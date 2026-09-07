"""
ServerApp Module - Flower Server Application for FlowerFL.

Implements the modern Flower ServerApp pattern with configurable
aggregation strategies (FedAvg, Krum, FedMedian, FedTrimmedAvg)
and pluggable Byzantine defense hooks.
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedMedian, FedTrimmedAvg, Krum
from flwr.common import Context, Metrics, ndarrays_to_parameters
from typing import List, Tuple

from flowerfl.task import get_num_clients, Net, detect_input_shape, get_weights, create_model, is_brfss_dataset
from flowerfl.byzantine_defense import (
    PluggableStrategy,
    KrumDefensePlugin,
    TrustScorePlugin,
    TGEnsemblePlugin,
)
from flowerfl.scenario_strategy import ScenarioStrategy
from flowerfl.mlflow_live import build_live_round_logger_from_env
from flowerfl.signal_logger import SignalLogger
from flowerfl.seeding import seed_everything
from flowerfl import fingerprint_registry as _fp_registry
from flowerfl.fingerprint_registry import (
    CalibrationCohort,
    FingerprintRegistry,
    TauNotLockedError,
)
import os
from pathlib import Path


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of client metrics.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    def get_weighted(key):
        valid = [(num, m) for num, m in metrics if key in m]
        if not valid:
            return 0.0
        total_examples = sum(num for num, _ in valid)
        if total_examples == 0:
            return 0.0
        weighted_sum = sum(num * m[key] for num, m in valid)
        return weighted_sum / total_examples

    return {
        "accuracy": get_weighted("accuracy"),
        "precision": get_weighted("precision"),
        "recall": get_weighted("recall"),
        "f1": get_weighted("f1"),
    }


def _signal_log_components(scenario_path: str, strategy_name: str, run_config):
    """The (seed, scenario_name, defense, exec_mode) that name a signal log."""
    seed = int(run_config.get("seed", 42))
    scenario_name = Path(scenario_path).stem
    defense = strategy_name.replace("Scenario", "").lower() or "unknown"
    optimizer_state = str(run_config.get("optimizer-state", "reset")).lower()
    exec_mode = "flower_persistent" if optimizer_state == "persistent" else "flower_reset"
    return seed, scenario_name, defense, exec_mode


def signal_log_path(scenario_path, strategy_name, run_config, signals_root=None):
    """The signals/ JSONL path this run will write, or None when signal logging
    is disabled or no scenario is set.

    Single source of truth for the filename (v23 Phase 1 P1.2 naming):
    `<exec_mode>__<scenario>__<defense>__seed<seed>.jsonl`. Shared with the
    runner's stale-log rotation (scripts/run_phase4_flower.py) so it targets
    EXACTLY the file the SignalLogger opens — the filename carries no SMOTE
    identity and the logger is append-only, so a rerun over a stale file would
    otherwise mix arms. ``signals_root`` overrides the default
    `signals/` dir (project root) for testability."""
    enabled = str(run_config.get("signal-log", "1")).lower() not in ("0", "false", "")
    if not enabled or not scenario_path:
        return None
    seed, scenario_name, defense, exec_mode = _signal_log_components(
        scenario_path, strategy_name, run_config
    )
    root = Path(signals_root) if signals_root is not None else Path(__file__).resolve().parent.parent / "signals"
    return root / f"{exec_mode}__{scenario_name}__{defense}__seed{seed}.jsonl"


def _maybe_create_signal_logger(
    run_config, dataset: str, scenario_path: str, strategy_name: str
):
    """Create a SignalLogger when the run_config enables signal logging.

    Default: ON. Disable with `signal-log=0` in run_config.
    Log directory: `signals/` under project root (v23 Phase 1 P1.2).
    Filename comes from signal_log_path (the single source shared with the
    runner's stale-log rotation). Distinct filenames prevent the two exec modes'
    signal data from being mixed (2026-05-26 finding: bug #4).
    """
    out_path = signal_log_path(scenario_path, strategy_name, run_config)
    if out_path is None:
        return None

    seed, scenario_name, defense, exec_mode = _signal_log_components(
        scenario_path, strategy_name, run_config
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Strategy] SignalLogger writing to: {out_path}")
    return SignalLogger(
        path=str(out_path),
        run_metadata={
            "seed": seed,
            "scenario": scenario_name,
            "exec_mode": exec_mode,
            "dataset": dataset,
            "defense": defense,
        },
    )


def _create_cs_plugin(model_path: str, k: int = 3, weighting: str = "linear"):
    """Load a trained cold-start detector and wrap it as a defense plugin."""
    if not model_path:
        print("[Strategy] WARNING: No cs-model path provided, cold-start plugin disabled")
        return None
    try:
        from flowerfl.cold_start_plugin import ColdStartDefensePlugin
        from flowerfl.cold_start_detector import ColdStartDetector
        detector = ColdStartDetector.load(model_path)
        plugin = ColdStartDefensePlugin(detector=detector, k=k, weighting=weighting)
        print(f"[Strategy] ColdStart plugin loaded: {model_path} (k={k}, weighting={weighting})")
        return plugin
    except Exception as e:
        print(f"[Strategy] WARNING: Could not load cold-start detector: {e}")
        return None


# ===========================================================================
# H3 fingerprint arm — registry construction and the PRE-LOCK posture
# ===========================================================================
# v1.10 § 5.1 gate (c) locks BOTH τ values in code before any H3 evaluation
# scenario runs, and `fingerprint_registry.locked_tau` / `locked_metric`
# raise `TauNotLockedError` until they are. But the acceptance smoke (EXP-049)
# runs the FP arm BEFORE τ exists, and per EMISSION_CONTRACT § 6.2 what it must
# still produce is: fingerprint emission at 100 % of client-rounds, both defense
# tokens, the aggregation coefficient, device ENROLMENT, custody, and the
# schema-v5 fields populated — everything except the matching branch, which is
# structurally unreachable pre-lock.
#
# That is implemented below as an explicit OBSERVE-ONLY registry. It is NOT a
# placeholder τ: inventing a numeric threshold before calibration is precisely
# what gate (c) forbids, and the dry-run measured how attractive a wrong τ looks
# (a cached-contract cohort produced `within_link_rate = 1.0` with no error at
# all — EMISSION_CONTRACT § 3.7). The observe-only object therefore carries NO
# finite threshold and cannot assert a match by any path.

#: Passed to the base constructor ONLY to clear its "positive finite tau"
#: validation, then immediately replaced by NaN in __init__. It is never a
#: threshold: no comparison in the registry ever sees this value.
_OBSERVE_ONLY_CONSTRUCTION_PLACEHOLDER = 1.0

#: Run-config key naming which locked τ/Σ pair a unit is scored under. Inert
#: before the lock (there is nothing to choose); REQUIRED after it.
_FP_COHORT_KEY = "fp-cohort"

#: Run-config key naming WHICH CANDIDATE POOL a re-entry decision considers
#: (v1.49 / the 2026-08-14 corrected-instrument amendment § 1). Absent = the
#: deployed flag-gated incumbent, byte for byte — the same absent-is-inert
#: design `fp-cohort` uses, so every pre-existing run is untouched.
_FP_REGISTRY_POLICY_KEY = "fp-registry-policy"


class ObserveOnlyFingerprintRegistry(FingerprintRegistry):
    """A registry that enrols, observes and logs — but can NEVER assert a match.

    The pre-τ-lock posture (EMISSION_CONTRACT § 6.2). Everything the smoke is
    for still happens: entries enrol, the EMA refreshes, generations are
    tracked, `MatchAssertion` rows are still emitted so the schema-v5 re-entry
    contract is exercised end to end. Only the *decision* is withheld, and it is
    withheld **structurally**, two independent ways over:

    1. :meth:`_nearest_flagged` returns ``(None, inf)`` — no candidate parent is
       ever considered, so ``matched = parent is not None and...`` is False
       before τ is even consulted. (This also avoids computing 10²³⁹-scale
       Mahalanobis distances under an uncalibrated Σ = I, which are meaningless.)
    2. ``self._tau`` is ``NaN`` — every ``min_d <= tau`` comparison is False by
       IEEE-754, including ``inf <= NaN``. There is no number here that could be
       mistaken for a pre-registered threshold, and `tau` logs as an explicit
       null (`scenario_strategy` maps non-finite floats to null, matching
       `signal_logger._jsonify`'s policy).

    Deleting either guard leaves the other; both must be removed to make a match
    reachable, which no accident does.

    When τ lands (EXP-050 → gate (c)), `build_fingerprint_registry` returns a
    plain :class:`FingerprintRegistry` carrying the locked τ and the locked
    metric from the SAME call site — no code change, no flag flip.
    """

    def __init__(self, metric=None, ema_alpha=_fp_registry.EMA_ALPHA,
                 dim=_fp_registry.FINGERPRINT_DIM, reason: str = "",
                 policy=_fp_registry.DEFAULT_REGISTRY_POLICY):
        super().__init__(
            tau=_OBSERVE_ONLY_CONSTRUCTION_PLACEHOLDER,
            metric=metric, ema_alpha=ema_alpha, dim=dim, policy=policy,
        )
        # After this line no finite threshold exists anywhere in the object.
        self._tau = float("nan")
        self._observe_only_reason = str(reason)
        print(
            "[Fingerprint] OBSERVE-ONLY REGISTRY: tau not locked — this arm "
            "enrols devices, emits fingerprints and writes the schema-v5 "
            "re-entry contract, but CANNOT assert a re-link match. No H3 "
            "re-link metric is computable from this run. "
            f"({reason or 'gate (c): tau is not calibrated'})",
            flush=True,
        )

    @property
    def is_observe_only(self) -> bool:
        return True

    @property
    def observe_only_reason(self) -> str:
        return self._observe_only_reason

    def _nearest_flagged(self, vector):
        """Guard 1: no candidate is ever considered pre-lock."""
        return (None, float("inf"))

    def _nearest_enrolled(self, vector, server_round):
        """Guard 1, for the identity-only pool.

        The pre-lock posture is a property of the RUN, not of the candidate
        policy, so a corrected-instrument arm smoked before its (τ, Σ) exists
        must be just as structurally unable to assert a match as a flag-gated
        one. Overriding only `_nearest_flagged` would have left guard 1 absent
        in exactly the mode the corrected H3 launches in.
        """
        return (None, float("inf"))


def _fingerprint_cohort(run_config):
    """The declared calibration cohort, or ``None`` when the run declares none."""
    raw = str(run_config.get(_FP_COHORT_KEY, "") or "").strip().lower()
    if not raw:
        return None
    try:
        return CalibrationCohort(raw)
    except ValueError as exc:
        raise ValueError(
            f"unknown {_FP_COHORT_KEY}={raw!r}; expected one of "
            f"{[c.value for c in CalibrationCohort]}"
        ) from exc


def _fingerprint_registry_policy(run_config):
    """The declared candidate policy, or the flag-gated default when absent.

    Unlike `fp-cohort` this has a legitimate default — the deployed incumbent —
    so an undeclared run is not refused; it simply runs the instrument every
    pre-existing experiment ran. A declared but UNKNOWN value is still a loud
    refusal: silently falling back to the incumbent would run the flag-gated
    pool under a design doc that asked for the corrected one.
    """
    raw = str(run_config.get(_FP_REGISTRY_POLICY_KEY, "") or "").strip().lower()
    if not raw:
        return _fp_registry.DEFAULT_REGISTRY_POLICY
    try:
        return _fp_registry.RegistryPolicy(raw)
    except ValueError as exc:
        raise ValueError(
            f"unknown {_FP_REGISTRY_POLICY_KEY}={raw!r}; expected one of "
            f"{[p.value for p in _fp_registry.RegistryPolicy]}"
        ) from exc


def build_fingerprint_registry(run_config):
    """The registry the FP arm runs with — locked when τ is, observe-only when not.

    Post-lock the cohort MUST be declared (`fp-cohort`): the validation and
    adjudicating τ/Σ pairs are different instruments (D9 axis (ii)), and
    silently defaulting to one of them would decide by accident which threshold
    a unit was scored under. Pre-lock the key is inert, because there is nothing
    to choose — so this refusal costs the acceptance smoke nothing and closes
    the gap the moment τ lands.
    """
    cohort = _fingerprint_cohort(run_config)
    policy = _fingerprint_registry_policy(run_config)
    if cohort is not None:
        try:
            tau = _fp_registry.locked_tau(cohort)
            metric = _fp_registry.locked_metric(cohort)
        except TauNotLockedError as exc:
            # Genuinely pre-lock only. TauLockIntegrityError (locked artifact
            # missing / off its SHA-256 pin / cohort block absent) is NOT
            # caught here on purpose: a damaged lock must stop the run, not
            # downgrade it to observe-only and produce results with no re-link
            # matching.
            return ObserveOnlyFingerprintRegistry(
                reason=f"cohort '{cohort.value}': {exc}", policy=policy
            )
        print(
            f"[Fingerprint] LOCKED registry: cohort={cohort.value} tau={tau:.6f} "
            f"metric_provenance={metric.provenance} policy={policy.value} "
            "(gate (c) satisfied)",
            flush=True,
        )
        return FingerprintRegistry(tau=tau, metric=metric, policy=policy)

    # No cohort declared. Legal only while nothing is locked.
    try:
        _fp_registry.locked_tau(CalibrationCohort.VALIDATION)
    except TauNotLockedError as exc:
        return ObserveOnlyFingerprintRegistry(
            reason=f"no {_FP_COHORT_KEY} declared and no tau locked: {exc}",
            policy=policy,
        )
    raise ValueError(
        f"tau is LOCKED but this run declares no {_FP_COHORT_KEY!r}. The "
        f"validation and adjudicating cohorts carry DIFFERENT locked tau/Sigma "
        f"pairs (v1.10 § 5.1 / D9 axis (ii)); refusing to pick one by default. "
        f"Set {_FP_COHORT_KEY} to "
        f"{[c.value for c in CalibrationCohort]}."
    )


#: Strategies that may host the H2' observer / detector plugin: the five H4
#: composition arms plus the erratum-B § B1 observer-attached BASE arms
#: (calibration units prepend the observer to Krum / TrustScore / the FedAvg
#: floor). `h2p-observe-only` on any other strategy is a loud refusal —
#: silently running a calibration unit observer-less would poison the census.
_H2P_OBSERVER_STRATEGIES = frozenset({
    "ScenarioH2PFPKrum", "ScenarioH2PFP", "ScenarioH2PKrum",
    "ScenarioH2PFPTS", "ScenarioH2PTS",
    "ScenarioKrum", "ScenarioTrustScore", "ScenarioNone",
})


def _h2p_observe_only_from_run_config(run_config) -> bool:
    """Strict bool coercion of the `h2p-observe-only` knob (default False).

    Values arrive as native bools (runner) or strings (CLI/pyproject). A
    typo'd value must NEVER coerce to the enforcing mode silently (erratum B
    § B1 — the observe/enforce distinction is the calibration design), so
    anything outside the closed true/false token sets refuses loudly.
    """
    val = run_config.get("h2p-observe-only", False)
    if isinstance(val, bool):
        return val
    token = str(val).strip().lower()
    if token in ("1", "true", "yes", "on"):
        return True
    if token in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(
        f"unknown h2p-observe-only={val!r}; expected a boolean "
        f"(true/false/1/0/yes/no/on/off) — erratum B refuses to guess "
        f"between observing and enforcing."
    )


def _h2p_cuts_version_from_run_config(run_config) -> str:
    """The EXPLICIT `h2p-cuts-version` knob; absent = the incumbent 'v1'.

    Validation of the closed {v1, v2} set lives in
    `h2prime_online._require_cuts_version` (single source): an unknown value
    refuses at bundle load, never auto-detects (erratum B § B2).
    """
    return (
        str(run_config.get("h2p-cuts-version", "") or "").strip().lower()
        or "v1"
    )


def _create_h2p_detector_plugin(run_config, scenario_path, strategy_name):
    """The H2' online detector (§ 7c-bis step 1 / erratum-B § B1 observer).

    Loads the frozen serving bundle at server startup — manifest-hash
    verification included — so a damaged or missing instrument stops the run
    before any round executes. The cut is resolved from the scenario name and
    the strategy's arm-class per the pinned `h2p-cuts-version` (v1: per-
    scenario cuts + the E3-bis C0 alias; v2: per-(scenario, arm-class) cuts,
    C0 calibrated directly, alias retired); no default operating point
    exists. `h2p-observe-only` switches the plugin to the erratum-B § B1
    calibration mode (scores + logs, drops nobody).
    """
    from pathlib import Path as _Path

    from flowerfl.h2prime_online import (
        ARM_CLASS_BY_STRATEGY,
        OnlineH2PrimeDetectorPlugin,
        load_serving_bundle,
        scenario_token_from_name,
    )

    bundle_dir = run_config.get("h2p-bundle-dir") or str(
        _Path(__file__).resolve().parent.parent / "data" / "h4_serving"
    )
    cuts_version = _h2p_cuts_version_from_run_config(run_config)
    observe_only = _h2p_observe_only_from_run_config(run_config)
    bundle = load_serving_bundle(bundle_dir, cuts_version=cuts_version)
    token = scenario_token_from_name(
        _Path(str(scenario_path)).stem, cuts_version=cuts_version
    )
    arm_class = ARM_CLASS_BY_STRATEGY.get(str(strategy_name))
    plugin = OnlineH2PrimeDetectorPlugin(
        bundle=bundle,
        scenario_token=token,
        observe_only=observe_only,
        arm_class=arm_class,
    )
    mode = "OBSERVE-ONLY (drops nobody)" if observe_only else "enforcing"
    print(
        f"[Strategy] H2' online detector attached (FIRST in chain; "
        f"mode={mode}; cuts_version={cuts_version} "
        f"bundle_sha256={bundle.bundle_sha256} scenario_cut {token}"
        f"[{arm_class}]={plugin.cut:.6f}, flag = STRICT score > cut)"
    )
    return plugin


def _maybe_h2p_observer_plugins(run_config, scenario_path, strategy_name,
                                plugins):
    """Prepend the observe-only H2' detector to a BASE arm's plugin chain
    when `h2p-observe-only` is on (erratum B § B1 calibration wiring).

    Returns a NEW list; the input is never mutated. When the knob is off the
    input chain is returned unchanged — byte-identical incumbent behavior.
    """
    if not _h2p_observe_only_from_run_config(run_config):
        return list(plugins)
    plugin = _create_h2p_detector_plugin(
        run_config, scenario_path, strategy_name
    )
    return [plugin, *plugins]


def _create_fingerprint_plugin(run_config):
    """The H3 fingerprint plugin, always LAST in the chain.

    Enforcement is the pre-registered D4 hard-drop (the plugin's default);
    `downweight`/`accept` exist only for the § 5.1 integrity test and are never
    constructed here.
    """
    from flowerfl.fingerprint_plugin import FingerprintDefensePlugin

    registry = build_fingerprint_registry(run_config)
    plugin = FingerprintDefensePlugin(registry=registry)
    print(
        "[Strategy] Fingerprint plugin attached (LAST in chain; observes the "
        "complete unfiltered cohort, D4 hard-drop enforcement)"
    )
    return plugin


def _holdout_disjoint_from_run_config(run_config) -> bool:
    """Coerce the ``holdout-disjoint`` run-config value to bool (default True).

    Values arrive as native bools (runner) or strings (CLI/pyproject). :
    the default is disjoint=True; only an explicit false-ish value opts back into
    the legacy overlapping holdout for prior-run reproduction.
    """
    val = run_config.get("holdout-disjoint", True)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")


def _create_eval_manager(dataset: str, run_config=None):
    """Create a FixedEvalManager for server-side evaluation.

    Raises RuntimeError on failure — v1.3 discipline. Returning None silently
    led to EXP-003 running 50 rounds with no eval trajectory, masquerading
    as success. Server-side eval is mandatory for trajectory parsing.

    : the holdout is disjoint from training by default (see FixedEvalManager).
    The train cap is read from the run's ``max-samples`` so the excluded rows
    match what load_data actually routed into training; 0/absent falls back to
    task.MAX_SAMPLES_PER_CLIENT (load_data's own fallback).
    """
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from rmc.fixed_eval import FixedEvalManager
    import flowerfl.task as _task
    eval_dataset = dataset.replace("_rmc", "")
    run_config = run_config or {}
    # H4 sealed-test evaluator (erratum-A E4): `eval-split=sealed_test`
    # evaluates on exactly the locked test indices of
    # data/val_test_split_manifest.json — no sampling, no seed. Absent or
    # "legacy" keeps the incumbent path byte-unchanged; anything else is a
    # loud refusal (never a silent fallback to the legacy sampler).
    eval_split = str(run_config.get("eval-split", "") or "").strip().lower()
    if eval_split not in ("", "legacy", "sealed_test"):
        raise ValueError(
            f"unknown eval-split={eval_split!r}; expected 'legacy' or "
            f"'sealed_test' (erratum-A E4)"
        )
    if eval_split == "sealed_test":
        from rmc.sealed_test_eval import SealedTestEvalManager

        return SealedTestEvalManager(dataset_name=eval_dataset)
    disjoint = _holdout_disjoint_from_run_config(run_config)
    max_samples = int(run_config.get("max-samples", 0))
    train_cap = max_samples if max_samples > 0 else _task.MAX_SAMPLES_PER_CLIENT
    try:
        mgr = FixedEvalManager(
            dataset_name=eval_dataset, samples_per_client=2000,
            disjoint=disjoint, train_max_samples=train_cap,
            # training genuinely ran on the un-mapped `dataset` (e.g.
            # edge_full_20_rmc); pass it so the reconstruction's is_full_dataset
            # cap gate matches load_data exactly (P1-4).
            train_dataset_name=dataset,
        )
    except Exception as e:
        raise RuntimeError(
            f"FixedEvalManager init failed for dataset={eval_dataset!r}: {e}. "
            f"This means server-side eval is broken and no trajectory will be parsed. "
            f"Most likely cause: data/{eval_dataset}/client_*.parquet files missing on this host."
        ) from e
    return mgr


def _scenario_defense_keep(cohort: int, num_malicious: int) -> int:
    """Multi-Krum keep count for scenario-mode defenses: ``max(1, cohort - f - 2)``.

    ``cohort`` is the per-round client cohort declared by the scenario
    (canonical RMC: 20 — NOT the 21 dataset partitions ``get_num_clients``
    returns for ``edge_full_20_rmc``), and ``num_malicious`` (f) is the
    scenario-declared sustained adversary count (canonical RMC: 9), giving
    keep = 20 - 9 - 2 = 9 — the documented deployed design.

    Raises ValueError loudly on a nonsensical (f, cohort) pair rather than
    silently clamping, so a misconfigured scenario fails fast instead of
    running a near-no-op defense (the keep-18/f-1 launch blocker this replaces
    was exactly such a silent near-no-op).
    """
    if num_malicious < 0:
        raise ValueError(
            f"num-malicious={num_malicious} is negative; expected the scenario's "
            f"sustained adversary count (>= 0)."
        )
    if num_malicious >= cohort - 2:
        raise ValueError(
            f"num-malicious={num_malicious} >= defense-cohort-size-2={cohort - 2}; "
            f"Multi-Krum keep would be <= 0. Check the scenario's declared adversary "
            f"count ({num_malicious}) against its per-round cohort ({cohort})."
        )
    return max(1, cohort - num_malicious - 2)


def server_fn(context: Context) -> ServerAppComponents:
    """
    Create server components with configured strategy.

    Reads configuration from context.run_config:
    - num-server-rounds: Number of FL rounds
    - dataset: Dataset name (edge/cic) for determining num_clients
    - strategy: Aggregation strategy (FedAvg/Krum/FedMedian/FedTrimmedAvg)
    - malicious-fraction: Fraction of malicious clients (for Krum config)
    """
    # Extract configuration
    run_config = context.run_config
    num_rounds = int(run_config.get("num-server-rounds", 5))
    dataset = run_config.get("dataset", "cic")
    strategy_name = run_config.get("strategy", "FedAvg")
    malicious_fraction = float(run_config.get("malicious-fraction", 0.0))
    scenario_path = run_config.get("scenario", "")

    # Get number of clients for this dataset
    num_clients = get_num_clients(dataset)
    # Honor an explicit scenario-derived adversary count when the runner
    # supplies one (methodology v1.19 launch-blocker fix): scenario-mode
    # Multi-Krum previously ran f=1/keep-18 because malicious-fraction
    # defaulted to 0.0. Absent the "num-malicious" key, fall back to the
    # legacy malicious-fraction derivation so non-scenario callers are
    # byte-for-byte unchanged.
    num_malicious = int(
        run_config.get("num-malicious", int(num_clients * malicious_fraction))
    )

    # Erratum B § B1: the observe-only knob is only meaningful on strategies
    # that host the H2' detector/observer. Refuse anything else LOUDLY (the
    # coercion itself also refuses garbage values) — a calibration unit that
    # silently ran observer-less would poison the calibration census.
    if (_h2p_observe_only_from_run_config(run_config)
            and strategy_name not in _H2P_OBSERVER_STRATEGIES):
        raise ValueError(
            f"h2p-observe-only=true is not supported for strategy "
            f"{strategy_name!r}; the observer attaches only to "
            f"{sorted(_H2P_OBSERVER_STRATEGIES)} (erratum B § B1)."
        )

    # seed all RNGs from the experiment seed BEFORE building the initial
    # global model. Prior to this the seed was logged but never reached torch's
    # RNG, so the initial model was drawn from OS entropy and identical-seed runs
    # diverged. Seeding here makes the broadcast round-0 model byte-reproducible.
    seed = int(run_config.get("seed", 42))
    seed_everything(seed)

    # Create initial model and get parameters
    # Use SzelagNet for BRFSS datasets, Net for everything else
    input_shape = detect_input_shape(dataset)
    initial_net = create_model(dataset, input_shape)
    initial_parameters = ndarrays_to_parameters(get_weights(initial_net))

    # Common strategy parameters
    base_params = {
        "initial_parameters": initial_parameters,
        "min_fit_clients": max(1, int(num_clients * 0.5)),
        "min_available_clients": num_clients,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": weighted_average,
        # thread the current round to clients via flwr's
        # standard on_fit_config_fn hook so FlowerClient.fit can derive its
        # per-(client, round) seed on NON-scenario strategy paths too (Krum,
        # FedTrimmedAvg, FedMedian, FedAvg, Plugin*). Without this, those paths
        # fell back to server_round=0 and replayed the same RNG stream every
        # round. Scenario* strategies ALSO inject this key in
        # ScenarioStrategy.configure_fit — same key, same value, so the double
        # injection is harmless (pinned by test_scenario_double_injection_is_harmless).
        "on_fit_config_fn": lambda server_round: {"server_round": server_round},
    }

    # Create strategy based on configuration
    if strategy_name == "Krum":
        # Krum requires knowing the expected number of malicious clients
        # num_clients_to_keep: Multi-Krum aggregates this many clients
        # Formula: max(1, n - f - 2) ensures we stay within theoretical bounds
        num_to_keep = max(1, num_clients - num_malicious - 2)
        print(f"[Strategy] Krum configured: num_malicious={num_malicious}, num_to_keep={num_to_keep}")
        strategy = Krum(
            **base_params,
            num_malicious_clients=num_malicious,
            num_clients_to_keep=num_to_keep,
        )

    elif strategy_name == "FedTrimmedAvg":
        # FedTrimmedAvg trims a fraction from each end
        # beta >= malicious_fraction/2 for robustness
        beta = max(0.1, malicious_fraction / 2)
        print(f"[Strategy] FedTrimmedAvg configured: beta={beta:.2f}")
        strategy = FedTrimmedAvg(
            **base_params,
            beta=beta,
        )

    elif strategy_name == "FedMedian":
        print(f"[Strategy] FedMedian configured")
        strategy = FedMedian(**base_params)

    elif strategy_name == "PluginKrum":
        # FedAvg base + Krum defense plugin (demonstrates hook architecture)
        base = FedAvg(**base_params)
        plugin = KrumDefensePlugin(
            num_malicious=num_malicious,
            num_to_keep=max(1, num_clients - num_malicious - 2),
        )
        print(f"[Strategy] PluginKrum: FedAvg + KrumDefensePlugin "
              f"(num_malicious={num_malicious})")
        strategy = PluggableStrategy(base, plugins=[plugin])

    elif strategy_name == "PluginTrust":
        # FedAvg base + TrustScore reputation plugin
        base = FedAvg(**base_params)
        plugin = TrustScorePlugin(decay_rate=0.9, outlier_threshold=2.0)
        print(f"[Strategy] PluginTrust: FedAvg + TrustScorePlugin")
        strategy = PluggableStrategy(base, plugins=[plugin])

    elif strategy_name == "PluginTGEnsemble":
        # FedAvg base + TGEnsemble plugin (GBDT + LSTM + tenure gate)
        base = FedAvg(**base_params)
        tge_plugin = TGEnsemblePlugin(
            num_malicious=num_malicious,
            num_to_keep=max(1, num_clients - num_malicious - 2),
        )
        print(f"[Strategy] PluginTGEnsemble: FedAvg + TGEnsemblePlugin")
        strategy = PluggableStrategy(base, plugins=[tge_plugin])

    elif strategy_name in ("ScenarioNone", "ScenarioKrum", "ScenarioTGEnsemble",
                           "ScenarioTrustScore", "ScenarioKrumCS", "ScenarioTrustScoreCS",
                           "ScenarioKrumTGE", "ScenarioTGEPrime", "ScenarioKrumTGEPrime",
                           "ScenarioFedMedian", "ScenarioFedTrimmedAvg",
                           "ScenarioTGEFP", "ScenarioKrumTGEFP",
                           "ScenarioH2PFPKrum", "ScenarioH2PFP",
                           "ScenarioH2PKrum", "ScenarioH2PFPTS",
                           "ScenarioH2PTS"):
        # Scenario-driven experiments: need ALL clients sampled every round
        # so ScenarioStrategy can do its own filtering per scenario schedule
        scenario_base_params = dict(base_params)
        scenario_base_params["min_fit_clients"] = num_clients
        scenario_base_params["fraction_fit"] = 1.0
        base = FedAvg(**scenario_base_params)
        # Per-round client cohort for the defenses' Multi-Krum keep math. The
        # runner supplies the scenario-declared cohort (20 for canonical RMC —
        # the 20-client per-round cohort, not the 21 dataset partitions);
        # fall back to the partition count for non-scenario callers
        # (methodology v1.19).
        cohort = int(run_config.get("defense-cohort-size", num_clients))
        eval_mgr = _create_eval_manager(dataset, run_config)
        sig_logger = _maybe_create_signal_logger(
            run_config, dataset, scenario_path, strategy_name
        )

        if strategy_name == "ScenarioNone":
            # Erratum B § B1: `h2p-observe-only` prepends the observe-only H2'
            # detector (fedavg_family calibration arm); off/absent = [] as ever.
            strategy = ScenarioStrategy(
                base,
                plugins=_maybe_h2p_observer_plugins(
                    run_config, scenario_path, strategy_name, []
                ),
                scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioNone: FedAvg + scenario scheduling")

        elif strategy_name == "ScenarioKrum":
            # Static values are provenance + the full-cohort operating-point
            # cross-check; deployment sizing is per-round dynamic
            # f=ceil(n/2)-1 (April anchor formula) so S3/S4 disconnect rounds
            # (n~11) stay computable. At the full cohort the two
            # coincide: ceil(20/2)-1 = 9 = num_malicious, keep = 9.
            num_to_keep = _scenario_defense_keep(cohort, num_malicious)
            plugin = KrumDefensePlugin(
                num_malicious=num_malicious,
                num_to_keep=num_to_keep,
                dynamic_f=True,
            )
            strategy = ScenarioStrategy(
                base,
                # Erratum B § B1: observe-only observer prepends when the
                # knob is on (krum_family calibration arm); off = incumbent.
                plugins=_maybe_h2p_observer_plugins(
                    run_config, scenario_path, strategy_name, [plugin]
                ),
                scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            # Two distinct facts, labeled distinctly ( round-2 P2):
            # scenario_declared_adversaries = ground truth from the scenario
            # (provenance + guard + full-cohort cross-check); krum_f_policy =
            # what the deployed defense actually sizes with (threat-model-
            # constant, independent of declared intensity).
            print(f"[Strategy] ScenarioKrum: FedAvg + KrumDefense "
                  f"(scenario_declared_adversaries={num_malicious}, cohort={cohort}, "
                  f"full-cohort keep={num_to_keep}, "
                  f'krum_f_policy="dynamic ceil(n/2)-1") + scenario scheduling')

        elif strategy_name == "ScenarioTrustScore":
            plugin = TrustScorePlugin(decay_rate=0.9, outlier_threshold=2.0)
            strategy = ScenarioStrategy(
                base,
                # Erratum B § B1: observe-only observer prepends when the
                # knob is on (ts_family calibration arm); off = incumbent.
                plugins=_maybe_h2p_observer_plugins(
                    run_config, scenario_path, strategy_name, [plugin]
                ),
                scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioTrustScore: FedAvg + TrustScore reputation + scenario scheduling")

        elif strategy_name == "ScenarioKrumCS":
            # Phase 4: Krum + Cold-Start Detector (layered)
            from flowerfl.cold_start_plugin import ColdStartDefensePlugin
            num_to_keep = _scenario_defense_keep(cohort, num_malicious)
            # dynamic_f: per-round ceil(n/2)-1 sizing; statics = provenance
            # (see ScenarioKrum branch comment; ).
            krum_plugin = KrumDefensePlugin(
                num_malicious=num_malicious,
                num_to_keep=num_to_keep,
                dynamic_f=True,
            )
            cs_model_path = run_config.get("cs-model", "")
            cs_weighting = run_config.get("cs-weighting", "linear")
            cs_k = int(run_config.get("cs-k", 3))
            cs_plugin = _create_cs_plugin(cs_model_path, cs_k, cs_weighting)
            plugins = [krum_plugin] + ([cs_plugin] if cs_plugin else [])
            strategy = ScenarioStrategy(
                base, plugins=plugins, scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioKrumCS: Krum "
                  f"(scenario_declared_adversaries={num_malicious}, cohort={cohort}, "
                  f"full-cohort keep={num_to_keep}, "
                  f'krum_f_policy="dynamic ceil(n/2)-1") '
                  f"+ ColdStart({cs_weighting}) + scenario scheduling")

        elif strategy_name == "ScenarioTrustScoreCS":
            # Phase 4: TrustScore + Cold-Start Detector (layered)
            from flowerfl.cold_start_plugin import ColdStartDefensePlugin
            trust_plugin = TrustScorePlugin(decay_rate=0.9, outlier_threshold=2.0)
            cs_model_path = run_config.get("cs-model", "")
            cs_weighting = run_config.get("cs-weighting", "linear")
            cs_k = int(run_config.get("cs-k", 3))
            cs_plugin = _create_cs_plugin(cs_model_path, cs_k, cs_weighting)
            plugins = [trust_plugin] + ([cs_plugin] if cs_plugin else [])
            strategy = ScenarioStrategy(
                base, plugins=plugins, scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioTrustScoreCS: TrustScore + ColdStart({cs_weighting}) + scenario scheduling")

        elif strategy_name == "ScenarioTGEnsemble":
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            # num_malicious / num_to_keep are INERT for TGEnsemblePlugin — the
            # 0.7 score threshold is its sole filter (the params are stored in
            # __init__ but never read by score_updates/filter_updates; see
            # flowerfl/byzantine_defense.py TGEnsemblePlugin). Passed for
            # construction-signature parity only; TGE semantics are unchanged
            # by the v1.19 keep-9 fix.
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
            )
            strategy = ScenarioStrategy(
                base, plugins=[tge_plugin], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioTGEnsemble: FedAvg + TGEnsemble (GBDT+LSTM, ramp_rounds={tge_ramp_rounds}, threshold-filter=0.7) + scenario scheduling")

        elif strategy_name == "ScenarioKrumTGE":
            # Krum layer: documented full-cohort operating point f=9 / keep-9
            # (cohort 20 - f 9 - 2); deployment sizing is per-round dynamic
            # f=ceil(n/2)-1 (April anchor formula, ) so disconnect
            # rounds stay computable. Statics = provenance.
            num_to_keep = _scenario_defense_keep(cohort, num_malicious)
            krum_plugin = KrumDefensePlugin(
                num_malicious=num_malicious,
                num_to_keep=num_to_keep,
                dynamic_f=True,
            )
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            # TGE layer: num_malicious / num_to_keep are INERT (0.7 score
            # threshold is its sole filter — stored but never read; see
            # flowerfl/byzantine_defense.py TGEnsemblePlugin). Passed for
            # signature parity only; TGE semantics unchanged by the v1.19 fix.
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
            )
            strategy = ScenarioStrategy(
                base, plugins=[krum_plugin, tge_plugin], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioKrumTGE: Krum "
                  f"(scenario_declared_adversaries={num_malicious}, cohort={cohort}, "
                  f"full-cohort keep={num_to_keep}, "
                  f'krum_f_policy="dynamic ceil(n/2)-1") + TGEnsemble '
                  f"(GBDT+LSTM, ramp_rounds={tge_ramp_rounds}, threshold-filter=0.7) + scenario scheduling")

        elif strategy_name == "ScenarioTGEPrime":
            # TGE′ : FedAvg + TGEnsemblePlugin with the two-leg
            # long-memory BANK (LSTM + EMA reputation, combined by min).
            # Identical wiring to ScenarioTGEnsemble; only long_memory_expert
            # differs. ramp/alpha are PROVISIONAL (amendment v1.7 pending).
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            tge_ema_alpha = float(run_config.get("tge-ema-alpha", 0.9))   # ADOPTED from TrustScore
            # Honor the configured long-memory mode : the prime
            # token deploys "bank" by default, but an explicit
            # tge-long-memory-expert override (e.g. "ema" for component
            # isolation) must actually execute — and match what
            # tge_provenance_fields records from the same key.
            tge_long_memory = str(run_config.get("tge-long-memory-expert", "bank"))
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
                long_memory_expert=tge_long_memory,
                ema_alpha=tge_ema_alpha,
            )
            strategy = ScenarioStrategy(
                base, plugins=[tge_plugin], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioTGEPrime: FedAvg + TGE′ ({tge_long_memory}: GBDT + long-memory leg, "
                  f"ramp_rounds={tge_ramp_rounds}, ema_alpha={tge_ema_alpha}, threshold-filter=0.7) + scenario scheduling")

        elif strategy_name == "ScenarioKrumTGEPrime":
            # TGE′ composed behind Krum. Krum layer identical to
            # ScenarioKrumTGE; the TGE layer uses the two-leg bank.
            num_to_keep = _scenario_defense_keep(cohort, num_malicious)
            krum_plugin = KrumDefensePlugin(
                num_malicious=num_malicious,
                num_to_keep=num_to_keep,
                dynamic_f=True,
            )
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            tge_ema_alpha = float(run_config.get("tge-ema-alpha", 0.9))   # ADOPTED from TrustScore
            # Honor the configured long-memory mode — see the
            # ScenarioTGEPrime branch above.
            tge_long_memory = str(run_config.get("tge-long-memory-expert", "bank"))
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
                long_memory_expert=tge_long_memory,
                ema_alpha=tge_ema_alpha,
            )
            strategy = ScenarioStrategy(
                base, plugins=[krum_plugin, tge_plugin], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioKrumTGEPrime: Krum "
                  f"(scenario_declared_adversaries={num_malicious}, cohort={cohort}, "
                  f"full-cohort keep={num_to_keep}, "
                  f'krum_f_policy="dynamic ceil(n/2)-1") + TGE′ '
                  f"({tge_long_memory}: GBDT + long-memory leg, ramp_rounds={tge_ramp_rounds}, ema_alpha={tge_ema_alpha}, threshold-filter=0.7) + scenario scheduling")

        elif strategy_name == "ScenarioTGEFP":
            # H3 scored arm (v1.10 § 5.1 ARM RULE): ScenarioTGEnsemble's chain
            # PLUS the fingerprint plugin, and nothing else — same TGE layer,
            # same gate-selected ramp — so `TGE` vs `TGE+FP` stays a
            # single-variable comparison. The FP plugin is LAST: it observes the
            # complete unfiltered cohort in observe_cohort, and being last is
            # what lets score_updates infer which clients TGE rejected (see
            # flowerfl/fingerprint_plugin.py module docstring).
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
            )
            fp_plugin = _create_fingerprint_plugin(run_config)
            strategy = ScenarioStrategy(
                base, plugins=[tge_plugin, fp_plugin], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioTGEFP: FedAvg + TGEnsemble "
                  f"(GBDT+LSTM, ramp_rounds={tge_ramp_rounds}, threshold-filter=0.7) "
                  f"+ Fingerprint (LAST) + scenario scheduling")

        elif strategy_name == "ScenarioKrumTGEFP":
            # H4-composable arm: ScenarioKrumTGE's chain PLUS the fingerprint
            # plugin, last. Krum layer built identically to ScenarioKrumTGE.
            num_to_keep = _scenario_defense_keep(cohort, num_malicious)
            krum_plugin = KrumDefensePlugin(
                num_malicious=num_malicious,
                num_to_keep=num_to_keep,
                dynamic_f=True,
            )
            tge_ramp_rounds = int(run_config.get("tge-ramp-rounds", 8))  # provisional canonical (v1.6 § 2)
            tge_plugin = TGEnsemblePlugin(
                num_malicious=num_malicious,
                num_to_keep=max(1, cohort - num_malicious - 2),
                ramp_rounds=tge_ramp_rounds,
            )
            fp_plugin = _create_fingerprint_plugin(run_config)
            strategy = ScenarioStrategy(
                base, plugins=[krum_plugin, tge_plugin, fp_plugin],
                scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioKrumTGEFP: Krum "
                  f"(scenario_declared_adversaries={num_malicious}, cohort={cohort}, "
                  f"full-cohort keep={num_to_keep}, "
                  f'krum_f_policy="dynamic ceil(n/2)-1") + TGEnsemble '
                  f"(GBDT+LSTM, ramp_rounds={tge_ramp_rounds}, threshold-filter=0.7) "
                  f"+ Fingerprint (LAST) + scenario scheduling")

        elif strategy_name in ("ScenarioH2PFPKrum", "ScenarioH2PFP",
                               "ScenarioH2PKrum", "ScenarioH2PFPTS",
                               "ScenarioH2PTS"):
            # H4 composition arms (spec 2026-08-16 § 2/§ 7c-bis as amended by
            # erratum-A; methodology v1.51/v1.52). The FROZEN per-round order
            # is the plugin chain order:
            #   (1) H2' online detector FIRST — scores the PRE-filter stream,
            #       flagged clients dropped from this round's aggregation;
            #   (2) FLAG_GATED fingerprint registry — matched re-entries of
            #       previously dropped identities hard-dropped at admission
            #       (the detector's drops are exactly the "upstream
            #       rejections" the FP plugin flags: it compares its full
            #       observe_cohort membership against the post-detector
            #       survivor stream it scores);
            #   (3) aggregator over the kept set — Krum (arms 1/5),
            #       TrustScore (arms 6/9), or the FedAvg base alone (arm 3).
            # Krum/TrustScore layers are constructed IDENTICALLY to their
            # standalone ScenarioKrum / ScenarioTrustScore arms.
            h2p_plugin = _create_h2p_detector_plugin(
                run_config, scenario_path, strategy_name
            )
            plugins = [h2p_plugin]
            if strategy_name in ("ScenarioH2PFPKrum", "ScenarioH2PFP",
                                 "ScenarioH2PFPTS"):
                plugins.append(_create_fingerprint_plugin(run_config))
            if strategy_name in ("ScenarioH2PFPKrum", "ScenarioH2PKrum"):
                num_to_keep = _scenario_defense_keep(cohort, num_malicious)
                plugins.append(KrumDefensePlugin(
                    num_malicious=num_malicious,
                    num_to_keep=num_to_keep,
                    dynamic_f=True,
                ))
                agg_desc = (
                    f"Krum (scenario_declared_adversaries={num_malicious}, "
                    f"cohort={cohort}, full-cohort keep={num_to_keep}, "
                    f'krum_f_policy="dynamic ceil(n/2)-1")'
                )
            elif strategy_name in ("ScenarioH2PFPTS", "ScenarioH2PTS"):
                plugins.append(
                    TrustScorePlugin(decay_rate=0.9, outlier_threshold=2.0)
                )
                agg_desc = "TrustScore reputation"
            else:
                agg_desc = "FedAvg (no step-3 filter)"
            strategy = ScenarioStrategy(
                base, plugins=plugins, scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            fp_desc = (
                " + Fingerprint (FLAG_GATED, hard-drop at admission)"
                if strategy_name in ("ScenarioH2PFPKrum", "ScenarioH2PFP",
                                     "ScenarioH2PFPTS") else ""
            )
            print(f"[Strategy] {strategy_name}: H2' online detector (FIRST, "
                  f"pre-filter stream){fp_desc} + {agg_desc} + scenario "
                  f"scheduling (frozen 7c-bis order)")

        elif strategy_name == "ScenarioFedMedian":
            strategy = ScenarioStrategy(
                FedMedian(**scenario_base_params), plugins=[], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioFedMedian: coordinate-wise median (utility baseline) + scenario scheduling")

        elif strategy_name == "ScenarioFedTrimmedAvg":
            strategy = ScenarioStrategy(
                FedTrimmedAvg(beta=0.4, **scenario_base_params), plugins=[], scenario_path=scenario_path,
                eval_manager=eval_mgr, signal_logger=sig_logger,
            )
            print(f"[Strategy] ScenarioFedTrimmedAvg: trimmed mean beta=0.4 (utility baseline) + scenario scheduling")

    else:  # Default to FedAvg
        print(f"[Strategy] FedAvg configured")
        strategy = FedAvg(**base_params)

    # Live per-round MLflow streaming (instrumentation only; spec Lane C item
    # 10). Attach the env-derived logger so ScenarioStrategy.evaluate streams
    # each round's metrics to MLflow as it completes. Returns None (→ no-op)
    # unless the unit was launched via `praxis exp launch` (PRAXIS_MLFLOW_RUN_ID
    # set). Best-effort and additive: only ScenarioStrategy exposes the hook.
    if isinstance(strategy, ScenarioStrategy):
        strategy.set_live_metric_logger(build_live_round_logger_from_env())

    # Print configuration summary
    print("=" * 60)
    print("FLOWERFL SERVER CONFIGURATION")
    print("=" * 60)
    print(f"Dataset:            {dataset}")
    print(f"Num Clients:        {num_clients}")
    print(f"Rounds:             {num_rounds}")
    print(f"Malicious Fraction: {malicious_fraction * 100:.0f}%")
    print(f"Strategy:           {strategy_name}")
    print("=" * 60)

    # Create server config
    server_config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )


# Create the ServerApp
app = ServerApp(server_fn=server_fn)

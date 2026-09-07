"""Phase 4 Track C — Paired comparison in Flower mode.

THIS IS THE PRAXIS'S UNIFIED PHASE-4 EXPERIMENT RUNNER. It supports both
execution modes (see `--optimizer-state` below) and every defense config in
SUPPORTED_CONFIGS. Real experiments MUST be launched via `praxis exp launch EXP-NNN`, which
wraps this script with git-tag provenance and MLflow run linkage — do NOT
invoke this file directly to start a fresh experiment (direct invocation is
fine for local debugging/dry-runs, but results from that path carry no
provenance and must not be cited as experiment output).

Runs N seeds x M configs on a given RMC scenario via Flower simulation
(`flwr.simulation.run_simulation`, called in-process — NOT the `flwr run`
CLI). Aggregates results into a `phase4_flower_paired_comparison.json`
summary alongside per-run JSONs, one per (config, seed) pair.

Execution modes (`--optimizer-state` / legacy `--modes` alias) — these are
NEVER pooled in analysis; each result JSON's `provenance.optimizer_state`
field records which mode produced it:
    - "reset"      (`--modes Flower`): vanilla per-round Adam reset — the
      client's optimizer state does not survive across rounds. This is
      Flower's off-the-shelf default behavior.
    - "persistent" (`--modes persistent_optimizer`): Adam momentum/variance
      persist across rounds for the same flower_cid, via the module-level
      singleton in flowerfl/persistent_optimizer.py. This models Szelag's
      actual training setup and is the mode used for the confirmatory H1/H2
      runs (spec v1.3 § 2) — client optimizer state accumulating across
      reconnects is part of what the RMC threat model exploits.
    Each mode reads its own locked learning rate from data/hparams_locked.json
    (see _load_locked_lr below); the two modes are NOT interchangeable and
    mixing their results in one comparison would confound the optimizer-state
    variable with whatever else is being measured.

Configurations (see SUPPORTED_CONFIGS / build_strategy_for_config below):
- Krum, Krum+CS, TrustScore, TrustScore+CS  (pre-correction-era + CS ablation)
- TGE, Krum+TGE                             (the thesis defense arms, spec § 6.1)
- FedMedian, FedTrimmedAvg                  (utility baselines; H4 deferred)
- TGE+FP, Krum+TGE+FP                       (H3 fingerprint arms, v1.10 § 5.1;
                                             the ONLY configs that emit
                                             `FitRes.metrics["fingerprint"]`)

Output schema (one JSON per run, see `result` dict in run_one() below):
    config, strategy, seed, return_code, elapsed_seconds, trajectory
    (per-round f1/accuracy/loss), alie_rounds, mean/final accuracy + f1,
    post_reconnect_accuracy, provenance (runner_version/commit, scenario_path,
    optimizer_state, flwr_version, ...), plus convergence / defense_overhead /
    confounder_control blocks appended post-hoc by _enrich_result_with_metrics().
    Per-round F1 trajectory is parsed from captured stdout (ScenarioStrategy's
    eval print lines), not read from a structured Flower API.

Integrity assertions (A1-A6, spec § 4.9): this runner cross-checks its own
output for known historical failure modes (silent lr drift, optimizer_state
mismatch, Ray actor-pool participant truncation, scenario/partition mapping
bugs, missing signal logs, empty eval trajectories) and raises AssertionError
rather than silently emitting a corrupted result — see the `_assert_*`
functions below, each of which cites the specific bug it guards against.

Usage:
    conda run -n flowerfl python scripts/run_phase4_flower.py
    conda run -n flowerfl python scripts/run_phase4_flower.py --seeds 42
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Local import path bootstrap so `from flowerfl.result_metrics import ...`
# resolves regardless of CWD.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT_BOOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))
from flowerfl.result_metrics import (  # noqa: E402
    compute_convergence_metrics,
    compute_confounder_control_metrics,
)


def _summarize_ms(ms_list: list[float]) -> dict:
    """Summary stats for a list of per-round wall-clock times (ms)."""
    if not ms_list:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(ms_list)
    n = len(s)
    return {
        "mean": sum(s) / n,
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
    }


def _fingerprint_custody(plugins: list) -> "dict | None":
    """Run-end custody record for the H3 fingerprint registry, or None.

    The registry is in-memory and server-process-scoped, so nothing about it
    survives the simulation unless it is exported here. Two things are exported:

    1. **Per-entry provenance** — identity / lifecycle / flag / generation, plus
       a SHA-256 of each entry's final EMA vector.
    2. **The per-observation log** — every AS-OBSERVED
       ``(server_round, session_key, 180-dim vector)`` draw the registry saw.

    `docs/reproduction/experiments.md` describes the public H3 workflow. The
    historical protocol's **A.6** requires (2), while A.5(a) pre-registers a
    NON-GATING naive-Euclidean
    baseline comparator on the un-whitened vector; the schema-v5 re-entry row
    carries `asserted_match` / `asserted_parent_logical_id` / `min_d` / `tau`
    but **not** the vector, so A.6 makes persisting it a blocking build
    prerequisite — "shall be logged (signal log or a sibling artifact) before
    the validation run launches; otherwise A.5(a) is unrecoverable without a
    re-run". This custody record IS that sibling artifact; the signal-log schema
    stays v5 and is deliberately not touched (the emission contract keeps the
    fingerprint on `FitRes.metrics`).

    An earlier revision of this docstring argued the opposite — that a second
    fingerprint channel in the result JSON would be "an unowned path into a
    locked pre-registration constant". A.6 supersedes that. The path is owned
    and it is one-way: the registry never reads the log back, nothing in this
    record is an input to τ, to Σ, to a re-link decision or to any gate. It is
    custody and offline-analysis evidence, full stop.

    Vector encoding: the raw little-endian float64 buffer, base64 encoded, so
    bit-exactness is a property of the bytes rather than of decimal formatting.
    Read it offline with the one line the record advertises in its own
    `read_offline` field::

        np.frombuffer(base64.b64decode(row["fingerprint_vec_b64"]), dtype="<f8")

    At the designed ~50-round × ~20-device shape that is ~1 000 rows ≈ 2.0 MB of
    `indent=2` JSON per unit (a decimal list would be ~5.7 MB, since `indent=2`
    puts every one of the 180 000 floats on its own line).
    `observation_log.truncated` / `dropped_count` report the bound: a truncated
    corpus must never be scored as if it were whole.

    `tau_posture` records whether the run executed with a LOCKED tau or in the
    pre-lock OBSERVE-ONLY posture, so no reader can mistake a run that could
    never assert a match for one that asserted none. The observe-only posture
    still logs every observation — that run must remain analysable.
    """
    import hashlib

    import numpy as np

    from flowerfl.fingerprint_registry import EMA_ALPHA

    def _calibration_artifact_sha256():
        from flowerfl.fingerprint_registry import CALIBRATION_ARTIFACT_SHA256

        return CALIBRATION_ARTIFACT_SHA256

    fp_plugins = [p for p in plugins if getattr(p, "name", "") == "Fingerprint"]
    if not fp_plugins:
        return None
    plugin = fp_plugins[0]
    registry = plugin.registry
    observe_only = bool(getattr(registry, "is_observe_only", False))

    entries = []
    for entry in sorted(registry.entries(), key=lambda e: e.entry_id):
        vector = np.asarray(entry.fingerprint_vec, dtype=np.float64)
        entries.append({
            "entry_id": entry.entry_id,
            "session_key": entry.session_key,
            "logical_id": entry.logical_id,
            "flower_cid": entry.flower_cid,
            "first_seen_round": int(entry.first_seen_round),
            "last_seen_round": int(entry.last_seen_round),
            "flag_status": bool(entry.flag_status),
            "flag_reason": entry.flag_reason,
            "flag_round": entry.flag_round,
            "generation": int(entry.generation),
            "parent_entry_id": entry.parent_entry_id,
            "fingerprint_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "fingerprint_finite": bool(np.all(np.isfinite(vector))),
        })

    observations = registry.observations()
    observation_log = {
        # A.6: how to read `rows` offline, stated in the artifact itself.
        "encoding": "base64_float64_le",
        "read_offline": (
            "np.frombuffer(base64.b64decode(row['fingerprint_vec_b64']), "
            "dtype='<f8')"
        ),
        "dim": int(registry.metric.dim),
        "ema_alpha": float(EMA_ALPHA),
        "row_count": len(observations),
        "max_rows": int(registry.observation_log_max_rows),
        "truncated": bool(registry.observation_log_truncated),
        "dropped_count": int(registry.observation_log_dropped_count),
        "rows": [o.as_custody_row() for o in observations],
    }

    rejections = registry.upstream_rejections()
    upstream_rejection_log = {
        "row_count": len(rejections),
        "max_rows": int(registry.upstream_rejection_log_max_rows),
        "truncated": bool(registry.upstream_rejection_log_truncated),
        "dropped_count": int(registry.upstream_rejection_log_dropped_count),
        "rows": [r.as_custody_row() for r in rejections],
    }

    return {
        "tau_posture": "observe_only_prelock" if observe_only else "locked",
        "tau": None if observe_only else float(registry.tau),
        # v1.49 / corrected-instrument § 1: WHICH candidate pool produced these
        # re-entry events. flag_gated and identity_only are different
        # instruments, so a reader must never have to infer it from launch
        # records. Always present, including for incumbent runs.
        "registry_policy": registry.policy.value,
        # The run identity stamped into every `reentry_event_key`
        # (`{run_uid}:{round}:{cid}`). Recorded so an offline scorer can bind
        # this result JSON to exactly the events it produced, rather than to
        # any run that happens to share its (scenario, seed) cell.
        "run_uid": getattr(plugin, "run_id", None) or None,
        "observe_only_reason": (
            getattr(registry, "observe_only_reason", "") if observe_only else None
        ),
        "metric_provenance": registry.metric.provenance,
        # Gate (c) corroboration for the offline scorer: the pinned hash of the
        # calibration artifact this image carried. The corrected-instrument
        # scorer requires it to EQUAL its own pin exactly — an absent value
        # refuses — so a unit can prove it scored under the exact locked
        # artifact, not merely a metric with a plausible provenance string.
        "calibration_artifact_sha256": (
            None if observe_only else _calibration_artifact_sha256()
        ),
        "enforcement_mode": getattr(plugin.enforcement_mode, "value", None),
        "enrollment_round": plugin.enrollment_round,
        "initial_enrollments": list(plugin.initial_enrollments),
        "entry_count": len(entries),
        "flagged_entry_count": sum(1 for e in entries if e["flag_status"]),
        # Gate (e) telemetry. The denominator is PARTICIPATING client-rounds —
        # it excludes discovery-round fits (never routed to the plugin, by
        # design) and scenario-dropped clients. EMISSION_CONTRACT § 4.5:
        # "100 % of client-rounds" reads as "100 % of participating
        # client-rounds". The names carry that so no reader can take these for
        # whole-run quantities, and the denominator is declared explicitly.
        "emission_denominator": "participating_client_rounds_excl_discovery",
        "participating_client_round_count": plugin.participating_client_round_count,
        "participating_missing_fingerprint_count": (
            plugin.participating_missing_fingerprint_count
        ),
        "participating_fingerprint_emission_rate": (
            plugin.participating_fingerprint_emission_rate()
        ),
        "reentry_assertion_count": len(plugin.reentry_events),
        "entries": entries,
        # A.6 (blocking prerequisite for the A.5(a) baseline comparator).
        "observation_log": observation_log,
        # A.6, second half: every upstream rejection, recorded unconditionally.
        # The entry lifecycle below records only each entry's FIRST flag, so
        # without this an already-inherited-flagged entry's later rejections are
        # unrecoverable — and those are what the Euclidean counterfactual needs.
        "upstream_rejection_log": upstream_rejection_log,
    }


def _enrich_result_with_metrics(result: dict, scenario_path: str, strategy_obj) -> None:
    """Inject convergence + confounder_control + defense_overhead blocks.

    Mutates ``result`` in place. Safe when ``strategy_obj`` is None or lacks
    expected attributes — degrades to zero-summaries / absent blocks.
    """
    trajectory = result.get("trajectory", [])
    result["convergence"] = compute_convergence_metrics(trajectory)

    # Defense overhead — Krum aggregation timing from the strategy itself,
    # plus per-plugin scoring timing where available. Each getattr() chain
    # safely returns [] when the plugin/attribute is absent, so configs
    # without TGE or CS still produce a structurally-valid block.
    krum_ms = list(getattr(strategy_obj, "_krum_timing_per_round", []) or [])
    plugins = list(getattr(strategy_obj, "_plugins", []) or [])
    tge_ms: list[float] = []
    cs_ms: list[float] = []
    for p in plugins:
        name = getattr(p, "name", "")
        if name == "TGEnsemble":
            tge_ms.extend(getattr(p, "_timing_per_round", []) or [])
        elif name == "ColdStartDetector":
            cs_ms.extend(getattr(p, "_timing_per_round", []) or [])
    result["defense_overhead"] = {
        "krum_aggregation_time_per_round_ms": _summarize_ms(krum_ms),
        "tge_score_time_per_round_ms": _summarize_ms(tge_ms),
        "fp_match_time_per_round_ms": _summarize_ms(cs_ms),
    }

    # Confounder-control — gated on scenario having honest_events. Load the
    # scenario JSON ourselves rather than trusting the strategy's cache to
    # avoid reaching into private state.
    scenario_dict: dict = {}
    try:
        with open(scenario_path) as _f:
            scenario_dict = json.load(_f)
    except (OSError, ValueError):
        scenario_dict = {}

    # Collect per-(round, cid, trust) scoring entries from CS plugins.
    raw_log: list[tuple[int, str, float]] = []
    for p in plugins:
        for entry in getattr(p, "_scoring_log", []) or []:
            raw_log.append(entry)

    # Translate flower_cid → logical_cid via the strategy's partition map.
    p2l = dict(getattr(strategy_obj, "_partition_to_logical", {}) or {})
    c2p = dict(getattr(strategy_obj, "_cid_to_partition", {}) or {})
    translated: list[tuple[int, str, float]] = []
    for round_, cid, trust in raw_log:
        pid = c2p.get(cid)
        logical = p2l.get(pid, cid) if pid is not None else cid
        translated.append((int(round_), str(logical), float(trust)))

    cc = compute_confounder_control_metrics(scenario_dict, trajectory, translated)
    if cc is not None:
        result["confounder_control"] = cc

    # Stage-F §5: lift the server-collected per-client
    # resampling manifest into the result JSON so completed Stage-F units carry
    # durable per-client resampling/step evidence. Absent (empty) for incumbent
    # runs — no manifest metric is emitted when prep_info was never requested.
    #
    # BLOCKER fix: before persisting, enforce the unit-level completeness gate.
    # When the strategy exposes BOTH the collection and the authoritative
    # dispatched-partition set (i.e. ScenarioStrategy on this image, whose
    # client ALWAYS emits a manifest), assert the collected rows EXACTLY cover
    # the dispatched set with schema-complete rows BEFORE writing the result
    # key. A violation raises (fails the unit) — "nonempty" is never enough,
    # and an empty collection against a nonempty dispatched set is a violation.
    # Strategies lacking either attr (legacy/tests/szelag paths) are exempt.
    manifest = getattr(strategy_obj, "_resampling_manifest", None) or {}
    if hasattr(strategy_obj, "_resampling_manifest") and hasattr(
        strategy_obj, "_dispatched_partitions"
    ):
        from flowerfl.resampling_manifest import assert_manifest_complete

        expected = set(getattr(strategy_obj, "_dispatched_partitions") or set())
        # PR #35 P1: also flag discovery-round clients that never resolved to a
        # partition (a discovery failure that would vanish from the evidence).
        # Strategies that expose the two gate attrs but not this method still
        # gate on the original three classes.
        unresolved = None
        _unres_fn = getattr(strategy_obj, "unresolved_dispatched_cids", None)
        if callable(_unres_fn):
            unresolved = _unres_fn()
        assert_manifest_complete(manifest, expected, unresolved_cids=unresolved)
    if manifest:
        result["resampling_manifest"] = [manifest[pid] for pid in sorted(manifest)]

    # H3 (v1.10 § 5.1): the fingerprint registry lives in the SERVER PROCESS
    # ONLY — it is in-memory, bounded, and dies with the simulation. Lift a
    # custody record into the result JSON at run end, the same way the
    # resampling manifest is lifted, so a completed FP unit carries durable
    # evidence of what enrolled, what was flagged, what generation each entry
    # reached, and which tau posture the run executed under. Absent for every
    # non-FP arm (no plugin -> no block).
    fp_custody = _fingerprint_custody(plugins)
    if fp_custody is not None:
        result["fingerprint_registry"] = fp_custody

    # H4 § 6 blackout/capture diagnostics — uniform across all arms whenever
    # the strategy ran the plugin path with a scenario schedule (the chain
    # trace + ground-truth join live in flowerfl/h4_diagnostics.py). A tally
    # whose layer is absent from the arm is null, never zero; nulls are not
    # findings. Absent entirely for non-scenario strategies.
    from flowerfl.h4_diagnostics import build_h4_diagnostics

    h4_block = build_h4_diagnostics(strategy_obj)
    if h4_block is not None:
        result["h4_diagnostics"] = h4_block

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is on sys.path so `flowerfl` package is importable
# regardless of where the script is invoked from.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "20260427"
LOG_DIR = PROJECT_ROOT / "results" / "rmc_flower" / "phase4"

DEV_SEEDS = [42, 137, 256, 314, 500]

# Phase 4 plan: Krum, Krum+CS, TrustScore, TrustScore+CS
CONFIGS = [
    ("Krum", "ScenarioKrum", False),
    ("Krum+CS", "ScenarioKrumCS", True),
    ("TrustScore", "ScenarioTrustScore", False),
    ("TrustScore+CS", "ScenarioTrustScoreCS", True),
]

# Ray backend configuration passed to flwr.simulation.run_simulation().
#
# Caps Ray's actor pool at 8 workers (instead of the implicit nproc default),
# disables the Ray dashboard process (saves ~150MB + a TCP port), and silences
# per-worker driver logging (reduces stdio thrash). This was added in response
# to repeated WSL2 crashes during Flower simulation activity (May 14-19, 2026):
# the host's hyper-v clock-skew + Ray's per-CPU actor spawning + the new AMD
# Zen 5 hybrid CPU's power-state transitions correlated with VM termination
# by the Windows host. See docs/HANDOFF_2026-05-19.md for the full diagnosis.
#
# Federation 'rmc-local' has 11 supernodes; 'rmc-20-local' has 21. Capping
# Ray at 8 actors means actors share supernodes via the scheduler — still
# correct, just more sequential per round.
_RAY_CPUS = int(os.environ.get("PRAXIS_RAY_CPUS", "8"))
_BACKEND_CONFIG: dict = {
    "init_args": {
        # WSL default: 8 (host stability cap). AWS / large hosts: set
        # PRAXIS_RAY_CPUS=24 (or up to vCPU count) to instantiate all 20
        # supernodes per round. See docs/AWS_SETUP.md for context — the WSL
        # cap silently dropped 13-17 of 20 partitions from every round
        # (2026-05-26 finding).
        "num_cpus": _RAY_CPUS,
        # log_to_driver stays False for LOG VOLUME: at 21 supernodes x ~41
        # client constructions/round x N rounds, forwarding every actor's stdout
        # to the driver floods CloudWatch and the captured buffer. Kept False in
        # image v9. CONSEQUENCE (diagnosed 2026-07-27): actor-side print() —
        # including every worker [SMOTE] provenance line — never reaches the
        # driver, so the fleet could not surface a single [SMOTE] record for the
        # entire EXP-018/019/020/021 era. v9 no longer DEPENDS on actor stdout for
        # provenance: the DRIVER-side prewarm (_prewarm_resample_cache) emits the
        # per-client [SMOTE] records from the driver process (whose stdout DOES
        # reach CloudWatch) before run_simulation(), and folds them into the
        # parsed output. Worker [SMOTE] prints remain a local-only backstop.
        "log_to_driver": False,
        "include_dashboard": False,
    },
    "client_resources": {
        "num_cpus": 1,
        "num_gpus": 0.0,
    },
}

# Configs supported by this runner. Adding new configs requires also wiring them
# into build_strategy_for_config() below.
SUPPORTED_CONFIGS = [
    "Krum",
    "Krum+CS",
    "TrustScore",
    "TrustScore+CS",
    "TGE",      # FedAvg + TGEnsemblePlugin (GBDT + LSTM tenure-gated ensemble, ramp_rounds=8 provisional per v1.6)
    "Krum+TGE", # Krum geometric filter + TGEnsemblePlugin (deployed primary defense, spec § 6.1)
    "TGEprime",      # FedAvg + TGEnsemblePlugin with the EMA-reputation long-memory expert (TGE′, GWU-53; PROVISIONAL)
    "Krum+TGEprime", # Krum geometric filter + TGE′ EMA-reputation ensemble (GWU-53; PROVISIONAL)
    "FedMedian",     # coordinate-wise median (utility baseline; H4 runs deferred)
    "FedTrimmedAvg", # trimmed mean beta=0.4 (utility baseline; H4 runs deferred)
    "TGE+FP",        # TGE + FingerprintDefensePlugin — the H3 scored arm (v1.10 § 5.1)
    "Krum+TGE+FP",   # Krum + TGE + FingerprintDefensePlugin — the H4-composable arm
    # ---- H4 composition arms (spec 2026-08-16 § 2 + erratum-A; v1.51/v1.52).
    # Unit-id tokens follow the existing config->filename rule
    # (label.replace('+','_').lower()): h2p_fp_krum / h2p_fp / h2p_krum /
    # h2p_fp_ts / h2p_ts / fedavg — the BUILD_CONTRACT arm tokens.
    "H2P+FP+Krum",   # arm 1 ADJUDICATING treatment: detect -> identity -> Krum
    "H2P+FP",        # arm 3: detect -> identity -> FedAvg
    "H2P+Krum",      # arm 5 attribution ablation: detect -> Krum (no FP)
    "H2P+FP+TS",     # arm 6 generality: detect -> identity -> TrustScore
    "H2P+TS",        # arm 9 (erratum-A E2): detect -> TrustScore (no FP)
    "FedAvg",        # arm 8 undefended floor: scenario scheduling, no defense
]


def build_strategy_for_config(config_name: str):
    """Return (strategy_token, plugin_config_dict) for the given config name.

    strategy_token is a class whose ``__name__`` matches the Flower run-config
    ``strategy`` string (e.g. ``ScenarioKrum``).  The token is passed to
    ``run_one`` which reads ``token.__name__`` to build the ``strategy=``
    run-config fragment.  The plugin_config_dict contains additional key-value
    pairs forwarded verbatim as run-config overrides.

    For TGE the full GBDT + LSTM tenure-gated ensemble is active: the
    provisional canonical tenure gate uses ``ramp_rounds=8`` (amendment v1.6
    § 2 — the value every prior LSTM-active run effectively used), so scoring
    is pure GBDT for client tenure < 2, a linear GBDT->LSTM blend over tenure
    2..8, and pure LSTM for tenure >= 8. The FINAL ramp is selected at the H2
    dev gate by the v1.6 § 3 pre-registered offline protocol
    (scripts/analyze_ramp_selection.py) before threshold freeze. This is spec
    § 6.1 family C (the H2 thesis method). Previously the runner forced
    ``tge_ramp_rounds=999``, silently disabling the LSTM (GBDT-only) — fixed
    2026-06-05 (blocker B2, methodology v1.8) so H2 actually tests the
    tenure-gated ensemble.

    Note: the TGE composition is FedAvg + TGEnsemblePlugin (score-threshold
    filter at 0.7). It does NOT layer KrumDefensePlugin — TGE's ensemble scorer
    is the sole filter. This differs from Krum+CS which composes
    KrumDefensePlugin (distance-based) with ColdStartDefensePlugin.
    """
    # Sentinel classes: __name__ == the Flower strategy string dispatched in
    # server_app.py.  No runtime deps — purely used as routing tokens.
    class ScenarioKrum:  # noqa: N801
        pass

    class ScenarioKrumCS:  # noqa: N801
        pass

    class ScenarioTrustScore:  # noqa: N801
        pass

    class ScenarioTrustScoreCS:  # noqa: N801
        pass

    class ScenarioTGEnsemble:  # noqa: N801
        pass

    class ScenarioKrumTGE:  # noqa: N801
        pass

    class ScenarioTGEPrime:  # noqa: N801
        pass

    class ScenarioKrumTGEPrime:  # noqa: N801
        pass

    class ScenarioFedMedian:  # noqa: N801
        pass

    class ScenarioFedTrimmedAvg:  # noqa: N801
        pass

    class ScenarioTGEFP:  # noqa: N801
        pass

    class ScenarioKrumTGEFP:  # noqa: N801
        pass

    class ScenarioH2PFPKrum:  # noqa: N801
        pass

    class ScenarioH2PFP:  # noqa: N801
        pass

    class ScenarioH2PKrum:  # noqa: N801
        pass

    class ScenarioH2PFPTS:  # noqa: N801
        pass

    class ScenarioH2PTS:  # noqa: N801
        pass

    class ScenarioNone:  # noqa: N801
        pass

    if config_name == "Krum":
        return (ScenarioKrum, {})
    if config_name == "Krum+CS":
        return (ScenarioKrumCS, {"cs_enabled": True})
    if config_name == "TrustScore":
        return (ScenarioTrustScore, {})
    if config_name == "TrustScore+CS":
        return (ScenarioTrustScoreCS, {"cs_enabled": True})
    if config_name == "TGE":
        return (ScenarioTGEnsemble, {
            "cs_enabled": True,
            # GATE-SELECTED (v1.6 § 3 frozen rule, executed 2026-07-23 dev gate):
            # r*=3 — results/20260723/ramp_selection (commit 61ef015); monotone,
            # +14.4pp primary vs incumbent 8, only candidate w/ nonzero cold-start
            # recall. § 4 contingency re-run = EXP-014. Never hand-edited.
            "tge_ramp_rounds": 3,
        })
    if config_name == "Krum+TGE":
        return (ScenarioKrumTGE, {
            "cs_enabled": False,
            # GATE-SELECTED (v1.6 § 3 frozen rule, executed 2026-07-23 dev gate):
            # r*=3 — results/20260723/ramp_selection (commit 61ef015); monotone,
            # +14.4pp primary vs incumbent 8, only candidate w/ nonzero cold-start
            # recall. § 4 contingency re-run = EXP-014. Never hand-edited.
            "tge_ramp_rounds": 3,
        })
    if config_name == "TGEprime":
        return (ScenarioTGEPrime, {
            "cs_enabled": True,
            # TGE′ (GWU-53): the incumbent LSTM is kept and a SECOND long-memory
            # leg — an EMA reputation over the cold-start score — is added
            # alongside it; the gate's long-memory leg = min(LSTM, EMA) (the
            # "bank" combiner). PROVISIONAL pending amendment v1.7 ratification.
            # ema_alpha=0.9 is ADOPTED from TrustScore's validated constant.
            # Ramp = 3 INHERITED (amendment v1.7): the bank RETAINS the LSTM leg
            # r*=3 was gate-selected on (2026-07-23, 61ef015), so holding it
            # keeps EXP-015 a single-variable A/B (added EMA leg + min combiner).
            "tge_long_memory_expert": "bank",
            "tge_ema_alpha": 0.9,
            "tge_ramp_rounds": 3,
        })
    if config_name == "Krum+TGEprime":
        return (ScenarioKrumTGEPrime, {
            "cs_enabled": False,
            # TGE′ (GWU-53) — PROVISIONAL; ramp 3 INHERITED; see TGEprime above.
            "tge_long_memory_expert": "bank",
            "tge_ema_alpha": 0.9,
            "tge_ramp_rounds": 3,
        })
    if config_name == "FedMedian":
        return (ScenarioFedMedian, {})
    if config_name == "FedTrimmedAvg":
        return (ScenarioFedTrimmedAvg, {})
    # ---- H3 fingerprint arms (dry-run F2 gap 2) ---------------------------
    # Each FP arm is its incumbent arm PLUS the fingerprint plugin and nothing
    # else: the same gate-selected ramp, the same cs_enabled sentinel. That is
    # what keeps `TGE` vs `TGE+FP` a single-variable comparison.
    #
    # `fingerprint-enabled` is already the hyphenated Flower run-config key
    # `flowerfl/client_app.py` reads (`_as_bool(run_config.get(
    # "fingerprint-enabled", False))`), so it is forwarded VERBATIM by main()'s
    # extras pass-through — it is deliberately NOT an underscore key routed
    # through `_hyphenate_tge_extra`, which is the TGE knob map. Emitting it
    # here, keyed off the config label, is what makes emission reachable ONLY
    # from a `+FP` label: no CLI flag, no scenario, no manifest `run_extras`
    # path turns it on for an incumbent arm (EMISSION_CONTRACT § 4.3 — the
    # fingerprint stream must stay inert for every sealed H1/H2 token).
    if config_name == "TGE+FP":
        return (ScenarioTGEFP, {
            "cs_enabled": True,          # inert for this branch; mirrors "TGE"
            "tge_ramp_rounds": 3,        # INHERITED from "TGE" (v1.6 § 3 gate-selected)
            "fingerprint-enabled": True,
        })
    if config_name == "Krum+TGE+FP":
        return (ScenarioKrumTGEFP, {
            "cs_enabled": False,         # mirrors "Krum+TGE"
            "tge_ramp_rounds": 3,        # INHERITED from "Krum+TGE"
            "fingerprint-enabled": True,
        })
    # ---- H4 composition arms (spec 2026-08-16 § 2 + erratum-A) -----------
    # These tokens are H4-ONLY, so the erratum-A frozen operating conditions
    # are BAKED into the config rather than left to CLI discipline:
    #   * eval-split=sealed_test — the E4 sealed-test evaluation population;
    #   * FP-bearing arms declare the § 5 frozen identity configuration
    #     explicitly (fp-cohort=validation -> TAU_VALIDATION_ALL_DEVICES via
    #     the locked-cohort mechanism, never a hardcoded float;
    #     fp-registry-policy=flag_gated — declared, not inherited, so the
    #     custody field is populated) plus client fingerprint emission.
    # Reused arms (Krum / TrustScore / Krum+TGE+FP / FedAvg) get
    # eval-split=sealed_test from the launch tooling's --eval-split flag —
    # their tokens predate H4 and stay byte-unchanged by default.
    _h4_fp_extras = {
        "fingerprint-enabled": True,
        "fp-cohort": "validation",
        "fp-registry-policy": "flag_gated",
    }
    if config_name == "H2P+FP+Krum":
        return (ScenarioH2PFPKrum, {
            "cs_enabled": False,
            "eval-split": "sealed_test",
            **_h4_fp_extras,
        })
    if config_name == "H2P+FP":
        return (ScenarioH2PFP, {
            "cs_enabled": False,
            "eval-split": "sealed_test",
            **_h4_fp_extras,
        })
    if config_name == "H2P+Krum":
        return (ScenarioH2PKrum, {
            "cs_enabled": False,
            "eval-split": "sealed_test",
        })
    if config_name == "H2P+FP+TS":
        return (ScenarioH2PFPTS, {
            "cs_enabled": False,
            "eval-split": "sealed_test",
            **_h4_fp_extras,
        })
    if config_name == "H2P+TS":
        return (ScenarioH2PTS, {
            "cs_enabled": False,
            "eval-split": "sealed_test",
        })
    if config_name == "FedAvg":
        # Arm 8 undefended floor: ScenarioNone = FedAvg + scenario scheduling,
        # no defense plugin. eval-split comes from the launch tooling so the
        # token stays usable outside H4 with the legacy evaluator.
        return (ScenarioNone, {"cs_enabled": False})
    raise ValueError(f"Unknown config: {config_name!r}. Supported: {SUPPORTED_CONFIGS}")


def _hyphenate_tge_extra(extra: dict) -> dict:
    """Map build_strategy_for_config's underscore TGE keys to the hyphenated
    Flower run-config keys server_app.py reads. Centralized so every TGE/TGE′
    knob (ramp, long-memory expert, EMA alpha) is translated consistently.
    """
    key_map = {
        "tge_ramp_rounds": "tge-ramp-rounds",
        "tge_long_memory_expert": "tge-long-memory-expert",
        "tge_ema_alpha": "tge-ema-alpha",
    }
    out = dict(extra)
    for underscore, hyphen in key_map.items():
        if underscore in out:
            out[hyphen] = out.pop(underscore)
    return out


def tge_provenance_fields(strategy: str, run_config: dict, rounds: int) -> dict:
    """True TGE provenance for result JSON (amendment v1.6 § 2).

    Replaces the Step-2-era hardcoded ``tge_lstm_state: "disabled"`` — flagged
    launch-blocking in docs/chapter3/TGE_METHOD_IDENTITY_MATRIX.md because it
    wrote false provenance into every result once the LSTM was re-enabled
    (methodology v1.8).

    Under the continuous tenure gate the LSTM contributes non-zero blend
    weight for any tenure > min_tenure once it has fitted (end of round
    warmup+2 — TGEnsembleModel.on_round_end), REGARDLESS of the ramp: even
    the legacy ramp=999 keeps ~5% LSTM weight at high tenure. So
    `tge_lstm_state` reports whether the LSTM can contribute AT ALL (run long
    enough for it to fit), and `tge_pure_lstm_reach` separately reports
    whether any client can reach the pure-LSTM regime (ramp < rounds). The
    effective ramp defaults to the provisional canonical 8 when the
    run-config key is absent, mirroring server_app.py's default. Gate
    settings not exposed via run-config (min_tenure, operational threshold,
    warmup) are read from the plugin/rule signatures so provenance cannot
    drift from the code (GWU-8 acceptance criteria). TGE has no serialized
    model artifact — both experts fit online within the run — so there is no
    artifact reference.
    """
    if "TGE" not in strategy:
        return {"tge_lstm_state": "n/a", "tge_ramp_rounds": None}
    import inspect
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    from rmc.tg_ensemble import TenureGatedDecisionRule
    ramp = int(run_config.get("tge-ramp-rounds", 8))
    # Long-memory expert identity (GWU-53). Defaults to the incumbent "lstm"
    # when the run-config key is absent, mirroring server_app.py / the plugin
    # default, so every pre-TGE′ result records "lstm" with a null alpha.
    # Default matches the server's construction default for the strategy: prime
    # tokens deploy "bank", incumbent TGE tokens "lstm" (GWU-53 — server
    # and provenance must agree even when the run-config key is absent).
    _default_expert = "bank" if "Prime" in strategy else "lstm"
    long_memory_expert = str(run_config.get("tge-long-memory-expert", _default_expert))
    # ema_alpha applies to the ema/bank legs; None for the incumbent lstm. When
    # the run-config key is absent the server constructs the EMA with the
    # plugin default, so provenance must record that same default, not None
    # (GWU-53). Read the default from the
    # plugin signature so it can never drift from the deployed value.
    _default_ema_alpha = inspect.signature(
        TGEnsemblePlugin.__init__).parameters["ema_alpha"].default
    ema_alpha = (
        float(run_config.get("tge-ema-alpha", _default_ema_alpha))
        if long_memory_expert in ("ema", "bank")
        else None
    )
    # Combiner identity for the gate's long-memory leg: "min" in bank mode,
    # else the single leg's name (GWU-53).
    long_memory_combiner = "min" if long_memory_expert == "bank" else long_memory_expert
    warmup = inspect.signature(
        TGEnsemblePlugin.__init__).parameters["warmup_rounds"].default
    # Round accounting: the runner adds a discovery round
    # (num-server-rounds = rounds + 1, see _build_run_config), and plugins
    # first score at server round 2 — so the run has server rounds 1..rounds+1
    # and a continuously-present client's max tenure is `rounds`. The LSTM
    # fits at the end of server round warmup+2 (TGEnsembleModel.on_round_end)
    # and first scores at server round warmup+3, which exists iff
    # rounds + 1 >= warmup + 3. The pure-LSTM gate is INCLUSIVE
    # (tenure >= ramp), so the regime is reachable iff ramp <= rounds.
    lstm_first_scoring_round = warmup + 3
    # Gate the legacy LSTM provenance on the EFFECTIVE mode (GWU-53,
    # run_phase4_flower.py:391). In "ema" component-isolation the LSTM is built
    # but never feeds the gate (_resolve_long_memory ignores it), so reporting
    # it enabled/reachable is internally contradictory. lstm/bank both use the
    # LSTM leg, so they keep the run-length/ramp-derived state.
    if long_memory_expert == "ema":
        tge_lstm_state = "disabled"
        tge_pure_lstm_reach = False
    else:
        tge_lstm_state = "enabled" if rounds + 1 >= lstm_first_scoring_round else "disabled"
        tge_pure_lstm_reach = ramp <= rounds
    return {
        "tge_lstm_state": tge_lstm_state,
        "tge_pure_lstm_reach": tge_pure_lstm_reach,
        "tge_ramp_rounds": ramp,
        # Long-memory bank identity (GWU-53). TGE′ keeps the LSTM and adds an
        # EMA reputation leg; tge_long_memory_expert is lstm | ema | bank, and
        # tge_long_memory_combiner is the combiner the gate's long-memory leg
        # uses ("min" in bank mode). tge_ema_alpha is the EMA retention weight
        # (None for the incumbent LSTM). The signal log records both raw leg
        # scores (tge_lstm_score, tge_ema_score) so the combiner is
        # reconstructable offline.
        "tge_long_memory_expert": long_memory_expert,
        "tge_long_memory_combiner": long_memory_combiner,
        "tge_ema_alpha": ema_alpha,
        # isolation forest per amendment v1.5 § 4 (class name GBDTColdStartExpert is legacy)
        "tge_cold_start_expert": "isolation_forest",
        "tge_min_tenure": inspect.signature(
            TenureGatedDecisionRule.__init__).parameters["min_tenure"].default,
        "tge_operational_threshold": inspect.signature(
            TGEnsemblePlugin.__init__).parameters["threshold"].default,
    }


# One structured, loud, parseable per-client record line emitted by
# flowerfl/task.py::load_data at data prep (DESIGN.md Stage-D provenance list):
#   [SMOTE] [WARNING ]client=<i> status=applied|skipped reason=<r> variant=<v>
#           target=<t> k=<int> n_before=<int> n_after=<int> synthetic=<int>
#           seed_component=<int>
# The runner lifts these from captured stdout into per-client provenance counts
# + the run-level skip flag + the reproducibility seed component, mirroring
# parse_alie_round_set's stdout-parse pattern.
_SMOTE_RECORD_RE = re.compile(r"^\[SMOTE\] .*$", re.M)
_SMOTE_KV_RE = re.compile(r"(\w+)=(\S+)")
_SMOTE_INT_FIELDS = frozenset(
    # "removed" (v8 change 2): rows dropped by the under-sampler; "cache_hits"
    # (v8 change 1): running process resample-cache hit count. Both are optional
    # trailing key=value tokens — absent tokens simply don't appear in the dict,
    # so older records without them parse unchanged.
    {"client", "k", "n_before", "n_after", "synthetic", "removed",
     "cache_hits", "seed_component"}
)


def parse_smote_records(full_output: str) -> list:
    """Parse the per-client ``[SMOTE] ...`` records from captured run stdout.

    Returns a list of dicts (one per client that ran the SMOTE path), with the
    numeric fields coerced to int. Empty list when SMOTE was off or emitted no
    records. Non-key=value tokens on the line (e.g. the ``WARNING`` loudness
    marker on skips) are ignored.
    """
    records = []
    for line in _SMOTE_RECORD_RE.findall(full_output or ""):
        rec = {}
        for key, val in _SMOTE_KV_RE.findall(line):
            if key in _SMOTE_INT_FIELDS:
                try:
                    rec[key] = int(val)
                except ValueError:
                    rec[key] = val
            else:
                rec[key] = val
        if rec:
            records.append(rec)
    return records


def _smote_dedupe_key(rec: dict) -> tuple:
    """Exact-emission identity of a [SMOTE] record: EVERY field except the
    ``cache_hits`` observability token.

    keying on only (client, status, variant, target,
    seed_component) collapsed two records for one client that share config but
    have DIFFERENT OUTCOMES (reason, n_before, n_after, synthetic, removed, k) —
    hiding the very anomaly smote_record_conflicts exists to surface. Keying on
    all fields means only BYTE-IDENTICAL emissions collapse (an LRU re-emit /
    second Ray worker re-running the SAME deterministic resample); any outcome
    difference survives dedup and lands in the per-client conflict grouping.

    ``cache_hits`` is excluded because it is a running per-process counter that
    legitimately differs between otherwise-identical re-emissions — including it
    would defeat the dedup (P2-1) and manufacture false conflicts."""
    return tuple(sorted((k, v) for k, v in rec.items() if k != "cache_hits"))


def _smote_run_summary(records: list) -> dict:
    """Aggregate per-client SMOTE records into run-level counts + skip flag.

    counts are over DISTINCT logical clients, not raw
    lines. load_data emits one [SMOTE] line per real resample, but LRU eviction
    (a worker touching > cache-size partitions) or multiple Ray workers can
    re-emit a line for the SAME logical client, so counting lines could push
    smote_applied_count above the client count — false provenance. Records are
    deduped by _smote_dedupe_key first. A SAME client appearing with CONFLICTING
    status/config (distinct deduped keys for one client) is a real anomaly:
    it is surfaced in ``smote_record_conflicts`` (nonzero = investigate) and
    EXCLUDED from applied/skipped rather than silently collapsed."""
    # dedupe exact re-emissions
    seen: set = set()
    deduped: list = []
    for r in records:
        key = _smote_dedupe_key(r)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # group distinct logical records by client to detect conflicts
    by_client: dict = {}
    for r in deduped:
        by_client.setdefault(r.get("client"), []).append(r)

    applied_count = 0
    skipped_count = 0
    conflicts = 0
    reasons: dict = {}
    for _client, recs in by_client.items():
        if len(recs) > 1:
            # same client, conflicting status/config — anomaly, not a count
            conflicts += 1
            continue
        r = recs[0]
        if r.get("status") == "applied":
            applied_count += 1
        elif r.get("status") == "skipped":
            skipped_count += 1
            reason = r.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
    skip_flag = ",".join(sorted(reasons)) if reasons else None
    return {
        "smote_applied_count": applied_count,
        "smote_skipped_count": skipped_count,
        "smote_skip_reasons": reasons,
        "smote_skipped_reason": skip_flag,
        "smote_record_conflicts": conflicts,
    }


def smote_provenance_fields(run_config: dict, records: "list | None" = None) -> dict:
    """SMOTE study-knob provenance for the result JSON (GWU-59).

    Every run declares its SMOTE status: ``smote_enabled`` is always present so
    a reader never has to distinguish "absent" from "off". When enabled, the
    normalized variant/target are echoed and validated LOUDLY (an unknown
    variant / invalid target raises here, matching client-side parse-time
    validation, so a mis-launched run fails before it writes false provenance),
    and the per-client ``records`` are aggregated into applied/skipped counts, a
    per-reason breakdown, and the run-level ``smote_skipped_reason`` flag
    (DESIGN.md §6c). ``smote_seed_component`` records the run base seed from
    which each client's SMOTE seed is derived as derive_seed(base, partition_id)
    — the reproducibility-audit anchor. OFF == incumbent: the minimal declaration
    only, no per-client fields.
    """
    from flowerfl.smote_resampler import validate_smote_variant, normalize_smote_target

    enabled = run_config.get("smote-enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
    enabled = bool(enabled)
    if not enabled:
        return {
            "smote_enabled": False,
            "smote_variant": None,
            "smote_target": None,
            "smote_skipped_reason": None,
        }

    variant = validate_smote_variant(str(run_config.get("smote-variant", "smote")))
    target = normalize_smote_target(run_config.get("smote-target", "balanced"))
    summary = _smote_run_summary(records or [])
    return {
        "smote_enabled": True,
        "smote_variant": variant,
        "smote_target": target,
        "smote_skipped_reason": summary["smote_skipped_reason"],
        "smote_applied_count": summary["smote_applied_count"],
        "smote_skipped_count": summary["smote_skipped_count"],
        "smote_skip_reasons": summary["smote_skip_reasons"],
        # P2-1: distinct clients emitting CONFLICTING [SMOTE] records; nonzero =
        # investigate (never silently collapsed into the applied/skipped counts).
        "smote_record_conflicts": summary["smote_record_conflicts"],
        # run base seed; per-client seed = derive_seed(this, partition_id)
        "smote_seed_component": int(run_config.get("seed", 42)),
    }


def _smote_mlflow_params(run_config: dict, records: "list | None" = None) -> dict:
    """MLflow params for a SMOTE-enabled run (empty dict when off).

    Kept as string values (MLflow params are strings) and disjoint from the
    result-JSON provenance so the two surfaces agree without coupling. The skip
    reason is logged only when at least one client skipped.
    """
    fields = smote_provenance_fields(run_config, records=records)
    if not fields["smote_enabled"]:
        return {}
    params = {
        "smote_enabled": str(fields["smote_enabled"]),
        "smote_variant": str(fields["smote_variant"]),
        "smote_target": str(fields["smote_target"]),
        "smote_applied_count": str(fields["smote_applied_count"]),
        "smote_skipped_count": str(fields["smote_skipped_count"]),
        "smote_seed_component": str(fields["smote_seed_component"]),
    }
    if fields["smote_skipped_reason"] is not None:
        params["smote_skipped_reason"] = str(fields["smote_skipped_reason"])
    # P2-1: only surface the conflict param when nonzero (absent = clean run).
    if fields.get("smote_record_conflicts"):
        params["smote_record_conflicts"] = str(fields["smote_record_conflicts"])
    return params


def _smote_enabled_in_run_config(run_config: dict) -> bool:
    """Coerce the run-config smote-enabled value (bool or string) to a bool."""
    enabled = run_config.get("smote-enabled", False)
    if isinstance(enabled, str):
        return enabled.strip().lower() in ("1", "true", "yes", "on")
    return bool(enabled)


def _prewarm_resample_cache(run_config: dict) -> "tuple[str, dict]":
    """Driver-side prewarm of the node-local resample DISK cache (image v9).

    Runs BEFORE run_simulation(). When SMOTE is enabled, iterate every client
    partition ONCE in the driver process, populating the container-local disk
    cache (flowerfl/task.py) so every Ray worker's load_data() becomes a 100%
    disk read (0% resample compute). This is the structural fix for the v8
    failure: Flower 1.29's ActorPool has no client->actor affinity, so a
    per-process in-RAM LRU missed ~84% of the time at fleet shape (32 actors >
    21 clients); a node-local disk cache is visible to every actor regardless of
    which one the scheduler hands a client.

    It ALSO closes the provenance-emission gap. run_phase4_flower sets
    ``log_to_driver=False`` (log-volume control), so a worker's [SMOTE] print()
    never reaches the driver stdout / CloudWatch — the entire EXP-018/019/020/021
    fleet era could not surface a single [SMOTE] line. The prewarm emits the
    per-client records from the DRIVER process (whose stdout DOES reach
    CloudWatch) and RETURNS them so the captured-stdout parser
    (parse_smote_records / _smote_run_summary) sees exactly one record per client.

    The driver must reconstruct load_data's call EXACTLY as client_app.py does —
    same dataset, batch size, per-run MAX_SAMPLES_PER_CLIENT cap (which enters the
    cache key), and per-partition smote_seed=derive_seed(base_seed, partition_id)
    — or the prewarmed keys will not match the workers' lookups.

    Returns (smote_record_text, summary_dict). No-op on a SMOTE-off run.
    """
    if not _smote_enabled_in_run_config(run_config):
        return "", {"smote_enabled": False, "prewarmed_clients": 0}

    from flowerfl import task as task_module
    from flowerfl.task import load_data
    from flowerfl.seeding import derive_seed
    from flowerfl.smote_resampler import validate_smote_variant, normalize_smote_target

    dataset = run_config.get("dataset", DATASET)
    batch_size = int(run_config.get("batch-size", 32))
    max_samples = int(run_config.get("max-samples", 0))
    base_seed = int(run_config.get("seed", 42))
    variant = validate_smote_variant(str(run_config.get("smote-variant", "smote")))
    target = normalize_smote_target(run_config.get("smote-target", "balanced"))
    # Stage-F §6: the semantic-target flag is part of the resample cache KEY, so
    # the prewarm MUST warm the SAME key the workers look up — omitting it here
    # warmed the legacy `False` key, forcing every worker to miss + recompute the
    # big resample AND writing driver-side [SMOTE] provenance for the LEGACY
    # arrays, not the semantic ones trained on.
    semantic = _stage_f_bool(run_config.get("smote-semantic-target", False))
    # m1 leakage fix: normalize-train-only is ALSO part of the resample cache KEY
    # (append-only-when-True), so the prewarm MUST warm and validate the SAME key
    # the workers look up. Omitting it here would prime + report durability for
    # the LEAK-ON key while every leak-free worker misses and concurrently
    # recomputes the big train-only resample (EXP-020-class memory failure; same
    # family as the semantic-target prewarm bug above).
    normalize_train_only = _stage_f_bool(run_config.get("normalize-train-only", False))

    # Mirror client_app.py: the per-run cap is set on the task-module global and
    # enters the resample cache KEY. Set the SAME value workers will, or the
    # prewarmed keys won't match.
    if max_samples > 0:
        task_module.MAX_SAMPLES_PER_CLIENT = max_samples

    cache_dir = task_module._resample_cache_dir()  # None if disk layer unavailable
    evictions_before = task_module._resample_disk_evictions
    persisted_before = task_module._resample_persisted_count
    compute_only_before = task_module._resample_compute_only_count
    stale_temps_before = task_module._resample_stale_temps_cleaned
    # ONCE, before the loop, enforce the CURRENT budget on the
    # existing dir contents — a reused/shared cache dir whose budget was lowered
    # could otherwise stay oversized on an all-hit workload (the store-time fast
    # path returns before any budget check). Evictions are counted in the block.
    task_module.enforce_resample_disk_budget()
    t0 = time.time()
    buf = io.StringIO()
    # Tee load_data's stdout: capture the [SMOTE] records AND the [RESAMPLE-CACHE]
    # warnings here so we can re-emit BOTH to the real driver stdout (CloudWatch),
    # dropping only the [Dataset] noise. The [SMOTE] records feed the provenance
    # parser; the [RESAMPLE-CACHE] warnings must NOT be silently
    # dropped or a degraded prime (failed stores) looks healthy in CloudWatch.
    with contextlib.redirect_stdout(buf):
        for partition_id in range(NUM_SUPERNODES):
            load_data(
                partition_id=partition_id,
                dataset_name=dataset,
                batch_size=batch_size,
                smote_enabled=True,
                smote_variant=variant,
                smote_target=target,
                smote_seed=derive_seed(base_seed, partition_id),
                smote_semantic_target=semantic,  # Stage-F §6: match worker key
                normalize_train_only=normalize_train_only,  # m1: match worker key
                # P1-1: emit the provenance record on EVERY partition, hit OR
                # miss, so a prewarm against a pre-populated cache (later config
                # in a multi-config run) still reports every client — never
                # applied_count=0 while training on resampled data.
                smote_record_always=True,
            )
    captured = buf.getvalue().splitlines()
    smote_lines = [ln for ln in captured if ln.startswith("[SMOTE]")]
    warn_lines = [ln for ln in captured if ln.startswith("[RESAMPLE-CACHE]")]
    smote_text = "\n".join(smote_lines)

    total_bytes = 0
    if cache_dir is not None:
        try:
            for name in os.listdir(cache_dir):
                if name.endswith(".npz"):
                    total_bytes += os.path.getsize(os.path.join(cache_dir, name))
        except OSError:
            pass
    wall = time.time() - t0
    evictions = task_module._resample_disk_evictions - evictions_before
    # Raw per-store success delta (diagnostic). NOT the durable count: the
    # counter is cumulative and never decremented, so under a budget smaller than
    # the working set a later store that EVICTS an earlier same-loop entry still
    # shows here as a success.
    stores_succeeded = task_module._resample_persisted_count - persisted_before
    stores_compute_only = task_module._resample_compute_only_count - compute_only_before
    stale_temps_cleaned = task_module._resample_stale_temps_cleaned - stale_temps_before
    budget = task_module._resample_cache_max_bytes()

    # FINAL DURABILITY PASS: count which partitions' expected
    # cache entries are actually LOADABLE on disk NOW — this is the truth a Ray
    # worker will see. Validate STRUCTURE (r9), not mere existence, so a corrupt
    # entry whose repair could not be stored is not miscounted as durable.
    # persisted = durable; compute_only = the rest (workers recompute them).
    durable = 0
    for partition_id in range(NUM_SUPERNODES):
        cpath = task_module.resample_cache_path(
            dataset_name=dataset, partition_id=partition_id, batch_size=batch_size,
            train_split=0.8, val_split=0.1, smote_variant=variant,
            smote_target=target, smote_seed=derive_seed(base_seed, partition_id),
            smote_semantic_target=semantic,  # Stage-F §6: durability pass on the SAME key
            normalize_train_only=normalize_train_only,  # m1: durability pass on the SAME key
        )
        if cpath is not None and task_module._resample_disk_valid(cpath):
            durable += 1
            # Refresh recency on the durability-validated hit: this
            # partition IS the current run's working set, so bump its mtime to
            # protect it from oldest-mtime eviction by later worker stores in a
            # shared/reused dir holding newer foreign-seed artifacts.
            task_module._touch_resample_entry(cpath)
    persisted = durable
    compute_only = NUM_SUPERNODES - durable

    # Emit provenance + cache warnings to the REAL driver stdout (reaches
    # CloudWatch under log_to_driver=False) + one summary line. persisted vs
    # compute_only makes a degraded prime unmistakable: compute_only>0 means those
    # clients are NOT on disk and every worker will recompute them.
    if smote_text:
        print(smote_text, flush=True)
    for wln in warn_lines:
        print(wln, flush=True)
    print(
        f"[PREWARM] smote resample cache primed: clients={len(smote_lines)} "
        f"persisted={persisted} compute_only={compute_only} "
        f"stores_succeeded={stores_succeeded} "
        f"variant={variant} target={target} dir={cache_dir} "
        f"bytes={total_bytes} budget_bytes={budget} evictions={evictions} "
        f"stale_temps_cleaned={stale_temps_cleaned} wall_s={wall:.1f}",
        flush=True,
    )

    # release the driver's in-process L1 RAM cache (the last
    # ~4 resampled partitions, up to ~4.7 GiB worst case) now that the durability
    # pass and provenance snapshot above are done — it must not sit resident for
    # the whole simulation, competing with Ray actors for the memory the prewarm
    # exists to free. Clears ONLY L1: the disk entries (which workers read) and
    # the counters stay intact.
    task_module._clear_resample_l1()

    return smote_text, {
        "smote_enabled": True,
        "prewarmed_clients": len(smote_lines),
        "persisted": persisted,
        "compute_only": compute_only,
        "stores_succeeded": stores_succeeded,
        "stores_compute_only": stores_compute_only,
        "cache_dir": cache_dir,
        "cache_bytes": total_bytes,
        "cache_budget_bytes": budget,
        "evictions": evictions,
        "stale_temps_cleaned": stale_temps_cleaned,
        "wall_s": wall,
    }


def _resample_prewarm_provenance(prewarm_summary: dict) -> "dict | None":
    """The resample_prewarm block persisted into the result JSON.

    None for a SMOTE-off run (no prewarm ran). Otherwise the durability-critical
    health of the prime — so a degraded run (compute_only>0) is distinguishable
    from a fully-primed one in the on-disk result, not just ephemeral console
    output.
    """
    if not prewarm_summary.get("smote_enabled"):
        return None
    return {
        # final-durability numbers (what workers actually see on disk)
        "persisted": prewarm_summary.get("persisted"),
        "compute_only": prewarm_summary.get("compute_only"),
        # raw per-store success (diagnostic; may exceed persisted when a later
        # store evicted an earlier same-loop entry under a tight budget)
        "stores_succeeded": prewarm_summary.get("stores_succeeded"),
        "evictions": prewarm_summary.get("evictions"),
        "stale_temps_cleaned": prewarm_summary.get("stale_temps_cleaned"),
        "cache_bytes": prewarm_summary.get("cache_bytes"),
        "budget_bytes": prewarm_summary.get("cache_budget_bytes"),
        "wall_clock_s": prewarm_summary.get("wall_s"),
    }


def add_smote_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the SMOTE study flags (GWU-59). Defaults preserve OFF exactly so
    a bare invocation is byte-identical to the incumbent. docker/entrypoint.py
    appends these from the manifest run_extras so the fleet path can reach the
    knob (it lived only on the pyproject/flwr-run path before)."""
    parser.add_argument("--smote-enabled", action="store_true",
                        help="Enable per-client training-split oversampling (GWU-59). "
                             "Default OFF; experiment-relevant, so inert unless set.")
    parser.add_argument("--smote-variant", default="smote",
                        help="Resampler: 'smote' (interpolation) | 'random_over' "
                             "(duplication) | 'random_under' (majority downsample, v8).")
    parser.add_argument("--smote-target", default="balanced",
                        help="'balanced' (50/50) or a minority-ratio float in (0, 1].")


def smote_run_config_from_cli(enabled: bool, variant: str, target: str) -> dict:
    """Map the SMOTE CLI flags to run-config overrides merged into extra_run_config.

    Returns {} when disabled — the merge is then a no-op and the run-config
    (hence provenance) is byte-identical to the incumbent. When enabled, emits
    the hyphenated Flower run-config keys client_fn reads; value validation is
    handled loudly downstream (smote_provenance_fields + client_fn)."""
    if not enabled:
        return {}
    return {
        "smote-enabled": True,
        "smote-variant": variant,
        "smote-target": target,
    }


def add_stage_f_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the Stage-F study flags (DESIGN_STAGE_F §4/§5/§6). Defaults
    preserve the incumbent EXACTLY so a bare invocation is byte-identical: update
    matching OFF (epochs loop), weight-mode resampled (post-resample count to
    FedAvg), semantic-target OFF (legacy label-agnostic min/max)."""
    parser.add_argument("--update-match", action="store_true",
                        help="Cap every arm/path to the off-arm's per-client "
                             "optimizer-step budget (Stage-F §4). Default OFF.")
    parser.add_argument("--weight-mode", choices=["resampled", "original"],
                        default="resampled",
                        help="FedAvg num_examples: 'resampled' (incumbent) or "
                             "'original' (pre-resample n_orig, Stage-F §5).")
    parser.add_argument("--smote-semantic-target", action="store_true",
                        help="Grow/shrink by attack-class (label 1) identity per "
                             "Stage-F §6 instead of the legacy min/max. Default OFF.")


def stage_f_run_config_from_cli(update_match: bool, weight_mode: str,
                                semantic_target: bool) -> dict:
    """Map the Stage-F CLI flags to run-config overrides (empty at defaults, so
    the merge is a no-op and provenance stays byte-identical to the incumbent)."""
    extra: dict = {}
    if update_match:
        extra["update-match"] = True
    if weight_mode and weight_mode != "resampled":
        extra["weight-mode"] = weight_mode
    if semantic_target:
        extra["smote-semantic-target"] = True
    return extra


def _stage_f_bool(value) -> bool:
    """Coerce a run-config Stage-F flag (native bool or string) to a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def add_leakage_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the m1 normalization-leak fix flag (normalization audit).

    Default preserves the incumbent EXACTLY: OFF means load_data fits the per-
    client Z-score on ALL rows (train+val+test) before the split — the sealed
    leak path. ``--normalize-train-only`` fits it on the seed-42 training rows
    only, so a bare invocation stays byte-identical to every pre-registered run.
    """
    parser.add_argument("--normalize-train-only", action="store_true",
                        help="Fit the per-client Z-score on the training split "
                             "only (m1 leakage fix). Default OFF (incumbent all-"
                             "rows fit); pre-registration-relevant, so inert "
                             "unless set.")


def leakage_run_config_from_cli(normalize_train_only: bool) -> dict:
    """Map the m1 leakage-fix CLI flag to run-config overrides.

    Returns {} when disabled — the merge is then a no-op and the run-config
    (hence provenance and every downstream artifact) is byte-identical to the
    incumbent. When enabled, emits the hyphenated key client_app reads."""
    if not normalize_train_only:
        return {}
    return {"normalize-train-only": True}


def add_fp_cohort_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the H3 post-τ-lock cohort declaration (v1.10 § 5.1 D9).

    Default None preserves the incumbent EXACTLY: no run-config key is emitted,
    so pre-lock runs and non-FP arms are byte-identical to every pre-registered
    run. Post-lock, an FP-arm run WITHOUT this flag is hard-refused by
    server_app.build_fingerprint_registry — the validation and adjudicating
    τ/Σ pairs are different instruments, and which one scored a unit must be a
    declared fact of the run, never a default.
    """
    from flowerfl.fingerprint_registry import CalibrationCohort
    parser.add_argument(
        "--fp-cohort",
        choices=[c.value for c in CalibrationCohort],
        default=None,
        help="Which LOCKED fingerprint tau/Sigma cohort scores this run "
             "(post-tau-lock FP arms MUST declare one; server refuses otherwise).",
    )


def fp_cohort_run_config_from_cli(fp_cohort: "str | None") -> dict:
    """Map --fp-cohort to run-config overrides.

    Returns {} when undeclared — the merge is then a no-op and the run-config
    (hence provenance and every downstream artifact) is byte-identical to the
    incumbent. When declared, emits the hyphenated key server_app reads."""
    if not fp_cohort:
        return {}
    return {"fp-cohort": fp_cohort}


def add_fp_registry_policy_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the H3 registry candidate-policy declaration (v1.49 § 1).

    Default None preserves the incumbent EXACTLY: no run-config key is emitted,
    so every pre-existing run is byte-identical and server_app builds the
    deployed flag-gated registry. `identity_only` selects the corrected
    enroll-everyone instrument the 2026-08-14 amendment makes THE H3 test;
    `flag_gated` is expressible explicitly so an H4 composition arm can DECLARE
    the incumbent rather than inherit it silently.
    """
    from flowerfl.fingerprint_registry import RegistryPolicy
    parser.add_argument(
        "--fp-registry-policy",
        choices=[p.value for p in RegistryPolicy],
        default=None,
        help="Which prior sessions a re-entry decision considers: "
             "flag_gated (deployed incumbent, default) or identity_only "
             "(corrected H3 instrument: enroll everyone, no flag gating).",
    )


def fp_registry_policy_run_config_from_cli(fp_registry_policy: "str | None") -> dict:
    """Map --fp-registry-policy to run-config overrides.

    Returns {} when undeclared — the merge is then a no-op and the run-config
    (hence provenance and every downstream artifact) is byte-identical to the
    incumbent. When declared, emits the hyphenated key server_app reads."""
    if not fp_registry_policy:
        return {}
    return {"fp-registry-policy": fp_registry_policy}


def add_eval_split_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the H4 sealed-test evaluator declaration (erratum-A E4).

    Default None preserves the incumbent EXACTLY: no run-config key is
    emitted, so every pre-H4 run keeps the legacy sampled holdout
    byte-identically. `sealed_test` evaluates the global model on exactly the
    test indices of data/val_test_split_manifest.json (the split locked
    2026-05-14, never re-cut). The launch tooling passes this fleet-wide for
    H4 units; the four H2P+* tokens additionally bake it into their configs.
    """
    parser.add_argument(
        "--eval-split",
        choices=["legacy", "sealed_test"],
        default=None,
        help="Evaluation population: legacy (sampled holdout, incumbent) or "
             "sealed_test (the locked val/test manifest's TEST indices — "
             "REQUIRED for H4 units per erratum-A E4).",
    )


def eval_split_run_config_from_cli(eval_split: "str | None") -> dict:
    """Map --eval-split to run-config overrides.

    Returns {} when undeclared — the merge is then a no-op and the run-config
    (hence provenance and every downstream artifact) is byte-identical to the
    incumbent. When declared, emits the hyphenated key server_app reads."""
    if not eval_split:
        return {}
    return {"eval-split": eval_split}


def add_h2p_cli_args(parser: "argparse.ArgumentParser") -> None:
    """Register the erratum-B detector knobs (RULED 2026-08-18, v1.53).

    Defaults preserve the incumbent EXACTLY: no run-config key is emitted.
    `--h2p-observe-only` switches the H2' detector/observer to the § B1
    calibration mode (scores + logs would-flags, drops NOBODY) — the EXP-062
    calibration fleet passes it fleet-wide via run_extras `h2p_observe_only`.
    `--h2p-cuts-version` pins the serving cut table EXPLICITLY (v1 =
    per-scenario § 7.1 cuts, the incumbent; v2 = per-(scenario, arm-class)
    erratum-B cuts) — never auto-detected from what exists on disk.
    """
    parser.add_argument(
        "--h2p-observe-only",
        action="store_true",
        default=False,
        help="Run the H2' detector/observer in erratum-B observe-only "
             "calibration mode: score every participating client and log "
             "per-(client, round) rows, but drop nobody.",
    )
    parser.add_argument(
        "--h2p-cuts-version",
        choices=["v1", "v2"],
        default=None,
        help="Serving cut table: v1 (per-scenario, incumbent § 7.1) or v2 "
             "(per-(scenario, arm-class), erratum B). Absent = v1; the "
             "selection is explicit, never auto-detected.",
    )


def h2p_run_config_from_cli(h2p_observe_only: bool,
                            h2p_cuts_version: "str | None") -> dict:
    """Map the erratum-B CLI knobs to run-config overrides.

    Returns {} at defaults — the merge is then a no-op and the run-config
    (hence provenance and every downstream artifact) is byte-identical to
    the incumbent. Declared knobs emit the hyphenated keys server_app reads.
    """
    out: dict = {}
    if h2p_observe_only:
        out["h2p-observe-only"] = True
    if h2p_cuts_version:
        out["h2p-cuts-version"] = h2p_cuts_version
    return out


def h2p_observe_provenance(run_config: dict) -> dict:
    """Erratum-B custody: whether this unit ran the detector observe-only.

    Always declared (False = enforcing/incumbent) so the cache-reuse
    identity and the calibration builder read the mode off the artifact,
    never off launch records. Coercion is the SAME strict single-source
    helper server_app dispatches on — a typo'd knob refuses identically at
    both ends instead of running one mode and declaring the other."""
    from flowerfl.server_app import _h2p_observe_only_from_run_config

    return {
        "h2p_observe_only": _h2p_observe_only_from_run_config(run_config)
    }


def h2p_observe_block(strategy_obj) -> "dict | None":
    """The erratum-B § B1 calibration log for the result JSON.

    Reads the ACTUAL detector plugin's accumulated observe rows and enriches
    each with `malicious_gt` from the strategy's scenario adversary set —
    the SAME run-static source `_maybe_log_signals` stamps signal-log rows
    from (`self._adv_ids`, membership by logical_cid) — so the cut builder
    pools ground-truth-honest rows without a cross-artifact join. Constants
    (mode, cut, scenario token, arm class, cuts version, bundle sha) live in
    the block header; rows carry {server_round, scenario_round, logical_cid,
    score, would_flag, malicious_gt}.

    Returns None (no block, byte-identical result JSON) unless an
    observe-only detector actually ran. A strategy that produced observe
    rows but exposes no adversary set REFUSES: silently labeling everyone
    honest would move the calibrated cuts.
    """
    if strategy_obj is None:
        return None
    detector = next(
        (p for p in getattr(strategy_obj, "_plugins", []) or []
         if getattr(p, "name", "") == "H2PrimeDetector"
         and getattr(p, "observe_only", False)),
        None,
    )
    if detector is None:
        return None
    adv_ids = getattr(strategy_obj, "_adv_ids", None)
    if adv_ids is None:
        raise RuntimeError(
            "h2p_observe_block: the strategy carries observe-only detector "
            "rows but no scenario adversary set (_adv_ids) — malicious_gt "
            "cannot be stamped and the calibration log would silently label "
            "every row honest. Refusing (erratum B: the cut pools "
            "GROUND-TRUTH-HONEST rows)."
        )
    rows = [
        {**row, "malicious_gt": (row["logical_cid"] in adv_ids)}
        for row in detector.observe_rows
    ]
    return {
        "mode": "observe_only",
        "cuts_version": detector.cuts_version,
        "scenario_token": detector.scenario_token,
        "arm_class": detector.arm_class,
        "cut": float(detector.cut),
        "serving_bundle_sha256": detector.bundle_sha256,
        "n_rows": len(rows),
        "rows": rows,
    }


def _h2p_cuts_version_cache_reusable(existing: dict,
                                     extra_run_config: "dict | None") -> bool:
    """Never reuse a cached result across the v1/v2 cut-table boundary.

    A v1-cut unit is a DIFFERENT instrument from a v2-cut unit of the same
    (config, scenario, seed); silently reusing one as the other would ship
    the wrong operating point into a fleet. Identity:

    * active side = the declared ``h2p-cuts-version`` normalized, absent =
      the effective ``v1`` default the run will actually serve;
    * cached side = ``provenance.h2p_cuts_version``. Key ABSENT = a legacy
      pre-erratum result (could only ever have served v1 cuts) — reusable
      only for an effective-v1 run, so legacy/non-H2P paths stay
      byte-unchanged while a declared-v2 run always recomputes. Value
      ``None`` = a non-detector arm: no cut table was in play and the knob
      cannot change its behavior, so it stays reusable.
    """
    extra = extra_run_config or {}
    active = (
        str(extra.get("h2p-cuts-version") or "").strip().lower() or "v1"
    )
    cached_prov = existing.get("provenance") or {}
    if "h2p_cuts_version" not in cached_prov:
        return active == "v1"
    cached = cached_prov["h2p_cuts_version"]
    if cached is None:
        return True
    return str(cached).strip().lower() == active


def _h2p_observe_cache_reusable(existing: dict,
                                extra_run_config: "dict | None") -> bool:
    """Never reuse a cached result across the observe/enforce boundary.

    An observe-only unit aggregates a DIFFERENT model trajectory than an
    enforcing unit of the same (config, scenario, seed) — and its result is
    the calibration artifact. Cached provenance absent the key = legacy
    enforcing (pre-erratum result)."""
    from flowerfl.server_app import _h2p_observe_only_from_run_config

    active = _h2p_observe_only_from_run_config(extra_run_config or {})
    cached_prov = existing.get("provenance") or {}
    cached = bool(cached_prov.get("h2p_observe_only", False))
    return active == cached


def eval_split_provenance(run_config: dict, eval_manager=None) -> dict:
    """E4 custody pair for the result JSON: which evaluation population
    scored this unit, and (for sealed_test) the manifest-bytes sha256 read
    from the ACTUAL eval manager — the artifact-of-record, not the launch
    intent. Always present; "legacy" with a null sha is the incumbent."""
    declared = str(run_config.get("eval-split", "") or "").strip().lower()
    sha = getattr(eval_manager, "manifest_sha256", None) if eval_manager else None
    return {
        "eval_split": declared if declared else "legacy",
        "eval_split_manifest_sha256": sha,
    }


def run_uid_provenance(strategy_obj) -> dict:
    """Universal per-unit run identity for the result JSON (Lane C custody
    audit): `provenance.run_uid` is exported for ALL arms, read from the
    ACTUAL SignalLogger the run wrote with — the same
    `{exec_mode}__{scenario}__{defense}__seed{seed}__{run_started_at}` string
    the FP arms already bind into `fingerprint_registry.run_uid` (stamped onto
    the plugin from this very logger at ScenarioStrategy construction), so on
    FP-bearing arms the two fields are EQUAL by construction. Without this the
    five non-FP arms carried no run identity and the H4 scorer could not
    enforce cross-unit uniqueness. Null — truthfully — when signal logging was
    disabled (the scorer refuses such units). The value embeds the launch
    seed, which is fine in the unit's own provenance (same exposure as the
    incumbent FP custody block); the scorer redacts on echo."""
    logger_obj = getattr(strategy_obj, "_signal_logger", None)
    return {"run_uid": getattr(logger_obj, "run_uid", None)}


def h4_serving_provenance(strategy_obj) -> dict:
    """H4 custody: the serving bundle's manifest sha256 plus the erratum-B
    cut-table identity, read from the ACTUAL detector plugin the run
    executed with (never from launch records).

    All three fields are null — not empty strings — for every arm without
    the online detector (BUILD_CONTRACT custody semantics). `h2p_cuts_version`
    names which cut table (v1 per-scenario / v2 per-(scenario, arm-class))
    resolved the operating point; `h2p_arm_class` is the erratum-B arm-class
    the cut was keyed by (null on v1 arms outside the arm-class table)."""
    for plugin in getattr(strategy_obj, "_plugins", []) or []:
        if getattr(plugin, "name", "") == "H2PrimeDetector":
            return {
                "serving_bundle_sha256": plugin.bundle_sha256,
                "h2p_cuts_version": getattr(plugin, "cuts_version", "v1"),
                "h2p_arm_class": getattr(plugin, "arm_class", None),
            }
    return {
        "serving_bundle_sha256": None,
        "h2p_cuts_version": None,
        "h2p_arm_class": None,
    }


def _eval_split_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's
    declared evaluation population (erratum-A E4).

    Same failure class as _fp_cohort_cache_reusable: the cache is keyed only
    by (config, seed), so a sealed-test rerun in a dir holding a legacy
    result would otherwise return the SAMPLED-holdout trajectory AS the
    sealed-test unit — a different evaluation population — without ever
    building the sealed evaluator. A cached result MISSING the field reads
    as "legacy" and is reusable only by a run that likewise declares none."""
    active = eval_split_provenance(extra_run_config or {})["eval_split"]
    cached_prov = existing.get("provenance") or {}
    cached = str(cached_prov.get("eval_split") or "legacy")
    return active == cached


def fp_registry_policy_provenance(run_config: dict) -> dict:
    """Declared registry candidate policy for the result JSON. Always present so
    a reader never confuses "absent" (incumbent) with an explicit declaration,
    and so adjudication can read WHICH candidate pool produced a unit's re-entry
    events off the artifact itself rather than off launch records."""
    value = run_config.get("fp-registry-policy") or None
    return {"fp_registry_policy": value}


def fp_cohort_provenance(run_config: dict) -> dict:
    """Declared fingerprint cohort for the result JSON. Always present so a
    reader never confuses "absent" (pre-lock / non-FP arm) with a declared
    cohort, and so adjudication can read WHICH locked τ scored the unit off
    the artifact itself rather than off launch records."""
    value = run_config.get("fp-cohort") or None
    return {"fp_cohort": value}


def normalize_train_only_provenance(run_config: dict) -> dict:
    """m1 leakage-fix knob provenance for the result JSON. Always present so a
    reader never confuses "absent" with "off", and so the cache-reuse identity
    can tell a leak-free result from a (default) leak-on one sharing a
    (config, seed) output filename."""
    return {
        "normalize_train_only": _stage_f_bool(
            run_config.get("normalize-train-only", False)
        ),
    }


def stage_f_provenance_fields(run_config: dict) -> dict:
    """Stage-F knob provenance for the result JSON.

    Always present so a reader never distinguishes "absent" from "off", and — the
    load-bearing reason — so the cache-reuse identity can compare them. Two runs
    that share the same sampler/variant/target but differ in update-match,
    weight-mode, or semantic-target are DIFFERENT arms; without these fields in
    provenance a legacy result could be reused as a semantic run (or vice versa),
    returning the wrong arm's trajectory without ever running."""
    return {
        "update_match": _stage_f_bool(run_config.get("update-match", False)),
        "weight_mode": str(run_config.get("weight-mode", "resampled")),
        "smote_semantic_target": _stage_f_bool(run_config.get("smote-semantic-target", False)),
    }


def _stage_f_identity(fields: dict) -> tuple:
    """The (update_match, weight_mode, semantic_target) triple identifying a
    Stage-F arm. A pre-Stage-F cached result (no such fields) reads as the
    incumbent identity (False, 'resampled', False), so it is reusable only by an
    incumbent run."""
    return (
        bool(fields.get("update_match", False)),
        fields.get("weight_mode", "resampled"),
        bool(fields.get("smote_semantic_target", False)),
    )


def _stage_f_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's Stage-F arm. Same failure class as _smote_cache_reusable: reuse only
    on an exact (update-match, weight-mode, semantic-target) identity match."""
    active = stage_f_provenance_fields(extra_run_config or {})
    cached = existing.get("provenance") or {}
    return _stage_f_identity(active) == _stage_f_identity(cached)


def _holdout_disjoint_provenance(run_config: dict) -> bool:
    """Coerce the ``holdout-disjoint`` run-config value to bool for the result
    JSON provenance (GWU-61). Default True; only an explicit false-ish value
    records the legacy overlapping holdout. Mirrors server_app's coercion so the
    provenance and the manager can't disagree."""
    val = run_config.get("holdout-disjoint", True)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")


def _holdout_provenance_fields(run_config: dict, eval_manager=None) -> dict:
    """Holdout provenance for the result JSON (GWU-61 + drift-investigation r-fix).

    Prefer the ACTUAL FixedEvalManager's durable record — holdout_rows_excluded,
    size, per-class counts — because the ``[FixedEval] holdout_disjoint=...``
    stdout line never reliably reaches CloudWatch (it prints inside a
    captured-stdout section, the same gap as the driver-prewarm health), so it
    was invisible on EXP-019/021 despite the disjoint construction running. The
    result JSON is the artifact-of-record. Falls back to the run-config flag
    alone (no rows_excluded) when the manager is unavailable — e.g. a failed or
    None strategy — so this never raises.
    """
    if eval_manager is not None and hasattr(eval_manager, "holdout_provenance"):
        try:
            fields = eval_manager.holdout_provenance()
            if isinstance(fields, dict) and "holdout_disjoint" in fields:
                return fields
        except Exception:  # noqa: BLE001 — provenance must never fail the run
            pass
    return {"holdout_disjoint": _holdout_disjoint_provenance(run_config)}


def _smote_identity(fields: dict) -> tuple:
    """The (enabled, variant, target) triple that identifies a SMOTE arm.

    Outcome fields (counts, seed_component, skipped_reason) are NOT part of
    identity — only the configured knob is."""
    return (
        bool(fields.get("smote_enabled", False)),
        fields.get("smote_variant"),
        fields.get("smote_target"),
    )


def _smote_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's SMOTE arm.

    The local result cache is keyed by (config, seed) in the filename only, so a
    rerun with --smote-enabled in an output dir holding a SMOTE-off result would
    otherwise return the incumbent JSON AS the SMOTE arm — before extra_run_config
    is even built. Compare the cached result's SMOTE provenance identity against
    the active run's; reuse only on an exact match. Provenance comparison (not a
    filename token) also catches the reverse (a SMOTE-on cache reused for an off
    run) and variant/target mismatches. A pre-SMOTE cached result (no smote_*
    provenance) reads as the off identity, so it is reusable only by an off run."""
    active = smote_provenance_fields(extra_run_config or {})
    cached = existing.get("provenance") or {}
    return _smote_identity(active) == _smote_identity(cached)


def _leakage_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's m1
    normalization mode (normalization audit).

    Same failure class as _smote_cache_reusable / _stage_f_cache_reusable: the
    local result cache is keyed only by (config, seed) in the filename, so a
    leak-free rerun in a dir holding a leak-on result would otherwise return the
    leak-on numbers AS the leak-free arm. Compare the cached result's
    ``normalize_train_only`` provenance against the active run's; reuse only on a
    match. A cached result MISSING the field is a pre-fix (leak-on) result, so it
    reads as False and is reusable only by a leak-on (default) run."""
    active = normalize_train_only_provenance(extra_run_config or {})["normalize_train_only"]
    cached_prov = existing.get("provenance") or {}
    cached = bool(cached_prov.get("normalize_train_only", False))  # missing => leak-on
    return bool(active) == cached


def _fp_cohort_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's declared
    fingerprint cohort.

    Same failure class as _smote_cache_reusable / _leakage_cache_reusable: the
    local result cache is keyed only by (config, seed) in the filename, so an
    adjudicating-cohort rerun in a dir holding a validation-cohort result would
    otherwise return the validation trajectory AS the adjudicating unit — a
    DIFFERENT locked instrument (different τ and Σ) — without ever constructing
    the adjudicating registry, while retaining the stale cohort provenance.
    Compare declared cohorts; reuse only on an exact match. A cached result
    MISSING the field is a pre-lock / non-FP result (reads as None) and is
    reusable only by a run that likewise declares no cohort."""
    active = fp_cohort_provenance(extra_run_config or {})["fp_cohort"]
    cached_prov = existing.get("provenance") or {}
    cached = cached_prov.get("fp_cohort") or None
    return active == cached


def _fp_registry_policy_cache_reusable(
    existing: dict, extra_run_config: "dict | None"
) -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's declared
    registry candidate policy.

    Same failure class as _fp_cohort_cache_reusable, one axis over: the two
    policies are DIFFERENT INSTRUMENTS producing different re-entry events from
    the same (config, seed), which is the whole point of the v1.49 ruling. A
    cached flag-gated trajectory returned as an identity_only unit would hand
    the corrected H3 exactly the invalid instrument it exists to replace. A
    cached result MISSING the field is an incumbent run (reads as None) and is
    reusable only by a run that likewise declares no policy."""
    active = fp_registry_policy_provenance(extra_run_config or {})["fp_registry_policy"]
    cached_prov = existing.get("provenance") or {}
    cached = cached_prov.get("fp_registry_policy") or None
    return active == cached


def _active_sealed_manifest_sha256() -> "str | None":
    """sha256 of the ACTIVE tree's sealed split manifest bytes, or None when
    the file is unreadable (the rerun would then refuse loudly at eval-manager
    build; the cache decision just declines reuse)."""
    import hashlib

    from rmc.sealed_test_eval import DEFAULT_MANIFEST_PATH

    try:
        return hashlib.sha256(DEFAULT_MANIFEST_PATH.read_bytes()).hexdigest()
    except OSError:
        return None


def _holdout_cache_reusable(existing: dict, extra_run_config: "dict | None") -> bool:
    """Whether a cached result JSON may be reused for the ACTIVE run's HOLDOUT
    mode.

    Same failure class as _smote_cache_reusable: the local result cache is keyed
    only by (config, seed) in the filename, so a v8 default-disjoint invocation in
    a dir holding a pre-v8 result would otherwise return the OVERLAPPING-holdout
    numbers this change exists to replace. Compare the cached result's
    ``holdout_disjoint`` provenance against the active run's requested mode; reuse
    only on a match. A cached result MISSING the field is a pre-v8 legacy result
    (overlapping holdout), so it reads as disjoint=False and is reusable only by a
    --no-holdout-disjoint run.

    SEALED-TEST branch (erratum-A E4): when BOTH the active run and the cached
    result declare ``eval_split=sealed_test``, the ``holdout_disjoint`` flag is
    meaningless — the sealed evaluator samples nothing and excludes nothing, so
    it truthfully records ``holdout_disjoint=false`` while the launch config
    carries the runner default ``holdout-disjoint=true``. Comparing the flag
    there declared every identical sealed-test rerun a mismatch, breaking the
    cache/self-heal reuse path (Spot-reclaim resume economics). The sealed
    identity is the SPLIT itself: reuse iff the cached
    ``eval_split_manifest_sha256`` equals the sha256 of the active tree's
    manifest bytes. The provenance stays truthful (holdout_disjoint=false);
    only the comparison changed. Legacy-vs-legacy semantics are byte-unchanged,
    and a legacy/sealed cross-pairing is already refused by
    ``_eval_split_cache_reusable`` regardless of what this function returns."""
    cached_prov = existing.get("provenance") or {}
    active_eval_split = eval_split_provenance(extra_run_config or {})["eval_split"]
    cached_eval_split = str(cached_prov.get("eval_split") or "legacy")
    if active_eval_split == "sealed_test" and cached_eval_split == "sealed_test":
        cached_sha = cached_prov.get("eval_split_manifest_sha256") or None
        active_sha = _active_sealed_manifest_sha256()
        return cached_sha is not None and cached_sha == active_sha
    active = _holdout_disjoint_provenance(extra_run_config or {})
    cached = bool(cached_prov.get("holdout_disjoint", False))  # missing => legacy/overlapping
    return active == cached


def _rotate_stale_signal_log(signal_path: "Path | None") -> "Path | None":
    """Rotate a pre-existing signal-log JSONL out of the way before a rerun.

    The SignalLogger opens append-only and is named without SMOTE identity
    (<exec_mode>__<scenario>__<defense>__seed<seed>.jsonl), so rerunning a unit
    — the SMOTE cache-identity RECOMPUTE, or a crashed prior run's leftover —
    would append the new arm's rows onto the old, producing a mixed-arm log
    whose result provenance declares only the active arm. Renaming to
    ``<stem>.superseded-<n>.jsonl`` (smallest free n>=1) lets the rerun start a
    fresh log while PRESERVING the prior rows — custody prefers rotation over
    destruction; nothing is ever deleted. Returns the rotated-to path, or None
    when there was nothing to rotate."""
    if signal_path is None or not signal_path.exists():
        return None
    n = 1
    while True:
        target = signal_path.with_name(f"{signal_path.stem}.superseded-{n}.jsonl")
        if not target.exists():
            break
        n += 1
    signal_path.rename(target)
    print(f"    ROTATED stale signal log {signal_path.name} -> {target.name} "
          f"(prior rows preserved; fresh log for this run)")
    return target


def _reuse_cached_or_rotate(
    json_path: Path,
    extra_run_config: "dict | None",
    scenario_path: str,
    strategy: str,
    seed: int,
    optimizer_state: str,
    exp_name: str,
) -> "dict | None":
    """Return a cached result to reuse, or None (having rotated any stale signal
    log) when a rerun is required.

    Cache reuse requires a non-empty trajectory, return_code 0, a matching SMOTE
    arm, AND a matching holdout mode. On the
    rerun path — mismatch, malformed
    cache, or no cache at all — rotate any stale signal log for this unit's stem
    so the append-only logger starts fresh. Reuse never
    rotates; rerun always does (a no-op when no stale file exists, e.g. the fleet
    where the container filesystem is fresh per attempt)."""
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
            if existing.get("trajectory") and existing.get("return_code") == 0:
                smote_ok = _smote_cache_reusable(existing, extra_run_config)
                holdout_ok = _holdout_cache_reusable(existing, extra_run_config)
                stage_f_ok = _stage_f_cache_reusable(existing, extra_run_config)
                leakage_ok = _leakage_cache_reusable(existing, extra_run_config)
                fp_cohort_ok = _fp_cohort_cache_reusable(existing, extra_run_config)
                fp_policy_ok = _fp_registry_policy_cache_reusable(
                    existing, extra_run_config
                )
                eval_split_ok = _eval_split_cache_reusable(existing, extra_run_config)
                h2p_observe_ok = _h2p_observe_cache_reusable(
                    existing, extra_run_config
                )
                h2p_cuts_ok = _h2p_cuts_version_cache_reusable(
                    existing, extra_run_config
                )
                if (smote_ok and holdout_ok and stage_f_ok and leakage_ok
                        and fp_cohort_ok and fp_policy_ok and eval_split_ok
                        and h2p_observe_ok and h2p_cuts_ok):
                    print(f"    SKIP {exp_name} (cached)")
                    return existing
                if not smote_ok:
                    which = "SMOTE arm"
                elif not holdout_ok:
                    which = "holdout mode"
                elif not stage_f_ok:
                    which = "Stage-F arm"
                elif not leakage_ok:
                    which = "normalization mode"
                elif not eval_split_ok:
                    which = "eval split"
                elif not h2p_observe_ok:
                    which = "h2p observe mode"
                elif not h2p_cuts_ok:
                    which = "h2p cuts version"
                else:
                    which = "fingerprint cohort"
                print(f"    RECOMPUTE {exp_name} (cached {which} differs from active)")
        except Exception:
            pass
    from flowerfl.server_app import signal_log_path
    _rotate_stale_signal_log(signal_log_path(
        scenario_path, strategy,
        {
            "seed": seed,
            "optimizer-state": optimizer_state,
            "signal-log": (extra_run_config or {}).get("signal-log", "1"),
        },
    ))
    return None


SCENARIO_PATH = "rmc/scenarios/rmc_main_50r.json"
DATASET = "edge_full_20_rmc"
FEDERATION = "rmc-20-local"
NUM_ROUNDS = 50
NUM_SUPERNODES = 21  # rmc-20-local federation: 20 client supernodes + 1 server slot (spec § 4.8)

# M1: named constants replacing magic-number literals (spec § 4.9).
A3_PARTICIPATION_FLOOR_FRAC = 0.95  # spec § 4.9: participants_per_round >= ceil(0.95 * num_supernodes)
A6_TRAJECTORY_MIN_FRAC = 0.5        # spec § 4.9: trajectory must cover >= 50% of declared rounds
CS_MODEL = "models/cold_start/flower_reset/S_k3_final.pkl"
CS_WEIGHTING = "linear"
CS_K = 3

# Provenance metadata for anchor 5 (every result JSON carries its origin).
RUNNER_VERSION = "unified-v1.0"

_HPARAMS_LOCKED_PATH = Path(__file__).resolve().parent.parent / "data" / "hparams_locked.json"


# M3: mtime-aware cache for _load_locked_lr.
# lru_cache caches forever; keying on (optimizer_state, file_mtime) ensures
# a mid-process edit to hparams_locked.json surfaces as a cache miss.
# Spec § 4.5 forbids mid-process edits, but defense-in-depth catches violations.
_LOAD_LOCKED_LR_CACHE: dict[tuple[str, float], float] = {}


def _load_locked_lr(optimizer_state: str) -> float:
    """Read mode-specific lr from data/hparams_locked.json (spec § 4.5).

    Cache keyed on (optimizer_state, file_mtime) so a mid-process file edit
    surfaces as a cache miss. Spec § 4.5 forbids mid-process edits, but
    defense-in-depth: this catches accidental violations.

    optimizer_state is the runner CLI value: "reset" or "persistent".
    Maps to hparams keys "flower_reset" or "persistent_optimizer".
    Raises ValueError on unknown mode, KeyError if section is absent.
    """
    mtime = _HPARAMS_LOCKED_PATH.stat().st_mtime
    key = (optimizer_state.lower(), mtime)
    if key in _LOAD_LOCKED_LR_CACHE:
        return _LOAD_LOCKED_LR_CACHE[key]

    mode_key = {"reset": "flower_reset", "persistent": "persistent_optimizer"}.get(
        optimizer_state.lower()
    )
    if mode_key is None:
        raise ValueError(
            f"Unknown optimizer_state={optimizer_state!r}; "
            f"expected 'reset' or 'persistent'"
        )
    hparams = json.loads(_HPARAMS_LOCKED_PATH.read_text())
    if mode_key not in hparams:
        raise KeyError(
            f"hparams_locked.json is missing section {mode_key!r}; "
            f"check {_HPARAMS_LOCKED_PATH}"
        )
    lr = float(hparams[mode_key]["lr"])
    _LOAD_LOCKED_LR_CACHE[key] = lr
    return lr


def _build_run_config(
    strategy: str,
    scenario_path: str,
    rounds: int,
    seed: int,
    optimizer_state: str,
    use_cs: bool,
    max_per_client: int,
) -> dict:
    """Construct the Flower run_config dict.

    Reads lr from data/hparams_locked.json based on optimizer_state mode.
    Raises ValueError for unknown optimizer_state.
    """
    max_samples = max_per_client if max_per_client > 0 else 5000
    lr = _load_locked_lr(optimizer_state)
    cfg: dict[str, bool | float | int | str] = {
        "dataset": DATASET,
        "strategy": strategy,
        "scenario": scenario_path,
        "num-server-rounds": rounds + 1,  # +1 for discovery round per scenario_strategy offset
        "batch-size": 32,
        "max-samples": max_samples,
        "learning-rate": lr,
        "local-epochs": 5,
        "weight-decay": 0.0,
        "seed": seed,
        "optimizer-state": optimizer_state,
    }
    if use_cs:
        cfg["cs-model"] = CS_MODEL
        cfg["cs-weighting"] = CS_WEIGHTING
        cfg["cs-k"] = CS_K
    # Scenario-derived defense sizing (methodology v1.19 launch-blocker fix):
    # emit the canonical RMC adversary count and per-round cohort so
    # server_app builds Multi-Krum at the documented f/keep (9/20 -> keep 9)
    # instead of the malicious-fraction=0 -> f=1/keep-18 fallback. Emitted only
    # when a scenario JSON is resolvable and declares participants; otherwise
    # the legacy path is left untouched.
    sizing = _scenario_defense_sizing(scenario_path)
    if sizing is not None:
        cfg["num-malicious"], cfg["defense-cohort-size"] = sizing
    return cfg


def _assert_lr_matches_locked(recorded_lr: float, optimizer_state: str) -> None:
    """Integrity A1: recorded lr matches data/hparams_locked.json for this mode.

    Prevents Phase 2 drift catch: runner config emitting a different lr than
    the locked-hparams file mandates for the chosen optimizer_state mode.
    """
    expected = _load_locked_lr(optimizer_state)
    if recorded_lr != expected:
        raise AssertionError(
            f"A1 [recorded lr drift]: optimizer_state={optimizer_state} "
            f"locked lr={expected}, recorded lr={recorded_lr}. "
            f"Spec § 4.9; data/hparams_locked.json is the source of truth."
        )


def _assert_optimizer_state_self_consistent(recorded_state: str, cli_state: str) -> None:
    """Integrity A2: recorded optimizer_state in run_config matches CLI flag.

    Catches the case where a run_config override (e.g., extra_run_config) silently
    overrides the CLI --optimizer-state, producing a result JSON whose mode tag
    does not match the operator's intent.
    """
    if recorded_state.lower() != cli_state.lower():
        raise AssertionError(
            f"A2 [optimizer_state drift]: CLI --optimizer-state={cli_state!r} "
            f"but run_config recorded {recorded_state!r}. Spec § 4.9."
        )


_INTEGRITY_MARKER_RE = re.compile(
    r"\[Integrity\] round=(\d+) participants=(\d+) n_malicious=(\d+)"
)


def _parse_participants_markers(captured_stdout: str) -> list[dict]:
    """Pull '[Integrity] round=R participants=P n_malicious=M' rows from stdout."""
    rows = []
    for m in _INTEGRITY_MARKER_RE.finditer(captured_stdout):
        rows.append(
            {
                "round": int(m.group(1)),
                "participants": int(m.group(2)),
                "n_malicious": int(m.group(3)),
            }
        )
    return rows


def _scenario_declared_participants_per_round(scenario_dict: dict) -> dict[int, int]:
    """Count declared participants per scenario_round from scenario JSON.

    Mirrors `_scenario_declared_malicious_per_round`. Each schedule block carries
    a `participants` list of logical client ids active that round; its length is
    the scheduled participant count. Scenarios with identity-reset/disconnect
    blocks (S3, S4) legitimately schedule fewer participants in some rounds.

    Blocks tagged `"skip_scheduling": True` are metadata-only and are skipped.
    """
    declared: dict[int, int] = {}
    for block in scenario_dict.get("schedule", []):
        if block.get("skip_scheduling", False):
            continue
        rounds_field = block.get("rounds")
        if not isinstance(rounds_field, list) or len(rounds_field) != 2:
            continue
        participants = block.get("participants")
        if isinstance(participants, list):
            n = len(participants)
        elif isinstance(participants, int):
            n = participants
        else:
            continue  # no participant info in this block — leave to floor fallback
        start, end = int(rounds_field[0]), int(rounds_field[1])
        for r in range(start, end + 1):
            declared[r] = n
    return declared


def _assert_participants_per_round(
    rows: list[dict], num_supernodes: int, scenario_dict: dict | None = None
) -> None:
    """Integrity A3: dispatched participants_per_round are not silently truncated.

    'Participants' here means clients DISPATCHED by `configure_fit` (i.e., the
    set the strategy intended to engage that round). A post-dispatch failure
    in `aggregate_fit` (e.g., a Ray actor dying mid-training) would not be
    caught by this assertion — that gap is documented for future hardening
    if it ever surfaces in practice.

    Expected participant count per round:
      - If `scenario_dict` is provided, use the per-round count DECLARED by the
        scenario schedule. This honors scenarios that legitimately disconnect
        clients (S3 identity-reset, S4 full-mix schedule fewer than full
        participation in some rounds — B5, caught by the S4 smoke 2026-06-05).
      - Otherwise (or for rounds the schedule does not declare), fall back to the
        flat floor ceil(A3_PARTICIPATION_FLOOR_FRAC * num_supernodes).

    Fails only on TRUE truncation: observed < expected. Prevents Bug #3 (Ray
    actor-pool truncation produced 7/20 participation at the configure_fit
    dispatch stage). Spec § 4.9.
    """
    floor = math.ceil(A3_PARTICIPATION_FLOOR_FRAC * num_supernodes)
    declared = (
        _scenario_declared_participants_per_round(scenario_dict)
        if scenario_dict is not None
        else {}
    )
    for row in rows:
        expected = declared.get(row["round"], floor)
        if row["participants"] < expected:
            basis = "scenario-declared" if row["round"] in declared else "floor"
            raise AssertionError(
                f"A3 [participants truncated]: round={row['round']} "
                f"participants={row['participants']} < expected={expected} "
                f"({basis}; num_supernodes={num_supernodes}). "
                f"Likely Ray actor-pool truncation. Spec § 4.9."
            )


def _assert_integrity_markers_emitted(
    integrity_rows: list[dict],
    trajectory: list[dict],
    discovery_rounds: int = 1,
) -> None:
    """Integrity A3.b: marker count is consistent with trajectory length.

    If the simulation produced eval trajectory rows for rounds beyond discovery,
    the strategy should also have emitted [Integrity] markers for those rounds.
    Detects the silent-pass failure mode where marker emission is suppressed
    (e.g., by an off-by-one in scenario_round, or a stdout-capture layer that
    swallows the print). Allows for legitimate cases where the marker is
    skipped on discovery rounds.
    """
    # If no scenario events at all (e.g., a smoke test), accept empty markers.
    if len(trajectory) <= discovery_rounds:
        return
    if not integrity_rows:
        raise AssertionError(
            f"A3.b [missing integrity markers]: trajectory has {len(trajectory)} "
            f"eval rows but no [Integrity] markers were emitted. Either marker "
            f"emission is broken or scenario_round indexing skipped every round. "
            f"Spec § 4.9."
        )


def _scenario_declared_malicious_per_round(scenario_dict: dict) -> dict[int, int]:
    """Count declared malicious clients per scenario_round from scenario JSON.

    Canonical scenario JSON shape (see existing rmc/scenarios/*.json):

        {"schedule": [{"rounds": [start, end], "attacks": {cid: {"type": ...}}}, ...]}

    `rounds` is a 2-element inclusive range, not a list of individual rounds.
    Each block's `attacks` maps logical_id -> {"type": str, "params": dict};
    counting non-empty truthy `type` values per round equals the scheduled
    n_malicious for that round.

    Blocks tagged `"skip_scheduling": True` are metadata-only (e.g., honest-
    reconnect annotations in v3 scenarios); skip them — they do not contribute
    malicious clients.
    """
    declared: dict[int, int] = {}
    for block in scenario_dict.get("schedule", []):
        if block.get("skip_scheduling", False):
            continue
        rounds_field = block.get("rounds")
        if not isinstance(rounds_field, list) or len(rounds_field) != 2:
            continue  # malformed block — skip rather than misattribute
        start, end = int(rounds_field[0]), int(rounds_field[1])
        attacks = block.get("attacks") or {}
        n = sum(1 for v in attacks.values() if v and v.get("type"))
        for r in range(start, end + 1):
            declared[r] = declared.get(r, 0) + n
    return declared


def _scenario_defense_sizing(scenario_path: str) -> "tuple[int, int] | None":
    """Derive ``(num_malicious, defense_cohort)`` from a scenario JSON schedule.

    - ``num_malicious`` = peak per-round declared adversary count — the
      canonical RMC sustained count (9 for S0-S4). Disconnect/discovery rounds
      declare fewer. NOTE (PR #12 P1): these values are PROVENANCE plus the
      loud misconfiguration guard and the full-cohort operating-point
      cross-check (ceil(20/2)-1 == 9); per-round sizing in the deployed
      KrumDefensePlugin is DYNAMIC (``dynamic_f=True``, April anchor formula
      ``ceil(n/2)-1``) so S3/S4 disconnect rounds (n~11, adversaries
      disconnected) remain computable rather than tripping the uncomputable
      fallback under a static f=9.
    - ``defense_cohort`` = peak per-round declared participant count — the
      20-client per-round cohort (NOT the 21 dataset partitions
      ``get_num_clients()`` returns for edge_full_20_rmc).

    Both are read via the same schedule-parsing helpers the A4 / A3 integrity
    assertions use (``_scenario_declared_malicious_per_round`` /
    ``_scenario_declared_participants_per_round``), so the deployed defense's f
    and keep cannot drift from the scenario the run actually executes. Verified
    robust across S0-S4 (all yield (9, 20)).

    Returns None when the path is empty, the file is unreadable, or the
    schedule declares no participants — the caller then falls back to the
    legacy malicious-fraction path so non-scenario behavior is unchanged.
    """
    if not scenario_path:
        return None
    try:
        with open(scenario_path) as fh:
            scenario_dict = json.load(fh)
    except (OSError, ValueError):
        return None
    participants = _scenario_declared_participants_per_round(scenario_dict)
    if not participants:
        return None
    malicious = _scenario_declared_malicious_per_round(scenario_dict)
    num_malicious = max(malicious.values()) if malicious else 0
    defense_cohort = max(participants.values())
    return num_malicious, defense_cohort


# Deployed Krum f policy string recorded in provenance (PR #12 round 2).
# Scenario-mode KrumDefensePlugin sizes f per round as ceil(n/2)-1 (April
# anchor formula, reproduce_szelag.py:388,739) regardless of the scenario's
# DECLARED adversary count — a deployed server has no oracle knowledge of the
# true count, and the sensitivity arms (rmc_intensity_*) deliberately vary
# intensity against a CONSTANT defense config.
KRUM_F_POLICY = "dynamic ceil(n/2)-1"


def _defense_provenance_fields(strategy: str, run_config: dict) -> dict:
    """Defense-sizing provenance for the result JSON (PR #12 round-2 P2).

    Records BOTH facts separately so the instrumentation audit can never
    conflate them:
    - ``scenario_declared_adversaries`` / ``defense_cohort_size``: the
      scenario-derived ground truth carried in the run config (None when the
      keys are absent, e.g. legacy non-scenario paths). These parameterize
      the misconfiguration guard and the full-cohort cross-check ONLY.
    - ``krum_f_policy``: the sizing policy the deployed Krum layer actually
      uses — threat-model-constant ``dynamic ceil(n/2)-1`` for scenario-mode
      Krum strategies, deliberately independent of the declared per-scenario
      intensity; "n/a" for strategies with no Krum layer.
    """
    has_scenario_krum = strategy.startswith("Scenario") and "Krum" in strategy
    declared = run_config.get("num-malicious")
    cohort = run_config.get("defense-cohort-size")
    return {
        "scenario_declared_adversaries": int(declared) if declared is not None else None,
        "defense_cohort_size": int(cohort) if cohort is not None else None,
        "krum_f_policy": KRUM_F_POLICY if has_scenario_krum else "n/a",
    }


def _assert_n_malicious_per_round(
    rows: list[dict], scenario_dict: dict
) -> None:
    """Integrity A4: observed n_malicious matches scenario declaration each round.

    Cross-references the [Integrity] markers (emitted by ScenarioStrategy in
    A3, indexed by scenario_round) against the scenario JSON's schedule
    declaration.

    Prevents Bug #1 (LOGICAL_TO_PARTITION silent drop of `client_N_newM`
    identities that produced 0 ground-truth malicious despite 9 declared).
    """
    declared = _scenario_declared_malicious_per_round(scenario_dict)
    for row in rows:
        r = row["round"]
        expected = declared.get(r, 0)
        observed = row["n_malicious"]
        if observed != expected:
            raise AssertionError(
                f"A4 [n_malicious mismatch]: round={r} expected={expected} "
                f"observed={observed}. Likely scenario→partition mapping bug "
                f"(see METHODOLOGY.md Bug #1). Spec § 4.9."
            )


def _assert_signal_log_filename(
    signal_dir: Path,
    scenario_name: str,
    defense: str,
    seed: int,
    cli_optimizer_state: str,
) -> None:
    """Integrity A5: signal log filename exec_mode token matches CLI state.

    Filename convention (server_app.py): <exec_mode>__<scenario>__<defense>__seed<N>.jsonl
    where exec_mode = flower_persistent | flower_reset.

    Prevents Bug #4 (filename collision aliasing two modes onto the same file).
    """
    expected_mode = (
        "flower_persistent" if cli_optimizer_state.lower() == "persistent" else "flower_reset"
    )
    # Normalize the defense token to match the signal logger's filename convention.
    # The runner call site passes the raw strategy class name (e.g. "ScenarioTGEnsemble",
    # "ScenarioKrum"); server_app.py:76 writes the file using
    # strategy_name.replace("Scenario", "").lower() (-> "tgensemble", "krum").
    # Without this, A5 mismatches on EVERY real run (B4, caught by S0 TGE smoke 2026-06-05).
    defense_token = defense.replace("Scenario", "").lower()
    expected_name = f"{expected_mode}__{scenario_name}__{defense_token}__seed{seed}.jsonl"
    expected_path = Path(signal_dir) / expected_name
    if not expected_path.exists():
        if Path(signal_dir).exists():
            actual = sorted(p.name for p in Path(signal_dir).glob("*.jsonl"))
        else:
            actual = []  # signals/ directory itself does not exist
        raise AssertionError(
            f"A5 [signal log filename mismatch]: expected {expected_path} "
            f"to exist after run with --optimizer-state={cli_optimizer_state}. "
            f"Found instead: {actual}. Spec § 4.9."
        )


def _assert_trajectory_non_empty(trajectory: list[dict], rounds: int) -> None:
    """Integrity A6: trajectory non-empty and covers >= 50% of declared rounds.

    Server-side eval (FixedEvalManager) must initialize and emit per-round eval
    lines parseable as trajectory rows. Empty trajectory is the EXP-002 / EXP-003
    silent-empty-trajectory failure (METHODOLOGY.md Bugs #5, #6).

    v1.3 raised FixedEvalManager init failure to RuntimeError; this assertion
    catches the partial case (e.g., manager died mid-run after some rounds
    emitted eval lines).
    Uses A6_TRAJECTORY_MIN_FRAC=0.5 per spec § 4.9.
    """
    if not trajectory:
        raise AssertionError(
            "A6 [empty trajectory]: 0 eval rows parsed despite return_code=0. "
            "FixedEvalManager likely failed silently. Spec § 4.9."
        )
    min_rows = max(1, math.ceil(A6_TRAJECTORY_MIN_FRAC * rounds))
    if len(trajectory) < min_rows:
        raise AssertionError(
            f"A6 [trajectory length]: {len(trajectory)} eval rows "
            f"for {rounds} declared rounds (< {A6_TRAJECTORY_MIN_FRAC * 100:.0f}%). "
            f"Server-side eval may have aborted mid-run. Spec § 4.9."
        )


def _git_rev() -> str:
    """Best-effort current git HEAD SHA. Returns 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _flwr_version() -> str:
    """Best-effort installed Flower version. Returns 'unknown' if not installed."""
    try:
        return importlib.metadata.version("flwr")
    except Exception:
        return "unknown"


def _runner_commit() -> str:
    """Best-effort commit SHA of the RUNNING code for result-JSON provenance
    (runtime provenance correction).

    The container bakes the repo WITHOUT ``.git`` (docker/Dockerfile via
    .dockerignore), so an in-container ``git rev-parse`` fails and the legacy
    ``_git_rev()`` returned a bare ``"unknown"``. Prefer ``PRAXIS_RUNNER_COMMIT``
    — now baked into the image at build time by the Dockerfile ``--build-arg``
    (scripts/aws/ecr/build_push.sh passes the repo SHA being built), NOT set by
    launch — fall back to the local git HEAD for dev runs, and record an
    explicit unavailable-with-reason otherwise so provenance never silently
    reads a meaningless value. Launch-time HEAD lives in ``_launch_commit``.
    """
    env_sha = os.environ.get("PRAXIS_RUNNER_COMMIT", "").strip()
    if env_sha:
        return env_sha
    rev = _git_rev()
    if rev and rev != "unknown":
        return rev
    return "unavailable (no .git in image; PRAXIS_RUNNER_COMMIT unset)"


def _launch_commit() -> str:
    """Best-effort launch-time git HEAD for result-JSON provenance.

    ``PRAXIS_LAUNCH_COMMIT`` is set by matrix_launch/matrix_refill to the git
    HEAD at launch time — legitimate provenance for the LAUNCH-side inputs
    (manifest, scenario JSONs), kept DISTINCT from the image-baked
    ``runner_commit`` (the running code) so neither masquerades as the other.
    Absent (e.g. a local dev run) => an explicit unavailable marker.
    """
    return (os.environ.get("PRAXIS_LAUNCH_COMMIT", "").strip()
            or "unavailable (PRAXIS_LAUNCH_COMMIT unset)")


def _image_digest() -> str:
    """Best-effort container image digest for result-JSON provenance.

    ``PRAXIS_IMAGE_DIGEST`` is set by matrix_launch/matrix_refill and reaches
    this runner subprocess because docker/entrypoint.py launches it with
    ``env=dict(os.environ, ...)``. Absent (e.g. a local dev run) => an explicit
    unavailable marker rather than a silent empty string.
    """
    return (os.environ.get("PRAXIS_IMAGE_DIGEST", "").strip()
            or "unavailable (PRAXIS_IMAGE_DIGEST unset)")


def _cpu_model() -> str:
    """First 'model name' line from /proc/cpuinfo; 'unavailable' on any failure."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 — provenance must never fail the run
        pass
    return "unavailable"


def _host_provenance_fields() -> dict:
    """Host CPU context for result-JSON provenance.

    Records the CPU model, logical CPU count, and torch thread count so a
    reproducibility claim can be tied to the host that produced it. Every field
    is best-effort — a failure degrades that field, never the run.
    """
    try:
        import torch  # noqa: PLC0415 — torch is a heavy import, keep it local
        torch_threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch_threads = None
    return {
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch_threads,
    }


def summarize_attack_recall(trajectory: list[dict], *, require: bool = False) -> dict:
    """mean/final attack_recall for the result summary (Stage-F).

    Built the same way as mean_f1/final_f1, but with loud completeness checks so
    a corrupted parse chain never passes silently:

    - MIXED presence (some rounds carry ``attack_recall``, some don't, in a
      non-empty trajectory) always raises ``ValueError`` — partial per-class
      instrumentation is never a legitimate state.
    - ``require=True`` (the in-run call): a non-empty trajectory with NO
      per-class fields also raises, because this runner's own image always emits
      per-class eval lines, so absence means the parse failed.
    - ``require=False`` (offline replay of legacy logs): all-absent still yields
      ``None`` for both, keeping the summary schema back-compatible.
    - Empty trajectory: ``None`` for both even under ``require=True`` (other
      summary machinery already screams on empty trajectories).
    """
    if not trajectory:
        return {"mean_attack_recall": None, "final_attack_recall": None}
    present = sum("attack_recall" in t for t in trajectory)
    if present == len(trajectory):
        mean_ar = sum(t["attack_recall"] for t in trajectory) / len(trajectory)
        final_ar = trajectory[-1]["attack_recall"]
        return {"mean_attack_recall": mean_ar, "final_attack_recall": final_ar}
    if present > 0:
        missing = len(trajectory) - present
        raise ValueError(
            f"attack_recall present on only {present}/{len(trajectory)} rounds "
            f"({missing} missing): partial per-class instrumentation means the "
            "parse chain is corrupted, never a legitimate state."
        )
    # present == 0 (all rounds lack per-class fields).
    if require:
        raise ValueError(
            f"attack_recall absent on all {len(trajectory)} rounds: this "
            "runner's image always emits per-class eval lines, so absence "
            "means the eval-log parse failed."
        )
    return {"mean_attack_recall": None, "final_attack_recall": None}


# Cache provenance values at import time — the git HEAD SHA, launch env vars,
# installed Flower version, and host CPU context cannot change mid-process, so
# running these in every run_one() call is wasted subprocess/metadata overhead.
_RUNNER_COMMIT = _runner_commit()
_LAUNCH_COMMIT = _launch_commit()
_IMAGE_DIGEST = _image_digest()
_FLWR_VERSION = _flwr_version()
_HOST_PROVENANCE = _host_provenance_fields()


def parse_eval_trajectory(output: str) -> list[dict]:
    """Extract per-round metrics from ScenarioStrategy eval output.

    F1/Acc/Loss are always present. Prec/Rec are appended by the new-image eval
    line and captured when present (older logs omit them) so the trajectory
    carries all five live-contract metrics for the fallback replay ().

    Per-class benign/attack metrics (AttP/AttR/AttF1/BenP/BenR/BenF1) are a
    further optional suffix (Stage-F). Both optional groups are
    independent, so a legacy or Prec/Rec-only line parses exactly as before —
    the new per-class keys are simply absent, never fabricated."""
    trajectory = []
    pattern = (
        r"\[ScenarioStrategy\] Round (\d+) eval: "
        r"F1=([\d.]+) Acc=([\d.]+) Loss=([\d.]+)"
        r"(?: Prec=([\d.]+) Rec=([\d.]+))?"
        r"(?: AttP=([\d.]+) AttR=([\d.]+) AttF1=([\d.]+)"
        r" BenP=([\d.]+) BenR=([\d.]+) BenF1=([\d.]+))?"
    )
    for match in re.finditer(pattern, output):
        entry = {
            "round": int(match.group(1)),
            "f1": float(match.group(2)),
            "accuracy": float(match.group(3)),
            "loss": float(match.group(4)),
        }
        if match.group(5) is not None:
            entry["precision"] = float(match.group(5))
            entry["recall"] = float(match.group(6))
        if match.group(7) is not None:
            entry["attack_precision"] = float(match.group(7))
            entry["attack_recall"] = float(match.group(8))
            entry["attack_f1"] = float(match.group(9))
            entry["benign_precision"] = float(match.group(10))
            entry["benign_recall"] = float(match.group(11))
            entry["benign_f1"] = float(match.group(12))
        trajectory.append(entry)
    return trajectory


_ALIE_ACTIVE_MARKER_RE = re.compile(
    r"\[Integrity\] round=(\d+) alie_active=1 n_alie=\d+"
)


def parse_alie_round_set(output: str) -> set[int]:
    """Return server_round numbers where ALIE was active that round.

    Primary signal: the structured ``[Integrity] round=R alie_active=1`` marker
    emitted by ScenarioStrategy.configure_fit. The marker uses scenario_round
    (= server_round - 1), so we add 1 to align with the trajectory which is
    keyed on server_round.

    Falls back to detecting any ``_new`` in the legacy ``(attacking: [...])``
    print so that legacy upper-slot scenarios still parse.
    """
    rounds: set[int] = set()
    for m in _ALIE_ACTIVE_MARKER_RE.finditer(output):
        scenario_round = int(m.group(1))
        rounds.add(scenario_round + 1)  # convert to server_round for trajectory lookup
    if rounds:
        return rounds
    # Fallback for legacy stdout shape
    legacy = re.compile(
        r"Round (\d+) \(scenario R(\d+)\): \d+/\d+ clients selected \(attacking: \[([^\]]*)\]\)"
    )
    for m in legacy.finditer(output):
        attacking = m.group(3)
        if "_new" in attacking:
            rounds.add(int(m.group(1)))  # server_round (group 1) for legacy format
    return rounds


def _persist_final_model(strategy_obj, dataset_name: str, model_path: Path) -> bool:
    """Best-effort persistence of the final global model (req 6, models).

    Ground-truthed before adding this: nothing in flowerfl/ or this script
    previously saved final model weights anywhere — only per-client
    state_dicts flow through get_weights/set_weights during training, and
    they're never written to disk. This reads the LAST aggregated
    Parameters PluggableStrategy now remembers (see
    flowerfl/byzantine_defense.py's ``_last_aggregated_parameters``,
    additive bookkeeping with zero effect on the FL aggregation math),
    reconstructs the model architecture via flowerfl.task.create_model, and
    torch.saves its state_dict (tiny MLP, low tens of KB) next to the
    per-run result JSON.

    Gated: this is a nice-to-have artifact, not part of the experimental
    record, so ANY failure (missing parameters, shape mismatch, unknown
    dataset, disk error) is caught, printed as a warning, and returns
    False — it must never fail the run.
    """
    try:
        last_params = getattr(strategy_obj, "_last_aggregated_parameters", None)
        if last_params is None:
            return False
        import torch
        from flwr.common import parameters_to_ndarrays
        from flowerfl.task import create_model, set_weights

        ndarrays = parameters_to_ndarrays(last_params)
        net = create_model(dataset_name)
        set_weights(net, ndarrays)
        torch.save(net.state_dict(), model_path)
        return True
    except Exception as e:
        print(f"[model-save] WARN: failed to persist final global model: {e}")
        return False


def run_one(
    config_label: str,
    strategy: str,
    use_cs: bool,
    seed: int,
    rounds: int = NUM_ROUNDS,
    timeout: int = 7200,
    scenario_path: str = SCENARIO_PATH,
    max_per_client: int = 0,
    extra_run_config: dict | None = None,
    output_dir: Path | None = None,
    optimizer_state: str = "reset",
) -> dict:
    """Run a single Flower experiment for one (config, seed) pair.

    Builds a run_config dict (see _build_run_config), runs the two integrity
    assertions that can be checked before launch (A1 lr-drift, A2
    optimizer_state self-consistency), then calls flwr.simulation.run_simulation()
    IN-PROCESS (not via subprocess/CLI) with stdout captured for trajectory
    parsing. After the run, checks the remaining integrity assertions (A3,
    A3.b, A4, A5, A6 — see module docstring), streams metrics to MLflow if
    launched under `praxis exp launch`, and writes the per-run result JSON.

    If a cached result JSON already exists at the expected path with a
    non-empty trajectory and return_code == 0, that cached result is
    returned immediately without re-running (see the cache check below).

    Args:
        config_label: Human-readable config name (e.g. "Krum+CS").
        strategy: Flower strategy string passed to run-config.
        use_cs: Whether to include cold-start plugin run-config keys.
        seed: RNG seed for this run.
        rounds: Number of server rounds.
        timeout: NOT CURRENTLY ENFORCED — accepted for interface
            compatibility but unused in the body below. run_simulation()
            is called directly in-process (see comment further down), not
            via a subprocess, so there is no watchdog that would kill a
            hung simulation at this timeout. A genuinely hung run must be
            killed externally (e.g. by the launching `praxis exp launch`
            process or by the operator).
        scenario_path: Path to scenario JSON (relative to project root).
        max_per_client: Max samples per client (0 = use default 5000; >0 overrides).
        extra_run_config: Additional key=value pairs merged into run-config.
        output_dir: If set, write per-run JSON here instead of LOG_DIR.
        optimizer_state: "reset" or "persistent" — see module docstring.

    Returns:
        The result dict written to json_path (see module docstring's "Output
        schema" section) — either freshly computed, or the cached one if a
        valid prior result already existed at this path.
    """
    exp_name = f"phase4_flower__{config_label.replace('+', '_').lower()}__seed{seed}"
    log_path = LOG_DIR / f"{exp_name}.log"
    # Use output_dir for result JSON if provided, else LOG_DIR
    result_dir = output_dir if output_dir is not None else LOG_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / f"{exp_name}.json"

    # Reuse a cached result only when its SMOTE arm matches this run's;  otherwise rerun, rotating any stale signal log first so the
    # append-only logger doesn't mix arms.
    cached = _reuse_cached_or_rotate(
        json_path, extra_run_config, scenario_path, strategy, seed, optimizer_state, exp_name
    )
    if cached is not None:
        return cached

    print(f"    {exp_name}: running...", end=" ", flush=True)
    t0 = time.time()

    # Build the run_config dict that mirrors what `flwr run --run-config` passes.
    # Reads learning-rate from data/hparams_locked.json per optimizer_state mode.
    run_config_dict = _build_run_config(
        strategy=strategy,
        scenario_path=scenario_path,
        rounds=rounds,
        seed=seed,
        optimizer_state=optimizer_state,
        use_cs=use_cs,
        max_per_client=max_per_client,
    )

    if extra_run_config:
        run_config_dict.update(extra_run_config)

    # Integrity assertion A1: recorded lr matches locked hparams (spec § 4.9).
    _assert_lr_matches_locked(
        recorded_lr=float(run_config_dict["learning-rate"]),
        optimizer_state=optimizer_state,
    )

    # Integrity assertion A2: recorded optimizer_state matches CLI flag (spec § 4.9).
    _assert_optimizer_state_self_consistent(
        recorded_state=str(run_config_dict["optimizer-state"]),
        cli_state=optimizer_state,
    )

    # Image v9: prewarm the node-local resample DISK cache in the DRIVER before
    # spinning up Ray. Populates the cache for every partition (so workers do
    # 100% disk reads, immune to ActorPool's no-affinity scheduling) AND emits
    # the per-client [SMOTE] provenance records from the driver stdout (which
    # reaches CloudWatch under log_to_driver=False). No-op when SMOTE is off.
    # prewarm_smote_text is folded into full_output below so the captured-stdout
    # provenance parser sees the driver-emitted records (workers stay silent —
    # their loads are disk HITS).
    prewarm_smote_text, prewarm_summary = _prewarm_resample_cache(run_config_dict)

    # persist the prewarm health into result provenance so a
    # DEGRADED run (store failures / lock timeouts / doesn't-fit skips => workers
    # recomputing every construction) is distinguishable from a fully-primed one
    # in the result JSON, not just ephemeral console output. Built only for a
    # SMOTE-enabled run; injected into whichever result dict is written below.
    resample_prewarm_block = _resample_prewarm_provenance(prewarm_summary)

    # Call run_simulation() directly — blocks until completion in Flower 1.29.
    # We create a closure-based ServerApp that injects run_config_dict into the
    # Context it receives, overriding whatever defaults came from pyproject.toml.
    # The ClientApp also needs run_config; we wrap it similarly.
    from flwr.simulation import run_simulation as _flwr_run_simulation
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.client import ClientApp
    from flwr.common import Context
    from flowerfl.server_app import server_fn as _server_fn
    from flowerfl.client_app import client_fn as _client_fn

    _rc = run_config_dict  # capture in closure
    # Mutable container so the runner can retrieve the ScenarioStrategy
    # instance after run_simulation() completes. Needed for post-hoc
    # extraction of defense_overhead / confounder_control metrics
    # (Tasks 4c.3 / 4c.4 / 4c.5).
    _strategy_holder: dict[str, object] = {}

    def _patched_server_fn(context: Context) -> ServerAppComponents:
        # Merge our run_config over the context's defaults
        patched_run_config = dict(context.run_config)
        patched_run_config.update(_rc)
        # Build a new Context with the merged config
        from flwr.common.record.recorddict import RecordDict
        patched_context = Context(
            run_id=context.run_id,
            node_id=context.node_id,
            node_config=context.node_config,
            state=RecordDict(),
            run_config=patched_run_config,
        )
        components = _server_fn(patched_context)
        # Stash the strategy reference so the runner can introspect timings,
        # scoring logs, and partition maps after the simulation completes.
        if "strategy" in _strategy_holder:
            import warnings
            warnings.warn(
                f"_patched_server_fn called more than once; strategy will be overwritten. "
                f"This may corrupt defense_overhead and confounder_control metrics. "
                f"Previous strategy was: {type(_strategy_holder['strategy']).__name__}",
                RuntimeWarning,
            )
        _strategy_holder["strategy"] = components.strategy
        return components

    def _patched_client_fn(context: Context):
        patched_run_config = dict(context.run_config)
        patched_run_config.update(_rc)
        from flwr.common.record.recorddict import RecordDict
        patched_context = Context(
            run_id=context.run_id,
            node_id=context.node_id,
            node_config=context.node_config,
            state=RecordDict(),
            run_config=patched_run_config,
        )
        return _client_fn(patched_context)

    patched_server_app = ServerApp(server_fn=_patched_server_fn)
    patched_client_app = ClientApp(client_fn=_patched_client_fn)

    captured_buf = io.StringIO()
    return_code = 0
    try:
        # Clear persistent_optimizer state between independent runs to prevent
        # cross-run leakage as a confounding variable.
        from flowerfl import persistent_optimizer as _po
        _po.clear()
        # Redirect stdout so per-round eval lines are captured for trajectory parsing.
        # Flower simulation also prints to stderr in some paths; capture both.
        with contextlib.redirect_stdout(captured_buf):
            _flwr_run_simulation(
                server_app=patched_server_app,
                client_app=patched_client_app,
                num_supernodes=NUM_SUPERNODES,
                backend_config=_BACKEND_CONFIG,
            )
    except Exception as exc:  # noqa: BLE001
        return_code = 1
        captured_buf.write(f"\nSIMULATION ERROR: {exc}\n")

    elapsed = time.time() - t0
    full_output = captured_buf.getvalue()
    # Image v9: fold the DRIVER-emitted prewarm [SMOTE] records into the parsed
    # output. On the fleet these are the ONLY [SMOTE] records that exist (workers
    # hit the disk cache and stay silent, and log_to_driver=False would suppress
    # them anyway), so provenance parsing now has a source it never had before.
    if prewarm_smote_text:
        full_output = prewarm_smote_text + "\n" + full_output
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(full_output, encoding="utf-8", errors="replace")

    # SMOTE (GWU-59): lift the per-client application records from the captured
    # data-prep stdout into per-client provenance counts + the run-level skip
    # flag + the reproducibility seed component (DESIGN.md Stage-D list).
    smote_records = parse_smote_records(full_output)

    trajectory = parse_eval_trajectory(full_output)
    _assert_trajectory_non_empty(trajectory, rounds=rounds)

    integrity_rows = _parse_participants_markers(full_output)
    _assert_integrity_markers_emitted(integrity_rows, trajectory)
    if integrity_rows:
        with open(scenario_path) as _fh:
            _scenario_dict = json.load(_fh)
        # A3 honors scenario-scheduled disconnects (S3/S4) via the declared
        # per-round participant count; falls back to the flat floor otherwise.
        _assert_participants_per_round(
            integrity_rows, num_supernodes=NUM_SUPERNODES, scenario_dict=_scenario_dict
        )
        _assert_n_malicious_per_round(integrity_rows, _scenario_dict)

    # A5: signal-log filename exec_mode check. Skip if signal logging was
    # explicitly disabled (signal-log run_config = 0/false/empty).
    signal_log_enabled = (
        str(run_config_dict.get("signal-log", "1")).lower() not in ("0", "false", "")
    )
    if signal_log_enabled:
        signal_dir = Path(__file__).resolve().parent.parent / "signals"
        _assert_signal_log_filename(
            signal_dir=signal_dir,
            scenario_name=Path(scenario_path).stem,
            defense=strategy,
            seed=seed,
            cli_optimizer_state=optimizer_state,
        )

    # Final-metric flush to MLflow if launched via `praxis exp launch`.
    #
    # Per-round metrics are now streamed LIVE from the server strategy
    # (flowerfl/scenario_strategy.py::evaluate → flowerfl.mlflow_live), so a
    # RUNNING unit already shows a live per-round trajectory and a mid-run
    # crash/Spot-reclaim keeps every completed round. This block therefore only
    # flushes the FINAL summary metrics as defense-in-depth (harmless if the
    # live hook already logged the rounds). Best-effort — a tracking failure
    # must never affect the trajectory or the on-disk/S3 result (contract
    # unchanged; spec § 3, item 10).
    mlflow_run_id = os.environ.get("PRAXIS_MLFLOW_RUN_ID")
    if mlflow_run_id and trajectory:
        try:
            import mlflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
            client = mlflow.tracking.MlflowClient()
            # Defense-in-depth trajectory replay. Per-round metrics are streamed
            # LIVE from the server strategy (flowerfl/mlflow_live.py), but a
            # DIRECT `praxis exp launch` (no Batch entrypoint to backfill the
            # trajectory afterward) has no other backstop — so any round the live
            # logger dropped on a transient error would be lost. Replaying the
            # full trajectory from the in-memory result keeps per-round metrics
            # complete (idempotent: re-logging an already-live round is a
            # harmless duplicate point; , run_phase4_flower.py:1145).
            for entry in trajectory:
                r = entry.get("round")
                if r is None:
                    continue
                for key in ("accuracy", "precision", "recall", "f1", "loss",
                            "attack_precision", "attack_recall", "attack_f1",
                            "benign_precision", "benign_recall", "benign_f1"):
                    if entry.get(key) is not None:
                        client.log_metric(mlflow_run_id, key, float(entry[key]), step=int(r))
            last = trajectory[-1]
            for key in ("f1", "accuracy", "loss"):
                if last.get(key) is not None:
                    client.log_metric(mlflow_run_id, f"final_{key}", float(last[key]))
            # SMOTE study knob (GWU-59): surface the input params when enabled so
            # a SMOTE run is filterable in MLflow (absent = incumbent).
            for pkey, pval in _smote_mlflow_params(run_config_dict, records=smote_records).items():
                client.log_param(mlflow_run_id, pkey, pval)
            # mirror ONLY the two health-critical prewarm numbers
            # into MLflow params so a degraded prime is filterable there too.
            if resample_prewarm_block is not None:
                client.log_param(mlflow_run_id, "resample_persisted",
                                 str(resample_prewarm_block["persisted"]))
                client.log_param(mlflow_run_id, "resample_compute_only",
                                 str(resample_prewarm_block["compute_only"]))
        except Exception as e:
            print(f"[mlflow] WARN: failed to flush metrics: {e}")

    alie_rounds = parse_alie_round_set(full_output)

    provenance = {
        "runner_version": RUNNER_VERSION,
        "runner_commit": _RUNNER_COMMIT,
        # launch-time git HEAD: provenance for the LAUNCH-side
        # inputs (manifest, scenario JSONs), kept distinct from runner_commit
        # (the image-baked running code) so neither masquerades as the other.
        "launch_commit": _LAUNCH_COMMIT,
        # runtime provenance: image digest + host CPU context so a
        # reproducibility claim can be pinned to a commit/image/host.
        "image_digest": _IMAGE_DIGEST,
        "code_path": "unified-flower-runner",
        "scenario_path": str(scenario_path),
        "optimizer_state": optimizer_state,
        "cs_model_path": CS_MODEL if use_cs else "",
        "flwr_version": _FLWR_VERSION,
        **_HOST_PROVENANCE,
        # true LSTM state + effective ramp (v1.6 § 2; replaces the Step-2-era
        # hardcoded "disabled" that wrote false provenance into every result)
        **tge_provenance_fields(strategy, run_config_dict, rounds),
        # declared-vs-policy defense sizing, kept distinct (PR #12 round-2 P2)
        **_defense_provenance_fields(strategy, run_config_dict),
        # SMOTE study knob (GWU-59): always declares status; absent-or-false = incumbent
        **smote_provenance_fields(run_config_dict, records=smote_records),
        # Stage-F knobs: always declared so the cache-reuse
        # identity can tell a legacy arm from a semantic / update-matched / original-
        # weight arm sharing the same sampler+target.
        **stage_f_provenance_fields(run_config_dict),
        # m1 leakage fix (normalization audit): always declares whether the
        # per-client Z-score was fit train-only (leak-free) or on all rows
        # (default, leak-on incumbent), so the cache-reuse identity can tell them
        # apart and the dissertation can read the exact arm off the artifact.
        **normalize_train_only_provenance(run_config_dict),
        # H3 post-tau-lock: which LOCKED tau/Sigma cohort scored this unit —
        # always declared (None = pre-lock / non-FP arm) so adjudication reads
        # the instrument identity off the artifact, not off launch records.
        **fp_cohort_provenance(run_config_dict),
        **fp_registry_policy_provenance(run_config_dict),
        # H4 (erratum-A E4): which evaluation population scored this unit +
        # the sealed manifest's sha256, read from the ACTUAL eval manager.
        **eval_split_provenance(
            run_config_dict,
            getattr(_strategy_holder.get("strategy"), "_eval_manager", None),
        ),
        # H4 § 7 item 1: the serving bundle's manifest sha256 + erratum-B cut
        # identity, from the ACTUAL detector plugin (null — not empty string —
        # for arms without the online detector).
        **h4_serving_provenance(_strategy_holder.get("strategy")),
        # Erratum B § B1: observe-only mode always declared (False =
        # enforcing/incumbent) — the calibration builder and the cache-reuse
        # identity read the mode off the artifact.
        **h2p_observe_provenance(run_config_dict),
        # Universal run identity (Lane C custody audit): present for ALL
        # arms; equals fingerprint_registry.run_uid on FP-bearing arms.
        **run_uid_provenance(_strategy_holder.get("strategy")),
        # GWU-61 (v8 change 3): eval holdout row-disjoint from training (default
        # True). Legacy overlapping holdout when --no-holdout-disjoint was set.
        # Durable holdout provenance (rows_excluded + size + per-class) is read
        # from the actual FixedEvalManager — the [FixedEval] stdout record never
        # reliably reaches CloudWatch, so the result JSON is the record.
        **_holdout_provenance_fields(
            run_config_dict,
            getattr(_strategy_holder.get("strategy"), "_eval_manager", None),
        ),
    }

    if not trajectory:
        print(f"FAILED ({elapsed:.0f}s, no trajectory)")
        tail = full_output[-300:] if full_output else ""
        if tail:
            print(f"      output tail: {tail}")
        result = {
            "config": config_label,
            "strategy": strategy,
            "seed": seed,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "trajectory": [],
            "error": "no trajectory parsed",
            "provenance": provenance,
        }
        if resample_prewarm_block is not None:
            result["resample_prewarm"] = resample_prewarm_block
        json_path.write_text(json.dumps(result, indent=2))
        return result

    accs = [t["accuracy"] for t in trajectory]
    final_acc = accs[-1] if accs else None
    mean_acc = sum(accs) / len(accs) if accs else None

    # Post-RMC accuracy: rounds where ALIE clients are attacking
    post_rmc = [t["accuracy"] for t in trajectory if t["round"] in alie_rounds]
    post_rmc_acc = sum(post_rmc) / len(post_rmc) if post_rmc else None

    result = {
        "config": config_label,
        "strategy": strategy,
        "use_cs": use_cs,
        "seed": seed,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "trajectory": trajectory,
        "alie_rounds": sorted(alie_rounds),
        "mean_accuracy": mean_acc,
        "final_accuracy": final_acc,
        "post_reconnect_accuracy": post_rmc_acc,
        "mean_f1": sum(t["f1"] for t in trajectory) / len(trajectory),
        "final_f1": trajectory[-1]["f1"],
        # attack-class recall summary: mean + final, same shape as
        # mean_f1/final_f1; None when the trajectory predates per-class logging.
        **summarize_attack_recall(trajectory, require=True),
        "provenance": provenance,
    }
    if resample_prewarm_block is not None:
        result["resample_prewarm"] = resample_prewarm_block

    # Task 4c.3-4c.5: enrich result with convergence + confounder_control +
    # defense_overhead blocks pulled from the captured strategy.
    strategy_obj = _strategy_holder.get("strategy")
    _enrich_result_with_metrics(result, scenario_path, strategy_obj)

    # Erratum B § B1: the observe-only calibration log rides the result JSON
    # (None — no key at all — for every non-observing unit, so the incumbent
    # result shape is byte-identical).
    observe_block = h2p_observe_block(strategy_obj)
    if observe_block is not None:
        result["h2p_observe"] = observe_block

    # req 6 (models): best-effort final-model persistence next to the result
    # JSON (see _persist_final_model docstring — never fails the run).
    model_path = result_dir / f"{exp_name}__model.pt"
    if _persist_final_model(strategy_obj, DATASET, model_path):
        result["model_path"] = str(model_path)

    json_path.write_text(json.dumps(result, indent=2))

    f1_str = f"F1={result['final_f1']:.4f}" if result.get('final_f1') is not None else "F1=?"
    print(f"{f1_str} mean_acc={mean_acc:.4f} final_acc={final_acc:.4f} ({elapsed:.0f}s)")
    return result


def _resolve_optimizer_state(args) -> str:
    """Resolve --optimizer-state and --modes alias.

    Precedence:
      1. --optimizer-state explicit → use it; if --modes also set non-default,
         error on conflict.
      2. --modes Flower → 'reset'
      3. --modes persistent_optimizer → 'persistent'
      4. Neither → default 'reset'
    """
    mode_to_state = {"Flower": "reset", "persistent_optimizer": "persistent"}
    modes_set = args.modes and args.modes != "Flower"  # 'Flower' is the legacy default
    if args.optimizer_state is not None:
        if modes_set:
            implied = mode_to_state.get(args.modes)
            if implied != args.optimizer_state:
                raise SystemExit(
                    f"--optimizer-state {args.optimizer_state} conflicts with "
                    f"--modes {args.modes}; specify only one.")
        return args.optimizer_state
    if args.modes in mode_to_state:
        return mode_to_state[args.modes]
    return "reset"


def main() -> int:
    """CLI entry point: parse args, run every (config, seed) pair, write summary.

    Iterates configs in the outer loop and seeds in the inner loop, calling
    run_one() for each pair and incrementally writing the partial summary
    JSON after every run (so a crash/interrupt partway through a sweep still
    leaves usable partial results on disk). Prints a final comparison table
    to stdout and writes the same data to
    <output_dir or RESULTS_DIR>/phase4_flower_paired_comparison.json.

    Not invoked directly for real experiments — see module docstring;
    `praxis exp launch` invokes this same entry point but wraps it with git
    tag + MLflow provenance.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default=",".join(str(s) for s in DEV_SEEDS))
    p.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    p.add_argument("--configs", default="all",
                   help="Comma-separated config labels or 'all'. Supported: "
                        + ", ".join(SUPPORTED_CONFIGS))
    p.add_argument("--modes", default="Flower",
                   help="Legacy single mode label ('Flower' or 'persistent_optimizer'), "
                        "mapped to the corresponding --optimizer-state value by "
                        "_resolve_optimizer_state() below. This runner supports BOTH "
                        "exec modes directly (unified since the 2026-05-15 runner "
                        "unification, see docs/METHODOLOGY_LOG.md) — prefer "
                        "--optimizer-state explicitly; --modes exists for CLI "
                        "backward compatibility.")
    p.add_argument("--scenario", default=SCENARIO_PATH,
                   help="Path to scenario JSON (relative to project root)")
    p.add_argument("--max-per-client", type=int, default=0,
                   help="Max samples per client (0 = use default 5000)")
    p.add_argument("--reporting-split", default="val",
                   help="Reporting split: val or test (passed through for "
                        "informational logging; actual split is controlled by "
                        "the dataset config)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for per-run output JSONs (default: LOG_DIR)")
    p.add_argument("--optimizer-state", choices=["reset", "persistent"], default=None,
                   help="Client-side Adam state lifetime. 'reset' = Flower default "
                        "(fresh optimizer each round); 'persistent' = Adam m/v survive "
                        "across rounds for the same flower_cid. If unspecified, falls "
                        "back to --modes alias (Flower→reset, persistent_optimizer→persistent).")
    add_smote_cli_args(p)  # GWU-59: fleet-reachable SMOTE knob (default OFF)
    add_stage_f_cli_args(p)  # GWU-59 Stage-F: update-match / weight-mode / semantic (default OFF)
    add_leakage_cli_args(p)  # m1 leakage fix: --normalize-train-only (default OFF)
    add_fp_cohort_cli_args(p)  # H3 post-tau-lock cohort declaration (default: undeclared)
    add_fp_registry_policy_cli_args(p)  # H3 candidate pool (default: flag_gated incumbent)
    add_eval_split_cli_args(p)  # H4 sealed-test evaluator (erratum-A E4; default: legacy)
    add_h2p_cli_args(p)  # erratum-B observe-only + cuts-version (defaults: incumbent)
    # GWU-61 (v8 change 3): server holdout is row-disjoint from training by
    # default; --no-holdout-disjoint restores the legacy overlapping holdout for
    # reproducing pre-v8 numbers.
    p.add_argument("--holdout-disjoint", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Exclude each client's TRAIN rows from the eval holdout "
                        "pool (default on). --no-holdout-disjoint = legacy overlap.")
    args = p.parse_args()

    optimizer_state = _resolve_optimizer_state(args)

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = args.output_dir

    # Resolve config list: support both legacy CONFIGS tuples and new SUPPORTED_CONFIGS
    if args.configs == "all":
        config_labels = [c[0] for c in CONFIGS]
    else:
        config_labels = [c.strip() for c in args.configs.split(",")]

    # Validate config names
    unknown = [c for c in config_labels if c not in SUPPORTED_CONFIGS]
    if unknown:
        print(f"ERROR: Unknown config(s): {unknown}. Supported: {SUPPORTED_CONFIGS}",
              file=sys.stderr)
        return 1

    # Host provenance line for CloudWatch: surfaces the CPU model of
    # the machine that ran this unit, since result JSONs are the durable record
    # but CloudWatch is where a fleet-wide host comparison is easiest to eyeball.
    print(f"[HOST] cpu_model={_HOST_PROVENANCE['cpu_model']!r} "
          f"cpu_count_logical={_HOST_PROVENANCE['cpu_count_logical']} "
          f"torch_num_threads={_HOST_PROVENANCE['torch_num_threads']} "
          f"image_digest={_IMAGE_DIGEST!r} runner_commit={_RUNNER_COMMIT!r} "
          f"launch_commit={_LAUNCH_COMMIT!r}")

    print(f"\n{'='*80}")
    print(f"  Phase 4 Track C — Paired comparison in Flower mode")
    print(f"  Scenario: {args.scenario}, Rounds: {args.rounds}")
    print(f"  Federation: {FEDERATION}, Dataset: {DATASET}")
    print(f"  Configs: {config_labels}")
    print(f"  Seeds: {seeds}")
    print(f"  Max-per-client: {args.max_per_client or 5000} (0=default)")
    print(f"  Reporting split: {args.reporting_split}")
    print(f"  Output dir: {output_dir or LOG_DIR}")
    print(f"{'='*80}")

    all_results: dict[str, list[dict]] = {}

    for label in config_labels:
        strategy_token, plugin_config = build_strategy_for_config(label)
        strategy_str = strategy_token.__name__
        use_cs = plugin_config.get("cs_enabled", False)
        # Extra run-config overrides from plugin_config (exclude cs_enabled sentinel)
        extra = {k: v for k, v in plugin_config.items() if k != "cs_enabled"}
        # Map underscore TGE/TGE′ keys -> hyphenated Flower run-config keys.
        extra = _hyphenate_tge_extra(extra)
        # GWU-59: merge the SMOTE run-config overrides (empty dict when the flag
        # is off, so extra is unchanged and the run is byte-identical to today).
        smote_extra = smote_run_config_from_cli(
            args.smote_enabled, args.smote_variant, args.smote_target
        )
        if smote_extra:
            extra = {**extra, **smote_extra}
        # Stage-F (§4/§5/§6): merge the update-match / weight-mode / semantic-target
        # overrides (empty dict at defaults, so extra is unchanged and the run is
        # byte-identical to the incumbent).
        stage_f_extra = stage_f_run_config_from_cli(
            args.update_match, args.weight_mode, args.smote_semantic_target
        )
        if stage_f_extra:
            extra = {**extra, **stage_f_extra}
        # m1 leakage fix: merge the normalize-train-only override (empty dict when
        # the flag is off, so extra is unchanged and the run is byte-identical to
        # the incumbent leak-on pipeline).
        leakage_extra = leakage_run_config_from_cli(args.normalize_train_only)
        if leakage_extra:
            extra = {**extra, **leakage_extra}
        # H3 post-tau-lock cohort declaration: merge the fp-cohort override
        # (empty dict when undeclared, so extra is unchanged and pre-lock /
        # non-FP runs are byte-identical to the incumbent).
        fp_cohort_extra = fp_cohort_run_config_from_cli(args.fp_cohort)
        if fp_cohort_extra:
            extra = {**extra, **fp_cohort_extra}
        # H3 registry candidate policy: absent leaves the run-config untouched,
        # so the incumbent flag-gated instrument stays byte-identical.
        fp_policy_extra = fp_registry_policy_run_config_from_cli(
            args.fp_registry_policy
        )
        if fp_policy_extra:
            extra = {**extra, **fp_policy_extra}
        # H4 sealed-test evaluator (erratum-A E4). The four H2P+* tokens BAKE
        # eval-split=sealed_test into their configs (a frozen operating
        # condition, not a knob); a CLI value that CONTRADICTS a baked one is
        # a config error and refuses loudly rather than silently overriding
        # the frozen evaluator. Reused arms carry no baked value and take the
        # CLI declaration verbatim.
        eval_split_extra = eval_split_run_config_from_cli(args.eval_split)
        if eval_split_extra:
            baked = extra.get("eval-split")
            if baked is not None and baked != eval_split_extra["eval-split"]:
                print(
                    f"ERROR: config {label!r} freezes eval-split={baked!r} "
                    f"but --eval-split={eval_split_extra['eval-split']!r} "
                    f"was passed; refusing the contradiction (erratum-A E4).",
                    file=sys.stderr,
                )
                return 1
            extra = {**extra, **eval_split_extra}
        # Erratum B: observe-only + cuts-version knobs (empty dict at
        # defaults, so extra is unchanged and the run is byte-identical to
        # the incumbent). No baked-value contradiction check needed: no
        # config label bakes either key (the EXP-062 calibration fleet
        # declares them via run_extras; the sealed fleet pins cuts-version
        # in its design doc).
        h2p_extra = h2p_run_config_from_cli(
            args.h2p_observe_only, args.h2p_cuts_version
        )
        if h2p_extra:
            extra = {**extra, **h2p_extra}
        # GWU-61: thread the disjoint-holdout flag into run-config so server_app
        # builds the eval manager in the requested mode and the result JSON
        # provenance records it. Always present (default True).
        extra = {**extra, "holdout-disjoint": args.holdout_disjoint}

        print(f"\n  --- {label} ({strategy_str}) ---")
        config_results = []
        for seed in seeds:
            result = run_one(
                label, strategy_str, use_cs, seed,
                rounds=args.rounds,
                scenario_path=args.scenario,
                max_per_client=args.max_per_client,
                extra_run_config=extra if extra else None,
                output_dir=output_dir,
                optimizer_state=optimizer_state,
            )
            config_results.append(result)
            # Save partial progress to --output-dir (or legacy RESULTS_DIR
            # if --output-dir not provided). Never clobber RESULTS_DIR when
            # the caller has supplied a target directory.
            summary_dir = output_dir or RESULTS_DIR
            partial_path = summary_dir / "phase4_flower_paired_comparison.json"
            all_results[label] = config_results
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(all_results, indent=2))

    # Final summary
    print(f"\n{'='*80}")
    print(f"  PHASE 4 FLOWER PAIRED COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':<18} {'Mean Acc':>10} {'Final Acc':>10} {'Post-RMC':>10} "
          f"{'Final F1':>10}")
    print(f"  {'-'*64}")
    import numpy as np
    for label in config_labels:
        results = all_results.get(label, [])
        valid = [r for r in results if r.get("trajectory")]
        if not valid:
            print(f"  {label:<18} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        mean_acc = np.mean([r["mean_accuracy"] for r in valid
                            if r["mean_accuracy"] is not None])
        final_acc = np.mean([r["final_accuracy"] for r in valid
                             if r["final_accuracy"] is not None])
        post_rmc_vals = [r["post_reconnect_accuracy"] for r in valid
                         if r.get("post_reconnect_accuracy") is not None]
        post_rmc = np.mean(post_rmc_vals) if post_rmc_vals else None
        final_f1 = np.mean([r["final_f1"] for r in valid if r.get("final_f1") is not None])
        post_str = f"{post_rmc:.4f}" if post_rmc is not None else "N/A"
        print(f"  {label:<18} {mean_acc:>10.4f} {final_acc:>10.4f} "
              f"{post_str:>10} {final_f1:>10.4f}")

    summary_dir = output_dir or RESULTS_DIR
    out_path = summary_dir / "phase4_flower_paired_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

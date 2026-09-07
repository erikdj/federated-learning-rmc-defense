"""Pure check logic for scripts/audit_calibration_gate.py.

Every function here takes already-loaded data (dicts / lists of dicts) and
returns CheckResult(s) -- no file or network I/O, so it is trivially unit
testable. I/O (reading result JSON / signal JSONL from S3 or local disk) lives
in scripts/_calibration_gate_store.py.

Ground truth citations (read directly from source, not assumed -- see
scripts/audit_calibration_gate.py module docstring for the full surprise list):
  - result JSON schema: scripts/run_phase4_flower.py run_one() ~L1028-1082.
    There is NO top-level "mode" field and NO persisted lr/local-epochs
    field anywhere in the result JSON -- only result["provenance"]["optimizer_state"]
    (short form "persistent"/"reset", NOT "persistent_optimizer") and
    result["seed"]/["final_accuracy"]/["elapsed_seconds"].
  - TGE provenance fields: scripts/run_phase4_flower.py tge_provenance_fields()
    ~L303-353.
  - signal-log row schema (v5, adds aggregation_coefficient + the H3 re-entry
    event contract; v4 adds tge_ema_score; v3 still accepted -- readers are
    version-GATED, never migrated, so historical v4/v3 logs keep parsing
    byte-identically and no v4 row is reinterpreted):
    flowerfl/signal_logger.py SignalLogger + the per-row tge_* fields built in
    flowerfl/scenario_strategy.py ScenarioStrategy._maybe_log_signals() ~L612-645.
  - participation floor: scripts/run_phase4_flower.py A3_PARTICIPATION_FLOOR_FRAC
    = 0.95, NUM_SUPERNODES = 21 (20 client supernodes + 1 server slot).
"""
from __future__ import annotations

import inspect
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from _calibration_gate_types import CheckResult, UnitRef

REPO_ROOT = Path(__file__).resolve().parent.parent
HPARAMS_LOCKED_PATH = REPO_ROOT / "data" / "hparams_locked.json"

# run_phase4_flower.py::_resolve_optimizer_state L1106 -- the ONLY place this
# mapping is defined; duplicated here (read-only comparison, not reusable as
# an import without pulling in the whole heavy runner module at CLI import
# time -- see docstring in scripts/audit_calibration_gate.py).
MODE_TO_OPTIMIZER_STATE = {"Flower": "reset", "persistent_optimizer": "persistent"}

# scripts/run_phase4_flower.py::_load_locked_lr L398 -- maps the runner's
# short-form optimizer_state to the hparams_locked.json section name.
OPTIMIZER_STATE_TO_HPARAMS_SECTION = {"reset": "flower_reset", "persistent": "persistent_optimizer"}

# Candidate (result-level, alias) hparam key names this audit will look for.
# GROUND-TRUTH SURPRISE: as of this writing, run_phase4_flower.py's result
# JSON persists NEITHER of these anywhere (checked run_one() end to end) --
# lr/epochs are only enforced pre-hoc by _assert_lr_matches_locked() at
# generation time and never written to the artifact. This check is written
# to work correctly IF a future runner change starts persisting hparams
# (e.g. under a "hparams" or "run_config" block), and reports a loud SKIP
# (not a silent pass) against real EXP-005c data today.
_HPARAM_ALIASES = {
    "lr": ("lr", "learning-rate", "learning_rate"),
    "local_epochs": ("local_epochs", "local-epochs", "epochs"),
}
_HPARAM_CONTAINERS = ("hparams", "run_config", "provenance")  # + top-level

# scripts/run_phase4_flower.py L363-364
A3_PARTICIPATION_FLOOR_FRAC = 0.95
NUM_CLIENT_SUPERNODES = 20  # NUM_SUPERNODES=21 minus 1 server slot (spec sec 4.8)

# flowerfl/scenario_strategy.py::_maybe_log_signals L635-642. tge_threshold is
# deliberately NOT in this list: it is a round-level constant applied to
# EVERY row once a TGE plugin is active (tge_threshold_seen), independent of
# whether that row's own client was individually scored -- see L616-620 and
# L641. Treating it as "must be null when unscored" would fail every real
# Krum+TGE signal log.
TGE_FIELDS_NULL_WHEN_UNSCORED = (
    "tge_score", "tge_gbdt_score", "tge_lstm_score",
    # tge_ema_score — the TGE′ bank's EMA-reputation leg (schema v4, GWU-53).
    # Null when the row's client was unscored, exactly like the other TGE
    # fields: the cohort-observation hook updates a filtered client's EMA
    # internally but never emits it as a scored row, so an unscored row
    # carrying a non-null EMA score is a population-leak bug. On pre-v4 logs
    # the field is simply absent (-> None), so this check passes there too.
    "tge_ema_score",
    "tge_tenure", "tge_phase", "tge_gate", "tge_decision",
)


_ENTRYPOINT_MOD = None


def _load_container_entrypoint():
    """Load docker/entrypoint.py by FILE PATH (cached). The local package is
    named 'docker', which collides with the PyPI docker SDK (an mlflow dependency
    mlflow imports and caches first), so `from docker.entrypoint import ...`
    resolves to the SDK in the installed CLI and raises ModuleNotFoundError
    (pytest masks it because the repo root is on sys.path). Loading by path
    bypasses the name."""
    global _ENTRYPOINT_MOD
    if _ENTRYPOINT_MOD is None:
        import importlib.util as _ilu
        import os as _os
        ep = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "docker", "entrypoint.py",
        )
        spec = _ilu.spec_from_file_location("_praxis_container_entrypoint", ep)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ENTRYPOINT_MOD = mod
    return _ENTRYPOINT_MOD


def defense_token_for(config: str) -> str:
    """Signal-log defense token for a config label -- reuses docker/entrypoint.py's
    verified _DEFENSE_TOKEN mapping (loaded by file path; see
    _load_container_entrypoint for why a plain import fails in the CLI)."""
    return _load_container_entrypoint().defense_token(config)


def tge_runtime_defaults() -> dict[str, Any]:
    """Live defaults for the TGE plugin/rule, mirroring
    scripts/run_phase4_flower.py::tge_provenance_fields()'s own approach of
    reading them via inspect.signature so this can never drift from the
    deployed code."""
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    from rmc.tg_ensemble import TenureGatedDecisionRule

    plugin_params = inspect.signature(TGEnsemblePlugin.__init__).parameters
    rule_params = inspect.signature(TenureGatedDecisionRule.__init__).parameters
    return {
        "warmup_rounds": plugin_params["warmup_rounds"].default,
        "ramp_rounds": plugin_params["ramp_rounds"].default,
        "threshold": plugin_params["threshold"].default,
        "min_tenure": rule_params["min_tenure"].default,
    }


def krum_dynamic_keep(n: int) -> int:
    """Multi-Krum's DYNAMIC per-round survivor count (methodology v1.19).

    THE single source for the auditor's keep expectation; the formula mirrors
    the deployed plugin exactly and must never be re-derived elsewhere:
      - f = ceil(n/2) - 1  -- KrumDefensePlugin._effective_f
        (flowerfl/byzantine_defense.py:217-222, dynamic_f=True branch; April
        anchor formula, reproduce_szelag.py:388,739)
      - keep m = max(1, n - f - 2)  -- KrumDefensePlugin.filter_updates
        dynamic branch (flowerfl/byzantine_defense.py:341-345)
    Documented operating points: n=20 -> f=9, keep=9; n=11 (S3/S4 disconnect
    round) -> f=5, keep=4. All Scenario*Krum* strategies construct the plugin
    with dynamic_f=True (flowerfl/server_app.py ScenarioKrum / ScenarioKrumTGE
    branches). A cross-check test drives the REAL plugin's score_updates +
    filter_updates and compares survivor counts against this helper
    (tests/test_audit_calibration_gate.py).
    """
    f = math.ceil(n / 2) - 1
    return max(1, n - f - 2)


def _runner_module():
    """Lazy import of scripts.run_phase4_flower (module-level cost: subprocess
    git rev-parse + flowerfl.result_metrics import). Imported under the
    package-qualified name so it shares the module instance with the rest of
    the test suite (tests/test_scenario_defense_sizing.py uses the same form)."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import run_phase4_flower as rp4

    return rp4


def _scenario_json_path(scenario: str) -> Path:
    # entrypoint.py builds --scenario as "<PRAXIS_SCENARIO_DIR>/<unit.scenario>.json"
    # with default dir "rmc/scenarios" (docker/entrypoint.py main / matrix_launch.py)
    return REPO_ROOT / "rmc" / "scenarios" / f"{scenario}.json"


def expected_defense_sizing(scenario: str) -> tuple[int, int] | None:
    """(declared adversaries, defense cohort) the runner derives for this
    scenario -- computed by the RUNNER'S OWN helper
    (run_phase4_flower.py::_scenario_defense_sizing: peak per-round declared
    malicious count, peak per-round declared participants), so the audit
    expectation can never drift from what _build_run_config emits as
    num-malicious / defense-cohort-size. None if the scenario JSON is
    missing/unreadable or declares no participants. S4_full_mix -> (9, 20)."""
    rp4 = _runner_module()
    return rp4._scenario_defense_sizing(str(_scenario_json_path(scenario)))


def krum_f_policy_string() -> str:
    """The deployed-Krum sizing-policy provenance string, imported from
    run_phase4_flower.py::KRUM_F_POLICY (currently "dynamic ceil(n/2)-1")."""
    return _runner_module().KRUM_F_POLICY


def scenario_declared_participants(scenario: str) -> dict[int, int] | None:
    """Scenario-declared participants per scenario_round, via the RUNNER'S OWN
    A3 helper (run_phase4_flower.py::_scenario_declared_participants_per_round).
    None when the scenario JSON is not available locally (the participation
    check then falls back to the flat floor)."""
    path = _scenario_json_path(scenario)
    if not path.is_file():
        return None
    rp4 = _runner_module()
    declared = rp4._scenario_declared_participants_per_round(json.loads(path.read_text()))
    return declared or None


# ---------------------------------------------------------------------------
# Check 2: baseline instrumentation (delegates to audit_run_instrumentation.audit)
# ---------------------------------------------------------------------------


def check_baseline_instrumentation(unit: UnitRef, result_path: Path, signal_path: Path) -> CheckResult:
    import audit_run_instrumentation as ari

    gaps = ari.audit(unit.config, str(result_path), str(signal_path))
    return CheckResult(
        name="baseline_instrumentation",
        unit_id=unit.unit_id,
        passed=not gaps,
        required=True,
        detail="; ".join(gaps) if gaps else "fully instrumented",
    )


# ---------------------------------------------------------------------------
# Check 3: result provenance
# ---------------------------------------------------------------------------


def check_provenance_optimizer_state(unit: UnitRef, result: dict) -> CheckResult:
    observed = (result.get("provenance") or {}).get("optimizer_state")
    passed = observed == unit.expected_optimizer_state
    return CheckResult(
        "provenance.optimizer_state", unit.unit_id, passed, True,
        f"expected {unit.expected_optimizer_state!r}, got {observed!r}"
        if not passed else f"optimizer_state={observed!r}",
    )


def check_provenance_seed(unit: UnitRef, result: dict) -> CheckResult:
    observed = result.get("seed")
    passed = observed == unit.seed
    return CheckResult(
        "provenance.seed", unit.unit_id, passed, True,
        f"expected seed={unit.seed}, got {observed!r}" if not passed else f"seed={observed}",
    )


def _find_hparam(result: dict, aliases: tuple[str, ...]) -> Any:
    containers: list[dict] = [result]
    for key in _HPARAM_CONTAINERS:
        sub = result.get(key)
        if isinstance(sub, dict):
            containers.append(sub)
    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    return _MISSING


_MISSING = object()


def check_provenance_hparams(unit: UnitRef, result: dict, locked_hparams: dict) -> CheckResult:
    section_name = OPTIMIZER_STATE_TO_HPARAMS_SECTION.get(unit.expected_optimizer_state)
    if section_name is None or section_name not in locked_hparams:
        return CheckResult(
            "provenance.hparams", unit.unit_id, False, True,
            f"no locked-hparams section for optimizer_state={unit.expected_optimizer_state!r}",
        )
    locked_section = locked_hparams[section_name]

    found: dict[str, Any] = {}
    mismatches: list[str] = []
    for canonical, aliases in _HPARAM_ALIASES.items():
        if canonical not in locked_section:
            continue
        observed = _find_hparam(result, aliases)
        if observed is _MISSING:
            continue
        found[canonical] = observed
        expected = locked_section[canonical]
        # Tolerant conversion (round-7 sweep, 3567232374 family): a
        # non-numeric recorded value must be a loud mismatch, not a float()
        # ValueError/TypeError that aborts the gate.
        try:
            matches = float(observed) == float(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(f"{canonical}: expected {expected!r}, got {observed!r}")

    if not found:
        # GROUND-TRUTH GAP (see module docstring): real run_phase4_flower.py
        # result JSONs do not carry lr/epochs at all. Loud, non-blocking SKIP.
        return CheckResult(
            "provenance.hparams", unit.unit_id, True, False,
            "SKIPPED: no hparam keys (lr/local_epochs or aliases) found in result "
            f"under {_HPARAM_CONTAINERS + ('<top-level>',)}; run_phase4_flower.py's "
            "result JSON does not persist hparams post-hoc (only enforces them at "
            "generation time via _assert_lr_matches_locked) -- cannot verify from "
            "the artifact alone",
        )
    passed = not mismatches
    return CheckResult(
        "provenance.hparams", unit.unit_id, passed, True,
        "; ".join(mismatches) if mismatches else f"hparams match locked ({found})",
    )


def check_provenance_tge_fields(unit: UnitRef, result: dict) -> CheckResult:
    prov = result.get("provenance") or {}
    is_tge_config = "TGE" in unit.config
    if not is_tge_config:
        problems = []
        if prov.get("tge_lstm_state") != "n/a":
            problems.append(f"tge_lstm_state expected 'n/a', got {prov.get('tge_lstm_state')!r}")
        if prov.get("tge_ramp_rounds") is not None:
            problems.append(f"tge_ramp_rounds expected None, got {prov.get('tge_ramp_rounds')!r}")
        return CheckResult(
            "provenance.tge_fields", unit.unit_id, not problems, True,
            "; ".join(problems) if problems else "non-TGE provenance correct (n/a, None)",
        )

    defaults = tge_runtime_defaults()
    expected = {
        "tge_ramp_rounds": defaults["ramp_rounds"],
        "tge_lstm_state": "enabled",
        "tge_pure_lstm_reach": True,
        "tge_cold_start_expert": "isolation_forest",
        "tge_min_tenure": defaults["min_tenure"],
    }
    problems = []
    for key, want in expected.items():
        got = prov.get(key)
        if got != want:
            problems.append(f"{key} expected {want!r}, got {got!r}")
    threshold = prov.get("tge_operational_threshold")
    # Tolerant conversion (round-7 sweep, 3567232374 family): non-numeric
    # threshold values must fail loudly, not crash float().
    try:
        threshold_ok = threshold is not None and abs(
            float(threshold) - float(defaults["threshold"])
        ) <= 1e-9
    except (TypeError, ValueError):
        threshold_ok = False
    if not threshold_ok:
        problems.append(
            f"tge_operational_threshold expected ~{defaults['threshold']!r}, got {threshold!r}"
        )
    return CheckResult(
        "provenance.tge_fields", unit.unit_id, not problems, True,
        "; ".join(problems) if problems else "TGE provenance fields correct",
    )


def check_provenance_defense_sizing(
    unit: UnitRef, result: dict, sizing: tuple[int, int] | None
) -> CheckResult:
    """Defense-sizing provenance (methodology v1.19 / PR #12 round-2 P2):
    run_phase4_flower.py::_defense_provenance_fields writes, for EVERY
    scenario-mode unit,
      - scenario_declared_adversaries / defense_cohort_size: scenario-derived
        ground truth (run-config num-malicious / defense-cohort-size, emitted
        by _build_run_config whenever the scenario resolves -- so present on
        ALL four EXP-005c units, not just the Krum ones),
      - krum_f_policy: KRUM_F_POLICY ("dynamic ceil(n/2)-1") iff the strategy
        has a Krum layer (strategy.startswith("Scenario") and "Krum" in
        strategy -- i.e. config Krum / Krum+TGE), else "n/a".
    `sizing` is expected_defense_sizing(unit.scenario); None fails loudly.
    """
    prov = result.get("provenance") or {}
    problems: list[str] = []

    expected_policy = krum_f_policy_string() if "Krum" in unit.config else "n/a"
    got_policy = prov.get("krum_f_policy")
    if got_policy != expected_policy:
        problems.append(f"krum_f_policy expected {expected_policy!r}, got {got_policy!r}")

    if sizing is None:
        problems.append(
            f"cannot derive expected sizing: scenario JSON for {unit.scenario!r} "
            "missing/unreadable or declares no participants"
        )
    else:
        declared, cohort = sizing
        for key, want in (
            ("scenario_declared_adversaries", declared),
            ("defense_cohort_size", cohort),
        ):
            got = prov.get(key)
            if got != want:
                problems.append(f"{key} expected {want!r}, got {got!r}")

    return CheckResult(
        "provenance.defense_sizing", unit.unit_id, not problems, True,
        "; ".join(problems) if problems else
        f"krum_f_policy={got_policy!r}, declared/cohort match scenario ({sizing})",
    )


def check_rounds_consistency(unit: UnitRef, result: dict) -> CheckResult:
    """Result trajectory must cover server rounds 0..unit.rounds+1 exactly.

    Off-by-one ground truth: _build_run_config sets num-server-rounds =
    rounds + 1 (discovery round, run_phase4_flower.py:437), and Flower's
    Server.fit evaluates the initial parameters at server round 0 PLUS every
    round 1..num_rounds -- ScenarioStrategy.evaluate prints the parsed
    "[ScenarioStrategy] Round N eval" line unconditionally per call
    (flowerfl/scenario_strategy.py:669-681). So a healthy run yields
    rounds + 2 trajectory rows covering exactly {0, ..., rounds+1}.

    Recurrence guard for the --rounds passthrough bug (PR #13): the manifest's
    declared rounds is the expectation, so a container that silently ran the
    runner's default round count instead of the manifest's shows up here.
    """
    expected_max = unit.rounds + 1
    expected = set(range(0, expected_max + 1))
    observed = {t.get("round") for t in (result.get("trajectory") or [])}
    missing = sorted(expected - observed)  # int-only (expected is ints)
    # key=repr: trajectory-derived rounds can mix int/str/None -- a raw sort
    # crashed before the failure was emitted (round-6 P2, 3567214958 sibling
    # site; also subsumes the earlier None special-case).
    extra = sorted(observed - expected, key=repr)
    passed = not missing and not extra
    return CheckResult(
        "rounds_consistency", unit.unit_id, passed, True,
        f"trajectory covers server rounds 0..{expected_max} "
        f"({len(expected)} rows incl. round-0 initial eval + discovery round)"
        if passed else
        f"trajectory must cover server rounds 0..{expected_max} exactly "
        f"(declared rounds {unit.rounds} + discovery round); "
        f"missing {missing[:6]}, unexpected {extra[:6]}",
    )


def check_result_provenance(
    unit: UnitRef, result: dict, locked_hparams: dict,
    sizing: tuple[int, int] | None,
) -> list[CheckResult]:
    return [
        check_provenance_optimizer_state(unit, result),
        check_provenance_seed(unit, result),
        check_provenance_hparams(unit, result, locked_hparams),
        check_provenance_tge_fields(unit, result),
        check_provenance_defense_sizing(unit, result, sizing),
        check_rounds_consistency(unit, result),
    ]


_METHODOLOGY_HEADER_RE = re.compile(r"^## (v\d+\.\d+) ", flags=re.M)


def active_methodology_version(log_path: Path) -> str:
    """The audit-time ACTIVE methodology version: the TOP entry of
    docs/METHODOLOGY_LOG.md (append-only, newest first). Header format is
    '## vX.Y — date — title' (em-dash after the version, so the regex anchors
    on the trailing space) -- same convention praxis_exp/scaffold.py's
    _current_methodology_version parses, but LOUD-FAIL here instead of that
    function's 'v0.1' scaffold default: an auditor must never fabricate the
    version it validates against."""
    log_path = Path(log_path)
    if not log_path.is_file():
        raise RuntimeError(f"methodology log not found: {log_path}")
    matches = _METHODOLOGY_HEADER_RE.findall(log_path.read_text())
    if not matches:
        raise RuntimeError(
            f"no '## vX.Y ' version headers found in {log_path} -- cannot "
            "determine the active methodology version"
        )
    return matches[0]  # newest entry is at the top (file is append-at-top)


def check_methodology_version(
    exp_id: str, meta: dict | None, expected_version: str
) -> CheckResult:
    """Gate-level (not per-unit) check: methodology_version lives on the
    manifest's meta block (praxis_exp/manifest.py write_manifest / matrix_launch.py
    L84), NEVER inside a per-unit result JSON or per-unit MLflow run tag (see
    docker/entrypoint.py's per-unit tag set: unit_id/config/scenario/seed/
    image_digest only -- confirmed by reading main() directly).

    The value must EQUAL the expected version (audit-time active version from
    docs/METHODOLOGY_LOG.md, or the --expect-methodology-version override for
    historical experiments) -- 'any non-empty string' let stale metadata pass."""
    if meta is None:
        return CheckResult(
            "provenance.methodology_version", None, True, False,
            "SKIPPED: no manifest available (local-dirs mode without --manifest-path)",
        )
    version = meta.get("methodology_version")
    passed = version == expected_version
    return CheckResult(
        "provenance.methodology_version", None, passed, True,
        f"methodology_version={version!r} matches expected" if passed else
        f"manifest meta methodology_version={version!r} != expected "
        f"{expected_version!r} (active version from docs/METHODOLOGY_LOG.md "
        "top entry; use --expect-methodology-version to audit a historical "
        "experiment frozen at an older methodology)",
    )


def check_unit_methodology_version(
    unit: UnitRef, result: dict, expected_version: str
) -> CheckResult | None:
    """Per-unit methodology field, IF one is present: the runner's result JSON
    carries none today (verified: run_one()'s provenance block has no
    methodology field), so this returns None (no check emitted) when absent --
    but if a future runner starts stamping one, a stale value must fail
    rather than ride along unexamined."""
    prov = result.get("provenance") or {}
    version = result.get("methodology_version", prov.get("methodology_version"))
    if version is None:
        return None
    passed = version == expected_version
    return CheckResult(
        "provenance.unit_methodology_version", unit.unit_id, passed, True,
        f"unit methodology_version={version!r} matches expected" if passed else
        f"unit methodology_version={version!r} != expected {expected_version!r}",
    )


# ---------------------------------------------------------------------------
# Check 4: signal-log hygiene
# ---------------------------------------------------------------------------


def check_signal_hygiene(
    unit: UnitRef, rows: list[dict],
    declared_participants: dict[int, int] | None = None,
) -> list[CheckResult]:
    """`declared_participants` maps scenario_round -> declared participant
    count (scenario_declared_participants(unit.scenario)). The participation
    sub-check mirrors runner integrity A3
    (run_phase4_flower.py::_assert_participants_per_round): when the scenario
    declares a round's cohort, that declaration IS the expectation -- S4
    legitimately schedules 19- and 11-participant rounds, which a flat
    ceil(0.95*20)=19 floor would false-fail; the flat floor applies only to
    rounds the schedule does not declare."""
    uid = unit.unit_id
    if not rows:
        empty = CheckResult("signal_hygiene.non_empty", uid, False, True, "signal log has 0 rows")
        return [empty]

    results: list[CheckResult] = []

    # the observed values are
    # row-derived and can mix incomparable types (missing field -> None,
    # string "2", int 3) -- a raw sorted() raised TypeError BEFORE the
    # intended required failure was emitted, aborting the whole report.
    # key=repr gives a deterministic display order over any value mix.
    observed_schema = sorted({r.get("signal_log_schema_version") for r in rows}, key=repr)
    # Accept v3 (pre-TGE′), v4 (adds tge_ema_score, GWU-53) and v5 (adds the
    # post-filter aggregation_coefficient + the H3 re-entry event contract,
    # GWU-9) so historical experiments (e.g. EXP-011) still audit while TGE′
    # logs are v4 and H3-era logs are v5. This is a version GATE, not a
    # migration: v4 rows keep their exact v4 meaning (notably `effective_weight`
    # = raw pre-filter num_examples) and are never reinterpreted as v5. Anything
    # else (missing, a stray "2", a malformed list) is flagged.
    _ACCEPTED_SCHEMA = (3, 4, 5)
    bad_schema = sorted((v for v in observed_schema if v not in _ACCEPTED_SCHEMA), key=repr)
    results.append(CheckResult(
        "signal_hygiene.schema_version", uid, not bad_schema, True,
        f"all rows schema_version in {_ACCEPTED_SCHEMA}" if not bad_schema else
        f"unsupported schema versions present: {bad_schema} "
        f"(accepted {_ACCEPTED_SCHEMA}; all distinct observed values: {observed_schema})",
    ))

    # the sole value must be PRESENT and
    # non-empty. A bare set-based len==1 test passed an all-None log as
    # "single run_started_at=None", and mixed None/str crashed sorted()
    # with a TypeError before the check could even fail.
    def _has_started_at(r: dict) -> bool:
        v = r.get("run_started_at")
        return isinstance(v, str) and bool(v.strip())

    n_missing = sum(1 for r in rows if not _has_started_at(r))
    valid_values = sorted({r["run_started_at"] for r in rows if _has_started_at(r)})
    started_ok = n_missing == 0 and len(valid_values) == 1
    if started_ok:
        started_detail = f"single run_started_at={valid_values[0]!r}"
    else:
        parts = []
        if n_missing:
            parts.append(f"{n_missing}/{len(rows)} rows missing/empty run_started_at")
        if len(valid_values) != 1:
            parts.append(f"{len(valid_values)} distinct non-empty values: {valid_values}")
        started_detail = "; ".join(parts)
    results.append(CheckResult(
        "signal_hygiene.run_started_at", uid, started_ok, True, started_detail,
    ))

    pair_counts: dict[tuple[Any, Any], int] = defaultdict(int)
    for r in rows:
        pair_counts[(r.get("server_round"), r.get("logical_cid"))] += 1
    # key=repr: pair tuples are row-derived; a duplicated pair with
    # server_round=None next to an int pair crashes a raw tuple sort
    # (round-6 P2, 3567214958 sibling site).
    dupes = sorted((k for k, c in pair_counts.items() if c > 1), key=repr)
    results.append(CheckResult(
        "signal_hygiene.duplicate_rows", uid, not dupes, True,
        "no duplicate (server_round, logical_cid) pairs" if not dupes
        else f"duplicate (server_round, logical_cid) pairs: {dupes[:5]}",
    ))

    by_round: dict[Any, set] = defaultdict(set)
    scenario_round_of: dict[Any, Any] = {}
    for r in rows:
        by_round[r.get("server_round")].add(r.get("logical_cid"))
        scenario_round_of.setdefault(r.get("server_round"), r.get("scenario_round"))
    floor = math.ceil(A3_PARTICIPATION_FLOOR_FRAC * NUM_CLIENT_SUPERNODES)
    shortfalls: dict[Any, str] = {}
    for rnd, cids in by_round.items():
        declared = (declared_participants or {}).get(scenario_round_of.get(rnd))
        expected = declared if declared is not None else floor
        if len(cids) < expected:
            src = "scenario-declared" if declared is not None else "flat floor"
            shortfalls[rnd] = f"{len(cids)} < {expected} ({src})"
    results.append(CheckResult(
        "signal_hygiene.participation", uid, not shortfalls, True,
        f"every round meets its scenario-declared cohort (fallback floor {floor})"
        if not shortfalls else f"rounds below declared/floor participation: {shortfalls}",
    ))

    # a log missing an ENTIRE server
    # round produced no by_round entry and sailed through -- silently costing
    # downstream recall/FPR that round. Expected signal server rounds, ground-
    # truthed from the producing code:
    #   - ScenarioStrategy._round_offset = 1 (flowerfl/scenario_strategy.py:135
    #     -- "round 1 is used for cid discovery, so the scenario schedule
    #     starts at round 2"; startup print: "Flower rounds 2-{num_rounds+1}").
    #   - _maybe_log_signals (scenario_strategy.py:533-536) emits rows IFF
    #     _schedule_cache[server_round - 1] is non-empty -- i.e. exactly the
    #     server rounds whose scenario_round is declared by the schedule.
    #     (Plugins also first score at server round 2, but row emission is
    #     schedule-gated, not plugin-gated.)
    #   - Generated scenarios (S0-S4) declare every scenario round
    #     1..num_rounds, so the expected set is {2..rounds+1}; when the
    #     scenario declaration is available the set derives from its actual
    #     declared rounds (shifted +1 for the discovery offset, clipped to
    #     the unit's horizon).
    # the comparison must run in BOTH
    # directions. Extra rounds beyond the manifest horizon (appended/longer
    # run) pass every other hygiene sub-check when well-formed -- same
    # run_started_at, non-duplicate (round, cid) pairs, declared/floor
    # participation (verified live) -- and would silently pool into
    # downstream recall/FPR.
    def _capped(rounds_list: list) -> str:
        shown = rounds_list[:8]
        n_more = len(rounds_list) - len(shown)
        return f"{shown}" + (f" +{n_more} more" if n_more > 0 else "")

    expected_rounds = expected_signal_rounds(unit, declared_participants)
    observed_rounds = set(by_round)
    missing_rounds = sorted(expected_rounds - observed_rounds)
    # key=repr: observed rounds are row-derived (None/str possible)
    # while expected are ints (round-6 P2, 3567214958 sibling site).
    unexpected_rounds = sorted(observed_rounds - expected_rounds, key=repr)
    coverage_problems: list[str] = []
    if missing_rounds:
        coverage_problems.append(
            f"{len(missing_rounds)} scheduled server round(s) have NO signal "
            f"rows: {_capped(missing_rounds)}"
        )
    if unexpected_rounds:
        coverage_problems.append(
            f"{len(unexpected_rounds)} UNEXPECTED server round(s) beyond the "
            f"declared schedule/horizon: {_capped(unexpected_rounds)} -- rows "
            "from an appended/longer run would silently pool into recall/FPR"
        )
    results.append(CheckResult(
        "signal_hygiene.round_coverage", uid, not coverage_problems, True,
        f"signal rows present for exactly the {len(expected_rounds)} scheduled "
        "server rounds (no missing, no unexpected)"
        if not coverage_problems else "; ".join(coverage_problems),
    ))

    return results


def expected_signal_rounds(
    unit: UnitRef, declared_participants: dict[int, int] | None
) -> set[int]:
    """Server rounds that MUST carry signal rows: the scenario's declared
    scenario rounds shifted +1 for the discovery offset, clipped to the
    unit's horizon; fallback {2..rounds+1} when the scenario JSON is not
    available (generated S0-S4 scenarios declare every round 1..num_rounds).
    Ground truth in check_signal_hygiene's round-coverage comment."""
    if declared_participants is not None:
        return {r + 1 for r in declared_participants if 1 <= r + 1 <= unit.rounds + 1}
    return set(range(2, unit.rounds + 2))


# ---------------------------------------------------------------------------
# Check 5: v1.17 Krum+TGE spot-check
# ---------------------------------------------------------------------------


def check_krumtge_spot_check(
    unit: UnitRef, rows: list[dict],
    declared_participants: dict[int, int] | None = None,
) -> CheckResult:
    by_round: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        by_round[r.get("server_round")].append(r)

    scored_rounds = {
        rnd: [r for r in group if r.get("tge_score") is not None]
        for rnd, group in by_round.items()
    }
    scored_rounds = {rnd: scored for rnd, scored in scored_rounds.items() if scored}

    problems: list[str] = []

    # iterating only rounds that HAPPEN
    # to have scored rows let TGE scoring silently vanish for most scheduled
    # rounds as long as one round looked right. EVERY expected scheduled
    # round must carry tge-scored rows -- ground truth:
    #   - TGEnsemblePlugin.score_updates populates _last_details for every
    #     client it receives, unconditionally per call
    #     (flowerfl/byzantine_defense.py:678-693);
    #   - TGEnsembleModel.score_client returns a NON-NULL final_score in
    #     EVERY phase, warmup included (warmup returns final_score=1.0,
    #     phase='warmup' -- rmc/tg_ensemble.py:801-810; pre_gbdt returns the
    #     geometric fallback :817-826) -- warmup changes the score VALUE,
    #     never whether details are populated, so no round is exempt;
    #   - upstream Krum's keep = max(1, n-f-2) >= 1 always hands TGE at
    #     least one survivor (byzantine_defense.py:341-345).
    expected = expected_signal_rounds(unit, declared_participants)
    unscored_expected = sorted(rnd for rnd in expected if rnd not in scored_rounds)
    if unscored_expected:
        shown = unscored_expected[:8]
        n_more = len(unscored_expected) - len(shown)
        problems.append(
            f"{len(unscored_expected)} scheduled round(s) have NO TGE-scored rows "
            f"(TGE scoring absent): {shown}"
            + (f" +{n_more} more" if n_more > 0 else "")
        )

    for rnd, scored in sorted(scored_rounds.items(), key=lambda kv: str(kv[0])):
        total = len(by_round[rnd])
        # DYNAMIC per-round expectation (methodology v1.19): the survivor
        # count is a function of THIS round's cohort n, not the full-cohort
        # constant -- n=20 -> 9, S3/S4 disconnect rounds n=11 -> 4.
        expected_keep = krum_dynamic_keep(total)
        if len(scored) != expected_keep:
            problems.append(
                f"round {rnd}: {len(scored)} tge-scored rows, expected exactly "
                f"{expected_keep} (dynamic Multi-Krum keep for n={total}: "
                f"f=ceil(n/2)-1, keep=max(1,n-f-2))"
            )
        if len(scored) >= total:
            problems.append(
                f"round {rnd}: {len(scored)} scored rows >= {total} total rows "
                "(expected strictly fewer -- positional-bug signature: every row "
                "carrying a tge_score means the cid-keyed join never filtered "
                "anyone, methodology v1.17)"
            )
        for row in by_round[rnd]:
            if row.get("tge_score") is not None:
                continue
            non_null = [f for f in TGE_FIELDS_NULL_WHEN_UNSCORED if row.get(f) is not None]
            if non_null:
                problems.append(
                    f"round {rnd} logical_cid={row.get('logical_cid')!r}: unscored row "
                    f"has non-null fields {non_null} (expected truthful nulls)"
                )

    return CheckResult(
        "krumtge_spot_check", unit.unit_id, not problems, True,
        "; ".join(problems) if problems else
        f"{len(scored_rounds)} scored round(s), per-round dynamic survivor count "
        "(f=ceil(n/2)-1) matched, unscored rows truthfully null",
    )


# ---------------------------------------------------------------------------
# Check: krum_score variance (pre-v1.19 flattening signature)
# ---------------------------------------------------------------------------


def check_krum_score_variance(unit: UnitRef, rows: list[dict]) -> CheckResult:
    """At least one round must show NON-CONSTANT krum_score values.

    The pre-v1.19 certification-bound guard (`n <= 2f+2` treated as a
    computability bound) returned uniform 1.0 scores at the canonical RMC
    operating point, degenerating Krum to keep-first-m-by-arrival and making
    H2 recall@10%FPR uncomputable for the arm (see the v1.19 note inside
    KrumDefensePlugin.score_updates, flowerfl/byzantine_defense.py:249-260).
    Uniform per-round scores in the signal log are that bug's signature.
    Applies to every unit deploying KrumDefensePlugin (config Krum and the
    composed Krum+TGE chain alike)."""
    by_round: dict[Any, set] = defaultdict(set)
    for r in rows:
        score = r.get("krum_score")
        if score is not None:
            by_round[r.get("server_round")].add(score)

    if not by_round:
        return CheckResult(
            "krum_score_variance", unit.unit_id, False, True,
            "krum_score is null on every row -- Krum scoring channel absent",
        )
    varied_rounds = [rnd for rnd, vals in by_round.items() if len(vals) > 1]
    passed = bool(varied_rounds)
    return CheckResult(
        "krum_score_variance", unit.unit_id, passed, True,
        f"{len(varied_rounds)}/{len(by_round)} scored round(s) show krum_score variance"
        if passed else
        f"krum_score constant within every one of {len(by_round)} scored round(s) "
        "(uniform-fallback signature: pre-v1.19 certification-bound guard flattened "
        "scores to 1.0 -- recall@10%FPR uncomputable)",
    )


# ---------------------------------------------------------------------------
# Check 7: Szeląg gate (report only, exit-code neutral)
# ---------------------------------------------------------------------------

SZELAG_ANCHOR = 0.9791
SZELAG_TOLERANCE = 0.01


def check_szelag_gate(result: dict) -> dict:
    observed = result.get("final_accuracy")
    if observed is None:
        return {"passed": False, "observed": None, "anchor": SZELAG_ANCHOR,
                "tolerance": SZELAG_TOLERANCE, "detail": "final_accuracy missing from result"}
    passed = abs(float(observed) - SZELAG_ANCHOR) <= SZELAG_TOLERANCE + 1e-9  # float-safe boundary
    return {
        "passed": passed, "observed": observed, "anchor": SZELAG_ANCHOR,
        "tolerance": SZELAG_TOLERANCE,
        "detail": f"final_accuracy={observed:.4f} vs anchor {SZELAG_ANCHOR} +/- {SZELAG_TOLERANCE}",
    }


# ---------------------------------------------------------------------------
# Check 8: wall-clock sizing table (report only, exit-code neutral)
# ---------------------------------------------------------------------------

FANOUT_TOTAL_VCPUS = 32
FANOUT_VCPUS_PER_JOB = 8
FANOUT_TARGET_UNITS = 100


def build_wall_clock_table(rows: list[dict]) -> tuple[list[dict], dict]:
    """rows: [{"unit_id":..., "config":..., "elapsed_seconds":...}, ...]
    (elapsed_seconds sourced from result["elapsed_seconds"], the only wall-clock
    field run_phase4_flower.py's run_one() actually records -- see
    `elapsed = time.time() - t0` in the ground-truth citations above)."""
    valid = [r for r in rows if isinstance(r.get("elapsed_seconds"), (int, float))]
    if not valid:
        return list(rows), {
            "note": "ESTIMATE: no numeric elapsed_seconds available; cannot project",
        }
    mean_elapsed = sum(r["elapsed_seconds"] for r in valid) / len(valid)
    concurrency = FANOUT_TOTAL_VCPUS // FANOUT_VCPUS_PER_JOB
    waves = math.ceil(FANOUT_TARGET_UNITS / concurrency)
    projected_seconds = waves * mean_elapsed
    projection = {
        "note": "ESTIMATE: naive projection, assumes uniform per-unit wall-clock "
                "and perfect concurrency (no queueing/startup overhead)",
        "n_units_observed": len(valid),
        "mean_elapsed_seconds": mean_elapsed,
        "fanout_target_units": FANOUT_TARGET_UNITS,
        "concurrency": concurrency,
        "waves": waves,
        "projected_wall_clock_seconds": projected_seconds,
        "projected_wall_clock_hours": projected_seconds / 3600.0,
    }
    return list(rows), projection

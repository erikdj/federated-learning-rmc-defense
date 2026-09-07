"""EXP-005c calibration-gate audit: validate every artifact the dissertation
analysis needs is present and correct BEFORE the 100-unit H2 dev-sweep fan-out.

Two modes:
    S3 mode (default):    python audit_calibration_gate.py EXP-005c
    local-dirs mode:      python audit_calibration_gate.py EXP-005c \\
                               --results-dir DIR --signals-dir DIR

Local-dirs mode expects already-downloaded artifacts (e.g. via `aws s3 sync`)
named by their S3 basenames: `<unit_id>.json` in --results-dir and
`<unit_id>.jsonl` in --signals-dir (matching praxis_exp.storage.result_key /
signal_key). It cannot run the S3-integrity (done-marker) or MLflow checks.
Its EXPECTED unit set comes from --manifest-path (units verbatim, meta feeds
methodology_version) or, absent that, from the experiment's design-matrix doc
(docs/experiments/EXP-NNN-*.md) -- never from globbing whatever files happen
to be present: every expected unit with a missing or unparseable artifact is
a REQUIRED failure.

Exit code is 0 iff every REQUIRED check passes. The Szeląg anchor gate and the
wall-clock sizing table are reported but are exit-code neutral (human review
gates, per spec).

GROUND-TRUTH SURPRISES found while building this auditor (all cited inline in
scripts/_calibration_gate_lib.py and scripts/_calibration_gate_store.py too):

1. Result JSON has NO top-level "mode" field. The runner's internal short-form
   vocabulary is result["provenance"]["optimizer_state"] == "persistent" (NOT
   "persistent_optimizer" -- that longer string is the --modes CLI alias /
   Unit.mode value; _resolve_optimizer_state() maps it to "persistent").
2. Result JSON never persists lr/local-epochs anywhere (top-level, provenance,
   or otherwise). run_phase4_flower.py only enforces them PRE-HOC via
   _assert_lr_matches_locked() at generation time; there is nothing to check
   post-hoc in the artifact. The hparams sub-check reports a loud, non-blocking
   SKIP against real data (see _calibration_gate_lib.check_provenance_hparams).
3. methodology_version is NEVER a per-unit field (not in the result JSON, not
   in the per-unit MLflow tags docker/entrypoint.py sets). It lives ONLY on
   the manifest's meta block (praxis_exp/manifest.py, written by
   praxis_exp/matrix_launch.py) and on the parent sweep's MLflow run tags.
   This audit checks it once at the gate level, not per-unit.
4. flowerfl/scenario_strategy.py's _maybe_log_signals sets tge_threshold to a
   ROUND-LEVEL constant for every row once a TGE plugin is active, even for
   rows whose client was never individually scored. The "truthful nulls for
   filtered clients" contract holds for every other tge_* field but NOT
   tge_threshold -- checking it as null-when-unscored would fail every real
   Krum+TGE signal log.
5. (Resolved in methodology v1.19.) An earlier revision of this
   auditor found that "Multi-Krum keeps 9/20" did not reproduce from the
   then-deployed code (malicious-fraction defaulted to 0.0 -> f=1 -> keep 18).
   The deployment was fixed: scenario-mode Krum now sizes f PER ROUND as
   ceil(n/2)-1 with keep = max(1, n-f-2) (KrumDefensePlugin._effective_f +
   filter_updates dynamic branch), giving keep 9 on full-cohort rounds (n=20)
   and keep 4 on S3/S4 disconnect rounds (n=11). The auditor's expectation
   lives in ONE helper (_calibration_gate_lib.krum_dynamic_keep) that mirrors
   the plugin formula, with a cross-check test that drives the real plugin.
6. S4_full_mix legitimately declares 19- and 11-participant rounds (histogram:
   38 rounds @ 20, 8 @ 19, 4 @ 11), so the signal-log participation check
   honors the scenario's declared per-round cohort (mirroring runner A3 --
   run_phase4_flower.py::_assert_participants_per_round) and only falls back
   to the flat ceil(0.95*20) floor for undeclared rounds. A flat floor would
   false-fail every real S4 signal log.
7. Trajectory round accounting: _build_run_config sets num-server-rounds =
   rounds + 1 (discovery round) and Flower's Server.fit evaluates round 0
   (initial parameters) plus every round 1..num_rounds, so a healthy result
   has rounds + 2 trajectory rows covering exactly {0..rounds+1}. (The 51-row
   0..50 artifacts under results/2026052x predate the discovery-round change,
   commit 254825e.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _calibration_gate_lib import (  # noqa: E402
    HPARAMS_LOCKED_PATH,
    build_wall_clock_table,
    check_baseline_instrumentation,
    active_methodology_version,
    check_krum_score_variance,
    check_krumtge_spot_check,
    check_methodology_version,
    check_result_provenance,
    check_signal_hygiene,
    check_szelag_gate,
    check_unit_methodology_version,
    expected_defense_sizing,
    scenario_declared_participants,
)
from _calibration_gate_report import render_report  # noqa: E402
from _calibration_gate_store import (  # noqa: E402
    DEFAULT_MLFLOW_URI,
    ExpectedUnitsError,
    S3GateStore,
    check_local_presence,
    check_mlflow_runs,
    check_s3_integrity,
    default_mlflow_fetcher,
    load_design_matrix_units,
    load_local_manifest,
    materialize_s3_artifacts,
    unit_ref_from_manifest_unit,
)
from _calibration_gate_types import CheckResult  # noqa: E402
from praxis_exp import storage as s3_storage  # noqa: E402  (key builders only; no AWS calls)
from _calibration_gate_types import GateReport  # noqa: E402

KRUM_TGE_CONFIG = "Krum+TGE"
SZELAG_ANCHOR_CONFIG = "Krum"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "aws_validation" / "EXP-005c-gate-report.md"


def _load_locked_hparams() -> dict:
    return json.loads(HPARAMS_LOCKED_PATH.read_text())


def _resolve_expected_methodology(expected: str | None) -> str:
    """None -> the audit-time active version (top entry of the repo's
    METHODOLOGY_LOG.md); a string -> the --expect-methodology-version
    override for auditing historical experiments frozen at an older
    methodology."""
    if expected is not None:
        return expected
    return active_methodology_version(REPO_ROOT / "docs" / "METHODOLOGY_LOG.md")


class _ScenarioCache:
    """Per-scenario memo for defense sizing + declared participants (both are
    derived from the scenario JSON via the runner's own helpers; EXP-005c has
    a single scenario but the dev sweep will not)."""

    def __init__(self) -> None:
        self._sizing: dict[str, tuple[int, int] | None] = {}
        self._participants: dict[str, dict[int, int] | None] = {}

    def sizing(self, scenario: str) -> tuple[int, int] | None:
        if scenario not in self._sizing:
            self._sizing[scenario] = expected_defense_sizing(scenario)
        return self._sizing[scenario]

    def participants(self, scenario: str) -> dict[int, int] | None:
        if scenario not in self._participants:
            self._participants[scenario] = scenario_declared_participants(scenario)
        return self._participants[scenario]


def _audit_unit(
    report: GateReport,
    unit,
    result_path: Path,
    signal_path: Path,
    result: dict,
    rows: list[dict],
    locked_hparams: dict,
    scenarios: _ScenarioCache,
    wall_clock_rows: list[dict],
    expected_methodology: str,
) -> None:
    """Per-unit checks shared verbatim by the S3 and local-dirs runners."""
    report.add(check_baseline_instrumentation(unit, result_path, signal_path))
    report.extend(check_result_provenance(
        unit, result, locked_hparams, scenarios.sizing(unit.scenario)
    ))
    report.extend(check_signal_hygiene(
        unit, rows, declared_participants=scenarios.participants(unit.scenario)
    ))
    unit_methodology = check_unit_methodology_version(unit, result, expected_methodology)
    if unit_methodology is not None:  # emitted only when the unit carries the field
        report.add(unit_methodology)
    if "Krum" in unit.config:
        # both Krum and the composed Krum+TGE chain deploy KrumDefensePlugin
        report.add(check_krum_score_variance(unit, rows))
    if unit.config == KRUM_TGE_CONFIG:
        report.add(check_krumtge_spot_check(
            unit, rows, declared_participants=scenarios.participants(unit.scenario)
        ))
    if unit.config == SZELAG_ANCHOR_CONFIG:
        report.szelag = check_szelag_gate(result)
    wall_clock_rows.append({
        "unit_id": unit.unit_id, "config": unit.config,
        "elapsed_seconds": result.get("elapsed_seconds"),
    })


def run_gate_s3(
    store,
    exp_id: str,
    *,
    mlflow_uri: str = DEFAULT_MLFLOW_URI,
    mlflow_fetcher=default_mlflow_fetcher,
    check_mlflow: bool = True,
    expected_methodology_version: str | None = None,
) -> GateReport:
    """S3 mode: manifest is the source of truth for the unit list + meta."""
    from praxis_exp.manifest import read_manifest

    manifest_exp_id, meta, manifest_units = read_manifest(store, exp_id)
    # do NOT discard the manifest's own
    # exp_id -- a copied/corrupt sweeps/<exp>/manifest.json naming another
    # experiment would drive the audit with that sweep's units while checking
    # this experiment's S3 prefix (same class as the --manifest-path guard).
    if manifest_exp_id != exp_id:
        raise ExpectedUnitsError(
            f"S3 manifest at {s3_storage.manifest_key(exp_id)} declares exp_id "
            f"{manifest_exp_id!r}, but this audit was invoked for {exp_id!r} -- "
            "refusing to audit another sweep's artifacts"
        )
    units = [unit_ref_from_manifest_unit(u) for u in manifest_units]
    locked_hparams = _load_locked_hparams()
    scenarios = _ScenarioCache()
    expected_methodology = _resolve_expected_methodology(expected_methodology_version)

    report = GateReport(exp_id=exp_id, mode="s3")
    report.add(check_methodology_version(exp_id, meta, expected_methodology))

    wall_clock_rows: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="calibration_gate_"))
    try:
        for unit in units:
            report.add(check_s3_integrity(store, exp_id, unit))
            result_path, signal_path = materialize_s3_artifacts(store, exp_id, unit, tmpdir)
            if result_path is None or signal_path is None:
                continue  # absence already flagged by check_s3_integrity

            # Parse AFTER download, converting corruption into a per-unit
            # required failure that names the S3 key -- the audit continues
            # to the remaining units.
            result, rows, parse_failure = _parse_unit_artifacts(
                unit.unit_id, result_path, signal_path,
                check_name="s3_artifact_parse",
                result_label=s3_storage.result_key(exp_id, unit.unit_id),
                signal_label=s3_storage.signal_key(exp_id, unit.unit_id),
            )
            if parse_failure is not None:
                report.add(parse_failure)
                continue

            _audit_unit(report, unit, result_path, signal_path, result, rows,
                        locked_hparams, scenarios, wall_clock_rows,
                        expected_methodology)

        if check_mlflow:
            report.extend(check_mlflow_runs(exp_id, units, meta, mlflow_uri, mlflow_fetcher))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    report.wall_clock_rows, report.fanout_projection = build_wall_clock_table(wall_clock_rows)
    return report


# --- Parse-time value-shape contract ( , 3567232374 +
# 3567232376, closing the whole corrupt-but-valid-JSON family) -------------
#
# EVERYTHING the lib's checks dereference, use as a dict key, set member, or
# sort input is validated HERE, once, at parse time. The contract every check
# downstream of _parse_unit_artifacts may assume:
#   result:  a dict; result["provenance"] a dict if present;
#            result["trajectory"] a list of dicts if present, each item's
#            "round" a hashable scalar; result["final_accuracy"] numeric or
#            absent/None (feeds float() in the Szelag gate).
#   rows:    each a dict; the identity/metadata fields in _ROW_SCALAR_FIELDS
#            are hashable scalars (str/int/float/bool/None) -- they are used
#            as dict keys, tuple keys, set members, and declared-schedule
#            lookup keys across the hygiene/variance/spot checks.
# Values NOT covered here (optional hparams values, tge_operational_threshold)
# are handled tolerantly in their consuming checks (loud mismatch, no crash).
# One corrupt unit NEVER aborts the gate: violations become the per-unit
# required *_artifact_parse failure and the audit continues.

_SCALAR_TYPES = (str, int, float, bool)
_ROW_SCALAR_FIELDS = (
    "server_round", "scenario_round", "logical_cid",
    "signal_log_schema_version", "run_started_at", "krum_score",
)


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, _SCALAR_TYPES)


def _validate_result_shape(
    result: dict, unit_id: str, check_name: str, result_label: str
) -> CheckResult | None:
    """Value-shape validation for the result JSON (see contract above)."""
    def fail(detail: str) -> CheckResult:
        return CheckResult(check_name, unit_id, False, True,
                           f"result JSON {result_label}: {detail}")

    prov = result.get("provenance")
    if prov is not None and not isinstance(prov, dict):
        return fail(f"'provenance' is not a JSON object (got {type(prov).__name__})")
    traj = result.get("trajectory")
    if traj is not None:
        if not isinstance(traj, list):
            return fail(f"'trajectory' is not a JSON array (got {type(traj).__name__})")
        for i, item in enumerate(traj):
            if not isinstance(item, dict):
                return fail(f"trajectory[{i}] is not a JSON object (got {type(item).__name__})")
            if not _is_scalar(item.get("round")):
                return fail(
                    f"trajectory[{i}].round is not a scalar "
                    f"(got {type(item.get('round')).__name__})"
                )
    final_acc = result.get("final_accuracy")
    if final_acc is not None and not isinstance(final_acc, (int, float)):
        return fail(f"'final_accuracy' is not numeric (got {type(final_acc).__name__})")
    return None


def _parse_unit_artifacts(
    unit_id: str,
    result_path: Path,
    signal_path: Path,
    *,
    check_name: str,
    result_label: str,
    signal_label: str,
) -> tuple[dict | None, list[dict] | None, CheckResult | None]:
    """Parse a unit's downloaded artifacts and ENFORCE the parse-time
    value-shape contract above. A malformed/unreadable/mis-shaped file is a
    LOUD required failure naming the artifact (its S3 key in S3 mode, its
    filename in local-dirs mode) and the error -- never a skip and never an
    exception that aborts the gate. Invalid UTF-8, non-object JSON or
    provenance, and unhashable row fields all become per-unit failures."""
    try:
        result = json.loads(result_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, CheckResult(
            check_name, unit_id, False, True,
            f"result JSON {result_label} is unreadable/malformed: {exc}",
        )
    # syntactically valid but non-object
    # JSON (top-level list/str/null) crashed every downstream .get(...) call
    # instead of failing per-unit. The result must be a JSON object.
    if not isinstance(result, dict):
        return None, None, CheckResult(
            check_name, unit_id, False, True,
            f"result JSON {result_label} is not a JSON object "
            f"(got {type(result).__name__})",
        )
    shape_failure = _validate_result_shape(result, unit_id, check_name, result_label)
    if shape_failure is not None:
        return None, None, shape_failure
    try:
        signal_lines = signal_path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return None, None, CheckResult(
            check_name, unit_id, False, True,
            f"signal log {signal_label} is unreadable: {exc}",
        )
    rows: list[dict] = []
    for lineno, line in enumerate(signal_lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, None, CheckResult(
                check_name, unit_id, False, True,
                f"signal log {signal_label} line {lineno} is malformed: {exc}",
            )
        # Each JSONL record must be an object -- `null`/`[]`/`"str"` parse
        # fine but crash every downstream r.get(...) (round-5 P2, 3567196082).
        if not isinstance(parsed, dict):
            return None, None, CheckResult(
                check_name, unit_id, False, True,
                f"signal log {signal_label} line {lineno} is not a JSON object "
                f"(got {type(parsed).__name__})",
            )
        # Round-7 (3567232376 + sweep): identity/metadata fields are used as
        # dict/tuple keys, set members, and schedule-lookup keys downstream;
        # array/object values crash those structures with unhashable-type
        # TypeErrors. Enforce the scalar contract per line.
        for field in _ROW_SCALAR_FIELDS:
            if not _is_scalar(parsed.get(field)):
                return None, None, CheckResult(
                    check_name, unit_id, False, True,
                    f"signal log {signal_label} line {lineno} field {field!r} "
                    f"is not a scalar (got {type(parsed.get(field)).__name__})",
                )
        rows.append(parsed)
    return result, rows, None


def run_gate_local(
    results_dir: Path,
    signals_dir: Path,
    exp_id: str,
    *,
    expected_units,
    meta: dict | None = None,
    expected_methodology_version: str | None = None,
) -> GateReport:
    """Local-dirs mode: no S3-integrity / MLflow checks (nothing to check them
    against offline). `expected_units` (list of praxis_exp.units.Unit) is the
    EXPECTED unit set -- resolved by main() from --manifest-path (verbatim) or
    from the experiment's design-matrix doc. Every expected unit whose
    artifacts are missing or unparseable is a REQUIRED failure; the gate
    never derives its expectations from whatever files happen to be present. `meta` is the manifest meta block when available."""
    locked_hparams = _load_locked_hparams()
    scenarios = _ScenarioCache()
    expected_methodology = _resolve_expected_methodology(expected_methodology_version)

    report = GateReport(exp_id=exp_id, mode="local")
    report.add(check_methodology_version(exp_id, meta, expected_methodology))

    wall_clock_rows: list[dict] = []
    expected_ids = set()
    for manifest_unit in expected_units:
        unit = unit_ref_from_manifest_unit(manifest_unit)
        expected_ids.add(unit.unit_id)
        result_path = results_dir / f"{unit.unit_id}.json"
        signal_path = signals_dir / f"{unit.unit_id}.jsonl"
        report.add(check_local_presence(unit, result_path, signal_path))
        if not (result_path.is_file() and signal_path.is_file()):
            continue  # missing artifact already recorded as a required failure

        result, rows, parse_failure = _parse_unit_artifacts(
            unit.unit_id, result_path, signal_path,
            check_name="local_artifact_parse",
            result_label=result_path.name,
            signal_label=signal_path.name,
        )
        if parse_failure is not None:
            report.add(parse_failure)
            continue

        _audit_unit(report, unit, result_path, signal_path, result, rows,
                    locked_hparams, scenarios, wall_clock_rows,
                    expected_methodology)

    extras = sorted(p.name for p in results_dir.glob("*.json") if p.stem not in expected_ids)
    if extras:
        report.add(CheckResult(
            "local_unexpected_files", None, True, False,
            f"NOTE: {len(extras)} result file(s) not in the expected unit set "
            f"(ignored): {extras[:6]}",
        ))

    report.wall_clock_rows, report.fanout_projection = build_wall_clock_table(wall_clock_rows)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_id", help="e.g. EXP-005c")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="local-dirs mode: directory of <unit_id>.json result files")
    p.add_argument("--signals-dir", type=Path, default=None,
                   help="local-dirs mode: directory of <unit_id>.jsonl signal logs")
    p.add_argument("--manifest-path", type=Path, default=None,
                   help="local-dirs mode: path to a downloaded manifest.json; its "
                        "units become the expected unit set VERBATIM and its meta "
                        "feeds the methodology_version/image_digest checks. Without "
                        "it, the expected set is derived from the experiment's "
                        "design-matrix doc (docs/experiments/EXP-NNN-*.md). All "
                        "per-unit expectations (config/scenario/seed/mode/rounds) "
                        "come from the unit spec, never from the artifacts under test.")
    p.add_argument("--bucket", default=None, help="S3 mode: artifact bucket (else praxis_exp.config.Config)")
    p.add_argument("--mlflow-uri", default=DEFAULT_MLFLOW_URI)
    p.add_argument("--skip-mlflow", action="store_true", help="S3 mode: skip check 6 (MLflow)")
    p.add_argument("--expect-methodology-version", default=None, metavar="vX.Y",
                   help="Expected methodology_version for the manifest meta (and any "
                        "per-unit methodology fields). Default: the audit-time ACTIVE "
                        "version, parsed from the top entry of docs/METHODOLOGY_LOG.md. "
                        "Set explicitly to audit a HISTORICAL experiment frozen at an "
                        "older methodology (e.g. --expect-methodology-version v1.11).")
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    local_mode = args.results_dir is not None or args.signals_dir is not None
    if local_mode:
        if args.results_dir is None or args.signals_dir is None:
            print("ERROR: --results-dir and --signals-dir must both be given for local-dirs mode",
                  file=sys.stderr)
            return 2
        # Expected unit set: manifest units verbatim when provided, else the
        # experiment's design-matrix doc. NEVER derived from present files
        # (glob-based discovery passed incomplete downloads).
        try:
            if args.manifest_path is not None:
                meta, expected_units = load_local_manifest(
                    args.manifest_path, expected_exp_id=args.exp_id
                )
            else:
                meta = None
                expected_units = load_design_matrix_units(REPO_ROOT, args.exp_id)
        except ExpectedUnitsError as exc:
            print(f"ERROR: cannot resolve the expected unit set: {exc}", file=sys.stderr)
            return 2
        report = run_gate_local(
            args.results_dir, args.signals_dir, args.exp_id,
            expected_units=expected_units, meta=meta,
            expected_methodology_version=args.expect_methodology_version,
        )
    else:
        import boto3
        from praxis_exp.config import Config

        bucket = args.bucket or Config().artifact_bucket
        store = S3GateStore(bucket, boto3.client("s3"))
        try:
            report = run_gate_s3(
                store, args.exp_id, mlflow_uri=args.mlflow_uri,
                check_mlflow=not args.skip_mlflow,
                expected_methodology_version=args.expect_methodology_version,
            )
        except ExpectedUnitsError as exc:
            # e.g. the S3 manifest declares a different exp_id (round-4 P2)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    text = render_report(report)
    print(text)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(text + "\n")
    print(f"\n[report written to {args.report_path}]")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

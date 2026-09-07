"""Tests for scripts/audit_calibration_gate.py -- the EXP-005c pre-launch gate.

Follows tests/test_analyze_ramp_selection.py's conventions: sys.path insert
for scripts/, synthetic in-memory fixtures, loud-fail assertions (specific
CheckResult.name / substrings in .detail, not just "did something fail").

Post-merge expectations (PRs #11/#12 on master, methodology v1.19):
- Multi-Krum keep is DYNAMIC per round: f = ceil(n/2)-1, keep = max(1, n-f-2)
  (KrumDefensePlugin._effective_f, flowerfl/byzantine_defense.py:217-222, and
  filter_updates' dynamic branch :341-345) -- n=20 -> 9, n=11 disconnect -> 4.
- Result provenance carries krum_f_policy / scenario_declared_adversaries /
  defense_cohort_size (run_phase4_flower.py::_defense_provenance_fields).
- Trajectory covers server rounds 0..rounds+1 (num-server-rounds = rounds+1,
  run_phase4_flower.py::_build_run_config L437; flwr Server.fit evaluates
  round 0 plus every round 1..num_rounds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from praxis_exp.integrity import persist_unit  # noqa: E402
from praxis_exp.manifest import write_manifest  # noqa: E402
from praxis_exp.units import expand_matrix  # noqa: E402

import _calibration_gate_lib as gl  # noqa: E402
import _calibration_gate_store as gs  # noqa: E402
import audit_calibration_gate as acg  # noqa: E402

from calibration_gate_fixtures import (  # noqa: E402
    ACTIVE_METHODOLOGY,
    EXP_ID,
    KRUM_F_POLICY,
    MODE,
    ROUNDS,
    SCENARIO,
    SEED,
    _uid,
    build_passing_local_fixture as _build_passing_local_fixture,
    expected_units,
    make_result,
    make_signal_rows_for_round,
    make_unit_ref,
    signal_rounds,
    write_local_manifest,
    write_result,
    write_signal_log,
)


# ---------------------------------------------------------------------------
# Local-dirs mode: full 4-unit passing gate
# ---------------------------------------------------------------------------


def _run_local(results_dir: Path, signals_dir: Path, **kw):
    kw.setdefault("expected_units", expected_units())
    return acg.run_gate_local(results_dir, signals_dir, EXP_ID, **kw)


def test_local_mode_full_gate_passes(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    report = _run_local(results_dir, signals_dir)
    required_failures = report.required_failures
    assert required_failures == [], [(r.name, r.unit_id, r.detail) for r in required_failures]
    assert report.ok is True
    assert report.szelag is not None and report.szelag["passed"] is True
    assert report.wall_clock_rows and len(report.wall_clock_rows) == 4


def test_local_mode_writes_report_file_and_exits_zero(tmp_path, capsys):
    """CLI end-to-end via --manifest-path (the documented offline workflow:
    the manifest's units are the expected set, its meta feeds the
    methodology_version check)."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    manifest_path = write_local_manifest(tmp_path)
    report_path = tmp_path / "gate-report.md"
    rc = acg.main([
        EXP_ID, "--results-dir", str(results_dir), "--signals-dir", str(signals_dir),
        "--manifest-path", str(manifest_path), "--report-path", str(report_path),
    ])
    assert rc == 0
    assert report_path.is_file()
    text = report_path.read_text()
    assert "Gate exit status: PASS" in text
    out = capsys.readouterr().out
    assert "SZELĄG GATE" in out


# ---------------------------------------------------------------------------
# Expected-unit-set discipline: the gate must fail loudly
# on units whose artifacts are missing or unparseable -- never skip them.
# ---------------------------------------------------------------------------


def test_local_mode_missing_result_file_is_required_failure(tmp_path):
    """An incomplete download (one unit's result JSON absent) must FAIL the
    gate: the expected unit set comes from the manifest/design matrix, not
    from globbing whatever happens to be present."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    (results_dir / f"{uid}.json").unlink()

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.unit_id == uid]
    assert failing, "missing result file for an expected unit produced no required failure"
    assert any("result" in r.detail and uid in (r.unit_id or "") for r in failing)


def test_local_mode_malformed_result_json_is_required_failure(tmp_path):
    """Malformed JSON must be a LOUD failure naming the file and parse error,
    never a silent skip."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    bad_path = results_dir / f"{uid}.json"
    bad_path.write_text("{this is not json")

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert bad_path.name in failing[0].detail
    assert "Expecting" in failing[0].detail or "line 1" in failing[0].detail


def test_local_mode_malformed_signal_line_is_required_failure(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    sig_path = signals_dir / f"{uid}.jsonl"
    with open(sig_path, "a") as fh:
        fh.write("not a json line\n")

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert sig_path.name in failing[0].detail


def test_local_mode_null_signal_line_is_required_failure(
        tmp_path):
    """`null` is syntactically valid JSON but not an object -- appending it
    to rows made every downstream r.get(...) crash the whole gate instead of
    a per-unit parse failure."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    sig_path = signals_dir / f"{uid}.jsonl"
    with open(sig_path, "a") as fh:
        fh.write("null\n")

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert sig_path.name in failing[0].detail
    assert "NoneType" in failing[0].detail or "object" in failing[0].detail
    # the OTHER units were still audited
    assert any(r.name == "baseline_instrumentation" and r.unit_id == _uid("Krum")
               for r in report.results)


def test_local_mode_non_dict_result_json_is_required_failure(tmp_path):
    """A top-level JSON array/string/null result must be a per-unit parse
    failure naming the offending type, not a downstream .get(...) crash."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad_path = results_dir / f"{uid}.json"
    bad_path.write_text("[1, 2, 3]")

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert bad_path.name in failing[0].detail
    assert "list" in failing[0].detail
    assert any(r.name == "baseline_instrumentation" and r.unit_id == _uid("TrustScore")
               for r in report.results)


def test_non_object_provenance_is_required_failure(
        tmp_path):
    """A syntactically valid result whose `provenance` is a string/list crashed
    every provenance check at (result.get('provenance') or {}).get(...) with
    AttributeError, aborting the gate. Must be a per-unit parse failure; the
    audit continues and the report renders."""
    from _calibration_gate_report import render_report

    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum")
    bad["provenance"] = "v1-string"
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "provenance" in failing[0].detail and "str" in failing[0].detail
    assert any(r.name == "baseline_instrumentation" and r.unit_id == _uid("TrustScore")
               for r in report.results)
    render_report(report)  # the report must render


@pytest.mark.parametrize("field,bad_value,type_name", [
    ("server_round", [], "list"),                  # round-7 named (3567232376)
    ("logical_cid", {}, "dict"),                   # round-7 named (3567232376)
    ("scenario_round", [], "list"),                # sweep: declared.get(unhashable) TypeError
    ("signal_log_schema_version", [3], "list"),    # sweep: set-build TypeError
    ("run_started_at", [], "list"),                # sweep: contract uniformity
    ("krum_score", [0.5], "list"),                 # sweep: variance-check set.add TypeError
])
def test_unhashable_row_field_is_required_failure(tmp_path, field, bad_value, type_name):
    """Array/object values on row identity/metadata fields either crash dict/
    set construction (unhashable) or downstream lookups -- each must be a
    per-unit parse failure naming line + field + offending type."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    rows[0][field] = bad_value
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert field in failing[0].detail
    assert type_name in failing[0].detail
    assert "line 1" in failing[0].detail


def test_non_dict_trajectory_item_is_required_failure(tmp_path):
    """A trajectory containing a non-object item crashed check_rounds_consistency
    and audit_run_instrumentation's traj[-1].get(...) -- per-unit failure."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    bad = make_result("TGE")
    bad["trajectory"].append("not-a-row")
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing and "trajectory" in failing[0].detail and "str" in failing[0].detail


def test_unhashable_trajectory_round_is_required_failure(tmp_path):
    """A trajectory item whose `round` is an array crashed the set build in
    check_rounds_consistency -- per-unit failure."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    bad = make_result("TrustScore")
    bad["trajectory"][0]["round"] = [0]
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing and "round" in failing[0].detail and "list" in failing[0].detail


def test_non_numeric_final_accuracy_is_required_failure(tmp_path):
    """float('high') in the Szelag gate crashed the report -- per-unit failure."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum")
    bad["final_accuracy"] = "high"
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing and "final_accuracy" in failing[0].detail


def test_non_numeric_hparams_value_fails_not_crashes():
    """A non-numeric hparam value must be a loud mismatch failure in
    check_provenance_hparams, not a float() ValueError crash."""
    unit = make_unit_ref("Krum")
    result = make_result("Krum", hparams_lr=None)
    result["hparams"]["lr"] = "abc"
    locked = {"persistent_optimizer": {"lr": 0.001, "local_epochs": 5}}
    check = gl.check_provenance_hparams(unit, result, locked)  # must not raise
    assert check.passed is False
    assert "abc" in check.detail


def test_non_numeric_tge_threshold_fails_not_crashes():
    """A non-numeric tge_operational_threshold must be a loud provenance
    failure, not a float() ValueError crash."""
    unit = make_unit_ref("TGE")
    result = make_result("TGE", provenance_overrides={"tge_operational_threshold": "x"})
    check = gl.check_provenance_tge_fields(unit, result)  # must not raise
    assert check.passed is False
    assert "tge_operational_threshold" in check.detail


def test_design_matrix_expected_units_for_exp005c(tmp_path):
    """With no --manifest-path the CLI derives the expected unit set from the
    EXP-005c design doc (docs/experiments/EXP-005c-calibration.md matrix
    front-matter). The doc's batch fields legitimately read TBD pre-deploy
    (a LAUNCH guard) -- the audit-time reader must tolerate that."""
    docs = tmp_path / "docs" / "experiments"
    docs.mkdir(parents=True)
    (docs / f"{EXP_ID}-calibration.md").write_text("""---
exp_id: EXP-005c
slug: calibration
hypothesis: H2
methodology_version: v1.19
matrix:
  defenses: [Krum, TrustScore, Krum+TGE, TGE]
  scenarios: [S4_full_mix]
  seeds: [42]
  mode: persistent_optimizer
  max_per_client: 2000000
  rounds: 50
batch:
  job_queue: TBD
  job_definition: TBD
---
# Synthetic calibration design fixture
""")
    units = gs.load_design_matrix_units(tmp_path, EXP_ID)
    assert len(units) == 4
    assert sorted(u.config for u in units) == sorted(["Krum", "TrustScore", "Krum+TGE", "TGE"])
    assert {u.scenario for u in units} == {"S4_full_mix"}
    assert {u.seed for u in units} == {42}
    assert {u.mode for u in units} == {"persistent_optimizer"}
    assert {u.rounds for u in units} == {50}


def test_local_manifest_units_and_meta(tmp_path):
    manifest_path = write_local_manifest(tmp_path)
    meta, units = gs.load_local_manifest(manifest_path, expected_exp_id=EXP_ID)
    assert meta["methodology_version"] == ACTIVE_METHODOLOGY
    assert [u.unit_id for u in units] == [u.unit_id for u in expected_units()]


def test_manifest_exp_id_mismatch_is_refused(
        tmp_path):
    """A manifest for ANOTHER sweep with the same matrix must be refused --
    unit_ids do not embed the exp id, so trusting the units verbatim would
    audit the wrong sweep's artifacts and could emit a passing report for
    the requested EXP-005c."""
    manifest_path = write_local_manifest(tmp_path, exp_id="EXP-999")
    with pytest.raises(gs.ExpectedUnitsError) as excinfo:
        gs.load_local_manifest(manifest_path, expected_exp_id=EXP_ID)
    msg = str(excinfo.value)
    assert "EXP-999" in msg and EXP_ID in msg  # names BOTH ids


def test_cli_exits_2_on_wrong_exp_id_manifest(tmp_path, capsys):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    manifest_path = write_local_manifest(tmp_path, exp_id="EXP-999")
    rc = acg.main([
        EXP_ID, "--results-dir", str(results_dir), "--signals-dir", str(signals_dir),
        "--manifest-path", str(manifest_path), "--report-path", str(tmp_path / "r.md"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "EXP-999" in err and EXP_ID in err


def test_local_mode_invalid_utf8_result_is_required_failure(
        tmp_path):
    """Invalid UTF-8 raises UnicodeDecodeError from read_text() BEFORE
    json.loads runs -- it must take the same per-unit required-failure path
    as JSON corruption, not abort the CLI."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad_path = results_dir / f"{uid}.json"
    bad_path.write_bytes(b"\x80\x81\xfe\xff")

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert bad_path.name in failing[0].detail
    assert "codec" in failing[0].detail or "decode" in failing[0].detail
    # the OTHER units were still audited
    assert any(r.name == "baseline_instrumentation" and r.unit_id == _uid("TrustScore")
               for r in report.results)


def test_local_mode_invalid_utf8_signal_is_required_failure(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    sig_path = signals_dir / f"{uid}.jsonl"
    sig_path.write_bytes(b"\x80\x81\xfe\xff")

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "local_artifact_parse" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert sig_path.name in failing[0].detail
    assert any(r.name == "baseline_instrumentation" and r.unit_id == _uid("Krum")
               for r in report.results)


# ---------------------------------------------------------------------------
# Failure fixtures (local-dirs mode unless noted)
# ---------------------------------------------------------------------------


def test_duplicate_run_started_at_fails_hygiene(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    rows = make_signal_rows_for_round("TrustScore", 2)
    rows += make_signal_rows_for_round("TrustScore", 3, run_started_at="2026-07-10T01:00:00Z")
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    names = [r.name for r in report.required_failures]
    assert "signal_hygiene.run_started_at" in names


def test_duplicate_round_cid_pair_fails_hygiene(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    rows = make_signal_rows_for_round("Krum", 2)
    rows.append(rows[0])  # exact duplicate (server_round, logical_cid)
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    names = [r.name for r in report.required_failures]
    assert "signal_hygiene.duplicate_rows" in names


def test_krumtge_all_rows_scored_is_positional_bug_signature(tmp_path):
    """Every row carrying a tge_score in the Krum+TGE chain is exactly the
    pre-v1.17 positional-bug signature (METHODOLOGY_LOG.md v1.17) -- must FAIL."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum+TGE")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum+TGE", rnd, n_scored=20))  # ALL scored
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "krumtge_spot_check"]
    assert failing, [(r.name, r.detail) for r in report.required_failures]
    assert "positional-bug signature" in failing[0].detail or ">=" in failing[0].detail


def test_missing_done_marker_fails_s3_integrity():
    """S3-mode-only check: done-marker missing must FAIL. Uses FakeGateStore +
    the real persist_unit()/write_manifest() -- NOT reimplemented."""
    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units, meta={"methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc"})
    unit = units[0]

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        result_path = tdp / "r.json"
        write_result(result_path, make_result("Krum"))
        signal_path = tdp / "s.jsonl"
        rows = []
        for rnd in signal_rounds():
            rows.extend(make_signal_rows_for_round("Krum", rnd))
        write_signal_log(signal_path, rows)

        # Upload result + signal directly WITHOUT ever calling persist_unit
        # (which would also write the done-marker) -- simulates a crash
        # between payload upload and marker commit.
        from praxis_exp import storage
        store.put_file(storage.result_key(EXP_ID, unit.unit_id), result_path)
        store.put_file(storage.signal_key(EXP_ID, unit.unit_id), signal_path)

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "s3_integrity"]
    assert failing and "done-marker missing" in failing[0].detail


def _seed_s3_store_two_units(tmp_path):
    """FakeGateStore with a 2-unit manifest; returns (store, krum_unit,
    trustscore_unit) with NO artifacts uploaded yet."""
    store = gs.FakeGateStore()
    units = expand_matrix(["Krum", "TrustScore"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units,
                   meta={"methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc"})
    return store, units[0], units[1]


def _persist_valid_unit(store, unit, tmp_path, config):
    result_path = tmp_path / f"{unit.unit_id}_r.json"
    write_result(result_path, make_result(config))
    signal_path = tmp_path / f"{unit.unit_id}_s.jsonl"
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round(config, rnd))
    write_signal_log(signal_path, rows)
    persist_unit(store, EXP_ID, unit.unit_id, result_path, signal_path)


def test_s3_mode_corrupt_result_json_is_required_failure(
        tmp_path):
    """A truncated/non-JSON result object in S3 must become a per-unit
    REQUIRED failure naming the S3 key and parse error; the audit must
    continue to the remaining units and the report must render -- not abort
    run_gate_s3 with a traceback."""
    from praxis_exp import storage as pstorage

    store, krum, trust = _seed_s3_store_two_units(tmp_path)
    # Krum unit: corrupt result, valid signal, marker present (persisted then overwritten)
    _persist_valid_unit(store, krum, tmp_path, "Krum")
    store.put_bytes(pstorage.result_key(EXP_ID, krum.unit_id), b'{"truncated": ')
    # TrustScore unit: fully valid
    _persist_valid_unit(store, trust, tmp_path, "TrustScore")

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "s3_artifact_parse" and r.unit_id == krum.unit_id]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert pstorage.result_key(EXP_ID, krum.unit_id) in failing[0].detail
    assert "Expecting" in failing[0].detail or "line 1" in failing[0].detail
    # the OTHER unit was still fully audited
    assert any(r.name == "baseline_instrumentation" and r.unit_id == trust.unit_id
               for r in report.results)


def test_s3_mode_corrupt_signal_line_is_required_failure(tmp_path):
    from praxis_exp import storage as pstorage

    store, krum, trust = _seed_s3_store_two_units(tmp_path)
    _persist_valid_unit(store, krum, tmp_path, "Krum")
    good_signal = store.get_bytes(pstorage.signal_key(EXP_ID, krum.unit_id))
    store.put_bytes(pstorage.signal_key(EXP_ID, krum.unit_id),
                    good_signal + b"not a json line\n")
    _persist_valid_unit(store, trust, tmp_path, "TrustScore")

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "s3_artifact_parse" and r.unit_id == krum.unit_id]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert pstorage.signal_key(EXP_ID, krum.unit_id) in failing[0].detail
    assert "line" in failing[0].detail


def test_ramp_5_provenance_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    bad = make_result("TGE", provenance_overrides={"tge_ramp_rounds": 5})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "provenance.tge_fields" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "tge_ramp_rounds" in failing[0].detail


def test_non_tge_unit_with_lstm_state_enabled_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum", provenance_overrides={"tge_lstm_state": "enabled"})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "provenance.tge_fields" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "tge_lstm_state" in failing[0].detail


def test_hparams_mismatch_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    bad = make_result("TrustScore", hparams_lr=0.9)  # wildly wrong lr
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "provenance.hparams" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "lr" in failing[0].detail


def test_hparams_absent_from_result_is_skipped_not_failed(tmp_path):
    """Ground-truth reality: real run_phase4_flower.py result JSONs carry no
    hparams at all. That must be a loud SKIP, not a silent pass and not a
    required failure."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum", include_hparams=False)
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    hp = [r for r in report.results if r.name == "provenance.hparams" and r.unit_id == uid]
    assert len(hp) == 1
    assert hp[0].required is False
    assert hp[0].passed is True
    assert "SKIPPED" in hp[0].detail
    assert report.ok is True  # a SKIP must not fail the gate


# ---------------------------------------------------------------------------
# Additional required-check coverage (mode / seed)
# ---------------------------------------------------------------------------


def test_wrong_optimizer_state_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum", optimizer_state="reset")
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "provenance.optimizer_state" and r.unit_id == uid]
    assert failing


def test_wrong_seed_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    bad = make_result("TrustScore", seed=137)
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "provenance.seed" and r.unit_id == uid]
    assert failing


def test_low_participation_round_fails_hygiene(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    # scenario_round 1 declares 20 participants in S4_full_mix -- 10 is TRUE truncation
    rows = make_signal_rows_for_round("Krum", 2, n_participants=10)
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "signal_hygiene.participation" and r.unit_id == uid]
    assert failing


def test_scenario_declared_disconnect_round_passes_participation():
    """S4 legitimately schedules 11-participant disconnect rounds; the
    participation check must honor the scenario's declared per-round count
    (mirroring runner A3), not a flat 0.95*20 floor that would false-fail
    every real S4 signal log (S4 histogram: 38 rounds @ 20, 8 @ 19, 4 @ 11)."""
    unit = make_unit_ref("Krum")
    # scenario_round 5 declared at 11 participants
    rows = make_signal_rows_for_round("Krum", 6, n_participants=11)
    declared = {5: 11}
    results = gl.check_signal_hygiene(unit, rows, declared_participants=declared)
    participation = [r for r in results if r.name == "signal_hygiene.participation"]
    assert len(participation) == 1
    assert participation[0].passed is True, participation[0].detail


def test_truncated_declared_round_fails_participation():
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 6, n_participants=9)  # declared 11, got 9
    declared = {5: 11}
    results = gl.check_signal_hygiene(unit, rows, declared_participants=declared)
    participation = [r for r in results if r.name == "signal_hygiene.participation"]
    assert participation[0].passed is False
    assert "11" in participation[0].detail


def test_missing_entire_round_fails_coverage(
        tmp_path):
    """A log missing an ENTIRE server round produced no by_round entry and
    sailed through -- silently costing recall/FPR that round. Expected signal
    rounds are {2..rounds+1}: server round 1 is cid discovery (_round_offset=1,
    scenario_strategy.py:135) and _maybe_log_signals logs iff the row's
    scenario_round (= server_round - 1) is schedule-declared
    (scenario_strategy.py:533-536)."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    rows = []
    for rnd in signal_rounds():
        if rnd == 3:
            continue  # drop server round 3 entirely (sync truncated mid-run)
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "signal_hygiene.round_coverage" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "3" in failing[0].detail


def test_round_coverage_expected_set_follows_schedule_convention():
    """Expected server rounds derive from the scenario's DECLARED scenario
    rounds shifted by the discovery offset (+1); rounds beyond the unit's
    horizon are clipped."""
    unit = make_unit_ref("Krum")  # rounds=3 -> server rounds 2..4
    declared = {1: 20, 2: 20, 3: 20}
    rows = make_signal_rows_for_round("Krum", 2)  # only server round 2 present
    results = gl.check_signal_hygiene(unit, rows, declared_participants=declared)
    coverage = [r for r in results if r.name == "signal_hygiene.round_coverage"]
    assert len(coverage) == 1
    assert coverage[0].passed is False
    assert "3" in coverage[0].detail and "4" in coverage[0].detail


def test_round_coverage_caps_missing_round_listing():
    import dataclasses

    unit = dataclasses.replace(make_unit_ref("Krum"), rounds=20)  # server rounds 2..21
    declared = {r: 20 for r in range(1, 21)}
    rows = make_signal_rows_for_round("Krum", 2)  # 19 rounds missing
    results = gl.check_signal_hygiene(unit, rows, declared_participants=declared)
    coverage = [r for r in results if r.name == "signal_hygiene.round_coverage"]
    assert coverage[0].passed is False
    assert "more" in coverage[0].detail  # capped listing: "[...] +N more"


def test_extra_rounds_beyond_horizon_fail_coverage(
        tmp_path):
    """A log containing every expected round PLUS rounds beyond the manifest
    horizon must FAIL: nothing else rejects well-formed extra rounds (same
    run_started_at, non-duplicate pairs, declared/floor participation --
    verified live against every hygiene sub-check), so downstream recall/FPR
    would silently pool rows from an appended/longer run."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    rows = []
    for rnd in list(signal_rounds()) + [5, 6]:  # 5, 6 beyond ROUNDS=3 horizon
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "signal_hygiene.round_coverage" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "5" in failing[0].detail and "unexpected" in failing[0].detail.lower()


def test_unexpected_rounds_capped_listing():
    unit = make_unit_ref("Krum")  # rounds=3 -> expected {2,3,4}
    declared = {1: 20, 2: 20, 3: 20}
    rows = []
    for rnd in list(range(2, 5)) + list(range(5, 16)):  # 11 unexpected rounds
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    results = gl.check_signal_hygiene(unit, rows, declared_participants=declared)
    coverage = [r for r in results if r.name == "signal_hygiene.round_coverage"]
    assert coverage[0].passed is False
    assert "more" in coverage[0].detail  # capped: 8 shown "+3 more"


def test_mixed_schema_version_types_fail_cleanly(
        tmp_path):
    """Mixed incomparable schema_version values (missing-field -> None, a '2'
    string, int 3) crashed the raw sorted() with TypeError BEFORE the intended
    required failure was emitted, aborting the whole report. The check must
    emit the failure naming the observed values, and the report must render."""
    from _calibration_gate_report import render_report

    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("TrustScore", rnd))
    del rows[0]["signal_log_schema_version"]          # -> None
    rows[1]["signal_log_schema_version"] = "2"        # -> str
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)  # rest stay int 3

    report = _run_local(results_dir, signals_dir)  # must not raise
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "signal_hygiene.schema_version" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]
    assert "None" in failing[0].detail
    assert "'2'" in failing[0].detail
    assert "3" in failing[0].detail  # all three distinct observed values named
    text = render_report(report)  # the report must render
    assert "schema_version" in text


def test_mixed_type_duplicate_pairs_fail_cleanly():
    """Two duplicated (server_round, logical_cid) pairs where one pair has
    server_round=None crashed the tuple sort -- must fail cleanly instead."""
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 2)
    broken = dict(rows[0])
    broken["server_round"] = None
    rows += [broken, dict(broken)]  # duplicate (None, cid) pair
    rows.append(dict(rows[1]))      # duplicate (2, cid) pair
    results = gl.check_signal_hygiene(unit, rows)  # must not raise
    dup = [r for r in results if r.name == "signal_hygiene.duplicate_rows"]
    assert dup[0].passed is False


def test_unexpected_round_mixed_types_fail_cleanly():
    """A row with server_round=None alongside an int extra round put mixed
    types into the unexpected-rounds sort -- must fail cleanly."""
    unit = make_unit_ref("Krum")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    broken = dict(rows[0])
    broken["server_round"] = None
    broken["logical_cid"] = "client_x"
    rows.append(broken)
    rows.extend(make_signal_rows_for_round("Krum", 7))  # int extra round
    results = gl.check_signal_hygiene(unit, rows)  # must not raise
    coverage = [r for r in results if r.name == "signal_hygiene.round_coverage"]
    assert coverage[0].passed is False
    assert "7" in coverage[0].detail


def test_rounds_consistency_mixed_type_trajectory_rounds_fail_cleanly():
    """A trajectory carrying BOTH a string round and an int extra round put
    mixed types into check_rounds_consistency's 'extra' sort -- must fail
    cleanly, not crash."""
    unit = make_unit_ref("Krum")
    result = make_result("Krum", trajectory_rounds=list(range(0, ROUNDS + 2)))
    result["trajectory"].append({"round": 7, "accuracy": 0.9, "f1": 0.9, "loss": 0.1})
    result["trajectory"].append({"round": "9", "accuracy": 0.9, "f1": 0.9, "loss": 0.1})
    check = gl.check_rounds_consistency(unit, result)  # must not raise
    assert check.passed is False
    assert "9" in check.detail and "7" in check.detail


def test_run_started_at_all_missing_fails(
):
    """A log where EVERY row lacks run_started_at must FAIL -- the old
    set-based check saw {None}, len==1, and passed it as 'single
    run_started_at=None'."""
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 1)
    for r in rows:
        del r["run_started_at"]
    results = gl.check_signal_hygiene(unit, rows)
    started = [r for r in results if r.name == "signal_hygiene.run_started_at"]
    assert len(started) == 1
    assert started[0].passed is False
    assert str(len(rows)) in started[0].detail  # counts the rows missing the field


def test_run_started_at_mixed_missing_fails_cleanly():
    """Mixed None/str must fail cleanly with a missing-row count -- the old
    sorted({None, 'str'}) crashed with TypeError before it could even fail."""
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 1)
    for r in rows[:5]:
        r["run_started_at"] = None
    results = gl.check_signal_hygiene(unit, rows)  # must not raise
    started = [r for r in results if r.name == "signal_hygiene.run_started_at"]
    assert started[0].passed is False
    assert "5" in started[0].detail


def test_run_started_at_empty_string_fails():
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 1, run_started_at="")
    results = gl.check_signal_hygiene(unit, rows)
    started = [r for r in results if r.name == "signal_hygiene.run_started_at"]
    assert started[0].passed is False


def test_missing_signal_log_fails_local_presence(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    (signals_dir / f"{uid}.jsonl").unlink()

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "local_presence" and r.unit_id == uid]
    assert failing and "signal log missing" in failing[0].detail


# ---------------------------------------------------------------------------
# Szeląg gate: informational only, never blocks exit code
# ---------------------------------------------------------------------------


def test_szelag_gate_fail_does_not_block_exit_code(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum", final_accuracy=0.5)  # way outside anchor +/- tolerance
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.szelag is not None
    assert report.szelag["passed"] is False
    assert report.ok is True  # no OTHER required check was broken by this edit


def test_szelag_gate_boundary():
    within = gl.check_szelag_gate({"final_accuracy": gl.SZELAG_ANCHOR + gl.SZELAG_TOLERANCE})
    outside = gl.check_szelag_gate({"final_accuracy": gl.SZELAG_ANCHOR + gl.SZELAG_TOLERANCE + 0.001})
    assert within["passed"] is True
    assert outside["passed"] is False


# ---------------------------------------------------------------------------
# Krum dynamic survivor count (v1.19) + cross-check vs the real plugin
# ---------------------------------------------------------------------------


def test_krum_dynamic_keep_matches_effective_f_formula():
    """The auditor's ONE keep-count helper must equal the deployed plugin's
    per-round math for every plausible cohort size: f from the REAL
    KrumDefensePlugin._effective_f (imported, not re-derived), keep = the
    dynamic branch of filter_updates (max(1, n-f-2))."""
    from flowerfl.byzantine_defense import KrumDefensePlugin

    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    for n in range(4, 26):
        f = plugin._effective_f(n)
        assert gl.krum_dynamic_keep(n) == max(1, n - f - 2), f"n={n}"
    # documented operating points (methodology v1.19)
    assert gl.krum_dynamic_keep(20) == 9
    assert gl.krum_dynamic_keep(11) == 4


def test_krum_dynamic_keep_matches_real_filter_updates_survivor_count():
    """Drive the REAL plugin end-to-end (score_updates + filter_updates) at
    the two documented operating points and count actual survivors."""
    import numpy as np
    from flwr.common import ndarrays_to_parameters
    from flowerfl.byzantine_defense import KrumDefensePlugin

    class _FakeClient:
        def __init__(self, cid: str):
            self.cid = cid

    class _FakeFitRes:
        def __init__(self, vec):
            self.parameters = ndarrays_to_parameters([vec.astype(np.float32)])
            self.num_examples = 100
            self.metrics = {}

    rng = np.random.default_rng(42)
    for n in (20, 11):
        results = [(_FakeClient(f"c{i}"), _FakeFitRes(rng.normal(0.0, 0.01, size=8)))
                   for i in range(n)]
        plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
        scores = plugin.score_updates(results, server_round=5)
        kept = plugin.filter_updates(results, scores)
        assert len(kept) == gl.krum_dynamic_keep(n), f"n={n}"


def _krumtge_rows_all_rounds(*, n_scored: int | None = None,
                             override_round: int | None = None,
                             override_kwargs: dict | None = None) -> list:
    """Krum+TGE rows for every expected server round (2..ROUNDS+1); one round
    may be overridden with different participant/scored counts."""
    rows = []
    for rnd in signal_rounds():
        if override_round is not None and rnd == override_round:
            rows.extend(make_signal_rows_for_round("Krum+TGE", rnd, **(override_kwargs or {})))
        else:
            rows.extend(make_signal_rows_for_round(
                "Krum+TGE", rnd, n_participants=20, n_scored=n_scored))
    return rows


def test_krumtge_spot_check_full_cohort_expects_dynamic_keep_9():
    rows = _krumtge_rows_all_rounds(n_scored=9)
    unit = make_unit_ref("Krum+TGE")
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is True, result.detail


def test_krumtge_spot_check_rejects_stale_static_keep_18():
    """The pre-v1.19 static sizing (malicious-fraction=0 -> f=1 -> keep 18 at
    n=20) must now FAIL -- regression guard against the auditor reverting to
    the old expectation."""
    rows = _krumtge_rows_all_rounds(n_scored=18)
    unit = make_unit_ref("Krum+TGE")
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is False
    assert "9" in result.detail  # detail names the dynamic expectation


def test_krumtge_spot_check_disconnect_round_expects_keep_4():
    """S3/S4 disconnect rounds (n=11): f=ceil(11/2)-1=5, keep=11-5-2=4. The
    check must size per ROUND, not per full cohort."""
    unit = make_unit_ref("Krum+TGE")
    ok_rows = _krumtge_rows_all_rounds(
        override_round=4, override_kwargs={"n_participants": 11, "n_scored": 4})
    assert gl.check_krumtge_spot_check(unit, ok_rows).passed is True
    bad_rows = _krumtge_rows_all_rounds(
        override_round=4, override_kwargs={"n_participants": 11, "n_scored": 9})
    result = gl.check_krumtge_spot_check(unit, bad_rows)
    assert result.passed is False
    assert "4" in result.detail


def test_signal_hygiene_accepts_schema_v3_v4_and_v5():
    """v4 (adds tge_ema_score, GWU-53) and v5 (adds the aggregation coefficient
    + the H3 re-entry event contract) must audit alongside historical v3 logs
    (e.g. EXP-011); an unsupported version still fails.

    v5 is version-GATED, never migrated: a v4 log keeps parsing byte-identically
    under the same readers, and no v4 row is reinterpreted."""
    unit = make_unit_ref("Krum")
    for ver in (3, 4, 5):
        rows = make_signal_rows_for_round("Krum", 2, schema_version=ver)
        sv = [r for r in gl.check_signal_hygiene(unit, rows)
              if r.name == "signal_hygiene.schema_version"]
        assert sv and sv[0].passed is True, f"v{ver}: {sv[0].detail if sv else 'MISSING'}"
    rows = make_signal_rows_for_round("Krum", 2, schema_version=2)
    sv = [r for r in gl.check_signal_hygiene(unit, rows)
          if r.name == "signal_hygiene.schema_version"]
    assert sv[0].passed is False and "2" in sv[0].detail


def test_krumtge_spot_check_flags_non_null_ema_on_unscored_row():
    """tge_ema_score joins TGE_FIELDS_NULL_WHEN_UNSCORED (schema v4): an unscored
    row carrying a non-null EMA score is a population leak and must fail, so a
    bank log's EMA leg can't be silently mistaken for legitimately null."""
    unit = make_unit_ref("Krum+TGE")
    rows = _krumtge_rows_all_rounds(n_scored=9)
    assert gl.check_krumtge_spot_check(unit, rows).passed is True  # clean run
    unscored = next(r for r in rows if r.get("tge_score") is None)
    unscored["tge_ema_score"] = 0.72                                # inject a leak
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is False
    assert "tge_ema_score" in result.detail


def test_krumtge_unscored_scheduled_round_fails(
):
    """A scheduled round whose rows are present but carry ZERO tge scores is
    TGE scoring silently vanishing -- it must FAIL, not be skipped. Ground
    truth that EVERY scheduled round must carry scores: score_updates
    populates _last_details unconditionally per call (byzantine_defense.py:
    678-693) and TGEnsembleModel.score_client returns a NON-NULL final_score
    in every phase, warmup included (final_score=1.0, phase='warmup' --
    rmc/tg_ensemble.py:801-810; pre_gbdt fallback :817-826), while Krum's
    keep = max(1, ...) always hands TGE >= 1 survivor."""
    unit = make_unit_ref("Krum+TGE")
    rows = _krumtge_rows_all_rounds(
        override_round=3, override_kwargs={"n_participants": 20, "n_scored": 0})
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is False
    assert "3" in result.detail
    assert "no tge-scored" in result.detail.lower()


def test_krumtge_absent_scheduled_round_fails_spot_check():
    """An expected round entirely absent from the log must also fail the
    spot-check (defense in depth alongside signal_hygiene.round_coverage)."""
    unit = make_unit_ref("Krum+TGE")
    rows = []
    for rnd in signal_rounds():
        if rnd == 3:
            continue
        rows.extend(make_signal_rows_for_round("Krum+TGE", rnd))
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is False
    assert "3" in result.detail


def test_krumtge_unscored_round_fails_full_gate(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum+TGE")
    rows = _krumtge_rows_all_rounds(
        override_round=3, override_kwargs={"n_participants": 20, "n_scored": 0})
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "krumtge_spot_check" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]


def test_krumtge_spot_check_null_fields_ignores_tge_threshold():
    """tge_threshold is a round-level constant even on unscored rows -- must
    NOT be flagged as an unexpected non-null field."""
    rows = _krumtge_rows_all_rounds(n_scored=9)
    unit = make_unit_ref("Krum+TGE")
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is True, result.detail


def test_krumtge_spot_check_flags_nonnull_score_field_on_unscored_row():
    rows = _krumtge_rows_all_rounds(n_scored=9)
    # corrupt: an unscored row leaks a gbdt_score despite tge_score being null
    for r in rows:
        if r["tge_score"] is None:
            r["tge_gbdt_score"] = 0.42
            break
    unit = make_unit_ref("Krum+TGE")
    result = gl.check_krumtge_spot_check(unit, rows)
    assert result.passed is False
    assert "tge_gbdt_score" in result.detail


# ---------------------------------------------------------------------------
# krum_score variance (pre-v1.19 certification-bound flattening signature)
# ---------------------------------------------------------------------------


def test_flat_krum_scores_fail_variance_check(tmp_path):
    """All krum_score == 1.0 is the old certification-bound guard's uniform
    fallback (methodology v1.19) -- recall@10%FPR uncomputable; must FAIL."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd, flat_krum_scores=True))
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures if r.name == "krum_score_variance" and r.unit_id == uid]
    assert failing, [(r.name, r.unit_id, r.detail) for r in report.required_failures]


def test_krum_score_variance_passes_with_varied_scores():
    unit = make_unit_ref("Krum")
    rows = make_signal_rows_for_round("Krum", 1)
    result = gl.check_krum_score_variance(unit, rows)
    assert result.passed is True, result.detail


def test_krum_score_variance_applies_to_krumtge_unit_too(tmp_path):
    """Krum+TGE deploys the same KrumDefensePlugin upstream -- the flattening
    bug would hit it identically, so the variance check covers it as well."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum+TGE")
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum+TGE", rnd, flat_krum_scores=True))
    write_signal_log(signals_dir / f"{uid}.jsonl", rows)

    report = _run_local(results_dir, signals_dir)
    failing = [r for r in report.required_failures if r.name == "krum_score_variance" and r.unit_id == uid]
    assert failing


# ---------------------------------------------------------------------------
# Defense-sizing provenance (v1.19 / PR #12 round-2)
# ---------------------------------------------------------------------------


def test_missing_krum_f_policy_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum")
    del bad["provenance"]["krum_f_policy"]
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "provenance.defense_sizing" and r.unit_id == uid]
    assert failing and "krum_f_policy" in failing[0].detail


def test_non_krum_unit_with_dynamic_policy_fails(tmp_path):
    """TGE-only deploys no Krum layer: krum_f_policy must be 'n/a'
    (run_phase4_flower.py::_defense_provenance_fields L703-709)."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TGE")
    bad = make_result("TGE", provenance_overrides={"krum_f_policy": KRUM_F_POLICY})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "provenance.defense_sizing" and r.unit_id == uid]
    assert failing and "krum_f_policy" in failing[0].detail


def test_wrong_declared_adversaries_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum+TGE")
    bad = make_result("Krum+TGE", provenance_overrides={"scenario_declared_adversaries": 1})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "provenance.defense_sizing" and r.unit_id == uid]
    assert failing and "scenario_declared_adversaries" in failing[0].detail


def test_wrong_cohort_size_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("TrustScore")
    bad = make_result("TrustScore", provenance_overrides={"defense_cohort_size": 18})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "provenance.defense_sizing" and r.unit_id == uid]
    assert failing and "defense_cohort_size" in failing[0].detail


# ---------------------------------------------------------------------------
# Rounds consistency (PR #13 --rounds passthrough recurrence guard)
# ---------------------------------------------------------------------------


def test_rounds_consistency_truncated_trajectory_fails(tmp_path):
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    # trajectory stops at round 2 despite declared ROUNDS=3 (expect 0..4)
    bad = make_result("Krum", trajectory_rounds=[0, 1, 2])
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "rounds_consistency" and r.unit_id == uid]
    assert failing


def test_rounds_consistency_missing_discovery_round_fails():
    """Trajectory covering only 0..rounds (missing the +1 discovery round's
    final eval) is exactly what a --rounds passthrough regression looks like
    when the container runs fewer rounds than the manifest declares."""
    unit = make_unit_ref("Krum")
    result = make_result("Krum", trajectory_rounds=list(range(0, ROUNDS + 1)))  # 0..3, not 0..4
    check = gl.check_rounds_consistency(unit, result)
    assert check.passed is False
    assert str(ROUNDS + 1) in check.detail


def test_rounds_consistency_passes_on_full_coverage():
    unit = make_unit_ref("Krum")
    result = make_result("Krum")  # trajectory 0..ROUNDS+1
    check = gl.check_rounds_consistency(unit, result)
    assert check.passed is True, check.detail


# ---------------------------------------------------------------------------
# Manifest meta tolerance (PR #13: prior_launches + launched_at)
# ---------------------------------------------------------------------------


def test_manifest_meta_tolerates_prior_launches_and_launched_at(tmp_path):
    """PR #13 adds prior_launches / launched_at to the manifest meta block;
    the gate's manifest reads must tolerate unknown meta keys."""
    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    meta = {
        "methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc",
        "git_sha": "deadbeef", "n_units": 1,
        "prior_launches": [{"launched_at": "2026-07-11T00:00:00Z", "git_sha": "cafe"}],
        "launched_at": "2026-07-12T00:00:00Z",
    }
    write_manifest(store, EXP_ID, units, meta=meta)
    unit = units[0]

    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("Krum"))
    signal_path = tmp_path / "s.jsonl"
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signal_path, rows)
    persist_unit(store, EXP_ID, unit.unit_id, result_path, signal_path)

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False)
    mv = [r for r in report.results if r.name == "provenance.methodology_version"]
    assert len(mv) == 1 and mv[0].passed is True, mv[0].detail
    assert report.ok is True, [(r.name, r.unit_id, r.detail) for r in report.required_failures]


# ---------------------------------------------------------------------------
# S3-mode manifest exp_id guard
# ---------------------------------------------------------------------------


def test_s3_mode_wrong_manifest_exp_id_is_refused():
    """S3 mode must not discard read_manifest's exp_id: a copied/corrupt
    sweeps/EXP-005c/manifest.json naming another experiment would drive the
    audit with that sweep's units while checking EXP-005c's prefix -- the
    same class as the --manifest-path guard."""
    import json as _json

    from praxis_exp import storage as pstorage

    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units,
                   meta={"methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc"})
    # corrupt the payload's exp_id in place (same S3 key, wrong experiment)
    payload = _json.loads(store.get_bytes(pstorage.manifest_key(EXP_ID)))
    payload["exp_id"] = "EXP-999"
    store.put_bytes(pstorage.manifest_key(EXP_ID), _json.dumps(payload).encode())

    with pytest.raises(gs.ExpectedUnitsError) as excinfo:
        acg.run_gate_s3(store, EXP_ID, check_mlflow=False)
    msg = str(excinfo.value)
    assert "EXP-999" in msg and EXP_ID in msg


# ---------------------------------------------------------------------------
# Active methodology version
# ---------------------------------------------------------------------------


def test_active_methodology_version_parses_top_entry(tmp_path):
    """Header format ground truth: '## vX.Y — date — title' (em-dash), newest
    entry at the top -- same append-only convention praxis_exp/scaffold.py's
    _current_methodology_version parses."""
    log = tmp_path / "METHODOLOGY_LOG.md"
    log.write_text(
        "# Methodology log — chronological evolution\n\n"
        "## v2.3 — 2026-08-01 — Newest entry\n\nbody\n\n"
        "## v2.2 — 2026-07-20 — Older entry\n\nbody\n"
    )
    assert gl.active_methodology_version(log) == "v2.3"


def test_active_methodology_version_missing_or_headerless_raises(tmp_path):
    with pytest.raises(RuntimeError):
        gl.active_methodology_version(tmp_path / "nope.md")
    empty = tmp_path / "empty.md"
    empty.write_text("# Methodology log\n\nno version headers here\n")
    with pytest.raises(RuntimeError):
        gl.active_methodology_version(empty)


def test_active_methodology_version_real_log_format():
    import re

    assert re.fullmatch(r"v\d+\.\d+", ACTIVE_METHODOLOGY)


def test_stale_manifest_methodology_version_fails(tmp_path):
    """A manifest carrying an out-of-date methodology_version must FAIL
    against the audit-time active version -- 'any non-empty string' let
    stale metadata pass."""
    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units,
                   meta={"methodology_version": "v1.11", "image_digest": "sha256:abc"})
    unit = units[0]
    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("Krum"))
    signal_path = tmp_path / "s.jsonl"
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signal_path, rows)
    persist_unit(store, EXP_ID, unit.unit_id, result_path, signal_path)

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False)
    assert report.ok is False
    mv = [r for r in report.required_failures if r.name == "provenance.methodology_version"]
    assert mv, [(r.name, r.detail) for r in report.required_failures]
    assert "v1.11" in mv[0].detail and ACTIVE_METHODOLOGY in mv[0].detail


def test_expect_methodology_version_override_allows_historical(tmp_path):
    """--expect-methodology-version lets a historical experiment (frozen at
    an older methodology) be audited without false version failures."""
    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units,
                   meta={"methodology_version": "v1.11", "image_digest": "sha256:abc"})
    unit = units[0]
    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("Krum"))
    signal_path = tmp_path / "s.jsonl"
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signal_path, rows)
    persist_unit(store, EXP_ID, unit.unit_id, result_path, signal_path)

    report = acg.run_gate_s3(store, EXP_ID, check_mlflow=False,
                             expected_methodology_version="v1.11")
    mv = [r for r in report.results if r.name == "provenance.methodology_version"]
    assert len(mv) == 1 and mv[0].passed is True, mv[0].detail


def test_unit_result_methodology_field_must_match_when_present(tmp_path):
    """The runner's result JSON carries no methodology field today (verified
    ground truth), but if one ever appears it must equal the active version --
    stale per-unit metadata must not pass."""
    results_dir, signals_dir = _build_passing_local_fixture(tmp_path)
    uid = _uid("Krum")
    bad = make_result("Krum", provenance_overrides={"methodology_version": "v1.11"})
    write_result(results_dir / f"{uid}.json", bad)

    report = _run_local(results_dir, signals_dir)
    assert report.ok is False
    failing = [r for r in report.required_failures
               if r.name == "provenance.unit_methodology_version" and r.unit_id == uid]
    assert failing and "v1.11" in failing[0].detail


# ---------------------------------------------------------------------------
# MLflow check (S3-mode only, fake fetcher)
# ---------------------------------------------------------------------------


def _fake_experiment_and_runs(exp_id: str, units, meta, *, unit_status="done",
                              tag_overrides=None, param_contract=False):
    experiment_name = f"{exp_id}__calibration"
    runs = []
    for unit in units:
        if param_contract:
            # PR #15 new-image contract: experiment inputs are PARAMS (config
            # label -> 'defense'); only metadata stays in tags.
            tags = {
                "unit_id": unit.unit_id, "image_digest": meta["image_digest"],
                "unit_status": unit_status, "defense_token": unit.config.lower(),
            }
            params = {"defense": unit.config, "scenario": unit.scenario, "seed": str(unit.seed)}
        else:
            # legacy pre-enrichment contract: inputs carried as tags.
            tags = {
                "unit_id": unit.unit_id, "config": unit.config, "scenario": unit.scenario,
                "seed": str(unit.seed), "image_digest": meta["image_digest"], "unit_status": unit_status,
            }
            params = {}
        if tag_overrides:
            tags.update(tag_overrides.get(unit.unit_id, {}))
        data = {"tags": [{"key": k, "value": v} for k, v in tags.items()]}
        if params:
            data["params"] = [{"key": k, "value": v} for k, v in params.items()]
        runs.append({"data": data})

    def fetcher(url, data=None):
        if "experiments/get-by-name" in url:
            assert experiment_name in url
            return {"experiment": {"experiment_id": "1", "name": experiment_name}}
        if "runs/search" in url:
            return {"runs": runs}
        raise AssertionError(f"unexpected URL: {url}")

    return fetcher


def test_mlflow_check_passes_when_tags_match():
    units = [make_unit_ref("Krum"), make_unit_ref("TrustScore")]
    meta = {"image_digest": "sha256:abc123"}
    fetcher = _fake_experiment_and_runs(EXP_ID, units, meta)
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert all(r.passed for r in results), [(r.unit_id, r.detail) for r in results if not r.passed]


def test_mlflow_check_passes_with_param_contract():
    """Configuration inputs moved from tags to params for config/scenario/seed from tags to params (config label ->
    'defense'). The gate must read the inputs from params so new-image and
    backfilled runs pass."""
    units = [make_unit_ref("Krum"), make_unit_ref("TrustScore")]
    meta = {"image_digest": "sha256:abc123"}
    fetcher = _fake_experiment_and_runs(EXP_ID, units, meta, param_contract=True)
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert all(r.passed for r in results), [(r.unit_id, r.detail) for r in results if not r.passed]


def _fetcher_with_runs(runs):
    def fetcher(url, data=None):
        if "experiments/get-by-name" in url:
            return {"experiment": {"experiment_id": "1", "name": f"{EXP_ID}__calibration"}}
        if "runs/search" in url:
            return {"runs": runs}
        raise AssertionError(f"unexpected URL: {url}")
    return fetcher


def _run_with(unit, meta, unit_status):
    return {"data": {"tags": [{"key": k, "value": v} for k, v in {
        "unit_id": unit.unit_id, "config": unit.config, "scenario": unit.scenario,
        "seed": str(unit.seed), "image_digest": meta["image_digest"],
        "unit_status": unit_status,
    }.items()]}}


def test_mlflow_check_counts_only_done_run_among_duplicates():
    """A reclaim+retry (or an aborted launch) leaves >1 run sharing a unit_id;
    the gate counts only the single unit_status=done run and passes — requiring
    exactly one *run* would wrongly fail 'found 2'."""
    units = [make_unit_ref("Krum")]
    meta = {"image_digest": "sha256:abc123"}
    u = units[0]
    # zombie reconciled_failed + the completing/enriched done run
    fetcher = _fetcher_with_runs([_run_with(u, meta, "reconciled_failed"),
                                  _run_with(u, meta, "done")])
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert all(r.passed for r in results), [(r.unit_id, r.detail) for r in results if not r.passed]

    # two done runs -> genuinely ambiguous -> fail
    fetcher2 = _fetcher_with_runs([_run_with(u, meta, "done"), _run_with(u, meta, "done")])
    results2 = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher2)
    assert not all(r.passed for r in results2)
    assert any("found 2 done" in r.detail for r in results2)

    # zero done runs (only a zombie) -> fail
    fetcher3 = _fetcher_with_runs([_run_with(u, meta, "reconciled_failed")])
    results3 = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher3)
    assert not all(r.passed for r in results3)


def test_mlflow_gate_scopes_to_current_parent():
    """A multi-launch experiment holds children from aborted + current launches
    sharing a unit_id. The gate resolves the newest non-FAILED parent and counts
    only ITS children, so a stale aborted-launch done-child can't false-pass."""
    units = [make_unit_ref("Krum")]
    meta = {"image_digest": "sha256:abc123"}
    u = units[0]

    def _parent(rid, status, start):
        return {"info": {"run_id": rid, "status": status, "start_time": start},
                "data": {"tags": [{"key": "exp_id", "value": EXP_ID}]}}

    def _child(rid, parent, unit_status="done"):
        return {"info": {"run_id": rid, "status": "FINISHED", "start_time": 1},
                "data": {"tags": [{"key": k, "value": v} for k, v in {
                    "unit_id": u.unit_id, "config": u.config, "scenario": u.scenario,
                    "seed": str(u.seed), "image_digest": meta["image_digest"],
                    "unit_status": unit_status, "mlflow.parentRunId": parent,
                }.items()]}}

    runs = [
        _parent("p-aborted", "FAILED", 100),
        _parent("p-current", "FINISHED", 200),   # newest non-FAILED -> current launch
        _child("c-stale", "p-aborted"),           # stale aborted-launch done child
        _child("c-current", "p-current"),         # current launch's done child
    ]
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, _fetcher_with_runs(runs))
    assert all(r.passed for r in results), [(r.unit_id, r.detail) for r in results if not r.passed]


def test_mlflow_check_fails_on_unit_status_not_done():
    units = [make_unit_ref("Krum")]
    meta = {"image_digest": "sha256:abc123"}
    fetcher = _fake_experiment_and_runs(EXP_ID, units, meta, unit_status="runner_failed")
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert not all(r.passed for r in results)
    assert any("unit_status" in r.detail for r in results)


def test_mlflow_check_fails_when_run_missing():
    units = [make_unit_ref("Krum"), make_unit_ref("TrustScore")]
    meta = {"image_digest": "sha256:abc123"}
    fetcher = _fake_experiment_and_runs(EXP_ID, [units[0]], meta)  # TrustScore run absent
    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    trust_results = [r for r in results if r.unit_id == units[1].unit_id]
    assert trust_results and not trust_results[0].passed
    assert "found 0" in trust_results[0].detail


def test_mlflow_check_fails_when_experiment_absent():
    units = [make_unit_ref("Krum")]
    meta = {"image_digest": "sha256:abc123"}

    def fetcher(url, data=None):
        return {"experiment": None}

    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert len(results) == 1 and not results[0].passed
    assert "not found" in results[0].detail


def test_mlflow_transport_error_is_required_failure_per_unit(
):
    """A dead tracking server (URLError) must become a per-unit REQUIRED
    failure carrying the underlying error -- not an exception that aborts
    the CLI before the report renders."""
    import urllib.error

    units = [make_unit_ref("Krum"), make_unit_ref("TrustScore")]
    meta = {"image_digest": "sha256:abc123"}

    def fetcher(url, data=None):
        raise urllib.error.URLError("connection refused (localhost:5001 down)")

    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert len(results) == len(units)
    assert all(r.required and not r.passed for r in results)
    assert all("connection refused" in r.detail for r in results)
    assert {r.unit_id for r in results} == {u.unit_id for u in units}


def test_mlflow_non_json_response_is_required_failure():
    import json as _json

    units = [make_unit_ref("Krum")]
    meta = {"image_digest": "sha256:abc123"}

    def fetcher(url, data=None):
        _json.loads("<html>502 Bad Gateway</html>")  # raises JSONDecodeError

    results = gs.check_mlflow_runs(EXP_ID, units, meta, gs.DEFAULT_MLFLOW_URI, fetcher)
    assert len(results) == 1
    assert results[0].required and not results[0].passed
    assert "JSONDecodeError" in results[0].detail


def test_s3_gate_survives_mlflow_outage_and_renders(tmp_path):
    """run_gate_s3 with an unreachable MLflow must still return a complete
    report (exit nonzero via report.ok False), never raise."""
    import urllib.error

    store = gs.FakeGateStore()
    units = expand_matrix(["Krum"], [SCENARIO], [SEED], MODE, 2_000_000, ROUNDS)
    write_manifest(store, EXP_ID, units,
                   meta={"methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc"})
    unit = units[0]
    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("Krum"))
    signal_path = tmp_path / "s.jsonl"
    rows = []
    for rnd in signal_rounds():
        rows.extend(make_signal_rows_for_round("Krum", rnd))
    write_signal_log(signal_path, rows)
    persist_unit(store, EXP_ID, unit.unit_id, result_path, signal_path)

    def dead_fetcher(url, data=None):
        raise urllib.error.URLError("no route to host")

    report = acg.run_gate_s3(store, EXP_ID, mlflow_fetcher=dead_fetcher, check_mlflow=True)
    assert report.ok is False
    mlflow_failures = [r for r in report.required_failures if r.name.startswith("mlflow")]
    assert mlflow_failures and "no route to host" in mlflow_failures[0].detail
    # the rest of the audit still ran and is present in the report
    assert any(r.name == "s3_integrity" for r in report.results)


# ---------------------------------------------------------------------------
# Wall-clock sizing table
# ---------------------------------------------------------------------------


def test_wall_clock_projection_estimate_shape():
    rows = [{"unit_id": "a", "config": "Krum", "elapsed_seconds": 100.0},
            {"unit_id": "b", "config": "TrustScore", "elapsed_seconds": 200.0}]
    table, projection = gl.build_wall_clock_table(rows)
    assert table == rows
    assert projection["concurrency"] == 4  # 32 vCPUs / 8 vCPUs per job
    assert projection["mean_elapsed_seconds"] == 150.0
    assert projection["waves"] == 25  # ceil(100/4)
    assert projection["projected_wall_clock_seconds"] == pytest.approx(25 * 150.0)
    assert "ESTIMATE" in projection["note"]


# ---------------------------------------------------------------------------
# Baseline instrumentation reuse (check 2)
# ---------------------------------------------------------------------------


def test_baseline_instrumentation_reuses_audit_run_instrumentation(tmp_path):
    import audit_run_instrumentation as ari

    unit = make_unit_ref("Krum")
    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("Krum"))
    signal_path = tmp_path / "s.jsonl"
    write_signal_log(signal_path, make_signal_rows_for_round("Krum", 1))

    direct_gaps = ari.audit(unit.config, str(result_path), str(signal_path))
    via_gate = gl.check_baseline_instrumentation(unit, result_path, signal_path)
    assert via_gate.passed == (not direct_gaps)


def test_baseline_instrumentation_gap_detected_when_score_field_all_none(tmp_path):
    unit = make_unit_ref("TrustScore")
    result_path = tmp_path / "r.json"
    write_result(result_path, make_result("TrustScore"))
    signal_path = tmp_path / "s.jsonl"
    rows = make_signal_rows_for_round("TrustScore", 1)
    for r in rows:
        r["trust_score"] = None
    write_signal_log(signal_path, rows)

    result = gl.check_baseline_instrumentation(unit, result_path, signal_path)
    assert result.passed is False
    assert "UNCOMPUTABLE" in result.detail or "trust_score" in result.detail

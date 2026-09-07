"""End-to-end verdicts on synthetic ceilings/failures, diagnostic verdict
withholding, the one-execution output gate, secondaries, § 6 diagnostics
aggregation (null-vs-zero), and whole-artifact redaction."""
from __future__ import annotations

import json

import pytest

from scripts import h4_scoring_lib as lib
from scripts.analyze_h4_composition import (
    load_units,
    main,
    score_units,
)
from scripts.h4_scoring_lib import ScoringError
from tests.h4_factory import pin_fake_bundle_v2
from tests.h4_factory import (
    CONFIG_LABEL_BY_ARM,
    TEST_SEEDS,
    confirmed_acc,
    falsified_acc,
    install_seed_manifest,
    install_split_manifest,
    make_diagnostics,
    make_unit,
    write_corpus,
    write_unit,
)

ALL_ARMS = tuple(CONFIG_LABEL_BY_ARM)


@pytest.fixture(autouse=True)
def _pinned_bundle_v2(monkeypatch):
    """Erratum-B pinned state: swap the TBD_BUNDLE_V2 sentinel for the fake
    bundle_v2 sha so pipeline tests exercise post-calibration custody
    (sentinel refusal is covered by tests/test_h4_scorer_pin_v2.py)."""
    pin_fake_bundle_v2(monkeypatch)


@pytest.fixture
def env(tmp_path, monkeypatch):
    install_split_manifest(tmp_path, monkeypatch)
    install_seed_manifest(tmp_path, monkeypatch)
    return tmp_path


def _run(env, paths, extra=(), out_name="verdict.json"):
    out = env / "read" / out_name
    argv = (["--units"] + [str(p) for p in paths]
            + ["--out", str(out)] + list(extra))
    rc = main(argv)
    report = json.loads(out.read_text()) if out.exists() else None
    return rc, out, report


# ===========================================================================
# terminal verdicts
# ===========================================================================

def test_e2e_confirmed_on_ceiling_corpus(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    assert len(paths) == 540
    rc, out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "CONFIRMED"
    assert report["verdict_status"] == "CONFIRMED"
    assert report["diagnostic_mode"] is False
    assert report["census"]["arms_dropped"] == []
    for scenario in ("S1", "S2", "S3", "S4"):
        component = report["primary"]["components"][scenario]
        assert component["status"] == "PASS"
        assert component["n_pairs"] == 10
        assert component["median_reduction"] == pytest.approx(0.20)
        assert component["wilcoxon"]["p_two_sided"] == pytest.approx(2 / 1024)
    assert len(report["units"]) == 540
    assert out.with_suffix(".md").exists()


def test_e2e_falsified_by_median(env):
    paths = write_corpus(env / "units", acc_fn=falsified_acc)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "FALSIFIED"
    components = report["primary"]["components"]
    assert components["S2"]["status"] == "FAIL"
    assert components["S2"]["median_reduction"] == pytest.approx(-0.20)
    assert components["S2"]["gate"]["median_pass"] is False
    # the component table is always emitted, other scenarios included
    for scenario in ("S1", "S3", "S4"):
        assert components[scenario]["status"] == "PASS"


def test_e2e_falsified_by_p_value(env):
    """S3: five seed-pairs +0.30, five -0.001 -> median 0.1495 (passes the
    5pp bar) but exact two-sided p = 224/1024 > 0.05 -> FAIL on p alone."""
    def acc(arm, scenario, seed):
        if scenario == "S3" and arm in ("krum", "h2p_fp_krum"):
            first_half = (seed - TEST_SEEDS[0]) < 5
            if arm == "krum":
                return 0.60 if first_half else 0.90
            return 0.90 if first_half else 0.899
        return confirmed_acc(arm, scenario, seed)

    paths = write_corpus(env / "units", acc_fn=acc)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "FALSIFIED"
    component = report["primary"]["components"]["S3"]
    assert component["status"] == "FAIL"
    assert component["median_reduction"] == pytest.approx(0.1495)
    assert component["gate"]["median_pass"] is True
    assert component["gate"]["p_pass"] is False
    assert component["wilcoxon"]["p_two_sided"] == pytest.approx(224 / 1024)


# ===========================================================================
# diagnostic mode / verdict withholding
# ===========================================================================

def test_partial_census_without_flag_refuses(env, capsys):
    paths = write_corpus(env / "units", arms=("h2p_fp_krum", "krum"),
                         scenarios=("C0", "S1"), acc_fn=confirmed_acc)
    rc, out, _report = _run(env, paths)
    assert rc == 2
    assert out is not None and not out.exists()
    assert "REFUSED" in capsys.readouterr().err


def test_allow_partial_census_withholds_verdict(env):
    paths = write_corpus(env / "units", arms=("h2p_fp_krum", "krum"),
                         scenarios=("C0", "S1"), acc_fn=confirmed_acc)
    rc, _out, report = _run(env, paths, extra=["--allow-partial-census"])
    assert rc == 0
    assert report["verdict"] is None
    assert report["verdict_status"] == "DIAGNOSTIC — VERDICT WITHHELD"
    assert report["diagnostic_mode"] is True
    assert "--allow-partial-census" in report["verdict_withheld_reason"]
    # component conjunction still reported for diagnosis
    assert report["component_conjunction"] in (
        "CONFIRMED", "FALSIFIED", "INCONCLUSIVE")
    assert report["census"]["census_gate_enforced"] is False


def test_diagnostic_components_inconclusive_when_pairs_missing(env):
    # comparator arm never supplied -> no pair can be formed
    units = load_units(write_corpus(
        env / "units", arms=("h2p_fp_krum",), acc_fn=confirmed_acc))
    report = score_units(units, require_census=False, diagnostic=True)
    for component in report["primary"]["components"].values():
        assert component["computed"] is False
        assert component["status"] == "INCONCLUSIVE"
    assert report["component_conjunction"] == "INCONCLUSIVE"
    assert report["verdict"] is None


# ===========================================================================
# one-execution discipline
# ===========================================================================

def test_one_execution_output_gate(env, capsys):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    rc1, out, report1 = _run(env, paths)
    assert rc1 == 0 and report1["verdict"] == "CONFIRMED"
    sealed_bytes = out.read_bytes()
    sealed_memo = out.with_suffix(".md").read_text()

    rc2, _out, _r = _run(env, paths)
    err = capsys.readouterr().err
    assert rc2 == 2
    assert "already exists" in err
    assert "ONE" in err
    # the terminal artifact is untouched
    assert out.read_bytes() == sealed_bytes

    # forced re-read: ROUTED to the.diagnostic sibling; the sealed
    # artifact (and its memo) are byte-identical afterwards
    rc3, _out, report3 = _run(env, paths, extra=["--force-diagnostic"])
    assert rc3 == 0
    assert out.read_bytes() == sealed_bytes            # inviolable
    assert out.with_suffix(".md").read_text() == sealed_memo
    assert report3["verdict"] == "CONFIRMED"           # re-read of `out`
    sibling = out.with_name("verdict.diagnostic.json")
    diagnostic = json.loads(sibling.read_text())
    assert diagnostic["verdict"] is None
    assert diagnostic["verdict_status"] == "DIAGNOSTIC — VERDICT WITHHELD"
    assert diagnostic["diagnostic_mode"] is True
    assert sibling.with_suffix(".md").exists()


def test_diagnostic_reread_refuses_when_sibling_also_protected(env, capsys):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    rc1, out, _report = _run(env, paths)
    assert rc1 == 0
    sibling = out.with_name("verdict.diagnostic.json")
    sibling.write_bytes(out.read_bytes())   # adjudicating artifact squats
    rc, _out, _r = _run(env, paths, extra=["--force-diagnostic"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "inviolable" in err
    # neither artifact was touched
    assert json.loads(out.read_text())["verdict"] == "CONFIRMED"
    assert json.loads(sibling.read_text())["verdict"] == "CONFIRMED"


def test_forced_reread_may_overwrite_a_previous_diagnostic(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, out, _report = _run(env, paths)
    rc2, _o, _r = _run(env, paths, extra=["--force-diagnostic"])
    rc3, _o, _r = _run(env, paths, extra=["--force-diagnostic"])
    assert rc2 == 0 and rc3 == 0
    sibling = out.with_name("verdict.diagnostic.json")
    assert json.loads(sibling.read_text())["verdict"] is None
    assert json.loads(out.read_text())["verdict"] == "CONFIRMED"


def test_unparseable_existing_out_is_protected_not_overwritten(env):
    # a corrupted file MIGHT be the terminal artifact — fail closed
    out = env / "read" / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{corrupted, not json")
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    rc = main(["--units"] + [str(p) for p in paths]
              + ["--out", str(out), "--force-diagnostic"])
    assert rc == 0
    assert out.read_text() == "{corrupted, not json"   # untouched
    sibling = out.with_name("verdict.diagnostic.json")
    assert json.loads(sibling.read_text())["verdict"] is None


# ===========================================================================
# secondaries
# ===========================================================================

def test_secondaries_reported_and_labeled(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, _out, report = _run(env, paths)
    secondaries = report["secondaries"]
    assert "NON-GATING" in secondaries["note"]
    # constant final-5 values: the final-round variant equals the primary
    s1 = secondaries["single_final_round"]["components"]["S1"]
    assert s1["computed"] and s1["median_reduction"] == pytest.approx(0.20)
    # factory f1 = accuracy - 0.05, so f1 reductions match too
    f1 = secondaries["f1_final5"]["components"]["S1"]
    assert f1["computed"] and f1["median_reduction"] == pytest.approx(0.20)
    # S0-referenced: deg(krum)=0.90-0.70, deg(h2p_fp_krum)=0.90-0.90
    s0 = secondaries["s0_referenced"]["components"]["S1"]
    assert s0["reference_scenario"] == "S0"
    assert s0["median_reduction"] == pytest.approx(0.20)
    # attributive contrasts all present and computed
    contrasts = secondaries["attributive_contrasts"]
    assert set(contrasts) == {"1v5", "6v9", "3v1", "6v7", "4v1"}
    # 1v5: deg(h2p_krum)=0.95-0.80=0.15, deg(h2p_fp_krum)=0.05 -> +0.10
    c15 = contrasts["1v5"]
    assert c15["computed"] is True
    assert c15["components"]["S1"]["median_reduction"] == pytest.approx(0.10)
    # no gate/pass flags on secondary components (they cannot gate)
    assert "gate" not in s1 and "status" not in s1


def test_f1_secondary_null_with_reason_when_f1_absent(env):
    def unit_fn(arm, scenario, seed, **kw):
        return make_unit(arm, scenario, seed, with_f1=False, **kw)

    paths = write_corpus(env / "units", acc_fn=confirmed_acc,
                         unit_fn=unit_fn)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "CONFIRMED"   # primary unaffected
    f1_s1 = report["secondaries"]["f1_final5"]["components"]["S1"]
    assert f1_s1["computed"] is False
    assert "null endpoint" in f1_s1["reason"]
    assert f1_s1["n_pairs_endpoint_null"] == 10


def test_dropped_arm_contrast_null_with_reason(env):
    arms = [a for a in ALL_ARMS if a not in ("fedavg", "krum_tge_fp")]
    paths = write_corpus(env / "units", arms=arms, acc_fn=confirmed_acc)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "CONFIRMED"
    assert report["census"]["arms_dropped"] == ["fedavg", "krum_tge_fp"]
    c41 = report["secondaries"]["attributive_contrasts"]["4v1"]
    assert c41["computed"] is False
    assert "drop order" in c41["reason"]
    # contrasts whose arms survive still compute
    assert report["secondaries"]["attributive_contrasts"]["1v5"]["computed"]


# ===========================================================================
# § 6 diagnostics aggregation
# ===========================================================================

def test_diagnostics_null_vs_zero_preserved(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, _out, report = _run(env, paths)
    per = report["diagnostics"]["per_arm_scenario"]
    # krum has no detector and no fp layer -> nulls, never zero
    krum_s1 = per["krum"]["S1"]["per_layer_removal_totals"]
    assert krum_s1["detector_dropped_honest"] is None
    assert krum_s1["detector_dropped_malicious"] is None
    assert krum_s1["fp_hard_dropped"] is None
    assert krum_s1["aggregator_rejected"] == 40   # 1/round x 4 rounds x 10
    # fedavg has NO layers at all
    fedavg_s1 = per["fedavg"]["S1"]["per_layer_removal_totals"]
    assert all(v is None for v in fedavg_s1.values())
    # the full composition has all three layers, zeros stay countable ints
    h2p_s1 = per["h2p_fp_krum"]["S1"]["per_layer_removal_totals"]
    assert h2p_s1["detector_dropped_honest"] == 40
    assert h2p_s1["detector_dropped_malicious"] == 80
    assert h2p_s1["fp_hard_dropped"] == 40
    assert per["h2p_fp_krum"]["S1"]["layer_presence_inconsistent"] == []


def test_diagnostics_blackout_rounds_aggregated(env):
    def unit_fn(arm, scenario, seed, **kw):
        diag = "auto"
        if arm == "h2p_fp":
            diag = make_diagnostics(arm, empty_rounds=(2, 3))
        return make_unit(arm, scenario, seed, diagnostics=diag, **kw)

    paths = write_corpus(env / "units", acc_fn=confirmed_acc,
                         unit_fn=unit_fn)
    _rc, _out, report = _run(env, paths)
    bucket = report["diagnostics"]["per_arm_scenario"]["h2p_fp"]["S2"]
    assert bucket["empty_aggregate_rounds_total"] == 20      # 2 x 10 units
    assert bucket["units_with_empty_rounds"] == 10
    frac = bucket["kept_set_malicious_fraction"]
    assert frac["rounds_null_blackout"] == 20   # 0/0 rounds stay null
    assert frac["rounds_total"] == 40
    assert frac["mean"] == pytest.approx(0.2)


def test_detector_recall_fpr_proxies(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, _out, report = _run(env, paths)
    proxies = report["diagnostics"]["detector_proxies"]
    assert "PROXIES" in proxies["caveat"]
    s1 = proxies["per_arm"]["h2p_fp_krum"]["S1"]
    # factory: 2 malicious flagged + 2 kept malicious per round
    assert s1["flagged_malicious"] == 80
    assert s1["kept_malicious"] == 80
    assert s1["recall_proxy"] == pytest.approx(0.5)
    assert s1["fpr_proxy"] == pytest.approx(40 / 360)
    # non-detector arms carry no proxies; krum_tge_fp (TGE) does
    assert "krum" not in proxies["per_arm"]
    assert "fedavg" not in proxies["per_arm"]
    assert "krum_tge_fp" in proxies["per_arm"]


def test_reference_anomaly_flagged_with_ordinal_not_seed(env):
    dipping = [{"round": r, "accuracy": (0.95 if r <= 5 else 0.70),
                "f1": 0.9, "loss": 0.1} for r in range(1, 11)]

    def unit_fn(arm, scenario, seed, **kw):
        if (arm, scenario, seed) == ("trustscore", "C0", TEST_SEEDS[0]):
            kw = dict(kw, trajectory=dipping)
        return make_unit(arm, scenario, seed, **kw)

    paths = write_corpus(env / "units", acc_fn=confirmed_acc,
                         unit_fn=unit_fn)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    anomaly = report["diagnostics"]["reference_anomaly"]
    assert anomaly["n_cells_checked"] == 180   # 9 arms x {C0, S0} x 10
    assert len(anomaly["flagged_cells"]) == 1
    cell = anomaly["flagged_cells"][0]
    assert cell["arm"] == "trustscore"
    assert cell["scenario"] == "C0"
    assert cell["seed_ordinal"] == 0
    assert "seed" not in {k for k in cell} - {"seed_ordinal"}
    assert cell["gap"] == pytest.approx(0.25)


def test_missing_diagnostics_block_reported_not_refused(env):
    def unit_fn(arm, scenario, seed, **kw):
        return make_unit(arm, scenario, seed,
                         diagnostics=None if arm == "fedavg" else "auto",
                         **kw)

    paths = write_corpus(env / "units", acc_fn=confirmed_acc,
                         unit_fn=unit_fn)
    rc, _out, report = _run(env, paths)
    assert rc == 0
    assert report["verdict"] == "CONFIRMED"   # diagnostics never gate
    missing = report["diagnostics"]["units_missing_diagnostics"]
    assert missing["count"] == 60
    assert all("sha256:" in ref for ref in missing["refs"])


# ===========================================================================
# whole-artifact redaction
# ===========================================================================

def test_no_seed_run_uid_or_path_leaks_anywhere(env, capsys):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    rc, out, report = _run(env, paths)
    assert rc == 0
    stdout = capsys.readouterr().out
    json_text = out.read_text()
    memo_text = out.with_suffix(".md").read_text()
    for surface in (stdout, json_text, memo_text):
        for seed in TEST_SEEDS:
            assert str(seed) not in surface
        assert "uid-" not in surface        # raw run_uids (factory prefix)
        assert "seed9000" not in surface    # seed-bearing filenames
    # units are still identifiable via redactions + ordinals
    row = report["units"][0]
    assert row["ref"].startswith("unit[arm=")
    assert "sha256:" in row["ref"]
    assert row["seed_ordinal"] in set(range(10))
    # universal run identity: EVERY unit row carries a redacted run_uid
    assert all(r["run_uid_redacted"].startswith("sha256:")
               for r in report["units"])


def test_memo_contents(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, out, _report = _run(env, paths)
    memo = out.with_suffix(".md").read_text()
    assert "**Verdict: CONFIRMED**" in memo
    assert "| S1 |" in memo and "| S4 |" in memo
    assert "NON-GATING" in memo
    assert lib.SERVING_BUNDLE_SHA256 in memo


def test_custody_pins_in_report(env):
    paths = write_corpus(env / "units", acc_fn=confirmed_acc)
    _rc, _out, report = _run(env, paths)
    custody = report["custody"]
    # erratum B: the report echoes the ACTIVE (v2) pin via the sentinel-guarded
    # accessor; here that is the fixture's fake bundle_v2 sha.
    assert custody["serving_bundle_sha256"] == lib.SERVING_BUNDLE_SHA256
    assert custody["serving_bundle_source"] == "data/h4_serving/manifest_v2.json"
    assert custody["eval_split"] == "sealed_test"
    assert custody["fp_registry_policy"] == "flag_gated"
    assert custody["fp_cohort"] == "validation"
    assert report["verdict_rule"].startswith("FROZEN")


def test_refusal_path_exit_code_via_main(env, capsys):
    unit = make_unit("krum", "S1", TEST_SEEDS[0])
    unit["provenance"]["eval_split"] = "legacy"
    path = write_unit(env / "units", unit)
    out = env / "read" / "v.json"
    rc = main(["--units", str(path), "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSED:")
    assert not out.exists()

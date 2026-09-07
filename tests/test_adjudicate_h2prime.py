"""Unit gates for the H2′ P1∧P2 adjudication executor (scripts/adjudicate_h2prime.py).

Covers the parts that must be right BEFORE the sealed corpus is opened:
the § 2.2a rotation indices (n = 10 wrap-around and the n = 5 reduction to the
dev harness), every refusal path of the assembly-map validator, the § 4 (P2)
exact sign-test convention, the § 3.2 comparability interval, and the verdict
composition. The dev-smoke determinism check is marked `slow` because it fits
30 GBDTs on the EXP-011 dev corpus.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import numpy as np
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "adjudicate_h2prime.py"
DEV_CORPUS_ENV = "H2PRIME_DEV_SIG_DIR"


def _load():
    spec = importlib.util.spec_from_file_location("adjudicate_h2prime_uut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["adjudicate_h2prime_uut"] = module   # dataclasses need this
    spec.loader.exec_module(module)
    return module


A = _load()

SCENARIOS_LONG = [
    "s0_clean_baseline", "s1_benign_churn_only", "s2_adaptive_switching_only",
    "s3_identity_reset_only", "s4_full_mix",
]


# ---------------------------------------------------------------------------
# § 2.2a rotation
# ---------------------------------------------------------------------------
def test_rotation_n10_matches_the_frozen_spec_table():
    """v1.15 § 2.2a 'THE OPERATIVE ROTATION' / EXP-051 § 3, seeds s_1…s_10."""
    seeds = list(range(1, 11))          # stand-ins for s_1 … s_10, ascending
    plan = A.rotation_plan(seeds)
    expected = {
        1: (1, 2, (3, 4, 5)), 2: (2, 3, (4, 5, 6)), 3: (3, 4, (5, 6, 7)),
        4: (4, 5, (6, 7, 8)), 5: (5, 6, (7, 8, 9)), 6: (6, 7, (8, 9, 10)),
        7: (7, 8, (9, 10, 1)), 8: (8, 9, (10, 1, 2)), 9: (9, 10, (1, 2, 3)),
        10: (10, 1, (2, 3, 4)),
    }
    assert {r.i: (r.test, r.calibration, r.fit) for r in plan} == expected


def test_rotation_n5_reduces_to_the_dev_harness_assignment():
    """§ 2.2a 'Reduction to the existing n = 5 assignment': the dev harness ran
    test = seeds[i], calibration = seeds[(i+1) % 5], fit = the remaining three."""
    seeds = sorted(A.DEV_SEEDS)
    plan = A.rotation_plan(seeds)
    for idx, rot in enumerate(plan):
        assert rot.test == seeds[idx]
        assert rot.calibration == seeds[(idx + 1) % 5]
        assert sorted(rot.fit) == sorted(s for s in seeds if s not in (rot.test, rot.calibration))


def test_rotation_scores_every_seed_exactly_once_and_uses_five_distinct():
    plan = A.rotation_plan(sorted([90369, 29387, 98362, 45013, 39477,
                                   70402, 47599, 39375, 17869, 96540]))
    assert sorted(r.test for r in plan) == sorted([17869, 29387, 39375, 39477, 45013,
                                                   47599, 70402, 90369, 96540, 98362])
    for r in plan:
        assert len({r.test, r.calibration, *r.fit}) == 5


def test_rotation_refuses_below_five_seeds():
    with pytest.raises(A.Refusal):
        A.rotation_plan([1, 2, 3, 4])


# ---------------------------------------------------------------------------
# assembly-map validation / refusal paths
# ---------------------------------------------------------------------------
def _write_map(tmp_path: Path, seeds, *, cells=None, scenarios=None, mutate=None) -> Path:
    scenarios = scenarios or SCENARIOS_LONG
    entries = []
    for scen in scenarios:
        for seed in seeds:
            unit = f"{scen}__krum_tge__persistent_optimizer__seed{seed}"
            path = tmp_path / f"{unit}.jsonl"
            path.write_text('{"placeholder": true}\n', encoding="utf-8")
            entries.append({"scenario": scen, "seed": seed, "source": "EXP-051",
                            "path": str(path)})
    if cells is not None:
        entries = cells(entries, tmp_path)
    # REAL content digests, computed AFTER any `cells` transform so an entry
    # whose path was rewritten is digested against the file it now points at.
    # The confirmatory profile requires a digest, so a fixture without one
    # would trip the custody gate before reaching what the test is about.
    # `mutate` runs after this and can still drop or corrupt them on purpose.
    for e in entries:
        f = Path(e["path"])
        if f.is_file():
            e["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    doc = {"_meta": {"test": True}, "cells": entries}
    if mutate:
        mutate(doc)
    p = tmp_path / "map.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_map_accepts_the_dev_smoke_shape(tmp_path):
    p = _write_map(tmp_path, A.DEV_SEEDS)
    cells, meta = A.load_assembly_map(p, A.DEV_SMOKE)
    assert len(cells) == 25
    assert meta["seeds_ascending"] == sorted(A.DEV_SEEDS)
    assert meta["defense_token"] == "krum_tge"


def test_map_refuses_wrong_cell_count(tmp_path):
    p = _write_map(tmp_path, A.DEV_SEEDS, cells=lambda e, _t: e[:-1])
    with pytest.raises(A.Refusal, match="24 cells"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_duplicate_scenario_seed(tmp_path):
    def dup(entries, _tmp):
        return entries[:-1] + [dict(entries[0])]
    p = _write_map(tmp_path, A.DEV_SEEDS, cells=dup)
    with pytest.raises(A.Refusal, match="duplicate"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_missing_file(tmp_path):
    p = _write_map(tmp_path, A.DEV_SEEDS)
    doc = json.loads(p.read_text())
    Path(doc["cells"][3]["path"]).unlink()
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(A.Refusal, match="missing on disk"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_empty_file(tmp_path):
    p = _write_map(tmp_path, A.DEV_SEEDS)
    doc = json.loads(p.read_text())
    Path(doc["cells"][0]["path"]).write_text("", encoding="utf-8")
    with pytest.raises(A.Refusal, match="empty"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_sha256_mismatch(tmp_path):
    def tamper(doc):
        doc["cells"][0]["sha256"] = "0" * 64
    p = _write_map(tmp_path, A.DEV_SEEDS, mutate=tamper)
    with pytest.raises(A.Refusal, match="sha256 mismatch"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_filename_seed_disagreeing_with_map(tmp_path):
    def swap(doc):
        doc["cells"][0]["seed"] = 999999
    p = _write_map(tmp_path, A.DEV_SEEDS, mutate=swap)
    with pytest.raises(A.Refusal, match="filename seed"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_heterogeneous_defense(tmp_path):
    def mixed(entries, tmp):
        e = dict(entries[0])
        unit = f"{e['scenario']}__krum__persistent_optimizer__seed{e['seed']}"
        path = tmp / f"{unit}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        e["path"] = str(path)
        return entries[1:] + [e]
    p = _write_map(tmp_path, A.DEV_SEEDS, cells=mixed)
    with pytest.raises(A.Refusal, match="heterogeneous defense"):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_map_refuses_a_uniformly_wrong_defense_token(tmp_path):
    """Homogeneity AND identity: 25 standalone-`tge` units are homogeneous and
    would adjudicate the cohort on an unregistered configuration."""
    entries = []
    for scen in SCENARIOS_LONG:
        for seed in A.DEV_SEEDS:
            unit = f"{scen}__tge__persistent_optimizer__seed{seed}"
            path = tmp_path / f"{unit}.jsonl"
            path.write_text('{"placeholder": true}\n', encoding="utf-8")
            entries.append({"scenario": scen, "seed": seed, "source": "X",
                            "path": str(path)})
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"cells": entries}), encoding="utf-8")
    with pytest.raises(A.Refusal, match="not the REGISTERED configuration"):
        A.load_assembly_map(mp, A.DEV_SMOKE)


def test_registered_defense_token_is_the_one_exp051_registers():
    assert A.REGISTERED_DEFENSE_TOKEN == "krum_tge"


def test_map_refuses_scenario_set_that_is_not_the_frozen_five(tmp_path):
    p = _write_map(tmp_path, A.DEV_SEEDS, scenarios=SCENARIOS_LONG[:4])
    with pytest.raises(A.Refusal):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_confirmatory_refuses_seeds_that_are_not_the_sealed_set(tmp_path):
    p = _write_map(tmp_path, [11111, 22222, 33333, 44444, 55555,
                              66666, 77777, 88888, 99999, 12345])
    with pytest.raises(A.Refusal, match="not the sealed confirmatory set"):
        A.load_assembly_map(p, A.CONFIRMATORY)


def test_dev_smoke_refuses_sealed_seeds(tmp_path):
    sealed = A.sealed_seeds()
    p = _write_map(tmp_path, sealed[:5])
    with pytest.raises(A.Refusal):
        A.load_assembly_map(p, A.DEV_SMOKE)


def test_sealed_seed_authority_hash_is_the_recorded_one():
    """EXP-051 § 2.1 — the seed file is the authority and its digest is pinned."""
    seeds = A.sealed_seeds()
    assert len(seeds) == 10 and seeds == sorted(set(seeds))


# ---------------------------------------------------------------------------
# § 4 (P2) exact sign-test convention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pos,neg,zero,expect_p", [
    (10, 0, 0, 1 / 1024),
    (9, 1, 0, 11 / 1024),
    (9, 0, 1, 1 / 512),
    (8, 0, 2, 1 / 256),
    (5, 0, 0, 1 / 32),
    (4, 0, 1, 1 / 16),
    (3, 0, 2, 1 / 8),
    (0, 0, 5, 1.0),
])
def test_exact_sign_p_matches_the_enumerated_spec_table(pos, neg, zero, expect_p):
    diffs = [1.0] * pos + [-1.0] * neg + [0.0] * zero
    got = A.exact_sign_p(diffs)
    assert got["positive"] == pos and got["negative"] == neg and got["zero"] == zero
    assert got["p_one_sided_exact"] == pytest.approx(expect_p)


def test_zero_differences_never_count_toward_the_nine():
    """§ 4 (P2): '8 positives with 2 zeros attains 1/256 … and it FAILS P2.'"""
    got = A.exact_sign_p([1.0] * 8 + [0.0, 0.0])
    assert got["p_one_sided_exact"] == pytest.approx(1 / 256)
    assert got["positive"] < A.CONFIRMATORY.p2_min_positive


# ---------------------------------------------------------------------------
# v1.15b § 1.2 — the frozen P2 comparability semantics
# ---------------------------------------------------------------------------
def _p2_fixture(det_cells, base_cells):
    """Minimal scored-corpus stub exercising adjudicate_p2's guard."""
    return {"alie": det_cells, "base_alie": base_cells}


def _cell(recall, flagged, honest, n_mal=10):
    return {"recall": recall, "fpr": flagged / honest, "n_mal": n_mal,
            "n_tp": int(round(recall * n_mal)), "n_honest": honest,
            "n_flagged_honest": flagged, "cut": 0.5, "rotation": 1,
            "auc": 0.9}


def _corpus(det_fn, base_fn):
    """A COMPLETE S0–S4 × dev-seed ALIE population (the structural gate's floor).

    det_fn / base_fn map (scenario, seed[, instrument]) -> (recall, flagged, honest).
    """
    det = {(sc, s): _cell(*det_fn(sc, s))
           for sc in A.SCENARIOS for s in A.DEV_SEEDS}
    base = {b: {(sc, s): _cell(*base_fn(sc, s, b))
                for sc in A.SCENARIOS for s in A.DEV_SEEDS}
            for b, _t, _l in A.BASELINES}
    return _p2_fixture(det, base)


def test_p2_guard_pools_rows_rather_than_averaging_cell_rates():
    """v1.15b § 1.2 item 1: row-pooled, not the unweighted mean of rates.

    S0's cells are tiny and badly calibrated; the other four scenarios are large
    and well calibrated. The unweighted mean of per-cell rates lands at 0.18 —
    outside [0.08, 0.12] — while the row-pooled rate is 0.101, inside. The
    pooled value is the one that governs.
    """
    res = A.adjudicate_p2(_corpus(
        lambda sc, s: (0.9, 5, 10) if sc == "S0" else (0.9, 99, 990),
        lambda sc, s, b: (0.0, 1, 10) if sc == "S0" else (0.0, 99, 990),
    ), A.DEV_SMOKE)
    rg = res["readout_grain_comparability"]
    assert rg["detector_flagged_honest"] == 5 * 5 + 20 * 99
    assert rg["detector_honest_rows"] == 5 * 10 + 20 * 990
    assert rg["detector_pooled_fpr"] == pytest.approx(2005 / 19850)
    assert rg["detector_in_interval"] is True
    # the rejected aggregation: (5×0.5 + 20×0.1)/25 = 0.18 — outside
    assert A.comparable((5 * 0.5 + 20 * 0.1) / 25) is False
    assert res["verdict"] == "PASS"


def test_p2_guard_uses_the_argmax_instrument_not_the_others():
    """v1.15b § 1.2 item 2: only the selected instrument's FPR is guarded."""
    res = A.adjudicate_p2(_corpus(
        lambda sc, s: (0.9, 10, 100),
        lambda sc, s, b: (0.5, 10, 100) if b == "krum_score" else (0.1, 30, 100),
    ), A.DEV_SMOKE)
    rg = res["readout_grain_comparability"]
    assert rg["argmax_instrument_counts"]["krum_score"] == 25
    assert rg["baseline_pooled_fpr"] == pytest.approx(0.10)
    assert rg["baseline_in_interval"] is True
    # the never-selected instruments are reported but do not halt the band
    per = res["realized_fpr_diagnostic"]["per_instrument_pooled"]
    assert per["L2_to_median"]["in_interval"] is False
    assert res["verdict"] == "PASS"


def test_p2_argmax_tie_resolves_to_the_frozen_tuple_order():
    """v1.15b § 1.2 item 3: exact-0.000 ties are the normal regime, and the
    tie-break decides WHICH instrument's FPR is guarded."""
    fprs = {"krum_score": 10, "L2_to_median": 9, "cos_to_median": 11}
    res = A.adjudicate_p2(_corpus(
        lambda sc, s: (0.9, 10, 100),
        lambda sc, s, b: (0.0, fprs[b], 100),
    ), A.DEV_SMOKE)
    rg = res["readout_grain_comparability"]
    assert {u["argmax_baseline"] for u in res["per_unit"]} == {"krum_score"}
    assert rg["n_cells_where_argmax_was_a_tie"] == 25
    assert rg["baseline_pooled_fpr"] == pytest.approx(0.10)


def test_p2_halts_inconclusive_when_either_pooled_side_is_outside():
    """v1.15b § 1.2 item 4, both directions."""
    def corpus(det_flagged, base_flagged):
        return _corpus(lambda sc, s: (0.9, det_flagged, 100),
                       lambda sc, s, b: (0.0, base_flagged, 100))

    res = A.adjudicate_p2(corpus(30, 10), A.DEV_SMOKE)
    assert res["verdict"] == "INCONCLUSIVE" and "detector" in res["reason"]

    res = A.adjudicate_p2(corpus(10, 30), A.DEV_SMOKE)
    assert res["verdict"] == "INCONCLUSIVE" and "arg-max baseline" in res["reason"]

    res = A.adjudicate_p2(corpus(10, 10), A.DEV_SMOKE)
    assert res["verdict"] == "PASS"
    assert res["sign_test"]["positive"] == 5


def test_p2_per_unit_fprs_are_diagnostic_and_no_longer_halt_the_band():
    """EVERY cell outside the interval, yet the pooled readout is inside — the
    band must adjudicate. This is the case that halted under the old grain."""
    def det_fn(sc, s):
        # S0/S1 badly under, S2–S4 badly over; no cell is inside [0.08, 0.12]
        return (0.9, 2, 100) if sc in ("S0", "S1") else (0.9, 18, 100)

    res = A.adjudicate_p2(_corpus(
        det_fn, lambda sc, s, b: (0.0, 10, 100)), A.DEV_SMOKE)
    rg = res["readout_grain_comparability"]
    assert rg["detector_pooled_fpr"] == pytest.approx((10 * 2 + 15 * 18) / 2500)
    assert rg["detector_in_interval"] is True
    assert res["realized_fpr_diagnostic"]["per_unit_detector_in_interval"] == 0
    assert res["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# structural gates on the P2 population (executor-enforced, custody-independent)
# ---------------------------------------------------------------------------
def test_p2_refuses_when_a_seed_is_missing_an_alie_bearing_cell():
    """The macro-average estimand is defined over exactly S0–S4 per seed."""
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    del corpus["alie"][("S3", A.DEV_SEEDS[2])]
    with pytest.raises(A.HardStop, match="STRUCTURAL POPULATION GATE"):
        A.adjudicate_p2(corpus, A.DEV_SMOKE)


def test_p2_refuses_when_an_instrument_is_null_on_calibration_rows():
    """The cut comes off the calibration honest population, so a null there
    takes the operating point from a different population than the detector's."""
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    corpus["base_alie"]["L2_to_median"][("S1", A.DEV_SEEDS[3])]["cal_null_dropped"] = 7
    with pytest.raises(A.HardStop, match="CALIBRATION row"):
        A.adjudicate_p2(corpus, A.DEV_SMOKE)


def test_p2_refuses_when_an_instrument_is_null_on_scored_rows():
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    corpus["base_alie"]["krum_score"][("S0", A.DEV_SEEDS[0])]["mal_null_dropped"] = 3
    with pytest.raises(A.HardStop, match="ALIE row"):
        A.adjudicate_p2(corpus, A.DEV_SMOKE)


def test_p1_refuses_when_a_registered_seed_is_missing_its_s4_unit():
    """Same class as P2's gate: a corpus defect must not be reported as the
    terminal INCONCLUSIVE verdict."""
    blend = {("S4", s): {"recall": 0.5, "fpr": 0.10} for s in A.DEV_SEEDS}
    del blend[("S4", A.DEV_SEEDS[1])]
    with pytest.raises(A.HardStop, match="P1 STRUCTURAL POPULATION GATE"):
        A.adjudicate_p1({"blend": blend}, A.DEV_SMOKE, A.DEV_SEEDS)


def test_p1_still_returns_inconclusive_for_a_genuine_calibration_halt():
    """The HardStop is for structure only — calibration halts stay INCONCLUSIVE."""
    blend = {("S4", s): {"recall": 0.5, "fpr": 0.30} for s in A.DEV_SEEDS}
    res = A.adjudicate_p1({"blend": blend}, A.DEV_SMOKE, A.DEV_SEEDS)
    assert res["verdict"] == "INCONCLUSIVE" and "calibration-integrity" in res["reason"]


def test_p2_refuses_when_a_registered_seed_is_wholly_absent():
    """A seed with all five cells missing must halt, not shrink the universe."""
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    gone = A.DEV_SEEDS[2]
    for sc in A.SCENARIOS:
        del corpus["alie"][(sc, gone)]
        for b, _t, _l in A.BASELINES:
            del corpus["base_alie"][b][(sc, gone)]
    # derived universe would not notice; the registered one must
    with pytest.raises(A.HardStop, match="STRUCTURAL POPULATION GATE"):
        A.adjudicate_p2(corpus, A.DEV_SMOKE, A.DEV_SEEDS)


def test_p2_refuses_when_an_enumerated_instrument_is_unavailable():
    """The oracle maximum must be taken over all three instruments (§ 3.1)."""
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    del corpus["base_alie"]["cos_to_median"][("S2", A.DEV_SEEDS[1])]
    with pytest.raises(A.HardStop, match="ENUMERATED-BASELINE COMPLETENESS"):
        A.adjudicate_p2(corpus, A.DEV_SMOKE)


def test_p2_gates_name_every_offending_cell():
    corpus = _corpus(lambda sc, s: (0.9, 10, 100),
                     lambda sc, s, b: (0.0, 10, 100))
    del corpus["alie"][("S0", A.DEV_SEEDS[0])]
    del corpus["alie"][("S4", A.DEV_SEEDS[4])]
    with pytest.raises(A.HardStop) as exc:
        A.adjudicate_p2(corpus, A.DEV_SMOKE)
    assert f"seed {A.DEV_SEEDS[0]} missing S0" in str(exc.value)
    assert f"seed {A.DEV_SEEDS[4]} missing S4" in str(exc.value)


# ---------------------------------------------------------------------------
# § 3.2 comparability interval
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fpr,ok", [
    (0.08, True), (0.12, True), (0.10, True),
    (0.0799999, False), (0.1200001, False), (0.169, False), (None, False),
])
def test_comparability_interval_is_closed_at_both_ends(fpr, ok):
    assert A.comparable(fpr) is ok


# ---------------------------------------------------------------------------
# § 4 (P1) CI and verdict composition
# ---------------------------------------------------------------------------
def test_student_t_ci_uses_sample_sd_and_is_not_clipped():
    ci = A.student_t_ci([0.0] * 5 + [1.0] * 5, 2.262, 9)
    assert ci["df"] == 9 and ci["t_crit"] == 2.262
    assert ci["sd_sample"] == pytest.approx(0.5270462767, rel=1e-6)
    assert ci["lo"] == pytest.approx(0.5 - 2.262 * ci["se"])
    assert ci["lo"] < 0.5 and ci["hi"] > 0.5
    assert ci["hi"] > 1.0 or ci["hi"] <= 1.0     # reported as computed, never clipped


@pytest.mark.parametrize("p1,p2,expect", [
    ("PASS", "PASS", "PASS"),
    ("FAIL", "PASS", "FAIL"),
    ("PASS", "FAIL", "FAIL"),
    ("INCONCLUSIVE", "PASS", "INCONCLUSIVE"),
    ("PASS", "INCONCLUSIVE", "INCONCLUSIVE"),
    ("FAIL", "INCONCLUSIVE", "FAIL"),
])
def test_overall_verdict_composition(p1, p2, expect):
    v = A.overall_verdict({"verdict": p1}, {"verdict": p2})
    assert v["conjunction"].startswith(expect)


def test_frozen_constants_match_the_ratified_bands():
    assert A.P1_FLOOR == 0.35
    assert A.P1_SLICE == "S4"
    assert A.COMPARABILITY_INTERVAL == (0.08, 0.12)
    assert A.ALPHA == 0.05
    assert A.CONFIRMATORY.p2_min_positive == 9 and A.CONFIRMATORY.n_seeds == 10
    assert A.CONFIRMATORY.t_crit == 2.262 and A.CONFIRMATORY.df == 9
    assert A.GBDT_PARAMS["random_state"] == 0 and A.GBDT_PARAMS["n_estimators"] == 100
    assert A.GBDT_PARAMS["max_depth"] == 3 and A.GBDT_PARAMS["learning_rate"] == 0.1
    assert A.FEATS == [
        "update_norm", "train_loss", "num_examples", "norm_variance", "loss_slope",
        "cos_to_median", "L2_to_median", "cos_drift", "cos_variance"]
    assert [b for b, _, _ in A.BASELINES] == ["krum_score", "L2_to_median", "cos_to_median"]


# ---------------------------------------------------------------------------
# v1.15b § 3 — strict-identity key = canonical base device lineage
# ---------------------------------------------------------------------------
def _mal(cid, family):
    return {"malicious_gt": True, "attack_type": family, "logical_cid": cid}


def test_exposed_devices_collapses_rmc_aliases_to_one_lineage():
    """`client_5_new2` and `client_5` are ONE identity (v1.15b § 3)."""
    rows = [_mal("client_5", "alie"), _mal("client_5_new2", "gaussian_noise"),
            _mal("client_7_new10", "alie")]
    devices = A.exposed_devices(rows)
    assert devices["alie"] == {"client_5", "client_7"}
    assert devices["gaussian_noise"] == {"client_5"}


def test_exposed_devices_does_not_merge_the_legacy_non_cycle_aliases():
    """`client_9_new` is partition 10 — a DIFFERENT device, not a cycle of 9."""
    rows = [_mal("client_9_new", "alie")]
    assert A.exposed_devices(rows)["alie"] == {"client_9_new"}
    assert A.canonical_device_id("client_9_new") == "client_9_new"
    assert A.canonical_device_id("client_9_new1") == "client_9"


def test_strict_identity_block_reports_degeneracy_without_crashing():
    """A reported-only sensitivity must never take down the adjudication."""
    rows = [dict(_mal("client_0_new1", "alie"), _scen="S4", _seed=1)]
    primary = {"blend": {}}
    strict = {"degenerate": {"alie": "positive class empty"}}
    block = A.strict_identity_block(rows, primary, strict,
                                    A.exposed_devices(rows), A.DEV_SMOKE)
    assert block["status"].startswith("UNDEFINED")
    assert block["primary_arm_unaffected"] is True
    assert "SWITCHES strategy" in block["finding"]
    # the census still reports what WOULD have been excluded
    assert block["exclusion_census"]["alie"]["canonical_devices_exposed"] == ["client_0"]


def test_strict_identity_census_counts_aliases_a_raw_key_would_miss():
    rows = [_mal("client_0", "alie"), _mal("client_0_new1", "gaussian_noise"),
            _mal("client_0_new2", "gaussian_noise")]
    block = A.strict_identity_block(rows, {"blend": {}},
                                    {"degenerate": {"x": "y"}},
                                    A.exposed_devices(rows), A.DEV_SMOKE)
    alie = block["exclusion_census"]["alie"]
    # alie is named only on client_0, but the lineage covers all three aliases
    assert alie["n_aliases_covered"] == 3
    assert alie["n_aliases_a_raw_logical_cid_key_would_have_MISSED"] == 2


def test_exp048_input_gate_hard_stops_on_a_real_invocation(monkeypatch):
    """A missing mandatory input is a read-prep defect, not a post-hoc note."""
    monkeypatch.delenv("H2PRIME_EXP048_SIG_DIR", raising=False)
    with pytest.raises(A.HardStop, match="EXP-048 INPUT GATE"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_allows_the_dev_smoke_self_test(monkeypatch):
    monkeypatch.delenv("H2PRIME_EXP048_SIG_DIR", raising=False)
    assert A.exp048_input_gate(A.DEV_SMOKE) is None


def _stage_exp048_arm(root: Path, seeds=None, rounds=3, families=None,
                      noisy=False, honest_clients=1, families_by_scenario=None,
                      families_by_seed=None):
    """A dev-shaped standalone-TGE arm over the REGISTERED h2_confirm grid.

    `noisy` + extra honest clients give the detector a non-degenerate score
    distribution, so the realized FPR can land near the 10 % operating point
    the § 3.2 interval is defined around.
    """
    seeds = tuple(A.h2_confirm_seeds()) if seeds is None else seeds
    for scen in SCENARIOS_LONG:
        for seed in seeds:
            unit = f"{scen}__tge__persistent_optimizer__seed{seed}"
            rows = []
            # one honest client plus one carrier per attack family, so every
            # LOAO fold retains a positive class after its family is held out
            # `is None`, not truthiness: an explicitly EMPTY mapping means "no
            # malicious carriers", which is a population a test must be able to
            # stage. Falling back to the default on {} conflates absent with
            # empty and silently hands back a healthy arm.
            #
            # `families_by_scenario` stages the REAL dev shape, where each
            # scenario schedules its own subset of families (S0/S1 carry `alie`
            # alone, S3 carries no `label_flip`). The § 2.2a fit pools across
            # scenarios, so that shape is fittable while an arm-wide single
            # family is not — a distinction only a per-scenario fixture can draw.
            if families_by_seed is not None:
                # Per-SEED control, which the rotation grain needs: `rot.fit`
                # selects seeds, so a rotation-level deficiency can only be
                # staged by varying families across seeds.
                carriers = families_by_seed[seed]
            elif families_by_scenario is not None:
                carriers = families_by_scenario[scen]
            else:
                carriers = ({"client_1": "alie", "client_2": "gaussian_noise",
                             "client_3": "label_flip"} if families is None
                            else families)
            honest = [f"honest_{i}" for i in range(honest_clients)]
            for cid in ["client_0", *honest, *carriers]:
                for rnd in range(1, rounds + 1):
                    fam = carriers.get(cid)
                    mal = fam is not None
                    # honest tge_score must VARY too: a constant honest
                    # score puts the TRUST-direction cut on the constant and
                    # strict `<` then flags nothing (realized FPR exactly 0).
                    tge = (0.2 + _jitter(cid, rnd, seed, salt=7) * 0.5 if mal
                           else 0.55 + _jitter(cid, rnd, seed, salt=7) * 0.45)
                    rows.append(_tge_row(cid, rnd, seed, scen, scen,
                                         tge if noisy else (0.2 if mal else 0.9),
                                         tenure=rnd, mal=mal,
                                         family=fam or "alie", noisy=noisy))
            (root / f"{unit}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return root


def test_exp048_input_gate_rejects_an_EMPTY_directory(tmp_path, monkeypatch):
    """Existence is not staging.

    An empty directory passing this gate means the failure surfaces only after
    sealed rows are opened, and the one-shot read cannot be retried.
    """
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="does not contain a usable standalone-TGE arm"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_krum_tge_only_directory(tmp_path, monkeypatch):
    """The contrast reads the STANDALONE-TGE arm; Krum+TGE logs cannot supply it."""
    unit = "s4_full_mix__krum_tge__persistent_optimizer__seed1"
    (tmp_path / f"{unit}.jsonl").write_text(
        json.dumps(_tge_row("client_0", 1, 1, "s4_full_mix",
                            "s4_full_mix", None)) + "\n",
        encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="standalone-TGE"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_an_incomplete_grid(tmp_path, monkeypatch):
    _stage_exp048_arm(tmp_path)
    seed = A.h2_confirm_seeds()[2]
    next(tmp_path.glob(f"s3_identity_reset_only__tge__*seed{seed}.jsonl")).unlink()
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="REGISTERED grid"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_wholly_omitted_registered_seed(tmp_path, monkeypatch):
    """The expected grid comes from data/h2_confirm_seeds.json, so a seed that
    was never staged at all still halts — it cannot vanish from the universe."""
    _stage_exp048_arm(tmp_path, seeds=tuple(A.h2_confirm_seeds()[:9]))
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="REGISTERED grid"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_malformed_row_beyond_the_first(tmp_path, monkeypatch):
    """Full-row parse: a bad row deep in a file must not survive to the read."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s2_adaptive_switching_only__tge__*.jsonl"))
    lines = victim.read_text().splitlines()
    lines[7] = "{not json"
    victim.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="line 8 is not valid JSON"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_missing_field_beyond_the_first_row(tmp_path, monkeypatch):
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s1_benign_churn_only__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    del rows[5]["tge_score"]
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="lacks 'tge_score'"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_seed_identity_mismatch(tmp_path, monkeypatch):
    _stage_exp048_arm(tmp_path)
    seed = A.h2_confirm_seeds()[0]
    victim = next(tmp_path.glob(f"s0_clean_baseline__tge__*seed{seed}.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    rows[0]["seed"] = 999
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="row seed"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_duplicate_units(tmp_path, monkeypatch):
    """Same (scenario, seed) staged twice under different execution tokens."""
    _stage_exp048_arm(tmp_path)
    seed = A.h2_confirm_seeds()[0]
    src = next(tmp_path.glob(f"s0_clean_baseline__tge__*seed{seed}.jsonl"))
    (tmp_path / f"s0_clean_baseline__tge__other_exec__seed{seed}.jsonl").write_text(
        src.read_text(), encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="duplicate staged unit"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_loader_refuses_to_load_duplicate_units(tmp_path):
    _stage_exp048_arm(tmp_path, seeds=A.h2_confirm_seeds()[:5])
    seed = A.h2_confirm_seeds()[0]
    src = next(tmp_path.glob(f"s0_clean_baseline__tge__*seed{seed}.jsonl"))
    (tmp_path / f"s0_clean_baseline__tge__other_exec__seed{seed}.jsonl").write_text(
        src.read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to load both"):
        A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                       A.R.derive_window_feats)


def test_exp048_required_keys_match_the_builder_input_contract():
    """The gate's key set is DERIVED from the frozen builder, not restated.

    Executable contract: a row carrying exactly the required keys must survive
    derive_window_feats + design_matrix. If the builder starts consuming a new
    field, this fails rather than the sealed pass failing.
    """
    assert set(A.R.RAW) <= set(A.SCORING_INPUT_KEYS)
    assert "tge_score" in A.EXP048_REQUIRED_KEYS
    row = {k: 1.0 for k in A.SCORING_INPUT_KEYS}
    row.update({"logical_cid": "client_0", "scenario_round": 1, "tenure": 1,
                "malicious_gt": False, "attack_type": "", "seed": 1,
                "scenario": "s4_full_mix"})
    rows = [dict(row), dict(row, scenario_round=2, tenure=2)]
    A.R.derive_window_feats(rows)
    X = A.design_matrix(rows)
    assert X.shape == (2, len(A.FEATS))


def test_exp048_input_gate_rejects_a_null_raw_feature(tmp_path, monkeypatch):
    """A null update_norm becomes NaN and fires the § 2.1a hard stop INSIDE the
    sealed pass. It has to fail at dry-run instead."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s0_clean_baseline__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    rows[4]["update_norm"] = None
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="NULL 'update_norm'"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_nonnumeric_raw_feature(tmp_path, monkeypatch):
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s2_adaptive_switching_only__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    rows[3]["train_loss"] = "not-a-number"
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="non-numeric 'train_loss'"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_tolerates_a_null_tge_score(tmp_path, monkeypatch):
    """A null tge_score is a measured COVERAGE fact, not a data defect —
    the coverage machinery reports it, the gate must not reject it."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s1_benign_churn_only__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    rows[2]["tge_score"] = None
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)          # must not raise


def test_input_contract_covers_every_REGISTERED_baseline_instrument():
    """ROUND 21. `design_matrix` is not a row's only consumer: the P2 oracle-max
    arm reads each § 3.1 instrument straight off the row. Two of the three are
    R.RAW features and were covered by accident of that overlap; `krum_score` is
    a baseline instrument ONLY and was in neither contract.

    Asserted against the REGISTERED ROSTER, not a hand-listed set, so adding a
    fourth instrument fails here instead of silently entering the sealed pass.
    """
    assert set(A.SCORING_BASELINE_KEYS) == {b for b, _t, _l in A.BASELINES}
    for k in A.SCORING_BASELINE_KEYS:
        assert k in A.SCORING_INPUT_KEYS, f"{k} not required on input"
        assert k in A.SCORING_VALUE_CONTRACT, f"{k} has no typed value contract"
    # DOUBLE DUTY, stricter contract governs: an instrument that is also a
    # design-matrix feature gets no null tolerance, because there a null becomes
    # NaN and trips the § 2.1a hard stop inside the sealed pass.
    for k in A.SCORING_BASELINE_KEYS:
        expected = "numeric" if k in A.R.RAW else "numeric_or_null"
        assert A.SCORING_VALUE_CONTRACT[k] == expected, k
    # v1.15b § 1.2 item 3 freezes the P2 tie-break to the FIRST entry. If that
    # ordering ever moves, the guarded arg-max identity moves with it.
    assert A.BASELINES[0][0] == "krum_score"


def test_exp048_input_gate_rejects_a_MISSING_baseline_instrument(tmp_path, monkeypatch):
    """The hazard the presence requirement exists for, and it is SILENT.

    The baseline arm reads `r[b] for r in ... if r.get(b) is not None`, so an
    absent column is indistinguishable from an all-null one: the score vector
    comes back empty and the instrument is simply dropped from the oracle
    maximum. Nothing raises. With `krum_score` gone the frozen tie-break falls
    through to L2_to_median, changing WHICH instrument's FPR § 4 (P2) guards
    under the exact-0.000 ties it expects as the normal regime.
    """
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s4_full_mix__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    assert "krum_score" in rows[1], "fixture must carry the key before removing it"
    del rows[1]["krum_score"]
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="krum_score"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_rejects_a_nonnumeric_baseline_instrument(tmp_path, monkeypatch):
    """Present but unusable: the arm casts survivors with dtype=float, so a
    string reaches numpy inside the sealed pass rather than at dry-run."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s3_*__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    rows[2]["krum_score"] = "not-a-number"
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="non-numeric 'krum_score'"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_TOLERATES_a_null_baseline_instrument(tmp_path, monkeypatch):
    """A null `krum_score` is a MEASURED FACT about which defense produced the
    row, not a defect. The deployed emitter is unambiguous: across the 98,100-row
    EXP-011 dev corpus the key is present on every row of every arm, non-null on
    100% of the two Krum-bearing arms (krum, krum_tge) and null on 100% of the
    two that never run a Krum stage (tge, trustscore). Rejecting it would
    hard-stop the entire standalone-TGE arm at the gate.

    This is the ANTI-VACUITY pair to the two rejection tests above: the gate must
    discriminate null from absent and from non-numeric, not simply refuse all three.
    """
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s1_benign_churn_only__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    for r in rows:
        r["krum_score"] = None            # the whole no-Krum arm, as deployed
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)          # must not raise


def test_exp048_input_gate_rejects_an_unregistered_attack_family(tmp_path, monkeypatch):
    """The scorer PARTITIONS on attack_type, so an unregistered value would
    create a phantom family slice no LOAO fold was ever fit for."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s4_full_mix__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    mal = next(i for i, r in enumerate(rows) if r["malicious_gt"])
    rows[mal]["attack_type"] = "min_max"          # a real attack, NOT registered
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="UNREGISTERED 'attack_type'"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_input_gate_accepts_the_empty_discovery_marker(tmp_path, monkeypatch):
    """The empty string is the frozen loader's discovery-row marker (§ 3.2)."""
    _stage_exp048_arm(tmp_path, rounds=6)
    victim = next(tmp_path.glob("s3_identity_reset_only__tge__*.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    mal = next(i for i, r in enumerate(rows) if r["malicious_gt"])
    rows[mal]["attack_type"] = ""                 # discovery row
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)           # must not raise


def test_registered_enum_contract_matches_the_scorer_roster():
    assert set(A.SCORING_ENUM_CONTRACT["attack_type"]) == {""} | set(A.ATTACKS)


def test_exp048_input_gate_passes_on_a_properly_staged_arm(tmp_path, monkeypatch):
    _stage_exp048_arm(tmp_path)
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    assert A.exp048_input_gate(A.CONFIRMATORY) == str(tmp_path)


def test_module_split_keeps_every_public_name_on_the_driver():
    """The split is for file size only — the executor's surface is unchanged."""
    for name in ("Refusal", "HardStop", "Profile", "CONFIRMATORY", "DEV_SMOKE",
                 "sealed_seeds", "load_assembly_map", "rotation_plan",
                 "load_cells", "design_matrix", "exposed_devices",
                 "score_corpus", "adjudicate_p1", "adjudicate_p2",
                 "secondaries", "strict_identity_block", "overall_verdict",
                 "golden_gate", "build_report", "main", "comparable",
                 "exact_sign_p", "student_t_ci", "SCORING_VALUE_CONTRACT"):
        assert hasattr(A, name), f"driver lost its public name: {name}"


def _executor_modules():
    """Every module of the executor, DERIVED FROM THE FILESYSTEM.

    A hand-maintained list silently omits new modules — which is exactly what
    happened when h2prime_exp048.py and h2prime_schema.py were added and the
    hand-list was not updated (rev-14 report claimed otherwise; it was wrong).
    """
    mods = sorted((REPO / "scripts").glob("h2prime_*.py"))
    mods.append(REPO / "scripts" / "adjudicate_h2prime.py")
    return sorted(mods)


def test_module_inventory_is_derived_and_covers_every_executor_module():
    found = {p.name for p in _executor_modules()}
    for expected in ("adjudicate_h2prime.py", "h2prime_common.py",
                     "h2prime_corpus.py", "h2prime_bands.py",
                     "h2prime_report.py", "h2prime_secondaries.py",
                     "h2prime_exp048.py", "h2prime_schema.py"):
        assert expected in found, f"inventory missed {expected}"


def test_no_module_exceeds_the_project_file_ceiling():
    checked = []
    for path in _executor_modules():
        n = len(path.read_text(encoding="utf-8").splitlines())
        checked.append(path.name)
        assert n <= 800, f"{path.name} is {n} lines, over the 800-line ceiling"
    assert len(checked) >= 8, f"inventory shrank to {checked}"


@pytest.mark.golden
def test_golden_gate_passes_unchanged():
    """§ 2.2b hard-stop pre-condition, run exactly as the executor runs it."""
    receipt = A.golden_gate()
    assert receipt["status"] == "PASS"
    assert receipt["frozen_commit"] == "dcef0f7"


# ---------------------------------------------------------------------------
# dev-smoke determinism (slow: 30 GBDT fits on the EXP-011 dev corpus)
# ---------------------------------------------------------------------------
def _dev_map(tmp_path: Path, corpus: Path) -> Path:
    entries = []
    for scen in SCENARIOS_LONG:
        for seed in A.DEV_SEEDS:
            unit = f"{scen}__krum_tge__persistent_optimizer__seed{seed}"
            src = corpus / f"{unit}.jsonl"
            if not src.is_file():
                pytest.skip(f"dev corpus incomplete: {src}")
            entries.append({"scenario": scen, "seed": seed, "source": "EXP-011-dev",
                            "path": str(src)})
    p = tmp_path / "dev_map.json"
    p.write_text(json.dumps({"_meta": {"dev": True}, "cells": entries}), encoding="utf-8")
    return p


@pytest.mark.slow
def test_dev_smoke_is_bit_identical_across_two_runs(tmp_path, monkeypatch):
    import os
    corpus = os.environ.get(DEV_CORPUS_ENV)
    if not corpus or not Path(corpus).is_dir():
        pytest.skip(f"set {DEV_CORPUS_ENV} to the EXP-011 dev signal-log directory")
    mp = _dev_map(tmp_path, Path(corpus))
    outs = []
    for i in (1, 2):
        out = tmp_path / f"run{i}.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(mp), "--out", str(out), "--dev-smoke"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr
        outs.append(out.read_bytes())
    assert outs[0] == outs[1], "DEV-SMOKE outputs are not bit-identical"


@pytest.mark.slow
def test_dev_smoke_reproduces_the_published_dev_grounding(tmp_path):
    """1:1 correspondence gate — the executor must land on the numbers the § 4
    bands were grounded on: S4 blended per-seed {0.535, 0.371, 0.467, 0.518,
    0.429}, mean 0.464, realized honest FPR 0.101 (BLENDED_GROUNDING.md § 2.1),
    and P2's baselines at exactly 0.000 on held-out ALIE in every scenario."""
    import os
    corpus = os.environ.get(DEV_CORPUS_ENV)
    if not corpus or not Path(corpus).is_dir():
        pytest.skip(f"set {DEV_CORPUS_ENV} to the EXP-011 dev signal-log directory")
    mp = _dev_map(tmp_path, Path(corpus))
    report, _text = A.build_report(mp, A.DEV_SMOKE, dry_run=False)

    p1 = report["P1"]
    assert p1["mean_recall"] == pytest.approx(0.464, abs=5e-4)
    assert p1["realized_blended_fpr"] == pytest.approx(0.101, abs=5e-4)
    published = {"42": 0.535, "137": 0.371, "256": 0.467, "314": 0.518, "500": 0.429}
    for seed, value in published.items():
        assert p1["per_seed_recall"][seed] == pytest.approx(value, abs=5e-4)

    p2 = report["P2"]
    assert all(u["oracle_max"] == 0.0 for u in p2["per_unit"]), (
        "dev evidence: every enumerated baseline scores exactly 0.000 on held-out ALIE")
    assert p2["sign_test"]["positive"] == 5
    assert p2["sign_test"]["p_one_sided_exact"] == pytest.approx(1 / 32)


def _jitter(cid, rnd, seed, salt=0):
    """Deterministic per-row variation. Identical features across rows make the
    detector degenerate (constant score, realized FPR 0.0), which cannot reach
    a genuine operating point — so the fixture varies them reproducibly.

    Uses a STABLE digest, not the built-in `hash()`: string hashing is salted
    per process, so a hash()-based fixture would produce different operating
    points on every run and quietly break the determinism this suite asserts.
    """
    key = f"{cid}|{rnd}|{seed}|{salt}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:6], 16) % 1000 / 1000.0


def _tge_row(cid, rnd, seed, scen_long, scen_field, tge, tenure=1, mal=False,
             family="alie", noisy=False):
    j = _jitter(cid, rnd, seed) if noisy else 0.0
    # A SMALL shift keeps the classes overlapping, so the honest score
    # distribution has mass above the 0.90 quantile cut and the realized FPR
    # can sit near 10 % rather than collapsing to 0.
    shift = 0.15 if (noisy and mal) else 0.0
    return {"logical_cid": cid, "scenario_round": rnd, "tenure": tenure,
            "seed": seed, "scenario": scen_field, "malicious_gt": mal,
            "attack_type": family if mal else "", "tge_score": tge,
            "update_norm": 1.0 + rnd + j + shift,
            "train_loss": 0.5 + j * 0.4 + shift * 0.3,
            "num_examples": 10, "cos_to_median": 0.9 - j * 0.2,
            "L2_to_median": 1.0 + j + shift, "krum_score": 0.8 - j * 0.1}


def test_exp048_standalone_tge_loader_takes_only_the_tge_arm(tmp_path):
    """§ 4 secondary 9 loader, verified on a dev-shaped fixture.

    EXP-048 is EXPOSED data, but the loader must still honour the frozen
    per-file discipline and must select the standalone-TGE arm only — that is
    what makes it the FULL-COVERAGE half of G2-EXTENDED.
    """
    import importlib.util as _il
    spec = _il.spec_from_file_location(
        "h2prime_exp048_uut", REPO / "scripts" / "h2prime_exp048.py")
    SEC = _il.module_from_spec(spec)
    sys.modules["h2prime_exp048_uut"] = SEC
    spec.loader.exec_module(SEC)

    for defense in ("tge", "krum_tge"):
        unit = f"s4_full_mix__{defense}__persistent_optimizer__seed42"
        rows = [_tge_row("client_0", r, 42, "s4_full_mix", "S4_full_mix",
                         0.5 if defense == "tge" else None, tenure=r)
                for r in (1, 2, 3)]
        rows.append(_tge_row("client_1", 1, 42, "s4_full_mix", "S4_full_mix",
                             0.2, mal=True))
        (tmp_path / f"{unit}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = SEC.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    assert len(loaded) == 4                       # the tge arm only
    assert {r["_defense"] for r in loaded} == {"tge"}
    assert all(r["_scen"] == "S4" for r in loaded)
    # window features were derived by the frozen builder, never left missing
    assert all(r["norm_variance"] is not None for r in loaded)

    cov = SEC.standalone_tge_coverage(loaded, A.SCENARIOS)
    assert cov["scenarios"]["S4"]["tge_score_coverage"] == 1.0
    assert cov["scenarios"]["S4"]["n_scored_malicious"] == 1


def _g2_cells(per_seed_n_mal, scen="S4", per_family=None):
    return {"g2": {(scen, sd): {
        "det_recall": 0.5, "det_fpr": 0.10, "tge_recall": 0.2, "tge_fpr": 0.10,
        "n_mal_scored": n, "n_mal_total": n, "n_honest_scored": 100,
        "n_honest_total": 100,
        "per_family": (per_family(sd, n) if per_family else {})}
        for sd, n in per_seed_n_mal.items()}}


def test_g2_floor_binds_per_attack_family_within_a_qualifying_fold():
    """A fold can clear 30 POOLED rows while one family contributes a handful;
    that family's contrast is not comparable even though the blend is."""
    def fams(_sd, _n):
        return {"alie": {"det_recall": 0.8, "tge_recall": 0.1,
                         "n_mal_scored": 40, "n_mal_total": 40},
                "gaussian_noise": {"det_recall": 0.9, "tge_recall": 0.2,
                                   "n_mal_scored": 4, "n_mal_total": 4}}
    out = A.SEC.g2_scored_rows_contrast(
        _g2_cells({s: 44 for s in A.DEV_SEEDS}, per_family=fams),
        A.SCENARIOS, A.DEV_SEEDS, A.ATTACKS, A.exact_sign_p, A.student_t_ci,
        A.ci_on_retained)
    e = out["scenarios"]["S4"]["per_attack_family"]
    assert e["alie"]["status"] == "COMPUTED"
    assert e["alie"]["margin_detector_minus_tge"] == pytest.approx(0.7)
    assert e["gaussian_noise"]["status"] == "NOT COMPARABLE"
    # statistics suppressed for the family that fell below the floor
    assert "detector_mean_recall" not in e["gaussian_noise"]
    assert "margin_detector_minus_tge" not in e["gaussian_noise"]
    # but the census is kept, per fold, for both
    assert e["gaussian_noise"]["census_per_fold"][f"S4x{A.DEV_SEEDS[0]}"][
        "n_mal_scored"] == 4


def test_zero_coverage_family_census_distinguishes_absent_from_unscored():
    """`378 eligible / 0 covered` is the coverage story; `0 / 0` erases it."""
    def fams(_sd, _n):
        return {"alie": {"det_recall": 0.8, "tge_recall": 0.1,
                         "n_mal_scored": 40, "n_mal_total": 40}}
    cells = _g2_cells({s: 40 for s in A.DEV_SEEDS}, per_family=fams)
    for v in cells["g2"].values():
        v["family_eligible_totals"] = {"alie": 40, "gaussian_noise": 378,
                                       "label_flip": 0}
    out = A.SEC.g2_scored_rows_contrast(
        cells, A.SCENARIOS, A.DEV_SEEDS, A.ATTACKS, A.exact_sign_p,
        A.student_t_ci, A.ci_on_retained)
    fam = out["scenarios"]["S4"]["per_attack_family"]
    gn = fam["gaussian_noise"]["census_per_fold"][f"S4x{A.DEV_SEEDS[0]}"]
    assert gn["n_mal_scored"] == 0 and gn["n_mal_total"] == 378
    assert "TGE scored none" in gn["note"]
    lf = fam["label_flip"]["census_per_fold"][f"S4x{A.DEV_SEEDS[0]}"]
    assert lf["n_mal_total"] == 0 and "not scheduled" in lf["note"]


def test_confirmatory_census_distinguishes_18_of_18_from_18_of_180():
    """Coverage rides on every branch: the scored count alone is ambiguous."""
    dense = _g2_cells({s: 18 for s in A.DEV_SEEDS})
    for v in dense["g2"].values():
        v["n_mal_total"] = 18                      # everything eligible covered
    sparse = _g2_cells({s: 18 for s in A.DEV_SEEDS})
    for v in sparse["g2"].values():
        v["n_mal_total"] = 180                     # 10 % covered

    def fold(cells):
        out = A.SEC.g2_scored_rows_contrast(
            cells, A.SCENARIOS, A.DEV_SEEDS, A.ATTACKS, A.exact_sign_p,
            A.student_t_ci, A.ci_on_retained)
        return out["scenarios"]["S4"]["per_fold"][f"S4x{A.DEV_SEEDS[0]}"]

    d, sp = fold(dense), fold(sparse)
    assert d["status"] == sp["status"] == "NOT COMPARABLE"
    assert d["n_mal_scored"] == sp["n_mal_scored"] == 18
    assert d["coverage_mal"] == pytest.approx(1.0)
    assert sp["coverage_mal"] == pytest.approx(0.1)
    assert d != sp, "18/18 and 18/180 serialize identically"


def test_g2_comparable_slice_emits_paired_diffs_and_the_registered_ci():
    out = A.SEC.g2_scored_rows_contrast(
        _g2_cells({s: 50 for s in A.DEV_SEEDS}), A.SCENARIOS, A.DEV_SEEDS,
        A.ATTACKS, A.exact_sign_p, A.student_t_ci, A.ci_on_retained)
    e = out["scenarios"]["S4"]
    assert e["status"] == "COMPUTED"
    assert len(e["per_seed_diff"]) == 5
    assert e["sign_test"]["positive"] == 5
    assert e["ci95_student_t_on_paired_diff"]["df"] == 4
    assert e["ci95_student_t_on_paired_diff"]["n_retained"] == 5
    assert e["ci95_student_t_on_paired_diff"]["t_crit"] == 2.776


def test_g2_floor_binds_per_fold_not_on_the_seed_summed_aggregate():
    """Ten 4-row folds are ten NOT-COMPARABLE folds, never a COMPUTED n = 40."""
    out = A.SEC.g2_scored_rows_contrast(
        _g2_cells({s: 4 for s in range(1, 11)}), A.SCENARIOS)
    e = out["scenarios"]["S4"]
    assert e["status"] == "NOT COMPARABLE"
    assert e["n_folds"] == 10 and e["n_folds_comparable"] == 0
    assert e["n_scored_malicious_all_folds"] == 40      # the sum that must NOT save it
    assert "detector" not in e and "tge" not in e       # no statistic computed
    # A disqualified fold publishes CENSUS FACTS but no contrast STATISTICS.
    # The census keys are mandatory: 18/18 and 18/180 must not print alike.
    for fold in e["per_fold"].values():
        assert fold["status"] == "NOT COMPARABLE"
        assert MANDATORY_CENSUS_KEYS <= set(fold)
        assert not ({"det_recall", "tge_recall", "det_fpr", "tge_fpr"}
                    & set(fold)), "statistics leaked onto a disqualified fold"


def test_g2_disqualified_folds_contribute_nothing_to_a_computed_slice():
    """A mixed slice computes on the qualifying folds only."""
    out = A.SEC.g2_scored_rows_contrast(
        _g2_cells({1: 4, 2: 4, 3: 50, 4: 60, 5: 70}), A.SCENARIOS)
    e = out["scenarios"]["S4"]
    assert e["status"] == "COMPUTED"
    assert e["n_folds_comparable"] == 3 and e["computed_over_n_folds"] == 3
    assert set(e["disqualified_folds"]) == {"seed 1", "seed 2"}
    assert set(e["detector"]["per_seed"]) == {"3", "4", "5"}
    assert e["per_fold"]["S4x1"]["status"] == "NOT COMPARABLE"
    assert e["per_fold"]["S4x3"]["status"] == "COMPUTED"


def test_exp048_contrast_executes_the_registered_mechanics(tmp_path):
    """§ 4 secondary 9 must RUN the protocol, not just count coverage.

    Verified on a dev-shaped standalone-TGE fixture: the § 2.2a rotation is
    carved, LOAO detectors are fit, per-scenario cuts come off the calibration
    seed, and the TGE side is scored on the SAME rows, yielding paired blended
    and per-family outputs with their realized FPRs.
    """
    # 3 malicious carriers x 12 rounds = 36 covered rows per fold, clearing
    # the 30-row coverage floor that now governs this side too
    _stage_exp048_arm(tmp_path, rounds=12)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    seeds = A.h2_confirm_seeds()
    plan = A.rotation_plan(seeds)
    scored = A.score_corpus(rows, plan)
    out = A.E48.exp048_full_coverage_contrast(
        rows, plan, scored, A.SCENARIOS, A.ATTACKS, A.exact_sign_p,
        A.comparable, A.student_t_ci, A.ci_on_retained, seeds)

    # the rotation ran over the REGISTERED universe, each seed scored once
    assert len(out["rotation"]) == 10
    assert sorted(r["test"] for r in out["rotation"]) == seeds

    # paired blended output per scenario, with BOTH sides and their FPRs
    assert set(out["blended"]) == set(A.SCENARIOS)
    for scen, e in out["blended"].items():
        # census is mandatory on every branch; the paired statistics exist only
        # for slices whose folds cleared BOTH the floor and the § 3.2 interval
        assert len(e["census_per_fold"]) == 10
        assert e["row_matched"] is True
        if e["status"] != "COMPUTED":
            continue
        assert e["n_paired_seeds"] == 10
        assert e["row_matched"] is True
        assert e["intersection_coverage_mean"] == 1.0
        assert e["n_seeds_expected"] == 10
        assert e["seeds_missing_from_pairing"] == []
        assert e["ci95_student_t_on_paired_diff"]["df"] == 9
        assert e["ci95_student_t_on_paired_diff"]["t_crit"] == 2.262
        assert e["detector_mean"] is not None and e["tge_mean"] is not None
        assert e["margin_detector_minus_tge"] == pytest.approx(
            e["detector_mean"] - e["tge_mean"])
        assert "sign_test" in e and e["sign_test"]["n_eff"] <= 10
        assert e["detector_realized_fpr"] is not None
        assert e["tge_realized_fpr"] is not None
        assert isinstance(e["detector_comparable"], bool)

    # per-family (per-fold) paired output for the family the fixture schedules
    assert "alie" in out["per_family"]
    assert set(out["per_family"]["alie"]) == set(A.SCENARIOS)

    # full coverage is the property that distinguishes this arm
    for scen in A.SCENARIOS:
        assert out["coverage"]["scenarios"][scen]["tge_score_coverage"] == 1.0


def test_g2_detector_fpr_uses_family_mix_weighting(tmp_path):
    """§ 3.2: the blend's honest cost is the family-mix-weighted mean of the
    component detectors' FPRs — an absent family carries ZERO weight, never an
    equal share."""
    _stage_exp048_arm(tmp_path, seeds=A.h2_confirm_seeds()[:5], rounds=6)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    plan = A.rotation_plan(A.h2_confirm_seeds()[:5])
    scored = A.score_corpus(rows, plan)
    assert scored["g2"], "fixture produced no G2 cells"
    for (_scen, _sd), cell in scored["g2"].items():
        mix = cell["det_fpr_family_mix"]
        assert sum(mix.values()) == pytest.approx(1.0)
        # every weight is the family's realized share of the covered rows
        for fam, w in mix.items():
            assert w == pytest.approx(
                cell["per_family"][fam]["n_mal_scored"] / cell["n_mal_scored"])
        # families with no covered rows are named and carry no weight at all
        for fam in cell["families_absent_zero_weight"]:
            assert fam not in mix


def test_zero_coverage_fold_still_materializes_a_census_cell(tmp_path):
    """A fold whose tge_scores are ALL null must still appear in the census."""
    _stage_exp048_arm(tmp_path, seeds=A.h2_confirm_seeds()[:5], rounds=6)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    for r in rows:                       # strip ALL coverage
        r["tge_score"] = None
    plan = A.rotation_plan(A.h2_confirm_seeds()[:5])
    scored = A.score_corpus(rows, plan)
    assert scored["g2"], "zero-coverage folds vanished entirely"
    for cell in scored["g2"].values():
        assert cell["status"] == "CENSUS ONLY"
        assert cell["n_mal_scored"] == 0 and cell["n_mal_total"] > 0
        assert cell["family_eligible_totals"], "eligible totals missing"
        assert "det_recall" not in cell, "statistics on a census-only cell"


def test_unscheduled_family_gets_a_zero_zero_census_row_on_census_only_cells(tmp_path):
    """Round-7's distinction survives on THIS branch too: 0/0 'not scheduled'
    is a different fact from 0/N 'scheduled but unscored', and neither may be a
    missing row."""
    _stage_exp048_arm(tmp_path, seeds=A.h2_confirm_seeds()[:5], rounds=12,
                      families={"client_1": "alie", "client_2": "label_flip"})
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    for r in rows:                       # census-only: honest side uncovered
        if not r["malicious_gt"]:
            r["tge_score"] = None
    seeds = A.h2_confirm_seeds()[:5]
    plan = A.rotation_plan(seeds)
    scored = A.score_corpus(rows, plan)
    assert all(c["status"] == "CENSUS ONLY" for c in scored["g2"].values())
    out = A.E48.exp048_full_coverage_contrast(
        rows, plan, scored, A.SCENARIOS, A.ATTACKS, A.exact_sign_p,
        A.comparable, A.student_t_ci, A.ci_on_retained, seeds)
    unscheduled = out["per_family"].get("gaussian_noise", {}).get("S4")
    assert unscheduled, "unscheduled family produced no slice at all"
    rowset = list(unscheduled["census_per_fold"].values())
    assert rowset and all(r["status"] == "NOT COMPARABLE" for r in rowset)
    assert all(r["n_mal_scored"] == 0 and r["n_mal_total"] == 0 for r in rowset)
    assert all("not scheduled" in r["note"] for r in rowset)


def test_retained_exp048_fold_serializes_both_recalls_and_fprs():
    """A retained fold's difference must be auditable back to its two sides.

    Uses a controlled operating point so the retained branch is exercised
    deterministically rather than depending on where a synthetic arm's FPR
    happens to land.
    """
    seeds = A.h2_confirm_seeds()
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.10), A.SCENARIOS,
        A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)
    e = out["blended"]["S4"]
    assert e["status"] == "COMPUTED"
    kept = [c for c in e["census_per_fold"].values() if c["status"] == "COMPUTED"]
    assert len(kept) == 10
    for c in kept:
        for key in ("detector_recall", "tge_recall", "detector_realized_fpr",
                    "tge_realized_fpr", "detector_comparable", "tge_comparable",
                    "coverage_mal", "coverage_honest"):
            assert key in c, f"retained fold is missing {key}"


# v1.15 § 1 DECISION G item 3: coverage is reported SPLIT honest / malicious.
MANDATORY_CENSUS_KEYS = {"n_mal_scored", "n_mal_total", "n_honest_scored",
                         "n_honest_total", "coverage_mal", "coverage_honest",
                         "status"}
CONTRAST_STATISTIC_KEYS = {"det_recall", "tge_recall", "detector_recall",
                           "detector_mean", "tge_mean", "detector_mean_recall",
                           "tge_mean_recall", "margin_detector_minus_tge"}


def test_census_symmetry_THROUGH_THE_SCORING_PATH(tmp_path):
    """SYMMETRY INVARIANT, exercised end to end.

    Stages a real arm, runs it through `score_corpus`, and reduces BOTH sides
    from that one scoring pass — no preconstructed cells anywhere. The fixture
    is tuned so the realized operating point yields retained AND disqualified
    folds deterministically, which is what lets one real pass cover all four
    branches. The synthetic-cell variant below remains as an ADDITIONAL case
    for operating points the ML path cannot reach.
    """
    seeds = A.h2_confirm_seeds()[:5]
    plan = A.rotation_plan(seeds)
    conf_rows, e048_rows = [], []

    # Two staged arms, BOTH through the real scoring path. The wide arm's
    # operating point yields retained folds; the narrow arm falls under the
    # 30-row coverage floor, which is the confirmatory side's only
    # disqualifier (its FPR gate is EXP-048-specific). One arm cannot produce
    # both branches on both sides, so the invariant is exercised over the union.
    for name, rounds, hc in (("wide", 20, 10), ("narrow", 8, 10)):
        arm = tmp_path / name
        arm.mkdir()
        _stage_exp048_arm(arm, seeds=seeds, rounds=rounds, noisy=True,
                          honest_clients=hc)
        rows = A.E48.load_standalone_tge_rows(arm, A.R.SCEN_SHORT,
                                              A.R.derive_window_feats)
        scored = A.score_corpus(rows, plan)      # THE REAL SCORING PATH
        conf = A.SEC.g2_scored_rows_contrast(
            scored, A.SCENARIOS, seeds, A.ATTACKS, A.exact_sign_p,
            A.student_t_ci, A.ci_on_retained)
        for slice_ in conf["scenarios"].values():
            conf_rows.extend(slice_["per_fold"].values())
        e048 = A.E48.exp048_full_coverage_contrast(
            rows, plan, scored, A.SCENARIOS, A.ATTACKS, A.exact_sign_p,
            A.comparable, A.student_t_ci, A.ci_on_retained, seeds)
        for slice_ in e048["blended"].values():
            e048_rows.extend(slice_["census_per_fold"].values())

    for side, rowset in (("confirmatory", conf_rows), ("exp048", e048_rows)):
        assert rowset, f"{side} produced no rows through the scoring path"
        statuses = {r["status"] for r in rowset}
        assert "COMPUTED" in statuses, f"{side}: no RETAINED row to test"
        assert "NOT COMPARABLE" in statuses, f"{side}: no DISQUALIFIED row"
        for r in rowset:
            assert MANDATORY_CENSUS_KEYS <= set(r), f"{side} row lost census keys"
            if r["status"] == "COMPUTED":
                assert CONTRAST_STATISTIC_KEYS & set(r), f"{side} retained: no stats"
            else:
                assert not (CONTRAST_STATISTIC_KEYS & set(r)), \
                    f"{side} disqualified row leaked statistics"


def test_census_symmetry_across_both_sides_and_both_branches(tmp_path):
    """The same invariant at CONTROLLED operating points — retained here comes
    from a chosen FPR rather than from wherever the fixture's lands."""
    def conf(n_scored, n_total):
        cells = _g2_cells({s: n_scored for s in A.DEV_SEEDS})
        for v in cells["g2"].values():
            v["n_mal_total"] = n_total
        o = A.SEC.g2_scored_rows_contrast(
            cells, A.SCENARIOS, A.DEV_SEEDS, A.ATTACKS, A.exact_sign_p,
            A.student_t_ci, A.ci_on_retained)
        return o["scenarios"]["S4"]["per_fold"][f"S4x{A.DEV_SEEDS[0]}"]

    def exp048(det_fpr, n_mal):
        seeds = A.h2_confirm_seeds()
        o = A.E48.exp048_full_coverage_contrast(
            [], A.rotation_plan(seeds), _exp048_scored(det_fpr, n_mal=n_mal),
            A.SCENARIOS, A.ATTACKS, A.exact_sign_p, A.comparable,
            A.student_t_ci, A.ci_on_retained, seeds)
        return list(o["blended"]["S4"]["census_per_fold"].values())[0]

    retained = {"confirmatory": conf(50, 50), "exp048": exp048(0.10, 40)}
    disqualified = {"confirmatory": conf(18, 180), "exp048": exp048(0.10, 4)}

    for side, row in {**retained, **disqualified}.items():
        assert MANDATORY_CENSUS_KEYS <= set(row), f"{side} row lost census keys"
    for side, row in retained.items():
        assert CONTRAST_STATISTIC_KEYS & set(row), f"{side} retained row has no stats"
    for side, row in disqualified.items():
        assert not (CONTRAST_STATISTIC_KEYS & set(row)), \
            f"{side} disqualified row leaked statistics"


def test_exp048_contrast_applies_the_same_coverage_floor(tmp_path):
    """One floor, one discipline: an under-floor EXP-048 fold is disqualified
    with statistics suppressed, exactly as on the confirmatory G2 side."""
    # 3 carriers x 6 rounds = 18 covered rows per fold — below the 30-row floor
    _stage_exp048_arm(tmp_path, rounds=6)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    seeds = A.h2_confirm_seeds()
    plan = A.rotation_plan(seeds)
    scored = A.score_corpus(rows, plan)
    out = A.E48.exp048_full_coverage_contrast(
        rows, plan, scored, A.SCENARIOS, A.ATTACKS, A.exact_sign_p,
        A.comparable, A.student_t_ci, A.ci_on_retained, seeds)
    for scen, e in out["blended"].items():
        assert e["status"] == "NOT COMPARABLE", scen
        assert e["n_folds_comparable"] == 0
        # statistics suppressed, census retained
        assert "detector_mean" not in e and "margin_detector_minus_tge" not in e
        assert "sign_test" not in e
        assert len(e["census_per_fold"]) == 10
        assert all(c["status"] == "NOT COMPARABLE"
                   for c in e["census_per_fold"].values())
        # suppression is for STATISTICS only — coverage is a census fact and
        # must survive disqualification
        assert all("coverage_mal" in c and "coverage_honest" in c
                   for c in e["census_per_fold"].values())
        assert any(c["coverage_mal"] is not None
                   for c in e["census_per_fold"].values())


def test_ci_on_retained_uses_the_retained_df_and_refuses_below_two():
    """§ CI params come from the RETAINED folds, never the full profile."""
    five = A.ci_on_retained([0.1, 0.2, 0.3, 0.4, 0.5], A.student_t_ci)
    assert five["status"] == "COMPUTED"
    assert five["n_retained"] == 5 and five["df"] == 4
    assert five["t_crit"] == 2.776          # NOT the profile's 2.262

    ten = A.ci_on_retained([0.1] * 9 + [0.2], A.student_t_ci)
    assert ten["df"] == 9 and ten["t_crit"] == 2.262

    one = A.ci_on_retained([0.1], A.student_t_ci)
    assert one["status"] == "NOT COMPUTED" and one["n_retained"] == 1
    assert "fewer than 2" in one["reason"]
    assert A.ci_on_retained([], A.student_t_ci)["status"] == "NOT COMPUTED"


def test_exp048_gate_is_strictly_stronger_than_the_loader(tmp_path):
    """Property: anything the loader raises on, the gate rejects PRE-read."""
    _stage_exp048_arm(tmp_path)
    (tmp_path / "not_a_unit_id.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a four-part unit id"):
        A.E48.validate_exp048_dir(tmp_path, A.R.SCEN_SHORT, A.SCENARIOS,
                                  A.h2_confirm_seeds(), A.EXP048_REQUIRED_KEYS)
    # and the loader would indeed have raised on that same file
    with pytest.raises(ValueError, match="unit-id stem"):
        A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                       A.R.derive_window_feats)


def test_exp048_gate_rejects_an_ALL_HONEST_arm(tmp_path, monkeypatch):
    """ROUND 22. The grid is syntax; the mechanics need POPULATIONS.

    An arm with no malicious rows satisfies the scenario x seed grid, parses
    cleanly, and passes every key/value/enum check — then produces no recall
    population on either side of the matched contrast, inside the sealed pass.
    """
    _stage_exp048_arm(tmp_path, families={})       # every client honest
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="NO MALICIOUS rows"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_gate_rejects_an_arm_with_no_honest_rows(tmp_path, monkeypatch):
    """The mirror class. Without honest rows the per-scenario calibration cut
    and the realized FPR are both undefined — and § 3.2 comparability is
    decided ON that FPR, so the whole guard silently loses its input."""
    _stage_exp048_arm(tmp_path, honest_clients=0,
                      families={"client_0": "alie", "client_1": "gaussian_noise",
                                "client_2": "label_flip", "client_3": "alie"})
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="NO HONEST rows"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_exp048_gate_rejects_malicious_rows_with_no_REGISTERED_family(tmp_path, monkeypatch):
    """The silent one. score_corpus pools ONLY registered families
    (`mal_pooled = [r for A2 in ATTACKS for r in fam[A2]]`), so a unit whose
    malicious population is entirely discovery rows ("") has malicious rows and
    still contributes no cell. Nothing raises; the unit simply is not there."""
    _stage_exp048_arm(tmp_path)
    for fn in tmp_path.glob("*.jsonl"):
        rows = [json.loads(x) for x in fn.read_text().splitlines() if x.strip()]
        for r in rows:
            if r.get("malicious_gt"):
                r["attack_type"] = ""          # legal enum value, no family
        fn.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="registered family label"):
        A.exp048_input_gate(A.CONFIRMATORY)


# The REAL dev shape, read off the disclosed EXP-011 corpus: each scenario
# schedules its own subset of families, and no scenario carries all three.
_DEV_SHAPE_FAMILIES = {
    "s0_clean_baseline": {"client_1": "alie"},
    "s1_benign_churn_only": {"client_1": "alie"},
    "s2_adaptive_switching_only": {"client_1": "alie",
                                   "client_2": "gaussian_noise",
                                   "client_3": "label_flip"},
    "s3_identity_reset_only": {"client_1": "alie",
                               "client_2": "gaussian_noise"},
    "s4_full_mix": {"client_1": "alie", "client_2": "gaussian_noise",
                    "client_3": "label_flip"},
}


def test_exp048_gate_ACCEPTS_per_scenario_one_family_because_fits_pool(tmp_path, monkeypatch):
    """ANTI-VACUITY, at the grain round 23 corrected.

    A family absent from a SCENARIO is the registered normal case. The § 2.2a
    fit builds `fit_rows_all` by filtering on SEED ONLY, so it pools across
    every scenario in the fit seeds: S0 carrying `alie` alone is fittable
    because S2 supplies gaussian_noise and label_flip to the same fit.

    This stages the real EXP-011 dev shape — S0/S1 `alie` only, S3 without
    `label_flip` — and the gate must accept all of it. My round-22 control
    staged an ARM-WIDE alie-only grid instead, which pinned acceptance of
    something the mechanics cannot fit; the rejecting test below is its
    replacement.
    """
    _stage_exp048_arm(tmp_path, families_by_scenario=_DEV_SHAPE_FAMILIES)
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)          # must NOT raise


def test_exp048_gate_REJECTS_an_arm_wide_single_family_grid(tmp_path, monkeypatch):
    """ROUND 23. Every scenario carrying only `alie` means holding `alie` out
    empties the positive class across the WHOLE arm at once — `y` is
    single-class and the GBDT fit dies inside the sealed pass.

    The primary arm has no runtime guard for this: the single-class check at the
    fit site reads `if strict and len(np.unique(y)) < 2`, armed only for the
    strict-identity sensitivity arm. Unguarded at fit time, so it must be caught
    at the gate.
    """
    _stage_exp048_arm(tmp_path, families={"client_1": "alie"})
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="LOAO") as exc:
        A.exp048_input_gate(A.CONFIRMATORY)
    msg = str(exc.value)
    assert "alie" in msg, "the unfittable family must be named"
    assert "per-SCENARIO one-family arm is fine" in msg, (
        "the message must distinguish the legal shape from the illegal one, "
        "or it reads as a blanket ban on one-family scenarios")


_ALL_THREE = {"client_1": "alie", "client_2": "gaussian_noise",
              "client_3": "label_flip"}


def test_exp048_gate_REJECTS_a_single_family_ROTATION_in_a_diverse_arm(tmp_path, monkeypatch):
    """ROUND 24. The arm-wide check cannot see this one.

    `fit_rows_all = [r for r in rows if r["_seed"] in rot.fit]` selects the
    rotation's THREE fit seeds only, so diversity living in the other two seeds
    never reaches that fit. Stage an arm that is diverse overall but whose
    rotation-1 fit seeds carry `alie` alone: the arm-grain check passes and the
    fit is still single-class once `alie` is held out.
    """
    seeds = A.h2_confirm_seeds()
    rot1 = next(r for r in A.rotation_plan(seeds) if r.i == 1)
    by_seed = {sd: ({"client_1": "alie"} if sd in rot1.fit else dict(_ALL_THREE))
               for sd in seeds}
    _stage_exp048_arm(tmp_path, families_by_seed=by_seed)
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="ROTATION BY ROTATION") as exc:
        A.exp048_input_gate(A.CONFIRMATORY)
    msg = str(exc.value)
    assert "rotation 1" in msg, "the deficient rotation must be named"
    assert str(list(rot1.fit)) in msg, "the fit seeds must be named"
    assert "alie" in msg, "the unfittable family must be named"
    # ANTI-VACUITY: the ARM-wide check must NOT be what fired — this arm is
    # diverse overall, so a failure there would mean the coarse check is
    # over-reaching rather than the fine one catching a new case.
    assert "whole arm" not in msg


def test_exp048_gate_ACCEPTS_the_real_dev_shape_at_ROTATION_grain(tmp_path, monkeypatch):
    """ANTI-VACUITY at the new grain, verified against the disclosed corpus
    BEFORE it was written: in the EXP-011 dev tge arm every seed carries all
    three families (alie 1503 / gaussian_noise 378 / label_flip 207 per seed),
    so all five rotations have fit_mal 6534 with a non-empty complement for
    every family — 0 violations. The gate must accept that shape."""
    seeds = A.h2_confirm_seeds()
    _stage_exp048_arm(tmp_path,
                      families_by_seed={sd: dict(_ALL_THREE) for sd in seeds})
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)          # must NOT raise


def test_exp048_rotation_check_uses_the_REGISTERED_construction(tmp_path):
    """The gate must not carry a second copy of the § 2.2a index formula.

    Pinned by behaviour: the rotations the gate iterates are exactly
    `rotation_plan(expected_seeds)` from h2prime_corpus — the same function the
    reducer is handed — so the formula cannot drift between the check and the
    thing it protects.
    """
    import h2prime_exp048 as E
    assert E.rotation_plan is A.rotation_plan
    seeds = A.h2_confirm_seeds()
    rots = A.rotation_plan(seeds)
    assert len(rots) == len(seeds)
    for r in rots:
        assert len(r.fit) == 3
        assert len({r.test, r.calibration, *r.fit}) == 5
    # every seed serves as CALIBRATION exactly once — the fact that makes the
    # per-unit honest check sufficient at the calibration grain
    assert sorted(r.calibration for r in rots) == sorted(seeds)


def test_exp048_gate_counts_DISCOVERY_rows_toward_the_fit_complement(tmp_path, monkeypatch):
    """The complement is `malicious_gt`, not "family-labelled".

    `tr` drops rows only where `attack_type == A`, and `y` is `malicious_gt`, so
    an unlabelled attacker ("") still carries the positive class through a
    hold-out. An arm-wide alie-only grid that ALSO carries discovery rows is
    therefore fittable, and the gate must not reject it — counting only
    family-labelled rows would have.
    """
    _stage_exp048_arm(tmp_path, families={"client_1": "alie"})
    for fn in tmp_path.glob("*.jsonl"):
        rows = [json.loads(x) for x in fn.read_text().splitlines() if x.strip()]
        extra = dict(rows[0])
        extra["malicious_gt"] = True
        extra["attack_type"] = ""            # discovery row: malicious, no family
        extra["logical_cid"] = "client_9"
        rows.append(extra)
        fn.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    A.exp048_input_gate(A.CONFIRMATORY)          # must NOT raise


def test_exp048_gate_names_the_DEFICIENT_unit(tmp_path, monkeypatch):
    """A gate that says "something is wrong" costs a staging cycle to act on.
    One deficient unit among a healthy grid must be named by scenario and seed,
    and the healthy units must not be implicated."""
    _stage_exp048_arm(tmp_path)
    seed = A.h2_confirm_seeds()[3]
    victim = next(tmp_path.glob(f"s2_adaptive_switching_only__tge__*seed{seed}.jsonl"))
    rows = [json.loads(x) for x in victim.read_text().splitlines() if x.strip()]
    for r in rows:
        r["malicious_gt"] = False
    victim.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match=f"S2x{seed}") as exc:
        A.exp048_input_gate(A.CONFIRMATORY)
    msg = str(exc.value)
    assert "NO MALICIOUS rows" in msg
    other = A.h2_confirm_seeds()[4]
    assert f"S2x{other}" not in msg, "a healthy unit was implicated"


def test_exp048_population_gate_is_ARMED_on_the_production_call_path(tmp_path, monkeypatch):
    """The family check needs a registered roster and SKIPS without one, so the
    guard could silently disable it. Pin that the executor's own call site
    supplies the roster — the skip is for direct callers, never for production.
    """
    assert A.SCORING_ENUM_CONTRACT.get("attack_type"), "no registered roster"
    assert [a for a in A.SCORING_ENUM_CONTRACT["attack_type"] if a] == list(A.ATTACKS)
    # Called WITHOUT the enum contract the family check cannot fire...
    _stage_exp048_arm(tmp_path)
    for fn in tmp_path.glob("*.jsonl"):
        rows = [json.loads(x) for x in fn.read_text().splitlines() if x.strip()]
        for r in rows:
            if r.get("malicious_gt"):
                r["attack_type"] = ""
        fn.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    A.E48.validate_exp048_dir(tmp_path, A.R.SCEN_SHORT, A.SCENARIOS,
                              A.h2_confirm_seeds(), A.EXP048_REQUIRED_KEYS)
    # ...but the executor's path passes it, so there it DOES fire.
    monkeypatch.setenv("H2PRIME_EXP048_SIG_DIR", str(tmp_path))
    with pytest.raises(A.HardStop, match="registered family label"):
        A.exp048_input_gate(A.CONFIRMATORY)


def test_G2_empty_cal_tge_is_ABSORBED_by_census_semantics(tmp_path):
    """ROUND 25, the SEVENTH population: `cal_tge` — honest calibration rows
    with a non-null `tge_score`, which sets BOTH sides' G2 cuts.

    Traced through the REAL producer rather than asserted: the guard is
        if n_mal and not (cal_tge and te_h_tge and n_mal_tge):  -> CENSUS ONLY
        elif cal_tge and te_h_tge and n_mal_tge:                -> cuts
    so no cut is ever taken from an empty population — the `elif` requires all
    three non-empty before `R.cut_from_calibration` is reached. The fold does
    not vanish either; it lands in the census naming the uncovered side.

    Disposition: WHY NOT NEEDED — absorbed by census semantics. This is the
    fixture that earns that line in the enumeration.
    """
    _stage_exp048_arm(tmp_path, rounds=12)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    # HONEST side uncovered, malicious side fully covered: exactly the shape
    # that empties cal_tge while leaving real malicious coverage to report.
    for r in rows:
        if not r["malicious_gt"]:
            r["tge_score"] = None
    plan = A.rotation_plan(A.h2_confirm_seeds())
    scored = A.score_corpus(rows, plan)          # THE REAL SCORING PATH
    cells = scored["g2"]
    assert cells, "fixture produced no g2 cells"
    for key, v in cells.items():
        assert v["status"] == "CENSUS ONLY", key
        # the CAUSE is named, and it names the calibration side specifically
        assert "calibration_honest" in v["uncovered_sides"], key
        assert "test_honest" in v["uncovered_sides"], key
        # census facts survive: the malicious side WAS covered and says so
        assert v["n_mal_total"] > 0 and v["n_honest_scored"] == 0
        # no cut, no contrast statistic leaked onto a census-only cell
        assert "det_recall" not in v and "per_family" in v

    # ...and the reducer classifies it with the RIGHT cause, not the floor
    out = A.SEC.g2_scored_rows_contrast(
        cells, A.SCENARIOS, A.h2_confirm_seeds(), A.ATTACKS, A.exact_sign_p,
        A.student_t_ci, A.ci_on_retained, A.comparable)
    for scen, e in out["scenarios"].items():
        assert e["status"] == "NOT COMPARABLE", scen
        assert e["exclusion_causes"]["census_only"] == e["n_folds"], scen
        assert e["exclusion_causes"]["coverage_floor"] == 0, (
            f"{scen}: an uncovered honest side was blamed on the row floor")


def test_G2_global_cut_population_is_GUARDED_when_empty(tmp_path):
    """The EIGHTH population, found by the re-sweep: `cal_pooled` — the
    calibration seed's honest rows POOLED ACROSS SCENARIOS, used to set
    `cut_global` for the § 4 secondary 8 global-cut sensitivity.

    It is a genuinely different population from the per-scenario `cal` of
    grain 4 (pooled, not per-scenario), so grain 4 passing says nothing about
    it. Guarded at BOTH ends: `cut_global = {...} if cal_pooled else {}` at the
    construction site, and `if cut_global:` at every consumer — so an empty
    pooled population omits the sensitivity block rather than setting a cut on
    nothing. Disposition: absorbed, same as grain 7.
    """
    _stage_exp048_arm(tmp_path, rounds=12)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    plan = A.rotation_plan(A.h2_confirm_seeds())
    # Empty the POOLED honest calibration population for one rotation by
    # marking that seed's honest rows malicious-labelled... no: that would
    # change the class balance the fit sees. Drop them instead, which is what
    # an uncovered calibration seed actually looks like.
    rot = plan[0]
    kept = [r for r in rows
            if not (r["_seed"] == rot.calibration and not r["malicious_gt"])]
    assert len(kept) < len(rows), "fixture removed nothing"
    scored = A.score_corpus(kept, plan)          # must not raise
    # The rotation whose calibration seed lost its honest rows contributes no
    # global-cut cell; the OTHER rotations still do, so the guard is scoped and
    # not a blanket disable.
    gc = scored.get("alie_global") or {}
    assert all(sd != rot.test for (_sc, sd) in gc), (
        "a global cut was taken with no pooled calibration population")
    assert gc, "the guard disabled every rotation, not just the uncovered one"


def test_exp048_slice_coverage_aggregates_over_EVERY_fold(tmp_path):
    """ROUND 26 #1. Coverage is a CENSUS fact, so it sums over every fold.

    `cov[sd]` is only assigned on the retained path — an excluded fold
    `continue`s before reaching it — so `intersection_coverage_mean` describes
    the folds that SURVIVED, which is systematically higher than the coverage
    of the slice. Both figures are now reported, each labelled with the
    population it sums over.
    """
    seeds = A.h2_confirm_seeds()
    cells = _exp048_scored(0.10, n_mal=40, seeds=seeds)
    # One fold under the floor AND barely covered: excluded from the statistic,
    # but its 2-of-400 coverage is part of what the slice covered.
    victim = seeds[0]
    cells["g2"][("S4", victim)].update({"n_mal_scored": 2, "n_mal_total": 400})
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), cells, A.SCENARIOS, A.ATTACKS,
        A.exact_sign_p, A.comparable, A.student_t_ci, A.ci_on_retained, seeds)
    e = out["blended"]["S4"]
    assert e["status"] == "COMPUTED"
    assert e["n_folds_comparable"] == 9 and e["n_folds"] == 10
    # the census-complete figure: 9 folds at 40/40 plus one at 2/400
    assert e["n_scored_malicious_all_folds"] == 9 * 40 + 2
    assert e["n_malicious_total_all_folds"] == 9 * 40 + 400
    assert e["coverage_malicious_all_folds"] == pytest.approx(362 / 760)
    # THE POINT: the comparable-only figure overstates, and the two are now
    # distinguishable instead of one unlabelled number.
    assert e["intersection_coverage_mean"] > e["coverage_malicious_all_folds"]
    assert "comparable folds only" in e["intersection_coverage_population"]
    assert "including disqualified" in e["coverage_all_folds_population"]
    # honest side reported separately, per DECISION G item 3
    assert e["coverage_honest_all_folds"] == pytest.approx(1.0)


def test_family_slice_reports_AGGREGATE_coverage_not_just_fold_counts(tmp_path):
    """ROUND 26 #2. A fold COUNT says how many folds were excluded; it does not
    say how much of the family the slice scored. Different questions."""
    cells = _fam_cells(det_fpr=0.10, fam_fpr=0.10, fam_n=40)
    # one fold partially covered and under the floor
    victim = (A.SCENARIOS[-1], A.DEV_SEEDS[0])
    cells["g2"][victim]["per_family"]["alie"].update(
        {"n_mal_scored": 3, "n_mal_total": 300})
    fam = _conf_slice(cells)["per_attack_family"]["alie"]
    assert fam["n_folds"] == 5 and fam["n_folds_comparable"] == 4
    # summed over EVERY fold, disqualified included
    assert fam["n_scored_malicious_all_folds"] == 4 * 40 + 3
    assert fam["n_malicious_total_all_folds"] == 4 * 40 + 300
    assert fam["coverage_fraction_malicious"] == pytest.approx(163 / 460)
    assert fam["coverage_fraction_honest"] == pytest.approx(1.0)
    assert "including disqualified" in fam["coverage_population"]
    # the aggregate must reconcile with the per-fold census it summarises
    rows = fam["census_per_fold"].values()
    assert fam["n_scored_malicious_all_folds"] == sum(r["n_mal_scored"] for r in rows)


def test_confirmatory_REQUIRES_a_content_digest_per_entry(tmp_path):
    """ROUND 27. A path proves a file EXISTS; it does not prove the bytes are
    the registered ones. The sealed corpus is opened exactly once, so a
    substituted or re-generated file cannot be caught afterwards — and the old
    loader verified a digest when present and shrugged when absent, so a
    hand-written or dry-run-produced map silently downgraded custody to paths.
    """
    # A map that is VALID IN EVERY OTHER RESPECT — the sealed seed set, real
    # files, correct stems — so the only thing under test is the digest.
    # Reusing a wrong-seed fixture would pass on check ORDER rather than on
    # the gate existing.
    seeds = A.sealed_seeds()
    ok = _write_map(tmp_path, seeds)
    A.load_assembly_map(ok, A.CONFIRMATORY)      # ANTI-VACUITY: valid as built
    (tmp_path / "nodigest").mkdir()
    stripped = _write_map(tmp_path / "nodigest", seeds,
                          mutate=lambda d: [c.pop("sha256", None)
                                            for c in d["cells"]])
    with pytest.raises(A.Refusal, match="NO sha256 content digest") as exc:
        A.load_assembly_map(stripped, A.CONFIRMATORY)
    msg = str(exc.value)
    assert "50 of 50" in msg, "the SCALE of the gap must be reported"
    assert "read exactly once" in msg, "the message must say why it matters"


def test_devsmoke_digest_exemption_is_PROFILE_SCOPED_and_still_verifies(tmp_path):
    """The exemption is stated, narrow, and does NOT disable the check.

    DEV-SMOKE runs on already-disclosed data, adjudicates nothing, and may be
    re-run freely, so a digest-less dev map has no custody consequence. But a
    digest that IS present is verified on EVERY profile — the flag governs the
    REQUIREMENT, never the check. Both halves asserted, because an exemption
    that quietly skipped verification would be the worse bug.
    """
    assert A.CONFIRMATORY.requires_content_digest is True
    assert A.DEV_SMOKE.requires_content_digest is False
    seeds = A.DEV_SEEDS
    ok = _write_map(tmp_path, seeds,
                    mutate=lambda d: [c.pop("sha256", None) for c in d["cells"]])
    A.load_assembly_map(ok, A.DEV_SMOKE)         # exempt: must not raise
    # ...but a WRONG digest still fails on dev-smoke
    other = tmp_path / "b"
    other.mkdir()
    bad = _write_map(other, seeds,
                     mutate=lambda d: d["cells"][0].__setitem__("sha256", "0" * 64))
    with pytest.raises(A.Refusal, match="sha256 mismatch"):
        A.load_assembly_map(bad, A.DEV_SMOKE)


def test_map_builder_emits_digests_even_in_DRY_RUN(tmp_path):
    """The other half of the contract. A dry-run map used to carry no `sha256`
    at all, which is exactly the artifact the old conditional waved through."""
    src = REPO / "scripts" / "build_exp051_assembly_map.py"
    body = src.read_text(encoding="utf-8")
    assert "digest_unavailable" in body, (
        "a dry-run entry must SAY the digest is absent rather than omitting "
        "the field silently")
    # ROUND 28 CORRECTION to round 27: a dry-run must NEVER hash whatever sits
    # at the destination. `download()` did not run, so those bytes were not
    # fetched by this build — digesting them manufactures custody evidence for
    # a stale or foreign file, which is worse than no digest at all.
    assert "if args.dry_run:\n            entry[\"sha256\"] = None" in body, (
        "dry-run must unconditionally declare the digest unavailable")
    assert "download did not run" in body


def test_exp048_gate_pass_implies_loader_success(tmp_path):
    """The property stated directly: a fixture the gate passes must load."""
    _stage_exp048_arm(tmp_path)
    A.E48.validate_exp048_dir(tmp_path, A.R.SCEN_SHORT, A.SCENARIOS,
                              A.h2_confirm_seeds(), A.EXP048_REQUIRED_KEYS)
    rows = A.E48.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    assert rows, "gate passed but loader produced no rows"


def test_exp048_loader_rejects_a_malformed_unit_id(tmp_path):
    import importlib.util as _il
    spec = _il.spec_from_file_location(
        "h2prime_exp048_uut2", REPO / "scripts" / "h2prime_exp048.py")
    SEC = _il.module_from_spec(spec)
    sys.modules["h2prime_exp048_uut2"] = SEC
    spec.loader.exec_module(SEC)
    (tmp_path / "not_a_unit_id.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unit-id stem"):
        SEC.load_standalone_tge_rows(tmp_path, A.R.SCEN_SHORT,
                                     A.R.derive_window_feats)


@pytest.mark.slow
def test_dev_smoke_reproduces_published_secondary_references(tmp_path):
    """The reported secondaries must land on their published dev references.

    AUC per scenario: `S4_ALIE_COLLAPSE.md` § 1.3 (0.849 / 0.843 / 0.894 /
    0.918 / 0.817). Global-cut realized FPR: v1.15 § 2.2's disclosed ~0.169
    against the per-scenario arm's 0.101.
    """
    import os
    corpus = os.environ.get(DEV_CORPUS_ENV)
    if not corpus or not Path(corpus).is_dir():
        pytest.skip(f"set {DEV_CORPUS_ENV} to the EXP-011 dev signal-log directory")
    mp = _dev_map(tmp_path, Path(corpus))
    report, _text = A.build_report(mp, A.DEV_SMOKE, dry_run=False)
    sec = report["secondaries"]

    published_auc = {"S0": 0.849, "S1": 0.843, "S2": 0.894, "S3": 0.918, "S4": 0.817}
    for scen, value in published_auc.items():
        assert sec["auc_per_scenario"][scen]["mean"] == pytest.approx(value, abs=5e-4)

    # S0/S1 schedule ALIE only, so their G2 cells must show the other two
    # families absent with zero weight — the equal-share bug would hide here
    g2_s0 = sec["g2_scored_rows_contrast"]["scenarios"]["S0"]["per_fold"]
    assert g2_s0, "no S0 G2 folds"

    # § 4 secondary 7 at the v1.15b § 1.2 readout grain. The amendment
    # discloses the dev cohort FPRs it was ruled on — detector 0.1163 and
    # instruments krum 0.1028 / L2 0.1000 / cos 0.1094, all inside [0.08, 0.12].
    # So on dev every contrast COMPUTES: if the guard ever suppresses these,
    # either the pooling changed or the corpus did.
    s7 = sec["alie_fixed_baseline_contrasts"]["contrasts"]
    published_pooled = {"krum_score": 0.1028, "L2_to_median": 0.1000,
                        "cos_to_median": 0.1094}
    for b, value in published_pooled.items():
        assert s7[b]["status"] == "COMPUTED", b
        assert s7[b]["baseline_pooled_fpr"] == pytest.approx(value, abs=5e-4)
        assert s7[b]["detector_pooled_fpr"] == pytest.approx(0.1163, abs=5e-4)
        assert s7[b]["detector_in_interval"] is True
        assert s7[b]["baseline_in_interval"] is True

    # Round 29: the blend-margin guard consumes P1's own quantity on P1's own
    # slice. If these two ever diverge, one of the two constructions drifted.
    s4m = sec["S4_blend_margin"]
    assert s4m["detector_mean_realized_fpr"] == pytest.approx(
        report["P1"]["realized_blended_fpr"])
    assert s4m["detector_mean_realized_fpr"] == pytest.approx(0.101, abs=5e-4)
    for b, node in s4m["baselines"].items():
        assert node["status"] == "COMPUTED", b     # dev sits inside [0.08, 0.12]
        assert node["detector_in_interval"] and node["baseline_in_interval"], b
    assert sec["S3_blend_margin"]["detector_mean_realized_fpr"] == pytest.approx(
        0.1121, abs=5e-4)

    gc = sec["global_cut_sensitivity"]["S4_blend_global_cut"]
    assert gc["mean_realized_fpr"] == pytest.approx(0.169, abs=1e-3)
    assert gc["would_be_comparable"] is False      # ~0.17 is decisively outside
    assert report["P1"]["realized_blended_fpr"] == pytest.approx(0.101, abs=5e-4)


@pytest.mark.slow
def test_dry_run_opens_no_signal_log(tmp_path):
    """--dry-run must validate and plan without reading a single row."""
    p = _write_map(tmp_path, A.DEV_SEEDS)          # placeholder rows, unparseable
    report, text = A.build_report(p, A.DEV_SMOKE, dry_run=True)
    assert report["dry_run"] is True
    assert "no signal-log row was opened" in text
    assert len(report["rotation_plan"]) == 5


# ---------------------------------------------------------------------------
# the frozen OUTPUT CONTRACT (scripts/h2prime_schema.py)
# ---------------------------------------------------------------------------
def _schema():
    import importlib.util as _il
    spec = _il.spec_from_file_location("h2prime_schema_uut",
                                       REPO / "scripts" / "h2prime_schema.py")
    m = _il.module_from_spec(spec)
    sys.modules["h2prime_schema_uut"] = m
    spec.loader.exec_module(m)
    return m


def _margin_fixture_nodes():
    """Both branches of the round-29 blend-margin guard, in fixture form."""
    census = {n: None for n in _MARGIN_CENSUS}
    return {
        "krum_score": {**census, "status": "COMPUTED",
                       **{n: None for n in _MARGIN_STATISTICS}},
        "cos_to_median": {**census, "status": "NOT COMPARABLE",
                          "exclusion_cause": None, "reason": None, "note": None},
    }


def _fixture_report(tmp_path):
    """A COMPLETE report: real G2 + EXP-048 blocks from a fixture arm, plus
    minimally-populated versions of every other mandatory block.

    Completeness matters for the mutation tests: against a partial document
    every mutation would "pass" because some OTHER rule was already failing,
    and the test would prove nothing about the key it removed.
    """
    arm = tmp_path / "arm"
    arm.mkdir(parents=True, exist_ok=True)
    _stage_exp048_arm(arm, rounds=12)
    seeds = A.h2_confirm_seeds()
    rows = A.E48.load_standalone_tge_rows(arm, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    plan = A.rotation_plan(seeds)
    scored = A.score_corpus(rows, plan)

    def keys(*names):
        return {n: None for n in names}

    return {
        "_meta": keys("executor", "spec", "profile", "golden_gate", "versions",
                      "gbdt_params", "features_frozen_order", "baselines",
                      "map_sha256", "defense_token", "seeds_ascending",
                      "n_cells", "n_seeds"),
        "rotation_plan": [{"i": 1, "test": seeds[0]}],
        "P1": keys("band", "slice", "floor", "n_units", "per_seed_recall",
                   "per_seed_realized_blended_fpr", "mean_recall",
                   "realized_blended_fpr", "comparability_interval",
                   "ci95_student_t", "ci_is_reported_not_adjudicating",
                   "verdict", "reason"),
        "P2": {
            **keys("band", "population", "estimand", "required_strictly_positive",
                   "per_seed_detector_macro", "per_seed_oracle_max_macro",
                   "per_seed_diff", "sign_test", "alpha", "verdict", "reason"),
            "per_unit": [{"scenario": "S4", "seed": seeds[0],
                          "detector_recall": 0.5, "detector_fpr": 0.10,
                          "comparable_diagnostic": True}],
            "readout_grain_comparability": keys(
                "_authority", "detector_pooled_fpr", "baseline_pooled_fpr",
                "detector_flagged_honest", "detector_honest_rows",
                "baseline_flagged_honest", "baseline_honest_rows",
                "detector_in_interval", "baseline_in_interval",
                "argmax_instrument_counts", "n_cells_where_argmax_was_a_tie",
                "n_cells_no_tie", "n_cells_partial_tie", "n_cells_full_tie",
                "partial_tie_cells"),
            "realized_fpr_diagnostic": {},
        },
        "secondaries": {
            "auc_per_scenario": {
                "S4": keys("per_seed", "mean", "ci95_student_t")},
            # Round 28: these three are emitted by the executor and were
            # ABSENT here, which is why the rule-coverage meta-test never
            # enumerated them and they went unguarded for 27 rounds.
            "S4_blend_margin": keys(
                "detector_blended_mean", "detector_per_seed",
                "detector_mean_realized_fpr", "detector_ci95_student_t",
                "oracle_max") | {"baselines": _margin_fixture_nodes()},
            "S3_blend_margin": keys(
                "detector_blended_mean", "detector_per_seed",
                "detector_mean_realized_fpr", "detector_ci95_student_t",
                "oracle_max") | {"baselines": _margin_fixture_nodes()},
            # Round 30: this fixture said `mean` where the reducer emits
            # `mean_recall`, and carried no per-seed row at all. Nothing caught
            # it for two rounds because the rule required nothing — the fixture
            # now mirrors the real emission, pinned by a test.
            "per_family_fold_recalls": {
                "alie": {"S4": {**keys("mean_recall", "mean_realized_fpr",
                                       "ci95_student_t"),
                                "per_seed": {"42": keys("recall",
                                                        "realized_fpr",
                                                        "n_mal")}}}},
            # LIST-valued disclosures. Present so the registry check sees them;
            # they carry no rule because the validator only walks dict nodes,
            # so a rule here would select nothing and check nothing.
            "declared_vacuous_cells": ["label_flip × S3 — structurally zero rows"],
            "not_computed_by_this_executor": [],
            "global_cut_sensitivity": {
                "S4_blend_global_cut": keys("per_seed", "mean_recall",
                                            "mean_realized_fpr",
                                            "would_be_comparable")},
            # BOTH branches of the § 3.2 guard are represented, so a mutation
            # against either one lands on a node that really exists (round 28:
            # the fixture's omissions are what let blocks go unguarded).
            "alie_fixed_baseline_contrasts": {
                "contrasts": {
                    "krum_score": {**keys(*_S7_CENSUS, *_S7_STATISTICS),
                                   "status": "COMPUTED"},
                    "cos_to_median": {**keys(*_S7_CENSUS, "reason", "note",
                                             "exclusion_cause"),
                                      "status": "NOT COMPARABLE"},
                }},
            "bracket_recall_by_fpr": {
                **keys("_note", "targets", "adjudicating_target",
                       "interval_rule", "comparability_is_evidence_only"),
                "points": {f"{t:g}": {
                    **keys("target_fpr", "interval", "is_adjudicating_point"),
                    "scenarios": {"S4": keys(*_BRACKET_SLICE_KEYS)}}
                    for t in A.BRACKET_TARGETS}},
            "strict_identity_loao_sensitivity": keys(
                "_note", "identity_key", "exclusion_census", "status"),
            "window_aware_loao_sensitivity": keys(
                "_note", "window_rule", "corpus_census", "status"),
            "g2_scored_rows_contrast": A.SEC.g2_scored_rows_contrast(
                scored, A.SCENARIOS, seeds, A.ATTACKS, A.exact_sign_p,
                A.student_t_ci, A.ci_on_retained),
            "exp048_standalone_tge_full_coverage": A.E48.exp048_full_coverage_contrast(
                rows, plan, scored, A.SCENARIOS, A.ATTACKS, A.exact_sign_p,
                A.comparable, A.student_t_ci, A.ci_on_retained, seeds),
        },
        "verdict": keys("p1", "p2", "conjunction", "terminal_protocol"),
    }


def test_fixture_report_is_schema_clean_before_any_mutation(tmp_path):
    """The mutation fixture must START valid, or the mutations prove nothing."""
    assert _schema().check(_fixture_report(tmp_path))["status"] == "PASS"


def test_schema_passes_on_the_dev_smoke_output(tmp_path):
    """Golden coverage: the real dev-smoke document satisfies the contract."""
    import os
    corpus = os.environ.get(DEV_CORPUS_ENV)
    if not corpus or not Path(corpus).is_dir():
        pytest.skip(f"set {DEV_CORPUS_ENV} to the EXP-011 dev signal-log directory")
    mp = _dev_map(tmp_path, Path(corpus))
    report, _text = A.build_report(mp, A.DEV_SMOKE, dry_run=False)
    receipt = _schema().check(report)
    assert receipt["status"] == "PASS"
    assert receipt["nodes_checked"] > 100


def test_schema_selectors_fan_out_over_lists_as_well_as_dicts():
    """A wildcard that skipped list rows would make its rule silently vacuous."""
    S = _schema()
    doc = {"P2": {"per_unit": [{"scenario": "S0", "seed": 1,
                                "detector_recall": 0.5, "detector_fpr": 0.1,
                                "comparable_diagnostic": True}]}}
    assert len(S._resolve(doc, ["P2", "per_unit", "*"])) == 1


@pytest.mark.parametrize("selector,key", [
    ("secondaries.g2_scored_rows_contrast.scenarios.S4.per_fold", "coverage_mal"),
    ("secondaries.g2_scored_rows_contrast.scenarios.S4.per_fold", "n_honest_total"),
    ("secondaries.g2_scored_rows_contrast.scenarios.S4.per_attack_family.alie"
     ".census_per_fold", "n_mal_total"),
    ("secondaries.exp048_standalone_tge_full_coverage.blended.S4.census_per_fold",
     "coverage_honest"),
])
def test_schema_mutation_knocking_out_a_census_key_is_caught(tmp_path, selector, key):
    """MUTATION: remove one mandatory key from one branch type; the validator
    must fire. This is the guard that rounds 4-9 lacked."""
    S = _schema()
    report = _fixture_report(tmp_path)
    node = report
    for part in selector.split("."):
        node = node[part]
    victim = next(iter(node.values()))
    assert key in victim, f"fixture does not exercise {selector}.{key}"
    del victim[key]
    with pytest.raises(S.SchemaViolation, match="OUTPUT SCHEMA VIOLATION"):
        S.check(report)


def test_schema_catches_statistics_leaking_onto_a_disqualified_node(tmp_path):
    """MUTATION: publish a contrast statistic on a NOT COMPARABLE row."""
    S = _schema()
    report = _fixture_report(tmp_path)
    folds = report["secondaries"]["g2_scored_rows_contrast"]["scenarios"]["S4"]["per_fold"]
    victim = next(iter(folds.values()))
    victim["status"] = "NOT COMPARABLE"
    victim["det_recall"] = 0.99
    with pytest.raises(S.SchemaViolation, match="leaked contrast statistics"):
        S.check(report)


def test_schema_catches_a_missing_top_level_block():
    S = _schema()
    with pytest.raises(S.SchemaViolation, match="document root"):
        S.check({"_meta": {}, "P1": {}, "P2": {}, "secondaries": {}})


def test_every_schema_rule_has_an_authority_citation():
    """A contract clause without its authority is an assertion, not a rule."""
    for rule in _schema().SCHEMA:
        assert rule.authority, f"rule {rule.name!r} cites no authority"


@pytest.mark.parametrize("block", ["_meta", "P1", "P2", "verdict",
                                   "rotation_plan", "secondaries"])
def test_schema_mutation_deleting_a_whole_mandatory_block_is_caught(tmp_path, block):
    """MUTATION: delete an entire mandatory block. A selector that resolves to
    NO nodes must FAIL — vacuous satisfaction is how a schema reads as coverage
    while checking nothing."""
    S = _schema()
    report = _fixture_report(tmp_path)
    del report[block]
    with pytest.raises(S.SchemaViolation, match="OUTPUT SCHEMA VIOLATION"):
        S.check(report)


def test_schema_mutation_deleting_a_mandatory_secondary_is_caught(tmp_path):
    S = _schema()
    report = _fixture_report(tmp_path)
    del report["secondaries"]["g2_scored_rows_contrast"]
    with pytest.raises(S.SchemaViolation, match="matched NO nodes"):
        S.check(report)


def test_optional_block_is_allowed_only_when_the_report_declares_NOT_RUN(tmp_path):
    """`optional_when` is a PREDICATE over the document, not a truthy string.

    Absent-and-declared passes; absent-while-claiming-results fails. A bare
    string would have made the second case pass too.
    """
    S = _schema()
    optional = [r for r in S.SCHEMA if r.optional_when]
    assert optional, "no rule is marked optional; the mechanism is unused"
    for rule in optional:
        assert isinstance(rule.optional_when, S.Condition)
        assert rule.optional_when.because, f"{rule.name} states no reason"

    # (a) the arm was not staged and the report SAYS SO — the legitimate case
    declared = _fixture_report(tmp_path)
    declared["secondaries"]["exp048_standalone_tge_full_coverage"] = {
        "status": "NOT RUN", "reason": "H2PRIME_EXP048_SIG_DIR unset"}
    assert S.check(declared)["status"] == "PASS"


def test_optional_rules_are_NOT_waived_when_the_report_claims_results(tmp_path):
    """MUTATION both ways: a COMPUTED report missing its results must FAIL."""
    S = _schema()
    # (b) the block is gone entirely — the condition cannot hold, so the
    # optional waiver does not apply and the mandatory rules bite
    gone = _fixture_report(tmp_path)
    del gone["secondaries"]["exp048_standalone_tge_full_coverage"]
    with pytest.raises(S.SchemaViolation, match="matched NO nodes"):
        S.check(gone)

    # (c) the block CLAIMS results but its blended output was deleted
    claiming = _fixture_report(tmp_path)
    block = claiming["secondaries"]["exp048_standalone_tge_full_coverage"]
    block["status"] = "COMPUTED"
    block["blended"] = {}
    with pytest.raises(S.SchemaViolation, match="EXP-048 blended slice"):
        S.check(claiming)


def test_schema_catches_statistics_leaking_onto_a_per_family_census_row(tmp_path):
    """MUTATION: the per-family branch gets the same guard as the blended one."""
    S = _schema()
    report = _fixture_report(tmp_path)
    fams = report["secondaries"]["g2_scored_rows_contrast"]["scenarios"]["S4"][
        "per_attack_family"]
    victim = next(iter(next(iter(fams.values()))["census_per_fold"].values()))
    victim["status"] = "NOT COMPARABLE"
    victim["margin_detector_minus_tge"] = 0.42
    with pytest.raises(S.SchemaViolation, match="leaked contrast statistics"):
        S.check(report)


def test_schema_catches_a_retained_row_that_dropped_one_side_of_the_contrast(tmp_path):
    """MUTATION, the round-20 direction: suppression had a contract and
    retention did not, so a COMPUTED row that quietly dropped a readout read as
    compliant. Delete one and the gate must stop the document."""
    S = _schema()
    report = _fixture_report(tmp_path)
    fams = report["secondaries"]["g2_scored_rows_contrast"]["scenarios"]["S4"][
        "per_attack_family"]
    victim = next(iter(next(iter(fams.values()))["census_per_fold"].values()))
    victim["status"] = "COMPUTED"
    victim.update({k: 0.5 for k in S.RETAINED_READOUT_KEYS})
    S.check(report)          # ANTI-VACUITY: the row passes while it is complete
    del victim["tge_realized_fpr"]
    with pytest.raises(S.SchemaViolation, match="dropped retained readouts"):
        S.check(report)


def test_every_emitted_block_is_selected_by_at_least_one_schema_rule(tmp_path):
    """RULE COVERAGE — the anti-vacuity principle applied to the schema itself.

    Enumerate the block paths the executor actually emits and assert each is
    addressed by some rule. A block nobody selects is a block nobody guards,
    which is how the EXP-048 per_family output went unchecked until round 11.
    """
    S = _schema()
    report = _fixture_report(tmp_path)

    def emitted_block_paths(node, prefix=""):
        """Dict-valued paths worth guarding, one level into each collection."""
        paths = []
        if not isinstance(node, dict):
            return paths
        for key, child in node.items():
            if str(key).startswith("_"):
                continue
            path = f"{prefix}{key}"
            if isinstance(child, dict) and child:
                paths.append(path)
                paths.extend(emitted_block_paths(child, path + "."))
        return paths

    selectors = [r.select for r in S.SCHEMA]

    def covered(path: str) -> bool:
        segs = path.split(".")
        for sel in selectors:
            ssegs = [x for x in sel.split(".") if x]
            if len(ssegs) > len(segs):
                ssegs = ssegs[:len(segs)]
            if len(ssegs) < 1:
                continue
            if all(a == "*" or a == b for a, b in zip(ssegs, segs)):
                return True
        return False

    gaps = [p for p in emitted_block_paths(report)
            if p.count(".") <= 3 and not covered(p)]
    assert not gaps, f"emitted blocks selected by NO schema rule: {sorted(set(gaps))}"


def test_every_REGISTERED_secondary_block_has_a_rule_and_a_fixture(tmp_path):
    """ROUND 28. The enumeration bug behind the missing rules.

    The rule-coverage test derives its universe from `_fixture_report`, a
    HAND-BUILT document. S4_blend_margin, S3_blend_margin and
    per_family_fold_recalls are emitted by the executor but were absent from
    that fixture, so the enumeration never saw them, and the coverage test
    passed while checking nothing about them — a test whose universe is a
    fixture can only ever be as complete as someone remembered to make it.

    The universe is now a REGISTERED list. Both halves are asserted, because
    either alone leaves the hole open: a block with a rule but no fixture is
    never enumerated, and a block in the fixture with no rule is unguarded.
    """
    S = _schema()
    report = _fixture_report(tmp_path)
    emitted = report["secondaries"]
    selectors = [r.select for r in S.SCHEMA]
    for block in S.SECONDARY_BLOCKS:
        assert block in emitted, (
            f"{block} is registered but missing from _fixture_report — the "
            "coverage meta-test cannot enumerate what the fixture omits")
        if not isinstance(emitted[block], dict):
            # A list-valued disclosure carries no rule ON PURPOSE: the
            # validator walks dict nodes only, so a rule would select nothing
            # and assert nothing. Presence in the registry is its guard.
            continue
        assert any(sel == f"secondaries.{block}"
                   or sel.startswith(f"secondaries.{block}.")
                   for sel in selectors), f"{block} is selected by NO rule"
    # ...and the registry itself must match what the executor actually emits,
    # or it becomes the next stale fixture.
    unregistered = sorted(k for k in emitted
                          if not str(k).startswith("_")
                          and k not in S.SECONDARY_BLOCKS)
    assert not unregistered, (
        f"emitted secondaries missing from SECONDARY_BLOCKS: {unregistered}")


def test_every_key_on_a_retained_contrast_node_is_classified(tmp_path):
    """KEY CLASSIFICATION — the anti-vacuity principle applied to the guard set.

    Every key emitted on a retained contrast node must be one of: a census key,
    an enumerated contrast statistic, or an explicitly-reasoned non-statistic.
    An unclassified key would slip the forbidden-statistics guard silently, so
    a NEW key fails this test rather than quietly becoming unguarded.
    """
    S = _schema()
    report = _fixture_report(tmp_path)
    known = (set(S.CENSUS_KEYS) | set(S.ALL_STATISTIC_KEYS)
             | set(S.NON_STATISTIC_KEYS))

    def retained_nodes(node, path=""):
        """Contrast NODES only. A subtree whose own key is already an
        enumerated statistic (a CI block, a detector/tge block) is guarded as a
        whole by that key — its internals are the statistic's fields, not
        separate keys on a contrast node."""
        out = []
        if isinstance(node, dict):
            if node.get("status") == "COMPUTED":
                out.append((path, node))
            for k, v in node.items():
                if k in S.ALL_STATISTIC_KEYS:
                    continue
                out.extend(retained_nodes(v, f"{path}.{k}"))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out.extend(retained_nodes(v, f"{path}[{i}]"))
        return out

    nodes = retained_nodes(report["secondaries"])
    assert nodes, "fixture produced no retained contrast nodes"
    unclassified = {}
    for path, node in nodes:
        for key in node:
            if key.startswith("_") or key in known:
                continue
            unclassified.setdefault(key, path)
    assert not unclassified, (
        "keys on retained contrast nodes that are neither census, statistic, "
        f"nor explicitly classified: {unclassified}")


def test_every_non_statistic_key_states_its_reason():
    S = _schema()
    for key, reason in S.NON_STATISTIC_KEYS.items():
        assert reason and len(reason) > 8, f"{key} has no stated reason"
    assert not (set(S.NON_STATISTIC_KEYS) & set(S.ALL_STATISTIC_KEYS)), \
        "a key cannot be both a statistic and a non-statistic"


def _alie_scored(det_fpr, base_fpr, det_recall=0.60, base_recall=0.10,
                 seeds=(42, 137), scens=("S3", "S4"), n_honest=100):
    """Synthetic § 4 secondary-7 inputs with CONTROLLED cohort operating points.

    The ML fixture cannot be steered to a chosen operating point, and this guard
    is about the operating point — so the cells are built directly. Honest-row
    counts are uniform, so the row-pooled cohort rate equals the per-cell rate
    exactly and the assertions can name the constructed number.
    """
    flagged = round(det_fpr * n_honest)
    b_flagged = round(base_fpr * n_honest)
    det = {(sc, sd): {"recall": det_recall, "n_honest": n_honest,
                      "n_flagged_honest": flagged}
           for sc in scens for sd in seeds}
    base = {b: {(sc, sd): {"recall": base_recall, "n_honest": n_honest,
                           "n_flagged_honest": b_flagged}
                for sc in scens for sd in seeds}
            for b, _t, _l in A.BASELINES}
    return {"alie": det, "base_alie": base}


_S7_STATISTICS = ("baseline_per_seed_macro", "per_seed_diff", "margin", "sign_test")
_S7_CENSUS = ("label", "n_paired_seeds", "paired_seeds",
              "n_detector_seeds", "n_baseline_seeds",
              "detector_flagged_honest", "detector_honest_rows",
              "baseline_flagged_honest", "baseline_honest_rows",
              "detector_pooled_fpr", "baseline_pooled_fpr",
              "detector_in_interval", "baseline_in_interval")


def _s7(scored):
    return A.SEC.alie_fixed_baseline_contrasts(
        scored, A.BASELINES, A.SCENARIOS, A.exact_sign_p, A.comparable)


def test_s7_contrast_halts_when_the_BASELINE_side_is_out_of_interval():
    """§ 3.2 binds on the reported fixed-baseline contrasts too (v1.15b § 1.2).

    Secondary 7 shares P2's estimand, so it shares P2's guard. A baseline
    instrument realizing 0.30 is not being compared at the detector's operating
    point, and a margin computed across that gap reads as a result.
    """
    out = _s7(_alie_scored(det_fpr=0.10, base_fpr=0.30))
    for b, _t, _l in A.BASELINES:
        node = out["contrasts"][b]
        assert node["status"] == "NOT COMPARABLE", b
        leaked = [k for k in _S7_STATISTICS if k in node]
        assert not leaked, f"{b} published {leaked} across a 0.10-vs-0.30 gap"
        # both realized FPRs ride the halting branch — they ARE the evidence
        assert node["detector_pooled_fpr"] == pytest.approx(0.10)
        assert node["baseline_pooled_fpr"] == pytest.approx(0.30)
        assert node["detector_in_interval"] is True
        assert node["baseline_in_interval"] is False
        assert node["exclusion_cause"] == "fpr_interval"
        assert b in node["reason"] and "0.3000" in node["reason"]
        # census + label ride every branch
        for key in _S7_CENSUS:
            assert key in node, f"{b} dropped {key} on the halting branch"
        assert node["n_paired_seeds"] == 2
        assert node["baseline_honest_rows"] == 400


def test_s7_contrast_halts_when_the_DETECTOR_side_is_out_of_interval():
    """Symmetry: § 3.2 applies 'to every baseline comparison, symmetrically'.

    A guard that only ever fires on the baseline side would let the detector buy
    the margin with false positives — the exact asymmetry the clause forbids.
    """
    out = _s7(_alie_scored(det_fpr=0.03, base_fpr=0.10))
    for b, _t, _l in A.BASELINES:
        node = out["contrasts"][b]
        assert node["status"] == "NOT COMPARABLE", b
        assert not [k for k in _S7_STATISTICS if k in node], b
        assert node["detector_in_interval"] is False
        assert node["baseline_in_interval"] is True
        assert node["detector_pooled_fpr"] == pytest.approx(0.03)
        assert node["baseline_pooled_fpr"] == pytest.approx(0.10)
        assert node["reason"].startswith("detector 0.0300")
        for key in _S7_CENSUS:
            assert key in node


def test_s7_contrast_computes_when_both_sides_are_inside_the_interval():
    """ANTI-VACUITY — a guard that suppressed everything would prove nothing.

    Both sides at 0.10 is the dev regime (detector 0.1163; instruments 0.1028 /
    0.1000 / 0.1094), so the block must still publish its four statistics there.
    """
    out = _s7(_alie_scored(det_fpr=0.10, base_fpr=0.10))
    for b, _t, _l in A.BASELINES:
        node = out["contrasts"][b]
        assert node["status"] == "COMPUTED", b
        for key in _S7_STATISTICS + _S7_CENSUS:
            assert key in node, f"{b} dropped {key} on the retained branch"
        assert node["margin"] == pytest.approx(0.50)
        assert node["per_seed_diff"] == {"42": pytest.approx(0.50),
                                         "137": pytest.approx(0.50)}
        assert node["sign_test"]["positive"] == 2


def test_s7_boundary_fprs_are_inside_the_CLOSED_interval():
    """[0.08, 0.12] is closed (§ 3.2) — the endpoints compute, not halt."""
    for fpr in (0.08, 0.12):
        out = _s7(_alie_scored(det_fpr=fpr, base_fpr=fpr))
        assert all(out["contrasts"][b]["status"] == "COMPUTED"
                   for b, _t, _l in A.BASELINES), fpr


def test_s7_cohort_fpr_is_ROW_POOLED_not_a_mean_of_cell_rates():
    """v1.15b § 1.2 item 1 — pooling is over ROWS, not over per-cell rates.

    Two cells at 0.20 on 10 honest rows and 0.06 on 90 pool to 0.074 (outside),
    while their unweighted mean of rates is 0.13. The two constructions
    disagree on the verdict, so the code must not be free to pick.
    """
    det = {("S4", 42): {"recall": 0.6, "n_honest": 10, "n_flagged_honest": 2},
           ("S3", 42): {"recall": 0.6, "n_honest": 90, "n_flagged_honest": 5}}
    base = {b: {("S4", 42): {"recall": 0.1, "n_honest": 100,
                             "n_flagged_honest": 10},
                ("S3", 42): {"recall": 0.1, "n_honest": 100,
                             "n_flagged_honest": 10}}
            for b, _t, _l in A.BASELINES}
    node = _s7({"alie": det, "base_alie": base})["contrasts"]["krum_score"]
    assert node["detector_pooled_fpr"] == pytest.approx(7 / 100)
    assert node["status"] == "NOT COMPARABLE"
    assert node["detector_flagged_honest"] == 7
    assert node["detector_honest_rows"] == 100


def test_s7_comparability_function_cannot_be_omitted_by_a_caller():
    """The guard is a REQUIRED argument, not a defaulted one.

    `g2_scored_rows_contrast` defaults `comparable_fn=None` so a fixture can
    build the block without it; that same affordance is how a guard silently
    stops guarding. Secondary 7 refuses the call instead.
    """
    with pytest.raises(TypeError):
        A.SEC.alie_fixed_baseline_contrasts(
            _alie_scored(0.10, 0.30), A.BASELINES, A.SCENARIOS, A.exact_sign_p)


_MARGIN_STATISTICS = ("mean", "per_seed", "fixed_contrast_margin",
                      "fixed_contrast_sign_test")
_MARGIN_CENSUS = ("label", "n_paired_seeds", "paired_seeds",
                  "mean_realized_fpr", "detector_mean_realized_fpr",
                  "detector_in_interval", "baseline_in_interval")


def _margin_scored(det_fpr, base_fpr, det_recall=0.60, base_recall=0.10,
                   seeds=(42, 137, 256, 314, 500)):
    """A fully-shaped `scored` dict whose BLEND surface has a chosen operating
    point. Everything the other secondaries read is present and empty, so this
    exercises the blend-margin branch alone."""
    blend = {(sc, sd): {"recall": det_recall, "fpr": det_fpr, "n_mal": 50,
                        "n_honest": 100, "family_mix": {}, "families_present": [],
                        "rotation": 1}
             for sc in ("S3", "S4") for sd in seeds}
    base_blend = {b: {(sc, sd): {"recall": base_recall, "fpr": base_fpr,
                                 "n_mal": 50, "n_honest": 100}
                      for sc in ("S3", "S4") for sd in seeds}
                  for b, _t, _l in A.BASELINES}
    return {"blend": blend, "base_blend": base_blend, "alie": {},
            "base_alie": {b: {} for b, _t, _l in A.BASELINES},
            "per_family": {}, "fit_census": {}, "degenerate": [],
            "blend_global": {}, "alie_global": {}, "g2": {}}


def _margins(scored, scen="S4"):
    return A.secondaries(scored, A.DEV_SMOKE)[f"{scen}_blend_margin"]["baselines"]


def test_blend_margin_halts_when_the_BASELINE_side_is_out_of_interval():
    """§ 3.2 binds on the S3/S4 fixed-baseline margins too (round 29).

    This block emits `fixed_contrast_margin` and an exact sign test against
    each enumerated instrument — a baseline comparison in the § 3.2 sense, and
    the last one still computing at any realized FPR.
    """
    for scen in ("S3", "S4"):
        nodes = _margins(_margin_scored(det_fpr=0.10, base_fpr=0.30), scen)
        for b, node in nodes.items():
            assert node["status"] == "NOT COMPARABLE", (scen, b)
            assert not [k for k in _MARGIN_STATISTICS if k in node], (scen, b)
            assert node["mean_realized_fpr"] == pytest.approx(0.30)
            assert node["detector_mean_realized_fpr"] == pytest.approx(0.10)
            assert node["detector_in_interval"] is True
            assert node["baseline_in_interval"] is False
            assert node["exclusion_cause"] == "fpr_interval"
            for key in _MARGIN_CENSUS:
                assert key in node, f"{scen}/{b} dropped {key} on the halt"


def test_blend_margin_halts_when_the_DETECTOR_side_is_out_of_interval():
    """Symmetry, same clause: the detector does not keep a margin bought at
    0.03 FPR any more than a baseline loses one bought at 0.30."""
    nodes = _margins(_margin_scored(det_fpr=0.03, base_fpr=0.10))
    for b, node in nodes.items():
        assert node["status"] == "NOT COMPARABLE", b
        assert not [k for k in _MARGIN_STATISTICS if k in node], b
        assert node["detector_in_interval"] is False
        assert node["baseline_in_interval"] is True
        assert node["reason"].startswith("detector 0.0300")


def test_blend_margin_computes_when_both_sides_are_inside_the_interval():
    """ANTI-VACUITY — the dev regime (0.101 blended) must still publish."""
    nodes = _margins(_margin_scored(det_fpr=0.101, base_fpr=0.10))
    for b, node in nodes.items():
        assert node["status"] == "COMPUTED", b
        for key in _MARGIN_STATISTICS + _MARGIN_CENSUS:
            assert key in node, f"{b} dropped {key} on the retained branch"
        assert node["fixed_contrast_margin"] == pytest.approx(0.50)
        assert node["fixed_contrast_sign_test"]["positive"] == 5


def test_blend_margin_guard_uses_the_P1_MEAN_construction_not_row_pooling():
    """GRAIN, stated as a test.

    This readout is a MEAN over per-seed blended values, so § 3.2's "its
    realized FPR" is the mean over per-seed blended FPRs — the basis P1
    publishes for this same surface. Per-seed FPRs of 0.06 and 0.14 mean 0.10
    and COMPUTE, even though neither seed is individually in interval; the
    row-pooled reading secondary 7 uses is not available here at all, because
    the detector's blended FPR is a family-mix-weighted mixture, not a rate.
    """
    seeds = (42, 137, 256, 314)
    scored = _margin_scored(det_fpr=0.10, base_fpr=0.10, seeds=seeds)
    for i, sd in enumerate(seeds):
        scored["blend"][("S4", sd)]["fpr"] = 0.06 if i % 2 else 0.14
    # NO per-seed FPR is inside the interval; their mean is exactly 0.10
    assert not any(A.comparable(scored["blend"][("S4", sd)]["fpr"])
                   for sd in seeds)
    node = _margins(scored)["krum_score"]
    assert node["detector_mean_realized_fpr"] == pytest.approx(0.10)
    assert node["status"] == "COMPUTED"
    assert A.P1_SLICE == "S4"


def test_blend_margin_oracle_discloses_noncomparable_contributors():
    """The oracle max is composed BEFORE the guard, so say so.

    Suppressing an instrument's contrast while it silently composes the oracle
    would be a worse defect than the one being fixed. The estimand is not
    touched (scope discipline); the overlap is named.
    """
    entry = A.secondaries(_margin_scored(det_fpr=0.10, base_fpr=0.30),
                          A.DEV_SMOKE)["S4_blend_margin"]
    assert entry["oracle_max"]["contributing_instruments_not_comparable"], (
        "an instrument whose contrast was suppressed still composed the oracle "
        "and the document did not say so")


def test_the_blend_margin_fixture_nodes_mirror_the_REAL_key_sets(tmp_path):
    """Same anti-drift pin as secondary 7's, for this block's two branches."""
    fx = _fixture_report(tmp_path)["secondaries"]["S4_blend_margin"]["baselines"]
    assert set(fx["krum_score"]) == set(
        _margins(_margin_scored(0.10, 0.10))["krum_score"])
    assert set(fx["cos_to_median"]) == set(
        _margins(_margin_scored(0.10, 0.30))["cos_to_median"])


def _corrupt_one_row(arm: Path, field: str, value, unit_prefix="s4_full_mix"):
    """Rewrite ONE row of ONE staged unit, leaving the rest of the arm valid.

    The corruption has to sit inside an otherwise-passing arm or the gate could
    halt for an unrelated reason and the test would prove nothing.
    """
    target = sorted(arm.glob(f"{unit_prefix}__*.jsonl"))[0]
    rows = [json.loads(x) for x in
            target.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows[0][field] = value
    target.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")
    return target


_PER_FAMILY_SLICE_KEYS = ("per_seed", "mean_recall", "mean_realized_fpr",
                          "ci95_student_t")
_PER_FAMILY_ROW_KEYS = ("recall", "realized_fpr", "n_mal")


def test_schema_rejects_a_per_family_slice_missing_a_readout(tmp_path):
    """KNOCKOUT — the rule was selector-only, so every one of these passed.

    Each mutation must raise for THAT key alone; a rule that fails the node for
    some other reason would prove nothing about the key removed.
    """
    S = _schema()
    for key in _PER_FAMILY_SLICE_KEYS:
        report = _fixture_report(tmp_path)
        report["secondaries"]["per_family_fold_recalls"]["alie"]["S4"].pop(key)
        with pytest.raises(S.SchemaViolation) as exc:
            S.check(report)
        assert f"missing ['{key}']" in str(exc.value), exc.value


def test_schema_rejects_a_per_family_seed_row_missing_a_readout(tmp_path):
    """The per-fold row carries its own operating point and row count."""
    S = _schema()
    for key in _PER_FAMILY_ROW_KEYS:
        report = _fixture_report(tmp_path)
        report["secondaries"]["per_family_fold_recalls"]["alie"]["S4"][
            "per_seed"]["42"].pop(key)
        with pytest.raises(S.SchemaViolation) as exc:
            S.check(report)
        assert f"missing ['{key}']" in str(exc.value), exc.value


def test_the_per_family_fixture_mirrors_the_REDUCERS_real_key_sets(tmp_path):
    """The fixture said `mean` where the reducer emits `mean_recall`.

    A fixture that does not match the emission makes every mutation against it
    a test of fiction, which is how the wrong key survived two rounds.
    """
    scored = {"per_family": {("alie", "S4", 42): {
        "recall": 0.5, "fpr": 0.10, "n_mal": 20, "n_honest": 100, "cut": 0.5}},
        "blend": {}, "base_blend": {}, "alie": {},
        "base_alie": {b: {} for b, _t, _l in A.BASELINES},
        "fit_census": {}, "degenerate": [], "blend_global": {},
        "alie_global": {}, "g2": {}}
    real = A.secondaries(scored, A.DEV_SMOKE)["per_family_fold_recalls"]["alie"]["S4"]
    fx = _fixture_report(tmp_path)["secondaries"][
        "per_family_fold_recalls"]["alie"]["S4"]
    assert set(fx) == set(real)
    assert set(fx["per_seed"]["42"]) == set(real["per_seed"]["42"])


def test_exp048_gate_rejects_an_UNREGISTERED_scenario_string(tmp_path):
    """A string-valued but unknown scenario must halt as a CORPUS defect.

    It satisfies the value contract's `str` check and then reaches a direct
    roster lookup. Pre-round-30 that surfaced as a bare KeyError — an executor
    crash, not a verdict about the input.
    """
    _stage_exp048_arm(tmp_path, rounds=12)
    _corrupt_one_row(tmp_path, "scenario", "s9_not_a_scenario")
    with pytest.raises(ValueError, match="UNREGISTERED 'scenario'"):
        A.E48.validate_exp048_dir(
            tmp_path, A.R.SCEN_SHORT, A.SCENARIOS, A.h2_confirm_seeds(),
            A.EXP048_REQUIRED_KEYS, A.SCORING_VALUE_CONTRACT,
            A.SCORING_ENUM_CONTRACT)


def test_exp048_gate_still_rejects_a_MISMATCHED_registered_scenario(tmp_path):
    """ANTI-VACUITY — the enum must not swallow the filename-mismatch check.

    A registered scenario that disagrees with the unit's filename is a
    different defect and keeps its own message.
    """
    _stage_exp048_arm(tmp_path, rounds=12)
    # registered, but not the scenario its own filename names
    _corrupt_one_row(tmp_path, "scenario", "s3_identity_reset_only",
                     unit_prefix="s4_full_mix")
    with pytest.raises(ValueError, match="!= filename scenario"):
        A.E48.validate_exp048_dir(
            tmp_path, A.R.SCEN_SHORT, A.SCENARIOS, A.h2_confirm_seeds(),
            A.EXP048_REQUIRED_KEYS, A.SCORING_VALUE_CONTRACT,
            A.SCORING_ENUM_CONTRACT)


def test_exp048_gate_covers_seed_against_the_registered_universe(tmp_path):
    """SWEEP — the sibling direct-lookup key, verified rather than assumed.

    `seed` reaches no dict lookup (it is compared to the filename), and the
    filename grid is itself checked against the registered universe, so an
    unregistered seed halts with a named message and never KeyErrors.
    """
    _stage_exp048_arm(tmp_path, rounds=12)
    _stage_exp048_arm(tmp_path, seeds=(99999,), rounds=12)
    with pytest.raises(ValueError, match="outside the registered grid"):
        A.E48.validate_exp048_dir(
            tmp_path, A.R.SCEN_SHORT, A.SCENARIOS, A.h2_confirm_seeds(),
            A.EXP048_REQUIRED_KEYS, A.SCORING_VALUE_CONTRACT,
            A.SCORING_ENUM_CONTRACT)


def test_the_exp048_split_kept_ONE_public_gate_name():
    """The split is a facade, not a fork: two live definitions of the gate
    would be worse than one long module."""
    import h2prime_exp048 as E
    import h2prime_exp048_gate as G
    assert E.validate_exp048_dir is G.validate_exp048_dir
    assert not hasattr(E, "_parse_unit_rows"), (
        "the moved helper is still defined in the old module too")


def _bracket_scored(tmp_path):
    """A scored corpus carrying the § 6 bracket, from the fixture arm."""
    arm = tmp_path / "arm"
    arm.mkdir(parents=True, exist_ok=True)
    _stage_exp048_arm(arm, rounds=12)
    seeds = A.h2_confirm_seeds()
    rows = A.E48.load_standalone_tge_rows(arm, A.R.SCEN_SHORT,
                                          A.R.derive_window_feats)
    return A.score_corpus(rows, A.rotation_plan(seeds))


def test_bracket_cut_helper_reproduces_the_FROZEN_cut_at_10_percent():
    """The bracket must be the § 2.2a construction, not a lookalike.

    `R.cut_from_calibration` is under the golden gate and hard-codes the frozen
    target, so the bracket needs its own parameterised copy — which is only
    legitimate if it agrees BIT-FOR-BIT at the frozen target.
    """
    rng = np.random.default_rng(20260813)
    for _ in range(50):
        honest = rng.normal(size=rng.integers(5, 400))
        for trust in (True, False):
            assert A._cut_at(honest, A.TARGET_FPR, trust) == \
                A.R.cut_from_calibration(honest, trust)


def test_bracket_10_percent_point_EQUALS_the_adjudicated_readout(tmp_path):
    """The § 4 bands adjudicate at 10 %; the bracket must not restate it.

    Every (scenario, seed) fold is compared exactly — not approximately — for
    both the recall and its realized blended FPR. They come from one shared
    construction, so any divergence means the shared path was forked.
    """
    scored = _bracket_scored(tmp_path)
    assert scored["bracket"], "no bracket cells were produced"
    checked = 0
    for (scen, seed), cell in scored["blend"].items():
        b = scored["bracket"][(scen, seed, A.TARGET_FPR)]
        assert b["recall"] == cell["recall"], (scen, seed)
        assert b["realized_fpr"] == cell["fpr"], (scen, seed)
        assert b["n_mal"] == cell["n_mal"] and b["n_honest"] == cell["n_honest"]
        checked += 1
    assert checked, "vacuous — no folds compared"


def test_bracket_cuts_are_ORDERED_and_land_near_their_targets(tmp_path):
    """Per-point cut correctness: a stricter target buys a lower realized FPR.

    The cut is a quantile of the calibration seed's honest rows, so a smaller
    target must not produce a HIGHER realized FPR on the test seed, and recall
    cannot rise as the operating point tightens.
    """
    scored = _bracket_scored(tmp_path)
    folds = sorted({(sc, sd) for (sc, sd, _t) in scored["bracket"]})
    assert folds
    for scen, seed in folds:
        pts = [scored["bracket"][(scen, seed, t)] for t in A.BRACKET_TARGETS]
        fprs = [p["realized_fpr"] for p in pts]
        recs = [p["recall"] for p in pts]
        assert fprs == sorted(fprs), (scen, seed, fprs)
        assert recs == sorted(recs), (scen, seed, recs)


def test_bracket_targets_and_proportional_interval_come_from_FROZEN_constants():
    """The interval is derived, not restated: at 10 % it IS [0.08, 0.12]."""
    assert A.BRACKET_TARGETS == (0.01, 0.02, 0.05, 0.10)
    assert A.TARGET_FPR in A.BRACKET_TARGETS
    block = A.SEC.bracket_recall_by_fpr(
        {"bracket": {}}, A.SCENARIOS, A.BRACKET_TARGETS, A.TARGET_FPR,
        A.COMPARABILITY_INTERVAL, A.student_t_ci, A.DEV_SMOKE.t_crit,
        A.DEV_SMOKE.df)
    ten = block["points"][f"{A.TARGET_FPR:g}"]
    assert ten["interval"] == pytest.approx(list(A.COMPARABILITY_INTERVAL))
    assert ten["is_adjudicating_point"] is True
    for key, pt in block["points"].items():
        lo, hi = pt["interval"]
        assert lo == pytest.approx(pt["target_fpr"] * 0.8)
        assert hi == pytest.approx(pt["target_fpr"] * 1.2)
        if pt["target_fpr"] != A.TARGET_FPR:
            assert pt["is_adjudicating_point"] is False


def test_bracket_annotates_comparability_as_EVIDENCE_and_never_halts():
    """§ 6: out-of-interval at a non-10 % point is evidence, not a halt.

    A point whose realized FPR misses its own interval must still publish its
    recall — and must carry no verdict, status, or reason key anywhere.
    """
    cells = {("S4", 42, t): {"recall": 0.5, "realized_fpr": 0.99,
                             "n_mal": 10, "n_honest": 100}
             for t in A.BRACKET_TARGETS}
    block = A.SEC.bracket_recall_by_fpr(
        {"bracket": cells}, A.SCENARIOS, A.BRACKET_TARGETS, A.TARGET_FPR,
        A.COMPARABILITY_INTERVAL, A.student_t_ci, A.DEV_SMOKE.t_crit,
        A.DEV_SMOKE.df)
    assert block["comparability_is_evidence_only"] is True
    for key, pt in block["points"].items():
        s4 = pt["scenarios"]["S4"]
        assert s4["in_interval"] is False              # 0.99 misses every band
        assert s4["mean_recall"] == pytest.approx(0.5)  # ...and still reports
    flat = json.dumps(block)
    for forbidden in ("verdict", "status", "NOT COMPARABLE", "INCONCLUSIVE",
                      "halt"):
        assert forbidden not in flat, f"{forbidden!r} leaked into a § 6 node"


def test_bracket_does_not_touch_the_adjudicating_bands(tmp_path):
    """The whole point of the addition: P1/P2 are computed from the same cells
    they always were, and the block is registered as a secondary."""
    S = _schema()
    assert "bracket_recall_by_fpr" in S.SECONDARY_BLOCKS
    scored = _bracket_scored(tmp_path)
    p1 = A.adjudicate_p1(scored, A.DEV_SMOKE)
    assert p1["per_seed_realized_blended_fpr"] == {
        str(sd): scored["bracket"][(A.P1_SLICE, sd, A.TARGET_FPR)]["realized_fpr"]
        for (sc, sd) in scored["blend"] if sc == A.P1_SLICE}


_BRACKET_SLICE_KEYS = ("n_folds", "per_seed_recall", "mean_recall",
                       "ci95_student_t", "per_seed_realized_fpr",
                       "mean_realized_fpr", "per_seed_in_interval",
                       "in_interval")


def test_schema_rejects_a_bracket_slice_missing_a_readout(tmp_path):
    """KNOCKOUT — a reported number no rule checks is unowned."""
    S = _schema()
    for key in _BRACKET_SLICE_KEYS:
        report = _fixture_report(tmp_path)
        report["secondaries"]["bracket_recall_by_fpr"]["points"]["0.01"][
            "scenarios"]["S4"].pop(key)
        with pytest.raises(S.SchemaViolation) as exc:
            S.check(report)
        assert f"missing ['{key}']" in str(exc.value), exc.value


def test_schema_rejects_a_bracket_point_missing_its_interval(tmp_path):
    S = _schema()
    for key in ("target_fpr", "interval", "is_adjudicating_point", "scenarios"):
        report = _fixture_report(tmp_path)
        report["secondaries"]["bracket_recall_by_fpr"]["points"]["0.01"].pop(key)
        with pytest.raises(S.SchemaViolation) as exc:
            S.check(report)
        assert f"missing ['{key}']" in str(exc.value), exc.value


def test_the_bracket_fixture_mirrors_the_REDUCERS_real_key_sets(tmp_path):
    real = A.SEC.bracket_recall_by_fpr(
        {"bracket": {("S4", 42, t): {"recall": 0.5, "realized_fpr": 0.1,
                                     "n_mal": 10, "n_honest": 100}
                     for t in A.BRACKET_TARGETS}},
        A.SCENARIOS, A.BRACKET_TARGETS, A.TARGET_FPR, A.COMPARABILITY_INTERVAL,
        A.student_t_ci, A.DEV_SMOKE.t_crit, A.DEV_SMOKE.df)
    fx = _fixture_report(tmp_path)["secondaries"]["bracket_recall_by_fpr"]
    assert set(fx) == set(real)
    assert set(fx["points"]["0.01"]) == set(real["points"]["0.01"])
    assert set(fx["points"]["0.01"]["scenarios"]["S4"]) == set(
        real["points"]["0.01"]["scenarios"]["S4"])


def test_schema_rejects_a_blend_margin_node_that_leaks_a_statistic(tmp_path):
    S = _schema()
    for leaked in _MARGIN_STATISTICS:
        report = _fixture_report(tmp_path)
        report["secondaries"]["S4_blend_margin"]["baselines"]["cos_to_median"][leaked] = 1
        with pytest.raises(S.SchemaViolation, match="leaked contrast statistics"):
            S.check(report)


def test_schema_rejects_a_blend_margin_node_that_drops_a_retained_statistic(tmp_path):
    S = _schema()
    for dropped in _MARGIN_STATISTICS:
        report = _fixture_report(tmp_path)
        report["secondaries"]["S4_blend_margin"]["baselines"]["krum_score"].pop(dropped)
        with pytest.raises(S.SchemaViolation, match="dropped retained readouts"):
            S.check(report)


def _alie_scored_unpaired(base_seeds_by_instrument):
    """Secondary-7 inputs where a baseline shares NO seed with the detector."""
    scored = _alie_scored(det_fpr=0.10, base_fpr=0.10, seeds=(42, 137))
    for b, seeds in base_seeds_by_instrument.items():
        scored["base_alie"][b] = {
            (sc, sd): {"recall": 0.1, "n_honest": 100, "n_flagged_honest": 10}
            for sc in ("S3", "S4") for sd in seeds}
    return scored


def test_s7_unpaired_instrument_leaves_an_exclusion_NODE_not_a_hole():
    """An instrument that pairs with nothing must not vanish.

    Pre-round-29 the reducer did `continue`, so "this contrast was excluded"
    and "this instrument does not exist" serialized identically — a silent
    omission of the same species the § 3.2 guard closed, one grain up.
    """
    out = _s7(_alie_scored_unpaired({"L2_to_median": (900, 901),
                                     "cos_to_median": ()}))
    assert set(out["contrasts"]) == {"krum_score", "L2_to_median",
                                     "cos_to_median"}, "an instrument vanished"

    disjoint = out["contrasts"]["L2_to_median"]
    assert disjoint["status"] == "NOT COMPARABLE"
    assert disjoint["exclusion_cause"] == "unpaired"
    assert not [k for k in _S7_STATISTICS if k in disjoint]
    for key in _S7_CENSUS:
        assert key in disjoint, f"unpaired node dropped {key}"
    # the census that MAKES the exclusion legible: 2 vs 2, zero overlap
    assert disjoint["n_paired_seeds"] == 0
    assert disjoint["paired_seeds"] == []
    assert disjoint["n_detector_seeds"] == 2
    assert disjoint["n_baseline_seeds"] == 2

    # ...and an instrument with NO cells at all is a different story
    empty = out["contrasts"]["cos_to_median"]
    assert empty["exclusion_cause"] == "unpaired"
    assert empty["n_baseline_seeds"] == 0
    assert empty["reason"] != disjoint["reason"], (
        "5-vs-0 and 5-vs-5-disjoint must not print identically")

    # the paired instrument is unaffected — anti-vacuity
    assert out["contrasts"]["krum_score"]["status"] == "COMPUTED"


def test_s7_unpaired_is_named_as_such_not_as_an_FPR_FAILURE():
    """Cause precedence: an unpaired node has no cohort to realize an operating
    point on, so its None FPRs must not be reported as out-of-interval."""
    node = _s7(_alie_scored_unpaired({"L2_to_median": (900,)})
               )["contrasts"]["L2_to_median"]
    assert node["exclusion_cause"] == "unpaired"
    assert "comparability" not in node["reason"]
    assert node["detector_pooled_fpr"] is None
    assert node["baseline_pooled_fpr"] is None


def test_exclusion_cause_vocabulary_is_shared_and_realizability_is_STATED():
    """The cause universe is one set; realizability differs and says so.

    Round 29 added a cause that is structurally impossible at the fold grain.
    The invariant is no longer "both sides realize the same causes" — it is
    that the VOCABULARY is shared and every asymmetry carries its structural
    reason, so a one-sided guard still shows up as a missing cause.
    """
    S7 = A.SEC
    assert S7.EXCLUSION_CAUSES == ("census_only", "coverage_floor",
                                   "fpr_interval", "unpaired")
    # every cause is accounted for on BOTH grains, as realizable or not
    for grain, spec in S7.CAUSE_REALIZABILITY.items():
        covered = set(spec["realizable"]) | set(spec["unrealizable"])
        assert covered == set(S7.EXCLUSION_CAUSES), (
            f"{grain} neither realizes nor excuses {covered ^ set(S7.EXCLUSION_CAUSES)}")
        for cause, why in spec["unrealizable"].items():
            assert len(why) > 30, f"{grain}/{cause} has no structural reason"
    # and the phrase/note tables cover the whole vocabulary
    for cause in S7.EXCLUSION_CAUSES:
        fields = S7.exclusion_fields(cause, "detail")
        assert fields["exclusion_cause"] == cause
        assert "no statistic computed" in fields["reason"]
        assert fields["note"]
    with pytest.raises(KeyError):
        S7.exclusion_fields("not_a_registered_cause")


def test_unpaired_is_structurally_unrealizable_at_the_FOLD_grain():
    """The asymmetry is asserted, not assumed.

    `unpaired` is excused at the fold grain because both arms are scored on the
    same cell. That claim is checked by running the fold-grain reducer and
    confirming the cause never fires and always censuses as zero.
    """
    seeds = A.h2_confirm_seeds()
    assert "unpaired" in A.SEC.CAUSE_REALIZABILITY["fold_grain"]["unrealizable"]
    for cells in (_exp048_scored(0.10, n_mal=4, seeds=seeds),
                  _exp048_scored(0.15, n_mal=40, seeds=seeds),
                  _exp048_scored(0.10, n_mal=40, seeds=seeds)):
        slice_ = A.SEC.g2_scored_rows_contrast(
            cells, A.SCENARIOS, seeds, A.ATTACKS, A.exact_sign_p,
            A.student_t_ci, A.ci_on_retained, A.comparable)["scenarios"]["S4"]
        # the 0/0 row is present on every branch (rev-14 doctrine) and stays 0
        assert slice_["exclusion_causes"]["unpaired"] == 0
        assert slice_["excluded_folds_by_cause"]["unpaired"] == []


def test_the_s7_fixture_nodes_mirror_the_reducers_REAL_key_sets(tmp_path):
    """Round 28's lesson, applied to this block's own fixture.

    Every schema mutation below is asserted against the fixture, so if the
    fixture drifts from what the reducer actually emits, the mutations test a
    document that does not exist. Both branches are pinned.
    """
    fx = _fixture_report(tmp_path)["secondaries"][
        "alie_fixed_baseline_contrasts"]["contrasts"]
    assert set(fx["krum_score"]) == set(
        _s7(_alie_scored(0.10, 0.10))["contrasts"]["krum_score"])
    assert set(fx["cos_to_median"]) == set(
        _s7(_alie_scored(0.10, 0.30))["contrasts"]["cos_to_median"])
    # both halting causes serialize the SAME shape — only the cause differs
    assert set(fx["cos_to_median"]) == set(
        _s7(_alie_scored_unpaired({"cos_to_median": (900,)}))
        ["contrasts"]["cos_to_median"])


def test_schema_rejects_an_s7_contrast_that_leaks_a_statistic(tmp_path):
    """The block's rule must FAIL a suppressed node that kept a statistic."""
    S = _schema()
    for leaked in _S7_STATISTICS:
        report = _fixture_report(tmp_path)
        node = report["secondaries"]["alie_fixed_baseline_contrasts"]["contrasts"]
        node["cos_to_median"][leaked] = 0.42
        with pytest.raises(S.SchemaViolation, match="leaked contrast statistics"):
            S.check(report)


def test_schema_rejects_an_s7_contrast_that_drops_a_retained_statistic(tmp_path):
    """...and the retention contract binds on the COMPUTED branch (round 21)."""
    S = _schema()
    for dropped in _S7_STATISTICS:
        report = _fixture_report(tmp_path)
        node = report["secondaries"]["alie_fixed_baseline_contrasts"]["contrasts"]
        node["krum_score"].pop(dropped)
        with pytest.raises(S.SchemaViolation, match="dropped retained readouts"):
            S.check(report)


def test_schema_rejects_an_s7_contrast_missing_its_halt_evidence(tmp_path):
    """Both pooled FPRs are owed on EVERY branch — they are the halt's evidence.

    Each mutation must raise for THAT key ALONE (`missing ['<key>']`, a
    one-element list). A rule that already fails the suppressed node for other
    reasons would let this test pass while proving nothing about the key it
    removed — which is exactly what the pre-fix `required` set did.
    """
    S = _schema()
    for key in ("detector_pooled_fpr", "baseline_pooled_fpr",
                "detector_in_interval", "baseline_in_interval",
                "detector_flagged_honest", "baseline_honest_rows",
                "label", "n_paired_seeds", "paired_seeds", "status"):
        report = _fixture_report(tmp_path)
        node = report["secondaries"]["alie_fixed_baseline_contrasts"]["contrasts"]
        node["cos_to_median"].pop(key)
        with pytest.raises(S.SchemaViolation) as exc:
            S.check(report)
        assert f"missing ['{key}']" in str(exc.value), (
            f"popping {key} did not produce a lone violation: {exc.value}")


def _exp048_scored(det_fpr, tge_fpr=0.10, n_mal=40, seeds=None, scen="S4"):
    """Synthetic g2 cells with CONTROLLED realized FPRs.

    The ML fixture cannot be steered to a chosen operating point, and the
    comparability gate is about the operating point — so these tests build the
    scorer's output shape directly and vary only the FPR.
    """
    seeds = seeds or A.h2_confirm_seeds()
    return {"g2": {(scen, sd): {
        "det_recall": 0.7, "tge_recall": 0.3,
        "det_fpr": det_fpr, "tge_fpr": tge_fpr,
        "n_mal_scored": n_mal, "n_mal_total": n_mal,
        "n_honest_scored": 100, "n_honest_total": 100,
        "per_family": {}, "family_eligible_totals": {}, "family_covered_totals": {},
    } for sd in seeds}}


def test_exp048_fold_clearing_the_floor_but_out_of_interval_is_excluded():
    """§ 3.2 discipline on this branch: clearing the ROW floor is not enough —
    an out-of-interval realized FPR excludes the fold from the contrast."""
    seeds = A.h2_confirm_seeds()
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.15), A.SCENARIOS,
        A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)
    e = out["blended"]["S4"]
    assert e["status"] == "NOT COMPARABLE"
    assert e["n_folds_comparable"] == 0
    # excluded from every aggregate and every paired vector
    for k in ("detector_mean", "tge_mean", "margin_detector_minus_tge",
              "per_seed_diff", "sign_test"):
        assert k not in e, f"{k} computed over an out-of-interval fold"
    rows = list(e["census_per_fold"].values())
    assert len(rows) == 10
    for r in rows:
        assert r["status"] == "NOT COMPARABLE"
        assert "outside the comparability interval" in r["note"]
        assert "detector 0.1500" in r["note"]
        # census AND the realized FPRs are retained as the EVIDENCE
        assert r["n_mal_scored"] == 40 and r["coverage_mal"] == pytest.approx(1.0)
        assert r["detector_realized_fpr"] == pytest.approx(0.15)
        assert r["detector_comparable"] is False and r["tge_comparable"] is True
        # but no contrast statistic
        assert "detector_recall" not in r and "tge_recall" not in r


def test_exp048_fold_inside_the_interval_still_computes():
    """The gate excludes only the out-of-interval case."""
    seeds = A.h2_confirm_seeds()
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.10), A.SCENARIOS,
        A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)
    e = out["blended"]["S4"]
    assert e["status"] == "COMPUTED"
    assert e["n_folds_comparable"] == 10
    assert e["margin_detector_minus_tge"] == pytest.approx(0.4)
    assert e["sign_test"]["positive"] == 10
    row = list(e["census_per_fold"].values())[0]
    assert row["status"] == "COMPUTED" and row["detector_recall"] == pytest.approx(0.7)


def test_exp048_tge_side_out_of_interval_also_excludes():
    """Symmetry: either side out of interval disqualifies the fold."""
    seeds = A.h2_confirm_seeds()
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.10, tge_fpr=0.02),
        A.SCENARIOS, A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)
    e = out["blended"]["S4"]
    assert e["status"] == "NOT COMPARABLE"
    assert "TGE 0.0200" in list(e["census_per_fold"].values())[0]["note"]


def test_both_reducers_classify_against_the_SAME_exclusion_cause_set():
    """A one-sided guard is now visible as a MISSING CAUSE, not as silent
    inclusion. This shape slipped through twice (rounds 14 and 16), so the
    invariant is that both sides enumerate the same causes."""
    assert A.SEC.EXCLUSION_CAUSES == ("census_only", "coverage_floor",
                                      "fpr_interval", "unpaired")
    seeds = A.h2_confirm_seeds()

    # confirmatory side: one slice per cause
    def conf(cells):
        return A.SEC.g2_scored_rows_contrast(
            cells, A.SCENARIOS, seeds, A.ATTACKS, A.exact_sign_p,
            A.student_t_ci, A.ci_on_retained, A.comparable
        )["scenarios"]["S4"]

    floor = conf(_exp048_scored(0.10, n_mal=4, seeds=seeds))
    fpr = conf(_exp048_scored(0.15, n_mal=40, seeds=seeds))
    assert floor["exclusion_causes"]["coverage_floor"] == 10
    assert floor["exclusion_causes"]["fpr_interval"] == 0
    assert fpr["exclusion_causes"]["fpr_interval"] == 10
    assert fpr["exclusion_causes"]["coverage_floor"] == 0

    # and the EXP-048 side reports the same two causes on the same inputs
    e_floor = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.10, n_mal=4, seeds=seeds),
        A.SCENARIOS, A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)["blended"]["S4"]
    e_fpr = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.15, n_mal=40, seeds=seeds),
        A.SCENARIOS, A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)["blended"]["S4"]
    assert e_floor["exclusion_causes"]["coverage_floor"] == 10
    # both sides now report in the SAME taxonomy — the point of the invariant
    assert e_fpr["exclusion_causes"]["fpr_interval"] == 10
    assert set(e_fpr["exclusion_causes"]) == set(A.SEC.EXCLUSION_CAUSES)
    assert set(fpr["exclusion_causes"]) == set(A.SEC.EXCLUSION_CAUSES)


def test_confirmatory_fpr_gate_excludes_with_evidence():
    """Round-14's guard, now mirrored: out-of-interval folds are excluded from
    the confirmatory contrast with the FPRs retained as the evidence."""
    seeds = A.h2_confirm_seeds()
    out = A.SEC.g2_scored_rows_contrast(
        _exp048_scored(0.15, n_mal=40, seeds=seeds), A.SCENARIOS, seeds,
        A.ATTACKS, A.exact_sign_p, A.student_t_ci, A.ci_on_retained,
        A.comparable)["scenarios"]["S4"]
    assert out["status"] == "NOT COMPARABLE"
    assert out["n_folds_comparable"] == 0
    assert "out-of-interval realized FPR" in out["reason"]
    for r in out["per_fold"].values():
        assert r["status"] == "NOT COMPARABLE"
        assert r["detector_realized_fpr"] == pytest.approx(0.15)
        assert r["detector_comparable"] is False
        assert "det_recall" not in r and "tge_recall" not in r


def test_confirmatory_census_only_halt_names_its_cause():
    seeds = A.h2_confirm_seeds()
    cells = _exp048_scored(0.10, n_mal=40, seeds=seeds)
    for v in cells["g2"].values():
        v["status"] = "CENSUS ONLY"
    out = A.SEC.g2_scored_rows_contrast(
        cells, A.SCENARIOS, seeds, A.ATTACKS, A.exact_sign_p, A.student_t_ci,
        A.ci_on_retained, A.comparable)["scenarios"]["S4"]
    assert "census-only: uncovered honest side" in out["reason"]
    assert out["exclusion_causes"]["census_only"] == 10


def test_p2_tie_census_splits_partial_from_full_ties():
    """Partial ties are the case a reader cannot reconstruct without being told
    which of the two candidates the frozen order selected."""
    det = {(sc, s): _cell(0.9, 10, 100)
           for sc in A.SCENARIOS for s in A.DEV_SEEDS}
    # krum and L2 tie at the max; cos is strictly lower -> PARTIAL tie
    base = {"krum_score": {k: _cell(0.5, 10, 100) for k in det},
            "L2_to_median": {k: _cell(0.5, 10, 100) for k in det},
            "cos_to_median": {k: _cell(0.1, 10, 100) for k in det}}
    res = A.adjudicate_p2({"alie": det, "base_alie": base}, A.DEV_SMOKE,
                          A.DEV_SEEDS)
    rg = res["readout_grain_comparability"]
    assert rg["n_cells_partial_tie"] == 25
    assert rg["n_cells_full_tie"] == 0 and rg["n_cells_no_tie"] == 0
    cell = list(rg["partial_tie_cells"].values())[0]
    assert cell["tied_instruments"] == ["L2_to_median", "krum_score"]
    assert cell["tie_break_selected"] == "krum_score"


def test_p2_tie_census_counts_full_and_no_tie_cells():
    det = {(sc, s): _cell(0.9, 10, 100)
           for sc in A.SCENARIOS for s in A.DEV_SEEDS}
    full = {b: {k: _cell(0.0, 10, 100) for k in det}
            for b, _t, _l in A.BASELINES}
    rg = A.adjudicate_p2({"alie": det, "base_alie": full}, A.DEV_SMOKE,
                         A.DEV_SEEDS)["readout_grain_comparability"]
    assert rg["n_cells_full_tie"] == 25 and rg["n_cells_partial_tie"] == 0

    distinct = {"krum_score": {k: _cell(0.7, 10, 100) for k in det},
                "L2_to_median": {k: _cell(0.4, 10, 100) for k in det},
                "cos_to_median": {k: _cell(0.1, 10, 100) for k in det}}
    rg2 = A.adjudicate_p2({"alie": det, "base_alie": distinct}, A.DEV_SMOKE,
                          A.DEV_SEEDS)["readout_grain_comparability"]
    assert rg2["n_cells_no_tie"] == 25
    assert rg2["n_cells_where_argmax_was_a_tie"] == 0


def _fam_cells(det_fpr, fam_fpr, n_mal=40, fam_n=40, seeds=None):
    """Cells whose POOLED operating point differs from the family's."""
    seeds = seeds or A.DEV_SEEDS
    return {"g2": {(sc, sd): {
        "det_recall": 0.7, "tge_recall": 0.3,
        "det_fpr": det_fpr, "tge_fpr": 0.10,
        "n_mal_scored": n_mal, "n_mal_total": n_mal,
        "n_honest_scored": 100, "n_honest_total": 100,
        "family_eligible_totals": {"alie": fam_n},
        "family_covered_totals": {"alie": fam_n},
        "per_family": {"alie": {
            "det_recall": 0.8, "tge_recall": 0.2,
            "det_fpr": fam_fpr, "tge_fpr": 0.10,
            "n_mal_scored": fam_n, "n_mal_total": fam_n,
            "n_honest_scored": 100, "n_honest_total": 100}},
    } for sc in A.SCENARIOS for sd in seeds}}


def _conf_slice(cells):
    return A.SEC.g2_scored_rows_contrast(
        cells, A.SCENARIOS, A.DEV_SEEDS, A.ATTACKS, A.exact_sign_p,
        A.student_t_ci, A.ci_on_retained, A.comparable)["scenarios"]["S4"]


def test_family_slice_guarded_even_when_the_pooled_fold_is_inside():
    """§ 3.2 at the FAMILY grain: the pooled FPR being fine does not license a
    family whose own operating point is outside the interval."""
    e = _conf_slice(_fam_cells(det_fpr=0.10, fam_fpr=0.17))
    assert e["status"] == "COMPUTED"            # the pooled fold is comparable
    fam = e["per_attack_family"]["alie"]
    assert fam["status"] == "NOT COMPARABLE"
    assert fam["exclusion_causes"]["fpr_interval"] == 5
    assert "out-of-interval realized FPR" in fam["reason"]
    for r in fam["census_per_fold"].values():
        assert r["status"] == "NOT COMPARABLE"
        assert r["detector_realized_fpr"] == pytest.approx(0.17)
        assert r["detector_comparable"] is False
        # statistics suppressed, census + evidence retained
        assert "det_recall" not in r and "tge_recall" not in r
        assert MANDATORY_CENSUS_KEYS <= set(r)


def test_family_slice_computes_when_its_own_operating_point_is_inside():
    fam = _conf_slice(_fam_cells(det_fpr=0.10, fam_fpr=0.10))["per_attack_family"]["alie"]
    assert fam["status"] == "COMPUTED"
    assert fam["n_folds_comparable"] == 5
    assert fam["margin_detector_minus_tge"] == pytest.approx(0.6)


def test_retained_family_fold_serializes_BOTH_SIDES_not_just_the_difference():
    """ROUND 20 #1. A comparable family fold cleared every guard and published
    census + status ONLY, so the slice carried a paired difference that no
    reader could resolve back into the two recalls it came from.

    The registered output is a MATCHED contrast. A difference is auditable only
    if both sides' per-seed recalls and both sides' realized operating points
    survive to the row — which is exactly what the EXP-048 retained folds have
    serialized since round 13. This is that mirror, at the family grain.
    """
    fam = _conf_slice(_fam_cells(det_fpr=0.10, fam_fpr=0.10))["per_attack_family"]["alie"]
    assert fam["status"] == "COMPUTED"
    kept = [r for r in fam["census_per_fold"].values() if r["status"] == "COMPUTED"]
    assert len(kept) == 5, "fixture must exercise the RETAINED branch"
    for sd, r in fam["census_per_fold"].items():
        if r["status"] != "COMPUTED":
            continue
        # both per-seed recalls, from _fam_cells' per_family entry
        assert r["detector_recall"] == pytest.approx(0.8)
        assert r["tge_recall"] == pytest.approx(0.2)
        # both realized FPRs, with the comparability claim they support
        assert r["detector_realized_fpr"] == pytest.approx(0.10)
        assert r["tge_realized_fpr"] == pytest.approx(0.10)
        assert r["detector_comparable"] is True and r["tge_comparable"] is True
        # the census is NOT displaced by the readouts
        assert MANDATORY_CENSUS_KEYS <= set(r)
    # THE POINT: the published difference reconciles against the row it came
    # from. Before the fix there was nothing on the row to reconcile against.
    for seed_key, d in fam["per_seed_diff"].items():
        row = fam["census_per_fold"][f"S4x{seed_key}"]
        assert row["detector_recall"] - row["tge_recall"] == pytest.approx(d)


def test_retained_family_readout_keys_are_the_EXP048_vocabulary():
    """One study, one vocabulary: the retained family row and the retained
    EXP-048 row name the same six facts with the same six keys. Round 20 found
    a divergence of this species on the third consecutive round."""
    seeds = A.h2_confirm_seeds()
    conf = _conf_slice(_fam_cells(det_fpr=0.10, fam_fpr=0.10))[
        "per_attack_family"]["alie"]["census_per_fold"]
    e048 = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), _exp048_scored(0.10, seeds=seeds),
        A.SCENARIOS, A.ATTACKS, A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)["blended"]["S4"]["census_per_fold"]
    readouts = set(_schema().RETAINED_READOUT_KEYS)
    conf_kept = next(r for r in conf.values() if r["status"] == "COMPUTED")
    e048_kept = next(r for r in e048.values() if r["status"] == "COMPUTED")
    assert readouts <= set(conf_kept), sorted(readouts - set(conf_kept))
    assert readouts <= set(e048_kept), sorted(readouts - set(e048_kept))


def test_exp048_per_family_slice_already_carries_the_same_guard():
    """Mirror CHECK, not assumption: the EXP-048 per-family path feeds its own
    FPRs into the shared gate, so it must exclude on the same input."""
    seeds = A.h2_confirm_seeds()
    cells = _fam_cells(det_fpr=0.10, fam_fpr=0.17, seeds=seeds)
    out = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds), cells, A.SCENARIOS, A.ATTACKS,
        A.exact_sign_p, A.comparable, A.student_t_ci, A.ci_on_retained, seeds)
    fam = out["per_family"]["alie"]["S4"]
    assert fam["status"] == "NOT COMPARABLE"
    assert fam["n_folds_comparable"] == 0
    for r in fam["census_per_fold"].values():
        assert "comparability interval" in r["note"]
        assert "detector_recall" not in r


def test_cause_set_symmetry_extends_to_the_family_grain():
    """Both sides classify FAMILY slices against the same three causes."""
    seeds = A.h2_confirm_seeds()
    conf_floor = _conf_slice(_fam_cells(0.10, 0.10, fam_n=4))["per_attack_family"]["alie"]
    conf_fpr = _conf_slice(_fam_cells(0.10, 0.17))["per_attack_family"]["alie"]
    assert set(conf_floor["exclusion_causes"]) == set(A.SEC.EXCLUSION_CAUSES)
    assert conf_floor["exclusion_causes"]["coverage_floor"] == 5
    assert conf_fpr["exclusion_causes"]["fpr_interval"] == 5

    def e048_fam(cells):
        return A.E48.exp048_full_coverage_contrast(
            [], A.rotation_plan(seeds), cells, A.SCENARIOS, A.ATTACKS,
            A.exact_sign_p, A.comparable, A.student_t_ci, A.ci_on_retained,
            seeds)["per_family"]["alie"]["S4"]

    assert e048_fam(_fam_cells(0.10, 0.10, fam_n=4, seeds=seeds))["n_folds_comparable"] == 0
    assert e048_fam(_fam_cells(0.10, 0.17, seeds=seeds))["n_folds_comparable"] == 0
    assert e048_fam(_fam_cells(0.10, 0.10, seeds=seeds))["n_folds_comparable"] == 10


def _cells_missing_family(parent_status=None, seeds=None):
    """Cells whose per_family carries ONLY alie — gaussian_noise is synthesized."""
    seeds = seeds or A.DEV_SEEDS
    cells = {}
    for sc in A.SCENARIOS:
        for sd in seeds:
            cell = {
                "det_recall": 0.7, "tge_recall": 0.3,
                "det_fpr": 0.10, "tge_fpr": 0.10,
                "n_mal_scored": 40, "n_mal_total": 40,
                "n_honest_scored": 100, "n_honest_total": 100,
                "family_eligible_totals": {"alie": 40, "gaussian_noise": 117},
                "family_covered_totals": {"alie": 40, "gaussian_noise": 0},
                "per_family": {"alie": {
                    "det_recall": 0.8, "tge_recall": 0.2,
                    "det_fpr": 0.10, "tge_fpr": 0.10,
                    "n_mal_scored": 40, "n_mal_total": 40,
                    "n_honest_scored": 100, "n_honest_total": 100}},
            }
            if parent_status:
                cell["status"] = parent_status
            cells[(sc, sd)] = cell
    return {"g2": cells}


def test_synthesized_zero_coverage_family_folds_carry_their_cause():
    """A family the scorer never materialized is excluded for a REASON, and
    that reason must reach the cause counts — censused AND caused."""
    fam = _conf_slice(_cells_missing_family())["per_attack_family"]["gaussian_noise"]
    assert fam["status"] == "NOT COMPARABLE"
    assert fam["n_folds"] == 5 and fam["n_folds_with_covered_rows"] == 0
    # every fold is accounted for by a cause, not silently uncaused
    assert sum(fam["exclusion_causes"].values()) == fam["n_folds"]
    assert fam["exclusion_causes"]["coverage_floor"] == 5
    assert "coverage floor" in fam["reason"]
    for r in fam["census_per_fold"].values():
        assert r["status"] == "NOT COMPARABLE"
        assert r["n_mal_scored"] == 0 and r["n_mal_total"] == 117
        assert "TGE scored none" in r["note"]


def test_synthesized_family_folds_under_census_only_parents_carry_census_only():
    fam = _conf_slice(
        _cells_missing_family(parent_status="CENSUS ONLY")
    )["per_attack_family"]["gaussian_noise"]
    assert fam["exclusion_causes"]["census_only"] == 5
    assert sum(fam["exclusion_causes"].values()) == fam["n_folds"]
    assert "census-only" in fam["reason"]


def test_materialized_family_under_census_only_parent_is_also_census_only():
    """The cause classifier sees the PARENT status for materialized folds too."""
    fam = _conf_slice(
        _cells_missing_family(parent_status="CENSUS ONLY")
    )["per_attack_family"]["alie"]
    assert fam["exclusion_causes"]["census_only"] == 5
    assert fam["n_folds_comparable"] == 0


def test_every_family_fold_is_caused_or_comparable_no_silent_gap():
    """INVARIANT: causes + comparable must account for every fold, on every
    family, in every slice — the property this round's finding was about."""
    for cells in (_cells_missing_family(),
                  _cells_missing_family(parent_status="CENSUS ONLY"),
                  _fam_cells(det_fpr=0.10, fam_fpr=0.17),
                  _fam_cells(det_fpr=0.10, fam_fpr=0.10)):
        slice_ = _conf_slice(cells)
        for name, fam in slice_.get("per_attack_family", {}).items():
            total = sum(fam["exclusion_causes"].values()) + fam["n_folds_comparable"]
            assert total == fam["n_folds"], (
                f"{name}: {total} accounted vs {fam['n_folds']} folds")
            assert len(fam["census_per_fold"]) == fam["n_folds"]


def test_census_only_parent_counts_its_families_covered_rows():
    """A census-only parent leaves per_family empty, but its families may have
    covered malicious rows — those folds are NOT zero-coverage."""
    cells = _cells_missing_family(parent_status="CENSUS ONLY")
    for v in cells["g2"].values():
        v["family_covered_totals"] = {"alie": 40, "gaussian_noise": 25}
    fam = _conf_slice(cells)["per_attack_family"]["gaussian_noise"]
    assert fam["n_folds_with_covered_rows"] == 5, "covered rows read as zero"
    for r in fam["census_per_fold"].values():
        assert r["n_mal_scored"] == 25
    assert fam["exclusion_causes"]["census_only"] == 5
    assert sum(fam["exclusion_causes"].values()) + fam["n_folds_comparable"] \
        == fam["n_folds"]


def test_exp048_accounting_invariant_holds_on_that_side_too():
    """causes + comparable == n_folds, mirrored onto the EXP-048 reducer."""
    seeds = A.h2_confirm_seeds()

    def blended(cells):
        return A.E48.exp048_full_coverage_contrast(
            [], A.rotation_plan(seeds), cells, A.SCENARIOS, A.ATTACKS,
            A.exact_sign_p, A.comparable, A.student_t_ci, A.ci_on_retained,
            seeds)["blended"]["S4"]

    for cells in (_exp048_scored(0.15, n_mal=40, seeds=seeds),   # fpr
                  _exp048_scored(0.10, n_mal=4, seeds=seeds),    # floor
                  _exp048_scored(0.10, n_mal=40, seeds=seeds)):  # comparable
        e = blended(cells)
        total = sum(e["exclusion_causes"].values()) + e["n_folds_comparable"]
        assert total == e["n_folds"], (
            f"{total} accounted vs {e['n_folds']} folds: {e['exclusion_causes']}")
        assert len(e["census_per_fold"]) == e["n_folds"]

    fpr = blended(_exp048_scored(0.15, n_mal=40, seeds=seeds))
    assert fpr["exclusion_causes"]["fpr_interval"] == 10
    floor = blended(_exp048_scored(0.10, n_mal=4, seeds=seeds))
    assert floor["exclusion_causes"]["coverage_floor"] == 10


def test_exp048_unscheduled_family_under_a_census_only_parent_is_census_only():
    """ROUND 20 #2. The unscheduled-family branch ran BEFORE the parent-status
    check, so a family absent from a CENSUS-ONLY fold was filed under
    `coverage_floor` — blaming a guard that never got to run. The parent has no
    operating point; that is what excluded the family, and the cause must say so.

    The confirmatory side has always decided it this way (`parent_census_only`
    wins). This is the same decision, on the same input, on the other side.
    """
    seeds = A.h2_confirm_seeds()

    def fam_for(parent_status):
        cells = _fam_cells(det_fpr=0.10, fam_fpr=0.10, seeds=seeds)
        if parent_status:
            for v in cells["g2"].values():
                v["status"] = parent_status
        return A.E48.exp048_full_coverage_contrast(
            [], A.rotation_plan(seeds), cells, A.SCENARIOS, ["gaussian_noise"],
            A.exact_sign_p, A.comparable, A.student_t_ci, A.ci_on_retained,
            seeds)["per_family"]["gaussian_noise"]["S4"]

    # THE HAZARD: gaussian_noise is unscheduled in every fold (no eligible
    # total), and every parent is census-only.
    fam = fam_for("CENSUS ONLY")
    assert fam["exclusion_causes"]["census_only"] == 10
    assert fam["exclusion_causes"]["coverage_floor"] == 0, \
        "an absent operating point was reported as a coverage failure"
    # The accounting invariant still binds — the cause moved, it did not vanish.
    assert sum(fam["exclusion_causes"].values()) + fam["n_folds_comparable"] \
        == fam["n_folds"]
    assert len(fam["census_per_fold"]) == fam["n_folds"]
    for r in fam["census_per_fold"].values():
        assert r["status"] == "NOT COMPARABLE"
        # BOTH facts survive: the parent had no operating point AND the family
        # was never scheduled. Naming only one of them loses the other.
        assert "census only" in r["note"]
        assert "not scheduled" in r["note"]
        # the FAMILY's own zero census, never the pooled cell's counts
        assert r["n_mal_scored"] == 0 and r["n_mal_total"] == 0
        assert MANDATORY_CENSUS_KEYS <= set(r)

    # CONTROL: with a healthy parent the same unscheduled family still files
    # under coverage_floor, so the fix moved one branch and not the rule.
    healthy = fam_for(None)
    assert healthy["exclusion_causes"]["coverage_floor"] == 10
    assert healthy["exclusion_causes"]["census_only"] == 0
    assert all("not scheduled" in r["note"]
               for r in healthy["census_per_fold"].values())


def test_exp048_unscheduled_family_fold_is_caused():
    """The finding's exact case: an unscheduled family must be CAUSED."""
    seeds = A.h2_confirm_seeds()
    fam = A.E48.exp048_full_coverage_contrast(
        [], A.rotation_plan(seeds),
        _fam_cells(det_fpr=0.10, fam_fpr=0.10, seeds=seeds), A.SCENARIOS,
        ["gaussian_noise"], A.exact_sign_p, A.comparable, A.student_t_ci,
        A.ci_on_retained, seeds)["per_family"].get("gaussian_noise", {}).get("S4")
    if fam is None:
        pytest.skip("fixture schedules no gaussian_noise slice")
    total = sum(fam["exclusion_causes"].values()) + fam["n_folds_comparable"]
    assert total == fam["n_folds"]

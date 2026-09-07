"""Tests for scripts/analyze_variance_envelope.py.

Pins the per-defense run-to-run variance-envelope reduction that gates the H2
dev-sweep sizing decision. The study runs later on AWS; these tests drive the
whole script from synthetic fixtures with HAND-COMPUTED mean / SD / CI /
decomposition so the statistics are pinned before any real data exists.

Pre-registration guardrails exercised here:
  - The PRIMARY metric (recall@10%FPR) is scored against a FIXED provisional
    per-defense threshold, applied UNCHANGED to every replicate. It is never
    re-derived per replicate (which would normalise away exactly the
    threshold-crossing variance the study measures), and the ramp-selection
    module is never imported/called (S4-only study; it raises on missing
    S0-S3 cells).
  - Score direction is TRUST (higher = kept); a malicious client is detected
    when its score is BELOW the cut (flag = score < threshold).
  - Variance decomposition sigma_seed^2 = sigma_total^2 - sigma_run^2 is
    reported HONESTLY when non-positive (never clamped, never crashed).
  - Cold-start scope (methodology v1.3 F6) is tenure in [1, k]; a rejoin
    resets tenure so it contributes rounds r..r+2.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_variance_envelope as ave  # noqa: E402


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _row(score_field, score, malicious, tenure, cid="c", server_round=1,
         scenario="S4_full_mix", seed=42, defense_token="krum"):
    row = {
        "logical_cid": cid,
        "malicious_gt": bool(malicious),
        "tenure": tenure,
        "server_round": server_round,
        "scenario": scenario,     # required field per flowerfl/signal_logger.py:89
        "seed": seed,             # required field per flowerfl/signal_logger.py:89
        "defense": defense_token, # strategy-class token per server_app.py:78
    }
    row[score_field] = score
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_unit(base: Path, name: str, score_field: str, rows: list[dict],
               *, final_accuracy=0.80, final_f1=0.75, scenario="S4_full_mix",
               seed=42, config="Krum") -> dict:
    """Write one unit's signal log + result JSON; return manifest path refs.

    Canonical result shape = the REAL runner output: scenario under
    provenance.scenario_path (scripts/run_phase4_flower.py:1176-1180), seed +
    config at top level (:1216-1228), NOT a top-level scenario key. Tests must
    exercise what AWS actually produces."""
    udir = base / name
    udir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(udir / "signals.jsonl", rows)
    (udir / "result.json").write_text(
        json.dumps({"config": config, "final_accuracy": final_accuracy,
                    "final_f1": final_f1, "seed": seed,
                    "provenance": {"scenario_path": scenario}})
    )
    return {"result": f"{name}/result.json", "signal": f"{name}/signals.jsonl"}


def _write_manifest(base: Path, entries: list[dict]) -> Path:
    p = base / "manifest.json"
    p.write_text(json.dumps({"units": entries}))
    return p


def _write_thresholds(base: Path, mapping: dict) -> Path:
    p = base / "thresholds.json"
    p.write_text(json.dumps(mapping))
    return p


def _write_anchors(base: Path, mapping: dict) -> Path:
    p = base / "anchors.json"
    p.write_text(json.dumps(mapping))
    return p


def _spec(defense="krum"):
    return ave.DEFENSE_SCORE_SPECS[defense]


# ---------------------------------------------------------------------------
# score-field / direction table (the Szelag-traceback landmine)
# ---------------------------------------------------------------------------

def test_defense_score_spec_table_fields_and_direction():
    specs = ave.DEFENSE_SCORE_SPECS
    assert set(specs) == {"krum", "trustscore", "tge", "krum_tge"}
    assert specs["krum"].score_field == "krum_score"
    assert specs["trustscore"].score_field == "trust_score"
    assert specs["tge"].score_field == "tge_score"
    # composed chain's final decision is the TGE score (Krum-filtered rows have
    # null tge_score and drop out of the population, is_tge_scored convention)
    assert specs["krum_tge"].score_field == "tge_score"
    # every praxis defense exposes a TRUST score: higher = kept, malicious
    # detected when score < threshold
    for s in specs.values():
        assert s.higher_is_trust is True
        assert s.source  # each direction cites its emitting source line


# ---------------------------------------------------------------------------
# recall @ fixed threshold — direction + hand-computed
# ---------------------------------------------------------------------------

def test_recall_fixed_threshold_hand_computed():
    sf = "krum_score"
    rows = (
        [_row(sf, 0.1, True, 1), _row(sf, 0.2, True, 1),
         _row(sf, 0.6, True, 1), _row(sf, 0.9, True, 1)]      # 2 of 4 below 0.5
        + [_row(sf, 0.7, False, 1), _row(sf, 0.8, False, 1),
           _row(sf, 0.3, False, 1)]                           # 1 of 3 below 0.5
    )
    res = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                        coldstart_only=True, k=3)
    assert res.recall == pytest.approx(0.5)
    assert res.fpr == pytest.approx(1 / 3)
    assert res.n_malicious == 4
    assert res.n_honest == 3


def test_recall_flag_is_low_score_direction():
    """A malicious client with HIGH trust is NOT detected; LOW trust IS."""
    sf = "trust_score"
    rows = [_row(sf, 0.95, True, 1), _row(sf, 0.05, True, 1),
            _row(sf, 0.9, False, 1)]
    res = ave.recall_at_fixed_threshold(rows, _spec("trustscore"), 0.5,
                                        coldstart_only=True, k=3)
    assert res.recall == pytest.approx(0.5)  # only the 0.05 malicious flagged


def test_recall_threshold_is_strict_less_than():
    sf = "krum_score"
    rows = [_row(sf, 0.5, True, 1), _row(sf, 0.4, True, 1),
            _row(sf, 0.9, False, 1)]
    res = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                        coldstart_only=True, k=3)
    # score == threshold is NOT flagged (strict <), so only 0.4 counts
    assert res.recall == pytest.approx(0.5)


def test_recall_none_when_no_malicious_in_scope():
    sf = "krum_score"
    rows = [_row(sf, 0.9, False, 1), _row(sf, 0.8, False, 1)]
    res = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                        coldstart_only=True, k=3)
    assert res is None  # graceful, not a crash


def test_null_scores_skipped_from_population():
    """Rows with a null score for the defense field drop out (compute_recall_fpr
    convention) — a Krum-filtered malicious row (null tge_score) does not
    inflate or deflate the tge population."""
    sf = "tge_score"
    rows = [
        {"logical_cid": "a", "malicious_gt": True, "tenure": 1, sf: None},
        {"logical_cid": "b", "malicious_gt": True, "tenure": 1, sf: 0.1},
        {"logical_cid": "c", "malicious_gt": False, "tenure": 1, sf: 0.9},
    ]
    res = ave.recall_at_fixed_threshold(rows, _spec("tge"), 0.5,
                                        coldstart_only=True, k=3)
    assert res.n_malicious == 1  # the null-score malicious row excluded
    assert res.recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cold-start windowing (methodology v1.3 F6) — rejoin resets tenure
# ---------------------------------------------------------------------------

def test_coldstart_window_is_tenure_1_to_k():
    assert ave.in_coldstart({"tenure": 1}, 3) is True
    assert ave.in_coldstart({"tenure": 3}, 3) is True
    assert ave.in_coldstart({"tenure": 4}, 3) is False
    assert ave.in_coldstart({"tenure": 0}, 3) is False


def test_coldstart_windowing_rejoin_contributes_r_to_r_plus_2():
    """A client rejoining at round r gets a fresh logical id with tenure reset
    to 1, so rounds r, r+1, r+2 (tenure 1,2,3) are in the primary scope and
    r+3 (tenure 4) is not."""
    sf = "krum_score"
    r = 7
    rows = [
        _row(sf, 0.1, True, 1, cid="client_5_new1", server_round=r),      # in
        _row(sf, 0.1, True, 2, cid="client_5_new1", server_round=r + 1),  # in
        _row(sf, 0.1, True, 3, cid="client_5_new1", server_round=r + 2),  # in
        _row(sf, 0.9, True, 4, cid="client_5_new1", server_round=r + 3),  # out
        _row(sf, 0.9, False, 1, cid="client_0", server_round=r),
    ]
    cs = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                       coldstart_only=True, k=3)
    allr = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                         coldstart_only=False, k=3)
    assert cs.n_malicious == 3          # tenure 1,2,3 only
    assert cs.recall == pytest.approx(1.0)   # all three low-trust => detected
    assert allr.n_malicious == 4        # tenure 4 included at all-rounds
    assert allr.recall == pytest.approx(0.75)  # the tenure-4 row (0.9) missed


# ---------------------------------------------------------------------------
# AUC (threshold-free cross-check) — direction-aware, hand-computed
# ---------------------------------------------------------------------------

def test_auc_perfect_detection():
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.4, True, 1),
            _row(sf, 0.5, False, 1), _row(sf, 0.9, False, 1)]
    auc = ave.detector_auc(rows, _spec("krum"), coldstart_only=True, k=3)
    assert auc == pytest.approx(1.0)  # all malicious trust below all honest


def test_auc_random_half():
    sf = "krum_score"
    rows = [_row(sf, 0.6, True, 1), _row(sf, 0.7, True, 1),
            _row(sf, 0.5, False, 1), _row(sf, 0.9, False, 1)]
    auc = ave.detector_auc(rows, _spec("krum"), coldstart_only=True, k=3)
    assert auc == pytest.approx(0.5)


def test_auc_ties_count_half():
    sf = "krum_score"
    rows = [_row(sf, 0.5, True, 1),
            _row(sf, 0.5, False, 1), _row(sf, 0.9, False, 1)]
    auc = ave.detector_auc(rows, _spec("krum"), coldstart_only=True, k=3)
    assert auc == pytest.approx(0.75)  # (tie=0.5 + win=1) / 2


def test_auc_none_without_both_classes():
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.2, True, 1)]
    assert ave.detector_auc(rows, _spec("krum"), coldstart_only=True, k=3) is None


# ---------------------------------------------------------------------------
# mean / SD / bootstrap CI
# ---------------------------------------------------------------------------

def test_mean_sd_ci_hand_computed():
    stat = ave.mean_sd_ci([0.4, 0.5, 0.6], bootstrap_n=500, rng_seed=1)
    assert stat.mean == pytest.approx(0.5)
    assert stat.sd == pytest.approx(0.1)  # ddof=1 sample SD
    assert stat.n == 3
    assert stat.ci_low <= stat.mean <= stat.ci_high


def test_bootstrap_ci_deterministic_and_seeded():
    a = ave.mean_sd_ci([0.2, 0.4, 0.9, 0.3, 0.5], bootstrap_n=400, rng_seed=7)
    b = ave.mean_sd_ci([0.2, 0.4, 0.9, 0.3, 0.5], bootstrap_n=400, rng_seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    c = ave.mean_sd_ci([0.2, 0.4, 0.9, 0.3, 0.5], bootstrap_n=400, rng_seed=8)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_mean_sd_ci_zero_variance_collapses():
    stat = ave.mean_sd_ci([0.7, 0.7, 0.7], bootstrap_n=200, rng_seed=3)
    assert stat.sd == pytest.approx(0.0)
    assert stat.ci_low == pytest.approx(0.7)
    assert stat.ci_high == pytest.approx(0.7)


def test_mean_sd_ci_single_value_sd_none():
    stat = ave.mean_sd_ci([0.7], bootstrap_n=200, rng_seed=3)
    assert stat.n == 1
    assert stat.sd is None            # ddof=1 undefined for n=1
    assert stat.mean == pytest.approx(0.7)


def test_mean_sd_ci_empty():
    stat = ave.mean_sd_ci([], bootstrap_n=50, rng_seed=1)
    assert stat.n == 0
    assert stat.mean is None
    assert stat.sd is None


# ---------------------------------------------------------------------------
# variance decomposition sigma_seed^2 = sigma_total^2 - sigma_run^2
# ---------------------------------------------------------------------------

def test_decompose_positive_sigma_seed():
    d = ave.decompose(sigma_run=0.1, sigma_total=0.2, tier="full")
    assert d.sigma_seed == pytest.approx(math.sqrt(0.04 - 0.01))
    assert d.sigma_seed_var == pytest.approx(0.03)
    assert d.note is None or "indistinguishable" not in d.note


def test_decompose_nonpositive_sigma_seed_honest_report():
    """sigma_total < sigma_run -> seed variance is not resolvable. Report it
    HONESTLY (note + the raw negative variance surfaced), never clamp silently,
    never crash."""
    d = ave.decompose(sigma_run=0.2, sigma_total=0.1, tier="full")
    assert d.sigma_seed is None                      # not clamped to a fake 0
    assert d.sigma_seed_var == pytest.approx(0.01 - 0.04)  # negative, surfaced
    assert d.sigma_seed_var < 0
    assert "indistinguishable" in d.note


def test_decompose_zero_boundary_is_nonpositive():
    d = ave.decompose(sigma_run=0.1, sigma_total=0.1, tier="full")
    assert d.sigma_seed is None
    assert "indistinguishable" in d.note


# ---------------------------------------------------------------------------
# end-to-end analyze — tiers, fixed-cut variance, JSON output
# ---------------------------------------------------------------------------

def _two_arm_manifest(base: Path):
    """krum defense, threshold 0.5. Arm A = seed42 x 3 repeats with SHIFTED
    malicious score distributions (pure run noise); Arm B = 3 seeds x 1."""
    sf = "krum_score"

    def mal_rows(mal_scores, seed):
        rows = [_row(sf, s, True, 1, cid=f"m{i}", seed=seed) for i, s in enumerate(mal_scores)]
        rows += [_row(sf, 0.9, False, 1, cid=f"h{i}", seed=seed) for i in range(4)]
        return rows

    entries = []
    # Arm A: same seed, three repeats; recalls 1.0, 0.5, 0.75 -> visibly vary
    arm_a = {
        0: [0.1, 0.2, 0.3, 0.4],   # all below 0.5 => recall 1.0
        1: [0.1, 0.2, 0.6, 0.7],   # 2/4 below      => recall 0.5
        2: [0.1, 0.2, 0.3, 0.7],   # 3/4 below      => recall 0.75
    }
    for rep, scores in arm_a.items():
        refs = _make_unit(base, f"A_krum_s42_r{rep}", sf, mal_rows(scores, 42),
                          final_accuracy=0.80 + 0.01 * rep, final_f1=0.75, seed=42)
        entries.append({"arm": "A", "defense": "krum", "seed": 42,
                        "replicate": rep, **refs})
    # Arm B: three seeds
    arm_b = {
        42: [0.1, 0.2, 0.3, 0.4],
        137: [0.1, 0.2, 0.3, 0.7],
        256: [0.1, 0.6, 0.7, 0.8],
    }
    for i, (seed, scores) in enumerate(arm_b.items()):
        refs = _make_unit(base, f"B_krum_s{seed}", sf, mal_rows(scores, seed),
                          final_accuracy=0.82, final_f1=0.76, seed=seed)
        entries.append({"arm": "B", "defense": "krum", "seed": seed,
                        "replicate": 0, **refs})
    manifest = _write_manifest(base, entries)
    thresholds = _write_thresholds(base, {"krum": 0.5})
    return manifest, thresholds


def test_analyze_recall_varies_under_single_fixed_cut(tmp_path):
    """The core pre-registration guard: per-replicate recall VARIES under one
    fixed cut when the score distribution shifts. If the threshold were
    re-centred to 10%FPR per replicate, recall would be normalised ~constant."""
    manifest, thresholds = _two_arm_manifest(tmp_path)
    report = ave.run_from_paths(manifest, thresholds)
    recalls = {
        (r["arm"], r["replicate"]): r["metrics"]["recall_coldstart"]
        for r in report["replicates"] if r["arm"] == "A"
    }
    assert recalls[("A", 0)] == pytest.approx(1.0)
    assert recalls[("A", 1)] == pytest.approx(0.5)
    assert recalls[("A", 2)] == pytest.approx(0.75)
    assert len(set(recalls.values())) > 1  # genuinely varies


def test_analyze_full_decomposition_and_summary(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    report = ave.run_from_paths(manifest, thresholds)

    # Arm A SD of recall_coldstart = sample SD of {1.0, 0.5, 0.75}
    import statistics
    exp_sigma_run = statistics.stdev([1.0, 0.5, 0.75])
    exp_sigma_total = statistics.stdev([1.0, 0.75, 0.25])  # arm B recalls

    summ = report["summary"]["krum"]
    assert summ["A"]["recall_coldstart"]["mean"] == pytest.approx(0.75)
    assert summ["A"]["recall_coldstart"]["sd"] == pytest.approx(exp_sigma_run)
    assert summ["A"]["recall_coldstart"]["n"] == 3

    dec = report["decomposition"]["krum"]["recall_coldstart"]
    assert dec["tier"] == "full"
    assert dec["sigma_run"] == pytest.approx(exp_sigma_run)
    assert dec["sigma_total"] == pytest.approx(exp_sigma_total)
    # sigma_total > sigma_run here so seed variance resolves positive
    assert dec["sigma_seed"] == pytest.approx(
        math.sqrt(exp_sigma_total ** 2 - exp_sigma_run ** 2))


def test_analyze_writes_json_out(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    out = tmp_path / "envelope.json"
    ave.run_from_paths(manifest, thresholds, out_path=out)
    doc = json.loads(out.read_text())
    for key in ("meta", "replicates", "summary", "decomposition"):
        assert key in doc
    # every required number present per (defense x metric)
    assert "recall_coldstart" in doc["decomposition"]["krum"]
    assert doc["meta"]["thresholds"] == {"krum": 0.5}
    assert doc["meta"]["coldstart_k"] == ave.DEFAULT_COLDSTART_K
    # secondary finals metrics carried through
    assert "final_accuracy" in doc["summary"]["krum"]["A"]
    assert "final_f1" in doc["summary"]["krum"]["A"]


# ---------------------------------------------------------------------------
# tier-conditional behaviour (never errors on a missing arm)
# ---------------------------------------------------------------------------

def _single_arm_manifest(base: Path, arm: str):
    sf = "krum_score"

    def rows(mal, seed):
        return ([_row(sf, s, True, 1, cid=f"m{i}", seed=seed) for i, s in enumerate(mal)]
                + [_row(sf, 0.9, False, 1, cid=f"h{i}", seed=seed) for i in range(4)])

    entries = []
    data = {0: [0.1, 0.2, 0.3, 0.4], 1: [0.1, 0.2, 0.6, 0.7],
            2: [0.1, 0.2, 0.3, 0.7]}
    for i, (rep, mal) in enumerate(data.items()):
        seed = 42 if arm == "A" else (42 + 100 * i)
        refs = _make_unit(base, f"{arm}_krum_{i}", sf, rows(mal, seed), seed=seed)
        entries.append({"arm": arm, "defense": "krum", "seed": seed,
                        "replicate": rep if arm == "A" else 0, **refs})
    return _write_manifest(base, entries), _write_thresholds(base, {"krum": 0.5})


def test_tier_arm_a_only_run_variance(tmp_path):
    manifest, thresholds = _single_arm_manifest(tmp_path, "A")
    report = ave.run_from_paths(manifest, thresholds)
    dec = report["decomposition"]["krum"]["recall_coldstart"]
    assert dec["tier"] == "run_only"
    assert dec["sigma_run"] is not None
    assert dec["sigma_total"] is None
    assert dec["sigma_seed"] is None
    assert "unavailable" in dec["note"].lower()


def test_tier_arm_b_only_no_anchor(tmp_path):
    manifest, thresholds = _single_arm_manifest(tmp_path, "B")
    report = ave.run_from_paths(manifest, thresholds)
    dec = report["decomposition"]["krum"]["recall_coldstart"]
    assert dec["tier"] == "total_only"
    assert dec["sigma_total"] is not None   # labelled seedrun
    assert dec["sigma_run"] is None
    assert dec["sigma_seed"] is None
    assert "anchor" in dec["note"].lower()


def test_tier_arm_b_only_with_anchor(tmp_path):
    manifest, thresholds = _single_arm_manifest(tmp_path, "B")
    anchors = _write_anchors(tmp_path, {"krum": 0.05})
    report = ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)
    dec = report["decomposition"]["krum"]["recall_coldstart"]
    assert dec["tier"] == "total_with_anchor"
    assert dec["sigma_run"] == pytest.approx(0.05)
    assert dec["sigma_run_source"] and "anchor" in dec["sigma_run_source"].lower()
    # sigma_seed resolves from the anchored run variance
    st = dec["sigma_total"]
    assert dec["sigma_seed"] == pytest.approx(math.sqrt(max(st ** 2 - 0.05 ** 2, 0.0)))


# ---------------------------------------------------------------------------
# per-defense sigma_run anchors (the EXP-005c/005e anchor is per-defense)
# ---------------------------------------------------------------------------

def _armB_unit(base: Path, name: str, score_field: str, mal_scores: list,
               *, seed=42, config="Krum", defense_token="krum") -> dict:
    rows = [_row(score_field, s, True, 1, cid=f"m{i}", seed=seed, defense_token=defense_token)
            for i, s in enumerate(mal_scores)]
    rows += [_row(score_field, 0.9, False, 1, cid=f"h{i}", seed=seed, defense_token=defense_token)
             for i in range(4)]
    return _make_unit(base, name, score_field, rows, seed=seed, config=config)


def _two_defense_armB_manifest(base: Path):
    """krum + tge, each Arm-B-only, with hand-set cold-start recalls so
    sigma_total is exactly computable per defense."""
    entries = []
    krum_recalls = {42: [0.1, 0.2, 0.3, 0.4],   # recall 1.0
                    137: [0.1, 0.2, 0.3, 0.7],  # recall 0.75
                    256: [0.1, 0.7, 0.8, 0.9]}  # recall 0.25
    for seed, mal in krum_recalls.items():
        refs = _armB_unit(base, f"kB_{seed}", "krum_score", mal, seed=seed,
                          config="Krum", defense_token="krum")
        entries.append({"arm": "B", "defense": "krum", "seed": seed, "repeat": 0, **refs})
    tge_recalls = {42: [0.1, 0.2, 0.3, 0.4],   # recall 1.0
                   137: [0.1, 0.2, 0.7, 0.8],  # recall 0.5
                   256: [0.6, 0.7, 0.8, 0.9]}  # recall 0.0
    for seed, mal in tge_recalls.items():
        refs = _armB_unit(base, f"tB_{seed}", "tge_score", mal, seed=seed,
                          config="TGE", defense_token="tgensemble")
        entries.append({"arm": "B", "defense": "tge", "seed": seed, "repeat": 0, **refs})
    return _write_manifest(base, entries), _write_thresholds(base, {"krum": 0.5, "tge": 0.5})


def test_per_defense_anchors_decompose_independently(tmp_path):
    import statistics
    manifest, thresholds = _two_defense_armB_manifest(tmp_path)
    anchors = _write_anchors(tmp_path, {"krum": 0.05, "tge": 0.10})
    report = ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)

    st_k = statistics.stdev([1.0, 0.75, 0.25])
    st_t = statistics.stdev([1.0, 0.5, 0.0])
    dk = report["decomposition"]["krum"]["recall_coldstart"]
    dt = report["decomposition"]["tge"]["recall_coldstart"]

    assert dk["tier"] == "total_with_anchor"
    assert dk["sigma_run"] == pytest.approx(0.05)
    assert dk["sigma_seed"] == pytest.approx(math.sqrt(st_k ** 2 - 0.05 ** 2))
    assert dt["tier"] == "total_with_anchor"
    assert dt["sigma_run"] == pytest.approx(0.10)
    assert dt["sigma_seed"] == pytest.approx(math.sqrt(st_t ** 2 - 0.10 ** 2))
    # the two defenses' anchors are applied independently (distinct sigma_run)
    assert dk["sigma_run"] != dt["sigma_run"]


def test_anchor_absent_defense_stays_total_only(tmp_path):
    manifest, thresholds = _two_defense_armB_manifest(tmp_path)
    anchors = _write_anchors(tmp_path, {"krum": 0.05})  # tge omitted
    report = ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)
    dt = report["decomposition"]["tge"]["recall_coldstart"]
    assert dt["tier"] == "total_only"
    assert dt["sigma_run"] is None
    assert "anchor" in dt["note"].lower()


def test_anchor_unknown_defense_key_raises(tmp_path):
    manifest, thresholds = _two_defense_armB_manifest(tmp_path)
    anchors = _write_anchors(tmp_path, {"mystery": 0.05})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)


def test_scalar_anchor_flag_removed_from_cli(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    with pytest.raises(SystemExit):  # old scalar flag no longer parses
        ave.main(["--manifest", str(manifest), "--thresholds", str(thresholds),
                  "--sigma-run-anchor", "0.05"])


def test_cli_accepts_sigma_run_anchors_path(tmp_path):
    manifest, thresholds = _two_defense_armB_manifest(tmp_path)
    anchors = _write_anchors(tmp_path, {"krum": 0.05, "tge": 0.10})
    out = tmp_path / "anchored.json"
    rc = ave.main(["--manifest", str(manifest), "--thresholds", str(thresholds),
                   "--sigma-run-anchors", str(anchors), "--out", str(out),
                   "--bootstrap-n", "50"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["meta"]["sigma_run_anchors"] == {"krum": 0.05, "tge": 0.10}


def test_missing_arm_never_raises(tmp_path):
    # Arm A only must not crash even though total/seed are unavailable
    manifest, thresholds = _single_arm_manifest(tmp_path, "A")
    report = ave.run_from_paths(manifest, thresholds)  # no exception
    assert report["decomposition"]["krum"]["recall_coldstart"]["tier"] == "run_only"


# ---------------------------------------------------------------------------
# finals-from-result helper
# ---------------------------------------------------------------------------

def test_finals_from_result_primary_keys():
    acc, f1 = ave.finals_from_result({"final_accuracy": 0.91, "final_f1": 0.88})
    assert (acc, f1) == (0.91, 0.88)


def test_finals_from_result_acc_fallback():
    acc, f1 = ave.finals_from_result({"final_acc": 0.5, "final_f1": 0.4})
    assert acc == 0.5


def test_finals_from_result_missing_is_none():
    acc, f1 = ave.finals_from_result({})
    assert acc is None and f1 is None


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_unknown_defense_raises(tmp_path):
    sf = "krum_score"
    refs = _make_unit(tmp_path, "u0", sf, [_row(sf, 0.1, True, 1)])
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "mystery", "seed": 42, "replicate": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"mystery": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_missing_threshold_for_defense_raises(tmp_path):
    sf = "krum_score"
    refs = _make_unit(tmp_path, "u0", sf, [_row(sf, 0.1, True, 1)])
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "replicate": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"tge": 0.5})  # missing krum
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


# ---------------------------------------------------------------------------
# fail-fast manifest preconditions (VarianceEnvelopeError class)
# ---------------------------------------------------------------------------

def _mini_unit(base, name):
    sf = "krum_score"
    return _make_unit(base, name, sf,
                      [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)])


def test_duplicate_unit_rejected(tmp_path):
    """A copy-pasted (arm, defense, seed, replicate) would double-count and
    bias sigma — must fail fast naming the duplicate key."""
    entry = {"arm": "A", "defense": "krum", "seed": 42, "replicate": 0,
             **_mini_unit(tmp_path, "u0")}
    manifest = _write_manifest(tmp_path, [entry, dict(entry)])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_arm_a_mixed_seed_rejected(tmp_path):
    """Arm A is DEFINED as fixed-seed repeats; mixed seeds would smuggle seed
    variance into sigma_run."""
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "replicate": 0,
         **_mini_unit(tmp_path, "a0")},
        {"arm": "A", "defense": "krum", "seed": 137, "replicate": 1,
         **_mini_unit(tmp_path, "a1")}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_arm_b_duplicate_seed_rejected(tmp_path):
    """Arm B is DEFINED as distinct-seed singles; a repeated seed (mislabeled
    repeat) would inflate sigma_total."""
    manifest = _write_manifest(tmp_path, [
        {"arm": "B", "defense": "krum", "seed": 42, "replicate": 0,
         **_mini_unit(tmp_path, "b0")},
        {"arm": "B", "defense": "krum", "seed": 42, "replicate": 1,
         **_mini_unit(tmp_path, "b1")}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_valid_arm_shapes_do_not_raise(tmp_path):
    """Fixed-seed Arm A repeats + distinct-seed Arm B singles is the canonical
    shape and must pass."""
    manifest, thresholds = _two_arm_manifest(tmp_path)
    ave.run_from_paths(manifest, thresholds)  # no exception


# ---------------------------------------------------------------------------
# strict-JSON diagnostics (no NaN/Infinity in --out)
# ---------------------------------------------------------------------------

def test_recall_fpr_none_when_no_honest_in_scope():
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.6, True, 1)]  # no honest rows
    res = ave.recall_at_fixed_threshold(rows, _spec("krum"), 0.5,
                                        coldstart_only=True, k=3)
    assert res is not None
    assert res.recall == pytest.approx(0.5)
    assert res.fpr is None          # None, never float('nan')
    assert res.n_honest == 0


def test_report_is_strict_json_no_nan(tmp_path):
    """An honest-empty cold-start scope must not leak NaN into the report —
    json.dumps(..., allow_nan=False) must round-trip and the --out file must
    be valid strict JSON."""
    refs = _make_unit(tmp_path, "u0", "krum_score",
                      [_row("krum_score", 0.1, True, 1),
                       _row("krum_score", 0.6, True, 1)])  # no honest
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "replicate": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    out = tmp_path / "strict.json"
    report = ave.run_from_paths(manifest, thresholds, out_path=out)
    diag = report["replicates"][0]["diagnostics"]
    assert diag["fpr_coldstart"] is None
    json.dumps(report, allow_nan=False)  # raises if any NaN/Infinity present
    reloaded = json.loads(out.read_text())
    assert reloaded["replicates"][0]["diagnostics"]["fpr_coldstart"] is None


def test_nonfinite_finals_sanitized_to_none():
    acc, f1 = ave.finals_from_result({"final_accuracy": float("nan"),
                                      "final_f1": float("inf")})
    assert acc is None and f1 is None


# ---------------------------------------------------------------------------
# launch-manifest interop: Unit asdict serializes the ordinal as 'repeat'
# ---------------------------------------------------------------------------

def test_manifest_repeat_key_parses_ordinal(tmp_path):
    """A manifest generated from a launch Unit (asdict) carries the replicate
    ordinal under key 'repeat'. It must be read so an Arm A pair with repeat
    1/2 does NOT collapse to 0/0 and collide as duplicates."""
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 1,
         **_mini_unit(tmp_path, "a0")},
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 2,
         **_mini_unit(tmp_path, "a1")}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    report = ave.run_from_paths(manifest, thresholds)  # no duplicate collision
    reps = sorted(r["replicate"] for r in report["replicates"])
    assert reps == [1, 2]


def test_manifest_both_ordinal_keys_equal_ok(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "replicate": 3,
         "repeat": 3, **_mini_unit(tmp_path, "u0")}])
    units = ave.load_manifest(manifest)
    assert units[0].replicate == 3


def test_manifest_conflicting_ordinal_keys_raise(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "replicate": 1,
         "repeat": 2, **_mini_unit(tmp_path, "u0")}])
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_manifest(manifest)
    msg = str(ei.value)
    assert "replicate" in msg and "repeat" in msg
    assert "1" in msg and "2" in msg  # names both disagreeing values


# ---------------------------------------------------------------------------
# frozen threshold artifact wrapper: {_meta, thresholds} (spec § 5)
# ---------------------------------------------------------------------------

def test_load_thresholds_plain_mapping_unchanged(tmp_path):
    p = _write_thresholds(tmp_path, {"krum": 0.5, "tge": 0.4})
    assert ave.load_thresholds(p) == {"krum": 0.5, "tge": 0.4}


def test_load_thresholds_wrapper_shape_parses_to_same_cuts(tmp_path):
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"_meta": {"frozen_from": "EXP-005c/005e"},
                             "thresholds": {"krum": 0.5, "tge": 0.4}}))
    assert ave.load_thresholds(p) == {"krum": 0.5, "tge": 0.4}


def test_load_thresholds_wrapper_with_stray_key_raises(tmp_path):
    p = tmp_path / "ambiguous.json"
    p.write_text(json.dumps({"thresholds": {"krum": 0.5}, "krum": 0.5}))
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.load_thresholds(p)


def test_load_thresholds_thresholds_non_dict_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"thresholds": 5}))
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.load_thresholds(p)


def test_run_from_paths_accepts_wrapped_thresholds(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    Path(thresholds).write_text(json.dumps(
        {"_meta": {"source": "frozen"}, "thresholds": {"krum": 0.5}}))
    report = ave.run_from_paths(manifest, thresholds)
    assert report["meta"]["thresholds"] == {"krum": 0.5}


# ---------------------------------------------------------------------------
# pre-registration: threshold not re-derived; ramp module never used
# ---------------------------------------------------------------------------

def test_source_never_references_ramp_or_threshold_selection():
    """Static guard: the module must NOT import analyze_ramp_selection nor any
    threshold-SELECTION helper (select_threshold / h2_threshold_pipeline).
    Re-deriving a 10%FPR cut from the analysed logs would normalise away the
    variance being measured."""
    src = (PROJECT_ROOT / "scripts" / "analyze_variance_envelope.py").read_text()
    assert "analyze_ramp_selection" not in src
    assert "select_threshold" not in src
    assert "h2_threshold_pipeline" not in src


def test_ramp_module_not_invoked_at_runtime(tmp_path, monkeypatch):
    """Runtime guard: booby-trap analyze_ramp_selection so ANY attribute access
    raises, then run a full analysis. It must complete untouched."""
    import types

    class _Boom(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(
                "analyze_variance_envelope must never touch analyze_ramp_selection")

    monkeypatch.setitem(sys.modules, "analyze_ramp_selection", _Boom("analyze_ramp_selection"))
    monkeypatch.setitem(sys.modules, "scripts.analyze_ramp_selection",
                        _Boom("scripts.analyze_ramp_selection"))
    manifest, thresholds = _two_arm_manifest(tmp_path)
    report = ave.run_from_paths(manifest, thresholds)  # no AssertionError
    assert report["replicates"]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_main_smoke(tmp_path, capsys):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    out = tmp_path / "cli_out.json"
    rc = ave.main(["--manifest", str(manifest), "--thresholds", str(thresholds),
                   "--out", str(out), "--bootstrap-n", "100"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "krum" in captured
    assert "recall_coldstart" in captured
    assert out.exists()


def test_cli_subprocess_exits_zero(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    script = PROJECT_ROOT / "scripts" / "analyze_variance_envelope.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest),
         "--thresholds", str(thresholds), "--bootstrap-n", "50"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# registered-scenario integrity (chain of custody on both inputs)
# ---------------------------------------------------------------------------

def _one_unit_manifest(base, sf, rows, *, result_scenario="S4_full_mix", defense="krum"):
    refs = _make_unit(base, "u0", sf, rows, scenario=result_scenario)
    manifest = _write_manifest(base, [
        {"arm": "A", "defense": defense, "seed": 42, "repeat": 0, **refs}])
    thresholds = _write_thresholds(base, {defense: 0.5})
    return manifest, thresholds


def test_scenario_scope_records_validated_stem(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    report = ave.run_from_paths(manifest, thresholds)
    assert report["meta"]["scenario_scope"] == "S4_full_mix"  # not the bare "S4"


def test_wrong_scenario_signal_row_rejected(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1, scenario="S0_honest_baseline"),
            _row(sf, 0.9, False, 1, scenario="S0_honest_baseline")]
    manifest, thresholds = _one_unit_manifest(tmp_path, sf, rows,
                                              result_scenario="S4_full_mix")
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "krum" in msg and "S0_honest_baseline" in msg  # names unit + value


def test_wrong_scenario_result_rejected(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)]  # rows correct
    manifest, thresholds = _one_unit_manifest(tmp_path, sf, rows,
                                              result_scenario="S1_intermittent")
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    assert "S1_intermittent" in str(ei.value)


def test_scenario_path_vs_stem_tolerance_passes(tmp_path):
    """Result records the full path '.../S4_full_mix.json'; rows carry the
    stem 'S4_full_mix'. Stem-normalization must reconcile them."""
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1, scenario="S4_full_mix"),
            _row(sf, 0.9, False, 1, scenario="S4_full_mix")]
    manifest, thresholds = _one_unit_manifest(
        tmp_path, sf, rows, result_scenario="rmc/scenarios/S4_full_mix.json")
    report = ave.run_from_paths(manifest, thresholds)  # no raise
    assert report["meta"]["scenario_scope"] == "S4_full_mix"


def test_missing_scenario_field_in_result_rejected(tmp_path):
    """Absent from BOTH top-level and provenance.scenario_path -> reject, and
    the message names both looked-in locations."""
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl",
                 [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)])
    # no top-level scenario AND no provenance.scenario_path
    (udir / "result.json").write_text(
        json.dumps({"final_accuracy": 0.8, "final_f1": 0.75,
                    "provenance": {"runner_version": "x"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "scenario" in msg and "provenance" in msg  # names both locations


def test_real_shape_result_provenance_scenario_passes(tmp_path):
    """The canonical AWS result carries scenario ONLY under
    provenance.scenario_path (no top-level key) — it must be accepted."""
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)]
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", rows)
    (udir / "result.json").write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75, "seed": 42,
        "provenance": {"runner_version": "v1", "scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    report = ave.run_from_paths(manifest, thresholds)  # no raise
    assert report["meta"]["scenario_scope"] == "S4_full_mix"


def test_provenance_scenario_mismatch_rejected(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)]  # rows correct
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", rows)
    (udir / "result.json").write_text(json.dumps({
        "final_accuracy": 0.8, "final_f1": 0.75,
        "provenance": {"scenario_path": "rmc/scenarios/S3_churn.json"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "krum" in msg and "S3_churn" in msg  # names unit + offending value


def test_top_level_scenario_compat_accepted(tmp_path):
    """Back-compat: a result with a top-level `scenario` (no provenance) is
    still accepted (preferred over provenance when both are present)."""
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)]
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", rows)
    (udir / "result.json").write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75, "seed": 42,
        "scenario": "S4_full_mix"}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    report = ave.run_from_paths(manifest, thresholds)  # no raise
    assert report["meta"]["scenario_scope"] == "S4_full_mix"


def test_missing_scenario_field_in_signal_row_rejected(tmp_path):
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", [
        {"logical_cid": "m", "malicious_gt": True, "tenure": 1, sf: 0.1},
        {"logical_cid": "h", "malicious_gt": False, "tenure": 1, sf: 0.9}])
    (udir / "result.json").write_text(
        json.dumps({"final_accuracy": 0.8, "final_f1": 0.75,
                    "scenario": "S4_full_mix"}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_mixed_scenario_one_stray_row_rejected(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1, scenario="S4_full_mix"),
            _row(sf, 0.2, True, 2, scenario="S4_full_mix"),
            _row(sf, 0.9, False, 1, scenario="S3_churn")]  # one stray row
    manifest, thresholds = _one_unit_manifest(tmp_path, sf, rows)
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    assert "S3_churn" in str(ei.value)


def test_custom_expected_scenario_arg(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1, scenario="S2_reconnect"),
            _row(sf, 0.9, False, 1, scenario="S2_reconnect")]
    manifest, thresholds = _one_unit_manifest(tmp_path, sf, rows,
                                              result_scenario="S2_reconnect")
    report = ave.run_from_paths(manifest, thresholds,
                                expected_scenario="S2_reconnect")
    assert report["meta"]["scenario_scope"] == "S2_reconnect"


def test_cli_scenario_flag_wires_through(tmp_path):
    sf = "krum_score"
    rows = [_row(sf, 0.1, True, 1, scenario="S2_reconnect"),
            _row(sf, 0.9, False, 1, scenario="S2_reconnect")]
    manifest, thresholds = _one_unit_manifest(tmp_path, sf, rows,
                                              result_scenario="S2_reconnect")
    out = tmp_path / "o.json"
    rc = ave.main(["--manifest", str(manifest), "--thresholds", str(thresholds),
                   "--scenario", "S2_reconnect", "--out", str(out),
                   "--bootstrap-n", "50"])
    assert rc == 0
    assert json.loads(out.read_text())["meta"]["scenario_scope"] == "S2_reconnect"


# ---------------------------------------------------------------------------
# seed-identity integrity (chain of custody, extended to seed)
# ---------------------------------------------------------------------------

def _seed_unit(base, *, result_seed, row_seed, declared_seed):
    """One Arm A unit whose result/rows carry given seeds, declared under
    `declared_seed` in the manifest."""
    sf = "krum_score"
    udir = base / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", [
        _row(sf, 0.1, True, 1, seed=row_seed), _row(sf, 0.9, False, 1, seed=row_seed)])
    (udir / "result.json").write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75,
        "seed": result_seed, "provenance": {"scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(base, [
        {"arm": "A", "defense": "krum", "seed": declared_seed, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    return manifest, _write_thresholds(base, {"krum": 0.5})


def test_matching_seed_passes(tmp_path):
    manifest, thresholds = _seed_unit(tmp_path, result_seed=42, row_seed=42,
                                      declared_seed=42)
    report = ave.run_from_paths(manifest, thresholds)  # no raise
    assert report["replicates"][0]["seed"] == 42


def test_wrong_seed_result_rejected(tmp_path):
    # result declares seed 137 but the manifest unit is seed 42
    manifest, thresholds = _seed_unit(tmp_path, result_seed=137, row_seed=42,
                                      declared_seed=42)
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "result" in msg and "137" in msg and "42" in msg  # location + both values


def test_wrong_seed_signal_row_rejected(tmp_path):
    # rows carry seed 137 but the manifest unit is seed 42
    manifest, thresholds = _seed_unit(tmp_path, result_seed=42, row_seed=137,
                                      declared_seed=42)
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "row 0" in msg and "137" in msg and "42" in msg  # names row index + values


def test_missing_seed_in_result_rejected(tmp_path):
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl",
                 [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)])
    (udir / "result.json").write_text(json.dumps({  # no seed
        "final_accuracy": 0.8, "final_f1": 0.75,
        "provenance": {"scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_missing_seed_in_signal_row_rejected(tmp_path):
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", [  # rows without seed
        {"logical_cid": "m", "malicious_gt": True, "tenure": 1,
         "scenario": "S4_full_mix", sf: 0.1},
        {"logical_cid": "h", "malicious_gt": False, "tenure": 1,
         "scenario": "S4_full_mix", sf: 0.9}])
    (udir / "result.json").write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75, "seed": 42,
        "provenance": {"scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


# ---------------------------------------------------------------------------
# defense-identity integrity (real strategy-class vocabulary, strict equality)
# ---------------------------------------------------------------------------

# The real map, independent of the module under test:
#   manifest defense -> (result config, signal-row defense token)
_DEFENSE_IDENTITY = {
    "krum": ("Krum", "krum"),
    "trustscore": ("TrustScore", "trustscore"),
    "tge": ("TGE", "tgensemble"),         # token is NOT "tge"
    "krum_tge": ("Krum+TGE", "krumtge"),  # token is NOT "krum_tge"
}
_SCORE_FIELD = {"krum": "krum_score", "trustscore": "trust_score",
                "tge": "tge_score", "krum_tge": "tge_score"}


def _defense_unit(base, name, defense, *, config=None, token=None, seed=42):
    """One Arm-A unit for `defense`; config/token override for swap tests."""
    real_cfg, real_tok = _DEFENSE_IDENTITY[defense]
    cfg = real_cfg if config is None else config
    tok = real_tok if token is None else token
    sf = _SCORE_FIELD[defense]
    rows = [_row(sf, 0.1, True, 1, cid="m", seed=seed, defense_token=tok),
            _row(sf, 0.9, False, 1, cid="h", seed=seed, defense_token=tok)]
    return _make_unit(base, name, sf, rows, seed=seed, config=cfg)


def test_defense_identity_vocabulary_map():
    """The spec pins the REAL strategy-class vocabulary, not the naive tokens."""
    specs = ave.DEFENSE_SCORE_SPECS
    for defense, (cfg, tok) in _DEFENSE_IDENTITY.items():
        assert specs[defense].result_config == cfg
        assert specs[defense].row_defense_token == tok
    assert specs["tge"].row_defense_token == "tgensemble"     # not "tge"
    assert specs["krum_tge"].row_defense_token == "krumtge"   # not "krum_tge"


def test_all_four_defenses_pass_with_correct_artifacts(tmp_path):
    entries = []
    for defense in ("krum", "trustscore", "tge", "krum_tge"):
        refs = _defense_unit(tmp_path, f"u_{defense}", defense)
        entries.append({"arm": "A", "defense": defense, "seed": 42,
                        "repeat": 0, **refs})
    manifest = _write_manifest(tmp_path, entries)
    thresholds = _write_thresholds(tmp_path, {d: 0.5 for d in
                                              ("krum", "trustscore", "tge", "krum_tge")})
    report = ave.run_from_paths(manifest, thresholds)  # no raise
    assert len(report["replicates"]) == 4


def test_swapped_tge_artifact_under_krum_tge_rejected_by_config(tmp_path):
    """A tge result config ('TGE') under a krum_tge unit is caught by the
    config check even though rows would carry the correct krum_tge token."""
    refs = _defense_unit(tmp_path, "u0", "krum_tge", config="TGE")  # wrong config
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum_tge", "seed": 42, "repeat": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"krum_tge": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "krum_tge" in msg and "TGE" in msg and "Krum+TGE" in msg


def test_swapped_tge_artifact_under_krum_tge_rejected_by_row(tmp_path):
    """tge row tokens ('tgensemble') under a krum_tge unit are caught by the
    row-token check even when the result config is correct."""
    refs = _defense_unit(tmp_path, "u0", "krum_tge", token="tgensemble")  # wrong token
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum_tge", "seed": 42, "repeat": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"krum_tge": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "krum_tge" in msg and "tgensemble" in msg and "krumtge" in msg


def test_missing_result_config_rejected(tmp_path):
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl",
                 [_row(sf, 0.1, True, 1), _row(sf, 0.9, False, 1)])
    (udir / "result.json").write_text(json.dumps({  # no config
        "final_accuracy": 0.8, "final_f1": 0.75, "seed": 42,
        "provenance": {"scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_missing_row_defense_token_rejected(tmp_path):
    sf = "krum_score"
    udir = tmp_path / "u0"
    udir.mkdir()
    _write_jsonl(udir / "signals.jsonl", [  # rows without defense token
        {"logical_cid": "m", "malicious_gt": True, "tenure": 1,
         "scenario": "S4_full_mix", "seed": 42, sf: 0.1},
        {"logical_cid": "h", "malicious_gt": False, "tenure": 1,
         "scenario": "S4_full_mix", "seed": 42, sf: 0.9}])
    (udir / "result.json").write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75, "seed": 42,
        "provenance": {"scenario_path": "S4_full_mix"}}))
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "u0/result.json", "signal": "u0/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds)


def test_naive_tge_token_under_tge_unit_rejected(tmp_path):
    """A row token 'tge' (the NAIVE guess) under a tge unit must be rejected —
    proving we pinned the REAL 'tgensemble' vocabulary, not the config token."""
    refs = _defense_unit(tmp_path, "u0", "tge", token="tge")  # naive wrong token
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "tge", "seed": 42, "repeat": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"tge": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "tgensemble" in msg and "'tge'" in msg  # expected real token vs got naive


# ---------------------------------------------------------------------------
# config-label key canonicalization in threshold / anchor inputs
# ---------------------------------------------------------------------------

def _all_four_manifest(base, seed=42):
    entries = []
    for defense in ("krum", "trustscore", "tge", "krum_tge"):
        refs = _defense_unit(base, f"u_{defense}", defense, seed=seed)
        entries.append({"arm": "A", "defense": defense, "seed": seed,
                        "repeat": 0, **refs})
    return _write_manifest(base, entries)


def _armB_defense_manifest(base, defense):
    """Three Arm-B seeds for one defense, varied recalls so sigma_total > 0."""
    patterns = {42: [0.1, 0.2, 0.3, 0.4], 137: [0.1, 0.2, 0.3, 0.7],
                256: [0.1, 0.7, 0.8, 0.9]}
    cfg, tok = _DEFENSE_IDENTITY[defense]
    sf = _SCORE_FIELD[defense]
    entries = []
    for seed, mal in patterns.items():
        rows = [_row(sf, s, True, 1, cid=f"m{i}", seed=seed, defense_token=tok)
                for i, s in enumerate(mal)]
        rows += [_row(sf, 0.9, False, 1, cid=f"h{i}", seed=seed, defense_token=tok)
                 for i in range(4)]
        refs = _make_unit(base, f"{defense}_{seed}", sf, rows, seed=seed, config=cfg)
        entries.append({"arm": "B", "defense": defense, "seed": seed,
                        "repeat": 0, **refs})
    return _write_manifest(base, entries)


def test_config_label_keyed_wrapped_thresholds_end_to_end(tmp_path):
    """The frozen-threshold pipeline writes config-label keys inside the
    wrapper; a convention-following artifact must reduce, not abort."""
    manifest = _all_four_manifest(tmp_path)
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"_meta": {"frozen": "EXP-005c/005e"},
                               "thresholds": {"Krum": 0.5, "TrustScore": 0.5,
                                              "TGE": 0.5, "Krum+TGE": 0.5}}))
    report = ave.run_from_paths(manifest, thr)
    assert len(report["replicates"]) == 4
    # canonicalized to defense ids in meta
    assert set(report["meta"]["thresholds"]) == {"krum", "trustscore", "tge", "krum_tge"}


def test_lowercase_id_threshold_keys_still_work(tmp_path):
    p = _write_thresholds(tmp_path, {"krum": 0.5, "tge": 0.4})
    assert ave.load_thresholds(p) == {"krum": 0.5, "tge": 0.4}


def test_threshold_key_collision_disagreeing_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"Krum": 0.5, "krum": 0.6}))  # both -> krum, differ
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_thresholds(p)
    msg = str(ei.value)
    assert "Krum" in msg and "krum" in msg  # names both originals


def test_threshold_key_collision_equal_values_accepted(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"Krum": 0.5, "krum": 0.5}))  # agree -> collapse
    assert ave.load_thresholds(p) == {"krum": 0.5}


def test_anchors_config_label_key_maps_and_decomposes(tmp_path):
    manifest = _armB_defense_manifest(tmp_path, "krum_tge")
    thresholds = _write_thresholds(tmp_path, {"krum_tge": 0.5})
    anchors = tmp_path / "a.json"
    anchors.write_text(json.dumps({"Krum+TGE": 0.05}))  # config label -> krum_tge
    report = ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)
    dec = report["decomposition"]["krum_tge"]["recall_coldstart"]
    assert dec["tier"] == "total_with_anchor"
    assert dec["sigma_run"] == pytest.approx(0.05)


def test_unknown_threshold_key_ignored(tmp_path):
    manifest, _ = _single_arm_manifest(tmp_path, "A")  # krum only
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"krum": 0.5, "bulyan": 0.9}))  # bulyan unknown
    report = ave.run_from_paths(manifest, thr)  # no raise: unknown key ignored
    assert "krum" in report["decomposition"]


def test_unknown_anchor_key_still_raises(tmp_path):
    manifest, thresholds = _single_arm_manifest(tmp_path, "B")
    anchors = tmp_path / "a.json"
    anchors.write_text(json.dumps({"bulyan": 0.05}))  # unknown -> fail-fast
    with pytest.raises(ave.VarianceEnvelopeError):
        ave.run_from_paths(manifest, thresholds, sigma_run_anchors_path=anchors)


# ---------------------------------------------------------------------------
# config-label MANIFEST defense keys (symmetric completion of R7)
# ---------------------------------------------------------------------------

def test_load_manifest_canonicalizes_config_label_defense(tmp_path):
    refs = _defense_unit(tmp_path, "u0", "krum_tge")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "Krum+TGE", "seed": 42, "repeat": 0, **refs}])
    units = ave.load_manifest(manifest)
    assert units[0].defense == "krum_tge"  # config label -> defense id


def test_config_label_keyed_manifest_end_to_end(tmp_path):
    """A manifest whose defense keys are config labels (from launch-side matrix
    docs) reduces, grouped under canonical defense ids."""
    entries = []
    for defense_id, (cfg, _tok) in _DEFENSE_IDENTITY.items():
        refs = _defense_unit(tmp_path, f"u_{defense_id}", defense_id)
        entries.append({"arm": "A", "defense": cfg, "seed": 42,  # config-label key
                        "repeat": 0, **refs})
    manifest = _write_manifest(tmp_path, entries)
    thresholds = _write_thresholds(tmp_path, {"Krum": 0.5, "TrustScore": 0.5,
                                              "TGE": 0.5, "Krum+TGE": 0.5})
    report = ave.run_from_paths(manifest, thresholds)
    ids = {"krum", "trustscore", "tge", "krum_tge"}
    assert set(report["summary"]) == ids
    assert set(report["decomposition"]) == ids
    assert {r["defense"] for r in report["replicates"]} == ids


def test_lowercase_id_manifest_defense_unchanged(tmp_path):
    manifest, _ = _two_arm_manifest(tmp_path)
    units = ave.load_manifest(manifest)
    assert {u.defense for u in units} == {"krum"}  # ids pass through unchanged


def test_unknown_manifest_defense_label_still_raises(tmp_path):
    refs = _defense_unit(tmp_path, "u0", "krum")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "Bulyan", "seed": 42, "repeat": 0, **refs}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError):  # neither id nor config label
        ave.run_from_paths(manifest, thresholds)


# ---------------------------------------------------------------------------
# launch-manifest 'config' key for the defense (Unit dataclass serializes it)
# ---------------------------------------------------------------------------

def test_manifest_config_key_only_resolves_defense(tmp_path):
    """Launch Unit serializes the defense under 'config', not 'defense'."""
    refs = _defense_unit(tmp_path, "u0", "krum_tge")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "config": "Krum+TGE", "seed": 42, "repeat": 0, **refs}])
    units = ave.load_manifest(manifest)
    assert units[0].defense == "krum_tge"


def test_manifest_defense_and_config_agree_ok(tmp_path):
    refs = _defense_unit(tmp_path, "u0", "krum")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "config": "Krum", "seed": 42,
         "repeat": 0, **refs}])
    units = ave.load_manifest(manifest)
    assert units[0].defense == "krum"


def test_manifest_defense_and_config_disagree_raises(tmp_path):
    refs = _defense_unit(tmp_path, "u0", "krum")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "config": "TGE", "seed": 42,
         "repeat": 0, **refs}])
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_manifest(manifest)
    msg = str(ei.value)
    assert "defense" in msg and "config" in msg  # names both keys


def test_manifest_missing_both_defense_and_config_raises(tmp_path):
    refs = _defense_unit(tmp_path, "u0", "krum")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "seed": 42, "repeat": 0, **refs}])  # neither key
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_manifest(manifest)
    assert "defense" in str(ei.value) and "config" in str(ei.value)


def test_manifest_config_key_end_to_end(tmp_path):
    """A fully launch-shaped manifest (config key + repeat key) reduces."""
    entries = []
    for defense_id, (cfg, _tok) in _DEFENSE_IDENTITY.items():
        refs = _defense_unit(tmp_path, f"u_{defense_id}", defense_id)
        entries.append({"arm": "A", "config": cfg, "seed": 42, "repeat": 0, **refs})
    manifest = _write_manifest(tmp_path, entries)
    thresholds = _write_thresholds(tmp_path, {"Krum": 0.5, "TrustScore": 0.5,
                                              "TGE": 0.5, "Krum+TGE": 0.5})
    report = ave.run_from_paths(manifest, thresholds)
    assert {r["defense"] for r in report["replicates"]} == {
        "krum", "trustscore", "tge", "krum_tge"}


# ---------------------------------------------------------------------------
# reused artifact paths across units (every unit is a distinct run)
# ---------------------------------------------------------------------------

def _krum_result(path: Path, seed=42):
    path.write_text(json.dumps({
        "config": "Krum", "final_accuracy": 0.8, "final_f1": 0.75, "seed": seed,
        "provenance": {"scenario_path": "S4_full_mix"}}))


def _krum_signals(path: Path, seed=42):
    _write_jsonl(path, [_row("krum_score", 0.1, True, 1, seed=seed),
                        _row("krum_score", 0.9, False, 1, seed=seed)])


def test_reused_result_path_across_units_raises(tmp_path):
    """Two Arm A repeats pointing at the SAME result file (distinct signals)
    would double-count one run — reject, naming both units and the path."""
    (tmp_path / "shared").mkdir()
    _krum_result(tmp_path / "shared" / "result.json")
    for name in ("s0", "s1"):
        (tmp_path / name).mkdir()
        _krum_signals(tmp_path / name / "signals.jsonl")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "shared/result.json", "signal": "s0/signals.jsonl"},
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 1,
         "result": "shared/result.json", "signal": "s1/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "rep0" in msg and "rep1" in msg and "result" in msg


def test_reused_signal_path_across_units_raises(tmp_path):
    """Two units sharing the SAME signal log (distinct results) — reject."""
    (tmp_path / "shared").mkdir()
    _krum_signals(tmp_path / "shared" / "signals.jsonl")
    for name in ("r0", "r1"):
        (tmp_path / name).mkdir()
        _krum_result(tmp_path / name / "result.json")
    manifest = _write_manifest(tmp_path, [
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 0,
         "result": "r0/result.json", "signal": "shared/signals.jsonl"},
        {"arm": "A", "defense": "krum", "seed": 42, "repeat": 1,
         "result": "r1/result.json", "signal": "shared/signals.jsonl"}])
    thresholds = _write_thresholds(tmp_path, {"krum": 0.5})
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.run_from_paths(manifest, thresholds)
    msg = str(ei.value)
    assert "rep0" in msg and "rep1" in msg and "signal" in msg


def test_distinct_artifact_paths_pass(tmp_path):
    manifest, thresholds = _two_arm_manifest(tmp_path)
    ave.run_from_paths(manifest, thresholds)  # canonical fixtures comply


# ---------------------------------------------------------------------------
# non-finite threshold / anchor values rejected at load
# ---------------------------------------------------------------------------

def test_neg_inf_threshold_rejected_at_load(tmp_path):
    """select_threshold returns -inf on an empty honest population; the freezer
    serializes it as -Infinity. A -inf cut silently reports zero recall — reject
    at load, naming the key."""
    p = tmp_path / "thr.json"
    p.write_text('{"krum": -Infinity}')  # json.loads accepts -Infinity
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_thresholds(p)
    msg = str(ei.value)
    assert "krum" in msg and "finite" in msg


def test_inf_threshold_in_wrapper_rejected_at_load(tmp_path):
    p = tmp_path / "thr.json"
    p.write_text('{"_meta": {}, "thresholds": {"TGE": Infinity}}')
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_thresholds(p)
    assert "finite" in str(ei.value)


def test_nan_anchor_rejected_at_load(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"krum": NaN}')  # json.loads accepts NaN
    with pytest.raises(ave.VarianceEnvelopeError) as ei:
        ave.load_sigma_run_anchors(p)
    assert "krum" in str(ei.value) and "finite" in str(ei.value)


def test_finite_threshold_values_pass_unchanged(tmp_path):
    p = _write_thresholds(tmp_path, {"krum": 0.5, "tge": 0.7})
    assert ave.load_thresholds(p) == {"krum": 0.5, "tge": 0.7}

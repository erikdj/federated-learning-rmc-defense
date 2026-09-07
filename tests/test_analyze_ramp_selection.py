"""Tests for the frozen dev-gate ramp-selection analysis (amendment v1.6 § 3).

The decision rule is pre-registered as EXECUTABLE CODE before any dev-sweep
data exists; these tests pin its semantics:

- Re-blend math matches the deployed TenureGatedDecisionRule exactly.
- Non-active rows are ramp-invariant (logged score passes through).
- Per-ramp threshold is derived ONCE from the pooled honest population
  (v1.3 F3/F8 frozen-threshold convention), then applied per (scenario, seed).
- The switch rule: 5/5 seed sweep AND >= 2 pp mean margin AND cold-start
  guard within 1 pp — otherwise the incumbent 8 locks.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_ramp_selection as ars
from rmc.tg_ensemble import TenureGatedDecisionRule


# ---------------------------------------------------------------------------
# Re-blend
# ---------------------------------------------------------------------------

import itertools

_ROW_SEQ = itertools.count()


def _row(phase="active", gbdt=0.2, lstm=0.9, tge_tenure=4, tge_score=0.5,
         tenure=4, malicious=False, logical_cid=None, server_round=None):
    # unique (server_round, logical_cid) defaults: one signal-log run never
    # repeats the pair, and the analysis loud-fails on duplicates (r6)
    i = next(_ROW_SEQ)
    return {
        "tge_phase": phase, "tge_gbdt_score": gbdt, "tge_lstm_score": lstm,
        "tge_tenure": tge_tenure, "tge_score": tge_score,
        "tenure": tenure, "malicious_gt": malicious,
        "logical_cid": logical_cid if logical_cid is not None else f"synth_{i}",
        "server_round": server_round if server_round is not None else i,
    }


def test_reblend_matches_deployed_gate_rule():
    for ramp in ars.RAMP_CANDIDATES:
        rule = TenureGatedDecisionRule(min_tenure=2, ramp_rounds=ramp)
        for tenure in range(1, 12):
            row = _row(tge_tenure=tenure)
            expected = rule.compute_score(0.2, 0.9, tenure)
            assert ars.reblend_score(row, ramp) == pytest.approx(expected)


def test_reblend_non_active_rows_pass_through_logged_score():
    for phase in ("warmup", "pre_gbdt", "gbdt_only"):
        row = _row(phase=phase, gbdt=None, lstm=None, tge_score=0.42)
        assert ars.reblend_score(row, ramp=3) == pytest.approx(0.42)


def test_reblend_active_row_missing_expert_score_fails_loudly():
    """The v1.6 § 5.4 contract says this cannot happen; if it does, the
    analysis must halt, not silently skip (no-silent-caps discipline)."""
    with pytest.raises(ValueError, match="active"):
        ars.reblend_score(_row(gbdt=None), ramp=5)


# ---------------------------------------------------------------------------
# Filename parsing / scenario keys
# ---------------------------------------------------------------------------

def test_parse_signal_filename():
    parsed = ars.parse_signal_filename(
        "flower_persistent__S4_full_mix__krumtge__seed42.jsonl")
    assert parsed == ("flower_persistent", "S4_full_mix", "krumtge", 42)
    assert ars.parse_signal_filename("not_a_signal_log.jsonl") is None


def test_scenario_key_from_stem():
    assert ars.scenario_key("S4_full_mix") == "S4"
    assert ars.scenario_key("S0_clean_baseline") == "S0"


# ---------------------------------------------------------------------------
# Decision rule (v1.6 § 3.4, frozen verbatim)
# ---------------------------------------------------------------------------

SEEDS = [42, 137, 256, 314, 500]


def _metrics(primary_by_ramp, guard_by_ramp=None):
    """Build {seed: {ramp: value}} dicts where every seed has the same value
    unless a per-seed override list is given."""
    def expand(by_ramp):
        out = {}
        for s_i, seed in enumerate(SEEDS):
            out[seed] = {}
            for ramp, val in by_ramp.items():
                out[seed][ramp] = val[s_i] if isinstance(val, list) else val
        return out
    primary = expand(primary_by_ramp)
    guard = expand(guard_by_ramp if guard_by_ramp is not None else primary_by_ramp)
    return primary, guard


def test_retains_incumbent_on_flat_curve():
    primary, guard = _metrics({r: 0.80 for r in ars.RAMP_CANDIDATES})
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 8 and verdict["switched"] is False


def test_switches_on_perfect_sweep_with_margin_and_guard():
    primary, guard = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES}, 5: 0.83},   # +3 pp, all seeds
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 5 and verdict["switched"] is True


def test_retains_incumbent_when_sweep_is_4_of_5():
    primary, guard = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES},
         5: [0.83, 0.83, 0.83, 0.83, 0.79]},                    # loses seed 500
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 8 and verdict["switched"] is False


def test_retains_incumbent_when_margin_below_2pp():
    primary, guard = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES}, 5: 0.815},   # +1.5 pp only
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 8 and verdict["switched"] is False


def test_retains_incumbent_when_coldstart_guard_violated():
    primary, _ = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES}, 5: 0.83},
    )
    _, guard = _metrics(
        {**{r: 0.90 for r in ars.RAMP_CANDIDATES}, 5: 0.885},   # -1.5 pp cold-start
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 8 and verdict["switched"] is False


def test_multiple_qualifiers_largest_margin_wins_then_closest_to_8():
    primary, guard = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES}, 5: 0.83, 6: 0.85},
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 6

    primary, guard = _metrics(
        {**{r: 0.80 for r in ars.RAMP_CANDIDATES}, 4: 0.83, 7: 0.83},  # exact tie
    )
    verdict = ars.decide(primary, guard, seeds=SEEDS)
    assert verdict["selected"] == 7  # closest to the incumbent


# ---------------------------------------------------------------------------
# End-to-end on synthetic logs
# ---------------------------------------------------------------------------

def _write_log(dirpath: Path, scenario: str, seed: int, rows: list[dict]):
    p = dirpath / f"flower_persistent__{scenario}__krumtge__seed{seed}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _synthetic_rows(n_honest=30, n_malicious=10, separable_at=None):
    """Active-phase rows across tenures 1..10. Honest scores high, malicious
    low; `separable_at` widens the honest/malicious gap in LSTM scores so
    LSTM-heavier ramps (smaller r) score strictly better."""
    rows = []
    for i in range(n_honest):
        tenure = (i % 10) + 1
        rows.append(_row(gbdt=0.75 + (i % 5) * 0.02, lstm=0.9,
                         tge_tenure=tenure, tenure=tenure, malicious=False))
    for i in range(n_malicious):
        tenure = (i % 10) + 1
        lstm = 0.05 if separable_at == "lstm" else 0.5
        gbdt = 0.5 if separable_at == "lstm" else 0.1
        rows.append(_row(gbdt=gbdt, lstm=lstm,
                         tge_tenure=tenure, tenure=tenure, malicious=True))
    return rows


def test_end_to_end_flat_case_retains_8(tmp_path):
    """Malicious separable in BOTH experts equally -> every ramp achieves the
    same recall -> incumbent retained."""
    rows = _synthetic_rows(separable_at=None)
    # make malicious clearly separable under any blend
    for r in rows:
        if r["malicious_gt"]:
            r["tge_gbdt_score"], r["tge_lstm_score"] = 0.05, 0.05
    seeds = [42, 137]
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        for seed in seeds:
            _write_log(tmp_path, scenario, seed, rows)

    result = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                              mode="flower_persistent", seeds=seeds,
                              scenarios=("S2", "S3", "S4"))
    assert result["verdict"]["selected"] == 8
    assert result["verdict"]["switched"] is False
    # curve is reported for every candidate (no silent truncation)
    assert sorted(result["curve"].keys()) == sorted(str(r) for r in ars.RAMP_CANDIDATES)


def test_end_to_end_missing_seed_fails_loudly(tmp_path):
    _write_log(tmp_path, "S4_full_mix", 42, _synthetic_rows())
    with pytest.raises(ars.RampAnalysisError, match="missing"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42, 137])


def test_unregistered_scenario_files_cannot_move_the_verdict(tmp_path):
    """v1.6 § 3.2: the selective input is the registered
    scenario set ONLY. A stray same-defense/mode/seed smoke log in the shared
    signals/ directory must be ignored (and counted) — its honest scores must
    not enter the pooled 10%-FPR threshold, or the pre-registered verdict
    stops being a pure function of the registered dev logs."""
    rows = _synthetic_rows()
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)
    baseline = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                                mode="flower_persistent", seeds=[42],
                                scenarios=("S2", "S3", "S4"))

    # stray smoke log: same defense/mode/seed, wildly different honest scores
    stray = [_row(gbdt=0.001 * i, lstm=0.001 * i, tge_tenure=1, tenure=1,
                  malicious=False) for i in range(50)]
    _write_log(tmp_path, "rmc_intensity_9_continuous_v2", 42, stray)

    result = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                              mode="flower_persistent", seeds=[42],
                              scenarios=("S2", "S3", "S4"))
    assert result["meta"]["n_files_ignored_unregistered"] == 1
    for r in ars.RAMP_CANDIDATES:
        assert result["curve"][str(r)]["threshold"] == pytest.approx(
            baseline["curve"][str(r)]["threshold"])
    assert result["verdict"] == baseline["verdict"]


def test_conflicting_stems_for_same_scenario_key_fail_loudly(tmp_path):
    """Two different stems mapping to the same S-key (e.g. S4_full_mix and a
    S4 smoke variant) must not silently merge into one cell."""
    rows = _synthetic_rows()
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)
    _write_log(tmp_path, "S4_smoke_variant", 42, rows)
    with pytest.raises(ars.RampAnalysisError, match="stem"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42],
                         scenarios=("S2", "S3", "S4"))


def test_appended_rerun_with_mixed_timestamps_fails_loudly(tmp_path):
    """SignalLogger opens files in append mode and a re-run of the same unit
    writes the same filename, so a cell can contain TWO runs' rows — which
    would overweight that cell in the pooled threshold and per-cell recall
    with no loud failure. Mixed run_started_at values are
    rejected."""
    rows = _synthetic_rows()
    appended = ([dict(r, run_started_at="2026-07-11T01:00:00Z") for r in rows]
                + [dict(r, run_started_at="2026-07-11T09:00:00Z") for r in rows])
    _write_log(tmp_path, "S2_adaptive", 42, appended)
    for scenario in ("S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, _synthetic_rows())
    with pytest.raises(ars.RampAnalysisError, match="run_started_at"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42],
                         scenarios=("S2", "S3", "S4"))


def test_duplicate_round_client_rows_fail_loudly(tmp_path):
    """Backstop for duplicates that share a timestamp (or lack the field):
    one run never repeats a (server_round, logical_cid) pair."""
    rows = _synthetic_rows()
    _write_log(tmp_path, "S2_adaptive", 42, rows + rows)  # verbatim append
    for scenario in ("S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, _synthetic_rows())
    with pytest.raises(ars.RampAnalysisError, match="duplicate"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42],
                         scenarios=("S2", "S3", "S4"))


def test_default_gate_run_requires_full_registered_grid(tmp_path):
    """The gate execution path (no scenario narrowing) must refuse to produce
    a verdict unless every registered S0–S4 × seed cell is present — the
    pooled threshold is only deterministic on the full registered input."""
    rows = _synthetic_rows()
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)
    with pytest.raises(ars.RampAnalysisError, match="missing"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42])


def test_descriptive_cell_without_malicious_rows_is_tolerated(tmp_path):
    """The verdict depends only on the primary S2/S3/S4 cells (v1.6 § 3.4);
    a descriptive cell with no malicious rows (e.g. an honest-control log
    dropped in the same directory) must be reported as n/a, not crash the
    gate. NOTE: the shipped S0/S1 are attack-bearing controls (amendment
    v1.4) — this guards the general descriptive path, not the documented
    dev-sweep inputs."""
    rows = _synthetic_rows()
    honest_only = [r for r in _synthetic_rows() if not r["malicious_gt"]]
    seeds = [42, 137]
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        for seed in seeds:
            _write_log(tmp_path, scenario, seed, rows)
    for seed in seeds:
        _write_log(tmp_path, "S0_clean_baseline", seed, honest_only)

    result = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                              mode="flower_persistent", seeds=seeds,
                              scenarios=("S0", "S2", "S3", "S4"))
    assert result["verdict"]["selected"] in ars.RAMP_CANDIDATES
    cell = result["curve"][str(ars.INCUMBENT_RAMP)]["per_cell_recall_all"]
    assert cell["S0__seed42"] is None  # reported n/a, not silently dropped


def test_threshold_derived_from_honest_scores_only(tmp_path):
    """v1.6 § 3.3.2: the per-ramp 10%-FPR threshold comes from the HONEST
    population only (the F3/F8 h2_threshold_pipeline convention). Moving the
    malicious scores around must not move the threshold — otherwise the
    cutoff choice leaks the very labels the ramp curve is scored on."""
    from h2_threshold_pipeline import select_threshold

    def rows_with_malicious_at(mal_score):
        rows = []
        for i in range(20):
            rows.append(_row(gbdt=0.30 + i * 0.02, lstm=0.30 + i * 0.02,
                             tge_tenure=1, tenure=1, malicious=False))
        for i in range(5):
            rows.append(_row(gbdt=mal_score, lstm=mal_score,
                             tge_tenure=1, tenure=1, malicious=True))
        return rows

    thresholds = {}
    for variant, mal_score in (("near", 0.31), ("far", 0.01)):
        d = tmp_path / variant
        d.mkdir()
        for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
            _write_log(d, scenario, 42, rows_with_malicious_at(mal_score))
        result = ars.run_analysis(signals_dir=d, defense="krumtge",
                                  mode="flower_persistent", seeds=[42],
                                  scenarios=("S2", "S3", "S4"))
        thresholds[variant] = {r: result["curve"][str(r)]["threshold"]
                               for r in ars.RAMP_CANDIDATES}

    assert thresholds["near"] == thresholds["far"]
    # and the value is exactly the honest-only convention (tenure=1 rows are
    # pure cold-start under every ramp, so the honest score set is known)
    honest = [0.30 + i * 0.02 for i in range(20)] * 3  # 3 scenarios
    expected = select_threshold(honest, ars.TARGET_FPR)
    assert thresholds["near"][8] == pytest.approx(expected)


def test_zero_recall_candidate_is_reported_not_fatal(tmp_path):
    """A ramp under which no malicious row falls below the honest-only
    threshold is a legitimate zero-recall result, not an error — the gate
    must produce a verdict (retain 8 on a flat-zero curve), not abort."""
    rows = []
    for i in range(20):
        rows.append(_row(gbdt=0.30 + i * 0.02, lstm=0.30 + i * 0.02,
                         tge_tenure=(i % 10) + 1, tenure=(i % 10) + 1,
                         malicious=False))
    for i in range(5):  # malicious score ABOVE every honest score
        rows.append(_row(gbdt=0.99, lstm=0.99,
                         tge_tenure=(i % 10) + 1, tenure=(i % 10) + 1,
                         malicious=True))
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)

    result = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                              mode="flower_persistent", seeds=[42],
                              scenarios=("S2", "S3", "S4"))
    assert result["verdict"]["selected"] == 8
    assert result["verdict"]["switched"] is False
    assert all(v == 0.0 for v in result["verdict"]["mean_primary"].values())


def test_upstream_filtered_rows_excluded_and_counted(tmp_path):
    """Krum+TGE rows for clients filtered BEFORE TGE scored them carry
    all-null TGE fields (truthful cid-keyed join, methodology v1.17). They
    are ramp-invariant — never TGE-scored under any candidate — so they are
    excluded from the re-blend identically for every ramp, but COUNTED and
    reported, never silently dropped."""
    rows = _synthetic_rows()
    for i in range(4):  # upstream-filtered: no TGE involvement at all
        rows.append({"tge_phase": None, "tge_gbdt_score": None,
                     "tge_lstm_score": None, "tge_tenure": None,
                     "tge_score": None, "tenure": (i % 10) + 1,
                     "malicious_gt": i % 2 == 0})
    for scenario in ("S2_adaptive", "S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)

    result = ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                              mode="flower_persistent", seeds=[42],
                              scenarios=("S2", "S3", "S4"))
    assert result["meta"]["n_rows_not_tge_scored"] == 12  # 4 rows x 3 scenarios
    assert result["verdict"]["selected"] in ars.RAMP_CANDIDATES


def test_primary_cell_without_malicious_rows_still_fails_loudly(tmp_path):
    """The loud-fail contract is unchanged where it matters: a PRIMARY cell
    with no malicious ground truth means the dev sweep is broken."""
    honest_only = [r for r in _synthetic_rows() if not r["malicious_gt"]]
    rows = _synthetic_rows()
    _write_log(tmp_path, "S2_adaptive", 42, honest_only)
    for scenario in ("S3_identity_reset", "S4_full_mix"):
        _write_log(tmp_path, scenario, 42, rows)
    with pytest.raises(ars.RampAnalysisError, match="malicious"):
        ars.run_analysis(signals_dir=tmp_path, defense="krumtge",
                         mode="flower_persistent", seeds=[42],
                         scenarios=("S2", "S3", "S4"))

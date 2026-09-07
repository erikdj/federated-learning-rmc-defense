"""H3 Step 6 — τ calibration on the two attack-free controls ONLY.

Design authority:
* v1.10 § 5.0 **D2** — "H3-A — existing `rmc/scenarios/control_honest.json` +
  `control_benign_churn.json`, **after a named freshness/integrity verify**
  (current-runner-compatible; zero malicious clients — G13). **S0/S1 are
  attack-bearing and are forbidden for calibration.**"
* v1.10 § 5.1 — "**BOTH τ values locked in code** (all-device validation-τ AND
  even-device adjudicating-τ, both derived from the calibration logs **before
  any eval scenario runs**, gate (c))"; D9 axis (ii) device hold-out.
* `docs/harness/architecture.md` "Threshold τ calibration" — within-vs-across
  distributions, τ at the dev-FPR=1% point, then LOCKED.

The allowlist refusal is a **pre-registration property**, not a convenience
check: calibrating on an attack-bearing scenario would silently move the
decision boundary onto data the metric is later scored against.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_fp_threshold import (
    ALLOWED_CALIBRATION_SCENARIOS,
    DEV_SEEDS,
    FPR_TARGET,
    PAIR_SAMPLING_SEED,
    CalibrationRefusal,
    build_cohort_calibration,
    calibrate,
    load_observations,
    main,
    select_tau,
    verify_scenarios_attack_free,
)
from flowerfl.fingerprint_registry import CalibrationCohort, MahalanobisMetric

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIM = 6


# ---------------------------------------------------------------------------
# Synthetic calibration corpus
# ---------------------------------------------------------------------------

def _observations(partitions, rounds=12, spread=0.05, gap=6.0, seed=0, scenario="control_honest"):
    """One well-separated cluster per device; `spread` is the within-device σ."""
    rng = np.random.default_rng(seed)
    rows = []
    for partition in partitions:
        centre = np.zeros(DIM)
        centre[0] = gap * partition
        for server_round in range(1, rounds + 1):
            rows.append(
                {
                    "run_id": f"run-{scenario}-{partition}",
                    "scenario": scenario,
                    "seed": 42,
                    "server_round": server_round,
                    "logical_id": f"client_{partition}",
                    "fingerprint": (centre + rng.normal(0, spread, DIM)).tolist(),
                }
            )
    return rows


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _full_corpus(partitions, rounds=6):
    """The complete pre-registered calibration design: both controls x 5 seeds."""
    rows = []
    for scenario in ALLOWED_CALIBRATION_SCENARIOS:
        for seed_index, seed in enumerate(DEV_SEEDS):
            for row in _observations(
                partitions, rounds=rounds, seed=seed_index, scenario=scenario
            ):
                rows.append(dict(row, seed=seed))
    return rows


# ---------------------------------------------------------------------------
# The pre-registered allowlist
# ---------------------------------------------------------------------------

def test_allowlist_is_exactly_the_two_attack_free_controls():
    assert ALLOWED_CALIBRATION_SCENARIOS == ("control_benign_churn", "control_honest")


def test_dev_seeds_and_fpr_target_are_the_pre_registered_values():
    assert DEV_SEEDS == (42, 137, 256, 314, 500)
    assert FPR_TARGET == 0.01


@pytest.mark.parametrize(
    "scenario",
    [
        "S0_clean_baseline",
        "S1_reset_no_attack",
        "S3_identity_reset_only",
        "S4_full_mix",
        "rmc_intensity_9_v1",
        "control_honest_v2",
    ],
)
def test_any_non_allowlisted_scenario_is_refused(tmp_path, scenario):
    path = _write_jsonl(
        tmp_path / "obs.jsonl", _observations(range(4), scenario=scenario)
    )
    with pytest.raises(CalibrationRefusal, match="scenario"):
        load_observations([path])


def test_a_single_forbidden_row_poisons_the_whole_input(tmp_path):
    rows = _observations(range(4))
    rows.append(dict(rows[0], scenario="S4_full_mix"))
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    with pytest.raises(CalibrationRefusal, match="S4_full_mix"):
        load_observations([path])


def test_non_dev_seeds_are_refused(tmp_path):
    rows = [dict(r, seed=999) for r in _observations(range(4))]
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    with pytest.raises(CalibrationRefusal, match="seed"):
        load_observations([path])


def test_the_allowlist_cannot_be_widened_from_the_command_line(tmp_path):
    """A CLI escape hatch would defeat the pre-registration property."""
    path = _write_jsonl(
        tmp_path / "obs.jsonl", _observations(range(4), scenario="S4_full_mix")
    )
    out = tmp_path / "tau.json"
    base = ["--observations", str(path), "--out", str(out),
            "--feature-selection", "none"]
    for attempt in (
        base + ["--scenario", "S4_full_mix"],
        base + ["--allow-scenario", "S4_full_mix"],
        base + ["--force"],
    ):
        with pytest.raises(SystemExit):
            main(attempt)
    assert not out.exists()


def test_attack_free_verify_accepts_the_two_controls():
    """D2's 'named freshness/integrity verify': zero malicious clients."""
    report = verify_scenarios_attack_free(ALLOWED_CALIBRATION_SCENARIOS)
    for scenario, entry in report.items():
        assert entry["attack_free"] is True, (scenario, entry)
        assert entry["num_attack_entries"] == 0
        assert entry["sha256"]


def test_attack_free_verify_rejects_an_attack_bearing_scenario():
    candidates = [
        p.stem
        for p in sorted((PROJECT_ROOT / "rmc" / "scenarios").glob("*.json"))
        if "control" not in p.stem
    ]
    if not candidates:
        pytest.skip("no attack-bearing scenario present to test against")
    with pytest.raises(CalibrationRefusal, match="not in the allowlist"):
        verify_scenarios_attack_free(candidates[:1])


# ---------------------------------------------------------------------------
# τ selection
# ---------------------------------------------------------------------------

def test_tau_lands_at_the_one_percent_false_positive_point():
    rng = np.random.default_rng(1)
    across = rng.normal(50.0, 5.0, size=20000)
    tau = select_tau(across, fpr_target=0.01)
    realised = float(np.mean(across <= tau))
    assert realised <= 0.01
    assert realised > 0.005, "τ is far more conservative than the 1% target"


def test_tau_selection_is_deterministic():
    rng = np.random.default_rng(2)
    across = rng.normal(10.0, 1.0, size=5000)
    assert select_tau(across, 0.01) == select_tau(across, 0.01)


def test_tau_selection_refuses_an_empty_distribution():
    with pytest.raises(CalibrationRefusal, match="across"):
        select_tau(np.array([]), 0.01)


# ---------------------------------------------------------------------------
# Cohort calibration
# ---------------------------------------------------------------------------

def test_calibration_recovers_a_usable_tau_on_separable_synthetic_data():
    observations = load_observations_from_rows(_observations(range(20)))
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    assert result["tau"] > 0.0
    assert result["realised_fpr"] <= FPR_TARGET
    # With σ_within ≪ device spacing, essentially every within-device pair links.
    assert result["within_link_rate"] > 0.95
    assert result["n_calibration_vectors"] == 20 * 12


def load_observations_from_rows(rows):
    """Helper: rows -> the loader's validated structure, via a temp file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(Path(tmp) / "obs.jsonl", rows)
        return load_observations([path])


def test_adjudicating_cohort_uses_even_partitions_only():
    """D9 axis (ii): odd-partition fingerprints must not reach the even τ/Σ.

    Stated as an invariance: the ODD devices' draws are replaced wholesale and
    every calibrated quantity must be unmoved. (The corpus cannot simply omit
    them — the § 5(1) homogeneity gate refuses a lock it cannot evaluate on
    every device it will score.)
    """
    even = _observations(range(0, 20, 2), seed=3)
    odd_a = _observations(range(1, 20, 2), seed=4)
    odd_b = _observations(range(1, 20, 2), seed=91)

    from_a = build_cohort_calibration(
        load_observations_from_rows(even + odd_a), CalibrationCohort.ADJUDICATING
    )
    from_b = build_cohort_calibration(
        load_observations_from_rows(even + odd_b), CalibrationCohort.ADJUDICATING
    )
    assert from_a["tau"] == from_b["tau"]
    assert from_a["metric"]["precision"] == from_b["metric"]["precision"]
    assert from_a["n_calibration_vectors"] == from_b["n_calibration_vectors"] == 120
    assert sorted(from_a["calibration_partitions"]) == list(range(0, 20, 2))


def test_validation_cohort_uses_every_partition():
    observations = load_observations_from_rows(
        _observations(range(0, 20, 2), seed=3) + _observations(range(1, 20, 2), seed=4)
    )
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    assert result["n_calibration_vectors"] == 240
    assert sorted(result["calibration_partitions"]) == list(range(20))


def test_pair_sampling_is_seeded_and_reproducible():
    assert isinstance(PAIR_SAMPLING_SEED, int)
    observations = load_observations_from_rows(_observations(range(20), rounds=20))
    first = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    second = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    assert first["tau"] == second["tau"]
    assert first["metric"]["precision"] == second["metric"]["precision"]


def test_a_cohort_with_one_device_cannot_be_calibrated():
    observations = load_observations_from_rows(_observations([0]))
    with pytest.raises(CalibrationRefusal, match="across-device"):
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)


def test_a_cohort_with_one_round_per_device_has_no_within_pairs():
    observations = load_observations_from_rows(_observations(range(6), rounds=1))
    with pytest.raises(CalibrationRefusal, match="within-device"):
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)


# ---------------------------------------------------------------------------
# The locked artifact
# ---------------------------------------------------------------------------

def test_calibrate_emits_both_cohorts_and_a_verifiable_artifact(tmp_path):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "fingerprint_tau_locked_v1.json"
    payload = calibrate([path], out)

    assert set(payload["cohorts"]) == {"validation", "adjudicating"}
    for cohort in payload["cohorts"].values():
        assert cohort["tau"] > 0.0
        assert cohort["metric"]["dim"] == DIM
    # 20 partitions x 4 rounds x 2 scenarios x 5 seeds = 800; even half = 400.
    assert payload["cohorts"]["validation"]["n_calibration_vectors"] == 800
    assert payload["cohorts"]["adjudicating"]["n_calibration_vectors"] == 400

    meta = payload["_meta"]
    assert meta["fpr_target"] == FPR_TARGET
    assert meta["allowed_scenarios"] == list(ALLOWED_CALIBRATION_SCENARIOS)
    assert meta["script_sha256"]
    assert meta["scenario_verification"]
    assert set(meta["observed_scenarios"]) <= set(ALLOWED_CALIBRATION_SCENARIOS)

    written = json.loads(out.read_text())
    assert written == payload
    assert payload["artifact_sha256_note"], "the lock snippet must be emitted"


def test_calibrate_refuses_a_single_control_scenario(tmp_path):
    """A nonempty SUBSET is not the pre-registered design.

    τ from `control_honest` alone would never have seen benign churn — and
    `load_observations` cannot catch this, because every row it saw was legal.
    """
    rows = [
        r for r in _full_corpus(range(20), rounds=4) if r["scenario"] == "control_honest"
    ]
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="control_benign_churn"):
        calibrate([path], out)
    assert not out.exists()


def test_calibrate_refuses_a_single_dev_seed(tmp_path):
    """One seed is one draw of the run-to-run variance the 5-seed design averages."""
    rows = [r for r in _full_corpus(range(20), rounds=4) if r["seed"] == 42]
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="missing dev seed"):
        calibrate([path], out)
    assert not out.exists()


def test_calibrate_refuses_a_missing_scenario_seed_cell(tmp_path):
    """Every cell of the design must be populated, not merely every label."""
    rows = [
        r
        for r in _full_corpus(range(20), rounds=4)
        if not (r["scenario"] == "control_benign_churn" and r["seed"] == 314)
    ]
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="cell"):
        calibrate([path], out)
    assert not out.exists()


def test_the_refusal_names_exactly_what_is_missing(tmp_path):
    rows = [
        r
        for r in _full_corpus(range(20), rounds=4)
        if r["scenario"] == "control_honest" and r["seed"] in (42, 137)
    ]
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    with pytest.raises(CalibrationRefusal) as excinfo:
        calibrate([path], tmp_path / "tau.json")
    message = str(excinfo.value)
    assert "control_benign_churn" in message
    assert "256" in message and "314" in message and "500" in message


def test_calibrate_accepts_the_complete_design(tmp_path):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    payload = calibrate([path], tmp_path / "tau.json")
    assert sorted(payload["_meta"]["observed_seeds"]) == list(DEV_SEEDS)
    assert payload["_meta"]["observed_scenarios"] == list(ALLOWED_CALIBRATION_SCENARIOS)


def test_calibrate_refuses_to_overwrite_an_existing_lock(tmp_path):
    """τ is NEVER re-derived after an eval scenario runs (v1.10 § 5.1)."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "fingerprint_tau_locked_v1.json"
    calibrate([path], out)
    with pytest.raises(CalibrationRefusal, match="already"):
        calibrate([path], out)


# ---------------------------------------------------------------------------
# Real-data separability (the construct-validity check for the metric choice)
# ---------------------------------------------------------------------------

EVAL_DATA_DIR = PROJECT_ROOT / "data" / "edge_full_20"


@pytest.mark.skipif(
    not (EVAL_DATA_DIR / "client_0.parquet").exists(),
    reason="data/edge_full_20 parquet not materialised (gitignored)",
)
def test_within_device_metric_separates_real_partitions(tmp_path):
    """End-to-end on REAL data: chunk each partition, calibrate, check separation.

    This is the check that the pooled WITHIN-device scatter (not the total
    covariance) is the right Mahalanobis basis for identity linking. It is a
    dry-run of the τ-calibration procedure, not an H3 result.

    It exercises the metric at `_calibrate_metric`, deliberately BELOW the
    § 5(1) lock gate. The fixture builds each observation from six SEQUENTIAL
    20 000-row chunks, so its within-device scatter carries the partition's real
    temporal drift; the emission contract instead draws M = 100 000 rows WITH
    REPLACEMENT from a fixed pool, whose per-round noise is far smaller. Five
    devices land above τ under the chunk fixture, which is a property of the
    fixture and says nothing about the emission contract — gating this dry-run
    on it would assert something the fixture cannot evidence.
    """
    import pyarrow.parquet as pq

    from flowerfl.fingerprint import compute_fingerprint, load_feature_spec

    spec = load_feature_spec()
    rows = []
    # All twenty base partitions: the § 5(1) gate refuses a lock it cannot
    # evaluate on every device the cohort will score, and a six-device dry-run
    # was never representative of the pre-registered cohort anyway.
    for partition in range(20):
        parquet = pq.ParquetFile(EVAL_DATA_DIR / f"client_{partition}.parquet")
        batches = parquet.iter_batches(batch_size=20_000, columns=list(spec.features))
        for chunk_index, batch in enumerate(batches):
            if chunk_index >= 6:
                break
            rows.append(
                {
                    "run_id": f"real-{partition}",
                    "scenario": "control_honest",
                    "seed": 42,
                    "server_round": chunk_index + 1,
                    "logical_id": f"client_{partition}",
                    "fingerprint": compute_fingerprint(
                        batch.to_pandas(), spec
                    ).tolist(),
                }
            )

    from scripts.calibrate_fp_threshold import (
        INCUMBENT_METRIC,
        SCOREABLE_PARTITIONS,
        _calibrate_metric,
    )

    observations = load_observations_from_rows(rows)
    result = _calibrate_metric(
        observations.vectors,
        list(observations.partitions),
        CalibrationCohort.VALIDATION,
        INCUMBENT_METRIC,
        corpus_vectors=observations.vectors,
        corpus_partitions=observations.partitions,
        scoreable_partitions=SCOREABLE_PARTITIONS[CalibrationCohort.VALIDATION.value],
    )
    assert result["realised_fpr"] <= FPR_TARGET
    assert result["within_link_rate"] > 0.5, (
        "the 180-dim construct does not separate real partitions at the 1%-FPR "
        f"threshold: {result['within_distance_summary']} vs "
        f"{result['across_distance_summary']}"
    )
    assert (
        result["within_distance_summary"]["median"]
        < result["across_distance_summary"]["median"]
    )


def test_main_writes_the_artifact_and_prints_the_lock_snippet(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "fingerprint_tau_locked_v1.json"
    assert main(["--observations", str(path), "--out", str(out),
                 "--feature-selection", "none"]) == 0
    captured = capsys.readouterr().out
    assert "TAU_VALIDATION_ALL_DEVICES" in captured
    assert "TAU_ADJUDICATING_EVEN_DEVICES" in captured
    assert "CALIBRATION_ARTIFACT_SHA256" in captured
    assert out.exists()


# ===========================================================================
# F3 — pair enumeration must not materialise the full combination list
# ===========================================================================
#
# `_pair_distances` previously built `np.array(list(itertools.combinations(...)))`
# in full before subsampling. On the pre-registered EXP-050 corpus (9 795
# observations) that is ~48 M pairs — ~3.2 GB of index array, built twice per
# cohort, for two cohorts, plus a (n_pairs x 180) delta matrix on top. The
# contract below is what the replacement must preserve.

def _reference_pairs(groups, same_group):
    """The old implementation's pair set, in its exact order."""
    import itertools as _itertools

    keys = list(groups)
    return [
        (i, j)
        for i, j in _itertools.combinations(range(len(keys)), 2)
        if (keys[i] == keys[j]) is bool(same_group)
    ]


def _reference_distances(whitened, groups, same_group):
    pairs = _reference_pairs(groups, same_group)
    if not pairs:
        return np.empty(0, dtype=np.float64)
    deltas = np.array([whitened[i] - whitened[j] for i, j in pairs])
    return np.sqrt(np.einsum("ij,ij->i", deltas, deltas))


def _grouped_points(n_devices, per_device, dim=4, seed=7):
    rng = np.random.default_rng(seed)
    groups = [d for d in range(n_devices) for _ in range(per_device)]
    points = rng.normal(size=(len(groups), dim)) + np.array(
        [[10.0 * g] + [0.0] * (dim - 1) for g in groups]
    )
    return points, groups


@pytest.mark.parametrize("same_group", [True, False])
def test_pair_distances_below_the_cap_are_bit_identical_to_full_enumeration(same_group):
    """At or below the cap the behaviour must be EXACTLY as before: all pairs."""
    from scripts.calibrate_fp_threshold import _pair_distances

    points, groups = _grouped_points(6, 5)
    rng = np.random.default_rng(PAIR_SAMPLING_SEED)
    got = _pair_distances(points, groups, same_group=same_group, rng=rng)
    expected = _reference_distances(points, groups, same_group)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("same_group", [True, False])
def test_pair_distances_below_the_cap_consume_no_randomness(same_group):
    """The un-subsampled path must not touch the generator (byte-for-byte reruns)."""
    from scripts.calibrate_fp_threshold import _pair_distances

    points, groups = _grouped_points(6, 5)
    rng = np.random.default_rng(PAIR_SAMPLING_SEED)
    before = rng.bit_generator.state
    _pair_distances(points, groups, same_group=same_group, rng=rng)
    assert rng.bit_generator.state == before


def test_pair_distances_with_fewer_than_two_rows_is_empty():
    from scripts.calibrate_fp_threshold import _pair_distances

    rng = np.random.default_rng(PAIR_SAMPLING_SEED)
    single = np.zeros((1, 4))
    assert _pair_distances(single, [0], same_group=True, rng=rng).size == 0
    assert _pair_distances(single, [0], same_group=False, rng=rng).size == 0


@pytest.mark.parametrize("same_group", [True, False])
def test_pair_distances_above_the_cap_sample_uniquely_and_reproducibly(
    monkeypatch, same_group
):
    """Above the cap: exactly `cap` distinct, self-pair-free, seeded samples."""
    import scripts.calibrate_fp_threshold as mod

    cap = 500
    monkeypatch.setattr(mod, "MAX_PAIRS_PER_POPULATION", cap)
    monkeypatch.setattr(mod, "EXACT_PAIR_ENUMERATION_LIMIT", 2 * cap)

    points, groups = _grouped_points(8, 40)  # 320 rows: 6 240 within, 44 800 across
    first = mod._pair_distances(
        points, groups, same_group=same_group, rng=np.random.default_rng(PAIR_SAMPLING_SEED)
    )
    second = mod._pair_distances(
        points, groups, same_group=same_group, rng=np.random.default_rng(PAIR_SAMPLING_SEED)
    )
    assert first.size == cap
    np.testing.assert_array_equal(first, second)

    # The sample must be drawn from the true qualifying population, with no
    # self-pairs and no repeats. Distances alone cannot prove that, so compare
    # against the reference multiset: every sampled distance must be present in
    # the reference, and no distance may appear more often than it does there.
    reference = _reference_distances(points, groups, same_group)
    ref_values, ref_counts = np.unique(reference, return_counts=True)
    got_values, got_counts = np.unique(first, return_counts=True)
    assert np.all(np.isin(got_values, ref_values))
    allowed = ref_counts[np.searchsorted(ref_values, got_values)]
    assert np.all(got_counts <= allowed), "a pair was sampled more than once"


def test_pair_sampling_above_the_cap_is_approximately_uniform(monkeypatch):
    """A uniform sample must not favour either half of the pair population."""
    import scripts.calibrate_fp_threshold as mod

    cap = 4000
    monkeypatch.setattr(mod, "MAX_PAIRS_PER_POPULATION", cap)
    monkeypatch.setattr(mod, "EXACT_PAIR_ENUMERATION_LIMIT", 2 * cap)

    # Two devices, well separated; the across population is 10 000 pairs.
    points, groups = _grouped_points(2, 100)
    sample = mod._pair_distances(
        points, groups, same_group=False, rng=np.random.default_rng(PAIR_SAMPLING_SEED)
    )
    reference = _reference_distances(points, groups, False)
    assert sample.size == cap
    # Mean of a 40% uniform sample of 10 000 values: well inside 4 standard errors.
    standard_error = float(np.std(reference)) / np.sqrt(cap)
    assert abs(float(np.mean(sample)) - float(np.mean(reference))) < 4 * standard_error


def test_pair_distances_never_materialise_the_full_combination_list(monkeypatch):
    """Structural proof: `itertools.combinations` is not on the pair path at all."""
    import itertools as _itertools

    import scripts.calibrate_fp_threshold as mod

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("full combination list materialised")

    monkeypatch.setattr(_itertools, "combinations", _forbidden)
    points, groups = _grouped_points(6, 8)
    for same_group in (True, False):
        mod._pair_distances(
            points, groups, same_group=same_group, rng=np.random.default_rng(1)
        )


#: 6 000 rows => 17 997 000 pairs. The old implementation built that index array
#: in full — 17.997 M x 2 x 8 B — before filtering it, copying the survivors, and
#: taking their deltas.
_FULL_MATERIALISATION_BYTES = 17_997_000 * 2 * 8


@pytest.mark.parametrize("same_group", [True, False])
@pytest.mark.parametrize("cap, budget_divisor", [(100_000, 8), (None, 2)])
def test_pair_distances_peak_memory_is_far_below_full_materialisation(
    monkeypatch, same_group, cap, budget_divisor
):
    """F3: measure the peak. numpy allocations are visible to `tracemalloc`.

    Two regimes, because the size of the ANSWER is not what is under test:
    * a small cap isolates the algorithm — nothing near O(n^2) may be allocated;
    * the production cap (2 M pairs) shows the real-corpus peak, which must
      still sit under the old implementation's index array alone.
    """
    import tracemalloc

    import scripts.calibrate_fp_threshold as mod

    if cap is not None:
        monkeypatch.setattr(mod, "MAX_PAIRS_PER_POPULATION", cap)
        monkeypatch.setattr(mod, "EXACT_PAIR_ENUMERATION_LIMIT", 2 * cap)

    points, groups = _grouped_points(20, 300, dim=4)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        mod._pair_distances(
            points,
            groups,
            same_group=same_group,
            rng=np.random.default_rng(PAIR_SAMPLING_SEED),
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    used = peak - baseline
    assert used < _FULL_MATERIALISATION_BYTES / budget_divisor, (
        f"peak {used / 1e6:.1f} MB is not a material reduction on the "
        f"{_FULL_MATERIALISATION_BYTES / 1e6:.1f} MB full materialisation"
    )


# ===========================================================================
# EMISSION_CONTRACT § 3.7 — explicit, SIZE-INDEPENDENT degeneracy refusal
# ===========================================================================
#
# Measured consequence of relying on the across-pair tie count instead: under a
# cached (non-varying) fingerprint contract the 10-device adjudicating cohort
# refused correctly, while the 20-device validation cohort produced a LOCKABLE
# tau with no error and `within_link_rate = 1.0` — the most attractive possible
# wrong answer. The tie-count guard fires on `floor(0.01 * n_pairs)`, which is a
# function of cohort size; this guard must not be.

def _cached_contract_rows(partitions, rounds=6, scenario="control_honest", seed=42):
    """The cached-fingerprint contract: one vector per device, repeated."""
    rng = np.random.default_rng(11)
    rows = []
    for partition in partitions:
        vector = (rng.normal(size=DIM) + 10.0 * partition).tolist()
        for server_round in range(1, rounds + 1):
            rows.append(
                {
                    "run_id": f"cached-{partition}",
                    "scenario": scenario,
                    "seed": seed,
                    "server_round": server_round,
                    "logical_id": f"client_{partition}",
                    "fingerprint": list(vector),
                }
            )
    return rows


@pytest.mark.parametrize(
    "cohort", [CalibrationCohort.VALIDATION, CalibrationCohort.ADJUDICATING]
)
def test_cached_fingerprints_refuse_on_both_cohorts(cohort):
    """§ 3.7: the guard must fire on the 20-device cohort too, not just the 10."""
    observations = load_observations_from_rows(_cached_contract_rows(range(20)))
    with pytest.raises(CalibrationRefusal, match="degenerate within-device"):
        build_cohort_calibration(observations, cohort)


def test_the_degeneracy_refusal_names_the_failing_condition():
    observations = load_observations_from_rows(_cached_contract_rows(range(20)))
    with pytest.raises(CalibrationRefusal) as excinfo:
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    message = str(excinfo.value)
    assert "identical" in message
    assert "client_0" in message or "0" in message


def test_a_single_degenerate_device_refuses_the_whole_cohort():
    """The silent partition-8 case: one device whose draw collapsed to a constant."""
    rows = _observations(range(10), rounds=6)
    frozen = rows[8 * 6]["fingerprint"]
    for row in rows:
        if row["logical_id"] == "client_8":
            row["fingerprint"] = list(frozen)
    observations = load_observations_from_rows(rows)
    with pytest.raises(CalibrationRefusal, match="degenerate within-device"):
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)


def test_zero_within_median_refuses_even_with_no_fully_identical_device():
    """The other § 3.7 tell: `within_median == 0.0` with a size-independent bar.

    Six observations per device, five of them identical: 10 of the 15
    within-device pairs are exactly zero, so the within MEDIAN is zero while no
    device is wholly constant.
    """
    rng = np.random.default_rng(5)
    rows = []
    for partition in range(8):
        centre = np.zeros(DIM)
        centre[0] = 6.0 * partition
        repeated = (centre + rng.normal(0, 0.05, DIM)).tolist()
        distinct = (centre + rng.normal(0, 0.05, DIM)).tolist()
        for server_round in range(1, 7):
            rows.append(
                {
                    "run_id": f"halfcached-{partition}",
                    "scenario": "control_honest",
                    "seed": 42,
                    "server_round": server_round,
                    "logical_id": f"client_{partition}",
                    "fingerprint": list(repeated if server_round < 6 else distinct),
                }
            )
    observations = load_observations_from_rows(rows)
    with pytest.raises(CalibrationRefusal, match="within_median"):
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)


def test_calibrate_refuses_a_cached_corpus_and_writes_no_artifact(tmp_path):
    rows = []
    for scenario in ALLOWED_CALIBRATION_SCENARIOS:
        for seed in DEV_SEEDS:
            rows.extend(_cached_contract_rows(range(20), scenario=scenario, seed=seed))
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="degenerate within-device"):
        calibrate([path], out)
    assert not out.exists()


def test_a_healthy_population_is_not_refused_by_the_degeneracy_guard():
    """The guard must not fire on the varying contract — no false refusals."""
    observations = load_observations_from_rows(_observations(range(20)))
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    assert result["within_distance_summary"]["median"] > 0.0


# ===========================================================================
# Addendum A — the PREDECLARED calibration-only metric comparison
# ===========================================================================

def test_the_estimator_declaration_strings_record_the_ratified_status():
    """Addendum A was ratified 2026-08-08 (methodology v1.46).

    The estimator-addenda spec's "On ratification" clause directs exactly this
    flip: the AMENDMENT-REQUIRED marker becomes a citation of the spec, while
    COVARIANCE_ESTIMATOR itself does NOT move — ratification authorises the
    selection, it does not change it. Every artifact this script writes carries
    the string, so a stale marker would make the locked deliverable self-report
    its own estimator as unauthorised.
    """
    from scripts.calibrate_fp_threshold import (
        ADDENDUM_STATUS,
        COVARIANCE_ESTIMATOR,
    )

    assert COVARIANCE_ESTIMATOR == "pooled_within_device"
    assert ADDENDUM_STATUS.startswith("RATIFIED 2026-08-08")
    assert "methodology v1.46" in ADDENDUM_STATUS
    assert "2026-08-08-h3-estimator-addenda.md" in ADDENDUM_STATUS
    # The historical marker survives only as narrated context, never as the
    # artifact's live status claim.
    assert not ADDENDUM_STATUS.startswith("AMENDMENT-REQUIRED")


def test_the_two_candidate_metrics_are_the_pre_declared_pair():
    from scripts.calibrate_fp_threshold import (
        ALTERNATIVE_METRIC,
        CANDIDATE_METRICS,
        INCUMBENT_METRIC,
        METRIC_TIE_MARGIN,
        SIMPLER_METRIC,
    )

    assert CANDIDATE_METRICS == (INCUMBENT_METRIC, ALTERNATIVE_METRIC)
    assert SIMPLER_METRIC == ALTERNATIVE_METRIC
    assert METRIC_TIE_MARGIN == 0.01


def test_both_metrics_are_computed_and_fully_summarised():
    from scripts.calibrate_fp_threshold import (
        ALTERNATIVE_METRIC,
        CANDIDATE_METRICS,
        INCUMBENT_METRIC,
    )

    observations = load_observations_from_rows(_observations(range(20)))
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    comparison = result["metric_comparison"]
    assert set(comparison) == set(CANDIDATE_METRICS)
    for name, summary in comparison.items():
        for field in (
            "tau",
            "realised_fpr",
            "within_link_rate",
            "shrinkage",
            "n_within_pairs",
            "n_across_pairs",
        ):
            assert field in summary, (name, field)
        assert summary["tau"] > 0.0
        assert summary["realised_fpr"] <= FPR_TARGET, (
            f"{name} must be scored at ITS OWN realised-FPR<=0.01 operating point"
        )
    assert comparison[INCUMBENT_METRIC]["metric"]["provenance"].endswith(INCUMBENT_METRIC)
    assert comparison[ALTERNATIVE_METRIC]["shrinkage"] == 1.0


def test_the_alternative_metric_is_shrinkage_to_identity():
    """Addendum A's claim, verified rather than assumed: full shrinkage collapses
    the precision to a multiple of the identity in the within-scale basis."""
    from scripts.calibrate_fp_threshold import ALTERNATIVE_METRIC

    observations = load_observations_from_rows(_observations(range(20)))
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    precision = np.asarray(
        result["metric_comparison"][ALTERNATIVE_METRIC]["metric"]["precision"]
    )
    diagonal = np.diag(precision)
    assert np.allclose(precision, np.diag(diagonal))
    assert np.allclose(diagonal, diagonal[0])


def _correlated_observations(partitions, rounds=12, seed=0):
    """Anisotropic, CORRELATED within-device noise.

    On isotropic noise Ledoit-Wolf legitimately picks shrinkage 1.0 and the two
    candidates coincide exactly — a real property of the estimator, not a bug.
    Distinguishing them therefore needs a within-device scatter that actually
    has off-diagonal structure to estimate.
    """
    rng = np.random.default_rng(seed)
    factor = np.triu(rng.normal(size=(DIM, DIM))) + np.eye(DIM)
    rows = []
    for partition in partitions:
        centre = np.zeros(DIM)
        centre[0] = 6.0 * partition
        for server_round in range(1, rounds + 1):
            noise = rng.normal(0, 0.05, DIM) @ factor
            rows.append(
                {
                    "run_id": f"corr-{partition}",
                    "scenario": "control_honest",
                    "seed": 42,
                    "server_round": server_round,
                    "logical_id": f"client_{partition}",
                    "fingerprint": (centre + noise).tolist(),
                }
            )
    return rows


def test_neither_metric_is_scored_at_the_other_threshold():
    from scripts.calibrate_fp_threshold import ALTERNATIVE_METRIC, INCUMBENT_METRIC

    observations = load_observations_from_rows(
        _correlated_observations(range(20), rounds=8)
    )
    comparison = build_cohort_calibration(
        observations, CalibrationCohort.VALIDATION
    )["metric_comparison"]
    assert comparison[INCUMBENT_METRIC]["shrinkage"] < 1.0, (
        "the two candidates are indistinguishable on this fixture"
    )
    assert comparison[ALTERNATIVE_METRIC]["shrinkage"] == 1.0
    assert comparison[INCUMBENT_METRIC]["tau"] != comparison[ALTERNATIVE_METRIC]["tau"]
    for summary in comparison.values():
        assert summary["realised_fpr"] <= FPR_TARGET


@pytest.mark.parametrize(
    "incumbent_rate, alternative_rate, expected_simpler",
    [
        (0.99, 0.50, False),   # incumbent wins outright
        (0.50, 0.99, True),    # alternative wins outright
        (0.90, 0.895, True),   # inside the tie margin -> simpler
        (0.90, 0.89, True),    # exactly 0.01 apart -> still a tie -> simpler
        (0.90, 0.88, False),   # outside the margin -> incumbent
        (0.90, 0.90, True),    # dead heat -> simpler
    ],
)
def test_the_selection_rule_is_total_and_ties_go_to_the_simpler_metric(
    incumbent_rate, alternative_rate, expected_simpler
):
    from scripts.calibrate_fp_threshold import (
        ALTERNATIVE_METRIC,
        INCUMBENT_METRIC,
        SIMPLER_METRIC,
        select_metric_by_rule,
    )

    outcome = select_metric_by_rule(
        {
            INCUMBENT_METRIC: {"within_link_rate": incumbent_rate},
            ALTERNATIVE_METRIC: {"within_link_rate": alternative_rate},
        }
    )
    expected = SIMPLER_METRIC if expected_simpler else INCUMBENT_METRIC
    assert outcome["selected_metric"] == expected
    assert outcome["rule"]
    assert outcome["reason"]
    assert outcome["within_link_rate_delta"] == pytest.approx(
        incumbent_rate - alternative_rate
    )


def test_the_selected_metric_is_promoted_to_the_cohort_top_level():
    observations = load_observations_from_rows(_observations(range(20)))
    result = build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    winner = result["metric_comparison"][result["selected_metric"]]
    assert result["tau"] == winner["tau"]
    assert result["within_link_rate"] == winner["within_link_rate"]
    assert result["realised_fpr"] == winner["realised_fpr"]
    assert result["metric"] == winner["metric"]


def test_the_losing_metric_is_recorded_never_deleted(tmp_path):
    from scripts.calibrate_fp_threshold import CANDIDATE_METRICS

    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    payload = calibrate([path], tmp_path / "tau.json")
    for cohort in payload["cohorts"].values():
        assert set(cohort["metric_comparison"]) == set(CANDIDATE_METRICS)
        loser = set(CANDIDATE_METRICS) - {cohort["selected_metric"]}
        for name in loser:
            assert cohort["metric_comparison"][name]["tau"] > 0.0


def test_the_artifact_records_the_rule_and_its_outcome_before_any_eval(tmp_path):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    payload = calibrate([path], out)
    meta = payload["_meta"]
    assert meta["candidate_metrics"]
    assert meta["metric_selection_rule"]
    assert "tie" in meta["metric_selection_rule"].lower()
    for cohort in payload["cohorts"].values():
        assert cohort["metric_selection"]["selected_metric"] == cohort["selected_metric"]
        assert cohort["metric_selection"]["reason"]
    assert json.loads(out.read_text()) == payload


def test_the_comparison_runs_inside_each_cohort_independently():
    """D9 axis (ii): no odd-partition fingerprint may enter the adjudicating
    comparison, so adding the odd devices must not move any of its numbers."""
    even = _observations(range(0, 20, 2), seed=3)
    from_even = build_cohort_calibration(
        load_observations_from_rows(even + _observations(range(1, 20, 2), seed=4)),
        CalibrationCohort.ADJUDICATING,
    )
    from_both = build_cohort_calibration(
        load_observations_from_rows(even + _observations(range(1, 20, 2), seed=91)),
        CalibrationCohort.ADJUDICATING,
    )
    assert from_even["selected_metric"] == from_both["selected_metric"]
    for name, summary in from_even["metric_comparison"].items():
        other = from_both["metric_comparison"][name]
        for field, value in summary.items():
            # `scoreable_homogeneity` is the ONE block that reads the odd rows
            # by design (ROOTCAUSE_MEMO § 5(1) — a hold-out device's scatter is
            # invisible to every estimate below). It is diagnostics: nothing in
            # it feeds tau, Sigma or the selection rule.
            if field == "scoreable_homogeneity":
                continue
            assert other[field] == value, f"{name}.{field} moved with the odd rows"


def test_the_cohorts_may_select_different_metrics_independently(tmp_path):
    """Nothing forces one cohort's winner onto the other."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    payload = calibrate([path], tmp_path / "tau.json")
    selected = {
        name: cohort["selected_metric"] for name, cohort in payload["cohorts"].items()
    }
    assert set(selected) == {"validation", "adjudicating"}
    for name in selected.values():
        assert name in payload["_meta"]["candidate_metrics"]


def test_main_prints_the_selected_metric_and_both_taus(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "fingerprint_tau_locked_v1.json"
    assert main(["--observations", str(path), "--out", str(out),
                 "--feature-selection", "none"]) == 0
    captured = capsys.readouterr().out
    from scripts.calibrate_fp_threshold import ALTERNATIVE_METRIC, INCUMBENT_METRIC

    assert INCUMBENT_METRIC in captured
    assert ALTERNATIVE_METRIC in captured
    assert "selected" in captured.lower()
    # the gate-(c) lock snippet must survive unchanged
    assert "TAU_VALIDATION_ALL_DEVICES" in captured
    assert "TAU_ADJUDICATING_EVEN_DEVICES" in captured


# ===========================================================================
# The SCOREABLE-DEVICE homogeneity refusal
# ===========================================================================
# `results/20260814/h3_rootcause/ROOTCAUSE_MEMO.md` § 4: the EXP-050 lock was
# produced from a corpus containing all twenty devices, but the adjudicating fit
# read only the ten EVEN ones — every one of them homogeneous — and reported a
# near-perfect 0.9939 within-link rate. Partitions 7 and 11, whose own
# next-round self-distance is 6.8 tau and 98 000 tau, are ODD: their scatter was
# structurally invisible to the estimator that has to whiten it. The two
# pre-existing degeneracy refusals guard the OPPOSITE direction (too little
# within-device variance) and run only on the calibration population, so neither
# could see it. This gate is § 5(1) of the memo's fix.

def _heterogeneous_corpus(pathological=7, scatter=50.0, partitions=range(20),
                          rounds=8, scenario="control_honest", seed=42):
    """A healthy corpus with ONE device whose within-device scatter dwarfs tau."""
    rows = []
    for row in _observations(partitions, rounds=rounds, seed=11, scenario=scenario):
        rows.append(dict(row, seed=seed))
    rng = np.random.default_rng(99)
    centre = np.zeros(DIM)
    centre[0] = 6.0 * pathological
    for row in rows:
        if row["logical_id"] == f"client_{pathological}":
            row["fingerprint"] = (centre + rng.normal(0, scatter, DIM)).tolist()
    return rows


def test_scoreable_partitions_are_the_pre_registered_sets():
    """VALIDATION scores all twenty devices; ADJUDICATING scores the odd hold-out."""
    from scripts.calibrate_fp_threshold import SCOREABLE_PARTITIONS

    assert SCOREABLE_PARTITIONS[CalibrationCohort.VALIDATION.value] == tuple(range(20))
    assert SCOREABLE_PARTITIONS[CalibrationCohort.ADJUDICATING.value] == tuple(
        range(1, 20, 2)
    )


def test_a_scoreable_device_whose_scatter_dwarfs_tau_refuses_the_lock():
    observations = load_observations_from_rows(_heterogeneous_corpus(pathological=7))
    with pytest.raises(CalibrationRefusal, match="within-device scatter"):
        build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)


def test_the_homogeneity_refusal_names_the_offending_device_and_the_numbers():
    observations = load_observations_from_rows(_heterogeneous_corpus(pathological=11))
    with pytest.raises(CalibrationRefusal) as excinfo:
        build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    message = str(excinfo.value)
    assert "11" in message
    assert "tau" in message or "τ" in message
    # the median and the threshold it exceeded must both be quotable from the
    # message alone — a refusal that says only "some device" is not actionable
    assert "median" in message


def test_the_gate_runs_for_devices_the_cohort_scores_but_never_calibrates_on():
    """The decisive property: partition 7 is ODD, so it is scored and NOT fitted.

    Neither pre-existing refusal can reach it — `_refuse_constant_devices` and
    `_refuse_zero_within_median` both operate on the cohort's CALIBRATION
    population, which for ADJUDICATING is the even partitions only.
    """
    from scripts.calibrate_fp_threshold import _refuse_constant_devices
    from flowerfl.fingerprint_registry import (
        ADJUDICATING_CALIBRATION_PARTITIONS,
        is_calibration_partition,
    )

    assert 7 not in ADJUDICATING_CALIBRATION_PARTITIONS
    assert not is_calibration_partition("client_7", CalibrationCohort.ADJUDICATING)

    observations = load_observations_from_rows(_heterogeneous_corpus(pathological=7))
    keep = [
        index
        for index, logical_id in enumerate(observations.logical_ids)
        if is_calibration_partition(logical_id, CalibrationCohort.ADJUDICATING)
    ]
    calibration_partitions = [observations.partitions[i] for i in keep]
    assert 7 not in calibration_partitions
    # the incumbent guard is silent on this corpus...
    _refuse_constant_devices(
        observations.vectors[keep],
        calibration_partitions,
        CalibrationCohort.ADJUDICATING,
    )
    #...and the new one is not.
    with pytest.raises(CalibrationRefusal, match="within-device scatter"):
        build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)


def test_a_homogeneous_corpus_passes_the_homogeneity_gate():
    observations = load_observations_from_rows(_observations(range(20), rounds=8))
    result = build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    homogeneity = result["metric_comparison"][
        result["selected_metric"]
    ]["scoreable_homogeneity"]
    assert homogeneity["refused"] is False
    assert homogeneity["n_devices_checked"] == 10  # the odd hold-out
    assert homogeneity["max_scoreable_median"] <= result["tau"]


def test_the_gate_reports_the_ratio_against_the_calibration_cohort_max():
    observations = load_observations_from_rows(_observations(range(20), rounds=8))
    result = build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    homogeneity = result["metric_comparison"][
        result["selected_metric"]
    ]["scoreable_homogeneity"]
    assert homogeneity["max_calibration_median"] > 0.0
    for device in homogeneity["scoreable_devices"]:
        assert device["ratio_to_calibration_max"] == pytest.approx(
            device["within_median"] / homogeneity["max_calibration_median"]
        )


def test_the_gate_runs_for_both_candidate_metrics():
    observations = load_observations_from_rows(_observations(range(20), rounds=8))
    result = build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    for summary in result["metric_comparison"].values():
        assert summary["scoreable_homogeneity"]["metric_name"] == summary["metric_name"]
        assert summary["scoreable_homogeneity"]["tau"] == summary["tau"]


def test_the_homogeneity_refusal_writes_no_artifact(tmp_path):
    rows = []
    for scenario in ALLOWED_CALIBRATION_SCENARIOS:
        for seed in DEV_SEEDS:
            rows.extend(
                _heterogeneous_corpus(pathological=7, rounds=4,
                                      scenario=scenario, seed=seed)
            )
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="within-device scatter"):
        calibrate([path], out)
    assert not out.exists()


def test_a_losing_candidate_metric_cannot_veto_a_healthy_lock():
    """Addendum A discards one candidate; a discarded estimator is not a lock.

    Measured on the real 180-dim construct (`_separates_real_partitions`): the
    shrinkage-to-identity candidate puts devices 0 and 1 at ~1.05 x its own tau
    while the selected pooled-within metric is comfortably homogeneous. Refusing
    on the loser would abort a lock whose actual instrument is healthy.
    """
    observations = load_observations_from_rows(_observations(range(20), rounds=8))
    result = build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    selected = result["metric_comparison"][result["selected_metric"]]
    assert selected["scoreable_homogeneity"]["refused"] is False
    # the gate that ran is the SELECTED metric's, promoted to the cohort level
    assert (
        result["scoreable_homogeneity"]["metric_name"] == result["selected_metric"]
    )


def test_both_candidates_homogeneity_reports_reach_the_artifact(tmp_path):
    """The loser is recorded, never deleted — same contract as its summary."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    calibrate([path], out)
    payload = json.loads(out.read_text())
    for cohort in payload["cohorts"].values():
        assert cohort["scoreable_partitions"]
        for summary in cohort["metric_comparison"].values():
            report = summary["scoreable_homogeneity"]
            assert report["refused"] is False
            assert report["offending_partitions"] == []


# ---------------------------------------------------------------------------
# the gate is FAIL-CLOSED on an unmeasurable scored device
# ---------------------------------------------------------------------------
# `_require_complete_calibration_corpus` checks (scenario, seed) cells, NOT
# partition coverage. So a corpus with zero rows for partition 7 would lock
# while the gate silently skipped exactly the device class it exists for. The
# gate therefore refuses rather than recording-and-passing.

def test_a_scoreable_device_absent_from_the_corpus_refuses_the_lock():
    rows = [r for r in _observations(range(20), rounds=8) if r["logical_id"] != "client_7"]
    observations = load_observations_from_rows(rows)
    with pytest.raises(CalibrationRefusal, match="CANNOT BE MEASURED") as excinfo:
        build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    assert "7" in str(excinfo.value)


def test_a_scoreable_device_with_a_single_observation_refuses_the_lock():
    """One draw carries no within-device distance — unmeasurable, not healthy."""
    rows = [
        r
        for r in _observations(range(20), rounds=8)
        if r["logical_id"] != "client_11" or r["server_round"] == 1
    ]
    observations = load_observations_from_rows(rows)
    with pytest.raises(CalibrationRefusal, match="CANNOT BE MEASURED") as excinfo:
        build_cohort_calibration(observations, CalibrationCohort.ADJUDICATING)
    message = str(excinfo.value)
    assert "11" in message
    # the refusal must quote the observation count, not just the device
    assert "1 observation" in message


def test_the_unmeasurable_refusal_fires_for_the_validation_cohort_too():
    rows = [r for r in _observations(range(20), rounds=8) if r["logical_id"] != "client_4"]
    observations = load_observations_from_rows(rows)
    with pytest.raises(CalibrationRefusal, match="CANNOT BE MEASURED") as excinfo:
        build_cohort_calibration(observations, CalibrationCohort.VALIDATION)
    assert "4" in str(excinfo.value)


def test_an_absent_scored_device_writes_no_artifact(tmp_path):
    rows = []
    for scenario in ALLOWED_CALIBRATION_SCENARIOS:
        for seed in DEV_SEEDS:
            rows.extend(
                dict(row, seed=seed)
                for row in _observations(range(20), rounds=4, scenario=scenario)
                if row["logical_id"] != "client_7"
            )
    path = _write_jsonl(tmp_path / "obs.jsonl", rows)
    out = tmp_path / "tau.json"
    with pytest.raises(CalibrationRefusal, match="CANNOT BE MEASURED"):
        calibrate([path], out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# the per-device pair cap must BOUND the allocation
# ---------------------------------------------------------------------------
# Materialising the full triangle before subsampling makes the cap bound
# nothing: 20 000 observations of one device would allocate ~200M pairs to
# then keep 200k of them.

def _one_device_whitened(n_rows, dim=DIM, seed=3):
    from scripts.calibrate_fp_threshold import _whiten

    rng = np.random.default_rng(seed)
    vectors = rng.normal(0, 1.0, size=(n_rows, dim))
    metric = MahalanobisMetric.identity(dim)
    return _whiten(vectors, metric), [0] * n_rows


def test_the_capped_path_never_materialises_the_full_triangle(monkeypatch):
    import scripts.calibrate_fp_threshold as mod

    whitened, partitions = _one_device_whitened(400)  # 79 800 exact pairs
    monkeypatch.setattr(mod, "MAX_WITHIN_PAIRS_PER_SCOREABLE_DEVICE", 500)

    def _explode(*args, **kwargs):
        raise AssertionError("the full pair triangle was materialised")

    monkeypatch.setattr(mod.np, "triu_indices", _explode)
    medians = mod._device_within_medians(
        whitened, partitions, [0], np.random.default_rng(mod.PAIR_SAMPLING_SEED)
    )
    assert medians[0][1] == 500


def test_the_capped_sample_is_deterministic():
    import scripts.calibrate_fp_threshold as mod

    whitened, partitions = _one_device_whitened(400)
    first, second = (
        mod._device_within_medians(
            whitened, partitions, [0],
            np.random.default_rng(mod.PAIR_SAMPLING_SEED),
        )
        for _ in range(2)
    )
    assert first[0] == second[0]


def test_the_capped_sample_estimates_the_exact_median(monkeypatch):
    """The cap bounds cost, not correctness: the gate reads a MEDIAN."""
    import scripts.calibrate_fp_threshold as mod

    whitened, partitions = _one_device_whitened(300)  # 44 850 exact pairs
    exact = mod._device_within_medians(
        whitened, partitions, [0], np.random.default_rng(mod.PAIR_SAMPLING_SEED)
    )[0][0]
    monkeypatch.setattr(mod, "MAX_WITHIN_PAIRS_PER_SCOREABLE_DEVICE", 5_000)
    sampled = mod._device_within_medians(
        whitened, partitions, [0], np.random.default_rng(mod.PAIR_SAMPLING_SEED)
    )[0][0]
    assert sampled == pytest.approx(exact, rel=0.05)


def test_the_exact_path_below_the_cap_is_unchanged():
    import scripts.calibrate_fp_threshold as mod

    whitened, partitions = _one_device_whitened(40)  # 780 exact pairs
    median, n_pairs = mod._device_within_medians(
        whitened, partitions, [0], np.random.default_rng(mod.PAIR_SAMPLING_SEED)
    )[0]
    left, right = np.triu_indices(40, k=1)
    deltas = whitened[left] - whitened[right]
    assert n_pairs == 780
    assert median == pytest.approx(float(np.median(np.linalg.norm(deltas, axis=1))))


# ===========================================================================
# The FROZEN R5 feature-selection rule (draft amendment § 3.1(b))
# ===========================================================================
# `results/20260814/h3_feature_eda/EDA_MEMO.md` § 6.1: surviving dims = NOT dead
# ∧ column not pool-flagged (top-1 second-moment share > 0.5 in ANY device's
# pool) ∧ not lattice-flagged ∧ within-device het ratio ≤ 100. The pool screen is
# distance-free and fingerprint-free; the control screens are degeneracy tests.
# NOTHING in the rule reads a distance, a re-link outcome or an eval scenario.

def _pool_frame(n_rows, column, values):
    import pandas as pd

    data = {column: values}
    return pd.DataFrame(data)


def _write_pool(tmp_path, per_partition_values, column="col_a", extra_columns=None):
    """Write client_{i}.parquet files carrying one screened column each."""
    import pandas as pd

    tmp_path.mkdir(parents=True, exist_ok=True)
    for partition, values in per_partition_values.items():
        frame = pd.DataFrame({column: np.asarray(values, dtype=np.float64)})
        for name, extra in (extra_columns or {}).items():
            frame[name] = np.asarray(extra, dtype=np.float64)
        frame.to_parquet(tmp_path / f"client_{partition}.parquet")
    return tmp_path


def test_the_rule_parameters_are_the_frozen_eda_values():
    from scripts.calibrate_fp_threshold import (
        FEATURE_SELECTION_RULE_ID,
        HET_RATIO_MAX,
        LATTICE_MAX_DISTINCT_LEVELS,
        LATTICE_MIN_RELATIVE_SPREAD,
        POOL_FLAG_TOP1_SHARE_THRESHOLD,
    )

    assert FEATURE_SELECTION_RULE_ID == "R5"
    assert POOL_FLAG_TOP1_SHARE_THRESHOLD == 0.5
    assert HET_RATIO_MAX == 100.0
    assert LATTICE_MAX_DISTINCT_LEVELS == 8
    assert LATTICE_MIN_RELATIVE_SPREAD == 0.5


def test_the_pool_screen_flags_a_column_one_row_dominates(tmp_path):
    """The capture-lottery mechanism: one row carrying >0.5 of the 2nd moment."""
    from scripts.calibrate_fp_threshold import pool_screen

    ordinary = np.ones(1000)
    dominated = np.concatenate([np.ones(999), [1e6]])
    pool = _write_pool(
        tmp_path / "pool",
        {0: ordinary, 1: ordinary},
        column="clean",
        extra_columns={"dominated": dominated},
    )
    report = pool_screen(["clean", "dominated"], pool, partitions=(0, 1))
    assert report["pool_flagged"] == ["dominated"]
    assert report["top1_share"]["dominated"] > 0.5
    assert report["top1_share"]["clean"] < 0.5
    assert report["worst_device"]["dominated"] in (0, 1)


def test_the_pool_screen_flags_on_ANY_device_not_the_average(tmp_path):
    """EDA § 3: http.content_length was caught via partition 3's pool — not even
    the failing device's. One device is enough."""
    from scripts.calibrate_fp_threshold import pool_screen

    pool = _write_pool(
        tmp_path / "pool",
        {
            0: np.ones(1000),
            1: np.ones(1000),
            2: np.concatenate([np.ones(999), [1e6]]),
        },
        column="col_a",
    )
    report = pool_screen(["col_a"], pool, partitions=(0, 1, 2))
    assert report["pool_flagged"] == ["col_a"]
    assert report["worst_device"]["col_a"] == 2


def test_the_pool_screen_refuses_a_missing_partition_parquet(tmp_path):
    from scripts.calibrate_fp_threshold import pool_screen

    pool = _write_pool(tmp_path / "pool", {0: np.ones(10)}, column="col_a")
    with pytest.raises(CalibrationRefusal, match="pool parquet"):
        pool_screen(["col_a"], pool, partitions=(0, 1))


def test_the_control_screens_reproduce_the_eda_definitions():
    from scripts.calibrate_fp_threshold import control_screens

    rng = np.random.default_rng(3)
    # dim 0: dead everywhere. dim 1: healthy. dim 2: a 3-level lattice on
    # device 1. dim 3: het ratio far above the cap.
    blocks = []
    partitions = []
    for device in range(4):
        rows = 40
        block = np.zeros((rows, 4))
        block[:, 1] = rng.normal(0, 1.0, rows)
        block[:, 2] = rng.normal(0, 1.0, rows)
        block[:, 3] = rng.normal(0, 1.0, rows)
        if device == 1:
            block[:, 2] = rng.integers(0, 3, rows) * 1000.0 + 1.0
            block[:, 3] = rng.normal(0, 1e5, rows)
        blocks.append(block)
        partitions.extend([device] * rows)
    screens = control_screens(np.vstack(blocks), partitions)

    assert screens["dead"][0] is True
    assert screens["dead"][1] is False
    assert screens["lattice_flag"][2] is True
    assert screens["lattice_flag"][1] is False
    assert screens["het_ratio"][3] > 100.0
    assert screens["het_ratio"][1] < 100.0


def test_a_dead_dimension_is_dead_only_when_constant_in_EVERY_device():
    from scripts.calibrate_fp_threshold import control_screens

    rng = np.random.default_rng(9)
    block_a = np.zeros((20, 2))
    block_b = np.zeros((20, 2))
    block_b[:, 0] = rng.normal(0, 1.0, 20)  # varies in device 1 only
    screens = control_screens(
        np.vstack([block_a, block_b]), [0] * 20 + [1] * 20
    )
    assert screens["dead"][0] is False
    assert screens["dead"][1] is True


def _mask_inputs(n_dims=4):
    """A control corpus whose dim 0 is dead and dim 3 is heteroscedastic."""
    rng = np.random.default_rng(17)
    blocks, partitions = [], []
    for device in range(4):
        block = rng.normal(0, 1.0, (40, n_dims))
        block[:, 0] = 0.0
        if device == 1:
            block[:, 3] = rng.normal(0, 1e5, 40)
        blocks.append(block)
        partitions.extend([device] * 40)
    return np.vstack(blocks), partitions


def test_the_mask_is_the_conjunction_of_all_four_screens():
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _mask_inputs(n_dims=6)
    names = ["c0__mean", "c0__std", "c1__mean", "c1__std",
             "c2__mean", "c2__std"]
    pool = {"threshold": 0.5, "pool_flagged": ["c1"],
            "top1_share": {"c0": 0.1, "c1": 0.9, "c2": 0.2},
            "worst_device": {"c0": 0, "c1": 2, "c2": 0}}
    report = build_feature_mask(vectors, partitions, pool, names)

    # dim 0 dead; dims 2,3 belong to pool-flagged column c1 (and dim 3 is ALSO
    # heteroscedastic); dims 1, 4, 5 survive every screen.
    assert report["mask"] == [1, 4, 5]
    by_dim = {d["dim"]: d for d in report["dims"]}
    assert by_dim[0]["dropped_by"] == ["dead"]
    assert by_dim[2]["dropped_by"] == ["pool_flagged"]
    # a dim can fail several screens; recording only the first would
    # misrepresent the rule
    assert by_dim[3]["dropped_by"] == ["pool_flagged", "het_ratio_above_cap"]
    assert by_dim[1]["surviving"] is True
    assert by_dim[1]["dropped_by"] == []


def test_every_dimension_gets_a_recorded_outcome_for_every_screen():
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _mask_inputs()
    names = ["c0__mean", "c0__std", "c1__mean", "c1__std"]
    pool = {"threshold": 0.5, "pool_flagged": [],
            "top1_share": {"c0": 0.1, "c1": 0.2},
            "worst_device": {"c0": 0, "c1": 1}}
    report = build_feature_mask(vectors, partitions, pool, names)
    assert len(report["dims"]) == 4
    for record in report["dims"]:
        for field in ("dim", "name", "column", "dead", "lattice_flag",
                      "lattice_worst_levels", "het_ratio", "pool_flagged",
                      "pool_top1_share", "surviving", "dropped_by"):
            assert field in record, field
    assert report["rule"] == "R5"
    assert report["parameters"]["pool_flag_top1_share_threshold"] == 0.5


def test_the_mask_refuses_when_too_few_dimensions_survive():
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _mask_inputs()
    names = ["c0__mean", "c0__std", "c1__mean", "c1__std"]
    pool = {"threshold": 0.5, "pool_flagged": ["c0", "c1"],
            "top1_share": {"c0": 0.9, "c1": 0.9},
            "worst_device": {"c0": 0, "c1": 1}}
    with pytest.raises(CalibrationRefusal, match="surviv"):
        build_feature_mask(vectors, partitions, pool, names)


def test_the_r4_ablation_is_recorded_beside_the_r5_mask():
    """EDA § 6.1: R4 (pool screen alone) is reported as R5's ablation."""
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _mask_inputs()
    names = ["c0__mean", "c0__std", "c1__mean", "c1__std"]
    pool = {"threshold": 0.5, "pool_flagged": [],
            "top1_share": {"c0": 0.1, "c1": 0.2},
            "worst_device": {"c0": 0, "c1": 1}}
    report = build_feature_mask(vectors, partitions, pool, names)
    # R4 = not dead and not pool-flagged: keeps the heteroscedastic dim 3
    assert report["ablation_R4_mask"] == [1, 2, 3]
    assert report["mask"] == [1, 2]


# ---------------------------------------------------------------------------
# The mask in the PIPELINE: screens -> mask -> fit -> selection -> gate -> lock
# ---------------------------------------------------------------------------

def _masked_pool(tmp_path, flagged_column="col_1"):
    """A pool whose `flagged_column` is dominated by a single row in one device."""
    import pandas as pd

    directory = tmp_path / "pool"
    directory.mkdir(parents=True, exist_ok=True)
    for partition in range(20):
        frame = pd.DataFrame({
            f"col_{index}": np.linspace(1.0, 2.0, 500) for index in range(DIM)
        })
        if partition == 3:
            values = np.ones(500)
            values[0] = 1e6
            frame[flagged_column] = values
        frame.to_parquet(directory / f"client_{partition}.parquet")
    return directory


def _dim_names():
    return [f"col_{index}__mean" for index in range(DIM)]


def test_the_pipeline_fits_sigma_and_tau_on_the_surviving_dims_only(tmp_path):
    observations = load_observations_from_rows(_observations(range(20), rounds=14))
    pool = _masked_pool(tmp_path)
    result = build_cohort_calibration(
        observations,
        CalibrationCohort.ADJUDICATING,
        feature_selection="R5",
        pool_dir=pool,
        dim_names=_dim_names(),
    )
    selection = result["feature_selection"]
    assert selection["rule"] == "R5"
    assert "col_1" in selection["pool_screen"]["pool_flagged"]
    assert selection["n_surviving"] < DIM
    # the fitted metric lives in the SUBSPACE but still accepts full vectors
    metric = MahalanobisMetric.from_dict(result["metric"])
    assert metric.dim == selection["n_surviving"]
    assert metric.expected_input_dim == DIM
    assert list(metric.mask) == selection["mask"]


def test_the_homogeneity_gate_runs_on_the_MASKED_metric(tmp_path):
    """The whole pipeline order: the § 5(1) gate must see the masked geometry,
    not the full-width one it would have refused."""
    observations = load_observations_from_rows(_observations(range(20), rounds=14))
    pool = _masked_pool(tmp_path)
    result = build_cohort_calibration(
        observations, CalibrationCohort.ADJUDICATING,
        feature_selection="R5", pool_dir=pool, dim_names=_dim_names(),
    )
    homogeneity = result["scoreable_homogeneity"]
    assert homogeneity["refused"] is False
    assert homogeneity["n_devices_checked"] == 10
    # the gate's tau is the masked fit's tau, not some full-width leftover
    assert homogeneity["tau"] == result["tau"]


def test_a_dimension_the_rule_drops_cannot_move_a_locked_distance(tmp_path):
    """End-to-end statement of the partitions-7/11 fix."""
    observations = load_observations_from_rows(_observations(range(20), rounds=14))
    pool = _masked_pool(tmp_path)
    result = build_cohort_calibration(
        observations, CalibrationCohort.ADJUDICATING,
        feature_selection="R5", pool_dir=pool, dim_names=_dim_names(),
    )
    metric = MahalanobisMetric.from_dict(result["metric"])
    dropped = [d["dim"] for d in result["feature_selection"]["dims"]
               if not d["surviving"]]
    assert dropped, "the fixture must drop at least one dimension"
    a = np.zeros(DIM)
    b = np.zeros(DIM)
    b[dropped[0]] = 1e12
    assert metric.distance(a, b) == 0.0


def test_the_lock_artifact_records_the_mask_and_every_screen(tmp_path):
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    calibrate([path], out, feature_selection="R5",
              pool_dir=_masked_pool(tmp_path), dim_names=_dim_names())
    payload = json.loads(out.read_text())

    assert payload["_meta"]["feature_selection_rule"] == "R5"
    assert payload["_meta"]["feature_selection_parameters"][
        "pool_flag_top1_share_threshold"] == 0.5
    for cohort in payload["cohorts"].values():
        selection = cohort["feature_selection"]
        assert len(selection["dims"]) == DIM
        assert selection["mask"] == sorted(selection["mask"])
        assert selection["mask_names"] == [
            selection["dims"][i]["name"] for i in selection["mask"]
        ]
        assert selection["ablation_R4_n_surviving"] >= selection["n_surviving"]
        # the metric written into the lock carries the mask
        assert cohort["metric"]["mask"] == selection["mask"]
        assert cohort["metric"]["input_dim"] == DIM


def test_the_locked_masked_metric_round_trips_into_a_registry(tmp_path):
    """Gate (c) end state: the artifact a registry actually loads."""
    from flowerfl.fingerprint_registry import FingerprintRegistry

    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    payload = calibrate(
        [path], out, feature_selection="R5",
        pool_dir=_masked_pool(tmp_path), dim_names=_dim_names()
    )
    cohort = payload["cohorts"][CalibrationCohort.ADJUDICATING.value]
    metric = MahalanobisMetric.from_dict(cohort["metric"])
    registry = FingerprintRegistry(tau=cohort["tau"], metric=metric, dim=DIM)
    vector = np.random.default_rng(5).normal(size=DIM)
    registry.observe("client_1", vector, 1, logical_id="client_1")
    result = registry.observe("client_1_new1", vector, 3, logical_id="client_1_new1")
    assert result.assertion is not None


def test_the_feature_selection_declaration_has_no_default(tmp_path):
    """Acquiring or losing the R5 mask by forgetting a flag would change the
    instrument silently — the declaration is REQUIRED, exactly like fp-cohort's
    post-lock refusal."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    with pytest.raises(SystemExit):
        main(["--observations", str(path), "--out", str(out)])
    assert not out.exists()


def test_an_unmasked_lock_records_the_absence_explicitly(tmp_path):
    """`none` is a declaration, not a gap: the artifact says so in as many words."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    payload = calibrate([path], out, feature_selection="none")
    assert payload["_meta"]["feature_selection_rule"] is None
    assert payload["_meta"]["pool_screen_dir"] is None
    for cohort in payload["cohorts"].values():
        assert cohort["feature_selection"]["rule"] is None
        assert "mask" not in cohort["metric"]   # byte-identical to the v1 shape


def test_an_unknown_feature_selection_is_refused():
    observations = load_observations_from_rows(_observations(range(20), rounds=14))
    with pytest.raises(CalibrationRefusal, match="feature selection"):
        build_cohort_calibration(
            observations, CalibrationCohort.VALIDATION, feature_selection="R9"
        )


# ---------------------------------------------------------------------------
# the lock artifact must be VALID JSON
# ---------------------------------------------------------------------------
# Dead dims carry het_ratio = inf, and `json.dumps` writes a bare `Infinity`,
# which is not JSON. Any strict parser doing custody or lock verification
# rejects the artifact outright.

def test_a_non_finite_het_ratio_serialises_as_null():
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _mask_inputs(n_dims=6)
    names = ["c0__mean", "c0__std", "c1__mean", "c1__std", "c2__mean", "c2__std"]
    pool = {"threshold": 0.5, "pool_flagged": [],
            "top1_share": {"c0": 0.1, "c1": 0.2, "c2": 0.2},
            "worst_device": {"c0": 0, "c1": 1, "c2": 0}}
    report = build_feature_mask(vectors, partitions, pool, names)
    dead = next(d for d in report["dims"] if d["dead"])
    assert dead["het_ratio"] is None
    # the dead outcome itself is untouched — it is the screen that dropped it
    assert dead["dropped_by"] == ["dead"]
    assert dead["surviving"] is False


def test_the_lock_artifact_parses_under_a_STRICT_json_parser(tmp_path):
    """`parse_constant` fires on Infinity/-Infinity/NaN, which json.loads
    otherwise accepts silently — the artifact must contain none of them."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    calibrate([path], out, feature_selection="R5",
              pool_dir=_masked_pool(tmp_path), dim_names=_dim_names())

    def _reject(value):
        raise AssertionError(f"artifact contains non-JSON constant {value!r}")

    payload = json.loads(out.read_text(), parse_constant=_reject)
    assert payload["_meta"]["feature_selection_rule"] == "R5"


def test_the_writer_refuses_to_emit_a_non_finite_value(tmp_path):
    """Backstop: allow_nan=False makes this whole class fail LOUDLY at write
    time rather than producing an artifact no strict reader can load."""
    import scripts.calibrate_fp_threshold as mod

    with pytest.raises(ValueError):
        json.dumps({"x": float("inf")}, allow_nan=False)
    assert "allow_nan=False" in Path(mod.__file__).read_text()


# ---------------------------------------------------------------------------
# The pool screen must not overflow into a silent non-flag
# ---------------------------------------------------------------------------
# `values * values` overflows float64 above ~1.3e154 (squaring lands past
# 1.8e308), which the raw-scale tcp.payload family reaches. It produces inf/NaN. The resulting NaN share never compares greater
# than the initialised 0.0, so `if share > top1[column]` is False and the column
# is left UNFLAGGED — recorded in the candidate artifact as
# pool_top1_share = 0.0, i.e. indistinguishable from a healthy column. The only
# tell was a RuntimeWarning in the calibration log.
#
# top1_share is scale-INVARIANT (it is a ratio of squares), so normalising by
# the column's max |value| before squaring is exact, not an approximation.

def _extreme_pool(tmp_path, magnitude=1e160, dominated=True):
    import pandas as pd

    directory = tmp_path / "pool"
    directory.mkdir(parents=True, exist_ok=True)
    values = np.full(500, magnitude / 1e6)
    if dominated:
        values[0] = magnitude          # one row carries the second moment
    for partition in (0, 1):
        pd.DataFrame({"huge": values}).to_parquet(
            directory / f"client_{partition}.parquet"
        )
    return directory


def test_an_overflowing_magnitude_column_produces_a_FINITE_share(tmp_path):
    from scripts.calibrate_fp_threshold import pool_screen

    report = pool_screen(["huge"], _extreme_pool(tmp_path), partitions=(0, 1))
    share = report["top1_share"]["huge"]
    assert np.isfinite(share)
    assert 0.0 < share <= 1.0


def test_a_dominated_overflowing_column_is_FLAGGED_not_silently_passed(tmp_path):
    """The exact defect: before the fix this column recorded share 0.0."""
    from scripts.calibrate_fp_threshold import pool_screen

    report = pool_screen(["huge"], _extreme_pool(tmp_path), partitions=(0, 1))
    assert report["pool_flagged"] == ["huge"]
    assert report["top1_share"]["huge"] > 0.5


def test_the_extreme_share_matches_the_exact_scale_invariant_value(tmp_path):
    """Normalising before squaring is EXACT, so the answer must equal the one
    computed in a scale where nothing overflows."""
    from scripts.calibrate_fp_threshold import pool_screen

    magnitude = 1e160
    report = pool_screen(
        ["huge"], _extreme_pool(tmp_path, magnitude=magnitude), partitions=(0, 1)
    )
    scaled = np.full(500, 1e-6)
    scaled[0] = 1.0
    squares = scaled * scaled
    expected = float(squares.max() / squares.sum())
    assert report["top1_share"]["huge"] == pytest.approx(expected, rel=1e-12)


def test_an_undominated_extreme_column_is_NOT_flagged(tmp_path):
    """No false flags: magnitude alone is not the defect, concentration is."""
    from scripts.calibrate_fp_threshold import pool_screen

    pool = _extreme_pool(tmp_path, dominated=False)
    report = pool_screen(["huge"], pool, partitions=(0, 1))
    assert report["pool_flagged"] == []
    assert report["top1_share"]["huge"] < 0.5


def test_the_pool_screen_emits_no_numpy_runtime_warning(tmp_path):
    """The overflow's only previous tell was a warning in the log. Make the
    absence of one a test, so a regression cannot hide there again."""
    from scripts.calibrate_fp_threshold import pool_screen

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pool_screen(["huge"], _extreme_pool(tmp_path), partitions=(0, 1))


def test_a_non_finite_share_FAILS_CLOSED_naming_the_column(tmp_path):
    """Fail closed: an unresolvable share must refuse the lock, never read as
    'this column is fine'."""
    import scripts.calibrate_fp_threshold as mod

    pool = _extreme_pool(tmp_path)

    def _poison(values):
        return np.full(values.shape, np.nan)

    original = mod._normalised_squares
    mod._normalised_squares = _poison
    try:
        with pytest.raises(CalibrationRefusal, match="huge") as excinfo:
            mod.pool_screen(["huge"], pool, partitions=(0, 1))
        assert "non-finite" in str(excinfo.value)
    finally:
        mod._normalised_squares = original


# ---------------------------------------------------------------------------
# FIX 3 — the het screen is the same overflow class, but it FAILS OPEN
# ---------------------------------------------------------------------------
# `np.var` overflows for a ~1e160-magnitude dim, so the per-device std is `inf`;
# the ratio is then inf/inf = NaN; and `NaN > HET_RATIO_MAX` is False, so the dim
# is NOT dropped. The artifact recorded `het_ratio: None, surviving: True` — an
# "unknown het" passing a cap it was never evaluated against. That is strictly
# worse than the pool-screen defect, which merely failed to flag.
#
# The mechanism is proven by the same column's scale-INVARIANT moments: skew and
# kurtosis of tcp.payload computed fine (104.66, 1384.2) and were both correctly
# dropped, while mean/std overflowed and both survived.

def _het_corpus(magnitude=1e160, n_dims=4, devices=4, rows=30, seed=23):
    """dim 0 healthy; dim 1 at `magnitude` and wildly heteroscedastic; dim 2 dead;
    dim 3 healthy (keeps the surviving count above the covariance floor so the
    drop-behaviour tests exercise the SCREEN, not MIN_SURVIVING_DIMS)."""
    rng = np.random.default_rng(seed)
    blocks, partitions = [], []
    for device in range(devices):
        block = np.zeros((rows, n_dims))
        block[:, 0] = rng.normal(0.0, 1.0, rows)
        spread = magnitude if device == 1 else magnitude * 1e-9
        block[:, 1] = rng.normal(0.0, spread, rows)
        block[:, 3] = rng.normal(5.0, 1.0, rows)
        blocks.append(block)
        partitions.extend([device] * rows)
    return np.vstack(blocks), partitions


def test_an_overflowing_dim_gets_a_FINITE_het_ratio():
    from scripts.calibrate_fp_threshold import control_screens

    vectors, partitions = _het_corpus()
    screens = control_screens(vectors, partitions)
    assert np.isfinite(screens["het_ratio"][1])
    assert screens["het_ratio"][1] > 100.0


def test_an_overflowing_heteroscedastic_dim_is_DROPPED_not_silently_kept():
    """THE FAIL-OPEN CASE: before the fix this dim recorded het_ratio None and
    surviving True — it passed a cap that was never evaluated."""
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _het_corpus()
    names = ["c0__mean", "c1__mean", "c2__mean", "c3__mean"]
    pool = {"threshold": 0.5, "pool_flagged": [],
            "top1_share": {"c0": 0.1, "c1": 0.1, "c2": 0.1, "c3": 0.1},
            "worst_device": {"c0": 0, "c1": 0, "c2": 0, "c3": 0}}
    report = build_feature_mask(vectors, partitions, pool, names)
    record = next(d for d in report["dims"] if d["dim"] == 1)
    assert record["surviving"] is False
    assert "het_ratio_above_cap" in record["dropped_by"]
    assert record["het_ratio"] is not None


def test_the_het_ratio_is_EXACT_against_the_unscaled_computation():
    """std is scale-EQUIVARIANT and the max/min ratio is scale-INVARIANT, so
    normalising before the variance is exact — same argument as
    `_normalised_squares`. Verified at a magnitude where nothing overflows, so
    the unscaled computation is itself trustworthy."""
    from scripts.calibrate_fp_threshold import control_screens

    vectors, partitions = _het_corpus(magnitude=1.0)
    screens = control_screens(vectors, partitions)

    keys = np.asarray(partitions)
    stds = np.vstack([
        np.sqrt(vectors[keys == device].var(axis=0, ddof=1))
        for device in sorted(set(partitions))
    ])
    for index in (0, 1):
        column = stds[:, index]
        expected = float(column.max() / column[column > 0].min())
        assert screens["het_ratio"][index] == pytest.approx(expected, rel=1e-12)


def test_a_non_finite_per_device_std_FAILS_CLOSED_naming_dim_and_device():
    from scripts.calibrate_fp_threshold import control_screens

    vectors, partitions = _het_corpus(magnitude=1.0)
    vectors[35, 0] = np.nan          # device 1, dim 0
    with pytest.raises(CalibrationRefusal) as excinfo:
        control_screens(vectors, partitions)
    message = str(excinfo.value)
    assert "non-finite" in message
    assert "dim 0" in message
    assert "device 1" in message


def test_the_dead_dim_infinite_path_is_STILL_ALLOWED():
    """A dim that never varies has no finite ratio by definition. That inf is
    documented and is caught by the `dead` screen; it must not be confused with
    an unresolved computation."""
    from scripts.calibrate_fp_threshold import control_screens

    vectors, partitions = _het_corpus(magnitude=1.0)
    screens = control_screens(vectors, partitions)
    assert screens["dead"][2] is True
    assert not np.isfinite(screens["het_ratio"][2])


def test_the_control_screens_emit_no_numpy_runtime_warning():
    from scripts.calibrate_fp_threshold import control_screens

    vectors, partitions = _het_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            control_screens(vectors, partitions)


def test_THE_INVARIANT_no_unresolved_screen_value_may_survive():
    """Structural: a dim whose screen value is null/non-finite has not been
    evaluated against its cap, and an unevaluated dim is not a passed one. This
    is the general form of the defect, asserted over the whole artifact."""
    from scripts.calibrate_fp_threshold import build_feature_mask

    vectors, partitions = _het_corpus()
    names = ["c0__mean", "c1__mean", "c2__mean", "c3__mean"]
    pool = {"threshold": 0.5, "pool_flagged": [],
            "top1_share": {"c0": 0.1, "c1": 0.1, "c2": 0.1, "c3": 0.1},
            "worst_device": {"c0": 0, "c1": 0, "c2": 0, "c3": 0}}
    report = build_feature_mask(vectors, partitions, pool, names)
    for record in report["dims"]:
        unresolved = [
            field for field in ("het_ratio", "pool_top1_share")
            if record[field] is None
            or not np.isfinite(float(record[field] if record[field] is not None else 0))
        ]
        if unresolved:
            assert record["surviving"] is False, (record["dim"], unresolved)


def test_a_full_calibration_emits_no_runtime_warning(tmp_path):
    """The candidate re-run must complete with ZERO RuntimeWarnings."""
    path = _write_jsonl(tmp_path / "obs.jsonl", _full_corpus(range(20), rounds=4))
    out = tmp_path / "tau.json"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        calibrate([path], out, feature_selection="R5",
                  pool_dir=_masked_pool(tmp_path), dim_names=_dim_names())
    assert out.exists()

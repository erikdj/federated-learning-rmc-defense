"""H3 Step 3 — the fingerprint registry (Mahalanobis matching + EMA + generation).

Design authority: v1.10 § 5.1 (the 180-dim fingerprint, Mahalanobis matching,
the EMA α=0.1 registry update, identity binding, and the two-stage τ-calibration
procedure are unchanged from base spec § 6.3; τ **and** the covariance are
calibrated per-cohort — all partitions for VALIDATION, even partitions only for
the ADJUDICATING device hold-out, D9 axis (ii)); `docs/harness/architecture.md`
("Server-side registry", "Mahalanobis distance", "Threshold τ calibration").

Property tests demanded by the execution plan's Step-3 validation column:

    "a device's own fingerprint matches itself under a new CID; two distinct
     partitions do not match at the calibrated τ; EMA converges; generation
     increments on inherited flags; the even-partition covariance is computed
     from even partitions only (asserted by construction, not by comment)."
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from flowerfl import fingerprint_registry as fpr
from flowerfl.fingerprint_registry import (
    ADJUDICATING_CALIBRATION_PARTITIONS,
    EMA_ALPHA,
    ODD_HOLDOUT_PARTITIONS,
    CalibrationCohort,
    FingerprintRegistry,
    MahalanobisMetric,
    MatchAssertion,
    RegistryEntry,
    TauNotLockedError,
    is_calibration_partition,
    is_holdout_partition,
    locked_metric,
    locked_tau,
    partition_of,
)

DIM = 8  # small stand-in for 180 in the synthetic property tests


def _metric(dim: int = DIM) -> MahalanobisMetric:
    return MahalanobisMetric.identity(dim)


def _registry(tau: float = 1.0, dim: int = DIM, **kwargs) -> FingerprintRegistry:
    return FingerprintRegistry(tau=tau, metric=_metric(dim), dim=dim, **kwargs)


def _vec(seed: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


# ---------------------------------------------------------------------------
# Immutability / dataclass contract
# ---------------------------------------------------------------------------

def test_registry_entry_is_frozen():
    entry = RegistryEntry(
        entry_id="e0",
        session_key="client_0",
        flower_cid="cid-0",
        logical_id="client_0",
        fingerprint_vec=np.zeros(DIM),
        first_seen_round=1,
        last_seen_round=1,
    )
    assert dataclasses.is_dataclass(entry)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.generation = 5  # type: ignore[misc]


def test_match_assertion_exposes_the_schema_v5_field_names():
    """The assertion IS the schema-v5 re-link contract (v1.10 § 5.1 table)."""
    fields = {f.name for f in dataclasses.fields(MatchAssertion)}
    assert {
        "asserted_match",
        "asserted_parent_entry_id",
        "asserted_parent_logical_id",
        "min_d",
        "tau",
        "generation",
    } <= fields


# ---------------------------------------------------------------------------
# Mahalanobis metric
# ---------------------------------------------------------------------------

def test_identity_metric_is_euclidean():
    metric = _metric()
    a, b = _vec(1), _vec(2)
    assert np.isclose(metric.distance(a, b), float(np.linalg.norm(a - b)))
    assert metric.distance(a, a) == 0.0


def test_metric_from_population_whitens_anisotropic_data():
    """A direction the honest population varies wildly in must not dominate d."""
    rng = np.random.default_rng(0)
    n = 4000
    population = np.column_stack(
        [rng.normal(0, 100.0, n), rng.normal(0, 0.01, n)]
    )
    metric = MahalanobisMetric.from_population(population)
    centre = population.mean(axis=0)
    # One population-sigma along each axis should give ~the same distance.
    d_wide = metric.distance(centre + np.array([100.0, 0.0]), centre)
    d_narrow = metric.distance(centre + np.array([0.0, 0.01]), centre)
    assert 0.5 < d_wide / d_narrow < 2.0
    #...whereas Euclidean distance would differ by four orders of magnitude.
    assert np.linalg.norm([100.0, 0.0]) / np.linalg.norm([0.0, 0.01]) > 1e3


def test_metric_distance_grows_with_population_sigmas():
    rng = np.random.default_rng(5)
    population = rng.normal(0, 1.0, size=(3000, 4))
    metric = MahalanobisMetric.from_population(population)
    centre = population.mean(axis=0)
    near = metric.distance(centre + np.array([1.0, 0, 0, 0]), centre)
    far = metric.distance(centre + np.array([3.0, 0, 0, 0]), centre)
    assert far > near > 0.0
    assert np.isclose(far / near, 3.0, rtol=0.15)


def test_metric_is_deterministic_and_serialisable():
    rng = np.random.default_rng(9)
    population = rng.normal(size=(500, DIM))
    first = MahalanobisMetric.from_population(population)
    second = MahalanobisMetric.from_population(population)
    assert first.precision.tobytes() == second.precision.tobytes()
    restored = MahalanobisMetric.from_dict(first.to_dict())
    probe_a, probe_b = _vec(21), _vec(22)
    assert np.isclose(
        restored.distance(probe_a, probe_b), first.distance(probe_a, probe_b)
    )


def test_metric_rejects_a_degenerate_population():
    with pytest.raises(ValueError, match="at least"):
        MahalanobisMetric.from_population(np.zeros((1, DIM)))


def test_metric_rejects_dimension_mismatch():
    metric = _metric()
    with pytest.raises(ValueError, match="dim"):
        metric.distance(np.zeros(DIM + 1), np.zeros(DIM + 1))


# ---------------------------------------------------------------------------
# Registry: first appearance, matching, generation inheritance
# ---------------------------------------------------------------------------

def test_first_ever_cid_has_no_flagged_candidates():
    registry = _registry()
    result = registry.observe("cid-a", _vec(1), server_round=1, logical_id="client_0")
    assert result.is_first_appearance is True
    assertion = result.assertion
    assert assertion is not None
    assert assertion.asserted_match is False
    assert assertion.asserted_parent_entry_id is None
    assert assertion.asserted_parent_logical_id is None
    assert assertion.min_d == float("inf")
    assert assertion.generation == 0
    assert registry.entry_for_session("cid-a").flag_status is False


def test_a_device_matches_itself_under_a_new_cid():
    registry = _registry(tau=1.0)
    fingerprint = _vec(1)
    registry.observe("cid-a", fingerprint, server_round=1, logical_id="client_3")
    registry.flag("cid-a", reason="krum", server_round=5)

    result = registry.observe(
        "cid-a-respawn", fingerprint, server_round=9, logical_id="client_3_new1"
    )
    assertion = result.assertion
    assert assertion.asserted_match is True
    assert assertion.min_d == pytest.approx(0.0, abs=1e-12)
    assert assertion.asserted_parent_logical_id == "client_3"
    assert assertion.asserted_parent_entry_id == registry.entry_for_session("cid-a").entry_id
    assert assertion.generation == 1

    respawn = registry.entry_for_session("cid-a-respawn")
    assert respawn.flag_status is True, "flag must be INHERITED on a match"
    assert respawn.flag_reason == "inherited"
    assert respawn.generation == 1
    assert respawn.parent_entry_id == registry.entry_for_session("cid-a").entry_id


def test_two_distinct_devices_do_not_match_at_tau():
    registry = _registry(tau=1.0)
    registry.observe("cid-a", _vec(1), server_round=1, logical_id="client_0")
    registry.flag("cid-a", reason="krum", server_round=2)

    result = registry.observe("cid-b", _vec(2), server_round=3, logical_id="client_1")
    assert result.assertion.asserted_match is False
    assert result.assertion.min_d > 1.0
    assert registry.entry_for_session("cid-b").flag_status is False
    assert registry.entry_for_session("cid-b").generation == 0


def test_only_flagged_entries_are_match_candidates():
    """An UNflagged incumbent must never be asserted as a parent."""
    registry = _registry(tau=1.0)
    fingerprint = _vec(1)
    registry.observe("cid-a", fingerprint, server_round=1, logical_id="client_0")
    # cid-a is never flagged.
    result = registry.observe(
        "cid-a-respawn", fingerprint, server_round=4, logical_id="client_0_new1"
    )
    assert result.assertion.asserted_match is False
    assert result.assertion.min_d == float("inf")


def test_generation_chains_across_successive_respawns():
    registry = _registry(tau=1.0)
    fingerprint = _vec(1)
    registry.observe("cid-g0", fingerprint, server_round=1, logical_id="client_2")
    registry.flag("cid-g0", reason="krum", server_round=2)
    registry.observe("cid-g1", fingerprint, server_round=3, logical_id="client_2_new1")
    result = registry.observe(
        "cid-g2", fingerprint, server_round=5, logical_id="client_2_new2"
    )
    assert result.assertion.generation == 2
    assert registry.entry_for_session("cid-g2").generation == 2
    # Pre-declared tie-break: at d == 0 from BOTH the root and the previous
    # respawn, the immediate predecessor is the asserted parent.
    assert result.assertion.asserted_parent_logical_id == "client_2_new1"


def test_exact_tie_break_is_deterministic_across_insertion_orders():
    """Identity resets copy a partition byte-for-byte, so d == 0 ties are real."""
    fingerprint = _vec(1)

    def run(order):
        registry = _registry(tau=1.0)
        registry.observe("cid-g0", fingerprint, server_round=1, logical_id="client_2")
        registry.flag("cid-g0", reason="krum", server_round=2)
        registry.observe("cid-g1", fingerprint, server_round=3, logical_id="client_2_new1")
        for cid, logical in order:
            registry.observe(cid, _vec(42), server_round=4, logical_id=logical)
        return registry.observe(
            "cid-g2", fingerprint, server_round=5, logical_id="client_2_new2"
        ).assertion

    first = run([("cid-x", "client_8"), ("cid-y", "client_9")])
    second = run([("cid-y", "client_9"), ("cid-x", "client_8")])
    assert first == second


def test_nearest_flagged_entry_wins_when_several_are_candidates():
    registry = _registry(tau=10.0)
    near = np.zeros(DIM)
    far = np.zeros(DIM)
    far[0] = 5.0
    registry.observe("cid-near", near, server_round=1, logical_id="client_4")
    registry.observe("cid-far", far, server_round=1, logical_id="client_5")
    registry.flag("cid-near", reason="krum", server_round=2)
    registry.flag("cid-far", reason="krum", server_round=2)

    probe = np.zeros(DIM)
    probe[0] = 1.0
    result = registry.observe("cid-new", probe, server_round=3, logical_id="client_4_new1")
    assert result.assertion.asserted_parent_logical_id == "client_4"
    assert result.assertion.min_d == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Registry: returning CID / EMA
# ---------------------------------------------------------------------------

def test_returning_cid_updates_ema_and_emits_no_new_assertion():
    registry = _registry()
    registry.observe("cid-a", np.zeros(DIM), server_round=1, logical_id="client_0")
    result = registry.observe("cid-a", np.ones(DIM), server_round=2, logical_id="client_0")
    assert result.is_first_appearance is False
    assert result.assertion is None, (
        "a re-entry event is one row per NEW cid — repeat participation must not "
        "manufacture extra scored events"
    )
    entry = registry.entry_for_session("cid-a")
    assert entry.last_seen_round == 2
    assert entry.first_seen_round == 1
    assert np.allclose(entry.fingerprint_vec, np.full(DIM, EMA_ALPHA))


def test_ema_converges_to_the_repeated_observation():
    registry = _registry()
    registry.observe("cid-a", np.zeros(DIM), server_round=1, logical_id="client_0")
    target = np.ones(DIM)
    for round_index in range(2, 202):
        registry.observe("cid-a", target, server_round=round_index, logical_id="client_0")
    assert np.allclose(registry.entry_for_session("cid-a").fingerprint_vec, target, atol=1e-6)


def test_ema_alpha_is_the_pre_registered_value():
    assert EMA_ALPHA == 0.1


def test_entries_are_replaced_not_mutated():
    registry = _registry()
    registry.observe("cid-a", np.zeros(DIM), server_round=1, logical_id="client_0")
    before = registry.entry_for_session("cid-a")
    registry.observe("cid-a", np.ones(DIM), server_round=2, logical_id="client_0")
    after = registry.entry_for_session("cid-a")
    assert before is not after
    assert np.allclose(before.fingerprint_vec, np.zeros(DIM)), "entry was mutated in place"


def test_flagging_an_unknown_cid_is_a_loud_error():
    registry = _registry()
    with pytest.raises(KeyError):
        registry.flag("nope", reason="krum", server_round=1)


def test_registry_rejects_a_wrong_dimension_fingerprint():
    registry = _registry()
    with pytest.raises(ValueError, match="dim"):
        registry.observe("cid-a", np.zeros(DIM + 3), server_round=1, logical_id="client_0")


def test_registry_rejects_a_non_finite_fingerprint():
    bad = np.zeros(DIM)
    bad[0] = np.nan
    registry = _registry()
    with pytest.raises(ValueError, match="finite"):
        registry.observe("cid-a", bad, server_round=1, logical_id="client_0")


def test_session_key_and_flower_cid_are_never_match_features():
    """Relabelling both keys must not move the decision by one bit.

    The registry may key on the CLAIMED identity and carry the Flower CID for
    audit, but the match itself is a function of the fingerprint and the
    registry's own internal entry ids alone.
    """
    fingerprint = _vec(1)

    def run(parent_session, child_session, parent_cid, child_cid):
        registry = _registry(tau=1.0)
        registry.observe(
            parent_session, fingerprint, server_round=1,
            logical_id="client_0", flower_cid=parent_cid,
        )
        registry.flag(parent_session, reason="krum", server_round=2)
        return registry.observe(
            child_session, fingerprint, server_round=3,
            logical_id="client_0_new1", flower_cid=child_cid,
        ).assertion

    baseline = run("client_0", "client_0_new1", "raw-a", "raw-a")
    relabelled = run("s-aaa", "s-zzz", "opaque-1", "opaque-2")
    assert baseline == relabelled
    assert baseline.asserted_match is True


def test_a_new_session_key_under_the_same_flower_cid_is_a_new_appearance():
    """Flower simulation reuses one raw cid across an identity reset (v1.15)."""
    registry = _registry(tau=1.0)
    fingerprint = _vec(1)
    registry.observe(
        "client_0", fingerprint, server_round=1, logical_id="client_0", flower_cid="raw0"
    )
    registry.flag("client_0", reason="krum", server_round=2)
    result = registry.observe(
        "client_0_new1", fingerprint, server_round=5,
        logical_id="client_0_new1", flower_cid="raw0",
    )
    assert result.is_first_appearance is True
    assert result.assertion.asserted_match is True
    assert len(registry) == 2


def test_observation_is_repeatable_for_the_same_inputs():
    """The registry's assertion must be a pure function of its inputs."""
    fingerprint = _vec(1)

    def run():
        registry = _registry(tau=1.0)
        registry.observe("cid-a", fingerprint, server_round=1, logical_id="client_0")
        registry.flag("cid-a", reason="krum", server_round=2)
        return registry.observe(
            "cid-b", fingerprint, server_round=3, logical_id="client_0_new1"
        ).assertion

    assert run() == run()


# ---------------------------------------------------------------------------
# Device hold-out (D9 axis ii) — parity partition, asserted by construction
# ---------------------------------------------------------------------------

def test_partition_of_agrees_with_the_ground_truth_identity_map():
    """`partition_of` must never diverge from `LOGICAL_TO_PARTITION`, the truth key."""
    from flowerfl.scenario_strategy import ScenarioStrategy

    truth = ScenarioStrategy.LOGICAL_TO_PARTITION
    assert truth, "ground-truth identity map is empty"
    for logical_id, partition in truth.items():
        assert partition_of(logical_id) == partition, logical_id


def test_partition_of_rejects_an_unmappable_identity():
    with pytest.raises(KeyError):
        partition_of("not_a_client")


def test_calibration_and_holdout_partitions_are_the_pre_registered_sets():
    assert ADJUDICATING_CALIBRATION_PARTITIONS == (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
    assert ODD_HOLDOUT_PARTITIONS == (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    assert not set(ADJUDICATING_CALIBRATION_PARTITIONS) & set(ODD_HOLDOUT_PARTITIONS)


@pytest.mark.parametrize(
    "logical_id,expected",
    [("client_0", True), ("client_18", True), ("client_1", False), ("client_7_new3", False)],
)
def test_adjudicating_calibration_admits_even_partitions_only(logical_id, expected):
    assert (
        is_calibration_partition(logical_id, CalibrationCohort.ADJUDICATING) is expected
    )


@pytest.mark.parametrize("logical_id", ["client_0", "client_1", "client_19_new2"])
def test_validation_calibration_admits_every_partition(logical_id):
    assert is_calibration_partition(logical_id, CalibrationCohort.VALIDATION) is True


@pytest.mark.parametrize(
    "logical_id,expected",
    [("client_1", True), ("client_7_new4", True), ("client_0", False), ("client_18", False)],
)
def test_holdout_scoring_admits_odd_partitions_only(logical_id, expected):
    assert is_holdout_partition(logical_id) is expected


def test_calibration_and_holdout_populations_are_disjoint_by_construction():
    """No logical identity may be both calibrated on and adjudicated on."""
    from flowerfl.scenario_strategy import ScenarioStrategy

    for logical_id in ScenarioStrategy.LOGICAL_TO_PARTITION:
        if partition_of(logical_id) > 19:
            continue  # the RMC duplicate partitions (20) are not base devices
        calibrated = is_calibration_partition(logical_id, CalibrationCohort.ADJUDICATING)
        adjudicated = is_holdout_partition(logical_id)
        assert not (calibrated and adjudicated), logical_id


def test_even_partition_metric_is_built_from_even_partitions_only():
    """Asserted by construction: odd-partition rows cannot reach the covariance."""
    from flowerfl.fingerprint_registry import select_calibration_vectors

    records = [
        {"logical_id": f"client_{p}", "fingerprint": np.full(DIM, float(p))}
        for p in range(20)
    ]
    even = select_calibration_vectors(records, CalibrationCohort.ADJUDICATING)
    assert even.shape == (10, DIM)
    assert sorted(float(row[0]) for row in even) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    all_devices = select_calibration_vectors(records, CalibrationCohort.VALIDATION)
    assert all_devices.shape == (20, DIM)


# ---------------------------------------------------------------------------
# τ lock (pre-registration gate c)
# ---------------------------------------------------------------------------

# Gate (c) history: until the EXP-050 τ-lock (2026-08-10) this test asserted
# that every accessor RAISED TauNotLockedError — the pre-lock refusal guard.
# The lock legitimately flips it: the test now pins the LOCKED values, so any
# drift in the committed τ (or the artifact hash it verifies against) fails
# the suite loudly. Same guard, opposite sign.
# v2 lock (2026-08-14, corrected-instrument pre-registration § 3.1): R5-masked
# re-calibration on the same EXP-050 corpus. The v1 values (24.378021816195496 /
# 28.01139899170691) are historical — pinned only in the untouched v1 artifact.
TAU_LOCKED_EXPECTED = {
    CalibrationCohort.VALIDATION: 18.639429816855873,
    CalibrationCohort.ADJUDICATING: 26.466874982783164,
}

#: The covariance estimator Addendum A's predeclared comparison SELECTED in each
#: cohort at EXP-050. Pinned separately from τ because τ and the estimator move
#: together: regenerating the artifact with the losing metric and refreshing
#: CALIBRATION_ARTIFACT_SHA256 alongside it would leave the τ pin satisfied
#: while silently changing every Mahalanobis decision. Asserting the metric name
#: verbatim makes that drift loud.
SELECTED_METRIC_EXPECTED = {
    CalibrationCohort.VALIDATION: "shrinkage_to_identity",
    # v2 lock: Addendum A's rule on the R5-masked corpus produced a TIE in the
    # adjudicating cohort (|delta| 0.000043 ≤ 0.01), which resolves to the
    # SIMPLER metric — the v1 pick was pooled_within_ledoit_wolf.
    CalibrationCohort.ADJUDICATING: "shrinkage_to_identity",
}


@pytest.mark.parametrize("cohort", list(CalibrationCohort))
def test_tau_is_locked_and_pinned_to_the_exp050_calibration(cohort):
    """Gate (c): τ committed to code BEFORE any eval run — now LOCKED.

    locked_tau also verifies the calibration artifact's SHA-256 against the
    in-module pin, so this test transitively asserts artifact integrity.
    """
    assert locked_tau(cohort) == TAU_LOCKED_EXPECTED[cohort]
    assert locked_metric(cohort) is not None


@pytest.mark.parametrize("cohort", list(CalibrationCohort))
@pytest.mark.parametrize("accessor", [locked_tau, locked_metric])
def test_both_locked_accessors_verify_the_artifact_hash_at_access(
    cohort, accessor, monkeypatch
):
    """The integrity guarantee must hold for τ as well as for Σ.

    Previously only `locked_metric` hashed the artifact, so a caller asking for
    the threshold alone got an answer even from a drifted or missing artifact —
    the advertised access-time guarantee was false for exactly the cheaper call
    site. Both now route through one verified loader.
    """
    monkeypatch.setattr(fpr, "CALIBRATION_ARTIFACT_SHA256", "00" * 32)
    with pytest.raises(fpr.TauLockIntegrityError, match="sha256 mismatch"):
        accessor(cohort)


@pytest.mark.parametrize("cohort", list(CalibrationCohort))
@pytest.mark.parametrize("accessor", [locked_tau, locked_metric])
def test_both_locked_accessors_refuse_a_missing_artifact(
    cohort, accessor, monkeypatch, tmp_path
):
    monkeypatch.setattr(fpr, "CALIBRATION_ARTIFACT_PATH", tmp_path / "absent.json")
    with pytest.raises(fpr.TauLockIntegrityError, match="missing"):
        accessor(cohort)


def test_integrity_failures_are_not_catchable_as_not_locked():
    """The load-bearing taxonomy split ( round-2 P1).

    `build_fingerprint_registry` downgrades `TauNotLockedError` to an
    observe-only registry. If `TauLockIntegrityError` were a subclass, a
    damaged locked artifact would ride that downgrade and an H3 evaluation
    would run with no re-link matching — invalid results instead of a stop.
    """
    assert not issubclass(fpr.TauLockIntegrityError, TauNotLockedError)
    assert not issubclass(TauNotLockedError, fpr.TauLockIntegrityError)


@pytest.mark.parametrize("cohort", list(CalibrationCohort))
def test_the_locked_artifact_pins_the_selected_covariance_metric(cohort):
    """The τ pin is only meaningful paired with the metric it was calibrated on."""
    payload = json.loads(fpr.CALIBRATION_ARTIFACT_PATH.read_text())
    block = payload["cohorts"][cohort.value]

    expected = SELECTED_METRIC_EXPECTED[cohort]
    assert block["selected_metric"] == expected
    assert block["metric"]["provenance"] == f"h3-calibration/{cohort.value}/{expected}"
    assert block["tau"] == TAU_LOCKED_EXPECTED[cohort]
    # The metric the code actually loads, not merely what the file says.
    assert locked_metric(cohort).provenance == f"h3-calibration/{cohort.value}/{expected}"
    # In-code record and artifact must agree on the selection, not just on τ.
    assert (
        fpr.TAU_LOCK_RECORD["selected_metric"][cohort.value]
        == SELECTED_METRIC_EXPECTED[cohort]
    )


def test_the_lock_record_carries_the_ten_calibration_run_ids():
    """P1: the corpus must be traceable to durable custody records, not a path."""
    record = fpr.TAU_LOCK_RECORD
    payload_meta = json.loads(fpr.CALIBRATION_ARTIFACT_PATH.read_text())["_meta"]

    assert len(record["observed_run_ids"]) == 10
    assert len(record["observed_units"]) == 10
    assert all(len(rid) == 32 for rid in record["observed_run_ids"])
    assert sorted(record["observed_units"].values()) == sorted(
        record["observed_run_ids"]
    )
    # Artifact and in-code record are the two authoritative provenance copies.
    assert payload_meta["observed_run_ids"] == sorted(record["observed_run_ids"])
    assert {
        entry["source_unit"]: entry["run_id"]
        for entry in payload_meta["observed_units"]
    } == record["observed_units"]
    assert payload_meta["git_commit"] == record["git_commit"] != "pending"
    # The artifact must not self-report its estimator as unauthorised. The
    # discharged marker may still appear downstream in the string as narrated
    # history; what matters is the status the artifact CLAIMS, which is first.
    assert payload_meta["addendum_status"].startswith("RATIFIED 2026-08-08")


def test_registry_refuses_a_non_positive_tau():
    with pytest.raises(ValueError, match="tau"):
        _registry(tau=0.0)


# ---------------------------------------------------------------------------
# A.5(a) / A.6 — the per-observation log (PASSIVE custody, never a decision input)
#
# THRESHOLD_GROUNDING_20260808 § 5 A.6: the schema-v5 re-entry row carries
# `asserted_match` / `asserted_parent_logical_id` / `min_d` / `tau` but NOT the
# 180-dim vector, so the A.5(a) naive-Euclidean comparator is only computable
# offline "if the vector is persisted... before the validation run launches".
# The registry sees every observed vector, so it is where they are retained.
# These tests pin the two properties the comparator depends on — the log is
# AS-OBSERVED (not the EMA state) and bit-exact — plus the property the
# pre-registration depends on: recording changes NOTHING about the decision.
# ---------------------------------------------------------------------------

def _log_triples(registry):
    return [(o.server_round, o.session_key) for o in registry.observations()]


def test_every_observation_is_logged_with_its_round_and_session_key():
    reg = _registry()
    reg.observe("dev-a", _vec(1), server_round=3)
    reg.observe("dev-b", _vec(2), server_round=3)
    reg.observe("dev-a", _vec(3), server_round=4)

    assert _log_triples(reg) == [(3, "dev-a"), (3, "dev-b"), (4, "dev-a")]
    assert not reg.observation_log_truncated
    assert reg.observation_log_dropped_count == 0


def test_logged_vector_is_the_as_observed_draw_not_the_ema_state():
    """The comparator needs the SAME per-round draws the matcher saw.

    With EMA_ALPHA = 0.1 the entry state and the second observation differ from
    the second observation onward, so this test is genuinely discriminating: a
    log that recorded `entry.fingerprint_vec` would fail on row 1.
    """
    reg = _registry()
    first = np.full(DIM, 1.0)
    second = np.full(DIM, 11.0)
    reg.observe("dev-a", first, server_round=1)
    reg.observe("dev-a", second, server_round=2)

    log = reg.observations()
    assert np.array_equal(log[0].vector, first)
    assert np.array_equal(log[1].vector, second)

    ema = reg.entry_for_session("dev-a").fingerprint_vec
    expected_ema = (1.0 - EMA_ALPHA) * first + EMA_ALPHA * second
    assert np.allclose(ema, expected_ema)
    # The discriminating assertion: EMA state != as-observed draw.
    assert not np.allclose(log[1].vector, ema)


def test_logged_vector_is_a_defensive_copy_of_the_caller_array():
    reg = _registry()
    caller = np.full(DIM, 2.0)
    reg.observe("dev-a", caller, server_round=1)
    caller[0] = 999.0
    assert reg.observations()[0].vector[0] == 2.0


def test_logged_vectors_round_trip_through_json_bit_exactly():
    """Full float64 fidelity: the comparator does exact nearest-neighbour work.

    The row carries the raw little-endian float64 buffer, base64-encoded, so
    exactness is a property of the encoding rather than of CPython's `repr`
    happening to be shortest-round-trip. This test decodes with the exact
    one-liner the artifact advertises, on adversarial magnitudes (1e150-scale
    `tcp.payload` moments, subnormals, ulp neighbours, signed zero).
    """
    import base64
    import json

    hard = np.array(
        [
            1.2345678901234567e150,
            np.nextafter(1.0, 2.0),
            -np.nextafter(0.1, 0.0),
            5e-324,
            0.0,
            -0.0,
            1e-308,
            123456789.123456789,
        ],
        dtype=np.float64,
    )
    assert hard.shape == (DIM,)
    reg = _registry()
    reg.observe("dev-a", hard, server_round=7)

    row = json.loads(json.dumps(reg.observations()[0].as_custody_row()))
    assert "fingerprint_vec" not in row, "the decimal-list encoding is retired"
    restored = np.frombuffer(
        base64.b64decode(row["fingerprint_vec_b64"]), dtype="<f8"
    )
    assert restored.tobytes() == hard.tobytes()
    assert restored.shape == (DIM,)


def test_observation_log_truncation_is_loud_and_bounded(caplog):
    reg = _registry(max_observation_rows=2)
    with caplog.at_level("ERROR"):
        for r in range(5):
            reg.observe(f"dev-{r}", _vec(r), server_round=r)

    assert len(reg.observations()) == 2
    assert reg.observation_log_truncated is True
    assert reg.observation_log_dropped_count == 3
    assert reg.observation_log_max_rows == 2
    assert any(
        "observation log" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    ), "truncation must be LOUD — a silently truncated corpus biases the comparator"


def test_recording_is_passive__decisions_identical_with_and_without_the_log():
    """The acceptance property: the log changes NOT ONE re-link decision.

    Two registries are driven through the identical scripted sequence; one keeps
    the full log, the other is capped at a single row so the log path is
    exercised AND truncated. Every assertion field — including `min_d` at full
    float64 precision — must be identical.
    """
    def drive(**kwargs):
        reg = _registry(tau=5.0, **kwargs)
        seen = []
        for r in range(1, 6):
            for name, seed in (("dev-a", 1), ("dev-b", 2), (f"dev-new{r}", 1)):
                res = reg.observe(name, _vec(seed), server_round=r)
                seen.append((res.entry_id, res.is_first_appearance, res.flagged,
                             res.assertion))
            if r == 1:
                reg.flag("dev-a", reason="upstream_filter", server_round=r)
            # Recorded EVERY round, including after dev-a is already flagged —
            # the unconditional call the plugin makes at its call site.
            reg.record_upstream_rejection("dev-a", server_round=r)
        return seen, reg

    full, reg_full = drive(max_observation_rows=10_000, max_upstream_rejection_rows=10_000)
    capped, reg_capped = drive(max_observation_rows=1, max_upstream_rejection_rows=1)

    assert len(reg_full.observations()) == 15
    assert len(reg_capped.observations()) == 1
    assert len(reg_full.upstream_rejections()) == 5
    assert len(reg_capped.upstream_rejections()) == 1
    assert full == capped
    # …and min_d is bit-identical, not merely equal-ish.
    for (_, _, _, a), (_, _, _, b) in zip(full, capped):
        if a is None:
            assert b is None
            continue
        assert np.float64(a.min_d).tobytes() == np.float64(b.min_d).tobytes()
    for e_full, e_capped in zip(reg_full.entries(), reg_capped.entries()):
        assert e_full.fingerprint_vec.tobytes() == e_capped.fingerprint_vec.tobytes()
        assert dataclasses.replace(e_full, fingerprint_vec=None) == \
            dataclasses.replace(e_capped, fingerprint_vec=None)


def test_observation_log_max_rows_must_be_positive():
    with pytest.raises(ValueError, match="max_observation_rows"):
        _registry(max_observation_rows=0)


# ---------------------------------------------------------------------------
# A.6 — the upstream-rejection log (PASSIVE)
#
# The observation log alone does not close A.5(a). The registry's own flag
# lifecycle records only the FIRST flag: `FingerprintDefensePlugin.score_updates`
# calls `flag` under `if not entry.flag_status`, so once an entry has been
# INHERITED-flagged by the Mahalanobis matcher, every later upstream rejection of
# it is invisible. The A.5(a) Euclidean counterfactual needs exactly those
# rejections — an entry Mahalanobis inherited-flagged might not be flagged at all
# under τ′, and then the question "when did the detector flag it?" has no answer
# in custody. This log records every rejection UNCONDITIONALLY.
# ---------------------------------------------------------------------------

def test_upstream_rejections_are_logged_with_round_and_session_key():
    reg = _registry()
    reg.observe("dev-a", _vec(1), server_round=1)
    reg.observe("dev-b", _vec(2), server_round=1)
    reg.record_upstream_rejection("dev-a", server_round=1)
    reg.record_upstream_rejection("dev-b", server_round=2)
    reg.record_upstream_rejection("dev-a", server_round=3)

    assert [(e.server_round, e.session_key) for e in reg.upstream_rejections()] == [
        (1, "dev-a"), (2, "dev-b"), (3, "dev-a"),
    ]
    assert not reg.upstream_rejection_log_truncated
    assert reg.upstream_rejection_log_dropped_count == 0


def test_upstream_rejection_log_records_rejections_of_an_ALREADY_flagged_entry():
    """The gap this log exists to close.

    `flag` is gated on `not entry.flag_status` at the call site and preserves
    an existing `flag_round`, so the entry lifecycle cannot express "rejected
    again at round 5". The rejection log can, and must.
    """
    reg = _registry()
    reg.observe("dev-a", _vec(1), server_round=1)
    reg.flag("dev-a", reason="inherited", server_round=1)
    reg.record_upstream_rejection("dev-a", server_round=5)

    entry = reg.entry_for_session("dev-a")
    assert entry.flag_round == 1, "flag lifecycle is unchanged"
    assert entry.flag_reason == "inherited"
    assert [(e.server_round, e.session_key) for e in reg.upstream_rejections()] == [
        (5, "dev-a")
    ]


def test_upstream_rejection_log_does_not_flag_or_otherwise_touch_the_entry():
    reg = _registry()
    reg.observe("dev-a", _vec(1), server_round=1)
    before = reg.entry_for_session("dev-a")
    reg.record_upstream_rejection("dev-a", server_round=2)
    assert reg.entry_for_session("dev-a") == before
    assert reg.flagged_entries() == ()


def test_upstream_rejection_log_accepts_an_unknown_session_without_raising():
    """Observation-only: it must never become a second way to raise on a key.

    `flag` raises KeyError on an unknown session by design; this log is not a
    lookup and must not add a new failure mode to the aggregation path.
    """
    reg = _registry()
    reg.record_upstream_rejection("never-observed", server_round=1)
    assert [e.session_key for e in reg.upstream_rejections()] == ["never-observed"]


def test_upstream_rejection_log_truncation_is_loud_and_bounded(caplog):
    reg = _registry(max_upstream_rejection_rows=2)
    with caplog.at_level("ERROR"):
        for r in range(5):
            reg.record_upstream_rejection(f"dev-{r}", server_round=r)

    assert len(reg.upstream_rejections()) == 2
    assert reg.upstream_rejection_log_truncated is True
    assert reg.upstream_rejection_log_dropped_count == 3
    assert reg.upstream_rejection_log_max_rows == 2
    assert any(
        "rejection log" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    ), "truncation must be LOUD — a silently truncated corpus biases the replay"


def test_upstream_rejection_log_max_rows_must_be_positive():
    with pytest.raises(ValueError, match="max_upstream_rejection_rows"):
        _registry(max_upstream_rejection_rows=0)


# ===========================================================================
# Registry candidate policy — the corrected-H3 identity-only instrument
# ===========================================================================
# `docs/reproduction/experiments.md`
# § 1: the deployed registry gates the re-entry candidate pool on upstream
# detector flags, so what it measures is P(flagged by TGE) x P(re-linked |
# flagged) — RQ2's detection performance multiplied into RQ3's identity
# question. The corrected instrument enrolls EVERY session and considers ALL
# sessions first seen strictly before the re-entrant. The flag-gated mode stays
# for H4's composition arms, and stays the default.

def _policy_registry(policy, tau=1.0, dim=DIM):
    return FingerprintRegistry(
        tau=tau, metric=_metric(dim), dim=dim, policy=policy
    )


def test_the_two_registry_policies_are_the_declared_modes():
    from flowerfl.fingerprint_registry import RegistryPolicy

    assert {p.value for p in RegistryPolicy} == {"flag_gated", "identity_only"}


def test_the_default_policy_is_the_deployed_flag_gated_incumbent():
    from flowerfl.fingerprint_registry import DEFAULT_REGISTRY_POLICY, RegistryPolicy

    assert DEFAULT_REGISTRY_POLICY is RegistryPolicy.FLAG_GATED
    assert _registry().policy is RegistryPolicy.FLAG_GATED


def test_an_unknown_policy_is_a_loud_refusal():
    with pytest.raises(ValueError, match="policy"):
        FingerprintRegistry(tau=1.0, metric=_metric(), dim=DIM, policy="everyone")


def _run_sequence(registry):
    """A fixed observe/flag script; returns every assertion and entry it made."""
    base = _vec(1)
    other = _vec(2) * 40.0
    log = []
    log.append(registry.observe("client_1", base, 1, logical_id="client_1"))
    log.append(registry.observe("client_2", other, 1, logical_id="client_2"))
    registry.flag("client_2", "upstream", 2)
    log.append(registry.observe("client_1", base + 0.01, 2, logical_id="client_1"))
    log.append(
        registry.observe("client_2_new1", other + 0.01, 3, logical_id="client_2_new1")
    )
    log.append(
        registry.observe("client_1_new1", base + 0.01, 4, logical_id="client_1_new1")
    )
    return log, registry.entries()


def test_declaring_flag_gated_is_identical_to_the_default_construction():
    """The knob's default is INERT: absent == flag_gated, decision for decision."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    default_log, default_entries = _run_sequence(_registry())
    declared_log, declared_entries = _run_sequence(
        _policy_registry(RegistryPolicy.FLAG_GATED)
    )
    assert default_log == declared_log
    assert len(default_entries) == len(declared_entries)
    for left, right in zip(default_entries, declared_entries):
        for field in dataclasses.fields(left):
            a = getattr(left, field.name)
            b = getattr(right, field.name)
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), field.name
            else:
                assert a == b, field.name


def test_identity_only_links_an_honest_device_the_flag_gated_mode_cannot_see():
    """The construct-validity fix, in one comparison: no flag anywhere."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    vector = _vec(7)
    gated = _registry()
    gated.observe("client_3", vector, 1, logical_id="client_3")
    gated_result = gated.observe(
        "client_3_new1", vector + 0.001, 3, logical_id="client_3_new1"
    )
    assert gated_result.assertion.asserted_match is False
    assert gated_result.assertion.min_d == float("inf")

    identity = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    identity.observe("client_3", vector, 1, logical_id="client_3")
    result = identity.observe(
        "client_3_new1", vector + 0.001, 3, logical_id="client_3_new1"
    )
    assert result.assertion.asserted_match is True
    assert result.assertion.asserted_parent_logical_id == "client_3"
    assert result.assertion.min_d < 1.0


def test_identity_only_never_flags_an_honest_self_reidentification():
    """Identity linking is NOT an enforcement decision.

    `fingerprint_plugin` hard-drops on `flag_status and flag_reason ==
    'inherited'`; an honest device correctly re-identified as itself must not
    satisfy that, or the corrected instrument would silently drop honest
    clients that the deployed one never touched.
    """
    from flowerfl.fingerprint_registry import RegistryPolicy

    vector = _vec(8)
    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    registry.observe("client_5", vector, 1, logical_id="client_5")
    result = registry.observe(
        "client_5_new1", vector + 0.001, 3, logical_id="client_5_new1"
    )
    entry = registry.entry_for_session("client_5_new1")
    assert result.assertion.asserted_match is True
    assert result.flagged is False
    assert entry.flag_status is False
    assert entry.flag_reason is None
    assert entry.flag_round is None
    # the identity link itself is still fully recorded
    assert entry.parent_entry_id == registry.entry_for_session("client_5").entry_id
    assert entry.generation == 1


def test_identity_only_still_inherits_a_flag_from_a_flagged_parent():
    """D4 enforcement stays scoped to FLAGGED devices in BOTH modes."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    vector = _vec(9)
    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    registry.observe("client_7", vector, 1, logical_id="client_7")
    registry.flag("client_7", "upstream", 2)
    result = registry.observe(
        "client_7_new1", vector + 0.001, 3, logical_id="client_7_new1"
    )
    entry = registry.entry_for_session("client_7_new1")
    assert result.flagged is True
    assert entry.flag_status is True
    assert entry.flag_reason == "inherited"
    assert entry.flag_round == 3
    assert entry.generation == 1


def test_identity_only_candidates_are_first_seen_STRICTLY_before_the_reentrant():
    """Policy B's pool rule: a session enrolled in the SAME round is not a
    candidate, mirroring `replay_engine.py` (`first_seen_round < rnd`)."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    vector = _vec(10)
    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    registry.observe("client_9", vector, 5, logical_id="client_9")
    same_round = registry.observe(
        "client_9_new1", vector + 0.001, 5, logical_id="client_9_new1"
    )
    assert same_round.assertion.asserted_match is False
    assert same_round.assertion.min_d == float("inf")

    later = registry.observe(
        "client_9_new2", vector + 0.001, 6, logical_id="client_9_new2"
    )
    assert later.assertion.asserted_match is True


def test_identity_only_tie_break_gives_an_exact_tie_to_the_LATEST_enrollment():
    """Policy B's disclosed rule: candidates ordered by (-first_seen_round,
    entry_id), replaced only on strictly smaller distance."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    offset = np.zeros(DIM)
    offset[0] = 0.5
    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    registry.observe("client_2", offset, 1, logical_id="client_2")     # fp-0000
    registry.observe("client_4", -offset, 2, logical_id="client_4")    # fp-0001
    result = registry.observe(
        "client_6", np.zeros(DIM), 3, logical_id="client_6"
    )
    assert result.assertion.min_d == pytest.approx(0.5)
    # both candidates sit at exactly 0.5; the LATER enrollment wins
    assert result.assertion.asserted_parent_logical_id == "client_4"


def test_identity_only_tie_break_falls_back_to_entry_id_within_a_round():
    from flowerfl.fingerprint_registry import RegistryPolicy

    offset = np.zeros(DIM)
    offset[0] = 0.5
    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    registry.observe("client_2", offset, 1, logical_id="client_2")     # fp-0000
    registry.observe("client_4", -offset, 1, logical_id="client_4")    # fp-0001
    result = registry.observe(
        "client_6", np.zeros(DIM), 3, logical_id="client_6"
    )
    assert result.assertion.asserted_parent_logical_id == "client_2"


def test_identity_only_pool_is_invariant_to_insertion_order():
    """A non-deterministic tie-break here would reproduce the defect
    inside the corrected H3 metric itself."""
    from flowerfl.fingerprint_registry import RegistryPolicy

    offset = np.zeros(DIM)
    offset[0] = 0.5
    parents = [("client_2", offset, 1), ("client_4", -offset, 2)]
    seen = set()
    for order in (parents, list(reversed(parents))):
        registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
        for key, vector, rnd in sorted(order, key=lambda item: item[2]):
            registry.observe(key, vector, rnd, logical_id=key)
        result = registry.observe("client_6", np.zeros(DIM), 3, logical_id="client_6")
        seen.add(result.assertion.asserted_parent_logical_id)
    assert seen == {"client_4"}


def test_a_returning_session_emits_no_assertion_under_either_policy():
    from flowerfl.fingerprint_registry import RegistryPolicy

    for policy in RegistryPolicy:
        registry = _policy_registry(policy)
        registry.observe("client_1", _vec(3), 1, logical_id="client_1")
        again = registry.observe("client_1", _vec(3) + 0.01, 2, logical_id="client_1")
        assert again.assertion is None
        assert again.is_first_appearance is False


# ===========================================================================
# Metric-level feature mask — the R5 selection rule's application point
# ===========================================================================
# `results/20260814/h3_feature_eda/EDA_MEMO.md` § 6.2: "Apply the mask at the
# METRIC level, not the emission level: raw 180-dim emission is unchanged (no
# emission-contract change, no bootstrap-noise concern); the locked calibration
# artifact carries the surviving-dim mask and the Σ/τ fit on those dims."
#
# The metric therefore owns the mask. Everything downstream — the registry, the
# custody export, any offline replay — keeps handing it full-width vectors and
# never has to know a subspace exists.

def _masked_metric(mask, input_dim=DIM, seed=4):
    """A within-device metric fitted on a SUBSPACE of a full-width population."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(40, input_dim))
    groups = [i % 4 for i in range(40)]
    return MahalanobisMetric.from_within_device_population(
        vectors, groups, shrinkage=1.0, mask=mask, input_dim=input_dim
    )


def test_an_unmasked_metric_expects_full_width_vectors():
    metric = MahalanobisMetric.identity(DIM)
    assert metric.mask is None
    assert metric.expected_input_dim == DIM
    assert metric.dim == DIM


def test_a_masked_metric_fits_the_subspace_but_accepts_full_width_vectors():
    mask = [0, 2, 5]
    metric = _masked_metric(mask)
    assert metric.dim == 3                      # precision lives in the subspace
    assert metric.expected_input_dim == DIM     # callers still pass 180-dim
    assert metric.precision.shape == (3, 3)
    assert metric.scale.shape == (3,)
    a, b = np.zeros(DIM), np.zeros(DIM)
    assert metric.distance(a, b) == 0.0


def test_a_masked_metric_ignores_the_dropped_dimensions_entirely():
    """The whole point: a pathological dim outside the mask cannot move a
    distance. Partition 7's `http.content_length__std` is exactly this case."""
    mask = [0, 2, 5]
    metric = _masked_metric(mask)
    a = np.zeros(DIM)
    b = np.zeros(DIM)
    b[1] = 1e9      # a dropped dimension, astronomically far away
    b[3] = -1e9
    assert metric.distance(a, b) == 0.0


def test_a_masked_metric_still_measures_the_surviving_dimensions():
    mask = [0, 2, 5]
    metric = _masked_metric(mask)
    a = np.zeros(DIM)
    b = np.zeros(DIM)
    b[2] = 5.0
    assert metric.distance(a, b) > 0.0


def test_a_masked_metric_refuses_a_wrongly_sized_vector():
    metric = _masked_metric([0, 2, 5])
    with pytest.raises(ValueError, match="dim mismatch"):
        metric.distance(np.zeros(3), np.zeros(3))


def test_the_mask_round_trips_through_the_artifact_form():
    metric = _masked_metric([0, 2, 5])
    restored = MahalanobisMetric.from_dict(metric.to_dict())
    assert list(restored.mask) == [0, 2, 5]
    assert restored.expected_input_dim == DIM
    assert restored.dim == metric.dim
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=DIM), rng.normal(size=DIM)
    assert restored.distance(a, b) == pytest.approx(metric.distance(a, b))


def test_an_unmasked_artifact_restores_unmasked():
    """The committed v1 lock carries no mask; it must keep behaving identically."""
    payload = MahalanobisMetric.identity(DIM).to_dict()
    payload.pop("mask", None)
    payload.pop("input_dim", None)
    restored = MahalanobisMetric.from_dict(payload)
    assert restored.mask is None
    assert restored.expected_input_dim == DIM


def test_a_mask_is_validated_against_the_input_width():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(20, DIM))
    groups = [i % 4 for i in range(20)]
    for bad in ([0, DIM], [-1, 2], [1, 1], []):
        with pytest.raises(ValueError, match="mask"):
            MahalanobisMetric.from_within_device_population(
                vectors, groups, shrinkage=1.0, mask=bad, input_dim=DIM
            )


def test_project_is_the_single_place_the_mask_is_applied():
    metric = _masked_metric([0, 2, 5])
    rng = np.random.default_rng(7)
    block = rng.normal(size=(6, DIM))
    assert np.array_equal(metric.project(block), block[:, [0, 2, 5]])
    assert np.array_equal(MahalanobisMetric.identity(DIM).project(block), block)


def test_the_registry_accepts_a_masked_metric_and_still_takes_full_vectors():
    mask = [0, 2, 5]
    registry = FingerprintRegistry(tau=1.0, metric=_masked_metric(mask), dim=DIM)
    vector = _vec(21)
    registry.observe("client_1", vector, 1, logical_id="client_1")
    result = registry.observe("client_1_new1", vector, 3, logical_id="client_1_new1")
    assert result.assertion is not None
    assert registry.metric.expected_input_dim == DIM


def test_the_registry_refuses_a_metric_whose_input_width_is_wrong():
    metric = _masked_metric([0, 2, 5], input_dim=DIM)
    with pytest.raises(ValueError, match="dim"):
        FingerprintRegistry(tau=1.0, metric=metric, dim=DIM + 1)


# ===========================================================================
# Unconditional nearest-candidate recording
# ===========================================================================
# The corrected instrument's P1 is threshold-FREE, but the frozen re-entry
# contract records `asserted_parent_*` only when the match fires, so an event
# whose nearest candidate sits beyond tau left rank-1 unanswerable. The registry
# now records the nearest candidate on EVERY re-entry decision. The `asserted_*`
# fields keep their frozen matched-only semantics untouched.

def test_the_nearest_candidate_is_recorded_even_when_no_match_fires():
    from flowerfl.fingerprint_registry import RegistryPolicy

    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY, tau=1e-9)
    registry.observe("client_3", _vec(31), 1, logical_id="client_3")
    result = registry.observe(
        "client_3_new1", _vec(31) + 5.0, 3, logical_id="client_3_new1"
    )
    assertion = result.assertion
    assert assertion.asserted_match is False
    assert assertion.asserted_parent_logical_id is None   # frozen semantics
    assert assertion.nearest_logical_id == "client_3"     # newly recoverable
    assert assertion.nearest_entry_id == registry.entry_for_session(
        "client_3"
    ).entry_id
    assert assertion.min_d > assertion.tau


def test_a_matched_event_asserts_exactly_the_nearest_candidate():
    from flowerfl.fingerprint_registry import RegistryPolicy

    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    vector = _vec(32)
    registry.observe("client_5", vector, 1, logical_id="client_5")
    result = registry.observe(
        "client_5_new1", vector + 0.001, 3, logical_id="client_5_new1"
    )
    assertion = result.assertion
    assert assertion.asserted_match is True
    assert assertion.asserted_parent_logical_id == assertion.nearest_logical_id
    assert assertion.asserted_parent_entry_id == assertion.nearest_entry_id


def test_the_nearest_pair_is_null_when_the_pool_is_empty():
    from flowerfl.fingerprint_registry import RegistryPolicy

    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY)
    result = registry.observe("client_1", _vec(33), 1, logical_id="client_1")
    assert result.assertion.nearest_entry_id is None
    assert result.assertion.nearest_logical_id is None
    assert result.assertion.min_d == float("inf")


def test_the_flag_gated_mode_also_records_its_nearest_candidate():
    """Recording is a property of the DECISION, not of the candidate policy."""
    registry = _registry(tau=1e-9)
    registry.observe("client_7", _vec(34), 1, logical_id="client_7")
    registry.flag("client_7", "upstream", 2)
    result = registry.observe(
        "client_7_new1", _vec(34) + 5.0, 3, logical_id="client_7_new1"
    )
    assert result.assertion.asserted_match is False
    assert result.assertion.nearest_logical_id == "client_7"


def test_as_log_fields_carries_the_nearest_pair_alongside_the_frozen_six():
    from flowerfl.fingerprint_registry import RegistryPolicy

    registry = _policy_registry(RegistryPolicy.IDENTITY_ONLY, tau=1e-9)
    registry.observe("client_9", _vec(35), 1, logical_id="client_9")
    result = registry.observe(
        "client_9_new1", _vec(35) + 5.0, 3, logical_id="client_9_new1"
    )
    fields = result.assertion.as_log_fields()
    # the frozen six are untouched
    for name in ("asserted_match", "asserted_parent_entry_id",
                 "asserted_parent_logical_id", "min_d", "tau", "generation"):
        assert name in fields
    assert fields["asserted_parent_logical_id"] is None
    # and the additive pair is there
    assert fields["nearest_logical_id"] == "client_9"
    assert fields["nearest_entry_id"] is not None

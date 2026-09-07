"""H3 Step 4 (library half) — the fingerprint defense plugin.

Design authority:
* v1.10 § 5.0 **D4** — "Hard-drop (coefficient = 0). The FP plugin sits **after**
  the soft-reweight incumbent; on a Mahalanobis-matched flagged re-entrant its
  action is a true **coeff = 0** hard-block. The incumbent `max(1,·)` floor would
  leave tiny non-zero mass → `re_entry_block_rate` structurally 0."
* v1.10 § 5.1 **INTEGRITY ASSERTION** — "A unit test asserts that flipping the
  FP enforcement action leaves the logged re-link fields (and therefore the D7
  metric) byte-identical. This is what makes 'detector-independent' a scoring
  rule, not just a phrase."
* v1.10 § 5.1 gate (e) — `FitRes.metrics["fingerprint"]` present for 100% of
  client-rounds.

`test_flipping_enforcement_leaves_the_logged_relink_fields_byte_identical` is
the single test the whole H3 primary metric rests on.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, Status, ndarrays_to_parameters

from flowerfl.byzantine_defense import ByzantineDefensePlugin
from flowerfl.fingerprint import encode_fingerprint
from flowerfl.fingerprint_plugin import (
    ENFORCEMENT_MODES,
    EnforcementMode,
    FingerprintDefensePlugin,
)
from flowerfl.fingerprint_registry import FingerprintRegistry, MahalanobisMetric

DIM = 8


def _fingerprint(seed: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


def _results(spec, dim=DIM):
    """spec: list of (cid, fingerprint | None). Returns (proxy, fit_res) pairs."""
    rng = np.random.default_rng(0)
    out = []
    for cid, fingerprint in spec:
        params = ndarrays_to_parameters([rng.normal(size=4).astype(np.float32)])
        metrics = {}
        if fingerprint is not None:
            metrics["fingerprint"] = encode_fingerprint(np.asarray(fingerprint))
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=100,
            metrics=metrics,
        )
        out.append((proxy, fit))
    return out


def _plugin(tau=1.0, **kwargs) -> FingerprintDefensePlugin:
    registry = FingerprintRegistry(
        tau=tau, metric=MahalanobisMetric.identity(DIM), dim=DIM
    )
    kwargs.setdefault("expected_dim", DIM)
    return FingerprintDefensePlugin(registry=registry, run_id="run-test", **kwargs)


# ---------------------------------------------------------------------------
# Plugin contract
# ---------------------------------------------------------------------------

def test_plugin_implements_the_abc():
    plugin = _plugin()
    assert isinstance(plugin, ByzantineDefensePlugin)
    assert plugin.name == "Fingerprint"


def test_default_enforcement_is_the_d4_hard_drop():
    assert _plugin().enforcement_mode is EnforcementMode.HARD_DROP
    assert set(ENFORCEMENT_MODES) == {"hard_drop", "downweight", "accept"}


def test_unknown_enforcement_mode_is_rejected():
    with pytest.raises(ValueError, match="enforcement"):
        _plugin(enforcement_mode="soft_ish")


# ---------------------------------------------------------------------------
# Observation happens on the FULL cohort, before any filtering
# ---------------------------------------------------------------------------

def test_observe_cohort_registers_every_participant():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    results = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(results, server_round=1)
    assert len(plugin.registry) == 2
    # Enrollment, not re-entry: nothing has RETURNED in the first cohort.
    assert plugin.reentry_events == []
    assert plugin.initial_enrollments == ("client_0", "client_1")
    assert plugin.enrollment_round == 1


def test_initial_enrollment_is_not_a_reentry_event():
    """§ 5.1: an event is a new CID for a **returning** logical device.

    Every device enrolls at the start of a run; those appearances are not
    returns. Emitting rows for them would put events in the metric's population
    that the ground-truth layer never emits, corrupting the join and the
    denominators.
    """
    plugin = _plugin()
    plugin.set_identity_map({f"raw{i}": f"client_{i}" for i in range(20)})
    cohort = _results([(f"raw{i}", _fingerprint(i)) for i in range(20)])
    plugin.observe_cohort(cohort, server_round=1)
    assert plugin.reentry_events == []
    assert len(plugin.initial_enrollments) == 20
    assert len(plugin.registry) == 20  # all still enrolled as match candidates


def test_a_claimed_identity_first_seen_after_enrollment_is_a_reentry_event():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    assert plugin.reentry_events == []

    plugin.set_identity_map({"raw0": "client_0_new1", "raw1": "client_1"})
    plugin.observe_cohort(
        _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))]), server_round=7
    )
    assert [row["gt_logical_id"] for row in plugin.reentry_events] == ["client_0_new1"]


def test_enrollment_round_assertions_could_never_have_matched_anyway():
    """Nothing is flagged until score_updates, which runs after observe_cohort,
    so suppressing enrollment-round rows costs no matching power."""
    plugin = _plugin(tau=1.0)
    shared = _fingerprint(1)
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    # Two clients with the SAME fingerprint in the enrollment cohort.
    plugin.observe_cohort(
        _results([("raw0", shared), ("raw1", shared)]), server_round=1
    )
    assert [e.flag_status for e in plugin.registry.entries()] == [False, False]


def test_observe_cohort_does_not_mutate_the_results():
    plugin = _plugin()
    results = _results([("raw0", _fingerprint(1))])
    before = (results[0][1].num_examples, dict(results[0][1].metrics))
    plugin.observe_cohort(results, server_round=1)
    assert (results[0][1].num_examples, dict(results[0][1].metrics)) == before


def test_a_reentry_event_row_carries_the_schema_v5_field_set():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0"})
    plugin.observe_cohort(_results([("raw0", _fingerprint(1))]), server_round=1)
    plugin.set_identity_map({"raw0": "client_0_new1"})
    plugin.observe_cohort(_results([("raw0", _fingerprint(1))]), server_round=3)
    row = plugin.reentry_events[0]
    assert row["reentry_event_key"] == "run-test:3:raw0"
    for field in (
        "server_round",
        "current_cid",
        "gt_logical_id",
        "asserted_match",
        "asserted_parent_entry_id",
        "asserted_parent_logical_id",
        "min_d",
        "tau",
        "generation",
    ):
        assert field in row, field
    assert row["gt_logical_id"] == "client_0_new1"


def test_repeat_participation_does_not_manufacture_extra_events():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0"})
    results = _results([("raw0", _fingerprint(1))])
    plugin.observe_cohort(results, server_round=1)          # enrollment
    plugin.set_identity_map({"raw0": "client_0_new1"})      # one identity change
    for round_index in (2, 3, 4, 5):
        plugin.observe_cohort(results, server_round=round_index)
    assert len(plugin.reentry_events) == 1


def test_event_keys_are_unique():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    plugin.observe_cohort(
        _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))]), server_round=1
    )
    plugin.set_identity_map({"raw0": "client_0_new1", "raw1": "client_1_new1"})
    plugin.observe_cohort(
        _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))]), server_round=2
    )
    plugin.set_identity_map({"raw0": "client_0_new2", "raw1": "client_1_new1"})
    plugin.observe_cohort(
        _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))]), server_round=3
    )
    keys = [row["reentry_event_key"] for row in plugin.reentry_events]
    assert len(keys) == len(set(keys)) == 3


# ---------------------------------------------------------------------------
# Fingerprint emission (gate (e) support)
# ---------------------------------------------------------------------------

def test_missing_fingerprint_is_counted_not_crashed():
    plugin = _plugin()
    plugin.observe_cohort(_results([("raw0", None)]), server_round=1)
    assert plugin.participating_missing_fingerprint_count == 1
    assert len(plugin.registry) == 0
    scores = plugin.score_updates(_results([("raw0", None)]), server_round=1)
    assert scores == {0: 1.0}


def test_malformed_fingerprint_is_counted_not_crashed():
    results = _results([("raw0", _fingerprint(1))])
    results[0][1].metrics["fingerprint"] = "{not json"
    plugin = _plugin()
    plugin.observe_cohort(results, server_round=1)
    assert plugin.participating_missing_fingerprint_count == 1


def test_wrong_dimension_fingerprint_is_counted_not_crashed():
    plugin = _plugin()
    plugin.observe_cohort(_results([("raw0", np.zeros(DIM + 2))]), server_round=1)
    assert plugin.participating_missing_fingerprint_count == 1


def test_emission_completeness_is_reported():
    plugin = _plugin()
    plugin.observe_cohort(
        _results([("raw0", _fingerprint(1)), ("raw1", None)]), server_round=1
    )
    assert plugin.participating_fingerprint_emission_rate() == 0.5
    assert plugin.participating_client_round_count == 2


# ---------------------------------------------------------------------------
# Scoring + flag inference
# ---------------------------------------------------------------------------

def test_upstream_dropped_clients_are_flagged_in_the_registry():
    """FP sits last: whoever the upstream chain removed was flagged malicious."""
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    # The upstream chain kept only raw1.
    plugin.score_updates([cohort[1]], server_round=1)
    assert plugin.registry.entry_for_session("client_0").flag_status is True
    assert plugin.registry.entry_for_session("client_0").flag_reason == "upstream_filter"
    assert plugin.registry.entry_for_session("client_1").flag_status is False


def test_a_client_absent_from_a_later_cohort_is_NOT_flagged():
    """Round-scoping regression.

    Against a run-long observed set, any client that merely sat out a sampling
    round would be subtracted from that round's survivors and permanently
    flagged as an upstream rejection — manufacturing match candidates out of
    honest devices. `control_benign_churn` excuses 3-5 clients EVERY round by
    design, so this would have fired on the calibration scenario itself.
    """
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    round_one = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(round_one, server_round=1)
    plugin.score_updates(round_one, server_round=1)  # nobody rejected

    # Round 2: client_0 simply does not participate (churn, not rejection).
    round_two = _results([("raw1", _fingerprint(2))])
    plugin.observe_cohort(round_two, server_round=2)
    plugin.score_updates(round_two, server_round=2)

    assert plugin.registry.entry_for_session("client_0").flag_status is False, (
        "a non-participating client was mistaken for an upstream rejection"
    )
    assert plugin.registry.entry_for_session("client_1").flag_status is False


def test_an_absent_then_returning_client_is_still_never_flagged():
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    both = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(both, server_round=1)
    plugin.score_updates(both, server_round=1)

    for absent_round in (2, 3, 4):
        only_one = _results([("raw1", _fingerprint(2))])
        plugin.observe_cohort(only_one, server_round=absent_round)
        plugin.score_updates(only_one, server_round=absent_round)

    plugin.observe_cohort(both, server_round=5)
    plugin.score_updates(both, server_round=5)
    assert plugin.registry.entry_for_session("client_0").flag_status is False


def test_upstream_rejection_in_the_SAME_round_is_still_flagged():
    """The round-scoping fix must not weaken genuine rejection detection."""
    plugin = _plugin()
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=3)
    plugin.score_updates([cohort[1]], server_round=3)  # raw0 dropped upstream
    assert plugin.registry.entry_for_session("client_0").flag_status is True
    assert plugin.registry.entry_for_session("client_0").flag_reason == "upstream_filter"


def test_matched_reentrant_scores_zero_and_everyone_else_scores_one():
    plugin = _plugin(tau=1.0)
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    device = _fingerprint(1)
    cohort = _results([("raw0", device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    plugin.score_updates([cohort[1]], server_round=1)  # raw0 dropped upstream => flagged

    plugin.set_identity_map({"raw0new": "client_0_new1", "raw1": "client_1"})
    respawn = _results([("raw0new", device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(respawn, server_round=5)
    scores = plugin.score_updates(respawn, server_round=5)
    assert scores == {0: 0.0, 1: 1.0}
    assert plugin._round_scores[5] == scores


def test_score_updates_handles_an_unobserved_client_conservatively():
    """A client score_updates sees but observe_cohort never did gets full trust."""
    plugin = _plugin()
    scores = plugin.score_updates(_results([("ghost", _fingerprint(9))]), server_round=1)
    assert scores == {0: 1.0}


# ---------------------------------------------------------------------------
# D4 enforcement
# ---------------------------------------------------------------------------

def _matched_setup(**kwargs):
    plugin = _plugin(tau=1.0, **kwargs)
    device = _fingerprint(1)
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    plugin.score_updates([cohort[1]], server_round=1)

    plugin.set_identity_map({"raw0new": "client_0_new1", "raw1": "client_1"})
    respawn = _results([("raw0new", device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(respawn, server_round=5)
    scores = plugin.score_updates(respawn, server_round=5)
    return plugin, respawn, scores


def test_hard_drop_removes_the_matched_reentrant_entirely():
    """D4: coefficient exactly 0, not the incumbent max(1, ·) soft floor."""
    plugin, respawn, scores = _matched_setup()
    survivors = plugin.filter_updates(respawn, scores)
    assert [proxy.cid for proxy, _ in survivors] == ["raw1"]
    assert plugin.blocked_cids(5) == ("raw0new",)


def test_hard_drop_leaves_the_survivors_untouched():
    plugin, respawn, scores = _matched_setup()
    survivors = plugin.filter_updates(respawn, scores)
    assert survivors[0][1] is respawn[1][1], "survivor FitRes must pass through as-is"


def test_downweight_mode_keeps_a_nonzero_floor_and_is_not_the_default():
    """Recorded for the integrity test only — it is what D4 rejects."""
    plugin, respawn, scores = _matched_setup(enforcement_mode="downweight")
    survivors = plugin.filter_updates(respawn, scores)
    assert [proxy.cid for proxy, _ in survivors] == ["raw0new", "raw1"]
    assert survivors[0][1].num_examples == 1  # the max(1, ·) floor D4 forbids


def test_accept_mode_passes_everything_through():
    plugin, respawn, scores = _matched_setup(enforcement_mode="accept")
    survivors = plugin.filter_updates(respawn, scores)
    assert [proxy.cid for proxy, _ in survivors] == ["raw0new", "raw1"]
    assert survivors[0][1].num_examples == 100


# ---------------------------------------------------------------------------
# THE pre-registered integrity assertion (v1.10 § 5.1)
# ---------------------------------------------------------------------------

def _run_scenario(enforcement_mode: str):
    """Identical FL trace under a different FP enforcement action."""
    plugin = _plugin(tau=1.0, enforcement_mode=enforcement_mode)
    device = _fingerprint(1)
    other = _fingerprint(2)

    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", device), ("raw1", other)])
    plugin.observe_cohort(cohort, server_round=1)
    scores = plugin.score_updates([cohort[1]], server_round=1)
    plugin.filter_updates([cohort[1]], scores)

    for round_index, (cid, logical) in enumerate(
        [("raw0new1", "client_0_new1"), ("raw0new2", "client_0_new2")], start=5
    ):
        plugin.set_identity_map({cid: logical, "raw1": "client_1"})
        respawn = _results([(cid, device), ("raw1", other)])
        plugin.observe_cohort(respawn, server_round=round_index)
        scores = plugin.score_updates(respawn, server_round=round_index)
        plugin.filter_updates(respawn, scores)
    return plugin


def test_flipping_enforcement_leaves_the_logged_relink_fields_byte_identical():
    """v1.10 § 5.1: the test the entire H3 primary metric rests on.

    The registry's re-link decisions must be computable from the log
    INDEPENDENT of enforcement — so hard-drop, downweight and accept must
    produce byte-identical `reentry_events`.
    """
    serialised = {
        mode: json.dumps(_run_scenario(mode).reentry_events, sort_keys=True, default=str)
        for mode in ENFORCEMENT_MODES
    }
    assert len(set(serialised.values())) == 1, (
        "FP enforcement action changed the logged re-link fields — the D7 metric "
        "would no longer be detector-independent"
    )
    # ...and the scenario really did exercise a link, so this is not vacuous.
    events = _run_scenario("hard_drop").reentry_events
    assert any(row["asserted_match"] for row in events)


def test_relink_assertions_survive_upstream_filtering_of_the_reentrant():
    """Assertions come from observe_cohort (full cohort), not from the survivors.

    If they were computed in score_updates, an upstream detector that dropped
    the re-entrant would erase the very event H3 is trying to score.
    """
    device = _fingerprint(1)

    def run(upstream_drops_reentrant: bool):
        plugin = _plugin(tau=1.0)
        plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
        cohort = _results([("raw0", device), ("raw1", _fingerprint(2))])
        plugin.observe_cohort(cohort, server_round=1)
        plugin.score_updates([cohort[1]], server_round=1)

        plugin.set_identity_map({"raw0new": "client_0_new1", "raw1": "client_1"})
        respawn = _results([("raw0new", device), ("raw1", _fingerprint(2))])
        plugin.observe_cohort(respawn, server_round=5)
        survivors = [respawn[1]] if upstream_drops_reentrant else respawn
        plugin.score_updates(survivors, server_round=5)
        return [
            {k: v for k, v in row.items() if k != "gt_logical_id"}
            for row in plugin.reentry_events
        ]

    assert run(True) == run(False)


# ---------------------------------------------------------------------------
# The registry must never key or match on a Flower CID (RMC threat model)
# ---------------------------------------------------------------------------

def test_identity_reset_under_a_STABLE_flower_cid_still_produces_an_event():
    """The harness fact this whole design has to survive.

    In Flower simulation the raw cid is stable per virtual client for the WHOLE
    run (v1.15) — it does not change at an identity reset. A registry keyed on
    the Flower CID would see one long-lived client, emit no new-identity event,
    and give the primary metric an empty numerator by construction.
    """
    plugin = _plugin(tau=1.0)
    device = _fingerprint(1)
    stable_cid = "raw0"  # SAME Flower CID before and after the reset

    plugin.set_identity_map({stable_cid: "client_0", "raw1": "client_1"})
    cohort = _results([(stable_cid, device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    plugin.score_updates([cohort[1]], server_round=1)  # flagged upstream

    # Identity reset: new CLAIMED identity, unchanged Flower CID.
    plugin.set_identity_map({stable_cid: "client_0_new1", "raw1": "client_1"})
    respawn = _results([(stable_cid, device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(respawn, server_round=5)

    reset_events = [
        row for row in plugin.reentry_events if row["gt_logical_id"] == "client_0_new1"
    ]
    assert len(reset_events) == 1, "the identity reset produced no re-entry event"
    assert reset_events[0]["asserted_match"] is True
    assert reset_events[0]["current_cid"] == stable_cid  # audit field, unchanged
    assert plugin.score_updates(respawn, server_round=5)[0] == 0.0


def test_the_match_decision_is_invariant_to_flower_cid_relabelling():
    """CIDs may appear in audit output but must never be a key or a feature."""
    device = _fingerprint(1)
    other = _fingerprint(2)

    def run(cid_a, cid_b):
        plugin = _plugin(tau=1.0)
        plugin.set_identity_map({cid_a: "client_0", cid_b: "client_1"})
        cohort = _results([(cid_a, device), (cid_b, other)])
        plugin.observe_cohort(cohort, server_round=1)
        plugin.score_updates([cohort[1]], server_round=1)

        plugin.set_identity_map({cid_a: "client_0_new1", cid_b: "client_1"})
        respawn = _results([(cid_a, device), (cid_b, other)])
        plugin.observe_cohort(respawn, server_round=5)
        scores = plugin.score_updates(respawn, server_round=5)
        decisions = [
            {k: v for k, v in row.items()
             if k not in ("current_cid", "reentry_event_key")}
            for row in plugin.reentry_events
        ]
        return decisions, scores

    stable, stable_scores = run("raw0", "raw1")
    relabelled, relabelled_scores = run("zzz-9999", "aaa-0001")
    assert stable == relabelled
    assert stable_scores == relabelled_scores


def test_registry_entries_are_keyed_by_claimed_identity_not_flower_cid():
    plugin = _plugin()
    plugin.set_identity_map({"opaque-cid": "client_7"})
    plugin.observe_cohort(_results([("opaque-cid", _fingerprint(1))]), server_round=1)
    entry = plugin.registry.entry_for_session("client_7")
    assert entry.session_key == "client_7"
    assert entry.flower_cid == "opaque-cid"  # audit only
    with pytest.raises(KeyError):
        plugin.registry.entry_for_session("opaque-cid")


# ---------------------------------------------------------------------------
# End-to-end through PluggableStrategy
# ---------------------------------------------------------------------------

def test_hard_drop_reaches_the_base_strategy_as_an_absent_client():
    """The blocked client must not appear in what the base strategy aggregates."""
    from flowerfl.byzantine_defense import PluggableStrategy

    seen = {}

    class _RecordingBase:
        def aggregate_fit(self, server_round, results, failures):
            seen[server_round] = [proxy.cid for proxy, _ in results]
            return None, {}

    device = _fingerprint(1)
    plugin = _plugin(tau=1.0)
    strategy = PluggableStrategy(_RecordingBase(), plugins=[plugin])

    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", device), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    plugin.score_updates([cohort[1]], server_round=1)  # flag raw0 upstream

    plugin.set_identity_map({"raw0new": "client_0_new1", "raw1": "client_1"})
    respawn = _results([("raw0new", device), ("raw1", _fingerprint(2))])
    strategy.aggregate_fit(5, respawn, [])
    assert seen[5] == ["raw1"]


# ---------------------------------------------------------------------------
# A.6 — the rejection record is written at the call site UNCONDITIONALLY
# ---------------------------------------------------------------------------

def test_score_updates_records_every_upstream_rejection_even_when_already_flagged():
    """The gated `flag()` call cannot carry this; the record must precede it.

    Round 1 rejects raw0 → it is flagged (`flag_round=1`). Round 2 rejects it
    AGAIN, but `flag()` is gated on `not entry.flag_status`, so nothing in the
    entry lifecycle moves. Only an unconditional record preserves round 2 — and
    round 2 is exactly what the A.5(a) Euclidean counterfactual needs, because
    under τ′ that entry may never have been flagged in round 1 at all.
    """
    plugin = _plugin()
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    for server_round in (1, 2):
        plugin.observe_cohort(cohort, server_round=server_round)
        # raw0 is dropped by the upstream detector in BOTH rounds.
        plugin.score_updates(_results([("raw1", _fingerprint(2))]),
                             server_round=server_round)

    entry = plugin.registry.entry_for_session(plugin._session_key("raw0"))
    assert entry.flag_status and entry.flag_round == 1, "flag gating is unchanged"

    rejections = [
        (e.server_round, e.session_key) for e in plugin.registry.upstream_rejections()
    ]
    assert rejections == [
        (1, plugin._session_key("raw0")),
        (2, plugin._session_key("raw0")),
    ], "the round-2 rejection of an already-flagged entry must survive"


def test_surviving_clients_are_never_recorded_as_rejections():
    plugin = _plugin()
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    plugin.score_updates(cohort, server_round=1)
    assert plugin.registry.upstream_rejections() == ()


# ---------------------------------------------------------------------------
# Gate (e) telemetry — the denominator is PARTICIPATING client-rounds
#
# `ScenarioStrategy.aggregate_fit` returns before `super()` on the discovery
# round, so `observe_cohort` never sees discovery fits. That bypass is
# deliberate (its own docstring: the discovery round dispatches ALL clients
# including unscheduled ones, and feeding them here would enrol devices the
# scenario never scheduled and set `_enrollment_round` off a non-scenario round,
# corrupting `_is_initial_enrollment` — the logic that keeps initial enrolments
# out of the re-entry numerator). EMISSION_CONTRACT_20260808 § 4.5 fixes the
# reading: gate (e)'s "100 % of client-rounds" is "100 % of *participating*
# client-rounds", since `control_benign_churn` deliberately drops clients
# (census 959, not 1 000). Discovery fits are not scenario participation, so
# excluding them is the contract-consistent denominator — which is why the field
# NAMES say `participating_`, rather than a docstring saying it quietly.
# ---------------------------------------------------------------------------

def test_emission_telemetry_names_state_the_participating_denominator():
    plugin = _plugin()
    assert hasattr(plugin, "participating_client_round_count")
    assert hasattr(plugin, "participating_missing_fingerprint_count")
    assert hasattr(plugin, "participating_fingerprint_emission_rate")
    # The bare names read as whole-run rates and must not survive.
    assert not hasattr(plugin, "client_round_count")
    assert not hasattr(plugin, "missing_fingerprint_count")
    assert not hasattr(plugin, "fingerprint_emission_rate")


def test_discovery_round_fits_never_reach_the_plugin_by_design():
    """Fails if discovery fits ever start arriving at `observe_cohort`.

    Drives a real `ScenarioStrategy` through its discovery round with a real
    fingerprint plugin attached. The plugin must see nothing: no participating
    client-rounds, no registry entries, no enrollment round.
    """
    from flwr.common import Code, Status, ndarrays_to_parameters

    from flowerfl.scenario_strategy import ScenarioStrategy

    plugin = _plugin()
    base = SimpleNamespace(aggregate_fit=lambda r, res, f: (None, {}))
    strategy = ScenarioStrategy(
        base,
        plugins=[plugin],
        scenario_path="scenarios/S0_clean_baseline.json",
        signal_logger=None,
    )

    rng = np.random.default_rng(0)
    discovery = []
    for pid in range(4):
        params = ndarrays_to_parameters([rng.normal(size=4).astype(np.float32)])
        discovery.append((
            SimpleNamespace(cid=f"raw{pid}"),
            SimpleNamespace(
                status=Status(code=Code.OK, message=""),
                parameters=params,
                num_examples=100,
                metrics={
                    "partition_id": pid,
                    "fingerprint": encode_fingerprint(_fingerprint(pid)),
                },
            ),
        ))

    strategy.aggregate_fit(1, discovery, [])

    assert plugin.participating_client_round_count == 0, (
        "discovery fits reached observe_cohort — the bypass that protects "
        "_is_initial_enrollment has been removed, or the denominator changed"
    )
    assert plugin.participating_missing_fingerprint_count == 0
    assert len(plugin.registry) == 0
    assert plugin.enrollment_round is None
    # …and the counter is live, so the assertion above is not vacuous.
    plugin.observe_cohort(_results([("raw0", _fingerprint(1))]), server_round=2)
    assert plugin.participating_client_round_count == 1

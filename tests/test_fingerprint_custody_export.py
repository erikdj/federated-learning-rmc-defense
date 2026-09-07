"""The run-end fingerprint custody record — A.5(a) / A.6 observation log.

Design authority
----------------
The H3 workflow in `docs/reproduction/experiments.md` summarizes these
historical protocol requirements:

* **A.5(a)** pre-registers a NON-GATING naive Euclidean baseline comparator —
  nearest-flagged matching on the UN-WHITENED 180-dim vector, τ′ derived by the
  identical two-stage dev-FPR = 1 % procedure, scored on the same held-out
  events and reported as Δrecall against the Mahalanobis registry.
* **A.6** is the blocking build prerequisite: the schema-v5 re-entry row carries
  `asserted_match` / `asserted_parent_logical_id` / `min_d` / `tau` but **not**
  the 180-dim vector, so A.5(a) is computable offline at zero extra compute only
  if the vector is persisted "before the validation run launches; otherwise
  A.5(a) is unrecoverable without a re-run".

The resolution: the signal-log schema stays **v5 and untouched**; the registry
custody export in `scripts/run_phase4_flower.py` is the *sibling artifact*.

Everything here is PASSIVE. Not one of these fields may reach τ, Σ, a match
decision or any gate — `tests/test_fingerprint_registry.py` carries the
byte-identical-decisions proof.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from flowerfl.fingerprint_registry import EMA_ALPHA, FingerprintRegistry, MahalanobisMetric  # noqa: E402

DIM = 8


def _registry(tau: float = 5.0, **kwargs) -> FingerprintRegistry:
    return FingerprintRegistry(
        tau=tau, metric=MahalanobisMetric.identity(DIM), dim=DIM, **kwargs
    )


class _FakeEnforcement:
    value = "observe"


class _FakePlugin:
    """The attribute surface `_fingerprint_custody` reads off the real plugin."""

    name = "Fingerprint"

    def __init__(self, registry):
        self.registry = registry
        self.enforcement_mode = _FakeEnforcement()
        self.enrollment_round = 1
        self.initial_enrollments = {"dev-a", "dev-b"}
        self.participating_client_round_count = 6
        self.participating_missing_fingerprint_count = 0
        self.reentry_events = [{"reentry_event_key": "r:2:c9"}]

    def participating_fingerprint_emission_rate(self):
        return 1.0


def _custody(registry):
    from run_phase4_flower import _fingerprint_custody

    return _fingerprint_custody([_FakePlugin(registry)])


def _decode(row):
    """The exact one-liner the artifact advertises in `read_offline`."""
    import base64

    return np.frombuffer(base64.b64decode(row["fingerprint_vec_b64"]), dtype="<f8")


def _drive(registry):
    """Two devices over three rounds, with one flag and one re-entry."""
    for r in (1, 2, 3):
        registry.observe("dev-a", np.full(DIM, 1.0 + r), server_round=r,
                         logical_id="client_0", flower_cid=f"cid-a{r}")
        registry.observe("dev-b", np.full(DIM, 5.0 + r), server_round=r,
                         logical_id="client_1", flower_cid=f"cid-b{r}")
        if r == 1:
            registry.flag("dev-a", reason="upstream_filter", server_round=r)
        # Upstream rejects dev-a EVERY round; only round 1 moves the entry.
        registry.record_upstream_rejection("dev-a", server_round=r)
    registry.observe("dev-a-new", np.full(DIM, 2.0), server_round=4,
                     logical_id="client_0_new", flower_cid="cid-a-new")
    return registry


# ---------------------------------------------------------------------------
# Pre-existing contract — every field that was exported before must survive
# ---------------------------------------------------------------------------

LEGACY_TOP_LEVEL = {
    "tau_posture", "tau", "observe_only_reason", "metric_provenance",
    "enforcement_mode", "enrollment_round", "initial_enrollments",
    "entry_count", "flagged_entry_count", "participating_client_round_count",
    "participating_missing_fingerprint_count",
    "participating_fingerprint_emission_rate", "emission_denominator",
    "reentry_assertion_count", "entries",
}
LEGACY_ENTRY = {
    "entry_id", "session_key", "logical_id", "flower_cid", "first_seen_round",
    "last_seen_round", "flag_status", "flag_reason", "flag_round", "generation",
    "parent_entry_id", "fingerprint_sha256", "fingerprint_finite",
}


def test_custody_preserves_every_pre_existing_field():
    custody = _custody(_drive(_registry()))
    assert LEGACY_TOP_LEVEL <= set(custody)
    assert custody["tau_posture"] == "locked"
    assert custody["tau"] == 5.0
    assert custody["entry_count"] == 3
    assert custody["flagged_entry_count"] >= 1
    assert custody["participating_fingerprint_emission_rate"] == 1.0
    assert "participating" in custody["emission_denominator"]
    assert "discovery" in custody["emission_denominator"]
    assert custody["reentry_assertion_count"] == 1
    for entry in custody["entries"]:
        assert LEGACY_ENTRY <= set(entry)


def test_custody_returns_none_without_a_fingerprint_plugin():
    from run_phase4_flower import _fingerprint_custody

    assert _fingerprint_custody([]) is None


# ---------------------------------------------------------------------------
# A.6 — the observation log rides in the custody record
# ---------------------------------------------------------------------------

def test_custody_carries_the_observation_log():
    registry = _drive(_registry())
    log = _custody(registry)["observation_log"]

    assert log["row_count"] == 7
    assert log["truncated"] is False
    assert log["dropped_count"] == 0
    assert log["dim"] == DIM
    assert log["ema_alpha"] == EMA_ALPHA
    assert log["encoding"] == "base64_float64_le"
    assert "b64decode" in log["read_offline"]
    assert [(r["server_round"], r["session_key"]) for r in log["rows"]] == [
        (1, "dev-a"), (1, "dev-b"),
        (2, "dev-a"), (2, "dev-b"),
        (3, "dev-a"), (3, "dev-b"),
        (4, "dev-a-new"),
    ]
    assert np.array_equal(_decode(log["rows"][0]), np.full(DIM, 2.0))


def test_custody_observation_log_is_as_observed_not_ema_state():
    registry = _drive(_registry())
    log = _custody(registry)["observation_log"]
    # dev-a's third draw is 4.0 across the board; its EMA state is not.
    assert np.array_equal(_decode(log["rows"][4]), np.full(DIM, 4.0))
    ema = registry.entry_for_session("dev-a").fingerprint_vec
    assert not np.allclose(ema, 4.0)


def test_custody_observation_log_round_trips_bit_exactly_through_the_result_json():
    """The custody record is written with a plain `json.dumps(result, indent=2)`."""
    hard = np.array(
        [1.2345678901234567e150, np.nextafter(1.0, 2.0), -np.nextafter(0.1, 0.0),
         5e-324, 0.0, -0.0, 1e-308, 123456789.123456789],
        dtype=np.float64,
    )
    registry = _registry()
    registry.observe("dev-a", hard, server_round=1, logical_id="client_0")

    restored = json.loads(json.dumps({"fingerprint_registry": _custody(registry)},
                                     indent=2))
    row = restored["fingerprint_registry"]["observation_log"]["rows"][0]
    assert _decode(row).tobytes() == hard.tobytes()


def test_custody_reports_truncation_loudly():
    registry = _drive(_registry(max_observation_rows=3))
    log = _custody(registry)["observation_log"]
    assert log["truncated"] is True
    assert log["dropped_count"] == 4
    assert log["row_count"] == 3
    assert log["max_rows"] == 3


# ---------------------------------------------------------------------------
# The pre-lock OBSERVE-ONLY posture must still be analysable
# ---------------------------------------------------------------------------

def test_observe_only_prelock_registry_still_logs_observations():
    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = _drive(
        ObserveOnlyFingerprintRegistry(
            metric=MahalanobisMetric.identity(DIM), dim=DIM, reason="gate (c)"
        )
    )
    custody = _custody(registry)
    assert custody["tau_posture"] == "observe_only_prelock"
    assert custody["tau"] is None
    assert custody["observation_log"]["row_count"] == 7
    assert np.array_equal(
        _decode(custody["observation_log"]["rows"][0]), np.full(DIM, 2.0)
    )


# ---------------------------------------------------------------------------
# A.5(a) sufficiency — the offline replay needs nothing this record lacks
# ---------------------------------------------------------------------------

def test_custody_alone_supports_the_a5a_offline_replay():
    """Replay smoke: reconstruct every candidate's state at every round.

    For each re-entry event the comparator needs (i) which entries were FLAGGED
    as of that round and (ii) each candidate's vector at that round. (i) comes
    from the per-entry `flag_status` / `flag_round` lifecycle (flagging is
    monotone — nothing is ever un-flagged); (ii) is reconstructed exactly from
    the observation log by replaying the pre-registered EMA in log order.
    """
    registry = _drive(_registry())
    custody = _custody(registry)
    log = custody["observation_log"]

    state: dict = {}
    for row in log["rows"]:
        vec = _decode(row)
        key = row["session_key"]
        if key not in state:
            state[key] = vec
        else:
            state[key] = (1.0 - log["ema_alpha"]) * state[key] + log["ema_alpha"] * vec

    by_key = {e["session_key"]: e for e in custody["entries"]}
    for key, replayed in state.items():
        live = registry.entry_for_session(key).fingerprint_vec
        assert np.array_equal(replayed, live), key
        import hashlib

        assert hashlib.sha256(replayed.tobytes()).hexdigest() == \
            by_key[key]["fingerprint_sha256"]

    # (i): flagged-as-of-round is derivable, and the flag round is present.
    flagged_at_round_4 = {
        e["session_key"] for e in custody["entries"]
        if e["flag_status"] and e["flag_round"] is not None and e["flag_round"] <= 4
    }
    assert "dev-a" in flagged_at_round_4


@pytest.mark.parametrize("field", ["row_count", "dropped_count", "max_rows"])
def test_custody_observation_log_counters_are_plain_ints(field):
    log = _custody(_drive(_registry()))["observation_log"]
    assert isinstance(log[field], int) and not isinstance(log[field], bool)


# ---------------------------------------------------------------------------
# A.6 — the upstream-rejection log rides in custody too
# ---------------------------------------------------------------------------

def test_custody_carries_the_upstream_rejection_log():
    log = _custody(_drive(_registry()))["upstream_rejection_log"]
    assert log["row_count"] == 3
    assert log["truncated"] is False
    assert log["dropped_count"] == 0
    assert [(r["server_round"], r["session_key"]) for r in log["rows"]] == [
        (1, "dev-a"), (2, "dev-a"), (3, "dev-a"),
    ]


def test_custody_upstream_rejection_log_reports_truncation_loudly():
    log = _custody(_drive(_registry(max_upstream_rejection_rows=2)))[
        "upstream_rejection_log"
    ]
    assert log["truncated"] is True
    assert log["dropped_count"] == 1
    assert log["row_count"] == 2
    assert log["max_rows"] == 2


def test_custody_replays_the_inherited_flag_counterfactual():
    """The gap that motivated this log, closed and asserted.

    `dev-a` is flagged once at round 1 and rejected again at rounds 2 and 3.
    The entry lifecycle alone reports a single event at round 1; the rejection
    log reports all three, so an offline Euclidean replay that does NOT
    inherit-flag `dev-a` can still recover the rounds at which the upstream
    detector rejected it.
    """
    custody = _custody(_drive(_registry()))
    entry = next(e for e in custody["entries"] if e["session_key"] == "dev-a")
    assert entry["flag_round"] == 1  # the lifecycle knows only this

    rounds = sorted(
        r["server_round"] for r in custody["upstream_rejection_log"]["rows"]
        if r["session_key"] == "dev-a"
    )
    assert rounds == [1, 2, 3]
    # No ground truth entered the log — the opaque session key and nothing else.
    assert set(custody["upstream_rejection_log"]["rows"][0]) == {
        "server_round", "session_key"
    }

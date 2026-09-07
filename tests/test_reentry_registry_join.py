"""The REGISTRY half of the schema-v5 re-entry contract (dry-run F2, gap 4).

`ScenarioStrategy._reentry_event_fields` used to populate only the ground-truth
half and document the gap in-code. This suite pins the join that closes it.

Design authority
----------------
* v1.10 § 5.1 **INTEGRITY ASSERTION** — the registry's re-link decisions must be
  "computable from the signal log **independent of enforcement**", written
  "**regardless of any downstream action** (block, downweight, or accept)".
  `test_the_registry_half_is_identical_under_all_three_enforcement_modes` is the
  test the whole D7 primary metric rests on, at the STRATEGY seam (the plugin's
  own copy lives in `tests/test_fingerprint_plugin.py`).
* v1.10 § 5.1 field table — the 10 re-entry fields; `flowerfl/signal_logger.py`
  `REENTRY_EVENT_FIELDS` is the single source. Schema stays **v5**: no field is
  added here.
* The join key is the frozen dedup unit `{run_id}:{round}:{cid}`
  (`signal_logger.build_reentry_event_key`), which the plugin's `_event_row`
  builds identically — so the two halves join by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, Status, ndarrays_to_parameters

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.fingerprint import encode_fingerprint  # noqa: E402
from flowerfl.fingerprint_plugin import (  # noqa: E402
    ENFORCEMENT_MODES,
    FingerprintDefensePlugin,
)
from flowerfl.fingerprint_registry import (  # noqa: E402
    FingerprintRegistry,
    MahalanobisMetric,
)
from flowerfl.scenario_strategy import ScenarioStrategy  # noqa: E402
from flowerfl.signal_logger import (  # noqa: E402
    REENTRY_EVENT_FIELDS,
    REENTRY_NEAREST_FIELDS,
)

S3 = str(PROJECT_ROOT / "rmc" / "scenarios" / "S3_identity_reset_only.json")

DIM = 8
RUN_UID = "run-join-test"

REGISTRY_HALF = (
    "asserted_match",
    "asserted_parent_entry_id",
    "asserted_parent_logical_id",
    "min_d",
    "tau",
    "generation",
)
GROUND_TRUTH_HALF = (
    "reentry_event_key",
    "current_cid",
    "gt_logical_id",
    "gt_is_malicious",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fingerprint(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=DIM)


def _results(spec):
    rng = np.random.default_rng(0)
    out = []
    for cid, fingerprint in spec:
        params = ndarrays_to_parameters([rng.normal(size=4).astype(np.float32)])
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=100,
            metrics={"fingerprint": encode_fingerprint(np.asarray(fingerprint))},
        )
        out.append((proxy, fit))
    return out


def _strategy(plugins, run_uid: str | None = RUN_UID) -> ScenarioStrategy:
    """A ScenarioStrategy wired only far enough to exercise the join."""
    logger = SimpleNamespace(run_uid=run_uid) if run_uid is not None else None
    return ScenarioStrategy(
        SimpleNamespace(), plugins=plugins, scenario_path=S3, signal_logger=logger,
    )


def _fake_plugin(rows):
    """A minimal duck-typed stand-in exposing the plugin's public surface."""
    return SimpleNamespace(
        name="Fingerprint",
        reentry_events=list(rows),
        set_run_id=lambda run_id: None,
    )


def _row(key, **overrides):
    row = {
        "reentry_event_key": key,
        "server_round": 5,
        "current_cid": "raw1",
        "gt_logical_id": "client_1_new1",
        "asserted_match": True,
        "asserted_parent_entry_id": "fp-0001",
        "asserted_parent_logical_id": "client_1",
        "min_d": 3.25,
        "tau": 19.09,
        "generation": 1,
        "nearest_entry_id": "fp-0001",
        "nearest_logical_id": "client_1",
    }
    row.update(overrides)
    return row


def _fields(strategy, *, server_round=5, logical_cid="client_1_new1",
            flower_cid="raw1", partition_id=1, tenure=1, run_uid=RUN_UID):
    return strategy._reentry_event_fields(
        server_round, logical_cid, flower_cid, partition_id, tenure, run_uid,
    )


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_registry_half_lands_on_the_joined_event_row():
    strategy = _strategy([_fake_plugin([_row(f"{RUN_UID}:5:raw1")])])
    fields = _fields(strategy)

    assert fields["reentry_event_key"] == f"{RUN_UID}:5:raw1"
    assert fields["gt_logical_id"] == "client_1"          # ground-truth half intact
    assert fields["asserted_match"] is True
    assert fields["asserted_parent_entry_id"] == "fp-0001"
    assert fields["asserted_parent_logical_id"] == "client_1"
    assert fields["min_d"] == pytest.approx(3.25)
    assert fields["tau"] == pytest.approx(19.09)
    assert fields["generation"] == 1


@pytest.mark.unit
def test_the_row_carries_exactly_the_declared_field_set():
    """Schema stays v5 — the join adds no field the logger does not declare.
    The declared set is the frozen v1.10 table PLUS the additive nearest pair
    (signal_logger.REENTRY_NEAREST_FIELDS, PR #57)."""
    strategy = _strategy([_fake_plugin([_row(f"{RUN_UID}:5:raw1")])])
    assert set(_fields(strategy)) == set(REENTRY_EVENT_FIELDS) | set(REENTRY_NEAREST_FIELDS)


@pytest.mark.unit
def test_the_nearest_pair_lands_on_the_joined_event_row():
    """EXP-057 smoke finding (2026-08-15): the registry records the nearest
    candidate unconditionally and the plugin row carries it, but the strategy
    merge projected onto the six frozen fields only — every fleet row logged
    nearest_* as null and an unmatched event would misread as 'empty pool'.
    The values must SURVIVE the join."""
    strategy = _strategy([_fake_plugin([_row(f"{RUN_UID}:5:raw1")])])
    fields = _fields(strategy)
    assert fields["nearest_entry_id"] == "fp-0001"
    assert fields["nearest_logical_id"] == "client_1"


@pytest.mark.unit
def test_an_unmatched_row_keeps_its_nearest_candidate_through_the_join():
    """The whole point of the pair: nearest survives even when no match fired."""
    strategy = _strategy([_fake_plugin([_row(
        f"{RUN_UID}:5:raw1", asserted_match=False,
        asserted_parent_entry_id=None, asserted_parent_logical_id=None,
        min_d=310.0, nearest_entry_id="fp-0007", nearest_logical_id="client_7",
    )])])
    fields = _fields(strategy)
    assert fields["asserted_match"] is False
    assert fields["nearest_entry_id"] == "fp-0007"
    assert fields["nearest_logical_id"] == "client_7"


@pytest.mark.unit
def test_a_row_with_no_registry_assertion_keeps_the_nearest_pair_null():
    strategy = _strategy([_fake_plugin([])])
    fields = _fields(strategy)
    assert fields["nearest_entry_id"] is None
    assert fields["nearest_logical_id"] is None


@pytest.mark.unit
def test_a_row_with_no_registry_assertion_keeps_the_registry_half_null():
    strategy = _strategy([_fake_plugin([])])
    fields = _fields(strategy)
    assert fields["reentry_event_key"] == f"{RUN_UID}:5:raw1"
    assert all(fields[name] is None for name in REGISTRY_HALF)


@pytest.mark.unit
def test_a_non_reentry_row_is_all_nulls():
    """Every non-event row carries the contract's fields as explicit nulls —
    unchanged behaviour, re-pinned so the join cannot leak onto ordinary rows."""
    strategy = _strategy([_fake_plugin([_row(f"{RUN_UID}:5:raw1")])])
    fields = _fields(strategy, logical_cid="client_1", tenure=4)
    assert all(value is None for value in fields.values())


@pytest.mark.unit
def test_the_join_is_by_key_not_by_position():
    """Two assertions in flight; the row must take ITS key's registry half."""
    rows = [
        _row(f"{RUN_UID}:5:raw9", current_cid="raw9", asserted_parent_entry_id="fp-9999",
             min_d=99.0, generation=7),
        _row(f"{RUN_UID}:5:raw1", asserted_parent_entry_id="fp-0001", min_d=3.25),
    ]
    strategy = _strategy([_fake_plugin(rows)])
    fields = _fields(strategy)
    assert fields["asserted_parent_entry_id"] == "fp-0001"
    assert fields["min_d"] == pytest.approx(3.25)
    assert fields["generation"] == 1


@pytest.mark.unit
def test_a_mismatched_run_id_does_not_join():
    """A registry row stamped with a different run id belongs to a different
    run's event population and must never be joined in."""
    strategy = _strategy([_fake_plugin([_row("some-other-run:5:raw1")])])
    fields = _fields(strategy)
    assert all(fields[name] is None for name in REGISTRY_HALF)


@pytest.mark.unit
def test_non_finite_min_d_and_tau_are_logged_as_explicit_null():
    """JSON has no `inf`/`NaN` literal, and `signal_logger._jsonify` already
    maps non-finite numpy floats to null. The observe-only pre-lock posture
    produces exactly this shape (`min_d = inf`, no τ)."""
    strategy = _strategy([
        _fake_plugin([_row(f"{RUN_UID}:5:raw1", asserted_match=False,
                           asserted_parent_entry_id=None,
                           asserted_parent_logical_id=None,
                           min_d=float("inf"), tau=float("nan"), generation=0)])
    ])
    fields = _fields(strategy)
    assert fields["asserted_match"] is False
    assert fields["min_d"] is None
    assert fields["tau"] is None
    json.dumps(fields, allow_nan=False)  # strictly valid JSON


@pytest.mark.unit
def test_the_run_id_is_stamped_from_the_signal_logger_run_uid():
    """The two halves join on `{run_id}:{round}:{cid}`; the plugin's run id must
    therefore be the SAME run_uid the signal logger writes."""
    stamped = {}
    plugin = SimpleNamespace(
        name="Fingerprint", reentry_events=[],
        set_run_id=lambda run_id: stamped.setdefault("run_id", run_id),
    )
    _strategy([plugin], run_uid="run-abc")
    assert stamped["run_id"] == "run-abc"


@pytest.mark.unit
def test_incumbent_plugins_are_untouched_by_the_run_id_stamp():
    """OFF BY DEFAULT: a plugin without `set_run_id` (every incumbent) must be
    left exactly as it was."""
    from flowerfl.byzantine_defense import TGEnsemblePlugin

    plugin = TGEnsemblePlugin(num_malicious=9, num_to_keep=9)
    _strategy([plugin])
    assert not hasattr(plugin, "set_run_id")


@pytest.mark.unit
def test_a_strategy_with_no_fingerprint_plugin_still_writes_the_null_half():
    from flowerfl.byzantine_defense import TGEnsemblePlugin

    strategy = _strategy([TGEnsemblePlugin(num_malicious=9, num_to_keep=9)])
    fields = _fields(strategy)
    assert fields["gt_logical_id"] == "client_1"
    assert all(fields[name] is None for name in REGISTRY_HALF)


@pytest.mark.unit
def test_the_truth_key_integrity_check_still_raises():
    """Preserved from before the join: a returning identity that does not
    resolve to its parent's partition fails LOUDLY rather than emitting a
    corrupt event (v1.17 misattribution-family rule)."""
    strategy = _strategy([_fake_plugin([_row(f"{RUN_UID}:5:raw1")])])
    with pytest.raises(RuntimeError, match="truth-key mismatch"):
        _fields(strategy, partition_id=17)


@pytest.mark.unit
def test_the_stale_gap_comment_is_gone():
    """The in-code note said the registry half 'stays null until the fingerprint
    plugin lands'. It has landed."""
    src = (PROJECT_ROOT / "flowerfl" / "scenario_strategy.py").read_text()
    assert "stays null until the fingerprint plugin lands" not in src


# ---------------------------------------------------------------------------
# The § 5.1 INTEGRITY ASSERTION, at the strategy seam
# ---------------------------------------------------------------------------

def _run_arm(enforcement_mode: str) -> dict:
    """One S3-shaped mini-run: enrol, flag via the upstream chain, re-enter.

    Returns the joined re-entry row for the re-entrant's CID-appearance.
    """
    registry = FingerprintRegistry(
        tau=50.0, metric=MahalanobisMetric.identity(DIM), dim=DIM
    )
    plugin = FingerprintDefensePlugin(
        registry=registry, run_id=RUN_UID, enforcement_mode=enforcement_mode,
        expected_dim=DIM,
    )
    strategy = _strategy([plugin])

    # Round 1 — enrolment cohort.
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1"})
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=1)
    # client_1 is rejected by an upstream detector -> flagged.
    scores = plugin.score_updates(cohort[:1], server_round=1)
    plugin.filter_updates(cohort[:1], scores)

    # Round 5 — client_1 returns under a new identity with the same device
    # fingerprint. The registry must link it.
    plugin.set_identity_map({"raw0": "client_0", "raw1": "client_1_new1"})
    cohort = _results([("raw0", _fingerprint(1)), ("raw1", _fingerprint(2))])
    plugin.observe_cohort(cohort, server_round=5)
    scores = plugin.score_updates(cohort, server_round=5)
    plugin.filter_updates(cohort, scores)

    return _fields(strategy)


@pytest.mark.unit
def test_the_registry_half_is_identical_under_all_three_enforcement_modes():
    """v1.10 § 5.1: the re-link fields are written REGARDLESS of the downstream
    action, so the D7 metric is computable from the signal log independent of
    enforcement. Flipping block -> downweight -> accept must leave the joined
    row byte-identical."""
    rendered = {
        mode: json.dumps(_run_arm(mode), sort_keys=True, default=str)
        for mode in ENFORCEMENT_MODES
    }
    assert len(set(rendered.values())) == 1, rendered


@pytest.mark.unit
@pytest.mark.parametrize("mode", sorted(ENFORCEMENT_MODES))
def test_a_real_match_is_written_under_every_enforcement_mode(mode):
    """Not merely equal — actually POPULATED. Three identical rows of nulls
    would satisfy the invariance test while proving nothing."""
    fields = _run_arm(mode)
    assert fields["reentry_event_key"] == f"{RUN_UID}:5:raw1"
    assert fields["gt_logical_id"] == "client_1"
    assert fields["gt_is_malicious"] is True    # S3 resets are adversarial
    assert fields["asserted_match"] is True
    assert fields["asserted_parent_logical_id"] == "client_1"
    assert fields["generation"] == 1
    assert fields["tau"] == pytest.approx(50.0)
    assert fields["min_d"] is not None

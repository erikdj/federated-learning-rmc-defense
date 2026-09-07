"""H4 § 6 diagnostics block — synthetic mini-run, null-vs-zero semantics.

Contract (spec 2026-08-16 § 6 + BUILD_CONTRACT): per-unit `h4_diagnostics`
with `kept_set_size[]`, `empty_aggregate_rounds` (count + rounds),
`kept_set_malicious_fraction[]` (ground truth from the scenario schedule),
per-layer removal tallies — UNIFORM across arms; a tally whose layer is
absent in the arm is null, never zero; an empty kept set yields a null
malicious fraction (0/0), never 0.0.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.h4_diagnostics import build_h4_diagnostics  # noqa: E402


def _strategy_stub(*, plugins, trace, schedule, adv_ids, cid_to_partition):
    return SimpleNamespace(
        _plugins=[SimpleNamespace(name=n) for n in plugins],
        _h4_chain_trace=trace,
        _schedule_cache=schedule,
        _adv_ids=set(adv_ids),
        _cid_to_partition=dict(cid_to_partition),
        _round_offset=1,
    )


def _schedule_entry(partition, logical):
    return {"partition_id": partition, "logical_id": logical,
            "attack_type": None, "attack_params": {}}


#: 3 clients: partitions 0 (malicious), 1, 2 (honest). Two scored rounds.
_SCHEDULE = {
    1: [_schedule_entry(0, "client_0"), _schedule_entry(1, "client_1"),
        _schedule_entry(2, "client_2")],
    2: [_schedule_entry(0, "client_0"), _schedule_entry(1, "client_1"),
        _schedule_entry(2, "client_2")],
}
_C2P = {"cid0": 0, "cid1": 1, "cid2": 2}
_ADV = {"client_0"}


def _trace(stages_by_round, kept_by_round):
    trace = {}
    for rnd in stages_by_round:
        trace[rnd] = {
            "input_cids": ["cid0", "cid1", "cid2"],
            "stages": stages_by_round[rnd],
            "kept_cids": kept_by_round[rnd],
        }
    return trace


@pytest.mark.unit
def test_full_chain_arm_tallies_and_capture_fraction():
    """Arm-1 shape: detector drops the malicious client in round 2 (server
    round 3); FP drops an honest one; aggregator rejects nothing."""
    stages = {
        2: [
            {"plugin": "H2PrimeDetector", "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": []},
            {"plugin": "Fingerprint", "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": []},
            {"plugin": "KrumDefense", "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": ["cid1"]},
        ],
        3: [
            {"plugin": "H2PrimeDetector", "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": ["cid0"]},
            {"plugin": "Fingerprint", "input_cids": ["cid1", "cid2"],
             "dropped_cids": ["cid2"]},
            {"plugin": "KrumDefense", "input_cids": ["cid1"],
             "dropped_cids": []},
        ],
    }
    kept = {2: ["cid0", "cid2"], 3: ["cid1"]}
    strategy = _strategy_stub(
        plugins=["H2PrimeDetector", "Fingerprint", "KrumDefense"],
        trace=_trace(stages, kept), schedule=_SCHEDULE,
        adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["layers_present"] == {
        "detector": "H2PrimeDetector", "fp": "Fingerprint",
        "aggregator": "KrumDefense",
    }
    assert block["rounds"]["server"] == [2, 3]
    assert block["rounds"]["scenario"] == [1, 2]
    assert block["kept_set_size"] == [2, 1]
    # round 2: kept = {cid0 (malicious), cid2} -> 1/2; round 3: kept = {cid1}.
    assert block["kept_set_malicious_fraction"] == [0.5, 0.0]
    removals = block["per_layer_removals_per_round"]
    assert removals["detector_dropped_malicious"] == [0, 1]  # cid0 in r3
    assert removals["detector_dropped_honest"] == [0, 0]
    assert removals["fp_hard_dropped"] == [0, 1]
    assert removals["aggregator_rejected"] == [1, 0]
    assert block["per_layer_removal_totals"] == {
        "detector_dropped_honest": 0,
        "detector_dropped_malicious": 1,
        "fp_hard_dropped": 1,
        "aggregator_rejected": 1,
    }
    assert block["empty_aggregate_rounds"]["count"] == 0


@pytest.mark.unit
def test_absent_layers_are_null_never_zero():
    """Arm-2 shape (Krum only): detector and FP tallies must be null."""
    stages = {
        2: [{"plugin": "KrumDefense",
             "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": ["cid0"]}],
    }
    strategy = _strategy_stub(
        plugins=["KrumDefense"], trace=_trace(stages, {2: ["cid1", "cid2"]}),
        schedule=_SCHEDULE, adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["layers_present"] == {
        "detector": None, "fp": None, "aggregator": "KrumDefense",
    }
    removals = block["per_layer_removals_per_round"]
    assert removals["detector_dropped_honest"] is None
    assert removals["detector_dropped_malicious"] is None
    assert removals["fp_hard_dropped"] is None
    assert removals["aggregator_rejected"] == [1]
    totals = block["per_layer_removal_totals"]
    assert totals["detector_dropped_honest"] is None
    assert totals["fp_hard_dropped"] is None
    assert totals["aggregator_rejected"] == 1


@pytest.mark.unit
def test_fedavg_arm_has_all_layer_tallies_null_but_kept_series_present():
    """Arm-8 shape (no plugins): every layer tally null; kept-set series and
    malicious fraction still computed — the block is UNIFORM across arms."""
    strategy = _strategy_stub(
        plugins=[], trace=_trace({2: []}, {2: ["cid0", "cid1", "cid2"]}),
        schedule=_SCHEDULE, adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["layers_present"] == {
        "detector": None, "fp": None, "aggregator": None,
    }
    assert all(
        v is None for v in block["per_layer_removals_per_round"].values()
    )
    assert block["kept_set_size"] == [3]
    assert block["kept_set_malicious_fraction"] == [1 / 3]


@pytest.mark.unit
def test_empty_aggregate_round_is_recorded_with_null_fraction():
    """Blackout round: kept set empty -> counted in empty_aggregate_rounds,
    malicious fraction null (0/0), never 0.0."""
    stages = {
        2: [{"plugin": "H2PrimeDetector",
             "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": ["cid0", "cid1", "cid2"]}],
    }
    strategy = _strategy_stub(
        plugins=["H2PrimeDetector"], trace=_trace(stages, {2: []}),
        schedule=_SCHEDULE, adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["kept_set_size"] == [0]
    assert block["kept_set_malicious_fraction"] == [None]
    assert block["empty_aggregate_rounds"] == {
        "count": 1, "server_rounds": [2], "scenario_rounds": [1],
    }
    removals = block["per_layer_removals_per_round"]
    assert removals["detector_dropped_malicious"] == [1]
    assert removals["detector_dropped_honest"] == [2]


@pytest.mark.unit
def test_krum_tge_fp_arm_maps_tge_as_detector_layer():
    """Arm 4 (legacy Krum-first order): TGE is the detector layer and Krum
    the aggregator layer regardless of chain position."""
    stages = {
        2: [
            {"plugin": "KrumDefense", "input_cids": ["cid0", "cid1", "cid2"],
             "dropped_cids": ["cid0"]},
            {"plugin": "TGEnsemble", "input_cids": ["cid1", "cid2"],
             "dropped_cids": ["cid2"]},
            {"plugin": "Fingerprint", "input_cids": ["cid1"],
             "dropped_cids": []},
        ],
    }
    strategy = _strategy_stub(
        plugins=["KrumDefense", "TGEnsemble", "Fingerprint"],
        trace=_trace(stages, {2: ["cid1"]}),
        schedule=_SCHEDULE, adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["layers_present"]["detector"] == "TGEnsemble"
    assert block["layers_present"]["aggregator"] == "KrumDefense"
    removals = block["per_layer_removals_per_round"]
    assert removals["detector_dropped_honest"] == [1]      # cid2
    assert removals["aggregator_rejected"] == [1]          # cid0
    assert removals["fp_hard_dropped"] == [0]


@pytest.mark.unit
def test_no_results_round_is_reported_not_omitted():
    """a round with ZERO incoming FitRes must appear in the
    block (kept_set_size 0, in empty_aggregate_rounds) with zero per-layer
    drops — distinguishable from an all-filtered blackout, never omitted."""
    trace = {
        2: {"input_cids": [], "stages": [], "kept_cids": []},   # no results
        3: {"input_cids": ["cid0", "cid1", "cid2"],
            "stages": [{"plugin": "H2PrimeDetector",
                        "input_cids": ["cid0", "cid1", "cid2"],
                        "dropped_cids": ["cid0", "cid1", "cid2"]}],
            "kept_cids": []},                                   # blackout
    }
    strategy = _strategy_stub(
        plugins=["H2PrimeDetector"], trace=trace, schedule=_SCHEDULE,
        adv_ids=_ADV, cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["rounds"]["server"] == [2, 3]
    assert block["kept_set_size"] == [0, 0]
    assert block["kept_set_malicious_fraction"] == [None, None]
    assert block["empty_aggregate_rounds"]["server_rounds"] == [2, 3]
    removals = block["per_layer_removals_per_round"]
    # no-results round: zero detector drops; blackout round: all three dropped
    assert removals["detector_dropped_honest"] == [0, 2]
    assert removals["detector_dropped_malicious"] == [0, 1]


@pytest.mark.unit
def test_pluggable_strategy_records_a_trace_for_an_empty_results_round():
    """The strategy-level half of the P2 fix: aggregate_fit([]) records the
    empty trace rather than early-returning past it."""
    from flwr.server.strategy import FedAvg

    from flowerfl.byzantine_defense import PluggableStrategy

    strategy = PluggableStrategy(FedAvg(), plugins=[])
    aggregated, metrics = strategy.aggregate_fit(5, [], [])
    assert aggregated is None and metrics == {}
    assert strategy._h4_chain_trace[5] == {
        "input_cids": [], "stages": [], "kept_cids": [],
    }


@pytest.mark.unit
def test_unresolvable_cid_refuses_loudly():
    strategy = _strategy_stub(
        plugins=[], trace=_trace({2: []}, {2: ["cid_unknown"]}),
        schedule=_SCHEDULE, adv_ids=_ADV, cid_to_partition=_C2P,
    )
    with pytest.raises(RuntimeError, match="misattribution"):
        build_h4_diagnostics(strategy)


@pytest.mark.unit
def test_non_scenario_strategy_returns_none():
    assert build_h4_diagnostics(SimpleNamespace()) is None


@pytest.mark.unit
def test_c0_like_schedule_yields_zero_fraction_series():
    """C0: no adversarial identities -> the fraction series is 0.0 by
    construction (not null — the kept sets are non-empty)."""
    strategy = _strategy_stub(
        plugins=[], trace=_trace({2: []}, {2: ["cid0", "cid1", "cid2"]}),
        schedule=_SCHEDULE, adv_ids=set(), cid_to_partition=_C2P,
    )
    block = build_h4_diagnostics(strategy)
    assert block["kept_set_malicious_fraction"] == [0.0]

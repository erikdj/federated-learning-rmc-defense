"""Signal-log attribution + discovery-round aggregation must be arrival-order-safe
(drift investigation of the aggregate_fit order pin).

The order pin canonicalizes `results` by ascending partition_id. If it lived ONLY
in PluggableStrategy.aggregate_fit (the parent), a ScenarioStrategy run would sort
the LOCAL list the plugins score, but ScenarioStrategy._maybe_log_signals would
still iterate the CALLER's arrival-ordered list — so `krum_scores.get(idx)` (keyed
by the sorted positions the plugin stored) would attach to the WRONG client's
ground-truth signal row whenever arrival != partition order. That is the v1.17
misattribution bug class, and it would silently corrupt recall@FPR — worse than
the tie/summation wobble the pin fixes.

The fix canonicalizes ONCE at ScenarioStrategy.aggregate_fit's OUTER entry, before
any branching, so super().aggregate_fit AND _maybe_log_signals (AND the discovery
round's direct base aggregation) all consume the SAME ordered list.

Both tests FAIL on the PluggableStrategy-only sort and PASS after the
ScenarioStrategy sort.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from flowerfl.byzantine_defense import KrumDefensePlugin
from flowerfl.scenario_strategy import ScenarioStrategy

DIM = 32
N = 6  # partitions 0..5; dynamic f=ceil(6/2)-1=2 -> num_closest=2 (computable, non-uniform)


def _vec(p):
    # distinct per-partition vectors -> distinct (order-independent) Krum scores
    return np.random.default_rng(500 + p).normal(p * 0.5, 0.1, DIM).astype(np.float32)


def _make_results(partitions_in_arrival_order):
    out = []
    for p in partitions_in_arrival_order:
        proxy = SimpleNamespace(cid=f"raw{p}")
        fit = SimpleNamespace(
            parameters=ndarrays_to_parameters([_vec(p)]),
            num_examples=100,
            metrics={"partition_id": float(p), "train_loss": 0.1},
        )
        out.append((proxy, fit))
    return out


def _krum():
    return KrumDefensePlugin(num_malicious=2, num_to_keep=2, dynamic_f=True)


def _base():
    return FedAvg(
        initial_parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
        min_fit_clients=1, min_available_clients=1, min_evaluate_clients=1,
    )


class _CapturingSignalLogger:
    def __init__(self):
        self.rows = None

    def log_round(self, server_round, scenario_round, per_client_records):
        self.rows = per_client_records


def _reference_krum_by_partition(server_round):
    """The Krum score FOR EACH CLIENT (score value is order-independent), keyed by
    partition_id — the independent ground truth the signal log must reproduce."""
    res = _make_results(list(range(N)))
    scores = _krum().score_updates(res, server_round)  # {position: score}
    return {int(res[pos][1].metrics["partition_id"]): scores[pos] for pos in scores}


def _scenario_strategy(signal_logger):
    strat = ScenarioStrategy(
        base_strategy=_base(), plugins=[_krum()],
        scenario_path=None, signal_logger=signal_logger,
    )
    # runtime discovery internals (normally filled during the discovery round)
    strat._mapping_ready = True
    strat._round_offset = 1
    strat._cid_to_partition = {f"raw{p}": p for p in range(N)}
    strat._partition_to_cid = {p: f"raw{p}" for p in range(N)}
    strat._schedule_cache = {
        1: [{"partition_id": p, "logical_id": f"client_{p}", "attack_type": ""} for p in range(N)]
    }
    strat._adv_ids = set()
    strat._tenure_first_seen = {}
    return strat


@pytest.mark.unit
def test_signal_log_krum_score_attribution_under_arrival_disorder():
    """LOAD-BEARING: each signal-log row's krum_score must be the score computed
    FOR THAT logical client (joined by partition), even when arrival order differs
    from partition order. Misattribution here silently corrupts recall@FPR."""
    server_round = 2  # scenario_round 1
    ref = _reference_krum_by_partition(server_round)

    cap = _CapturingSignalLogger()
    strat = _scenario_strategy(cap)
    arrival = list(reversed(range(N)))  # 5,4,3,2,1,0 -> arrival != partition order
    strat.aggregate_fit(server_round, _make_results(arrival), failures=[])

    assert cap.rows, "no signal rows were logged"
    assert len(cap.rows) == N
    for row in cap.rows:
        pid = int(row["physical_partition_id"])
        assert row["krum_score"] is not None
        assert row["krum_score"] == pytest.approx(ref[pid], rel=1e-6, abs=1e-9), (
            f"krum_score misattributed for partition {pid}: "
            f"logged {row['krum_score']} != client's own score {ref[pid]}"
        )


@pytest.mark.unit
def test_discovery_round_aggregate_hash_order_invariant():
    """P1-2: the discovery round aggregates through self._base.aggregate_fit
    directly (bypassing the plugin path), so it too must consume the canonical
    order — its FedAvg reduction must be identical across arrival permutations."""
    server_round = 1  # discovery round (offset 1)

    def _run(arrival):
        strat = ScenarioStrategy(
            base_strategy=_base(), plugins=[_krum()],
            scenario_path=None, signal_logger=None,
        )
        strat._mapping_ready = False  # force the discovery branch
        strat._round_offset = 1
        agg, _ = strat.aggregate_fit(server_round, _make_results(arrival), failures=[])
        return parameters_to_ndarrays(agg)[0].tobytes()

    ref = _run(list(range(N)))
    assert _run(list(reversed(range(N)))) == ref
    for s in range(3):
        perm = [int(i) for i in np.random.default_rng(10 + s).permutation(N)]
        assert _run(perm) == ref, "discovery-round aggregate wobbled under permutation"

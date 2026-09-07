"""Signal-log score join when the aggregator is DOWNSTREAM of a filter.

In the H4 § 7c-bis chains, Krum/TrustScore score only the detector's
survivors, so their positional `_round_scores` index a SUBSET — a positional
join against the full round ordering would misattribute scores across
clients (the v1.17 bug class). The fix keys the join by CID through the
chain trace's per-stage input order. This suite pins:

  * a client dropped upstream carries a truthful null krum_score;
  * every survivor's krum_score is the score Krum computed FOR THAT client
    on the survivor subset (cross-checked against an independent
    same-subset reference), under arrival disorder.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from flowerfl.byzantine_defense import KrumDefensePlugin
from flowerfl.scenario_strategy import ScenarioStrategy

DIM = 32
N = 6


def _vec(p):
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


class _DropPartitionZero:
    """Stand-in detector layer: hard-drops partition 0's client."""

    name = "H2PrimeDetector"

    def on_round_start(self, *a):
        pass

    def observe_cohort(self, *a):
        pass

    def set_identity_map(self, m):
        pass

    def score_updates(self, results, server_round):
        return {
            i: (0.0 if int(f.metrics["partition_id"]) == 0 else 1.0)
            for i, (_p, f) in enumerate(results)
        }

    def filter_updates(self, results, scores, threshold=0.0):
        return [r for i, r in enumerate(results) if scores[i] > 0.0]

    def on_round_end(self, *a):
        pass


class _CapturingSignalLogger:
    def __init__(self):
        self.rows = None

    def log_round(self, server_round, scenario_round, per_client_records):
        self.rows = per_client_records


def _krum():
    return KrumDefensePlugin(num_malicious=2, num_to_keep=2, dynamic_f=True)


def _scenario_strategy(signal_logger, plugins):
    base = FedAvg(
        initial_parameters=ndarrays_to_parameters(
            [np.zeros(DIM, dtype=np.float32)]
        ),
        min_fit_clients=1, min_available_clients=1, min_evaluate_clients=1,
    )
    strat = ScenarioStrategy(
        base_strategy=base, plugins=plugins,
        scenario_path=None, signal_logger=signal_logger,
    )
    strat._mapping_ready = True
    strat._round_offset = 1
    strat._cid_to_partition = {f"raw{p}": p for p in range(N)}
    strat._partition_to_cid = {p: f"raw{p}" for p in range(N)}
    strat._schedule_cache = {
        1: [{"partition_id": p, "logical_id": f"client_{p}",
             "attack_type": ""} for p in range(N)]
    }
    strat._adv_ids = set()
    strat._tenure_first_seen = {}
    return strat


def _reference_survivor_krum(server_round):
    """Krum scores computed independently on the SAME survivor subset
    (partitions 1..5, canonical order), keyed by partition."""
    res = _make_results(list(range(1, N)))
    scores = _krum().score_updates(res, server_round)
    return {
        int(res[pos][1].metrics["partition_id"]): scores[pos]
        for pos in scores
    }


@pytest.mark.unit
def test_downstream_krum_scores_join_by_cid_not_position():
    server_round = 2
    ref = _reference_survivor_krum(server_round)

    cap = _CapturingSignalLogger()
    strat = _scenario_strategy(cap, [_DropPartitionZero(), _krum()])
    arrival = list(reversed(range(N)))  # arrival != partition order
    strat.aggregate_fit(server_round, _make_results(arrival), failures=[])

    assert cap.rows, "no signal rows were logged"
    by_partition = {row["physical_partition_id"]: row for row in cap.rows}
    assert set(by_partition) == set(range(N))
    # The upstream-dropped client was never scored by Krum: truthful null.
    assert by_partition[0]["krum_score"] is None
    # Every survivor's krum_score is the score computed FOR THAT client on
    # the survivor subset — not a positional neighbour's.
    for p in range(1, N):
        assert by_partition[p]["krum_score"] == pytest.approx(ref[p]), p


@pytest.mark.unit
def test_incumbent_first_position_krum_join_is_unchanged():
    """With Krum FIRST (the incumbent arms), the cid-keyed join must equal
    the historical positional join value-for-value."""
    server_round = 2
    res = _make_results(list(range(N)))
    positional = _krum().score_updates(res, server_round)
    ref = {
        int(res[pos][1].metrics["partition_id"]): positional[pos]
        for pos in positional
    }

    cap = _CapturingSignalLogger()
    strat = _scenario_strategy(cap, [_krum()])
    strat.aggregate_fit(server_round, _make_results(list(reversed(range(N)))),
                        failures=[])
    by_partition = {row["physical_partition_id"]: row for row in cap.rows}
    for p in range(N):
        assert by_partition[p]["krum_score"] == pytest.approx(ref[p]), p

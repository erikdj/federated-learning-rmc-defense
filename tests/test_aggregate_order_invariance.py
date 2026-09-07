"""Order-invariance of PluggableStrategy.aggregate_fit (drift investigation, 2026-07-27).

aggregate_fit consumed Flower results in Ray ARRIVAL order, which the async
simulation scheduler does not pin across runs. Two order-sensitivities followed:

  * KrumDefensePlugin.filter_updates breaks EXACT score ties with Python's stable
    sort over the positionally-indexed ``results`` list, so a tie straddling the
    keep/drop cutoff silently changed WHICH client survived — purely by arrival
    order. Byte-identical updates are a real occurrence: RMC's duplicate-partition
    identities (DATASET_CONFIGS[...]["duplicate_partitions"]) produce literal
    duplicate clients.
  * FedAvg's weighted-sum reduction is not float-summation-order-invariant, so the
    aggregate itself wobbles at the last bits even with a fixed keep-set.

The fix pins arrival order to ascending ``partition_id`` (reported on EVERY
FitRes: flowerfl/client_app.py:149) at the very top of aggregate_fit — the single
shared entry point for ALL defense compositions. These tests assert keep-set AND
aggregate-hash invariance across permutations, including the exact-tie boundary.
The BOUNDARY case FAILS on the unsorted code and passes with the fix.
"""
import logging

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from flowerfl.byzantine_defense import (
    KrumDefensePlugin,
    PluggableStrategy,
    TGEnsemblePlugin,
)

LOGGER_NAME = "flowerfl.byzantine_defense"
DIM = 64
N_KEEP, N_TIED, N_OUT = 8, 3, 9  # n=20; dynamic f=9 -> keep=9 -> 1 slot contested by the 3 tied


class _FakeClient:
    def __init__(self, cid):
        self.cid = cid


class _FakeFitRes:
    def __init__(self, vec, partition_id=None, include_pid=True):
        self.parameters = ndarrays_to_parameters([vec.astype(np.float32)])
        self.num_examples = 1000
        self.metrics = {"train_loss": 0.1}
        if include_pid:
            self.metrics["partition_id"] = float(partition_id)


class _CapturingFedAvg(FedAvg):
    """Records the cid ORDER of the filtered set aggregate_fit hands the base.

    frozenset(captured_order) is the KEEP-SET; the tuple preserves the summation
    order FedAvg actually reduces in.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.captured_order = None

    def aggregate_fit(self, server_round, results, failures):
        self.captured_order = tuple(c.cid for c, _ in results)
        return super().aggregate_fit(server_round, results, failures)


def _population(tie):
    """8 clear-keep + 3 boundary (byte-identical when tie=True) + 9 clear-drop."""
    pop = []  # (cid, partition_id, vec)
    for i in range(N_KEEP):
        rng = np.random.default_rng(1000 + i)
        pop.append((f"keep_{i}", i, rng.normal(0.0, 0.01, DIM)))
    if tie:
        tied = np.random.default_rng(1100).normal(0.3, 0.05, DIM)
        for i in range(N_TIED):
            pop.append((f"mid_{i}", N_KEEP + i, tied.copy()))  # exact tie at the cutoff
    else:
        for i in range(N_TIED):
            rng = np.random.default_rng(1100 + i)
            pop.append((f"mid_{i}", N_KEEP + i, rng.normal(0.3, 0.05, DIM)))
    for i in range(N_OUT):
        rng = np.random.default_rng(1200 + i)
        pop.append((f"out_{i}", N_KEEP + N_TIED + i, rng.normal(3.0, 0.01, DIM)))
    return pop


def _orders(cids):
    orders = [list(cids), list(reversed(cids))]
    for s in range(3):
        perm = np.random.default_rng(100 + s).permutation(len(cids))
        orders.append([cids[i] for i in perm])
    return orders


def _results(pop, order):
    by_cid = {cid: (cid, pid, vec) for cid, pid, vec in pop}
    out = []
    for cid in order:
        _c, pid, vec = by_cid[cid]
        out.append((_FakeClient(cid), _FakeFitRes(vec, pid)))
    return out


def _run(pop, order, plugins_factory):
    base = _CapturingFedAvg(
        initial_parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
    )
    strategy = PluggableStrategy(base, plugins=plugins_factory())
    aggregated, _ = strategy.aggregate_fit(2, _results(pop, order), failures=[])
    keep_set = frozenset(base.captured_order) if base.captured_order else frozenset()
    agg_hash = parameters_to_ndarrays(aggregated)[0].tobytes() if aggregated is not None else None
    return keep_set, agg_hash


def _krum_only():
    return [KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)]


def _krum_tge():
    return [
        KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True),
        TGEnsemblePlugin(num_malicious=9, num_to_keep=9, ramp_rounds=8),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("tie", [False, True], ids=["separated", "boundary"])
def test_krum_keep_set_order_invariant(tie):
    """Krum's keep-set must not depend on arrival order — the crisp signal: with a
    byte-identical tie at the cutoff, the unsorted code keeps whichever tied client
    arrived first, so permuting flips the keep-set (this fails pre-fix)."""
    pop = _population(tie)
    cids = [c for c, _, _ in pop]
    ref_keep, _ = _run(pop, cids, _krum_only)
    assert len(ref_keep) == 9
    for order in _orders(cids):
        keep, _ = _run(pop, order, _krum_only)
        assert keep == ref_keep, f"Krum keep-set flipped under permutation (tie={tie})"


@pytest.mark.unit
@pytest.mark.parametrize("tie", [False, True], ids=["separated", "boundary"])
def test_composed_aggregate_order_invariant(tie):
    """The real production composition (Krum+TGE): keep-set AND the aggregated
    parameters must be identical across arrival-order permutations."""
    pop = _population(tie)
    cids = [c for c, _, _ in pop]
    ref_keep, ref_hash = _run(pop, cids, _krum_tge)
    assert ref_hash is not None
    for order in _orders(cids):
        keep, agg = _run(pop, order, _krum_tge)
        assert keep == ref_keep, f"composed keep-set flipped under permutation (tie={tie})"
        assert agg == ref_hash, f"aggregate hash wobbled under permutation (tie={tie})"


@pytest.mark.unit
def test_missing_partition_id_falls_back_stably_and_warns(caplog):
    """If partition_id is ever absent, the sort falls back to a STABLE cid order
    (never arrival order) and warns loudly — still order-invariant."""
    rng = np.random.default_rng(3)
    pop = [(f"c{i}", None, rng.normal(size=DIM)) for i in range(6)]
    cids = [c for c, _, _ in pop]

    def _results_no_pid(order):
        by_cid = {cid: vec for cid, _, vec in pop}
        return [(_FakeClient(cid), _FakeFitRes(by_cid[cid], include_pid=False)) for cid in order]

    def _run_no_pid(order):
        base = _CapturingFedAvg(
            initial_parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
            min_fit_clients=1, min_available_clients=1, min_evaluate_clients=1,
        )
        strategy = PluggableStrategy(base, plugins=[])
        strategy.aggregate_fit(2, _results_no_pid(order), failures=[])
        return base.captured_order

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        ref = _run_no_pid(cids)
        rev = _run_no_pid(list(reversed(cids)))

    assert ref == rev, "fallback ordering was not stable (leaked arrival order)"
    assert any("partition_id" in r.getMessage() for r in caplog.records), \
        "missing partition_id did not warn"

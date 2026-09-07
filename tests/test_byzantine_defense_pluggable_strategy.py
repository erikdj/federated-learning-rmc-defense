"""Tests for PluggableStrategy's post-aggregation bookkeeping.

req 6 (models): the runner has never persisted the final global model
anywhere (ground-truthed: no state_dict/torch.save call exists in
flowerfl/ or scripts/run_phase4_flower.py before this change). The minimal,
additive fix is for PluggableStrategy to remember the last aggregated
Parameters object across rounds so the runner can convert it to a
state_dict and torch.save it after the simulation ends — with zero change
to aggregate_fit's return value or the FL aggregation math itself.
"""
from types import SimpleNamespace

import numpy as np


def _fake_base_strategy(aggregated):
    return SimpleNamespace(
        aggregate_fit=lambda server_round, results, failures: (aggregated, {"m": 1.0}),
        initialize_parameters=lambda client_manager: None,
        configure_fit=lambda server_round, parameters, client_manager: [],
        configure_evaluate=lambda server_round, parameters, client_manager: [],
        aggregate_evaluate=lambda server_round, results, failures: (None, {}),
        evaluate=lambda server_round, parameters: None,
    )


def _fake_results(n=2, dim=4):
    from flwr.common import ndarrays_to_parameters
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        params = ndarrays_to_parameters([rng.normal(size=dim).astype(np.float32)])
        proxy = SimpleNamespace(cid=str(i))
        fit = SimpleNamespace(parameters=params, num_examples=10, metrics={})
        out.append((proxy, fit))
    return out


def test_aggregate_fit_remembers_last_aggregated_parameters():
    from flwr.common import ndarrays_to_parameters
    from flowerfl.byzantine_defense import PluggableStrategy

    aggregated = ndarrays_to_parameters([np.ones(3, dtype=np.float32)])
    strategy = PluggableStrategy(base_strategy=_fake_base_strategy(aggregated), plugins=[])
    assert strategy._last_aggregated_parameters is None

    strategy.aggregate_fit(1, _fake_results(), [])
    assert strategy._last_aggregated_parameters is aggregated


def test_last_aggregated_parameters_survives_across_rounds():
    """A later round's None-aggregation (e.g. all results filtered) must not
    erase the previous round's remembered parameters — the runner wants the
    LAST successfully aggregated model, not necessarily the final round's."""
    from flwr.common import ndarrays_to_parameters
    from flowerfl.byzantine_defense import PluggableStrategy

    round1 = ndarrays_to_parameters([np.ones(3, dtype=np.float32)])
    strategy = PluggableStrategy(base_strategy=_fake_base_strategy(round1), plugins=[])
    strategy.aggregate_fit(1, _fake_results(), [])
    assert strategy._last_aggregated_parameters is round1

    # Round 2: base strategy returns None (nothing aggregated this round).
    strategy._base = _fake_base_strategy(None)
    strategy.aggregate_fit(2, _fake_results(), [])
    assert strategy._last_aggregated_parameters is round1


def test_aggregate_fit_return_value_unchanged_by_bookkeeping():
    """The bookkeeping must be purely additive: aggregate_fit's return value
    (what Flower's simulation loop actually consumes) is untouched."""
    from flwr.common import ndarrays_to_parameters
    from flowerfl.byzantine_defense import PluggableStrategy

    aggregated = ndarrays_to_parameters([np.ones(3, dtype=np.float32)])
    strategy = PluggableStrategy(base_strategy=_fake_base_strategy(aggregated), plugins=[])
    params, metrics = strategy.aggregate_fit(1, _fake_results(), [])
    assert params is aggregated
    assert metrics == {"m": 1.0}

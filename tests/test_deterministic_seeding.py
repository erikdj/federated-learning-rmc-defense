"""Determinism tests for the seeding fix.

Root cause (confirmed): the experiment `seed` from run_config was recorded in
filenames/logs but never wired into torch's global RNG, so the initial global
model built in `server_app.server_fn` (and every client's per-round training)
was drawn from OS entropy — identical-seed runs were only statistically
reproducible, never byte-identical.

These tests are fast (no Ray, no data) and pin the three guarantees the fix
provides:

  (a) `seed_everything` makes `create_model` weight init reproducible.
  (b) `derive_seed` is stable, order-sensitive, and per-(client, round) unique.
  (c) the real `server_fn` builds a byte-identical initial model for two
      same-seed constructions — this is the production-level identity that was
      RED before the fix (server_fn did not seed before `create_model`).

The end-to-end proof (a full 2-round simulation reproducing byte-identically)
lives in `tests/test_run_phase4_flower_unification.py::test_anchor_2_self_determinism`.
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# (a) seed_everything makes model init reproducible
# ---------------------------------------------------------------------------

def _state_dict_bytes(net):
    """Deterministic, comparable snapshot of a model's parameters."""
    return [v.detach().cpu().clone() for v in net.state_dict().values()]


def _tensors_equal(a, b):
    return len(a) == len(b) and all(
        x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )


def test_seed_everything_makes_create_model_deterministic():
    from flowerfl.seeding import seed_everything
    from flowerfl.task import create_model

    seed_everything(42)
    first = _state_dict_bytes(create_model("edge", input_shape=20))

    # Reseed to the same value: a fresh construction must be byte-identical.
    seed_everything(42)
    second = _state_dict_bytes(create_model("edge", input_shape=20))

    assert _tensors_equal(first, second), (
        "seed_everything(42) before create_model must yield identical weights"
    )


def test_unseeded_create_model_diverges_without_reseed():
    """Control: without reseeding, consecutive create_model calls differ.

    This is exactly the pre-fix production defect — server_fn/client_fn built
    models with no prior seeding, so identical-seed runs diverged.
    """
    from flowerfl.task import create_model

    torch.manual_seed(1234)
    a = _state_dict_bytes(create_model("edge", input_shape=20))
    b = _state_dict_bytes(create_model("edge", input_shape=20))
    assert not _tensors_equal(a, b), (
        "two create_model calls with no intervening seed should differ "
        "(demonstrates why seeding is required)"
    )


def test_different_seeds_give_different_models():
    from flowerfl.seeding import seed_everything
    from flowerfl.task import create_model

    seed_everything(1)
    m1 = _state_dict_bytes(create_model("edge", input_shape=20))
    seed_everything(2)
    m2 = _state_dict_bytes(create_model("edge", input_shape=20))
    assert not _tensors_equal(m1, m2), "distinct seeds must give distinct init"


# ---------------------------------------------------------------------------
# (b) derive_seed properties
# ---------------------------------------------------------------------------

def test_derive_seed_is_stable():
    from flowerfl.seeding import derive_seed

    assert derive_seed(42, 3, 7) == derive_seed(42, 3, 7)


def test_derive_seed_is_order_sensitive():
    from flowerfl.seeding import derive_seed

    assert derive_seed(42, 3, 7) != derive_seed(42, 7, 3)


def test_derive_seed_varies_across_clients_and_rounds():
    from flowerfl.seeding import derive_seed

    base = 42
    # Distinct client at same round.
    assert derive_seed(base, 0, 1) != derive_seed(base, 1, 1)
    # Same client across rounds.
    assert derive_seed(base, 0, 1) != derive_seed(base, 0, 2)
    # Distinct base seeds.
    assert derive_seed(1, 0, 1) != derive_seed(2, 0, 1)


def test_derive_seed_in_valid_range():
    from flowerfl.seeding import derive_seed

    for comps in [(0, 0), (5, 9), (19, 12), (12345, 678)]:
        s = derive_seed(42, *comps)
        assert 0 <= s < 2**31, "derived seed must be a valid manual_seed input"
        # accepted by torch without raising
        torch.Generator().manual_seed(s)


def test_seed_everything_seeds_numpy_and_random():
    import random as _random

    from flowerfl.seeding import seed_everything

    seed_everything(7)
    t1, n1, r1 = torch.rand(3), np.random.rand(3), _random.random()
    seed_everything(7)
    t2, n2, r2 = torch.rand(3), np.random.rand(3), _random.random()
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)
    assert r1 == r2


# ---------------------------------------------------------------------------
# (c) production server_fn builds a deterministic initial model (RED pre-fix)
# ---------------------------------------------------------------------------

def _server_initial_ndarrays(seed: int):
    from types import SimpleNamespace

    from flwr.common import parameters_to_ndarrays

    from flowerfl.server_app import server_fn

    run_config = {
        "strategy": "Krum",
        "dataset": "edge",
        "num-server-rounds": 1,
        "malicious-fraction": 0.1,
        "seed": seed,
    }
    ctx = SimpleNamespace(run_config=dict(run_config), node_config={})
    components = server_fn(ctx)
    return parameters_to_ndarrays(components.strategy.initial_parameters)


def test_server_fn_initial_model_is_deterministic_across_constructions():
    """RED before the fix: server_fn did not seed before create_model, so two
    same-seed constructions produced different initial global models."""
    a = _server_initial_ndarrays(seed=123)
    b = _server_initial_ndarrays(seed=123)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)), (
        "server_fn must build a byte-identical initial model for a fixed seed"
    )


def test_server_fn_initial_model_depends_on_seed():
    a = _server_initial_ndarrays(seed=123)
    c = _server_initial_ndarrays(seed=456)
    assert not (len(a) == len(c) and all(np.array_equal(x, y) for x, y in zip(a, c))), (
        "different seeds should give different initial models"
    )


# ---------------------------------------------------------------------------
# (d) server_round reaches fit config on NON-scenario strategy paths too
#     (: only ScenarioStrategy injected it; legacy strategies
#     fell back to server_round=0 every round -> the same derived seed, i.e.
#     identical shuffle/dropout/noise streams replayed each round)
# ---------------------------------------------------------------------------

import pytest


class _FakeClientManager:
    """Minimal stand-in satisfying what flwr strategies call in configure_fit:
    num_available and sample. Returned proxies are never invoked."""

    def __init__(self, n: int):
        from types import SimpleNamespace

        self._clients = [SimpleNamespace(cid=str(i)) for i in range(n)]

    def num_available(self):
        return len(self._clients)

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return self._clients[:num_clients]


def _build_strategy(strategy_name: str):
    from types import SimpleNamespace

    from flowerfl.server_app import server_fn

    ctx = SimpleNamespace(
        run_config={
            "strategy": strategy_name,
            "dataset": "edge",
            "num-server-rounds": 3,
            "malicious-fraction": 0.1,
            "seed": 42,
        },
        node_config={},
    )
    return server_fn(ctx).strategy


@pytest.mark.parametrize(
    "strategy_name",
    ["Krum", "FedTrimmedAvg", "FedMedian", "FedAvg", "PluginKrum"],
)
def test_non_scenario_strategy_delivers_server_round_in_fit_config(strategy_name):
    """RED pre-fix: legacy (non-Scenario*) strategies never sent server_round,
    so FlowerClient.fit derived the SAME seed every round on those paths."""
    from flwr.common import ndarrays_to_parameters

    from flowerfl.seeding import seed_everything
    from flowerfl.task import create_model, get_weights

    strategy = _build_strategy(strategy_name)
    seed_everything(42)
    params = ndarrays_to_parameters(get_weights(create_model("edge", input_shape=20)))
    manager = _FakeClientManager(4)
    for rnd in (1, 2, 3):
        configs = strategy.configure_fit(rnd, params, manager)
        assert configs, f"{strategy_name}: no clients configured"
        for _, fit_ins in configs:
            assert fit_ins.config.get("server_round") == rnd, (
                f"{strategy_name}: fit config must carry server_round={rnd}, "
                f"got {fit_ins.config!r}"
            )


def test_scenario_double_injection_is_harmless():
    """ScenarioStrategy.configure_fit injects server_round on top of the base
    strategy's on_fit_config_fn (server_fn installs both on Scenario* paths).
    Both set the same key to the same value, so the composition must deliver
    exactly server_round."""
    from flwr.common import ndarrays_to_parameters
    from flwr.server.strategy import FedAvg

    from flowerfl.scenario_strategy import ScenarioStrategy
    from flowerfl.seeding import seed_everything
    from flowerfl.task import create_model, get_weights

    seed_everything(42)
    params = ndarrays_to_parameters(get_weights(create_model("edge", input_shape=20)))
    base = FedAvg(
        initial_parameters=params,
        min_fit_clients=4,
        min_available_clients=4,
        fraction_fit=1.0,
        # Same hook server_fn installs via base_params ( P2 fix).
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )
    strategy = ScenarioStrategy(base, plugins=[], scenario_path=None)
    manager = _FakeClientManager(4)
    for rnd in (1, 2):
        configs = strategy.configure_fit(rnd, params, manager)
        assert configs
        for _, fit_ins in configs:
            assert fit_ins.config.get("server_round") == rnd, (
                f"double injection must resolve to server_round={rnd}, "
                f"got {fit_ins.config!r}"
            )

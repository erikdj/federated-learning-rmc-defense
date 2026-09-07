"""Persistent per-client Adam optimizer state across federation rounds.

Implements the persistent-optimizer mode described in
docs/reproduction/experiments.md.

This module is the ONLY file in the codebase that patches Flower's default
per-round optimizer reset behavior. The default Flower simulation re-instantiates
client objects each round, which means a fresh Adam optimizer with zero-state
m/v tensors at every round — destroying the momentum estimates that smooth out
noisy minibatch gradients. The persistent_optimizer mode of the unified runner
opts into reading/writing this module's _OPTIMIZER_STATE dict at fit-time, so
Adam state survives across rounds for the same client.

Keying:
    Adam state is keyed by Flower's per-supernode node_id (an int provided via
    flwr.common.Context.node_id). Per flowerfl/scenario_strategy.py:62-65, each
    logical identity in a scenario (e.g., client_19 vs client_19_new1) maps to
    a distinct Flower partition_id, and therefore a distinct node_id. Rejoining
    under a new logical identity therefore receives fresh optimizer state, which
    is the semantics we want.

Lifecycle:
    The runner calls clear() before each (config, seed) tuple's run_simulation()
    invocation to eliminate state leakage as a confounding variable across
    independent runs.
"""
from __future__ import annotations

import copy

import torch

_OPTIMIZER_STATE: dict[int, dict] = {}


def save_state(node_id: int, optimizer: torch.optim.Optimizer) -> None:
    """Snapshot the optimizer's state_dict under node_id.

    Uses ``copy.deepcopy`` on the full ``state_dict()`` payload so the saved
    snapshot is fully independent of the live optimizer. ``state_dict()``
    already detaches tensors (the returned dict is a fresh container), but
    inner per-param state dicts and the ``param_groups`` list-of-dicts may
    still share references with the live optimizer's internals (e.g. Adam's
    ``step`` tensor). ``deepcopy`` is the only fully version-agnostic way to
    guarantee isolation.
    """
    _OPTIMIZER_STATE[node_id] = copy.deepcopy(optimizer.state_dict())


def load_state(node_id: int, optimizer: torch.optim.Optimizer) -> bool:
    """Restore previously-saved state into `optimizer`.

    Returns True if state was found and loaded, False if no state exists for
    this node_id (first round for this client; optimizer remains at default).
    """
    if node_id not in _OPTIMIZER_STATE:
        return False
    optimizer.load_state_dict(_OPTIMIZER_STATE[node_id])
    return True


def clear() -> None:
    """Drop all stored state.

    Called between independent runs (each unique (config, seed) tuple) by the
    runner, to prevent cross-run state leakage.
    """
    _OPTIMIZER_STATE.clear()

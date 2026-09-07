"""Unit tests for the persistent optimizer state singleton.

Tests the three-function API: save_state, load_state, clear.
"""
import torch
import pytest


def test_load_state_returns_false_for_missing_key():
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    assert po.load_state(node_id=12345, optimizer=opt) is False


def test_save_then_load_restores_state():
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    # Drive Adam's m/v away from zero by one step
    loss = net(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    saved = {k: v.clone() if torch.is_tensor(v) else v
             for k, v in opt.state_dict()['state'].get(0, {}).items()}
    po.save_state(node_id=42, optimizer=opt)

    # Fresh optimizer should be zero-state initially
    net2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(net2.parameters(), lr=0.001)
    ok = po.load_state(node_id=42, optimizer=opt2)
    assert ok is True
    # After load, opt2's state for param 0 should match the saved state
    loaded = opt2.state_dict()['state'].get(0, {})
    assert 'exp_avg' in loaded, "Adam state should have exp_avg after load"
    assert torch.allclose(loaded['exp_avg'], saved['exp_avg'])
    assert torch.allclose(loaded['exp_avg_sq'], saved['exp_avg_sq'])
    assert loaded['step'] == saved['step']


def test_clear_removes_all_state():
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = net(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    po.save_state(node_id=1, optimizer=opt)
    po.save_state(node_id=2, optimizer=opt)
    po.clear()

    net2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(net2.parameters(), lr=0.001)
    assert po.load_state(node_id=1, optimizer=opt2) is False
    assert po.load_state(node_id=2, optimizer=opt2) is False


def test_independent_states_per_node_id():
    from flowerfl import persistent_optimizer as po
    po.clear()
    # Drive two different optimizers to two different states
    net_a = torch.nn.Linear(4, 2)
    opt_a = torch.optim.Adam(net_a.parameters(), lr=0.001)
    for _ in range(3):
        loss = net_a(torch.randn(8, 4)).sum()
        loss.backward()
        opt_a.step()
        opt_a.zero_grad()
    po.save_state(node_id=10, optimizer=opt_a)

    net_b = torch.nn.Linear(4, 2)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=0.001)
    for _ in range(7):
        loss = net_b(torch.randn(8, 4)).sum()
        loss.backward()
        opt_b.step()
        opt_b.zero_grad()
    po.save_state(node_id=20, optimizer=opt_b)

    # Reload independently
    net_a2 = torch.nn.Linear(4, 2)
    opt_a2 = torch.optim.Adam(net_a2.parameters(), lr=0.001)
    assert po.load_state(node_id=10, optimizer=opt_a2) is True

    net_b2 = torch.nn.Linear(4, 2)
    opt_b2 = torch.optim.Adam(net_b2.parameters(), lr=0.001)
    assert po.load_state(node_id=20, optimizer=opt_b2) is True

    state_a = opt_a2.state_dict()['state'][0]
    state_b = opt_b2.state_dict()['state'][0]
    # Step counts differ (3 vs 7), proving independence
    assert state_a['step'] != state_b['step']


def test_save_state_isolates_param_groups():
    """param_groups list must be deep-copied, not stored by reference.

    Regression test for an issue where save_state() only deep-cloned
    state_dict()["state"] entries (which are dicts) and stored
    state_dict()["param_groups"] (a list-of-dicts) BY REFERENCE.
    A subsequent mutation of the live optimizer's param_groups (e.g.
    LR schedule update) would silently bleed into the saved snapshot.
    """
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    po.save_state(node_id=1, optimizer=opt)
    # Mutate live optimizer's param_groups
    opt.param_groups[0]["lr"] = 0.999
    # Saved snapshot must retain original lr
    snap = po._OPTIMIZER_STATE[1]
    assert snap["param_groups"][0]["lr"] == 0.001, \
        f"saved param_groups must be independent of live optimizer; got lr={snap['param_groups'][0]['lr']}"


def test_save_state_isolates_inner_state_tensors():
    """Inner state-dict tensors (e.g. Adam's `step`) must not share refs with
    the live optimizer's tensors.

    Regression test for a bug where save_state() only cloned tensors at the
    OUTER level of state_dict()'s 'state' map, but stored the per-param state
    dicts (containing live tensors like `step`) by reference. A subsequent
    .step() on the live optimizer then silently advanced the saved snapshot's
    step counter — a hidden mutation of supposedly-immutable saved state.
    """
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    # Drive Adam to step=1
    loss = net(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    po.save_state(node_id=1, optimizer=opt)
    snap = po._OPTIMIZER_STATE[1]
    saved_step = float(snap["state"][0]["step"])
    # Now step the live optimizer again
    loss = net(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    # The saved snapshot's step must NOT have advanced
    snap_step_after = float(snap["state"][0]["step"])
    assert snap_step_after == saved_step, (
        f"saved state.step must be independent of live optimizer; "
        f"saved at {saved_step}, now {snap_step_after} after live opt.step()"
    )


def test_train_accepts_external_optimizer_and_persists_state():
    """train() should accept an external Adam optimizer; state must survive
    across calls when the same optimizer instance is reused, OR when state
    is saved/loaded via persistent_optimizer between fresh instances."""
    from flowerfl.task import train
    from flowerfl.task import Net
    from torch.utils.data import DataLoader, TensorDataset
    from flowerfl import persistent_optimizer as po
    po.clear()

    # Build a tiny edge-shape dataset (45-dim → 2-class)
    X = torch.randn(64, 45)
    y = torch.randint(0, 2, (64,)).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    net = Net(input_shape=45)
    opt_first = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=3e-3)
    train(net, loader, epochs=1, optimizer=opt_first)
    po.save_state(node_id=1, optimizer=opt_first)
    # NOTE: Adam stores `step` as a 0-dim tensor that is shared by reference
    # across optimizers wrapping the same `net.parameters()`. Capture the
    # current value by .item() so a subsequent .step() on a sibling optimizer
    # doesn't retroactively mutate this snapshot.
    step_a = float(opt_first.state_dict()['state'][0]['step'])
    assert step_a > 0, "Adam should have stepped at least once"

    # Round 2 — fresh optimizer instance, but load state via persistent_optimizer
    opt_second = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=3e-3)
    assert po.load_state(node_id=1, optimizer=opt_second) is True
    train(net, loader, epochs=1, optimizer=opt_second)
    step_b = float(opt_second.state_dict()['state'][0]['step'])
    assert step_b > step_a, "step counter should advance from persisted base"

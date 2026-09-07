"""Tests for FlowerClient's optimizer-state wiring."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def _tiny_loader():
    X = torch.randn(64, 45)
    y = torch.randint(0, 2, (64,)).long()
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


def test_flower_client_reset_mode_does_not_persist_state():
    from flowerfl.client_app import FlowerClient
    from flowerfl.task import Net
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = Net(input_shape=45)
    client = FlowerClient(
        trainloader=_tiny_loader(),
        valloader=_tiny_loader(),
        net=net,
        is_malicious=False,
        partition_id=0,
        lr=0.001,
        local_epochs=1,
        weight_decay=3e-3,
        optimizer_state="reset",
        node_id=42,
    )
    params = client.get_parameters({})
    client.fit(params, {})
    client.fit(params, {})  # second round — fresh Adam each round
    assert 42 not in po._OPTIMIZER_STATE, "reset mode must NOT write to persistent_optimizer"


def test_flower_client_persistent_mode_round_trips_state():
    from flowerfl.client_app import FlowerClient
    from flowerfl.task import Net
    from flowerfl import persistent_optimizer as po
    po.clear()
    net = Net(input_shape=45)
    client = FlowerClient(
        trainloader=_tiny_loader(),
        valloader=_tiny_loader(),
        net=net,
        is_malicious=False,
        partition_id=0,
        lr=0.001,
        local_epochs=1,
        weight_decay=3e-3,
        optimizer_state="persistent",
        node_id=99,
    )
    params = client.get_parameters({})
    client.fit(params, {})
    assert 99 in po._OPTIMIZER_STATE, "persistent mode must save state under node_id"
    step_round1 = float(po._OPTIMIZER_STATE[99]['state'][0]['step'])
    client.fit(params, {})
    step_round2 = float(po._OPTIMIZER_STATE[99]['state'][0]['step'])
    assert step_round2 > step_round1, "persistent mode must continue from round 1's Adam state"

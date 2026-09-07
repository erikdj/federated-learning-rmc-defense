"""label_flip epoch equalization (GWU-72, director ruling 2026-08-04).

train_label_flip() historically ran a SINGLE natural pass over the trainloader
while train() and every other training path ran ``for _ in range(epochs)`` with
local_epochs=5 — label_flip attackers trained 5x less than everyone else.

Asserts the equalized contract:
  (a) natural path honors ``epochs``: epochs=E takes E * len(loader) optimizer
      steps, identical to train();
  (b) the default (epochs omitted) preserves the legacy single pass for any
      external caller of the old signature, and ``epochs`` is KEYWORD-ONLY
      appended after the legacy parameters so every pre-change positional slot
      is preserved exactly (``train_label_flip(net, loader, 0.001)`` still
      means lr=0.001 — the compatibility constraint);
  (c) the fleet dispatch (FlowerClient scenario branch AND legacy static
      branch) passes local_epochs, exactly as _honest_train() does for train();
  (d) the update-match branch (Stage-F §4) is unaffected: with max_steps set,
      steps == max_steps regardless of epochs.

Per-batch attack semantics (which labels flip, loss, optimizer stepping) are
unchanged — only the epoch count changes.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flowerfl.task import train, train_label_flip

B = 32
N_ROWS = 96          # -> exactly 3 batches per pass
BATCHES_PER_PASS = 3
E = 5                # canonical local_epochs


def _loader(n_rows: int = N_ROWS, n_features: int = 6, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_rows, n_features, generator=g)
    y = torch.randint(0, 2, (n_rows,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=B, shuffle=False)


class _TwoLogit(nn.Module):
    def __init__(self, n_features: int = 6):
        super().__init__()
        self.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# (a) Natural path honors epochs — equalized with train()
# ---------------------------------------------------------------------------

def test_label_flip_natural_path_takes_epochs_times_loader_steps():
    loader = _loader()
    m = {}
    train_label_flip(_TwoLogit(), loader, epochs=E, metrics_out=m)
    assert m["steps_taken"] == E * len(loader) == E * BATCHES_PER_PASS


def test_label_flip_step_count_equals_honest_train_at_same_epochs():
    loader = _loader()
    m_honest, m_flip = {}, {}
    train(_TwoLogit(), loader, epochs=E, metrics_out=m_honest)
    train_label_flip(_TwoLogit(), loader, epochs=E, metrics_out=m_flip)
    assert m_flip["steps_taken"] == m_honest["steps_taken"] == E * BATCHES_PER_PASS


# ---------------------------------------------------------------------------
# (b) Legacy default: caller passing nothing still gets a single pass
# ---------------------------------------------------------------------------

def test_label_flip_default_epochs_preserves_single_pass():
    loader = _loader()
    m = {}
    train_label_flip(_TwoLogit(), loader, metrics_out=m)
    assert m["steps_taken"] == len(loader) == BATCHES_PER_PASS


def test_label_flip_epochs_1_matches_legacy_single_pass_weights():
    # epochs=1 must be byte-identical to the legacy single-pass behavior.
    loader = _loader()
    torch.manual_seed(7)
    net_default = _TwoLogit()
    torch.manual_seed(7)
    net_explicit = _TwoLogit()

    torch.manual_seed(99)
    loss_default = train_label_flip(net_default, loader)
    torch.manual_seed(99)
    loss_explicit = train_label_flip(net_explicit, loader, epochs=1)

    assert loss_default == loss_explicit
    for p_a, p_b in zip(net_default.parameters(), net_explicit.parameters()):
        assert torch.equal(p_a, p_b)


def test_label_flip_third_positional_arg_still_means_lr():
    #  epochs must be keyword-only so the pre-change public
    # calling convention survives — the third positional argument is lr.
    import inspect
    sig = inspect.signature(train_label_flip)
    params = list(sig.parameters.values())
    # Legacy positional order preserved exactly.
    assert [p.name for p in params[:8]] == [
        "net", "trainloader", "lr", "weight_decay",
        "max_steps", "partition_id", "arm", "metrics_out",
    ]
    assert all(p.kind == p.POSITIONAL_OR_KEYWORD for p in params[:8])
    assert sig.parameters["epochs"].kind == inspect.Parameter.KEYWORD_ONLY
    bound = sig.bind(object(), object(), 0.001)
    assert bound.arguments["lr"] == 0.001
    assert "epochs" not in bound.arguments

    # Behavioral: positional lr call is byte-identical to keyword lr call.
    loader = _loader()
    torch.manual_seed(7)
    net_pos = _TwoLogit()
    torch.manual_seed(7)
    net_kw = _TwoLogit()
    torch.manual_seed(99)
    loss_pos = train_label_flip(net_pos, loader, 0.5)
    torch.manual_seed(99)
    loss_kw = train_label_flip(net_kw, loader, lr=0.5)
    assert loss_pos == loss_kw
    for p_a, p_b in zip(net_pos.parameters(), net_kw.parameters()):
        assert torch.equal(p_a, p_b)


# ---------------------------------------------------------------------------
# (c) Fleet dispatch passes local_epochs (scenario branch + legacy static)
# ---------------------------------------------------------------------------

def _client(local_epochs: int, **kwargs):
    from flowerfl.client_app import FlowerClient
    from flowerfl.task import Net
    net = Net(input_shape=6)
    return FlowerClient(
        trainloader=_loader(),
        valloader=_loader(),
        net=net,
        partition_id=0,
        use_brfss=False,
        local_epochs=local_epochs,
        **kwargs,
    )


def test_scenario_label_flip_dispatch_trains_local_epochs_passes():
    client = _client(local_epochs=E)
    params = client.get_parameters({})
    client.fit(params, {"server_round": 1, "attack_type": "label_flip"})
    assert client._step_metrics["steps_taken"] == E * BATCHES_PER_PASS


def test_legacy_static_label_flip_dispatch_trains_local_epochs_passes():
    client = _client(local_epochs=E, is_malicious=True, attack_type="label_flip")
    params = client.get_parameters({})
    client.fit(params, {})
    assert client._step_metrics["steps_taken"] == E * BATCHES_PER_PASS


def test_scenario_label_flip_matches_honest_step_count():
    # The equalization claim itself: same client config, honest vs label_flip
    # rounds now take the identical number of optimizer steps.
    honest = _client(local_epochs=E)
    flip = _client(local_epochs=E)
    p_h = honest.get_parameters({})
    p_f = flip.get_parameters({})
    honest.fit(p_h, {"server_round": 1, "attack_type": ""})
    flip.fit(p_f, {"server_round": 1, "attack_type": "label_flip"})
    assert flip._step_metrics["steps_taken"] == honest._step_metrics["steps_taken"]


# ---------------------------------------------------------------------------
# (d) Update-match branch unaffected: steps == max_steps regardless of epochs
# ---------------------------------------------------------------------------

def test_update_match_branch_ignores_epochs():
    loader = _loader()
    K = 7  # deliberately not a multiple of the pass length
    for epochs in (1, E):
        m = {}
        train_label_flip(_TwoLogit(), loader, epochs=epochs, max_steps=K,
                         partition_id=3, arm="label_flip", metrics_out=m)
        assert m["steps_taken"] == K

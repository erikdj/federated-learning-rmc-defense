"""Pre-launch equivalence gates (DESIGN_STAGE_F §11) as tests.

Gate A — matched-cap off equivalence: legacy 5-epoch off (max_steps=None) and
matched-cap off (max_steps=ceil(n/B)*E) must be NUMERICALLY IDENTICAL when n is a
multiple of the batch size (the cap is then exactly the legacy step count) — same
final params, reported loss, persistent-Adam optimizer state, and torch RNG state.
This proves the Stage-F off arm is the incumbent, so cross-arm contrasts are clean.

Gate B — instrumentation macro-equivalence: the per-class-instrumented evaluator
(test_detailed, ) reproduces the macro accuracy/precision/recall/F1 of the
pre-branch macro evaluator (test) bit-for-bit, and old (pre-suffix) eval lines
still replay unchanged with the per-class fields ABSENT (never fabricated).

Also includes the byte-unchanged spot check: with the new flag at its default
(max_steps=None) each train fn reproduces the classic epochs loop exactly.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from flowerfl.task import (
    train, train_label_flip, train_brfss, train_brfss_label_flip,
)
# Alias so pytest does not collect these library functions as test cases.
from flowerfl.task import test as macro_eval
from flowerfl.task import test_detailed as detailed_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

B = 32
E = 5
NF = 6


class _TwoLogit(nn.Module):
    def __init__(self, n_features: int = NF):
        super().__init__()
        self.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        return self.fc(x)


def _cic_loader(n_rows: int, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_rows, NF, generator=g)
    y = torch.randint(0, 2, (n_rows,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)


def _opt_state_equal(a: dict, b: dict) -> bool:
    sa, sb = a["state"], b["state"]
    if sa.keys() != sb.keys():
        return False
    for k in sa:
        for field, va in sa[k].items():
            vb = sb[k][field]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


# ---------------------------------------------------------------------------
# Gate A — matched-cap off equivalence (n a multiple of B)
# ---------------------------------------------------------------------------

def _run_off_arm(loader, max_steps, *, model_seed=0, shuffle_seed=123):
    torch.manual_seed(model_seed)
    net = _TwoLogit()
    opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=3e-3)
    torch.manual_seed(shuffle_seed)
    loss = train(net, loader, epochs=E, optimizer=opt, max_steps=max_steps)
    return net, opt, loss, torch.get_rng_state()


def test_gate_a_matched_cap_off_is_numerically_identical():
    n = 4 * B  # 128 — a multiple of B, so ceil(n/B)*E == legacy step count
    loader = _cic_loader(n)
    K = math.ceil(n / B) * E
    assert K == 4 * E

    net_legacy, opt_legacy, loss_legacy, rng_legacy = _run_off_arm(loader, None)
    net_matched, opt_matched, loss_matched, rng_matched = _run_off_arm(loader, K)

    # Final params identical.
    for p_l, p_m in zip(net_legacy.parameters(), net_matched.parameters()):
        assert torch.equal(p_l, p_m)
    # Reported loss identical (exact float).
    assert loss_legacy == loss_matched
    # Persistent-Adam optimizer state identical.
    assert _opt_state_equal(opt_legacy.state_dict(), opt_matched.state_dict())
    # torch RNG state identical (same number of shuffle draws consumed).
    assert torch.equal(rng_legacy, rng_matched)


def test_gate_a_holds_even_when_n_not_multiple_of_b():
    # The steps-driven loop reaches K by completing E full passes for ANY n
    # (ceil(n/B) steps/pass), so the identity is actually stronger than §11's
    # multiple-of-B scoping — assert it here too.
    n = 150  # not a multiple of 32
    loader = _cic_loader(n, seed=1)
    K = math.ceil(n / B) * E
    net_legacy, opt_legacy, loss_legacy, rng_legacy = _run_off_arm(loader, None)
    net_matched, opt_matched, loss_matched, rng_matched = _run_off_arm(loader, K)
    for p_l, p_m in zip(net_legacy.parameters(), net_matched.parameters()):
        assert torch.equal(p_l, p_m)
    assert loss_legacy == loss_matched
    assert _opt_state_equal(opt_legacy.state_dict(), opt_matched.state_dict())
    assert torch.equal(rng_legacy, rng_matched)


# ---------------------------------------------------------------------------
# Gate B — instrumented eval macro-equivalence + old-log replay
# ---------------------------------------------------------------------------

def test_gate_b_macro_metrics_bit_identical():
    torch.manual_seed(7)
    net = _TwoLogit()
    # Eval loaders are NOT shuffled (load_data builds val/test without shuffle),
    # so the two evaluators sum batches in identical order -> bit-identical loss.
    g = torch.Generator().manual_seed(3)
    X = torch.randn(96, NF, generator=g)
    y = torch.randint(0, 2, (96,), generator=g)
    loader = DataLoader(TensorDataset(X, y), batch_size=B)
    loss, acc, prec, rec, f1 = macro_eval(net, loader)
    d = detailed_eval(net, loader)
    assert d["loss"] == loss
    assert d["accuracy"] == acc
    assert d["precision"] == prec
    assert d["recall"] == rec
    assert d["f1"] == f1


def test_gate_b_old_log_replays_with_per_class_absent():
    from run_phase4_flower import parse_eval_trajectory

    old_line = "[ScenarioStrategy] Round 3 eval: F1=0.812 Acc=0.844 Loss=0.331 Prec=0.805 Rec=0.822"
    stage_f_line = (
        "[ScenarioStrategy] Round 3 eval: F1=0.812 Acc=0.844 Loss=0.331 Prec=0.805 Rec=0.822 "
        "AttP=0.777 AttR=0.690 AttF1=0.731 BenP=0.833 BenR=0.900 BenF1=0.865"
    )

    old = parse_eval_trajectory(old_line)
    assert len(old) == 1
    entry = old[0]
    # macro fields replay unchanged...
    assert entry["f1"] == 0.812 and entry["accuracy"] == 0.844 and entry["loss"] == 0.331
    assert entry["precision"] == 0.805 and entry["recall"] == 0.822
    #...and the per-class fields are ABSENT, never fabricated.
    for k in ("attack_recall", "attack_precision", "attack_f1",
              "benign_recall", "benign_precision", "benign_f1"):
        assert k not in entry

    sf = parse_eval_trajectory(stage_f_line)[0]
    assert sf["attack_recall"] == 0.690 and sf["attack_precision"] == 0.777
    assert sf["benign_f1"] == 0.865


# ---------------------------------------------------------------------------
# Byte-unchanged spot check: default (max_steps=None) == classic epochs loop
# ---------------------------------------------------------------------------

def _classic_cic(net, loader, epochs, lr, wd):
    """Reference re-implementation of the pre-Stage-F train epochs loop."""
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    net.train()
    total, nb = 0.0, 0
    for _ in range(epochs):
        for features, labels in loader:
            opt.zero_grad()
            out = net(features)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
    return total / max(nb, 1)


def test_byte_unchanged_default_path_matches_classic_loop():
    loader = _cic_loader(100, seed=5)
    torch.manual_seed(11)
    net_a = _TwoLogit()
    torch.manual_seed(11)
    net_b = _TwoLogit()

    torch.manual_seed(99)
    loss_new = train(net_a, loader, epochs=3, lr=0.01, weight_decay=3e-3)  # max_steps default None
    torch.manual_seed(99)
    loss_ref = _classic_cic(net_b, loader, epochs=3, lr=0.01, wd=3e-3)

    assert loss_new == loss_ref
    for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(p_a, p_b)


@pytest.mark.parametrize("fn,model,n", [
    (train, _TwoLogit, 100),
    (train_label_flip, _TwoLogit, 100),
])
def test_default_path_is_deterministic(fn, model, n):
    loader = _cic_loader(n, seed=2)
    torch.manual_seed(4)
    net1 = model()
    torch.manual_seed(4)
    net2 = model()
    torch.manual_seed(21)
    l1 = fn(net1, loader)
    torch.manual_seed(21)
    l2 = fn(net2, loader)
    assert l1 == l2

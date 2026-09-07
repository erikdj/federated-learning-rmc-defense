"""Per-class eval metrics (Stage-F).

The server-side fixed-eval trajectory historically stored only macro
precision/recall/F1. The macro fields are kept verbatim (append-only schema);
per-class benign(0)/attack(1) fields are ADDED so an attack-class recall/
precision claim is auditable from the trajectory rather than mislabelled from
macro. These tests pin exact per-class values from a known confusion matrix and
assert the macro contract is unchanged.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Alias on import: bare names `test`/`test_detailed` would otherwise be
# collected by pytest as test cases (they match test*).
from flowerfl.task import (
    _binary_per_class_prf,
    test as run_test,
    test_detailed as run_test_detailed,
)


# Known confusion matrix (benign=0, attack=1):
#   attack (1): TP=3, FN=1  -> 4 attack rows
#   benign (0): TN=4, FP=2  -> 6 benign rows
# attack  precision=3/5=0.60  recall=3/4=0.75      f1=0.6667
# benign  precision=4/5=0.80  recall=4/6=0.6667    f1=0.7273
_LABELS = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
_PREDS = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]


def test_per_class_exact_values_from_confusion_matrix():
    d = _binary_per_class_prf(_LABELS, _PREDS)
    assert d["attack_precision"] == pytest.approx(0.60)
    assert d["attack_recall"] == pytest.approx(0.75)
    assert d["attack_f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert d["benign_precision"] == pytest.approx(0.80)
    assert d["benign_recall"] == pytest.approx(4 / 6)
    assert d["benign_f1"] == pytest.approx(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6))


def test_per_class_keys_are_exactly_the_six_expected():
    assert set(_binary_per_class_prf(_LABELS, _PREDS)) == {
        "attack_precision", "attack_recall", "attack_f1",
        "benign_precision", "benign_recall", "benign_f1",
    }


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _tiny_loader(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(24, 4, generator=g)
    y = torch.randint(0, 2, (24,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=8)


def test_test_detailed_is_macro_superset():
    net, loader = _TinyNet(), _tiny_loader()
    d = run_test_detailed(net, loader)
    # macro keys present and unchanged in meaning
    for k in ("loss", "accuracy", "precision", "recall", "f1"):
        assert k in d
    # per-class keys appended
    for k in ("attack_precision", "attack_recall", "attack_f1",
              "benign_precision", "benign_recall", "benign_f1"):
        assert k in d


def test_test_tuple_contract_unchanged():
    """test() must still return the 5-tuple its existing callers unpack."""
    net, loader = _TinyNet(), _tiny_loader()
    out = run_test(net, loader)
    assert len(out) == 5
    loss, acc, prec, rec, f1 = out  # must not raise
    d = run_test_detailed(net, loader)
    assert (loss, acc, prec, rec, f1) == pytest.approx(
        (d["loss"], d["accuracy"], d["precision"], d["recall"], d["f1"])
    )

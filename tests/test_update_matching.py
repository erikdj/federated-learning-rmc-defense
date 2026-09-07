"""Steps-driven update matching (DESIGN_STAGE_F §4).

Asserts the load-bearing invariant of Stage F's dosage isolation: with a fixed
per-client ``max_steps`` cap, EVERY training path — honest, label_flip, and the
gaussian/norm-matched/ALIE attacks that route through the honest train fn — takes
the IDENTICAL number of optimizer steps, regardless of how resampling changed the
loader length. Covers the generic cycling loop, the worked 160-step example, the
loud empty-loader RuntimeError, and step-equality across all four train fns.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flowerfl.task import (
    train,
    train_label_flip,
    train_brfss,
    train_brfss_label_flip,
)
from flowerfl.update_matching import run_matched_steps

B = 32
E = 5


def _budget(n_orig: int) -> int:
    return math.ceil(n_orig / B) * E


def _loader(n_rows: int, n_features: int = 6, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_rows, n_features, generator=g)
    y = torch.randint(0, 2, (n_rows,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)


class _TwoLogit(nn.Module):
    def __init__(self, n_features: int = 6):
        super().__init__()
        self.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        return self.fc(x)


class _OneLogit(nn.Module):
    def __init__(self, n_features: int = 6):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Generic cycling loop + worked example
# ---------------------------------------------------------------------------

def test_run_matched_steps_worked_example_1000_to_280():
    # n_orig=1000, B=32, E=5 -> K=160. A resampled loader of 280 rows holds
    # ceil(280/32)=9 batches/pass; reaching 160 cycles 17 full passes (153) + 7.
    K = _budget(1000)
    assert K == 160
    loader = _loader(280)
    assert len(loader) == 9
    calls = {"n": 0}

    def step_fn(_batch):
        calls["n"] += 1
        return 0.0

    total_loss, steps = run_matched_steps(loader, K, step_fn=step_fn)
    assert steps == 160
    assert calls["n"] == 160
    assert 160 == 17 * 9 + 7  # 17 full passes + 7 batches of the 18th


def test_run_matched_steps_empty_loader_raises():
    empty = DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long)),
                       batch_size=B)
    with pytest.raises(RuntimeError, match="empty post-resampling loader.*client 7.*arm random_under"):
        run_matched_steps(empty, 160, step_fn=lambda b: 0.0, partition_id=7, arm="random_under")


def test_run_matched_steps_zero_cap_is_noop_even_when_empty():
    empty = DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long)),
                       batch_size=B)
    total_loss, steps = run_matched_steps(empty, 0, step_fn=lambda b: 0.0)
    assert (total_loss, steps) == (0.0, 0)


# ---------------------------------------------------------------------------
# Step-equality across every S3/S4 attack type + all four train fns
# ---------------------------------------------------------------------------
# The five S3/S4 arms map onto two CE train fns: honest / gaussian_noise /
# norm_matched_noise / ALIE all route through train(); label_flip routes through
# train_label_flip(). BRFSS mirrors this with train_brfss / train_brfss_label_flip.

@pytest.mark.parametrize("n_orig,n_resampled", [
    (1000, 280),   # under-sampled: loader shorter than n_orig, must cycle
    (200, 200),    # off/matched: n multiple of B
    (200, 640),    # over-sampled: loader longer than n_orig, stops mid-pass
    (150, 150),    # n NOT a multiple of B (ceil budget)
])
def test_all_five_arms_identical_step_counts_cic(n_orig, n_resampled):
    K = _budget(n_orig)
    loader = _loader(n_resampled)
    counts = {}

    for arm, fn in (("honest", train), ("label_flip", train_label_flip)):
        m = {}
        net = _TwoLogit()
        fn(net, loader, max_steps=K, partition_id=3, arm=arm, metrics_out=m)
        counts[arm] = m["steps_taken"]

    # gaussian / norm-matched / ALIE all use the honest train fn -> same count.
    assert counts["honest"] == counts["label_flip"] == K


@pytest.mark.parametrize("n_orig,n_resampled", [(1000, 280), (200, 200), (200, 640)])
def test_all_five_arms_identical_step_counts_brfss(n_orig, n_resampled):
    K = _budget(n_orig)
    loader = _loader(n_resampled)

    m_h, m_lf = {}, {}
    train_brfss(_OneLogit(), loader, max_steps=K, partition_id=3, arm="honest", metrics_out=m_h)
    train_brfss_label_flip(_OneLogit(), loader, max_steps=K, partition_id=3,
                           arm="label_flip", metrics_out=m_lf)
    assert m_h["steps_taken"] == m_lf["steps_taken"] == K


@pytest.mark.parametrize("fn,model", [
    (train, _TwoLogit), (train_label_flip, _TwoLogit),
    (train_brfss, _OneLogit), (train_brfss_label_flip, _OneLogit),
])
def test_each_train_fn_raises_on_empty_loader_with_cap(fn, model):
    empty = DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long)),
                       batch_size=B)
    with pytest.raises(RuntimeError, match="empty post-resampling loader"):
        fn(model(), empty, max_steps=160, partition_id=5, arm="smote@0.5")

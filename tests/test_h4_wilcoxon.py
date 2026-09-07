"""Exact paired two-sided Wilcoxon signed-rank — hand-computed small cases
plus ranking-convention parity with the frozen one-sided implementation
(reproduction/protocol/h2prime/revalidate_v115.py).
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from scripts.h4_scoring_lib import exact_wilcoxon_two_sided

_ROOT = Path(__file__).resolve().parent.parent


def _load_frozen_onesided():
    path = (_ROOT / "reproduction" / "protocol" / "h2prime"
            / "revalidate_v115.py")
    src = path.read_text()
    # Extract just the function — importing the module would execute its
    # heavyweight top-level (results loading). The function is self-contained
    # apart from `combinations`.
    start = src.index("def exact_wilcoxon_onesided")
    end = src.index("\ndef ", start + 1)
    namespace: dict = {}
    exec("from itertools import combinations\n" + src[start:end], namespace)
    return namespace["exact_wilcoxon_onesided"]


# ---------------------------------------------------------------------------
# hand-computed cases
# ---------------------------------------------------------------------------

def test_all_positive_three_diffs():
    # ranks 1,2,3; W+ = 6; P(W+>=6) = 1/8; two-sided = 2/8
    out = exact_wilcoxon_two_sided([0.01, 0.02, 0.03])
    assert out["w_plus"] == 6.0
    assert out["p_ge"] == pytest.approx(1 / 8)
    assert out["p_two_sided"] == pytest.approx(2 / 8)


def test_mixed_signs_hand_case():
    # diffs 1,-2,3 -> ranks 1,2,3; W+ = 1+3 = 4.
    # subset sums of {1,2,3}: 0,1,2,3,3,4,5,6 -> P(>=4)=3/8, P(<=4)=6/8
    # two-sided = 2*min(3/8, 6/8) = 6/8
    out = exact_wilcoxon_two_sided([1.0, -2.0, 3.0])
    assert out["w_plus"] == 4.0
    assert out["p_ge"] == pytest.approx(3 / 8)
    assert out["p_le"] == pytest.approx(6 / 8)
    assert out["p_two_sided"] == pytest.approx(6 / 8)


def test_tied_magnitudes_get_mid_ranks():
    # diffs 1,1,-1 -> |d| all tied -> every rank = 2; W+ = 4.
    # subset sums of {2,2,2}: 0,2,2,2,4,4,4,6 -> P(>=4)=4/8, P(<=4)=7/8
    # two-sided = min(1, 2*4/8) = 1.0
    out = exact_wilcoxon_two_sided([1.0, 1.0, -1.0])
    assert out["w_plus"] == 4.0
    assert out["p_ge"] == pytest.approx(4 / 8)
    assert out["p_two_sided"] == 1.0


def test_zero_diffs_dropped_before_ranking():
    with_zero = exact_wilcoxon_two_sided([0.0, 1.0, 2.0])
    without = exact_wilcoxon_two_sided([1.0, 2.0])
    assert with_zero["n_zero_dropped"] == 1
    assert with_zero["n_effective"] == 2
    assert with_zero["w_plus"] == without["w_plus"]
    assert with_zero["p_two_sided"] == without["p_two_sided"]


def test_all_zero_diffs_degenerate():
    out = exact_wilcoxon_two_sided([0.0, 0.0, 0.0])
    assert out["n_effective"] == 0
    assert out["w_plus"] == 0.0
    assert out["p_two_sided"] == 1.0


def test_empty_diffs_degenerate():
    out = exact_wilcoxon_two_sided([])
    assert out["n_effective"] == 0
    assert out["p_two_sided"] == 1.0


def test_sign_symmetry():
    diffs = [0.3, -0.1, 0.25, 0.07, -0.4, 0.02]
    a = exact_wilcoxon_two_sided(diffs)
    b = exact_wilcoxon_two_sided([-d for d in diffs])
    assert a["p_two_sided"] == pytest.approx(b["p_two_sided"])


def test_p_is_capped_at_one_and_positive():
    for diffs in ([1.0], [1.0, -1.0], [0.5, -0.5, 0.25, -0.25]):
        out = exact_wilcoxon_two_sided(diffs)
        assert 0.0 < out["p_two_sided"] <= 1.0


def test_ten_identical_positive_diffs():
    # The CONFIRMED-ceiling fixture shape: all |d| tied -> mid-rank 5.5 each,
    # W+ = 55 (the maximum); only the full subset reaches 55 -> P(>=)=1/1024.
    out = exact_wilcoxon_two_sided([0.2] * 10)
    assert out["w_plus"] == pytest.approx(55.0)
    assert out["p_ge"] == pytest.approx(1 / 1024)
    assert out["p_two_sided"] == pytest.approx(2 / 1024)


def test_ten_distinct_positive_diffs():
    out = exact_wilcoxon_two_sided([0.01 * k for k in range(1, 11)])
    assert out["w_plus"] == 55.0
    assert out["p_two_sided"] == pytest.approx(2 / 1024)


def test_five_up_five_down_hand_case():
    # five diffs +0.3 (ranks 6..10 -> mid 8), five -0.001 (ranks 1..5 -> mid 3)
    # W+ = 40; count(>=40) = 112/1024 (k=5 eights: 32; k=4 eights and >=3
    # threes: 5*16=80); two-sided = 2*112/1024
    out = exact_wilcoxon_two_sided([0.3] * 5 + [-0.001] * 5)
    assert out["w_plus"] == pytest.approx(40.0)
    assert out["p_ge"] == pytest.approx(112 / 1024)
    assert out["p_two_sided"] == pytest.approx(224 / 1024)


# ---------------------------------------------------------------------------
# parity with the frozen one-sided implementation
# ---------------------------------------------------------------------------

def test_upper_tail_matches_frozen_onesided_on_random_vectors():
    frozen = _load_frozen_onesided()
    rng = random.Random(1234)
    for _ in range(12):
        n = rng.randint(1, 9)
        diffs = [round(rng.uniform(-1, 1), 2) for _ in range(n)]
        w_frozen, p_frozen = frozen(diffs)
        ours = exact_wilcoxon_two_sided(diffs)
        if ours["n_effective"] == 0:
            assert (w_frozen, p_frozen) == (0.0, 1.0)
            continue
        assert ours["w_plus"] == pytest.approx(w_frozen)
        assert ours["p_ge"] == pytest.approx(p_frozen)


def test_upper_tail_matches_frozen_onesided_with_ties_and_zeros():
    frozen = _load_frozen_onesided()
    for diffs in ([0.1, 0.1, -0.1, 0.2, 0.0],
                  [0.05, -0.05, 0.05, -0.05],
                  [1.0, 2.0, 2.0, -2.0, 3.0, 0.0, 0.0]):
        w_frozen, p_frozen = frozen(diffs)
        ours = exact_wilcoxon_two_sided(diffs)
        assert ours["w_plus"] == pytest.approx(w_frozen)
        assert ours["p_ge"] == pytest.approx(p_frozen)

"""Semantic attack-class (label 1) rebalancing target policy (Stage-F §6).

Stage F re-defines the resampling target as a SEMANTIC attack-fraction ``f`` on
label 1 (attack) with the benign class (label 0) held fixed by the over-samplers
and shrunk by the under-sampler — replacing the label-agnostic min/max ratio the
incumbent path still uses. The policy is OPT-IN (``attack_target_policy=True``);
the legacy min/max semantics are untouched when it is off, so the pre-existing
resampler tests keep asserting the incumbent behaviour.

These are the FIVE required test classes from §6 (attack-minority, attack-
majority, equal, single-class, already-above-target), each asserting applied-vs-
skip and EXACT after-counts, plus the ``ceil`` fraction-≥f guarantee and its
``round`` counterexample.
"""
import numpy as np
import pytest

from flowerfl.smote_resampler import (
    resample_training_split,
    semantic_attack_target_count,
    SKIP_SINGLE_CLASS,
    SKIP_MINORITY_STARVED,
    SKIP_UNDERSAMPLE_DEGENERATE,
    SKIP_ATTACK_AT_OR_ABOVE_TARGET,
)

OVER = ("smote", "random_over")
_F_BALANCED = 0.50
_F_GLOBAL = 0.47


def _partition(n_benign: int, n_attack: int, n_features: int = 8, seed: int = 0):
    """A binary partition with exactly ``n_benign`` label-0 and ``n_attack``
    label-1 rows. Attack rows form a tight cluster so SMOTE's interpolation hull
    is well-defined."""
    rng = np.random.default_rng(seed)
    X_ben = rng.normal(0.0, 1.0, (n_benign, n_features)).astype(np.float32)
    X_att = rng.normal(5.0, 0.5, (n_attack, n_features)).astype(np.float32)
    X = np.vstack([X_ben, X_att]).astype(np.float32)
    y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)]).astype(np.int64)
    return X, y


def _counts(y):
    return int((y == 0).sum()), int((y == 1).sum())


def _run(variant, X, y, f):
    return resample_training_split(
        X, y, variant=variant, target=f, seed=13, attack_target_policy=True
    )


# ---------------------------------------------------------------------------
# ceil target arithmetic (§6 target + proof)
# ---------------------------------------------------------------------------

def test_semantic_target_count_balanced_grows_attack_to_benign():
    assert semantic_attack_target_count(0.50, 90) == 90


def test_semantic_target_count_global_uses_ceil():
    # f=0.47, n_benign=90 -> ceil(0.886792 * 90) = ceil(79.81) = 80.
    assert semantic_attack_target_count(0.47, 90) == 80


def test_ceil_guarantees_fraction_at_least_f_where_round_fails():
    # §6 counterexample: f=.47, n_benign=5. round(4.434)=4 -> 4/9=.444 < .47;
    # ceil=5 -> 5/10=.50 >= .47.
    f, n_b = 0.47, 5
    n_a_ceil = semantic_attack_target_count(f, n_b)
    assert n_a_ceil == 5
    assert n_a_ceil / (n_a_ceil + n_b) >= f
    n_a_round = round(f / (1.0 - f) * n_b)
    assert n_a_round == 4
    assert n_a_round / (n_a_round + n_b) < f


def test_ceil_fraction_at_least_f_exhaustive():
    # ceil guarantee holds over many (f, n_benign) draws (design §verification).
    rng = np.random.default_rng(20260731)
    for _ in range(2000):
        f = float(rng.uniform(0.05, 0.95))
        n_b = int(rng.integers(1, 500))
        n_a = semantic_attack_target_count(f, n_b)
        assert n_a / (n_a + n_b) >= f - 1e-12


# ---------------------------------------------------------------------------
# Case 1 — attack-minority (90 benign / 10 attack): everything applies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", OVER)
def test_case1_attack_minority_oversampler_balanced(variant):
    X, y = _partition(90, 10)
    _, yr, skipped = _run(variant, X, y, _F_BALANCED)
    assert skipped is None
    assert _counts(yr) == (90, 90)


@pytest.mark.parametrize("variant", OVER)
def test_case1_attack_minority_oversampler_global(variant):
    X, y = _partition(90, 10)
    _, yr, skipped = _run(variant, X, y, _F_GLOBAL)
    assert skipped is None
    # benign untouched at 90; attack grown to ceil(0.886792*90)=80.
    assert _counts(yr) == (90, 80)


def test_case1_attack_minority_undersampler_shrinks_benign():
    X, y = _partition(90, 10)
    _, yr, skipped = _run("random_under", X, y, _F_BALANCED)
    assert skipped is None
    # benign shrunk to n_attack=10; attack untouched.
    assert _counts(yr) == (10, 10)


# ---------------------------------------------------------------------------
# Case 2 — attack-majority (10 benign / 90 attack): NEVER grow benign
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", OVER)
@pytest.mark.parametrize("f", [_F_BALANCED, _F_GLOBAL])
def test_case2_attack_majority_oversampler_skips(variant, f):
    X, y = _partition(10, 90)
    _, yr, skipped = _run(variant, X, y, f)
    assert skipped == SKIP_ATTACK_AT_OR_ABOVE_TARGET
    assert _counts(yr) == (10, 90)  # unchanged — benign never grown


def test_case2_attack_majority_undersampler_degenerate():
    X, y = _partition(10, 90)
    _, yr, skipped = _run("random_under", X, y, _F_BALANCED)
    assert skipped == SKIP_UNDERSAMPLE_DEGENERATE
    assert _counts(yr) == (10, 90)


# ---------------------------------------------------------------------------
# Case 3 — equal (50/50): over-samplers at/above target, under degenerate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", OVER)
@pytest.mark.parametrize("f", [_F_BALANCED, _F_GLOBAL])
def test_case3_equal_oversampler_skips(variant, f):
    X, y = _partition(50, 50)
    _, yr, skipped = _run(variant, X, y, f)
    assert skipped == SKIP_ATTACK_AT_OR_ABOVE_TARGET
    assert _counts(yr) == (50, 50)


def test_case3_equal_undersampler_degenerate():
    X, y = _partition(50, 50)
    _, yr, skipped = _run("random_under", X, y, _F_BALANCED)
    assert skipped == SKIP_UNDERSAMPLE_DEGENERATE
    assert _counts(yr) == (50, 50)


# ---------------------------------------------------------------------------
# Case 4 — single class: SKIP_SINGLE_CLASS for all variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ("smote", "random_over", "random_under"))
@pytest.mark.parametrize("only_label", (0, 1))
def test_case4_single_class_skips(variant, only_label):
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, (40, 6)).astype(np.float32)
    y = np.full(40, only_label, dtype=np.int64)
    _, yr, skipped = _run(variant, X, y, _F_BALANCED)
    assert skipped == SKIP_SINGLE_CLASS
    assert np.array_equal(yr, y)


# ---------------------------------------------------------------------------
# Case 5 — already-above-target (f=0.47, 40 benign / 60 attack -> frac 0.60)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", OVER)
def test_case5_already_above_target_oversampler_skips(variant):
    X, y = _partition(40, 60)
    _, yr, skipped = _run(variant, X, y, _F_GLOBAL)
    assert skipped == SKIP_ATTACK_AT_OR_ABOVE_TARGET
    assert _counts(yr) == (40, 60)


# ---------------------------------------------------------------------------
# SMOTE minority-starvation under the semantic policy (attack present but < 2)
# ---------------------------------------------------------------------------

def test_smote_attack_starved_skips_but_random_over_grows():
    # 50 benign / 1 attack: SMOTE needs >=2 attack rows for k-NN -> starve;
    # random_over can duplicate a single attack row -> applies.
    X, y = _partition(50, 1)
    _, ys, skipped_s = _run("smote", X, y, _F_BALANCED)
    assert skipped_s == SKIP_MINORITY_STARVED
    assert _counts(ys) == (50, 1)
    _, yo, skipped_o = _run("random_over", X, y, _F_BALANCED)
    assert skipped_o is None
    assert _counts(yo) == (50, 50)


# ---------------------------------------------------------------------------
# The opt-in nature: legacy min/max path is untouched when policy is off
# ---------------------------------------------------------------------------

def test_legacy_min_max_path_unchanged_when_policy_off():
    # target=0.5 under the LEGACY (min/max ratio) path grows attack to ~0.5*benign
    # (minority/majority ratio), NOT to a 0.5 attack fraction.
    X, y = _partition(200, 40)
    _, yr, skipped = resample_training_split(
        X, y, variant="smote", target=0.5, seed=7  # attack_target_policy defaults False
    )
    assert skipped is None
    n_benign, n_attack = _counts(yr)
    assert n_attack == pytest.approx(0.5 * n_benign, abs=1)

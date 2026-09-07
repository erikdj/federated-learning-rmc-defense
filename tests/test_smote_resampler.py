"""Unit tests for flowerfl/smote_resampler.py ( SMOTE study).

The resampler is a flag-gated, per-client training-split oversampler. These
tests exercise the pure library in isolation (no FL run, no network, synthetic
data only):

  - balance achieved (target ratio),
  - determinism (same seed -> byte-identical; different seed -> different rows),
  - synthetic rows are interpolations within the minority feature hull,
  - loud validation of unknown variant / invalid target.
"""
import numpy as np
import pytest

from flowerfl.smote_resampler import (
    SUPPORTED_SMOTE_VARIANTS,
    normalize_smote_target,
    validate_smote_variant,
    resample_training_split,
    effective_k_neighbors,
    coerce_smote_enabled,
)


def _skewed_partition(n_majority=200, n_minority=40, n_features=8, seed=0):
    """A 2-class partition skewed toward the majority (benign) class.

    Minority points are drawn from a tight cluster so the SMOTE interpolation
    hull is well-defined for the bounds check.
    """
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(0.0, 1.0, (n_majority, n_features)).astype(np.float32)
    X_min = rng.normal(5.0, 0.5, (n_minority, n_features)).astype(np.float32)
    X = np.vstack([X_maj, X_min]).astype(np.float32)
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)]).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Variant / target vocabulary
# ---------------------------------------------------------------------------

def test_supported_variants_include_minimum_set():
    assert "smote" in SUPPORTED_SMOTE_VARIANTS
    assert "random_over" in SUPPORTED_SMOTE_VARIANTS
    # image v8 change 2 (advisor-directed 2026-07-26 item 2)
    assert "random_under" in SUPPORTED_SMOTE_VARIANTS


def test_validate_variant_accepts_supported():
    for v in SUPPORTED_SMOTE_VARIANTS:
        assert validate_smote_variant(v) == v


@pytest.mark.parametrize("bad", ["adasyn", "SMOTE ", "", "over", None, 3])
def test_validate_variant_rejects_unknown_loudly(bad):
    with pytest.raises(ValueError, match="smote_variant"):
        validate_smote_variant(bad)


def test_normalize_target_balanced_passthrough():
    assert normalize_smote_target("balanced") == "balanced"


@pytest.mark.parametrize("value,expected", [("0.5", 0.5), (0.75, 0.75), ("1.0", 1.0), (1, 1.0)])
def test_normalize_target_float(value, expected):
    assert normalize_smote_target(value) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["banana", 0, 0.0, -0.3, 1.5, "2", None])
def test_normalize_target_rejects_invalid_loudly(bad):
    with pytest.raises(ValueError, match="smote_target"):
        normalize_smote_target(bad)


# ---------------------------------------------------------------------------
# Balance achieved
# ---------------------------------------------------------------------------

def test_balanced_target_gives_5050():
    X, y = _skewed_partition()
    Xr, yr, skipped = resample_training_split(X, y, variant="smote", target="balanced", seed=42)
    assert skipped is None
    n0 = int((yr == 0).sum())
    n1 = int((yr == 1).sum())
    assert n0 == n1, f"expected 50/50 after balanced SMOTE, got {n0}/{n1}"
    # majority class is untouched (over-sampler only grows the minority)
    assert n0 == int((y == 0).sum())


def test_float_target_gives_requested_minority_ratio():
    X, y = _skewed_partition(n_majority=200, n_minority=20)
    Xr, yr, skipped = resample_training_split(X, y, variant="smote", target=0.5, seed=7)
    assert skipped is None
    n0 = int((yr == 0).sum())
    n1 = int((yr == 1).sum())
    # minority/majority ~= 0.5
    assert n1 == pytest.approx(0.5 * n0, abs=1)


def test_random_over_balances_by_duplication():
    X, y = _skewed_partition()
    Xr, yr, skipped = resample_training_split(X, y, variant="random_over", target="balanced", seed=1)
    assert skipped is None
    assert int((yr == 0).sum()) == int((yr == 1).sum())
    # RandomOverSampler duplicates existing rows: every resampled minority row
    # is an exact copy of some original minority row (no interpolation).
    orig_min_rows = {tuple(r) for r in X[y == 1]}
    for r in Xr[yr == 1]:
        assert tuple(r) in orig_min_rows


# ---------------------------------------------------------------------------
# random_under (image v8 change 2): minority PRESERVED, majority SHRUNK to hit
# the same ratio the over-samplers grow the minority to. No synthesis.
# ---------------------------------------------------------------------------

def test_random_under_balanced_downsamples_majority():
    """balanced -> majority downsampled to the minority count (50/50), minority
    untouched, total rows SHRINK, and every kept row is an original row."""
    X, y = _skewed_partition(n_majority=200, n_minority=40)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target="balanced", seed=4)
    assert skipped is None
    n0 = int((yr == 0).sum())
    n1 = int((yr == 1).sum())
    assert n0 == n1 == 40, f"expected 40/40 after balanced under-sampling, got {n0}/{n1}"
    assert len(yr) < len(y), "under-sampling must reduce total rows"
    # minority preserved exactly; kept rows are originals (no synthesis)
    orig_rows = {tuple(r) for r in X}
    for r in Xr:
        assert tuple(r) in orig_rows


def test_random_under_float_target_shrinks_majority():
    """float target r sets minority/majority = r AFTER resampling by shrinking
    the majority (mirror of the over-sampler ratio, opposite mechanism)."""
    X, y = _skewed_partition(n_majority=200, n_minority=40)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target=0.5, seed=7)
    assert skipped is None
    n0 = int((yr == 0).sum())
    n1 = int((yr == 1).sum())
    assert n1 == 40, "minority untouched under-sampling"
    assert n1 == pytest.approx(0.5 * n0, abs=1)  # majority shrunk to ~80


def test_random_under_determinism():
    X, y = _skewed_partition()
    Xa, ya, _ = resample_training_split(X, y, variant="random_under", target="balanced", seed=11)
    Xb, yb, _ = resample_training_split(X, y, variant="random_under", target="balanced", seed=11)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)


def test_random_under_single_class_skips():
    """Only one class -> single_class skip, untouched (shared policy)."""
    X, y = _single_class_partition()
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target="balanced", seed=5)
    assert skipped == "single_class"
    assert np.array_equal(Xr, X) and np.array_equal(yr, y)


def test_random_under_minority_of_one_proceeds():
    """Under-sampling cannot 'starve' like SMOTE-k: minority of 1 is valid — the
    majority is shrunk to 1 and the split proceeds (no skip)."""
    X, y = _skewed_partition(n_majority=50, n_minority=1)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target="balanced", seed=9)
    assert skipped is None
    assert int((yr == 0).sum()) == 1 and int((yr == 1).sum()) == 1


def test_random_under_infeasible_float_skips_degenerate():
    """A float ratio that would require GROWING the majority (impossible for an
    under-sampler) skips with the distinct 'undersample_degenerate' reason,
    untouched — never a crash."""
    # data ratio = 40/200 = 0.2; asking for 0.1 needs majority = 400 > 200.
    X, y = _skewed_partition(n_majority=200, n_minority=40)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target=0.1, seed=3)
    assert skipped == "undersample_degenerate"
    assert np.array_equal(Xr, X) and np.array_equal(yr, y)


def test_random_under_rounding_hidden_noop_skips():
    """n_min=2, n_maj=3, target 0.6 rounds the majority
    target back to 3 == current majority — a no-op that MUST NOT report applied.
    The ratio is compared DIRECTLY (0.6 <= observed 2/3), so it SKIPs."""
    X = np.vstack([np.zeros((3, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32)])
    y = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target=0.6, seed=1)
    assert skipped == "undersample_degenerate"
    assert np.array_equal(Xr, X) and np.array_equal(yr, y)


def test_random_under_target_equal_observed_ratio_skips():
    """Boundary: target == observed minority/majority ratio is a no-op, not
    'applied'. n_maj=4, n_min=2 -> observed 0.5; target 0.5 -> SKIP."""
    X, y = _skewed_partition(n_majority=4, n_minority=2)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target=0.5, seed=1)
    assert skipped == "undersample_degenerate"
    assert np.array_equal(Xr, X) and np.array_equal(yr, y)


def test_random_under_feasible_target_above_observed_applies():
    """A target strictly above the observed ratio is feasible and applies (real
    majority reduction)."""
    # observed = 40/200 = 0.2; target 0.4 > 0.2 -> majority shrinks to 100.
    X, y = _skewed_partition(n_majority=200, n_minority=40)
    Xr, yr, skipped = resample_training_split(X, y, variant="random_under", target=0.4, seed=1)
    assert skipped is None
    n0 = int((yr == 0).sum())
    n1 = int((yr == 1).sum())
    assert n1 == 40
    assert n1 == pytest.approx(0.4 * n0, abs=1)


# ---------------------------------------------------------------------------
# Synthetic rows are interpolations within the minority hull
# ---------------------------------------------------------------------------

def test_smote_synthetic_rows_within_minority_bounds():
    X, y = _skewed_partition()
    Xr, yr, skipped = resample_training_split(X, y, variant="smote", target="balanced", seed=42)
    assert skipped is None
    orig_min = X[y == 1]
    lo = orig_min.min(axis=0)
    hi = orig_min.max(axis=0)
    res_min = Xr[yr == 1]
    eps = 1e-4
    assert (res_min >= lo - eps).all(), "SMOTE row below minority feature min"
    assert (res_min <= hi + eps).all(), "SMOTE row above minority feature max"
    # and there really are new rows (interpolation happened)
    assert len(res_min) > len(orig_min)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_byte_identical():
    X, y = _skewed_partition()
    Xa, ya, _ = resample_training_split(X, y, variant="smote", target="balanced", seed=123)
    Xb, yb, _ = resample_training_split(X, y, variant="smote", target="balanced", seed=123)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)


def test_different_seed_different_synthetic_rows():
    X, y = _skewed_partition()
    Xa, _, _ = resample_training_split(X, y, variant="smote", target="balanced", seed=1)
    Xb, _, _ = resample_training_split(X, y, variant="smote", target="balanced", seed=2)
    assert not np.array_equal(Xa, Xb), "distinct seeds must yield distinct synthetic rows"


# ---------------------------------------------------------------------------
# Starvation policy: skip-with-provenance, never crash (DESIGN.md §6c)
# ---------------------------------------------------------------------------

def _single_class_partition(n=100, n_features=8, label=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, n_features)).astype(np.float32)
    y = np.full(n, label, dtype=np.int64)
    return X, y


@pytest.mark.parametrize("variant", ["smote", "random_over"])
def test_single_class_partition_skips_untouched(variant):
    """Only one class present -> skip BEFORE building the sampler, no crash,
    output returned untouched, reason 'single_class' (both variants)."""
    X, y = _single_class_partition()
    Xr, yr, skipped = resample_training_split(X, y, variant=variant, target="balanced", seed=5)
    assert skipped == "single_class"
    assert np.array_equal(Xr, X)
    assert np.array_equal(yr, y)


@pytest.mark.parametrize("variant", ["smote", "random_over"])
def test_minority_of_one_starved_skip(variant):
    """A minority class of exactly 1 sample is un-oversamplable under the pinned
    policy -> skip with 'minority_starved', untouched, no crash (both variants)."""
    X, y = _skewed_partition(n_majority=50, n_minority=1)
    Xr, yr, skipped = resample_training_split(X, y, variant=variant, target="balanced", seed=9)
    assert skipped == "minority_starved"
    assert np.array_equal(Xr, X)
    assert np.array_equal(yr, y)


def test_minority_of_two_proceeds_with_k1():
    """minority_count == 2 -> k_neighbors clamps to 1 and SMOTE proceeds (no
    skip); the split is balanced."""
    X, y = _skewed_partition(n_majority=50, n_minority=2)
    Xr, yr, skipped = resample_training_split(X, y, variant="smote", target="balanced", seed=3)
    assert skipped is None
    assert int((yr == 0).sum()) == int((yr == 1).sum())
    assert len(yr) > len(y)


# ---------------------------------------------------------------------------
# effective_k_neighbors — single source of truth for the k clamp, reused by
# load_data to report the applied k in the per-client provenance record.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_minority,expected_k", [(40, 5), (6, 5), (5, 4), (3, 2), (2, 1)])
def test_effective_k_neighbors_clamp(n_minority, expected_k):
    _, y = _skewed_partition(n_majority=80, n_minority=n_minority)
    assert effective_k_neighbors(y) == expected_k


# ---------------------------------------------------------------------------
# coerce_smote_enabled — strict bool gate. A typo must
# NEVER silently disable SMOTE; it raises so the whole arm doesn't run incumbent.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("false", False),
    ("True", True), ("FALSE", False), (" true ", True),
])
def test_coerce_smote_enabled_accepts_bool_and_canonical_tokens(value, expected):
    assert coerce_smote_enabled(value) is expected


@pytest.mark.parametrize("bad", ["ture", "yes", "yes  ", "1", "on", "", 1, 0, 1.5, None, ["true"]])
def test_coerce_smote_enabled_rejects_garbage_loudly(bad):
    with pytest.raises(ValueError, match="smote_enabled"):
        coerce_smote_enabled(bad)

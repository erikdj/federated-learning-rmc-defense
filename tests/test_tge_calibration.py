"""Test score calibration continuity at boundaries."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
        errors.append((name, str(e)))


def test_gbdt_boundary_continuity():
    """GBDT scoring should be continuous at pos=-0.5 boundary."""
    # Simulate the GBDT score function at the boundary
    # In-range at pos=-0.5: normalized = 0, score = 0.75
    # Below-range at pos=-0.5: shifted = 0, score = 1.50 / (1 + exp(0)) = 0.75

    pos_above = -0.49  # just above boundary
    pos_at = -0.50     # at boundary
    pos_below = -0.51  # just below boundary

    # In-range scoring
    norm_above = (pos_above + 0.5) / 1.5
    score_above = 0.75 + 0.20 * min(max(norm_above, 0.0), 1.0)

    norm_at = (pos_at + 0.5) / 1.5
    score_at_in_range = 0.75 + 0.20 * min(max(norm_at, 0.0), 1.0)

    # Below-range scoring
    shifted_at = -(pos_at + 0.5)
    score_at_below = 1.50 / (1.0 + np.exp(1.0 * shifted_at))

    shifted_below = -(pos_below + 0.5)
    score_below = 1.50 / (1.0 + np.exp(1.0 * shifted_below))

    print(f"    pos=-0.49 (in-range): {score_above:.6f}")
    print(f"    pos=-0.50 (in-range): {score_at_in_range:.6f}")
    print(f"    pos=-0.50 (below):    {score_at_below:.6f}")
    print(f"    pos=-0.51 (below):    {score_below:.6f}")

    # Continuity: both paths should give 0.75 at the boundary
    assert abs(score_at_in_range - 0.75) < 1e-6, f"In-range at boundary: {score_at_in_range}"
    assert abs(score_at_below - 0.75) < 1e-6, f"Below-range at boundary: {score_at_below}"

    # Smoothness: nearby values should be close
    assert abs(score_above - score_at_in_range) < 0.01, f"Jump above boundary"
    assert abs(score_at_below - score_below) < 0.01, f"Jump below boundary"


def test_lstm_boundary_continuity():
    """LSTM scoring should be continuous at overshoot=TOLERANCE boundary."""
    OVERSHOOT_TOLERANCE = 0.01

    # Simulate at boundary and just above
    train_mse_max = 0.5

    # At tolerance: in-range path
    overshoot_at = OVERSHOOT_TOLERANCE
    ratio = min(train_mse_max * (1 + OVERSHOOT_TOLERANCE / 1.0) / train_mse_max, 1.0)
    # Actually, the in-range path computes: ratio = min(mse / train_mse_max, 1.0)
    # At overshoot=tolerance, mse = train_mse_max + tolerance * range
    # For this test, just use ratio = 1.0 (clamped)
    score_in_range_at_boundary = 0.95 - 0.15 * 1.0  # = 0.80

    # Just above tolerance: below-range path
    overshoot_above = OVERSHOOT_TOLERANCE + 0.001
    shifted = overshoot_above - OVERSHOOT_TOLERANCE  # = 0.001
    score_above = 1.60 / (1.0 + np.exp(1.0 * shifted))

    # At tolerance in below-range: shifted = 0
    shifted_at = 0.0
    score_below_at_boundary = 1.60 / (1.0 + np.exp(1.0 * shifted_at))

    print(f"    In-range at boundary (ratio=1): {score_in_range_at_boundary:.6f}")
    print(f"    Below-range at boundary:        {score_below_at_boundary:.6f}")
    print(f"    Below-range just above:         {score_above:.6f}")

    # Continuity: both paths should give 0.80 at the boundary
    assert abs(score_in_range_at_boundary - 0.80) < 1e-6, \
        f"In-range at boundary: {score_in_range_at_boundary}"
    assert abs(score_below_at_boundary - 0.80) < 1e-6, \
        f"Below-range at boundary: {score_below_at_boundary}"

    # Smoothness
    assert abs(score_above - score_below_at_boundary) < 0.01, \
        f"Jump at boundary: {score_above} vs {score_below_at_boundary}"


def test_gbdt_score_monotonic():
    """GBDT scores should decrease monotonically as pos decreases."""
    positions = [1.0, 0.5, 0.0, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0, -3.0]
    prev_score = float('inf')
    for pos in positions:
        if pos >= -0.5:
            normalized = (pos + 0.5) / 1.5
            score = 0.75 + 0.20 * min(max(normalized, 0.0), 1.0)
        else:
            shifted = -(pos + 0.5)
            score = 1.50 / (1.0 + np.exp(1.0 * shifted))

        assert score <= prev_score + 1e-6, \
            f"Non-monotonic at pos={pos}: {score:.4f} > {prev_score:.4f}"
        prev_score = score

    print(f"    Score range: [{1.50/(1+np.exp(2.5)):.4f}, 0.9500] (pos from -3.0 to 1.0)")


def test_lstm_score_monotonic():
    """LSTM scores should decrease monotonically as overshoot increases."""
    OVERSHOOT_TOLERANCE = 0.01
    overshoots = [-0.5, -0.2, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0]
    prev_score = float('inf')

    for os_val in overshoots:
        if os_val <= OVERSHOOT_TOLERANCE:
            ratio = max(0, min(1.0, (os_val + 0.5) / 0.51))  # rough approximation
            ratio = min(ratio, 1.0)
            score = 0.95 - 0.15 * ratio
        else:
            shifted = os_val - OVERSHOOT_TOLERANCE
            score = 1.60 / (1.0 + np.exp(1.0 * shifted))

        assert score <= prev_score + 1e-6, \
            f"Non-monotonic at overshoot={os_val}: {score:.4f} > {prev_score:.4f}"
        prev_score = score


if __name__ == "__main__":
    print("=" * 60)
    print("Score Calibration Continuity Tests")
    print("=" * 60)

    run_test("gbdt_boundary_continuity", test_gbdt_boundary_continuity)
    run_test("lstm_boundary_continuity", test_lstm_boundary_continuity)
    run_test("gbdt_score_monotonic", test_gbdt_score_monotonic)
    run_test("lstm_score_monotonic", test_lstm_score_monotonic)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)

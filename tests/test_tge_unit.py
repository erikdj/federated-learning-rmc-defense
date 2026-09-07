"""Unit tests for TGEnsembleModel correctness (no pytest required)."""

import numpy as np
import sys
import os
import traceback

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmc.tg_ensemble import (
    TGEnsembleModel, extract_geometric_features, compute_layer_boundaries,
    GBDTColdStartExpert, LSTMTemporalExpert, TenureGatedDecisionRule,
    geometric_fallback_score,
)

# Net architecture: 45->64->32->2
PARAM_SHAPES = [(45, 64), (64,), (64, 32), (32,), (32, 2), (2,)]
NUM_LAYERS = len(PARAM_SHAPES)
EXPECTED_FEATURES = 3 + NUM_LAYERS + 3  # = 12


def _total_params():
    total = 0
    for shape in PARAM_SHAPES:
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


TOTAL_PARAMS = _total_params()


def _make_updates(num_honest=9, num_anomalous=1, anomaly_scale=5.0, seed=42):
    """Generate fake client updates: honest + anomalous."""
    np.random.seed(seed)
    base = np.random.randn(TOTAL_PARAMS).astype(np.float32) * 0.1
    honest = [base + np.random.randn(TOTAL_PARAMS).astype(np.float32) * 0.01
              for _ in range(num_honest)]
    anomalous = [base + np.random.randn(TOTAL_PARAMS).astype(np.float32) * anomaly_scale
                 for _ in range(num_anomalous)]
    return honest + anomalous


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
        traceback.print_exc()
        failed += 1
        errors.append((name, str(e)))


# ===== TESTS =====

def test_layer_boundaries():
    boundaries = compute_layer_boundaries(PARAM_SHAPES)
    expected = [2880, 2944, 4992, 5024, 5088, 5090]
    assert boundaries == expected, f"Expected {expected}, got {boundaries}"
    assert TOTAL_PARAMS == 5090, f"Expected 5090, got {TOTAL_PARAMS}"


def test_feature_dimension():
    assert EXPECTED_FEATURES == 12, f"Expected 12, got {EXPECTED_FEATURES}"
    updates = _make_updates()
    boundaries = compute_layer_boundaries(PARAM_SHAPES)
    features = extract_geometric_features(updates[0], updates, boundaries)
    assert features.shape == (12,), f"Expected (12,), got {features.shape}"
    assert features.dtype == np.float32


def test_gbdt_warmup():
    gbdt = GBDTColdStartExpert(warmup_rounds=3)
    assert not gbdt.is_ready
    score = gbdt.score(np.zeros(12))
    assert score == 0.85, f"Expected 0.85, got {score}"


def test_gbdt_fit_and_score():
    gbdt = GBDTColdStartExpert(warmup_rounds=3)
    updates = _make_updates()
    boundaries = compute_layer_boundaries(PARAM_SHAPES)

    for u in updates[:9]:
        feat = extract_geometric_features(u, updates, boundaries)
        gbdt.accumulate(feat)

    gbdt.fit(server_round=3)
    assert gbdt.is_ready, "GBDT should be fitted"

    feat_honest = extract_geometric_features(updates[0], updates, boundaries)
    score_honest = gbdt.score(feat_honest)
    assert 0.0 <= score_honest <= 1.0, f"Score out of range: {score_honest}"

    feat_anomalous = extract_geometric_features(updates[9], updates, boundaries)
    score_anomalous = gbdt.score(feat_anomalous)
    assert 0.0 <= score_anomalous <= 1.0, f"Score out of range: {score_anomalous}"

    print(f"    GBDT honest: {score_honest:.4f}, anomalous: {score_anomalous:.4f}")


def test_lstm_warmup():
    lstm = LSTMTemporalExpert(input_dim=12, warmup_rounds=5)
    assert not lstm.is_ready
    score = lstm.score("client_0")
    assert score == 0.85, f"Expected 0.85, got {score}"


def test_lstm_fit_and_score():
    lstm = LSTMTemporalExpert(input_dim=12, warmup_rounds=5, min_sequences=3)
    updates = _make_updates()
    boundaries = compute_layer_boundaries(PARAM_SHAPES)

    for round_num in range(1, 8):
        for i, u in enumerate(updates[:5]):
            feat = extract_geometric_features(u, updates, boundaries)
            lstm.record_features(f"client_{i}", feat)

    for i in range(5):
        lstm.accumulate_training_sequence(f"client_{i}")

    lstm.fit(server_round=7)
    assert lstm.is_ready, "LSTM should be fitted"

    score = lstm.score("client_0")
    assert 0.0 <= score <= 1.0, f"LSTM score out of range: {score}"
    print(f"    LSTM client_0 score: {score:.4f}")


def test_tenure_gate_pure_gbdt():
    gate = TenureGatedDecisionRule(min_tenure=2, ramp_rounds=5)
    assert gate.compute_score(0.8, 0.3, tenure=1) == 0.8


def test_tenure_gate_pure_lstm():
    gate = TenureGatedDecisionRule(min_tenure=2, ramp_rounds=5)
    assert gate.compute_score(0.8, 0.3, tenure=5) == 0.3


def test_tenure_gate_blend():
    gate = TenureGatedDecisionRule(min_tenure=2, ramp_rounds=5)
    score = gate.compute_score(0.8, 0.3, tenure=3)
    expected = 2/3 * 0.8 + 1/3 * 0.3
    assert abs(score - expected) < 1e-6, f"Expected {expected:.4f}, got {score:.4f}"


def test_warmup_all_accepted():
    """During warmup, all clients should score 1.0."""
    model = TGEnsembleModel(num_features=12, warmup_rounds=3)
    model.discover_layer_boundaries(PARAM_SHAPES)
    updates = _make_updates()

    for server_round in range(1, 4):
        for i, update in enumerate(updates):
            cid = f"client_{i}"
            features = model.extract_features(update, updates)
            score, details = model.score_client(cid, features, server_round)
            assert score == 1.0, f"Warmup score != 1.0: {score} (round={server_round}, cid={cid})"
            assert details["phase"] == "warmup", f"Phase should be warmup, got {details['phase']}"

        for i in range(len(updates)):
            model.record_scored(f"client_{i}", server_round)
            model.record_accepted(f"client_{i}", server_round)
        model.on_round_end(server_round)


def test_post_warmup_scores_in_range():
    """Post-warmup, all scores should be in [0, 1]."""
    model = TGEnsembleModel(num_features=12, warmup_rounds=3)
    model.discover_layer_boundaries(PARAM_SHAPES)
    updates = _make_updates()

    for server_round in range(1, 4):
        for i, update in enumerate(updates):
            features = model.extract_features(update, updates)
            model.score_client(f"client_{i}", features, server_round)
        for i in range(len(updates)):
            model.record_scored(f"client_{i}", server_round)
            model.record_accepted(f"client_{i}", server_round)
        model.on_round_end(server_round)

    for server_round in range(4, 10):
        for i, update in enumerate(updates):
            features = model.extract_features(update, updates)
            score, details = model.score_client(f"client_{i}", features, server_round)
            assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score} (round={server_round}, client={i})"
        for i in range(len(updates)):
            model.record_scored(f"client_{i}", server_round)
            model.record_accepted(f"client_{i}", server_round)
        model.on_round_end(server_round)


def test_anomalous_detection():
    """After sufficient rounds, anomalous client should score lower than honest avg."""
    model = TGEnsembleModel(num_features=12, warmup_rounds=3)
    model.discover_layer_boundaries(PARAM_SHAPES)
    updates = _make_updates(anomaly_scale=10.0)

    scores = {}
    for server_round in range(1, 12):
        for i, update in enumerate(updates):
            cid = f"client_{i}"
            features = model.extract_features(update, updates)
            score, details = model.score_client(cid, features, server_round)
            scores[cid] = score
        for i in range(len(updates)):
            cid = f"client_{i}"
            model.record_scored(cid, server_round)
            if scores[cid] >= 0.7:
                model.record_accepted(cid, server_round)
        model.on_round_end(server_round)

    honest_scores = [scores[f"client_{i}"] for i in range(9)]
    anomalous_score = scores["client_9"]
    avg_honest = np.mean(honest_scores)

    print(f"    Honest avg: {avg_honest:.4f}, Anomalous: {anomalous_score:.4f}")
    print(f"    Honest: {[f'{s:.3f}' for s in honest_scores]}")

    assert anomalous_score < avg_honest, (
        f"Anomalous ({anomalous_score:.4f}) should be < honest avg ({avg_honest:.4f})"
    )


def test_no_false_rejections_baseline():
    """In a baseline (no attacks), all honest clients should score >= 0.7."""
    model = TGEnsembleModel(num_features=12, warmup_rounds=3)
    model.discover_layer_boundaries(PARAM_SHAPES)
    updates = _make_updates(num_honest=10, num_anomalous=0)

    scores = {}
    for server_round in range(1, 12):
        for i, update in enumerate(updates):
            cid = f"client_{i}"
            features = model.extract_features(update, updates)
            score, details = model.score_client(cid, features, server_round)
            scores[cid] = score
        for i in range(len(updates)):
            cid = f"client_{i}"
            model.record_scored(cid, server_round)
            model.record_accepted(cid, server_round)
        model.on_round_end(server_round)

    below_threshold = [(cid, s) for cid, s in scores.items() if s < 0.7]
    if below_threshold:
        print(f"    Clients below 0.7: {below_threshold}")
    assert len(below_threshold) == 0, f"False rejections: {below_threshold}"


def test_geometric_fallback_perfect():
    features = np.array([0.0, 1.0, 0.0] + [1.0]*6 + [0.1, 0.05, 0.3], dtype=np.float32)
    score = geometric_fallback_score(features)
    assert abs(score - 1.0) < 0.01, f"Expected ~1.0, got {score}"


def test_geometric_fallback_anomalous():
    features = np.array([3.0, -0.5, 2.0] + [5.0]*6 + [10.0, 5.0, 20.0], dtype=np.float32)
    score = geometric_fallback_score(features)
    assert score < 0.5, f"Expected < 0.5, got {score}"


# ===== RUN ALL TESTS =====

if __name__ == "__main__":
    print("=" * 60)
    print("TGEnsembleModel Unit Tests")
    print("=" * 60)

    print("\n--- Layer Boundaries ---")
    run_test("layer_boundaries", test_layer_boundaries)

    print("\n--- Feature Extraction ---")
    run_test("feature_dimension", test_feature_dimension)

    print("\n--- GBDT Cold-Start Expert ---")
    run_test("gbdt_warmup", test_gbdt_warmup)
    run_test("gbdt_fit_and_score", test_gbdt_fit_and_score)

    print("\n--- LSTM Temporal Expert ---")
    run_test("lstm_warmup", test_lstm_warmup)
    run_test("lstm_fit_and_score", test_lstm_fit_and_score)

    print("\n--- Tenure Gate ---")
    run_test("tenure_gate_pure_gbdt", test_tenure_gate_pure_gbdt)
    run_test("tenure_gate_pure_lstm", test_tenure_gate_pure_lstm)
    run_test("tenure_gate_blend", test_tenure_gate_blend)

    print("\n--- TGEnsembleModel Integration ---")
    run_test("warmup_all_accepted", test_warmup_all_accepted)
    run_test("post_warmup_scores_in_range", test_post_warmup_scores_in_range)
    run_test("anomalous_detection", test_anomalous_detection)
    run_test("no_false_rejections_baseline", test_no_false_rejections_baseline)

    print("\n--- Geometric Fallback ---")
    run_test("geometric_fallback_perfect", test_geometric_fallback_perfect)
    run_test("geometric_fallback_anomalous", test_geometric_fallback_anomalous)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)

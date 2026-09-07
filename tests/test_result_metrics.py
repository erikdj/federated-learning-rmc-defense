"""Tests for flowerfl.result_metrics helper functions."""
import pytest


def test_compute_convergence_metrics_basic():
    from flowerfl.result_metrics import compute_convergence_metrics
    # Synthetic 10-round trajectory; F1 climbs from 0.2 to 0.9
    trajectory = [
        {"round": i, "f1": 0.2 + 0.07 * i, "accuracy": 0.3, "loss": 1.0}
        for i in range(10)
    ]
    # final_f1 = 0.83, 90% threshold = 0.747, reached at round where 0.2 + 0.07*r >= 0.747 → r >= 7.8 → round 8
    m = compute_convergence_metrics(trajectory)
    assert "rounds_to_reach_90pct_final_F1" in m
    assert m["rounds_to_reach_90pct_final_F1"] == 8
    assert m["rounds_to_reach_50pct_final_F1"] >= 3
    assert 0.0 <= m["stability_last10"] <= 1.0
    assert m["monotonicity_score"] == 1.0  # strictly increasing


def test_compute_convergence_metrics_non_monotonic():
    from flowerfl.result_metrics import compute_convergence_metrics
    # Synthetic trajectory with backslides
    trajectory = [{"round": i, "f1": f1, "accuracy": 0.3, "loss": 1.0}
                  for i, f1 in enumerate([0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8, 0.8, 0.9])]
    m = compute_convergence_metrics(trajectory)
    # 7 of 9 round-pairs are non-decreasing → monotonicity 7/9 ≈ 0.778
    assert 0.7 < m["monotonicity_score"] < 0.9


def test_compute_confounder_control_no_honest_events_returns_none():
    from flowerfl.result_metrics import compute_confounder_control_metrics
    scenario = {"schedule": [], "honest_events": []}
    trajectory = [{"round": i, "f1": 0.5} for i in range(50)]
    scoring_log = []
    m = compute_confounder_control_metrics(scenario, trajectory, scoring_log)
    # No honest events → no confounder-control to report → returns None or {}
    assert m is None or m.get("n_honest_reconnect_events", 0) == 0


def test_compute_confounder_control_with_synthetic_events():
    from flowerfl.result_metrics import compute_confounder_control_metrics
    # Synthetic scenario with 2 honest reconnect events at rounds 15, 30
    scenario = {
        "honest_events": [
            {"offline_id": "honest_5_offline1", "rejoin_round": 15, "victim_original_id": "client_5"},
            {"offline_id": "honest_2_offline2", "rejoin_round": 30, "victim_original_id": "client_2"},
        ],
        "schedule": [],
        "clients": {
            "client_11": {"physical_id": 11},
            "client_11_new1": {"physical_id": 20},
            "honest_5_offline1": {"physical_id": 22},
            "honest_2_offline2": {"physical_id": 23},
        },
    }
    trajectory = [{"round": i, "f1": 0.5} for i in range(50)]
    # Scoring log: list of (round, logical_cid, cs_trust) tuples emitted by ColdStartPlugin
    scoring_log = [
        # Honest reconnects flagged at rejoin rounds (trust < 0.5 → flagged)
        (15, "honest_5_offline1", 0.3),  # honest flagged: contributes to FPR
        (30, "honest_2_offline2", 0.7),  # honest unflagged: no FPR contribution
        # Adversary reconnects at rounds 13 and 25
        (13, "client_11_new1", 0.2),     # adversary flagged: contributes to TPR
        (25, "client_12_new2", 0.4),     # adversary flagged: contributes to TPR
    ]
    m = compute_confounder_control_metrics(scenario, trajectory, scoring_log)
    assert m is not None
    assert m["n_honest_reconnect_events"] == 2
    assert m["n_adversary_reconnect_events"] == 2  # extracted from scenario or scoring_log; spec the impl
    assert 0.0 <= m["fpr_honest_reconnect"] <= 1.0
    assert 0.0 <= m["tpr_adversary_reconnect"] <= 1.0


def test_discrimination_ratio_undefined_when_fpr_zero_but_tpr_positive():
    """When TPR>0 and FPR=0 (perfect discrimination), return None (JSON-safe)."""
    import json
    from flowerfl.result_metrics import compute_confounder_control_metrics
    scenario = {
        "honest_events": [
            {"offline_id": "honest_5_offline1", "rejoin_round": 15, "victim_original_id": "client_5"},
        ],
        "schedule": [],
        "clients": {
            "client_11": {"physical_id": 11},
            "client_11_new1": {"physical_id": 20},
            "honest_5_offline1": {"physical_id": 22},
        },
    }
    scoring_log = [
        (15, "honest_5_offline1", 0.9),   # honest unflagged → FPR contribution = 0
        (13, "client_11_new1", 0.2),      # adversary flagged → TPR contribution > 0
    ]
    m = compute_confounder_control_metrics(scenario, [], scoring_log)
    assert m is not None
    assert m["fpr_honest_reconnect"] == 0.0
    assert m["tpr_adversary_reconnect"] > 0.0
    assert m["discrimination_ratio"] is None, \
        "perfect-discrimination case (TPR>0, FPR=0) must be None (JSON-safe), not float('inf')"
    # And the result must round-trip through JSON cleanly
    serialized = json.dumps(m, default=str)
    parsed = json.loads(serialized)
    assert parsed["discrimination_ratio"] is None

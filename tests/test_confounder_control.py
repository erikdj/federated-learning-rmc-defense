"""Synthetic test for confounder_control TPR/FPR wiring.

Resolves the open question from results/20260520/SWEEP_ANALYSIS.md:
is the v3 sweep's discrimination_ratio=0.0 a legitimate finding (TGE filters on
content, not identity) or a wiring bug?

Key findings from reading compute_confounder_control_metrics():
- honest_events entries must use field name "offline_id" (not "client")
- adversary clients are detected by naming: must start with "client_" AND contain "_new"
  (e.g. "client_11_new1", "client_12_new2")
- n_adversary_reconnect_events = count of unique adversary offline IDs (from clients dict
  + scoring_log), NOT event count
- n_honest_reconnect_events = len(honest_events)
- TPR/FPR are computed over scoring_log entries that match the ID sets
- discrimination_ratio is None when fpr=0 but tpr>0 (perfect discrimination)
- discrimination_ratio is 0.0 when both tpr=0 and fpr=0 (no signal)
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mock_scenario():
    """3 honest + 2 adversary identities.

    honest_5_offline1 reconnects at round 5.
    client_11_new1 reconnects at rounds 3 and 7.
    client_12_new2 reconnects at round 7.

    Adversary IDs must follow naming convention client_K_newN (contains "_new")
    so they are picked up by compute_confounder_control_metrics.
    n_adversary_reconnect_events counts unique adversary offline IDs = 2.
    n_honest_reconnect_events = len(honest_events) = 1.
    """
    return {
        "name": "synthetic_confounder_test",
        "num_rounds": 10,
        "clients": {
            # Honest clients — no "_new" in name, so not adversary-detected
            "client_5": {"physical_id": 5},
            "client_6": {"physical_id": 6},
            "client_7": {"physical_id": 7},
            # Adversary reconnect identities — naming must have "_new"
            "client_11_new1": {"physical_id": 20},
            "client_12_new2": {"physical_id": 21},
        },
        "honest_events": [
            {
                "offline_id": "honest_5_offline1",
                "rejoin_round": 5,
                "victim_original_id": "client_5",
            },
        ],
    }


def test_confounder_control_records_tge_rejections():
    """Mock TGE rejects adversaries (score=0.0), keeps honest (score=1.0).

    Drive a 10-round scenario. Assert confounder_control captures the rejections:
    - adversary IDs appear in scoring_log with trust=0.0 (< FLAG_THRESHOLD=0.5) → flagged
    - honest ID appears with trust=1.0 (>= FLAG_THRESHOLD) → not flagged
    - tpr_adversary_reconnect == 1.0 (both adversary IDs flagged every round)
    - fpr_honest_reconnect == 0.0 (honest never flagged)
    - discrimination_ratio is None (tpr>0, fpr=0 → perfect discrimination, undefined)
    """
    from flowerfl.result_metrics import compute_confounder_control_metrics

    scenario = _make_mock_scenario()
    trajectory = [{"round": r, "f1": 0.5, "accuracy": 0.5, "loss": 0.7} for r in range(10)]

    # Scoring log: (round, logical_cid, cs_trust_score)
    # Honest offline ID must match honest_events[*]["offline_id"]
    # Adversary IDs must match clients dict keys that contain "_new"
    scoring_log = []
    for r in range(10):
        scoring_log.append((r, "honest_5_offline1", 1.0))   # honest: not flagged
        scoring_log.append((r, "client_11_new1", 0.0))      # adversary: flagged
        scoring_log.append((r, "client_12_new2", 0.0))      # adversary: flagged

    metrics = compute_confounder_control_metrics(scenario, trajectory, scoring_log)

    assert metrics is not None
    # n_adversary = unique adversary offline IDs from clients dict = 2
    assert metrics["n_adversary_reconnect_events"] == 2, metrics
    # n_honest = len(honest_events) = 1
    assert metrics["n_honest_reconnect_events"] == 1, metrics
    # All adversary log entries flagged → tpr = 1.0
    assert metrics["tpr_adversary_reconnect"] == 1.0, metrics
    # No honest log entries flagged → fpr = 0.0
    assert metrics["fpr_honest_reconnect"] == 0.0, metrics
    # tpr>0 and fpr=0 → perfect discrimination, serialized as None (not inf)
    assert metrics["discrimination_ratio"] is None, metrics


def test_confounder_control_records_fpr_when_honest_flagged():
    """Worst-case mock TGE that rejects EVERYONE: discrimination_ratio = 1.0.

    When both adversary and honest are flagged at rate 1.0:
    - tpr_adversary_reconnect = 1.0
    - fpr_honest_reconnect = 1.0
    - discrimination_ratio = tpr / fpr = 1.0
    """
    from flowerfl.result_metrics import compute_confounder_control_metrics

    scenario = _make_mock_scenario()
    trajectory = [{"round": r, "f1": 0.5, "accuracy": 0.5, "loss": 0.7} for r in range(10)]

    # All clients score 0.0 → everyone flagged
    scoring_log = []
    for r in range(10):
        scoring_log.append((r, "honest_5_offline1", 0.0))   # honest: flagged → FPR=1.0
        scoring_log.append((r, "client_11_new1", 0.0))      # adversary: flagged
        scoring_log.append((r, "client_12_new2", 0.0))      # adversary: flagged

    metrics = compute_confounder_control_metrics(scenario, trajectory, scoring_log)

    assert metrics is not None
    assert metrics["tpr_adversary_reconnect"] == 1.0, metrics
    assert metrics["fpr_honest_reconnect"] == 1.0, metrics
    # fpr > 0 → discrimination_ratio = tpr / fpr = 1.0 / 1.0 = 1.0
    assert metrics["discrimination_ratio"] == 1.0, metrics

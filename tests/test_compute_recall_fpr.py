"""Unit tests for scripts/compute_recall_fpr.py."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_recall_at_target_fpr_perfect_separator():
    """If malicious scores are all 0.1 and honest are all 0.9, recall@FPR=10% = 1.0."""
    from compute_recall_fpr import recall_at_target_fpr

    records = (
        [{"score": 0.1, "malicious": True} for _ in range(20)]
        + [{"score": 0.9, "malicious": False} for _ in range(180)]
    )
    recall, threshold, actual_fpr = recall_at_target_fpr(records, target_fpr=0.10)
    assert recall == 1.0
    assert actual_fpr <= 0.10
    assert 0.1 < threshold < 0.9


def test_recall_at_target_fpr_random_scores():
    """If scores are uniformly random regardless of label, recall@FPR=10% ≈ 0.10."""
    import random
    from compute_recall_fpr import recall_at_target_fpr
    random.seed(42)
    records = (
        [{"score": random.random(), "malicious": True} for _ in range(100)]
        + [{"score": random.random(), "malicious": False} for _ in range(900)]
    )
    recall, threshold, actual_fpr = recall_at_target_fpr(records, target_fpr=0.10)
    assert 0.05 <= recall <= 0.15, f"recall@10%FPR for random scores should be near 0.10; got {recall}"


def test_recall_at_target_fpr_empty_malicious():
    """If there are no malicious records, recall is undefined (return None)."""
    from compute_recall_fpr import recall_at_target_fpr

    records = [{"score": 0.5, "malicious": False} for _ in range(10)]
    recall, threshold, actual_fpr = recall_at_target_fpr(records, target_fpr=0.10)
    assert recall is None

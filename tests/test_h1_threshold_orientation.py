"""H1 malicious-risk thresholds use the upper tail and can be frozen externally."""
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for _path in (str(REPO / "scripts"), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TARGET_FPR = 0.10


def _honest_scores(n=1000, seed=0):
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)


def test_select_threshold_risk_uses_upper_quantile():
    from h1_signal_family_eval import select_threshold_risk

    scores = _honest_scores(seed=7)
    assert select_threshold_risk(scores, TARGET_FPR) == float(
        np.quantile(scores, 1.0 - TARGET_FPR)
    )
    flagged = float(np.mean(scores > select_threshold_risk(scores, TARGET_FPR)))
    assert TARGET_FPR - 2.0 / len(scores) <= flagged <= TARGET_FPR


def test_select_threshold_risk_empty_calibration_flags_nothing():
    from h1_signal_family_eval import select_threshold_risk

    assert select_threshold_risk([], TARGET_FPR) == float("inf")


class _StubClassifier:
    def __init__(self, risk):
        self._risk = np.asarray(risk, dtype=float)

    def predict_proba(self, features):
        risk = self._risk[: features.shape[0]]
        return np.column_stack([1.0 - risk, risk])


def _synthetic_rows(n_clients=4, k=3):
    rows = []
    for client in range(n_clients):
        malicious = client >= n_clients // 2
        for round_number in range(1, k + 1):
            rows.append(
                {
                    "logical_cid": f"client_{client}",
                    "scenario_round": round_number,
                    "malicious_gt": malicious,
                    "update_norm": 1.0 + 0.1 * round_number + (0.5 if malicious else 0.0),
                    "cos_to_median": 0.9 - 0.05 * round_number,
                    "L2_to_median": 0.4 + 0.01 * round_number,
                    "train_loss": 0.7 - 0.02 * round_number,
                    "num_examples": 100,
                    "seed": 42,
                }
            )
    return rows


def test_compute_recall_applies_supplied_threshold_unchanged():
    from h1_signal_family_eval import _compute_recall_at_fpr

    rows = _synthetic_rows()
    classifier = _StubClassifier([0.10] * 6 + [0.90] * 6)

    result = _compute_recall_at_fpr(
        classifier, rows, "S", 3, TARGET_FPR, threshold=0.5
    )

    assert result == {
        "recall_at_fpr": 1.0,
        "n_eval": 12,
        "threshold": 0.5,
        "actual_fpr": 0.0,
    }


def test_supplied_threshold_prevents_heldout_recalibration():
    from h1_signal_family_eval import _compute_recall_at_fpr

    rows = _synthetic_rows()
    classifier = _StubClassifier([0.10] * 6 + [0.90] * 6)

    result = _compute_recall_at_fpr(
        classifier, rows, "S", 3, TARGET_FPR, threshold=0.99
    )

    assert result["threshold"] == 0.99
    assert result["recall_at_fpr"] == 0.0
    assert result["actual_fpr"] == 0.0


def _load_analyzer():
    path = REPO / "reproduction" / "analyze_h1.py"
    spec = importlib.util.spec_from_file_location("public_analyze_h1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyzer_freezes_one_cut_from_pooled_development_cells(tmp_path):
    analyzer = _load_analyzer()
    cells = {}
    for seed in (42, 137):
        path = tmp_path / (
            f"s0_clean_baseline__krum_tge__persistent_optimizer__seed{seed}.jsonl"
        )
        rows = _synthetic_rows()
        for row in rows:
            row["seed"] = seed
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        cells[("s0_clean_baseline", seed)] = path

    classifiers = {
        family: _StubClassifier([0.10, 0.20, 0.30, 0.10, 0.20, 0.30] * 2)
        for family in ("S", "W", "C")
    }
    frozen = analyzer.freeze_dev_thresholds(classifiers, cells, 3, TARGET_FPR)

    expected = float(np.quantile([0.10, 0.20, 0.30] * 4, 0.90))
    for family in ("S", "W", "C"):
        assert frozen[family]["threshold"] == expected
        assert frozen[family]["n_honest_dev"] == 12
        assert frozen[family]["realized_dev_fpr"] <= TARGET_FPR


def test_analyzer_rejects_models_outside_the_published_custody(tmp_path):
    analyzer = _load_analyzer()
    for family in ("S", "W", "C"):
        (tmp_path / f"{family}_k3_final.pkl").write_bytes(b"not a published model")

    with pytest.raises(SystemExit, match="model hash mismatch"):
        analyzer.load_classifiers(tmp_path, 3)

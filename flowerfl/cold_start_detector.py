"""
Phase 3 P3.1/P3.2 — Cold-start detector for early-round malicious client detection.

Architecture (v23 §B6/B7):
    - XGBoost gradient-boosted decision tree
    - Produces calibrated risk score ∈ [0, 1] per (client, round)
    - Active during the first k observed rounds after join or rejoin
    - Three feature families: S (single-round), W (short-window), C (combined)

Signal families (v23 §B5):

    Family S — single-round server-observed signals:
        - update_norm: L2 norm of client's parameter update vector
        - cos_to_median: cosine similarity between client update and round median
        - L2_to_median: L2 distance between client update and round median
        - train_loss: client-reported training loss
        - num_examples: number of training samples used

    Family W — short-window signals (computed over first k observed rounds):
        - norm_change: update_norm[last] - update_norm[first]
        - norm_variance: variance of update_norm across k rounds
        - norm_mean: mean update_norm across k rounds
        - cos_drift: cos_to_median[last] - cos_to_median[first]
        - cos_variance: variance of cos_to_median across k rounds
        - cos_mean: mean cos_to_median across k rounds
        - loss_slope: linear slope of train_loss over k rounds
        - loss_variance: variance of train_loss across k rounds
        - tenure_count: number of observed rounds (≤ k)

    Family C — combined:
        - all S features + all W features

Usage:
    from flowerfl.cold_start_detector import ColdStartDetector, FeatureExtractor

    # Training
    extractor = FeatureExtractor()
    X, y = extractor.build_training_set(signal_log_paths, family="C", k=3)
    detector = ColdStartDetector(family="C", k=3)
    detector.fit(X, y)
    detector.save("models/cold_start/flower_reset/C_seed42.pkl")

    # Inference
    detector = ColdStartDetector.load("models/cold_start/flower_reset/C_seed42.pkl")
    risk = detector.predict_risk(feature_vector)  # float ∈ [0, 1]
"""
from __future__ import annotations

import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Feature families
# ============================================================================

class Family(str, Enum):
    S = "S"   # single-round
    W = "W"   # short-window
    C = "C"   # combined

# Feature name lists (for interpretability / SHAP)
S_FEATURES = [
    "update_norm",
    "cos_to_median",
    "L2_to_median",
    "train_loss",
    "num_examples",
]

W_FEATURES = [
    "norm_change",
    "norm_variance",
    "norm_mean",
    "cos_drift",
    "cos_variance",
    "cos_mean",
    "loss_slope",
    "loss_variance",
    "tenure_count",
]

C_FEATURES = S_FEATURES + W_FEATURES


def feature_names(family: Family) -> List[str]:
    if family == Family.S:
        return list(S_FEATURES)
    elif family == Family.W:
        return list(W_FEATURES)
    else:
        return list(C_FEATURES)


# ============================================================================
# Feature extraction
# ============================================================================

class FeatureExtractor:
    """Extracts S, W, and C feature vectors from signal log rows.

    A signal log row is a dict with keys matching the JSONL schema from
    `flowerfl/signal_logger.py`.
    """

    def extract_s(self, row: Dict[str, Any]) -> np.ndarray:
        """Single-round features from one signal log row."""
        return np.array([
            _safe_float(row.get("update_norm")),
            _safe_float(row.get("cos_to_median")),
            _safe_float(row.get("L2_to_median")),
            _safe_float(row.get("train_loss")),
            _safe_float(row.get("num_examples")),
        ], dtype=np.float32)

    def extract_w(self, history: List[Dict[str, Any]], k: int = 3) -> np.ndarray:
        """Short-window features from the first k observed rounds for a client.

        `history` is a list of signal log rows for the SAME logical client,
        sorted by scenario_round. Only the first min(len(history), k) rows
        are used.
        """
        h = history[:k]
        n = len(h)
        if n == 0:
            return np.zeros(len(W_FEATURES), dtype=np.float32)

        norms = np.array([_safe_float(r.get("update_norm")) for r in h])
        coss = np.array([_safe_float(r.get("cos_to_median")) for r in h])
        losses = np.array([_safe_float(r.get("train_loss")) for r in h])

        norm_change = float(norms[-1] - norms[0]) if n > 1 else 0.0
        norm_var = float(np.var(norms)) if n > 1 else 0.0
        norm_mean = float(np.mean(norms))

        cos_drift = float(coss[-1] - coss[0]) if n > 1 else 0.0
        cos_var = float(np.var(coss)) if n > 1 else 0.0
        cos_mean = float(np.mean(coss))

        # Linear slope of train_loss over k rounds
        if n > 1:
            x = np.arange(n, dtype=np.float32)
            # Least-squares slope: cov(x, y) / var(x)
            loss_slope = float(np.cov(x, losses)[0, 1] / max(np.var(x), 1e-12))
        else:
            loss_slope = 0.0
        loss_var = float(np.var(losses)) if n > 1 else 0.0

        return np.array([
            norm_change,
            norm_var,
            norm_mean,
            cos_drift,
            cos_var,
            cos_mean,
            loss_slope,
            loss_var,
            float(n),  # tenure_count
        ], dtype=np.float32)

    def extract_c(
        self, row: Dict[str, Any], history: List[Dict[str, Any]], k: int = 3
    ) -> np.ndarray:
        """Combined features: S (current round) + W (first k rounds)."""
        s = self.extract_s(row)
        w = self.extract_w(history, k)
        return np.concatenate([s, w])

    def build_training_set(
        self,
        signal_log_paths: List[str],
        family: Family = Family.C,
        k: int = 3,
        cold_start_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) training set from multiple signal log JSONL files.

        Args:
            signal_log_paths: paths to JSONL files from dev runs
            family: which feature family to extract
            k: cold-start window size
            cold_start_only: if True, only include examples from the first k
                observed rounds per client per run. This focuses the detector
                on the cold-start period (v23 §B4).

        Returns:
            X: (n_examples, n_features) feature matrix
            y: (n_examples,) binary labels (1=malicious, 0=honest)
        """
        all_X = []
        all_y = []

        for path in signal_log_paths:
            rows = _load_jsonl(path)
            if not rows:
                continue

            # Group rows by (logical_cid) to build per-client histories
            # Each run (file) is one independent experiment
            client_rows: Dict[str, List[Dict]] = {}
            for r in rows:
                cid = r.get("logical_cid", "")
                client_rows.setdefault(cid, []).append(r)

            # Sort each client's rows by scenario_round
            for cid in client_rows:
                client_rows[cid].sort(key=lambda r: r.get("scenario_round", 0))

            # Extract features for each client-round
            for cid, hist in client_rows.items():
                for i, row in enumerate(hist):
                    obs_round = i + 1  # 1-indexed observed round for this client

                    if cold_start_only and obs_round > k:
                        break  # only keep first k rounds per client

                    if family == Family.S:
                        feat = self.extract_s(row)
                    elif family == Family.W:
                        feat = self.extract_w(hist[:obs_round], k)
                    else:  # Family.C
                        feat = self.extract_c(row, hist[:obs_round], k)

                    label = 1.0 if row.get("malicious_gt", False) else 0.0
                    all_X.append(feat)
                    all_y.append(label)

        if not all_X:
            return np.empty((0, len(feature_names(family)))), np.empty(0)

        return np.stack(all_X), np.array(all_y, dtype=np.float32)


# ============================================================================
# Cold-start detector
# ============================================================================

class ColdStartDetector:
    """XGBoost-backed cold-start malicious client detector.

    Produces a calibrated risk score ∈ [0, 1] for each (client, round).
    Higher score = more likely malicious.
    """

    def __init__(self, family: Family = Family.C, k: int = 3):
        self.family = family
        self.k = k
        self._model = None
        self._calibrator = None  # optional Platt scaling
        self._feature_names = feature_names(family)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        scale_pos_weight: Optional[float] = None,
    ) -> "ColdStartDetector":
        """Train the XGBoost detector on labeled examples.

        Args:
            X: (n, d) feature matrix
            y: (n,) binary labels (1=malicious)
            scale_pos_weight: if None, auto-compute from class balance
        """
        import xgboost as xgb

        if scale_pos_weight is None:
            n_pos = max(y.sum(), 1)
            n_neg = max(len(y) - n_pos, 1)
            scale_pos_weight = float(n_neg / n_pos)

        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=0,
        )
        self._model.fit(X, y)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict malicious risk score ∈ [0, 1].

        Args:
            X: (n, d) or (d,) feature vector(s)

        Returns:
            risk: (n,) or scalar float — probability of malicious
        """
        if self._model is None:
            raise RuntimeError("Detector not fitted; call fit() or load() first")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba = self._model.predict_proba(X)[:, 1]
        return proba

    def feature_importance(self, importance_type: str = "gain") -> List[Tuple[str, float]]:
        """Return feature importance ranking."""
        if self._model is None:
            return []
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        ranked = []
        for i, name in enumerate(self._feature_names):
            key = f"f{i}"
            ranked.append((name, scores.get(key, 0.0)))
        ranked.sort(key=lambda kv: -kv[1])
        return ranked

    def save(self, path: str) -> None:
        """Persist detector to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "family": self.family.value,
                "k": self.k,
                "model": self._model,
                "feature_names": self._feature_names,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ColdStartDetector":
        """Load a persisted detector."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(family=Family(data["family"]), k=data["k"])
        det._model = data["model"]
        det._feature_names = data.get("feature_names", feature_names(det.family))
        return det


# ============================================================================
# Helpers
# ============================================================================

def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

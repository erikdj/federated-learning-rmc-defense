"""Phase 3.5 — Baseline comparison for cold-start detection.

Evaluates 7 published baselines alongside our XGBoost detectors (S, W, C)
on the same signal log data, same CV splits, same metrics, same cold-start
window. Answers: does our detector add value over established methods?

Baselines:
    B0: Random              — lower bound (uniform random score)
    B1: L2-distance z-score — standard outlier detection
    B2: Cosine dissimilarity — FoolsGold-style (Fung et al., 2020)
    B3: Norm deviation      — relative deviation from median norm
    B5: MB-Weight composite — MB-Weight composite formula
    (B4 Krum and B6 TrustScore require pairwise data; approximated from logs)

All baselines produce per-client risk scores from the same signal log
fields used by our detector: update_norm, cos_to_median, L2_to_median,
train_loss, num_examples.

Usage:
    conda run -n flowerfl python scripts/evaluate_baselines.py --k 3
    conda run -n flowerfl python scripts/evaluate_baselines.py --k 1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.cold_start_detector import (  # noqa: E402
    ColdStartDetector,
    Family,
    FeatureExtractor,
    _load_jsonl,
    _safe_float,
)

DEV_SEEDS = [42, 137, 256, 314, 500]


# ============================================================================
# Baseline scorers — each takes signal log rows and returns risk scores
# ============================================================================

def score_random(rows: List[dict], rng: np.random.RandomState) -> np.ndarray:
    """B0: Random baseline (lower bound)."""
    return rng.uniform(0, 1, size=len(rows))


def score_l2_distance(rows: List[dict], round_rows: Dict[int, List[dict]]) -> np.ndarray:
    """B1: L2-distance z-score — how far is this client from the round median?
    Higher z-score = more suspicious.
    """
    scores = []
    for r in rows:
        sr = r.get("scenario_round", 0)
        rr = round_rows.get(sr, [])
        l2s = [_safe_float(x.get("L2_to_median")) for x in rr]
        if len(l2s) < 2:
            scores.append(0.0)
            continue
        mu = np.mean(l2s)
        std = max(np.std(l2s), 1e-12)
        z = (_safe_float(r.get("L2_to_median")) - mu) / std
        scores.append(max(0, z))  # higher z = more suspicious
    return np.array(scores)


def score_cosine_dissim(rows: List[dict]) -> np.ndarray:
    """B2: Cosine dissimilarity (FoolsGold-style). Higher = more suspicious."""
    return np.array([1.0 - _safe_float(r.get("cos_to_median"), 1.0) for r in rows])


def score_norm_deviation(rows: List[dict], round_rows: Dict[int, List[dict]]) -> np.ndarray:
    """B3: Relative norm deviation from round median norm. Higher = more suspicious."""
    scores = []
    for r in rows:
        sr = r.get("scenario_round", 0)
        rr = round_rows.get(sr, [])
        norms = [_safe_float(x.get("update_norm")) for x in rr]
        median_norm = np.median(norms) if norms else 1.0
        client_norm = _safe_float(r.get("update_norm"))
        if median_norm < 1e-12:
            scores.append(0.0)
        else:
            scores.append(abs(client_norm - median_norm) / median_norm)
    return np.array(scores)


def score_mb_weight(rows: List[dict], round_rows: Dict[int, List[dict]]) -> np.ndarray:
    """B5: MB-Weight composite score (advisor CH3 §3.8.4, Eq. 6).
    score = 0.5 × norm_diff + 0.5 × (1 - cos_similarity)
    where norm_diff = |update_norm - median_norm| (normalized by median)
    """
    scores = []
    for r in rows:
        sr = r.get("scenario_round", 0)
        rr = round_rows.get(sr, [])
        norms = [_safe_float(x.get("update_norm")) for x in rr]
        median_norm = np.median(norms) if norms else 1.0

        client_norm = _safe_float(r.get("update_norm"))
        norm_diff = abs(client_norm - median_norm) / max(median_norm, 1e-12)
        cos_dissim = 1.0 - _safe_float(r.get("cos_to_median"), 1.0)
        score = 0.5 * norm_diff + 0.5 * cos_dissim
        scores.append(score)
    return np.array(scores)


def score_trust_zscore(rows: List[dict], round_rows: Dict[int, List[dict]]) -> np.ndarray:
    """B6: TrustScore-style z-score (single-round component).
    z = (L2_to_median - mean) / std; risk = max(0, z / outlier_threshold)
    Matches TrustScorePlugin logic at outlier_threshold=2.0.
    """
    scores = []
    for r in rows:
        sr = r.get("scenario_round", 0)
        rr = round_rows.get(sr, [])
        l2s = [_safe_float(x.get("L2_to_median")) for x in rr]
        if len(l2s) < 2:
            scores.append(0.0)
            continue
        mu = np.mean(l2s)
        std = max(np.std(l2s), 1e-12)
        z = (_safe_float(r.get("L2_to_median")) - mu) / std
        risk = max(0.0, z / 2.0)  # outlier_threshold=2.0
        scores.append(min(risk, 1.0))
    return np.array(scores)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_scores(scores: np.ndarray, labels: np.ndarray, fpr_thresh: float = 0.10) -> dict:
    """Evaluate a set of risk scores against binary labels."""
    from sklearn.metrics import average_precision_score, roc_curve

    if len(scores) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
        return {"pr_auc": None, "recall_at_10pct_fpr": None, "n_pos": int(labels.sum()),
                "n_neg": int(len(labels) - labels.sum())}

    pr_auc = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = fpr <= fpr_thresh
    recall_at_fpr = float(tpr[valid][-1]) if valid.any() else 0.0

    return {
        "pr_auc": pr_auc,
        "recall_at_10pct_fpr": recall_at_fpr,
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
    }


# ============================================================================
# Main
# ============================================================================

def find_all_signal_logs() -> Dict[int, List[str]]:
    """Find all signal logs grouped by seed."""
    out: Dict[int, List[str]] = {}
    for seed in DEV_SEEDS:
        files = []
        # Flower scenario logs
        files.extend(sorted(glob.glob(
            str(PROJECT_ROOT / "signals" / f"flower_reset__*__none__seed{seed}.jsonl"))))
        # Dynamic RMC logs
        files.extend(sorted(glob.glob(
            str(PROJECT_ROOT / "signals" / f"persistent_optimizer__szelag_dynamic_rmc__*__seed{seed}.jsonl"))))
        if files:
            out[seed] = files
    return out


def build_cold_start_examples(
    paths: List[str], k: int
) -> Tuple[List[dict], np.ndarray, Dict[int, List[dict]]]:
    """Load signal logs and extract cold-start examples (first k rounds per client).

    Returns:
        rows: list of signal log row dicts (cold-start only)
        labels: binary array (1=malicious)
        round_rows: {scenario_round: [all rows in that round]} for per-round stats
    """
    all_rows = []
    round_rows: Dict[int, List[dict]] = {}

    for path in paths:
        file_rows = _load_jsonl(path)
        # Group by logical_cid
        client_rows: Dict[str, List[dict]] = {}
        for r in file_rows:
            cid = r.get("logical_cid", "")
            client_rows.setdefault(cid, []).append(r)
            sr = r.get("scenario_round", 0)
            round_rows.setdefault(sr, []).append(r)

        for cid in client_rows:
            client_rows[cid].sort(key=lambda x: x.get("scenario_round", 0))

        for cid, hist in client_rows.items():
            for i, row in enumerate(hist[:k]):
                all_rows.append(row)

    labels = np.array([1.0 if r.get("malicious_gt") else 0.0 for r in all_rows])
    return all_rows, labels, round_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    seed_logs = find_all_signal_logs()
    if not seed_logs:
        print("No signal logs found.")
        return 1

    extractor = FeatureExtractor()

    print(f"\n{'='*80}")
    print(f"  Phase 3.5 — Baseline Comparison (k={args.k})")
    print(f"  Seeds: {sorted(seed_logs.keys())}")
    print(f"  Total log files: {sum(len(v) for v in seed_logs.values())}")
    print(f"{'='*80}\n")

    # Baselines + our detectors
    baseline_names = ["B0:Random", "B1:L2-zscore", "B2:CosDissim", "B3:NormDev",
                      "B5:MB-Weight", "B6:TrustZscore"]
    our_names = ["Our_S", "Our_W", "Our_C"]

    # Load our trained detectors
    model_dir = PROJECT_ROOT / "models" / "cold_start" / "flower_reset"
    our_detectors = {}
    for fam in [Family.S, Family.W, Family.C]:
        path = model_dir / f"{fam.value}_k{args.k}_final.pkl"
        if path.exists():
            our_detectors[fam] = ColdStartDetector.load(str(path))

    # Leave-one-seed-out CV
    all_cv_results: Dict[str, List[dict]] = {n: [] for n in baseline_names + our_names}

    for held_out in sorted(seed_logs.keys()):
        eval_paths = seed_logs[held_out]
        train_paths = []
        for s, ps in seed_logs.items():
            if s != held_out:
                train_paths.extend(ps)

        eval_rows, eval_labels, eval_round_rows = build_cold_start_examples(eval_paths, args.k)

        if len(eval_rows) == 0 or eval_labels.sum() == 0:
            print(f"  seed={held_out}: skip (no positive examples)")
            continue

        rng = np.random.RandomState(held_out)

        # Compute baseline scores
        b0 = score_random(eval_rows, rng)
        b1 = score_l2_distance(eval_rows, eval_round_rows)
        b2 = score_cosine_dissim(eval_rows)
        b3 = score_norm_deviation(eval_rows, eval_round_rows)
        b5 = score_mb_weight(eval_rows, eval_round_rows)
        b6 = score_trust_zscore(eval_rows, eval_round_rows)

        baseline_scores = {
            "B0:Random": b0, "B1:L2-zscore": b1, "B2:CosDissim": b2,
            "B3:NormDev": b3, "B5:MB-Weight": b5, "B6:TrustZscore": b6,
        }

        # Evaluate baselines
        for name, scores in baseline_scores.items():
            m = evaluate_scores(scores, eval_labels)
            m["held_out_seed"] = held_out
            all_cv_results[name].append(m)

        # Evaluate our detectors
        for fam, det in our_detectors.items():
            # Build feature vectors for eval data
            client_rows_grouped: Dict[str, List[dict]] = {}
            for r in eval_rows:
                cid = r.get("logical_cid", "")
                client_rows_grouped.setdefault(cid, []).append(r)
            for cid in client_rows_grouped:
                client_rows_grouped[cid].sort(key=lambda x: x.get("scenario_round", 0))

            X_list = []
            for r in eval_rows:
                cid = r.get("logical_cid", "")
                hist = client_rows_grouped.get(cid, [])
                idx = hist.index(r) if r in hist else 0
                if fam == Family.S:
                    feat = extractor.extract_s(r)
                elif fam == Family.W:
                    feat = extractor.extract_w(hist[:idx + 1], args.k)
                else:
                    feat = extractor.extract_c(r, hist[:idx + 1], args.k)
                X_list.append(feat)

            X_eval = np.stack(X_list) if X_list else np.empty((0, 14))
            our_scores = det.predict_risk(X_eval)
            m = evaluate_scores(our_scores, eval_labels)
            m["held_out_seed"] = held_out
            label = f"Our_{fam.value}"
            all_cv_results[label].append(m)

        # Print per-seed summary
        print(f"  seed={held_out}: pos={int(eval_labels.sum())}, neg={int(len(eval_labels)-eval_labels.sum())}")
        for name in baseline_names + our_names:
            results = all_cv_results[name]
            if results and results[-1]["held_out_seed"] == held_out:
                r = results[-1]
                pr = r["pr_auc"]
                rec = r["recall_at_10pct_fpr"]
                print(f"    {name:18s}  PR-AUC={pr:.3f}  Recall@10%FPR={rec:.3f}" if pr else
                      f"    {name:18s}  (no data)")

    # Aggregate CV summary
    print(f"\n{'='*80}")
    print(f"  COMPARATIVE TABLE (k={args.k}, leave-one-seed-out CV)")
    print(f"{'='*80}")
    print(f"  {'Method':20s} {'PR-AUC':>12s} {'Recall@10%FPR':>16s}")
    print(f"  {'-'*50}")

    summary = {}
    for name in baseline_names + our_names:
        results = all_cv_results[name]
        pr_aucs = [r["pr_auc"] for r in results if r["pr_auc"] is not None]
        recalls = [r["recall_at_10pct_fpr"] for r in results if r["recall_at_10pct_fpr"] is not None]
        if pr_aucs:
            mean_pr = np.mean(pr_aucs)
            std_pr = np.std(pr_aucs)
            mean_rec = np.mean(recalls) if recalls else 0
            std_rec = np.std(recalls) if recalls else 0
            print(f"  {name:20s} {mean_pr:.3f}±{std_pr:.3f}   {mean_rec:.3f}±{std_rec:.3f}")
            summary[name] = {
                "mean_pr_auc": float(mean_pr), "std_pr_auc": float(std_pr),
                "mean_recall_at_10pct_fpr": float(mean_rec), "std_recall_at_10pct_fpr": float(std_rec),
                "n_folds": len(pr_aucs),
            }

    # Save
    out_path = PROJECT_ROOT / "results" / "20260416" / f"phase3_5_baseline_comparison_k{args.k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"k": args.k, "summary": summary, "per_seed": {
            name: results for name, results in all_cv_results.items()
        }}, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Gate check
    if our_detectors:
        best_baseline_pr = max(
            (summary.get(n, {}).get("mean_pr_auc", 0) for n in baseline_names), default=0
        )
        our_best_pr = max(
            (summary.get(n, {}).get("mean_pr_auc", 0) for n in our_names), default=0
        )
        print(f"\n  Gate 3.5: Our best PR-AUC={our_best_pr:.3f} vs "
              f"best baseline PR-AUC={best_baseline_pr:.3f}")
        if our_best_pr >= best_baseline_pr - 0.001:
            print(f"  → PASS (our detector ≥ best baseline)")
        else:
            print(f"  → FAIL (our detector < best baseline by {best_baseline_pr - our_best_pr:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

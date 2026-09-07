"""Phase 3 P3.5 — Train and evaluate cold-start detectors.

Trains S-only, W-only, and C (combined) detectors on Flower-mode dev signal
logs, evaluates each on the cold-start window (first k=3 observed rounds),
and reports PR-AUC, recall@10%FPR, median TTD, and FPR on benign.

The training set is built from all 25 JSONL files (5 seeds × 5 scenarios).
Cross-validation: leave-one-seed-out (train on 4 seeds, eval on 1; repeat
for all 5; report mean ± std).

Output:
    models/cold_start/flower_reset/{family}_cv_results.json
    models/cold_start/flower_reset/{family}_final.pkl (trained on all 5 seeds)

Usage:
    conda run -n flowerfl python scripts/train_cold_start_detector.py \
        --mode flower_reset --k 3

    # Quick test on k=1 or k=5 sensitivity
    conda run -n flowerfl python scripts/train_cold_start_detector.py \
        --mode flower_reset --k 1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.cold_start_detector import (  # noqa: E402
    ColdStartDetector,
    Family,
    FeatureExtractor,
    feature_names,
)


DEV_SEEDS = [42, 137, 256, 314, 500]


def find_signal_logs(mode: str, include_dynamic: bool = True,
                     sweep_zmax: str = "") -> dict[int, list[str]]:
    """Return {seed: [list of JSONL paths]} for all dev signal logs.

    Searches for both scenario-based Flower logs (defense=none) and
    dynamic RMC persistent-opt logs (defense=krum).

    Args:
        sweep_zmax: If set (e.g. "0.9"), also include real ALIE sweep
            signals at that z_max (Phase 3c data).
    """
    out: dict[int, list[str]] = {}
    for seed in DEV_SEEDS:
        files = []
        # Flower scenario logs
        pat1 = str(PROJECT_ROOT / "signals" / f"{mode}__*__none__seed{seed}.jsonl")
        files.extend(sorted(glob.glob(pat1)))
        # Dynamic RMC logs (persistent-opt mode)
        if include_dynamic:
            pat2 = str(PROJECT_ROOT / "signals" / f"persistent_optimizer__szelag_dynamic_rmc__*__seed{seed}.jsonl")
            files.extend(sorted(glob.glob(pat2)))
        # Real ALIE sweep signals (Phase 3c)
        if sweep_zmax:
            pat3 = str(PROJECT_ROOT / "signals" / f"sweep_zmax{sweep_zmax}__seed{seed}.jsonl")
            files.extend(sorted(glob.glob(pat3)))
        if files:
            out[seed] = files
    return out


def evaluate_detector(
    detector: ColdStartDetector,
    X: np.ndarray,
    y: np.ndarray,
    fpr_threshold: float = 0.10,
) -> dict:
    """Compute cold-start detection metrics.

    Returns:
        pr_auc: precision-recall AUC
        recall_at_fpr: recall when FPR ≤ fpr_threshold
        fpr_on_benign: FPR on the benign-only subset
        precision_at_fpr: precision at the FPR threshold
        n_pos: number of malicious examples
        n_neg: number of honest examples
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    if len(X) == 0 or y.sum() == 0 or (1 - y).sum() == 0:
        return {"pr_auc": None, "recall_at_fpr": None, "fpr_on_benign": None,
                "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}

    risk = detector.predict_risk(X)

    # PR-AUC
    pr_auc = float(average_precision_score(y, risk))

    # ROC for recall@FPR threshold
    fpr, tpr, thresholds = roc_curve(y, risk)
    # Find the highest TPR (recall) where FPR ≤ fpr_threshold
    valid = fpr <= fpr_threshold
    recall_at_fpr = float(tpr[valid][-1]) if valid.any() else 0.0

    # FPR on benign examples (those labeled 0): what fraction get risk > 0.5?
    benign_mask = y == 0
    if benign_mask.sum() > 0:
        benign_risk = risk[benign_mask]
        fpr_on_benign = float((benign_risk > 0.5).mean())
    else:
        fpr_on_benign = None

    # Precision at the FPR threshold point
    prec_at_fpr = None
    if valid.any():
        threshold_at_fpr = thresholds[valid][-1] if len(thresholds[valid]) > 0 else 0.5
        predicted_pos = risk >= threshold_at_fpr
        if predicted_pos.sum() > 0:
            prec_at_fpr = float(y[predicted_pos].mean())

    return {
        "pr_auc": pr_auc,
        "recall_at_10pct_fpr": recall_at_fpr,
        "fpr_on_benign": fpr_on_benign,
        "precision_at_10pct_fpr": prec_at_fpr,
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_total": int(len(y)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="flower_reset",
                   choices=["flower_reset", "persistent_optimizer"])
    p.add_argument("--k", type=int, default=3, help="Cold-start window size")
    p.add_argument("--sweep-zmax", type=str, default="",
                   help="Include real ALIE sweep signals at this z_max (e.g. 0.9)")
    args = p.parse_args()

    extractor = FeatureExtractor()
    seed_logs = find_signal_logs(args.mode, sweep_zmax=args.sweep_zmax)

    if not seed_logs:
        print(f"No signal logs found for mode={args.mode}. Run collect_dev_signals.py first.")
        return 1

    print(f"\n{'='*72}")
    print(f"  Phase 3 P3.5 — Train Cold-Start Detectors")
    print(f"  mode: {args.mode}  k={args.k}")
    print(f"  seeds with logs: {sorted(seed_logs.keys())}")
    total_files = sum(len(v) for v in seed_logs.values())
    print(f"  total signal log files: {total_files}")
    print(f"{'='*72}\n")

    for family in [Family.S, Family.W, Family.C]:
        print(f"\n--- Family {family.value} ({len(feature_names(family))} features) ---")

        # Leave-one-seed-out cross-validation
        cv_results = []
        for held_out_seed in sorted(seed_logs.keys()):
            train_paths = []
            eval_paths = []
            for seed, paths in seed_logs.items():
                if seed == held_out_seed:
                    eval_paths.extend(paths)
                else:
                    train_paths.extend(paths)

            if not train_paths or not eval_paths:
                continue

            X_train, y_train = extractor.build_training_set(
                train_paths, family=family, k=args.k, cold_start_only=True
            )
            X_eval, y_eval = extractor.build_training_set(
                eval_paths, family=family, k=args.k, cold_start_only=True
            )

            if len(X_train) < 10 or y_train.sum() < 2:
                print(f"  seed={held_out_seed}: skip (too few training examples: "
                      f"{len(X_train)} total, {int(y_train.sum())} pos)")
                continue

            det = ColdStartDetector(family=family, k=args.k)
            det.fit(X_train, y_train)
            metrics = evaluate_detector(det, X_eval, y_eval)
            metrics["held_out_seed"] = held_out_seed
            metrics["n_train"] = len(X_train)
            metrics["n_eval"] = len(X_eval)
            cv_results.append(metrics)

            print(f"  seed={held_out_seed}: PR-AUC={metrics['pr_auc']:.3f}  "
                  f"recall@10%FPR={metrics['recall_at_10pct_fpr']:.3f}  "
                  f"FPR-benign={metrics.get('fpr_on_benign', '?')}  "
                  f"(train={len(X_train)}, eval={len(X_eval)}, "
                  f"pos={metrics['n_pos']}, neg={metrics['n_neg']})")

        if not cv_results:
            print(f"  No CV results for Family {family.value}")
            continue

        # Aggregate CV stats
        pr_aucs = [r["pr_auc"] for r in cv_results if r["pr_auc"] is not None]
        recalls = [r["recall_at_10pct_fpr"] for r in cv_results if r["recall_at_10pct_fpr"] is not None]

        print(f"\n  CV Summary (Family {family.value}):")
        if pr_aucs:
            print(f"    PR-AUC:          {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f}")
        if recalls:
            print(f"    Recall@10%FPR:   {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

        # Train final model on ALL seeds
        all_paths = []
        for paths in seed_logs.values():
            all_paths.extend(paths)
        X_all, y_all = extractor.build_training_set(
            all_paths, family=family, k=args.k, cold_start_only=True
        )
        final_det = ColdStartDetector(family=family, k=args.k)
        final_det.fit(X_all, y_all)

        # Feature importance
        imp = final_det.feature_importance()
        print(f"\n  Feature importance (Family {family.value}):")
        for name, score in imp[:5]:
            print(f"    {name:20s} {score:.4f}")

        # Save
        out_dir = PROJECT_ROOT / "models" / "cold_start" / args.mode
        out_dir.mkdir(parents=True, exist_ok=True)

        final_det.save(str(out_dir / f"{family.value}_k{args.k}_final.pkl"))

        cv_out = {
            "family": family.value,
            "k": args.k,
            "mode": args.mode,
            "n_seeds": len(seed_logs),
            "cv_results": cv_results,
            "cv_mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
            "cv_std_pr_auc": float(np.std(pr_aucs)) if pr_aucs else None,
            "cv_mean_recall_at_10pct_fpr": float(np.mean(recalls)) if recalls else None,
            "cv_std_recall_at_10pct_fpr": float(np.std(recalls)) if recalls else None,
            "final_model_path": str(out_dir / f"{family.value}_k{args.k}_final.pkl"),
            "final_n_train": int(len(X_all)),
            "final_n_pos": int(y_all.sum()),
            "final_feature_importance": [
                {"feature": n, "importance": float(s)} for n, s in imp
            ],
        }
        cv_path = out_dir / f"{family.value}_k{args.k}_cv_results.json"
        with open(cv_path, "w") as f:
            json.dump(cv_out, f, indent=2)
        print(f"\n  Saved: {cv_path}")
        print(f"  Saved: {out_dir / f'{family.value}_k{args.k}_final.pkl'}")

    # Decision Gate 1 check
    print(f"\n{'='*72}")
    print(f"  Decision Gate 1 (v23 §10.1):")
    print(f"  Combined (C) detector must outperform both S and W ablations")
    print(f"  on PR-AUC and recall@10%FPR in first k={args.k} rounds.")
    print(f"  → Review the CV summaries above to determine PASS/FAIL.")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

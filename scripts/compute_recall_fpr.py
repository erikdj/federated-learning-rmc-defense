"""Compute recall@FPR=target from signal-log JSONL files.

H2 (per docs/METHODOLOGY.md) is operationalized as recall@FPR=10%. This script
pools per-(round, client) records from signal logs, builds an ROC curve over the
specified detector score (krum_score / trust_score / tge_score), and reports
recall at the threshold where FPR <= target.

Usage:
    python scripts/compute_recall_fpr.py \\
        --signal-glob 'signals/flower_reset__*__tgensemble__*.jsonl' \\
        --score-field tge_score \\
        --target-fpr 0.10
"""
import argparse
import json
import sys
from pathlib import Path


def recall_at_target_fpr(records: list[dict], target_fpr: float = 0.10):
    """records: list of {"score": float, "malicious": bool}.
    Returns (recall_at_target, threshold, actual_fpr) or (None, None, None) if undefined.

    Defenders REJECT clients with LOW scores (below threshold = flagged as malicious).
    So:
      - True Positive: malicious client with score < threshold (correctly flagged)
      - False Positive: honest client with score < threshold (wrongly flagged)
    """
    malicious = [r for r in records if r["malicious"]]
    honest = [r for r in records if not r["malicious"]]
    if not malicious or not honest:
        return None, None, None

    unique_scores = sorted({r["score"] for r in records})

    # Use midpoints between adjacent unique scores as candidate thresholds.
    # This ensures the threshold sits strictly between score clusters rather than
    # landing on an observed score value, which makes semantics unambiguous:
    # "reject clients whose score < threshold" is evaluated with a threshold that
    # cleanly separates the two nearest score groups.
    # Also include sentinels just below the minimum and just above the maximum.
    sentinels = [unique_scores[0] - 1e-9] + [
        (unique_scores[i] + unique_scores[i + 1]) / 2
        for i in range(len(unique_scores) - 1)
    ] + [unique_scores[-1] + 1e-9]

    best_recall = 0.0
    best_threshold = None
    best_actual_fpr = 0.0

    for t in sentinels:
        tp = sum(1 for r in malicious if r["score"] < t)
        fp = sum(1 for r in honest if r["score"] < t)
        fpr = fp / len(honest)
        tpr = tp / len(malicious)
        if fpr <= target_fpr and tpr > best_recall:
            best_recall = tpr
            best_threshold = t
            best_actual_fpr = fpr

    return best_recall, best_threshold, best_actual_fpr


def filter_scope(rows: list[dict], scope: str = "coldstart", k: int = 3) -> list[dict]:
    """coldstart = rows within the first k rounds of an identity's tenure; all = no filter."""
    if scope == "all":
        return rows
    return [r for r in rows if r.get("tenure") is not None and 1 <= int(r["tenure"]) <= k]


def load_records(signal_glob: str, score_field: str) -> list[dict]:
    """Pool records across all JSONL files matching the glob (relative to cwd)."""
    records = []
    for p in Path(".").glob(signal_glob):
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                score = rec.get(score_field)
                if score is None:
                    continue
                records.append({
                    "score": float(score),
                    "malicious": bool(rec["malicious_gt"]),
                    "seed": rec.get("seed"),
                    "round": rec.get("server_round"),
                    "logical_cid": rec.get("logical_cid"),
                    "tenure": rec.get("tenure"),
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-glob", required=True)
    parser.add_argument("--score-field", required=True,
                        choices=["krum_score", "trust_score", "tge_score", "tge_gbdt_score"])
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--scope", choices=["coldstart", "all"], default="coldstart")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Frozen threshold (from h2_threshold_pipeline freeze step). "
                             "When provided, skips FPR-based threshold selection and computes "
                             "recall directly at this fixed value (flag = score < threshold).")
    args = parser.parse_args()

    records = load_records(args.signal_glob, args.score_field)
    print(f"Loaded {len(records)} records from {args.signal_glob} using {args.score_field}")
    records = filter_scope(records, scope=args.scope, k=args.k)
    print(f"Scope={args.scope!r} k={args.k}: {len(records)} rows remain after filtering")

    if args.threshold is not None:
        # Confirmatory path: apply the frozen threshold directly.
        # Flag = score < threshold (low score = flagged malicious, per defender convention).
        malicious = [r for r in records if r["malicious"]]
        if not malicious:
            print("UNDEFINED (no malicious records in input)")
            sys.exit(1)
        tp = sum(1 for r in malicious if r["score"] < args.threshold)
        recall = tp / len(malicious)
        honest = [r for r in records if not r["malicious"]]
        actual_fpr = (sum(1 for r in honest if r["score"] < args.threshold) / len(honest)
                      if honest else float("nan"))
        print(f"recall@frozen_threshold={args.threshold}: {recall:.4f}")
        print(f"  frozen threshold:             {args.threshold:.4f}")
        print(f"  actual FPR at this threshold: {actual_fpr:.4f}")
    else:
        recall, threshold, actual_fpr = recall_at_target_fpr(records, args.target_fpr)
        if recall is None:
            print("UNDEFINED (no malicious or no honest records in input)")
            sys.exit(1)

        print(f"recall@FPR<={args.target_fpr}: {recall:.4f}")
        print(f"  threshold where this is achieved: {threshold:.4f}")
        print(f"  actual FPR at this threshold:    {actual_fpr:.4f}")


if __name__ == "__main__":
    main()

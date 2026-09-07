"""H1 signal-family evaluation — Finding F1/F5.

Design (v1.3 spec § 2, F1/F5):
    Train S/W/C detectors on the DEPLOYED-config (Krum+TGE) DEV-seed signal logs,
    then evaluate them OUT-OF-SAMPLE on the CONFIRMATORY-seed logs (cold-start window).

Signal source: single named deployed config (Krum+TGE) only — one config, two seed
partitions.  This ensures the family comparison is not confounded by defense choice.

Families:
    S — single-round features (update_norm, cos_to_median, L2_to_median,
        train_loss, num_examples); trained per-round, one row per client-round.
    W — short-window temporal features (norm_change, norm_variance, norm_mean,
        cos_drift, cos_variance, cos_mean, loss_slope, loss_variance, tenure_count);
        one row per client (summarises first k rounds).
    C — combined S + W (all 14 features); one row per client-round up to k.

Seed split (F1 — out-of-sample):
    Dev seeds   [42, 137, 256, 314, 500]        → train detector + select threshold
    Confirm seeds [1009, 1733, 2521, ...]        → eval recall@10%FPR (frozen threshold)

Threshold protocol (F3/F8):
    1. Collect GBDT risk scores on HONEST development rows.
    2. Call select_threshold_risk(honest_scores, target_fpr) to freeze a cut at the
       1 - target_fpr quantile of honest scores; a row is flagged when
       score > cut.  (GBDT scores are malicious-RISK probabilities, so HIGH is
       the suspicious end.  h2_threshold_pipeline.select_threshold is the mirror
       selector for native defense scores where LOW is suspicious, and is not
       interchangeable with this one.)
    3. Apply filter_scope(..., scope="coldstart", k=args.k) to restrict to the
       cold-start window (first k rounds post join/rejoin) — primary scope per F6.
    4. recall = flagged-malicious / total-malicious in that window.

Feature extraction delegates entirely to flowerfl.cold_start_detector.FeatureExtractor
and flowerfl.cold_start_detector.Family — no feature math is re-derived here.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Seed-split contract (hard-gated unit test covers this)
# ---------------------------------------------------------------------------

def split_by_seed(
    rows: List[Dict[str, Any]],
    train_seeds: List[int],
    eval_seeds: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition rows into disjoint train / eval sets by seed field.

    Args:
        rows: list of dicts, each must have a ``seed`` key.
        train_seeds: seeds assigned to the training partition.
        eval_seeds: seeds assigned to the evaluation partition.

    Returns:
        (train_rows, eval_rows) — guaranteed disjoint by construction because
        train_seeds and eval_seeds are tested for disjointness by the caller.
    """
    tr = set(train_seeds)
    ev = set(eval_seeds)
    return (
        [r for r in rows if r.get("seed") in tr],
        [r for r in rows if r.get("seed") in ev],
    )


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _load_jsonl_glob(pattern: str) -> List[Dict[str, Any]]:
    """Load all JSONL files matching glob pattern; attach path-level seed if present."""
    rows: List[Dict[str, Any]] = []
    for path in glob.glob(pattern):
        p = Path(path)
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Feature extraction — delegates to cold_start_detector.FeatureExtractor
# ---------------------------------------------------------------------------

def extract_family_features(
    rows: List[Dict[str, Any]],
    family: str,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) feature matrix for one signal family from flat signal-log rows.

    Delegates to ``flowerfl.cold_start_detector.FeatureExtractor``:
        - Family S: ``extractor.extract_s(row)`` — one vector per client-round row.
        - Family W: ``extractor.extract_w(history, k)`` — one vector per client
          (history = all rows for that logical_cid, sorted by scenario_round).
        - Family C: ``extractor.extract_c(row, history[:obs_idx], k)`` — one vector
          per client-round row within the first k rounds.

    Only rows within the first k observed rounds per logical_cid are included
    (cold_start_only=True logic, matching FeatureExtractor.build_training_set behaviour).

    Label: ``malicious_gt`` field (bool/int); dormant adversary rounds count positive
    per F7 (per-identity ground truth is already encoded in the signal log).

    Args:
        rows: flat list of signal-log dicts (all seeds merged before this call).
        family: one of "S", "W", "C".
        k: cold-start window size.

    Returns:
        X: (n_examples, n_features) float32 array.
        y: (n_examples,) float32 binary labels.
    """
    # Import here so the module is importable even without the flowerfl package
    # on sys.path (the import-sanity check in main() will catch problems early).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flowerfl.cold_start_detector import Family, FeatureExtractor, feature_names

    fam = Family(family)
    extractor = FeatureExtractor()

    # Group rows by logical_cid; sort each group by scenario_round.
    client_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = str(r.get("logical_cid", ""))
        client_rows.setdefault(cid, []).append(r)
    for cid in client_rows:
        client_rows[cid].sort(key=lambda r: r.get("scenario_round", 0))

    all_X: List[np.ndarray] = []
    all_y: List[float] = []

    for cid, hist in client_rows.items():
        for i, row in enumerate(hist):
            obs_round = i + 1  # 1-indexed observed round for this client
            if obs_round > k:
                break  # cold_start_only: only first k rounds

            if fam == Family.S:
                feat = extractor.extract_s(row)
            elif fam == Family.W:
                feat = extractor.extract_w(hist[:obs_round], k)
            else:  # Family.C
                feat = extractor.extract_c(row, hist[:obs_round], k)

            label = 1.0 if row.get("malicious_gt", False) else 0.0
            all_X.append(feat)
            all_y.append(label)

    if not all_X:
        n_feat = len(feature_names(fam))
        return np.empty((0, n_feat), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.stack(all_X).astype(np.float32), np.array(all_y, dtype=np.float32)


# ---------------------------------------------------------------------------
# GBDT training (sklearn GradientBoostingClassifier, seeded random_state=0)
# ---------------------------------------------------------------------------

def _train_gbdt(X: np.ndarray, y: np.ndarray):
    """Train a sklearn GradientBoostingClassifier on (X, y)."""
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Threshold selection for RISK scores (higher = more suspicious)
# ---------------------------------------------------------------------------

def select_threshold_risk(honest_scores, target_fpr: float = 0.10) -> float:
    """Calibration cut for RISK scores, where HIGH score = suspicious.

    Mirrors the project-wide high-score convention (cf. ``cut_from_calibration``
    with ``higher_is_trust=False`` in the H2′ protocol revalidation): the cut is
    the ``1 - target_fpr`` quantile of the honest calibration scores (numpy
    default linear interpolation), and a row is flagged when ``score > cut``.

    Note this is the mirror image of ``h2_threshold_pipeline.select_threshold``,
    which is the correct selector for native defense scores where a LOW score is
    the suspicious end and the flag test is ``score < threshold``.

    Args:
        honest_scores: development-calibration scores from honest rows.
        target_fpr: nominal false-positive rate the cut is calibrated to.

    Returns:
        The cut. An empty calibration set yields ``+inf`` (nothing flagged),
        the conservative choice when no calibration support exists.
    """
    a = np.asarray(list(honest_scores), dtype=float)
    if a.size == 0:
        return float("inf")
    return float(np.quantile(a, 1.0 - target_fpr))


# ---------------------------------------------------------------------------
# Recall computation (reuses filter_scope + select_threshold_risk)
# ---------------------------------------------------------------------------

def score_and_scope(
    clf,
    confirm_rows: List[Dict[str, Any]],
    family: str,
    k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score rows with clf and restrict to the cold-start window.

    Returns (scored_records, scoped) where ``scored_records`` is one record per
    scored client-round (score, malicious, tenure, logical_cid, seed) in scoring
    order, and ``scoped`` is that list after ``filter_scope(scope="coldstart")``.
    """
    from compute_recall_fpr import filter_scope

    # Build feature matrix for confirm set.
    X_ev, _y_ev = extract_family_features(confirm_rows, family, k)

    if X_ev.shape[0] == 0:
        return [], []

    # Raw GBDT risk scores ∈ [0, 1]; higher = more likely malicious.
    scores = clf.predict_proba(X_ev)[:, 1]

    # Attach scores back to rows for filter_scope (which needs "tenure" field).
    # We rebuild a flat record list with score + malicious flag + tenure.
    # tenure = obs_round (1-indexed) — re-derive from the same grouping logic.
    client_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in confirm_rows:
        cid = str(r.get("logical_cid", ""))
        client_rows.setdefault(cid, []).append(r)
    for cid in client_rows:
        client_rows[cid].sort(key=lambda r: r.get("scenario_round", 0))

    scored_records: List[Dict[str, Any]] = []
    score_idx = 0
    for cid, hist in client_rows.items():
        for i, row in enumerate(hist):
            obs_round = i + 1
            if obs_round > k:
                break
            scored_records.append({
                "score": float(scores[score_idx]),
                "malicious": bool(row.get("malicious_gt", False)),
                "tenure": obs_round,
                "logical_cid": cid,
                "seed": row.get("seed"),
            })
            score_idx += 1

    # Apply cold-start filter (primary scope per F6).
    return scored_records, filter_scope(scored_records, scope="coldstart", k=k)


def _compute_recall_at_fpr(
    clf,
    confirm_rows: List[Dict[str, Any]],
    family: str,
    k: int,
    target_fpr: float,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Score confirm rows with clf, apply cold-start filter, compute recall@FPR.

    Uses:
        filter_scope (compute_recall_fpr.py) — restrict to first k rounds per identity.
        select_threshold_risk — target the nominal FPR on the honest scores of
            this row set when no external threshold is supplied; report the
            realized FPR separately.

    Args:
        threshold: when given, this cut is applied unchanged and no calibration
            is performed on ``confirm_rows`` — the path used for a threshold
            frozen on development data before a held-out read.

    Returns dict with recall_at_fpr, n_train (not known here; caller fills),
    n_eval, threshold, actual_fpr.
    """
    # Import shared utilities (scripts/ dir is on sys.path from main()).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    scored_records, scoped = score_and_scope(clf, confirm_rows, family, k)

    if not scoped:
        return {"recall_at_fpr": None, "n_eval": len(scored_records), "threshold": None, "actual_fpr": None}

    if threshold is None:
        honest_scores = [r["score"] for r in scoped if not r["malicious"]]
        threshold = select_threshold_risk(honest_scores, target_fpr)
    else:
        threshold = float(threshold)

    malicious_scoped = [r for r in scoped if r["malicious"]]
    n_mal = len(malicious_scoped)
    if n_mal == 0:
        return {"recall_at_fpr": None, "n_eval": len(scoped), "threshold": threshold, "actual_fpr": None}

    # Defender flags clients with HIGH risk score (opposite of low-score convention
    # used for native defense scores).  GBDT score is probability of malicious,
    # so flag = score > threshold.
    tp = sum(1 for r in malicious_scoped if r["score"] > threshold)
    recall = tp / n_mal

    honest_scoped = [r for r in scoped if not r["malicious"]]
    actual_fpr = (
        sum(1 for r in honest_scoped if r["score"] > threshold) / len(honest_scoped)
        if honest_scoped else float("nan")
    )

    return {
        "recall_at_fpr": recall,
        "n_eval": len(scoped),
        "threshold": threshold,
        "actual_fpr": actual_fpr,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """H1 signal-family evaluation pipeline (F1/F5).

    Loads dev signal logs (train) and confirmatory signal logs (eval),
    trains a GBDT per family {S, W, C}, evaluates recall@target-FPR in the
    cold-start window, and writes a JSON summary.
    """
    # Ensure scripts/ is on path for sibling imports.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    ap = argparse.ArgumentParser(
        description="H1 signal-family evaluation: train on dev logs, eval on confirmatory logs."
    )
    ap.add_argument(
        "--dev-signal-glob",
        required=True,
        help="Glob for Krum+TGE DEV-seed signal-log JSONL files (train partition).",
    )
    ap.add_argument(
        "--confirm-signal-glob",
        required=True,
        help="Glob for Krum+TGE CONFIRMATORY-seed signal-log JSONL files (eval partition).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="Cold-start window size (number of rounds post join/rejoin). Default: 3.",
    )
    ap.add_argument(
        "--target-fpr",
        type=float,
        default=0.10,
        help="Target FPR for recall@FPR computation. Default: 0.10.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path for family summary {family: {recall_at_fpr, n_train, n_eval}}.",
    )
    args = ap.parse_args()

    print(f"[h1_signal_family_eval] Loading dev logs: {args.dev_signal_glob}")
    dev_rows = _load_jsonl_glob(args.dev_signal_glob)
    print(f"  {len(dev_rows)} dev rows loaded.")

    print(f"[h1_signal_family_eval] Loading confirmatory logs: {args.confirm_signal_glob}")
    confirm_rows = _load_jsonl_glob(args.confirm_signal_glob)
    print(f"  {len(confirm_rows)} confirmatory rows loaded.")

    summary: Dict[str, Any] = {}

    for family in ("S", "W", "C"):
        print(f"\n[h1_signal_family_eval] Family {family}")

        X_tr, y_tr = extract_family_features(dev_rows, family, args.k)
        n_train = X_tr.shape[0]
        print(f"  Train: {n_train} examples, {X_tr.shape[1] if n_train else 0} features, "
              f"{int(y_tr.sum())} malicious.")

        if n_train == 0:
            print(f"  WARNING: no training examples for family {family}; skipping.")
            summary[family] = {"recall_at_fpr": None, "n_train": 0, "n_eval": 0}
            continue

        clf = _train_gbdt(X_tr, y_tr)
        _all_dev, scoped_dev = score_and_scope(clf, dev_rows, family, args.k)
        honest_dev_scores = [
            row["score"] for row in scoped_dev if not row["malicious"]
        ]
        threshold = select_threshold_risk(honest_dev_scores, args.target_fpr)

        result = _compute_recall_at_fpr(
            clf,
            confirm_rows,
            family,
            args.k,
            args.target_fpr,
            threshold=threshold,
        )
        result["n_train"] = n_train
        summary[family] = result

        print(f"  Eval n={result['n_eval']}, recall@{args.target_fpr:.0%}FPR="
              f"{result['recall_at_fpr']}, threshold={result['threshold']}, "
              f"actual_fpr={result['actual_fpr']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[h1_signal_family_eval] Summary written to {out_path}")


if __name__ == "__main__":
    main()

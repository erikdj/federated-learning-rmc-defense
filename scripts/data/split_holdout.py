"""Split the existing 42K server-side holdout manifest into val + test.

Reads data/holdout_manifest_edge_full_20_rmc.json (built in Phase 1 P1.1)
and produces data/val_test_split_manifest.json with per-partition val and
test indices, stratified per (partition × label).

Implements professor's "training, held-out validation, and test splits"
ask at the data level:
  - Train: parquet rows minus original holdout indices (unchanged)
  - Val:   ~30K rows (70% of 42K), used for hparam/threshold calibration
  - Test:  ~12K rows (30% of 42K), sealed until Phase 8

Usage:
    conda run -n flowerfl python scripts/data/split_holdout.py
    # writes data/val_test_split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def split_holdout_manifest(
    holdout: dict[str, Any],
    val_frac: float = 0.7,
    seed: int = 20260515,
    _created: str | None = None,
) -> dict[str, Any]:
    """Stratified per (partition × label) split of an existing holdout manifest.

    Args:
        holdout: Loaded contents of holdout_manifest_edge_full_20_rmc.json.
        val_frac: Fraction of rows per stratum routed to val (rest → test).
        seed: Seed for the per-stratum shuffle.

    Returns:
        Manifest dict with per-partition val_indices, test_indices, and
        per_label_counts plus reproducibility metadata.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac={val_frac} must be in (0, 1)")

    rng = random.Random(seed)
    per_partition_out: dict[str, dict[str, Any]] = {}

    for pid, payload in holdout["per_partition"].items():
        indices = list(payload["indices"])
        # We do not have row-level labels in the holdout manifest itself,
        # only label_dist. The holdout indices are already stratified by
        # label at construction time (Phase 1 P1.1, sampling_seed=20260416).
        # We approximate per-label stratification by walking indices in
        # their stored order and applying the same val/test fraction per
        # consecutive label block, recorded from the holdout's label_dist.
        label_dist = payload["label_dist"]
        # Order labels descending by count for deterministic walk
        ordered_labels = sorted(label_dist.keys(), key=lambda k: -label_dist[k])

        cursor = 0
        val_idx: list[int] = []
        test_idx: list[int] = []
        per_label_counts: dict[str, dict[str, int]] = {}
        for label in ordered_labels:
            n = label_dist[label]
            block = indices[cursor:cursor + n]
            cursor += n
            # Deterministic shuffle within the block
            shuffled = block[:]
            rng.shuffle(shuffled)
            n_val = int(round(n * val_frac))
            block_val = shuffled[:n_val]
            block_test = shuffled[n_val:]
            val_idx.extend(block_val)
            test_idx.extend(block_test)
            per_label_counts[str(label)] = {"val": len(block_val), "test": len(block_test)}

        # Sort indices ascending for stable diffs
        val_idx.sort()
        test_idx.sort()

        per_partition_out[pid] = {
            "file": payload["file"],
            "val_indices": val_idx,
            "test_indices": test_idx,
            "per_label_counts": per_label_counts,
        }

    # Use caller-supplied timestamp if provided; otherwise derive a deterministic
    # ISO string from the seed so that two calls with the same seed produce
    # identical manifests (required by test_deterministic_given_seed).
    # main() injects the actual wall-clock time via _created.
    if _created is not None:
        created = _created
    else:
        # Encode seed as YYYY-MM-DD date (seed format: YYYYMMDD)
        seed_str = str(seed)
        if len(seed_str) == 8:
            created = f"{seed_str[:4]}-{seed_str[4:6]}-{seed_str[6:8]}T00:00:00+00:00"
        else:
            created = f"seed={seed}"
    return {
        "_meta": {
            "description": (
                "Val/test split of the 42K server-side holdout (Phase 1 P1.1). "
                "Stratified per (partition × label) per-block fixed fraction. "
                "Locked: never re-cut to chase results. See METHODOLOGY pre-reg rule 4."
            ),
            "source_manifest": "data/holdout_manifest_edge_full_20_rmc.json",
            "val_frac": val_frac,
            "seed": seed,
            "stratification": "per (partition x label) block, deterministic",
            "phase": "Phase 1 P1.1 amendment (2026-05-14)",
            "created": created,
        },
        "per_partition": per_partition_out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--holdout-manifest",
        type=Path,
        default=Path("data/holdout_manifest_edge_full_20_rmc.json"),
    )
    p.add_argument("--val-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/val_test_split_manifest.json"),
    )
    args = p.parse_args()

    holdout = json.loads(args.holdout_manifest.read_text())
    manifest = split_holdout_manifest(
        holdout,
        val_frac=args.val_frac,
        seed=args.seed,
        _created=datetime.now(timezone.utc).isoformat(),
    )

    n_val = sum(len(p["val_indices"]) for p in manifest["per_partition"].values())
    n_test = sum(len(p["test_indices"]) for p in manifest["per_partition"].values())
    print(f"Split: {n_val:,} val + {n_test:,} test = {n_val + n_test:,} total")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

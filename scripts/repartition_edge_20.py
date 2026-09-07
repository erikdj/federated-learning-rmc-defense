"""
Repartition Edge-IIoT Dataset: 10 Clients -> 20 Clients (+ 1 RMC duplicate).

Matches Szelag et al. (2504.03077v1) experimental setup:
  - 20 total clients: 11 honest + 9 malicious (45% malicious fraction)
  - Non-IID distribution via Dirichlet partitioning (alpha=0.5)
  - client_20.parquet = duplicate of client_19.parquet (for RMC identity reset)

Input:  data/edge_full/client_0.parquet through client_9.parquet (~20.9M rows)
Output: data/edge_full_20/client_0.parquet through client_20.parquet + metadata.json + features.json

The original 10-client dataset uses sensor-type partitioning (natural non-IID).
This script concatenates all data and re-partitions using Dirichlet allocation
to create 20 non-IID partitions while preserving label balance within each.

Dirichlet Partitioning (alpha=0.5):
  - Samples a probability vector from Dir(alpha) for each class
  - Assigns rows of each class to clients according to these probabilities
  - alpha=0.5 produces moderate heterogeneity (neither IID nor extreme non-IID)
  - Each client gets a different proportion of benign vs attack samples

Client Assignment:
  - Clients 0-10:  Honest clients (will train faithfully)
  - Clients 11-19: Malicious clients (will execute attacks per scenario)
  - Client 20:     Duplicate of client 19 (RMC reconnection identity)

Usage:
    conda run -n flowerfl python scripts/repartition_edge_20.py
    conda run -n flowerfl python scripts/repartition_edge_20.py --alpha 0.3  # more non-IID
    conda run -n flowerfl python scripts/repartition_edge_20.py --dry-run    # stats only

Author: Erik D.J., GWU Doctoral Research (Praxis)
Date:   2026-04-02
"""

import os
import sys
import json
import gc
import time
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "edge_full"
OUTPUT_DIR = PROJECT_ROOT / "data" / "edge_full_20"

# Original 10 clients (skip client_10 which is a duplicate of client_9)
ORIGINAL_CLIENTS = [f"client_{i}.parquet" for i in range(10)]

NUM_PARTITIONS = 20       # 11 honest + 9 malicious
NUM_HONEST = 11           # clients 0-10
NUM_MALICIOUS = 9         # clients 11-19
RMC_DUPLICATE_SRC = 19    # client_19 -> client_20 (identity reset)
RMC_DUPLICATE_DST = 20

DEFAULT_ALPHA = 0.5       # Dirichlet concentration parameter
SEED = 42                 # Reproducibility
LABEL_COLUMN = "Attack_label"


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dirichlet_partition(
    df: pd.DataFrame,
    num_partitions: int,
    alpha: float,
    seed: int,
    label_col: str = LABEL_COLUMN,
) -> list[pd.DataFrame]:
    """
    Partition a DataFrame into non-IID subsets using Dirichlet allocation.

    For each unique label value, samples a probability vector from Dir(alpha)
    and assigns rows to partitions according to those probabilities. This
    ensures every partition gets some data from every class, but in different
    proportions (controlled by alpha).

    Args:
        df:              Full dataset to partition
        num_partitions:  Number of output partitions
        alpha:           Dirichlet concentration. Lower = more non-IID.
                         0.1 = extreme non-IID, 0.5 = moderate, 10.0 ~ IID
        seed:            Random seed for reproducibility
        label_col:       Name of the label column

    Returns:
        List of DataFrames, one per partition
    """
    rng = np.random.default_rng(seed)
    labels = df[label_col].values
    unique_labels = np.unique(labels)

    # Initialize empty index lists for each partition
    partition_indices = [[] for _ in range(num_partitions)]

    for label in unique_labels:
        # Indices of all rows with this label
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        # Sample Dirichlet distribution for this label
        proportions = rng.dirichlet(np.full(num_partitions, alpha))

        # Convert proportions to counts
        counts = (proportions * len(label_indices)).astype(int)

        # Distribute any remainder due to rounding
        remainder = len(label_indices) - counts.sum()
        if remainder > 0:
            # Add remainder to the largest partitions
            top_indices = np.argsort(proportions)[-remainder:]
            counts[top_indices] += 1
        elif remainder < 0:
            # Remove excess from the largest partitions
            top_indices = np.argsort(proportions)[remainder:]
            counts[top_indices] -= 1

        # Assign indices to partitions
        offset = 0
        for p in range(num_partitions):
            partition_indices[p].extend(label_indices[offset:offset + counts[p]])
            offset += counts[p]

    # Create DataFrames from indices
    partitions = []
    for p in range(num_partitions):
        idx = partition_indices[p]
        part_df = df.iloc[idx].copy()
        # Shuffle within partition for good measure
        part_df = part_df.sample(frac=1, random_state=seed + p).reset_index(drop=True)
        partitions.append(part_df)

    return partitions


def load_all_clients(input_dir: Path) -> pd.DataFrame:
    """Load and concatenate all 10 original client parquets."""
    dfs = []
    total_rows = 0
    for client_file in ORIGINAL_CLIENTS:
        filepath = input_dir / client_file
        if not filepath.exists():
            log(f"  WARNING: {filepath} not found, skipping")
            continue
        df = pd.read_parquet(filepath)
        log(f"  Loaded {client_file}: {len(df):,} rows, {df.shape[1]} columns")
        dfs.append(df)
        total_rows += len(df)

    if not dfs:
        raise RuntimeError(f"No client parquets found in {input_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Combined: {len(combined):,} rows (from {len(dfs)} clients)")

    # Verify no data loss
    assert len(combined) == total_rows, (
        f"Row count mismatch: {len(combined)} != {total_rows}"
    )

    del dfs
    gc.collect()
    return combined


def compute_stats(df: pd.DataFrame, label_col: str = LABEL_COLUMN) -> dict:
    """Compute partition statistics."""
    benign = int((df[label_col] == 0).sum())
    attack = int((df[label_col] == 1).sum())
    total = len(df)
    return {
        "rows": total,
        "label_dist": {"0": benign, "1": attack},
        "benign_pct": round(100 * benign / total, 1) if total > 0 else 0.0,
        "attack_pct": round(100 * attack / total, 1) if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Repartition Edge-IIoT from 10 to 20 clients using Dirichlet allocation"
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Dirichlet concentration parameter (default: {DEFAULT_ALPHA})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and display stats without writing files"
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(INPUT_DIR),
        help=f"Input directory (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    alpha = args.alpha
    seed = args.seed

    log("=" * 70)
    log("REPARTITION Edge-IIoT: 10 clients -> 20 clients (Szelag et al. setup)")
    log("=" * 70)
    log(f"  Input:      {input_dir}")
    log(f"  Output:     {output_dir}")
    log(f"  Partitions: {NUM_PARTITIONS} ({NUM_HONEST} honest + {NUM_MALICIOUS} malicious)")
    log(f"  Malicious:  {NUM_MALICIOUS}/{NUM_PARTITIONS} = {100*NUM_MALICIOUS/NUM_PARTITIONS:.0f}%")
    log(f"  Alpha:      {alpha} (Dirichlet concentration)")
    log(f"  Seed:       {seed}")
    log(f"  RMC dup:    client_{RMC_DUPLICATE_SRC} -> client_{RMC_DUPLICATE_DST}")
    log(f"  Dry run:    {args.dry_run}")
    log("")

    # =========================================================================
    # Phase 1: Load all original data
    # =========================================================================
    log("PHASE 1: Loading original 10-client data")
    log("-" * 50)
    combined = load_all_clients(input_dir)

    # Verify features
    feature_cols = sorted([c for c in combined.columns if c != LABEL_COLUMN])
    log(f"  Features: {len(feature_cols)}")
    log(f"  Label column: {LABEL_COLUMN}")

    overall_stats = compute_stats(combined)
    log(f"  Overall: {overall_stats['rows']:,} rows "
        f"({overall_stats['benign_pct']}% benign, {overall_stats['attack_pct']}% attack)")
    log("")

    # =========================================================================
    # Phase 2: Dirichlet partitioning
    # =========================================================================
    log("PHASE 2: Dirichlet partitioning into 20 non-IID clients")
    log("-" * 50)
    log(f"  Using alpha={alpha} (lower = more heterogeneous)")

    partitions = dirichlet_partition(
        combined, NUM_PARTITIONS, alpha=alpha, seed=seed
    )

    # Free the combined DataFrame
    del combined
    gc.collect()

    # Report per-partition stats
    metadata = {}
    log("")
    log(f"  {'Client':<12} {'Rows':>10} {'Benign%':>8} {'Attack%':>8} {'Role':<12}")
    log(f"  {'------':<12} {'----':>10} {'-------':>8} {'-------':>8} {'----':<12}")

    for i, part_df in enumerate(partitions):
        stats = compute_stats(part_df)
        role = "honest" if i < NUM_HONEST else "malicious"
        metadata[str(i)] = {
            "file": f"client_{i}.parquet",
            "rows": stats["rows"],
            "label_dist": stats["label_dist"],
            "benign_pct": stats["benign_pct"],
            "attack_pct": stats["attack_pct"],
            "role": role,
        }
        log(f"  client_{i:<4} {stats['rows']:>10,} {stats['benign_pct']:>7.1f}% "
            f"{stats['attack_pct']:>7.1f}% {role:<12}")

    # Summary statistics
    all_rows = [m["rows"] for m in metadata.values()]
    log("")
    log(f"  Total rows: {sum(all_rows):,}")
    log(f"  Min/Max partition size: {min(all_rows):,} / {max(all_rows):,}")
    log(f"  Std dev partition size: {np.std(all_rows):,.0f}")

    # Heterogeneity metric: std dev of benign_pct across partitions
    benign_pcts = [m["benign_pct"] for m in metadata.values()]
    log(f"  Benign% range: {min(benign_pcts):.1f}% - {max(benign_pcts):.1f}% "
        f"(std={np.std(benign_pcts):.1f}%)")

    if args.dry_run:
        log("")
        log("DRY RUN complete — no files written.")
        return

    # =========================================================================
    # Phase 3: Save partitions
    # =========================================================================
    log("")
    log("PHASE 3: Saving partitions to disk")
    log("-" * 50)

    os.makedirs(output_dir, exist_ok=True)

    for i, part_df in enumerate(partitions):
        outpath = output_dir / f"client_{i}.parquet"
        part_df.to_parquet(outpath, index=False)
        log(f"  Saved client_{i}.parquet ({len(part_df):,} rows)")

    # =========================================================================
    # Phase 4: Create RMC duplicate (client_20 = copy of client_19)
    # =========================================================================
    log("")
    log("PHASE 4: Creating RMC identity-reset duplicate")
    log("-" * 50)

    src_path = output_dir / f"client_{RMC_DUPLICATE_SRC}.parquet"
    dst_path = output_dir / f"client_{RMC_DUPLICATE_DST}.parquet"
    shutil.copy2(src_path, dst_path)

    # Add to metadata
    src_meta = metadata[str(RMC_DUPLICATE_SRC)]
    metadata[str(RMC_DUPLICATE_DST)] = {
        "file": f"client_{RMC_DUPLICATE_DST}.parquet",
        "rows": src_meta["rows"],
        "label_dist": dict(src_meta["label_dist"]),
        "benign_pct": src_meta["benign_pct"],
        "attack_pct": src_meta["attack_pct"],
        "role": "rmc_duplicate",
        "duplicate_of": f"client_{RMC_DUPLICATE_SRC}",
        "purpose": "Identity reset for Reconnecting Malicious Client experiment",
    }
    log(f"  Copied client_{RMC_DUPLICATE_SRC}.parquet -> client_{RMC_DUPLICATE_DST}.parquet")
    log(f"  ({src_meta['rows']:,} rows)")

    # =========================================================================
    # Phase 5: Save metadata and features
    # =========================================================================
    log("")
    log("PHASE 5: Saving metadata and features")
    log("-" * 50)

    # Build comprehensive metadata
    full_metadata = {
        "_meta": {
            "description": (
                "Edge-IIoT dataset repartitioned into 20 non-IID clients "
                "using Dirichlet allocation, matching Szelag et al. (2504.03077v1) "
                "experimental setup with 45% malicious fraction."
            ),
            "source": str(input_dir),
            "partitioning": {
                "method": "dirichlet",
                "alpha": alpha,
                "seed": seed,
                "num_partitions": NUM_PARTITIONS,
            },
            "client_roles": {
                "honest": list(range(NUM_HONEST)),
                "malicious": list(range(NUM_HONEST, NUM_PARTITIONS)),
                "rmc_duplicate": [RMC_DUPLICATE_DST],
            },
            "malicious_fraction": f"{NUM_MALICIOUS}/{NUM_PARTITIONS} = {100*NUM_MALICIOUS/NUM_PARTITIONS:.0f}%",
            "total_rows": sum(all_rows),
            "num_features": len(feature_cols),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reference": "Szelag et al., 'Reconnecting Malicious Client', arXiv:2504.03077v1",
        },
        "partitions": metadata,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    log(f"  Saved metadata.json")

    # Save features.json (same feature set as original)
    features_info = {
        "features": feature_cols,
        "num_features": len(feature_cols),
    }
    feat_path = output_dir / "features.json"
    with open(feat_path, "w") as f:
        json.dump(features_info, f, indent=2)
    log(f"  Saved features.json ({len(feature_cols)} features)")

    # =========================================================================
    # Summary
    # =========================================================================
    log("")
    log("=" * 70)
    log("REPARTITIONING COMPLETE")
    log("=" * 70)
    log(f"  Output directory: {output_dir}")
    log(f"  Client parquets:  client_0.parquet through client_{RMC_DUPLICATE_DST}.parquet")
    log(f"  Total files:      {NUM_PARTITIONS + 1} parquets + metadata.json + features.json")
    log(f"  Total rows:       {sum(all_rows):,}")
    log(f"  Features:         {len(feature_cols)}")
    log(f"  Honest clients:   0-{NUM_HONEST - 1} ({NUM_HONEST} clients)")
    log(f"  Malicious:        {NUM_HONEST}-{NUM_PARTITIONS - 1} ({NUM_MALICIOUS} clients)")
    log(f"  RMC duplicate:    client_{RMC_DUPLICATE_DST} (copy of client_{RMC_DUPLICATE_SRC})")
    log(f"  Alpha:            {alpha}")
    log(f"  Seed:             {seed}")
    log("")
    log("Next steps:")
    log("  1. Verify: conda run -n flowerfl python scripts/repartition_edge_20.py --dry-run")
    log('  2. Run experiment: flwr run . rmc-20-local --run-config \'dataset="edge_full_20_rmc" scenario="rmc/scenarios/rmc_szelag_20.json"\'')
    log("")


if __name__ == "__main__":
    main()

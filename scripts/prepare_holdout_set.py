"""
Phase 1 P1.1 — Build a fixed server-side holdout manifest (v23 rule 3.3).

Produces:
    data/holdout_manifest_edge_full_20.json  — provenance + per-partition row indices

The holdout is used as the canonical fixed evaluation set for ALL Phase 2+
experiments. Built once, version-locked, never changed between compared
methods.

Sampling policy:
    - Target 40,000 total rows (stratified by label across partitions)
    - ~2,000 rows per partition (21 partitions for edge_full_20_rmc)
    - Stratification: sample proportionally from each label class within each
      partition, so label distribution of the holdout mirrors the global
      label distribution of the source data.
    - Deterministic: sampling seed is fixed (20260416), so the manifest is
      exactly reproducible by re-running this script.

Usage:
    conda run -n flowerfl python scripts/prepare_holdout_set.py \
        --dataset edge_full_20_rmc \
        --samples-per-partition 2000 \
        --seed 20260416
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.task import DATASET_CONFIGS  # noqa: E402


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str:
    """sha256 of file contents (first 1 MB, enough to detect drift)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_000_000))
    return h.hexdigest()[:16]


def stratified_sample(
    labels: np.ndarray, n_total: int, rng: np.random.RandomState
) -> np.ndarray:
    """Return indices of a stratified-by-label sample of size ~n_total.

    Proportional allocation: each label class contributes a fraction of
    n_total equal to its frequency in `labels`. Rounds to integer per class;
    any shortfall is absorbed by the majority class.
    """
    if n_total >= len(labels):
        return np.arange(len(labels))

    unique, counts = np.unique(labels, return_counts=True)
    freqs = counts / counts.sum()
    per_class = np.maximum(1, np.round(freqs * n_total).astype(int))

    # Cap per_class so the total doesn't exceed n_total
    if per_class.sum() > n_total:
        # trim from the majority class
        over = per_class.sum() - n_total
        per_class[np.argmax(counts)] -= over

    out = []
    for cls, k in zip(unique, per_class):
        class_idx = np.where(labels == cls)[0]
        k = min(k, len(class_idx))
        if k > 0:
            out.append(rng.choice(class_idx, size=k, replace=False))
    if not out:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(out)).astype(np.int64)


def build_manifest(dataset_name: str, samples_per_partition: int, seed: int) -> dict:
    if dataset_name not in DATASET_CONFIGS:
        raise SystemExit(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]
    data_dir = PROJECT_ROOT / cfg["data_dir"]
    label_col = cfg["label_column"]
    client_files = cfg["client_files"]

    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    rng = np.random.RandomState(seed)

    per_partition = {}
    total_rows_sampled = 0
    total_label_dist: dict[str, int] = {}

    for i, fname in enumerate(client_files):
        path = data_dir / fname
        if not path.exists():
            print(f"  [WARN] missing: {path}; skipping")
            continue

        df = pd.read_parquet(path)
        labels = df[label_col].values
        if labels.max() > 1:
            labels_binary = (labels > 0).astype(np.int64)
        else:
            labels_binary = labels.astype(np.int64)

        indices = stratified_sample(labels_binary, samples_per_partition, rng)

        # Compute sampled label distribution for provenance
        sampled_labels = labels_binary[indices]
        unique, counts = np.unique(sampled_labels, return_counts=True)
        label_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        for k, v in label_dist.items():
            total_label_dist[k] = total_label_dist.get(k, 0) + v

        per_partition[str(i)] = {
            "file": fname,
            "total_rows_source": int(len(df)),
            "rows_sampled": int(len(indices)),
            "label_dist": label_dist,
            "source_file_hash": file_hash(path),
            "indices": indices.tolist(),
        }
        total_rows_sampled += len(indices)
        print(
            f"  partition {i:2d} ({fname}): "
            f"sampled {len(indices)}/{len(df)} "
            f"(labels: {label_dist})"
        )

    manifest = {
        "_meta": {
            "description": (
                "Fixed server-side holdout manifest for stratified evaluation. "
                "Built once in Phase 1 P1.1; never changed between compared "
                "methods per v23 rule 3.3."
            ),
            "dataset": dataset_name,
            "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
            "label_column": label_col,
            "stratification": "by_label (proportional per partition)",
            "samples_per_partition_target": samples_per_partition,
            "sampling_seed": seed,
            "total_rows_sampled": total_rows_sampled,
            "total_label_dist": total_label_dist,
            "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git_commit": get_git_commit(),
            "phase": "Phase 1 P1.1",
            "v23_rule": "3.3 — Use a fixed server-side evaluation set",
        },
        "per_partition": per_partition,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="edge_full_20_rmc",
        help="dataset config name in flowerfl.task.DATASET_CONFIGS",
    )
    parser.add_argument(
        "--samples-per-partition",
        type=int,
        default=2000,
        help="target rows sampled per partition (stratified by label)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260416,
        help="sampling RNG seed (date-based, distinct from training seeds)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output manifest path (default: data/holdout_manifest_<dataset>.json)",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset, args.samples_per_partition, args.seed)

    output = args.output or str(
        PROJECT_ROOT / f"data/holdout_manifest_{args.dataset}.json"
    )
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("=" * 70)
    print(f"Holdout manifest written: {output}")
    print(
        f"  total rows sampled: {manifest['_meta']['total_rows_sampled']:,}"
    )
    print(f"  total label dist:   {manifest['_meta']['total_label_dist']}")
    print(f"  git commit:         {manifest['_meta']['git_commit']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

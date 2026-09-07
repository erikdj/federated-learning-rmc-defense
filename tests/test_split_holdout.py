"""Test the val/test holdout splitter (Phase 1 P1.1 amendment).

Validates that scripts/data/split_holdout.py produces a deterministic,
stratified split of the existing 42K holdout into ~30K val and ~12K test
without overlap and preserving per-partition label proportions.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from split_holdout import split_holdout_manifest  # noqa: E402

HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout_manifest_edge_full_20_rmc.json"


def test_holdout_manifest_exists():
    assert HOLDOUT_PATH.exists(), "Required input manifest missing"


def test_split_no_overlap_and_total_count():
    """val ∪ test = original holdout, no duplicates."""
    holdout = json.loads(HOLDOUT_PATH.read_text())
    manifest = split_holdout_manifest(holdout, val_frac=0.7, seed=20260515)

    val_pairs = set()
    test_pairs = set()
    for pid, payload in manifest["per_partition"].items():
        for idx in payload["val_indices"]:
            val_pairs.add((pid, idx))
        for idx in payload["test_indices"]:
            test_pairs.add((pid, idx))

    assert val_pairs.isdisjoint(test_pairs), "val and test must not overlap"

    original_pairs = set()
    for pid, payload in holdout["per_partition"].items():
        for idx in payload["indices"]:
            original_pairs.add((pid, idx))

    assert val_pairs | test_pairs == original_pairs, "Union must equal original holdout"


def test_stratification_within_partition():
    """Per (partition, label) stratum: val fraction within ±2% of target."""
    holdout = json.loads(HOLDOUT_PATH.read_text())
    manifest = split_holdout_manifest(holdout, val_frac=0.7, seed=20260515)

    for pid, payload in manifest["per_partition"].items():
        for label_str, counts in payload["per_label_counts"].items():
            total = counts["val"] + counts["test"]
            if total < 10:
                # Tiny strata can deviate
                continue
            val_frac = counts["val"] / total
            assert 0.68 <= val_frac <= 0.72, (
                f"partition {pid} label {label_str}: "
                f"val_frac={val_frac:.3f} outside [0.68, 0.72]"
            )


def test_deterministic_given_seed():
    """Same seed → identical splits."""
    holdout = json.loads(HOLDOUT_PATH.read_text())
    m1 = split_holdout_manifest(holdout, val_frac=0.7, seed=20260515)
    m2 = split_holdout_manifest(holdout, val_frac=0.7, seed=20260515)
    assert m1 == m2, "Splitter must be deterministic given seed"


def test_metadata_present():
    """Output has required metadata for reproducibility."""
    holdout = json.loads(HOLDOUT_PATH.read_text())
    manifest = split_holdout_manifest(holdout, val_frac=0.7, seed=20260515)
    meta = manifest["_meta"]
    required = {"description", "val_frac", "seed", "source_manifest", "created", "stratification"}
    assert required.issubset(set(meta.keys())), f"Missing metadata keys: {required - set(meta.keys())}"

"""Sealed-test evaluator — the H4 primary endpoint's evaluation population.

Design authority (erratum-A E4, methodology v1.52):
    H4 units evaluate the global model on EXACTLY the test indices of
    `data/val_test_split_manifest.json` — the val/test split of the 42K
    server-side holdout locked 2026-05-14 ("never re-cut", METHODOLOGY
    pre-reg rule 4). The test portion (the 30% complement of val_frac 0.7)
    is the split the base registration sealed for H4's accuracy endpoint.
    Unit custody exports `eval_split=sealed_test` +
    `eval_split_manifest_sha256`; the H4 scorer refuses units without them.

The legacy evaluator (`rmc/fixed_eval.py::FixedEvalManager`) samples a
generic 2,000-row/client holdout with an RNG (seed 42). This manager does
NO sampling and consumes NO seed: the manifest's per-partition
`test_indices` are parquet row positions, taken verbatim, in manifest
order. Determinism is structural, not seeded.

Selected via run-config `eval-split=sealed_test`
(`flowerfl/server_app.py::_create_eval_manager`); the legacy path is
byte-unchanged for every non-H4 config.

Refusal policy: a missing, malformed, or internally inconsistent manifest
(val/test overlap, empty test set, duplicate or out-of-range indices,
missing parquet) raises `SealedTestManifestError` and stops the run —
never a silent fallback to the legacy sampler.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rmc.fixed_eval import FixedEvalManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "val_test_split_manifest.json"


class SealedTestManifestError(RuntimeError):
    """The sealed val/test split manifest is missing or malformed. Refuse."""


def load_split_manifest(path: "Path | str") -> Tuple[Dict[str, Any], str]:
    """Load `val_test_split_manifest.json`; return (manifest, sha256-of-bytes).

    The sha256 is over the FILE BYTES — the custody pin exported as
    `eval_split_manifest_sha256` so the scorer can gate on the exact split.
    """
    path = Path(path)
    if not path.is_file():
        raise SealedTestManifestError(
            f"sealed val/test split manifest missing: {path} — the H4 "
            f"primary endpoint cannot be evaluated without the locked split "
            f"(erratum-A E4)."
        )
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise SealedTestManifestError(
            f"sealed split manifest is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("per_partition"), dict
    ) or not manifest["per_partition"]:
        raise SealedTestManifestError(
            f"sealed split manifest must be an object with a non-empty "
            f"'per_partition' map: {path}"
        )
    return manifest, sha


def sealed_test_selection(
    manifest: Dict[str, Any],
) -> List[Tuple[int, str, List[int]]]:
    """Validated per-partition (partition, file, test_indices) selection.

    Ordered by integer partition key. Validates each entry: `file` is a
    non-empty string; `test_indices` is a non-empty list of unique,
    non-negative ints; `val_indices` (when present) is disjoint from
    `test_indices` — an overlap means the locked split is corrupt.
    """
    selection: List[Tuple[int, str, List[int]]] = []
    per_partition = manifest["per_partition"]
    try:
        keys = sorted(per_partition, key=int)
    except (TypeError, ValueError) as exc:
        raise SealedTestManifestError(
            f"per_partition keys must be integer-like, got "
            f"{sorted(map(str, per_partition))}"
        ) from exc
    for key in keys:
        entry = per_partition[key]
        if not isinstance(entry, dict):
            raise SealedTestManifestError(
                f"per_partition[{key!r}] must be an object, got "
                f"{type(entry).__name__}"
            )
        fname = entry.get("file")
        if not isinstance(fname, str) or not fname:
            raise SealedTestManifestError(
                f"per_partition[{key!r}].file must be a non-empty string, "
                f"got {fname!r}"
            )
        test_raw = entry.get("test_indices")
        if not isinstance(test_raw, list) or not test_raw:
            raise SealedTestManifestError(
                f"per_partition[{key!r}].test_indices must be a non-empty "
                f"list — an empty sealed test set is a corrupt split."
            )
        try:
            test_indices = [int(i) for i in test_raw]
        except (TypeError, ValueError) as exc:
            raise SealedTestManifestError(
                f"per_partition[{key!r}].test_indices must be integers"
            ) from exc
        if any(i < 0 for i in test_indices):
            raise SealedTestManifestError(
                f"per_partition[{key!r}].test_indices contains a negative "
                f"index"
            )
        if len(set(test_indices)) != len(test_indices):
            raise SealedTestManifestError(
                f"per_partition[{key!r}].test_indices contains duplicates"
            )
        val_raw = entry.get("val_indices")
        if isinstance(val_raw, list):
            overlap = set(test_indices) & {int(i) for i in val_raw}
            if overlap:
                raise SealedTestManifestError(
                    f"per_partition[{key!r}]: val/test overlap of "
                    f"{len(overlap)} indices — the locked split is corrupt "
                    f"and must not be scored."
                )
        selection.append((int(key), fname, test_indices))
    return selection


class SealedTestEvalManager(FixedEvalManager):
    """Server-side evaluation on exactly the manifest's sealed TEST indices.

    Subclasses `FixedEvalManager` ONLY for its `evaluate()` (model build +
    metric computation) — `FixedEvalManager.__init__` and `_build_holdout`
    are deliberately NOT called: this manager performs no sampling, consults
    no RNG, and excludes nothing; the manifest IS the selection.

    Z-score normalization over the assembled evaluation set mirrors the
    legacy manager's convention ("matching training pipeline"), so the two
    evaluators differ ONLY in which rows they score.
    """

    def __init__(
        self,
        dataset_name: str = "edge_full_20",
        manifest_path: "Path | str | None" = None,
        data_dir: "str | None" = None,
        label_column: "str | None" = None,
        batch_size: int = 256,
        input_shape: "int | None" = None,
    ):
        """
        Args:
            dataset_name: eval dataset config name (drives model construction
                and, when `data_dir`/`label_column` are not given, the parquet
                directory + label column).
            manifest_path: the locked split manifest (default
                `data/val_test_split_manifest.json`).
            data_dir / label_column / input_shape: explicit overrides for
                tests with synthetic parquet fixtures; production callers
                leave them None and the dataset config governs.
        """
        # NOTE: FixedEvalManager.__init__ intentionally NOT called (see class
        # docstring). Set the attributes evaluate()/provenance need directly.
        self.dataset_name = dataset_name
        self.batch_size = int(batch_size)
        self.eval_split = "sealed_test"
        self.holdout_disjoint = False  # no train-row exclusion is performed;
        # disjointness-from-training is not a property this sealed split
        # claims — `eval_split` is the identity field (erratum-A E4).
        self.rows_excluded = 0
        self.holdout_indices: Dict[int, np.ndarray] = {}
        self.holdout_size = 0
        self.per_class_counts: Dict[int, int] = {}
        self._testloader = None

        manifest_path = (
            Path(manifest_path) if manifest_path is not None
            else DEFAULT_MANIFEST_PATH
        )
        self.manifest_path = manifest_path
        manifest, sha = load_split_manifest(manifest_path)
        self.manifest_sha256 = sha
        selection = sealed_test_selection(manifest)

        if data_dir is None or label_column is None:
            from flowerfl.task import get_dataset_config

            config = get_dataset_config(dataset_name)
            data_dir = data_dir if data_dir is not None else config["data_dir"]
            label_column = (
                label_column if label_column is not None
                else config["label_column"]
            )
        self.data_dir = str(data_dir)
        self.label_column = str(label_column)

        if input_shape is not None:
            self.input_shape = int(input_shape)
        else:
            from flowerfl.task import detect_input_shape

            self.input_shape = detect_input_shape(dataset_name)

        self._build_sealed_holdout(selection)

    def _build_sealed_holdout(
        self, selection: List[Tuple[int, str, List[int]]]
    ) -> None:
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        all_features: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for partition, fname, test_indices in selection:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                raise SealedTestManifestError(
                    f"sealed split names {fname} for partition {partition} "
                    f"but {path} does not exist — refusing to evaluate on a "
                    f"partial sealed test set."
                )
            df = pd.read_parquet(path)
            indices = np.asarray(test_indices, dtype=np.int64)
            if indices.max() >= len(df):
                raise SealedTestManifestError(
                    f"partition {partition}: test index {int(indices.max())} "
                    f"is out of range for {fname} ({len(df)} rows) — the "
                    f"manifest does not describe this file."
                )
            labels = df[self.label_column].values.copy()
            features = df.drop(columns=[self.label_column]).values.astype(
                np.float32
            )
            # Binary mapping — identical to FixedEvalManager._build_holdout.
            if labels.max() > 1:
                labels = (labels > 0).astype(np.int64)
            else:
                labels = labels.astype(np.int64)
            self.holdout_indices[partition] = indices
            all_features.append(features[indices])
            all_labels.append(labels[indices])

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # Z-score normalization — same construction as the legacy manager.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        self._testloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        total = len(y)
        pos = int(y.sum())
        self.holdout_size = int(total)
        self.per_class_counts = {0: int(total - pos), 1: int(pos)}
        print(
            f"[SealedTestEval] Built SEALED-TEST eval set: {total} rows over "
            f"{len(selection)} partitions ({pos} positive, {total - pos} "
            f"negative) from {self.manifest_path.name} "
            f"sha256={self.manifest_sha256[:12]} (no sampling, no seed)"
        )

    def holdout_provenance(self) -> dict:
        """Durable provenance: legacy-shaped fields plus the E4 custody pair
        (`eval_split`, `eval_split_manifest_sha256`) the H4 scorer gates on."""
        return {
            "holdout_disjoint": bool(self.holdout_disjoint),
            "holdout_rows_excluded": int(self.rows_excluded),
            "holdout_size": int(self.holdout_size),
            "holdout_pos": int(self.per_class_counts.get(1, 0)),
            "holdout_neg": int(self.per_class_counts.get(0, 0)),
            "eval_split": self.eval_split,
            "eval_split_manifest_sha256": self.manifest_sha256,
            "eval_split_manifest_path": str(self.manifest_path),
        }

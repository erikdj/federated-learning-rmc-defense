"""Fixed evaluation manager for RMC POC experiments.

Builds a global holdout test set from stratified samples across all clients,
applying the same Z-score normalization used in training. Evaluates the
global model after each round on this fixed set for consistent metrics.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowerfl.task import Net, get_dataset_config, detect_input_shape, create_model, is_brfss_dataset


class FixedEvalManager:
    """Manages a fixed global holdout set for consistent evaluation."""

    def __init__(
        self,
        dataset_name: str = "edge_full",
        samples_per_client: int = 2000,
        batch_size: int = 256,
        seed: int = 42,
        disjoint: bool = False,
        train_max_samples: "int | None" = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        train_dataset_name: "str | None" = None,
    ):
        """Server-side fixed global holdout.

        disjoint (GWU-61): exclude each eval file's corresponding
        training-partition TRAIN row indices from the sampling pool before the
        stratified draw, so the reported holdout accuracy is a generalization
        number rather than one optimistically inflated by 77.5% train/holdout
        row overlap (audit results/20260727/holdout_overlap_audit/). Legacy
        behaviour (the pre-v8 overlapping holdout, for reproducing prior-run
        numbers) is `disjoint=False`, which is byte-identical to master.

        CLASS DEFAULT IS False (semantics-safe). Disjointness
        is only truthful when the training rows this excludes were actually
        produced by flowerfl/task.py::load_data — the reconstruction here mirrors
        load_data's cap + fixed-seed random_split exactly. The FLEET path
        (flowerfl/server_app.py::_create_eval_manager) trains through load_data,
        so IT passes disjoint=True by default from the run-config flag — fleet
        result JSONs still record holdout_disjoint=true by default. Other callers
        whose training uses DIFFERENT index semantics (e.g. rmc/simulate.py's
        load_client_data: 50k cap, RandomState, first-80%) must NOT flip this on,
        or they would falsely LABEL an overlapping holdout as disjoint — worse
        than honestly overlapping. They keep the safe default.

        train_max_samples: the per-client cap load_data used for the run whose
        training rows must be excluded. None -> flowerfl.task.MAX_SAMPLES_PER_CLIENT
        (the module default), which is what load_data falls back to when a run
        sets no max-samples. Only consulted when disjoint=True.

        DATASET MAPPING SUBTLETY (edge_full_20 / edge_full_20_rmc). Eval reads
        edge_full_20 (client_0..19). Training uses edge_full_20_rmc whose
        partition p reads client_p.parquet (p % 21); its extra partition
        client_20 is a BYTE-IDENTICAL duplicate of client_19 (md5 verified). The
        holdout never samples client_20.parquet, and because client_20 is the
        same data with the same row count, load_data's fixed-seed random_split
        assigns it the SAME train row positions as partition 19 — so excluding
        partition 19's train rows from the client_19 holdout pool already covers
        the client_20 duplicate. Per-eval-file exclusion of the matching-index
        training partition is therefore both sufficient and complete.
        """
        self.dataset_name = dataset_name
        self.samples_per_client = samples_per_client
        self.batch_size = batch_size
        self.seed = seed
        self.disjoint = bool(disjoint)
        if train_max_samples is None:
            from flowerfl.task import MAX_SAMPLES_PER_CLIENT as _default_cap
            train_max_samples = _default_cap
        self.train_max_samples = int(train_max_samples)
        self.train_split = float(train_split)
        self.val_split = float(val_split)
        # The TRAINING dataset name drives the is_full_dataset cap gate in the
        # reconstruction (P1-4). Server maps train `_rmc` -> eval base for
        # dataset_name, but training genuinely ran on the `_rmc` name; the fleet
        # passes it explicitly. Fall back to the eval name (membership in
        # _FULL_DATASETS is preserved under the `_rmc` mapping, so this is safe
        # for the canonical datasets even without the explicit pass-through).
        self.train_dataset_name = train_dataset_name or dataset_name
        self.input_shape = detect_input_shape(dataset_name)

        # provenance / test surfaces
        self.holdout_disjoint = self.disjoint
        self.rows_excluded = 0
        self.holdout_indices: dict[int, np.ndarray] = {}
        self.holdout_size = 0
        self.per_class_counts: dict[int, int] = {}

        self._testloader = None
        self._build_holdout()

    def _backing_train_partitions(self, eval_fname):
        """Training partitions whose TRAIN rows land in this eval file's row space.

        Derived from the TRAINING dataset config (NOT a filename heuristic):
          - the DIRECT backer is the training partition reading the same filename;
          - DUPLICATE backers come from the config's explicit ``duplicate_partitions``
            map ({dup_ordinal: source_ordinal}) on the _rmc entries, where the dup
            partition is a byte-identical copy of its source (md5-verified in the
            Stage-C record), so row positions correspond 1:1.

        Why the union matters: at a cap below file size
        the source and duplicate partitions draw DIFFERENT capped subsets
        (random_state 42+src vs 42+dup), so the duplicate's train rows are NOT
        covered by excluding only the source's. Each backer is reconstructed with
        ITS OWN ordinal seed and cap decision, then the positions are unioned.
        Non-_rmc datasets declare no duplicates -> direct backer only (unchanged).

        Returns a list of (ordinal, train_fname).
        """
        train_cfg = get_dataset_config(self.train_dataset_name)
        train_files = train_cfg["client_files"]
        dup_map = {int(k): int(v)
                   for k, v in train_cfg.get("duplicate_partitions", {}).items()}
        backers = [(j, tf) for j, tf in enumerate(train_files) if tf == eval_fname]
        direct_ordinals = {j for j, _ in backers}
        for dup_ord, src_ord in dup_map.items():
            if 0 <= src_ord < len(train_files) and train_files[src_ord] == eval_fname:
                if dup_ord not in direct_ordinals and 0 <= dup_ord < len(train_files):
                    backers.append((dup_ord, train_files[dup_ord]))
        return backers

    def _reconstruct_train_positions(self, data_dir, label_col, ordinal, fname):
        """Original-parquet-row positions load_data routes into this partition's
        TRAIN split. Reuses the fidelity-checked reconstruction from
        scripts/analysis/audit_holdout_overlap.py (do NOT re-derive).

        ``ordinal`` is the partition's position in client_files (== load_data's
        cap random_state offset); ``fname`` is the actual parquet filename to
        read — never a numeric id parsed out of the filename (P1-3)."""
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "scripts", "analysis", "audit_holdout_overlap.py")
        spec = importlib.util.spec_from_file_location("audit_holdout_overlap", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res, _meta = mod.reconstruct_train_indices(
            ordinal, data_dir, label_col,
            max_samples=self.train_max_samples,
            train_split=self.train_split,
            val_split=self.val_split,
            fname=fname,
            dataset_name=self.train_dataset_name,
        )
        return res["train"]

    def _build_holdout(self):
        """Build global holdout from stratified samples per client.

        When self.disjoint, each eval file's corresponding training-partition
        TRAIN rows are removed from the per-class sampling pool before the draw
        (same spc, same shared-RandomState seed path -> deterministic).
        """
        config = get_dataset_config(self.dataset_name)
        data_dir = config["data_dir"]
        label_col = config["label_column"]
        client_files = config["client_files"]

        rng = np.random.RandomState(self.seed)
        all_features = []
        all_labels = []

        for ordinal, fname in enumerate(client_files):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                print(f"[FixedEval] WARNING: {path} not found, skipping")
                continue

            df = pd.read_parquet(path)

            # Extract labels
            labels = df[label_col].values.copy()
            features = df.drop(columns=[label_col]).values.astype(np.float32)

            # Convert multi-class to binary if needed
            if labels.max() > 1:
                labels = (labels > 0).astype(np.int64)
            else:
                labels = labels.astype(np.int64)

            # Partition index is the ORDINAL position in client_files (matches
            # load_data's client_idx = partition_id % num_clients and its cap
            # random_state), NOT a numeric id parsed from the filename — the
            # latter breaks on IP-style names like client_192_168_0_101.parquet
            # (P1-3). Keyed by ordinal throughout.
            client_key = ordinal

            # Disjoint mode: exclude the UNION of TRAIN row positions over EVERY
            # training partition backing this eval file — the direct partition
            # plus any byte-duplicate partition (e.g. client_20 backing client_19)
            # whose capped draw differs at sub-file caps (GWU-61). Each backer is reconstructed from the TRAINING config's data_dir
            # with its own ordinal seed. Legacy mode: empty set -> byte-identical
            # to master.
            exclude = set()
            if self.disjoint:
                train_cfg = get_dataset_config(self.train_dataset_name)
                train_data_dir = train_cfg["data_dir"]
                for back_ordinal, back_fname in self._backing_train_partitions(fname):
                    exclude |= self._reconstruct_train_positions(
                        train_data_dir, label_col, back_ordinal, back_fname)
                self.rows_excluded += len(exclude)

            # Stratified sample
            n = min(self.samples_per_client, len(df))
            if n < len(df):
                # Stratified: sample proportionally from each class. The per-class
                # target n_sample uses the FULL label counts (so the holdout's
                # class balance and size match legacy), but the DRAW is from the
                # disjoint pool.
                unique_labels = np.unique(labels)
                indices = []
                for lbl in unique_labels:
                    lbl_indices = np.where(labels == lbl)[0]
                    proportion = len(lbl_indices) / len(labels)
                    n_sample = max(1, int(n * proportion))
                    if n_sample > len(lbl_indices):
                        n_sample = len(lbl_indices)
                    pool = lbl_indices
                    if exclude:
                        pool = lbl_indices[~np.isin(lbl_indices, list(exclude))]
                    if len(pool) == 0:
                        raise RuntimeError(
                            f"[FixedEval] disjoint holdout: client_{client_key} "
                            f"class {int(lbl)} has NO rows left after excluding "
                            f"{len(exclude)} train rows — cannot sample."
                        )
                    if len(pool) < n_sample:
                        # LOUD, never a silent shrink below the stratified target.
                        print(
                            f"[FixedEval] WARNING disjoint holdout shortfall: "
                            f"client_{client_key} class {int(lbl)} requested "
                            f"{n_sample} but only {len(pool)} disjoint rows "
                            f"available; clamping to {len(pool)}."
                        )
                        n_sample = len(pool)
                    sampled = rng.choice(pool, size=n_sample, replace=False)
                    indices.extend(sampled)
                indices = np.array(indices)
            elif not exclude:
                # take-all, legacy/no-exclusion: byte-identical to master
                # (original parquet row order preserved).
                indices = np.arange(len(df))
            else:
                # take-all UNDER disjoint exclusion (P2-3): apply the SAME
                # per-class guards as the stratified branch so a small/rare class
                # can't silently vanish from the holdout while still recording
                # holdout_disjoint=true. Here the effective per-class target is
                # the FULL class count (take-all), so any exclusion is a
                # shortfall and a fully-excluded class is a hard error.
                unique_labels = np.unique(labels)
                indices = []
                for lbl in unique_labels:
                    lbl_indices = np.where(labels == lbl)[0]
                    pool = lbl_indices[~np.isin(lbl_indices, list(exclude))]
                    if len(pool) == 0:
                        raise RuntimeError(
                            f"[FixedEval] disjoint holdout: client_{client_key} "
                            f"class {int(lbl)} has NO rows left after excluding "
                            f"{len(exclude)} train rows — cannot sample."
                        )
                    if len(pool) < len(lbl_indices):
                        # LOUD, never a silent shrink.
                        print(
                            f"[FixedEval] WARNING disjoint holdout shortfall: "
                            f"client_{client_key} class {int(lbl)} take-all "
                            f"reduced from {len(lbl_indices)} to {len(pool)} rows "
                            f"after excluding train rows."
                        )
                    indices.extend(pool)
                indices = np.array(indices, dtype=np.int64)

            # in-code disjointness assertion (guard, not just by construction)
            if self.disjoint and exclude:
                assert set(int(i) for i in indices).isdisjoint(exclude), (
                    f"[FixedEval] disjoint invariant violated for client_{client_key}"
                )

            self.holdout_indices[client_key] = np.asarray(indices, dtype=np.int64)
            all_features.append(features[indices])
            all_labels.append(labels[indices])

        if not all_features:
            raise RuntimeError(f"No data loaded for {self.dataset_name}")

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # Z-score normalization (matching training pipeline)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        # Build DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        self._testloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total = len(y)
        pos = int(y.sum())
        self.holdout_size = int(total)
        self.per_class_counts = {0: int(total - pos), 1: int(pos)}
        print(f"[FixedEval] Built holdout: {total} samples ({pos} positive, {total - pos} negative)")
        # one-line provenance record (GWU-61): disjoint flag + rows excluded, so
        # a reader can tell a disjoint holdout from a legacy overlapping one.
        print(f"[FixedEval] holdout_disjoint={'true' if self.disjoint else 'false'} "
              f"size={total} pos={pos} neg={total - pos} "
              f"rows_excluded={self.rows_excluded}")

    def holdout_provenance(self) -> dict:
        """Durable holdout provenance for the result-JSON provenance dict.

        The one-line ``[FixedEval] holdout_disjoint=...`` stdout record is a
        BONUS — like the driver-prewarm health, actor/driver stdout emitted
        during a captured-stdout section never reliably reaches CloudWatch, so it
        was invisible on EXP-019/021 despite the disjoint construction running.
        This method is the artifact-of-record: the runner copies it into the
        result JSON next to ``holdout_disjoint``. ``holdout_rows_excluded`` is
        >0 only under disjoint mode (legacy/overlapping mode excludes nothing).
        """
        return {
            "holdout_disjoint": bool(self.holdout_disjoint),
            "holdout_rows_excluded": int(self.rows_excluded),
            "holdout_size": int(self.holdout_size),
            "holdout_pos": int(self.per_class_counts.get(1, 0)),
            "holdout_neg": int(self.per_class_counts.get(0, 0)),
        }

    def evaluate(self, weights) -> dict:
        """Evaluate global model weights on the fixed holdout set.

        Args:
            weights: List of numpy arrays (global model parameters).

        Returns:
            Dict with keys: loss, accuracy, precision, recall, f1 (macro), plus
            per-class benign(0)/attack(1) precision/recall/f1 (Stage-F; append-only, with historical macro-key semantics).
        """
        from flowerfl.task import set_weights, test_detailed, test_brfss_detailed

        net = create_model(self.dataset_name, self.input_shape)
        set_weights(net, weights)

        if is_brfss_dataset(self.dataset_name):
            m = test_brfss_detailed(net, self._testloader)
        else:
            m = test_detailed(net, self._testloader)

        return {
            "loss": float(m["loss"]),
            "accuracy": float(m["accuracy"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "attack_precision": float(m["attack_precision"]),
            "attack_recall": float(m["attack_recall"]),
            "attack_f1": float(m["attack_f1"]),
            "benign_precision": float(m["benign_precision"]),
            "benign_recall": float(m["benign_recall"]),
            "benign_f1": float(m["benign_f1"]),
        }

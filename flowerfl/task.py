"""
Task Module - Model, Training, and Data Loading for FlowerFL.

This module is the shared client-side and server-side-eval code path for
every RMC experiment in the praxis (Chapter 3, hypotheses H1-H4): it is
imported by flowerfl/client_app.py and flowerfl/server_app.py (the Flower
ClientApp/ServerApp), by rmc/fixed_eval.py (the server's held-out evaluator,
FixedEvalManager), and by non-Flower reproduction scripts (e.g.
the original baseline reproduction) so the same model/train/test code is
exercised both inside and outside the Flower simulation harness.

Contains:
- Net: the standard feedforward classifier used for the Edge-IIoT /
  CIC-IoT2023 datasets across all RMC scenarios. Its per-exec-mode
  hyperparameters (learning rate, local epochs, batch size) are locked in
  data/hparams_locked.json — see that file for the frozen values per mode
  (flower_reset vs persistent_optimizer) and the grid-search provenance
  behind the lock. Both modes share this same architecture; only lr differs.
- SzelagNet: exact-reproduction architecture for the BRFSS anchor experiment
  (Szelag et al., arXiv:2504.03077v1) — used only for the "brfss_*" dataset
  configs below, not for the Edge-IIoT/CIC-IoT2023 RMC scenario suite.
- train/train_brfss (+ train_label_flip/train_brfss_label_flip and
  add_parameter_noise attack-simulation variants) and test/test_brfss:
  local-training and evaluation loops invoked once per FL round, either by
  a simulated client (attack variants for malicious clients) or by the
  server's held-out evaluator.
- DATASET_CONFIGS: per-dataset partition layout (client parquet files,
  label column, malicious-client ordering) for every dataset variant used
  across the praxis, including the "_rmc" variants that add one duplicate
  client partition to support RMC identity-reset scenarios (see comment on
  DATASET_CONFIGS below).
- Data loading utilities (load_data, generate_synthetic_data) shared by both
  Flower simulation clients and the non-Flower reproduction scripts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
import hashlib
import tempfile
import time
from pathlib import Path

from flowerfl.update_matching import run_matched_steps

try:
    import fcntl  # POSIX-only; the fleet/dev hosts are Linux
except ImportError:  # pragma: no cover — non-POSIX: interprocess lock unavailable
    fcntl = None


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================
#
# Each entry describes one dataset partition layout: which parquet files
# back each client, the label column name, and `malicious_order` — the
# ordering used by get_malicious_clients below to pick the first N clients
# as malicious for a given malicious_fraction (lower-slot convention, also
# used by scripts/data/generate_scenarios.py's Design D scenario generators
# for S0-S4, which hardcode client_0..client_8 as the 9 adversaries).
#
# The "_rmc" variants (edge_full_rmc, cic_full_rmc, edge_full_20_rmc,
# brfss_20_rmc) each add ONE extra client partition beyond the base dataset
# (e.g. edge_full has 10 clients / edge_full_rmc has 11) whose parquet file
# is a duplicate of the last base client. This extra partition supplies the
# data used by RMC identity-reset "_new" logical identities (see
# scripts/data/generate_scenarios.py) — the same physical data, reassigned
# to a new logical client_id, so a reconnecting malicious client is modeled
# as "same device, new identity" rather than fabricating a data source.
DATASET_CONFIGS = {
    # --- Original POC datasets (small subsets) ---
    "edge": {
        "data_dir": "data/edge",
        "label_column": "attack_label",
        "client_files": [
            "client_0.parquet",
            "client_192_168_0_101.parquet",
            "client_192_168_0_128.parquet",
            "client_192_168_0_170.parquet",
        ],
        "client_ids": ["0", "192_168_0_101", "192_168_0_128", "192_168_0_170"],
        "num_classes": 2,
        "description": "Edge-IIoT dataset partitioned by source IP",
        "malicious_order": [3, 2, 0, 1],
    },
    "cic": {
        "data_dir": "data/cic",
        "label_column": "label",
        "client_files": [f"client_{i}.parquet" for i in range(10)],
        "client_ids": [str(i) for i in range(10)],
        "num_classes": 2,
        "description": "CIC-IoT2023 dataset with Dirichlet partitioning",
        "malicious_order": list(range(10)),
    },
    # --- Full datasets ---
    "edge_full": {
        "data_dir": "data/edge_full",
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(10)],
        "client_ids": [str(i) for i in range(10)],
        "num_classes": 2,
        "description": "Full Edge-IIoT dataset, 10 sensor-type clients (~20.9M rows)",
        "malicious_order": list(range(10)),
    },
    "cic_full": {
        "data_dir": "data/cic_full",
        "label_column": "label",
        "client_files": [f"client_{i}.parquet" for i in range(15)],
        "client_ids": [str(i) for i in range(15)],
        "num_classes": 2,
        "description": "Full CIC-IoT2023 dataset, 15 Dirichlet clients (~46.7M rows)",
        "malicious_order": list(range(15)),
    },
    "edge_full_enc": {
        "data_dir": "data/edge_full_enc",
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(10)],
        "client_ids": [str(i) for i in range(10)],
        "num_classes": 2,
        "description": "Full Edge-IIoT with label-encoded string features, 10 clients, 58 features",
        "malicious_order": list(range(10)),
    },
    # --- RMC scenario dataset (11 clients: 0-9 + duplicated partition 10) ---
    "edge_full_rmc": {
        "data_dir": "data/edge_full",
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(11)],
        "client_ids": [str(i) for i in range(11)],
        "num_classes": 2,
        "description": "Edge-IIoT for RMC scenarios: 11 clients (client_10 = duplicate of client_9 for identity reset)",
        "malicious_order": list(range(11)),
        # {duplicate_ordinal: source_ordinal} — client_10 is a byte-identical copy
        # of client_9. Used by the disjoint holdout to exclude BOTH partitions'
        # train rows from client_9's eval pool.
        "duplicate_partitions": {10: 9},
    },
    "cic_full_rmc": {
        "data_dir": "data/cic_full",
        "label_column": "label",
        "client_files": [f"client_{i}.parquet" for i in range(16)],
        "client_ids": [str(i) for i in range(16)],
        "num_classes": 2,
        "description": "CIC-IoT2023 for RMC scenarios: 16 clients (client_15 = duplicate of client_14 for identity reset)",
        "malicious_order": list(range(16)),
        "duplicate_partitions": {15: 14},  # client_15 == byte-dup of client_14
    },
    # --- 20-client Szelag setup (11 honest + 9 malicious, 45% malicious fraction) ---
    "edge_full_20": {
        "data_dir": "data/edge_full_20",
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(20)],
        "client_ids": [str(i) for i in range(20)],
        "num_classes": 2,
        "description": "Edge-IIoT repartitioned to 20 Dirichlet clients (Szelag et al. setup)",
        "malicious_order": list(range(20)),
    },
    "edge_full_20_rmc": {
        "data_dir": "data/edge_full_20",
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(21)],
        "client_ids": [str(i) for i in range(21)],
        "num_classes": 2,
        "description": "Edge-IIoT 20-client for RMC: 21 partitions (client_20 = duplicate of client_19 for identity reset)",
        "malicious_order": list(range(21)),
        "duplicate_partitions": {20: 19},  # client_20 == byte-dup of client_19 (md5-verified)
    },
    # --- BRFSS dataset (Szelag et al. reproduction) ---
    "brfss_20": {
        "data_dir": "data/brfss_20",
        "label_column": "Diabetes",
        "client_files": [f"client_{i}.parquet" for i in range(20)],
        "client_ids": [str(i) for i in range(20)],
        "num_classes": 1,  # Binary with BCELoss (Sigmoid output, not softmax)
        "description": "BRFSS dataset from Szelag et al., 20 clients (11 honest + 9 malicious)",
        "malicious_order": list(range(20)),
        "model": "szelag",  # Use SzelagNet instead of Net
    },
    "brfss_20_rmc": {
        "data_dir": "data/brfss_20",
        "label_column": "Diabetes",
        "client_files": [f"client_{i}.parquet" for i in range(21)],
        "client_ids": [str(i) for i in range(21)],
        "num_classes": 1,  # Binary with BCELoss (Sigmoid output, not softmax)
        "description": "BRFSS 20-client for RMC: 21 partitions (client_20 = duplicate of client_19)",
        "malicious_order": list(range(21)),
        "model": "szelag",  # Use SzelagNet instead of Net
        "duplicate_partitions": {20: 19},  # client_20 == byte-dup of client_19
    },
}

# Memoizes detect_input_shape results per dataset_name so repeated client
# instantiations within one process (e.g. 20 simulated clients) only read the
# parquet schema once.
_input_shape_cache = {}


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class Net(nn.Module):
    """
    Feedforward neural network for binary classification.
    Architecture: input -> 64 -> Dropout(0.3) -> 32 -> Dropout(0.3) -> 2

    Dropout=0.3 was added 2026-04-16 (Phase 2 P2.0) to align with the locked architecture in data/hparams_locked.json. Promotes generalization
    under non-IID FL where each client holds only a slice of data.

    This is the model architecture for every RMC scenario run (Edge-IIoT /
    CIC-IoT2023 datasets, all non-BRFSS entries in DATASET_CONFIGS). Its
    identity is fixed across both exec modes (flower_reset and
    persistent_optimizer) per data/hparams_locked.json — only the optimizer
    hyperparameters differ between modes, not this architecture. `input_shape`
    is dataset-dependent and resolved at runtime via detect_input_shape.
    """

    def __init__(self, input_shape: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class SzelagNet(nn.Module):
    """
    Exact reproduction of Szelag's NeuralNetwork(17).
    Architecture: 17 -> 8 -> 16 -> 8 -> 1 with Sigmoid activation and Dropout.
    Used for BRFSS dataset experiments to match Szelag et al. (arXiv:2504.03077v1).

    NOTE: Output is a single logit (no activation) — use BCEWithLogitsLoss for training.
    For inference, apply sigmoid and threshold at 0.5.
    """

    def __init__(self, input_shape: int = 17):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropoutMiddle = nn.Dropout(p=0.5)
        self.dropoutEnd = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.dropoutMiddle(x)
        x = self.sigmoid(self.fc2(x))
        x = self.dropoutMiddle(x)
        x = self.sigmoid(self.fc3(x))
        x = self.dropoutEnd(x)
        return self.fc4(x)


def is_brfss_dataset(dataset_name: str) -> bool:
    """Check if a dataset uses the Szelag/BRFSS model architecture."""
    config = DATASET_CONFIGS.get(dataset_name, {})
    return config.get("model") == "szelag"


# Datasets to which load_data applies the MAX_SAMPLES_PER_CLIENT stratified cap.
# SINGLE SOURCE OF TRUTH — shared with the disjoint-holdout train-index
# reconstruction (scripts/analysis/audit_holdout_overlap.py::reconstruct_train_indices)
# so the excluded rows can NEVER diverge from what load_data actually caps (the
# v8 integrity claim rides on that seam being exact).
_FULL_DATASETS = frozenset({
    "edge_full", "edge_full_rmc",
    "edge_full_20", "edge_full_20_rmc",
    "cic_full", "cic_full_rmc",
})


def is_full_dataset(dataset_name: str) -> bool:
    """Whether load_data stratified-caps this dataset at MAX_SAMPLES_PER_CLIENT.

    load_data only caps the large "_full"/"_rmc" partitions; small/enc/BRFSS
    datasets are used whole. The disjoint-holdout reconstruction MUST gate its
    cap on this exact predicate keyed on the TRAINING dataset name, or it would
    exclude the wrong rows and falsely label an overlapping holdout disjoint.
    """
    return dataset_name in _FULL_DATASETS


def create_model(dataset_name: str, input_shape: int = None):
    """Create the appropriate model for the given dataset.

    Returns Net for standard datasets, SzelagNet for BRFSS datasets.
    """
    if input_shape is None:
        input_shape = detect_input_shape(dataset_name)

    if is_brfss_dataset(dataset_name):
        return SzelagNet(input_shape)
    else:
        return Net(input_shape)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
#
# Two families of functions below, mirrored for Net (CrossEntropyLoss,
# *_flip suffix) and SzelagNet (BCEWithLogitsLoss, *_brfss suffix):
#   - train / train_brfss: the honest-client local-training loop, called
#     once per FL round for every non-attacking client.
#   - train_label_flip / train_brfss_label_flip: label-poisoning attack
#     simulation — trains on the SAME data but with labels flipped, used by
#     malicious clients configured for the "label_flip" attack type (see
#     scripts/data/generate_scenarios.py attack schedules).
#   - add_parameter_noise: a separate model-poisoning attack that perturbs
#     already-trained weights rather than the training signal.
#   - test / test_brfss: held-out evaluation, called by the server-side
#     evaluator (rmc/fixed_eval.py) after each round's aggregation.
# All four training variants report loss as the running total divided by
# the number of batches seen (max(num_batches, 1) guards the degenerate
# empty-loader case).

def train(net, trainloader, epochs: int = 1, lr: float = 0.01, weight_decay: float = 3e-3,
          optimizer: "torch.optim.Optimizer | None" = None,
          max_steps: "int | None" = None, partition_id: "int | None" = None,
          arm: "str | None" = None, metrics_out: "dict | None" = None):
    """Train the local model for `epochs` epochs.

    If `optimizer` is provided, it is used as-is (caller is responsible for its
    state lifecycle — see flowerfl/persistent_optimizer.py for the persistent
    mode integration). If `optimizer` is None, a fresh Adam is constructed with
    (lr, weight_decay) — this is the legacy Flower-native per-round reset
    behavior.

    Uses Adam(lr, weight_decay=3e-3) to align with the locked hyperparameters and Szelag's actual code. Pre-2026-04-16 used SGD+momentum without
    weight_decay; migrated in Phase 2 P2.0.

    Update-matching (Stage-F §4): when `max_steps` is set, the epochs-bounded
    loop is replaced by the generic steps-driven cap (flowerfl/update_matching.py)
    that runs EXACTLY `max_steps` optimizer steps by cycling the loader — the
    per-batch body is unchanged. `partition_id`/`arm` name the client in the loud
    empty-loader error; `metrics_out`, if given, receives `steps_taken`. When
    `max_steps` is None the original epochs loop runs byte-for-byte.
    """
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    if max_steps is not None:
        def _step(batch):
            features, labels = batch
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        total_loss, num_batches = run_matched_steps(
            trainloader, max_steps, step_fn=_step,
            partition_id=partition_id, arm=arm)
    else:
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for features, labels in trainloader:
                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

    if metrics_out is not None:
        metrics_out["steps_taken"] = num_batches
    return total_loss / max(num_batches, 1)


def train_label_flip(net, trainloader, lr: float = 0.01, weight_decay: float = 3e-3,
                     max_steps: "int | None" = None, partition_id: "int | None" = None,
                     arm: "str | None" = None, metrics_out: "dict | None" = None,
                     *, epochs: int = 1):
    """Train with flipped labels (label poisoning attack) for `epochs` epochs.

    Uses Adam(lr, weight_decay=3e-3) — same optimizer migration as train
    in Phase 2 P2.0.

    Honors `epochs`
    exactly like train. Historically this function ran a SINGLE natural pass
    while every other path ran `for _ in range(epochs)` with local_epochs=5 —
    label_flip attackers under-trained 5x. `epochs` is KEYWORD-ONLY and
    appended after the legacy parameters, so every pre-change positional slot
    is preserved exactly (`train_label_flip(net, loader, 0.001)` still means
    lr=0.001) and callers that pass nothing (e.g. rmc/attacks.py, where the
    honest arm also runs 1 epoch) keep the historical single pass via the
    default epochs=1; the fleet client passes epochs=local_epochs by keyword.
    Per-batch attack semantics (which labels flip, loss, optimizer stepping)
    are unchanged — only the epoch count changed.

    Update-matching (Stage-F §4): identical steps-driven cap as train; when
    `max_steps` is set the epochs loop is bypassed and EXACTLY `max_steps`
    optimizer steps run, regardless of `epochs`.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    if max_steps is not None:
        def _step(batch):
            features, labels = batch
            optimizer.zero_grad()
            flipped_labels = 1 - labels  # Flip: 0 -> 1, 1 -> 0
            outputs = net(features)
            loss = criterion(outputs, flipped_labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        total_loss, num_batches = run_matched_steps(
            trainloader, max_steps, step_fn=_step,
            partition_id=partition_id, arm=arm)
    else:
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for features, labels in trainloader:
                optimizer.zero_grad()
                flipped_labels = 1 - labels  # Flip: 0 -> 1, 1 -> 0
                outputs = net(features)
                loss = criterion(outputs, flipped_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

    if metrics_out is not None:
        metrics_out["steps_taken"] = num_batches
    return total_loss / max(num_batches, 1)


def add_parameter_noise(net, noise_scale: float = 0.1):
    """Add Gaussian noise to model parameters (model poisoning attack)."""
    with torch.no_grad():
        for param in net.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)


# Metric key groups for the eval-result dict (single source so callers and
# tests agree on the schema). MACRO keys keep their historical semantics; the
# PER_CLASS keys are append-only (Stage-F).
_MACRO_METRIC_KEYS = ("loss", "accuracy", "precision", "recall", "f1")
_PER_CLASS_METRIC_KEYS = (
    "benign_precision", "benign_recall", "benign_f1",
    "attack_precision", "attack_recall", "attack_f1",
)


def _binary_per_class_prf(all_labels, all_preds) -> dict:
    """Per-class precision/recall/F1 for the binary benign(0)/attack(1) task.

    Uses ``average=None`` with an explicit ``labels=[0, 1]`` so the returned
    arrays are positionally [benign, attack] even if a class is absent from
    the predictions or labels (the fixed holdout always carries both, but the
    explicit-labels guard keeps the schema stable). Complements — never
    replaces — the macro fields, so an attack-class recall/precision claim is
    auditable rather than inferred from a macro average.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "benign_f1": float(f1[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
    }


def test_detailed(net, testloader) -> dict:
    """Evaluate on the test set, returning macro AND per-class metrics.

    Returns a dict with the macro keys (loss, accuracy, precision, recall, f1)
    plus the per-class keys from ``_binary_per_class_prf``. The macro values
    are computed exactly as the legacy ``test`` did, so the macro trajectory
    numbers are byte-identical; ``test`` is a thin wrapper over this that
    extracts the 5-tuple its existing callers unpack.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = 0.0
    all_labels = []
    all_preds = []

    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            outputs = net(features)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    if total == 0:
        return {k: 0.0 for k in _MACRO_METRIC_KEYS + _PER_CLASS_METRIC_KEYS}

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    result = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    result.update(_binary_per_class_prf(all_labels, all_preds))
    return result


def test(net, testloader):
    """
    Evaluate the model on the test set.
    Returns: (loss, accuracy, precision, recall, f1)

    Thin wrapper over ``test_detailed`` — kept for callers that only need the
    macro 5-tuple (the honest per-client val path in client_app.py). The
    per-class fields are computed by ``test_detailed`` and discarded here.
    """
    d = test_detailed(net, testloader)
    return d["loss"], d["accuracy"], d["precision"], d["recall"], d["f1"]


def train_brfss(net, trainloader, epochs: int = 2, lr: float = 0.01, weight_decay: float = 3e-3,
                optimizer: "torch.optim.Optimizer | None" = None,
                max_steps: "int | None" = None, partition_id: "int | None" = None,
                arm: "str | None" = None, metrics_out: "dict | None" = None):
    """Train the SzelagNet model with BCEWithLogitsLoss and Adam optimizer.

    Matches Szelag's training setup with advisor-aligned weight_decay:
    - Adam optimizer with lr and weight_decay=3e-3 (advisor CH3 Table 3-2)
    - BCEWithLogitsLoss (logit output, no sigmoid in loss)
    - 2 local epochs
    - batch_size=32

    If `optimizer` is provided, it is used as-is (persistent-state mode);
    otherwise a fresh Adam(lr, weight_decay) is constructed each call
    (default Flower-reset mode).

    NOTE: In Flower-reset mode, the optimizer is NOT persistent across rounds
    (unlike Szelag's implementation). This means Adam's momentum/variance don't
    accumulate, which should result in HIGHER accuracy (the persistent
    optimizer is part of the "ticking time bomb" that degrades Krum in
    Szelag's dynamic mode). The persistent-state mode of the unified runner
    routes through flowerfl/persistent_optimizer.py to restore Szelag's
    cross-round Adam continuity.

    weight_decay added 2026-04-16 (Phase 2 P2.0). Will verify via P2.8
    that BRFSS Szelag anchor (0.504) still reproduces within tolerance.
    """
    criterion = nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    if max_steps is not None:
        def _step(batch):
            features, labels = batch
            labels_float = labels.float().view(-1, 1)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels_float)
            loss.backward()
            optimizer.step()
            return loss.item()

        total_loss, num_batches = run_matched_steps(
            trainloader, max_steps, step_fn=_step,
            partition_id=partition_id, arm=arm)
    else:
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for features, labels in trainloader:
                # Labels should be float for BCE
                labels_float = labels.float().view(-1, 1)
                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, labels_float)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

    if metrics_out is not None:
        metrics_out["steps_taken"] = num_batches
    return total_loss / max(num_batches, 1)


def train_brfss_label_flip(net, trainloader, lr: float = 0.01, weight_decay: float = 3e-3,
                           max_steps: "int | None" = None, partition_id: "int | None" = None,
                           arm: "str | None" = None, metrics_out: "dict | None" = None):
    """Train SzelagNet with flipped labels (label poisoning for BRFSS).

    Update-matching (Stage-F §4): identical steps-driven cap as train_brfss;
    when `max_steps` is None the original single-pass loop runs byte-for-byte.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    if max_steps is not None:
        def _step(batch):
            features, labels = batch
            labels_float = labels.float().view(-1, 1)
            flipped = 1.0 - labels_float  # Flip: 0 -> 1, 1 -> 0
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, flipped)
            loss.backward()
            optimizer.step()
            return loss.item()

        total_loss, num_batches = run_matched_steps(
            trainloader, max_steps, step_fn=_step,
            partition_id=partition_id, arm=arm)
    else:
        total_loss = 0.0
        num_batches = 0

        for features, labels in trainloader:
            labels_float = labels.float().view(-1, 1)
            flipped = 1.0 - labels_float  # Flip: 0 -> 1, 1 -> 0
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, flipped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    if metrics_out is not None:
        metrics_out["steps_taken"] = num_batches
    return total_loss / max(num_batches, 1)


def test_brfss_detailed(net, testloader) -> dict:
    """Evaluate SzelagNet on the test set, returning macro AND per-class BCE
    metrics. Macro values match the legacy ``test_brfss`` exactly; ``test_brfss``
    wraps this to keep its 5-tuple contract (Stage-F)."""
    criterion = nn.BCEWithLogitsLoss()
    correct, total = 0, 0
    total_loss = 0.0
    all_labels = []
    all_preds = []

    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            labels_float = labels.float().view(-1, 1)
            outputs = net(features)
            total_loss += criterion(outputs, labels_float).item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels.float()).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.view(-1).cpu().numpy().astype(int))

    if total == 0:
        return {k: 0.0 for k in _MACRO_METRIC_KEYS + _PER_CLASS_METRIC_KEYS}

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    result = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    result.update(_binary_per_class_prf(all_labels, all_preds))
    return result


def test_brfss(net, testloader):
    """
    Evaluate SzelagNet on test set with BCE metrics.
    Returns: (loss, accuracy, precision, recall, f1)

    Thin wrapper over ``test_brfss_detailed`` (see ``test`` for the rationale).
    """
    d = test_brfss_detailed(net, testloader)
    return d["loss"], d["accuracy"], d["precision"], d["recall"], d["f1"]


# ============================================================================
# PARAMETER UTILITIES
# ============================================================================

def get_weights(net):
    """Return model parameters as a list of NumPy arrays, in state_dict order.

    This is the (de)serialization boundary between PyTorch and Flower's wire
    format (flwr.common.Parameters, built via ndarrays_to_parameters). Called
    both by clients (to report a fit result) and by defense plugins in
    flowerfl/byzantine_defense.py (to flatten updates for distance scoring).
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Load model parameters from a list of NumPy arrays (inverse of get_weights).

    Reconstructs the OrderedDict by zipping against the model's own
    state_dict.keys, so `parameters` must be in the same order get_weights
    would have produced for this same model instance. strict=True ensures a
    shape/key mismatch (e.g. a stale checkpoint against a resized model)
    fails loudly rather than silently loading a partial state.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_dataset_config(dataset_name: str) -> dict:
    """Look up a DATASET_CONFIGS entry by name; raises on unknown dataset_name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'edge' or 'cic'.")
    return DATASET_CONFIGS[dataset_name]


def get_num_clients(dataset_name: str) -> int:
    """Number of client partitions for a dataset (len of its client_files list)."""
    config = get_dataset_config(dataset_name)
    return len(config["client_files"])


def detect_input_shape(dataset_name: str) -> int:
    """Determine the model's input feature count for a dataset.

    Reads the first client's parquet schema and subtracts 1 for the label
    column (input_shape = num_columns - 1). Result is memoized in
    _input_shape_cache since every client in a run shares the same schema.

    Falls back to a hardcoded per-dataset default (below) when the parquet
    files aren't present locally — e.g. when running unit tests or doing a
    dry run without the (gitignored) data/ directory checked out. These
    defaults must stay in sync with the actual schemas; they are last-resort
    scaffolding, not a substitute for real data during an experiment.
    """
    if dataset_name in _input_shape_cache:
        return _input_shape_cache[dataset_name]

    config = get_dataset_config(dataset_name)
    sample_file = os.path.join(config["data_dir"], config["client_files"][0])

    if os.path.exists(sample_file):
        df = pd.read_parquet(sample_file, columns=None)
        input_shape = len(df.columns) - 1
        _input_shape_cache[dataset_name] = input_shape
        print(f"[Dataset] Detected {input_shape} features for {dataset_name}")
        return input_shape
    else:
        defaults = {"edge": 59, "cic": 46, "edge_full": 45, "edge_full_rmc": 45, "edge_full_20": 45, "edge_full_20_rmc": 45, "cic_full": 46, "cic_full_rmc": 46, "edge_full_enc": 58, "brfss_20": 17, "brfss_20_rmc": 17}
        print(f"[Dataset] Using default input shape {defaults.get(dataset_name, 46)}")
        return defaults.get(dataset_name, 46)



# Default cap on rows sampled per client for the large "_full"/"_rmc" dataset
# variants (stratified downsampling in load_data below); keeps a single
# simulated round tractable on full-scale Edge-IIoT/CIC-IoT2023 partitions.
# NOTE: this module-level constant is intentionally mutable at runtime —
# flowerfl/client_app.py overrides it (`task_module.MAX_SAMPLES_PER_CLIENT =
# max_samples`) from the Flower run-config's "max-samples" key before calling
# load_data, so an experiment's actual per-client cap (e.g. 5,000 for a
# smoke run or 2,000,000 for a full-data confirmatory run) is set per-run,
# not by editing this default.
MAX_SAMPLES_PER_CLIENT = 200_000


# ============================================================================
# SMOTE RESAMPLE CACHE (image v9 — node-local DISK cache + tiny in-process L1)
# ============================================================================
#
# Flower's Ray simulation calls client_fn -> load_data on EVERY client
# construction (~41x/round: 20 fit + 21 evaluate). With SMOTE enabled the
# kNN synthesis (imblearn fit_resample) re-ran uncached on every one of those
# calls — measured ~26 min/round overhead on EXP-017. The resampled arrays are
# a pure deterministic function of the full argument tuple, so ANY memo is
# bit-identical by construction.
#
# WHY v8's process-local LRU was replaced (empirically refuted at fleet scale):
# Flower 1.29's VirtualClientEngineActorPool is a bare LIFO idle-actor stack
# with NO client->actor affinity. At production shape (32 actors > 21 concurrent
# clients, client_resources num_cpus=1 vs PRAXIS_RAY_CPUS=32) which actor picks
# up client N on round R is set by completion order, not client identity. A
# per-PROCESS LRU therefore churns at random: a real Ray ActorPool probe measured
# a 16.3% hit rate / 3.8% pinning with NO warm-up trend (see
# tests/test_resample_cache_fleet_shape.py and tests/test_resample_cache_fleet_shape.py).
# The cache "worked" in unit tests (single process) and silently no-op'd on the
# fleet.
#
# v9 design — the DISK layer is the correctness/perf layer:
#   * L2 (authoritative): a NODE-LOCAL disk cache keyed by a stable hash of the
#     SAME full deterministic key tuple as v8. It is shared by every Ray actor
#     PROCESS on the container, so a resample computed by ANY actor is reusable by
#     EVERY actor regardless of the scheduler's (non-)affinity. Writes are ATOMIC
#     (temp file in the same dir, then os.replace) so a concurrent actor can never
#     read a torn file.
#   * L1 (tiny fast path): a per-process OrderedDict capped at
#     _RESAMPLE_CACHE_MAXSIZE=1, purely to skip a redundant np.load on the ONE
#     locality that reliably exists — the same actor re-touching the SAME
#     partition back-to-back (a fit→evaluate double-construction burst). MEMORY
#     BOUND: at fleet shape the retained L1 arrays must not
#     re-create the v8 H-C memory-pressure failure. With cap=4 the worst case is
#     32 persistent actors x 4 entries x ~1.18 GB = ~151 GB > the 120 GiB
#     container — v9's own L1 reintroducing the very failure it fixed. cap=1
#     bounds it to 32 x 1 x ~1.18 GB ≈ 38 GB (float32 X halves it to ~19 GB),
#     comfortably under the container. A larger L1 buys little: np.load from a
#     page-cache-warm node-local.npz is fast, and the OS page cache already
#     shares those bytes across actor processes WITHOUT per-process retention.
#     RAM is now actor-count-independent per entry AND bounded to a single entry.
#   * Driver PREWARM: scripts/run_phase4_flower.py populates L2 once, in the
#     driver, before run_simulation — so the worker hot path is 100% disk reads
#     — then RELEASES its own L1 before the simulation (does not sit resident).
#
# CATASTROPHIC failure mode: a cache keyed too
# loosely serves a STALE ARM. The key includes EVERYTHING that determines the
# arrays — dataset, partition, batch size, both split fractions, the per-call
# MAX_SAMPLES_PER_CLIENT cap, and every SMOTE argument (variant, normalized
# target, seed). Over-keying can only cause extra misses, never a wrong hit.
_RESAMPLE_CACHE_MAXSIZE = 1
# key -> (X_res, y_res, skipped_reason)   [L1, per-process]
_resample_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_resample_cache_hits = 0
_resample_cache_misses = 0
# Count of L2 disk entries evicted for the byte budget (observability; reset with
# the cache). Surfaced in the driver prewarm summary line.
_resample_disk_evictions = 0
# Best-effort persistence bookkeeping. An entry is "persisted"
# when it is durably on the disk layer for OTHER actor processes to read (a disk
# hit, or a miss whose store succeeded); "compute_only" when the disk layer was
# unavailable or the store failed, so every worker will recompute it. The driver
# prewarm reports the delta so a degraded (compute_only>0) prime is unmistakable
# in CloudWatch even though the per-worker recompute itself is invisible under
# log_to_driver=False.
_resample_persisted_count = 0
_resample_compute_only_count = 0
# Count of STALE abandoned temp files cleaned during budget enforcement.
# A writer killed mid-np.savez (OOM-killed Ray actor) can't run its
# cleanup, leaving a potentially GB-sized temp that would otherwise be invisible
# to the budget and grow the cache past its bound.
_resample_stale_temps_cleaned = 0
# Disk-cache dirs whose creation/verification already failed this process — used
# to warn ONCE per dir instead of on every (frequent) construction.
_resample_disk_disabled_dirs: set = set()

# Cache serialization/semantics version. Folded into the key so a change to the
# resampler output or the.npz layout invalidates EVERY prior entry (a long-lived
# container or a shared PRAXIS_RESAMPLE_CACHE_DIR must never serve an entry written
# by a different code version). Bump on ANY resampler/serialization change.
_RESAMPLE_CACHE_FORMAT_VERSION = 1

# L2 node-local disk cache. Default lives under the job's scratch; overridable so
# each container/run can scope it (and so tests can isolate). Temp files written
# during an atomic store carry this prefix and are never mistaken for entries.
_RESAMPLE_DISK_CACHE_DEFAULT = "/tmp/praxis_resample_cache"  # nosec B108
_RESAMPLE_DISK_TMP_PREFIX = ".praxis_resample_tmp_"
# LOCK INVARIANT: every temp file is created and written ONLY
# inside _resample_disk_store, which holds the EXCLUSIVE budget lock for the
# whole mkstemp->np.savez->os.replace sequence (verify: no other code path
# writes a _RESAMPLE_DISK_TMP_PREFIX file). Therefore, while a process holds the
# lock, ANY temp other than its own belongs to a DEAD writer — a crashed
# process's flock auto-releases, so no live writer can hold a temp without also
# holding the lock we currently own. Such abandoned temps are removed IMMEDIATELY
# (no age gate) so dead gigabytes never charge against the budget.

# Byte budget for the L2 disk cache. The key includes the seed, so
# a multi-seed runner invocation would otherwise accumulate partitions x seeds x
# hundreds-of-MB artifacts forever until /tmp exhausts — at which point stores
# fail, the best-effort fallback recomputes every construction, and the v8
# slowdown regime returns. Bounded by evicting oldest-mtime entries before each
# store.
#
# Default sized to comfortably hold ONE full-data run's working set so a single
# run never evicts its own hot entries (which would reintroduce intra-run
# thrash). Arithmetic (worst case, full-data confirmatory):
#   MAX_SAMPLES_PER_CLIENT cap......... 2,000,000 rows/client
#   train split (0.8).................. 1,600,000 train rows pre-resample
#   SMOTE "balanced" -> minority grown to majority => <= 2x majority
#                       => <= ~3,200,000 post-resample rows
#   per artifact: X (<=3.2M x 45 feat x 8B f64 worst) + y (3.2M x 8B)
#                 ~= 1.18 GB  (float32 X halves this to ~0.6 GB)
#   working set: 21 partitions x ~1.18 GB ~= 24.8 GB
# Default 32 GiB gives headroom over that worst case and sits far under the
# 122,880 MiB container. Dev-stage smoke (200k cap) artifacts are ~10x smaller
# (~2.5 GB working set), well within. Operators on constrained scratch (e.g. a
# memory-backed /tmp) can lower it via PRAXIS_RESAMPLE_CACHE_MAX_BYTES; the floor
# that avoids intra-run thrash is one run's working set.
_RESAMPLE_DISK_CACHE_MAX_BYTES_DEFAULT = 32 * 1024**3  # 32 GiB

# Filesystem free-space reserve. The configured budget is a
# POLICY cap, not provisioned storage: on a host with less free space than the
# budget, publishing up to the budget would fill the filesystem and starve Ray's
# object-spill directory and other container writes BEFORE run_simulation even
# starts — degrading or killing the very run the cache exists to speed up. So the
# EFFECTIVE budget is min(configured, statvfs_free - reserve), recomputed at each
# locked store (free space moves as others write). The reserve is headroom kept
# for Ray/system writes.
#
# Default 8 GiB: Ray spills serialized objects to disk under memory pressure —
# for the 21-supernode/32-actor fleet each round materializes ~21 client
# updates plus aggregation temporaries, and Ray's default object-store spill can
# reach GB-scale on a busy round; 8 GiB leaves comfortable room for that plus the
# result JSON, signal logs, and model checkpoints without starving them. Tunable
# via PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES for hosts with a different spill
# profile (set 0 to disable the reserve; the free-space cap itself still applies).
_RESAMPLE_DISK_CACHE_FS_RESERVE_DEFAULT = 8 * 1024**3  # 8 GiB
# Cache dirs where the free-space term has already been reported as the binding
# constraint this process — warn ONCE per dir instead of on every store.
_resample_fs_bound_warned_dirs: set = set()


def _resample_cache_dir() -> "str | None":
    """Return the node-local disk cache directory, or None if unavailable.

    Read from PRAXIS_RESAMPLE_CACHE_DIR on EVERY call (not cached at import) so a
    per-run/per-test override always takes effect. This directory is shared by
    all Ray actor processes on the same container — that shared visibility is the
    whole point of the disk layer.

    Directory setup is BEST-EFFORT: a read-only or otherwise
    uncreatable dir must NEVER raise. On failure it warns ONCE per dir (loud,
    [RESAMPLE-CACHE], NOT parseable as a [SMOTE] record) that the disk layer is
    disabled — resamples are recomputed every construction (v7 perf) but
    correctness and provenance are unaffected — and returns None. Callers treat
    None as "no disk layer": compute + L1 only, no persistence, no raise.
    """
    d = os.environ.get("PRAXIS_RESAMPLE_CACHE_DIR", _RESAMPLE_DISK_CACHE_DEFAULT)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as exc:
        if d not in _resample_disk_disabled_dirs:
            _resample_disk_disabled_dirs.add(d)
            print(
                f"[RESAMPLE-CACHE] WARNING cannot create/verify cache dir "
                f"errno={getattr(exc, 'errno', None)} path={d} err={exc} — "
                f"DISK CACHE DISABLED for this process: resamples will be "
                f"recomputed on every client construction (v7 perf, no "
                f"persistence). Correctness and provenance are unaffected.",
                flush=True,
            )
        return None
    return d


def _resample_disk_path(key: tuple) -> "str | None":
    """Deterministic.npz path for a cache key, or None if the disk layer is
    unavailable (best-effort dir setup failed)."""
    cache_dir = _resample_cache_dir()
    if cache_dir is None:
        return None
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, digest + ".npz")


def _resample_cache_key(
    dataset_name: str,
    partition_id: int,
    batch_size: int,
    train_split: float,
    val_split: float,
    smote_variant: str,
    smote_target,
    smote_seed: int,
    smote_semantic_target: bool = False,
    normalize_train_only: bool = False,
) -> tuple:
    """Build the full deterministic resample cache key.

    Single source of truth shared by load_data (which stores the entry) and
    resample_cache_path (which the driver prewarm uses for its final durability
    pass), so the two can never drift. Over-keyed on purpose: any
    component that could change the resampled arrays is present. Leads with the
    CACHE_FORMAT_VERSION and folds in the (size, mtime_ns) source-data
    fingerprint of the parquet actually read.
    """
    from flowerfl.smote_resampler import normalize_smote_target

    config = get_dataset_config(dataset_name)
    num_clients = len(config["client_files"])
    client_idx = partition_id % num_clients
    file_path = os.path.join(config["data_dir"], config["client_files"][client_idx])
    try:
        _src = os.stat(file_path)
        src_fingerprint = (int(_src.st_size), int(_src.st_mtime_ns))
    except OSError:
        src_fingerprint = (0, 0)
    return (
        _RESAMPLE_CACHE_FORMAT_VERSION,
        dataset_name, int(client_idx), int(batch_size),
        float(train_split), float(val_split),
        int(MAX_SAMPLES_PER_CLIENT), True,  # smote_enabled (this key is SMOTE-only)
        str(smote_variant), normalize_smote_target(smote_target),
        int(smote_seed), src_fingerprint,
        # Stage-F semantic attack-class policy changes the resampled arrays, so it
        # MUST partition the cache — a legacy-min/max entry can never alias it.
        bool(smote_semantic_target),
    ) + (
        # m1 leakage fix: leak-free normalization rescales the train rows SMOTE
        # interpolates in, so a leak-free resample must never alias a leak-on
        # entry. Appended ONLY when True so every existing (leak-on) SMOTE cache
        # key stays byte-identical — the in-flight Stage-F sweep is unaffected.
        ("normalize_train_only",) if normalize_train_only else ()
    )


def resample_cache_path(
    dataset_name: str,
    partition_id: int,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    smote_variant: str = "smote",
    smote_target="balanced",
    smote_seed: int = 42,
    smote_semantic_target: bool = False,
    normalize_train_only: bool = False,
) -> "str | None":
    """The disk-cache.npz path a SMOTE resample WOULD occupy for these args, or
    None if the disk layer is unavailable. Lets the driver prewarm check which
    partitions are actually durable on disk NOW, independent of
    the cumulative per-store counters.

    ``smote_semantic_target`` (Stage-F §6) MUST be threaded here identically to
    the worker-side load_data call — it is part of the cache key, so a
    durability pass that omitted it would validate the LEGACY key while workers
    look up the semantic key.

    ``normalize_train_only`` (m1 leakage fix) has the SAME contract: it partitions
    the resample cache key (append-only-when-True), so a durability pass that
    omitted it would validate the LEAK-ON key while leak-free workers look up the
    leak-free key — every worker would then miss and concurrently recompute the
    large train-only resample, defeating the prewarm's memory protection
    (EXP-020-class; same family as the  semantic-target finding)."""
    key = _resample_cache_key(
        dataset_name, partition_id, batch_size, train_split, val_split,
        smote_variant, smote_target, smote_seed,
        smote_semantic_target=smote_semantic_target,
        normalize_train_only=normalize_train_only,
    )
    return _resample_disk_path(key)


def _resample_cache_max_bytes() -> int:
    """The L2 disk-cache byte budget (env override, else the documented default).

    An unset/invalid/non-positive PRAXIS_RESAMPLE_CACHE_MAX_BYTES falls back to
    the default so a typo can never disable the bound.
    """
    raw = os.environ.get("PRAXIS_RESAMPLE_CACHE_MAX_BYTES")
    if raw is None:
        return _RESAMPLE_DISK_CACHE_MAX_BYTES_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _RESAMPLE_DISK_CACHE_MAX_BYTES_DEFAULT
    return val if val > 0 else _RESAMPLE_DISK_CACHE_MAX_BYTES_DEFAULT


def _resample_fs_reserve_bytes() -> int:
    """Free-space reserve kept for Ray/system writes (env override, else default).

    A negative/invalid value falls back to the default; 0 is honored (disables the
    reserve, keeping only the raw free-space cap).
    """
    raw = os.environ.get("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES")
    if raw is None:
        return _RESAMPLE_DISK_CACHE_FS_RESERVE_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _RESAMPLE_DISK_CACHE_FS_RESERVE_DEFAULT
    return val if val >= 0 else _RESAMPLE_DISK_CACHE_FS_RESERVE_DEFAULT


def _resample_fs_free_bytes(cache_dir: str) -> "int | None":
    """Bytes free to an UNPRIVILEGED writer under cache_dir, or None if unknown."""
    try:
        st = os.statvfs(cache_dir)
    except OSError:
        return None
    return int(st.f_bavail) * int(st.f_frsize)


def _resample_current_cache_bytes(cache_dir: str) -> int:
    """Bytes currently occupied under cache_dir by cache entries + temp files."""
    total = sum(size for _p, size, _m in _resample_disk_entries(cache_dir))
    total += sum(size for _p, size, _m in _resample_temp_files(cache_dir))
    return total


def _resample_effective_budget(cache_dir: str, cache_bytes_now: int = 0) -> int:
    """The max TOTAL cache size enforced this instant.

    The configured PRAXIS_RESAMPLE_CACHE_MAX_BYTES is only a policy cap; a host
    may have less disk than that. CRUCIAL accounting point: statvfs free space ALREADY EXCLUDES the bytes the cache's own
    entries occupy, so the fs-derived MAX TOTAL size is what the cache already
    holds PLUS what is still free, minus the reserve — NOT `free - reserve`
    (which double-counts existing entries and would evict usable / reject fitting
    artifacts). So:

        effective = min(configured, cache_bytes_now + fs_free - reserve)

    Recomputed on every (locked) store because free space changes as Ray spills
    and others write. Callers pass cache_bytes_now (entries + temps) enumerated
    under the lock. Warn ONCE per dir when the fs term binds. Floors at 0. If
    free space cannot be measured, the configured budget governs (never raises).
    """
    configured = _resample_cache_max_bytes()
    free = _resample_fs_free_bytes(cache_dir)
    if free is None:
        return configured
    reserve = _resample_fs_reserve_bytes()
    fs_capacity = max(0, int(cache_bytes_now) + free - reserve)
    if fs_capacity < configured:
        if cache_dir not in _resample_fs_bound_warned_dirs:
            _resample_fs_bound_warned_dirs.add(cache_dir)
            print(
                f"[RESAMPLE-CACHE] WARNING free-space bound is binding: "
                f"cache_bytes={int(cache_bytes_now)} fs_free={free} reserve={reserve} "
                f"effective_budget={fs_capacity} < configured={configured} "
                f"dir={cache_dir} — the disk cache is capped by available disk "
                f"(existing entries + free - reserve), not "
                f"PRAXIS_RESAMPLE_CACHE_MAX_BYTES (protecting Ray spill / system "
                f"writes).",
                flush=True,
            )
        return fs_capacity
    return configured


def _resample_disk_entries(cache_dir: str) -> list:
    """Return [(path, size_bytes, mtime_ns)] for published *.npz cache ENTRIES.

    Race-tolerant (a file vanishing between listdir and stat is skipped). Temp
    files (``_RESAMPLE_DISK_TMP_PREFIX*``) are EXCLUDED — they are not cache
    entries and are accounted separately by _resample_temp_files /
    _clean_stale_temps.
    """
    out = []
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".npz") or name.startswith(_RESAMPLE_DISK_TMP_PREFIX):
            continue
        path = os.path.join(cache_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue  # evicted by another process between listdir and stat
        out.append((path, st.st_size, st.st_mtime_ns))
    return out


def _resample_temp_files(cache_dir: str) -> list:
    """Return [(path, size_bytes, mtime_ns)] for temp files, race-tolerant.

    Temp files consume disk but are not cache entries. A LIVE temp (a store in
    progress under the budget lock) still counts toward the budget total; a STALE
    one (writer died) is cleaned by _clean_stale_temps.
    """
    out = []
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return out
    for name in names:
        if not name.startswith(_RESAMPLE_DISK_TMP_PREFIX):
            continue
        path = os.path.join(cache_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        out.append((path, st.st_size, st.st_mtime_ns))
    return out


def _clean_stale_temps(cache_dir: str, exclude: "str | None" = None) -> int:
    """Immediately unlink EVERY abandoned temp file (except ``exclude``).

    MUST be called while holding the exclusive budget lock. Per the lock
    invariant (see _RESAMPLE_DISK_TMP_PREFIX above), any temp other than the
    caller's own (``exclude``) is a DEAD writer's orphan — a crashed writer's
    flock auto-releases, so no live writer can own a temp without also holding
    the lock we currently hold. So there is NO age gate: a
    fresh-mtime orphan is removed at once, since dead gigabytes must not charge
    against the budget while we wait out an arbitrary timer. ENOENT-tolerant;
    a temp that cannot be unlinked is left in place (it still counts toward the
    budget total). Returns the count cleaned (accrued into
    _resample_stale_temps_cleaned).
    """
    global _resample_stale_temps_cleaned
    cleaned = 0
    for path, _size, _mtime_ns in _resample_temp_files(cache_dir):
        if path == exclude:
            continue  # our own in-flight temp (becomes the published entry)
        try:
            os.unlink(path)
            cleaned += 1
        except FileNotFoundError:
            pass  # already cleaned by another process
        except OSError:
            continue
    _resample_stale_temps_cleaned += cleaned
    return cleaned


# Interprocess budget lock. The byte budget is only sound if
# evict + size-check + publish is SERIALIZED across processes: otherwise
# concurrent worker-side stores each snapshot the same total, all decide they
# fit, and all publish — exceeding the bound by up to (writers-1) artifacts.
# A best-effort fcntl.flock on this file in the cache dir serializes them.
# INVARIANT: the bound is HARD whenever the lock is acquired; the lock-unavailable
# path never publishes (it skips), so it also never violates the bound.
_RESAMPLE_BUDGET_LOCK_NAME = ".budget.lock"
_RESAMPLE_BUDGET_LOCK_TIMEOUT_S = 2.0
_RESAMPLE_BUDGET_LOCK_POLL_S = 0.02


def _acquire_budget_lock(cache_dir: str):
    """Best-effort NON-BLOCKING interprocess lock over the cache dir's budget.

    Returns an open fd holding an exclusive flock, or None if the lock could not
    be acquired within _RESAMPLE_BUDGET_LOCK_TIMEOUT_S (a peer is mid-store) or
    could not be created (no fcntl / uncreatable lock file — same best-effort
    degradation as dir setup). A None return means the caller SKIPS the store
    (compute_only) rather than block a training client or risk exceeding the
    bound.
    """
    if fcntl is None:
        return None
    lock_path = os.path.join(cache_dir, _RESAMPLE_BUDGET_LOCK_NAME)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    deadline = time.monotonic() + _RESAMPLE_BUDGET_LOCK_TIMEOUT_S
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except OSError:
            if time.monotonic() >= deadline:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
                return None
            time.sleep(_RESAMPLE_BUDGET_LOCK_POLL_S)


def _release_budget_lock(lock_fd) -> None:
    """Release + close a budget lock fd (no-op for None). Never raises."""
    if lock_fd is None:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(lock_fd)
    except OSError:
        pass


def _evict_for_budget(cache_dir: str, incoming_bytes: int, *, exclude_temp: "str | None" = None) -> bool:
    """Evict oldest-mtime entries until incoming_bytes fits under the budget.

    Enforced BEFORE each store so total on-disk bytes + the new artifact stay
    within PRAXIS_RESAMPLE_CACHE_MAX_BYTES. Concurrency-safe: a file already
    removed by another process (ENOENT) is treated as freed, never raised; an
    entry that cannot be unlinked (perms/busy) is left in place and still counts
    against the total.

    MUST be called while holding the budget lock (all production callers —
    _resample_disk_store, enforce_resample_disk_budget — do), because it removes
    other processes' temp files, which is only safe under the lock invariant.

    ``incoming_bytes`` is the TRUE serialized size of the artifact about to be
    published: the caller writes the temp first, then passes its
    on-disk size and its path as ``exclude_temp`` so it is not double-counted
    (the temp becomes the published entry).

    Returns whether the incoming artifact now fits under the budget. It may NOT fit if the artifact alone exceeds the budget (a small-scratch
    config) or if eviction candidates were exhausted (unremovable entries) — the
    caller then SKIPS the store rather than blowing past the bound. Evictions are
    accrued into the module-level _resample_disk_evictions counter.
    """
    global _resample_disk_evictions
    # Remove EVERY other (abandoned) temp immediately — under the
    # lock, any temp but our own is a dead writer's orphan. Only a temp we could
    # not unlink lingers, and it still counts toward the total below.
    _clean_stale_temps(cache_dir, exclude=exclude_temp)
    all_temps = _resample_temp_files(cache_dir)
    temp_bytes = sum(size for path, size, _ in all_temps if path != exclude_temp)
    temp_bytes_all = sum(size for _p, size, _m in all_temps)  # incl our own in-flight temp
    entries = _resample_disk_entries(cache_dir)
    entries_total = sum(size for _, size, _ in entries)
    # Effective budget uses the CURRENT cache bytes INCLUDING our own temp (which
    # occupies real disk that statvfs already subtracted, and which becomes the
    # published entry) so capacity = cache_now + free - reserve is correct — the
    # earlier code double-counted existing entries.
    budget = _resample_effective_budget(cache_dir, entries_total + temp_bytes_all)
    # `total` is cache bytes EXCLUDING our own temp; `incoming_bytes` (== our temp
    # size on the store path) stands in for it, so total + incoming == cache_now.
    total = entries_total + temp_bytes
    if total + incoming_bytes <= budget:
        return True
    # Impossible even with an EMPTY entry set (only our temp/other-unremovable
    # temps remain): preserve existing entries and skip WITHOUT evicting.
    # temp_bytes here is only unremovable other-temps (~0).
    if temp_bytes + incoming_bytes > budget:
        return False
    entries.sort(key=lambda e: e[2])  # oldest mtime first
    evicted = 0
    for path, size, _ in entries:
        if total + incoming_bytes <= budget:
            break
        try:
            os.unlink(path)
            evicted += 1
        except FileNotFoundError:
            pass  # another process evicted it first — its bytes are already gone
        except OSError:
            continue  # cannot remove (perms/busy) — leave it counting against total
        total -= size  # freed whether we or a racing process removed it
    _resample_disk_evictions += evicted
    return total + incoming_bytes <= budget


def enforce_resample_disk_budget() -> int:
    """Evict oldest entries until the disk cache is within the CURRENT budget.

    Best-effort, under the budget lock. Called ONCE per driver prewarm before the
    per-partition loop so a REUSED cache dir whose budget was
    lowered between runs — or a shared dir written by a larger-budget process —
    does not stay oversized on an all-hit (all-cached) workload, where the
    store-time fast path returns before any budget check. Returns the number of
    entries evicted (0 if the disk layer is unavailable or the lock can't be
    acquired). Never raises.
    """
    cache_dir = _resample_cache_dir()
    if cache_dir is None:
        return 0
    before = _resample_disk_evictions
    lock_fd = _acquire_budget_lock(cache_dir)
    if lock_fd is None:
        return 0
    try:
        # incoming=0: evict until the EXISTING contents alone fit the budget.
        _evict_for_budget(cache_dir, 0)
    finally:
        _release_budget_lock(lock_fd)
    return _resample_disk_evictions - before


def _resample_disk_load(path: str):
    """Load (X_res, y_res, skipped_reason) from an.npz, or None if unreadable.

    A cache READ failure of ANY kind is BY DEFINITION a miss: the
    except is intentionally broad. This is a PURE cache read — nothing about it
    may propagate into the caller, or a damaged cache file (a torn/partial write,
    a legacy layout, a `zipfile.BadZipFile` CRC/central-directory corruption, or
    any future np.load surprise) would escape load_data's broad handler and
    SILENTLY substitute generate_synthetic_data for the client's REAL partition.
    We warn once (naming the exception class) and return None so the caller
    recomputes and self-heals the entry via the overwrite path.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            X_res = data["X_res"]
            y_res = data["y_res"]
            has_reason = bool(data["has_reason"][0])
            reason = str(data["reason"][0]) if has_reason else None
        return X_res, y_res, reason
    except Exception as exc:  # noqa: BLE001 — a cache read failure IS a miss
        print(
            f"[RESAMPLE-CACHE] WARNING unreadable entry ({type(exc).__name__}): "
            f"treating as miss path={path}",
            flush=True,
        )
        return None


def _resample_disk_valid(path: str) -> bool:
    """Cheap STRUCTURAL validity check for a cache entry.

    Opens the.npz as a zip and verifies the expected members exist via the
    central directory — NO CRC / data read, so it scales to many large entries at
    prewarm time (a full np.load of 21 x ~1 GB entries on the prewarm hot path is
    not acceptable). Used by the driver prewarm's durability pass so a
    structurally-broken entry (torn central directory, missing member — the
    failure a killed writer / partial publish produces) is not miscounted as
    durable. TRADEOFF (documented): it does NOT detect data-region (CRC)
    corruption; the worker-side _resample_disk_load still full-validates on read
    and self-heals via the overwrite path if such corruption is ever hit.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except Exception:  # noqa: BLE001 — any failure means "not a usable entry"
        return False
    return {"X_res.npy", "y_res.npy", "has_reason.npy", "reason.npy"} <= names


def _touch_resample_entry(path: str) -> None:
    """Refresh an entry's mtime on a successful USE.

    Byte-budget eviction is oldest-mtime, but mtime is set at WRITE time and disk
    loads never updated it — so a just-USED current-run entry kept its old write
    mtime and became an eviction target while newer FOREIGN-seed artifacts (in a
    shared/reused dir) survived, evicting the live working set out from under the
    durability pass and worker stores. Touching on a successful load/validation
    makes "oldest mtime" mean "least recently USED", the LRU intent. Best-effort:
    a failed touch degrades recency ordering, never correctness.
    """
    try:
        os.utime(path, None)
    except OSError:
        pass


def _resample_disk_store(path: str, X_res, y_res, skipped_reason, *, overwrite: bool = False) -> bool:
    """Atomically persist the resample tuple to `path` (temp-then-rename).

    The temp file is written in the SAME directory (so os.replace is atomic on
    the same filesystem). skipped_reason is stored WITHOUT pickle (a unicode
    array + a boolean sentinel) so reads stay allow_pickle=False.

    Returns True if the entry was persisted (durable on disk), False if the store
    was SKIPPED — because the artifact does not fit the byte budget, the interprocess budget lock could not be acquired, or
    serialization failed (for example, ENOSPC). The bound is never exceeded to
    make room, and space-preevict + serialize + size-check + publish all run
    under the lock so concurrent stores cannot each snapshot the same total and
    all publish.

    Publish policy:
      * overwrite=False (default) — plain lost-a-race case: if an entry already
        exists we discard our temp. The content is a deterministic function of
        the key, so any winner's bytes are identical; a reader can never observe
        a torn or mismatched file. (Counts as persisted — the entry is on disk.)
      * overwrite=True — the miss followed a load-REJECTION of an existing
        (torn/legacy) file: publish UNCONDITIONALLY via
        os.replace. Replacing corrupt bytes with valid, deterministic bytes is
        strictly correct and self-heals the entry, instead of the discard guard
        leaving the corrupt file immortal (every future process recomputing it).
    """
    cache_dir = os.path.dirname(path)
    has_reason = np.array([skipped_reason is not None])
    reason = np.array([skipped_reason if skipped_reason is not None else ""], dtype="U64")

    # Serialize evict + size-check + publish across processes. A
    # best-effort non-blocking lock: if a peer holds it past the timeout, SKIP
    # (compute_only) rather than block a training client or risk exceeding the
    # bound. The invariant — the budget is HARD under the lock — is what makes
    # the check-then-publish safe under worker concurrency.
    lock_fd = _acquire_budget_lock(cache_dir)
    if lock_fd is None:
        print(
            f"[RESAMPLE-CACHE] WARNING could not acquire budget lock within "
            f"{_RESAMPLE_BUDGET_LOCK_TIMEOUT_S}s path={path} — NOT persisted "
            f"(workers will recompute this entry)",
            flush=True,
        )
        return False
    try:
        # Lost-race fast path: if a peer already published this
        # exact entry (non-overwrite), we have NOTHING to do — return persisted
        # BEFORE writing a temp or running eviction. Reserving budget for a
        # duplicate we would only discard could needlessly evict unrelated HOT
        # entries. Deterministic bytes mean the existing file is already correct.
        if not overwrite and os.path.exists(path):
            return True

        # Two-phase store — fixes the r8 regression where writing
        # the full temp BEFORE any eviction could np.savez-ENOSPC on a near-full
        # filesystem (less than one artifact free), leaving the OLD working set
        # intact so a new seed/config can never populate.
        # Effective budget = min(configured, current_cache_bytes + free - reserve),
        # recomputed here under the lock: the configured
        # budget is a policy cap, not provisioned storage, and statvfs free already
        # excludes the cache's own entries — so the max TOTAL size is what the cache
        # already holds plus what is still free, minus the reserve.
        budget = _resample_effective_budget(cache_dir, _resample_current_cache_bytes(cache_dir))
        nbytes_est = int(getattr(X_res, "nbytes", 0)) + int(getattr(y_res, "nbytes", 0))

        # Phase 0: an IMPOSSIBLE-to-fit artifact must never
        # flush the cache. The true.npz is strictly larger than the raw arrays, so
        # if the raw bytes already exceed the max TOTAL size it can NEVER fit even
        # after evicting everything — skip WITHOUT evicting (existing entries live).
        if nbytes_est > budget:
            print(
                f"[RESAMPLE-CACHE] WARNING artifact does not fit cache budget "
                f"incoming_bytes={nbytes_est} budget_bytes={budget} path={path} — "
                f"NOT persisted (workers will recompute this entry)",
                flush=True,
            )
            return False

        #   Phase 1: pre-evict for SPACE using a generous ESTIMATE (raw nbytes +
        #     margin) so the temp write has room BEFORE it runs. This is about
        #     freeing disk, not the exact bound (return ignored). Guarded so a
        #     tiny/impossible fit (estimate > budget) never flushes the cache to
        #     make room for an artifact Phase 3 would reject: skip the pre-evict
        #     there (a sub-margin budget has no ENOSPC risk; Phase 3 still decides
        #     the exact bound and preserves entries on an impossible true size).
        space_estimate = nbytes_est + max(nbytes_est // 100, 1024 * 1024)
        if space_estimate <= budget:
            _evict_for_budget(cache_dir, space_estimate)

        # Phase 2: serialize to the temp.
        fd, tmp = tempfile.mkstemp(prefix=_RESAMPLE_DISK_TMP_PREFIX, suffix=".npz", dir=cache_dir)
        try:
            try:
                with os.fdopen(fd, "wb") as fh:
                    np.savez(fh, X_res=X_res, y_res=y_res, has_reason=has_reason, reason=reason)
            except OSError as exc:
                # ENOSPC (or another write error) mid-serialize even after the
                # space pre-evict — best-effort SKIP (clean the partial temp),
                # never propagate into load_data's synthetic-data fallback.
                print(
                    f"[RESAMPLE-CACHE] WARNING serialization failed (likely ENOSPC) "
                    f"errno={getattr(exc, 'errno', None)} path={path} err={exc} — "
                    f"NOT persisted (workers will recompute this entry)",
                    flush=True,
                )
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return False
            # Phase 3: decide publish/skip on the temp's TRUE on-disk size
            # — np.savez adds ZIP headers + the has_reason/reason
            # arrays, so a raw-nbytes estimate could publish slightly OVER the
            # hard limit. Our own temp is excluded from the total (it becomes the
            # published entry, accounted as `actual`).
            actual = os.path.getsize(tmp)
            # Phase 3: _evict_for_budget decides on the TRUE on-disk size against
            # the corrected effective budget (cache_now incl our temp + free -
            # reserve). It evicts oldest entries to fit, and — on an IMPOSSIBLE
            # artifact (bigger than the max total size even with an empty entry
            # set) — returns False WITHOUT evicting, so existing entries survive
            # (Fix-1 spirit). Either skip cleans our temp and never publishes.
            if not _evict_for_budget(cache_dir, actual, exclude_temp=tmp):
                print(
                    f"[RESAMPLE-CACHE] WARNING artifact does not fit cache budget "
                    f"incoming_bytes={actual} "
                    f"budget_bytes={_resample_effective_budget(cache_dir, _resample_current_cache_bytes(cache_dir))} "
                    f"path={path} — NOT persisted (workers will recompute this entry)",
                    flush=True,
                )
                os.unlink(tmp)
                return False
            os.replace(tmp, path)  # atomic publish (also overwrites a rejected entry)
        except BaseException:
            # Never leave a temp behind on any failure path.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    finally:
        _release_budget_lock(lock_fd)


def _store_l1(key, value) -> None:
    """Insert into the bounded per-process L1 LRU (most-recent at the end)."""
    _resample_cache[key] = value
    _resample_cache.move_to_end(key)
    while len(_resample_cache) > _RESAMPLE_CACHE_MAXSIZE:
        _resample_cache.popitem(last=False)


def _clear_resample_l1() -> None:
    """Drop ONLY the in-process L1 RAM cache — NOT the disk layer, NOT the
    counters. The driver prewarm calls this after its final
    durability pass + provenance snapshot so the ~4-entry (up to ~4.7 GiB
    worst-case) RAM L1 does not sit resident for the whole simulation, competing
    with Ray actors for exactly the memory the prewarm exists to free.
    _reset_resample_cache is the WRONG tool here — it also wipes the disk entries
    (which workers must read) and the provenance counters.
    """
    _resample_cache.clear()


def _clear_resample_disk_cache() -> None:
    """Remove this module's entries/temp files from the disk cache directory.

    Only removes ``*.npz`` and ``_RESAMPLE_DISK_TMP_PREFIX*`` files — never the
    directory itself or foreign files. Called only from _reset_resample_cache
    (test hook / explicit process reset) — never on the fleet hot path.
    """
    d = os.environ.get("PRAXIS_RESAMPLE_CACHE_DIR", _RESAMPLE_DISK_CACHE_DEFAULT)
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        if name.endswith(".npz") or name.startswith(_RESAMPLE_DISK_TMP_PREFIX):
            try:
                os.unlink(os.path.join(d, name))
            except OSError:
                pass


def _reset_resample_cache() -> None:
    """Clear BOTH cache layers + counters (test hook / process reset).

    Clears the in-process L1 LRU and its counters AND wipes the node-local disk
    cache (L2) so a fresh process starts genuinely cold. This is a test/reset
    hook with NO production caller — a prewarmed disk cache is never wiped
    mid-run on the fleet.
    """
    global _resample_cache_hits, _resample_cache_misses, _resample_disk_evictions
    global _resample_persisted_count, _resample_compute_only_count
    global _resample_stale_temps_cleaned
    _resample_cache.clear()
    _resample_cache_hits = 0
    _resample_cache_misses = 0
    _resample_disk_evictions = 0
    _resample_persisted_count = 0
    _resample_compute_only_count = 0
    _resample_stale_temps_cleaned = 0
    _cache_env_dir = os.environ.get("PRAXIS_RESAMPLE_CACHE_DIR", _RESAMPLE_DISK_CACHE_DEFAULT)
    _resample_disk_disabled_dirs.discard(_cache_env_dir)
    _resample_fs_bound_warned_dirs.discard(_cache_env_dir)
    _clear_resample_disk_cache()


def _resample_cached(X_tr, y_tr, *, key, variant, target, seed, attack_target_policy=False):
    """Return (X_res, y_res, skipped_reason, was_cached) for the training split.

    Two-layer memo over the deterministic resampler:
      1. L1 (in-process LRU): a hit skips even the disk read.
      2. L2 (node-local disk): a hit np.loads the entry — this is the layer
         that survives Ray's unpinned actor scheduling, because the file is
         visible to every actor process on the container.
      3. Miss: run the resampler, write-through to disk (atomic) AND L1.

    was_cached is True for BOTH an L1 and an L2 hit (no recompute happened) and
    False only when the resampler actually ran — load_data gates the one-per-
    client [SMOTE] provenance record on ``not was_cached`` so a run emits exactly
    one record per distinct resample (the driver prewarm's misses), never a
    duplicate per worker re-touch. Correctness never depends on the disk layer:
    an unreadable entry, an unavailable cache dir, or a failed store all fall
    through to a recompute that still returns the real arrays.

    Persistence bookkeeping (best-effort contract): _resample_persisted_count is
    incremented when the entry is durably on disk for OTHER actor processes (a
    disk hit, or a miss/L1-hit whose store succeeded); _resample_compute_only_count
    when the disk layer was unavailable or the store failed (every worker will
    recompute). An L1 hit is NOT assumed durable — Ray actors cannot see this
    process's L1, and byte-budget eviction may have removed the.npz — so the
    disk entry is verified and RESTORED if missing before counting (P2-1, r4).
    """
    global _resample_cache_hits, _resample_cache_misses
    global _resample_persisted_count, _resample_compute_only_count
    # L1: process-local fast path
    if key in _resample_cache:
        _resample_cache.move_to_end(key)
        _resample_cache_hits += 1
        X_res, y_res, skipped_reason = _resample_cache[key]
        _account_l1_hit_durability(key, X_res, y_res, skipped_reason)
        return X_res, y_res, skipped_reason, True

    # L2: node-local disk, shared across actor processes. disk_path is None when
    # the cache dir could not be created/verified (best-effort, P1-1) — then there
    # is simply no disk layer and we compute + L1 without persistence.
    disk_path = _resample_disk_path(key)
    rejected_corrupt = False
    if disk_path is not None and os.path.exists(disk_path):
        loaded = _resample_disk_load(disk_path)
        if loaded is not None:
            _touch_resample_entry(disk_path)  # refresh recency on USE (LRU-by-mtime)
            X_res, y_res, skipped_reason = loaded
            _store_l1(key, (X_res, y_res, skipped_reason))
            _resample_cache_hits += 1
            _resample_persisted_count += 1
            return X_res, y_res, skipped_reason, True
        # File exists but is torn/legacy — recompute AND overwrite it (P2) so a
        # corrupt entry self-heals instead of forcing perpetual recomputes.
        rejected_corrupt = True

    # Miss: compute the deterministic resample and write through both layers.
    from flowerfl.smote_resampler import resample_training_split

    X_res, y_res, skipped_reason = resample_training_split(
        X_tr, y_tr, variant=variant, target=target, seed=seed,
        attack_target_policy=attack_target_policy,
    )
    # Disk persistence is BEST-EFFORT: a store failure
    # (ENOSPC, read-only fs,...) OR an unavailable cache dir must NEVER lose the
    # already-computed valid resample. If a store raised, the exception would
    # propagate into load_data's broad handler and SILENTLY substitute
    # generate_synthetic_data for the client's real partition. Warn loudly (NOT
    # with a [SMOTE] prefix, so it is never parsed as a provenance record) and
    # return the real arrays regardless. Track persisted vs compute_only so the
    # prewarm can surface a degraded prime.
    persisted = False
    if disk_path is None:
        # No disk layer this process — already warned once by _resample_cache_dir.
        _resample_compute_only_count += 1
    else:
        try:
            # Returns False (already warned) when the artifact does not fit the
            # budget; raises only on an unexpected write error.
            persisted = _resample_disk_store(
                disk_path, X_res, y_res, skipped_reason, overwrite=rejected_corrupt
            )
        except Exception as exc:  # noqa: BLE001 — best-effort cache write
            print(
                f"[RESAMPLE-CACHE] WARNING disk store failed "
                f"errno={getattr(exc, 'errno', None)} path={disk_path} err={exc} "
                f"(returning computed arrays; the resample is intact, only the "
                f"cache write was skipped — workers will recompute this entry)",
                flush=True,
            )
        if persisted:
            _resample_persisted_count += 1
        else:
            _resample_compute_only_count += 1
            if rejected_corrupt:
                # The repair of a REJECTED corrupt entry could not be persisted
                # (doesn't-fit / lock timeout / write error), so the corrupt file
                # is still on disk. Remove it: a MISSING entry is
                # strictly better than a POISONED one — every worker would reject
                # the corrupt file and recompute anyway, and the prewarm
                # durability pass must not count it as durable. Best-effort.
                try:
                    os.unlink(disk_path)
                except OSError:
                    pass
    _store_l1(key, (X_res, y_res, skipped_reason))
    _resample_cache_misses += 1
    return X_res, y_res, skipped_reason, False


def _account_l1_hit_durability(key, X_res, y_res, skipped_reason) -> None:
    """Count an L1 hit as persisted only if the L2 entry is actually durable.

    An L1 hit means THIS process has the arrays, but Ray actors run in separate
    processes and cannot see this L1 — so the entry is only useful fleet-wide if
    the.npz is on disk. Byte-budget eviction (or any external removal) may have
    deleted it. Verify (cheap os.path.exists) and RESTORE via the normal atomic
    store if missing; count persisted only when durable, else compute_only with a
    loud warning.
    """
    global _resample_persisted_count, _resample_compute_only_count
    disk_path = _resample_disk_path(key)
    if disk_path is None:
        _resample_compute_only_count += 1  # no disk layer (already warned once)
        return
    if os.path.exists(disk_path):
        _resample_persisted_count += 1
        return
    # L1-resident but the durable entry is gone — restore it for the other actors.
    try:
        if _resample_disk_store(disk_path, X_res, y_res, skipped_reason):
            _resample_persisted_count += 1
        else:
            _resample_compute_only_count += 1  # did not fit budget (already warned)
    except Exception as exc:  # noqa: BLE001 — best-effort restore
        print(
            f"[RESAMPLE-CACHE] WARNING failed to restore evicted entry "
            f"errno={getattr(exc, 'errno', None)} path={disk_path} err={exc} "
            f"(workers will recompute this partition)",
            flush=True,
        )
        _resample_compute_only_count += 1


def _make_prep_info(
    *, n_orig, n_resampled, n_benign_before, n_attack_before,
    n_benign_after, n_attack_after, status, skip_reason, k_eff, variant, target,
) -> dict:
    """Assemble the per-client resampling-manifest fragment load_data owns (§5).

    These are the fields knowable at data-prep time — the client
    (flowerfl/client_app.py) adds partition_id, arm, weight_mode, update_match,
    max_steps, actual_steps and reported num_examples to complete the row.
    """
    return {
        "n_orig": int(n_orig),
        "n_resampled": int(n_resampled),
        "n_benign_before": int(n_benign_before),
        "n_attack_before": int(n_attack_before),
        "n_benign_after": int(n_benign_after),
        "n_attack_after": int(n_attack_after),
        "sampler_status": status,
        "skip_reason": skip_reason,
        "k_eff": int(k_eff),
        "variant": variant,
        "target_fraction": target,
    }


def _prep_info_from_loader(train_loader) -> dict:
    """Off-path prep_info for the synthetic-data fallbacks (no resampling ran).

    n_orig == n_resampled == the synthetic train count; before == after. Class
    counts come straight from the loader's dataset so the manifest schema is
    still complete on the fallback path.
    """
    ds = train_loader.dataset
    labels = torch.stack([torch.as_tensor(ds[i][1]) for i in range(len(ds))]) if len(ds) else torch.empty(0)
    n_benign = int((labels == 0).sum()) if len(ds) else 0
    n_attack = int((labels == 1).sum()) if len(ds) else 0
    return _make_prep_info(
        n_orig=len(ds), n_resampled=len(ds),
        n_benign_before=n_benign, n_attack_before=n_attack,
        n_benign_after=n_benign, n_attack_after=n_attack,
        status="off", skip_reason=None, k_eff=0, variant=None, target=None,
    )


def _load_data_result(
    train_loader, val_loader, test_loader, prep_info, fingerprint_pool,
    return_prep_info: bool, return_fingerprint_pool: bool,
):
    """Assemble load_data's return tuple from the two OPT-IN extras.

    Single source of truth for the arity contract so the real-data path and both
    synthetic fallbacks can never drift: the 3-tuple is the incumbent, then
    ``prep_info`` and then the fingerprint pool are appended, in that order, for
    whichever extras the caller asked for. A caller that asked for neither gets
    the byte-identical incumbent 3-tuple.
    """
    extras = []
    if return_prep_info:
        extras.append(prep_info)
    if return_fingerprint_pool:
        extras.append(fingerprint_pool)
    return (train_loader, val_loader, test_loader, *extras)


def compute_zscore_stats(X, train_split: float, normalize_train_only: bool):
    """Per-feature Z-score (mean, std) for the normalization transform.

    ``normalize_train_only`` False (DEFAULT, incumbent): fit on ALL rows — the
    historical m1 LEAK path (normalization audit), byte-identical to every
    sealed pre-registered run: exactly ``X.mean(axis=0)`` / ``X.std(axis=0)``
    with a std==0 -> 1.0 guard.

    ``normalize_train_only`` True: fit EXCLUSIVELY on the training rows of the
    fixed seed-42 split. random_split(dataset, [...], seed=42) assigns the train
    subset ``randperm(n_total, seed42)[:n_train]``; we reconstruct that exact
    index set here so the fit rows are identical to the rows that later train the
    model, with no val/test contribution.

    Returns fresh (mean, std) numpy arrays and never mutates ``X``. Zero-variance
    features get std=1.0 so the divide is well-defined.
    """
    if normalize_train_only:
        n_total = X.shape[0]
        n_train = int(train_split * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(42)
        ).numpy()
        fit_rows = X[perm[:n_train]]
    else:
        fit_rows = X
    mean = fit_rows.mean(axis=0)
    std = fit_rows.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def load_data(
    partition_id: int,
    dataset_name: str = "cic",
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    smote_enabled: bool = False,
    smote_variant: str = "smote",
    smote_target="balanced",
    smote_seed: int = 42,
    smote_record_always: bool = False,
    smote_semantic_target: bool = False,
    normalize_train_only: bool = False,
    return_prep_info: bool = False,
    return_fingerprint_pool: bool = False,
):
    """
    Load data partition for federated learning.

    For full datasets (edge_full, cic_full), stratified sampling caps each
    client at MAX_SAMPLES_PER_CLIENT rows to keep training feasible while
    preserving class distribution.

    SMOTE : when ``smote_enabled`` is True, the TRAINING split ONLY is
    oversampled after the per-client cap and the fixed-seed train/val/test
    partition, so the val/test streams never contain synthetic rows. The knob
    is inert (byte-identical to the incumbent) when disabled. ``smote_seed`` is
    the per-(base_seed, client) value from derive_seed — load_data runs once per
    client per run, before the server round is known, so seeding is per-client
    not per-round.

    Stage-F (§4/§5/§6): ``smote_semantic_target`` selects the semantic attack-
    class (label 1) resampling policy (§6) instead of the legacy min/max ratio.
    ``return_prep_info`` is an OPT-IN that appends a 4th return element — a dict
    carrying ``n_orig`` (the PRE-resampling train row count, the update-matching
    budget input) plus the per-client resampling manifest fragment (§5); it does
    NOT change the loaders, so every legacy 3-tuple caller is untouched.

    H3 fingerprint emission (EMISSION_CONTRACT § 4.2): ``return_fingerprint_pool``
    is a second OPT-IN that appends a ``FingerprintPool`` — the device's
    training-split rows in the locked 45-feature order, as **raw float64**,
    captured HERE because this is the only place the un-normalised frame exists.
    It must not be reconstructed from the returned loaders: those hold the
    Z-scored float32 tensor, which zeroes every value past float32's range
    (52,331 cells on partition 3) and forces mean~0/std~1 per device. The pool is
    bounded at ``min(n_train, 100_000)`` rows taken as the first P of the SAME
    fixed seed-42 train permutation used for the split, so it is seed-invariant
    and depends on ``partition_id`` only. It is deliberately NOT put in
    ``prep_info``, which is JSON-serialised into the resampling manifest.

    Returns: (train_loader, val_loader, test_loader), with ``prep_info`` and then
    the fingerprint pool appended, in that order, for whichever of
    ``return_prep_info`` / ``return_fingerprint_pool`` was requested.
    """
    config = get_dataset_config(dataset_name)
    num_clients = len(config["client_files"])

    # partition_id may exceed num_clients (e.g. a federation configured with
    # more supernodes than data partitions); modulo wraps it back into range
    # rather than raising, so client_idx always indexes a valid client_files
    # entry.
    client_idx = partition_id % num_clients
    file_name = config["client_files"][client_idx]
    file_path = os.path.join(config["data_dir"], file_name)

    if os.path.exists(file_path):
        print(f"[Dataset] Loading {dataset_name} client {client_idx}: {file_name}")
        try:
            df = pd.read_parquet(file_path)

            label_col = config["label_column"]
            if label_col not in df.columns:
                raise KeyError(f"Label column '{label_col}' not found")

            # Stratified downsampling for large datasets (single-source predicate
            # shared with the disjoint-holdout reconstruction).
            if is_full_dataset(dataset_name) and len(df) > MAX_SAMPLES_PER_CLIENT:
                print(f"[Dataset] Stratified sampling: {len(df):,} -> {MAX_SAMPLES_PER_CLIENT:,}")
                # Group by label and sample proportionally
                sampled_parts = []
                for label_val, group in df.groupby(label_col):
                    n_samples = max(1, int(MAX_SAMPLES_PER_CLIENT * len(group) / len(df)))
                    sampled_parts.append(
                        group.sample(n=n_samples, random_state=42 + client_idx)
                    )
                df = pd.concat(sampled_parts).reset_index(drop=True)

            raw_labels = df[label_col].values

            # CIC-IoT2023's label column is multi-class (0 = benign, 1..N =
            # distinct attack categories); collapse to binary attack/benign.
            # Edge-IIoT and BRFSS label columns are already binary, so no
            # collapsing is needed for them.
            if dataset_name in ("cic", "cic_full", "cic_full_rmc"):
                binary_labels = np.where(raw_labels == 0, 0, 1)
            else:
                binary_labels = raw_labels.astype(int)

            y = torch.tensor(binary_labels, dtype=torch.long)

            X_df = df.drop(columns=[label_col], errors='ignore')
            X = X_df.astype(np.float32).values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Z-score normalization
            # BRFSS data is already standardized by Szelag's preprocessing,
            # but we apply Z-score anyway for consistency with the pipeline.
            # Szelag's original code also uses independent fit_transform on
            # train/test, so re-normalizing is acceptable.
            #
            # m1 leakage fix (opt-in, default OFF): with normalize_train_only
            # False the statistics are fitted on ALL rows before the split — the
            # sealed incumbent LEAK path (normalization audit), byte-
            # identical to every pre-registered run. With it True the mean/std
            # are fitted EXCLUSIVELY on the seed-42 training rows (the SAME
            # indices random_split assigns below), so no local val/test row
            # influences the training transform. compute_zscore_stats returns
            # fresh arrays and never mutates X.
            mean, std = compute_zscore_stats(X, train_split, normalize_train_only)
            X = (X - mean) / std

            X = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(X, y)

            total_size = len(dataset)
            train_size = int(train_split * total_size)
            val_size = int(val_split * total_size)
            test_size = total_size - train_size - val_size

            # The per-client train/val/test split uses a FIXED seed (42),
            # independent of the experiment's own RNG seed argument, so every
            # run partitions each client's local data identically — only
            # model init and attack randomness vary across seeds. This
            # `test_loader` is local-only and is discarded by the caller
            # (flowerfl/client_app.py passes `_` for it); the server-side
            # held-out evaluation uses a separate global holdout set managed
            # by rmc/fixed_eval.py, not this per-client split.
            train_set, val_set, test_set = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Stage-F §4/§5: capture n_orig (the PRE-resampling train row count)
            # and the before-resampling benign/attack split now, BEFORE any SMOTE
            # rebuild replaces train_set. n_orig is the update-matching budget
            # input; the class counts seed the resampling manifest fragment.
            _train_idx0 = train_set.indices
            n_orig = len(_train_idx0)

            # H3 (EMISSION_CONTRACT § 4.2): capture the RAW float64 fingerprint
            # pool from `df` — before the label drop, the float32 cast, the
            # nan_to_num and the Z-score above — over the FIRST P entries of the
            # SAME seed-42 train permutation. No new RNG, no extra parquet pass,
            # and no dependence on any experiment seed. Built once per client per
            # run; ~36 MB at the 100k bound. Off by default: zero work, zero
            # memory when the flag is absent.
            fingerprint_pool = None
            if return_fingerprint_pool:
                from flowerfl.fingerprint_emission import build_fingerprint_pool
                fingerprint_pool = build_fingerprint_pool(
                    df, train_indices=_train_idx0, partition_id=partition_id
                )

            _ytr0 = y[_train_idx0].numpy()
            n_benign_before = int((_ytr0 == 0).sum())
            n_attack_before = int((_ytr0 == 1).sum())
            # Manifest defaults for the SMOTE-off (and skip) paths: after == before.
            n_resampled = n_orig
            n_benign_after, n_attack_after = n_benign_before, n_attack_before
            manifest_status = "off"
            manifest_skip_reason = None
            manifest_k_eff = 0
            manifest_variant = None
            manifest_target = None

            # shuffle=True with no explicit generator draws from torch's
            # GLOBAL RNG at iteration time. That is intentional here: this loader
            # is built once per client at construction (load_data), before the
            # client knows the server round, so a per-(client, round) generator
            # can't be pinned at this call site. Instead, FlowerClient.fit
            # calls seed_everything(derive_seed(base_seed, partition_id,
            # server_round)) immediately before iterating this loader in train,
            # so the shuffle order is deterministic and derives from that
            # fit-scoped global seed. (num_workers=0, so the main-process global
            # RNG is authoritative — no worker-RNG divergence.)
            # SMOTE : oversample the TRAINING split only. Kept strictly
            # gated so the disabled path is byte-identical to the incumbent
            # (train_set flows straight into the DataLoader below). val/test are
            # built from the untouched val_set/test_set, so no synthetic row can
            # reach evaluation.
            if smote_enabled:
                # normalize_smote_target is applied inside _resample_cache_key.
                from flowerfl.smote_resampler import effective_k_neighbors

                train_idx = train_set.indices
                Xtr = X[train_idx].numpy()
                ytr = y[train_idx].numpy()
                n_before = len(train_idx)
                # Full-argument cache key via the single-source-of-truth builder
                # (also used by resample_cache_path for the prewarm durability
                # pass, so they can never drift). It folds in the source-data
                # fingerprint and the CACHE_FORMAT_VERSION.
                cache_key = _resample_cache_key(
                    dataset_name=dataset_name,
                    partition_id=partition_id,
                    batch_size=batch_size,
                    train_split=train_split,
                    val_split=val_split,
                    smote_variant=smote_variant,
                    smote_target=smote_target,
                    smote_seed=smote_seed,
                    smote_semantic_target=smote_semantic_target,
                    normalize_train_only=normalize_train_only,
                )
                Xtr_res, ytr_res, skipped_reason, was_cached = _resample_cached(
                    Xtr, ytr,
                    key=cache_key,
                    variant=smote_variant,
                    target=smote_target,
                    seed=smote_seed,
                    attack_target_policy=smote_semantic_target,
                )
                manifest_variant = smote_variant
                manifest_target = smote_target
                if skipped_reason is None:
                    train_set = TensorDataset(
                        torch.tensor(Xtr_res, dtype=torch.float32),
                        torch.tensor(ytr_res, dtype=torch.long),
                    )
                    n_after = len(train_set)
                    # k is SMOTE-specific (neighbor interpolation); random_over
                    # has no neighbor concept, so report 0 there.
                    k_eff = effective_k_neighbors(ytr) if smote_variant == "smote" else 0
                    status, reason, marker = "applied", "none", ""
                    # Stage-F §5 manifest: post-resampling class split from ytr_res.
                    n_resampled = n_after
                    n_benign_after = int((ytr_res == 0).sum())
                    n_attack_after = int((ytr_res == 1).sum())
                    manifest_status = "applied"
                    manifest_k_eff = k_eff
                else:
                    # Starvation skip (DESIGN.md §6c): training split left
                    # untouched. A skip is the designed outcome, never a crash
                    # that would lose the whole cloud unit.
                    n_after, k_eff = n_before, 0
                    status, reason, marker = "skipped", skipped_reason, "WARNING "
                    # after == before (split untouched); record the skip reason.
                    manifest_status = "skipped"
                    manifest_skip_reason = skipped_reason

                # ONE structured, loud, parseable record per client at data prep
                # (DESIGN.md Stage-D provenance list). The runner
                # (scripts/run_phase4_flower.py::parse_smote_records) lifts these
                # from captured stdout into per-client provenance counts + the
                # run-level skip flag + the reproducibility seed component.
                # Emission gate:
                #   * WORKER path (smote_record_always=False): print on the REAL
                #     resample (a cache MISS) only. On the fleet this runs inside a
                #     Ray actor and run_phase4_flower.py sets log_to_driver=False,
                #     so actor stdout never reaches the driver/CloudWatch anyway —
                #     this is only a LOCAL-visibility backstop, so suppressing the
                #     duplicate lines on hits is pure noise reduction.
                #   * DRIVER PREWARM path (smote_record_always=True): print on
                #     EVERY call, hit OR miss. The prewarm is the
                #     SOLE provenance source on the fleet; if it emitted only on
                #     misses, a second config in a multi-config run (or any run
                #     against a pre-populated disk cache) would be all-hits and
                #     report smote_applied_count=0 while training on resampled
                #     data — false provenance. All record fields are reconstructed
                #     deterministically on a hit (n_before/ytr from the partition,
                #     n_after from the cached arrays), so the emitted record is
                #     byte-identical to the miss-time record modulo the mutable
                #     cache_hits observability token (which parse_smote_records and
                #     _smote_dedupe_key both ignore).
                if smote_record_always or not was_cached:
                    # synthetic = rows ADDED (over-samplers); removed = rows
                    # REMOVED (under-sampler). Exactly one is > 0 on the applied
                    # path; both 0 on a skip. Floored so an under-sampling
                    # record never reports a negative synthetic count.
                    synthetic = max(0, n_after - n_before)
                    removed = max(0, n_before - n_after)
                    print(f"[SMOTE] {marker}client={client_idx} status={status} "
                          f"reason={reason} variant={smote_variant} target={smote_target} "
                          f"k={k_eff} n_before={n_before} n_after={n_after} "
                          f"synthetic={synthetic} removed={removed} "
                          f"seed_component={int(smote_seed)} "
                          f"cache_hits={_resample_cache_hits}")

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size)
            test_loader = DataLoader(test_set, batch_size=batch_size)

            benign_count = (binary_labels == 0).sum()
            attack_count = (binary_labels == 1).sum()
            print(f"[Dataset] Client {client_idx}: {total_size} samples "
                  f"(Benign: {benign_count}, Attack: {attack_count})")

            prep_info = None
            if return_prep_info:
                prep_info = _make_prep_info(
                    n_orig=n_orig, n_resampled=n_resampled,
                    n_benign_before=n_benign_before, n_attack_before=n_attack_before,
                    n_benign_after=n_benign_after, n_attack_after=n_attack_after,
                    status=manifest_status, skip_reason=manifest_skip_reason,
                    k_eff=manifest_k_eff, variant=manifest_variant, target=manifest_target,
                )
            return _load_data_result(
                train_loader, val_loader, test_loader, prep_info, fingerprint_pool,
                return_prep_info, return_fingerprint_pool,
            )

        except Exception as e:
            print(f"[Dataset] Error loading {file_path}: {e}")
            tr, va, te = generate_synthetic_data(partition_id, num_clients, batch_size, dataset_name)
            return _load_data_result(
                tr, va, te,
                _prep_info_from_loader(tr) if return_prep_info else None, None,
                return_prep_info, return_fingerprint_pool,
            )
    else:
        print(f"[Dataset] File not found: {file_path}. Using synthetic data.")
        tr, va, te = generate_synthetic_data(partition_id, num_clients, batch_size, dataset_name)
        return _load_data_result(
            tr, va, te,
            _prep_info_from_loader(tr) if return_prep_info else None, None,
            return_prep_info, return_fingerprint_pool,
        )


def generate_synthetic_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_name: str = "cic"
):
    """Generate synthetic random data as a fallback when real data is unavailable.

    Called by load_data when the expected parquet file is missing or fails
    to load. This exists so client code can be exercised (unit tests, smoke
    runs, CI without the gitignored data/ directory) without crashing —
    accuracy/F1 numbers from this path are meaningless and must never be
    treated as a real experiment result.
    """
    print(f"[Dataset] Generating synthetic data for partition {partition_id}")

    input_shape = detect_input_shape(dataset_name)
    num_samples_total = 1000

    X = torch.randn(num_samples_total, input_shape)
    y = torch.randint(0, 2, (num_samples_total,))

    partition_size = num_samples_total // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else num_samples_total

    X_part = X[start_idx:end_idx]
    y_part = y[start_idx:end_idx]

    num_samples = len(X_part)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    train_dataset = TensorDataset(X_part[:train_size], y_part[:train_size])
    val_dataset = TensorDataset(X_part[train_size:train_size+val_size],
                                 y_part[train_size:train_size+val_size])
    test_dataset = TensorDataset(X_part[train_size+val_size:],
                                  y_part[train_size+val_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_malicious_clients(dataset_name: str, malicious_fraction: float) -> list:
    """Return the client indices designated malicious for a given fraction.

    Takes the first `num_clients * malicious_fraction` entries from the
    dataset config's `malicious_order` list (lower-slot convention — for the
    20-client Edge-IIoT RMC datasets this is simply [0, 1,..., num_malicious-1],
    matching the client_0..client_8 adversary convention used throughout
    scripts/data/generate_scenarios.py's Design D scenarios). This function
    is used by non-scenario-driven experiments (e.g. reproduction scripts);
    scenario-JSON-driven runs instead read malicious assignments directly
    from the scenario's `schedule[].attacks` blocks.
    """
    config = get_dataset_config(dataset_name)
    num_clients = len(config["client_files"])
    num_malicious = int(num_clients * malicious_fraction)
    malicious_order = config.get("malicious_order", list(range(num_clients)))
    return malicious_order[:num_malicious]

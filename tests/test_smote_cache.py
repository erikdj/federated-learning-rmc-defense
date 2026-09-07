"""Process-local SMOTE resample cache (image v8, change 1).

Flower's Ray simulation calls client_fn (hence task.load_data) ~41x/round (20
fit + 21 evaluate). With SMOTE enabled the kNN synthesis re-ran on every call
(~26 min/round overhead measured on EXP-017). The resample is deterministic per
the full argument tuple, so a process-local memo of the resampled TRAIN arrays
is bit-identical by construction.

Load-bearing safety property (the catastrophic observed failure mode in the
runner cache,  R1): a wrong-key cache must NEVER serve a stale arm.
test_cache_miss_on_every_key_flip flips each key component and asserts a MISS.

The SMOTE-off path never touches the cache and stays byte-identical to master.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import flowerfl.task as task_module
from flowerfl.task import load_data

TMP_DATASET = "smote_cache_ds"


@pytest.fixture
def skewed_dataset(tmp_path, monkeypatch):
    """A temp DATASET_CONFIGS entry backed by one skewed parquet partition."""
    rng = np.random.default_rng(0)
    n_maj, n_min, n_feat = 300, 60, 6
    X = np.vstack([rng.normal(0.0, 1.0, (n_maj, n_feat)),
                   rng.normal(4.0, 0.4, (n_min, n_feat))])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(n_feat)}
    cols["Attack_label"] = y
    data_dir = tmp_path / "smote_ds"
    data_dir.mkdir()
    pd.DataFrame(cols).to_parquet(data_dir / "client_0.parquet")
    pd.DataFrame(cols).to_parquet(data_dir / "client_1.parquet")

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[TMP_DATASET] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": ["client_0.parquet", "client_1.parquet"],
        "client_ids": ["0", "1"],
        "num_classes": 2,
        "description": "temp skewed partition for SMOTE cache tests",
        "malicious_order": [0, 1],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(TMP_DATASET, None)
    task_module._reset_resample_cache()
    yield TMP_DATASET
    task_module._reset_resample_cache()


def _train_rows(loader):
    ds = loader.dataset
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([torch.as_tensor(ds[i][1]) for i in range(len(ds))])
    return X, y


_BASE = dict(
    dataset_name=TMP_DATASET, batch_size=32, train_split=0.8, val_split=0.1,
    smote_enabled=True, smote_variant="smote", smote_target="balanced", smote_seed=42,
)


def test_cache_hit_is_bit_identical_and_counts(skewed_dataset):
    """Second identical SMOTE-on construction is a cache HIT yielding
    bit-identical train tensors, and the hit counter advances."""
    tr0, _, _ = load_data(0, **_BASE)
    assert task_module._resample_cache_misses == 1
    assert task_module._resample_cache_hits == 0

    tr1, _, _ = load_data(0, **_BASE)
    assert task_module._resample_cache_hits == 1
    assert task_module._resample_cache_misses == 1  # no new resample

    X0, y0 = _train_rows(tr0)
    X1, y1 = _train_rows(tr1)
    assert torch.equal(X0, X1)
    assert torch.equal(y0, y1)


@pytest.mark.parametrize("flip", [
    {"partition_id": 1},
    {"batch_size": 64},
    {"train_split": 0.7},
    {"val_split": 0.2},
    {"smote_variant": "random_over"},
    {"smote_target": 0.8},
    {"smote_seed": 43},
])
def test_cache_miss_on_every_key_flip(skewed_dataset, flip):
    """Flipping ANY key component forces a cache MISS (never serves a stale arm).
    partition_id is passed positionally; all others are kwargs."""
    task_module._reset_resample_cache()
    load_data(0, **_BASE)  # prime: 1 miss
    assert task_module._resample_cache_misses == 1

    kwargs = dict(_BASE)
    partition = flip.pop("partition_id", 0)
    kwargs.update(flip)
    load_data(partition, **kwargs)
    assert task_module._resample_cache_misses == 2, f"flip {flip} did not miss"
    assert task_module._resample_cache_hits == 0


def test_max_samples_flip_misses(skewed_dataset, monkeypatch):
    """MAX_SAMPLES_PER_CLIENT at call time is part of the key: changing it
    (even on a small partition where the cap never fires) forces a MISS."""
    task_module._reset_resample_cache()
    load_data(0, **_BASE)
    monkeypatch.setattr(task_module, "MAX_SAMPLES_PER_CLIENT", 12345)
    load_data(0, **_BASE)
    assert task_module._resample_cache_misses == 2


def test_off_path_never_touches_cache(skewed_dataset):
    """SMOTE-off construction leaves the cache empty and counters at zero —
    the OFF path is unchanged by the cache layer."""
    task_module._reset_resample_cache()
    load_data(0, dataset_name=TMP_DATASET, batch_size=32)
    load_data(0, dataset_name=TMP_DATASET, batch_size=32, smote_enabled=False)
    assert len(task_module._resample_cache) == 0
    assert task_module._resample_cache_hits == 0
    assert task_module._resample_cache_misses == 0


def test_cache_is_lru_bounded(skewed_dataset):
    """The cache holds at most _RESAMPLE_CACHE_MAXSIZE entries (a Ray actor may
    host many partitions over its lifetime; memory must stay bounded)."""
    task_module._reset_resample_cache()
    for seed in range(task_module._RESAMPLE_CACHE_MAXSIZE + 3):
        kwargs = dict(_BASE)
        kwargs["smote_seed"] = 1000 + seed
        load_data(0, **kwargs)
    assert len(task_module._resample_cache) <= task_module._RESAMPLE_CACHE_MAXSIZE


def test_cache_hit_suppresses_record_line(skewed_dataset, capsys):
    """The [SMOTE] provenance record prints on the real resample (miss) only,
    not on cache hits — noise reduction for the distinct-partition acceptance
    analysis."""
    load_data(0, **_BASE)
    capsys.readouterr()  # drain the miss record
    load_data(0, **_BASE)  # hit
    out = capsys.readouterr().out
    assert "[SMOTE]" not in out


def test_miss_emits_one_applied_record(skewed_dataset, capsys):
    """The miss path still emits exactly one structured [SMOTE] applied record."""
    load_data(0, **_BASE)
    out = capsys.readouterr().out
    assert out.count("[SMOTE]") == 1
    assert "status=applied" in out

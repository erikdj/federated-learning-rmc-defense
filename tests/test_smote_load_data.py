"""Integration tests for SMOTE wired into the FL data pipeline.

Covers the two load-bearing safety properties:
  1. OFF (flag false/absent) is bit-identical to the incumbent load_data output.
  2. ON applies ONLY to the training split — the val/test streams are byte-
     identical to the OFF run (zero synthetic rows leak into evaluation).

Plus client_fn's loud config validation at parse time. Synthetic parquet
partition only; no FL run, no network.
"""
import re

import numpy as np
import pandas as pd
import pytest
import torch

import flowerfl.task as task_module
from flowerfl.task import load_data

TMP_DATASET = "smote_test_ds"


@pytest.fixture
def skewed_dataset(tmp_path, monkeypatch):
    """Register a temp DATASET_CONFIGS entry backed by one skewed parquet file."""
    rng = np.random.default_rng(0)
    n_maj, n_min, n_feat = 300, 60, 6
    X_maj = rng.normal(0.0, 1.0, (n_maj, n_feat))
    X_min = rng.normal(4.0, 0.4, (n_min, n_feat))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(n_feat)}
    cols["Attack_label"] = y
    df = pd.DataFrame(cols)
    data_dir = tmp_path / "smote_ds"
    data_dir.mkdir()
    df.to_parquet(data_dir / "client_0.parquet")

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[TMP_DATASET] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": ["client_0.parquet"],
        "client_ids": ["0"],
        "num_classes": 2,
        "description": "temp skewed partition for SMOTE tests",
        "malicious_order": [0],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(TMP_DATASET, None)
    return TMP_DATASET


def _rows(dataset):
    """Materialize a dataset in index order (shuffle-independent)."""
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y = torch.stack([torch.as_tensor(dataset[i][1]) for i in range(len(dataset))])
    return X, y


def test_off_is_bit_identical_to_incumbent(skewed_dataset):
    """Default call (no smote args) vs explicit smote_enabled=False must produce
    byte-identical train/val/test datasets — the flag is inert when off."""
    tr0, va0, te0 = load_data(0, dataset_name=skewed_dataset, batch_size=32)
    tr1, va1, te1 = load_data(
        0, dataset_name=skewed_dataset, batch_size=32,
        smote_enabled=False, smote_variant="smote", smote_target="balanced", smote_seed=99,
    )
    for a, b in ((tr0, tr1), (va0, va1), (te0, te1)):
        Xa, ya = _rows(a.dataset)
        Xb, yb = _rows(b.dataset)
        assert torch.equal(Xa, Xb)
        assert torch.equal(ya, yb)


def test_on_applies_to_train_only(skewed_dataset):
    """SMOTE ON: val/test byte-identical to OFF (no synthetic leak); train grows
    and is class-balanced."""
    tr_off, va_off, te_off = load_data(0, dataset_name=skewed_dataset, batch_size=32)
    tr_on, va_on, te_on = load_data(
        0, dataset_name=skewed_dataset, batch_size=32,
        smote_enabled=True, smote_variant="smote", smote_target="balanced", smote_seed=42,
    )

    # val/test untouched
    for off, on in ((va_off, va_on), (te_off, te_on)):
        Xa, ya = _rows(off.dataset)
        Xb, yb = _rows(on.dataset)
        assert torch.equal(Xa, Xb), "eval split changed under SMOTE"
        assert torch.equal(ya, yb), "eval labels changed under SMOTE"

    # train grew and is balanced
    _, ytr_off = _rows(tr_off.dataset)
    Xtr_on, ytr_on = _rows(tr_on.dataset)
    assert len(ytr_on) > len(ytr_off), "SMOTE did not add training rows"
    n0 = int((ytr_on == 0).sum())
    n1 = int((ytr_on == 1).sum())
    assert n0 == n1, f"training split not balanced after SMOTE: {n0}/{n1}"


def test_on_train_split_determinism(skewed_dataset):
    """Same smote_seed -> identical training tensors."""
    tr_a, _, _ = load_data(0, dataset_name=skewed_dataset, batch_size=32,
                           smote_enabled=True, smote_seed=7)
    tr_b, _, _ = load_data(0, dataset_name=skewed_dataset, batch_size=32,
                           smote_enabled=True, smote_seed=7)
    Xa, ya = _rows(tr_a.dataset)
    Xb, yb = _rows(tr_b.dataset)
    assert torch.equal(Xa, Xb)
    assert torch.equal(ya, yb)


@pytest.fixture
def single_class_dataset(tmp_path, monkeypatch):
    """A partition whose every row is the benign class — SMOTE must skip, not crash."""
    rng = np.random.default_rng(1)
    n, n_feat = 200, 6
    X = rng.normal(0.0, 1.0, (n, n_feat))
    cols = {f"f{i}": X[:, i] for i in range(n_feat)}
    cols["Attack_label"] = np.zeros(n, dtype=int)
    df = pd.DataFrame(cols)
    data_dir = tmp_path / "single_ds"
    data_dir.mkdir()
    df.to_parquet(data_dir / "client_0.parquet")

    name = "smote_single_ds"
    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[name] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": ["client_0.parquet"],
        "client_ids": ["0"],
        "num_classes": 2,
        "description": "single-class partition for SMOTE starvation test",
        "malicious_order": [0],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(name, None)
    return name


def test_smote_on_single_class_skips_without_crash(single_class_dataset, capsys):
    """SMOTE ON over a single-class partition: no exception, train split left
    untouched (byte-identical to OFF), and a loud parseable WARNING is emitted."""
    tr_off, _, _ = load_data(0, dataset_name=single_class_dataset, batch_size=32)
    tr_on, _, _ = load_data(
        0, dataset_name=single_class_dataset, batch_size=32,
        smote_enabled=True, smote_variant="smote", smote_target="balanced", smote_seed=42,
    )
    Xoff, yoff = _rows(tr_off.dataset)
    Xon, yon = _rows(tr_on.dataset)
    assert torch.equal(Xoff, Xon), "train split changed despite starvation skip"
    assert torch.equal(yoff, yon)
    out = capsys.readouterr().out
    # one structured, loud, parseable per-client record on the skip path
    assert "[SMOTE]" in out
    assert "WARNING" in out
    assert "status=skipped" in out
    assert "reason=single_class" in out


def test_smote_on_emits_applied_record(skewed_dataset, capsys):
    """The applied path emits one structured per-client record with the counts
    and the reproducibility seed component (DESIGN.md Stage-D provenance list)."""
    load_data(
        0, dataset_name=skewed_dataset, batch_size=32,
        smote_enabled=True, smote_variant="smote", smote_target="balanced", smote_seed=1234,
    )
    out = capsys.readouterr().out
    assert "[SMOTE] client=0 status=applied" in out
    assert "variant=smote" in out
    assert "seed_component=1234" in out
    # n_after > n_before and a positive synthetic count are recorded
    m = re.search(r"n_before=(\d+) n_after=(\d+) synthetic=(\d+)", out)
    assert m is not None
    n_before, n_after, synthetic = map(int, m.groups())
    assert n_after > n_before
    assert synthetic == n_after - n_before


def test_random_under_records_removed_not_synthetic(skewed_dataset, capsys):
    """random_under (v8 change 2): the applied record reports synthetic=0 and a
    positive removed count; the train split SHRINKS and stays class-balanced.
    val/test are untouched (byte-identical to OFF)."""
    _, va_off, te_off = load_data(0, dataset_name=skewed_dataset, batch_size=32)
    tr_off, _, _ = load_data(0, dataset_name=skewed_dataset, batch_size=32)
    tr_on, va_on, te_on = load_data(
        0, dataset_name=skewed_dataset, batch_size=32,
        smote_enabled=True, smote_variant="random_under", smote_target="balanced", smote_seed=5,
    )
    out = capsys.readouterr().out
    assert "[SMOTE] client=0 status=applied" in out
    assert "variant=random_under" in out
    m = re.search(r"synthetic=(\d+) removed=(\d+)", out)
    assert m is not None, out
    synthetic, removed = map(int, m.groups())
    assert synthetic == 0
    assert removed > 0

    # eval splits untouched
    for off, on in ((va_off, va_on), (te_off, te_on)):
        Xa, ya = _rows(off.dataset)
        Xb, yb = _rows(on.dataset)
        assert torch.equal(Xa, Xb) and torch.equal(ya, yb)

    # train shrank and is balanced
    _, ytr_off = _rows(tr_off.dataset)
    _, ytr_on = _rows(tr_on.dataset)
    assert len(ytr_on) < len(ytr_off), "under-sampling did not remove rows"
    assert int((ytr_on == 0).sum()) == int((ytr_on == 1).sum())


# ---------------------------------------------------------------------------
# client_fn loud config validation (parse-time)
# ---------------------------------------------------------------------------

def _ctx(run_config):
    from flwr.common import Context
    from flwr.common.record.recorddict import RecordDict
    return Context(
        run_id=1, node_id=1,
        node_config={"partition-id": 0},
        state=RecordDict(),
        run_config=run_config,
    )


def test_client_fn_rejects_unknown_variant(skewed_dataset):
    from flowerfl.client_app import client_fn
    ctx = _ctx({"dataset": skewed_dataset, "smote-enabled": True, "smote-variant": "adasyn"})
    with pytest.raises(ValueError, match="smote_variant"):
        client_fn(ctx)


def test_client_fn_rejects_invalid_target(skewed_dataset):
    from flowerfl.client_app import client_fn
    ctx = _ctx({"dataset": skewed_dataset, "smote-enabled": True, "smote-target": "banana"})
    with pytest.raises(ValueError, match="smote_target"):
        client_fn(ctx)


def test_client_fn_off_ignores_smote_config(skewed_dataset):
    """When disabled, an (irrelevant) variant/target string is not validated
    and the client is built normally."""
    from flowerfl.client_app import client_fn
    ctx = _ctx({"dataset": skewed_dataset, "smote-enabled": False, "smote-variant": "adasyn"})
    client = client_fn(ctx)  # must not raise
    assert client is not None

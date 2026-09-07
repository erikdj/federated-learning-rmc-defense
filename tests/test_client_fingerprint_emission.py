"""FlowerClient fingerprint emission — wiring, off-by-default, RNG isolation.

Design authority
----------------
The public workflow is described in `docs/reproduction/experiments.md`.
Historical protocol references: § 4.1 (emitted in `fit()`
**before** training so a training failure cannot silently suppress it), § 4.2
(the pool is built once, keyed on partition_id only), § 4.3 (arm invariance +
**RNG isolation**, "the one defect in this contract that would corrupt H1/H2
artefacts rather than merely H3's"), § 4.4 (`FitRes.metrics["fingerprint"]`).

The RNG-isolation test here is the regression the contract demands *by name*:
"an existing scenario's per-round training draws are byte-identical with the
fingerprint emission on and off" — not a code review.

Small synthetic parquet partitions only. No FL run, no real `data/` parquet, no
cloud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import flowerfl.task as task_module
from flowerfl.client_app import FlowerClient
from flowerfl.fingerprint import FINGERPRINT_DIM, decode_fingerprint, load_feature_spec
from flowerfl.fingerprint_emission import (
    FINGERPRINT_METRIC_KEY,
    build_fingerprint_pool,
)
from flowerfl.task import create_model, detect_input_shape, load_data

SPEC = load_feature_spec(verify=True)
FEATURES = SPEC.features
TMP_DATASET = "fp_emission_test_ds"
B = 32
N_ROWS = 320


def _partition_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {
        name: rng.normal(loc=float(seed + i), scale=1.0 + 0.1 * i, size=N_ROWS)
        for i, name in enumerate(FEATURES)
    }
    cols["Attack_label"] = np.concatenate(
        [np.zeros(N_ROWS - 60), np.ones(60)]
    ).astype(int)
    return pd.DataFrame(cols)


@pytest.fixture
def fp_dataset(tmp_path, monkeypatch):
    """Two synthetic partitions carrying the locked 45 feature columns."""
    data_dir = tmp_path / "fp_ds"
    data_dir.mkdir()
    for pid in (0, 1):
        _partition_frame(seed=pid).to_parquet(data_dir / f"client_{pid}.parquet")

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[TMP_DATASET] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": ["client_0.parquet", "client_1.parquet"],
        "client_ids": ["0", "1"],
        "num_classes": 2,
        "description": "temp 45-feature partitions for fingerprint emission tests",
        "malicious_order": [0, 1],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(TMP_DATASET, None)
    return TMP_DATASET


def _pool_for(pid: int, seed_offset: int = 0):
    """A pool built directly from a synthetic frame (no parquet round-trip)."""
    df = _partition_frame(seed=pid + seed_offset)
    return build_fingerprint_pool(
        df, train_indices=list(range(256)), partition_id=pid
    )


def _client(ds, *, pool=None, arm="off", partition_id=0, is_malicious=False):
    tr, va, _ = load_data(partition_id, dataset_name=ds, batch_size=B)
    torch.manual_seed(7)  # identical init across the clients we compare
    net = create_model(ds, detect_input_shape(ds))
    return FlowerClient(
        tr, va, net,
        partition_id=partition_id,
        is_malicious=is_malicious,
        use_brfss=False,
        local_epochs=1,
        arm_label=arm,
        base_seed=42,
        fingerprint_pool=pool,
        m_fp=400,
    )


# ---------------------------------------------------------------------------
# OFF BY DEFAULT — the incumbent path must be byte-identical
# ---------------------------------------------------------------------------

def test_no_fingerprint_key_when_no_pool_is_configured(fp_dataset):
    client = _client(fp_dataset)
    _, _, metrics = client.fit(client.get_parameters({}), {"server_round": 1})
    assert set(metrics) == {"train_loss", "partition_id"}
    assert FINGERPRINT_METRIC_KEY not in metrics


def test_flower_client_defaults_to_no_fingerprint_pool(fp_dataset):
    tr, va, _ = load_data(0, dataset_name=fp_dataset, batch_size=B)
    net = create_model(fp_dataset, detect_input_shape(fp_dataset))
    client = FlowerClient(tr, va, net, partition_id=0)
    assert client.fingerprint_pool is None
    _, _, metrics = client.fit(client.get_parameters({}), {"server_round": 1})
    assert FINGERPRINT_METRIC_KEY not in metrics


# ---------------------------------------------------------------------------
# § 4.1 / § 4.4 — emission
# ---------------------------------------------------------------------------

def test_fit_emits_a_decodable_180_dim_fingerprint(fp_dataset):
    client = _client(fp_dataset, pool=_pool_for(0))
    _, _, metrics = client.fit(client.get_parameters({}), {"server_round": 1})

    assert FINGERPRINT_METRIC_KEY in metrics
    vec = decode_fingerprint(metrics[FINGERPRINT_METRIC_KEY], expected_dim=FINGERPRINT_DIM)
    assert vec.shape == (180,)
    assert np.all(np.isfinite(vec))


def test_emitted_fingerprint_varies_across_rounds(fp_dataset):
    client = _client(fp_dataset, pool=_pool_for(0))
    params = client.get_parameters({})
    payloads = [
        client.fit(params, {"server_round": r})[2][FINGERPRINT_METRIC_KEY]
        for r in (1, 2, 3)
    ]
    assert len(set(payloads)) == 3


def test_emission_happens_before_training(fp_dataset, monkeypatch):
    """§ 4.1: 'before training (so a training failure cannot silently suppress
    the emission)'."""
    import flowerfl.client_app as client_app

    order: list[str] = []
    real_payload = client_app.compute_round_fingerprint_payload

    def _spy_payload(*args, **kwargs):
        order.append("fingerprint")
        return real_payload(*args, **kwargs)

    def _boom(*args, **kwargs):
        order.append("train")
        raise RuntimeError("training blew up")

    monkeypatch.setattr(client_app, "compute_round_fingerprint_payload", _spy_payload)
    monkeypatch.setattr(client_app, "train", _boom)

    client = _client(fp_dataset, pool=_pool_for(0))
    with pytest.raises(RuntimeError, match="training blew up"):
        client.fit(client.get_parameters({}), {"server_round": 1})
    assert order == ["fingerprint", "train"]


def test_emission_failure_fails_the_fit_loudly(fp_dataset, monkeypatch):
    """Gate (e) is 100% emission: a fingerprint that cannot be computed must
    fail the unit, never emit nothing and carry on."""
    import flowerfl.client_app as client_app
    from flowerfl.fingerprint_emission import FingerprintEmissionError

    def _boom(*args, **kwargs):
        raise FingerprintEmissionError("pool is unusable")

    monkeypatch.setattr(client_app, "compute_round_fingerprint_payload", _boom)
    client = _client(fp_dataset, pool=_pool_for(0))
    with pytest.raises(FingerprintEmissionError):
        client.fit(client.get_parameters({}), {"server_round": 1})


# ---------------------------------------------------------------------------
# § 4.3 — ARM INVARIANCE
# ---------------------------------------------------------------------------

def test_fingerprint_is_byte_identical_across_defense_arms(fp_dataset):
    """§ 4.3 / § 6.2: 'the same (scenario, seed) under tgefp and krumtgefp must
    produce BYTE-IDENTICAL fingerprints for every client-round'.

    Emission is a pure function of (partition_id, base_seed, server_round); it
    reads no model, no server state, no defense config and no enforcement
    outcome — so a different arm, a different attack branch and a different
    model cannot move it.
    """
    tge = _client(fp_dataset, pool=_pool_for(0), arm="tgefp")
    krumtge = _client(
        fp_dataset, pool=_pool_for(0), arm="krumtgefp", is_malicious=True
    )

    for server_round in (1, 2, 7):
        a = tge.fit(tge.get_parameters({}), {"server_round": server_round})[2]
        b = krumtge.fit(
            krumtge.get_parameters({}),
            {"server_round": server_round, "attack_type": "gaussian_noise"},
        )[2]
        assert a[FINGERPRINT_METRIC_KEY] == b[FINGERPRINT_METRIC_KEY]


# ---------------------------------------------------------------------------
# § 4.3 — RNG ISOLATION REGRESSION (the H1/H2-protecting test)
# ---------------------------------------------------------------------------

def _fit_and_capture(client, server_round: int):
    """Run one fit() and return (trained params, global numpy state, torch state)."""
    params = client.get_parameters({})
    out_params, _, _ = client.fit(params, {"server_round": server_round})
    return out_params, np.random.get_state(), torch.get_rng_state().clone()


def test_training_draws_are_byte_identical_with_emission_on_and_off(fp_dataset):
    """THE regression the contract demands by name (§ 4.3, § 7.4).

    If the fingerprint draw ever consumed the global NumPy/torch RNG, every
    subsequent dropout mask and DataLoader shuffle would shift and every sealed
    H1/H2 run would stop reproducing — silently.
    """
    off = _client(fp_dataset, pool=None)
    on = _client(fp_dataset, pool=_pool_for(0))

    off_params, off_np, off_torch = _fit_and_capture(off, 5)
    on_params, on_np, on_torch = _fit_and_capture(on, 5)

    # 1. The trained weights — the observable consequence of every training draw.
    assert len(off_params) == len(on_params)
    for a, b in zip(off_params, on_params):
        np.testing.assert_array_equal(a, b)

    # 2. The global RNG states themselves, after fit().
    assert off_np[0] == on_np[0]
    np.testing.assert_array_equal(off_np[1], on_np[1])
    assert off_np[2:] == on_np[2:]
    assert torch.equal(off_torch, on_torch)


def test_repeated_emission_does_not_drift_the_global_rng(fp_dataset):
    """Even many emissions in a row must leave the training stream untouched."""
    client = _client(fp_dataset, pool=_pool_for(0))
    params = client.get_parameters({})

    np.random.seed(2024)
    torch.manual_seed(2024)
    for r in range(1, 6):
        client.fit(params, {"server_round": r})
    seq_np, seq_torch = np.random.get_state(), torch.get_rng_state().clone()

    off = _client(fp_dataset, pool=None)
    np.random.seed(2024)
    torch.manual_seed(2024)
    for r in range(1, 6):
        off.fit(params, {"server_round": r})

    after_np = np.random.get_state()
    assert seq_np[0] == after_np[0]
    np.testing.assert_array_equal(seq_np[1], after_np[1])
    assert torch.equal(seq_torch, torch.get_rng_state())


# ---------------------------------------------------------------------------
# § 4.2 — the pool through the real data path
# ---------------------------------------------------------------------------

def test_load_data_pool_is_seed_invariant(fp_dataset):
    """§ 4.2 keying: the pool depends on partition_id ONLY, never on base_seed."""
    from flowerfl.seeding import derive_seed

    _, _, _, pool_a = load_data(
        0, dataset_name=fp_dataset, batch_size=B,
        smote_seed=derive_seed(42, 0), return_fingerprint_pool=True,
    )
    _, _, _, pool_b = load_data(
        0, dataset_name=fp_dataset, batch_size=B,
        smote_seed=derive_seed(43, 0), return_fingerprint_pool=True,
    )
    np.testing.assert_array_equal(pool_a.values, pool_b.values)
    assert pool_a.features == pool_b.features == FEATURES


def test_load_data_pool_is_partition_keyed(fp_dataset):
    _, _, _, pool_0 = load_data(
        0, dataset_name=fp_dataset, batch_size=B, return_fingerprint_pool=True
    )
    _, _, _, pool_1 = load_data(
        1, dataset_name=fp_dataset, batch_size=B, return_fingerprint_pool=True
    )
    assert pool_0.partition_id == 0 and pool_1.partition_id == 1
    assert not np.array_equal(pool_0.values, pool_1.values)


def test_load_data_pool_is_raw_not_the_normalised_float32_tensor(fp_dataset):
    """§ 2.3: the pool must NOT be the Z-scored float32 training tensor (whose
    per-client mean is ~0 and std ~1 for EVERY device by construction)."""
    train_loader, _, _, pool = load_data(
        0, dataset_name=fp_dataset, batch_size=B, return_fingerprint_pool=True
    )
    assert pool.values.dtype == np.float64
    # Column means track the raw synthetic loc values (0, 1, 2, ...), not ~0.
    col_means = pool.values.mean(axis=0)
    assert abs(col_means[10] - 10.0) < 1.0
    assert abs(col_means[40] - 40.0) < 1.0


def test_load_data_pool_is_the_train_split_only(fp_dataset):
    _, _, _, pool = load_data(
        0, dataset_name=fp_dataset, batch_size=B, return_fingerprint_pool=True
    )
    assert pool.n_train == int(0.8 * N_ROWS)
    assert len(pool) == pool.n_train


def test_load_data_without_the_flag_returns_the_incumbent_arity(fp_dataset):
    out = load_data(0, dataset_name=fp_dataset, batch_size=B)
    assert len(out) == 3
    out4 = load_data(0, dataset_name=fp_dataset, batch_size=B, return_prep_info=True)
    assert len(out4) == 4


def test_load_data_pool_comes_after_prep_info_when_both_requested(fp_dataset):
    from flowerfl.fingerprint_emission import FingerprintPool

    out = load_data(
        0, dataset_name=fp_dataset, batch_size=B,
        return_prep_info=True, return_fingerprint_pool=True,
    )
    assert len(out) == 5
    assert isinstance(out[3], dict) and "n_orig" in out[3]
    assert isinstance(out[4], FingerprintPool)


# ---------------------------------------------------------------------------
# client_fn gating
# ---------------------------------------------------------------------------

def _ctx(run_config, partition_id=0):
    from flwr.common import Context
    from flwr.common.record.recorddict import RecordDict

    return Context(
        run_id=1, node_id=1,
        node_config={"partition-id": partition_id},
        state=RecordDict(),
        run_config=run_config,
    )


def test_client_fn_off_by_default_builds_no_pool(fp_dataset):
    from flowerfl.client_app import client_fn

    client = client_fn(_ctx({"dataset": fp_dataset}))
    assert client.numpy_client.fingerprint_pool is None


def test_client_fn_builds_the_pool_when_enabled(fp_dataset):
    from flowerfl.client_app import client_fn
    from flowerfl.fingerprint_emission import FingerprintPool

    client = client_fn(_ctx({"dataset": fp_dataset, "fingerprint-enabled": True}))
    pool = client.numpy_client.fingerprint_pool
    assert isinstance(pool, FingerprintPool)
    assert pool.partition_id == 0


def test_client_fn_raises_loudly_when_the_pool_cannot_be_built(tmp_path, monkeypatch):
    """Gate (e): a fingerprint-enabled run whose partition falls back to
    synthetic data must FAIL, not run silently without emission."""
    from flowerfl.client_app import client_fn

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg["fp_missing_ds"] = {
        "data_dir": str(tmp_path / "nope"),
        "label_column": "Attack_label",
        "client_files": ["client_0.parquet"],
        "client_ids": ["0"],
        "num_classes": 2,
        "description": "missing partition",
        "malicious_order": [0],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)

    ctx = _ctx({"dataset": "fp_missing_ds", "fingerprint-enabled": True})
    with pytest.raises(RuntimeError, match="fingerprint"):
        client_fn(ctx)

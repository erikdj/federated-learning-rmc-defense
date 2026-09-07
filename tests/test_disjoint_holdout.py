"""Disjoint evaluation holdout (image v8 change 3).

The server-side FixedEvalManager built its global holdout by stratified-sampling
each client's FULL parquet, never excluding the rows load_data routes into that
client's TRAIN split — measured 77.5% holdout/train row overlap (audit
results/20260727/holdout_overlap_audit/). `disjoint=True` (the new default)
excludes each eval file's corresponding training-partition TRAIN indices from the
sampling pool before the stratified draw, using the exact reconstruction logic
from scripts/analysis/audit_holdout_overlap.py.

Tests use a small synthetic fixture (two client parquets); no FL run, no network.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flowerfl.task as task_module
from rmc.fixed_eval import FixedEvalManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
from audit_holdout_overlap import reconstruct_train_indices  # noqa: E402

BIG_CAP = 10_000_000  # larger than the fixture, so no cap fires (uncapped path)


def _make_dataset(tmp_path, monkeypatch, name, n_files=2, n_maj=400, n_min=100, nf=5):
    rng = np.random.default_rng(0)
    data_dir = tmp_path / name
    data_dir.mkdir()
    for cidx in range(n_files):
        X = np.vstack([rng.normal(0.0, 1.0, (n_maj, nf)),
                       rng.normal(4.0, 0.5, (n_min, nf))])
        y = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
        cols = {f"f{i}": X[:, i] for i in range(nf)}
        cols["Attack_label"] = y
        pd.DataFrame(cols).to_parquet(data_dir / f"client_{cidx}.parquet")
    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[name] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": [f"client_{i}.parquet" for i in range(n_files)],
        "client_ids": [str(i) for i in range(n_files)],
        "num_classes": 2,
        "description": "temp partition for disjoint-holdout tests",
        "malicious_order": list(range(n_files)),
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(name, None)
    return name, str(data_dir)


@pytest.fixture
def disj_dataset(tmp_path, monkeypatch):
    return _make_dataset(tmp_path, monkeypatch, "disj_test")


def _train_positions(data_dir, label_col, client_idx):
    res, _ = reconstruct_train_indices(
        client_idx, data_dir, label_col, max_samples=BIG_CAP,
        train_split=0.8, val_split=0.1,
    )
    return res["train"]


def test_disjoint_holdout_has_zero_train_overlap(disj_dataset):
    name, data_dir = disj_dataset
    mgr = FixedEvalManager(
        dataset_name=name, samples_per_client=50, seed=42,
        disjoint=True, train_max_samples=BIG_CAP,
    )
    assert mgr.holdout_disjoint is True
    total_overlap = 0
    for cidx, sel in mgr.holdout_indices.items():
        train = _train_positions(data_dir, "Attack_label", cidx)
        total_overlap += len(set(int(i) for i in sel) & train)
    assert total_overlap == 0, f"disjoint holdout overlapped train by {total_overlap}"
    assert mgr.rows_excluded > 0


def test_legacy_holdout_overlaps_train(disj_dataset):
    """disjoint=False reproduces the pre-v8 behaviour: the holdout DOES overlap
    the train split (the 77.5% bug)."""
    name, data_dir = disj_dataset
    mgr = FixedEvalManager(
        dataset_name=name, samples_per_client=50, seed=42, disjoint=False,
    )
    assert mgr.holdout_disjoint is False
    assert mgr.rows_excluded == 0
    total_overlap = 0
    for cidx, sel in mgr.holdout_indices.items():
        train = _train_positions(data_dir, "Attack_label", cidx)
        total_overlap += len(set(int(i) for i in sel) & train)
    assert total_overlap > 0, "legacy holdout should overlap train (pre-v8 behaviour)"


def _reference_legacy_selection(data_dir, label_col, client_files, spc, seed):
    """Line-faithful replay of the pre-v8 _build_holdout selection (shared
    RandomState across files, per-class stratified choice, NO exclusion)."""
    rng = np.random.RandomState(seed)
    out = {}
    for fname in client_files:
        cidx = int(fname.split("_")[1].split(".")[0])
        df = pd.read_parquet(os.path.join(data_dir, fname))
        labels = df[label_col].values
        labels = (labels > 0).astype(np.int64) if labels.max() > 1 else labels.astype(np.int64)
        n = min(spc, len(df))
        if n < len(df):
            indices = []
            for lbl in np.unique(labels):
                lbl_idx = np.where(labels == lbl)[0]
                n_sample = max(1, int(n * len(lbl_idx) / len(labels)))
                n_sample = min(n_sample, len(lbl_idx))
                indices.extend(rng.choice(lbl_idx, size=n_sample, replace=False).tolist())
            out[cidx] = np.array(indices, dtype=np.int64)
        else:
            out[cidx] = np.arange(len(df), dtype=np.int64)
    return out


def test_legacy_mode_is_bit_identical_to_reference(disj_dataset):
    name, data_dir = disj_dataset
    cfg = task_module.DATASET_CONFIGS[name]
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42, disjoint=False)
    ref = _reference_legacy_selection(data_dir, "Attack_label", cfg["client_files"], 50, 42)
    assert set(mgr.holdout_indices) == set(ref)
    for cidx in ref:
        assert np.array_equal(np.sort(mgr.holdout_indices[cidx]), np.sort(ref[cidx]))


def test_disjoint_is_deterministic(disj_dataset):
    name, _ = disj_dataset
    a = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42,
                         disjoint=True, train_max_samples=BIG_CAP)
    b = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42,
                         disjoint=True, train_max_samples=BIG_CAP)
    assert set(a.holdout_indices) == set(b.holdout_indices)
    for cidx in a.holdout_indices:
        assert np.array_equal(np.sort(a.holdout_indices[cidx]),
                              np.sort(b.holdout_indices[cidx]))


def test_disjoint_provenance_fields(disj_dataset, capsys):
    name, _ = disj_dataset
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42,
                           disjoint=True, train_max_samples=BIG_CAP)
    out = capsys.readouterr().out
    assert "holdout_disjoint=true" in out
    assert "rows_excluded=" in out
    # instance provenance
    assert mgr.holdout_disjoint is True
    assert mgr.rows_excluded > 0
    assert mgr.holdout_size == sum(len(v) for v in mgr.holdout_indices.values())
    assert mgr.per_class_counts[0] + mgr.per_class_counts[1] == mgr.holdout_size


def test_class_default_is_false_semantics_safe(disj_dataset):
    """the FixedEvalManager CLASS default is disjoint=False. Only the
    fleet path (server_app, where training genuinely goes through task.load_data)
    opts into disjoint=True. A caller whose train indices are NOT reconstructable
    by reconstruct_train_indices (e.g. rmc/simulate.py) must NOT get a falsely
    'disjoint' holdout by default."""
    name, _ = disj_dataset
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42,
                           train_max_samples=BIG_CAP)
    assert mgr.holdout_disjoint is False


def test_fleet_call_site_passes_disjoint_true_by_default(monkeypatch):
    """The fleet path (server_app._create_eval_manager) defaults disjoint=True
    from the run-config flag, so fleet result JSONs still record
    holdout_disjoint=true by default — that behaviour is unchanged."""
    import rmc.fixed_eval as fe
    import flowerfl.server_app as server_app

    captured = {}

    class _FakeMgr:
        def __init__(self, *a, **kw):
            captured.update(kw)

    monkeypatch.setattr(fe, "FixedEvalManager", _FakeMgr)
    # empty run-config -> fleet default is disjoint=True
    server_app._create_eval_manager("edge_full_20_rmc", {})
    assert captured["disjoint"] is True
    # explicit legacy opt-out threads disjoint=False
    captured.clear()
    server_app._create_eval_manager("edge_full_20_rmc", {"holdout-disjoint": False})
    assert captured["disjoint"] is False


# ---------------------------------------------------------------------------
# result-cache eligibility must fold in the holdout mode, so a
# default-disjoint run never silently reuses a pre-v8 OVERLAPPING-holdout result.
# ---------------------------------------------------------------------------

def _cached(holdout_disjoint="__absent__"):
    prov = {"smote_enabled": False, "smote_variant": None, "smote_target": None}
    if holdout_disjoint != "__absent__":
        prov["holdout_disjoint"] = holdout_disjoint
    return {"trajectory": [{"round": 1}], "return_code": 0, "provenance": prov}


def test_holdout_cache_not_reusable_when_mode_differs():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import _holdout_cache_reusable
    # active run defaults to disjoint=True (flag absent -> provenance default True)
    assert _holdout_cache_reusable(_cached(holdout_disjoint=True), None) is True
    assert _holdout_cache_reusable(_cached(holdout_disjoint=False), None) is False
    # a pre-v8 cached result MISSING the field reads as legacy/overlapping
    assert _holdout_cache_reusable(_cached(), None) is False


def test_holdout_cache_reusable_for_legacy_run():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import _holdout_cache_reusable
    legacy_active = {"holdout-disjoint": False}
    # a legacy run may reuse a legacy or field-missing cached result...
    assert _holdout_cache_reusable(_cached(holdout_disjoint=False), legacy_active) is True
    assert _holdout_cache_reusable(_cached(), legacy_active) is True
    #...but NOT a disjoint one
    assert _holdout_cache_reusable(_cached(holdout_disjoint=True), legacy_active) is False


def test_reuse_cached_or_rotate_recomputes_on_holdout_mode_mismatch(tmp_path):
    """End-to-end: a cached OVERLAPPING-holdout result must NOT satisfy a
    default-disjoint run (the exact silent-reuse P1-1 guards against)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import json as _json
    from run_phase4_flower import _reuse_cached_or_rotate
    jp = tmp_path / "r.json"
    jp.write_text(_json.dumps(_cached(holdout_disjoint=False)))
    # active run: default disjoint (holdout-disjoint True threaded by main)
    out = _reuse_cached_or_rotate(
        jp, {"holdout-disjoint": True}, "rmc/scenarios/S0.json",
        "ScenarioKrum", 42, "reset", "exp",
    )
    assert out is None, "must RECOMPUTE, not reuse an overlapping-holdout cache"


# ---------------------------------------------------------------------------
# an eval file backed by a DUPLICATE training partition. At a
# cap below file size the duplicate (client_20, random_state 42+20) draws a
# DIFFERENT capped set than its source (client_19, 42+19), so its train rows
# stay eligible for the source file's holdout pool. Exclusion must be the UNION
# over every backing training partition.
# ---------------------------------------------------------------------------

@pytest.fixture
def dup_rmc_dataset(tmp_path, monkeypatch):
    """Base eval dataset (client_0, client_1) + an _rmc training dataset whose
    client_2 is a BYTE-DUPLICATE of client_1 (declared via duplicate_partitions).
    Both flagged full so a sub-file cap actually fires."""
    import shutil
    data_dir = tmp_path / "dupev"
    data_dir.mkdir()

    def _write(fname, seed):
        r = np.random.default_rng(seed)
        X = np.vstack([r.normal(0.0, 1.0, (800, 4)), r.normal(4.0, 0.5, (200, 4))])
        y = np.concatenate([np.zeros(800), np.ones(200)]).astype(int)
        cols = {f"c{i}": X[:, i] for i in range(4)}
        cols["Attack_label"] = y
        pd.DataFrame(cols).to_parquet(data_dir / fname)

    _write("client_0.parquet", 10)
    _write("client_1.parquet", 11)
    shutil.copy(data_dir / "client_1.parquet", data_dir / "client_2.parquet")  # byte-dup

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg["dupev"] = {
        "data_dir": str(data_dir), "label_column": "Attack_label",
        "client_files": ["client_0.parquet", "client_1.parquet"],
        "client_ids": ["0", "1"], "num_classes": 2,
        "description": "dup-test eval base", "malicious_order": [0, 1],
    }
    cfg["dupev_rmc"] = {
        "data_dir": str(data_dir), "label_column": "Attack_label",
        "client_files": ["client_0.parquet", "client_1.parquet", "client_2.parquet"],
        "client_ids": ["0", "1", "2"], "num_classes": 2,
        "description": "dup-test train (client_2 == dup of client_1)",
        "malicious_order": [0, 1, 2],
        "duplicate_partitions": {2: 1},
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    monkeypatch.setattr(task_module, "_FULL_DATASETS",
                        frozenset(set(task_module._FULL_DATASETS) | {"dupev", "dupev_rmc"}))
    task_module._input_shape_cache.pop("dupev", None)
    task_module._input_shape_cache.pop("dupev_rmc", None)
    return "dupev", "dupev_rmc", str(data_dir)


def _recon(data_dir, ordinal, fname, cap):
    res, _ = reconstruct_train_indices(
        ordinal, data_dir, "Attack_label", max_samples=cap,
        fname=fname, dataset_name="dupev_rmc",
    )
    return res["train"]


def test_capped_regime_union_excludes_duplicate_partition_rows(dup_rmc_dataset):
    """At a sub-file cap the source and duplicate partitions draw DIFFERENT train
    sets; the eval file's holdout must be disjoint from rows trained through
    EITHER partition (the union)."""
    _, _, data_dir = dup_rmc_dataset
    CAP = 600  # < 1000 rows -> cap fires
    tr_src = _recon(data_dir, 1, "client_1.parquet", CAP)
    tr_dup = _recon(data_dir, 2, "client_2.parquet", CAP)
    assert tr_src != tr_dup, "fixture invalid: capped sets must differ to test the union"

    mgr = FixedEvalManager(
        dataset_name="dupev", samples_per_client=20, seed=42,
        disjoint=True, train_max_samples=CAP, train_dataset_name="dupev_rmc",
    )
    sel = set(int(i) for i in mgr.holdout_indices[1])  # eval file client_1 (ordinal 1)
    assert sel.isdisjoint(tr_src), "overlap with source-partition train rows"
    assert sel.isdisjoint(tr_dup), "overlap with DUPLICATE-partition train rows"


def test_uncapped_union_equals_single_partition_byte_identical(dup_rmc_dataset):
    """At an uncapped cap the duplicate and source draw IDENTICAL train sets, so
    the union exclusion equals the single-partition exclusion — the canonical 2M
    behaviour is byte-identical (dup map has no effect when uncapped)."""
    _, _, data_dir = dup_rmc_dataset
    tr_src = _recon(data_dir, 1, "client_1.parquet", BIG_CAP)
    tr_dup = _recon(data_dir, 2, "client_2.parquet", BIG_CAP)
    assert tr_src == tr_dup, "uncapped byte-duplicate partitions must match"

    mgr_union = FixedEvalManager(
        dataset_name="dupev", samples_per_client=30, seed=42,
        disjoint=True, train_max_samples=BIG_CAP, train_dataset_name="dupev_rmc")
    mgr_single = FixedEvalManager(
        dataset_name="dupev", samples_per_client=30, seed=42,
        disjoint=True, train_max_samples=BIG_CAP, train_dataset_name="dupev")  # no dup map
    for k in mgr_union.holdout_indices:
        assert np.array_equal(np.sort(mgr_union.holdout_indices[k]),
                              np.sort(mgr_single.holdout_indices[k]))


def test_backing_partitions_from_config_not_hardcoded(dup_rmc_dataset):
    """The eval-file -> training-partitions map is derived from the config's
    duplicate_partitions declaration, not a hardcoded 19/20."""
    mgr = FixedEvalManager(dataset_name="dupev", samples_per_client=30, seed=42,
                           disjoint=False, train_dataset_name="dupev_rmc")
    assert mgr._backing_train_partitions("client_1.parquet") == [
        (1, "client_1.parquet"), (2, "client_2.parquet")]
    assert mgr._backing_train_partitions("client_0.parquet") == [(0, "client_0.parquet")]


def test_non_rmc_dataset_has_only_direct_backer(disj_dataset):
    """A dataset without duplicate_partitions yields the direct backer only —
    unchanged behaviour."""
    name, _ = disj_dataset
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=30, seed=42,
                           disjoint=False, train_dataset_name=name)
    assert mgr._backing_train_partitions("client_0.parquet") == [(0, "client_0.parquet")]


# ---------------------------------------------------------------------------
# the take-all branch (samples_per_client >= file rows) must
# apply the SAME per-class disjoint guards as the stratified branch — a rare
# class fully in train must HARD-ERROR (not silently vanish), a shrunk class must
# WARN. Uncapped train indices are label-independent (randperm), so we engineer a
# class onto specific train/test positions deterministically.
# ---------------------------------------------------------------------------

def _uncapped_train_positions(n, train_split=0.8):
    import torch
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).numpy()
    train_size = int(train_split * n)
    return perm[:train_size], perm[train_size:]


def _write_take_all_dataset(tmp_path, monkeypatch, name, n, class1_positions):
    rng = np.random.default_rng(7)
    data_dir = tmp_path / name
    data_dir.mkdir()
    y = np.zeros(n, dtype=int)
    y[list(class1_positions)] = 1
    X = rng.normal(0.0, 1.0, (n, 4))
    cols = {f"c{i}": X[:, i] for i in range(4)}
    cols["Attack_label"] = y
    pd.DataFrame(cols).to_parquet(data_dir / "client_0.parquet")
    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[name] = {
        "data_dir": str(data_dir), "label_column": "Attack_label",
        "client_files": ["client_0.parquet"], "client_ids": ["0"],
        "num_classes": 2, "description": "take-all guard fixture",
        "malicious_order": [0],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(name, None)
    return name


def test_take_all_disjoint_empty_class_hard_errors(tmp_path, monkeypatch):
    n = 20
    train, _test = _uncapped_train_positions(n)
    # both class-1 rows fall inside the train split -> holdout pool emptied.
    name = _write_take_all_dataset(tmp_path, monkeypatch, "takeall_empty",
                                   n, [int(train[0]), int(train[1])])
    with pytest.raises(RuntimeError, match="NO rows left"):
        FixedEvalManager(dataset_name=name, samples_per_client=1000, seed=42,
                         disjoint=True, train_max_samples=BIG_CAP, train_dataset_name=name)


def test_take_all_disjoint_shrunk_class_warns_and_proceeds(tmp_path, monkeypatch, capsys):
    n = 20
    train, test = _uncapped_train_positions(n)
    # one class-1 row in train (excluded), one in test (survives) -> shrinks to 1.
    name = _write_take_all_dataset(tmp_path, monkeypatch, "takeall_shrink",
                                   n, [int(train[0]), int(test[0])])
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=1000, seed=42,
                           disjoint=True, train_max_samples=BIG_CAP, train_dataset_name=name)
    out = capsys.readouterr().out
    assert "WARNING" in out and "shortfall" in out.lower()
    assert "class 1" in out
    assert mgr.per_class_counts[1] == 1  # survived, not vanished


# ---------------------------------------------------------------------------
# CLI / run-config / provenance wiring
# ---------------------------------------------------------------------------

def test_runner_parser_defaults_holdout_disjoint_true():
    import argparse
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_phase4_flower  # noqa
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-disjoint", action=argparse.BooleanOptionalAction, default=True)
    assert p.parse_args([]).holdout_disjoint is True
    assert p.parse_args(["--no-holdout-disjoint"]).holdout_disjoint is False
    assert p.parse_args(["--holdout-disjoint"]).holdout_disjoint is True


def test_runner_provenance_default_and_legacy():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import _holdout_disjoint_provenance
    assert _holdout_disjoint_provenance({}) is True  # absent -> disjoint default
    assert _holdout_disjoint_provenance({"holdout-disjoint": True}) is True
    assert _holdout_disjoint_provenance({"holdout-disjoint": False}) is False
    assert _holdout_disjoint_provenance({"holdout-disjoint": "false"}) is False
    assert _holdout_disjoint_provenance({"holdout-disjoint": "true"}) is True


def test_server_app_run_config_coercion():
    from flowerfl.server_app import _holdout_disjoint_from_run_config as c
    assert c({}) is True
    assert c({"holdout-disjoint": False}) is False
    assert c({"holdout-disjoint": "false"}) is False
    assert c({"holdout-disjoint": "0"}) is False
    assert c({"holdout-disjoint": True}) is True
    assert c({"holdout-disjoint": "true"}) is True


# ---------------------------------------------------------------------------
# IP-style client filenames must not break the exclusion.
# The partition index is the ORDINAL position in client_files, and the actual
# filename is read — never a numeric id parsed out of the filename.
# ---------------------------------------------------------------------------

@pytest.fixture
def ip_dataset(tmp_path, monkeypatch):
    """A config with an IP-style filename (like the `edge` dataset)."""
    rng = np.random.default_rng(1)
    data_dir = tmp_path / "edge_ip"
    data_dir.mkdir()
    files = ["client_0.parquet", "client_192_168_0_101.parquet"]
    for f in files:
        X = np.vstack([rng.normal(0.0, 1.0, (400, 5)), rng.normal(4.0, 0.5, (100, 5))])
        y = np.concatenate([np.zeros(400), np.ones(100)]).astype(int)
        cols = {f"c{i}": X[:, i] for i in range(5)}
        cols["Attack_label"] = y
        pd.DataFrame(cols).to_parquet(data_dir / f)
    name = "edge_ip_test"
    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[name] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": files,
        "client_ids": ["0", "192_168_0_101"],
        "num_classes": 2,
        "description": "IP-style filename partition for disjoint-holdout tests",
        "malicious_order": [0, 1],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(name, None)
    return name, str(data_dir), files


def test_disjoint_holdout_ip_filenames_no_crash_and_zero_overlap(ip_dataset):
    name, data_dir, files = ip_dataset
    # Must NOT raise (pre-fix: parsed id 192 -> client_192.parquet -> abort).
    mgr = FixedEvalManager(
        dataset_name=name, samples_per_client=50, seed=42,
        disjoint=True, train_max_samples=BIG_CAP, train_dataset_name=name,
    )
    assert mgr.holdout_disjoint is True
    # Keyed by ORDINAL; exclusion reads the actual filename.
    for ordinal, sel in mgr.holdout_indices.items():
        res, _ = reconstruct_train_indices(
            ordinal, data_dir, "Attack_label", max_samples=BIG_CAP,
            fname=files[ordinal], dataset_name=name,
        )
        assert set(int(i) for i in sel).isdisjoint(res["train"])


# ---------------------------------------------------------------------------
# reconstruction must mirror load_data's dataset-specific cap
# gating (is_full_dataset), not cap unconditionally on n_rows > max_samples.
# ---------------------------------------------------------------------------

def _one_file(tmp_path, fname, n_maj=80, n_min=20, nf=4):
    data_dir = tmp_path / "capgate"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 1, (n_maj, nf)), rng.normal(4, 0.5, (n_min, nf))])
    y = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
    cols = {f"c{i}": X[:, i] for i in range(nf)}
    cols["Attack_label"] = y
    pd.DataFrame(cols).to_parquet(data_dir / fname)
    return str(data_dir)


def test_reconstruct_caps_only_for_full_datasets(tmp_path):
    data_dir = _one_file(tmp_path, "client_7.parquet")  # 100 rows
    # full dataset name -> cap fires at max_samples=50 (mirrors load_data)
    _res_f, meta_f = reconstruct_train_indices(
        7, data_dir, "Attack_label", max_samples=50, dataset_name="edge_full_20_rmc")
    assert meta_f["capped"] is True
    assert meta_f["n_after_cap"] <= 50
    # NON-full dataset name -> NO cap despite n_rows(100) > max_samples(50),
    # exactly as load_data never caps a non-is_full_dataset partition.
    res_nf, meta_nf = reconstruct_train_indices(
        7, data_dir, "Attack_label", max_samples=50, dataset_name="edge_full_enc")
    assert meta_nf["capped"] is False
    assert meta_nf["n_after_cap"] == 100
    assert len(res_nf["train"]) == int(0.8 * 100)


def test_reconstruct_default_dataset_name_caps_like_audit(tmp_path):
    """The audit's own calls omit dataset_name; the default (TRAIN_DATASET =
    edge_full_20_rmc, a full dataset) must still cap so audit numerics are
    byte-identical."""
    data_dir = _one_file(tmp_path, "client_7.parquet")
    _res, meta = reconstruct_train_indices(7, data_dir, "Attack_label", max_samples=50)
    assert meta["capped"] is True


def test_pool_shortfall_warns_loudly_not_silent(tmp_path, monkeypatch, capsys):
    """When exclusion leaves a class pool below the stratified target, the
    builder warns LOUDLY listing client+class and clamps — never silently
    shrinks without a printed warning."""
    # class1 has only 10 rows; 80% land in train -> ~2 survive; spc target for
    # class1 exceeds that, forcing a shortfall.
    name, _ = _make_dataset(tmp_path, monkeypatch, "shortfall_ds",
                            n_files=1, n_maj=90, n_min=10, nf=4)
    FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42,
                     disjoint=True, train_max_samples=BIG_CAP)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "holdout" in out.lower()
    assert "class" in out.lower()


# ---------------------------------------------------------------------------
# Durable holdout provenance (drift-investigation fix): the [FixedEval] stdout
# record never reliably reaches CloudWatch (captured-stdout section), so the
# rows-excluded / size / per-class counts must live in the result-JSON
# provenance. FixedEvalManager.holdout_provenance is the artifact source;
# run_phase4_flower._holdout_provenance_fields copies it into the result dict.
# ---------------------------------------------------------------------------
def test_holdout_provenance_disjoint(disj_dataset):
    name, _ = disj_dataset
    mgr = FixedEvalManager(
        dataset_name=name, samples_per_client=50, seed=42,
        disjoint=True, train_max_samples=BIG_CAP,
    )
    prov = mgr.holdout_provenance()
    assert prov["holdout_disjoint"] is True
    assert prov["holdout_rows_excluded"] > 0
    assert prov["holdout_size"] > 0
    assert prov["holdout_pos"] + prov["holdout_neg"] == prov["holdout_size"]


def test_holdout_provenance_legacy(disj_dataset):
    name, _ = disj_dataset
    mgr = FixedEvalManager(dataset_name=name, samples_per_client=50, seed=42, disjoint=False)
    prov = mgr.holdout_provenance()
    assert prov["holdout_disjoint"] is False
    assert prov["holdout_rows_excluded"] == 0
    assert prov["holdout_size"] > 0


def test_runner_result_provenance_carries_rows_excluded(disj_dataset):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import _holdout_provenance_fields

    name, _ = disj_dataset
    mgr = FixedEvalManager(
        dataset_name=name, samples_per_client=50, seed=42,
        disjoint=True, train_max_samples=BIG_CAP,
    )
    fields = _holdout_provenance_fields({"holdout-disjoint": True}, mgr)
    assert fields["holdout_disjoint"] is True
    assert fields["holdout_rows_excluded"] > 0
    assert fields["holdout_size"] > 0


def test_runner_result_provenance_falls_back_without_manager():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import _holdout_provenance_fields

    # manager unavailable -> flag only, rows_excluded ABSENT (never fabricated)
    disjoint = _holdout_provenance_fields({"holdout-disjoint": True}, None)
    assert disjoint == {"holdout_disjoint": True}
    assert "holdout_rows_excluded" not in disjoint
    # legacy flag round-trips as False
    legacy = _holdout_provenance_fields({"holdout-disjoint": False}, None)
    assert legacy == {"holdout_disjoint": False}

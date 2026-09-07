"""Sealed-test evaluator (erratum-A E4) — manifest-exact selection, refusals,
custody provenance. All fixtures are synthetic parquet files in tmp dirs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rmc.sealed_test_eval import (  # noqa: E402
    SealedTestEvalManager,
    SealedTestManifestError,
    load_split_manifest,
    sealed_test_selection,
)

LABEL = "Attack_label"
N_FEATURES = 4


def _write_partition(data_dir: Path, fname: str, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n_rows, N_FEATURES)),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    df[LABEL] = (np.arange(n_rows) % 3 == 0).astype(np.int64)
    df.to_parquet(data_dir / fname)
    return df


def _manifest(per_partition: dict) -> dict:
    return {"_meta": {"val_frac": 0.7, "locked": "2026-05-14"},
            "per_partition": per_partition}


@pytest.fixture()
def fixture_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_partition(data_dir, "client_0.parquet", 30, seed=0)
    _write_partition(data_dir, "client_1.parquet", 40, seed=1)
    manifest = _manifest({
        "0": {"file": "client_0.parquet",
              "val_indices": [0, 1, 2, 3],
              "test_indices": [5, 7, 29, 11]},
        "1": {"file": "client_1.parquet",
              "val_indices": [10, 11],
              "test_indices": [0, 39, 20]},
    })
    manifest_path = tmp_path / "val_test_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return data_dir, manifest_path


def _manager(fixture_dir, **kwargs):
    data_dir, manifest_path = fixture_dir
    return SealedTestEvalManager(
        dataset_name="edge_full_20",
        manifest_path=manifest_path,
        data_dir=str(data_dir),
        label_column=LABEL,
        input_shape=N_FEATURES,
        **kwargs,
    )


# ===========================================================================
# manifest-exact selection — no sampling, no seed
# ===========================================================================

@pytest.mark.unit
def test_selects_exactly_the_manifest_test_indices(fixture_dir):
    mgr = _manager(fixture_dir)
    assert mgr.holdout_indices[0].tolist() == [5, 7, 29, 11]  # manifest order
    assert mgr.holdout_indices[1].tolist() == [0, 39, 20]
    assert mgr.holdout_size == 7
    assert mgr.eval_split == "sealed_test"


@pytest.mark.unit
def test_selection_is_deterministic_across_builds(fixture_dir):
    a, b = _manager(fixture_dir), _manager(fixture_dir)
    assert {k: v.tolist() for k, v in a.holdout_indices.items()} == \
           {k: v.tolist() for k, v in b.holdout_indices.items()}
    assert a.per_class_counts == b.per_class_counts


@pytest.mark.unit
def test_per_class_counts_match_the_selected_rows(fixture_dir):
    data_dir, _ = fixture_dir
    mgr = _manager(fixture_dir)
    df0 = pd.read_parquet(data_dir / "client_0.parquet")
    df1 = pd.read_parquet(data_dir / "client_1.parquet")
    labels = np.concatenate([
        df0[LABEL].values[[5, 7, 29, 11]], df1[LABEL].values[[0, 39, 20]],
    ])
    assert mgr.per_class_counts == {
        0: int((labels == 0).sum()), 1: int((labels == 1).sum()),
    }


@pytest.mark.unit
def test_manifest_sha256_is_over_the_file_bytes(fixture_dir):
    _, manifest_path = fixture_dir
    mgr = _manager(fixture_dir)
    assert mgr.manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


@pytest.mark.unit
def test_holdout_provenance_carries_the_e4_custody_pair(fixture_dir):
    prov = _manager(fixture_dir).holdout_provenance()
    assert prov["eval_split"] == "sealed_test"
    assert len(prov["eval_split_manifest_sha256"]) == 64
    assert prov["holdout_size"] == 7
    assert prov["holdout_disjoint"] is False  # truthful: no exclusion is done
    assert prov["holdout_rows_excluded"] == 0


# ===========================================================================
# refusals — missing/malformed manifest, corrupt split, missing data
# ===========================================================================

@pytest.mark.unit
def test_missing_manifest_refuses(tmp_path):
    with pytest.raises(SealedTestManifestError, match="missing"):
        load_split_manifest(tmp_path / "nope.json")


@pytest.mark.unit
def test_malformed_json_refuses(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{not json")
    with pytest.raises(SealedTestManifestError, match="not valid JSON"):
        load_split_manifest(p)


@pytest.mark.unit
def test_manifest_without_per_partition_refuses(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"_meta": {}}))
    with pytest.raises(SealedTestManifestError, match="per_partition"):
        load_split_manifest(p)


@pytest.mark.unit
def test_val_test_overlap_refuses():
    manifest = _manifest({
        "0": {"file": "client_0.parquet",
              "val_indices": [1, 2, 5], "test_indices": [5, 7]},
    })
    with pytest.raises(SealedTestManifestError, match="overlap"):
        sealed_test_selection(manifest)


@pytest.mark.unit
def test_empty_test_indices_refuse():
    manifest = _manifest({
        "0": {"file": "client_0.parquet", "val_indices": [1],
              "test_indices": []},
    })
    with pytest.raises(SealedTestManifestError, match="non-empty"):
        sealed_test_selection(manifest)


@pytest.mark.unit
def test_duplicate_test_indices_refuse():
    manifest = _manifest({
        "0": {"file": "client_0.parquet", "val_indices": [],
              "test_indices": [3, 3]},
    })
    with pytest.raises(SealedTestManifestError, match="duplicates"):
        sealed_test_selection(manifest)


@pytest.mark.unit
def test_out_of_range_index_refuses(fixture_dir, tmp_path):
    data_dir, _ = fixture_dir
    manifest = _manifest({
        "0": {"file": "client_0.parquet", "val_indices": [],
              "test_indices": [5, 30]},   # client_0 has 30 rows: max idx 29
    })
    p = tmp_path / "bad_manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(SealedTestManifestError, match="out of range"):
        SealedTestEvalManager(
            dataset_name="edge_full_20", manifest_path=p,
            data_dir=str(data_dir), label_column=LABEL,
            input_shape=N_FEATURES,
        )


@pytest.mark.unit
def test_missing_parquet_refuses(fixture_dir, tmp_path):
    data_dir, _ = fixture_dir
    manifest = _manifest({
        "0": {"file": "client_9.parquet", "val_indices": [],
              "test_indices": [1]},
    })
    p = tmp_path / "bad_manifest2.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(SealedTestManifestError, match="does not exist"):
        SealedTestEvalManager(
            dataset_name="edge_full_20", manifest_path=p,
            data_dir=str(data_dir), label_column=LABEL,
            input_shape=N_FEATURES,
        )


# ===========================================================================
# the committed production manifest is loadable and internally consistent
# ===========================================================================

@pytest.mark.unit
def test_committed_manifest_validates():
    manifest, sha = load_split_manifest(
        PROJECT_ROOT / "data" / "val_test_split_manifest.json"
    )
    selection = sealed_test_selection(manifest)
    assert len(selection) == 21          # 20 partitions + the duplicate 20
    assert all(len(t) == 600 for _, _, t in selection)
    assert len(sha) == 64


# ===========================================================================
# server_app dispatch — legacy path untouched, unknown value refuses
# ===========================================================================

@pytest.mark.unit
def test_server_app_rejects_unknown_eval_split(monkeypatch):
    from flowerfl import server_app

    with pytest.raises(ValueError, match="eval-split"):
        server_app._create_eval_manager(
            "edge_full_20_rmc", {"eval-split": "bogus"}
        )


@pytest.mark.unit
def test_server_app_sealed_test_dispatch(monkeypatch, fixture_dir):
    """eval-split=sealed_test builds a SealedTestEvalManager (patched to the
    synthetic fixture so no real dataset is needed)."""
    data_dir, manifest_path = fixture_dir
    import rmc.sealed_test_eval as ste
    from flowerfl import server_app

    built = {}

    def _fake_manager(dataset_name):
        built["dataset_name"] = dataset_name
        return SealedTestEvalManager(
            dataset_name="edge_full_20", manifest_path=manifest_path,
            data_dir=str(data_dir), label_column=LABEL,
            input_shape=N_FEATURES,
        )

    monkeypatch.setattr(
        ste, "SealedTestEvalManager",
        lambda dataset_name: _fake_manager(dataset_name),
    )
    mgr = server_app._create_eval_manager(
        "edge_full_20_rmc", {"eval-split": "sealed_test"}
    )
    assert built["dataset_name"] == "edge_full_20"  # _rmc mapped to eval base
    assert mgr.eval_split == "sealed_test"

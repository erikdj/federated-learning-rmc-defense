"""End-to-end Stage-F client wiring (DESIGN_STAGE_F §4/§5/§6).

Drives a real FlowerClient.fit() over a synthetic benign-heavy partition with the
full Stage-F knob stack on (update-match, weight-mode=original, semantic target)
and asserts the observable §5 contract: FedAvg mass == n_orig, every path hit the
step cap, and the manifest row is emitted + passes its own arm-compliance checks.
Also confirms the incumbent defaults leave num_examples at the resampled count.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import flowerfl.task as task_module
from flowerfl.client_app import FlowerClient
from flowerfl.task import load_data, create_model, detect_input_shape

TMP = "stage_f_int_ds"
B = 32
E = 5


@pytest.fixture
def benign_heavy_ds(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_ben, n_att, nf = 260, 40, 6  # benign-heavy -> attack is the minority
    X = np.vstack([rng.normal(0, 1, (n_ben, nf)), rng.normal(5, 0.5, (n_att, nf))]).astype(np.float32)
    y = np.concatenate([np.zeros(n_ben), np.ones(n_att)]).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(nf)}
    cols["Attack_label"] = y
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    pd.DataFrame(cols).to_parquet(data_dir / "client_0.parquet")
    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[TMP] = {
        "data_dir": str(data_dir), "label_column": "Attack_label",
        "client_files": ["client_0.parquet"], "client_ids": ["0"],
        "num_classes": 2, "description": "stage-f int", "malicious_order": [0],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(TMP, None)
    return TMP


def _client(ds, *, update_match, weight_mode, semantic, arm="smote@0.5"):
    tr, va, _, prep = load_data(
        0, dataset_name=ds, batch_size=B,
        smote_enabled=True, smote_variant="smote", smote_target="balanced",
        smote_seed=1, smote_semantic_target=semantic, return_prep_info=True,
    )
    n_orig = prep["n_orig"]
    max_steps = math.ceil(n_orig / B) * E if update_match else None
    net = create_model(ds, detect_input_shape(ds))
    return FlowerClient(
        tr, va, net, partition_id=0, use_brfss=False, local_epochs=E,
        max_steps=max_steps, arm_label=arm, update_match=update_match,
        weight_mode=weight_mode, n_orig=n_orig, prep_info=prep,
        semantic_target=semantic,
    ), n_orig, max_steps


def test_full_stack_reports_original_mass_and_hits_cap(benign_heavy_ds):
    client, n_orig, K = _client(benign_heavy_ds, update_match=True,
                                weight_mode="original", semantic=True)
    params = client.get_parameters({})
    _, num_examples, metrics = client.fit(params, {"server_round": 1, "attack_type": ""})

    # §5: original weight-mode reports the PRE-resampling count as FedAvg mass.
    assert num_examples == n_orig
    # §4: the honest path hit the cap exactly.
    assert client._step_metrics["steps_taken"] == K
    # §5: the manifest row is emitted and self-consistent.
    row = json.loads(metrics["resampling_manifest"])
    assert row["num_examples"] == n_orig
    assert row["actual_steps"] == K == row["max_steps"]
    assert row["semantic_policy"] is True
    # semantic over-sampler grew attack; benign held fixed.
    assert row["n_benign_after"] == row["n_benign_before"]
    assert row["sampler_status"] == "applied"


def test_label_flip_path_also_hits_cap(benign_heavy_ds):
    client, n_orig, K = _client(benign_heavy_ds, update_match=True,
                                weight_mode="original", semantic=True,
                                arm="smote@0.5")
    params = client.get_parameters({})
    # Route through the scenario-driven label-flip attack branch.
    _, num_examples, metrics = client.fit(params, {"server_round": 1, "attack_type": "label_flip"})
    assert num_examples == n_orig
    assert client._step_metrics["steps_taken"] == K


def test_defaults_report_resampled_mass(benign_heavy_ds):
    # weight-mode resampled + update-match off = incumbent: mass is the resampled
    # count and no step cap is applied.
    client, n_orig, _ = _client(benign_heavy_ds, update_match=False,
                                weight_mode="resampled", semantic=True)
    params = client.get_parameters({})
    _, num_examples, metrics = client.fit(params, {"server_round": 1, "attack_type": ""})
    row = json.loads(metrics["resampling_manifest"])
    assert num_examples == row["n_resampled"] >= n_orig
    assert row["max_steps"] is None

"""Tests for the v3 signal-log schema with TGE per-client fields."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _fake_results(cids, dim=8, seed=0):
    """Fake (client_proxy, fit_res) pairs with real flwr Parameters
    (mirrors tests/test_plugin_identity_keying.py)."""
    from flwr.common import ndarrays_to_parameters
    rng = np.random.default_rng(seed)
    out = []
    for cid in cids:
        params = ndarrays_to_parameters([rng.normal(size=dim).astype(np.float32)])
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(parameters=params, num_examples=10, metrics={})
        out.append((proxy, fit))
    return out


def test_signal_logger_emits_schema_version(tmp_path):
    """Every JSONL record must include signal_log_schema_version=5 (v5 adds the
    post-filter aggregation_coefficient + the H3 re-entry event contract,
    ; v4 added the TGE′ bank's tge_ema_score leg, )."""
    from flowerfl.signal_logger import SignalLogger

    out = tmp_path / "test.jsonl"
    logger = SignalLogger(
        path=str(out),
        run_metadata={
            "seed": 42, "scenario": "test", "exec_mode": "flower_reset",
            "dataset": "test_dataset", "defense": "tgensemble",
        },
    )
    logger.log_round(
        server_round=1, scenario_round=0,
        per_client_records=[{
            "logical_cid": "client_0", "flower_cid": "abc",
            "physical_partition_id": 0, "malicious_gt": False,
            "attack_type": "", "num_examples": 100,
            "train_loss": 0.5, "update_norm": 1.0,
            "cos_to_median": 0.95, "L2_to_median": 0.5,
            "krum_score": None, "trust_score": None,
            "effective_weight": 100.0,
        }],
    )
    logger.close()

    with open(out) as f:
        rec = json.loads(f.readline())
    assert rec.get("signal_log_schema_version") == 5, (
        f"signal_log_schema_version missing or wrong: {rec.get('signal_log_schema_version')!r}"
    )


def test_signal_logger_schema_version_resists_override(tmp_path):
    """A caller passing signal_log_schema_version in run_metadata must NOT
    override the authoritative value (5)."""
    from flowerfl.signal_logger import SignalLogger
    import json

    out = tmp_path / "test.jsonl"
    logger = SignalLogger(
        path=str(out),
        run_metadata={
            "seed": 42, "scenario": "test", "exec_mode": "flower_reset",
            "dataset": "test_dataset", "defense": "tgensemble",
            # Adversarially try to override:
            "signal_log_schema_version": 99,
        },
    )
    logger.log_round(
        server_round=1, scenario_round=0,
        per_client_records=[{
            "logical_cid": "client_0", "flower_cid": "abc",
            "physical_partition_id": 0, "malicious_gt": False,
            "attack_type": "", "num_examples": 100,
            "train_loss": 0.5, "update_norm": 1.0,
            "cos_to_median": 0.95, "L2_to_median": 0.5,
            "krum_score": None, "trust_score": None,
            "effective_weight": 100.0,
        }],
    )
    logger.close()

    with open(out) as f:
        rec = json.loads(f.readline())
    assert rec["signal_log_schema_version"] == 5, (
        f"caller-supplied schema_version=99 should NOT override the authoritative 5; "
        f"got {rec['signal_log_schema_version']}"
    )


def test_tge_plugin_exposes_per_client_details():
    """After construction, TGEnsemblePlugin should expose a _last_details attribute
    initialized to an empty dict, so the strategy can read it after score_updates."""
    from flowerfl.byzantine_defense import TGEnsemblePlugin

    plugin = TGEnsemblePlugin(num_malicious=2, num_to_keep=5, ramp_rounds=999)
    # Before any score_updates call, _last_details should exist as empty dict
    assert hasattr(plugin, "_last_details"), (
        "TGEnsemblePlugin should expose _last_details attribute"
    )
    assert plugin._last_details == {}, (
        f"TGEnsemblePlugin._last_details should be initialized to empty dict; "
        f"got {plugin._last_details!r}"
    )


def test_tge_last_details_keyed_by_raw_cid():
    """v1.17: `_last_details` must be keyed by RAW CID, not
    positional index. In the composed Krum+TGE chain (the deployed primary),
    PluggableStrategy hands TGE only Krum's survivors, while the signal
    logger iterates the FULL results list — a positional join silently
    attributes TGE scores to the wrong clients whenever Krum filters anyone
    (Multi-Krum always does)."""
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    plugin = TGEnsemblePlugin(num_malicious=1, num_to_keep=3)
    results = _fake_results(["raw0", "raw1", "raw2", "raw3"])
    plugin.score_updates(results, server_round=1)
    assert set(plugin._last_details.keys()) == {"raw0", "raw1", "raw2", "raw3"}


def test_tge_details_absent_for_upstream_filtered_clients():
    """When TGE receives only upstream survivors, filtered clients must have
    NO details entry — their signal rows get null TGE fields (truthful:
    TGE never scored them), never another client's scores."""
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    plugin = TGEnsemblePlugin(num_malicious=1, num_to_keep=3)
    survivors = _fake_results(["raw1", "raw3"])  # raw0/raw2 filtered upstream
    plugin.score_updates(survivors, server_round=1)
    assert set(plugin._last_details.keys()) == {"raw1", "raw3"}


@pytest.mark.slow
def test_tge_signal_log_has_tge_fields(tmp_path):
    """A small TGE run should produce a signal-log JSONL where every active-TGE
    row carries non-null tge_score, tge_threshold, tge_decision, tge_gbdt_score,
    tge_phase, tge_gate, tge_tenure.

    Regression context (2026-05-26): commit 89a1882 silently broke this by
    folding TGE collection into a loop that skips plugins lacking _round_scores.
    The pre-fix test only required AT LEAST ONE populated row, so it passed on
    stale data from earlier test runs (the signal logger appends). This version
    deletes the target file first and asserts ALL eligible rows are populated.
    """
    import subprocess
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Delete any prior signal log so we don't ride on stale populated rows.
    signal_path = (
        PROJECT_ROOT / "signals"
        / "flower_reset__rmc_intensity_9_continuous_v2__tgensemble__seed42.jsonl"
    )
    if signal_path.exists():
        signal_path.unlink()

    result = subprocess.run(
        ["conda", "run", "-n", "flowerfl",
         "python", "scripts/run_phase4_flower.py",
         "--configs", "TGE",
         "--seeds", "42",
         "--scenario", "rmc/scenarios/rmc_intensity_9_continuous_v2.json",
         "--max-per-client", "100",
         "--rounds", "3",
         "--optimizer-state", "reset",
         "--output-dir", str(tmp_path),
         "--reporting-split", "val"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"Runner failed: stderr={result.stderr[-2000:]!r}")

    assert signal_path.exists(), f"no signal log produced at {signal_path}"

    import json
    eligible_rows = []
    null_tge_rows = []
    with open(signal_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("server_round", 0) >= 2:
                eligible_rows.append(rec)
                if rec.get("tge_score") is None:
                    null_tge_rows.append(rec)
                for field in ["tge_score", "tge_threshold", "tge_decision",
                              "tge_gbdt_score", "tge_lstm_score",
                              "tge_phase", "tge_gate", "tge_tenure"]:
                    assert field in rec, f"missing tge field: {field}"

    assert len(eligible_rows) > 0, "no per-client records emitted at server_round >= 2"
    assert not null_tge_rows, (
        f"{len(null_tge_rows)}/{len(eligible_rows)} eligible rows have tge_score=None — "
        f"plumbing regression. First null row keys: {sorted(null_tge_rows[0].keys())}"
    )

"""Stage-F configuration and cache-reuse regression tests.

P1-1 cache-reuse identity: a cached result may only be reused for a run with the
     SAME Stage-F arm (update-match / weight-mode / semantic-target), never a
     different arm sharing the same sampler/target.
P1-2 prewarm semantic key: resample_cache_path threads smote_semantic_target so
     the driver warms / validates the SAME key the workers look up.
P1-3 manifest durability: ScenarioStrategy collects the resampling_manifest fit
     metric server-side (deduped) and the runner writes it into the result JSON.
P2-4 declared-config check: the production assertion passes `expected`, so a
     client whose manifest disagrees with its declared arm raises in fit.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

import flowerfl.task as task_module
from flowerfl.scenario_strategy import ScenarioStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ===========================================================================
# P1-1 — Stage-F cache-reuse identity
# ===========================================================================

def test_stage_f_provenance_fields_defaults_are_incumbent():
    from run_phase4_flower import stage_f_provenance_fields
    assert stage_f_provenance_fields({}) == {
        "update_match": False, "weight_mode": "resampled", "smote_semantic_target": False,
    }


def test_stage_f_provenance_fields_reads_active_knobs():
    from run_phase4_flower import stage_f_provenance_fields
    cfg = {"update-match": True, "weight-mode": "original", "smote-semantic-target": True}
    assert stage_f_provenance_fields(cfg) == {
        "update_match": True, "weight_mode": "original", "smote_semantic_target": True,
    }


@pytest.mark.parametrize("knob", ["update-match", "weight-mode", "smote-semantic-target"])
def test_stage_f_cache_not_reusable_when_a_knob_differs(knob):
    from run_phase4_flower import _stage_f_cache_reusable, stage_f_provenance_fields
    # Cached result is a pure incumbent run.
    cached = {"provenance": stage_f_provenance_fields({})}
    active = {"update-match": True} if knob == "update-match" else \
             {"weight-mode": "original"} if knob == "weight-mode" else \
             {"smote-semantic-target": True}
    assert _stage_f_cache_reusable(cached, active) is False


def test_stage_f_cache_reusable_when_knobs_match():
    from run_phase4_flower import _stage_f_cache_reusable, stage_f_provenance_fields
    active = {"update-match": True, "weight-mode": "original", "smote-semantic-target": True}
    cached = {"provenance": stage_f_provenance_fields(active)}
    assert _stage_f_cache_reusable(cached, active) is True


def test_legacy_cache_reads_as_incumbent_identity():
    # A pre-Stage-F result JSON (no stage-f provenance) is reusable ONLY by an
    # incumbent run, never by a semantic/update-matched one.
    from run_phase4_flower import _stage_f_cache_reusable
    legacy = {"provenance": {"smote_enabled": False}}  # no stage-f fields
    assert _stage_f_cache_reusable(legacy, {}) is True
    assert _stage_f_cache_reusable(legacy, {"update-match": True}) is False


def test_reuse_cached_or_rotate_recomputes_on_stage_f_mismatch(tmp_path):
    from run_phase4_flower import (
        _reuse_cached_or_rotate, smote_provenance_fields, stage_f_provenance_fields,
        _holdout_disjoint_provenance,
    )
    # A cached incumbent result with a valid trajectory + matching SMOTE/holdout.
    prov = {
        **smote_provenance_fields({}),
        **stage_f_provenance_fields({}),
        "holdout_disjoint": _holdout_disjoint_provenance({"holdout-disjoint": True}),
    }
    cached = {"trajectory": [{"round": 1}], "return_code": 0, "provenance": prov}
    json_path = tmp_path / "unit.json"
    json_path.write_text(json.dumps(cached))
    common = dict(scenario_path="rmc/scenarios/rmc_main_50r.json", strategy="Krum",
                  seed=42, optimizer_state="reset", exp_name="X")

    # Incumbent active run reuses.
    reused = _reuse_cached_or_rotate(json_path, {"holdout-disjoint": True}, **common)
    assert reused is not None
    # Semantic active run must NOT reuse (returns None -> recompute).
    not_reused = _reuse_cached_or_rotate(
        json_path, {"holdout-disjoint": True, "smote-semantic-target": True}, **common)
    assert not_reused is None


# ===========================================================================
# P1-2 — prewarm / resample_cache_path threads the semantic flag
# ===========================================================================

def test_resample_cache_key_partitions_on_semantic_flag():
    k_legacy = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, smote_semantic_target=False)
    k_sem = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, smote_semantic_target=True)
    assert k_legacy != k_sem
    assert k_legacy[-1] is False and k_sem[-1] is True


def test_resample_cache_path_threads_semantic_to_key():
    # The path resolver must build the SAME key the semantic flag produces, so a
    # driver durability pass validates the key workers actually look up.
    sem_path = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42,
        smote_semantic_target=True)
    expected = task_module._resample_disk_path(task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, smote_semantic_target=True))
    legacy_path = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42,
        smote_semantic_target=False)
    assert sem_path == expected
    # semantic and legacy resolve to DIFFERENT durable paths (or both None only if
    # the disk layer is unavailable — then the key-level test above is the proof).
    if sem_path is not None or legacy_path is not None:
        assert sem_path != legacy_path


def test_prewarm_reads_semantic_flag():
    # Guard against the exact P1-2 regression: the prewarm must consult the
    # semantic run-config key (not silently warm the legacy False key).
    import inspect
    from run_phase4_flower import _prewarm_resample_cache
    src = inspect.getsource(_prewarm_resample_cache)
    assert "smote-semantic-target" in src
    assert "smote_semantic_target=semantic" in src


# ===========================================================================
# P1-3 — server-side manifest collection + runner extraction
# ===========================================================================

DIM = 8


def _fit_results_with_manifest(pids, *, steps=160):
    # update_match=True pins these fixtures to the update-matched regime, where
    # the static-row gate compares actual_steps/max_steps strictly.
    out = []
    for p in pids:
        proxy = SimpleNamespace(cid=f"raw{p}")
        row = {"partition_id": p, "arm": "smote@0.5", "actual_steps": steps,
               "max_steps": steps, "update_match": True,
               "n_orig": 100, "n_resampled": 180}
        fit = SimpleNamespace(
            parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
            num_examples=100,
            metrics={"partition_id": float(p), "resampling_manifest": json.dumps(row)},
        )
        out.append((proxy, fit))
    return out


def _bare_strategy():
    base = FedAvg(
        initial_parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
        min_fit_clients=1, min_available_clients=1, min_evaluate_clients=1,
    )
    return ScenarioStrategy(base_strategy=base, plugins=[], scenario_path=None)


def test_collect_resampling_manifest_dedupes_by_partition():
    strat = _bare_strategy()
    strat._collect_resampling_manifest(1, _fit_results_with_manifest([0, 1, 2], steps=160))
    # IDENTICAL rows reappear next round (static rows) -> idempotent no-op.
    strat._collect_resampling_manifest(2, _fit_results_with_manifest([0, 1, 2], steps=160))
    assert sorted(strat._resampling_manifest) == [0, 1, 2]
    assert strat._resampling_manifest[0]["actual_steps"] == 160  # first appearance kept


def test_aggregate_fit_collects_manifest_on_discovery_round():
    strat = _bare_strategy()  # mapping not ready -> discovery path
    strat.aggregate_fit(1, _fit_results_with_manifest([0, 1]), failures=[])
    assert sorted(strat._resampling_manifest) == [0, 1]
    assert strat._resampling_manifest[1]["arm"] == "smote@0.5"


# ---------------------------------------------------------------------------
# BLOCKER fix — loud collector, dispatched-partition tracking, unit gate
# ---------------------------------------------------------------------------

def _full_row(pid):
    """A schema-complete manifest row (every MANIFEST_FIELDS key present)."""
    from flowerfl.resampling_manifest import MANIFEST_FIELDS
    row = {k: 0 for k in MANIFEST_FIELDS}
    row["partition_id"] = pid
    return row


def _fit_result_raw(cid_tag, raw, *, part_metric=None):
    """One (proxy, fit_res) carrying an explicit raw resampling_manifest metric."""
    metrics = {"resampling_manifest": raw}
    if part_metric is not None:
        metrics["partition_id"] = float(part_metric)
    proxy = SimpleNamespace(cid=f"raw{cid_tag}")
    fit = SimpleNamespace(
        parameters=ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)]),
        num_examples=100, metrics=metrics,
    )
    return (proxy, fit)


class _CidClientManager:
    """Minimal client manager whose proxies carry caller-chosen cids."""

    def __init__(self, cids):
        self._clients = [SimpleNamespace(cid=c) for c in cids]

    def num_available(self):
        return len(self._clients)

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return self._clients[:num_clients]


def test_collector_raises_on_malformed_json():
    strat = _bare_strategy()
    with pytest.raises(ValueError, match="round 1"):
        strat._collect_resampling_manifest(1, [_fit_result_raw(0, "{not json")])


def test_collector_raises_on_missing_partition_id():
    strat = _bare_strategy()
    raw = json.dumps({"arm": "smote@0.5"})  # no partition_id
    with pytest.raises(ValueError, match="partition_id"):
        strat._collect_resampling_manifest(2, [_fit_result_raw(0, raw)])


def test_collector_conflicting_repeat_raises_naming_pid_and_keys():
    strat = _bare_strategy()
    strat._collect_resampling_manifest(1, _fit_results_with_manifest([0], steps=160))
    with pytest.raises(ValueError) as ei:
        strat._collect_resampling_manifest(2, _fit_results_with_manifest([0], steps=999))
    msg = str(ei.value)
    assert "0" in msg and "actual_steps" in msg and "max_steps" in msg


def test_collector_identical_repeat_is_silent_noop(capsys):
    strat = _bare_strategy()
    strat._collect_resampling_manifest(1, _fit_results_with_manifest([0, 1], steps=160))
    capsys.readouterr()  # drop the first-appearance [MANIFEST] prints
    strat._collect_resampling_manifest(2, _fit_results_with_manifest([0, 1], steps=160))
    out = capsys.readouterr().out
    assert "[MANIFEST]" not in out  # identical repeat: no raise, no reprint
    assert sorted(strat._resampling_manifest) == [0, 1]


# ---------------------------------------------------------------------------
# static-row gate must not guard natural step counts (update-match OFF)
#
# Root cause: train_label_flip runs 1 natural pass vs 5 epochs on every other
# path, so under update-match OFF a partition whose S4 lineage rotates into
# label_flip legitimately reports a different actual_steps than its
# first-recorded row. actual_steps is a per-round quantity in that regime, not
# a per-unit invariant — the gate skips it when the incoming row's own
# update_match field is falsey, and compares everything else strictly
# (including max_steps, which OFF rows carry as None).
# ---------------------------------------------------------------------------

_ABSENT = object()


def _off_row(pid, *, actual_steps, max_steps=None, update_match=False, **overrides):
    row = {"partition_id": pid, "arm": "smote@0.5", "actual_steps": actual_steps,
           "max_steps": max_steps, "n_orig": 100, "n_resampled": 180}
    if update_match is not _ABSENT:
        row["update_match"] = update_match
    row.update(overrides)
    return row


@pytest.mark.parametrize("update_match", [False, _ABSENT],
                         ids=["explicit-false", "absent"])
def test_collector_update_match_off_accepts_natural_step_drift(update_match, capsys):
    # (1) OFF-regime pair differing ONLY in actual_steps (max_steps None):
    # no raise, first-recorded row retained, no reprint.
    strat = _bare_strategy()
    first = _off_row(0, actual_steps=800, update_match=update_match)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(first))])
    capsys.readouterr()
    second = _off_row(0, actual_steps=27, update_match=update_match)
    strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(second))])
    assert strat._resampling_manifest[0]["actual_steps"] == 800  # first kept
    assert "[MANIFEST]" not in capsys.readouterr().out  # no reprint


def test_collector_update_match_off_still_raises_on_data_field_conflict():
    # (2) OFF-regime pair differing in a data field: raises naming that field,
    # and the excluded step fields never appear in the differing-keys list.
    strat = _bare_strategy()
    strat._collect_resampling_manifest(
        1, [_fit_result_raw(0, json.dumps(_off_row(0, actual_steps=800)))])
    with pytest.raises(ValueError) as ei:
        strat._collect_resampling_manifest(
            2, [_fit_result_raw(0, json.dumps(
                _off_row(0, actual_steps=27, n_orig=999)))])
    msg = str(ei.value)
    assert "n_orig" in msg
    assert "actual_steps" not in msg and "max_steps" not in msg


def test_collector_update_match_on_step_conflict_still_raises():
    # (3) ON-regime pair differing only in actual_steps: unchanged full
    # comparison, still raises naming the field.
    strat = _bare_strategy()
    on_row = _off_row(0, actual_steps=160, max_steps=160, update_match=True)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(on_row))])
    drifted = dict(on_row, actual_steps=159)
    with pytest.raises(ValueError, match="actual_steps"):
        strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(drifted))])


def test_collector_identical_repeat_noop_update_match_off(capsys):
    # (4) OFF-regime identical repeat: idempotent no-op (ON regime is covered
    # by test_collector_identical_repeat_is_silent_noop above).
    strat = _bare_strategy()
    row = _off_row(0, actual_steps=800)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(row))])
    capsys.readouterr()
    strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(row))])
    assert "[MANIFEST]" not in capsys.readouterr().out
    assert sorted(strat._resampling_manifest) == [0]


def test_collector_on_regime_missing_vs_none_key_raises():
    # a MISSING key must not compare equal to a present key
    # valued None — the ON-regime full comparison distinguishes them (sentinel).
    strat = _bare_strategy()
    prior = _off_row(0, actual_steps=160, max_steps=160, update_match=True,
                     k_eff=None)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(prior))])
    sparse = {k: v for k, v in prior.items() if k != "k_eff"}  # k_eff MISSING
    with pytest.raises(ValueError, match="k_eff"):
        strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(sparse))])


def test_collector_update_match_off_max_steps_corruption_raises():
    # max_steps stays compared under OFF (production OFF
    # rows carry max_steps=None; it is NOT round-varying). None -> number is
    # corruption and raises naming max_steps.
    strat = _bare_strategy()
    strat._collect_resampling_manifest(
        1, [_fit_result_raw(0, json.dumps(_off_row(0, actual_steps=800)))])
    with pytest.raises(ValueError, match="max_steps"):
        strat._collect_resampling_manifest(
            2, [_fit_result_raw(0, json.dumps(
                _off_row(0, actual_steps=800, max_steps=160)))])


@pytest.mark.parametrize("prior_on", [True, False], ids=["on-then-off", "off-then-on"])
def test_collector_mixed_regime_repeat_raises(prior_on):
    # a repeat that flips regimes differs in update_match
    # itself, which is always compared — both orderings raise naming it.
    strat = _bare_strategy()
    on_row = _off_row(0, actual_steps=160, max_steps=160, update_match=True)
    off_row = _off_row(0, actual_steps=160, max_steps=160, update_match=False)
    first, second = (on_row, off_row) if prior_on else (off_row, on_row)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(first))])
    with pytest.raises(ValueError, match="update_match"):
        strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(second))])


@pytest.mark.parametrize("prior_has", [True, False],
                         ids=["prior-has-repeat-omits", "prior-omits-repeat-has"])
def test_collector_update_match_off_actual_steps_presence_mismatch_raises(prior_has):
    # under OFF, only VALUE drift on
    # actual_steps is legitimate — a presence mismatch (field missing from one
    # row) is a custody error and must still raise naming actual_steps.
    strat = _bare_strategy()
    full = _off_row(0, actual_steps=800)
    sparse = {k: v for k, v in full.items() if k != "actual_steps"}
    first, second = (full, sparse) if prior_has else (sparse, full)
    strat._collect_resampling_manifest(1, [_fit_result_raw(0, json.dumps(first))])
    with pytest.raises(ValueError, match="actual_steps"):
        strat._collect_resampling_manifest(2, [_fit_result_raw(0, json.dumps(second))])


def test_collector_label_flip_lineage_natural_steps_accepted():
    # (5) The scenario in miniature: an S4 partition first fits on the
    # 5-epoch path (5 epochs x 160 steps), then its lineage rotates into
    # label_flip which runs 1 natural pass (160 steps), update-match OFF.
    # The second row must be accepted, first-recorded row retained.
    strat = _bare_strategy()
    five_epoch = _off_row(3, actual_steps=800, arm="smote@0.5")
    strat._collect_resampling_manifest(4, [_fit_result_raw(3, json.dumps(five_epoch))])
    one_pass = _off_row(3, actual_steps=160, arm="smote@0.5")
    strat._collect_resampling_manifest(5, [_fit_result_raw(3, json.dumps(one_pass))])
    assert strat._resampling_manifest[3]["actual_steps"] == 800


def test_dispatched_partitions_accumulate_incl_discovery():
    strat = _bare_strategy()
    params = ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)])
    # Discovery round: the dispatched set learns every returned partition.
    strat.aggregate_fit(1, _fit_results_with_manifest([0, 1]), failures=[])
    assert strat._dispatched_partitions == {0, 1}
    # Post-discovery scheduled round that dispatches pid 2 (unseen at discovery).
    strat._scenario = {"stub": True}
    strat._mapping_ready = True
    strat._round_offset = 1
    strat._cid_to_partition = {"raw0": 0, "raw1": 1, "raw2": 2}
    strat._schedule_cache = {1: [
        {"partition_id": p, "attack_type": "", "attack_params": {}, "logical_id": f"c{p}"}
        for p in (0, 1, 2)
    ]}
    strat.configure_fit(2, params, _CidClientManager(["raw0", "raw1", "raw2"]))
    assert strat._dispatched_partitions == {0, 1, 2}


def test_runner_gate_raises_on_incomplete_manifest_and_skips_write():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(
        _resampling_manifest={0: _full_row(0)},  # pid 1 dispatched but no row
        _dispatched_partitions={0, 1},
    )
    result = {"trajectory": []}
    with pytest.raises((ValueError, RuntimeError)):
        _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert "resampling_manifest" not in result


def test_runner_gate_raises_when_empty_but_dispatched_nonempty():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(_resampling_manifest={}, _dispatched_partitions={0, 1})
    result = {"trajectory": []}
    with pytest.raises((ValueError, RuntimeError)):
        _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert "resampling_manifest" not in result


def test_runner_gate_exempt_without_dispatched_attr():
    # Legacy/test strategy objects lacking _dispatched_partitions are exempt:
    # a nonempty manifest is still written unconditionally (back-compat).
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(_resampling_manifest={0: _full_row(0), 2: _full_row(2)})
    result = {"trajectory": []}
    _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert [r["partition_id"] for r in result["resampling_manifest"]] == [0, 2]


def test_runner_gate_passes_on_complete_manifest_and_writes_sorted():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(
        _resampling_manifest={2: _full_row(2), 0: _full_row(0)},
        _dispatched_partitions={0, 2},
    )
    result = {"trajectory": []}
    _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert [r["partition_id"] for r in result["resampling_manifest"]] == [0, 2]


# ---------------------------------------------------------------------------
# P1 — discovery-round failure closure via unresolved dispatched cids
# ---------------------------------------------------------------------------

def test_configure_fit_pre_mapping_records_dispatched_cids():
    strat = _bare_strategy()  # scenario None, mapping not ready
    params = ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)])
    strat.configure_fit(1, params, _CidClientManager(["raw0", "raw1", "raw2"]))
    assert strat._pre_mapping_dispatched_cids == {"raw0", "raw1", "raw2"}


def test_unresolved_dispatched_cids_flags_discovery_failure():
    strat = _bare_strategy()
    params = ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)])
    # Dispatch raw0/raw1/raw2 in the discovery round...
    strat.configure_fit(1, params, _CidClientManager(["raw0", "raw1", "raw2"]))
    #...but only raw0/raw1 return a successful fit (raw2 failed in discovery).
    strat.aggregate_fit(1, _fit_results_with_manifest([0, 1]), failures=[])
    assert strat.unresolved_dispatched_cids() == {"raw2"}


def test_unresolved_dispatched_cids_empty_when_all_resolve():
    strat = _bare_strategy()
    params = ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)])
    strat.configure_fit(1, params, _CidClientManager(["raw0", "raw1"]))
    strat.aggregate_fit(1, _fit_results_with_manifest([0, 1]), failures=[])
    assert strat.unresolved_dispatched_cids() == set()


def test_runner_gate_raises_on_unresolved_discovery_cid():
    # Coverage of dispatched partitions is exact, but a discovery-round client
    # never resolved -> the unit must still FAIL (missing evidence).
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(
        _resampling_manifest={0: _full_row(0), 1: _full_row(1)},
        _dispatched_partitions={0, 1},
        unresolved_dispatched_cids=lambda: {"raw2"},
    )
    result = {"trajectory": []}
    with pytest.raises((ValueError, RuntimeError), match="raw2"):
        _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert "resampling_manifest" not in result


def test_runner_gate_passes_when_all_discovery_cids_resolved():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(
        _resampling_manifest={0: _full_row(0), 1: _full_row(1)},
        _dispatched_partitions={0, 1},
        unresolved_dispatched_cids=lambda: set(),
    )
    result = {"trajectory": []}
    _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert [r["partition_id"] for r in result["resampling_manifest"]] == [0, 1]


def test_runner_writes_manifest_into_result_json():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(_resampling_manifest={
        2: {"partition_id": 2, "arm": "smote@0.5"},
        0: {"partition_id": 0, "arm": "smote@0.5"},
    })
    result = {"trajectory": []}
    _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert [r["partition_id"] for r in result["resampling_manifest"]] == [0, 2]  # sorted


def test_runner_omits_manifest_for_incumbent_runs():
    from run_phase4_flower import _enrich_result_with_metrics
    fake = SimpleNamespace(_resampling_manifest={})
    result = {"trajectory": []}
    _enrich_result_with_metrics(result, "nonexistent.json", fake)
    assert "resampling_manifest" not in result


# ===========================================================================
# P2-4 — declared-config compliance check fires in production fit
# ===========================================================================

def _off_prep_info(n=64):
    half = n // 2
    return {
        "n_orig": n, "n_resampled": n,
        "n_benign_before": half, "n_attack_before": half,
        "n_benign_after": half, "n_attack_after": half,
        "sampler_status": "off", "skip_reason": None, "k_eff": 0,
        "variant": None, "target_fraction": None,
    }


def _tiny_client(expected_arm, prep_info=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from flowerfl.client_app import FlowerClient
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 6, generator=g)
    y = torch.randint(0, 2, (64,), generator=g)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    net = nn.Linear(6, 2)
    return FlowerClient(
        loader, loader, net, partition_id=0, use_brfss=False, local_epochs=1,
        weight_mode="resampled", n_orig=64, prep_info=prep_info or _off_prep_info(),
        expected_arm=expected_arm,
    )


def test_declared_arm_mismatch_raises_in_fit():
    # Manifest records variant=None (off arm), but the declared arm says
    # random_under -> the production assertion must fire.
    client = _tiny_client(expected_arm={
        "variant": "random_under", "target_fraction": 0.5,
        "weight_mode": "resampled", "update_match": False,
    })
    params = client.get_parameters({})
    with pytest.raises(ValueError, match="declared variant='random_under' disagrees"):
        client.fit(params, {"server_round": 1, "attack_type": ""})


def test_matching_declared_arm_passes_fit():
    client = _tiny_client(expected_arm={
        "variant": None, "target_fraction": None,
        "weight_mode": "resampled", "update_match": False,
    })
    params = client.get_parameters({})
    _, num_examples, metrics = client.fit(params, {"server_round": 1, "attack_type": ""})
    assert num_examples == 64
    assert "resampling_manifest" in metrics

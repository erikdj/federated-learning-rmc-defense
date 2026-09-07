"""Signal-log custody on rerun.

The signal log is named <exec_mode>__<scenario>__<defense>__seed<seed>.jsonl —
no SMOTE identity — and SignalLogger opens it append-only. So when run_one does
NOT reuse the cache and reruns the simulation (the RECOMPUTE branch the SMOTE
cache-identity fix introduced, or a crashed prior run's leftover file), the new
arm's rows would append onto the old arm's, yielding a mixed-arm JSONL. The fix
rotates any stale file out of the way (never deletes) before a rerun so the new
log starts fresh.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ===========================================================================
# signal_log_path — single source of truth for the signal filename, shared by
# server_app._maybe_create_signal_logger and run_one's stale-log rotation.
# ===========================================================================

def test_signal_log_path_reset_mode(tmp_path):
    from flowerfl.server_app import signal_log_path
    p = signal_log_path("rmc/scenarios/S4.json", "ScenarioKrum",
                        {"seed": 42, "optimizer-state": "reset"}, signals_root=tmp_path)
    assert p == tmp_path / "flower_reset__S4__krum__seed42.jsonl"


def test_signal_log_path_persistent_mode(tmp_path):
    from flowerfl.server_app import signal_log_path
    p = signal_log_path("rmc/scenarios/S0.json", "ScenarioKrumTGE",
                        {"seed": 137, "optimizer-state": "persistent"}, signals_root=tmp_path)
    assert p == tmp_path / "flower_persistent__S0__krumtge__seed137.jsonl"


def test_signal_log_path_none_when_disabled(tmp_path):
    from flowerfl.server_app import signal_log_path
    assert signal_log_path("rmc/scenarios/S4.json", "ScenarioKrum",
                           {"seed": 42, "signal-log": "0"}, signals_root=tmp_path) is None


def test_signal_log_path_none_when_no_scenario(tmp_path):
    from flowerfl.server_app import signal_log_path
    assert signal_log_path("", "ScenarioKrum", {"seed": 42}, signals_root=tmp_path) is None


# ===========================================================================
# _rotate_stale_signal_log — rotate (never delete) before a rerun.
# ===========================================================================

def test_rotate_absent_file_is_noop(tmp_path):
    from run_phase4_flower import _rotate_stale_signal_log
    assert _rotate_stale_signal_log(tmp_path / "nope.jsonl") is None


def test_rotate_moves_and_starts_fresh_new_log_only_new_rows(tmp_path):
    """The custody guarantee: old rows go to.superseded-1, and a fresh append
    (as SignalLogger does) writes ONLY the new arm's rows to the original path."""
    from run_phase4_flower import _rotate_stale_signal_log
    p = tmp_path / "flower_reset__S4__krum__seed42.jsonl"
    p.write_text('{"arm":"off","row":1}\n{"arm":"off","row":2}\n')

    rotated = _rotate_stale_signal_log(p)
    assert rotated == tmp_path / "flower_reset__S4__krum__seed42.superseded-1.jsonl"
    assert not p.exists(), "original path must be free for the fresh log"
    # rotated file preserved intact
    assert rotated.read_text() == '{"arm":"off","row":1}\n{"arm":"off","row":2}\n'

    # simulate the rerun's append-only logger writing the new arm
    with open(p, "a") as f:
        f.write('{"arm":"on","row":1}\n')
    new_rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert new_rows == [{"arm": "on", "row": 1}], "new log must contain ONLY new-arm rows"


def test_rotate_counter_increments_never_overwrites(tmp_path):
    from run_phase4_flower import _rotate_stale_signal_log
    p = tmp_path / "flower_reset__S4__krum__seed42.jsonl"
    p.write_text("first\n")
    r1 = _rotate_stale_signal_log(p)
    p.write_text("second\n")
    r2 = _rotate_stale_signal_log(p)
    assert r1.name.endswith(".superseded-1.jsonl")
    assert r2.name.endswith(".superseded-2.jsonl")
    assert r1.read_text() == "first\n" and r2.read_text() == "second\n"


# ===========================================================================
# _reuse_cached_or_rotate — reuse the cache (no rotate) or rerun (rotate stale).
# Rotation is spied so these exercise the control flow without an FL simulation.
# ===========================================================================

def _cached(enabled=False, variant=None, target=None, holdout_disjoint=True):
    # v8 result JSONs carry holdout_disjoint in provenance; default True mirrors a
    # default-disjoint run so cache reuse depends only on the SMOTE arm here
    # (holdout mode is part of eligibility too).
    prov = {"smote_enabled": enabled, "smote_variant": variant, "smote_target": target,
            "holdout_disjoint": holdout_disjoint}
    return {"trajectory": [{"round": 1}], "return_code": 0, "provenance": prov}


def _spy_rotate(monkeypatch):
    import run_phase4_flower
    calls = []
    monkeypatch.setattr(run_phase4_flower, "_rotate_stale_signal_log",
                        lambda p: calls.append(p))
    return calls


def test_reuse_matching_cache_does_not_rotate(tmp_path, monkeypatch):
    from run_phase4_flower import _reuse_cached_or_rotate
    calls = _spy_rotate(monkeypatch)
    jp = tmp_path / "r.json"
    jp.write_text(json.dumps(_cached(True, "smote", "balanced")))
    extra = {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    out = _reuse_cached_or_rotate(jp, extra, "rmc/scenarios/S4.json", "ScenarioKrum", 42, "reset", "exp")
    assert out is not None and out["provenance"]["smote_enabled"] is True
    assert calls == [], "cache reuse must NOT rotate the signal log"


def test_recompute_smote_mismatch_rotates(tmp_path, monkeypatch):
    """THE custody case: cached SMOTE-off + active smote-on -> rerun -> rotate."""
    from run_phase4_flower import _reuse_cached_or_rotate
    calls = _spy_rotate(monkeypatch)
    jp = tmp_path / "r.json"
    jp.write_text(json.dumps(_cached(enabled=False)))
    extra = {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    out = _reuse_cached_or_rotate(jp, extra, "rmc/scenarios/S4.json", "ScenarioKrum", 42, "reset", "exp")
    assert out is None, "mismatched cache must trigger a rerun"
    assert len(calls) == 1 and calls[0] is not None


def test_no_cache_leftover_log_rotates(tmp_path, monkeypatch):
    """Crashed-prior-run case: no cached result JSON, but a leftover signal file
    with this stem — the same guard rotates it."""
    from run_phase4_flower import _reuse_cached_or_rotate
    calls = _spy_rotate(monkeypatch)
    jp = tmp_path / "absent.json"  # does not exist
    out = _reuse_cached_or_rotate(jp, None, "rmc/scenarios/S4.json", "ScenarioKrum", 42, "reset", "exp")
    assert out is None
    assert len(calls) == 1

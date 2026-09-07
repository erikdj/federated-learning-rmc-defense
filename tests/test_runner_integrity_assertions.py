"""Spec § 4.9 integrity assertions (one test per assertion A1-A6).

Each test forces the violation condition and asserts the runner fails loudly
with a diagnostic message naming the assertion.
"""
from __future__ import annotations

import pytest

from scripts.run_phase4_flower import (
    NUM_SUPERNODES,
    _assert_integrity_markers_emitted,
    _assert_lr_matches_locked,
    _assert_n_malicious_per_round,
    _assert_optimizer_state_self_consistent,
    _assert_participants_per_round,
    _assert_signal_log_filename,
    _assert_trajectory_non_empty,
    _load_locked_lr,
    _parse_participants_markers,
    _scenario_declared_malicious_per_round,
)


@pytest.mark.unit
def test_a1_lr_matches_locked_persistent_ok():
    locked = _load_locked_lr("persistent")
    _assert_lr_matches_locked(recorded_lr=locked, optimizer_state="persistent")


@pytest.mark.unit
def test_a1_lr_matches_locked_reset_ok():
    locked = _load_locked_lr("reset")
    _assert_lr_matches_locked(recorded_lr=locked, optimizer_state="reset")


@pytest.mark.unit
def test_a1_lr_mismatch_persistent_raises():
    wrong = _load_locked_lr("persistent") + 1.0  # guaranteed mismatch
    with pytest.raises(AssertionError, match="A1.*lr.*persistent"):
        _assert_lr_matches_locked(recorded_lr=wrong, optimizer_state="persistent")


@pytest.mark.unit
def test_a1_lr_mismatch_reset_raises():
    wrong = _load_locked_lr("reset") + 1.0
    with pytest.raises(AssertionError, match="A1.*lr.*reset"):
        _assert_lr_matches_locked(recorded_lr=wrong, optimizer_state="reset")


@pytest.mark.unit
def test_a2_optimizer_state_matches_ok():
    _assert_optimizer_state_self_consistent(
        recorded_state="persistent",
        cli_state="persistent",
    )


@pytest.mark.unit
def test_a2_optimizer_state_mismatch_raises():
    with pytest.raises(AssertionError, match="A2.*optimizer_state"):
        _assert_optimizer_state_self_consistent(
            recorded_state="reset",
            cli_state="persistent",
        )


@pytest.mark.unit
def test_a2_optimizer_state_matches_ok_reset():
    _assert_optimizer_state_self_consistent(
        recorded_state="reset",
        cli_state="reset",
    )


@pytest.mark.unit
def test_a2_case_insensitive_does_not_raise():
    # run_config may capitalise the value; that is not a drift violation.
    _assert_optimizer_state_self_consistent(
        recorded_state="PERSISTENT",
        cli_state="persistent",
    )


@pytest.mark.unit
def test_a3_participants_markers_parsed():
    log = (
        "[FL] some other log line\n"
        "[Integrity] round=1 participants=21 n_malicious=9\n"
        "[FL] another line\n"
        "[Integrity] round=2 participants=20 n_malicious=9\n"
    )
    rows = _parse_participants_markers(log)
    assert rows == [
        {"round": 1, "participants": 21, "n_malicious": 9},
        {"round": 2, "participants": 20, "n_malicious": 9},
    ]


@pytest.mark.unit
def test_a3_participants_ok():
    rows = [
        {"round": 1, "participants": 21, "n_malicious": 9},
        {"round": 2, "participants": 20, "n_malicious": 9},
    ]
    _assert_participants_per_round(rows, num_supernodes=NUM_SUPERNODES)


@pytest.mark.unit
def test_a3_participants_truncated_raises():
    rows = [
        {"round": 1, "participants": 21, "n_malicious": 9},
        {"round": 2, "participants": 7, "n_malicious": 3},  # Ray truncation
    ]
    with pytest.raises(AssertionError, match="A3.*participants.*round=2"):
        _assert_participants_per_round(rows, num_supernodes=NUM_SUPERNODES)


@pytest.mark.unit
def test_a3_empty_rows_does_not_raise():
    _assert_participants_per_round([], num_supernodes=NUM_SUPERNODES)


@pytest.mark.unit
def test_a3_honors_scenario_scheduled_disconnects():
    """Regression (B5, 2026-06-05): scenarios with identity-reset/disconnect blocks
    (S3, S4) legitimately schedule FEWER participants in some rounds (e.g. round 7 of
    S4 = 11 by design). A3's flat 0.95*num_supernodes floor false-positived on those,
    which would abort every S3/S4 run. When a scenario_dict is supplied, A3 must use
    the per-round DECLARED participant count and fail only on true truncation
    (observed < declared). Caught by the S4 TGE smoke.
    """
    scenario = {
        "schedule": [
            {"rounds": [1, 6], "participants": ["c%d" % i for i in range(20)]},
            {"rounds": [7, 7], "participants": ["c%d" % i for i in range(11)]},  # disconnect block
            {"rounds": [8, 8], "participants": ["c%d" % i for i in range(20)]},
        ]
    }
    # Observed exactly matches the schedule (incl. the legit 11 at round 7) -> no raise.
    ok_rows = [
        {"round": 6, "participants": 20, "n_malicious": 0},
        {"round": 7, "participants": 11, "n_malicious": 0},
        {"round": 8, "participants": 20, "n_malicious": 0},
    ]
    _assert_participants_per_round(ok_rows, num_supernodes=NUM_SUPERNODES, scenario_dict=scenario)

    # Real truncation: round 8 declared 20 but only 11 observed -> must raise.
    trunc_rows = [{"round": 8, "participants": 11, "n_malicious": 0}]
    with pytest.raises(AssertionError, match="A3.*round=8"):
        _assert_participants_per_round(trunc_rows, num_supernodes=NUM_SUPERNODES, scenario_dict=scenario)


@pytest.mark.unit
def test_a3b_short_trajectory_allows_empty_markers():
    # Smoke test or discovery-only run — no markers OK
    _assert_integrity_markers_emitted(
        integrity_rows=[],
        trajectory=[{"round": 1, "accuracy": 0.5, "f1": 0.5}],
    )


@pytest.mark.unit
def test_a3b_long_trajectory_no_markers_raises():
    with pytest.raises(AssertionError, match="A3.b.*missing integrity markers"):
        _assert_integrity_markers_emitted(
            integrity_rows=[],
            trajectory=[
                {"round": r, "accuracy": 0.5, "f1": 0.5} for r in range(1, 11)
            ],
        )


@pytest.mark.unit
def test_a4_scenario_declared_counts_canonical_schedule_format():
    """Validate the schedule-format parser per existing rmc/scenarios/*.json shape."""
    sc = {
        "name": "test_scenario",
        "num_rounds": 12,
        "clients": {f"client_{i}": {"physical_id": i} for i in range(20)},
        "schedule": [
            # Discovery block: no attacks
            {"rounds": [1, 5], "participants": [f"client_{i}" for i in range(20)], "attacks": {}},
            # Attack block rounds 6..10: 9 ALIE adversaries
            {
                "rounds": [6, 10],
                "participants": [f"client_{i}" for i in range(20)],
                "attacks": {
                    f"client_{i}": {"type": "alie", "params": {"z_max": 0.9}}
                    for i in range(9)
                },
            },
            # Honest block rounds 11..12
            {"rounds": [11, 12], "participants": [f"client_{i}" for i in range(20)], "attacks": {}},
        ],
    }
    declared = _scenario_declared_malicious_per_round(sc)
    assert declared[1] == 0
    assert declared[5] == 0
    assert declared[6] == 9
    assert declared[10] == 9
    assert declared[11] == 0
    assert declared[12] == 0


@pytest.mark.unit
def test_a4_skip_scheduling_blocks_not_counted():
    """Blocks with skip_scheduling=True must not contribute to declared counts."""
    sc = {
        "schedule": [
            {
                "rounds": [1, 5],
                "participants": [],
                "attacks": {"client_0": {"type": "alie"}},
                "skip_scheduling": True,  # metadata-only, must be ignored
            },
            {
                "rounds": [3, 7],
                "participants": [],
                "attacks": {"client_0": {"type": "alie"}, "client_1": {"type": "alie"}},
            },
        ],
    }
    declared = _scenario_declared_malicious_per_round(sc)
    assert declared.get(1, 0) == 0  # only the skip_scheduling block covers round 1
    assert declared[3] == 2
    assert declared[7] == 2


@pytest.mark.unit
def test_a4_n_malicious_match_ok():
    """Observed counts that match the scenario declaration should not raise."""
    sc = {
        "schedule": [
            {
                "rounds": [1, 2],
                "participants": [],
                "attacks": {f"client_{i}": {"type": "alie"} for i in range(9)},
            },
        ],
    }
    rows = [
        {"round": 1, "participants": 21, "n_malicious": 9},
        {"round": 2, "participants": 21, "n_malicious": 9},
    ]
    _assert_n_malicious_per_round(rows, sc)


@pytest.mark.unit
def test_a4_n_malicious_silent_drop_raises():
    """Observed n_malicious lower than declared = Bug #1 silent drop."""
    sc = {
        "schedule": [
            {
                "rounds": [1, 2],
                "participants": [],
                "attacks": {f"client_{i}": {"type": "alie"} for i in range(9)},
            },
        ],
    }
    rows = [
        {"round": 1, "participants": 21, "n_malicious": 9},
        {"round": 2, "participants": 21, "n_malicious": 7},  # silent drop
    ]
    with pytest.raises(AssertionError, match="A4.*n_malicious.*round=2.*expected=9.*observed=7"):
        _assert_n_malicious_per_round(rows, sc)


@pytest.mark.unit
def test_a4_round_not_in_schedule_treated_as_zero():
    """A marker round with no schedule coverage must compare against 0."""
    sc = {"schedule": []}
    rows = [{"round": 5, "participants": 21, "n_malicious": 0}]
    _assert_n_malicious_per_round(rows, sc)  # 0 expected, 0 observed → ok

    rows_bad = [{"round": 5, "participants": 21, "n_malicious": 3}]
    with pytest.raises(AssertionError, match="A4.*expected=0.*observed=3"):
        _assert_n_malicious_per_round(rows_bad, sc)


@pytest.mark.unit
def test_a5_filename_persistent_ok(tmp_path):
    sig = tmp_path / "flower_persistent__S3_identity_reset_only__krum__seed42.jsonl"
    sig.write_text("{}\n")
    _assert_signal_log_filename(
        signal_dir=tmp_path,
        scenario_name="S3_identity_reset_only",
        defense="krum",
        seed=42,
        cli_optimizer_state="persistent",
    )


@pytest.mark.unit
def test_a5_filename_reset_ok(tmp_path):
    sig = tmp_path / "flower_reset__S3_identity_reset_only__krum__seed42.jsonl"
    sig.write_text("{}\n")
    _assert_signal_log_filename(
        signal_dir=tmp_path,
        scenario_name="S3_identity_reset_only",
        defense="krum",
        seed=42,
        cli_optimizer_state="reset",
    )


@pytest.mark.unit
def test_a5_filename_mode_aliased_raises(tmp_path):
    """Persistent run found only a reset signal log → mode-aliasing bug."""
    (tmp_path / "flower_reset__S3_identity_reset_only__krum__seed42.jsonl").write_text("{}\n")
    with pytest.raises(AssertionError, match="A5.*signal log.*flower_persistent"):
        _assert_signal_log_filename(
            signal_dir=tmp_path,
            scenario_name="S3_identity_reset_only",
            defense="krum",
            seed=42,
            cli_optimizer_state="persistent",
        )


@pytest.mark.unit
def test_a5_accepts_raw_scenario_strategy_token(tmp_path):
    """Regression (B4, 2026-06-05): the runner call site passes the RAW strategy
    class name (e.g. 'ScenarioTGEnsemble', 'ScenarioKrum') as `defense`, but the
    signal logger writes the NORMALIZED token (server_app.py:76 does
    strategy_name.replace('Scenario','').lower() -> 'tgensemble', 'krum').

    A5 must normalize the same way so it tolerates both forms; otherwise it aborts
    EVERY real run at the final gate. Caught by the S0 TGE local smoke.
    """
    # Actual filename uses the normalized token 'tgensemble'.
    (tmp_path / "flower_persistent__S0_clean_baseline__tgensemble__seed42.jsonl").write_text("{}\n")
    # Caller passes the raw strategy class name — must NOT raise.
    _assert_signal_log_filename(
        signal_dir=tmp_path,
        scenario_name="S0_clean_baseline",
        defense="ScenarioTGEnsemble",
        seed=42,
        cli_optimizer_state="persistent",
    )
    # Same for Krum/TrustScore raw tokens.
    (tmp_path / "flower_persistent__S0_clean_baseline__krum__seed7.jsonl").write_text("{}\n")
    _assert_signal_log_filename(
        signal_dir=tmp_path,
        scenario_name="S0_clean_baseline",
        defense="ScenarioKrum",
        seed=7,
        cli_optimizer_state="persistent",
    )


@pytest.mark.unit
def test_a5_filename_case_insensitive_cli(tmp_path):
    """CLI value may be uppercase; expected_mode mapping must lower it first."""
    (tmp_path / "flower_persistent__S0_clean_baseline__trustscore__seed137.jsonl").write_text("{}\n")
    _assert_signal_log_filename(
        signal_dir=tmp_path,
        scenario_name="S0_clean_baseline",
        defense="trustscore",
        seed=137,
        cli_optimizer_state="PERSISTENT",
    )


@pytest.mark.unit
def test_a5_signal_dir_absent_raises(tmp_path):
    """If signal logging is enabled but the dir was never created, A5 must raise."""
    nonexistent = tmp_path / "no_such_dir"
    assert not nonexistent.exists()
    with pytest.raises(AssertionError, match="A5.*signal log.*flower_persistent"):
        _assert_signal_log_filename(
            signal_dir=nonexistent,
            scenario_name="S0_clean_baseline",
            defense="krum",
            seed=42,
            cli_optimizer_state="persistent",
        )


@pytest.mark.unit
def test_a6_trajectory_non_empty_ok():
    # Single trajectory with 50% of declared rounds passes
    traj = [{"round": r, "accuracy": 0.8, "f1": 0.7} for r in range(1, 26)]  # 25 of 50 = 50%
    _assert_trajectory_non_empty(traj, rounds=50)
    # Full trajectory also OK
    traj_full = [{"round": r, "accuracy": 0.5, "f1": 0.5} for r in range(1, 51)]
    _assert_trajectory_non_empty(traj_full, rounds=50)


@pytest.mark.unit
def test_a6_empty_trajectory_raises():
    with pytest.raises(AssertionError, match="A6.*empty trajectory"):
        _assert_trajectory_non_empty([], rounds=50)


@pytest.mark.unit
def test_a6_short_trajectory_raises():
    # Only 5 of 50 rounds emitted eval lines — almost certainly a FixedEvalManager fault
    traj = [{"round": r, "accuracy": 0.5, "f1": 0.5} for r in range(1, 6)]
    with pytest.raises(AssertionError, match="A6.*trajectory length"):
        _assert_trajectory_non_empty(traj, rounds=50)


@pytest.mark.unit
def test_a6_short_run_threshold():
    # 10-round run with 5 eval rows = 50% — should pass (exactly at threshold).
    traj = [{"round": r, "accuracy": 0.5, "f1": 0.5} for r in range(1, 6)]
    _assert_trajectory_non_empty(traj, rounds=10)
    # 10-round run with 4 eval rows = 40% — should fail (below 50% threshold).
    traj2 = [{"round": r, "accuracy": 0.5, "f1": 0.5} for r in range(1, 5)]
    with pytest.raises(AssertionError, match="A6.*trajectory length"):
        _assert_trajectory_non_empty(traj2, rounds=10)


@pytest.mark.unit
def test_parse_alie_round_set_structured_marker():
    """C2 regression: must detect ALIE rounds via the structured marker,
    not via the legacy upper-slot substring."""
    from scripts.run_phase4_flower import parse_alie_round_set

    log = (
        "[ScenarioStrategy] Round 8 (scenario R7): 20/21 clients selected (attacking: ['client_0', 'client_1'])\n"
        "[Integrity] round=7 participants=20 n_malicious=9\n"
        "[Integrity] round=7 alie_active=1 n_alie=9\n"
        "[ScenarioStrategy] Round 14 (scenario R13): 20/21 clients selected (attacking: ['client_0_new1'])\n"
        "[Integrity] round=13 participants=20 n_malicious=9\n"
        "[Integrity] round=13 alie_active=1 n_alie=9\n"
    )
    rounds = parse_alie_round_set(log)
    # structured marker returns server_round = scenario_round + 1
    assert rounds == {8, 14}


@pytest.mark.unit
def test_load_locked_lr_invalidates_on_file_mtime_change(tmp_path, monkeypatch):
    """M3 regression: lru_cache replaced with mtime-aware cache so mid-process
    edits to data/hparams_locked.json don't silently use stale values."""
    import json as _json
    import time
    from scripts import run_phase4_flower as rpf

    # Point the runner at a tmp hparams file
    fake = tmp_path / "hparams_locked.json"
    fake.write_text(_json.dumps({
        "flower_reset": {"lr": 0.005},
        "persistent_optimizer": {"lr": 0.001},
    }))
    monkeypatch.setattr(rpf, "_HPARAMS_LOCKED_PATH", fake)
    rpf._LOAD_LOCKED_LR_CACHE.clear()

    assert rpf._load_locked_lr("persistent") == 0.001

    # Bump mtime + change value — sleep to guarantee mtime differs on FAT/ext4
    time.sleep(0.05)
    fake.write_text(_json.dumps({
        "flower_reset": {"lr": 0.005},
        "persistent_optimizer": {"lr": 0.999},  # changed
    }))

    assert rpf._load_locked_lr("persistent") == 0.999, (
        "cache must invalidate when file mtime changes"
    )

    # Restore for other tests
    rpf._LOAD_LOCKED_LR_CACHE.clear()


@pytest.mark.unit
def test_parse_alie_round_set_legacy_fallback():
    """Legacy upper-slot scenarios with no structured marker still parse."""
    from scripts.run_phase4_flower import parse_alie_round_set

    log = (
        "Round 15 (scenario R14): 20/21 clients selected (attacking: ['client_11_new1', 'client_12_new1'])\n"
    )
    rounds = parse_alie_round_set(log)
    assert rounds == {15}  # server_round from group(1) in legacy format

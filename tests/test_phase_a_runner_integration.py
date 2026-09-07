"""Integration test for the Phase A runner's stdout-parsing + assertion pipeline.

Does NOT boot Flower (too heavy). Instead, builds synthetic captured-stdout
matching what ScenarioStrategy.configure_fit would emit for each S0..S4
schedule, and exercises the runner's parsers/assertions end-to-end.

This is the test class that would have caught C1 (S3 disconnect-block
participants leak) and C2 (parse_alie_round_set hardcoded to upper-slot).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "rmc" / "scenarios"


def _scenario(name: str) -> dict:
    return json.loads((SCENARIOS_DIR / name).read_text())


def _expected_markers(scenario_dict: dict) -> list[str]:
    """Build the integrity markers we'd expect the strategy to emit for
    every scheduled round."""
    lines = []
    for block in scenario_dict.get("schedule", []):
        if block.get("skip_scheduling"):
            continue
        start, end = block["rounds"]
        attacks = block.get("attacks") or {}
        n_malicious = sum(1 for v in attacks.values() if v and v.get("type"))
        n_alie = sum(1 for v in attacks.values() if v and v.get("type") == "alie")
        n_participants = len(block.get("participants", []))
        for r in range(start, end + 1):
            lines.append(
                f"[Integrity] round={r} participants={n_participants} "
                f"n_malicious={n_malicious}"
            )
            if n_alie > 0:
                lines.append(
                    f"[Integrity] round={r} alie_active=1 n_alie={n_alie}"
                )
    return lines


@pytest.mark.parametrize("scenario_file", [
    "S0_clean_baseline.json",
    "S1_benign_churn_only.json",
    "S2_adaptive_switching_only.json",
    "S3_identity_reset_only.json",
    "S4_full_mix.json",
])
def test_full_integrity_path_synthetic(scenario_file):
    from scripts.run_phase4_flower import (
        _assert_integrity_markers_emitted,
        _assert_n_malicious_per_round,
        _assert_trajectory_non_empty,
        _parse_participants_markers,
    )

    sc = _scenario(scenario_file)
    stdout = "\n".join(_expected_markers(sc)) + "\n"
    rows = _parse_participants_markers(stdout)
    assert rows, f"{scenario_file}: no integrity markers parsed"

    # Build a synthetic trajectory of correct length (matches scenario_round 1..N-1)
    trajectory = [
        {"round": r, "accuracy": 0.5, "f1": 0.5}
        for r in range(1, sc["num_rounds"])
    ]

    _assert_trajectory_non_empty(trajectory, rounds=sc["num_rounds"])
    _assert_integrity_markers_emitted(rows, trajectory)
    # NOTE: _assert_participants_per_round is intentionally omitted here.
    # That assertion checks for Ray actor-pool truncation in live Flower runs
    # (floor = ceil(0.95 * num_supernodes)). Synthetic schedule-based participant
    # counts are legitimately smaller during benign churn blocks (S1, S3, S4),
    # so the floor check would produce false positives in this synthetic context.
    _assert_n_malicious_per_round(rows, sc)


@pytest.mark.parametrize("scenario_file,expected_min_alie_rounds", [
    ("S0_clean_baseline.json", 30),
    ("S2_adaptive_switching_only.json", 10),
    ("S3_identity_reset_only.json", 20),
    ("S4_full_mix.json", 10),
])
def test_alie_round_set_per_scenario(scenario_file, expected_min_alie_rounds):
    """C2 regression: parse_alie_round_set must find ALIE rounds for S0..S4
    (lower-slot adversary convention)."""
    from scripts.run_phase4_flower import parse_alie_round_set

    sc = _scenario(scenario_file)
    stdout = "\n".join(_expected_markers(sc)) + "\n"
    rounds = parse_alie_round_set(stdout)
    assert len(rounds) >= expected_min_alie_rounds, (
        f"{scenario_file}: parsed only {len(rounds)} ALIE rounds, "
        f"expected >= {expected_min_alie_rounds}"
    )


def test_all_scenarios_cover_every_round_without_gaps():
    """H1 regression + schedule-coherency: every scenario_round 1..num_rounds
    is covered by exactly one (non-skip_scheduling) block."""
    for fname in [
        "S0_clean_baseline.json",
        "S1_benign_churn_only.json",
        "S2_adaptive_switching_only.json",
        "S3_identity_reset_only.json",
        "S4_full_mix.json",
    ]:
        sc = _scenario(fname)
        covered: dict[int, int] = {}
        for block in sc["schedule"]:
            if block.get("skip_scheduling"):
                continue
            start, end = block["rounds"]
            for r in range(start, end + 1):
                covered[r] = covered.get(r, 0) + 1
        for r in range(1, sc["num_rounds"] + 1):
            assert covered.get(r, 0) == 1, (
                f"{fname} round {r} coverage = {covered.get(r, 0)} "
                f"(expected exactly 1)"
            )

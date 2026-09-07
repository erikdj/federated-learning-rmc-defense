"""Test the parameterized scenario generator (Option C from May 13 brainstorm).

Validates that scripts/data/generate_scenarios.py produces faithful Szelag-style
scenarios at configurable adversary intensity, matching the schema of the
existing rmc_main_50r scenarios.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from generate_scenarios import (  # noqa: E402
    generate_intensity_scenario,
    generate_continuous_reconnect_scenario,
)


@pytest.mark.parametrize("n_adversaries", [1, 3, 5, 7, 9])
def test_intensity_produces_expected_adversary_count(n_adversaries: int):
    """Generator at intensity=n must produce exactly n unique ALIE adversaries."""
    scenario = generate_intensity_scenario(
        n_adversaries=n_adversaries,
        n_clients=20,
        n_rounds=50,
        scenario_name=f"test_intensity_{n_adversaries}",
    )

    alie_clients = set()
    for block in scenario["schedule"]:
        for cid, atk in (block.get("attacks") or {}).items():
            if atk.get("type") == "alie":
                alie_clients.add(cid)

    assert len(alie_clients) == n_adversaries, (
        f"Expected {n_adversaries} ALIE adversaries, "
        f"got {len(alie_clients)}: {sorted(alie_clients)}"
    )


def test_schema_matches_existing_scenario():
    """Generated scenarios satisfy the runtime schedule schema."""

    generated = generate_intensity_scenario(
        n_adversaries=1,
        n_clients=20,
        n_rounds=50,
        scenario_name="rmc_test",
    )

    # Same top-level keys
    assert set(generated.keys()) >= {"name", "num_rounds", "dataset", "clients", "schedule"}
    # Schedule blocks have rounds + participants + attacks
    for block in generated["schedule"]:
        assert "rounds" in block
        assert "participants" in block


def test_szelag_faithful_at_intensity_9():
    """Intensity=9 must match Szelag paper (45% malicious of 20 clients)."""
    scenario = generate_intensity_scenario(
        n_adversaries=9,
        n_clients=20,
        n_rounds=50,
        scenario_name="rmc_szelag_faithful",
    )

    alie_clients = set()
    gaussian_clients = set()
    for block in scenario["schedule"]:
        for cid, atk in (block.get("attacks") or {}).items():
            if atk.get("type") == "alie":
                alie_clients.add(cid)
            elif atk.get("type") == "gaussian_noise":
                gaussian_clients.add(cid)

    # All 9 adversaries appear as ALIE on reconnect (post-rejoin rounds)
    assert len(alie_clients) == 9
    # Initial phase uses gaussian noise (per Szelag's design)
    assert len(gaussian_clients) >= 8


# ---------------------------------------------------------------------------
# v2 continuous-reconnect scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_adversaries", [1, 3, 5, 7, 9])
def test_v2_cumulative_identity_count(n_adversaries: int):
    """v2 with N reconnect cycles produces n_adversaries × (1 + N) cumulative malicious identities."""
    reconnect_cycles = 4
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=n_adversaries,
        n_clients=20,
        n_rounds=50,
        reconnect_cycles=reconnect_cycles,
    )
    alie_ids = set()
    gauss_ids = set()
    for blk in sc["schedule"]:
        for cid, atk in (blk.get("attacks") or {}).items():
            if atk.get("type") == "alie":
                alie_ids.add(cid)
            elif atk.get("type") == "gaussian_noise":
                gauss_ids.add(cid)
    # Each cycle has its own _newK suffix → cycles × n_adversaries unique ALIE identities
    assert len(alie_ids) == reconnect_cycles * n_adversaries
    # Initial Gaussian wave uses the n original adversary identities
    assert len(gauss_ids) == n_adversaries
    cumulative = n_adversaries * (1 + reconnect_cycles)
    assert len(alie_ids) + len(gauss_ids) == cumulative


def test_v2_schedule_block_count_default():
    """v2 default (4 reconnect cycles) produces 2 (warmup, initial) + 2 × 4 (disconnect+ALIE per cycle) = 10 blocks."""
    sc = generate_continuous_reconnect_scenario(n_adversaries=9, reconnect_cycles=4)
    assert len(sc["schedule"]) == 10


def test_v2_each_cycle_has_unique_suffix():
    """Each reconnect cycle's identities use a distinct `_newK` suffix; no overlap."""
    sc = generate_continuous_reconnect_scenario(n_adversaries=9, reconnect_cycles=4)
    seen_suffixes: dict[str, int] = {}
    for blk in sc["schedule"]:
        for cid in (blk.get("attacks") or {}):
            if "_new" in cid:
                suffix = cid.split("_new")[1]
                seen_suffixes.setdefault(suffix, 0)
                seen_suffixes[suffix] += 1
    assert set(seen_suffixes.keys()) == {"1", "2", "3", "4"}
    # Each cycle's identity-set has the same count as n_adversaries
    for suffix, count in seen_suffixes.items():
        assert count == 9, f"_new{suffix}: expected 9, got {count}"


def test_v2_rounds_cover_n_rounds_exactly():
    """v2 schedule blocks tile [1, n_rounds] without gaps or overlaps."""
    sc = generate_continuous_reconnect_scenario(n_adversaries=9, n_rounds=50, reconnect_cycles=4)
    covered: set[int] = set()
    for blk in sc["schedule"]:
        start, end = blk["rounds"]
        for r in range(start, end + 1):
            assert r not in covered, f"round {r} appears in multiple blocks"
            covered.add(r)
    assert covered == set(range(1, 51))


def test_v2_too_few_rounds_raises():
    """v2 raises ValueError when n_rounds is too small for the requested cycle count."""
    with pytest.raises(ValueError, match="too small"):
        generate_continuous_reconnect_scenario(
            n_adversaries=9, n_rounds=12, reconnect_cycles=4,
        )


def test_v2_physical_id_slots_unique():
    """Every logical identity in the clients dict has a unique physical_id slot."""
    sc = generate_continuous_reconnect_scenario(n_adversaries=9, reconnect_cycles=4)
    physical_ids = [c["physical_id"] for c in sc["clients"].values()]
    assert len(physical_ids) == len(set(physical_ids)), "physical_ids must be unique"
    # 20 originals + 9 adversaries × 4 cycles = 56
    assert len(physical_ids) == 56


def test_continuous_reconnect_default_no_honest_churn():
    """Backward compatibility: default args produce zero honest reconnections."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
    )
    # No client name should have the honest-offline suffix.
    assert not any("offline" in cid for cid in sc["clients"]), \
        "default args must produce no honest_offline identities"


def test_continuous_reconnect_honest_cycles_create_new_identities():
    """honest_disconnect_cycles=3 should add 3 honest_*_offline* identities."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
        honest_reconnect_window=(10, 45),
    )
    offline_ids = [cid for cid in sc["clients"] if "offline" in cid]
    assert len(offline_ids) == 3, f"expected 3 offline identities, got {offline_ids}"
    # All honest reconnect identities must be derived from honest physical IDs (0..10)
    for cid in offline_ids:
        # Pattern: honest_K_offlineN where K is in [0, n_honest)
        assert cid.startswith("honest_"), f"unexpected naming: {cid}"


def test_continuous_reconnect_honest_schedule_in_window():
    """Honest reconnect events must fall within honest_reconnect_window."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
        honest_reconnect_window=(10, 45),
    )
    honest_events = [
        entry for entry in sc["schedule"]
        if entry.get("honest_reconnect", False)
    ]
    assert len(honest_events) >= 3, \
        f"expected at least 3 honest-reconnect schedule entries, got {len(honest_events)}"
    for entry in honest_events:
        start, end = entry["rounds"]
        assert 10 <= start <= 45, f"honest reconnect at round {start} outside [10, 45]"


def test_continuous_reconnect_honest_deterministic_given_seed():
    """Same scenario_name + same parameters → identical honest event schedule."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc_a = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
        scenario_name="test_het",
    )
    sc_b = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
        scenario_name="test_het",
    )
    # Order-preserving comparison of schedule entries
    honest_a = [e for e in sc_a["schedule"] if e.get("honest_reconnect")]
    honest_b = [e for e in sc_b["schedule"] if e.get("honest_reconnect")]
    assert honest_a == honest_b, \
        f"deterministic generation failed: {honest_a!r} != {honest_b!r}"


def test_continuous_reconnect_honest_events_do_not_empty_main_schedule():
    """Honest-reconnect schedule entries must not overwrite main-schedule participants."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
    )
    # For each honest event, the rounds spanning offline → rejoin-1 must still
    # have non-empty participants in SOME non-skip_scheduling schedule entry
    # (the adversary schedule should cover them).
    honest_rounds = set()
    for ev in sc["honest_events"]:
        for r in range(ev["offline_round"], ev["rejoin_round"]):
            honest_rounds.add(r)
    # Find which rounds have an executable (non-skip_scheduling) schedule entry
    for r in honest_rounds:
        executable_entries = [
            e for e in sc["schedule"]
            if not e.get("skip_scheduling", False)
            and e["rounds"][0] <= r <= e["rounds"][1]
        ]
        executable_participants = []
        for e in executable_entries:
            executable_participants.extend(e["participants"])
        assert executable_participants, (
            f"Round {r} (during honest offline window) has no executable "
            f"schedule entry with participants. Honest-reconnect entries must "
            f"carry skip_scheduling=True to avoid overwriting main schedule."
        )


def test_continuous_reconnect_honest_entries_marked_skip_scheduling():
    """Honest-reconnect schedule entries must carry skip_scheduling: True."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        honest_disconnect_cycles=3, honest_cycle_size=1,
    )
    honest_entries = [e for e in sc["schedule"] if e.get("honest_reconnect")]
    assert honest_entries, "no honest_reconnect entries found"
    for e in honest_entries:
        assert e.get("skip_scheduling") is True, (
            f"honest_reconnect schedule entry missing skip_scheduling flag: {e}"
        )


def test_continuous_reconnect_narrow_window_raises_descriptive_error():
    """Window with effective span < 2 rounds must raise a clear ValueError.

    Use a window with win_lo < win_hi (so it bypasses the no-rounds-after-warmup
    guard) but win_hi - win_lo < 2 (so it triggers the new too-narrow guard).
    With warmup=5 + initial_gaussian=5 = 10, win_lo gets clamped UP to 11.
    Setting honest_reconnect_window=(11, 12) → win_lo=11, win_hi=12 → span=1.
    """
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    import pytest
    with pytest.raises(ValueError, match="too narrow"):
        generate_continuous_reconnect_scenario(
            n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
            honest_disconnect_cycles=1, honest_cycle_size=1,
            honest_reconnect_window=(11, 12),
        )


def test_continuous_reconnect_honest_cycle_size_too_large_raises():
    """honest_cycle_size > n_honest must raise a clear ValueError."""
    from scripts.data.generate_scenarios import generate_continuous_reconnect_scenario
    import pytest
    with pytest.raises(ValueError, match="honest_cycle_size"):
        generate_continuous_reconnect_scenario(
            n_adversaries=11, n_clients=20, n_rounds=50, reconnect_cycles=4,
            honest_disconnect_cycles=1, honest_cycle_size=20,  # > n_honest=9
        )


# ---------------------------------------------------------------------------
# PhA-3.1: Canonical ALIE z_max default = 0.9 (Szelag-exact, spec § 4.2)
# ---------------------------------------------------------------------------

def test_default_alie_z_max_is_canonical_09():
    """Spec § 4.2: ALIE z_max canonical = 0.9 (Szelag-exact)."""
    sc1 = generate_intensity_scenario(n_adversaries=9, n_clients=20, n_rounds=50)
    z_values = set()
    for block in sc1["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                z_values.add(attack["params"]["z_max"])
    assert z_values == {0.9}, f"expected ALIE z_max=0.9 canonical, got {z_values}"

    sc2 = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4
    )
    z_values = set()
    for block in sc2["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                z_values.add(attack["params"]["z_max"])
    assert z_values == {0.9}


def test_explicit_alie_z_max_05_still_works_for_sensitivity_arm():
    """Operators can still request z_max=0.5 explicitly for the sensitivity arm."""
    sc = generate_continuous_reconnect_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4,
        alie_z_max=0.5,
    )
    z_values = set()
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                z_values.add(attack["params"]["z_max"])
    assert z_values == {0.5}

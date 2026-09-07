"""Property tests for Design D scenarios (spec § 5).

Validates S0 (clean baseline), S1-S4 scenario generators and emitted JSON files.
All scenarios use the canonical schedule-format per spec § 5 + existing RMC precedent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from generate_scenarios import generate_clean_baseline_scenario  # noqa: E402

SCENARIOS_DIR = PROJECT_ROOT / "rmc" / "scenarios"


def _load(name: str) -> dict:
    """Load scenario JSON file."""
    return json.loads((SCENARIOS_DIR / name).read_text())


@pytest.mark.unit
def test_s0_clean_baseline_shape_from_generator():
    """Generator emits canonical schedule-format JSON for S0 baseline."""
    sc = generate_clean_baseline_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50
    )
    assert sc["num_rounds"] == 50
    # Only the original 20 clients — no `_new1..N` reconnect identities
    assert len(sc["clients"]) == 20
    for cid in sc["clients"]:
        assert "_new" not in cid, f"S0 must have no reconnect identities, got {cid}"

    # Canonical format: "schedule" top-level, "rounds" as range pairs
    assert "schedule" in sc
    for block in sc["schedule"]:
        rounds_field = block["rounds"]
        assert isinstance(rounds_field, list)
        assert len(rounds_field) == 2, (
            f"rounds must be [start, end] inclusive range, got {rounds_field}"
        )
        assert isinstance(rounds_field[0], int) and isinstance(rounds_field[1], int)

    # All ALIE attacks use z_max=0.9 (canonical, spec § 4.2)
    z_values = set()
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                z_values.add(attack["params"]["z_max"])
    assert z_values == {0.9}, f"S0 must use ALIE z_max=0.9, got {z_values}"

    # Each attacking round has exactly 9 adversaries (45% per § 4.1)
    declared_per_round = {}
    for block in sc["schedule"]:
        rounds_field = block["rounds"]
        n = sum(
            1
            for v in (block.get("attacks") or {}).values()
            if v and v.get("type")
        )
        for r in range(rounds_field[0], rounds_field[1] + 1):
            declared_per_round[r] = declared_per_round.get(r, 0) + n
    for r, n in declared_per_round.items():
        assert n in (0, 9), f"round {r} declares {n} attackers (expected 0 or 9)"

    # Single attack type — no strategy switching (no label_flip)
    attack_types = set()
    for block in sc["schedule"]:
        for v in (block.get("attacks") or {}).values():
            if v and v.get("type"):
                attack_types.add(v["type"])
    assert "label_flip" not in attack_types, "S0 must not switch to label_flip"
    # Allow gaussian_noise preface + alie, or just alie
    assert attack_types.issubset({"alie", "gaussian_noise"})


@pytest.mark.unit
def test_s0_clean_baseline_file_exists():
    """The generator output is committed to rmc/scenarios/S0_clean_baseline.json."""
    sc = _load("S0_clean_baseline.json")
    assert sc["num_rounds"] == 50
    assert sc["name"] == "S0_clean_baseline"
    assert sc["dataset"] == "edge_full_20_rmc"
    assert "schedule" in sc


@pytest.mark.unit
def test_s1_benign_churn_only_shape_from_generator():
    from scripts.data.generate_scenarios import generate_benign_churn_only_scenario

    sc = generate_benign_churn_only_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, n_honest_reconnect_cycles=3
    )
    assert sc["num_rounds"] == 50
    assert "schedule" in sc
    for block in sc["schedule"]:
        assert isinstance(block["rounds"], list) and len(block["rounds"]) == 2

    # Honest reconnect identities present
    new_ids = [cid for cid in sc["clients"] if "_new" in cid]
    assert len(new_ids) == 3, f"S1 expects 3 honest reconnect ids, got {new_ids}"

    # ALL `_new` ids must have a base index >= n_adversaries (i.e. honest only)
    for cid in new_ids:
        base = int(cid.split("_")[1])
        assert base >= 9, f"S1 forbids malicious reconnect; got {cid}"

    # No label_flip (no strategy switching)
    attack_types = set()
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type"):
                attack_types.add(attack["type"])
    assert "label_flip" not in attack_types
    assert attack_types == {"alie"}, (
        f"S1 should use only ALIE post-discovery, got {attack_types}"
    )

    # ALIE z_max=0.9 canonical
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                assert attack["params"]["z_max"] == 0.9

    # Every attacking round has exactly 9 adversaries
    declared_per_round = {}
    for block in sc["schedule"]:
        start, end = block["rounds"]
        n = sum(
            1 for v in (block.get("attacks") or {}).values()
            if v and v.get("type")
        )
        for r in range(start, end + 1):
            declared_per_round[r] = declared_per_round.get(r, 0) + n
    for r, n in declared_per_round.items():
        assert n in (0, 9), f"round {r} has {n} attackers (expected 0 or 9)"


@pytest.mark.unit
def test_s1_benign_churn_only_file_exists():
    sc = _load("S1_benign_churn_only.json")
    assert sc["name"] == "S1_benign_churn_only"
    assert sc["num_rounds"] == 50


@pytest.mark.unit
def test_s2_adaptive_switching_only_shape_from_generator():
    from scripts.data.generate_scenarios import generate_adaptive_switching_only_scenario

    sc = generate_adaptive_switching_only_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50, n_switch_cycles=4
    )
    # No reconnect identities (no churn, no reset)
    assert all("_new" not in cid for cid in sc["clients"])
    assert len(sc["clients"]) == 20

    # All three attack types appear (rotation visible)
    seen = set()
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type"):
                seen.add(attack["type"])
    assert seen == {"gaussian_noise", "alie", "label_flip"}, (
        f"S2 must rotate all three attack types; saw {seen}"
    )

    # ALIE blocks use z_max=0.9
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                assert attack["params"]["z_max"] == 0.9

    # Every attacking round has exactly 9 adversaries (zero rotation cycles
    # with mixed types — all 9 adversaries use the same type per cycle)
    declared_per_round = {}
    for block in sc["schedule"]:
        start, end = block["rounds"]
        n = sum(
            1 for v in (block.get("attacks") or {}).values()
            if v and v.get("type")
        )
        for r in range(start, end + 1):
            declared_per_round[r] = declared_per_round.get(r, 0) + n
    for r, n in declared_per_round.items():
        assert n in (0, 9), f"round {r} has {n} attackers (expected 0 or 9)"

    # All blocks have rounds as inclusive range pair
    for block in sc["schedule"]:
        assert isinstance(block["rounds"], list) and len(block["rounds"]) == 2


@pytest.mark.unit
def test_s2_adaptive_switching_only_file_exists():
    sc = _load("S2_adaptive_switching_only.json")
    assert sc["name"] == "S2_adaptive_switching_only"
    assert sc["num_rounds"] == 50


@pytest.mark.unit
def test_s3_identity_reset_only_shape():
    sc = _load("S3_identity_reset_only.json")

    # Roster has malicious reconnect identities (lower-slot base 0..8)
    mal_new = [
        cid for cid in sc["clients"]
        if "_new" in cid and int(cid.split("_")[1]) < 9
    ]
    assert len(mal_new) == 36, (
        f"S3 expects 9 adversaries × 4 cycles = 36 _new ids, got {len(mal_new)}"
    )

    # No honest _new identities (no benign churn)
    honest_new = [
        cid for cid in sc["clients"]
        if "_new" in cid and int(cid.split("_")[1]) >= 9
    ]
    assert honest_new == [], f"S3 forbids honest reconnects, got {honest_new}"

    # No label_flip (no strategy switching) — only alie + gaussian_noise
    seen_types = set()
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type"):
                seen_types.add(attack["type"])
    assert "label_flip" not in seen_types
    assert seen_types.issubset({"alie", "gaussian_noise"})

    # Canonical ALIE z_max=0.9
    for block in sc["schedule"]:
        for attack in (block.get("attacks") or {}).values():
            if attack and attack.get("type") == "alie":
                assert attack["params"]["z_max"] == 0.9

    # COHERENCY: every attacker referenced in the schedule must exist in the
    # roster (clients dict). The previous bug had _new1 attackers without
    # corresponding roster entries.
    roster_ids = set(sc["clients"].keys())
    for block in sc["schedule"]:
        for cid in (block.get("attacks") or {}):
            assert cid in roster_ids, f"S3 attacker {cid} not in clients roster"
        for cid in block.get("participants", []):
            assert cid in roster_ids, f"S3 participant {cid} not in clients roster"

    # Every attacking block has exactly 9 attackers (45%)
    for block in sc["schedule"]:
        atks = block.get("attacks") or {}
        n = sum(1 for v in atks.values() if v and v.get("type"))
        assert n in (0, 9), f"block {block['rounds']} has {n} attackers"


@pytest.mark.unit
def test_s3_identity_reset_only_file_exists():
    """The S3 JSON file is committed and has expected structure."""
    sc = _load("S3_identity_reset_only.json")
    assert sc["name"] == "S3_identity_reset_only"
    assert sc["num_rounds"] == 50


@pytest.mark.unit
def test_s4_full_mix_shape_from_generator():
    from scripts.data.generate_scenarios import generate_full_mix_scenario

    sc = generate_full_mix_scenario(
        n_adversaries=9, n_clients=20, n_rounds=50,
        malicious_reconnect_cycles=4, n_honest_reconnect_cycles=3,
    )
    assert sc["num_rounds"] == 50

    # Malicious reconnect identities (lower-slot)
    mal_new = [
        cid for cid in sc["clients"]
        if "_new" in cid and int(cid.split("_")[1]) < 9
    ]
    assert len(mal_new) == 36, f"expected 9 × 4 = 36 mal _new ids, got {len(mal_new)}"

    # Honest reconnect identities (upper-slot)
    honest_new = [
        cid for cid in sc["clients"]
        if "_new" in cid and int(cid.split("_")[1]) >= 9
    ]
    assert len(honest_new) == 3, f"expected 3 honest _new ids, got {len(honest_new)}"

    # All three confounders → all three attack types present
    seen = set()
    for block in sc["schedule"]:
        for v in (block.get("attacks") or {}).values():
            if v and v.get("type"):
                seen.add(v["type"])
    assert seen == {"gaussian_noise", "alie", "label_flip"}, (
        f"S4 must exhibit all 3 attack types via rotation; saw {seen}"
    )

    # ALIE blocks use z_max=0.9 canonical
    for block in sc["schedule"]:
        for v in (block.get("attacks") or {}).values():
            if v and v.get("type") == "alie":
                assert v["params"]["z_max"] == 0.9

    # Coherency: every attacker is in roster
    roster = set(sc["clients"].keys())
    for block in sc["schedule"]:
        for cid in (block.get("attacks") or {}):
            assert cid in roster, f"S4 attacker {cid} not in roster"
        for cid in block.get("participants", []):
            assert cid in roster, f"S4 participant {cid} not in roster"


@pytest.mark.unit
def test_s4_full_mix_file_exists():
    sc = _load("S4_full_mix.json")
    assert sc["name"] == "S4_full_mix"


@pytest.mark.unit
def test_s3_disconnect_blocks_exclude_all_malicious_originals():
    """C1 regression: disconnect blocks must NOT include client_0..client_8 (the
    malicious originals that conceptually left at cycle 1). Prior bug had them
    running as honest during disconnect windows, contaminating the federated
    average."""
    sc = _load("S3_identity_reset_only.json")
    for block in sc["schedule"]:
        if "disconnect" in (block.get("comment") or "").lower():
            for i in range(9):
                assert f"client_{i}" not in block["participants"], (
                    f"S3 disconnect block {block['rounds']}: client_{i} "
                    f"present in participants (would run as honest)"
                )
    assert sc["num_rounds"] == 50

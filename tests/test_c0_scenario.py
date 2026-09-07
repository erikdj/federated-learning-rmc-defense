"""C0_clean_no_attack — the v1.10 D1 clean-utility reference (erratum-A E1).

Asserts the committed scenario JSON is S0's exact population/rounds/schedule
shape with ZERO malicious entries and no attack blocks anywhere, that the
generator regenerates it byte-consistently from its spec, and that the
scenario layer derives an empty adversary set from it (so
kept_set_malicious_fraction is 0.0 by construction for every kept set).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

C0_PATH = PROJECT_ROOT / "rmc" / "scenarios" / "C0_clean_no_attack.json"
C0_SPEC = PROJECT_ROOT / "rmc" / "scenario_specs" / "C0_clean_no_attack.spec.json"
S0_PATH = PROJECT_ROOT / "rmc" / "scenarios" / "S0_clean_baseline.json"


def _c0() -> dict:
    return json.loads(C0_PATH.read_text())


def _s0() -> dict:
    return json.loads(S0_PATH.read_text())


@pytest.mark.unit
def test_c0_files_are_committed():
    assert C0_PATH.is_file()
    assert C0_SPEC.is_file()


@pytest.mark.unit
def test_c0_matches_s0_population_and_schedule_shape():
    c0, s0 = _c0(), _s0()
    assert c0["num_rounds"] == s0["num_rounds"] == 50
    assert c0["dataset"] == s0["dataset"]
    assert list(c0["clients"]) == list(s0["clients"])
    assert [b["rounds"] for b in c0["schedule"]] == \
           [b["rounds"] for b in s0["schedule"]]
    assert [b["participants"] for b in c0["schedule"]] == \
           [b["participants"] for b in s0["schedule"]]


@pytest.mark.unit
def test_c0_has_zero_malicious_entries_and_no_attack_blocks():
    c0 = _c0()
    for block in c0["schedule"]:
        assert block.get("attacks") in ({}, None) or not block["attacks"]
    # And through the scenario layer's own ground-truth derivation:
    from flowerfl.scenario_strategy import adversarial_identities

    assert adversarial_identities(c0["schedule"]) == set()


@pytest.mark.unit
def test_s0_is_not_a_no_attack_control():
    """The reason C0 exists (v1.10 D1): S0 carries sustained attackers."""
    from flowerfl.scenario_strategy import adversarial_identities

    assert len(adversarial_identities(_s0()["schedule"])) == 9


@pytest.mark.unit
def test_generator_reproduces_the_committed_c0():
    from generate_scenarios import generate_scenario_from_spec

    regenerated = generate_scenario_from_spec(C0_SPEC)
    assert regenerated == _c0()


@pytest.mark.unit
def test_generator_refuses_nonzero_adversaries():
    from generate_scenarios import generate_clean_no_attack_scenario

    with pytest.raises(ValueError):
        generate_clean_no_attack_scenario(n_clients=10)
    with pytest.raises(ValueError):
        generate_clean_no_attack_scenario(n_rounds=40)


@pytest.mark.unit
def test_c0_participant_counts_are_full_every_round():
    """A3-style shape check: 20 participants declared for every round 1..50."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_flower import (
        _scenario_declared_malicious_per_round,
        _scenario_declared_participants_per_round,
    )

    c0 = _c0()
    participants = _scenario_declared_participants_per_round(c0)
    assert set(participants) == set(range(1, 51))
    assert set(participants.values()) == {20}
    malicious = _scenario_declared_malicious_per_round(c0)
    assert all(v == 0 for v in malicious.values())

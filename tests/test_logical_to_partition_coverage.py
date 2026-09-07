"""Regression test for LOGICAL_TO_PARTITION coverage gap (2026-05-26 finding).

Background: ScenarioStrategy._build_schedule used to silently warn-and-skip
unknown logical_ids. The v2 RMC scenarios use `client_N_newM` (M=1..4) for
reconnection cycles, but LOGICAL_TO_PARTITION only knew `client_N_new` (no
suffix). The skip silently dropped all 36 ALIE adversary entries from the
schedule, so adversaries never participated. The cascade broke malicious_gt
tagging and made recall@FPR=10% uncomputable.

These tests assert:
  1. All v2 RMC scenarios build without raising "Unknown logical_id".
  2. `client_N_newM` maps to partition N (same physical, new identity).
  3. Loading a scenario with an unmapped ID raises ValueError immediately
     (no more silent skip).
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_base_strategy():
    """ScenarioStrategy requires a base Strategy; supply a permissive mock."""
    return MagicMock()


def _build_scenario_strategy(scenario_path, mock_base):
    from flowerfl.scenario_strategy import ScenarioStrategy
    return ScenarioStrategy(
        base_strategy=mock_base,
        plugins=[],
        scenario_path=str(scenario_path),
    )


@pytest.mark.parametrize("scenario_name", [
    "rmc_intensity_1_continuous_v2.json",
    "rmc_intensity_3_continuous_v2.json",
    "rmc_intensity_5_continuous_v2.json",
    "rmc_intensity_7_continuous_v2.json",
    "rmc_intensity_9_continuous_v2.json",
])
def test_v2_scenarios_build_without_unknown_logical_ids(
    scenario_name, mock_base_strategy
):
    """All v2 RMC scenarios should construct cleanly. If LOGICAL_TO_PARTITION
    misses any logical_id used in the schedule, this raises ValueError."""
    scenario_path = PROJECT_ROOT / "rmc" / "scenarios" / scenario_name
    if not scenario_path.exists():
        pytest.skip(f"scenario not present: {scenario_name}")
    # Should not raise.
    strategy = _build_scenario_strategy(scenario_path, mock_base_strategy)
    assert strategy._scenario is not None


def test_cycle_identity_maps_to_same_physical_partition():
    """`client_N_newM` should map back to partition N (same physical client
    returning under a new identity) for the RMC threat model."""
    from flowerfl.scenario_strategy import ScenarioStrategy
    ltp = ScenarioStrategy.LOGICAL_TO_PARTITION
    for n in range(21):
        for m in range(1, 11):
            key = f"client_{n}_new{m}"
            assert key in ltp, f"missing cycle identity: {key}"
            assert ltp[key] == n, (
                f"{key} maps to partition {ltp[key]}, expected {n} "
                f"(same physical, new identity)"
            )


def test_v2_intensity_9_schedule_contains_alie_adversaries(mock_base_strategy):
    """After _build_schedule, the v2 intensity-9 scenario must include the 9
    ALIE adversaries in attack rounds 13-20 (block 3). Pre-fix, these were
    silently dropped."""
    scenario_path = (
        PROJECT_ROOT / "rmc" / "scenarios"
        / "rmc_intensity_9_continuous_v2.json"
    )
    strategy = _build_scenario_strategy(scenario_path, mock_base_strategy)

    # Block 3 rounds are 13-20. Pick round 13.
    schedule_r13 = strategy._schedule_cache.get(13, [])
    alie_attackers = [
        e for e in schedule_r13
        if e["attack_type"] == "alie"
    ]
    assert len(alie_attackers) == 9, (
        f"expected 9 ALIE adversaries in round 13, got {len(alie_attackers)}. "
        f"Full schedule: {schedule_r13}"
    )
    # Their logical_ids must be the _new1 cycle identities.
    expected_ids = {f"client_{n}_new1" for n in range(11, 20)}
    actual_ids = {e["logical_id"] for e in alie_attackers}
    assert actual_ids == expected_ids, (
        f"adversary IDs mismatch: expected {expected_ids}, got {actual_ids}"
    )


def test_unknown_logical_id_raises_immediately(tmp_path, mock_base_strategy):
    """A scenario with an unmapped logical_id should fail fast with ValueError,
    not silently skip. This was the 2026-05-26 root cause."""
    bad_scenario = {
        "name": "bad_test_scenario",
        "dataset": "edge_full_20_rmc",
        "num_rounds": 5,
        "seed": 0,
        "clients": [f"client_{i}" for i in range(20)],
        "schedule": [
            {
                "rounds": [1, 5],
                "participants": ["client_0", "client_99_bogus"],
                "attacks": {},
                "comment": "this should fail",
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad_scenario))
    with pytest.raises(ValueError, match="Unknown logical_id"):
        _build_scenario_strategy(p, mock_base_strategy)


def test_logical_to_partition_rejects_m_eleven_or_higher():
    """M2: LOGICAL_TO_PARTITION must cover M in {1..10} per _build_schedule design.
    Higher M values must be absent so _build_schedule's raise-on-unmapped check
    fires loudly (Bug #1 prevention)."""
    from flowerfl.scenario_strategy import ScenarioStrategy

    ltp = ScenarioStrategy.LOGICAL_TO_PARTITION

    # Positive: M=1..10 present for client_0 and client_19 (boundary check)
    for m in range(1, 11):
        assert f"client_0_new{m}" in ltp, (
            f"client_0_new{m} missing from LOGICAL_TO_PARTITION"
        )
        assert f"client_19_new{m}" in ltp, (
            f"client_19_new{m} missing from LOGICAL_TO_PARTITION"
        )

    # Negative: M=11 must be absent so _build_schedule raises on unmapped id
    assert "client_0_new11" not in ltp, (
        "client_0_new11 is present in LOGICAL_TO_PARTITION but should be absent; "
        "_build_schedule must raise ValueError if a scenario ever uses M=11"
    )
    assert "client_5_new50" not in ltp, (
        "client_5_new50 is present but should be absent"
    )


def test_build_schedule_raises_on_overlap():
    """M5: overlapping schedule blocks must surface as ValueError rather than
    silently overwriting _schedule_cache (defense-in-depth for A4 invariant).
    """
    import json as _json
    import tempfile

    from flowerfl.scenario_strategy import ScenarioStrategy

    sc = {
        "name": "overlap_test",
        "dataset": "edge_full_20_rmc",
        "num_rounds": 10,
        "seed": 42,
        "clients": {f"client_{i}": {"physical_id": i} for i in range(20)},
        "schedule": [
            {
                "rounds": [1, 5],
                "participants": [f"client_{i}" for i in range(20)],
                "attacks": {},
            },
            {
                "rounds": [4, 8],  # overlaps with first block at rounds 4-5
                "participants": [f"client_{i}" for i in range(20)],
                "attacks": {
                    "client_0": {"type": "alie", "params": {"z_max": 0.9}}
                },
            },
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        _json.dump(sc, f)
        path = f.name

    with pytest.raises(ValueError, match="schedule overlap at round"):
        strategy = ScenarioStrategy.__new__(ScenarioStrategy)
        strategy._scenario = sc
        strategy._schedule_cache = {}
        strategy._partition_to_logical = {}
        strategy._build_schedule()


@pytest.mark.parametrize(
    "scenario_file",
    [
        "rmc/scenarios/S0_clean_baseline.json",
        "rmc/scenarios/S1_benign_churn_only.json",
        "rmc/scenarios/S2_adaptive_switching_only.json",
        "rmc/scenarios/S3_identity_reset_only.json",
        "rmc/scenarios/S4_full_mix.json",
    ],
)
def test_phase_a_scenario_logical_ids_all_mapped(scenario_file):
    """Every logical_id used by S0..S4 must be in LOGICAL_TO_PARTITION
    (per v1.1 Bug #1 fix). No silent adversary drop on the new scenarios."""
    scenario_path = PROJECT_ROOT / scenario_file
    if not scenario_path.exists():
        pytest.skip(f"scenario not present: {scenario_file}")

    from flowerfl.scenario_strategy import ScenarioStrategy

    sc = json.loads(scenario_path.read_text())
    seen_cids = set()
    for block in sc["schedule"]:
        for cid in block.get("participants", []):
            seen_cids.add(cid)
        for cid in (block.get("attacks") or {}):
            seen_cids.add(cid)

    for cid in seen_cids:
        assert cid in ScenarioStrategy.LOGICAL_TO_PARTITION, (
            f"{scenario_file}: '{cid}' missing from LOGICAL_TO_PARTITION "
            f"(would silently drop in _build_schedule per Bug #1)"
        )

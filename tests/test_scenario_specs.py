"""Validate that JSON scenario specs deterministically produce the committed
scenario JSON files. Closes the audit-trail loop: changes to a spec JSON
will fail this test until the corresponding scenario JSON is regenerated.

Sweep 1 — spec-driven generation (2026-05-28).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "rmc" / "scenario_specs"
SCENARIOS_DIR = REPO_ROOT / "rmc" / "scenarios"


def _normalize_for_comparison(sc: dict) -> dict:
    """Strip free-text comment fields before comparing for behavioural equivalence."""
    out = {k: v for k, v in sc.items() if k not in ("description",)}
    out["schedule"] = [
        {k: v for k, v in block.items() if k != "comment"}
        for block in sc["schedule"]
    ]
    return out


@pytest.mark.parametrize("spec_name,scenario_name", [
    ("S0_clean_baseline.spec.json", "S0_clean_baseline.json"),
    ("S1_benign_churn_only.spec.json", "S1_benign_churn_only.json"),
    ("S2_adaptive_switching_only.spec.json", "S2_adaptive_switching_only.json"),
    ("S3_identity_reset_only.spec.json", "S3_identity_reset_only.json"),
    ("S4_full_mix.spec.json", "S4_full_mix.json"),
])
def test_spec_generates_committed_scenario(spec_name, scenario_name):
    """Every spec must deterministically reproduce its committed scenario JSON.

    Comparison is behavioural: same clients dict, same num_rounds, same
    schedule length, and per-block: same rounds, same participant sets,
    same attacks dict.  Free-text comment fields and the description string
    are excluded (they may diverge without breaking the experiment).
    """
    from scripts.data.generate_scenarios import generate_scenario_from_spec

    spec_path = SPECS_DIR / spec_name
    expected_path = SCENARIOS_DIR / scenario_name

    if not spec_path.exists():
        pytest.skip(f"spec not present: {spec_name}")
    if not expected_path.exists():
        pytest.skip(f"scenario not present: {scenario_name}")

    actual = generate_scenario_from_spec(spec_path)
    expected = json.loads(expected_path.read_text())

    actual_norm = _normalize_for_comparison(actual)
    expected_norm = _normalize_for_comparison(expected)

    # Top-level fields
    assert actual_norm["clients"] == expected_norm["clients"], (
        f"{spec_name}: clients dict mismatch"
    )
    assert actual_norm["num_rounds"] == expected_norm["num_rounds"], (
        f"{spec_name}: num_rounds mismatch "
        f"({actual_norm['num_rounds']} vs {expected_norm['num_rounds']})"
    )
    assert actual_norm["seed"] == expected_norm["seed"], (
        f"{spec_name}: seed mismatch"
    )
    assert actual_norm["name"] == expected_norm["name"], (
        f"{spec_name}: name mismatch"
    )

    # Schedule length
    assert len(actual_norm["schedule"]) == len(expected_norm["schedule"]), (
        f"{spec_name}: schedule block count mismatch "
        f"(got {len(actual_norm['schedule'])}, expected {len(expected_norm['schedule'])})"
    )

    # Per-block comparison
    for i, (ab, eb) in enumerate(
        zip(actual_norm["schedule"], expected_norm["schedule"])
    ):
        assert ab["rounds"] == eb["rounds"], (
            f"{spec_name}: block {i} rounds differ: {ab['rounds']} vs {eb['rounds']}"
        )
        assert set(ab.get("participants", [])) == set(eb.get("participants", [])), (
            f"{spec_name}: block {i} participants differ\n"
            f"  actual:   {sorted(ab.get('participants', []))}\n"
            f"  expected: {sorted(eb.get('participants', []))}"
        )
        assert ab.get("attacks", {}) == eb.get("attacks", {}), (
            f"{spec_name}: block {i} attacks differ\n"
            f"  actual:   {ab.get('attacks', {})}\n"
            f"  expected: {eb.get('attacks', {})}"
        )


def test_generate_scenario_from_spec_unknown_type(tmp_path):
    """generate_scenario_from_spec must raise ValueError on unknown type."""
    from scripts.data.generate_scenarios import generate_scenario_from_spec

    bad_spec = {
        "$schema_version": 1,
        "name": "bad_type_test",
        "type": "nonexistent_type",
        "output_file": "rmc/scenarios/bad.json",
        "params": {
            "n_adversaries": 9, "n_clients": 20, "n_rounds": 50,
            "discovery_rounds": 2, "alie_z_max": 0.9, "seed": 42,
        },
    }
    p = tmp_path / "bad.spec.json"
    p.write_text(json.dumps(bad_spec))

    with pytest.raises(ValueError, match="Unknown scenario spec type"):
        generate_scenario_from_spec(p)


def test_all_spec_files_are_valid_json():
    """All *.spec.json files in rmc/scenario_specs/ must be valid JSON with required fields."""
    required_fields = {"$schema_version", "name", "type", "output_file", "params"}
    for spec_path in sorted(SPECS_DIR.glob("*.spec.json")):
        spec = json.loads(spec_path.read_text())
        missing = required_fields - spec.keys()
        assert not missing, (
            f"{spec_path.name}: missing required fields: {missing}"
        )
        assert spec["$schema_version"] == 1, (
            f"{spec_path.name}: unexpected $schema_version {spec['$schema_version']}"
        )

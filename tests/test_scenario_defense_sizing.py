"""Runner derives scenario-mode defense sizing (num-malicious / defense-cohort-size).

Launch-blocker regression (methodology v1.19): scenario-mode Multi-Krum ran
f=1/keep-18 because the runner never sent an adversary count and server_app
defaulted ``malicious-fraction`` to 0.0. The runner must derive the canonical
RMC sustained adversary count (9) and the per-round cohort (20) from the
scenario schedule and emit them as ``num-malicious`` / ``defense-cohort-size``
so the deployed defense matches the documented keep-9 / f-9 design.

The derivation reuses the same schedule-parsing helpers the A3/A4 integrity
assertions use (peak per-round declared malicious count = 9; peak per-round
declared participant count = 20), so it is robust across S0-S4 and cannot
drift from the scenario the run actually executes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_phase4_flower import _build_run_config, _scenario_defense_sizing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCEN = PROJECT_ROOT / "rmc" / "scenarios"


def _cfg(scenario: Path) -> dict:
    return _build_run_config(
        strategy="ScenarioKrum",
        scenario_path=str(scenario),
        rounds=50,
        seed=42,
        optimizer_state="reset",
        use_cs=False,
        max_per_client=2000,
    )


@pytest.mark.unit
def test_build_run_config_emits_sizing_for_s4():
    cfg = _cfg(SCEN / "S4_full_mix.json")
    assert cfg["num-malicious"] == 9
    assert cfg["defense-cohort-size"] == 20


@pytest.mark.unit
def test_build_run_config_emits_sizing_for_s0():
    # S0 declared adversary count verified from its JSON: 9 sustained ALIE
    # adversaries (client_0..client_8, z_max=0.9) over rounds 3..50.
    cfg = _cfg(SCEN / "S0_clean_baseline.json")
    assert cfg["num-malicious"] == 9
    assert cfg["defense-cohort-size"] == 20


@pytest.mark.unit
def test_sizing_helper_direct_s4():
    assert _scenario_defense_sizing(str(SCEN / "S4_full_mix.json")) == (9, 20)


@pytest.mark.unit
def test_defense_provenance_fields_krum_policy():
    """PR #12 round-2 P2 (labeling): result provenance must carry BOTH facts
    separately — the scenario-declared adversary count (ground truth) and the
    deployed Krum f policy (threat-model-constant dynamic ceil(n/2)-1) — so
    the audit can never conflate declared intensity with deployed f."""
    from scripts.run_phase4_flower import _defense_provenance_fields

    fields = _defense_provenance_fields(
        "ScenarioKrum", {"num-malicious": 3, "defense-cohort-size": 20}
    )
    assert fields["scenario_declared_adversaries"] == 3
    assert fields["defense_cohort_size"] == 20
    assert fields["krum_f_policy"] == "dynamic ceil(n/2)-1"

    # Krum+TGE chain: Krum layer present -> same policy.
    fields = _defense_provenance_fields(
        "ScenarioKrumTGE", {"num-malicious": 9, "defense-cohort-size": 20}
    )
    assert fields["krum_f_policy"] == "dynamic ceil(n/2)-1"

    # No Krum layer -> policy n/a; declared counts still recorded.
    fields = _defense_provenance_fields(
        "ScenarioTGEnsemble", {"num-malicious": 9, "defense-cohort-size": 20}
    )
    assert fields["krum_f_policy"] == "n/a"
    assert fields["scenario_declared_adversaries"] == 9

    # Keys absent (legacy/non-scenario) -> None, policy still truthful.
    fields = _defense_provenance_fields("ScenarioKrum", {})
    assert fields["scenario_declared_adversaries"] is None
    assert fields["defense_cohort_size"] is None
    assert fields["krum_f_policy"] == "dynamic ceil(n/2)-1"


@pytest.mark.unit
def test_sizing_absent_when_scenario_unreadable():
    # Missing/unreadable scenario -> helper returns None -> runner emits no
    # sizing keys, so the legacy malicious-fraction path stays intact.
    assert _scenario_defense_sizing(str(SCEN / "does_not_exist.json")) is None
    cfg = _build_run_config(
        strategy="ScenarioKrum",
        scenario_path=str(SCEN / "does_not_exist.json"),
        rounds=50,
        seed=42,
        optimizer_state="reset",
        use_cs=False,
        max_per_client=2000,
    )
    assert "num-malicious" not in cfg
    assert "defense-cohort-size" not in cfg

"""Tests for the Krum+TGE config routing in run_phase4_flower.py.

Verifies that:
  - The runner advertises 'Krum+TGE' as a supported config.
  - The strategy factory routes 'Krum+TGE' to a ScenarioKrumTGE token with
    the provisional canonical tenure-gate ramp (tge_ramp_rounds == 8,
    amendment v1.6 § 2; final value selected at the dev gate per v1.6 § 3)
    and ramp < NUM_ROUNDS (regression guard: ensures the LSTM temporal
    expert actually blends in).

This is the deployed primary defense for the H2 thesis test: Krum geometric
filter chained with the TGEnsemble GBDT+LSTM plugin (spec § 6.1 family C,
methodology v1.9 amendment).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_krumtge_in_supported_configs():
    from run_phase4_flower import SUPPORTED_CONFIGS
    assert "Krum+TGE" in SUPPORTED_CONFIGS


def test_krumtge_routes_to_scenario_token_with_real_ramp():
    from run_phase4_flower import build_strategy_for_config, NUM_ROUNDS
    token, cfg = build_strategy_for_config("Krum+TGE")
    assert "KrumTGE" in token.__name__
    assert cfg.get("tge_ramp_rounds") == 3  # gate-selected r*=3 (v1.6 § 3, 2026-07-23; 61ef015)
    assert cfg["tge_ramp_rounds"] < NUM_ROUNDS

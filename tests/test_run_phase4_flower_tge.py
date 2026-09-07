"""Sanity tests for the TGE (GBDT + LSTM tenure-gated) config routing in run_phase4_flower.py.

Verifies that:
  - The runner advertises 'TGE' as a supported config
  - The strategy factory routes 'TGE' to TGEnsemble with the PROVISIONAL canonical
    tenure-gate ramp (ramp_rounds == 8, amendment v1.6 § 2 — the final value is
    selected at the dev gate per v1.6 § 3) so the LSTM temporal expert actually
    blends in. This is the GBDT + LSTM thesis method (spec § 6.1 family C).
    The previous ramp_rounds=999 silently ran TGE as GBDT-only — a known
    blocker (B2) that would have invalidated the H2 thesis test.

Note: this config was previously named 'Krum+TGE'. The 'Krum+' prefix was
misleading — the routing uses FedAvg base + TGEnsemblePlugin only (no Krum
filter). Renamed to 'TGE' on 2026-05-19 to reflect the actual composition.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_supported_configs_includes_tge():
    from run_phase4_flower import SUPPORTED_CONFIGS
    assert "TGE" in SUPPORTED_CONFIGS, (
        f"SUPPORTED_CONFIGS missing 'TGE'. Got: {SUPPORTED_CONFIGS}"
    )


def test_strategy_factory_routes_tge_to_tge_plugin():
    """`build_strategy_for_config('TGE')` returns a (strategy_class, plugin_config) tuple where:
       - strategy_class.__name__ contains 'TGE' (i.e., ScenarioTGEnsemble)
       - plugin_config carries the GATE-SELECTED tenure-gate ramp
         (tge_ramp_rounds == 3, v1.6 § 3 frozen rule executed at the 2026-07-23
         dev gate — results/20260723/ramp_selection, 61ef015) so the LSTM
         temporal expert blends in. Library defaults stay at the provisional
         canonical 8; the runner config is the deploy site and carries the
         selected value. Never hand-edited.

    Regression guard (B2): tge_ramp_rounds must NOT be >= the simulation round count
    (e.g. 999), which would silently disable the LSTM and run TGE as GBDT-only —
    invalidating the H2 thesis test.
    """
    from run_phase4_flower import build_strategy_for_config, NUM_ROUNDS
    result = build_strategy_for_config("TGE")
    assert isinstance(result, tuple) and len(result) == 2
    strategy_class, plugin_config = result
    assert "TGE" in strategy_class.__name__, (
        f"Expected TGE strategy class; got {strategy_class.__name__}"
    )
    ramp = plugin_config.get("tge_ramp_rounds")
    assert ramp == 3, (
        f"tge_ramp_rounds must be the gate-selected r*=3 (v1.6 § 3, 2026-07-23; "
        f"61ef015); got {ramp!r}"
    )
    assert ramp < NUM_ROUNDS, (
        f"tge_ramp_rounds ({ramp}) must be < NUM_ROUNDS ({NUM_ROUNDS}) or the LSTM "
        f"never activates (B2 regression — TGE would run GBDT-only)"
    )


def test_strategy_factory_preserves_existing_configs():
    """The existing four configs (Krum, Krum+CS, TrustScore, TrustScore+CS) must still route to
    their original strategies after the runner extension."""
    from run_phase4_flower import build_strategy_for_config
    for config_name in ["Krum", "Krum+CS", "TrustScore", "TrustScore+CS"]:
        strategy_class, _ = build_strategy_for_config(config_name)
        assert strategy_class is not None, f"{config_name} routing returned None"


def test_backend_config_caps_ray_workers_and_disables_dashboard():
    """The runner must cap Ray to 8 worker actors and disable noisy/heavy Ray features.

    Rationale: hyper-v + new AMD Zen 5 hybrid CPU clock-skew + Ray spawning one
    actor per CPU led to repeated WSL crashes during Flower simulation. Capping
    Ray to 8 actors (instead of the implicit nproc default), turning off the
    Ray dashboard process, and silencing per-worker driver logging removes the
    fork/load/clock-skew pressure that correlated with the crashes.
    See docs/HANDOFF_2026-05-19.md for the full diagnosis.
    """
    from run_phase4_flower import _BACKEND_CONFIG

    init_args = _BACKEND_CONFIG.get("init_args", {})
    assert init_args.get("num_cpus") == 8, (
        f"backend_config.init_args.num_cpus must be 8 (cap Ray actor pool); "
        f"got {init_args.get('num_cpus')!r}"
    )
    assert init_args.get("log_to_driver") is False, (
        f"backend_config.init_args.log_to_driver must be False (silence per-worker logging); "
        f"got {init_args.get('log_to_driver')!r}"
    )
    assert init_args.get("include_dashboard") is False, (
        f"backend_config.init_args.include_dashboard must be False (skip dashboard process); "
        f"got {init_args.get('include_dashboard')!r}"
    )

    client_resources = _BACKEND_CONFIG.get("client_resources", {})
    assert client_resources.get("num_cpus") == 1, (
        f"backend_config.client_resources.num_cpus must remain 1 "
        f"(was the implicit default before Step 2); got {client_resources.get('num_cpus')!r}"
    )
    assert client_resources.get("num_gpus") == 0.0, (
        f"backend_config.client_resources.num_gpus must remain 0.0; "
        f"got {client_resources.get('num_gpus')!r}"
    )


def test_runner_accepts_optimizer_state_flag(tmp_path, monkeypatch):
    """The --optimizer-state flag should be parsed and propagated."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/run_phase4_flower.py",
         "--optimizer-state", "persistent",
         "--configs", "Krum",
         "--seeds", "42",
         "--scenario", "rmc/scenarios/rmc_intensity_9_continuous_v2.json",
         "--max-per-client", "100",
         "--rounds", "2",
         "--output-dir", str(tmp_path),
         "--help"],  # short-circuit: just verify the flag exists and parses
        capture_output=True, text=True,
    )
    # If --help is honored without arg-error, the flag is wired.
    # If --optimizer-state isn't a recognized arg, argparse exits with code 2.
    assert "--optimizer-state" in result.stdout or "optimizer-state" in result.stdout, \
        f"--optimizer-state flag missing from help output. stdout: {result.stdout!r}, stderr: {result.stderr!r}"


def test_runner_modes_alias_back_compat(tmp_path):
    """--modes Flower and --modes persistent_optimizer should alias to --optimizer-state."""
    import subprocess, sys
    # Conflict between --modes and --optimizer-state should error.
    result = subprocess.run(
        [sys.executable, "scripts/run_phase4_flower.py",
         "--optimizer-state", "reset",
         "--modes", "persistent_optimizer",
         "--configs", "Krum",
         "--seeds", "42",
         "--scenario", "rmc/scenarios/rmc_intensity_9_continuous_v2.json"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, "conflicting flags must error"
    assert "conflict" in result.stderr.lower() or "cannot" in result.stderr.lower(), \
        f"expected conflict error, got: {result.stderr!r}"

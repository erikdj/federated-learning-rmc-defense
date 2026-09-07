"""Verify runner picks the locked lr for each optimizer mode (spec § 4.5, § 8.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_phase4_flower import _build_run_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
HPARAMS_FILE = _REPO_ROOT / "data" / "hparams_locked.json"


def _locked_lr(mode_key: str) -> float:
    return float(json.loads(HPARAMS_FILE.read_text())[mode_key]["lr"])


@pytest.mark.unit
def test_flower_reset_lr_matches_locked():
    cfg = _build_run_config(
        strategy="krum",
        scenario_path="rmc/scenarios/S0_clean_baseline.json",
        rounds=50,
        seed=42,
        optimizer_state="reset",
        use_cs=False,
        max_per_client=2_000_000,
    )
    assert cfg["learning-rate"] == _locked_lr("flower_reset")


@pytest.mark.unit
def test_persistent_optimizer_lr_matches_locked():
    cfg = _build_run_config(
        strategy="krum",
        scenario_path="rmc/scenarios/S0_clean_baseline.json",
        rounds=50,
        seed=42,
        optimizer_state="persistent",
        use_cs=False,
        max_per_client=2_000_000,
    )
    assert cfg["learning-rate"] == _locked_lr("persistent_optimizer")


@pytest.mark.unit
def test_locked_file_contains_spec_values():
    """Confirm hparams_locked.json holds spec § 4.5 mandated values.

    Update these literals only when a spec amendment changes the locked hparams
    (per spec § 12 amendment protocol — new dated spec file, not in-place edit).
    """
    assert _locked_lr("flower_reset") == pytest.approx(0.005)
    assert _locked_lr("persistent_optimizer") == pytest.approx(0.001)


@pytest.mark.unit
def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="optimizer_state"):
        _build_run_config(
            strategy="krum",
            scenario_path="rmc/scenarios/S0_clean_baseline.json",
            rounds=50,
            seed=42,
            optimizer_state="nonsense",
            use_cs=False,
            max_per_client=2_000_000,
        )


@pytest.mark.unit
def test_h1_runner_adds_one_for_discovery_offset():
    """H1 regression: runner must pass num-server-rounds = scenario_rounds + 1
    so the strategy's scenario_round = server_round - 1 covers the full
    declared range."""
    cfg = _build_run_config(
        strategy="krum",
        scenario_path="rmc/scenarios/S0_clean_baseline.json",
        rounds=50,
        seed=42,
        optimizer_state="persistent",
        use_cs=False,
        max_per_client=2_000_000,
    )
    assert cfg["num-server-rounds"] == 51, (
        f"runner must add 1 to declared rounds for the strategy offset; "
        f"got {cfg['num-server-rounds']}"
    )

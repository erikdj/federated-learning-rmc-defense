"""Regression tests for the unified Phase 4 runner.

Anchor 2: self-determinism — two runs with identical (seed, scenario, config,
optimizer_state) produce byte-identical trajectory arrays.

Anchor 3 was deleted on 2026-05-19 after root-cause analysis (see
results/anchors/README.md and the 2026-05-19 anchor-3 investigation notes).
The frozen JSON it referenced was produced by the old `ScenarioKrumCS` code
path but had been retroactively relabeled for the `ScenarioTGEnsemble` path —
which is a materially different defense composition. The regression-vs-pilot
gate now relies on the Szelag reproduction anchor (anchor 1).

These tests run a 2-round simulation with --max-per-client 200 (a deliberately
tiny scale; the goal is determinism + schema parity, not real F1 scores).
Total wall-clock per test: ~30 seconds.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO = PROJECT_ROOT / "rmc/scenarios/rmc_intensity_9_continuous_v2.json"


def _run(out_dir: Path, seed: int = 42, opt_state: str = "reset",
         config: str = "Krum") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["conda", "run", "-n", "flowerfl",
         "python", "scripts/run_phase4_flower.py",
         "--configs", config,
         "--seeds", str(seed),
         "--scenario", str(SCENARIO),
         "--max-per-client", "200",
         "--rounds", "2",
         "--optimizer-state", opt_state,
         "--output-dir", str(out_dir),
         "--reporting-split", "val"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"Runner failed: stdout={result.stdout!r}, stderr={result.stderr!r}")
    # Find the produced result file
    matches = sorted(out_dir.glob(f"*{config.lower().replace('+','_')}*seed{seed}*.json"))
    if not matches:
        pytest.fail(f"No result JSON found in {out_dir}; runner stdout: {result.stdout!r}")
    return json.loads(matches[0].read_text())


@pytest.mark.slow
def test_anchor_2_self_determinism(tmp_path):
    """Anchor 2: two runs with identical inputs produce identical trajectory."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    res_a = _run(out_a, seed=42, opt_state="reset", config="Krum")
    res_b = _run(out_b, seed=42, opt_state="reset", config="Krum")
    # Compare trajectory list element-wise; allow byte-identical match
    assert res_a["trajectory"] == res_b["trajectory"], \
        f"Self-determinism violated: trajectory_a={res_a['trajectory']!r}, " \
        f"trajectory_b={res_b['trajectory']!r}"


@pytest.mark.slow
def test_provenance_metadata_complete(tmp_path):
    """Every result JSON must include a `provenance` block with required keys."""
    res = _run(tmp_path, seed=42, opt_state="persistent", config="Krum")
    assert "provenance" in res, "result JSON missing provenance block"
    p = res["provenance"]
    required = {"runner_version", "runner_commit", "code_path",
                "scenario_path", "optimizer_state", "cs_model_path",
                "flwr_version", "tge_lstm_state"}
    missing = required - set(p.keys())
    assert not missing, f"provenance missing keys: {sorted(missing)}"
    assert p["optimizer_state"] == "persistent"
    assert p["code_path"] == "unified-flower-runner"
    assert p["tge_lstm_state"] == "disabled"  # Step 2 stage


@pytest.mark.slow
def test_runner_emits_convergence_block(tmp_path):
    res = _run(tmp_path, seed=42, opt_state="reset", config="Krum")
    assert "convergence" in res
    c = res["convergence"]
    assert "rounds_to_reach_90pct_final_F1" in c
    assert "stability_last10" in c
    assert "monotonicity_score" in c


@pytest.mark.slow
def test_runner_emits_confounder_control_only_for_v3(tmp_path):
    """v2 scenario → no confounder_control block (or null). v3 → populated."""
    res_v2 = _run(tmp_path, seed=42, opt_state="reset", config="Krum")
    # v2 has no honest_events → confounder_control may be absent or None
    cc = res_v2.get("confounder_control")
    assert cc is None or cc.get("n_honest_reconnect_events", 0) == 0


@pytest.mark.slow
def test_runner_emits_defense_overhead(tmp_path):
    res = _run(tmp_path, seed=42, opt_state="reset", config="Krum")
    assert "defense_overhead" in res
    do = res["defense_overhead"]
    assert "krum_aggregation_time_per_round_ms" in do
    s = do["krum_aggregation_time_per_round_ms"]
    assert "mean" in s and "p50" in s and "p95" in s
    assert s["mean"] >= 0.0


# Anchor 3 tests removed 2026-05-19. See module docstring + results/anchors/README.md.

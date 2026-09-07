"""Erratum-B fleet reachability: run_extras allowlist + entrypoint argv.

The EXP-062 calibration fleet declares `h2p_observe_only: true` (and later
fleets pin `h2p_cuts_version`) in the matrix design doc's run_extras; the
keys must parse loudly at pre-registration (closed values, eval_split
discipline) and reach the runner via the entrypoint argv passthrough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from praxis_exp.matrix_doc import MatrixDocError, parse_matrix  # noqa: E402


def _doc(tmp_path: Path, run_extras_yaml: str = "") -> Path:
    path = tmp_path / "design.md"
    path.write_text(f"""---
exp_id: EXP-062
slug: h4-observe-calibration
hypothesis: H4
methodology_version: v1.53
matrix:
  defenses: [Krum, TrustScore, FedAvg]
  scenarios: [C0_clean_no_attack, S0_clean_baseline]
  seeds: [42, 137]
  mode: Flower
  max_per_client: 5000
  rounds: 50
batch:
  job_queue: praxis-queue
  job_definition: praxis-jobdef
{run_extras_yaml}---
body text
""")
    return path


# ---------------------------------------------------------------------------
# matrix_doc allowlist + closed-value validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h2p_observe_only_parses_and_normalizes_to_bool(tmp_path):
    doc = parse_matrix(_doc(tmp_path, "run_extras:\n  h2p_observe_only: true\n"))
    assert doc.run_extras == {"h2p_observe_only": True}


@pytest.mark.unit
def test_h2p_observe_only_typo_refuses_at_parse(tmp_path):
    with pytest.raises(MatrixDocError, match="h2p_observe_only"):
        parse_matrix(_doc(tmp_path, "run_extras:\n  h2p_observe_only: ture\n"))


@pytest.mark.unit
def test_h2p_cuts_version_parses_and_canonicalizes(tmp_path):
    doc = parse_matrix(_doc(tmp_path, "run_extras:\n  h2p_cuts_version: V2\n"))
    assert doc.run_extras == {"h2p_cuts_version": "v2"}


@pytest.mark.unit
def test_h2p_cuts_version_unknown_refuses_at_parse(tmp_path):
    with pytest.raises(MatrixDocError, match="h2p_cuts_version"):
        parse_matrix(_doc(tmp_path, "run_extras:\n  h2p_cuts_version: v9\n"))


@pytest.mark.unit
def test_unknown_run_extras_key_still_refuses(tmp_path):
    with pytest.raises(MatrixDocError, match="unknown run_extras"):
        parse_matrix(_doc(tmp_path, "run_extras:\n  h2p_observer: on\n"))


# ---------------------------------------------------------------------------
# entrypoint argv passthrough
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_entrypoint_h2p_argv_absent_is_empty():
    from docker.entrypoint import h2p_argv_from_run_extras

    assert h2p_argv_from_run_extras(None) == []
    assert h2p_argv_from_run_extras({}) == []
    assert h2p_argv_from_run_extras({"eval_split": "sealed_test"}) == []


@pytest.mark.unit
def test_entrypoint_h2p_observe_argv():
    from docker.entrypoint import h2p_argv_from_run_extras

    assert h2p_argv_from_run_extras(
        {"h2p_observe_only": True}) == ["--h2p-observe-only"]
    assert h2p_argv_from_run_extras({"h2p_observe_only": False}) == []
    assert h2p_argv_from_run_extras(
        {"h2p_observe_only": "true"}) == ["--h2p-observe-only"]


@pytest.mark.unit
def test_entrypoint_h2p_observe_garbage_refuses():
    from docker.entrypoint import h2p_argv_from_run_extras

    with pytest.raises(ValueError, match="h2p_observe_only"):
        h2p_argv_from_run_extras({"h2p_observe_only": "ture"})


@pytest.mark.unit
def test_entrypoint_h2p_cuts_version_argv():
    from docker.entrypoint import h2p_argv_from_run_extras

    assert h2p_argv_from_run_extras(
        {"h2p_cuts_version": "v2"}) == ["--h2p-cuts-version", "v2"]
    assert h2p_argv_from_run_extras(
        {"h2p_observe_only": True, "h2p_cuts_version": "v1"}
    ) == ["--h2p-observe-only", "--h2p-cuts-version", "v1"]


@pytest.mark.unit
def test_entrypoint_h2p_cuts_version_garbage_refuses():
    from docker.entrypoint import h2p_argv_from_run_extras

    with pytest.raises(ValueError, match="h2p_cuts_version"):
        h2p_argv_from_run_extras({"h2p_cuts_version": "v9"})


@pytest.mark.unit
def test_runner_argv_carries_h2p_flags():
    from docker.entrypoint import runner_argv
    from praxis_exp.units import Unit

    unit = Unit(config="Krum", scenario="C0_clean_no_attack", mode="Flower",
                seed=42, max_per_client=5000, rounds=50, array_index=0)
    argv = runner_argv(
        unit, scenario_dir="rmc/scenarios", out_dir=Path("/tmp/out"),
        run_extras={"h2p_observe_only": True, "h2p_cuts_version": "v1"},
    )
    assert "--h2p-observe-only" in argv
    i = argv.index("--h2p-cuts-version")
    assert argv[i + 1] == "v1"


@pytest.mark.unit
def test_runner_argv_byte_identical_without_h2p_extras():
    from docker.entrypoint import runner_argv
    from praxis_exp.units import Unit

    unit = Unit(config="Krum", scenario="S3_identity_reset_only",
                mode="Flower", seed=42, max_per_client=5000, rounds=50,
                array_index=0)
    base = runner_argv(unit, scenario_dir="rmc/scenarios",
                       out_dir=Path("/tmp/out"), run_extras=None)
    empty = runner_argv(unit, scenario_dir="rmc/scenarios",
                        out_dir=Path("/tmp/out"), run_extras={})
    assert base == empty
    assert "--h2p-observe-only" not in base

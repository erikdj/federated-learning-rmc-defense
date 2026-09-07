import pytest

@pytest.fixture(autouse=True)
def isolated_cloud_services(monkeypatch):
    """CLI routing tests use explicit public configuration and fake services."""
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setenv("PRAXIS_CONTAINER_MLFLOW_URI", "http://tracking.example:5000")
    with patch("boto3.Session"), patch("praxis_exp.cli.PraxisMlflowClient"), \
         patch("praxis_exp.cli.tabulate_experiment", return_value={
             "units_tabulated": 0, "units_skipped": 0, "units_failed": 0}):
        yield

from unittest.mock import patch
from click.testing import CliRunner
from praxis_exp.cli import main


def test_launch_matrix_command_invokes_orchestrator():
    runner = CliRunner()
    with patch("praxis_exp.cli.launch_matrix") as ml:
        ml.return_value = {"exp_id": "EXP-005", "n_units": 100, "array_job_id": "job-1",
                           "experiment_name": "EXP-005__h2-dev-sweep",
                           "manifest_key": "sweeps/EXP-005/manifest.json",
                           "git_sha": "abc", "image_digest": "sha256:x"}
        result = runner.invoke(main, ["exp", "launch-matrix", "EXP-005",
                                      "--image-digest", "sha256:x"])
    assert result.exit_code == 0, result.output
    ml.assert_called_once()
    assert ml.call_args.kwargs["image_digest"] == "sha256:x"
    assert {"_store", "_batch", "_mlflow"} <= set(ml.call_args.kwargs)
    assert "EXP-005" in result.output and "100" in result.output


def test_launch_matrix_forwards_branch_and_no_push():
    runner = CliRunner()
    with patch("praxis_exp.cli.launch_matrix") as ml:
        ml.return_value = {
            "exp_id": "EXP-005", "n_units": 2, "array_job_id": "job-1",
            "manifest_key": "sweeps/EXP-005/manifest.json",
        }
        result = runner.invoke(
            main,
            ["exp", "launch-matrix", "EXP-005", "--image-digest", "sha256:x",
             "--branch", "release", "--no-push"],
        )

    assert result.exit_code == 0, result.output
    assert ml.call_args.kwargs["branch"] == "release"
    assert ml.call_args.kwargs["_no_push"] is True


def test_launch_matrix_refill_routes_to_refill_matrix():
    """: --refill routes to refill_matrix with the parsed --cells list."""
    runner = CliRunner()
    with patch("praxis_exp.cli.refill_matrix") as rm, patch("praxis_exp.cli.launch_matrix") as lm:
        rm.return_value = {"exp_id": "EXP-005", "serial": "r2", "n_refilled": 2,
                           "refilled_cells": ["s0__krum__persistent_optimizer__seed42",
                                              "s4__trustscore__persistent_optimizer__seed42"],
                           "array_job_id": "job-9", "array_size": 8,
                           "parent_run_id": "parent-orig"}
        result = runner.invoke(main, ["exp", "launch-matrix", "EXP-005",
                                      "--image-digest", "sha256:x",
                                      "--refill", "--cells", "0,6",
                                      "--branch", "maintenance", "--no-push"])
    assert result.exit_code == 0, result.output
    rm.assert_called_once()
    lm.assert_not_called()
    assert rm.call_args.args[2] == ["0", "6"]  # parsed cell list
    assert rm.call_args.kwargs["image_digest"] == "sha256:x"
    assert rm.call_args.kwargs["branch"] == "maintenance"
    assert rm.call_args.kwargs["_no_push"] is True
    assert "Refilled EXP-005" in result.output and "r2" in result.output


def test_launch_matrix_cells_without_refill_errors():
    runner = CliRunner()
    result = runner.invoke(main, ["exp", "launch-matrix", "EXP-005",
                                  "--image-digest", "sha256:x", "--cells", "0,6"])
    assert result.exit_code != 0
    assert "--cells requires --refill" in result.output


def test_launch_matrix_surfaces_error_and_nonzero_exit():
    runner = CliRunner()
    with patch("praxis_exp.cli.launch_matrix") as ml:
        ml.side_effect = RuntimeError("dirty tree")
        result = runner.invoke(main, ["exp", "launch-matrix", "EXP-005", "--image-digest", "sha256:x"])
    assert result.exit_code != 0
    assert "dirty tree" in result.output


def test_enrich_command_invokes_backfill():
    runner = CliRunner()
    with patch("praxis_exp.cli.enrich_experiment") as en:
        en.return_value = {"experiment_id": "e", "units_enriched": 3,
                           "units_reconciled": 1, "units_skipped": 0}
        result = runner.invoke(main, ["exp", "enrich", "EXP-005"])
    assert result.exit_code == 0, result.output
    en.assert_called_once()
    assert en.call_args.args[1] == "EXP-005"
    assert {"bucket", "_store", "_client"} <= set(en.call_args.kwargs)
    assert "3 enriched" in result.output and "1 reconciled" in result.output


def test_enrich_command_surfaces_error_and_nonzero_exit():
    runner = CliRunner()
    with patch("praxis_exp.cli.enrich_experiment") as en:
        en.side_effect = RuntimeError("no manifest")
        result = runner.invoke(main, ["exp", "enrich", "EXP-005"])
    assert result.exit_code != 0
    assert "no manifest" in result.output

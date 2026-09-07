"""Launch command tests with mocked external systems."""
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def repo_with_design(tmp_path):
    # Build a clean git repo with a valid design doc
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-001-my.md").write_text(
        """---
exp_id: EXP-001
slug: my
hypothesis: test hypothesis
methodology_version: v1.2
params:
  defense: Krum
  scenario: foo.json
  seed: 42
  mode: flower_reset
  max_per_client: 100
predictions:
  final_accuracy_min: 0.4
---

body
"""
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_launch_validates_design_doc(repo_with_design, monkeypatch):
    """A missing design doc fails fast."""
    from praxis_exp.launch import LaunchError, launch_experiment
    with pytest.raises(LaunchError, match="design doc"):
        launch_experiment(repo_with_design, "EXP-999")


def test_launch_refuses_dirty_tree(repo_with_design, monkeypatch):
    """Uncommitted changes block launch."""
    from praxis_exp.launch import LaunchError, launch_experiment
    (repo_with_design / "dirty.py").write_text("x")
    with pytest.raises(LaunchError, match="working tree"):
        launch_experiment(repo_with_design, "EXP-001", _no_push=True, _skip_preflight=True)


def _seed_scenario(repo_with_design):
    """Create the foo.json scenario referenced by the design-doc fixture."""
    (repo_with_design / "foo.json").write_text("{}")
    subprocess.run(["git", "add", "foo.json"], cwd=repo_with_design, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scenario"], cwd=repo_with_design, check=True)


def test_launch_preflight_blocks_when_data_missing(repo_with_design):
    """v1.3 — launch refuses if data/edge_full_20/client_*.parquet is missing."""
    from praxis_exp.launch import LaunchError, launch_experiment
    _seed_scenario(repo_with_design)
    with pytest.raises(LaunchError, match="data dir missing|client parquet"):
        launch_experiment(repo_with_design, "EXP-001", _no_push=True)


def test_launch_preflight_passes_with_data(repo_with_design):
    """v1.3 preflight accepts a populated data dir."""
    from praxis_exp.launch import launch_experiment
    from unittest.mock import MagicMock
    _seed_scenario(repo_with_design)
    data_dir = repo_with_design / "data" / "edge_full_20"
    data_dir.mkdir(parents=True)
    for i in range(21):
        (data_dir / f"client_{i}.parquet").write_bytes(b"")
    subprocess.run(["git", "add", "."], cwd=repo_with_design, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "data"], cwd=repo_with_design, check=True)

    mock_client = MagicMock()
    mock_client.get_or_create_experiment.return_value = "exp-id-1"
    mock_client.create_run.return_value = "run-id-1"
    out = launch_experiment(
        repo_with_design, "EXP-001",
        _client=mock_client, _runner=lambda r, rid, p: 0, _no_push=True,
    )
    assert out["run_id"] == "run-id-1"


def test_launch_preflight_blocks_when_scenario_missing(repo_with_design):
    """v1.3 — launch refuses if the scenario file referenced by the design doc is missing."""
    from praxis_exp.launch import LaunchError, launch_experiment
    with pytest.raises(LaunchError, match="scenario file missing"):
        launch_experiment(repo_with_design, "EXP-001", _no_push=True)


def test_launch_creates_tag_and_mlflow_run(repo_with_design, monkeypatch):
    """Happy path: tag created, MLflow run started, runner invoked."""
    from praxis_exp.launch import launch_experiment
    mock_client = MagicMock()
    mock_client.get_or_create_experiment.return_value = "exp-id-1"
    mock_client.create_run.return_value = "run-id-1"

    def _fake_run_runner(repo, run_id, params):
        return 0

    out = launch_experiment(
        repo_with_design,
        "EXP-001",
        _client=mock_client,
        _runner=_fake_run_runner,
        _no_push=True,
        _skip_preflight=True,
    )
    assert out["run_id"] == "run-id-1"
    # Tag was created locally
    tags = subprocess.check_output(["git", "tag", "-l"], cwd=repo_with_design, text=True)
    assert "exp/EXP-001" in tags

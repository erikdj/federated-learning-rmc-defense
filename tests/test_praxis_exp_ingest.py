"""Ingest command tests."""
from pathlib import Path
from unittest.mock import MagicMock
import json
import pytest


@pytest.fixture
def repo_with_result(tmp_path):
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-001-my.md").write_text(
        """---
exp_id: EXP-001
slug: my
hypothesis: t
methodology_version: v1.2
params:
  defense: Krum
predictions:
  final_accuracy_min: 0.5
---

body
"""
    )
    res_dir = tmp_path / "results" / "EXP-001"
    res_dir.mkdir(parents=True)
    (res_dir / "phase4_flower__krum__seed42.json").write_text(
        json.dumps({
            "final_f1": 0.79,
            "final_accuracy": 0.79,
            "mean_accuracy": 0.90,
            "trajectory": [],
        })
    )
    return tmp_path


def test_ingest_logs_final_metrics(repo_with_result):
    from praxis_exp.ingest import ingest_experiment
    client = MagicMock()
    # Fake a tag-stored run id
    client.find_run_id_for_exp.return_value = "run-id-1"
    summary = ingest_experiment(
        repo_with_result, "EXP-001",
        result_path=repo_with_result / "results" / "EXP-001" / "phase4_flower__krum__seed42.json",
        _client=client,
    )
    client.log_metric.assert_any_call("run-id-1", "final_accuracy", 0.79)
    client.log_metric.assert_any_call("run-id-1", "mean_accuracy", 0.90)
    client.log_metric.assert_any_call("run-id-1", "final_f1", 0.79)
    assert summary["criteria_ok"] is True  # 0.79 >= 0.5


def test_ingest_sets_s3_link_tags_when_client_supports_artifact_uri(repo_with_result):
    """req G: ingest gains the same S3-link tags (s3_result_uri +
    s3_console_url) for the run it ingests, derived from the run's real
    MLflow artifact_uri (the SAME s3:// location log_artifact just uploaded
    the result JSON to) — idempotent (plain set_tag calls, safe to re-run)."""
    from praxis_exp.ingest import ingest_experiment
    client = MagicMock()
    client.find_run_id_for_exp.return_value = "run-id-1"
    client.artifact_uri_for_run.return_value = \
        "s3://praxis-bucket/mlflow/artifacts/my/run-id-1/artifacts"
    ingest_experiment(
        repo_with_result, "EXP-001",
        result_path=repo_with_result / "results" / "EXP-001" / "phase4_flower__krum__seed42.json",
        _client=client,
    )
    client.artifact_uri_for_run.assert_called_once_with("run-id-1")
    tag_calls = {c.args[1]: c.args[2] for c in client.set_tag.call_args_list}
    assert tag_calls["s3_result_uri"] == \
        "s3://praxis-bucket/mlflow/artifacts/my/run-id-1/artifacts/phase4_flower__krum__seed42.json"
    assert tag_calls["s3_console_url"] == (
        "https://us-east-1.console.aws.amazon.com/s3/buckets/praxis-bucket"
        "?prefix=mlflow/artifacts/my/run-id-1/artifacts/"
    )


def test_ingest_degrades_gracefully_when_client_lacks_artifact_uri_support(repo_with_result):
    """Graceful degradation: a client without artifact_uri_for_run (e.g. the
    _AttrClient minimal interface) must not fail ingestion — S3 tags are
    simply skipped."""
    from praxis_exp.ingest import ingest_experiment

    class _MinimalClient:
        def find_run_id_for_exp(self, exp_id):
            return "run-id-1"

        def log_metric(self, run_id, key, value, step=0):
            pass

        def set_tag(self, run_id, key, value):
            pass

        def log_artifact(self, run_id, path):
            pass

    summary = ingest_experiment(
        repo_with_result, "EXP-001",
        result_path=repo_with_result / "results" / "EXP-001" / "phase4_flower__krum__seed42.json",
        _client=_MinimalClient(),
    )
    assert summary["criteria_ok"] is True


def test_ingest_degrades_gracefully_when_artifact_uri_lookup_raises(repo_with_result, capsys):
    """A server error resolving artifact_uri must not fail ingestion —
    printed as a warning, ingestion still completes."""
    from praxis_exp.ingest import ingest_experiment
    client = MagicMock()
    client.find_run_id_for_exp.return_value = "run-id-1"
    client.artifact_uri_for_run.side_effect = RuntimeError("server unreachable")
    summary = ingest_experiment(
        repo_with_result, "EXP-001",
        result_path=repo_with_result / "results" / "EXP-001" / "phase4_flower__krum__seed42.json",
        _client=client,
    )
    assert summary["criteria_ok"] is True
    assert "WARN" in capsys.readouterr().out


def test_ingest_marks_criteria_failed(repo_with_result):
    from praxis_exp.ingest import ingest_experiment
    # tweak result to be below threshold
    res = repo_with_result / "results" / "EXP-001" / "phase4_flower__krum__seed42.json"
    res.write_text(json.dumps({"final_f1": 0.3, "final_accuracy": 0.3, "mean_accuracy": 0.3, "trajectory": []}))
    client = MagicMock()
    client.find_run_id_for_exp.return_value = "run-id-1"
    summary = ingest_experiment(
        repo_with_result, "EXP-001",
        result_path=res, _client=client,
    )
    assert summary["criteria_ok"] is False

"""Unit tests for the MLflow client wrapper (mocked)."""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import pytest


@pytest.fixture(autouse=True)
def _fake_site_config(monkeypatch):
    """Keep wrapper tests independent of operator AWS/site configuration."""
    from praxis_exp import mlflow_client

    monkeypatch.setattr(
        mlflow_client,
        "Config",
        lambda: SimpleNamespace(
            tracking_uri="http://unused.test",
            artifact_bucket="test-bucket",
            aws_profile="test-profile",
        ),
    )


def test_get_or_create_experiment_creates_new(tmp_path, monkeypatch):
    """If an MLflow experiment doesn't exist, create it."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_mlflow_client = MagicMock()
    fake_mlflow_client.get_experiment_by_name.return_value = None
    fake_mlflow_client.create_experiment.return_value = "exp-id-123"
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_mlflow_client)
    eid = pc.get_or_create_experiment("EXP-001__test")
    fake_mlflow_client.create_experiment.assert_called_once_with(
        name="EXP-001__test", artifact_location="s3://test-bucket/mlflow/artifacts/EXP-001__test/"
    )
    assert eid == "exp-id-123"


def test_get_or_create_experiment_returns_existing(tmp_path):
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    fake_exp = MagicMock()
    fake_exp.experiment_id = "existing-456"
    fake_client.get_experiment_by_name.return_value = fake_exp
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    assert pc.get_or_create_experiment("EXP-001__test") == "existing-456"
    fake_client.create_experiment.assert_not_called()


def test_set_experiment_tag_delegates_to_underlying_client():
    """req 3 (metadata): experiment-level description/hypothesis/dataset tags
    are set via MlflowClient.set_experiment_tag — a real, stable MLflow API
    (verified against mlflow-skinny 3.12.0 client / MLflow 2.18 server)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    pc.set_experiment_tag("exp-1", "mlflow.note.content", "some description")
    fake_client.set_experiment_tag.assert_called_once_with(
        "exp-1", "mlflow.note.content", "some description"
    )


def test_artifact_uri_for_run_reads_run_info():
    """req 5/G (S3 links): ingest.py needs the run's real artifact root to
    build s3_result_uri/s3_console_url tags without hand-rolling the MLflow
    artifact-location convention."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    fake_run = MagicMock()
    fake_run.info.artifact_uri = "s3://praxis-bucket/mlflow/artifacts/my-slug/run-1/artifacts"
    fake_client.get_run.return_value = fake_run
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    assert pc.artifact_uri_for_run("run-1") == \
        "s3://praxis-bucket/mlflow/artifacts/my-slug/run-1/artifacts"
    fake_client.get_run.assert_called_once_with("run-1")


def test_build_meta_dataset_resolves_s3_source_offline():
    """redesign item 4: the native dataset is built from the S3 URI without
    touching the network — the source resolves to an S3 artifact source."""
    from praxis_exp.mlflow_client import build_meta_dataset
    ds = build_meta_dataset(
        name="edge_full_20_rmc", source_uri="s3://praxis-bucket/data/edge_full_20/",
        digest="abc123",
    )
    assert ds.name == "edge_full_20_rmc"
    assert ds.digest == "abc123"


def test_log_input_delegates_to_log_inputs_with_context_tag():
    """redesign item 4/11: log a native Dataset to a specific run (backfill can't
    use the fluent active-run API)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient, build_meta_dataset
    fake_client = MagicMock()
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    ds = build_meta_dataset(name="edge_full_20_rmc", source_uri="s3://b/data/edge_full_20/")
    pc.log_input("run-1", ds, context="training")
    fake_client.log_inputs.assert_called_once()
    args = fake_client.log_inputs.call_args
    assert args.args[0] == "run-1"
    dataset_inputs = args.args[1]
    assert dataset_inputs[0].dataset.name == "edge_full_20_rmc"
    assert any(t.key == "mlflow.data.context" and t.value == "training"
               for t in dataset_inputs[0].tags)


def test_find_run_by_unit_returns_run_id_or_none():
    """redesign item 11: the backfill reuses a unit's existing child run
    (idempotency) by searching for its unit_id tag."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    found = MagicMock()
    found.info.run_id = "run-9"
    fake_client.search_runs.return_value = [found]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    assert pc.find_run_by_unit("exp-1", "s0__krum__persistent_optimizer__seed42") == "run-9"
    fake_client.search_runs.assert_called_once()
    assert "s0__krum__persistent_optimizer__seed42" in \
        fake_client.search_runs.call_args.kwargs["filter_string"]

    fake_client.search_runs.return_value = []
    assert pc.find_run_by_unit("exp-1", "missing") is None


def test_find_parent_run_returns_newest_non_failed():
    """find_parent_run returns the current launch's parent — the newest
    non-FAILED run tagged exp_id (aborted launches are FAILED) ."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    failed = MagicMock(); failed.info.run_id = "p-old"; failed.info.status = "FAILED"
    live = MagicMock(); live.info.run_id = "p-live"; live.info.status = "RUNNING"
    fake_client.search_runs.return_value = [failed, live]  # DESC: newest first
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    assert pc.find_parent_run("exp-1", "EXP-005c") == "p-live"
    assert "EXP-005c" in fake_client.search_runs.call_args.kwargs["filter_string"]

    fake_client.search_runs.return_value = []
    assert pc.find_parent_run("exp-1", "EXP-none") is None


def test_find_runs_by_unit_scopes_to_parent_when_given():
    """find_runs_by_unit adds a mlflow.parentRunId filter when a parent is given,
    so only the launch's children are returned (mlflow_client.py:93)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    r1 = MagicMock(); r1.info.run_id = "c1"
    fake_client.search_runs.return_value = [r1]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)

    assert pc.find_runs_by_unit("exp-1", "u1", parent_run_id="p-live") == ["c1"]
    filt = fake_client.search_runs.call_args.kwargs["filter_string"]
    assert "tags.unit_id = 'u1'" in filt
    assert "mlflow.parentRunId" in filt and "p-live" in filt

    pc.find_runs_by_unit("exp-1", "u1")  # no parent -> no parentRunId clause
    assert "parentRunId" not in fake_client.search_runs.call_args.kwargs["filter_string"]


# --- : signal dataset-by-source + artifact passthroughs ---

def test_build_signal_dataset_resolves_s3_signal_source_offline():
    """: the signal log is referenced as a dataset-by-source (no
    byte copy). Source resolves to S3ArtifactDatasetSource at the storage
    signal_key; default digest is a deterministic hash of that key (dedups on
    re-log). Assert via type.__name__ — the source class is NOT importable."""
    import hashlib
    from praxis_exp.mlflow_client import build_signal_dataset
    from praxis_exp import storage
    unit = "control_honest__krum__persistent_optimizer__seed42"
    ds = build_signal_dataset(exp_id="EXP-006", unit_id=unit, bucket="praxis-bucket",
                              defense_token="krum")
    key = storage.signal_key("EXP-006", unit)
    assert ds.name == "signal_krum"
    assert type(ds.source).__name__ == "S3ArtifactDatasetSource"
    assert ds.source._get_source_type() == "s3"
    assert ds.source.uri == f"s3://praxis-bucket/{key}"
    assert ds.digest == hashlib.sha1(key.encode()).hexdigest()[:12]


def test_build_signal_dataset_honors_explicit_digest():
    from praxis_exp.mlflow_client import build_signal_dataset
    ds = build_signal_dataset(exp_id="EXP-006", unit_id="u1", bucket="b",
                              defense_token="tge", digest="fixed123")
    assert ds.digest == "fixed123"
    assert ds.name == "signal_tge"


def test_list_artifacts_delegates():
    """emit_round_table's refresh needs to see existing artifacts (log_table
    APPENDS, so round_timeline.json must be deleted before re-logging)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    fake_client.list_artifacts.return_value = ["a", "b"]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    assert pc.list_artifacts("run-1") == ["a", "b"]
    fake_client.list_artifacts.assert_called_once_with("run-1")


def test_delete_artifact_uses_run_artifact_repository():
    """No per-artifact MlflowClient delete + mlflow.artifacts.delete_artifacts
    does not exist, so delete via the run's artifact repository."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    fake_run = MagicMock()
    fake_run.info.artifact_uri = "s3://b/mlflow/artifacts/slug/run-1/artifacts"
    fake_client.get_run.return_value = fake_run
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    with patch("praxis_exp.mlflow_client.get_artifact_repository") as g:
        repo = MagicMock(); g.return_value = repo
        pc.delete_artifact("run-1", "round_timeline.json")
        g.assert_called_once_with("s3://b/mlflow/artifacts/slug/run-1/artifacts")
        repo.delete_artifacts.assert_called_once_with("round_timeline.json")


def test_log_table_delegates():
    """: log_table passthrough (renders as a table in the 3.14 UI).
    It APPENDS, so emit_round_table refreshes first — tested there."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fake_client = MagicMock()
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fake_client)
    data = {"server_round": [1, 2], "f1": [0.1, 0.2]}
    pc.log_table("run-1", data, artifact_file="round_timeline.json")
    fake_client.log_table.assert_called_once_with(
        "run-1", data, artifact_file="round_timeline.json"
    )


def test_set_model_alias_delegates():
    """: sweep-scoped champion__/challenger__ alias assignment."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fc = MagicMock()
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fc)
    pc.set_model_alias("praxis-krum", "champion__EXP-006", "3")
    fc.set_registered_model_alias.assert_called_once_with("praxis-krum", "champion__EXP-006", "3")


def test_search_model_versions_forwards_filter_string():
    """The passthrough forwards a FILTER QUERY (name='...'), not a bare name."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fc = MagicMock()
    fc.search_model_versions.return_value = ["v1"]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fc)
    assert pc.search_model_versions("name='praxis-krum'") == ["v1"]
    fc.search_model_versions.assert_called_once_with("name='praxis-krum'")


# --- C: raw-search passthroughs for the self-heal finalizer/reaper ---

def test_search_runs_passthrough_forwards_query():
    """The self-heal finalizer/reaper locate parent runs by batch_array_job_id /
    exp_id tags and enumerate a parent's children — they need the raw Run
    objects (status, start_time, experiment_id, tags), so search_runs is a thin
    passthrough that forwards the query verbatim ( Lanes B/C)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fc = MagicMock()
    run = MagicMock(); run.info.run_id = "r1"
    fc.search_runs.return_value = [run]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fc)
    out = pc.search_runs(
        ["e1", "e2"], filter_string="tags.batch_array_job_id = 'job-1'",
        order_by=["attributes.start_time DESC"], max_results=50,
    )
    assert out == [run]
    call = fc.search_runs.call_args
    assert call.args[0] == ["e1", "e2"]
    assert call.kwargs["filter_string"] == "tags.batch_array_job_id = 'job-1'"
    assert call.kwargs["order_by"] == ["attributes.start_time DESC"]
    assert call.kwargs["max_results"] == 50


def test_list_experiment_ids_enumerates_experiments():
    """The finalizer/reaper search parent runs by tag across EVERY experiment (the
    Batch event / schedule carries no experiment id), so they need the full id
    list to pass to search_runs ( Lanes B/C)."""
    from praxis_exp.mlflow_client import PraxisMlflowClient
    fc = MagicMock()
    e1 = MagicMock(); e1.experiment_id = "1"
    e2 = MagicMock(); e2.experiment_id = "2"
    fc.search_experiments.return_value = [e1, e2]
    pc = PraxisMlflowClient(tracking_uri="http://test", _client=fc)
    assert pc.list_experiment_ids() == ["1", "2"]
    fc.search_experiments.assert_called_once()

from unittest.mock import MagicMock
from praxis_exp.batch import FakeBatchSubmitter, Boto3BatchSubmitter


def test_fake_records_submit_call():
    fake = FakeBatchSubmitter()
    job_id = fake.submit_array(
        job_name="EXP-005-dev", job_queue="q", job_definition="d", size=100,
        environment={"PRAXIS_EXP_ID": "EXP-005"},
    )
    assert job_id == "fake-array-job-id"
    assert fake.calls[0]["size"] == 100
    assert fake.calls[0]["environment"]["PRAXIS_EXP_ID"] == "EXP-005"


def test_boto3_submitter_maps_to_submit_job():
    client = MagicMock()
    client.submit_job.return_value = {"jobId": "real-array-id"}
    sub = Boto3BatchSubmitter(client)
    job_id = sub.submit_array(
        job_name="EXP-005-dev", job_queue="q", job_definition="d", size=100,
        environment={"PRAXIS_EXP_ID": "EXP-005", "PRAXIS_BUCKET": "b"},
    )
    assert job_id == "real-array-id"
    _, kwargs = client.submit_job.call_args
    assert kwargs["arrayProperties"] == {"size": 100}
    assert kwargs["jobQueue"] == "q" and kwargs["jobDefinition"] == "d"
    env = kwargs["containerOverrides"]["environment"]
    assert sorted(env, key=lambda e: e["name"]) == [
        {"name": "PRAXIS_BUCKET", "value": "b"},
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005"},
    ]


def test_fake_copies_environment_defensively():
    fake = FakeBatchSubmitter()
    env = {"K": "v"}
    fake.submit_array(job_name="n", job_queue="q", job_definition="d", size=2, environment=env)
    env["K"] = "MUTATED"
    assert fake.calls[0]["environment"]["K"] == "v"


def test_fake_job_definition_image_defaults_none_and_is_configurable():
    """GWU-48: None (default) opts the fake out of the digest guard; a
    configured reference drives it."""
    assert FakeBatchSubmitter().job_definition_image("jobdef:7") is None
    fake = FakeBatchSubmitter(job_def_image="repo@sha256:abc")
    assert fake.job_definition_image("jobdef:7") == "repo@sha256:abc"


def test_boto3_job_definition_image_resolves_via_describe():
    """GWU-48: resolve the job def's containerProperties.image so launch-matrix
    can compare it against the requested digest."""
    client = MagicMock()
    client.describe_job_definitions.return_value = {
        "jobDefinitions": [{"containerProperties": {"image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/praxis-flowerfl@sha256:deadbeef"}}]
    }
    sub = Boto3BatchSubmitter(client)
    image = sub.job_definition_image("arn:...:job-definition/JobDef:7")
    assert image.endswith("@sha256:deadbeef")
    _, kwargs = client.describe_job_definitions.call_args
    assert kwargs["jobDefinitions"] == ["arn:...:job-definition/JobDef:7"]


def test_boto3_job_definition_image_raises_when_unresolvable():
    """An unresolvable job def (empty list) or one with no image is
    unaccountable state for a launch about to run on it — raise, never return
    None (None is the fake opt-out only)."""
    import pytest
    client = MagicMock()
    client.describe_job_definitions.return_value = {"jobDefinitions": []}
    with pytest.raises(RuntimeError, match="no job definition"):
        Boto3BatchSubmitter(client).job_definition_image("missing:1")

    client.describe_job_definitions.return_value = {"jobDefinitions": [{"containerProperties": {}}]}
    with pytest.raises(RuntimeError, match="no containerProperties.image"):
        Boto3BatchSubmitter(client).job_definition_image("jobdef:7")

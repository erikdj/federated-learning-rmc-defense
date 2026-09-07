import textwrap
from unittest.mock import MagicMock
import pytest
from praxis_exp.storage import InMemoryObjectStore
from praxis_exp.batch import FakeBatchSubmitter
from praxis_exp.manifest import read_manifest
from praxis_exp.matrix_launch import launch_matrix, MatrixLaunchError


@pytest.fixture(autouse=True)
def _site_config_env(monkeypatch):
    """launch_matrix resolves its bucket via Config when not passed explicitly;
    these tests must not depend on the private, unshipped local_defaults."""
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")

DOC = textwrap.dedent('''\
    ---
    exp_id: EXP-005
    slug: h2-dev-sweep
    hypothesis: H2
    methodology_version: v1.9
    matrix:
      defenses: [Krum, TrustScore]
      scenarios: [S0, S4]
      seeds: [42, 137]
      mode: persistent_optimizer
      max_per_client: 2000000
      rounds: 50
    batch:
      job_queue: praxis-spot-queue
      job_definition: praxis-flowerfl-unit
    ---
    Dev sweep.
    ''')


def _repo(tmp_path):
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "experiments" / "EXP-005-h2-dev-sweep.md").write_text(DOC)
    return tmp_path


def _mlflow():
    c = MagicMock()
    c.get_or_create_experiment.return_value = "mlexp-1"
    c.create_run.return_value = "run-1"
    return c


def _git():
    g = MagicMock()
    g.working_tree_clean.return_value = True
    g.head_sha.return_value = "abc1234"
    g.resolve_push_branch.return_value = "main"
    # Default: no exp/ tag exists yet (first launch). Without this, MagicMock's
    # truthy auto-return would route every test down the serial-suffix path.
    g.tag_target_sha.return_value = None
    return g


def test_launch_matrix_writes_manifest_creates_experiment_tag_and_submits(tmp_path):
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    out = launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    exp_id, meta, units = read_manifest(store, "EXP-005")
    assert len(units) == 8 and meta["image_digest"] == "sha256:deadbeef"
    mlflow.get_or_create_experiment.assert_called_once()
    git.create_annotated_tag.assert_called_once()
    assert git.create_annotated_tag.call_args[0][1] == "exp/EXP-005"
    assert batch.calls[0]["size"] == 8
    assert batch.calls[0]["environment"]["PRAXIS_EXP_ID"] == "EXP-005"
    # entrypoint.py builds --scenario from this; omitting it KeyErrors every container
    assert batch.calls[0]["environment"]["PRAXIS_SCENARIO_DIR"] == "rmc/scenarios"
    # Portable defaults remain identifiable without embedding a private owner.
    assert batch.calls[0]["tags"]["EXP"] == "EXP-005"
    assert batch.calls[0]["tags"]["Project"] == "federated-learning-rmc-defense"
    assert batch.calls[0]["tags"]["Owner"] == "researcher"
    assert out["array_job_id"] == "fake-array-job-id" and out["n_units"] == 8


def test_launch_matrix_uses_environment_resource_tags(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    store, batch = InMemoryObjectStore(), FakeBatchSubmitter()
    monkeypatch.setenv("PRAXIS_PROJECT_TAG", "public-project")
    monkeypatch.setenv("PRAXIS_OWNER_TAG", "public-owner")
    monkeypatch.setenv("PRAXIS_PURPOSE_TAG", "reproducibility")

    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=_mlflow(), _git=_git(), _no_push=True,
    )

    assert batch.calls[0]["tags"] == {
        "EXP": "EXP-005", "Project": "public-project",
        "Owner": "public-owner", "Purpose": "reproducibility",
    }


def test_launch_matrix_stamps_parent_run_id_and_array_job_id_into_meta(tmp_path):
    """: a POST-SUBMIT best-effort meta update records this launch's
    parent_run_id and array_job_id so a later refill can nest children under the
    SAME parent run and verify the prior array is terminal."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    _, meta, _ = read_manifest(store, "EXP-005")
    assert meta["parent_run_id"] == "run-1"
    assert meta["array_job_id"] == "fake-array-job-id"


def test_launch_matrix_refuses_dirty_tree(tmp_path):
    repo = _repo(tmp_path)
    git = _git(); git.working_tree_clean.return_value = False
    with pytest.raises(MatrixLaunchError, match="uncommitted"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
                      _mlflow=_mlflow(), _git=git, _no_push=True)


def test_launch_matrix_pushes_when_not_no_push(tmp_path):
    repo = _repo(tmp_path)
    git = _git()
    launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                  container_tracking_uri="http://10.0.0.10:5000",
                  _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
                  _mlflow=_mlflow(), _git=git, _no_push=False)
    git.resolve_push_branch.assert_called_once_with(repo, None)
    git.push_with_tags.assert_called_once_with(repo, branch="main")


def test_launch_matrix_pushes_to_explicit_branch_override(tmp_path):
    repo = _repo(tmp_path)
    git = _git()
    git.resolve_push_branch.return_value = "release"

    launch_matrix(
        repo, "EXP-005", image_digest="sha256:x", branch="release",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
        _mlflow=_mlflow(), _git=git,
    )

    git.resolve_push_branch.assert_called_once_with(repo, "release")
    git.push_with_tags.assert_called_once_with(repo, branch="release")


def test_launch_matrix_no_push_skips_branch_resolution_and_remote_push(tmp_path):
    repo = _repo(tmp_path)
    git = _git()
    batch = FakeBatchSubmitter()

    launch_matrix(
        repo, "EXP-005", image_digest="sha256:x", branch="ignored",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=InMemoryObjectStore(), _batch=batch,
        _mlflow=_mlflow(), _git=git, _no_push=True,
    )

    git.resolve_push_branch.assert_not_called()
    git.push_with_tags.assert_not_called()
    git.create_annotated_tag.assert_called_once()
    assert batch.calls


def test_existing_manifest_message_routes_operator_to_refill(tmp_path):
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    units = _doc_units()
    _seed_prior_launch(store, "EXP-005", units, done_count=1)

    with pytest.raises(MatrixLaunchError) as exc:
        launch_matrix(
            repo, "EXP-005", image_digest="sha256:deadbeef",
            container_tracking_uri="http://10.0.0.10:5000",
            _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(),
            _git=_git(), _no_push=True,
        )

    message = str(exc.value)
    assert "--refill" in message
    assert "refill support disabled" not in message


def test_launch_matrix_rolls_back_on_tag_push_failure(tmp_path):
    """Rollback symmetry : tag/push failure
    terminates the parent FAILED, deletes the tag this launch created, AND
    deletes this launch's manifest — so the retry is not refused by the
    one-launch-per-EXP guard."""
    from praxis_exp.storage import manifest_key
    repo = _repo(tmp_path)
    git = _git()
    git.push_with_tags.side_effect = RuntimeError("push rejected")
    mlflow, store = _mlflow(), InMemoryObjectStore()
    with pytest.raises(MatrixLaunchError, match="tag/push failed"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(),
                      _mlflow=mlflow, _git=git, _no_push=False)
    git.delete_local_tag.assert_called_once()
    mlflow.set_terminated.assert_called_once_with("run-1", "FAILED")
    assert store.head(manifest_key("EXP-005")) is False  # manifest rolled back


def test_launch_matrix_rolls_back_on_submit_failure(tmp_path):
    from unittest.mock import MagicMock
    from praxis_exp.storage import manifest_key
    repo = _repo(tmp_path)
    mlflow, store = _mlflow(), InMemoryObjectStore()
    batch = MagicMock()
    batch.submit_array.side_effect = RuntimeError("batch down")
    with pytest.raises(MatrixLaunchError, match="Batch submit failed"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=batch,
                      _mlflow=mlflow, _git=_git(), _no_push=True)
    mlflow.set_terminated.assert_called_once_with("run-1", "FAILED")
    assert store.head(manifest_key("EXP-005")) is False  # manifest rolled back


def test_launch_matrix_enrichment_failure_rolls_back(tmp_path):
    """ : the post-parent-run enrichment
    (log_params here) previously sat outside all rollback handlers — a
    transient MLflow error exited with the manifest written and the parent
    stuck RUNNING, and the close-out's prior-manifest guard then refused
    the retry. Now: parent FAILED, manifest deleted, no tag created,
    MatrixLaunchError raised with the cause."""
    from praxis_exp.storage import manifest_key
    repo = _repo(tmp_path)
    mlflow, git, store = _mlflow(), _git(), InMemoryObjectStore()
    mlflow.log_params.side_effect = RuntimeError("mlflow down")
    with pytest.raises(MatrixLaunchError, match="mlflow down"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(),
                      _mlflow=mlflow, _git=git, _no_push=True)
    mlflow.set_terminated.assert_called_once_with("run-1", "FAILED")
    assert store.head(manifest_key("EXP-005")) is False  # manifest rolled back
    git.create_annotated_tag.assert_not_called()  # failed before the tag step


def test_launch_matrix_rollback_step_failure_does_not_mask_original(tmp_path):
    """A rollback step failing (manifest delete raises) must not mask the
    original error: the raised message carries the original cause AND names
    the leftover manifest with the manual-cleanup runbook."""
    repo = _repo(tmp_path)

    class _DeleteRaisesStore(InMemoryObjectStore):
        def delete(self, key):
            raise RuntimeError("s3 delete denied")

    mlflow, store = _mlflow(), _DeleteRaisesStore()
    mlflow.log_params.side_effect = RuntimeError("mlflow down")
    with pytest.raises(MatrixLaunchError) as exc:
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(),
                      _mlflow=mlflow, _git=_git(), _no_push=True)
    msg = str(exc.value)
    assert "mlflow down" in msg          # original error preserved
    assert "manifest.json" in msg        # leftover named
    assert "manually delete" in msg      # falls back to the manual runbook


def test_launch_matrix_uses_container_tracking_uri_for_batch_env(tmp_path):
    """The AWS Batch container environment gets the VPC-reachable MLflow URI,
    not the operator's localhost SSM-tunnel URI used by the local client."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        tracking_uri="http://localhost:5001",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    assert batch.calls[0]["environment"]["MLFLOW_TRACKING_URI"] == "http://10.0.0.10:5000"


def test_launch_matrix_refuses_localhost_container_uri(tmp_path):
    """Without an explicit container URI, the default tracking_uri (localhost)
    would be unreachable from inside the VPC; launch_matrix must refuse before
    any side effects (no manifest, no parent run, no tag)."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    with pytest.raises(MatrixLaunchError, match="PRAXIS_CONTAINER_MLFLOW_URI"):
        launch_matrix(
            repo, "EXP-005", image_digest="sha256:deadbeef",
            _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
        )
    from praxis_exp.storage import manifest_key
    assert store.head(manifest_key("EXP-005")) is False
    mlflow.get_or_create_experiment.assert_not_called()
    mlflow.create_run.assert_not_called()
    git.create_annotated_tag.assert_not_called()


def test_launch_matrix_refuses_127_0_0_1_container_uri(tmp_path):
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    with pytest.raises(MatrixLaunchError, match="PRAXIS_CONTAINER_MLFLOW_URI"):
        launch_matrix(
            repo, "EXP-005", image_digest="sha256:deadbeef",
            container_tracking_uri="http://127.0.0.1:5001",
            _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
        )
    from praxis_exp.storage import manifest_key
    assert store.head(manifest_key("EXP-005")) is False
    git.create_annotated_tag.assert_not_called()


def test_launch_matrix_container_uri_falls_back_to_non_localhost_tracking_uri(tmp_path):
    """When container_tracking_uri is not passed, launch_matrix falls back to
    tracking_uri; if that is already VPC-reachable (non-localhost), no error."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        tracking_uri="http://10.0.0.5:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    assert batch.calls[0]["environment"]["MLFLOW_TRACKING_URI"] == "http://10.0.0.5:5000"


def test_launch_matrix_names_experiment_by_slug(tmp_path):
    """req 1: MLflow experiment = doc.slug (the design family), not
    f'{exp_id}__{slug}' — so re-launches of the same design land as new RUNS
    in the SAME experiment instead of spawning a new experiment per exp_id."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    mlflow.get_or_create_experiment.assert_called_once_with("h2-dev-sweep")


def test_launch_matrix_parent_run_named_exp_id(tmp_path):
    """Parent run is NAMED EXP-NNN (mlflow.runName) — re-launches of the same
    slug become additional parent runs in the same experiment, distinguished
    by run name/launched_at rather than by experiment identity."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    tags = mlflow.create_run.call_args[1]["tags"]
    assert tags["mlflow.runName"] == "EXP-005"


def test_launch_matrix_second_launch_reuses_experiment_new_parent_run(tmp_path):
    """Repeated launches of the same experiment use distinct parent runs.

    Re-launching EXP-005's design doc must reuse the slug-named experiment and
    create a second parent run rather than a second experiment.
    """
    repo = _repo(tmp_path)
    mlflow = _mlflow()
    mlflow.get_or_create_experiment.return_value = "mlexp-1"
    mlflow.create_run.side_effect = ["run-1", "run-2"]

    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
        _mlflow=mlflow, _git=_git(), _no_push=True,
    )
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
        _mlflow=mlflow, _git=_git(), _no_push=True,
    )
    assert mlflow.get_or_create_experiment.call_count == 2
    assert [c.args[0] for c in mlflow.get_or_create_experiment.call_args_list] == \
        ["h2-dev-sweep", "h2-dev-sweep"]
    assert mlflow.create_run.call_count == 2


def test_launch_matrix_tags_experiment_with_description_hypothesis_dataset(tmp_path):
    """req 3: experiment-level metadata — mlflow.note.content (description),
    hypothesis, dataset — set once per launch via set_experiment_tag (real
    MlflowClient API), idempotent on relaunch."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    mlflow.get_or_create_experiment.return_value = "mlexp-1"
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    tag_calls = {c.args[1]: c.args[2] for c in mlflow.set_experiment_tag.call_args_list
                 if c.args[0] == "mlexp-1"}
    assert "mlflow.note.content" in tag_calls
    assert "Dev sweep" in tag_calls["mlflow.note.content"] or "h2-dev-sweep" in tag_calls["mlflow.note.content"]
    assert tag_calls["hypothesis"] == "H2"
    assert tag_calls["dataset"] == "edge_full_20_rmc"


def test_launch_matrix_parent_run_params_and_s3_tags(tmp_path):
    """req 3 (matrix params) + req 5 (S3 links): parent run gets the sweep
    matrix as PARAMS (not tags) and s3_manifest_uri/s3_console_url as TAGS."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        artifact_bucket="test-bucket",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    params = mlflow.log_params.call_args[0][1]
    assert params["defenses"] == "Krum,TrustScore"
    assert params["scenarios"] == "S0,S4"
    assert params["seeds"] == "42,137"
    assert params["n_units"] == "8"
    # the replicate axis is always recorded; this
    # repeats-less DOC parses to repeats=1 (axis inactive).
    assert params["repeats"] == "1"

    tags = mlflow.create_run.call_args[1]["tags"]
    assert tags["s3_manifest_uri"] == "s3://test-bucket/sweeps/EXP-005/manifest.json"
    assert tags["s3_console_url"] == \
        "https://us-east-1.console.aws.amazon.com/s3/buckets/test-bucket?prefix=sweeps/EXP-005/"


def test_launch_matrix_passes_parent_run_id_and_methodology_env(tmp_path):
    """Batch env carries PRAXIS_PARENT_RUN_ID (so entrypoint.py can nest child
    runs under the sweep parent) and PRAXIS_METHODOLOGY_VERSION (so entrypoint
    can tag units without re-parsing the design doc)."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    mlflow.create_run.return_value = "run-xyz"
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    env = batch.calls[0]["environment"]
    assert env["PRAXIS_PARENT_RUN_ID"] == "run-xyz"
    assert env["PRAXIS_METHODOLOGY_VERSION"] == "v1.9"


def test_launch_matrix_injects_launch_commit_not_runner_commit(tmp_path):
    """Launch HEAD is legitimate provenance for the LAUNCH-side inputs (manifest,
    scenario JSONs) → PRAXIS_LAUNCH_COMMIT. It must NOT be injected as
    PRAXIS_RUNNER_COMMIT: that env would override the image-baked runner commit
    and record false provenance for the running code."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    env = batch.calls[0]["environment"]
    assert env["PRAXIS_LAUNCH_COMMIT"] == "abc1234"
    assert "PRAXIS_RUNNER_COMMIT" not in env


def test_launch_matrix_creates_serial_tag_when_existing_tag_at_different_sha(tmp_path):
    """Existing exp/EXP-NNN at a DIFFERENT SHA is an immutable custody anchor
    — never force-moved. The relaunch records its own SHA under a serial
    suffixed tag exp/EXP-NNN.r2, and the parent run is tagged with the
    actual tag name used."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    git.tag_target_sha.side_effect = lambda _repo, name: {
        "exp/EXP-005": "0ld00000",
    }.get(name)
    out = launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    git.create_annotated_tag.assert_called_once()
    assert git.create_annotated_tag.call_args[0][1] == "exp/EXP-005.r2"
    assert out["git_tag"] == "exp/EXP-005.r2"
    assert mlflow.create_run.call_args[1]["tags"]["git_tag"] == "exp/EXP-005.r2"


def test_launch_matrix_serial_tag_skips_taken_suffixes(tmp_path):
    """Third relaunch at a third SHA:.r2 is taken at another SHA, so.r3."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    git.tag_target_sha.side_effect = lambda _repo, name: {
        "exp/EXP-005": "0ld00001",
        "exp/EXP-005.r2": "0ld00002",
    }.get(name)
    out = launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    assert git.create_annotated_tag.call_args[0][1] == "exp/EXP-005.r3"
    assert out["git_tag"] == "exp/EXP-005.r3"


def test_launch_matrix_rollback_deletes_the_serial_tag_it_created(tmp_path):
    """Rollback must delete the tag THIS launch created (the.r2 suffix),
    not the base exp/EXP-NNN custody anchor from the prior launch."""
    repo = _repo(tmp_path)
    git = _git()
    git.tag_target_sha.side_effect = lambda _repo, name: {
        "exp/EXP-005": "0ld00000",
    }.get(name)
    git.push_with_tags.side_effect = RuntimeError("push rejected")
    with pytest.raises(MatrixLaunchError, match="tag/push failed"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=InMemoryObjectStore(), _batch=FakeBatchSubmitter(),
                      _mlflow=_mlflow(), _git=git, _no_push=False)
    git.delete_local_tag.assert_called_once()
    assert git.delete_local_tag.call_args[0][1] == "exp/EXP-005.r2"


def _doc_units():
    """The Units the DOC's matrix expands to (mirrors expand_matrix order)."""
    from praxis_exp.units import expand_matrix
    return expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
                         "persistent_optimizer", 2_000_000, 50)


def _seed_done_markers(store, exp_id, units):
    from praxis_exp import storage
    for u in units:
        store.put_bytes(storage.marker_key(exp_id, u.unit_id), b"")


def _seed_prior_launch(store, exp_id, units, done_count, meta=None):
    """Arrange a realistic prior partial launch: its manifest + some markers.

    Default meta mirrors what launch_matrix itself writes for the doc/args
    these tests use (methodology v1.9, head sha abc1234, image
    sha256:deadbeef) — i.e. an identical-provenance prior launch. Override
    fields via ``meta`` to simulate provenance drift."""
    from praxis_exp.manifest import write_manifest
    full_meta = {
        "methodology_version": "v1.9",
        "git_sha": "abc1234",
        "image_digest": "sha256:deadbeef",
        "n_units": len(units),
    }
    if meta:
        full_meta.update(meta)
    write_manifest(store, exp_id, units, meta=full_meta)
    _seed_done_markers(store, exp_id, units[:done_count])


def _assert_refill_refused_no_side_effects(repo, store, match):
    """Run launch_matrix expecting a refill refusal; assert zero side effects
    and return the raised error message."""
    from praxis_exp.manifest import read_manifest
    mlflow, git, batch = _mlflow(), _git(), FakeBatchSubmitter()
    with pytest.raises(MatrixLaunchError, match=match) as exc:
        launch_matrix(
            repo, "EXP-005", image_digest="sha256:deadbeef",
            container_tracking_uri="http://10.0.0.10:5000",
            _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
        )
    mlflow.get_or_create_experiment.assert_not_called()
    mlflow.create_run.assert_not_called()
    git.create_annotated_tag.assert_not_called()
    assert batch.calls == []
    return str(exc.value)


def test_launch_matrix_refuses_orphaned_marker_under_old_unit_id(tmp_path):
    """ : the freshness probe was unit-id-keyed
    (is_done over the NEW expansion only) — after a manual manifest removal
    with a CHANGED matrix, old markers under other unit ids evaded it and
    the namespace was treated as fresh. The check is now namespace-level:
    ANY object under sweeps/{exp_id}/ refuses, naming the found key."""
    from praxis_exp import storage
    repo = _repo(tmp_path)  # DOC expands to seed42/137 unit ids
    store = InMemoryObjectStore()
    # orphaned marker from a pre-manifest-delete launch with a DIFFERENT matrix
    old_key = storage.marker_key("EXP-005", "s0__krum__persistent_optimizer__seed7")
    store.put_bytes(old_key, b"")
    msg = _assert_refill_refused_no_side_effects(repo, store, match="not empty")
    assert old_key in msg  # found key named
    assert "NEW EXP id" in msg


def test_launch_matrix_refuses_stray_result_object(tmp_path):
    """Any stray object — here a result JSON alone, a key class the old
    is_done probe never looked at — refuses too: 'empty' has no unit-id
    blind spots."""
    from praxis_exp import storage
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    stray = storage.result_key("EXP-005", "s0__krum__persistent_optimizer__seed42")
    store.put_bytes(stray, b"{}")
    msg = _assert_refill_refused_no_side_effects(repo, store, match="not empty")
    assert stray in msg


def test_launch_matrix_rollback_then_retry_succeeds(tmp_path):
    """CRITICAL lock ( x ): the transient-retry
    flow the rollback work exists for. First attempt fails in enrichment ->
    rollback deletes this launch's manifest (and writes nothing else to the
    namespace by construction) -> the retry passes BOTH the prior-manifest
    guard and the namespace-emptiness check and launches."""
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    flaky_mlflow = _mlflow()
    flaky_mlflow.log_params.side_effect = RuntimeError("mlflow down")
    with pytest.raises(MatrixLaunchError, match="mlflow down"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(),
                      _mlflow=flaky_mlflow, _git=_git(), _no_push=True)
    # retry on the SAME store with healthy MLflow
    batch = FakeBatchSubmitter()
    out = launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                        container_tracking_uri="http://10.0.0.10:5000",
                        _store=store, _batch=batch,
                        _mlflow=_mlflow(), _git=_git(), _no_push=True)
    assert batch.calls and out["n_units"] == 8  # retry launched cleanly


def test_launch_matrix_fresh_launch_keeps_expansion_order(tmp_path):
    """Fresh namespace: manifest ordering is the plain expansion order with
    contiguous indices — index stabilization only applies to refills."""
    from praxis_exp.manifest import read_manifest
    from praxis_exp.units import expand_matrix
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(), _git=_git(),
        _no_push=True,
    )
    _, _, new_units = read_manifest(store, "EXP-005")
    expected = expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
                             "persistent_optimizer", 2_000_000, 50)
    by_index = sorted(new_units, key=lambda u: u.array_index)
    assert [u.unit_id for u in by_index] == [u.unit_id for u in expected]
    assert sorted(u.array_index for u in new_units) == list(range(8))


def _repo_with_repeats(tmp_path, n):
    """A repo whose EXP-005 design doc carries matrix.repeats = n."""
    doc = DOC.replace("  rounds: 50\n", f"  rounds: 50\n  repeats: {n}\n")
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "experiments" / "EXP-005-h2-dev-sweep.md").write_text(doc)
    return tmp_path


def test_launch_matrix_threads_repeats_into_expansion(tmp_path):
    """: matrix.repeats reaches expand_matrix — a repeats=2 doc
    expands each (config,scenario,seed) into two __rep-suffixed replicate
    units with a single contiguous array_index across the whole product."""
    repo = _repo_with_repeats(tmp_path, 2)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    out = launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    assert out["n_units"] == 16  # 2 defenses x 2 scenarios x 2 seeds x 2 repeats
    _, _, units = read_manifest(store, "EXP-005")
    ids = {u.unit_id for u in units}
    assert len(ids) == 16
    assert all("__rep" in i for i in ids)
    assert "s0__krum__persistent_optimizer__seed42__rep1" in ids
    assert "s0__krum__persistent_optimizer__seed42__rep2" in ids
    assert sorted(u.array_index for u in units) == list(range(16))


def test_launch_matrix_refuses_any_prior_manifest_with_runbook(tmp_path):
    """A normal second launch refuses and directs a partial prior launch to
    the explicit refill workflow while preserving the debris runbook."""
    from praxis_exp.manifest import read_manifest
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    _seed_prior_launch(store, "EXP-005", _doc_units(), done_count=0)
    msg = _assert_refill_refused_no_side_effects(repo, store, match="--refill")
    assert "NEW EXP id" in msg
    assert "manifest.json" in msg  # manual-delete runbook for verified debris
    # prior manifest untouched (launch_matrix's own write stamps launched_at)
    _, meta, _ = read_manifest(store, "EXP-005")
    assert "launched_at" not in meta


def test_launch_matrix_existing_tag_fresh_namespace_mints_serial(tmp_path):
    """ : tag annotations embed launch-specific
    facts (parent run id, image digest, manifest key), so NO existing
    annotation is ever reused — even at the same sha (e.g. relaunching after
    a manual namespace clear). The launch mints the next free serial tag
    with ITS OWN tag_msg; the old tag is untouched."""
    repo = _repo(tmp_path)
    store, batch, mlflow, git = InMemoryObjectStore(), FakeBatchSubmitter(), _mlflow(), _git()
    git.tag_target_sha.side_effect = lambda _repo, name: {
        "exp/EXP-005": "abc1234",  # exists, even at HEAD's sha
    }.get(name)
    out = launch_matrix(
        repo, "EXP-005", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True,
    )
    git.create_annotated_tag.assert_called_once()
    assert git.create_annotated_tag.call_args[0][1] == "exp/EXP-005.r2"
    assert out["git_tag"] == "exp/EXP-005.r2"
    assert mlflow.create_run.call_args[1]["tags"]["git_tag"] == "exp/EXP-005.r2"
    git.delete_local_tag.assert_not_called()  # old tag untouched


def test_launch_matrix_refuses_when_job_def_image_digest_mismatches(tmp_path):
    """: --image-digest is provenance-only; the job def selects the image
    AWS Batch runs. A job def pinned to a DIFFERENT digest than requested is a
    silent chain-of-custody break — refuse before any side effect."""
    from praxis_exp.storage import manifest_key
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    batch = FakeBatchSubmitter(job_def_image="repo@sha256:oldimage")
    git = _git()
    with pytest.raises(MatrixLaunchError, match="does not match the requested"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=batch, _mlflow=_mlflow(), _git=git, _no_push=True)
    # zero side effects: no manifest, no tag, no submit
    assert store.head(manifest_key("EXP-005")) is False
    git.create_annotated_tag.assert_not_called()
    assert batch.calls == []


def test_launch_matrix_proceeds_when_job_def_image_contains_digest(tmp_path):
    """A job def pinned to exactly the requested digest passes the guard."""
    repo = _repo(tmp_path)
    batch = FakeBatchSubmitter(job_def_image="repo@sha256:deadbeef")
    out = launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                        container_tracking_uri="http://10.0.0.10:5000",
                        _store=InMemoryObjectStore(), _batch=batch, _mlflow=_mlflow(),
                        _git=_git(), _no_push=True)
    assert out["n_units"] == 8 and batch.calls[0]["size"] == 8


def test_launch_matrix_refuses_truncated_digest_prefix(tmp_path):
    """the guard compares the parsed digest by EQUALITY, not
    substring — a truncated request (sha256:dead) must NOT pass against a longer
    real digest (sha256:deadbeef...) or the manifest records provenance that does
    not identify the image actually run."""
    repo = _repo(tmp_path)
    batch = FakeBatchSubmitter(job_def_image="repo@sha256:deadbeef01234567")
    with pytest.raises(MatrixLaunchError, match="does not match the requested"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",  # a prefix of the real digest
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=InMemoryObjectStore(), _batch=batch, _mlflow=_mlflow(),
                      _git=_git(), _no_push=True)


def test_launch_matrix_matches_digest_when_job_def_ref_has_repo_and_tag(tmp_path):
    """A full repo@sha256 reference matches when the digest token is equal."""
    repo = _repo(tmp_path)
    batch = FakeBatchSubmitter(
        job_def_image="123.dkr.ecr.us-east-1.amazonaws.com/praxis-flowerfl:v3@sha256:deadbeef")
    out = launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                        container_tracking_uri="http://10.0.0.10:5000",
                        _store=InMemoryObjectStore(), _batch=batch, _mlflow=_mlflow(),
                        _git=_git(), _no_push=True)
    assert out["n_units"] == 8


def test_launch_matrix_refuses_tag_pinned_job_def_as_unverifiable(tmp_path):
    """A tag-pinned job def (no @sha256) cannot be proven to run the requested
    digest — treat as a mismatch and refuse."""
    repo = _repo(tmp_path)
    batch = FakeBatchSubmitter(job_def_image="repo:v3")
    with pytest.raises(MatrixLaunchError, match="does not match the requested"):
        launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=InMemoryObjectStore(), _batch=batch, _mlflow=_mlflow(),
                      _git=_git(), _no_push=True)


def test_launch_matrix_allow_digest_mismatch_overrides_loudly(tmp_path, capsys):
    """--allow-digest-mismatch downgrades the hard failure to a LOUD warning and
    proceeds (escape hatch, never silent)."""
    repo = _repo(tmp_path)
    batch = FakeBatchSubmitter(job_def_image="repo@sha256:oldimage")
    out = launch_matrix(repo, "EXP-005", image_digest="sha256:deadbeef",
                        container_tracking_uri="http://10.0.0.10:5000",
                        _store=InMemoryObjectStore(), _batch=batch, _mlflow=_mlflow(),
                        _git=_git(), _no_push=True, allow_digest_mismatch=True)
    assert out["n_units"] == 8 and batch.calls  # launched anyway
    assert "WARNING: --allow-digest-mismatch" in capsys.readouterr().out


def test_launch_matrix_rejects_size_below_two(tmp_path):
    one_unit_doc = DOC.replace("defenses: [Krum, TrustScore]", "defenses: [Krum]") \
                      .replace("scenarios: [S0, S4]", "scenarios: [S0]") \
                      .replace("seeds: [42, 137]", "seeds: [42]")
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "experiments" / "EXP-005-h2-dev-sweep.md").write_text(one_unit_doc)
    store = InMemoryObjectStore()
    with pytest.raises(MatrixLaunchError, match="size"):
        launch_matrix(tmp_path, "EXP-005", image_digest="sha256:x",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(), _git=_git(), _no_push=True)
    # guard fired before side effects: no manifest written
    from praxis_exp.storage import manifest_key
    assert store.head(manifest_key("EXP-005")) is False

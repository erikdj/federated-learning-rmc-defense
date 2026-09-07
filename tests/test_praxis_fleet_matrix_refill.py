import json
import textwrap
from unittest.mock import MagicMock

import pytest

from praxis_exp.batch import FakeBatchSubmitter
from praxis_exp.manifest import write_manifest
from praxis_exp.matrix_launch import MatrixLaunchError
from praxis_exp.matrix_refill import refill_matrix
from praxis_exp.storage import InMemoryObjectStore, marker_key
from praxis_exp.units import expand_matrix


@pytest.fixture(autouse=True)
def _site_config_env(monkeypatch):
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


def _doc_units(rounds=50, max_per_client=2_000_000):
    return expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
                         "persistent_optimizer", max_per_client, rounds)


def _mlflow():
    c = MagicMock()
    c.get_or_create_experiment.return_value = "mlexp-1"
    c.create_run.return_value = "refill-parent-1"
    return c


def _git():
    """A prior sweep already minted exp/EXP-005, so the base tag EXISTS — a
    refill must mint the next serial suffix."""
    g = MagicMock()
    g.working_tree_clean.return_value = True
    g.head_sha.return_value = "abc1234"
    g.tag_target_sha.side_effect = lambda _repo, name: {"exp/EXP-005": "0ldsha00"}.get(name)
    return g


def _seed_launched(store, exp_id, units, *, done_ids, meta_overrides=None):
    """Arrange a prior launch: its manifest (with GWU-41 parent_run_id +
    array_job_id recorded) plus done-markers for the completed cells."""
    meta = {
        "methodology_version": "v1.9",
        "image_digest": "sha256:deadbeef",
        "git_sha": "abc1234",
        "n_units": len(units),
        "launched_at": "2026-07-20T00:00:00Z",
        "parent_run_id": "parent-orig",
        "array_job_id": "arr-orig",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    write_manifest(store, exp_id, units, meta)
    for uid in done_ids:
        store.put_bytes(marker_key(exp_id, uid), b"")
    return meta


def _all_but(units, missing_indices):
    """unit_ids of every cell EXCEPT the given array indices (i.e. the done set)."""
    return [u.unit_id for u in units if u.array_index not in missing_indices]


def _refill(repo, store, cells, **kw):
    batch = kw.pop("batch", FakeBatchSubmitter())
    mlflow = kw.pop("mlflow", _mlflow())
    git = kw.pop("git", _git())
    return refill_matrix(
        repo, "EXP-005", cells, image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=batch, _mlflow=mlflow, _git=git, _no_push=True, **kw,
    ), batch, mlflow, git


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

def test_refill_resubmits_full_array_and_links_original_parent(tmp_path):
    """The refill re-submits the FULL canonical array (done cells skip on their
    markers) with children nesting under the ORIGINAL parent run, and mints the
    next serial tag."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    # missing = indices {0, 6} -> name them by unit_id
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    out, batch, mlflow, git = _refill(repo, store, missing_ids)

    assert batch.calls[0]["size"] == 8  # full array; done cells skip
    env = batch.calls[0]["environment"]
    assert env["PRAXIS_EXP_ID"] == "EXP-005"
    assert env["PRAXIS_PARENT_RUN_ID"] == "parent-orig"  # SAME parent run (req d)
    assert env["PRAXIS_MANIFEST_KEY"] == "sweeps/EXP-005/manifest.json"  # original manifest
    # launch HEAD is LAUNCH-side provenance, never masquerades as the running
    # code's commit — must not clobber the image-baked runner commit
    assert env["PRAXIS_LAUNCH_COMMIT"] == "abc1234"
    assert "PRAXIS_RUNNER_COMMIT" not in env
    git.create_annotated_tag.assert_called_once()
    assert git.create_annotated_tag.call_args[0][1] == "exp/EXP-005.r2"  # serial (invariant 6)
    assert out["git_tag"] == "exp/EXP-005.r2" and out["serial"] == "r2"
    assert sorted(out["refilled_cells"]) == sorted(missing_ids)
    assert out["parent_run_id"] == "parent-orig" and out["minted_parent"] is False
    assert batch.calls[0]["tags"]["Project"] == "federated-learning-rmc-defense"
    assert batch.calls[0]["tags"]["Owner"] == "researcher"
    # no new parent run minted
    mlflow.create_run.assert_not_called()


def test_refill_uses_environment_resource_tags(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    monkeypatch.setenv("PRAXIS_PROJECT_TAG", "public-project")
    monkeypatch.setenv("PRAXIS_OWNER_TAG", "public-owner")
    monkeypatch.setenv("PRAXIS_PURPOSE_TAG", "reproducibility")

    _, batch, *_ = _refill(repo, store, None)

    assert batch.calls[0]["tags"] == {
        "EXP": "EXP-005", "Project": "public-project",
        "Owner": "public-owner", "Purpose": "reproducibility",
    }


def test_refill_full_array_leaves_done_cells_as_idempotent_skips(tmp_path):
    """Item 3 (design): the refill submits the FULL array; the already-DONE cells
    are the idempotent no-ops (the container's should_skip returns 0 before any
    MLflow contact — pinned by test_fleet_entrypoint.test_main_skip_path_never_
    touches_mlflow). Here: only the missing cells are analysis-bearing, and the
    array size minus the refilled count equals the done-cell count that will
    skip. No padding is needed because the in-namespace full-array resubmit
    always meets the Batch size>=2 floor."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {2, 5}))  # 6 done, 2 missing
    out, batch, *_ = _refill(repo, store, None)
    array_size = batch.calls[0]["size"]
    assert array_size == 8                       # full array resubmitted
    assert out["n_refilled"] == 2                # only the missing cells are analysis-bearing
    assert array_size - out["n_refilled"] == 6   # the 6 done cells skip (idempotent no-op)


def test_refill_writes_pre_registered_contract_record(tmp_path):
    """(c) The refill writes a refill-scoped record naming the analysis-bearing
    cells, WITHOUT touching the original manifest (invariant 4)."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    before = store.get_bytes("sweeps/EXP-005/manifest.json")
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    out, *_ = _refill(repo, store, missing_ids)

    record = json.loads(store.get_bytes(out["refill_record_key"]))
    assert record["meta"]["refill_of"] == "EXP-005"
    assert record["meta"]["serial"] == "r2"
    assert sorted(record["meta"]["refilled_cells"]) == sorted(missing_ids)
    assert record["meta"]["parent_run_id"] == "parent-orig"
    # original manifest is byte-for-byte untouched
    assert store.get_bytes("sweeps/EXP-005/manifest.json") == before


def test_refill_tags_parent_run_with_bookkeeping(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    _, batch, mlflow, _ = _refill(repo, store, missing_ids)
    tagged = {c.args[1]: c.args[2] for c in mlflow.set_tag.call_args_list}
    assert tagged["refill_r2_array_job_id"] == "fake-array-job-id"
    assert set(tagged["refill_r2_cells"].split(",")) == set(missing_ids)


def test_refill_without_cells_refills_all_missing(tmp_path):
    """--cells omitted → refill every currently-missing cell."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {2, 5}))
    out, batch, *_ = _refill(repo, store, None)
    expected = [u.unit_id for u in units if u.array_index in {2, 5}]
    assert sorted(out["refilled_cells"]) == sorted(expected)
    assert batch.calls[0]["size"] == 8


def test_refill_resolves_cells_by_array_index(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    out, *_ = _refill(repo, store, ["0", "6"])  # array indices
    expected = [u.unit_id for u in units if u.array_index in {0, 6}]
    assert sorted(out["refilled_cells"]) == sorted(expected)


# --------------------------------------------------------------------------
# Refusals — invariants 1, 2, 3, and (a)/(b)
# --------------------------------------------------------------------------

def test_refill_refuses_when_no_prior_manifest(tmp_path):
    repo = _repo(tmp_path)
    store = InMemoryObjectStore()
    with pytest.raises(MatrixLaunchError, match="nothing to refill"):
        _refill(repo, store, ["0", "1"])


def test_refill_refuses_when_fully_complete(tmp_path):
    """Invariant 1: nothing missing → refuse."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=[u.unit_id for u in units])
    with pytest.raises(MatrixLaunchError, match="fully complete"):
        _refill(repo, store, None)


def test_refill_refuses_named_cell_that_is_already_done(tmp_path):
    """(b) Refuse a named cell that already has artifacts — no silent duplicate."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))  # only idx0 missing
    done_id = units[6].unit_id  # a completed cell
    with pytest.raises(MatrixLaunchError, match="already have artifacts"):
        _refill(repo, store, [units[0].unit_id, done_id])


def test_refill_refuses_when_a_missing_cell_is_undeclared(tmp_path):
    """--cells must name the FULL missing set (explicit contract)."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))  # 2 missing
    with pytest.raises(MatrixLaunchError, match="ALSO missing but not in --cells"):
        _refill(repo, store, [units[0].unit_id])  # names only 1 of 2


def test_refill_refuses_when_matrix_changed(tmp_path):
    """Invariant 2: the doc's rounds changed since launch → refuse (unit_id does
    not encode rounds, so this would otherwise silently mix provenance)."""
    repo = _repo(tmp_path)  # DOC has rounds: 50
    prior = _doc_units(rounds=40)
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", prior, done_ids=_all_but(prior, {0}))
    with pytest.raises(MatrixLaunchError, match="changed since launch"):
        _refill(repo, store, None)


def test_refill_refuses_bad_cell_token(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))
    with pytest.raises(MatrixLaunchError, match="no cell at array_index 99"):
        _refill(repo, store, ["99"])
    with pytest.raises(MatrixLaunchError, match="no cell with unit_id"):
        _refill(repo, store, ["not_a_real_unit"])


# --------------------------------------------------------------------------
# Provenance (invariant 3) + terminal-array (invariant 5)
# --------------------------------------------------------------------------

def test_refill_refuses_on_methodology_change(tmp_path):
    repo = _repo(tmp_path)  # DOC methodology v1.9
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"methodology_version": "v1.8"})
    with pytest.raises(MatrixLaunchError, match="methodology"):
        _refill(repo, store, None)


def test_refill_git_sha_change_appends_prior_launch_and_checks_terminal(tmp_path):
    """git_sha drift with SAME methodology = the normal bugfix-rebuild refill:
    proceed, append the prior launch to the record's prior_launches chain, tag
    the change — and verify the prior array is terminal first (invariant 5)."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"git_sha": "0ldsha00"})  # HEAD is abc1234
    batch = FakeBatchSubmitter(array_terminal=True)
    out, _, mlflow, _ = _refill(repo, store, None, batch=batch)
    assert batch.array_terminal_queries == ["arr-orig"]  # invariant 5 ran
    record = json.loads(store.get_bytes(out["refill_record_key"]))
    assert record["meta"]["prior_launches"][0]["git_sha"] == "0ldsha00"
    assert out["provenance_change"] == "git_sha 0ldsha00 -> abc1234"
    tagged = {c.args[1]: c.args[2] for c in mlflow.set_tag.call_args_list}
    assert tagged["refill_r2_provenance_change"] == "git_sha 0ldsha00 -> abc1234"


def test_refill_refuses_provenance_change_when_prior_array_not_terminal(tmp_path):
    """Invariant 5: a provenance-changing refill while the prior array is still
    live could let an old child win the done-marker race under the old image."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"image_digest": "sha256:oldimage"})  # provenance change
    batch = FakeBatchSubmitter(array_terminal=False)
    with pytest.raises(MatrixLaunchError, match="not terminal"):
        _refill(repo, store, None, batch=batch)
    assert batch.calls == []  # nothing submitted


def test_refill_refuses_provenance_change_when_prior_array_id_unknown(tmp_path):
    """A legacy manifest without array_job_id cannot be verified terminal — a
    provenance-changing refill refuses rather than risk the race."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"git_sha": "0ldsha00", "array_job_id": None})
    with pytest.raises(MatrixLaunchError, match="no array_job_id is recorded"):
        _refill(repo, store, None)


def _write_prior_refill_record(store, exp_id, serial, *, array_job_id):
    key = f"sweeps/{exp_id}/refills/{serial}/manifest.json"
    store.put_bytes(key, json.dumps({"exp_id": exp_id, "meta": {
        "serial": serial, "array_job_id": array_job_id}, "units": []}).encode())


def _git_tags_taken(*taken):
    """A git mock where the named tags already exist (so the next refill mints the
    next free serial)."""
    g = MagicMock()
    g.working_tree_clean.return_value = True
    g.head_sha.return_value = "abc1234"
    g.tag_target_sha.side_effect = lambda _repo, name: "0ldsha00" if name in taken else None
    return g


def test_refill_repoints_parent_batch_array_job_id_to_live_array(tmp_path):
    """self-heal gates reconciliation on the parent's
    batch_array_job_id. A refill must repoint it to the LIVE refill array or
    self-heal, seeing the terminal original array, could seal the live refill
    children FAILED."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    _, batch, mlflow, _ = _refill(repo, store, missing_ids)
    tagged = {c.args[1]: c.args[2] for c in mlflow.set_tag.call_args_list}
    assert tagged["batch_array_job_id"] == "fake-array-job-id"  # repointed to the refill array


def test_refill_persists_array_id_into_its_record(tmp_path):
    """each refill's array id is persisted into its record so a
    later provenance-changing refill can use it as the terminality baseline."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))
    out, *_ = _refill(repo, store, None)
    rec = json.loads(store.get_bytes(out["refill_record_key"]))
    assert rec["meta"]["array_job_id"] == "fake-array-job-id"


def test_refill_provenance_check_uses_latest_refill_array_not_original(tmp_path):
    """on a 2nd+ provenance-changing refill the terminality
    baseline is the most recent refill's array (still possibly running old
    provenance), NOT the always-terminal original manifest array."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"git_sha": "0ldsha00"})  # provenance change vs HEAD
    _write_prior_refill_record(store, "EXP-005", "r2", array_job_id="arr-r2")
    batch = FakeBatchSubmitter(array_terminal=True)
    _refill(repo, store, None, batch=batch, git=_git_tags_taken("exp/EXP-005", "exp/EXP-005.r2"))
    assert batch.array_terminal_queries == ["arr-r2"]  # latest refill array, not arr-orig


def test_refill_refuses_when_latest_refill_array_not_terminal(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"git_sha": "0ldsha00"})
    _write_prior_refill_record(store, "EXP-005", "r2", array_job_id="arr-r2")
    batch = FakeBatchSubmitter(array_terminal=False)
    with pytest.raises(MatrixLaunchError, match="arr-r2.*not terminal"):
        _refill(repo, store, None, batch=batch)


def test_refill_post_submit_bookkeeping_failure_does_not_fail_launch(tmp_path):
    """a post-submit MLflow tag failure must NOT surface as a
    failed launch — the array is already live, and a retry would submit a second
    concurrent array. Return success carrying the warnings instead."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))
    mlflow = _mlflow()
    mlflow.set_tag.side_effect = RuntimeError("mlflow down")
    batch = FakeBatchSubmitter()
    out, *_ = _refill(repo, store, None, batch=batch, mlflow=mlflow)
    assert out["array_job_id"] == "fake-array-job-id"   # launch succeeded
    assert out["bookkeeping_warnings"]                    # failure surfaced, not raised
    assert batch.calls and batch.calls[0]["size"] == 8   # array was submitted exactly once


def test_refill_same_provenance_does_not_check_terminal(tmp_path):
    """Identical provenance (Spot retry of the same code/image) has no
    old-image race, so the terminal-array check is skipped."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))
    batch = FakeBatchSubmitter()
    _refill(repo, store, None, batch=batch)
    assert batch.array_terminal_queries == []  # not consulted


# --------------------------------------------------------------------------
# Parent-run linkage (d) — legacy fallback
# --------------------------------------------------------------------------

def test_refill_legacy_manifest_mints_cross_linked_parent_and_warns(tmp_path, capsys):
    """A manifest predating GWU-41 (no parent_run_id) mints a NEW cross-linked
    refill parent run and warns loudly; children nest under it."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}),
                   meta_overrides={"parent_run_id": None})
    out, batch, mlflow, _ = _refill(repo, store, None)
    assert out["minted_parent"] is True
    assert out["parent_run_id"] == "refill-parent-1"
    assert batch.calls[0]["environment"]["PRAXIS_PARENT_RUN_ID"] == "refill-parent-1"
    mlflow.create_run.assert_called_once()
    tags = mlflow.create_run.call_args[1]["tags"]
    assert tags["refill_of_exp"] == "EXP-005"
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------
# GWU-48 guard applies to refills too
# --------------------------------------------------------------------------

def test_refill_refuses_on_job_def_image_mismatch(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0}))
    batch = FakeBatchSubmitter(job_def_image="repo@sha256:oldimage")
    with pytest.raises(MatrixLaunchError, match="does not match the requested"):
        _refill(repo, store, None, batch=batch)
    assert batch.calls == []


# --------------------------------------------------------------------------
# Rollback symmetry
# --------------------------------------------------------------------------

def test_refill_rolls_back_record_and_tag_on_push_failure(tmp_path):
    """A tag/push failure deletes the refill record AND the tag this refill
    created; the original manifest is untouched throughout."""
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    git = _git()
    git.push_with_tags.side_effect = RuntimeError("push rejected")
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    with pytest.raises(MatrixLaunchError, match="tag/push failed"):
        refill_matrix(repo, "EXP-005", missing_ids, image_digest="sha256:deadbeef",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(),
                      _git=git, _no_push=False)
    git.delete_local_tag.assert_called_once()
    assert store.head("sweeps/EXP-005/refills/r2/manifest.json") is False  # record rolled back


def test_refill_rollback_on_submit_failure_keeps_pushed_tag_but_clears_record(tmp_path):
    repo = _repo(tmp_path)
    units = _doc_units()
    store = InMemoryObjectStore()
    _seed_launched(store, "EXP-005", units, done_ids=_all_but(units, {0, 6}))
    batch = MagicMock()
    batch.job_definition_image.return_value = None
    batch.submit_array.side_effect = RuntimeError("batch down")
    missing_ids = [u.unit_id for u in units if u.array_index in {0, 6}]
    with pytest.raises(MatrixLaunchError, match="Batch submit failed"):
        refill_matrix(repo, "EXP-005", missing_ids, image_digest="sha256:deadbeef",
                      container_tracking_uri="http://10.0.0.10:5000",
                      _store=store, _batch=batch, _mlflow=_mlflow(), _git=_git(), _no_push=True)
    # pushed tag stays as the record of the attempt; refill record is cleared
    assert store.head("sweeps/EXP-005/refills/r2/manifest.json") is False

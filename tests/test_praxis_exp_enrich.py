"""Tests for praxis_exp/enrich.py — the ``praxis exp enrich`` S3 backfill.

Uses the DI-fake pattern (InMemoryObjectStore + a recording client) so no
network / AWS / MLflow server is touched."""
import json
from pathlib import Path

from praxis_exp import storage
from praxis_exp.enrich import enrich_experiment
from praxis_exp.manifest import write_manifest
from praxis_exp.units import expand_matrix

_DOC = """---
exp_id: EXP-005
slug: h2-dev-sweep
hypothesis: Krum vs TrustScore under RMC
methodology_version: v1.17
matrix:
  defenses: [Krum, TrustScore]
  scenarios: [S0, S4]
  seeds: [42]
  mode: persistent_optimizer
  max_per_client: 2000000
  rounds: 50
batch:
  job_queue: q
  job_definition: jd
---
body
"""


def _units():
    return expand_matrix(
        ["Krum", "TrustScore"], ["S0", "S4"], [42], "persistent_optimizer", 2_000_000, 50
    )


def _build_repo(tmp_path):
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-005-h2-dev-sweep.md").write_text(_DOC)
    return tmp_path


def _result_json(return_code=0, trajectory=None):
    return json.dumps({
        "return_code": return_code,
        "elapsed_seconds": 30.0,
        "final_accuracy": 0.97, "final_f1": 0.95, "mean_accuracy": 0.96,
        "trajectory": trajectory if trajectory is not None else [
            {"round": 1, "accuracy": 0.90, "f1": 0.90, "loss": 0.2},
            {"round": 2, "accuracy": 0.97, "f1": 0.95, "loss": 0.1},
        ],
        "provenance": {"cs_model_path": ""},
    }).encode()


def _seed(store, units, done_units, meta=None):
    write_manifest(store, "EXP-005", units,
                   meta=meta or {"methodology_version": "v1.17", "image_digest": "sha256:abc"})
    for u in done_units:
        store.put_bytes(storage.result_key("EXP-005", u.unit_id), _result_json())
        store.put_bytes(storage.signal_key("EXP-005", u.unit_id), b"{}\n")
        store.put_bytes(storage.marker_key("EXP-005", u.unit_id), b"")


class _RunObj:
    """Minimal Run stand-in for the H1 skip-complete status check (search_runs),
    exposing the ``.info.status`` / ``.info.run_id`` / ``.data.tags`` / ``.data.metrics``
    shape the real PraxisMlflowClient.search_runs passthrough returns."""

    def __init__(self, run_id, status, tags, metrics=None):
        self.info = type("_Info", (), {"run_id": run_id, "status": status})()
        self.data = type(
            "_Data", (), {"tags": dict(tags), "metrics": dict(metrics or {})})()


class _FakeEnrichClient:
    """Records every MLflow call; simulates find/create for idempotency. Each
    unit_id maps to a LIST of run ids (oldest first) so the duplicate-run
    (reclaim + retry) path is exercisable."""

    def __init__(self, existing=None):
        # unit_id -> list[run_id]. `existing` values may be a single id or a list.
        self.runs = {}
        for uid, rid in (existing or {}).items():
            self.runs[uid] = list(rid) if isinstance(rid, list) else [rid]
        self.created = []
        self.tags = {}
        self.params = {}
        self.metrics = {}
        self.artifacts = {}
        self.inputs = {}
        self.deleted = {}
        self.terminated = {}
        self.statuses = {}       # run_id -> MLflow status (for the skip-complete check)
        self.run_metrics = {}    # run_id -> {metric_key: value} (skip-complete final-metric check)
        self._next = 0

    def get_or_create_experiment(self, name):
        return f"exp-{name}"

    def find_parent_run(self, experiment_id, exp_id):
        return None  # tests don't model parents -> unscoped lookup (all runs for a unit)

    def find_runs_by_unit(self, experiment_id, unit_id, parent_run_id=None):
        return list(self.runs.get(unit_id, []))

    def search_runs(self, experiment_ids, *, filter_string="", order_by=None,
                    max_results=1000):
        """H1 skip-complete check: return Run OBJECTS for the unit_id in the filter
        (oldest-first, mirroring find_runs_by_unit's order_by start_time ASC), with
        status from ``self.statuses`` (default RUNNING) and the recorded tags."""
        import re
        m = re.search(r"tags\.unit_id = '([^']*)'", filter_string)
        uid = m.group(1) if m else None
        return [_RunObj(rid, self.statuses.get(rid, "RUNNING"), self.tags.get(rid, {}),
                        self.run_metrics.get(rid, {}))
                for rid in self.runs.get(uid, [])]

    def create_run(self, experiment_id, tags):
        self._next += 1
        rid = f"run-{self._next}"
        self.runs.setdefault(tags.get("unit_id"), []).append(rid)
        self.created.append(rid)
        self.tags.setdefault(rid, {}).update(tags)
        return rid

    def log_params(self, run_id, params):
        self.params.setdefault(run_id, {}).update(params)

    def log_metric(self, run_id, key, value, step=0):
        self.metrics.setdefault(run_id, []).append((key, value, step))
        # Mirror the materialized run view (search_runs reads run_metrics): a logged
        # metric becomes visible on the run, as real MLflow does — so a run backfilled
        # in one pass presents its final_f1 to the skip-complete check on the NEXT pass
        # (the round-trip). run_metrics keeps the latest value per key (a dict);
        # self.metrics stays the full append-only call log used to prove (non-)re-log.
        self.run_metrics.setdefault(run_id, {})[key] = value

    def set_tag(self, run_id, key, value):
        self.tags.setdefault(run_id, {})[key] = value

    def delete_tag(self, run_id, key):
        self.tags.get(run_id, {}).pop(key, None)

    def log_artifact(self, run_id, local_path):
        self.artifacts.setdefault(run_id, []).append(Path(local_path).name)

    def log_input(self, run_id, dataset, context="training"):
        self.inputs.setdefault(run_id, []).append((getattr(dataset, "name", None), context))

    def delete_artifact(self, run_id, path):
        self.deleted.setdefault(run_id, []).append(path)

    def set_terminated(self, run_id, status="FINISHED"):
        self.terminated[run_id] = status
        self.statuses[run_id] = status


def test_enrich_reconstructs_completed_units(tmp_path):
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_enriched"] == 4
    assert out["units_reconciled"] == 0 and out["units_skipped"] == 0
    assert len(client.created) == 4
    assert set(client.terminated.values()) == {"FINISHED"}

    rid = client.created[0]
    # params (inputs) vs tags (metadata) disjoint
    assert set(client.params[rid]) & set(client.tags[rid]) == set()
    assert client.params[rid]["defense"] in ("Krum", "TrustScore")
    assert client.tags[rid]["defense_token"] in ("krum", "trustscore")
    metric_keys = {m[0] for m in client.metrics[rid]}
    assert {"final_accuracy", "final_f1", "rounds_completed", "wall_clock_sec"} <= metric_keys
    # per-round metrics carry the round step
    assert any(m[0] == "accuracy" and m[2] == 2 for m in client.metrics[rid])
    assert client.tags[rid]["criteria_ok"] == "true"
    assert client.tags[rid]["unit_status"] == "done"
    assert client.tags[rid]["s3_result_uri"].startswith("s3://b/")
    assert client.inputs[rid][0] == ("edge_full_20_rmc", "training")
    names = client.artifacts[rid]
    assert any(n.endswith(".json") for n in names)         # result.json still copied
    assert not any(n.endswith(".jsonl") for n in names)    # signal NO LONGER copied
    # signal is now referenced as a dataset-by-source input (context "signal")
    assert any(ctx == "signal" and name and name.startswith("signal_")
               for name, ctx in client.inputs[rid])


def test_enrich_migrates_legacy_signal_artifact_to_dataset(tmp_path):
    """ migration: enrich best-effort deletes any pre-existing
    signal.jsonl artifact copy (so re-enrich removes the duplicate, not just
    stops future copies) and attaches the signal dataset-by-source instead."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()
    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    rid = client.created[0]
    assert any(p.endswith(".jsonl") for p in client.deleted.get(rid, []))


def test_enrich_sets_note_content_with_rmc_params_and_verdict(tmp_path):
    """: the backfill path sets mlflow.note.content (RMC params +
    gate verdict), parity with the live post_persist path."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()
    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    rid = client.created[0]
    note = client.tags[rid]["mlflow.note.content"]
    assert "Scenario:" in note and "Mode:" in note and "criteria_ok:" in note


def test_enrich_is_idempotent_no_duplicate_runs(tmp_path):
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()

    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    created_first = list(client.created)
    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert client.created == created_first  # re-run reuses runs, creates none


def test_enrich_reconciles_zombie_run_without_result(tmp_path):
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # only unit 0 completed
    zombie = units[1]                 # unit 1 has a run but no durable result
    client = _FakeEnrichClient(existing={zombie.unit_id: "zombie-run"})

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_enriched"] == 1
    assert out["units_reconciled"] == 1
    assert out["units_skipped"] == 2  # units 2 & 3: no result, no run
    assert client.terminated["zombie-run"] == "FAILED"
    assert client.tags["zombie-run"]["unit_status"] == "reconciled_failed"
    assert "reclaim_reason" in client.tags["zombie-run"]


def test_enrich_marks_criteria_false_for_unclean_result(tmp_path):
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    write_manifest(store, "EXP-005", units,
                   meta={"methodology_version": "v1.17", "image_digest": "x"})
    u = units[0]
    # A COMMITTED unit (done-marker present) whose result fails the clean-run
    # criteria via an empty trajectory -> enriched FINISHED but criteria_ok=false.
    store.put_bytes(storage.result_key("EXP-005", u.unit_id),
                    json.dumps({"return_code": 0, "trajectory": []}).encode())
    store.put_bytes(storage.signal_key("EXP-005", u.unit_id), b"{}\n")
    store.put_bytes(storage.marker_key("EXP-005", u.unit_id), b"")
    client = _FakeEnrichClient()

    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    rid = client.created[0]  # one run minted for the committed unit
    assert client.tags[rid]["criteria_ok"] == "false"
    assert client.terminated[rid] == "FINISHED"


def test_enrich_reconciles_duplicate_runs_keeping_one_finished(tmp_path):
    """A reclaimed+retried unit has TWO runs with the same unit_id; enrich marks
    the newest FINISHED with the durable result and reconciles the older
    duplicate FAILED."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # unit 0 committed (durable result present)
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: ["zombie-old", "retry-new"]})

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_enriched"] == 1
    assert out["duplicate_runs_reconciled"] == 1
    assert client.created == []  # both runs reused, none minted
    assert client.terminated["retry-new"] == "FINISHED"      # newest carries result
    assert client.terminated["zombie-old"] == "FAILED"       # older duplicate reconciled
    assert client.tags["zombie-old"]["unit_status"] == "reconciled_failed"


def test_enrich_strips_legacy_input_tags(tmp_path):
    """Reusing an OLD-entrypoint run: its input tags (config/scenario/seed/mode/
    dataset) are deleted so the recovered run honours param/tag isolation — the
    inputs live as params instead."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: "old-run"})
    client.tags["old-run"] = {  # legacy tags the old entrypoint left behind
        "unit_id": u0.unit_id, "config": u0.config, "scenario": u0.scenario,
        "seed": str(u0.seed), "mode": u0.mode, "dataset": "edge_full_20_rmc",
    }

    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    remaining = client.tags["old-run"]
    for legacy in ("config", "scenario", "seed", "mode", "dataset"):
        assert legacy not in remaining
    assert client.params["old-run"]["defense"] == u0.config
    assert client.params["old-run"]["scenario"] == u0.scenario


def test_enrich_creates_run_with_parent_linkage(tmp_path):
    """A minted backfill run carries mlflow.parentRunId so the next enrich's
    parent-scoped lookup finds it instead of minting another duplicate outside
    the launch hierarchy."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # committed, but no preexisting MLflow runs

    class _WithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

    client = _WithParent()
    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert client.created  # runs were minted
    for rid in client.created:
        assert client.tags[rid]["mlflow.parentRunId"] == "parent-XYZ"


def test_enrich_seals_parent_run_finished(tmp_path):
    """The launch parent run is created RUNNING at launch and the array children
    never own its lifecycle, so without this it lingers RUNNING forever (the Runs
    tab shows an open-ended parent). enrich seals it FINISHED after backfilling."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)

    class _WithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

    client = _WithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert client.terminated["parent-XYZ"] == "FINISHED"
    assert out["parent_run_id"] == "parent-XYZ"


def test_enrich_reconciles_refilled_sweep_before_sealing(tmp_path):
    """: when a refilled sweep is now complete, the enrich seal path clears
    the finalizer's stale sweep_incomplete/missing_cells (preserving
    refill_history) so the sealed parent does not claim incomplete forever."""
    import json
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # all cells complete (refill landed)
    # a durable refill contract record from the recovery launch
    store.put_bytes(
        "sweeps/EXP-005/refills/r2/manifest.json",
        json.dumps({"exp_id": "EXP-005", "meta": {
            "serial": "r2", "refilled_cells": [units[0].unit_id],
            "launched_at": "2026-07-24T00:00:00Z", "git_sha": "abc1234"}, "units": []}).encode(),
    )

    class _WithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

    client = _WithParent()
    enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    t = client.tags["parent-XYZ"]
    assert t["sweep_incomplete"] == "false"      # reconciled, not stale-true
    assert t["missing_cells"] == ""
    assert json.loads(t["refill_history"])[0]["serial"] == "r2"
    assert client.terminated["parent-XYZ"] == "FINISHED"


def test_enrich_no_parent_seal_when_parent_absent(tmp_path):
    """When no parent run is found (unscoped fallback), there is nothing to seal —
    enrich must not fabricate or crash on parent housekeeping."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()  # find_parent_run -> None

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["parent_run_id"] is None
    # only child runs were terminated; no phantom parent id sealed
    assert set(client.terminated) == set(client.created)


def test_enrich_continues_after_per_unit_failure(tmp_path):
    """One unit's MLflow failure (e.g. an immutable-param conflict) must not
    abort the backfill — later units are still enriched and the failure is
    counted."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # all 4 committed (done-marker present)
    fail_uid = units[0].unit_id

    class _FailFirst(_FakeEnrichClient):
        def log_params(self, run_id, params):
            if self.tags.get(run_id, {}).get("unit_id") == fail_uid:
                raise RuntimeError("immutable param conflict")
            super().log_params(run_id, params)

    client = _FailFirst()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_failed"] == 1
    assert out["units_enriched"] == 3  # the other three still enriched
    assert sum(1 for s in client.terminated.values() if s == "FINISHED") == 3


def test_enrich_partial_failure_leaves_parent_open(tmp_path):
    """a partial repair failure (>=1 unit) must leave the launch
    parent run OPEN — NOT sealed FINISHED. A FINISHED parent claims 'reconciliation
    complete' and drops the sweep out of the reaper's open-parent scan forever,
    PERMANENTLY stranding a transient mid-repair failure (paradoxically worse than a
    full enrich crash, which leaves the parent open). enrich is idempotent, so leaving
    the parent open lets the reaper (or a CLI re-run) heal it."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # all 4 committed
    fail_uid = units[0].unit_id

    class _FailFirstWithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

        def log_params(self, run_id, params):
            if self.tags.get(run_id, {}).get("unit_id") == fail_uid:
                raise RuntimeError("immutable param conflict")
            super().log_params(run_id, params)

    client = _FailFirstWithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_failed"] == 1
    assert out["parent_run_id"] == "parent-XYZ"
    assert "parent-XYZ" not in client.terminated  # parent LEFT OPEN while a unit failed


def test_enrich_seals_parent_when_all_units_repaired(tmp_path):
    """With zero failures the parent is STILL sealed
    FINISHED (the truthful 'reconciliation complete' claim)."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)

    class _WithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

    client = _WithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert out["units_failed"] == 0
    assert out["parent_seal_failed"] is False              # clean seal -> not flagged
    assert client.terminated["parent-XYZ"] == "FINISHED"  # clean sweep -> parent sealed


def test_enrich_surfaces_parent_seal_failure(tmp_path):
    """the parent seal (set_terminated FINISHED) still swallows a
    transient failure so a CLI enrich never crashes on parent housekeeping — but the
    outcome is now SURFACED via summary ``parent_seal_failed=True`` (units_failed stays
    0), so the Lambda paths can apply the backstoppability doctrine. A swallowed seal on
    a tagless parent would otherwise leave it RUNNING forever, reaper unable to retry."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # all units repair cleanly

    class _FailSealWithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

        def set_terminated(self, run_id, status="FINISHED"):
            if run_id == "parent-XYZ":
                raise RuntimeError("transient MLflow seal failure")
            super().set_terminated(run_id, status)

    client = _FailSealWithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)  # must NOT crash
    assert out["units_failed"] == 0
    assert out["parent_seal_failed"] is True               # seal raised -> surfaced
    assert "parent-XYZ" not in client.terminated           # parent NOT sealed


def test_enrich_counts_unit_failed_when_artifact_upload_fails(tmp_path):
    """A result.json artifact upload failure is a
    REPAIR-step failure — its internal swallow inside _log_artifacts is REMOVED, so it
    propagates to the per-unit handler (units_failed += 1) and leaves the parent OPEN.
    Previously it was swallowed, failed stayed 0, and the sealed parent dropped the
    sweep from the reaper scan with the artifact permanently missing."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)  # all 4 committed
    fail_uid = units[0].unit_id

    class _FailArtifactWithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

        def log_artifact(self, run_id, local_path):
            if self.tags.get(run_id, {}).get("unit_id") == fail_uid:
                raise RuntimeError("result artifact upload failed")
            super().log_artifact(run_id, local_path)

    client = _FailArtifactWithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert out["units_failed"] == 1
    assert out["units_enriched"] == 3
    assert "parent-XYZ" not in client.terminated  # repair-step failure -> parent left OPEN


def test_enrich_counts_unit_failed_when_dataset_input_fails(tmp_path):
    """A dataset/signal log_input failure is a
    REPAIR-step failure — its swallow inside _log_dataset_input/_log_signal_dataset is
    REMOVED, so it propagates to units_failed and leaves the parent OPEN."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    fail_uid = units[0].unit_id

    class _FailInputWithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

        def log_input(self, run_id, dataset, context="training"):
            if self.tags.get(run_id, {}).get("unit_id") == fail_uid:
                raise RuntimeError("log_input failed")
            super().log_input(run_id, dataset, context=context)

    client = _FailInputWithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert out["units_failed"] == 1
    assert "parent-XYZ" not in client.terminated  # repair-step failure -> parent left OPEN


def test_enrich_cosmetic_cleanup_failure_still_repairs(tmp_path):
    """Cosmetic cleanup stays best-effort — a
    delete_artifact (legacy signal.jsonl) or delete_tag (legacy input tags) failure
    must NOT count the unit failed, since it only removes redundant data. The unit is
    still fully enriched and the clean parent is sealed."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)

    class _FailCleanupWithParent(_FakeEnrichClient):
        def find_parent_run(self, experiment_id, exp_id):
            return "parent-XYZ"

        def delete_artifact(self, run_id, path):
            raise RuntimeError("delete_artifact unavailable")

        def delete_tag(self, run_id, key):
            raise RuntimeError("delete_tag unavailable")

    client = _FailCleanupWithParent()
    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert out["units_failed"] == 0                         # cosmetic failures tolerated
    assert out["units_enriched"] == 4
    assert client.terminated["parent-XYZ"] == "FINISHED"    # clean sweep -> parent sealed


def test_enrich_uses_passed_experiment_id_without_design_doc(tmp_path):
    """: the finalizer/reaper pass the parent run's experiment_id, so
    enrich skips _experiment_name -> _find_design_doc entirely. A sweep whose
    design doc was never baked (the acceptance EXP + every future EXP) still heals
    instead of raising from the missing docs/experiments doc (design § 5.4)."""
    # Deliberately NO docs/experiments/ dir: _find_design_doc would raise if hit.
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, units)
    client = _FakeEnrichClient()

    out = enrich_experiment(
        tmp_path, "EXP-005", bucket="b",
        experiment_id="exp-passed-123", _store=store, _client=client,
    )

    assert out["experiment_id"] == "exp-passed-123"  # passed id used verbatim
    assert out["units_enriched"] == 4                 # ran to completion, no EnrichError


# ---------------------------------------------------------------------------
# H1 skip-complete fast path (authorized design-§11 deviation, Erik 2026-07-16)
# ---------------------------------------------------------------------------

def test_enrich_skip_complete_skips_relog_for_committed_finished_unit(tmp_path):
    """H1: with skip_complete=True, a unit whose PRIMARY run is already fully
    logged in-container (FINISHED + unit_status=done) AND whose done-marker is
    present is NOT re-logged — ZERO log_metric/log_param calls for it — and it is
    counted in the new units_skipped_complete summary key. This is what lets the
    Lambda finalizer/reaper fit the hard 900s ceiling at 100-unit×50-round scale."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # only unit 0 committed (done-marker + result)
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: "done-run"})
    client.statuses["done-run"] = "FINISHED"                       # already logged in-container
    client.tags["done-run"] = {"unit_id": u0.unit_id, "unit_status": "done",
                               "live_enrichment": "complete"}       # full in-container enrichment
    client.run_metrics["done-run"] = {"final_f1": 0.95}             # canonical final metric present

    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)

    assert out["units_skipped_complete"] == 1
    assert out["units_enriched"] == 0            # primary NOT re-enriched
    assert "done-run" not in client.metrics      # ZERO metric re-log
    assert "done-run" not in client.params       # ZERO param re-log


def test_enrich_skip_complete_still_reconciles_zombie_dup(tmp_path):
    """H1 CRITICAL invariant: skipping the primary's re-log must NOT skip zombie
    reconciliation — a reclaimed unit can have a healthy FINISHED primary AND a
    RUNNING zombie sibling, and sealing that residue is the finalizer's whole job.
    Under skip_complete=True the primary is skipped but the duplicate is STILL
    sealed FAILED."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # unit 0 committed (durable result present)
    u0 = units[0]
    # oldest-first: a RUNNING zombie attempt + the completing FINISHED+done primary
    client = _FakeEnrichClient(existing={u0.unit_id: ["zombie-old", "done-new"]})
    client.statuses["zombie-old"] = "RUNNING"
    client.statuses["done-new"] = "FINISHED"
    client.tags["zombie-old"] = {"unit_id": u0.unit_id}
    client.tags["done-new"] = {"unit_id": u0.unit_id, "unit_status": "done",
                               "live_enrichment": "complete"}
    client.run_metrics["done-new"] = {"final_f1": 0.95}

    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)

    assert out["units_skipped_complete"] == 1
    assert out["duplicate_runs_reconciled"] == 1
    assert client.terminated["zombie-old"] == "FAILED"                 # zombie STILL sealed
    assert client.tags["zombie-old"]["unit_status"] == "reconciled_failed"
    assert "done-new" not in client.metrics                            # healthy primary not re-logged


def test_enrich_skip_complete_default_off_relogs(tmp_path):
    """Guard: default (skip_complete=False) re-logs EXACTLY as today — even an
    already-complete unit is re-enriched (the CLI/agent/ re-migration paths
    must keep re-authoring FINISHED runs; that is why the fast path is opt-in)."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: "done-run"})
    client.statuses["done-run"] = "FINISHED"
    client.tags["done-run"] = {"unit_id": u0.unit_id, "unit_status": "done"}

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out.get("units_skipped_complete", 0) == 0
    assert out["units_enriched"] == 1        # re-enriched despite being already complete
    assert "done-run" in client.metrics      # metrics re-logged (default behaviour unchanged)


def test_enrich_skip_complete_requires_live_enrichment_marker(tmp_path):
    """FINISHED + unit_status=done is NOT sufficient to skip.
     GUARANTEES unit_status=done for a committed unit even when its
    post-persist decoration partially failed (and the SIGTERM-mid-decoration path
    sets done too), so such a FINISHED+done run can be missing metrics/tags. Without
    the live_enrichment=complete marker the unit is re-logged — self-heal must be
    able to repair exactly these partially-decorated / pre-img-v2 runs."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: "done-run"})
    client.statuses["done-run"] = "FINISHED"
    client.tags["done-run"] = {"unit_id": u0.unit_id, "unit_status": "done"}  # marker ABSENT
    client.run_metrics["done-run"] = {"final_f1": 0.95}

    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)

    assert out["units_skipped_complete"] == 0
    assert out["units_enriched"] == 1        # re-logged despite FINISHED + unit_status=done
    assert "done-run" in client.metrics


def test_enrich_skip_complete_requires_final_metric(tmp_path):
    """The marker is present but the canonical
    final metric (final_f1) is absent from the run's metrics -> NOT skipped. Guards
    the live per-round metric stream, which no flag can track."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])
    u0 = units[0]
    client = _FakeEnrichClient(existing={u0.unit_id: "done-run"})
    client.statuses["done-run"] = "FINISHED"
    client.tags["done-run"] = {"unit_id": u0.unit_id, "unit_status": "done",
                               "live_enrichment": "complete"}
    client.run_metrics["done-run"] = {}       # marker present but NO final_f1 metric

    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)

    assert out["units_skipped_complete"] == 0
    assert out["units_enriched"] == 1        # re-logged: final metric missing -> fail-safe re-log
    assert "done-run" in client.metrics


def test_enrich_treats_result_without_marker_as_uncommitted(tmp_path):
    """persist_unit writes result+signal BEFORE the done-marker (the commit
    point). A crash in that window leaves a readable result with no marker;
    enrich must NOT seal it FINISHED — it reconciles a preexisting run to FAILED
    or skips, never fabricating a completed run."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    write_manifest(store, "EXP-005", units,
                   meta={"methodology_version": "v1.17", "image_digest": "x"})
    # unit 0: result+signal but NO marker, WITH a preexisting run -> reconcile FAILED
    u0 = units[0]
    store.put_bytes(storage.result_key("EXP-005", u0.unit_id), _result_json())
    store.put_bytes(storage.signal_key("EXP-005", u0.unit_id), b"{}\n")
    # unit 1: result+signal but NO marker, no run -> skipped (no run minted)
    u1 = units[1]
    store.put_bytes(storage.result_key("EXP-005", u1.unit_id), _result_json())
    store.put_bytes(storage.signal_key("EXP-005", u1.unit_id), b"{}\n")
    client = _FakeEnrichClient(existing={u0.unit_id: "partial-run"})

    out = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)

    assert out["units_enriched"] == 0          # no marker -> nothing sealed FINISHED
    assert out["units_reconciled"] == 1        # u0 had a run -> FAILED
    assert out["units_skipped"] == 3           # u1 (+ units 2,3): no marker, no run
    assert client.terminated["partial-run"] == "FAILED"
    assert client.created == []                 # never mints a run for an uncommitted unit
    assert u1.unit_id not in client.runs


# ---------------------------------------------------------------------------
# : skip fast path recognizes BACKFILL-completed units (not just live)
# ---------------------------------------------------------------------------

def test_enrich_skip_complete_skips_backfill_repaired_unit(tmp_path):
    """: a unit REPAIRED by the backfill path (_enrich_completed_unit) carries
    ``backfill_enrichment=complete`` as its LAST write, so a SECOND enrich pass with
    skip_complete=True recognizes it as already-complete and does NOT re-repair it.
    Without this, every reaper tick / EventBridge retry on a partial-failure sweep
    re-logs ~250 REST calls for each already-backfilled unit before reaching the
    still-failed one — the H1 non-convergence loop fixed for the live path,
    here closed for the backfill path."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # only unit 0 committed (done-marker + result)
    # No preexisting run: pass 1 MINTS + fully backfills the primary via the backfill
    # path (default skip_complete=False), stamping the marker as its last write.
    client = _FakeEnrichClient()
    first = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert first["units_enriched"] == 1
    rid = client.created[0]
    assert client.tags[rid]["backfill_enrichment"] == "complete"  # marker stamped

    # Second pass: the backfilled unit must be recognized and skipped (append-only
    # metric call log stays flat -> zero re-log).
    metric_calls_before = len(client.metrics.get(rid, []))
    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)

    assert out["units_skipped_complete"] == 1
    assert out["units_enriched"] == 0                                 # NOT re-repaired
    assert len(client.metrics.get(rid, [])) == metric_calls_before   # ZERO metric re-log


def test_enrich_backfill_marker_absent_on_midway_failure_prevents_skip(tmp_path):
    """ propagation guard: if a REPAIR step raises mid-way through the backfill
    (here the result.json upload), _enrich_completed_unit never reaches its LAST write,
    so NO ``backfill_enrichment`` marker is stamped — the partially-logged run is left
    un-skippable and a later skip_complete pass re-attempts the repair instead of
    falsely skipping it. This is exactly why the marker is safe as a bare last-write
    with no witness-gating: reaching it PROVES every repair step succeeded, and its
    placement AFTER every repair step is load-bearing (moving it earlier would stamp a
    lie here)."""
    repo = _build_repo(tmp_path)
    store = storage.InMemoryObjectStore()
    units = _units()
    _seed(store, units, [units[0]])  # only unit 0 committed
    u0 = units[0]

    class _FailArtifact(_FakeEnrichClient):
        def log_artifact(self, run_id, local_path):
            if self.tags.get(run_id, {}).get("unit_id") == u0.unit_id:
                raise RuntimeError("result artifact upload failed")
            super().log_artifact(run_id, local_path)

    client = _FailArtifact()
    first = enrich_experiment(repo, "EXP-005", bucket="b", _store=store, _client=client)
    assert first["units_failed"] == 1
    rid = client.created[0]
    assert "backfill_enrichment" not in client.tags.get(rid, {})   # marker NOT stamped
    # the marker's write is strictly AFTER unit_status=done/set_terminated, which the
    # failing upload also short-circuits — so the run is nowhere near skippable.
    assert client.tags.get(rid, {}).get("unit_status") != "done"

    # Next pass with skip_complete=True must NOT skip the partially-logged run.
    out = enrich_experiment(
        repo, "EXP-005", bucket="b", _store=store, _client=client, skip_complete=True)
    assert out["units_skipped_complete"] == 0

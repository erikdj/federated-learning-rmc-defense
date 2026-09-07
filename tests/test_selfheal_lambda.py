"""Tests for praxis_exp/selfheal_lambda.py — the EventBridge finalizer (Lane B)
and the scheduled reaper (Lane C). All mocked: no AWS, no MLflow server, no
Batch. The finalizer/reaper never write the sweep S3 namespace — they read S3
and write MLflow only — so the fakes model an object store + an MLflow client."""
import re

from praxis_exp import storage
from praxis_exp.manifest import write_manifest
from praxis_exp.units import expand_matrix


# --- fakes -----------------------------------------------------------------

class _Info:
    def __init__(self, run_id, status, experiment_id, start_time):
        self.run_id = run_id
        self.status = status
        self.experiment_id = experiment_id
        self.start_time = start_time


class _Data:
    def __init__(self, tags):
        self.tags = dict(tags)


class _Run:
    """Minimal stand-in for an mlflow Run (``.info`` + ``.data.tags``)."""

    def __init__(self, run_id, *, status="RUNNING", experiment_id="e1",
                 start_time=0, tags=None):
        self.info = _Info(run_id, status, experiment_id, start_time)
        self.data = _Data(tags or {})


def _quoted_after(filter_string, key):
    """The single-quoted value in the filter clause mentioning ``key`` (how a
    real MLflow backend keys results off the query — the fake mirrors that)."""
    m = re.search(re.escape(key) + r"[^']*'([^']*)'", filter_string)
    return m.group(1) if m else None


class _FakeClient:
    """The PraxisMlflowClient surface selfheal_lambda uses: the two raw
    passthroughs (``list_experiment_ids`` / ``search_runs``, routed off the
    filter string exactly as the real backend would) plus ``set_tag`` /
    ``set_terminated``. Also handed to the stubbed ``_enrich``, which ignores it."""

    def __init__(self, *, experiment_ids=None, parents=None, children=None):
        self._experiment_ids = experiment_ids or ["e1"]
        self._parents = list(parents or [])       # parent _Run objects
        self._children = dict(children or {})      # parent_run_id -> [child _Run]
        self.tags = {}                             # run_id -> {k: v}
        self.terminated = {}                       # run_id -> status
        self.searches = []                         # every filter_string seen

    def list_experiment_ids(self):
        return list(self._experiment_ids)

    def search_runs(self, experiment_ids, *, filter_string="", order_by=None,
                    max_results=1000):
        self.searches.append(filter_string)
        if "mlflow.parentRunId" in filter_string:
            pid = _quoted_after(filter_string, "mlflow.parentRunId")
            return list(self._children.get(pid, []))
        if "batch_array_job_id" in filter_string:
            jid = _quoted_after(filter_string, "batch_array_job_id")
            return [r for r in self._parents
                    if r.data.tags.get("batch_array_job_id") == jid]
        if "exp_id = '" in filter_string:   # exp_id EQUALITY (not the != '' open-parent scan)
            xid = _quoted_after(filter_string, "exp_id")
            return [r for r in self._parents if r.data.tags.get("exp_id") == xid]
        # open-parent scan (tags.exp_id != '')
        return list(self._parents)

    def set_tag(self, run_id, key, value):
        self.tags.setdefault(run_id, {})[key] = value

    def set_terminated(self, run_id, status="FINISHED"):
        self.terminated[run_id] = status


class _RecordingEnrich:
    """Stands in for enrich_experiment; records each call and returns a summary."""

    def __init__(self, summary=None):
        self.calls = []
        self._summary = summary or {}

    def __call__(self, repo_root, exp_id, *, bucket, experiment_id=None,
                 _store=None, _client=None, skip_complete=False):
        self.calls.append({
            "repo_root": repo_root, "exp_id": exp_id, "bucket": bucket,
            "experiment_id": experiment_id, "_store": _store, "_client": _client,
            "skip_complete": skip_complete,
        })
        summary = {
            "experiment_id": experiment_id, "parent_run_id": None,
            "units_enriched": 0, "units_reconciled": 0, "units_skipped": 0,
            "units_failed": 0, "duplicate_runs_reconciled": 0,
            "parent_seal_failed": False,
        }
        summary.update(self._summary)
        return summary


class _FakeBatch:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def describe_jobs(self, jobs):  # boto3 Batch is keyword-only -> jobs=[...]
        self.calls.append(jobs)
        if self._raises is not None:
            raise self._raises
        return self._response


def _finalizer_event(job_id, status="SUCCEEDED", *, index=None, size=2):
    detail = {"jobId": job_id, "status": status, "arrayProperties": {"size": size}}
    if index is not None:
        detail["arrayProperties"]["index"] = index
    return {"source": "aws.batch", "detail-type": "Batch Job State Change",
            "detail": detail}


def _seed_manifest(store, exp_id, defenses, done_count):
    units = expand_matrix(defenses, ["S0"], [42], "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, exp_id, units,
                   meta={"methodology_version": "v1", "image_digest": "x"})
    for u in units[:done_count]:
        store.put_bytes(storage.marker_key(exp_id, u.unit_id), b"")
    return units


# --- Task 6: finalizer handler --------------------------------------------

def test_handler_resolves_exp_id_from_parent_run():
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)  # annotation reads it (round-11)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    handler(_finalizer_event("job-1"), None,
            _enrich=enrich, _client=client, _store=store)
    assert len(enrich.calls) == 1
    assert enrich.calls[0]["exp_id"] == "EXP-005d"


def test_handler_passes_parent_run_experiment_id():
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)  # annotation reads it (round-11)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    handler(_finalizer_event("job-1"), None,
            _enrich=enrich, _client=client, _store=store)
    assert enrich.calls[0]["experiment_id"] == "exp-77"


def test_handler_falls_back_to_describe_jobs():
    from praxis_exp.selfheal_lambda import handler
    client = _FakeClient(parents=[])  # no matching parent run
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005d"},
        {"name": "PRAXIS_BUCKET", "value": "b"},
    ]}}]})
    enrich = _RecordingEnrich()
    handler(_finalizer_event("job-9", status="FAILED"), None,
            _enrich=enrich, _client=client, _store=storage.InMemoryObjectStore(),
            _batch=batch)
    assert batch.calls == [["job-9"]]           # keyword-only jobs=[jobId]
    assert enrich.calls[0]["exp_id"] == "EXP-005d"
    assert enrich.calls[0]["experiment_id"] is None  # no parent run exists AT ALL for this exp_id


def test_handler_raises_on_describe_jobs_transient_failure():
    """on the tag-less-parent fallback, a TRANSIENT DescribeJobs
    failure must RAISE (not return unresolved_exp_id as a success) so EventBridge's
    bounded async retry re-runs the finalizer with the ONLY job-id-bearing event — the
    reaper cannot backstop a parent missing the batch_array_job_id tag (it only ever
    stale-seals children)."""
    import pytest
    from praxis_exp.selfheal_lambda import handler
    client = _FakeClient(parents=[])  # no parent by the batch_array_job_id tag
    batch = _FakeBatch(raises=RuntimeError("throttled: transient DescribeJobs error"))
    enrich = _RecordingEnrich()
    with pytest.raises(RuntimeError):
        handler(_finalizer_event("job-9", status="FAILED"), None,
                _enrich=enrich, _client=client, _store=storage.InMemoryObjectStore(),
                _batch=batch)
    assert enrich.calls == []                    # never reached enrich; will be retried


def test_handler_unresolved_exp_id_is_deterministic_no_raise():
    """a SUCCESSFUL DescribeJobs whose parent job carries no
    PRAXIS_EXP_ID is DETERMINISTICALLY unresolvable — a retry cannot help — so the
    finalizer returns unresolved_exp_id (a genuine successful no-op), never raises."""
    from praxis_exp.selfheal_lambda import handler
    client = _FakeClient(parents=[])  # no parent by the batch_array_job_id tag
    # DescribeJobs succeeds but the parent job env has NO PRAXIS_EXP_ID entry
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_BUCKET", "value": "b"},
    ]}}]})
    enrich = _RecordingEnrich()
    out = handler(_finalizer_event("job-9", status="FAILED"), None,
                  _enrich=enrich, _client=client, _store=storage.InMemoryObjectStore(),
                  _batch=batch)
    assert out == {"status": "unresolved_exp_id", "job_id": "job-9"}
    assert enrich.calls == []


def test_handler_raises_on_enrich_failure_for_tagless_parent():
    """on the TAGLESS fallback path (parent lacks
    batch_array_job_id; exp_id recovered via DescribeJobs + the exp_id-tag parent
    lookup) a transient enrich failure must RE-RAISE, NOT return the honest deferred
    status — the reaper can NEVER prove terminality for a tagless parent (it only
    stale-seals children), so 'deferred to reaper' would be a false claim. Raising
    lets EventBridge redeliver the only job-id-bearing event."""
    import pytest
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=1)
    # parent tagged exp_id but NOT batch_array_job_id (the tag write that failed)
    parent = _Run("p-exp", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d"})
    client = _FakeClient(parents=[parent])
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005d"},
    ]}}]})

    def boom_enrich(repo_root, exp_id, *, bucket, experiment_id=None,
                    _store=None, _client=None, skip_complete=False):
        raise RuntimeError("transient MLflow error on the tagless path")

    with pytest.raises(RuntimeError):
        handler(_finalizer_event("job-1", status="FAILED"), None,
                _enrich=boom_enrich, _client=client, _store=store, _batch=batch)


def test_handler_primary_partial_defers_to_reaper():
    """a PARTIAL enrich (units_failed > 0) on the PRIMARY
    (job-id-tagged) path still annotates the parent (loud tags land even if retries
    exhaust) and returns the honest ``enrich_partial_deferred_to_reaper`` — truthful
    now that enrich leaves the parent OPEN, so the reaper's scan re-finds it. No raise."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"units_failed": 1, "parent_run_id": "p1"})
    out = handler(_finalizer_event("job-1"), None, _enrich=enrich, _client=client, _store=store)
    assert out["status"] == "enrich_partial_deferred_to_reaper"
    tags = client.tags["p1"]                              # annotation still landed
    assert tags["done_count"] == "2" and tags["sweep_incomplete"] == "true"
    assert tags["missing_cells"] == units[2].unit_id


def test_handler_tagless_partial_raises():
    """a PARTIAL enrich on the TAGLESS fallback path must RAISE
    (the reaper can never backstop a tagless parent) — the annotation is attempted
    BEFORE the raise so the loud tags still land."""
    import pytest
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p-exp", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d"})  # NO batch_array_job_id -> tagless
    client = _FakeClient(parents=[parent])
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005d"},
    ]}}]})
    enrich = _RecordingEnrich(summary={"units_failed": 1, "parent_run_id": "p-exp"})
    with pytest.raises(RuntimeError):
        handler(_finalizer_event("job-1", status="FAILED"), None,
                _enrich=enrich, _client=client, _store=store, _batch=batch)
    assert client.tags["p-exp"]["sweep_incomplete"] == "true"  # annotation landed before the raise


def test_handler_primary_parent_seal_failure_defers_to_reaper():
    """a swallowed parent-seal failure (summary
    parent_seal_failed=True) on the PRIMARY path defers to the reaper — the reaper
    genuinely re-finds the UNSEALED parent — returning enrich_partial_deferred_to_reaper,
    no raise. units_failed stays 0."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_seal_failed": True, "units_failed": 0,
                                       "parent_run_id": "p1"})
    out = handler(_finalizer_event("job-1"), None, _enrich=enrich, _client=client, _store=store)
    assert out["status"] == "enrich_partial_deferred_to_reaper"


def test_handler_tagless_parent_seal_failure_raises():
    """a swallowed parent-seal failure on the TAGLESS path must
    RAISE (the reaper can never retry a tagless parent's seal) — annotation is attempted
    before the raise."""
    import pytest
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p-exp", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d"})  # NO batch_array_job_id -> tagless
    client = _FakeClient(parents=[parent])
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005d"},
    ]}}]})
    enrich = _RecordingEnrich(summary={"parent_seal_failed": True, "units_failed": 0,
                                       "parent_run_id": "p-exp"})
    with pytest.raises(RuntimeError):
        handler(_finalizer_event("job-1", status="FAILED"), None,
                _enrich=enrich, _client=client, _store=store, _batch=batch)
    assert client.tags["p-exp"]["sweep_incomplete"] == "true"  # annotation landed before the raise


def test_handler_fallback_resolves_parent_by_exp_id():
    """when the batch_array_job_id tag write failed, the PRIMARY
    parent lookup misses and DescribeJobs recovers exp_id — but the parent run is
    tagged exp_id at CREATION (matrix_launch.py:297, pre-submit), so it is still
    findable by exp_id. The handler must resolve experiment_id + run_id from that
    parent so enrich runs WITH the experiment_id (never None -> _find_design_doc,
    which the Lambda image can't satisfy) and the incomplete-sweep annotation
    targets that parent's run DIRECTLY (not via the enrich summary)."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    # parent tagged exp_id but NOT batch_array_job_id (the tag write that failed)
    parent = _Run("p-exp", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d"})
    client = _FakeClient(parents=[parent])
    batch = _FakeBatch(response={"jobs": [{"container": {"environment": [
        {"name": "PRAXIS_EXP_ID", "value": "EXP-005d"},
    ]}}]})
    # summary.parent_run_id defaults None -> the annotation MUST use the resolved parent
    enrich = _RecordingEnrich()
    out = handler(_finalizer_event("job-1", status="FAILED"), None,
                  _enrich=enrich, _client=client, _store=store, _batch=batch)
    assert batch.calls == [["job-1"]]                    # primary missed -> DescribeJobs hit
    assert enrich.calls[0]["exp_id"] == "EXP-005d"
    assert enrich.calls[0]["experiment_id"] == "exp-77"  # resolved from the exp_id-tagged parent
    tags = client.tags["p-exp"]                          # annotation landed on that parent's run
    assert tags["done_count"] == "2"
    assert tags["n_units"] == "3"
    assert tags["missing_cells"] == units[2].unit_id
    assert tags["sweep_incomplete"] == "true"
    assert out["status"] == "reconciled"


def test_handler_ignores_child_events():
    from praxis_exp.selfheal_lambda import handler
    parent = _Run("p1", tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich()
    handler(_finalizer_event("job-1", index=0), None,
            _enrich=enrich, _client=client, _store=storage.InMemoryObjectStore())
    assert enrich.calls == []  # belt-and-suspenders vs the EventBridge filter


def test_handler_status_reconciled_on_success():
    """FIX-2 invariant: a SUCCESSFUL enrich keeps ``status='reconciled''``."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)  # annotation reads it (round-11)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    out = handler(_finalizer_event("job-1"), None,
                  _enrich=enrich, _client=client, _store=store)
    assert out["status"] == "reconciled"


def test_handler_passes_skip_complete_true():
    """H1: the Lambda finalizer opts into enrich's skip-complete fast path so a
    healthy 100-unit sweep fits the hard 900s ceiling (units already fully logged
    in-container are not redundantly re-logged)."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)  # annotation reads it (round-11)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    handler(_finalizer_event("job-1"), None,
            _enrich=enrich, _client=client, _store=store)
    assert enrich.calls[0]["skip_complete"] is True


def test_reaper_passes_skip_complete_true():
    """H1: the reaper (backstop trigger of the SAME enrich) also opts into the
    skip-complete fast path — one reconcile implementation, both triggers fit 900s."""
    from praxis_exp.selfheal_lambda import reaper
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    child_running = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child_running]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal
    enrich = _RecordingEnrich()
    reaper({}, None, _enrich=enrich, _client=client,
           _store=storage.InMemoryObjectStore(), _batch=batch, _now_ms=10_000)
    assert enrich.calls[0]["skip_complete"] is True


def test_handler_status_enrich_failed_deferred_to_reaper():
    """FIX-2 (M1): when enrich RAISES, the finalizer must report an HONEST status
    (``enrich_failed_deferred_to_reaper``), NOT ``reconciled`` — nothing was
    reconciled. It deliberately does NOT re-raise (a re-raise triggers an
    EventBridge retry storm); the scheduled reaper backstops the deferral."""
    from praxis_exp.selfheal_lambda import handler
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])

    def boom_enrich(repo_root, exp_id, *, bucket, experiment_id=None,
                    _store=None, _client=None):
        raise RuntimeError("transient mlflow error")

    out = handler(_finalizer_event("job-1"), None,  # must NOT raise
                  _enrich=boom_enrich, _client=client, _store=storage.InMemoryObjectStore())
    assert out["status"] == "enrich_failed_deferred_to_reaper"


# --- Task 7: loud-not-silent parent annotation ----------------------------

def test_finalizer_flags_incomplete_sweep():
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    handler(_finalizer_event("job-1"), None, _enrich=enrich, _client=client, _store=store)
    tags = client.tags["p1"]
    assert tags["done_count"] == "2"
    assert tags["n_units"] == "3"
    assert tags["missing_cells"] == units[2].unit_id  # the one without a marker
    assert tags["sweep_incomplete"] == "true"


def test_finalizer_no_incomplete_flag_when_complete():
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=3)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    handler(_finalizer_event("job-1"), None, _enrich=enrich, _client=client, _store=store)
    tags = client.tags["p1"]
    assert tags["done_count"] == "3"
    assert tags["n_units"] == "3"
    assert "missing_cells" not in tags       # complete -> absent (not "false")
    assert "sweep_incomplete" not in tags


def test_annotate_incomplete_sweep_returns_bool():
    """_annotate_incomplete_sweep returns True when the tags
    write (or there is legitimately no parent to annotate — a no-op, not a failure) and
    False when set_tag raises (so the caller can surface a lost σ-safeguard). It never
    raises."""
    from praxis_exp.selfheal_lambda import _annotate_incomplete_sweep
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)

    ok = _FakeClient(parents=[])
    assert _annotate_incomplete_sweep(ok, store, "EXP-005d", None, "p1") is True  # tags wrote
    assert ok.tags["p1"]["sweep_incomplete"] == "true"
    # no parent resolved -> legitimate no-op -> True (nothing to annotate, not a failure)
    assert _annotate_incomplete_sweep(ok, store, "EXP-005d", None, None) is True

    class _BoomClient(_FakeClient):
        def set_tag(self, run_id, key, value):
            raise RuntimeError("transient set_tag failure")

    boom = _BoomClient(parents=[])
    assert _annotate_incomplete_sweep(boom, store, "EXP-005d", None, "p1") is False  # never raises


def test_handler_raises_when_annotation_fails():
    """a CLEAN enrich seals the parent FINISHED, so if the
    incomplete-sweep annotation then fails transiently the reaper's open-parent scan
    NEVER revisits — the σ-safeguard tags (done_count/n_units/missing_cells/
    sweep_incomplete) are permanently lost, exactly the silent-incomplete-sweep failure
    GWU-45 exists to prevent. The finalizer must RAISE so the bounded EventBridge retry
    re-runs the idempotent enrich + re-attempts the annotation. The raise is AFTER
    enrich (the enrich seam WAS called)."""
    import pytest
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})

    class _AnnotateBoom(_FakeClient):
        def set_tag(self, run_id, key, value):
            if run_id == "p1":
                raise RuntimeError("transient annotation set_tag failure")
            super().set_tag(run_id, key, value)

    client = _AnnotateBoom(parents=[parent])
    enrich = _RecordingEnrich(summary={"units_failed": 0, "parent_run_id": "p1"})  # CLEAN enrich
    with pytest.raises(RuntimeError):
        handler(_finalizer_event("job-1"), None, _enrich=enrich, _client=client, _store=store)
    assert len(enrich.calls) == 1  # the raise happened AFTER enrich was called


def test_handler_sets_and_clears_inflight_marker():
    """the finalizer stamps the advisory selfheal_inflight marker
    BEFORE enrich (so a concurrent reaper defers) and clears it after a clean completion.
    Verified via the value seen AT enrich time (set) + the final tag value (cleared)."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent])
    seen = {}

    def capturing_enrich(repo_root, exp_id, *, bucket, experiment_id=None,
                         _store=None, _client=None, skip_complete=False):
        seen["inflight_at_enrich"] = _client.tags.get("p1", {}).get("selfheal_inflight")
        return {"units_failed": 0, "parent_run_id": "p1", "parent_seal_failed": False}

    out = handler(_finalizer_event("job-1"), None,
                  _enrich=capturing_enrich, _client=client, _store=store)
    # marker was SET (a timestamp) BEFORE enrich, then CLEARED after clean completion
    assert seen["inflight_at_enrich"] and seen["inflight_at_enrich"].isdigit()
    assert client.tags["p1"]["selfheal_inflight"] == ""
    assert out["status"] == "reconciled"


def test_handler_inflight_marker_failure_does_not_block():
    """the marker is advisory — a set_tag failure on it must NOT
    block or fail healing. enrich still runs and the finalizer still returns reconciled."""
    from praxis_exp.selfheal_lambda import handler
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77",
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})

    class _InflightBoom(_FakeClient):
        def set_tag(self, run_id, key, value):
            if key == "selfheal_inflight":
                raise RuntimeError("transient marker set_tag failure")
            super().set_tag(run_id, key, value)

    client = _InflightBoom(parents=[parent])
    enrich = _RecordingEnrich(summary={"units_failed": 0, "parent_run_id": "p1"})
    out = handler(_finalizer_event("job-1"), None,
                  _enrich=enrich, _client=client, _store=store)
    assert len(enrich.calls) == 1          # enrich still ran despite the marker failure
    assert out["status"] == "reconciled"   # advisory marker never blocks


# --- Task 9: scheduled reaper backstop ------------------------------------

_STALE_AGE_MS = 86_400 * 1000  # AttemptDurationSeconds (batch-stack.yaml:86)


def test_reaper_enriches_terminal_array_with_open_runs():
    from praxis_exp.selfheal_lambda import reaper
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    child_running = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child_running]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal
    enrich = _RecordingEnrich()
    reaper({}, None, _enrich=enrich, _client=client,
           _store=storage.InMemoryObjectStore(), _batch=batch, _now_ms=10_000)
    assert len(enrich.calls) == 1
    assert enrich.calls[0]["exp_id"] == "EXP-005d"
    assert enrich.calls[0]["experiment_id"] == "exp-77"  # parent's own experiment_id


def test_reaper_annotates_incomplete_sweep():
    """FIX-1 (H2): the reaper is the backstop for a MISSED finalizer event, so
    after a successful enrich it must ALSO write the loud-not-silent σ-safeguard
    tags (done_count/n_units/missing_cells/sweep_incomplete) on the parent —
    otherwise a sweep whose finalizer event was dropped never gets flagged
    incomplete (design § 5.6)."""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    child_running = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child_running]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal
    enrich = _RecordingEnrich(summary={"parent_run_id": "p1"})
    reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=10_000)
    tags = client.tags["p1"]
    assert tags["done_count"] == "2"
    assert tags["n_units"] == "3"
    assert tags["missing_cells"] == units[2].unit_id  # the one without a marker
    assert tags["sweep_incomplete"] == "true"


def test_reaper_partial_not_counted_healed():
    """when the reaper's enrich reports units_failed > 0, that
    parent is NOT counted healed — it is counted in ``parents_partial`` — and the
    reaper does NOT raise (enrich left the parent open, so the next scheduled tick
    naturally retries)."""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore", "Bulyan"], done_count=2)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    child_running = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child_running]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal
    enrich = _RecordingEnrich(summary={"units_failed": 1, "parent_run_id": "p1"})
    out = reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=10_000)
    assert out["parents_partial"] == 1
    assert out["units_healed"] == 0            # partial repair not counted healed
    assert len(enrich.calls) == 1              # enrich still attempted (no raise)


def test_reaper_parent_seal_failure_counted_partial():
    """a summary parent_seal_failed makes the reaper count the
    parent in parents_partial (not healed) — the UNSEALED parent stays in the next
    tick's scan for a natural retry, no raise."""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    child_running = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child_running]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal
    enrich = _RecordingEnrich(summary={"parent_seal_failed": True, "units_failed": 0,
                                       "parent_run_id": "p1"})
    out = reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=10_000)
    assert out["parents_partial"] == 1
    assert out["units_healed"] == 0


def test_reaper_annotation_failure_counted_partial():
    """a clean reaper enrich whose annotation then FAILS is
    counted in parents_partial (not healed), with NO raise so the remaining parents in
    the same tick are still processed. (Accepted asymmetry: a sealed parent won't be
    re-found next tick — surfaced via parents_partial + the loud log.)"""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    _seed_manifest(store, "EXP-006d", ["Krum", "TrustScore"], done_count=2)
    p_fail = _Run("p-fail", status="FINISHED", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    p_ok = _Run("p-ok", status="FINISHED", experiment_id="exp-77", start_time=1000,
                tags={"exp_id": "EXP-006d", "batch_array_job_id": "job-2"})
    child_fail = _Run("c-fail", status="RUNNING", experiment_id="exp-77", start_time=1000)
    child_ok = _Run("c-ok", status="RUNNING", experiment_id="exp-77", start_time=1000)

    class _AnnotateBoomForP(_FakeClient):
        def set_tag(self, run_id, key, value):
            if run_id == "p-fail":
                raise RuntimeError("annotation failure on p-fail")
            super().set_tag(run_id, key, value)

    client = _AnnotateBoomForP(parents=[p_fail, p_ok],
                               children={"p-fail": [child_fail], "p-ok": [child_ok]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})  # array terminal for both
    enrich = _RecordingEnrich(summary={"units_failed": 0})  # clean enrich for both
    out = reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=10_000)
    assert out["parents_partial"] == 1   # p-fail: annotation failed -> partial, not healed
    assert out["units_healed"] == 1      # p-ok healed -> loop CONTINUED past p-fail (no raise)
    assert len(enrich.calls) == 2        # both parents' enrich ran


def test_reaper_skips_parent_with_fresh_inflight_marker():
    """a parent whose advisory selfheal_inflight marker is FRESH
    (a finalizer is enriching it) is SKIPPED — no enrich — and counted in
    parents_skipped_inflight, narrowing the concurrent-enrich race."""
    from praxis_exp.selfheal_lambda import reaper, _INFLIGHT_STALE_MS
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    now = 10 * _INFLIGHT_STALE_MS
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1",
                        "selfheal_inflight": str(now - 60_000)})  # 1 min ago -> fresh
    child = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})
    enrich = _RecordingEnrich()
    out = reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=now)
    assert enrich.calls == []                        # skipped -> no enrich
    assert out["parents_skipped_inflight"] == 1
    assert out["units_healed"] == 0


def test_reaper_processes_parent_with_stale_inflight_marker():
    """a STALE selfheal_inflight marker (older than 30 min — a
    crashed invocation) must NOT deadlock healing — the reaper processes the parent."""
    from praxis_exp.selfheal_lambda import reaper, _INFLIGHT_STALE_MS
    store = storage.InMemoryObjectStore()
    _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=2)
    now = 10 * _INFLIGHT_STALE_MS
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1",
                        "selfheal_inflight": str(now - _INFLIGHT_STALE_MS - 60_000)})  # > 30 min old
    child = _Run("c1", status="RUNNING", experiment_id="exp-77", start_time=1000)
    client = _FakeClient(parents=[parent], children={"p1": [child]})
    batch = _FakeBatch(response={"jobs": [{"status": "SUCCEEDED"}]})
    enrich = _RecordingEnrich()
    out = reaper({}, None, _enrich=enrich, _client=client, _store=store, _batch=batch, _now_ms=now)
    assert len(enrich.calls) == 1                    # stale marker -> enrich runs
    assert out["parents_skipped_inflight"] == 0
    assert out["units_healed"] == 1


def test_reaper_skips_live_array():
    from praxis_exp.selfheal_lambda import reaper
    parent = _Run("p1", status="RUNNING", experiment_id="exp-77", start_time=1000,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    client = _FakeClient(parents=[parent], children={"p1": []})
    batch = _FakeBatch(response={"jobs": [{"status": "RUNNING"}]})  # array still live
    enrich = _RecordingEnrich()
    reaper({}, None, _enrich=enrich, _client=client,
           _store=storage.InMemoryObjectStore(), _batch=batch, _now_ms=10_000)
    assert enrich.calls == []
    assert client.terminated == {}  # nothing sealed on a live array


def test_reaper_seals_stale_run():
    """FIX-3 (M2) + : a stale RUNNING child whose S3 done-marker
    proves it committed is sealed **FINISHED** (unit_status=done) — NOT FAILED — with a
    truthful stale_seal_reason, and the parent gets the stale_seal_decoration_pending
    breadcrumb. The done-marker proves durable SUCCESS (the container died between the
    persist and the MLflow seal), so FAILED would permanently misrecord a successful
    unit (once terminal, the open-parent scan never revisits to self-correct)."""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=1)
    now = 5 * _STALE_AGE_MS
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=now - 10,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    stale = _Run("c1", status="RUNNING", experiment_id="exp-77",
                 start_time=now - _STALE_AGE_MS - 1000,       # older than AttemptDurationSeconds
                 tags={"unit_id": units[0].unit_id})          # committed unit (done_count=1)
    client = _FakeClient(parents=[parent], children={"p1": [stale]})
    batch = _FakeBatch(raises=RuntimeError("no batch VPC endpoint"))  # array lookup unavailable
    enrich = _RecordingEnrich()
    reaper({}, None, _enrich=enrich, _client=client,
           _store=store, _batch=batch, _now_ms=now)
    assert client.terminated["c1"] == "FINISHED"          # committed -> durable SUCCESS, not FAILED
    assert client.tags["c1"]["unit_status"] == "done"     # truthful status, not reconciled_failed
    assert "stale_seal_reason" in client.tags["c1"]       # truthful reason tag
    assert client.tags["p1"]["stale_seal_decoration_pending"] == "true"  # loud breadcrumb on parent
    assert enrich.calls == []                             # no enrich when the array lookup fails


def test_reaper_leaves_stale_run_without_done_marker():
    """FIX-3 (M2): without an S3 done-marker, age alone does NOT prove death.
    Under Lane A resume-by-unit a child's ``start_time`` is FROZEN at attempt 1
    while later attempts legitimately run, so a cumulatively-old RUNNING child may
    be a LIVE resumed unit. With no marker it is LEFT ALONE (zero false positives);
    the array-terminality enrich path / endpoint recovery heals it later."""
    from praxis_exp.selfheal_lambda import reaper
    store = storage.InMemoryObjectStore()
    units = _seed_manifest(store, "EXP-005d", ["Krum", "TrustScore"], done_count=0)  # nothing committed
    now = 5 * _STALE_AGE_MS
    parent = _Run("p1", status="FINISHED", experiment_id="exp-77", start_time=now - 10,
                  tags={"exp_id": "EXP-005d", "batch_array_job_id": "job-1"})
    stale = _Run("c1", status="RUNNING", experiment_id="exp-77",
                 start_time=now - _STALE_AGE_MS - 1000,
                 tags={"unit_id": units[0].unit_id})  # tag resolves but NO done-marker
    client = _FakeClient(parents=[parent], children={"p1": [stale]})
    batch = _FakeBatch(raises=RuntimeError("no batch VPC endpoint"))
    enrich = _RecordingEnrich()
    reaper({}, None, _enrich=enrich, _client=client,
           _store=store, _batch=batch, _now_ms=now)
    assert client.terminated == {}   # no done-marker -> live resumed attempt, left alone
    assert enrich.calls == []

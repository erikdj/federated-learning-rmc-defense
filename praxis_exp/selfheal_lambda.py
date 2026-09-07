"""AWS Lambda self-healing for the praxis fleet.

This module is two of the three self-heal layers (design § 3), with **zero agent
in the loop**:

  - ``handler``  — Layer 1, the **EventBridge finalizer**. On the array PARENT's
    terminal ``Batch Job State Change`` event it reconciles the sweep's MLflow
    record ONCE (zombies -> FAILED, parent -> FINISHED, incomplete-sweep
    annotation).
  - ``reaper``   — Layer 3, the **scheduled backstop** (Lane C). On a clock it
    reconciles any sweep whose Batch array is terminal but whose MLflow record is
    not, and seals stale RUNNING children even when the Batch API is unreachable.

Both run the SAME existing ``enrich_experiment`` — one reconcile implementation,
two triggers (a Batch terminal event; a clock). Invariants (design § 4):

  - **Never writes the sweep S3 namespace.** Self-healing reads S3 (result/signal/
    done) and writes MLflow only; the ``result -> signal -> done`` contract is
    untouched.
  - **Idempotent + best-effort.** ``enrich_experiment`` is idempotent, so a
    duplicate delivery is a harmless no-op; every internal failure is caught so a
    finalizer/reaper error never corrupts a good run. Runs are re-terminated, never
    deleted: enrich's duplicate/zombie handling seals superseded attempts FAILED,
    while the reaper's stale fallback seals a COMMITTED unit's run FINISHED (its S3
    done-marker proves durable success — FAILED would misrecord it).
  - **No refills**: a Spot-exhausted cell is made *loud* (parent
    annotation), never re-submitted.

Credentials come from the Lambda **execution role** (``PRAXIS_USE_INSTANCE_ROLE``
makes ``Config`` role-tolerant); the finalizer's primary exp_id resolution keys
off the MLflow parent run's ``batch_array_job_id`` tag and makes **no Batch API
call** (design § 5.4 / § 8), so only the rare DescribeJobs fallback and the
reaper need the Batch VPC endpoint.

**Serialization.** ``enrich_experiment``'s find-then-create
of a child run is not atomic, so two self-heal invocations enriching the SAME parent
could each mint a duplicate child (a triple-coincidence, cosmetic race — self-healing
on any later enrich via the dup-reconcile loop). MLflow offers no atomic CAS, so this
is mitigated, not locked: (1) ``ReservedConcurrentExecutions: 1`` on BOTH Lambdas
(selfheal-stack.yaml) serializes each function against itself — EventBridge
redeliveries queue and overlapping reaper ticks are impossible; (2) an ADVISORY
``selfheal_inflight`` timestamp tag on the parent run narrows the residual
cross-function overlap — the finalizer stamps it before its enrich and the reaper
SKIPS a parent whose marker is fresher than ``_INFLIGHT_STALE_MS``. Accepted residual:
a finalizer event landing mid-reaper-enrich within tag-propagation latency (advisory,
not exclusive) — cosmetic and self-healing on the next enrich.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from praxis_exp.config import Config
from praxis_exp.enrich import enrich_experiment
from praxis_exp.integrity import is_done
from praxis_exp.mlflow_client import PraxisMlflowClient
from praxis_exp.refill_reconcile import reconcile_sweep_tags
from praxis_exp.storage import S3ObjectStore

# MLflow terminal run statuses — anything else (RUNNING/SCHEDULED) is "open".
_TERMINAL_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED"})
_DEFAULT_REPO_ROOT = "/app"

# Reaper bounds (design § 5.7): scan only parent runs launched within the
# lookback window (keeps the scan cheap as experiments accumulate). A RUNNING
# child older than the Batch AttemptDurationSeconds is NOT definitively dead on
# age alone — under Lane A resume-by-unit ``start_time`` is frozen at attempt 1
# while later attempts legitimately run, so cumulative age can exceed the timeout
# on a LIVE unit. It is treated as stale only when its S3 done-marker proves the
# unit committed (see ``_seal_stale_children``).
_REAPER_LOOKBACK_MS = 7 * 24 * 60 * 60 * 1000     # 7d
_STALE_RUN_AGE_MS = 86_400 * 1000                 # AttemptDurationSeconds (batch-stack.yaml:86)

# Advisory in-flight marker. Reserved-concurrency=1 serializes
# each Lambda against ITSELF; this tag narrows the residual cross-function race (a
# finalizer event landing mid-reaper-enrich for the same parent). It is ADVISORY only —
# MLflow offers no atomic CAS, so it never guarantees mutual exclusion; every set/clear
# is best-effort. Stale window > the 15-min Lambda max runtime so a crashed invocation's
# marker can never deadlock healing.
_INFLIGHT_TAG = "selfheal_inflight"
_INFLIGHT_STALE_MS = 30 * 60 * 1000               # 30 min (> Lambda max runtime)


@dataclass(frozen=True)
class _ParentRun:
    """The self-heal view of an array-parent MLflow run."""
    run_id: str
    exp_id: Optional[str]
    experiment_id: str
    array_job_id: Optional[str] = None
    status: str = ""
    start_time: int = 0
    inflight: str = ""   # the selfheal_inflight tag value (advisory), read at scan time


# --- dependency builders (role creds + env; each overridable via a seam) ----

def _default_store() -> S3ObjectStore:
    import boto3
    cfg = Config()  # role mode makes this constructible with no AWS_PROFILE
    session = boto3.Session(region_name=cfg.aws_region)  # default chain = execution role
    return S3ObjectStore(cfg.artifact_bucket, session.client("s3"))


def _default_client() -> PraxisMlflowClient:
    cfg = Config()
    return PraxisMlflowClient(tracking_uri=cfg.tracking_uri)


def _default_batch() -> Any:
    import boto3
    return boto3.client("batch", region_name=Config().aws_region)


def _artifact_bucket() -> str:
    """The bare bucket NAME the enrich upload / S3 reads consume. The stack sets
    ``PRAXIS_ARTIFACT_BUCKET`` on the Lambda; fall back to ``Config`` (local runs)."""
    bucket = os.environ.get("PRAXIS_ARTIFACT_BUCKET")
    if bucket:
        return bucket
    try:
        return Config().artifact_bucket
    except Exception:
        return ""


def _repo_root(override: Optional[str]) -> Path:
    return Path(override or os.environ.get("PRAXIS_REPO_ROOT", _DEFAULT_REPO_ROOT))


# --- exp_id / experiment_id resolution -------------------------------------

def _find_parent_by_tag(client: Any, tag_key: str, value: str) -> Optional[_ParentRun]:
    """The array-parent MLflow run whose ``tags.<tag_key>`` equals ``value``, or
    None. Cross-experiment search (the Batch event / DescribeJobs carries no
    experiment id), newest-first, preferring a non-FAILED parent (an aborted launch
    is FAILED) and else falling back to the newest. NO Batch API call.

    Two resolution keys share this logic (design § 5.4):
      - ``batch_array_job_id`` — PRIMARY (the parent is tagged the array job id).
      - ``exp_id`` — FALLBACK when the job-id tag write failed: the parent is tagged
        ``exp_id`` at CREATION (matrix_launch.py:297, before submit), so it is still
        findable, letting the DescribeJobs-recovered exp_id resolve experiment_id +
        run_id instead of leaving them None (None would send enrich into
        ``_find_design_doc``, which the Lambda image does NOT bake)."""
    try:
        exp_ids = client.list_experiment_ids()
        runs = client.search_runs(
            exp_ids,
            filter_string=f"tags.{tag_key} = '{value}'",
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
    except Exception as e:
        print(f"[selfheal] WARN: parent-run lookup ({tag_key}={value}) failed ({e})")
        return None
    chosen = None
    for r in runs:
        if r.info.status != "FAILED":
            chosen = r
            break
    if chosen is None and runs:
        chosen = runs[0]
    if chosen is None:
        return None
    return _ParentRun(
        run_id=chosen.info.run_id,
        exp_id=chosen.data.tags.get("exp_id"),
        experiment_id=chosen.info.experiment_id,
        array_job_id=chosen.data.tags.get("batch_array_job_id"),
        status=chosen.info.status,
        start_time=chosen.info.start_time,
    )


def _find_parent_by_job_id(client: Any, job_id: str) -> Optional[_ParentRun]:
    """PRIMARY parent resolution (design § 5.4): the parent run tagged
    ``batch_array_job_id`` — NO Batch API call, so the primary path needs no Batch
    VPC route. A thin wrapper over ``_find_parent_by_tag``."""
    return _find_parent_by_tag(client, "batch_array_job_id", job_id)


def _exp_id_from_describe_jobs(batch: Any, job_id: str) -> Optional[str]:
    """FALLBACK exp_id resolution when the parent run tag is missing: read
    ``PRAXIS_EXP_ID`` from the array-parent job's container env. boto3 Batch is
    keyword-only (``jobs=[...]``); ``container.environment`` is a LIST of
    ``{name,value}`` dicts (batch.py:50-52), NOT a ``container.environment.X``
    dict path (design § 5.4).

    A ``describe_jobs`` exception is TRANSIENT and DELIBERATELY propagates — there is NO blanket try/except here — so the caller can
    re-raise and let EventBridge retry the only job-id-bearing event. Returns None
    ONLY for the two DETERMINISTIC/permanent cases where a retry cannot help: an
    empty ``jobs`` list, or no ``PRAXIS_EXP_ID`` entry in the parent job's env."""
    resp = batch.describe_jobs(jobs=[job_id])
    jobs = resp.get("jobs") or []
    if not jobs:
        return None
    env = ((jobs[0].get("container") or {}).get("environment")) or []
    for item in env:
        if item.get("name") == "PRAXIS_EXP_ID":
            return item.get("value")
    return None


# --- loud-not-silent annotation --------------------------------------------

def _annotate_incomplete_sweep(
    client: Any, store: Any, exp_id: str, summary: Optional[dict], parent_run_id: Optional[str],
) -> bool:
    """Tag the parent run ``done_count`` / ``n_units``, and when any cell lacks a
    done-marker also ``missing_cells`` + ``sweep_incomplete=true`` (design § 5.6).
    A complete sweep gets ``done_count`` / ``n_units`` only — ``missing_cells`` and
    ``sweep_incomplete`` stay ABSENT. It never re-submits a cell.

    Returns True iff the tags were written — OR there was legitimately nothing to
    annotate (no ``parent_run_id`` resolved: no parent means nothing to tag, a no-op,
    not a failure). Returns False from the guarded except. The
    never-raise contract is UNCHANGED — the bool lets the caller surface a LOST
    σ-safeguard: after a clean enrich the parent is sealed FINISHED, so a swallowed
    annotation failure would otherwise vanish (the reaper's open-parent scan never
    revisits a sealed sweep), silently losing exactly the incomplete-sweep signal
    this finalizer exists to guarantee."""
    parent_run_id = parent_run_id or (summary or {}).get("parent_run_id")
    if not parent_run_id:
        return True  # nothing to annotate (no parent) -> legitimate no-op, not a failure
    try:
        # Shared with the CLI enrich seal path: recomputes completeness
        # and, for a refilled sweep that is now complete, clears the stale
        # sweep_incomplete/missing_cells while preserving refill_history.
        reconcile_sweep_tags(client, store, exp_id, parent_run_id)
        return True
    except Exception as e:
        print(f"[selfheal] WARN: incomplete-sweep annotation failed for {exp_id}: {e}")
        return False


# --- advisory in-flight marker ---------------------

def _set_inflight(client: Any, parent_run_id: str, now_ms: int) -> None:
    """Stamp the advisory ``selfheal_inflight`` marker with ``now_ms`` so a concurrent
    self-heal invocation for the SAME parent can narrow the race window. STRICTLY
    best-effort — a tag failure must NEVER block or fail healing (the marker is
    advisory, not a lock)."""
    try:
        client.set_tag(parent_run_id, _INFLIGHT_TAG, str(now_ms))
    except Exception as e:
        print(f"[selfheal] WARN: could not set inflight marker on {parent_run_id}: {e}")


def _clear_inflight(client: Any, parent_run_id: str) -> None:
    """Clear the advisory in-flight marker (set to ""). Best-effort — see
    ``_set_inflight``."""
    try:
        client.set_tag(parent_run_id, _INFLIGHT_TAG, "")
    except Exception as e:
        print(f"[selfheal] WARN: could not clear inflight marker on {parent_run_id}: {e}")


def _inflight_fresh(tag_value: Optional[str], now_ms: int) -> bool:
    """True iff ``tag_value`` is a timestamp fresher than ``_INFLIGHT_STALE_MS`` (an
    enrich is likely in flight for this parent, so a second worker should defer). An
    empty / absent / malformed marker, or a stale timestamp, returns False (proceed) —
    a crashed invocation's stale marker must never deadlock healing."""
    if not tag_value:
        return False
    try:
        return (now_ms - int(tag_value)) < _INFLIGHT_STALE_MS
    except (TypeError, ValueError):
        return False


# --- Layer 1: EventBridge finalizer ----------------------------------------

def handler(
    event: dict, context: Any = None, *,
    _enrich: Any = None, _client: Any = None, _store: Any = None,
    _batch: Any = None, _repo_root_override: Optional[str] = None,
) -> dict:
    """Reconcile a completed Batch array's MLflow record on its terminal event.

    Fires once per array (the EventBridge rule matches the array PARENT — no
    ``arrayProperties.index`` — design § 6). Child events (index present) are
    ignored belt-and-suspenders. Idempotent: a duplicate delivery just re-runs
    the idempotent enrich.

    Transient vs permanent: on the tag-less-parent fallback a
    TRANSIENT ``DescribeJobs`` failure RE-RAISES so EventBridge retries the only
    job-id-bearing event (the reaper cannot backstop a parent missing the
    ``batch_array_job_id`` tag); a DETERMINISTIC miss (DescribeJobs OK but no
    ``PRAXIS_EXP_ID``) returns ``unresolved_exp_id`` without raising.

    The return-don't-raise rule for an ENRICH failure applies ONLY when the reaper can
    actually backstop the parent — i.e. the PRIMARY (job-id-tagged) path, which returns
    ``enrich_failed_deferred_to_reaper``. On the TAGLESS fallback
    path the reaper can never prove terminality, so an enrich failure RE-RAISES too.
    Likewise a failed incomplete-sweep ANNOTATION after a (clean) enrich RE-RAISES on both
    paths: a sealed parent has no reaper backstop for a missing
    σ-safeguard tag, so the bounded EventBridge retry is the only vehicle. Accepted
    residual: a sustained outage across all EventBridge delivery attempts still strands
    the sweep (no DLQ configured — candidate future hardening)."""
    enrich = _enrich or enrich_experiment
    detail = (event or {}).get("detail", {}) or {}
    array_props = detail.get("arrayProperties", {}) or {}

    # belt-and-suspenders vs the EventBridge filter: never finalize on a child.
    if "index" in array_props:
        print("[selfheal] finalizer: child array event (arrayProperties.index present) — ignoring")
        return {"status": "ignored_child_event"}

    job_id = detail.get("jobId")
    if not job_id:
        print("[selfheal] finalizer: event has no detail.jobId — ignoring")
        return {"status": "ignored_no_job_id"}

    client = _client if _client is not None else _default_client()
    store = _store if _store is not None else _default_store()
    bucket = _artifact_bucket()
    repo_root = _repo_root(_repo_root_override)

    # PRIMARY: parent run's batch_array_job_id tag (no Batch API call).
    parent = _find_parent_by_job_id(client, job_id)
    # tagless_fallback: the parent lacks batch_array_job_id, so the reaper can NEVER
    # prove its array terminal (it only stale-seals children) — nothing backstops this
    # parent's enrich. Gates the enrich-failure handling below (raise vs M1 return).
    tagless_fallback = False
    if parent is not None and parent.exp_id:
        exp_id: Optional[str] = parent.exp_id
        experiment_id = parent.experiment_id
        parent_run_id: Optional[str] = parent.run_id
    else:
        # FALLBACK: the batch_array_job_id tag write failed, so recover exp_id from
        # the array-parent job's Batch env (DescribeJobs).
        tagless_fallback = True
        batch = _batch if _batch is not None else _default_batch()
        try:
            exp_id = _exp_id_from_describe_jobs(batch, job_id)
        except Exception as e:
            # TRANSIENT DescribeJobs failure -> RE-RAISE so the Lambda invocation
            # FAILS and EventBridge's bounded async retry (2 redeliveries) re-runs the
            # finalizer with the SAME job-id-bearing event. That is the ONLY recovery
            # path: the reaper CANNOT prove terminality for a parent missing the
            # batch_array_job_id tag (it only ever stale-seals children). This is
            # DELIBERATELY OPPOSITE to the enrich-failure handling below, which returns
            # WITHOUT raising (there the reaper CAN backstop, so a raise would only
            # cause retry churn). Bounded (2 retries), not a storm.
            print(f"[selfheal] finalizer: transient DescribeJobs failure for job "
                  f"{job_id} ({e}) — raising so EventBridge retries the event")
            raise
        experiment_id = None
        parent_run_id = None
        if not exp_id:
            # DETERMINISTIC unresolvable: DescribeJobs SUCCEEDED but the parent job
            # carries no PRAXIS_EXP_ID (or no job) — a retry cannot help, so this is a
            # genuine successful no-op, NOT a deferral.
            print(f"[selfheal] finalizer: could not resolve exp_id for job {job_id} "
                  "(no parent-run tag, no PRAXIS_EXP_ID in Batch env) — skipping")
            return {"status": "unresolved_exp_id", "job_id": job_id}
        # The parent run is tagged exp_id at CREATION (matrix_launch.py:297, before
        # submit), so it is findable by exp_id even though the batch_array_job_id tag
        # write failed. Recover its experiment_id + run_id so enrich uses them
        # directly (NOT experiment_id=None -> _find_design_doc, absent from the
        # Lambda image) and the annotation targets the right parent run. The residual
        # deferral now covers ONLY the case where NO parent run exists at all for
        # this exp_id (then experiment_id stays None -> enrich attempt -> M1 status).
        parent = _find_parent_by_tag(client, "exp_id", exp_id)
        if parent is not None:
            experiment_id = parent.experiment_id
            parent_run_id = parent.run_id

    # Advisory in-flight marker (round-12): stamp BEFORE enrich so a concurrent reaper
    # tick defers on this parent. Cleared only on the non-raising exit below — the raise
    # paths deliberately leave it set (the EventBridge redelivery lands within minutes,
    # and 30-min staleness is the cleanup; clearing on a raise would widen the race
    # window for that redelivery). Best-effort — never blocks healing.
    now_ms = int(time.time() * 1000)
    if parent_run_id:
        _set_inflight(client, parent_run_id, now_ms)

    print(f"[selfheal] finalizer: reconciling {exp_id} (job {job_id}, "
          f"experiment_id={experiment_id})")
    summary: Optional[dict] = None
    enrich_failed = False
    try:
        # skip_complete=True (H1 authorized design-§11 deviation, Erik 2026-07-16):
        # skip the redundant re-log of units already fully logged in-container so a
        # healthy 100-unit×50-round sweep fits the hard 900s Lambda ceiling.
        summary = enrich(repo_root, exp_id, bucket=bucket, experiment_id=experiment_id,
                         _store=store, _client=client, skip_complete=True)
        print(f"[selfheal] finalizer: enrich summary for {exp_id}: {summary} "
              f"(skip-complete fast path: "
              f"{(summary or {}).get('units_skipped_complete', 0)} already-logged "
              "units left untouched)")
    except Exception as e:
        if tagless_fallback:
            # TAGLESS fallback path: the parent lacks batch_array_job_id, so the reaper
            # can NEVER prove its array terminal (it only stale-seals children) — the
            # "defer to reaper" backstop does NOT exist here, so returning
            # "enrich_failed_deferred_to_reaper" would be a FALSE claim and strand the
            # sweep. RE-RAISE so the Lambda invocation FAILS and EventBridge's bounded
            # async retry redelivers the only job-id-bearing event (same rationale as
            # the transient-DescribeJobs raise — the backstoppability test applied to the
            # ENRICH failure too, not just DescribeJobs). Bounded (2 retries), not a storm.
            print(f"[selfheal] finalizer: enrich failed for {exp_id} on the TAGLESS "
                  f"fallback path (job {job_id}, {e}) — raising so EventBridge retries "
                  "(the reaper cannot backstop a parent missing batch_array_job_id)")
            raise
        # PRIMARY (job-id-tagged) path: a doc-less sweep or transient MLflow error must
        # not crash the Lambda — the reaper CAN reconcile this parent (it carries
        # array_job_id). Deliberately NOT re-raised: a raise would make EventBridge retry
        # the finalizer (churn); the scheduled reaper backstops the honest deferral (M1).
        enrich_failed = True
        print(f"[selfheal] WARN: enrich failed for {exp_id} (job {job_id}): {e}")

    # Annotate FIRST (loud σ-safeguard tags must land even if a raise below exhausts
    # retries), then decide the status.
    annotated = _annotate_incomplete_sweep(client, store, exp_id, summary, parent_run_id)
    # If the annotation did NOT land (and enrich did not already crash — that is the M1
    # deferral, which leaves the parent OPEN so the reaper backstops it): RAISE on BOTH
    # paths. After a clean enrich the parent is SEALED, so the reaper's open-parent scan
    # NEVER re-finds it — the bounded EventBridge retry is the ONLY vehicle to re-run the
    # idempotent enrich (cheap via skip_complete) and re-attempt the annotation. Accepted
    # residual: a sustained annotation failure across all delivery attempts loses the tags
    # until a manual/CLI enrich.
    if not enrich_failed and not annotated:
        print(f"[selfheal] finalizer: incomplete-sweep annotation FAILED to land for "
              f"{exp_id} (job {job_id}) — raising so EventBridge retries; a sealed parent "
              "has no reaper backstop for a missing σ-safeguard annotation")
        raise RuntimeError(
            f"incomplete-sweep annotation failed for {exp_id} (job {job_id}); raising for "
            "EventBridge retry so the σ-safeguard tags are not silently lost")
    units_failed = (summary or {}).get("units_failed", 0)
    parent_seal_failed = (summary or {}).get("parent_seal_failed", False)
    # needs_backstop: enrich COMPLETED but the record is not fully sealed — either N unit
    # repairs failed (parent left OPEN, round-8) OR the parent seal itself raised (round-10).
    # Both leave an UNSEALED parent the reaper re-finds on a job-id-tagged parent (defer);
    # a TAGLESS parent has no backstop, so RAISE for an EventBridge retry (annotation
    # already landed above). Same doctrine as an enrich crash.
    needs_backstop = units_failed > 0 or parent_seal_failed
    if not enrich_failed and needs_backstop and tagless_fallback:
        if units_failed > 0 and parent_seal_failed:
            reason = f"{units_failed} unit(s) failed repair + parent seal failed"
        elif parent_seal_failed:
            reason = "parent seal failed"
        else:
            reason = f"{units_failed} unit(s) failed repair"
        print(f"[selfheal] finalizer: {reason} for {exp_id} on the TAGLESS fallback path "
              f"(job {job_id}) — raising so EventBridge retries (the reaper cannot backstop "
              "a tagless parent)")
        raise RuntimeError(
            f"enrich left {exp_id} unsealed ({reason}) on the tagless fallback path; "
            "raising for EventBridge retry")
    # Honest status: nothing was reconciled when enrich raised ("failed"); a partial
    # enrich OR a failed parent seal on a job-id-tagged parent is truthfully deferred to
    # the reaper (it genuinely re-finds the UNSEALED parent); a clean, fully-sealed pass
    # is "reconciled".
    if enrich_failed:
        status = "enrich_failed_deferred_to_reaper"
    elif needs_backstop:
        status = "enrich_partial_deferred_to_reaper"
    else:
        status = "reconciled"
    # Non-raising exit -> clear the advisory marker (round-12). The raise paths above
    # deliberately skip this (see the set-inflight note).
    if parent_run_id:
        _clear_inflight(client, parent_run_id)
    return {"status": status, "exp_id": exp_id, "job_id": job_id,
            "experiment_id": experiment_id, "summary": summary}


# --- Layer 3: scheduled reaper backstop ------------------------------------

def _child_runs(client: Any, experiment_id: str, parent_run_id: str) -> list:
    """A parent run's child runs (scoped by ``mlflow.parentRunId``), or []."""
    try:
        return list(client.search_runs(
            [experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            order_by=["attributes.start_time ASC"],
            max_results=1000,
        ))
    except Exception as e:
        print(f"[selfheal] WARN: child lookup for parent {parent_run_id} failed ({e})")
        return []


def _search_open_parents(client: Any, cutoff_ms: int) -> list[_ParentRun]:
    """Parent runs (``tags.exp_id`` set) launched since ``cutoff_ms`` that are
    themselves non-terminal OR still have a non-terminal child — the sweeps whose
    MLflow record may need reconciling (design § 5.7)."""
    try:
        exp_ids = client.list_experiment_ids()
        runs = client.search_runs(
            exp_ids,
            filter_string=f"tags.exp_id != '' and attributes.start_time > {cutoff_ms}",
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
    except Exception as e:
        print(f"[selfheal] WARN: reaper parent scan failed ({e})")
        return []
    open_parents: list[_ParentRun] = []
    for r in runs:
        exp_id = r.data.tags.get("exp_id")
        if not exp_id:
            continue
        parent_open = r.info.status not in _TERMINAL_STATUSES
        children = _child_runs(client, r.info.experiment_id, r.info.run_id)
        child_open = any(c.info.status not in _TERMINAL_STATUSES for c in children)
        if parent_open or child_open:
            open_parents.append(_ParentRun(
                run_id=r.info.run_id, exp_id=exp_id,
                experiment_id=r.info.experiment_id,
                array_job_id=r.data.tags.get("batch_array_job_id"),
                status=r.info.status, start_time=r.info.start_time,
                # advisory in-flight marker read from the SAME run object -> zero extra
                # REST calls (round-12).
                inflight=r.data.tags.get(_INFLIGHT_TAG, ""),
            ))
    return open_parents


def _array_is_terminal(batch: Any, job_id: str) -> Optional[bool]:
    """True/False if the Batch array's status is known; None if the lookup is
    unavailable (no Batch VPC endpoint / API error) so the caller uses the
    stale-age fallback instead (design § 5.7)."""
    try:
        resp = batch.describe_jobs(jobs=[job_id])
    except Exception as e:
        print(f"[selfheal] WARN: reaper DescribeJobs for job {job_id} failed ({e})")
        return None
    jobs = resp.get("jobs") or []
    if not jobs:
        return None
    return jobs[0].get("status") in ("SUCCEEDED", "FAILED")


def _seal_stale_children(client: Any, store: Any, parent: _ParentRun, now_ms: int) -> int:
    """Endpoint-independent fallback: seal a stale RUNNING child **FINISHED** (NOT
    FAILED) ONLY when its S3 done-marker proves the unit committed (design § 5.7).

    Age alone is NOT proof of death under Lane A resume-by-unit: a resumed unit's
    run keeps its attempt-1 ``start_time``, so a still-live unit can be older than
    ``_STALE_RUN_AGE_MS``. So a child is sealed only when age > ``_STALE_RUN_AGE_MS``
    AND its ``unit_id`` tag resolves AND ``is_done(store, exp_id, unit_id)``. The
    done-marker (which rides the S3 gateway endpoint, reachable even when the Batch
    endpoint is down) proves the unit committed its ``result -> signal -> done`` — it
    durably SUCCEEDED. The ONLY truthful terminal status is therefore ``FINISHED``
    with ``unit_status=done``: the container merely died between the persist and the
    MLflow seal. Sealing FAILED would PERMANENTLY misrecord a successful unit — once
    a run is terminal the open-parent scan stops revisiting it, so the lie never
    self-corrects.

    Decoration may be incomplete (the container may have died before the
    ``live_enrichment=complete`` marker), so a later enrich fully re-logs the run; as
    a loud-not-silent breadcrumb that decoration repair may still be owed (the reaper
    may never revisit once everything is terminal), the PARENT is tagged
    ``stale_seal_decoration_pending=true`` when ≥1 child is sealed here. Markerless or
    tagless non-terminal children are LEFT ALONE. Reconcile (re-terminate), never
    delete. Returns the count sealed."""
    sealed = 0
    exp_id = parent.exp_id
    for c in _child_runs(client, parent.experiment_id, parent.run_id):
        if c.info.status in _TERMINAL_STATUSES:
            continue
        if now_ms - (c.info.start_time or 0) <= _STALE_RUN_AGE_MS:
            continue
        # done-marker gate: without proof the unit committed, an old RUNNING child
        # may be a LIVE resumed attempt (frozen start_time) — never seal it.
        unit_id = c.data.tags.get("unit_id")
        if not unit_id or not exp_id or not is_done(store, exp_id, unit_id):
            continue
        try:
            # committed -> durably SUCCEEDED; seal FINISHED/done (not FAILED), with a
            # truthful reason so the FINISHED status is not mistaken for a clean live seal.
            client.set_tag(c.info.run_id, "unit_status", "done")
            client.set_tag(c.info.run_id, "stale_seal_reason",
                           "stale RUNNING child with committed S3 done-marker; sealed "
                           "FINISHED by reaper stale fallback (container died between "
                           "persist and seal; decoration may be incomplete — no "
                           "live_enrichment marker, so any later enrich fully re-logs it)")
            client.set_terminated(c.info.run_id, "FINISHED")
            sealed += 1
        except Exception as e:
            print(f"[selfheal] WARN: could not seal stale run {c.info.run_id}: {e}")
    if sealed:
        # loud-not-silent breadcrumb: decoration repair may still be owed for this
        # sweep (the reaper may never revisit once everything is terminal; a manual/CLI
        # enrich is signposted). Best-effort — never fail the reaper on this tag.
        try:
            client.set_tag(parent.run_id, "stale_seal_decoration_pending", "true")
        except Exception as e:
            print(f"[selfheal] WARN: could not tag parent {parent.run_id} "
                  f"stale_seal_decoration_pending: {e}")
    return sealed


def reaper(
    event: dict, context: Any = None, *,
    _enrich: Any = None, _client: Any = None, _store: Any = None,
    _batch: Any = None, _repo_root_override: Optional[str] = None,
    _now_ms: Optional[int] = None,
) -> dict:
    """The finalizer, triggered by a clock instead of an event: reconcile any
    sweep whose Batch array is terminal but whose MLflow record still shows open
    runs, and seal stale RUNNING children even when the Batch API is unreachable
    (design § 3 / § 5.7). Bounded to a ``_REAPER_LOOKBACK_MS`` window. A parent whose
    advisory ``selfheal_inflight`` marker is fresh is SKIPPED (a finalizer is enriching
    it) and counted in ``parents_skipped_inflight`` (round-12 serialization)."""
    enrich = _enrich or enrich_experiment
    now_ms = _now_ms if _now_ms is not None else int(time.time() * 1000)
    client = _client if _client is not None else _default_client()
    store = _store if _store is not None else _default_store()
    bucket = _artifact_bucket()
    repo_root = _repo_root(_repo_root_override)
    cutoff_ms = now_ms - _REAPER_LOOKBACK_MS

    parents = _search_open_parents(client, cutoff_ms)
    healed = sealed = partial = skipped_inflight = 0
    for p in parents:
        # Advisory in-flight skip (round-12): a FRESH selfheal_inflight marker means a
        # finalizer is (or was very recently) enriching this parent — defer to avoid the
        # concurrent-enrich race. A stale/absent marker proceeds (a crashed invocation
        # must never deadlock healing). Read from the scan's run object (zero extra REST).
        if _inflight_fresh(p.inflight, now_ms):
            skipped_inflight += 1
            print(f"[selfheal] reaper: {p.exp_id} skipped: self-heal in flight")
            continue
        # Array-terminality check (needs the Batch VPC endpoint). None => unavailable.
        array_terminal: Optional[bool] = None
        if p.array_job_id:
            batch = _batch if _batch is not None else _default_batch()
            array_terminal = _array_is_terminal(batch, p.array_job_id)

        if array_terminal is True:
            # Terminal array but open MLflow runs -> reconcile (pass the parent's
            # own experiment_id so a doc-less future EXP still heals, design § 5.4).
            # Set the advisory marker BEFORE enrich (round-12) so a concurrent finalizer
            # event narrows the race; cleared on the non-except completion below. On the
            # except path the marker is left set (staleness cleans up; the NEXT-next tick
            # retries) — same "don't clear on failure" choice as the finalizer.
            _set_inflight(client, p.run_id, now_ms)
            try:
                # skip_complete=True (H1): same fast path as the finalizer — the
                # reaper is a trigger of the SAME enrich, so it must also fit 900s.
                summary = enrich(repo_root, p.exp_id, bucket=bucket, experiment_id=p.experiment_id,
                                 _store=store, _client=client, skip_complete=True)
                # The reaper is the backstop for a MISSED finalizer event, so it is the
                # ONLY writer of the loud-not-silent σ-safeguard tags on that path —
                # annotate here too (design § 5.6). Capture the outcome to fold into the
                # partial count. Never raises (best-effort bool).
                annotated = _annotate_incomplete_sweep(client, store, p.exp_id, summary, p.run_id)
                s = summary or {}
                enrich_partial = s.get("units_failed", 0) > 0 or s.get("parent_seal_failed", False)
                if enrich_partial or not annotated:
                    # PARTIAL: unit repairs failed and/or the parent seal raised (parent
                    # stays OPEN/UNSEALED -> next tick retries) OR the annotation failed to
                    # land. Do NOT count it healed; count separately + log loudly. NO raise
                    # (a raise would abort the remaining parents in this same tick). NOTE
                    # the honest asymmetry (round-11): on a clean-enrich-but-failed-annotation
                    # the parent IS sealed, so the next tick will NOT re-find it — an accepted
                    # deeper residual (the reaper is the backstop-of-backstops; its own
                    # annotation failing after a successful heal is a double-failure),
                    # surfaced here via parents_partial + the loud log.
                    partial += 1
                    print(f"[selfheal] reaper: {s.get('units_failed', 0)} unit(s) failed "
                          f"repair, parent_seal_failed={s.get('parent_seal_failed', False)}, "
                          f"annotation_ok={annotated} for {p.exp_id} — NOT counted healed")
                else:
                    healed += 1
                    print(f"[selfheal] reaper: reconciled {p.exp_id} (terminal array, open runs)")
                # Non-except completion -> clear the advisory marker (round-12) so the
                # next tick re-processes a still-open (partial) parent without deferring.
                _clear_inflight(client, p.run_id)
            except Exception as e:
                print(f"[selfheal] WARN: reaper enrich failed for {p.exp_id}: {e}")
        elif array_terminal is None:
            # Array lookup unavailable -> endpoint-independent, done-marker-gated
            # stale seal (age alone is not proof of death under resume-by-unit).
            sealed += _seal_stale_children(client, store, p, now_ms)
        # array_terminal is False (still live) -> leave it for a later interval.

    return {"parents_scanned": len(parents), "units_healed": healed,
            "parents_partial": partial, "parents_skipped_inflight": skipped_inflight,
            "stale_runs_sealed": sealed}

"""``praxis exp enrich EXP-NNN`` — backfill / repair a sweep's MLflow record from S3.

The MLflow enrichment workflow is documented in ``docs/harness/README.md``. The container entrypoint sets rich metadata live; this reconstructs
the FULL record for units that ran under an older/thinner entrypoint (or whose
run was mis-sealed by the pre-fix lifecycle bug) directly from the durable S3
artifacts — the result JSON carries the whole trajectory.

Design constraints honoured here:

  - **Idempotent.** Re-running never mints a duplicate child run: existing runs
    are located by their ``unit_id`` tag (``find_runs_by_unit``) and reused.
    Re-setting a metric/tag/param is harmless.
  - **Repairs duplicate runs.** A reclaimed attempt that was retried leaves two
    runs sharing a ``unit_id`` (the mis-sealed zombie + the completing retry);
    the run carrying the durable result is enriched FINISHED and the older
    duplicate(s) are reconciled FAILED — plus legacy input tags on a reused
    old run are stripped so the recovered run honours the param/tag isolation.
  - **Corrects mis-sealed runs.** A run that exists but whose unit has no
    durable result (e.g. a Spot reclaim the pre-fix code sealed FINISHED) is
    re-terminated FAILED with a ``unit_status`` note.
  - **No duplicated S3-key logic.** All keys come from ``praxis_exp.storage``
    via the pure helpers in ``mlflow_enrichment`` / ``docker.entrypoint`` — this
    module reuses them rather than re-deriving strings.
  - **Best-effort, never destructive.** A zombie run is re-terminated, never
    deleted; every enrichment sub-step is individually guarded.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from praxis_exp import storage
from praxis_exp.integrity import is_done
from praxis_exp.manifest import read_manifest
from praxis_exp.matrix_doc import parse_matrix
from praxis_exp.mlflow_client import PraxisMlflowClient, build_meta_dataset, build_signal_dataset
from praxis_exp.refill_reconcile import reconcile_sweep_tags
from praxis_exp.mlflow_enrichment import (
    dataset_digest_from_metadata,
    dataset_source_uri,
    unit_s3_tags,
)
from praxis_exp.runner_paths import result_filename, signal_filename
from praxis_exp.storage import ObjectNotFoundError, ObjectStore
from praxis_exp.units import Unit

# Reuse the container's PURE unit-enrichment helpers rather than duplicating
# them (they are already unit-tested in tests/test_fleet_entrypoint.py).
#
# The local package is named `docker`, which COLLIDES with the PyPI docker SDK —
# a transitive mlflow dependency that mlflow imports (and caches in sys.modules)
# before this line runs. So a plain `from docker.entrypoint import ...` resolves
# to the SDK in the installed `praxis` CLI and raises ModuleNotFoundError (pytest
# masks it because the repo root is on sys.path from startup). Load the local
# docker/entrypoint.py directly by file path to bypass the name entirely.
import importlib.util as _ilu
import os as _os
_ENTRYPOINT_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "docker", "entrypoint.py"
)
_spec = _ilu.spec_from_file_location("_praxis_container_entrypoint", _ENTRYPOINT_PATH)
_entrypoint = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_entrypoint)
build_unit_params = _entrypoint.build_unit_params
build_unit_tags = _entrypoint.build_unit_tags
cold_start_model_tag = _entrypoint.cold_start_model_tag
defense_token = _entrypoint.defense_token
final_metrics = _entrypoint.final_metrics
trajectory_metrics = _entrypoint.trajectory_metrics
unit_criteria_ok = _entrypoint.unit_criteria_ok
build_unit_note = _entrypoint.build_unit_note

_DATASET_NAME = "edge_full_20_rmc"
_DATASET_DIR = "edge_full_20"


class EnrichError(RuntimeError):
    """Raised when a backfill precondition (missing manifest/doc) fails."""


class _EnrichClient:
    """The MLflow surface ``enrich_experiment`` needs (see PraxisMlflowClient).

    ``PraxisMlflowClient`` implements all of these; the protocol is documented
    here (mirroring ``ingest._AttrClient``) so a test fake is obvious."""

    def get_or_create_experiment(self, name: str) -> str: ...
    def find_parent_run(self, experiment_id: str, exp_id: str) -> Optional[str]: ...
    def find_runs_by_unit(
        self, experiment_id: str, unit_id: str, parent_run_id: Optional[str] = None,
    ) -> list[str]: ...
    def search_runs(
        self, experiment_ids: list[str], *, filter_string: str = "",
        order_by: Optional[list[str]] = None, max_results: int = 1000,
    ) -> Any: ...  # returns Run objects (info.status / data.tags) — H1 skip-complete check
    def create_run(self, experiment_id: str, tags: dict[str, str]) -> str: ...
    def log_params(self, run_id: str, params: dict[str, str]) -> None: ...
    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None: ...
    def set_tag(self, run_id: str, key: str, value: str) -> None: ...
    def delete_tag(self, run_id: str, key: str) -> None: ...
    def log_artifact(self, run_id: str, local_path: str) -> None: ...
    def log_input(self, run_id: str, dataset: Any, context: str = "training") -> None: ...
    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None: ...


# Input/renamed keys the OLD entrypoint stored as TAGS that the PR #15 param/tag
# isolation moved to PARAMS (config->defense PARAM) or renamed (defense->
# defense_token, dataset->dataset_name). Stripped from a reused old run so the
# recovered run is not left with duplicated param/tag fields.
_LEGACY_INPUT_TAGS = (
    "config", "scenario", "seed", "mode", "max_per_client", "dataset",
    "dataset_dir_s3", "defense",
)


def _find_design_doc(repo_root: Path, exp_id: str) -> Path:
    exp_dir = repo_root / "docs" / "experiments"
    for f in exp_dir.iterdir():
        if f.is_file() and f.name.startswith(f"{exp_id}-") and not f.name.endswith("-result.md"):
            return f
    raise EnrichError(f"no design doc for {exp_id} in {exp_dir}")


def _experiment_name(repo_root: Path, exp_id: str) -> str:
    """The MLflow experiment name for a sweep = its matrix doc's slug (the same
    value ``matrix_launch`` used as ``exp_name``)."""
    return parse_matrix(_find_design_doc(repo_root, exp_id)).slug


def _read_result(store: ObjectStore, exp_id: str, unit_id: str) -> Optional[dict]:
    """The unit's durable result JSON from S3, or None if it never landed."""
    try:
        raw = store.get_bytes(storage.result_key(exp_id, unit_id))
    except ObjectNotFoundError:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[enrich] WARN: result for {unit_id} is not valid JSON ({e}); skipping")
        return None


def _read_local_metadata(repo_root: Path) -> Optional[dict]:
    path = repo_root / "data" / _DATASET_DIR / "metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _log_dataset_input(client: _EnrichClient, run_id: str, bucket: str, metadata: Optional[dict]) -> None:
    """REPAIR step (native dataset input): NO internal swallow — a failure PROPAGATES
    to the per-unit handler in ``enrich_experiment`` so the unit counts as failed and
    the parent is left open."""
    dataset = build_meta_dataset(
        name=_DATASET_NAME,
        source_uri=dataset_source_uri(bucket, _DATASET_DIR),
        digest=dataset_digest_from_metadata(metadata),
    )
    client.log_input(run_id, dataset, context="training")


def _log_signal_dataset(
    client: _EnrichClient, run_id: str, exp_id: str, bucket: str, unit: Unit,
) -> None:
    """Attach the unit's signal log as a **dataset-by-source** (context ``"signal"``)
    — no byte copy (GWU-47 Lane A). REPAIR step: NO internal swallow — a failure
    PROPAGATES to the per-unit handler so the unit counts as failed."""
    dataset = build_signal_dataset(
        exp_id=exp_id, unit_id=unit.unit_id, bucket=bucket,
        defense_token=defense_token(unit.config),
    )
    client.log_input(run_id, dataset, context="signal")


def _log_artifacts(
    client: _EnrichClient, store: ObjectStore, run_id: str, exp_id: str, unit: Unit,
) -> None:
    """Download the unit's durable **result** artifact from S3 and attach it to
    the run (a REPAIR step). The signal log is NO LONGER copied — it is referenced as
    a dataset-by-source (``_log_signal_dataset``). Any pre-existing ``signal.jsonl``
    copy left by the old copy-everything scheme is best-effort deleted so re-enrich
    MIGRATES old runs (not just stops future copies) — this is what makes the "drops
    the duplicate signal artifact" migration promise true (GWU-47 Lane A)."""
    # COSMETIC cleanup: only REMOVES a redundant legacy copy (writes no part of the
    # record), so per _enrich_completed_unit's rule it is the one kind of step that may
    # swallow — an absent artifact / a client lacking delete_artifact is a no-op.
    try:
        client.delete_artifact(
            run_id, signal_filename(unit, defense_token=defense_token(unit.config))
        )
    except Exception:
        pass
    # REPAIR: read result.json from S3 and upload it. NO swallow — a read or upload
    # failure PROPAGATES so the unit counts as failed. For a
    # committed unit the result key exists (the loop already read it), so a raise here
    # is a genuine transient failure worth retrying, not a missing artifact.
    key = storage.result_key(exp_id, unit.unit_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / result_filename(unit)
        local.write_bytes(store.get_bytes(key))
        client.log_artifact(run_id, str(local))


def _create_run(
    client: _EnrichClient, experiment_id: str, unit: Unit,
    parent_run_id: Optional[str] = None,
) -> str:
    """Mint a child run for a committed unit that has none. It MUST carry
    ``mlflow.parentRunId`` when the launch parent is known, otherwise the next
    parent-scoped ``find_runs_by_unit`` can't see it and would mint yet another
    duplicate outside the launch hierarchy."""
    tags = {"mlflow.runName": unit.unit_id, "unit_id": unit.unit_id}
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    return client.create_run(experiment_id, tags=tags)


def _strip_legacy_input_tags(client: _EnrichClient, run_id: str) -> None:
    """Delete the input tags an OLD-entrypoint run carried that are now params
    (or renamed tags), so a recovered run honours the param/tag isolation rather
    than keeping duplicated fields. COSMETIC cleanup
    (removes redundant data, writes no part of the record) -> best-effort per tag
    per _enrich_completed_unit's rule: an absent tag (new-image run) or a client
    without delete_tag is fine."""
    for key in _LEGACY_INPUT_TAGS:
        try:
            client.delete_tag(run_id, key)
        except Exception:
            pass


def _enrich_completed_unit(
    client: _EnrichClient, store: ObjectStore, *, run_id: str, exp_id: str,
    bucket: str, unit: Unit, result: dict, meta: dict, metadata: Optional[dict],
) -> None:
    """Re-log the FULL repairable record for one committed unit: identity/provenance
    tags, S3-link tags, ``note.content``, params, final + per-round metrics, the
    cold-start tag, native + signal dataset inputs, and the result.json artifact.

    CLASSIFICATION RULE (terminates the swallowed-error audit class):
    within a unit's repair, ONLY *cosmetic cleanup* may swallow its own errors — the
    legacy ``signal.jsonl`` ``delete_artifact`` and ``_strip_legacy_input_tags``,
    which only REMOVE redundant data and write no part of the record. EVERY *repair*
    step (anything that writes part of the record) propagates to the per-unit
    ``try/except`` in ``enrich_experiment``, which does ``failed += 1`` (unit
    isolation) and leaves the parent OPEN (round-8). A unit counts as failed iff any
    part of its repairable record was not written."""
    for k, v in build_unit_tags(
        unit,
        methodology_version=meta.get("methodology_version", ""),
        image_digest=meta.get("image_digest", ""),
    ).items():
        client.set_tag(run_id, k, v)
    s3_tags = unit_s3_tags(bucket, exp_id, unit.unit_id)
    for k, v in s3_tags.items():
        client.set_tag(run_id, k, v)
    # note.content on the backfill path too (GWU-47 Lane C) — RMC params + gate verdict,
    # parity with the live post_persist_enrichment path.
    client.set_tag(run_id, "mlflow.note.content", build_unit_note(
        unit, result=result, console_url=s3_tags.get("s3_console_url", ""),
        result_filename=result_filename(unit),
    ))
    client.log_params(run_id, build_unit_params(unit))
    _strip_legacy_input_tags(client, run_id)

    for key, value in final_metrics(result).items():
        client.log_metric(run_id, key, value)
    for key, value, step in trajectory_metrics(result):
        client.log_metric(run_id, key, value, step=step)

    cs_model = cold_start_model_tag(result)
    if cs_model:
        client.set_tag(run_id, "cold_start_model", cs_model)

    _log_dataset_input(client, run_id, bucket, metadata)
    _log_signal_dataset(client, run_id, exp_id, bucket, unit)
    _log_artifacts(client, store, run_id, exp_id, unit)

    client.set_tag(run_id, "criteria_ok", "true" if unit_criteria_ok(result) else "false")
    client.set_tag(run_id, "unit_status", "done")
    client.set_terminated(run_id, "FINISHED")
    # backfill_enrichment=complete (GWU-50) — the LAST write of this function, on
    # purpose. Post-round-9 every repair step above raises/propagates on failure (only
    # cosmetic cleanup — the legacy signal.jsonl delete and _strip_legacy_input_tags —
    # may swallow), so REACHING this line proves every repair step succeeded; the marker
    # is therefore inherently truthful with no extra witness-gating (unlike the live
    # entrypoint, whose swallowing decoration block must gate live_enrichment on
    # start_ok/metrics_ok/artifacts_ok/post_ok). It lets the skip-complete fast path
    # (_primary_already_complete) recognize a backfill-repaired unit exactly as it
    # recognizes a live one, closing the reaper/EventBridge re-log loop on
    # backfill-heavy partial-failure sweeps (GWU-45's H1 fix, extended to this path).
    client.set_tag(run_id, "backfill_enrichment", "complete")


def _reconcile_zombie_run(
    client: _EnrichClient, run_id: str, unit: Unit, *, reason: Optional[str] = None,
) -> None:
    """A run that must NOT be FINISHED: either the unit produced no durable
    result (pre-fix lifecycle bug sealed it FINISHED), or it is a duplicate
    reclaimed attempt superseded by the run that carries the result. Re-terminate
    FAILED and annotate, never delete."""
    try:
        client.set_tag(run_id, "unit_status", "reconciled_failed")
        client.set_tag(
            run_id, "reclaim_reason",
            reason or "no durable S3 result for unit; run mis-sealed by pre-fix lifecycle",
        )
    except Exception as e:
        print(f"[enrich] WARN: could not tag reconciled run {run_id}: {e}")
    client.set_terminated(run_id, "FAILED")


# The canonical final metric the entrypoint logs for a completed unit
# (docker/entrypoint.py final_metrics -> mlflow.log_metric). Its presence in a
# run's metrics is the belt-and-suspenders half of the skip-complete predicate:
# it witnesses the live per-round/final metric stream, which no tag/flag tracks.
_SKIP_COMPLETE_FINAL_METRIC = "final_f1"


def _primary_already_complete(
    client: _EnrichClient, experiment_id: str, unit_id: str,
    parent_run_id: Optional[str],
) -> bool:
    """H1: True when the unit's PRIMARY (newest) run is PROVABLY fully logged
    in-container, so the Lambda finalizer/reaper can skip its redundant per-unit
    re-log. Requires ALL of:
      - ``info.status == "FINISHED"``,
      - tag ``unit_status == "done"``,
      - EITHER completion marker: tag ``live_enrichment == "complete"`` (the live
        entrypoint sets it ONLY when the FULL repairable set logged in-container:
        start-metadata AND result + per-round metrics AND the result.json artifact AND
        the post-persist decoration block ALL succeeded) OR tag
        ``backfill_enrichment == "complete"`` (``_enrich_completed_unit`` sets it as its
        LAST write, so reaching it proves every propagating repair step succeeded —
        GWU-50; a unit healed by the backfill path is as fully logged as a live one).
        ``unit_status=done`` alone is INSUFFICIENT:
         guarantees
        ``done`` for a committed unit even when decoration partially failed, and the
        SIGTERM-mid-decoration path sets ``done`` too — such a ``FINISHED``+``done``
        run can be MISSING metrics/tags/artifacts, and skipping it would prevent the
        self-heal from ever repairing exactly those runs,
      - the canonical final metric (``_SKIP_COMPLETE_FINAL_METRIC``) present in the
        run's metrics — belt-and-suspenders for the live metric stream.
    Absence of BOTH markers (ALL pre-img-v2 runs) -> never skipped -> always re-logged.
    Fail-safe direction: any doubt costs a redundant re-log, NEVER a missed repair.

    Fetches run OBJECTS via the client's ``search_runs`` passthrough (a REAL
    ``PraxisMlflowClient`` method — deliberately NOT a nonexistent ``get_run``, which
    would AttributeError in production, be swallowed by the guard below, and silently
    disable the fast path while fake-based tests passed) using the SAME ``unit_id`` /
    parent-scope filter ``find_runs_by_unit`` issues, so status + tags + metrics come
    from ONE search (~1 extra call/unit — cheap vs the ~250 log calls/unit it avoids).
    Best-effort: any lookup failure returns False, so a check error only costs a
    redundant re-log, never a wrongly-skipped unit."""
    try:
        filt = f"tags.unit_id = '{unit_id}'"
        if parent_run_id:
            filt += f" and tags.`mlflow.parentRunId` = '{parent_run_id}'"
        objs = client.search_runs(
            [experiment_id], filter_string=filt,
            order_by=["attributes.start_time ASC"], max_results=1000,
        )
        if not objs:
            return False
        newest = objs[-1]  # start_time ASC -> newest is last (== find_runs_by_unit[-1])
        tags = newest.data.tags
        metrics = getattr(newest.data, "metrics", {}) or {}
        return (newest.info.status == "FINISHED"
                and tags.get("unit_status") == "done"
                and (tags.get("live_enrichment") == "complete"
                     or tags.get("backfill_enrichment") == "complete")
                and _SKIP_COMPLETE_FINAL_METRIC in metrics)
    except Exception as e:
        print(f"[enrich] WARN: skip-complete status check for {unit_id} failed "
              f"({e}); will re-log")
        return False


def enrich_experiment(
    repo_root: Path,
    exp_id: str,
    *,
    bucket: str,
    experiment_id: Optional[str] = None,
    skip_complete: bool = False,
    _store: ObjectStore,
    _client: Optional[_EnrichClient] = None,
) -> dict[str, Any]:
    """Backfill/repair every unit of a completed sweep's MLflow record from S3.

    ``experiment_id`` (optional, keyword-only): when supplied — the self-heal
    finalizer/reaper read it off the resolved parent run (GWU-45 Lane B) — enrich
    uses it directly and SKIPS ``_experiment_name`` -> ``_find_design_doc``, so a
    sweep whose design doc was never baked (the acceptance EXP + every future EXP)
    still heals instead of raising. The CLI/agent path passes no id and falls back
    to the doc-derived experiment name, unchanged (design § 5.4).

    ``skip_complete`` (keyword-only, default ``False``): the H1 fast path — an
    AUTHORIZED design-§11 deviation (Erik, 2026-07-16) for the **Lambda triggers
    ONLY**. When True, a unit whose primary run is already fully logged
    in-container (``FINISHED`` + ``unit_status=done``) is NOT re-logged (its ~250
    redundant metric/tag REST calls would blow the Lambda 900s ceiling at
    100-unit×50-round scale); it is counted in ``units_skipped_complete`` and its
    zombie/duplicate siblings are STILL reconciled. Default False keeps the CLI /
    agent / GWU-47 re-migration paths byte-for-byte identical (they MUST keep
    re-authoring FINISHED runs — that is why the fast path is opt-in).

    The launch parent run is sealed FINISHED ONLY when every unit repaired cleanly
    (``units_failed == 0``); on a partial failure the parent is left OPEN so its
    terminal status never lies about incomplete reconciliation and the reaper (or a
    CLI re-run — enrich is idempotent) re-finds the sweep.

    Returns a summary: how many child runs were fully enriched vs reconciled
    (zombie -> FAILED) vs skipped (no result and no run to fix) vs
    skipped-complete (already-logged primaries left untouched under
    ``skip_complete``) vs failed (``units_failed`` — repair errors that left the
    parent open)."""
    repo_root = Path(repo_root)
    _, meta, units = read_manifest(_store, exp_id)
    client = _client or PraxisMlflowClient()

    experiment_id = experiment_id or client.get_or_create_experiment(
        _experiment_name(repo_root, exp_id)
    )
    # Scope child lookups to THIS launch's parent run so a unit_id shared across
    # launches (aborted attempts) or a foreign EXP reusing the slug/matrix is not
    # enriched/overwritten by mistake.
    parent_run_id = client.find_parent_run(experiment_id, exp_id)
    if parent_run_id is None:
        print(f"[enrich] WARN: no non-FAILED parent run found for {exp_id}; child "
              "lookup will be UNSCOPED and may include other launches' runs")
    metadata = _read_local_metadata(repo_root)

    enriched = reconciled = skipped = failed = dup_reconciled = 0
    skipped_complete = 0
    for unit in units:
        try:
            # The done-marker is persist_unit's commit point (is_done), written
            # strictly AFTER the result + signal. Gate on the marker — NOT on mere
            # result-readability — so a unit whose host died between the result
            # upload and the marker write (partial persist / IntegrityError) is
            # treated as UNcommitted and reconciled/skipped, never sealed FINISHED

            committed = is_done(_store, exp_id, unit.unit_id)
            result = _read_result(_store, exp_id, unit.unit_id) if committed else None
            runs = client.find_runs_by_unit(
                experiment_id, unit.unit_id, parent_run_id=parent_run_id,
            )
            if committed and result is not None:
                # Enrich ONE run (the newest, or a fresh one) with the durable
                # result; reconcile any duplicate attempts — a reclaim zombie and
                # its completing retry share a unit_id — so exactly one FINISHED
                # run carries the result.
                primary = runs[-1] if runs else _create_run(
                    client, experiment_id, unit, parent_run_id=parent_run_id,
                )
                # H1 skip-complete fast path (AUTHORIZED design-§11 deviation, Erik
                # 2026-07-16; Lambda triggers ONLY, default OFF): a primary that is
                # ALREADY fully logged in-container (FINISHED + unit_status=done)
                # needs no re-log — re-logging every unit is ~250 redundant REST
                # calls/unit, blowing the Lambda 900s ceiling at 100-unit×50-round
                # scale. Only the PRIMARY's re-log is skipped; the zombie/duplicate
                # reconciliation below STILL runs (sealing reclaim residue is the
                # finalizer's real job). ``runs`` truthy: a freshly-created primary
                # (empty runs) is never already-complete, so it always re-logs.
                if (skip_complete and runs
                        and _primary_already_complete(
                            client, experiment_id, unit.unit_id, parent_run_id)):
                    skipped_complete += 1
                else:
                    _enrich_completed_unit(
                        client, _store, run_id=primary, exp_id=exp_id,
                        bucket=bucket, unit=unit, result=result, meta=meta, metadata=metadata,
                    )
                    enriched += 1
                for dup in (runs[:-1] if runs else []):
                    _reconcile_zombie_run(
                        client, dup, unit,
                        reason="superseded duplicate attempt; the completing run carries the result",
                    )
                    dup_reconciled += 1
                continue
            # uncommitted: reconcile every run for the unit (all mis-sealed)
            if runs:
                for r in runs:
                    _reconcile_zombie_run(client, r, unit)
                reconciled += 1
            else:
                skipped += 1
        except Exception as e:
            # One unit's failure (e.g. an immutable-param conflict on a
            # preexisting run, or a transient set_tag/log_metric error) must not
            # abort the whole backfill and leave later units + zombie runs
            # unrepaired.
            failed += 1
            print(f"[enrich] WARN: unit {unit.unit_id} enrichment failed ({e}); continuing")

    # Seal the launch parent run FINISHED — but ONLY when every unit repaired cleanly
    # (failed == 0). The parent is created RUNNING at launch and the array children
    # never own its lifecycle, so a clean sweep must seal it or it lingers RUNNING
    # forever. But on a PARTIAL failure a FINISHED parent would be a LIE
    # ("reconciliation complete"), and once terminal the reaper's open-parent scan
    # drops the sweep FOREVER — permanently stranding a transient mid-repair failure
    # (paradoxically worse than a full enrich crash, which leaves the parent open and
    # reaper-visible). So leave the parent OPEN + loud WARN: the reaper re-finds it (on
    # a job-id-tagged parent) and enrich is idempotent, so a re-run heals.
    # A partial CLI
    # enrich now leaves the parent RUNNING, which is truthful + visible; the operator
    # re-runs the idempotent enrich.) Best-effort: parent housekeeping never fails the run.
    # parent_seal_failed: the seal was ATTEMPTED (clean pass) and RAISED. The swallow is
    # RETAINED so a transient MLflow failure never crashes a CLI enrich on parent
    # housekeeping — but the outcome is no longer INVISIBLE: it
    # rides the summary so the Lambda paths apply the backstoppability doctrine. Left
    # False when the parent sealed cleanly OR when it was deliberately left open for
    # failed > 0 (which units_failed already surfaces — no seal was attempted there).
    parent_seal_failed = False
    if parent_run_id is not None:
        if failed == 0:
            # GWU-41: reconcile §5.6 completeness BEFORE sealing FINISHED — a
            # refill that completed previously-missing cells must clear the
            # finalizer's stale sweep_incomplete/missing_cells (preserving
            # refill_history), or the sealed parent claims incomplete forever.
            # Best-effort: parent housekeeping never fails the run.
            try:
                reconcile_sweep_tags(client, _store, exp_id, parent_run_id)
            except Exception as e:
                print(f"[enrich] WARN: sweep-completeness reconciliation failed for "
                      f"{exp_id}: {e}")
            try:
                client.set_terminated(parent_run_id, "FINISHED")
            except Exception as e:
                parent_seal_failed = True
                print(f"[enrich] WARN: could not seal parent run {parent_run_id} FINISHED: {e}")
        else:
            print(f"[enrich] WARN: {failed} unit(s) failed repair for {exp_id} — parent "
                  f"run {parent_run_id} left OPEN so the reaper re-finds this sweep; "
                  "enrich is idempotent, re-run heals")

    return {
        "experiment_id": experiment_id,
        "parent_run_id": parent_run_id,
        "units_enriched": enriched,
        "units_reconciled": reconciled,
        "units_skipped": skipped,
        "units_skipped_complete": skipped_complete,
        "units_failed": failed,
        "duplicate_runs_reconciled": dup_reconciled,
        "parent_seal_failed": parent_seal_failed,
    }

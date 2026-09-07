"""``praxis exp launch-matrix --refill EXP-NNN --cells …`` — re-run the missing
cells of a prior sweep, in place, with full chain-of-custody hardening.

This module re-enables explicit refills while preserving terminal-array and
per-launch provenance safeguards:

  A refill re-submits the FULL canonical array into the SAME namespace against
  the ORIGINAL (immutable) manifest. The container resolves each
  ``AWS_BATCH_JOB_ARRAY_INDEX`` against ``sweeps/{exp_id}/manifest.json`` and
  exits 0 on any cell whose done-marker already exists (``should_skip`` returns
  BEFORE any MLflow contact — proven by EXP-011's own Batch Attempts=10
  retries), so only the genuinely-missing cells actually train. Refilled
  children nest under the ORIGINAL parent run (``PRAXIS_PARENT_RUN_ID``), so the
  sweep's parent shows all cells.

Why not a small k-sized array: the container reads the canonical manifest by
``exp_id`` (it ignores ``PRAXIS_MANIFEST_KEY``), so array indices only line up
against the full original manifest. A k-sized array would remap indices onto
the wrong units. Re-submitting full-size is the same idempotent-skip path Batch
retries already exercise — no container change, no manifest divergence.

The six validated invariants:
  1. refuse-if-complete / refill-if-partial (nothing missing → refuse).
  2. matrix-identity — the current doc must expand to EXACTLY the prior
     manifest (same unit_ids, indices, specs); a changed matrix needs a new EXP id.
  3. provenance policy — methodology change → refuse; git_sha/image_digest change
     → allowed, prior launch appended to a ``prior_launches`` chain in the refill
     record + a ``refill_*_provenance_change`` parent tag.
  4. array-index stability — trivially satisfied: the original manifest is never
     rewritten, so a still-queued old child resolves the SAME unit it always did.
  5. terminal-prior-array — a provenance-changing refill verifies the prior Batch
     array is terminal (injected ``array_terminal``) before launching.
  6. per-launch tag — a fresh serial ``exp/EXP-NNN.rN`` with its own annotation.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import json

from praxis_exp import git_helper, storage
from praxis_exp.batch import BatchSubmitter
from praxis_exp.config import Config
from praxis_exp.integrity import is_done
from praxis_exp.matrix_doc import parse_matrix


def _batch_resource_tags(exp_id: str) -> dict[str, str]:
    """Portable Batch tags, overridable without embedding site identities."""
    return {
        "EXP": exp_id,
        "Project": os.environ.get(
            "PRAXIS_PROJECT_TAG", "federated-learning-rmc-defense"
        ),
        "Owner": os.environ.get("PRAXIS_OWNER_TAG", "researcher"),
        "Purpose": os.environ.get(
            "PRAXIS_PURPOSE_TAG", "federated-learning-research"
        ),
    }
from praxis_exp.matrix_launch import (
    MatrixLaunchError,
    _find_matrix_doc,
    _read_ignore,
    _read_prior_manifest,
    _resolve_launch_tag,
    _verify_job_def_image,
)
from praxis_exp.refill_reconcile import read_refill_records
from praxis_exp.storage import ObjectStore, manifest_key, sweep_prefix
from praxis_exp.units import Unit, expand_matrix

# The full unit spec compared for matrix identity (invariant 2). unit_id encodes
# scenario/config/mode/seed/repeat but NOT rounds/max_per_client, so identity
# must compare those explicitly or a doc edit to rounds would silently pass.
_UNIT_SPEC_FIELDS = ("config", "scenario", "mode", "seed", "max_per_client", "rounds", "repeat")

# Prior-launch meta preserved in the refill record's prior_launches chain.
_LAUNCH_PROVENANCE_FIELDS = ("git_sha", "image_digest", "methodology_version", "n_units", "launched_at")


def _assert_matrix_unchanged(exp_id: str, prior_units: list[Unit], current_units: list[Unit]) -> None:
    """Invariant 2 (as EQUALITY for a full-array refill): the current doc must
    expand to exactly the prior manifest — same array_index → same unit_id and
    spec. A refill re-submits the full original array against the original
    manifest, so any divergence (added/removed/re-specced/re-ordered unit) would
    mismatch what the container resolves. A changed matrix requires a new EXP id.
    """
    prior_by_idx = {u.array_index: u for u in prior_units}
    cur_by_idx = {u.array_index: u for u in current_units}
    if len(prior_units) != len(current_units) or set(prior_by_idx) != set(cur_by_idx):
        raise MatrixLaunchError(
            f"{exp_id} refill refused: the current design doc expands to "
            f"{len(current_units)} unit(s) at indices {sorted(cur_by_idx)[:5]}…, but the "
            f"prior manifest has {len(prior_units)} at {sorted(prior_by_idx)[:5]}…. A "
            "refill re-runs the ORIGINAL matrix; a changed matrix requires a NEW EXP id."
        )
    for idx, prior in prior_by_idx.items():
        cur = cur_by_idx[idx]
        diffs = [
            f"{field}: {getattr(prior, field)} -> {getattr(cur, field)}"
            for field in _UNIT_SPEC_FIELDS
            if getattr(prior, field) != getattr(cur, field)
        ]
        if cur.unit_id != prior.unit_id:
            diffs.insert(0, f"unit_id: {prior.unit_id} -> {cur.unit_id}")
        if diffs:
            raise MatrixLaunchError(
                f"{exp_id} refill refused: cell at array_index {idx} changed since launch "
                f"({'; '.join(diffs)}). A refill must re-run the ORIGINAL design; changed "
                "settings require a NEW EXP id."
            )


def _resolve_cells(exp_id: str, cells: list[str], prior_units: list[Unit]) -> list[Unit]:
    """Resolve a ``--cells`` spec (comma-separated tokens; each an integer
    array_index or a unit_id) to prior-manifest Units, de-duplicated in the
    order given. Every token must name a cell that exists in the prior
    manifest."""
    by_id = {u.unit_id: u for u in prior_units}
    by_idx = {u.array_index: u for u in prior_units}
    resolved: list[Unit] = []
    seen: set[str] = set()
    for raw in cells:
        tok = raw.strip()
        if not tok:
            continue
        if tok.lstrip("-").isdigit():
            unit = by_idx.get(int(tok))
            if unit is None:
                raise MatrixLaunchError(
                    f"{exp_id} refill refused: no cell at array_index {tok} in the prior "
                    f"manifest (valid indices 0..{len(prior_units) - 1})."
                )
        else:
            unit = by_id.get(tok)
            if unit is None:
                raise MatrixLaunchError(
                    f"{exp_id} refill refused: no cell with unit_id {tok!r} in the prior "
                    "manifest. Use an array_index or a manifest unit_id."
                )
        if unit.unit_id not in seen:
            seen.add(unit.unit_id)
            resolved.append(unit)
    if not resolved:
        raise MatrixLaunchError(f"{exp_id} refill refused: --cells resolved to no cells.")
    return resolved


def _refill_provenance(
    exp_id: str, prior_meta: dict[str, Any], *,
    methodology_version: str, git_sha: str, image_digest: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Invariant 3. methodology change → refuse (results under two methodology
    versions are not comparable within one EXP). git_sha/image_digest change
    (same methodology) → allowed: the normal recover-after-fix refill. Returns
    ``(change_summary_or_None, prior_launches_chain)`` with the prior launch's
    meta appended (oldest first; an existing chain is preserved)."""
    prior_chain = list(prior_meta.get("prior_launches", []))
    prior_mv = prior_meta.get("methodology_version")
    if prior_mv != methodology_version:
        raise MatrixLaunchError(
            f"{exp_id} refill refused: prior launch ran methodology_version {prior_mv}, "
            f"this launch is {methodology_version} — results under two methodology versions "
            "are not comparable within one experiment. A methodology change requires a NEW EXP id."
        )
    changes = [
        f"{field} {prior_meta.get(field)} -> {new}"
        for field, new in (("git_sha", git_sha), ("image_digest", image_digest))
        if prior_meta.get(field) != new
    ]
    if not changes:
        return None, prior_chain
    prior_entry = {f: prior_meta[f] for f in _LAUNCH_PROVENANCE_FIELDS if f in prior_meta}
    return "; ".join(changes), prior_chain + [prior_entry]


def _serial_suffix(launch_tag: str, exp_id: str) -> str:
    """The ``rN`` label for this refill, from the minted serial tag
    (``exp/EXP-NNN.r2`` → ``r2``). The base tag (no suffix) should not occur for
    a refill — the original launch created it — but defaults to ``r1`` if it does."""
    base = f"exp/{exp_id}"
    suffix = launch_tag[len(base):].lstrip(".")
    return suffix or "r1"


def _refill_record_key(exp_id: str, serial: str) -> str:
    return f"{sweep_prefix(exp_id)}/refills/{serial}/manifest.json"


def _latest_array_id(
    store: ObjectStore, exp_id: str, prior_meta: dict[str, Any],
) -> tuple[str | None, str]:
    """The array id of the MOST RECENT launch for this sweep, as the terminality
    baseline for invariant 5. On a second-or-later refill the
    original manifest's array is long terminal, but an EARLIER refill's array may
    still be running old-provenance children — so the baseline must be the newest
    refill array (persisted post-submit into each refill record), falling back to
    the original manifest only when no refill has recorded one yet. Returns
    ``(array_id_or_None, source_label)``."""
    for rec in reversed(read_refill_records(store, exp_id)):  # newest serial first
        aid = rec.get("array_job_id")
        if aid:
            return aid, f"refill {rec.get('serial', '?')}"
    return prior_meta.get("array_job_id"), "original launch"


def _rollback_refill(
    *, _store: ObjectStore, _mlflow: Any, _git: Any, repo_root: Path,
    record_key: str | None, created_tag: str | None,
    parent_run: str | None, terminate_parent: bool,
) -> str:
    """Best-effort rollback of everything THIS refill created, each step
    guarded so a failing step never masks the original error. The refill NEVER
    touches the original manifest, so rollback only removes the refill record,
    the tag this launch created, and (only if this refill MINTED a new parent
    run) that run. Returns a leftover suffix ('' when clean)."""
    leftovers: list[str] = []
    if terminate_parent and parent_run is not None:
        try:
            _mlflow.set_terminated(parent_run, "FAILED")
        except Exception as e:
            leftovers.append(f"refill parent run {parent_run} could not be terminated ({e})")
    if created_tag is not None:
        try:
            _git.delete_local_tag(repo_root, created_tag)
        except Exception as e:
            leftovers.append(f"local tag {created_tag} could not be deleted ({e})")
    if record_key is not None:
        try:
            _store.delete(record_key)
        except Exception as e:
            leftovers.append(f"refill record {record_key} could not be deleted ({e})")
    if leftovers:
        return " ROLLBACK INCOMPLETE: " + "; ".join(leftovers) + "."
    return ""


def refill_matrix(
    repo_root: Path,
    exp_id: str,
    cells: list[str] | None,
    *,
    image_digest: str,
    artifact_bucket: str | None = None,
    tracking_uri: str = "http://localhost:5001",
    container_tracking_uri: str | None = None,
    scenario_dir: str = "rmc/scenarios",
    _store: ObjectStore,
    _batch: BatchSubmitter,
    _mlflow: Any,
    _git: Any = git_helper,
    _no_push: bool = False,
    branch: str | None = None,
    allow_digest_mismatch: bool = False,
) -> dict[str, Any]:
    """Re-run the missing cells of a prior sweep (see module docstring).

    ``cells`` names the analysis-bearing cells to refill (array indices or
    unit_ids). Omit it (None/empty) to refill EVERY currently-missing cell. When
    given, it must name exactly the missing set: any named cell that is already
    done refuses (no silent duplicate), and any missing cell not named refuses
    (forcing an explicit, pre-registered refill contract).
    """
    repo_root = Path(repo_root)
    if artifact_bucket is None:
        artifact_bucket = Config().artifact_bucket

    effective_container_tracking_uri = container_tracking_uri or tracking_uri
    container_host = urlsplit(effective_container_tracking_uri).hostname
    if container_host in ("localhost", "127.0.0.1"):
        raise MatrixLaunchError(
            f"container_tracking_uri resolves to {effective_container_tracking_uri!r}, "
            f"whose host ({container_host!r}) is the operator's localhost — an AWS Batch "
            "container runs inside the VPC and can never reach it."
        )

    doc = parse_matrix(_find_matrix_doc(repo_root, exp_id))

    if not _git.working_tree_clean(repo_root, ignore=_read_ignore(repo_root)):
        raise MatrixLaunchError("working tree has uncommitted changes; commit before refill")
    sha = _git.head_sha(repo_root)
    push_branch = None
    if not _no_push:
        try:
            push_branch = _git.resolve_push_branch(repo_root, branch)
        except Exception as e:
            raise MatrixLaunchError(f"cannot determine Git push branch: {e}") from e

    # The image guard applies to refills too — a refill on a stale job definition is
    # the same silent-wrong-image hazard.
    _verify_job_def_image(_batch, doc.job_definition, image_digest, allow_mismatch=allow_digest_mismatch)

    current_units = expand_matrix(
        doc.defenses, doc.scenarios, doc.seeds, doc.mode, doc.max_per_client,
        doc.rounds, repeats=doc.repeats,
    )

    prior = _read_prior_manifest(_store, exp_id)
    if prior is None:
        raise MatrixLaunchError(
            f"{exp_id} has no launch manifest — there is nothing to refill. Launch it "
            "with `praxis exp launch-matrix` first, or register a NEW EXP id."
        )
    prior_meta, prior_units = prior

    # invariant 2: the doc must still describe the ORIGINAL matrix.
    _assert_matrix_unchanged(exp_id, prior_units, current_units)

    # Which cells are actually missing (read-only S3 done-marker probe).
    missing = [u for u in current_units if not is_done(_store, exp_id, u.unit_id)]
    if not missing:
        raise MatrixLaunchError(
            f"{exp_id} is fully complete (all {len(current_units)} cells have done-markers) "
            "— nothing to refill. New data requires a NEW EXP id."
        )

    if cells:
        targets = _resolve_cells(exp_id, cells, prior_units)
        # (b) refuse any named cell that already has artifacts — no silent duplicate.
        already = sorted(t.unit_id for t in targets if is_done(_store, exp_id, t.unit_id))
        if already:
            raise MatrixLaunchError(
                f"{exp_id} refill refused: named cell(s) already have artifacts (done-marker "
                f"present): {', '.join(already[:5])}{' …' if len(already) > 5 else ''}. Refilling "
                "them would risk a silent duplicate — remove them from --cells."
            )
        # Force an explicit contract: every missing cell must be named.
        target_ids = {t.unit_id for t in targets}
        undeclared = sorted(u.unit_id for u in missing if u.unit_id not in target_ids)
        if undeclared:
            raise MatrixLaunchError(
                f"{exp_id} refill refused: these cell(s) are ALSO missing but not in --cells: "
                f"{', '.join(undeclared[:5])}{' …' if len(undeclared) > 5 else ''}. The full-array "
                "resubmit would refill them too — add them to --cells (or omit --cells to refill "
                "all missing) so the refill contract is explicit."
            )
    else:
        # No --cells: refill every currently-missing cell.
        targets = list(missing)

    target_ids = sorted(t.unit_id for t in targets)

    # invariant 3: provenance policy.
    provenance_change, prior_launches = _refill_provenance(
        exp_id, prior_meta,
        methodology_version=doc.methodology_version, git_sha=sha, image_digest=image_digest,
    )

    # invariant 5: a provenance-changing refill must wait for the prior array to
    # be terminal (an old RUNNABLE child could win the done-marker race under
    # the old image while this refill advertises new provenance).
    if provenance_change:
        # Baseline = the MOST RECENT array for this sweep (newest refill, else the
        # original), not always the original manifest — a still-running earlier
        # refill under old provenance must block this one.
        baseline_array_id, baseline_src = _latest_array_id(_store, exp_id, prior_meta)
        if not baseline_array_id:
            raise MatrixLaunchError(
                f"{exp_id} refill refused: provenance changed ({provenance_change}) but no "
                "array_job_id is recorded for the most recent launch, so its Batch array cannot "
                "be verified terminal (invariant 5). This manifest predates array-id recording — "
                "a NEW EXP id is the safe path for a provenance-changing refill here."
            )
        if not _batch.array_terminal(baseline_array_id):
            raise MatrixLaunchError(
                f"{exp_id} refill refused: the most recent Batch array {baseline_array_id} "
                f"({baseline_src}) is not terminal, and this refill changes provenance "
                f"({provenance_change}) — an old-provenance child could win the done-marker race. "
                "Cancel/await that array first."
            )

    # invariant 6: this refill mints its OWN serial tag with its OWN annotation.
    launch_tag = _resolve_launch_tag(_git, repo_root, exp_id)
    serial = _serial_suffix(launch_tag, exp_id)

    # (d) parent-run linkage: nest refilled children under the ORIGINAL parent
    # run so the sweep's parent shows all cells. Legacy manifests without a
    # recorded parent_run_id get a cross-linked refill parent + a LOUD warning.
    exp_name = doc.slug
    parent_run = prior_meta.get("parent_run_id")
    minted_parent = False
    if not parent_run:
        print(
            f"[refill] WARNING: {exp_id}'s manifest records no parent_run_id (pre-dates "
            "refill metadata) — minting a NEW cross-linked refill parent run instead of "
            "reusing the original. The refilled children will nest under it, not the "
            f"original sweep parent. AUDIT TRAIL: the new parent carries refill_of_exp="
            f"{exp_id} and refill_serial={serial} tags (and git_tag {launch_tag}) — walk "
            "those back to this sweep."
        )
        ml_exp_id = _mlflow.get_or_create_experiment(exp_name)
        parent_run = _mlflow.create_run(ml_exp_id, tags={
            "mlflow.runName": f"{exp_id} (refill {serial})",
            "exp_id": exp_id,
            "slug": doc.slug,
            "refill_of_exp": exp_id,
            "refill_serial": serial,
            "methodology_version": doc.methodology_version,
            "git_sha": sha,
            "image_digest": image_digest,
            "git_tag": launch_tag,
        })
        minted_parent = True

    # (c) refill contract: the pre-registered record of which cells this refill
    # is analysis-bearing for. Written to a refill-scoped key — the ORIGINAL
    # manifest is never touched (invariant 4).
    launched_at = datetime.utcnow().isoformat() + "Z"
    refill_meta: dict[str, Any] = {
        "refill_of": exp_id,
        "serial": serial,
        "methodology_version": doc.methodology_version,
        "git_sha": sha,
        "image_digest": image_digest,
        "git_tag": launch_tag,
        "parent_run_id": parent_run,
        "minted_parent": minted_parent,
        "n_units_full_array": len(current_units),
        "refilled_cells": target_ids,
        "launched_at": launched_at,
        "provenance_changed": provenance_change,
        # A refill re-runs against the original manifest, so the
        # container reproduces the original run_extras (e.g. SMOTE) by
        # construction — never dropping it and corrupting the A/B. Copied into
        # the refill record for the audit trail so the record self-documents the
        # arm it reproduced.
        "run_extras": prior_meta.get("run_extras", {}),
        # Record the mechanism so a future analyst does not look for padding
        # replicates (ratified 2026-07-24): in-namespace refills obsolete the
        # EXP-012/013 pad-to-array-floor workaround entirely.
        "refill_semantics": (
            "in-namespace full-array resubmit; already-done cells skip on their "
            "done-markers (idempotent no-op, no re-run); NO padding replicates"
        ),
    }
    if provenance_change:
        refill_meta["prior_launches"] = prior_launches
    record_key = _refill_record_key(exp_id, serial)
    record_payload = {
        "exp_id": exp_id, "meta": refill_meta,
        "units": [
            {"unit_id": u.unit_id, "array_index": u.array_index} for u in targets
        ],
    }
    _store.put_bytes(record_key, json.dumps(record_payload, indent=2).encode())

    tag_msg = (
        f"{launch_tag} - {doc.slug} (refill {serial}: {len(target_ids)} cell(s))\n\n"
        f"refill_of: {exp_id}\ngit_sha: {sha}\nimage_digest: {image_digest}\n"
        f"mlflow_experiment: {exp_name}\nmlflow_parent_run_id: {parent_run}\n"
        f"refilled_cells: {', '.join(target_ids)}\nrefill_record: {record_key}\n"
    )
    created_tag: str | None = None
    try:
        _git.create_annotated_tag(repo_root, launch_tag, tag_msg)
        created_tag = launch_tag
        if not _no_push:
            _git.push_with_tags(repo_root, branch=push_branch)
    except Exception as e:
        leftover = _rollback_refill(
            _store=_store, _mlflow=_mlflow, _git=_git, repo_root=repo_root,
            record_key=record_key, created_tag=created_tag,
            parent_run=parent_run, terminate_parent=minted_parent,
        )
        raise MatrixLaunchError(
            f"git tag/push failed — refill rolled back (record deleted"
            f"{', refill parent terminated FAILED' if minted_parent else ''}); the array "
            f"was NOT submitted: {e}{leftover}"
        ) from e

    # Re-submit the FULL canonical array; done cells skip on their markers,
    # missing cells run. PRAXIS_EXP_ID stays exp_id so results/markers land in
    # sweeps/{exp_id}/ at the same unit_id keys; PRAXIS_PARENT_RUN_ID nests the
    # refilled children under the original (or minted) parent run.
    try:
        array_job_id = _batch.submit_array(
            job_name=f"{exp_id}-{doc.slug}-refill-{serial}",
            job_queue=doc.job_queue,
            job_definition=doc.job_definition,
            size=len(current_units),
            tags=_batch_resource_tags(exp_id),
            environment={
                "PRAXIS_EXP_ID": exp_id,
                "PRAXIS_BUCKET": artifact_bucket,
                "PRAXIS_MANIFEST_KEY": manifest_key(exp_id),
                "PRAXIS_MLFLOW_EXPERIMENT": exp_name,
                "MLFLOW_TRACKING_URI": effective_container_tracking_uri,
                "PRAXIS_IMAGE_DIGEST": image_digest,
                # Launch-side provenance: the launch git HEAD
                # is provenance for the LAUNCH inputs (manifest, scenario JSONs),
                # recorded as launch_commit. NOT injected as PRAXIS_RUNNER_COMMIT
                # — that would override the image-baked runner commit and record
                # false provenance for the running code.
                "PRAXIS_LAUNCH_COMMIT": sha,
                "PRAXIS_SCENARIO_DIR": scenario_dir,
                "PRAXIS_PARENT_RUN_ID": parent_run,
                "PRAXIS_METHODOLOGY_VERSION": doc.methodology_version,
            },
        )
    except Exception as e:
        # The launch tag stays as the record of the attempt; the refill record
        # is deleted so a retry is clean.
        leftover = _rollback_refill(
            _store=_store, _mlflow=_mlflow, _git=_git, repo_root=repo_root,
            record_key=record_key, created_tag=None,  # launch tag stays
            parent_run=parent_run, terminate_parent=minted_parent,
        )
        raise MatrixLaunchError(
            f"Batch submit failed — refill rolled back (record deleted"
            f"{', refill parent terminated FAILED' if minted_parent else ''}; launch tag "
            f"{launch_tag} remains): {e}{leftover}"
        ) from e

    # POST-SUBMIT bookkeeping: the array is now LIVE. Every
    # write below is BEST-EFFORT — a failure here must NOT surface as a failed
    # launch, because an operator retry would submit a SECOND concurrent array
    # (same-provenance refills skip the terminal-array check). Log loudly and
    # return success carrying any warnings instead.
    bookkeeping_warnings: list[str] = []

    def _bk(desc: str, fn: Any) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - deliberately swallow post-submit
            bookkeeping_warnings.append(f"{desc} ({e})")
            print(f"[refill] WARN: post-submit bookkeeping failed — {desc}: {e}")

    # repoint the parent's batch_array_job_id to THIS refill
    # array so self-heal (finalizer + scheduled reaper) gates reconciliation on
    # the LIVE array, not the terminal original — otherwise it can conclude the
    # sweep is done and seal the live, markerless refill children FAILED.
    _bk(
        "repoint parent batch_array_job_id to the refill array (IF THIS DID NOT "
        "LAND, self-heal may prematurely reconcile the live refill children as "
        f"failed zombies — verify batch_array_job_id={array_job_id} on parent "
        f"{parent_run})",
        lambda: _mlflow.set_tag(parent_run, "batch_array_job_id", array_job_id),
    )
    _bk("set refill array-id tag",
        lambda: _mlflow.set_tag(parent_run, f"refill_{serial}_array_job_id", array_job_id))
    _bk("set refill cells tag",
        lambda: _mlflow.set_tag(parent_run, f"refill_{serial}_cells", ",".join(target_ids)))
    if provenance_change:
        _bk("set refill provenance-change tag",
            lambda: _mlflow.set_tag(parent_run, f"refill_{serial}_provenance_change", provenance_change))

    # persist THIS refill's array id into its own record so a
    # LATER provenance-changing refill uses it (not the always-terminal original)
    # as the terminality baseline.
    def _persist_array_id() -> None:
        rec = json.loads(_store.get_bytes(record_key))
        rec["meta"]["array_job_id"] = array_job_id
        _store.put_bytes(record_key, json.dumps(rec, indent=2).encode())

    _bk(
        "persist array_job_id into the refill record (IF THIS DID NOT LAND, a "
        "later provenance-changing refill falls back to the original array as its "
        "terminality baseline and may not wait for this one)",
        _persist_array_id,
    )

    return {
        "exp_id": exp_id, "experiment_name": exp_name, "serial": serial,
        "array_job_id": array_job_id, "array_size": len(current_units),
        "refilled_cells": target_ids, "n_refilled": len(target_ids),
        "parent_run_id": parent_run, "minted_parent": minted_parent,
        "git_sha": sha, "git_tag": launch_tag, "image_digest": image_digest,
        "refill_record_key": record_key, "provenance_change": provenance_change,
        "bookkeeping_warnings": bookkeeping_warnings,
    }

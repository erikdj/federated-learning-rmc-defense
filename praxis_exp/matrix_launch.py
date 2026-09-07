"""``praxis exp launch-matrix EXP-NNN`` — fan out a sweep matrix to AWS Batch.

Mirrors launch.py's dependency-injection contract: all external systems
(object store, Batch, MLflow, git) are injected so the orchestration is unit
tested without AWS. The container (SP3) creates per-unit MLflow runs and does
the idempotent-skip; this function creates the experiment, writes the manifest,
tags the launch, and submits the array.

The initial launch path permits one launch per EXP ID. A prior manifest routes
operators to the explicit ``--refill`` workflow, which verifies the original
matrix and provenance before resubmitting missing cells.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from praxis_exp import git_helper
from praxis_exp.batch import BatchSubmitter
from praxis_exp.config import Config
from praxis_exp.manifest import read_manifest, write_manifest
from praxis_exp.matrix_doc import parse_matrix
from praxis_exp.mlflow_enrichment import (
    build_experiment_description,
    build_parent_run_params,
    parent_run_s3_tags,
)
from praxis_exp.storage import (
    ObjectNotFoundError,
    ObjectStore,
    manifest_key,
    sweep_prefix,
)
from praxis_exp.units import Unit, expand_matrix


class MatrixLaunchError(RuntimeError):
    """Raised when a matrix launch precondition fails."""


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


def _read_prior_manifest(
    store: ObjectStore, exp_id: str,
) -> tuple[dict[str, Any], list[Unit]] | None:
    """The prior launch's (meta, units) if a manifest exists in the namespace,
    None if genuinely absent. The probe is unconditional: it is never gated on
    done-marker counts.

    A manifest that exists but cannot be parsed refuses outright: an
    unreadable registry over a possibly-populated namespace is unaccountable
    state.
    """
    try:
        _, meta, units = read_manifest(store, exp_id)
        return meta, units
    except ObjectNotFoundError:
        return None
    except Exception as e:
        raise MatrixLaunchError(
            f"{exp_id} has a prior manifest that cannot be parsed ({e}); "
            "nothing reliable records what the namespace contains — "
            "unaccountable state. New data requires a new EXP id."
        ) from e


def _digest_token(ref: str) -> str:
    """The bare digest token of a reference: the part after the last ``@`` if the
    caller passed a full ``repo@sha256:…`` reference, else the string itself
    (already a ``sha256:…`` digest). Used so the guard compares digest EQUALITY,
    never a substring."""
    return ref.rsplit("@", 1)[-1].strip()


def _verify_job_def_image(
    _batch: BatchSubmitter, job_definition: str, image_digest: str,
    *, allow_mismatch: bool,
) -> None:
    """Fail fast when the job definition's container image does not
    match the requested ``--image-digest``.

    ``--image-digest`` is provenance-only — it is recorded in tags/env but does
    NOT select the image AWS Batch runs; the job definition's
    ``containerProperties.image`` does. Launching with a fresh digest while the
    job def still points at the old image would silently run the OLD image while
    recording the NEW digest: a chain-of-custody break (and, for a manifest that
    depends on the new image, an outright wrong measurement).

    A job def pinned by digest (``…@sha256:…``) must match ``image_digest`` by
    EQUALITY of the parsed digest — not a substring test, which would accept a
    truncated request (``sha256:abc`` against a real ``sha256:abcdef…``) and then
    record provenance that does not identify the image actually run. A tag-pinned job def (no ``@sha256:``) is UNVERIFIABLE — the tag could
    resolve to any image — and is treated as a mismatch. ``None`` from the
    submitter is the fake's opt-out (guard skipped). ``--allow-digest-mismatch``
    downgrades the hard failure to a LOUD warning.
    """
    image_ref = _batch.job_definition_image(job_definition)
    if not isinstance(image_ref, str):
        return  # submitter did not resolve a concrete image (fake None / mock) — opt out
    requested = _digest_token(image_digest)
    # The job def image digest is the reference AFTER the '@'; a tag-pinned ref
    # (no '@') has no verifiable digest and falls through to the mismatch path.
    actual = _digest_token(image_ref.rsplit("@", 1)[1]) if "@" in image_ref else None
    if actual is not None and actual == requested:
        return  # job def is pinned to EXACTLY this digest
    detail = (
        f"job definition {job_definition!r} runs image {image_ref!r}, which does "
        f"not match the requested --image-digest {image_digest!r}. The job def, "
        "not --image-digest, selects the image AWS Batch runs — this launch would "
        "run the job def's image while recording the requested digest (silent "
        "chain-of-custody break). Redeploy the job def to the requested image "
        "(scripts/aws/batch/deploy_stack.sh) and point the doc's job_definition "
        "at the new revision, or pass --allow-digest-mismatch to override."
    )
    if allow_mismatch:
        print(
            "[launch-matrix] *** WARNING: --allow-digest-mismatch — LAUNCHING ANYWAY "
            f"despite an image mismatch. {detail}"
        )
        return
    raise MatrixLaunchError(detail)


def _read_ignore(repo_root: Path) -> list[str]:
    f = repo_root / ".experiment-ignore"
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def _find_matrix_doc(repo_root: Path, exp_id: str) -> Path:
    exp_dir = repo_root / "docs" / "experiments"
    for f in exp_dir.iterdir():
        if f.is_file() and f.name.startswith(f"{exp_id}-") and not f.name.endswith("-result.md"):
            return f
    raise MatrixLaunchError(f"no design doc found for {exp_id} in {exp_dir}")


# Cap on serial tag suffixes — purely a runaway guard.
_MAX_TAG_SERIAL = 100


def _resolve_launch_tag(_git: Any, repo_root: Path, exp_id: str) -> str:
    """Pick the immutable git tag this launch will create.

    Existing ``exp/`` tags are immutable chain-of-custody anchors — never
    force-moved — and their annotations embed launch-specific facts (parent
    run id, image digest, manifest key). NO existing annotation is ever
    reused: reuse would point auditors at a previous launch's record (e.g.
    after a manual namespace clear). ``exp/{exp_id}`` absent -> use it;
    otherwise the next free serial ``exp/{exp_id}.rN`` (N = 2, 3, ...).
    Every launch creates its own tag with its own annotation.
    """
    base = f"exp/{exp_id}"
    if _git.tag_target_sha(repo_root, base) is None:
        return base
    for n in range(2, _MAX_TAG_SERIAL + 1):
        candidate = f"{base}.r{n}"
        if _git.tag_target_sha(repo_root, candidate) is None:
            return candidate
    raise MatrixLaunchError(
        f"more than {_MAX_TAG_SERIAL} serial launch tags exist for {base}; refusing to continue"
    )


def _rollback_launch(
    exp_id: str, *, _store: ObjectStore, _mlflow: Any,
    _git: Any = None, repo_root: Path | None = None,
    parent_run: str | None = None, created_tag: str | None = None,
) -> str:
    """Best-effort rollback of everything this launch created.

    Order: terminate parent FAILED -> delete the tag this launch created ->
    delete this launch's manifest. Deleting the manifest is safe by
    construction: the one-launch-per-EXP guard proved the namespace fresh at
    launch start, THIS launch wrote the manifest, and no Batch array was
    submitted on any path that reaches here — so no child can be reading it.

    Every step is individually guarded: a failing rollback step must never
    mask the original error. Returns a suffix describing anything left
    behind ('' when the rollback was clean) for the caller to append to its
    raised message.
    """
    leftovers: list[str] = []
    if parent_run is not None:
        try:
            _mlflow.set_terminated(parent_run, "FAILED")
        except Exception as e:
            leftovers.append(f"parent run {parent_run} could not be terminated ({e})")
    if created_tag is not None:
        try:
            _git.delete_local_tag(repo_root, created_tag)
        except Exception as e:
            leftovers.append(f"local tag {created_tag} could not be deleted ({e})")
    try:
        _store.delete(manifest_key(exp_id))
    except Exception as e:
        leftovers.append(
            f"manifest {manifest_key(exp_id)} could not be deleted ({e}) — "
            "manually delete it before relaunching (see the one-launch-per-EXP runbook)"
        )
    if leftovers:
        return " ROLLBACK INCOMPLETE: " + "; ".join(leftovers) + "."
    return ""


def launch_matrix(
    repo_root: Path,
    exp_id: str,
    *,
    image_digest: str,
    # None -> resolved via Config (env var > praxis_exp/local_defaults);
    # no private bucket default may be baked here (ships in the public release)
    artifact_bucket: str | None = None,
    tracking_uri: str = "http://localhost:5001",
    # MLflow URI for the Batch CONTAINER environment (VPC-reachable). Containers
    # can never reach the operator's localhost SSM tunnel used by tracking_uri
    # above. None -> falls back to tracking_uri (guarded below).
    container_tracking_uri: str | None = None,
    # repo-relative; entrypoint.py builds --scenario as <dir>/<unit.scenario>.json
    scenario_dir: str = "rmc/scenarios",
    _store: ObjectStore,
    _batch: BatchSubmitter,
    # _mlflow must provide: get_or_create_experiment, create_run, set_tag,
    # set_terminated, set_experiment_tag, log_params
    _mlflow: Any,
    _git: Any = git_helper,
    _no_push: bool = False,
    branch: str | None = None,
    # Override the job-definition image-digest guard (logs loudly, never silently).
    allow_digest_mismatch: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    if artifact_bucket is None:
        artifact_bucket = Config().artifact_bucket

    effective_container_tracking_uri = container_tracking_uri or tracking_uri
    container_host = urlsplit(effective_container_tracking_uri).hostname
    if container_host in ("localhost", "127.0.0.1"):
        raise MatrixLaunchError(
            f"container_tracking_uri resolves to {effective_container_tracking_uri!r}, "
            f"whose host ({container_host!r}) is the operator's localhost — an AWS "
            "Batch container runs inside the VPC and can never reach it. Set "
            "PRAXIS_CONTAINER_MLFLOW_URI to a VPC-reachable MLflow address (or "
            "configure praxis_exp/local_defaults.CONTAINER_MLFLOW_URI)."
        )

    doc = parse_matrix(_find_matrix_doc(repo_root, exp_id))

    if not _git.working_tree_clean(repo_root, ignore=_read_ignore(repo_root)):
        raise MatrixLaunchError("working tree has uncommitted changes; commit before launch")

    sha = _git.head_sha(repo_root)
    push_branch = None
    if not _no_push:
        try:
            push_branch = _git.resolve_push_branch(repo_root, branch)
        except Exception as e:
            raise MatrixLaunchError(f"cannot determine Git push branch: {e}") from e

    units = expand_matrix(
        doc.defenses, doc.scenarios, doc.seeds, doc.mode, doc.max_per_client,
        doc.rounds, repeats=doc.repeats,
    )

    if len(units) < 2:
        raise MatrixLaunchError(
            f"AWS Batch requires array size >= 2; this matrix expands to {len(units)} unit(s). "
            "Use `praxis exp launch` for a single-unit run."
        )

    # The job definition, not --image-digest, selects the image AWS Batch
    # runs — verify they agree BEFORE any side effect so a stale job def fails
    # fast instead of silently running the wrong image.
    _verify_job_def_image(
        _batch, doc.job_definition, image_digest, allow_mismatch=allow_digest_mismatch,
    )

    # Namespace pre-flight, BEFORE any side effect. ONE LAUNCH PER EXP ID
    # (see module docstring): any prior manifest refuses; done-markers without a
    # readable manifest are unaccountable state and also refuse.
    if _read_prior_manifest(_store, exp_id) is not None:
        raise MatrixLaunchError(
            f"{exp_id} already has a launch manifest ({manifest_key(exp_id)}) — "
            "a normal launch remains one launch per EXP id. Use "
            f"`praxis exp launch-matrix {exp_id} --refill` to resubmit the "
            "missing cells under the existing manifest, optionally with --cells. "
            "For a changed matrix, register a NEW EXP id. For hard-crash "
            "debris only (transient failures roll their own "
            "manifest back — a leftover manifest means the process was killed "
            "before rollback could run): verify the parent run is FAILED or "
            f"absent and no Batch array exists for {exp_id}, then archive and "
            f"clear the ENTIRE sweeps/{exp_id}/ prefix (not just manifest.json) "
            "and relaunch."
        )
    # Namespace-emptiness check: the old
    # probe asked is_done only for the NEW expansion's unit_ids — after a
    # manual manifest removal with a changed matrix, old markers/results
    # under OTHER unit ids evaded it and artifacts would mix. "No object
    # under the prefix" has no unit-id blind spots and subsumes the old
    # markers-without-manifest refusal (markers live under this prefix).
    namespace_prefix = f"{sweep_prefix(exp_id)}/"
    stray = _store.find_keys(namespace_prefix, limit=25)
    if stray:
        count = f"{len(stray)}{'+' if len(stray) == 25 else ''}"
        raise MatrixLaunchError(
            f"{exp_id} namespace is not empty: {count} object(s) under "
            f"{namespace_prefix} with no readable manifest (first: {stray[0]}) — "
            "unaccountable state (e.g. debris left by a manual manifest delete "
            "with a changed matrix). Register a NEW EXP id, or for verified "
            f"debris archive and clear the ENTIRE {namespace_prefix} prefix, "
            "then relaunch."
        )

    meta: dict[str, Any] = {
        "methodology_version": doc.methodology_version,
        "image_digest": image_digest,
        "git_sha": sha,
        "n_units": len(units),
        "launched_at": datetime.utcnow().isoformat() + "Z",
        # Experiment-level run-config overrides (e.g. SMOTE). Carried in
        # the manifest so docker/entrypoint.py::runner_argv can turn them into
        # runner CLI flags for every unit; {} keeps the argv byte-identical to
        # the incumbent when the doc declares no run_extras.
        "run_extras": doc.run_extras,
    }
    # Rollback contract: the manifest is
    # written first, and from here to the Batch submit EVERY failure path
    # rolls back what this launch created (parent run FAILED -> created tag
    # -> manifest) via _rollback_launch — a transient error leaves a clean
    # namespace for retry instead of tripping the one-launch-per-EXP guard.
    # (The old "benign, overwritten on retry" note died with the round-9
    # hard refusal.) Rollback leaves NO other namespace objects by
    # construction: write_manifest below is this function's ONLY store
    # write pre-submit (everything else is a read or the rollback delete),
    # so a rolled-back launch also passes the namespace-emptiness check
    # above on retry.
    write_manifest(_store, exp_id, units, meta)

    exp_name = doc.slug
    parent_run: str | None = None
    try:
        # Requirement: if we're running the same experiment multiple
        # times, it should be a new RUN" — the experiment is the design
        # FAMILY (doc.slug); each launch adds a new parent run instead of
        # minting a fresh experiment per exp_id.
        ml_exp_id = _mlflow.get_or_create_experiment(exp_name)
        # req 3: experiment-level metadata (description, hypothesis,
        # dataset). Set on every launch (idempotent overwrite) so design-doc
        # edits keep the experiment's description current.
        _mlflow.set_experiment_tag(
            ml_exp_id, "mlflow.note.content",
            build_experiment_description(doc, repo_root, doc.path),
        )
        _mlflow.set_experiment_tag(ml_exp_id, "hypothesis", doc.hypothesis)
        _mlflow.set_experiment_tag(ml_exp_id, "dataset", "edge_full_20_rmc")

        # Every launch creates its OWN tag (annotations are never reused —
        # they embed this launch's parent run id, image digest, manifest key).
        launch_tag = _resolve_launch_tag(_git, repo_root, exp_id)

        parent_run = _mlflow.create_run(ml_exp_id, tags={
            # req 1: parent run is NAMED EXP-NNN — re-launches become
            # additional runs in this experiment, distinguished by launched_at.
            "mlflow.runName": exp_id,
            "exp_id": exp_id,
            "slug": doc.slug,
            "hypothesis": doc.hypothesis,
            "methodology_version": doc.methodology_version,
            "git_sha": sha,
            "image_digest": image_digest,
            "n_units": str(len(units)),
            "launched_at": datetime.utcnow().isoformat() + "Z",
            # the ACTUAL tag anchoring this launch (may be exp/EXP-NNN.rN)
            "git_tag": launch_tag,
            # req 5: S3 links so MLflow is a functional directory, not just
            # a bucket-name param.
            **parent_run_s3_tags(artifact_bucket, exp_id),
        })
        # req 3: the swept matrix as PARAMS (what was actually run),
        # separate from the identity tags above.
        _mlflow.log_params(parent_run, build_parent_run_params(doc, len(units)))
    except Exception as e:
        leftover = _rollback_launch(
            exp_id, _store=_store, _mlflow=_mlflow, parent_run=parent_run,
        )
        raise MatrixLaunchError(
            f"MLflow setup/enrichment failed — rolled back (parent run "
            f"{'terminated FAILED' if parent_run else 'was never created'}, this "
            "launch's manifest deleted); no tag was created and the array was "
            f"NOT submitted: {e}{leftover}"
        ) from e

    tag_msg = (
        f"{launch_tag} - {doc.slug} (matrix sweep, {len(units)} units)\n\n"
        f"git_sha: {sha}\nimage_digest: {image_digest}\n"
        f"mlflow_experiment: {exp_name}\nmlflow_parent_run_id: {parent_run}\n"
        f"manifest: {manifest_key(exp_id)}\n"
    )
    created_tag: str | None = None
    try:
        _git.create_annotated_tag(repo_root, launch_tag, tag_msg)
        created_tag = launch_tag
        if not _no_push:
            _git.push_with_tags(repo_root, branch=push_branch)
    except Exception as e:
        # Roll back only what this launch created; pre-existing exp/ tags
        # are other launches' custody anchors and are never touched.
        leftover = _rollback_launch(
            exp_id, _store=_store, _mlflow=_mlflow, _git=_git,
            repo_root=repo_root, parent_run=parent_run, created_tag=created_tag,
        )
        raise MatrixLaunchError(
            f"git tag/push failed — rolled back (parent run {parent_run} terminated "
            "FAILED, this launch's tag and manifest deleted); the array was NOT "
            f"submitted: {e}{leftover}"
        ) from e

    try:
        array_job_id = _batch.submit_array(
            job_name=f"{exp_id}-{doc.slug}",
            job_queue=doc.job_queue,
            job_definition=doc.job_definition,
            size=len(units),
            tags=_batch_resource_tags(exp_id),
            environment={
                "PRAXIS_EXP_ID": exp_id,
                "PRAXIS_BUCKET": artifact_bucket,
                "PRAXIS_MANIFEST_KEY": manifest_key(exp_id),
                "PRAXIS_MLFLOW_EXPERIMENT": exp_name,
                "MLFLOW_TRACKING_URI": effective_container_tracking_uri,
                "PRAXIS_IMAGE_DIGEST": image_digest,
                # Launch-side provenance: the launch git HEAD
                # is legitimate provenance for the LAUNCH inputs (manifest,
                # scenario JSONs) — recorded as launch_commit. It must NOT be
                # injected as PRAXIS_RUNNER_COMMIT: that Batch env would override
                # the image-baked runner commit and record false provenance for
                # the running code (the container bakes the SHA at build time).
                "PRAXIS_LAUNCH_COMMIT": sha,
                "PRAXIS_SCENARIO_DIR": scenario_dir,
                # req C: lets entrypoint.py nest each unit's child run under
                # this sweep's parent run (mlflow.parentRunId tag) instead of
                # a flat list of unrelated runs.
                "PRAXIS_PARENT_RUN_ID": parent_run,
                # req C: units tag themselves with the methodology version
                # active at launch without re-parsing the design doc.
                "PRAXIS_METHODOLOGY_VERSION": doc.methodology_version,
            },
        )
    except Exception as e:
        # The launch tag stays as the record of the attempt; the
        # manifest is deleted so a retry passes the one-launch-per-EXP guard
        # (a subsequent launch mints the next serial tag).
        leftover = _rollback_launch(
            exp_id, _store=_store, _mlflow=_mlflow, parent_run=parent_run,
        )
        raise MatrixLaunchError(
            f"Batch submit failed — rolled back (parent run {parent_run} terminated "
            f"FAILED, this launch's manifest deleted; launch tag {launch_tag} "
            f"remains): {e}{leftover}"
        ) from e
    _mlflow.set_tag(parent_run, "batch_array_job_id", array_job_id)

    # POST-SUBMIT (past the rollback window — the launch has already succeeded):
    # stamp the manifest meta with this launch's parent_run_id and array_job_id
    # so a later `refill` can (d) nest refilled children under the SAME parent
    # run and (invariant 5) verify this array is terminal before a
    # provenance-changing refill. Best-effort: a failure here does not fail the
    # launch (the array is live); the refill path degrades loudly if absent.
    try:
        meta_with_ids = {**meta, "parent_run_id": parent_run, "array_job_id": array_job_id}
        write_manifest(_store, exp_id, units, meta_with_ids)
    except Exception as e:  # pragma: no cover - defensive; launch already succeeded
        print(
            f"[launch-matrix] WARN: could not stamp parent_run_id/array_job_id into "
            f"the manifest meta ({e}); a later refill will mint a cross-linked parent "
            "run instead of reusing this one"
        )

    return {
        "exp_id": exp_id, "experiment_name": exp_name, "n_units": len(units),
        "array_job_id": array_job_id, "manifest_key": manifest_key(exp_id),
        "git_sha": sha, "git_tag": launch_tag, "image_digest": image_digest,
    }

"""``praxis`` CLI entry point dispatching to subcommands."""
from pathlib import Path
import os
import sys
import click

from praxis_exp.scaffold import scaffold_experiment
from praxis_exp.launch import launch_experiment
from praxis_exp.ingest import ingest_experiment
from praxis_exp.enrich import enrich_experiment
from praxis_exp.round_trace import tabulate_experiment, cleanup_traces
from praxis_exp.promote_models import promote_models
from praxis_exp.listing import list_experiments
from praxis_exp.matrix_launch import launch_matrix
from praxis_exp.matrix_refill import refill_matrix
from praxis_exp.batch import Boto3BatchSubmitter
from praxis_exp.storage import S3ObjectStore
from praxis_exp.mlflow_client import PraxisMlflowClient
from praxis_exp.config import Config


REPO_ROOT = Path(__file__).resolve().parent.parent


@click.group()
def main():
    """praxis - praxis experiment-tracking CLI."""


@main.group()
def exp():
    """Experiment-lifecycle subcommands."""


@exp.command("new")
@click.argument("slug")
def cmd_new(slug):
    """Allocate next EXP-NNN and scaffold a design doc."""
    path = scaffold_experiment(REPO_ROOT, slug)
    click.echo(f"Created: {path}")
    click.echo("Edit the doc, then: praxis exp launch <EXP-ID>")


@exp.command("launch")
@click.argument("exp_id")
def cmd_launch(exp_id):
    """Validate, tag, push, create MLflow run, invoke runner."""
    try:
        out = launch_experiment(REPO_ROOT, exp_id)
        click.echo(f"Launched {exp_id}: run_id={out['run_id']}, sha={out['git_sha'][:8]}")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@exp.command("launch-matrix")
@click.argument("exp_id")
@click.option("--image-digest", required=True,
              help="Pinned container image digest (sha256:...) recorded for chain of custody.")
@click.option("--allow-digest-mismatch", is_flag=True, default=False,
              help="Override the job-definition image guard when its image != --image-digest "
                   "(logs LOUDLY; use only when you know the job def is correct).")
@click.option("--refill", is_flag=True, default=False,
              help="Refill mode: re-run the missing cells of EXP_ID's prior launch "
                   "in place (done cells skip on their markers). Use with --cells.")
@click.option("--cells", default=None,
              help="Refill only: comma-separated cells to refill (array indices or unit_ids). "
                   "Must name exactly the missing set. Omit to refill ALL missing cells.")
@click.option("--branch", default=None, metavar="NAME",
              help="Remote branch to receive the committed launch HEAD. Defaults to the "
                   "current branch; required for a detached HEAD.")
@click.option("--no-push", is_flag=True, default=False,
              help="Create the local custody tag without pushing commits or tags.")
def cmd_launch_matrix(
    exp_id, image_digest, allow_digest_mismatch, refill, cells, branch, no_push,
):
    """Fan a sweep matrix out to AWS Batch (one EXP owns the matrix).

    With --refill, re-run only the missing cells of a prior launch (chain-of-
    custody-preserving refill).
    """
    import boto3
    if cells and not refill:
        click.echo("ERROR: --cells requires --refill", err=True)
        sys.exit(1)
    try:
        cfg = Config()
        # mlflow.log_artifact uploads via boto's DEFAULT credential chain (not our
        # profile session below), so export AWS_PROFILE for it — otherwise artifact
        # upload fails "Unable to locate credentials" even though the S3 reads (which
        # use the profile session) succeed.
        if cfg.aws_profile:
            os.environ.setdefault("AWS_PROFILE", cfg.aws_profile)
        session = boto3.Session(profile_name=cfg.aws_profile, region_name=cfg.aws_region)
        store = S3ObjectStore(cfg.artifact_bucket, session.client("s3"))
        batch = Boto3BatchSubmitter(session.client("batch"))
        mlflow_client = PraxisMlflowClient(tracking_uri=cfg.tracking_uri)
        if refill:
            cell_list = [c for c in cells.split(",")] if cells else None
            out = refill_matrix(
                REPO_ROOT, exp_id, cell_list, image_digest=image_digest,
                artifact_bucket=cfg.artifact_bucket, tracking_uri=cfg.tracking_uri,
                container_tracking_uri=cfg.container_tracking_uri,
                _store=store, _batch=batch, _mlflow=mlflow_client,
                allow_digest_mismatch=allow_digest_mismatch,
                branch=branch, _no_push=no_push,
            )
            click.echo(
                f"Refilled {out['exp_id']} ({out['serial']}): {out['n_refilled']} cell(s) "
                f"[{', '.join(out['refilled_cells'])}] via full-array job {out['array_job_id']} "
                f"(size {out['array_size']}), parent run {out['parent_run_id']}"
            )
            return
        out = launch_matrix(
            REPO_ROOT, exp_id, image_digest=image_digest,
            artifact_bucket=cfg.artifact_bucket, tracking_uri=cfg.tracking_uri,
            container_tracking_uri=cfg.container_tracking_uri,
            _store=store, _batch=batch, _mlflow=mlflow_client,
            allow_digest_mismatch=allow_digest_mismatch,
            branch=branch, _no_push=no_push,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    click.echo(
        f"Launched {out['exp_id']}: {out['n_units']} units, "
        f"array job {out['array_job_id']}, manifest {out['manifest_key']}"
    )


@exp.command("ingest")
@click.argument("exp_id")
@click.option("--result", type=click.Path(exists=True), default=None)
def cmd_ingest(exp_id, result):
    """Ingest a completed run's artifacts + metrics into MLflow."""
    out = ingest_experiment(REPO_ROOT, exp_id, result_path=Path(result) if result else None)
    click.echo(f"Ingested {exp_id}: criteria_ok={out['criteria_ok']}")


@exp.command("enrich")
@click.argument("exp_id")
def cmd_enrich(exp_id):
    """Backfill/repair the full MLflow record for a completed sweep from S3."""
    import boto3
    try:
        cfg = Config()
        # mlflow.log_artifact uploads via boto's DEFAULT credential chain (not our
        # profile session below), so export AWS_PROFILE for it — otherwise artifact
        # upload fails "Unable to locate credentials" even though the S3 reads (which
        # use the profile session) succeed.
        if cfg.aws_profile:
            os.environ.setdefault("AWS_PROFILE", cfg.aws_profile)
        session = boto3.Session(profile_name=cfg.aws_profile, region_name=cfg.aws_region)
        store = S3ObjectStore(cfg.artifact_bucket, session.client("s3"))
        client = PraxisMlflowClient(tracking_uri=cfg.tracking_uri)
        out = enrich_experiment(
            REPO_ROOT, exp_id, bucket=cfg.artifact_bucket, _store=store, _client=client,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    failed = out.get("units_failed", 0)
    dup = out.get("duplicate_runs_reconciled", 0)
    parent = out.get("parent_run_id")
    click.echo(
        f"Enriched {exp_id}: {out['units_enriched']} enriched, "
        f"{out['units_reconciled']} reconciled, {out['units_skipped']} skipped, "
        f"{failed} failed"
        + (f", {dup} duplicate run(s) reconciled" if dup else "")
        + (f"; parent run {parent} sealed FINISHED" if parent else "")
    )
    # Also (re)build the per-round timeline TABLE (best-effort — a failure must
    # never fail enrich; the dedicated command is `praxis exp timeline`).
    try:
        tout = tabulate_experiment(REPO_ROOT, exp_id, _store=store, _client=client)
        click.echo(
            f"  timeline: {tout['units_tabulated']} tabulated, "
            f"{tout['units_skipped']} skipped, {tout['units_failed']} failed"
        )
    except Exception as e:
        click.echo(f"  timeline: skipped ({e}) — run `praxis exp timeline {exp_id}` to retry", err=True)
    if failed:
        # A partial-failure summary must not read as success — surface it and
        # exit non-zero so the operator re-runs enrich.
        click.echo(f"WARNING: {failed} unit(s) failed to enrich — see logs above", err=True)
        sys.exit(1)


@exp.command("timeline")
@click.argument("exp_id")
def cmd_timeline(exp_id):
    """(Re)build the per-round timeline TABLE (round_timeline.json) for a
    completed sweep from S3.

    Idempotent: mlflow.log_table appends, so each unit's table is refreshed
    (list->delete->log) — re-running never duplicates rows."""
    import boto3
    try:
        cfg = Config()
        # mlflow.log_artifact uploads via boto's DEFAULT credential chain (not our
        # profile session below), so export AWS_PROFILE for it — otherwise artifact
        # upload fails "Unable to locate credentials" even though the S3 reads (which
        # use the profile session) succeed.
        if cfg.aws_profile:
            os.environ.setdefault("AWS_PROFILE", cfg.aws_profile)
        session = boto3.Session(profile_name=cfg.aws_profile, region_name=cfg.aws_region)
        store = S3ObjectStore(cfg.artifact_bucket, session.client("s3"))
        client = PraxisMlflowClient(tracking_uri=cfg.tracking_uri)
        out = tabulate_experiment(
            REPO_ROOT, exp_id, _store=store, _client=client,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    failed = out.get("units_failed", 0)
    click.echo(
        f"Tabulated {exp_id}: {out['units_tabulated']} tabulated, "
        f"{out['units_skipped']} skipped, {failed} failed"
    )
    if failed:
        click.echo(f"WARNING: {failed} unit(s) failed to tabulate — see logs above", err=True)
        sys.exit(1)


@exp.command("cleanup-traces")
@click.argument("exp_id")
def cmd_cleanup_traces(exp_id):
    """One-time removal of RETIRED MLflow traces for a completed sweep — both the
    per-round fl_training__* traces and the coarse run_phase4_flower/persist_unit
    spans. These traces were retired because an FL run has no call tree. Idempotent."""
    try:
        cfg = Config()
        client = PraxisMlflowClient(tracking_uri=cfg.tracking_uri)
        out = cleanup_traces(REPO_ROOT, exp_id, _client=client)
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    click.echo(f"Cleaned traces for {exp_id}: {out['traces_deleted']} deleted")


@exp.command("promote-models")
@click.argument("exp_id")
def cmd_promote_models(exp_id):
    """Set sweep-scoped champion__/challenger__ registry aliases on each defense's
    model (praxis-{defense_token}), ranked by final_f1. Best-effort, idempotent
    operation. Applies only to sweeps whose models were logged in-container."""
    import boto3
    try:
        cfg = Config()
        if cfg.aws_profile:
            os.environ.setdefault("AWS_PROFILE", cfg.aws_profile)
        session = boto3.Session(profile_name=cfg.aws_profile, region_name=cfg.aws_region)
        store = S3ObjectStore(cfg.artifact_bucket, session.client("s3"))
        client = PraxisMlflowClient(tracking_uri=cfg.tracking_uri)
        out = promote_models(REPO_ROOT, exp_id, _store=store, _client=client)
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    click.echo(
        f"Promoted {exp_id}: {out['defenses_promoted']} defense(s) aliased "
        "(champion__/challenger__)"
    )


@exp.command("list")
@click.option("--status", default=None)
@click.option("--methodology", default=None)
@click.option("--hypothesis", default=None)
def cmd_list(status, methodology, hypothesis):
    """List experiments and regenerate docs/experiments/INDEX.md."""
    rows = list_experiments(REPO_ROOT, status=status, methodology=methodology, hypothesis=hypothesis)
    for r in rows:
        click.echo(f"{r['exp_id']:8s}  {r['slug']:40s}  {r['status']:10s}  {r['methodology_version']}")


if __name__ == "__main__":
    main()

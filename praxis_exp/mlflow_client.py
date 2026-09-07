"""Thin wrapper around mlflow.tracking.MlflowClient with praxis defaults."""
from typing import Any, Optional

import hashlib

import mlflow
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import MlflowClient

from praxis_exp import storage
from praxis_exp.config import Config


def build_meta_dataset(*, name: str, source_uri: str, digest: Optional[str] = None) -> Any:
    """Build a native MLflow ``MetaDataset`` for an S3-hosted dataset directory.

    Kept as a module function (not a pure helper in ``mlflow_enrichment``)
    because it imports ``mlflow.data`` — used by both the container entrypoint
    (fluent ``mlflow.log_input``) and the backfill client (``log_input`` below)
    so the dataset object is constructed in exactly one place.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset

    source = resolve_dataset_source(source_uri)
    return MetaDataset(source, name=name, digest=digest)


def build_signal_dataset(
    *, exp_id: str, unit_id: str, bucket: str, defense_token: str,
    digest: Optional[str] = None,
) -> Any:
    """Build a ``MetaDataset`` referencing a unit's signal log **by S3 source**
    (no byte copy). The default digest is a deterministic hash
    of the S3 key so re-logging the same signal dedups in ``log_inputs`` (an
    idempotent re-enrich must not duplicate the input). The signal S3 key is
    derived by ``praxis_exp.storage`` (single source of the key string)."""
    key = storage.signal_key(exp_id, unit_id)
    if digest is None:
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return build_meta_dataset(
        name=f"signal_{defense_token}", source_uri=f"s3://{bucket}/{key}", digest=digest,
    )


class PraxisMlflowClient:
    """Adapter that pre-fills artifact locations and standard tags."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_bucket: Optional[str] = None,
        _client: Optional[MlflowClient] = None,
    ):
        cfg = Config()
        self.tracking_uri = tracking_uri or cfg.tracking_uri
        self.artifact_bucket = artifact_bucket or cfg.artifact_bucket
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = _client or MlflowClient(tracking_uri=self.tracking_uri)

    def get_or_create_experiment(self, name: str) -> str:
        existing = self._client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return self._client.create_experiment(
            name=name,
            artifact_location=f"s3://{self.artifact_bucket}/mlflow/artifacts/{name}/",
        )

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> str:
        run = self._client.create_run(experiment_id=experiment_id, tags=tags)
        return run.info.run_id

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        for k, v in params.items():
            self._client.log_param(run_id, k, str(v))

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self._client.log_metric(run_id, key, value, step=step)

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self._client.set_tag(run_id, key, value)

    def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
        """Tag the experiment itself (not a run) — used for req 3's
        experiment-level metadata: mlflow.note.content (description),
        hypothesis, dataset. Real MlflowClient API, stable since MLflow 1.x."""
        self._client.set_experiment_tag(experiment_id, key, value)

    def log_artifact(self, run_id: str, local_path: str) -> None:
        self._client.log_artifact(run_id, local_path)

    def list_artifacts(self, run_id: str) -> Any:
        """A run's artifacts. ``emit_round_table``'s refresh uses this to detect
        an existing ``round_timeline.json`` — ``log_table`` APPENDS, so the table
        must be deleted before re-logging or rows duplicate."""
        return self._client.list_artifacts(run_id)

    def delete_artifact(self, run_id: str, path: str) -> None:
        """Delete a single artifact from a run via its artifact repository.
        ``MlflowClient`` has no per-artifact delete and
        ``mlflow.artifacts.delete_artifacts`` does not exist, so go through
        ``get_artifact_repository``. Used to refresh ``round_timeline.json`` and
        to drop the legacy ``signal.jsonl`` copy on old-scheme runs (Lane A/B)."""
        get_artifact_repository(self.artifact_uri_for_run(run_id)).delete_artifacts(path)

    def log_table(self, run_id: str, data: Any, artifact_file: str) -> None:
        """Log a column-oriented dict (or DataFrame) as an artifact table that
        renders in the 3.14 artifact browser. NOTE: ``MlflowClient.log_table``
        APPENDS to an existing ``artifact_file`` — callers (``emit_round_table``)
        refresh (delete) first so re-runs do not duplicate rows."""
        self._client.log_table(run_id, data, artifact_file=artifact_file)

    def log_input(self, run_id: str, dataset: Any, context: str = "training") -> None:
        """Attach a native MLflow Dataset (see ``build_meta_dataset``) to a
        specific run via the public ``MlflowClient.log_inputs`` API — the
        backfill path (``praxis_exp.enrich``) needs to log an input to a run
        it did not itself start, so it cannot use the fluent ``mlflow.log_input``
        (which targets the active run)."""
        from mlflow.entities import DatasetInput, InputTag

        entity = dataset._to_mlflow_entity()
        dataset_input = DatasetInput(
            dataset=entity, tags=[InputTag("mlflow.data.context", context)]
        )
        self._client.log_inputs(run_id, [dataset_input])

    def search_runs(
        self, experiment_ids: list[str], *, filter_string: str = "",
        order_by: Optional[list[str]] = None, max_results: int = 1000,
    ) -> Any:
        """Raw passthrough to ``MlflowClient.search_runs`` returning Run objects.

        The self-heal finalizer/reaper (``praxis_exp.selfheal_lambda``) locates
        parent runs by ``batch_array_job_id`` / ``exp_id`` tags and enumerates a
        parent's children; these operations need the full Run objects
        (``info.status``/``start_time``/``experiment_id`` + ``data.tags``), not the
        reduced projections ``find_parent_run``/``find_runs_by_unit`` return. Kept
        generic so the query string lives with the caller."""
        return self._client.search_runs(
            experiment_ids, filter_string=filter_string,
            order_by=order_by or [], max_results=max_results,
        )

    def list_experiment_ids(self) -> list[str]:
        """All (active) experiment ids. The finalizer/reaper searches parent runs by
        tag across EVERY experiment — the Batch event / schedule carries no
        experiment id — so they need the full id list to pass to ``search_runs``."""
        return [e.experiment_id for e in self._client.search_experiments()]

    def find_parent_run(self, experiment_id: str, exp_id: str) -> Optional[str]:
        """The current launch's PARENT run id for an EXP, or ``None``.

        A design-family experiment (slug) holds one parent run per launch —
        aborted launches are terminated FAILED, so the live/completed launch is
        the newest non-FAILED parent tagged ``exp_id=<exp_id>`` (only parent runs
        carry that tag). Used to scope child lookups to the launch being
        enriched, so a unit_id shared across launches (or a foreign EXP reusing
        the slug/matrix) cannot cross-contaminate."""
        runs = self._client.search_runs(
            [experiment_id],
            filter_string=f"tags.exp_id = '{exp_id}'",
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
        for r in runs:
            if r.info.status != "FAILED":
                return r.info.run_id
        return None

    def find_runs_by_unit(
        self, experiment_id: str, unit_id: str, parent_run_id: Optional[str] = None,
    ) -> list[str]:
        """ALL run ids tagged ``unit_id=<unit_id>``, oldest-started first. A
        reclaimed Batch attempt that was retried leaves TWO runs with the same
        unit_id (the mis-sealed zombie + the completing retry), so the backfill
        must see every one — it enriches the run carrying the durable result and
        reconciles the rest. When ``parent_run_id`` is given the search is scoped
        to that launch's children (``mlflow.parentRunId``), so aborted launches'
        or a foreign EXP's same-unit_id runs are excluded."""
        filt = f"tags.unit_id = '{unit_id}'"
        if parent_run_id:
            filt += f" and tags.`mlflow.parentRunId` = '{parent_run_id}'"
        runs = self._client.search_runs(
            [experiment_id],
            filter_string=filt,
            order_by=["attributes.start_time ASC"],
            max_results=1000,
        )
        return [r.info.run_id for r in runs]

    def find_run_by_unit(self, experiment_id: str, unit_id: str) -> Optional[str]:
        """The most-recently-started run for a unit, or ``None``. Retained for
        single-run callers; the backfill uses ``find_runs_by_unit``."""
        runs = self.find_runs_by_unit(experiment_id, unit_id)
        return runs[-1] if runs else None

    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag from a run. The backfill uses this to strip legacy input
        tags (scenario/seed/mode/...) that the param/tag isolation moved to
        params, so a recovered old run is not left with duplicated fields."""
        self._client.delete_tag(run_id, key)

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self._client.set_terminated(run_id, status)

    def search_traces(self, experiment_ids: list[str], max_results: int = 1000) -> Any:
        """Traces in the given experiment(s). Used by ``round_trace`` to find a
        run's prior ``fl_training`` trace for idempotent re-emission."""
        return self._client.search_traces(experiment_ids=experiment_ids, max_results=max_results)

    def delete_traces(self, experiment_id: str, trace_ids: list[str]) -> Any:
        """Delete traces by id (idempotent re-trace strips the stale one first)."""
        return self._client.delete_traces(experiment_id=experiment_id, trace_ids=trace_ids)

    def set_model_alias(self, name: str, alias: str, version: str) -> None:
        """Assign a registry alias to a model version. Sweep-scoped names
        (``champion__{exp_id}`` / ``challenger__{exp_id}``) — a
        bare ``champion``/``challenger`` is a MUTABLE model-level ref the next
        sweep's promotion would overwrite, destroying the 'best within this
        sweep' record."""
        self._client.set_registered_model_alias(name, alias, version)

    def search_model_versions(self, filter_string: str) -> Any:
        """Model versions matching a FILTER QUERY STRING (e.g. ``name='praxis-krum'``)
        — the documented first arg is a filter, NOT a bare model name (passing a
        bare name errors or searches the wrong set). Verified against 3.12.0."""
        return self._client.search_model_versions(filter_string)

    def artifact_uri_for_run(self, run_id: str) -> str:
        """The run's real artifact root (e.g. s3://bucket/mlflow/artifacts/
        <exp>/<run_id>/artifacts), read from the server rather than
        reconstructed client-side. Used by ingest.py to build s3_result_uri /
        s3_console_url tags for artifacts it just uploaded via log_artifact."""
        return self._client.get_run(run_id).info.artifact_uri

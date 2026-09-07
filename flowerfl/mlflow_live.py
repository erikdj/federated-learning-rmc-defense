"""Live per-round MLflow metric streaming from the server strategy.

Instrumentation only — Lane C, item 10 of the MLflow enrichment redesign
(``docs/harness/README.md``).
This is observability, NOT experimental methodology: it changes nothing about
aggregation, defense logic, seeding, the parsed trajectory, or anything written
to disk/S3.

Motivation
----------
Previously the runner logged every round's metrics to MLflow only *after* the
whole run completed (``scripts/run_phase4_flower.py`` iterated an already-complete
``trajectory`` and batch-dumped it). So a RUNNING unit showed ``metrics(0)`` and a
crash / Spot reclaim lost everything. This module logs each round's centralized
evaluation metrics to MLflow *as the round completes*, driven from
``flowerfl.scenario_strategy.ScenarioStrategy.evaluate``.

Design constraints (spec § 3)
-----------------------------
- **Guarded by ``PRAXIS_MLFLOW_RUN_ID``.** Only active when the unit was launched
  via ``praxis exp launch`` (the entrypoint sets that env var to the child run
  id before spawning the runner). No run id → no logger → no-op.
- **Best-effort, never fail the unit.** Every MLflow call is wrapped; a tracking
  failure only warns and never propagates into the simulation, the trajectory,
  or what is persisted. The S3 result→signal→done contract is untouched.
- **One client, created lazily and reused.** The ``MlflowClient`` is built on the
  first successful log and reused for every subsequent round; a failed init is
  remembered so we do not retry (and re-import mlflow) every round.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)

#: Canonical per-round metric keys logged live, in a stable order. Missing or
#: ``None`` values are skipped so a partially-populated eval never raises.
LIVE_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "loss",
)

#: Fallback tracking URI matching the runner's historical default. Only used
#: when ``MLFLOW_TRACKING_URI`` is absent from the environment.
_DEFAULT_TRACKING_URI = "http://localhost:5001"


class MetricSink(Protocol):
    """Minimal structural type for the object that receives metrics.

    Both ``mlflow.tracking.MlflowClient`` and
    ``praxis_exp.mlflow_client.PraxisMlflowClient`` satisfy this signature, which
    is also trivial to fake in a unit test.
    """

    def log_metric(
        self, run_id: str, key: str, value: float, step: int = 0
    ) -> None:  # pragma: no cover - structural typing only
        ...


def _build_default_client_factory(
    tracking_uri: Optional[str],
) -> Callable[[], MetricSink]:
    """Return a zero-arg factory that builds a real ``MlflowClient``.

    The mlflow import is deferred into the factory so that importing this module
    (and, in turn, the server strategy) never hard-depends on mlflow being
    installed — a unit under test or a bare local run has no mlflow requirement.
    """

    def _factory() -> MetricSink:
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            return MlflowClient(tracking_uri=tracking_uri)
        return MlflowClient()

    return _factory


class LiveRoundMetricLogger:
    """Best-effort, lazily-connected sink for per-round MLflow metrics.

    Args:
        run_id: The MLflow run id to log against (the child run created by the
            entrypoint, surfaced as ``PRAXIS_MLFLOW_RUN_ID``).
        tracking_uri: Optional tracking URI for the default client factory.
            Ignored when ``client_factory`` is supplied.
        client_factory: Optional zero-arg callable returning a ``MetricSink``.
            Injected by tests to supply a fake client; defaults to a real
            ``MlflowClient`` builder.
    """

    def __init__(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        client_factory: Optional[Callable[[], MetricSink]] = None,
    ) -> None:
        self._run_id = run_id
        self._tracking_uri = tracking_uri
        self._client_factory = client_factory or _build_default_client_factory(
            tracking_uri
        )
        self._client: Optional[MetricSink] = None
        self._client_failed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    def _get_client(self) -> Optional[MetricSink]:
        """Return the cached client, building it lazily on first use.

        A failed build is remembered (``_client_failed``) so we do not re-import
        mlflow and retry a broken connection on every round.
        """
        if self._client is not None:
            return self._client
        if self._client_failed:
            return None
        try:
            self._client = self._client_factory()
        except Exception as exc:  # noqa: BLE001 - best-effort; must not raise
            self._client_failed = True
            logger.warning("live MLflow client init failed: %s", exc)
            return None
        return self._client

    def log_round(self, server_round: int, metrics: Mapping[str, object]) -> None:
        """Log one round's metrics at ``step=server_round``.

        Only the canonical :data:`LIVE_METRIC_KEYS` that are present and non-None
        are logged. Every failure mode — no client, a bad value, a server error —
        is swallowed with a warning; this method never raises into the caller.
        """
        client = self._get_client()
        if client is None:
            return
        for key in LIVE_METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                client.log_metric(
                    self._run_id, key, float(value), step=int(server_round)
                )
            except Exception as exc:  # noqa: BLE001 - best-effort; must not raise
                logger.warning(
                    "live MLflow log_metric(%s, round=%s) failed: %s",
                    key,
                    server_round,
                    exc,
                )


def build_live_round_logger_from_env(
    *,
    client_factory: Optional[Callable[[], MetricSink]] = None,
    default_tracking_uri: str = _DEFAULT_TRACKING_URI,
) -> Optional[LiveRoundMetricLogger]:
    """Build a logger from the environment, or ``None`` if not launched via praxis.

    Returns ``None`` unless ``PRAXIS_MLFLOW_RUN_ID`` is set (i.e. the unit was
    launched via ``praxis exp launch``). The tracking URI is read from
    ``MLFLOW_TRACKING_URI`` and falls back to ``default_tracking_uri`` to match
    the runner's historical behavior.
    """
    run_id = os.environ.get("PRAXIS_MLFLOW_RUN_ID")
    if not run_id:
        return None
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri
    return LiveRoundMetricLogger(
        run_id,
        tracking_uri=tracking_uri,
        client_factory=client_factory,
    )

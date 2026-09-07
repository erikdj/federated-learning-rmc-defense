"""Unit tests for live per-round MLflow metric streaming (Lane C, item 10).

Covers ``flowerfl.mlflow_live`` (the injectable best-effort logger) and the
``ScenarioStrategy.evaluate`` hook that drives it. All MLflow interaction is
faked via dependency injection — no network, no real MlflowClient — matching the
repo's existing DI/fake test idiom (see test_praxis_exp_mlflow_client.py and
test_byzantine_defense_pluggable_strategy.py).

Invariants pinned here:
    - per-round metrics are logged with ``step=server_round``;
    - the client is created lazily and reused (one build across many rounds);
    - a failing client / factory never raises into the caller;
    - ``ScenarioStrategy.evaluate`` streams live and returns metrics unchanged,
      and a raising logger never breaks the round.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeMlflowClient:
    """Records every log_metric call; matches MlflowClient.log_metric's shape."""

    def __init__(self):
        self.calls = []

    def log_metric(self, run_id, key, value, step=0):
        self.calls.append(
            {"run_id": run_id, "key": key, "value": value, "step": step}
        )


class RaisingMlflowClient:
    """Every log_metric raises — simulates an unreachable/broken tracking server."""

    def __init__(self):
        self.calls = 0

    def log_metric(self, run_id, key, value, step=0):
        self.calls += 1
        raise RuntimeError("tracking server unreachable")


_FULL_METRICS = {
    "accuracy": 0.91,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.86,
    "loss": 0.42,
}


# --------------------------------------------------------------------------- #
# LiveRoundMetricLogger
# --------------------------------------------------------------------------- #
def test_log_round_logs_each_metric_with_step_equal_round():
    from flowerfl.mlflow_live import LiveRoundMetricLogger, LIVE_METRIC_KEYS

    fake = FakeMlflowClient()
    logger = LiveRoundMetricLogger("run-123", client_factory=lambda: fake)

    logger.log_round(7, _FULL_METRICS)

    assert len(fake.calls) == len(LIVE_METRIC_KEYS)
    for call in fake.calls:
        assert call["run_id"] == "run-123"
        assert call["step"] == 7  # step must be the server_round
        assert call["key"] in LIVE_METRIC_KEYS
        assert isinstance(call["value"], float)
    logged = {c["key"]: c["value"] for c in fake.calls}
    assert logged == {k: float(v) for k, v in _FULL_METRICS.items()}


def test_log_round_skips_missing_and_none_values():
    from flowerfl.mlflow_live import LiveRoundMetricLogger

    fake = FakeMlflowClient()
    logger = LiveRoundMetricLogger("run-1", client_factory=lambda: fake)

    # precision absent, recall explicitly None → neither logged.
    logger.log_round(2, {"accuracy": 0.5, "recall": None, "f1": 0.4, "loss": 0.9})

    logged_keys = {c["key"] for c in fake.calls}
    assert logged_keys == {"accuracy", "f1", "loss"}


def test_client_built_lazily_and_reused_across_rounds():
    from flowerfl.mlflow_live import LiveRoundMetricLogger

    builds = {"n": 0}
    fake = FakeMlflowClient()

    def factory():
        builds["n"] += 1
        return fake

    logger = LiveRoundMetricLogger("run-1", client_factory=factory)
    assert builds["n"] == 0  # not built until first log

    logger.log_round(1, _FULL_METRICS)
    logger.log_round(2, _FULL_METRICS)
    logger.log_round(3, _FULL_METRICS)

    assert builds["n"] == 1  # built exactly once, reused thereafter
    steps = sorted({c["step"] for c in fake.calls})
    assert steps == [1, 2, 3]


def test_failing_client_does_not_raise_and_isolates_per_key():
    """A log_metric that raises must not propagate; every key is still attempted."""
    from flowerfl.mlflow_live import LiveRoundMetricLogger, LIVE_METRIC_KEYS

    raising = RaisingMlflowClient()
    logger = LiveRoundMetricLogger("run-1", client_factory=lambda: raising)

    logger.log_round(4, _FULL_METRICS)  # must NOT raise

    # Per-key isolation: one raising key does not abort the rest.
    assert raising.calls == len(LIVE_METRIC_KEYS)


def test_failing_client_factory_does_not_raise_and_is_not_retried():
    from flowerfl.mlflow_live import LiveRoundMetricLogger

    attempts = {"n": 0}

    def broken_factory():
        attempts["n"] += 1
        raise RuntimeError("cannot import mlflow")

    logger = LiveRoundMetricLogger("run-1", client_factory=broken_factory)

    logger.log_round(1, _FULL_METRICS)  # must NOT raise
    logger.log_round(2, _FULL_METRICS)  # still must NOT raise

    # A failed build is remembered — the factory is not retried every round.
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- #
# build_live_round_logger_from_env
# --------------------------------------------------------------------------- #
def test_build_from_env_returns_none_without_run_id(monkeypatch):
    from flowerfl.mlflow_live import build_live_round_logger_from_env

    monkeypatch.delenv("PRAXIS_MLFLOW_RUN_ID", raising=False)
    assert build_live_round_logger_from_env() is None


def test_build_from_env_returns_logger_with_run_id(monkeypatch):
    from flowerfl.mlflow_live import build_live_round_logger_from_env

    monkeypatch.setenv("PRAXIS_MLFLOW_RUN_ID", "child-run-42")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    fake = FakeMlflowClient()

    logger = build_live_round_logger_from_env(client_factory=lambda: fake)
    assert logger is not None
    assert logger.run_id == "child-run-42"

    logger.log_round(1, _FULL_METRICS)
    assert all(c["run_id"] == "child-run-42" for c in fake.calls)


def test_build_from_env_default_client_uses_tracking_uri(monkeypatch):
    """With no injected factory, the default factory must build against the env
    tracking URI. We stub the mlflow module so no network/real client is used."""
    from flowerfl.mlflow_live import build_live_round_logger_from_env

    monkeypatch.setenv("PRAXIS_MLFLOW_RUN_ID", "child-run-7")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")

    set_uris = []
    made_with = []

    class _StubClient:
        def __init__(self, tracking_uri=None):
            made_with.append(tracking_uri)

        def log_metric(self, run_id, key, value, step=0):
            pass

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: set_uris.append(uri),
        tracking=SimpleNamespace(MlflowClient=_StubClient),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_mlflow.tracking)

    logger = build_live_round_logger_from_env()
    assert logger is not None
    logger.log_round(1, _FULL_METRICS)

    assert set_uris == ["http://tracking:5000"]
    assert made_with == ["http://tracking:5000"]


# --------------------------------------------------------------------------- #
# ScenarioStrategy.evaluate hook
# --------------------------------------------------------------------------- #
def _fake_base_strategy():
    return SimpleNamespace(
        aggregate_fit=lambda server_round, results, failures: (None, {}),
        initialize_parameters=lambda client_manager: None,
        configure_fit=lambda server_round, parameters, client_manager: [],
        configure_evaluate=lambda server_round, parameters, client_manager: [],
        aggregate_evaluate=lambda server_round, results, failures: (None, {}),
        evaluate=lambda server_round, parameters: None,
    )


def _fake_eval_manager():
    # Returns the five-metric dict the real FixedEvalManager produces.
    return SimpleNamespace(evaluate=lambda weights: dict(_FULL_METRICS))


def _make_parameters():
    from flwr.common import ndarrays_to_parameters

    return ndarrays_to_parameters([np.zeros(2, dtype=np.float32)])


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log_round(self, server_round, metrics):
        self.calls.append((server_round, dict(metrics)))


class _RaisingLogger:
    def log_round(self, server_round, metrics):
        raise RuntimeError("logger blew up")


def test_evaluate_streams_live_and_returns_metrics_unchanged():
    from flowerfl.scenario_strategy import ScenarioStrategy

    rec = _RecordingLogger()
    strategy = ScenarioStrategy(
        _fake_base_strategy(),
        plugins=[],
        scenario_path=None,
        eval_manager=_fake_eval_manager(),
        live_metric_logger=rec,
    )

    loss, metrics = strategy.evaluate(5, _make_parameters())

    # Return value is byte-identical to the pre-instrumentation contract.
    assert loss == _FULL_METRICS["loss"]
    assert metrics == {
        "accuracy": _FULL_METRICS["accuracy"],
        "precision": _FULL_METRICS["precision"],
        "recall": _FULL_METRICS["recall"],
        "f1": _FULL_METRICS["f1"],
    }

    # The live hook fired once with step=server_round and all five metrics.
    assert len(rec.calls) == 1
    server_round, logged = rec.calls[0]
    assert server_round == 5
    assert logged == _FULL_METRICS


def test_evaluate_without_logger_is_noop_and_returns_metrics():
    from flowerfl.scenario_strategy import ScenarioStrategy

    strategy = ScenarioStrategy(
        _fake_base_strategy(),
        plugins=[],
        scenario_path=None,
        eval_manager=_fake_eval_manager(),
    )
    assert strategy._live_metric_logger is None

    loss, metrics = strategy.evaluate(1, _make_parameters())
    assert loss == _FULL_METRICS["loss"]
    assert metrics["f1"] == _FULL_METRICS["f1"]


def test_evaluate_survives_a_raising_logger():
    """A tracking failure must NEVER affect the round: evaluate still returns the
    correct metrics even when the injected logger raises."""
    from flowerfl.scenario_strategy import ScenarioStrategy

    strategy = ScenarioStrategy(
        _fake_base_strategy(),
        plugins=[],
        scenario_path=None,
        eval_manager=_fake_eval_manager(),
        live_metric_logger=_RaisingLogger(),
    )

    loss, metrics = strategy.evaluate(3, _make_parameters())  # must NOT raise
    assert loss == _FULL_METRICS["loss"]
    assert metrics["accuracy"] == _FULL_METRICS["accuracy"]


def test_set_live_metric_logger_wires_after_construction():
    """server_fn attaches the env-derived logger through this setter."""
    from flowerfl.scenario_strategy import ScenarioStrategy

    strategy = ScenarioStrategy(
        _fake_base_strategy(),
        plugins=[],
        scenario_path=None,
        eval_manager=_fake_eval_manager(),
    )
    rec = _RecordingLogger()
    strategy.set_live_metric_logger(rec)

    strategy.evaluate(9, _make_parameters())
    assert rec.calls and rec.calls[0][0] == 9

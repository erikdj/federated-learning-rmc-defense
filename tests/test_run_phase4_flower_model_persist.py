"""Tests for scripts/run_phase4_flower.py's _persist_final_model helper.

req 6 (models): ground-truthed that nothing in flowerfl/ or
scripts/run_phase4_flower.py previously saved the final global model
anywhere (no state_dict/torch.save call existed). This is the minimal
additive fix: after a run's simulation completes, best-effort convert the
strategy's last aggregated Parameters (see PluggableStrategy bookkeeping in
flowerfl/byzantine_defense.py) into a state_dict and torch.save it next to
the result JSON. Gated: any failure here must never fail the run, so
_persist_final_model always returns bool rather than raising.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_persist_final_model_returns_false_when_no_parameters(tmp_path):
    from run_phase4_flower import _persist_final_model

    class NoParamsStrategy:
        pass

    ok = _persist_final_model(NoParamsStrategy(), "edge_full_20_rmc", tmp_path / "m.pt")
    assert ok is False
    assert not (tmp_path / "m.pt").exists()


def test_persist_final_model_returns_false_when_strategy_is_none(tmp_path):
    from run_phase4_flower import _persist_final_model
    assert _persist_final_model(None, "edge_full_20_rmc", tmp_path / "m.pt") is False


def test_persist_final_model_saves_state_dict(tmp_path, monkeypatch):
    import torch
    from run_phase4_flower import _persist_final_model

    class FakeNet:
        def state_dict(self):
            return {"w": torch.zeros(2)}

        def load_state_dict(self, sd, strict=True):
            self.loaded = sd

    fake_net = FakeNet()
    monkeypatch.setattr("flowerfl.task.create_model", lambda name: fake_net)
    monkeypatch.setattr("flowerfl.task.set_weights", lambda net, params: None)
    monkeypatch.setattr("flwr.common.parameters_to_ndarrays", lambda p: [])

    class FakeStrategy:
        _last_aggregated_parameters = object()

    model_path = tmp_path / "model.pt"
    ok = _persist_final_model(FakeStrategy(), "edge_full_20_rmc", model_path)
    assert ok is True
    assert model_path.exists()
    loaded = torch.load(model_path, weights_only=True)
    assert "w" in loaded


def test_persist_final_model_never_raises_on_internal_failure(tmp_path, monkeypatch, capsys):
    from run_phase4_flower import _persist_final_model

    def _boom(name):
        raise RuntimeError("no such dataset")

    monkeypatch.setattr("flowerfl.task.create_model", _boom)

    class FakeStrategy:
        _last_aggregated_parameters = object()

    ok = _persist_final_model(FakeStrategy(), "bogus_dataset", tmp_path / "m.pt")
    assert ok is False
    assert not (tmp_path / "m.pt").exists()
    assert "WARN" in capsys.readouterr().out

"""Runtime provenance in the result JSON (Stage-F).

The result-JSON provenance historically recorded runner_commit="unknown" (the
container bakes the repo WITHOUT.git) and carried no image digest or host CPU
context, so a reproducibility claim could not be tied to a commit/image/host.
These tests pin the best-effort helpers: env-sourced commit/digest, /proc CPU
model, logical CPU count, torch thread count, and their never-crash fallbacks.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4_flower import (
    _runner_commit,
    _launch_commit,
    _image_digest,
    _cpu_model,
    _host_provenance_fields,
    summarize_attack_recall,
)

import pytest


def test_runner_commit_prefers_env(monkeypatch):
    # PRAXIS_RUNNER_COMMIT is now the IMAGE-BAKED commit (Dockerfile --build-arg),
    # NOT a launch-time value — the runner still reads it as its own code commit.
    monkeypatch.setenv("PRAXIS_RUNNER_COMMIT", "abc1234")
    assert _runner_commit() == "abc1234"


def test_runner_commit_falls_back_to_reasoned_unavailable(monkeypatch):
    monkeypatch.delenv("PRAXIS_RUNNER_COMMIT", raising=False)
    # Force the local-git fallback to report unavailable.
    monkeypatch.setattr("run_phase4_flower._git_rev", lambda: "unknown")
    out = _runner_commit()
    assert out != "unknown"  # never a bare 'unknown'
    assert "unavailable" in out.lower()


def test_launch_commit_prefers_env(monkeypatch):
    # PRAXIS_LAUNCH_COMMIT is the launch git HEAD (manifest/scenario inputs);
    # distinct from the image-baked runner commit.
    monkeypatch.setenv("PRAXIS_LAUNCH_COMMIT", "def5678")
    assert _launch_commit() == "def5678"


def test_launch_commit_falls_back_to_reasoned_unavailable(monkeypatch):
    monkeypatch.delenv("PRAXIS_LAUNCH_COMMIT", raising=False)
    out = _launch_commit()
    assert out != "unknown"  # never a bare 'unknown'
    assert "unavailable" in out.lower()


def test_image_digest_from_env(monkeypatch):
    monkeypatch.setenv("PRAXIS_IMAGE_DIGEST", "sha256:deadbeef")
    assert _image_digest() == "sha256:deadbeef"


def test_image_digest_fallback_when_unset(monkeypatch):
    monkeypatch.delenv("PRAXIS_IMAGE_DIGEST", raising=False)
    out = _image_digest()
    assert "unavailable" in out.lower()


def test_host_provenance_fields_present():
    d = _host_provenance_fields()
    assert set(d) == {"cpu_model", "cpu_count_logical", "torch_num_threads"}
    # cpu_model is a string (real model name or 'unavailable')
    assert isinstance(d["cpu_model"], str) and d["cpu_model"]


def test_cpu_model_never_raises(monkeypatch):
    # Simulate an unreadable /proc/cpuinfo — must degrade, not crash.
    def _boom(*a, **k):
        raise OSError("no /proc")
    monkeypatch.setattr("builtins.open", _boom)
    assert _cpu_model() == "unavailable"


# --- summary aggregation (mean/final attack_recall built like mean/final f1) ---

def test_summarize_attack_recall_present():
    traj = [
        {"round": 0, "attack_recall": 0.8},
        {"round": 1, "attack_recall": 0.6},
    ]
    s = summarize_attack_recall(traj)
    assert s["final_attack_recall"] == 0.6
    assert s["mean_attack_recall"] == (0.8 + 0.6) / 2


def test_summarize_attack_recall_none_when_absent():
    # Legacy trajectory (older image) has no per-class fields -> None, not crash.
    # Lenient default (require=False) is the offline-replay path for legacy logs.
    traj = [{"round": 0, "f1": 0.9}, {"round": 1, "f1": 0.9}]
    s = summarize_attack_recall(traj)
    assert s["mean_attack_recall"] is None
    assert s["final_attack_recall"] is None


def test_summarize_attack_recall_all_absent_require_raises():
    # require=True: this runner's own image always emits per-class eval lines, so
    # a fully-absent non-empty trajectory means the parse chain failed (prereg
    # completeness assertion, ).
    traj = [{"round": 0, "f1": 0.9}, {"round": 1, "f1": 0.9}]
    with pytest.raises(ValueError):
        summarize_attack_recall(traj, require=True)


def test_summarize_attack_recall_mixed_raises_regardless_of_require():
    # Partial per-class instrumentation is never a legitimate state — the parse
    # chain is corrupted; the message names the count of missing rounds.
    traj = [{"round": 0, "attack_recall": 0.8}, {"round": 1, "f1": 0.9}]
    with pytest.raises(ValueError, match="1"):
        summarize_attack_recall(traj)
    with pytest.raises(ValueError, match="1"):
        summarize_attack_recall(traj, require=True)


def test_summarize_attack_recall_empty_is_none_even_with_require():
    # Empty trajectory keeps the Nones even under require=True — other summary
    # machinery already screams on empty trajectories.
    assert summarize_attack_recall([]) == {
        "mean_attack_recall": None, "final_attack_recall": None,
    }
    assert summarize_attack_recall([], require=True) == {
        "mean_attack_recall": None, "final_attack_recall": None,
    }

"""Reconstruct the on-disk filenames scripts/run_phase4_flower.py writes for a Unit.

Mirrors run_phase4_flower.py result-JSON naming (~line 720) and the signal-log
naming (~lines 555-590 / server_app.py). The defense_token (config-label ->
strategy token) is resolved by the caller from server_app.py, so signal_filename
takes it explicitly.
"""
from __future__ import annotations

from praxis_exp.units import Unit

__all__ = ["exec_mode_token", "result_filename", "signal_filename", "model_filename"]


def exec_mode_token(mode: str) -> str:
    """Map a Unit.mode (the runner --modes value) to the signal-log exec_mode token.

    The runner (_resolve_optimizer_state, run_phase4_flower.py:971-981) maps
    --modes persistent_optimizer -> state "persistent" -> signal token "flower_persistent",
    and --modes Flower -> "reset" -> "flower_reset". Unit.mode carries the raw --modes
    value, so startswith("persistent") is correct and robust across the vocabulary
    {persistent_optimizer, persistent, Flower, flower_reset, reset}. Do NOT change this to
    `== "persistent"`: that would map "persistent_optimizer" to flower_reset and break
    persist_unit's signal-file lookup. Keep in sync with run_phase4_flower.py:570.
    """
    return "flower_persistent" if mode.lower().startswith("persistent") else "flower_reset"


def result_filename(unit: Unit) -> str:
    label = unit.config.replace("+", "_").lower()
    return f"phase4_flower__{label}__seed{unit.seed}.json"


def signal_filename(unit: Unit, *, defense_token: str) -> str:
    return f"{exec_mode_token(unit.mode)}__{unit.scenario}__{defense_token}__seed{unit.seed}.jsonl"


def model_filename(unit: Unit) -> str:
    """req 6 (models): mirrors result_filename's naming with a __model.pt
    suffix. run_phase4_flower.py's _persist_final_model writes here
    (best-effort, gated); entrypoint.py checks for its existence before
    logging it to the child run — never assume it exists."""
    label = unit.config.replace("+", "_").lower()
    return f"phase4_flower__{label}__seed{unit.seed}__model.pt"

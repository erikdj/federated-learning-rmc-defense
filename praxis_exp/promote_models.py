"""Sweep-level model promotion through champion/challenger aliases.

``log_native_model`` runs per-unit in the container with no cross-unit view, so it
cannot pick the best model for a defense — an in-container alias set would be a
last-writer-wins race. This sweep-level pass groups a sweep's registered model
versions by defense, ranks them by ``final_f1``, and sets **sweep-scoped** aliases
``champion__{exp_id}`` / ``challenger__{exp_id}``. Bare ``champion``/``challenger``
would be overwritten by the next sweep's promotion (aliases are mutable, model-level
refs), silently destroying the "best within THIS sweep" record. Idempotent, best-effort.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from praxis_exp import storage
from praxis_exp.manifest import read_manifest
from praxis_exp.mlflow_client import PraxisMlflowClient
from praxis_exp.round_trace import _defense_token, _experiment_name, _read_json
from praxis_exp.storage import ObjectStore


def promote_models(
    repo_root: Path, exp_id: str, *, _store: ObjectStore, _client: Optional[Any] = None,
) -> dict[str, Any]:
    """Set sweep-scoped ``champion__{exp_id}``/``challenger__{exp_id}`` aliases on
    each defense's registered model (``praxis-{defense_token}``), ranked by the
    unit's ``final_f1`` (from the durable ``result.json``). Only this sweep's
    versions are considered (mapped via the version's ``run_id``). Best-effort;
    idempotent (re-run assigns the same aliases)."""
    repo_root = Path(repo_root)
    _, _meta, units = read_manifest(_store, exp_id)
    client = _client or PraxisMlflowClient()
    experiment_id = client.get_or_create_experiment(_experiment_name(repo_root, exp_id))
    parent_run_id = client.find_parent_run(experiment_id, exp_id)

    # this sweep's runs -> their unit (scoped to the launch's children)
    run_to_unit: dict[str, Any] = {}
    dtokens: list[str] = []
    for unit in units:
        dt = _defense_token(unit)
        if dt not in dtokens:
            dtokens.append(dt)
        for rid in client.find_runs_by_unit(experiment_id, unit.unit_id, parent_run_id=parent_run_id):
            run_to_unit[rid] = unit

    promoted = 0
    for dt in dtokens:
        name = f"praxis-{dt}"
        try:
            versions = client.search_model_versions(f"name='{name}'")
        except Exception as e:
            print(f"[promote] WARN: version search failed for {name} ({e}); skipping")
            continue
        scored: list[tuple[float, str]] = []
        for v in versions:
            rid = getattr(v, "run_id", None)
            unit = run_to_unit.get(rid)
            if unit is None or _defense_token(unit) != dt:
                continue  # a version from another sweep (or another defense) — skip
            result = _read_json(_store, storage.result_key(exp_id, unit.unit_id))
            f1 = (result or {}).get("final_f1")
            if f1 is not None:
                scored.append((float(f1), str(v.version)))
        if not scored:
            continue
        scored.sort(key=lambda t: -t[0])  # highest final_f1 first
        try:
            client.set_model_alias(name, f"champion__{exp_id}", scored[0][1])
            if len(scored) > 1:
                client.set_model_alias(name, f"challenger__{exp_id}", scored[1][1])
            promoted += 1
        except Exception as e:
            print(f"[promote] WARN: alias set failed for {name} ({e})")
    return {"experiment_id": experiment_id, "defenses_promoted": promoted}

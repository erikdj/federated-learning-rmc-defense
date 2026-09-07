"""The S3 manifest: the single source of truth mapping array index -> Unit."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from praxis_exp import storage
from praxis_exp.storage import ObjectStore
from praxis_exp.units import Unit


class ManifestSchemaError(ValueError):
    """Raised when a manifest is missing required keys or has an incompatible unit schema."""


def write_manifest(store: ObjectStore, exp_id: str, units: list[Unit], meta: dict[str, Any]) -> None:
    if not units:
        raise ValueError("write_manifest: units must be non-empty")
    payload = {"exp_id": exp_id, "meta": meta, "units": [asdict(u) for u in units]}
    store.put_bytes(storage.manifest_key(exp_id), json.dumps(payload, indent=2).encode())


def read_manifest(store: ObjectStore, exp_id: str) -> tuple[str, dict[str, Any], list[Unit]]:
    data = json.loads(store.get_bytes(storage.manifest_key(exp_id)))
    for key in ("exp_id", "meta", "units"):
        if key not in data:
            raise ManifestSchemaError(f"manifest for {exp_id} missing top-level key {key!r}")
    try:
        units = [Unit(**u) for u in data["units"]]
    except TypeError as exc:
        raise ManifestSchemaError(
            f"manifest for {exp_id} has an incompatible unit schema: {exc}"
        ) from exc
    return data["exp_id"], data["meta"], units


def unit_for_index(units: list[Unit], array_index: int) -> Unit:
    # Linear scan is intentional: <=200 units, called once per container startup.
    for u in units:
        if u.array_index == array_index:
            return u
    raise IndexError(f"no unit with array_index={array_index} (manifest has {len(units)} units)")

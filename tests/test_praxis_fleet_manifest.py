import json

import pytest
from praxis_exp import storage
from praxis_exp.storage import InMemoryObjectStore, ObjectNotFoundError
from praxis_exp.units import expand_matrix
from praxis_exp.manifest import (
    write_manifest,
    read_manifest,
    unit_for_index,
    ManifestSchemaError,
)


def test_manifest_round_trips_units_and_meta():
    store = InMemoryObjectStore()
    units = expand_matrix(["Krum"], ["S0", "S4"], [42, 137], "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, "EXP-005", units, meta={"methodology_version": "v1.9", "image_digest": "sha256:abc"})
    exp_id, meta, loaded = read_manifest(store, "EXP-005")
    assert exp_id == "EXP-005"
    assert meta["image_digest"] == "sha256:abc"
    assert [u.unit_id for u in loaded] == [u.unit_id for u in units]


def test_unit_for_index_maps_array_index():
    units = expand_matrix(["Krum", "TrustScore"], ["S0"], [42], "persistent_optimizer", 2_000_000, 50)
    assert unit_for_index(units, 1).config == "TrustScore"
    with pytest.raises(IndexError):
        unit_for_index(units, 99)


def test_read_manifest_raises_on_absent_manifest():
    with pytest.raises(ObjectNotFoundError):
        read_manifest(InMemoryObjectStore(), "EXP-NEVER")


def test_read_manifest_raises_schema_error_on_missing_top_key():
    store = InMemoryObjectStore()
    store.put_bytes(storage.manifest_key("EXP-X"), json.dumps({"exp_id": "EXP-X", "meta": {}}).encode())
    with pytest.raises(ManifestSchemaError, match="units"):
        read_manifest(store, "EXP-X")


def test_read_manifest_raises_schema_error_on_unknown_unit_field():
    store = InMemoryObjectStore()
    bad = {"exp_id": "EXP-X", "meta": {}, "units": [{"config": "Krum", "scenario": "S0", "mode": "m",
            "seed": 42, "max_per_client": 1, "rounds": 50, "array_index": 0, "ghost": "boom"}]}
    store.put_bytes(storage.manifest_key("EXP-X"), json.dumps(bad).encode())
    with pytest.raises(ManifestSchemaError):
        read_manifest(store, "EXP-X")


def test_write_manifest_rejects_empty_units():
    with pytest.raises(ValueError):
        write_manifest(InMemoryObjectStore(), "EXP-X", [], meta={})

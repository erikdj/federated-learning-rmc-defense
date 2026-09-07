import json

from praxis_exp.manifest import write_manifest
from praxis_exp.refill_reconcile import reconcile_sweep_tags, read_refill_records
from praxis_exp.storage import InMemoryObjectStore, marker_key
from praxis_exp.units import expand_matrix


class _FakeClient:
    def __init__(self):
        self.tags = {}  # run_id -> {k: v}

    def set_tag(self, run_id, key, value):
        self.tags.setdefault(run_id, {})[key] = value


def _units():
    return expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
                         "persistent_optimizer", 2_000_000, 50)


def _seed_manifest(store, exp_id, units, *, done):
    write_manifest(store, exp_id, units, {"methodology_version": "v1.9", "n_units": len(units)})
    for u in units:
        if u.array_index in done:
            store.put_bytes(marker_key(exp_id, u.unit_id), b"")


def _write_refill_record(store, exp_id, serial, cells, **meta):
    key = f"sweeps/{exp_id}/refills/{serial}/manifest.json"
    payload = {"exp_id": exp_id, "meta": {"serial": serial, "refilled_cells": cells,
                                          "launched_at": f"2026-07-2{serial[-1]}T00:00:00Z", **meta},
               "units": []}
    store.put_bytes(key, json.dumps(payload).encode())


# --------------------------------------------------------------------------

def test_reconcile_clears_stale_incomplete_when_refilled_sweep_completes(tmp_path):
    """A sweep that HAD a refill and is now fully complete: clear the finalizer's
    stale sweep_incomplete/missing_cells, preserve refill_history, stamp reconciled."""
    store = InMemoryObjectStore()
    units = _units()
    _seed_manifest(store, "EXP-005", units, done=set(range(8)))  # all complete
    _write_refill_record(store, "EXP-005", "r2",
                         [units[0].unit_id, units[6].unit_id],
                         git_sha="abc1234", image_digest="sha256:deadbeef",
                         provenance_changed=None)
    client = _FakeClient()
    reconcile_sweep_tags(client, store, "EXP-005", "parent-1")
    t = client.tags["parent-1"]
    assert t["done_count"] == "8" and t["n_units"] == "8"
    assert t["sweep_incomplete"] == "false"   # cleared, not left "true"
    assert t["missing_cells"] == ""
    assert "sweep_reconciled_at" in t
    history = json.loads(t["refill_history"])
    assert history[0]["serial"] == "r2"
    assert set(history[0]["refilled_cells"]) == {units[0].unit_id, units[6].unit_id}


def test_reconcile_pristine_complete_sweep_leaves_incomplete_tags_absent(tmp_path):
    """A sweep that NEVER had a refill and is complete keeps the legacy invariant:
    done_count/n_units only, no sweep_incomplete / missing_cells (not 'false')."""
    store = InMemoryObjectStore()
    units = _units()
    _seed_manifest(store, "EXP-005", units, done=set(range(8)))  # complete, no refill record
    client = _FakeClient()
    reconcile_sweep_tags(client, store, "EXP-005", "parent-1")
    t = client.tags["parent-1"]
    assert t["done_count"] == "8" and t["n_units"] == "8"
    assert "sweep_incomplete" not in t
    assert "missing_cells" not in t
    assert "refill_history" not in t


def test_reconcile_still_incomplete_keeps_flags_and_shows_history(tmp_path):
    """A refill that did NOT complete every cell: keep sweep_incomplete=true with the
    remaining missing cells, and still surface refill_history."""
    store = InMemoryObjectStore()
    units = _units()
    _seed_manifest(store, "EXP-005", units, done=set(range(7)))  # idx7 still missing
    _write_refill_record(store, "EXP-005", "r2", [units[0].unit_id])
    client = _FakeClient()
    reconcile_sweep_tags(client, store, "EXP-005", "parent-1")
    t = client.tags["parent-1"]
    assert t["sweep_incomplete"] == "true"
    assert t["missing_cells"] == units[7].unit_id
    assert t["done_count"] == "7"
    assert json.loads(t["refill_history"])[0]["serial"] == "r2"


def test_read_refill_records_orders_by_serial(tmp_path):
    store = InMemoryObjectStore()
    _write_refill_record(store, "EXP-005", "r3", ["c3"])
    _write_refill_record(store, "EXP-005", "r2", ["c2"])
    records = read_refill_records(store, "EXP-005")
    assert [r["serial"] for r in records] == ["r2", "r3"]


def test_read_refill_records_empty_when_none(tmp_path):
    assert read_refill_records(InMemoryObjectStore(), "EXP-005") == []


def test_read_refill_records_orders_r10_after_r2_numerically(tmp_path):
    """lexicographic key ordering places r10 before r2; the
    audit trail must be oldest-first by NUMERIC serial."""
    store = InMemoryObjectStore()
    for serial in ("r2", "r10", "r3"):
        _write_refill_record(store, "EXP-005", serial, [f"c-{serial}"])
    assert [r["serial"] for r in read_refill_records(store, "EXP-005")] == ["r2", "r3", "r10"]

from praxis_exp.storage import InMemoryObjectStore
from praxis_exp.units import expand_matrix
from praxis_exp.manifest import write_manifest, read_manifest, unit_for_index
from praxis_exp.integrity import persist_unit, is_done


def _fake_run_outputs(tmp_path, unit):
    """Stand in for what scripts/run_phase4_flower.py writes on disk."""
    r = tmp_path / f"{unit.unit_id}.json"; r.write_text('{"return_code": 0, "final_f1": 0.97}')
    s = tmp_path / f"{unit.unit_id}.jsonl"; s.write_text('{"logical_cid": 0, "malicious_gt": true}\n')
    return r, s


def test_full_sweep_lifecycle_with_idempotent_skip(tmp_path):
    store = InMemoryObjectStore()
    exp_id = "EXP-005"
    units = expand_matrix(["Krum", "Krum+TGE"], ["S0", "S4"], [42], "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, exp_id, units, meta={"methodology_version": "v1.9"})

    _, _, manifest_units = read_manifest(store, exp_id)
    runs_executed = 0
    for attempt in range(2):  # run the whole sweep twice; second pass must skip everything
        for i in range(len(manifest_units)):
            unit = unit_for_index(manifest_units, i)
            if is_done(store, exp_id, unit.unit_id):
                continue
            r, s = _fake_run_outputs(tmp_path, unit)
            persist_unit(store, exp_id, unit.unit_id, r, s)
            runs_executed += 1

    assert runs_executed == 4
    assert all(is_done(store, exp_id, u.unit_id) for u in units)

"""Tests for praxis_exp/promote_models.py — sweep-level champion/challenger aliases."""
import json

from praxis_exp import storage
from praxis_exp.manifest import write_manifest
from praxis_exp.units import expand_matrix


class _FakeVersion:
    def __init__(self, version, run_id):
        self.version = version
        self.run_id = run_id


def _seed_store(store, exp_id, units, f1_by_unit):
    write_manifest(store, exp_id, units, meta={})
    for u in units:
        store.put_bytes(
            storage.result_key(exp_id, u.unit_id),
            json.dumps({"final_f1": f1_by_unit[u.unit_id]}).encode(),
        )


class _FakePromoteClient:
    def __init__(self, unit_run_ids, versions):
        self._runs = unit_run_ids           # unit_id -> [run_id]
        self._versions = versions           # filter_string -> [ _FakeVersion ]
        self.aliases = []
        self.searched = []

    def get_or_create_experiment(self, name):
        return "40"

    def find_parent_run(self, experiment_id, exp_id):
        return "parent-1"

    def find_runs_by_unit(self, experiment_id, unit_id, parent_run_id=None):
        return list(self._runs.get(unit_id, []))

    def search_model_versions(self, filter_string):
        self.searched.append(filter_string)
        return self._versions.get(filter_string, [])

    def set_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, version))


def test_promote_models_sets_sweep_scoped_champion_and_challenger(monkeypatch, tmp_path):
    import praxis_exp.promote_models as pm
    monkeypatch.setattr(pm, "_experiment_name", lambda repo, eid: "calibration")

    store = storage.InMemoryObjectStore()
    units = expand_matrix(["Krum"], ["S0", "S4"], [42], "persistent_optimizer", 5000, 2)
    lo, hi = units[0], units[1]
    _seed_store(store, "EXP-006", units, {lo.unit_id: 0.60, hi.unit_id: 0.90})

    client = _FakePromoteClient(
        unit_run_ids={lo.unit_id: ["run-a"], hi.unit_id: ["run-b"]},
        versions={"name='praxis-krum'": [_FakeVersion("1", "run-a"), _FakeVersion("2", "run-b")]},
    )
    out = pm.promote_models(tmp_path, "EXP-006", _store=store, _client=client)

    assert out["defenses_promoted"] == 1
    # champion = version 2 (run-b, final_f1 0.90); challenger = version 1 (run-a, 0.60)
    assert ("praxis-krum", "champion__EXP-006", "2") in client.aliases
    assert ("praxis-krum", "challenger__EXP-006", "1") in client.aliases
    # SWEEP-SCOPED (never a bare champion/challenger the next sweep would clobber)
    assert all(alias.endswith("__EXP-006") for _, alias, _ in client.aliases)
    # search used a filter query string, not a bare name
    assert client.searched == ["name='praxis-krum'"]


def test_promote_models_single_version_sets_champion_only(monkeypatch, tmp_path):
    import praxis_exp.promote_models as pm
    monkeypatch.setattr(pm, "_experiment_name", lambda repo, eid: "calibration")
    store = storage.InMemoryObjectStore()
    units = expand_matrix(["Krum"], ["S0"], [42], "persistent_optimizer", 5000, 2)
    _seed_store(store, "EXP-006", units, {units[0].unit_id: 0.7})
    client = _FakePromoteClient(
        unit_run_ids={units[0].unit_id: ["run-a"]},
        versions={"name='praxis-krum'": [_FakeVersion("1", "run-a")]},
    )
    pm.promote_models(tmp_path, "EXP-006", _store=store, _client=client)
    assert [a[1] for a in client.aliases] == ["champion__EXP-006"]  # no challenger


def test_promote_models_no_matching_versions_is_noop(monkeypatch, tmp_path):
    import praxis_exp.promote_models as pm
    monkeypatch.setattr(pm, "_experiment_name", lambda repo, eid: "calibration")
    store = storage.InMemoryObjectStore()
    units = expand_matrix(["Krum"], ["S0"], [42], "persistent_optimizer", 5000, 2)
    _seed_store(store, "EXP-006", units, {units[0].unit_id: 0.7})
    client = _FakePromoteClient(
        unit_run_ids={units[0].unit_id: ["run-a"]},
        versions={"name='praxis-krum'": []},   # no registered versions yet
    )
    out = pm.promote_models(tmp_path, "EXP-006", _store=store, _client=client)
    assert out["defenses_promoted"] == 0 and client.aliases == []

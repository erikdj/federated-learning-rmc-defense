"""server_app builds scenario-mode Krum defenses at the scenario-derived f/keep.

Launch-blocker fix (methodology v1.19). Before the fix, ``ScenarioKrum`` built
``KrumDefensePlugin(num_malicious=1, num_to_keep=18)`` because the run config
carried no adversary count (``malicious-fraction`` defaulted to 0.0 and the
branch clamped with ``max(1, 0)``). With the runner now emitting
``num-malicious`` / ``defense-cohort-size``, the deployed Multi-Krum must run
at f=9 / keep=9 (cohort 20 - f 9 - 2 = 9), matching the documented design.

TGE's ``num_to_keep`` / ``num_malicious`` are inert (never read; the 0.7 score
threshold is TGE's sole filter — see flowerfl/byzantine_defense.py
TGEnsemblePlugin), so this suite asserts only the Krum layer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import flowerfl.server_app as server_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S4 = str(PROJECT_ROOT / "rmc" / "scenarios" / "S4_full_mix.json")


class _CapturingScenarioStrategy:
    """Stand-in for ScenarioStrategy that records the plugins it was built with.

    Lets us introspect the real KrumDefensePlugin / TGEnsemblePlugin instances
    server_fn constructs without running a Flower simulation.
    """

    last: "_CapturingScenarioStrategy | None" = None

    def __init__(self, base, plugins=None, scenario_path=None,
                 eval_manager=None, signal_logger=None,
                 live_metric_logger=None):
        _CapturingScenarioStrategy.last = self
        self.base = base
        self.plugins = list(plugins or [])
        self.scenario_path = scenario_path
        self.live_metric_logger = live_metric_logger

    def set_live_metric_logger(self, logger_obj):
        # Mirror ScenarioStrategy's interface: server_fn attaches the
        # env-derived live MLflow logger here after construction.
        self.live_metric_logger = logger_obj


@pytest.fixture
def light_server(monkeypatch):
    """Strip server_fn of its data-loading deps so the strategy it builds can
    be introspected without parquet files or a FixedEvalManager."""
    monkeypatch.setattr(server_app, "detect_input_shape", lambda dataset: 45)
    monkeypatch.setattr(server_app, "_create_eval_manager", lambda dataset, run_config=None: object())
    monkeypatch.setattr(server_app, "ScenarioStrategy", _CapturingScenarioStrategy)
    _CapturingScenarioStrategy.last = None
    return _CapturingScenarioStrategy


def _ctx(run_config: dict):
    from flwr.common import Context
    from flwr.common.record.recorddict import RecordDict

    return Context(
        run_id=1,
        node_id=1,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )


def _config(**overrides) -> dict:
    cfg = {
        "dataset": "edge_full_20_rmc",
        "strategy": "ScenarioKrum",
        "scenario": S4,
        "num-server-rounds": 3,
        "signal-log": 0,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.unit
def test_scenario_krum_uses_derived_f_and_keep(light_server):
    cfg = _config(**{"num-malicious": 9, "defense-cohort-size": 20})
    server_app.server_fn(_ctx(cfg))
    plugin = light_server.last.plugins[0]
    assert plugin.name == "KrumDefense"
    assert plugin._num_malicious == 9
    assert plugin._num_to_keep == 9  # cohort 20 - f 9 - 2


@pytest.mark.unit
def test_scenario_krumtge_krum_layer_derived(light_server):
    cfg = _config(
        strategy="ScenarioKrumTGE",
        **{"num-malicious": 9, "defense-cohort-size": 20, "tge-ramp-rounds": 8},
    )
    server_app.server_fn(_ctx(cfg))
    plugins = light_server.last.plugins
    krum = plugins[0]
    assert krum.name == "KrumDefense"
    assert krum._num_malicious == 9
    assert krum._num_to_keep == 9
    # TGE layer is present; its keep is inert (0.7 threshold is the filter).
    assert plugins[1].name == "TGEnsemble"


@pytest.mark.unit
def test_legacy_fallback_num_malicious_from_fraction(light_server):
    # No num-malicious / defense-cohort-size keys -> num_malicious derived from
    # malicious-fraction exactly as before the fix. With 21 partitions and
    # fraction 0.2: f = int(21 * 0.2) = 4, keep = max(1, 21 - 4 - 2) = 15.
    cfg = _config(**{"malicious-fraction": 0.2})
    server_app.server_fn(_ctx(cfg))
    plugin = light_server.last.plugins[0]
    assert plugin._num_malicious == 4
    assert plugin._num_to_keep == 15


@pytest.mark.unit
def test_invalid_num_malicious_raises(light_server):
    # f >= cohort - 2 would drive keep to <= 0; the branch must fail loudly
    # rather than silently clamp to a near-no-op defense.
    cfg = _config(**{"num-malicious": 19, "defense-cohort-size": 20})
    with pytest.raises(ValueError):
        server_app.server_fn(_ctx(cfg))


@pytest.mark.unit
def test_scenario_krum_branches_enable_dynamic_f(light_server):
    """: Scenario* Krum layers must use per-round dynamic f
    (April formula ceil(n/2)-1) so S3/S4 disconnect rounds (n~11) stay
    computable; scenario-derived static values remain as provenance."""
    cfg = _config(**{"num-malicious": 9, "defense-cohort-size": 20})
    server_app.server_fn(_ctx(cfg))
    plugin = light_server.last.plugins[0]
    assert plugin._dynamic_f is True
    # Provenance statics preserved.
    assert plugin._num_malicious == 9
    assert plugin._num_to_keep == 9

    cfg = _config(
        strategy="ScenarioKrumTGE",
        **{"num-malicious": 9, "defense-cohort-size": 20, "tge-ramp-rounds": 8},
    )
    server_app.server_fn(_ctx(cfg))
    krum = light_server.last.plugins[0]
    assert krum.name == "KrumDefense"
    assert krum._dynamic_f is True


@pytest.mark.unit
def test_noncanonical_declared_count_keeps_dynamic_policy(light_server, capsys):
    """ round-2 P2 (labeling): with a non-canonical declared count
    (intensity arms declare 1/3/5/7 adversaries), the deployed f policy is
    threat-model-constant — dynamic ceil(n/2)-1, NOT the declared count —
    while the declared value is preserved as provenance and the print labels
    the two facts distinctly so the audit cannot conflate them."""
    cfg = _config(**{"num-malicious": 3, "defense-cohort-size": 20})
    server_app.server_fn(_ctx(cfg))
    plugin = light_server.last.plugins[0]

    # Deployed sizing: dynamic policy, independent of declared=3.
    assert plugin._dynamic_f is True
    assert plugin._effective_f(20) == 9  # f=9/keep=9 at a 20-client round

    # Declared count preserved as provenance statics.
    assert plugin._num_malicious == 3
    assert plugin._num_to_keep == 15  # max(1, 20 - 3 - 2), provenance only

    # Print carries both facts, distinctly labeled.
    out = capsys.readouterr().out
    assert "scenario_declared_adversaries=3" in out
    assert 'krum_f_policy="dynamic ceil(n/2)-1"' in out


@pytest.mark.unit
def test_legacy_scenario_caller_without_keys_f0_accepted(light_server):
    """: legacy scenario dev-scripts (e.g. run_rmc_flower.py) emit no
    sizing keys and default malicious-fraction=0.0 -> raw f=0. The validation
    must accept f=0 (only f<0 or f>=cohort-2 raise), and with dynamic_f=True
    the per-round sizing no longer depends on the static f."""
    cfg = _config()  # no num-malicious / defense-cohort-size / malicious-fraction
    server_app.server_fn(_ctx(cfg))
    plugin = light_server.last.plugins[0]
    assert plugin._num_malicious == 0        # raw, no max(1,.) clamp
    assert plugin._num_to_keep == 19         # max(1, 21 - 0 - 2), provenance only
    assert plugin._dynamic_f is True         # deployment sizing is per-round
    # Sanity: per-round effective f at a 20-client round is Szelag-faithful 9.
    assert plugin._effective_f(20) == 9

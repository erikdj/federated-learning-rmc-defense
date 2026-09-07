"""Erratum-B server_app wiring: observe-only knob + cuts-version knob.

The public workflow is described in `docs/reproduction/experiments.md`.
Historical protocol references: § B1
(RULED, methodology v1.53): the § B1 calibration units run the online H2'
detector in OBSERVE-ONLY mode PREPENDED to the existing BASE arms (Krum /
TrustScore / FedAvg-floor) — no new strategy classes, a generic
`h2p-observe-only` run-config knob. § B2: the cut table is selected
explicitly via `h2p-cuts-version` (closed {v1, v2}), never auto-detected.

Uses the light_server introspection harness (test_h4_chain_wiring.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import flowerfl.server_app as server_app  # noqa: E402
from h4_bundle_fixture import (  # noqa: E402
    make_synthetic_bundle,
    make_synthetic_bundle_v2,
)

S3 = str(PROJECT_ROOT / "rmc" / "scenarios" / "S3_identity_reset_only.json")
C0 = str(PROJECT_ROOT / "rmc" / "scenarios" / "C0_clean_no_attack.json")


class _CapturingScenarioStrategy:
    last = None

    def __init__(self, base, plugins=None, scenario_path=None,
                 eval_manager=None, signal_logger=None,
                 live_metric_logger=None):
        _CapturingScenarioStrategy.last = self
        self.base = base
        self.plugins = list(plugins or [])
        self.scenario_path = scenario_path

    def set_live_metric_logger(self, logger_obj):
        pass


@pytest.fixture
def light_server(monkeypatch):
    monkeypatch.setattr(server_app, "detect_input_shape", lambda dataset: 45)
    monkeypatch.setattr(
        server_app, "_create_eval_manager",
        lambda dataset, run_config=None: object(),
    )
    monkeypatch.setattr(
        server_app, "ScenarioStrategy", _CapturingScenarioStrategy
    )
    _CapturingScenarioStrategy.last = None
    return _CapturingScenarioStrategy


@pytest.fixture
def bundle_dir(tmp_path):
    return make_synthetic_bundle(tmp_path / "h4_serving")


@pytest.fixture
def bundle_v2_dir(tmp_path):
    return make_synthetic_bundle_v2(tmp_path / "h4_serving_v2")


def _ctx(run_config: dict):
    from flwr.common import Context
    from flwr.common.record.recorddict import RecordDict

    return Context(run_id=1, node_id=1, node_config={}, state=RecordDict(),
                   run_config=run_config)


def _config(strategy, bundle_dir, scenario=S3, **overrides) -> dict:
    cfg = {
        "dataset": "edge_full_20_rmc",
        "strategy": strategy,
        "scenario": scenario,
        "num-server-rounds": 3,
        "signal-log": 0,
        "seed": 42,
        "num-malicious": 9,
        "defense-cohort-size": 20,
        "h2p-bundle-dir": str(bundle_dir),
        "fp-cohort": "validation",
        "fp-registry-policy": "flag_gated",
    }
    cfg.update(overrides)
    return cfg


def _build(light_server, strategy, bundle_dir, **overrides):
    server_app.server_fn(_ctx(_config(strategy, bundle_dir, **overrides)))
    return light_server.last


#: observer-attached BASE arms: strategy -> (expected chain, arm_class).
OBSERVER_BASE = {
    "ScenarioKrum": (["H2PrimeDetector", "KrumDefense"], "krum_family"),
    "ScenarioTrustScore": (["H2PrimeDetector", "TrustScore"], "ts_family"),
    "ScenarioNone": (["H2PrimeDetector"], "fedavg_family"),
}


# ---------------------------------------------------------------------------
# observe-only knob on the base arms (erratum B § B1 calibration wiring)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("strategy", sorted(OBSERVER_BASE))
def test_observer_prepended_to_base_arm(light_server, bundle_dir, strategy):
    chain, arm_class = OBSERVER_BASE[strategy]
    built = _build(light_server, strategy, bundle_dir,
                   **{"h2p-observe-only": True})
    assert [p.name for p in built.plugins] == chain
    det = built.plugins[0]
    assert det.observe_only is True
    assert det.arm_class == arm_class


@pytest.mark.unit
@pytest.mark.parametrize("strategy", sorted(OBSERVER_BASE))
def test_base_arm_without_knob_is_unchanged(light_server, bundle_dir, strategy):
    built = _build(light_server, strategy, bundle_dir)
    chain, _ = OBSERVER_BASE[strategy]
    assert [p.name for p in built.plugins] == chain[1:]  # no detector


@pytest.mark.unit
def test_observer_base_arm_krum_layer_matches_standalone(light_server,
                                                         bundle_dir):
    built = _build(light_server, "ScenarioKrum", bundle_dir,
                   **{"h2p-observe-only": True})
    krum = built.plugins[-1]
    assert krum.name == "KrumDefense"
    assert krum._dynamic_f is True
    assert krum._num_to_keep == 9  # 20 - 9 - 2, unchanged by the observer


@pytest.mark.unit
def test_observer_on_c0_resolves_the_v1_alias(light_server, bundle_dir):
    """Calibration units run C0 with v1 cuts: the E3-bis alias still governs
    the LOGGED would-flag cut on the v1 path (retired only for v2)."""
    built = _build(light_server, "ScenarioNone", bundle_dir, scenario=C0,
                   **{"h2p-observe-only": True})
    det = built.plugins[0]
    assert det.scenario_token == "S0"
    assert det.observe_only is True


@pytest.mark.unit
def test_observe_knob_string_values_coerce_strictly(light_server, bundle_dir):
    built = _build(light_server, "ScenarioKrum", bundle_dir,
                   **{"h2p-observe-only": "true"})
    assert built.plugins[0].name == "H2PrimeDetector"
    built = _build(light_server, "ScenarioKrum", bundle_dir,
                   **{"h2p-observe-only": "0"})
    assert [p.name for p in built.plugins] == ["KrumDefense"]


@pytest.mark.unit
def test_observe_knob_garbage_value_refuses(light_server, bundle_dir):
    with pytest.raises(ValueError, match="h2p-observe-only"):
        _build(light_server, "ScenarioKrum", bundle_dir,
               **{"h2p-observe-only": "ture"})


@pytest.mark.unit
def test_observe_knob_on_unsupported_strategy_refuses(light_server, bundle_dir):
    """A calibration unit whose arm cannot host the observer must refuse
    loudly — silently running it observer-less would poison the census."""
    with pytest.raises(ValueError, match="h2p-observe-only"):
        _build(light_server, "ScenarioTGEnsemble", bundle_dir,
               **{"h2p-observe-only": True})


@pytest.mark.unit
def test_observe_knob_on_h2p_arm_makes_the_detector_observe(light_server,
                                                            bundle_dir):
    built = _build(light_server, "ScenarioH2PKrum", bundle_dir,
                   **{"h2p-observe-only": True})
    assert [p.name for p in built.plugins] == ["H2PrimeDetector",
                                               "KrumDefense"]
    assert built.plugins[0].observe_only is True


@pytest.mark.unit
def test_h2p_arm_without_knob_enforces(light_server, bundle_dir):
    built = _build(light_server, "ScenarioH2PKrum", bundle_dir)
    assert built.plugins[0].observe_only is False


# ---------------------------------------------------------------------------
# cuts-version knob (erratum B § B2 — explicit, never auto-detect)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h2p_arm_default_is_v1(light_server, bundle_v2_dir):
    built = _build(light_server, "ScenarioH2PKrum", bundle_v2_dir)
    det = built.plugins[0]
    assert det.cuts_version == "v1"  # v2 files present but NOT auto-selected


@pytest.mark.unit
def test_h2p_arm_pins_v2_and_selects_the_pair_cut(light_server, bundle_v2_dir):
    built = _build(light_server, "ScenarioH2PKrum", bundle_v2_dir,
                   **{"h2p-cuts-version": "v2"})
    det = built.plugins[0]
    assert det.cuts_version == "v2"
    assert det.arm_class == "krum_family"
    assert det.scenario_token == "S3"


@pytest.mark.unit
def test_v2_c0_scenario_uses_its_own_token(light_server, bundle_v2_dir):
    built = _build(light_server, "ScenarioH2PFP", bundle_v2_dir, scenario=C0,
                   **{"h2p-cuts-version": "v2"})
    det = built.plugins[0]
    assert det.scenario_token == "C0"
    assert det.arm_class == "fedavg_family"


@pytest.mark.unit
def test_unknown_cuts_version_refuses(light_server, bundle_v2_dir):
    from flowerfl.h2prime_online import H2PrimeBundleError

    with pytest.raises(H2PrimeBundleError, match="cuts.version|cuts_version"):
        _build(light_server, "ScenarioH2PKrum", bundle_v2_dir,
               **{"h2p-cuts-version": "v9"})


@pytest.mark.unit
def test_observer_base_arm_with_v2_cuts(light_server, bundle_v2_dir):
    """EXP-063+ style: observer on a base arm can pin v2 cuts too — the
    arm-class comes from the base strategy."""
    built = _build(light_server, "ScenarioTrustScore", bundle_v2_dir,
                   **{"h2p-observe-only": True, "h2p-cuts-version": "v2"})
    det = built.plugins[0]
    assert det.cuts_version == "v2"
    assert det.arm_class == "ts_family"

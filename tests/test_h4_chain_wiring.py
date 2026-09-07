"""H4 arm wiring in server_app — frozen § 7c-bis chain per strategy token.

Asserts, for each of the five new H4 strategies, the exact plugin chain
order (detector FIRST, FP second where present, aggregator plugin last),
the aggregator construction parity with the standalone arms (Krum
dynamic_f / TrustScore constants), the FLAG_GATED + validation-τ identity
configuration on FP-bearing arms, and that the incumbent chains are
untouched. Uses the established light_server introspection harness
(tests/test_server_scenario_krum_params.py) plus a synthetic serving
bundle routed via the `h2p-bundle-dir` run-config key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import flowerfl.server_app as server_app  # noqa: E402
from h4_bundle_fixture import make_synthetic_bundle  # noqa: E402

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


#: strategy -> expected plugin-name chain (§ 7c-bis order, frozen).
EXPECTED_CHAINS = {
    "ScenarioH2PFPKrum": ["H2PrimeDetector", "Fingerprint", "KrumDefense"],
    "ScenarioH2PFP": ["H2PrimeDetector", "Fingerprint"],
    "ScenarioH2PKrum": ["H2PrimeDetector", "KrumDefense"],
    "ScenarioH2PFPTS": ["H2PrimeDetector", "Fingerprint", "TrustScore"],
    "ScenarioH2PTS": ["H2PrimeDetector", "TrustScore"],
}


@pytest.mark.unit
@pytest.mark.parametrize("strategy", sorted(EXPECTED_CHAINS))
def test_chain_order_is_the_frozen_7cbis_order(light_server, bundle_dir,
                                               strategy):
    built = _build(light_server, strategy, bundle_dir)
    assert [p.name for p in built.plugins] == EXPECTED_CHAINS[strategy]
    # The base aggregation is FedAvg in every arm (Krum/TrustScore act as
    # plugins over the kept set, mirroring the standalone arms).
    from flwr.server.strategy import FedAvg

    assert type(built.base) is FedAvg


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["ScenarioH2PFPKrum", "ScenarioH2PKrum"])
def test_krum_layer_matches_standalone_scenario_krum(light_server, bundle_dir,
                                                     strategy):
    built = _build(light_server, strategy, bundle_dir)
    krum = built.plugins[-1]
    assert krum.name == "KrumDefense"
    assert krum._dynamic_f is True
    assert krum._num_malicious == 9
    assert krum._num_to_keep == 9  # 20 - 9 - 2


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["ScenarioH2PFPTS", "ScenarioH2PTS"])
def test_trustscore_layer_matches_standalone_arm(light_server, bundle_dir,
                                                 strategy):
    built = _build(light_server, strategy, bundle_dir)
    ts = built.plugins[-1]
    assert ts.name == "TrustScore"
    assert ts._decay_rate == 0.9
    assert ts._outlier_threshold == 2.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "strategy", ["ScenarioH2PFPKrum", "ScenarioH2PFP", "ScenarioH2PFPTS"]
)
def test_fp_layer_is_flag_gated_with_the_validation_tau(light_server,
                                                        bundle_dir, strategy):
    """§ 5 frozen identity configuration: FLAG_GATED policy, v2 validation-
    cohort τ via the locked-cohort mechanism (never a hardcoded float)."""
    from flowerfl.fingerprint_registry import (
        CalibrationCohort,
        RegistryPolicy,
        locked_tau,
    )

    built = _build(light_server, strategy, bundle_dir)
    fp = built.plugins[1]
    assert fp.name == "Fingerprint"
    registry = fp.registry
    assert registry.policy is RegistryPolicy.FLAG_GATED
    assert registry.tau == locked_tau(CalibrationCohort.VALIDATION)
    assert not getattr(registry, "is_observe_only", False)
    assert fp.enforcement_mode.value == "hard_drop"


@pytest.mark.unit
def test_detector_carries_bundle_sha_and_scenario_cut(light_server,
                                                      bundle_dir):
    import hashlib

    built = _build(light_server, "ScenarioH2PKrum", bundle_dir)
    det = built.plugins[0]
    expected_sha = hashlib.sha256(
        (bundle_dir / "manifest.json").read_bytes()
    ).hexdigest()
    assert det.bundle_sha256 == expected_sha
    assert det.scenario_token == "S3"


@pytest.mark.unit
def test_c0_scenario_uses_the_s0_cut_alias(light_server, bundle_dir):
    """E3-bis: on C0_clean_no_attack the detector resolves the S0 cut via
    the explicit declared alias."""
    built = _build(light_server, "ScenarioH2PTS", bundle_dir, scenario=C0)
    det = built.plugins[0]
    assert det.scenario_token == "S0"


@pytest.mark.unit
def test_missing_bundle_fails_at_startup(light_server, tmp_path):
    from flowerfl.h2prime_online import H2PrimeBundleError

    with pytest.raises(H2PrimeBundleError, match="manifest missing"):
        _build(light_server, "ScenarioH2PKrum", tmp_path / "no_bundle")


@pytest.mark.unit
def test_incumbent_chains_are_untouched(light_server, bundle_dir):
    """Reused arms (krum / trustscore / krum_tge_fp / fedavg=ScenarioNone)
    still build their frozen chains."""
    for strategy, chain in {
        "ScenarioKrum": ["KrumDefense"],
        "ScenarioTrustScore": ["TrustScore"],
        "ScenarioKrumTGEFP": ["KrumDefense", "TGEnsemble", "Fingerprint"],
        "ScenarioNone": [],
    }.items():
        built = _build(light_server, strategy, bundle_dir)
        assert [p.name for p in built.plugins] == chain, strategy

"""H3 fleet wiring — tokens, SUPPORTED_CONFIGS, plugin chains, pre-lock posture.

Closes the H3 dry-run gaps and emission-contract flag plumbing described in
`docs/reproduction/experiments.md`.

Design authority
----------------
* Historical emission contract — at smoke time **no τ and
  no Σ exist**; the FP arm must still run and produce emission at 100 % of
  client-rounds, both defense tokens, the aggregation coefficient, device
  ENROLMENT, custody and the schema-v5 fields — but **not** the matching branch,
  which is structurally unreachable pre-lock.
* v1.10 § 5.1 gate **(c)** — τ is locked in code before any eval run. A silent
  default τ is exactly what the gate forbids, so the pre-lock posture must be
  OBSERVE-ONLY, never a placeholder threshold.
* `flowerfl/fingerprint_plugin.py` module docstring — chain order
  `[Krum, TGE, Fingerprint]`, fingerprint **always last**: it observes the
  complete unfiltered cohort in `observe_cohort`, and being last is what lets
  `score_updates` infer which clients an upstream detector rejected.
* EXP-016 attempt-1 postmortem — an unmapped `_DEFENSE_TOKEN` entry trains for
  hours and then dies in finalization. `test_defense_token_covers_every_supported_config`
  in `tests/test_fleet_entrypoint.py` is the standing drift canary; these tests
  pin the two H3 tokens by name.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

S4 = str(PROJECT_ROOT / "rmc" / "scenarios" / "S4_full_mix.json")

#: The two H3 configuration labels and the signal-log tokens they must resolve
#: to. The token is the strategy class name with "Scenario" stripped and
#: lowercased — `flowerfl/server_app.py` line ~64.
H3_CONFIGS = {
    "TGE+FP": ("ScenarioTGEFP", "tgefp"),
    "Krum+TGE+FP": ("ScenarioKrumTGEFP", "krumtgefp"),
}

#: Plugin chains for the pre-existing configs, frozen here so a change to the
#: H3 branches that perturbs an incumbent arm fails loudly. EXP-048 is running
#: on this codebase's image lineage: every existing token must stay
#: byte-identical in behaviour.
INCUMBENT_CHAINS = {
    "ScenarioKrum": ["KrumDefense"],
    "ScenarioTGEnsemble": ["TGEnsemble"],
    "ScenarioKrumTGE": ["KrumDefense", "TGEnsemble"],
    "ScenarioTGEPrime": ["TGEnsemble"],
    "ScenarioKrumTGEPrime": ["KrumDefense", "TGEnsemble"],
    "ScenarioTrustScore": ["TrustScore"],
}


# ===========================================================================
# Gap 1 — defense tokens
# ===========================================================================

@pytest.mark.unit
def test_defense_token_round_trips_both_h3_tokens():
    """EXP-016 attempt-1 class: an unmapped token dies in FINALIZATION."""
    from docker.entrypoint import defense_token

    assert defense_token("TGE+FP") == "tgefp"
    assert defense_token("Krum+TGE+FP") == "krumtgefp"


@pytest.mark.unit
def test_h3_tokens_follow_the_class_name_lowered_convention():
    """The token IS `strategy_name.replace("Scenario", "").lower()` — the same
    derivation `server_app._signal_log_components` performs at runtime, so the
    map cannot drift from what the signal log actually writes."""
    for config, (strategy_name, token) in H3_CONFIGS.items():
        assert strategy_name.replace("Scenario", "").lower() == token, config


@pytest.mark.unit
def test_every_supported_config_still_has_a_token():
    """The standing drift canary, re-asserted here so this suite fails on its
    own if a config lands without a token."""
    from docker.entrypoint import defense_token
    from run_phase4_flower import SUPPORTED_CONFIGS

    for config in SUPPORTED_CONFIGS:
        assert defense_token(config)


# ===========================================================================
# Gap 2 — SUPPORTED_CONFIGS + strategy routing
# ===========================================================================

@pytest.mark.unit
def test_h3_configs_are_supported_by_the_runner():
    from run_phase4_flower import SUPPORTED_CONFIGS

    assert "TGE+FP" in SUPPORTED_CONFIGS
    assert "Krum+TGE+FP" in SUPPORTED_CONFIGS


@pytest.mark.unit
@pytest.mark.parametrize("config", sorted(H3_CONFIGS))
def test_h3_config_routes_to_its_scenario_token(config):
    from run_phase4_flower import build_strategy_for_config

    token, _cfg = build_strategy_for_config(config)
    assert token.__name__ == H3_CONFIGS[config][0]


@pytest.mark.unit
def test_h3_configs_inherit_the_gate_selected_ramp():
    """The FP arms are the incumbent TGE arms + the fingerprint plugin. The
    ramp is the v1.6 § 3 gate-selected r*=3, inherited verbatim so the FP arm
    differs from its incumbent by exactly one plugin."""
    from run_phase4_flower import build_strategy_for_config

    for config in H3_CONFIGS:
        _token, cfg = build_strategy_for_config(config)
        assert cfg["tge_ramp_rounds"] == 3, config


# ===========================================================================
# Gap 2 — plugin chains, fingerprint LAST
# ===========================================================================

class _CapturingScenarioStrategy:
    """Records the plugin chain `server_fn` builds, without a simulation."""

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
        self.live_metric_logger = logger_obj


@pytest.fixture
def light_server(monkeypatch):
    """Strip `server_fn` of its data-loading deps so the chain it builds can be
    introspected without parquet files or a FixedEvalManager."""
    import flowerfl.server_app as server_app

    monkeypatch.setattr(server_app, "detect_input_shape", lambda dataset: 45)
    monkeypatch.setattr(
        server_app, "_create_eval_manager", lambda dataset, run_config=None: object()
    )
    monkeypatch.setattr(server_app, "ScenarioStrategy", _CapturingScenarioStrategy)
    _CapturingScenarioStrategy.last = None
    return _CapturingScenarioStrategy


def _ctx(run_config: dict):
    from flwr.common import Context
    from flwr.common.record.recorddict import RecordDict

    return Context(
        run_id=1, node_id=1, node_config={}, state=RecordDict(),
        run_config=run_config,
    )


def _config(**overrides) -> dict:
    cfg = {
        "dataset": "edge_full_20_rmc",
        "strategy": "ScenarioKrum",
        "scenario": S4,
        "num-server-rounds": 3,
        "signal-log": 0,
        "num-malicious": 9,
        "defense-cohort-size": 20,
    }
    cfg.update(overrides)
    return cfg


def _chain(light, strategy_name: str, **overrides) -> list:
    """Build an FP chain. Declares a cohort by default (post-EXP-050).

    Before the τ-lock the `fp-cohort` key was inert, so these shape tests could
    omit it. Post-lock `build_fingerprint_registry` REFUSES an undeclared
    cohort — deliberately, since the validation and adjudicating τ/Σ pairs are
    different instruments. That refusal is the lock working, not a wiring bug,
    so the shape tests declare a cohort and the refusal itself is asserted
    directly by `test_an_undeclared_cohort_refuses_once_tau_is_locked`.
    """
    import flowerfl.server_app as server_app

    overrides.setdefault("fp-cohort", "validation")
    server_app.server_fn(_ctx(_config(strategy=strategy_name, **overrides)))
    return light.last.plugins


def _simulate_unlocked(monkeypatch) -> None:
    """Restore the pre-lock tree so the observe-only posture stays under test.

    τ is now permanently locked in this tree (gate (c) is one-way), but the
    observe-only fallback is still live code — anything that makes the artifact
    unverifiable lands on it. Simulating the unlocked state is the only way to
    keep exercising it; the alternative is deleting coverage of the path a
    corrupted artifact would take.
    """
    from flowerfl import fingerprint_registry as fpr

    monkeypatch.setattr(fpr, "TAU_VALIDATION_ALL_DEVICES", None)
    monkeypatch.setattr(fpr, "TAU_ADJUDICATING_EVEN_DEVICES", None)


@pytest.mark.unit
def test_tgefp_chain_is_tge_then_fingerprint_last(light_server):
    plugins = _chain(light_server, "ScenarioTGEFP")
    assert [p.name for p in plugins] == ["TGEnsemble", "Fingerprint"]


@pytest.mark.unit
def test_krumtgefp_chain_is_krum_tge_then_fingerprint_last(light_server):
    plugins = _chain(light_server, "ScenarioKrumTGEFP")
    assert [p.name for p in plugins] == ["KrumDefense", "TGEnsemble", "Fingerprint"]


@pytest.mark.unit
@pytest.mark.parametrize("strategy_name", ["ScenarioTGEFP", "ScenarioKrumTGEFP"])
def test_the_fingerprint_plugin_is_always_last(light_server, strategy_name):
    """Load-bearing (plugin module docstring): the plugin observes the complete
    unfiltered cohort, and being LAST is what lets `score_updates` infer which
    clients an upstream detector rejected."""
    plugins = _chain(light_server, strategy_name)
    assert plugins[-1].name == "Fingerprint"
    assert [p.name for p in plugins].count("Fingerprint") == 1


@pytest.mark.unit
def test_the_fp_arms_are_their_incumbents_plus_one_plugin(light_server):
    """`TGE+FP` == `TGE` + FP, `Krum+TGE+FP` == `Krum+TGE` + FP — asserted on
    the chains themselves so the FP arm can never silently differ by a second
    variable."""
    tge = [p.name for p in _chain(light_server, "ScenarioTGEnsemble")]
    tgefp = [p.name for p in _chain(light_server, "ScenarioTGEFP")]
    assert tgefp == tge + ["Fingerprint"]

    krumtge = [p.name for p in _chain(light_server, "ScenarioKrumTGE")]
    krumtgefp = [p.name for p in _chain(light_server, "ScenarioKrumTGEFP")]
    assert krumtgefp == krumtge + ["Fingerprint"]


@pytest.mark.unit
@pytest.mark.parametrize("strategy_name", sorted(INCUMBENT_CHAINS))
def test_preexisting_chains_are_unchanged(light_server, strategy_name):
    """OFF BY DEFAULT: every EXISTING config token must be byte-identical in
    behaviour. EXP-048 is running on this codebase's image lineage."""
    plugins = _chain(light_server, strategy_name)
    assert [p.name for p in plugins] == INCUMBENT_CHAINS[strategy_name]
    assert all(p.name != "Fingerprint" for p in plugins)


@pytest.mark.unit
def test_krum_layer_sizing_is_identical_in_the_fp_arm(light_server):
    """The Krum layer of `Krum+TGE+FP` must be built exactly as `Krum+TGE`'s —
    same f, same keep, same dynamic policy."""
    krum_incumbent = _chain(light_server, "ScenarioKrumTGE")[0]
    krum_fp = _chain(light_server, "ScenarioKrumTGEFP")[0]
    assert (krum_fp._num_malicious, krum_fp._num_to_keep, krum_fp._dynamic_f) == (
        krum_incumbent._num_malicious,
        krum_incumbent._num_to_keep,
        krum_incumbent._dynamic_f,
    )


# ===========================================================================
# The `fingerprint-enabled` run-config flag
# ===========================================================================

@pytest.mark.unit
def test_fingerprint_enabled_is_declared_in_pyproject_and_defaults_off():
    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    m = re.search(r"^fingerprint-enabled\s*=\s*(\w+)", text, re.M)
    assert m is not None, "fingerprint-enabled missing from pyproject.toml"
    assert m.group(1) == "false", (
        f"a bare `flwr run` must never emit fingerprints; got {m.group(1)!r}"
    )


@pytest.mark.unit
def test_only_fp_configs_turn_fingerprint_emission_on():
    """No other config can turn emission on accidentally.

    The FP-bearing set now includes the three H4 composition arms that carry
    the fingerprint plugin (spec 2026-08-16 § 5 / BUILD_CONTRACT): emission
    stays reachable ONLY from a config label whose chain includes the FP
    layer — never from an incumbent or detector-only token."""
    from run_phase4_flower import SUPPORTED_CONFIGS, build_strategy_for_config

    fp_bearing = set(H3_CONFIGS) | {"H2P+FP+Krum", "H2P+FP", "H2P+FP+TS"}
    for config in SUPPORTED_CONFIGS:
        _token, cfg = build_strategy_for_config(config)
        enabled = cfg.get("fingerprint-enabled", False)
        if config in fp_bearing:
            assert enabled is True, config
        else:
            assert enabled is False, config


@pytest.mark.unit
def test_the_config_label_is_the_only_path_that_enables_emission():
    """No CLI flag, no manifest `run_extras` family, no entrypoint argv carries
    the fingerprint knob — unlike SMOTE / Stage-F / leakage, whose run-config
    overrides all have an operator-reachable path. Emission is reachable ONLY by
    selecting a `+FP` config label, so an incumbent token cannot be talked into
    emitting fingerprints."""
    runner_src = (PROJECT_ROOT / "scripts" / "run_phase4_flower.py").read_text()
    assert "--fingerprint" not in runner_src

    entrypoint_src = (PROJECT_ROOT / "docker" / "entrypoint.py").read_text()
    assert "fingerprint-enabled" not in entrypoint_src
    assert "fingerprint_enabled" not in entrypoint_src


@pytest.mark.unit
def test_the_flag_key_is_the_one_the_client_reads():
    """Source assertion: the runner's key and `client_app`'s lookup are the same
    string. A mismatch would run the FP arm with ZERO fingerprints emitted —
    gate (e) requires 100 % of client-rounds."""
    client_src = (PROJECT_ROOT / "flowerfl" / "client_app.py").read_text()
    assert 'run_config.get("fingerprint-enabled", False)' in client_src

    runner_src = (PROJECT_ROOT / "scripts" / "run_phase4_flower.py").read_text()
    assert '"fingerprint-enabled": True' in runner_src


# ===========================================================================
# Gap 3 — the PRE-LOCK OBSERVE-ONLY posture
# ===========================================================================

def _tau_is_locked() -> bool:
    from flowerfl import fingerprint_registry as fpr

    return fpr.TAU_VALIDATION_ALL_DEVICES is not None


@pytest.mark.unit
def test_tau_is_locked_in_this_tree():
    """The canary this section was written around — now FLIPPED.

    It previously asserted `not _tau_is_locked()`, guarding the premise of the
    pre-lock tests below. EXP-050 locked both τ (2026-08-10), which is exactly
    the event the original docstring anticipated ("when τ lands, this canary
    flips and the locked-posture tests take over"). Same guard, opposite sign:
    it now fails loudly if τ is ever un-locked, which gate (c) forbids.
    """
    assert _tau_is_locked()


@pytest.mark.unit
def test_locked_registry_is_a_real_matching_registry(light_server):
    """Post-lock the FP arm must MATCH, not merely observe."""
    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    fp_plugin = _chain(light_server, "ScenarioTGEFP")[-1]
    assert not isinstance(fp_plugin.registry, ObserveOnlyFingerprintRegistry)
    assert getattr(fp_plugin.registry, "is_observe_only", False) is False
    assert fp_plugin.registry.tau == pytest.approx(18.639429816855873)


@pytest.mark.unit
def test_the_locked_posture_is_loud_at_construction(light_server, capsys):
    """The operator must SEE which instrument the run is scored under."""
    _chain(light_server, "ScenarioTGEFP")
    out = capsys.readouterr().out.lower()
    assert "locked registry" in out
    assert "cohort=validation" in out
    assert "gate (c) satisfied" in out


@pytest.mark.unit
def test_prelock_registry_is_observe_only(light_server, monkeypatch):
    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    _simulate_unlocked(monkeypatch)
    fp_plugin = _chain(light_server, "ScenarioTGEFP", **{"fp-cohort": None})[-1]
    assert isinstance(fp_plugin.registry, ObserveOnlyFingerprintRegistry)
    assert fp_plugin.registry.is_observe_only is True


@pytest.mark.unit
def test_prelock_posture_is_loud_at_construction(light_server, capsys, monkeypatch):
    """The operator must SEE the posture at construction — a silent observe-only
    arm looks exactly like a working one until the metric comes back empty."""
    _simulate_unlocked(monkeypatch)
    _chain(light_server, "ScenarioTGEFP", **{"fp-cohort": None})
    out = capsys.readouterr().out.lower()
    assert "observe-only" in out
    assert "tau not locked" in out


@pytest.mark.unit
def test_observe_only_registry_has_no_numeric_tau():
    """Gate (c): an un-pre-registered default τ must never exist. There is no
    placeholder threshold anywhere in the object."""
    import numpy as np

    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = ObserveOnlyFingerprintRegistry(dim=4)
    assert not np.isfinite(registry.tau)


@pytest.mark.unit
def test_observe_only_registry_still_enrols_and_observes():
    """§ 6.2: the pre-lock smoke must still get device ENROLMENT and custody."""
    import numpy as np

    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = ObserveOnlyFingerprintRegistry(dim=4)
    registry.observe("client_0", np.arange(4.0), server_round=1, logical_id="client_0")
    registry.observe("client_1", np.arange(4.0) + 1, server_round=1, logical_id="client_1")
    assert len(registry) == 2
    entry = registry.entry_for_session("client_0")
    assert entry.first_seen_round == 1
    assert entry.generation == 0


@pytest.mark.unit
def test_observe_only_registry_can_never_assert_a_match():
    """Structurally impossible, two ways over: no candidate is ever considered,
    AND there is no threshold a distance could fall under. Fed a BYTE-IDENTICAL
    fingerprint to a flagged entry — the most favourable possible input."""
    import numpy as np

    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = ObserveOnlyFingerprintRegistry(dim=4)
    vector = np.arange(4.0)
    registry.observe("client_0", vector, server_round=1, logical_id="client_0")
    registry.flag("client_0", reason="upstream_filter", server_round=1)

    result = registry.observe(
        "client_0_new1", vector.copy(), server_round=5, logical_id="client_0_new1"
    )
    assert result.assertion is not None          # the row IS written (custody)
    assert result.assertion.asserted_match is False
    assert result.assertion.asserted_parent_entry_id is None
    assert result.assertion.generation == 0
    assert result.flagged is False


@pytest.mark.unit
def test_observe_only_never_inherits_a_flag_even_across_many_generations():
    import numpy as np

    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = ObserveOnlyFingerprintRegistry(dim=4)
    vector = np.arange(4.0)
    for index in range(6):
        key = "client_0" if index == 0 else f"client_0_new{index}"
        registry.observe(key, vector.copy(), server_round=index + 1, logical_id=key)
        registry.flag(key, reason="upstream_filter", server_round=index + 1)
    assert all(e.flag_reason == "upstream_filter" for e in registry.entries())
    assert all(e.generation == 0 for e in registry.entries())


@pytest.mark.unit
def test_the_same_code_path_picks_up_a_locked_tau_with_no_further_change(monkeypatch):
    """When τ IS locked the factory returns a plain, matching registry carrying
    the LOCKED τ and the LOCKED metric — no code change, no flag flip."""
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import build_fingerprint_registry

    metric = fpr.MahalanobisMetric.identity(fpr.FINGERPRINT_DIM)
    monkeypatch.setattr(fpr, "TAU_VALIDATION_ALL_DEVICES", 19.09)
    monkeypatch.setattr(fpr, "locked_metric", lambda cohort: metric)

    registry = build_fingerprint_registry({"fp-cohort": "validation"})
    assert not getattr(registry, "is_observe_only", False)
    assert registry.tau == pytest.approx(19.09)
    assert registry.metric is metric


@pytest.mark.unit
def test_an_undeclared_cohort_refuses_once_tau_is_locked(monkeypatch):
    """Post-lock, the run must SAY which locked τ it is scored under. Silently
    defaulting to one of the two cohorts is the failure gate (c) exists to
    prevent. Pre-lock the key is inert (nothing to choose)."""
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import build_fingerprint_registry

    monkeypatch.setattr(fpr, "TAU_VALIDATION_ALL_DEVICES", 19.09)
    monkeypatch.setattr(fpr, "TAU_ADJUDICATING_EVEN_DEVICES", 26.15)
    monkeypatch.setattr(
        fpr, "locked_metric", lambda cohort: fpr.MahalanobisMetric.identity(fpr.FINGERPRINT_DIM)
    )
    with pytest.raises(ValueError, match="fp-cohort"):
        build_fingerprint_registry({})


@pytest.mark.unit
def test_an_undeclared_cohort_is_inert_before_the_lock(monkeypatch):
    from flowerfl.server_app import build_fingerprint_registry

    _simulate_unlocked(monkeypatch)
    registry = build_fingerprint_registry({})
    assert registry.is_observe_only is True


@pytest.mark.unit
def test_an_unknown_cohort_label_is_rejected():
    from flowerfl.server_app import build_fingerprint_registry

    with pytest.raises(ValueError):
        build_fingerprint_registry({"fp-cohort": "whichever"})


@pytest.mark.unit
def test_a_damaged_locked_artifact_stops_the_run_instead_of_downgrading(monkeypatch):
    """PR #52 round-2 P1: post-lock integrity failure must PROPAGATE.

    The observe-only downgrade is for the genuinely pre-lock state only. A
    locked artifact whose bytes are off the SHA-256 pin (or missing entirely)
    must refuse to build a registry at all — otherwise an H3 evaluation with a
    damaged artifact runs to completion with no re-link matching and the unit
    looks like a valid result.
    """
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import build_fingerprint_registry

    monkeypatch.setattr(fpr, "CALIBRATION_ARTIFACT_SHA256", "00" * 32)
    with pytest.raises(fpr.TauLockIntegrityError):
        build_fingerprint_registry({"fp-cohort": "validation"})


@pytest.mark.unit
def test_the_fp_plugin_runs_the_pre_registered_hard_drop_action(light_server):
    """D4 is the H3 arm; `downweight`/`accept` exist only for the § 5.1
    integrity test and must never be what the fleet deploys."""
    from flowerfl.fingerprint_plugin import EnforcementMode

    for strategy_name in ("ScenarioTGEFP", "ScenarioKrumTGEFP"):
        fp_plugin = _chain(light_server, strategy_name)[-1]
        assert fp_plugin.enforcement_mode is EnforcementMode.HARD_DROP

"""H4 arm tokens in the runner + custody export semantics.

Covers: the nine-arm token set (BUILD_CONTRACT + erratum-A), the config ->
unit-id token rule (label.replace('+','_').lower()), the baked frozen
operating conditions per H4 token, the E4 eval-split CLI/custody plumbing,
`serving_bundle_sha256` null semantics, the eval-split cache-reuse guard,
and the entrypoint's fleet reachability (defense token + eval_split argv).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

#: config label -> (strategy class name, unit-id token, signal defense token)
H4_TOKENS = {
    "H2P+FP+Krum": ("ScenarioH2PFPKrum", "h2p_fp_krum", "h2pfpkrum"),
    "H2P+FP":      ("ScenarioH2PFP", "h2p_fp", "h2pfp"),
    "H2P+Krum":    ("ScenarioH2PKrum", "h2p_krum", "h2pkrum"),
    "H2P+FP+TS":   ("ScenarioH2PFPTS", "h2p_fp_ts", "h2pfpts"),
    "H2P+TS":      ("ScenarioH2PTS", "h2p_ts", "h2pts"),
    "FedAvg":      ("ScenarioNone", "fedavg", "none"),
}

#: The reused arms every H4 launch also needs (must stay supported).
REUSED = ["Krum", "TrustScore", "Krum+TGE+FP"]


@pytest.mark.unit
def test_all_nine_arm_tokens_are_supported():
    from run_phase4_flower import SUPPORTED_CONFIGS

    for label in list(H4_TOKENS) + REUSED:
        assert label in SUPPORTED_CONFIGS, label


@pytest.mark.unit
@pytest.mark.parametrize("label", sorted(H4_TOKENS))
def test_config_routes_to_strategy_and_unit_token(label):
    from run_phase4_flower import build_strategy_for_config

    token, _cfg = build_strategy_for_config(label)
    strategy_name, unit_token, defense_token = H4_TOKENS[label]
    assert token.__name__ == strategy_name
    # unit-id token = the exp_name rule in run_one (replace('+','_').lower())
    assert label.replace("+", "_").lower() == unit_token
    # signal-row defense = class name lowered (existing convention)
    assert strategy_name.replace("Scenario", "").lower() == defense_token


@pytest.mark.unit
@pytest.mark.parametrize("label", ["H2P+FP+Krum", "H2P+FP", "H2P+FP+TS"])
def test_fp_bearing_h2p_arms_bake_the_frozen_identity_config(label):
    """§ 5 frozen: FLAG_GATED + validation cohort (τ via the locked-cohort
    mechanism) + client fingerprint emission + sealed-test eval."""
    from run_phase4_flower import build_strategy_for_config

    _tok, cfg = build_strategy_for_config(label)
    assert cfg["fingerprint-enabled"] is True
    assert cfg["fp-cohort"] == "validation"
    assert cfg["fp-registry-policy"] == "flag_gated"
    assert cfg["eval-split"] == "sealed_test"
    assert cfg["cs_enabled"] is False


@pytest.mark.unit
@pytest.mark.parametrize("label", ["H2P+Krum", "H2P+TS"])
def test_no_fp_h2p_arms_carry_no_fingerprint_keys(label):
    from run_phase4_flower import build_strategy_for_config

    _tok, cfg = build_strategy_for_config(label)
    assert "fingerprint-enabled" not in cfg
    assert "fp-cohort" not in cfg
    assert "fp-registry-policy" not in cfg
    assert cfg["eval-split"] == "sealed_test"


@pytest.mark.unit
def test_fedavg_token_is_a_pure_no_defense_baseline():
    from run_phase4_flower import build_strategy_for_config

    _tok, cfg = build_strategy_for_config("FedAvg")
    assert cfg == {"cs_enabled": False}  # no baked eval-split: CLI supplies it


# ===========================================================================
# E4 eval-split plumbing + custody
# ===========================================================================

@pytest.mark.unit
def test_eval_split_cli_mapping():
    from run_phase4_flower import eval_split_run_config_from_cli

    assert eval_split_run_config_from_cli(None) == {}
    assert eval_split_run_config_from_cli("") == {}
    assert eval_split_run_config_from_cli("sealed_test") == {
        "eval-split": "sealed_test"
    }


@pytest.mark.unit
def test_eval_split_provenance_defaults_to_legacy_with_null_sha():
    from run_phase4_flower import eval_split_provenance

    assert eval_split_provenance({}) == {
        "eval_split": "legacy", "eval_split_manifest_sha256": None,
    }


@pytest.mark.unit
def test_eval_split_provenance_reads_sha_from_the_actual_manager():
    from run_phase4_flower import eval_split_provenance

    mgr = SimpleNamespace(manifest_sha256="ff" * 32)
    prov = eval_split_provenance({"eval-split": "sealed_test"}, mgr)
    assert prov == {
        "eval_split": "sealed_test",
        "eval_split_manifest_sha256": "ff" * 32,
    }


@pytest.mark.unit
def test_eval_split_cache_guard_blocks_cross_population_reuse():
    from run_phase4_flower import _eval_split_cache_reusable

    legacy_result = {"provenance": {}}
    sealed_result = {"provenance": {"eval_split": "sealed_test"}}
    active_sealed = {"eval-split": "sealed_test"}
    assert not _eval_split_cache_reusable(legacy_result, active_sealed)
    assert _eval_split_cache_reusable(sealed_result, active_sealed)
    assert _eval_split_cache_reusable(legacy_result, None)
    assert not _eval_split_cache_reusable(sealed_result, None)


@pytest.mark.unit
def test_sealed_test_rerun_is_cache_reusable_despite_holdout_flag():
    """An IDENTICAL sealed-test rerun must be
    reusable. The launch config carries the runner default
    holdout-disjoint=true while the sealed evaluator truthfully records
    holdout_disjoint=false (it samples/excludes nothing); comparing that flag
    broke cache/self-heal reuse for every sealed unit. The sealed identity is
    the manifest sha, compared against the ACTIVE tree's manifest bytes."""
    from run_phase4_flower import (
        _active_sealed_manifest_sha256,
        _holdout_cache_reusable,
    )

    active_sha = _active_sealed_manifest_sha256()
    assert active_sha is not None and len(active_sha) == 64
    cached = {"provenance": {
        "eval_split": "sealed_test",
        "eval_split_manifest_sha256": active_sha,
        "holdout_disjoint": False,   # truthful sealed provenance, unchanged
    }}
    active_cfg = {"eval-split": "sealed_test", "holdout-disjoint": True}
    assert _holdout_cache_reusable(cached, active_cfg) is True


@pytest.mark.unit
def test_sealed_test_cache_refuses_a_different_manifest_sha():
    from run_phase4_flower import _holdout_cache_reusable

    active_cfg = {"eval-split": "sealed_test", "holdout-disjoint": True}
    wrong_sha = {"provenance": {
        "eval_split": "sealed_test",
        "eval_split_manifest_sha256": "0" * 64,
        "holdout_disjoint": False,
    }}
    assert _holdout_cache_reusable(wrong_sha, active_cfg) is False
    missing_sha = {"provenance": {
        "eval_split": "sealed_test",
        "holdout_disjoint": False,
    }}
    assert _holdout_cache_reusable(missing_sha, active_cfg) is False


@pytest.mark.unit
def test_legacy_holdout_cache_semantics_are_byte_unchanged():
    """The legacy comparison is untouched: disjoint-vs-disjoint reuses,
    a pre-v8 result (missing flag) is reusable only by a
    --no-holdout-disjoint run, and the sealed branch never engages when
    either side is legacy."""
    from run_phase4_flower import _holdout_cache_reusable

    disjoint_cached = {"provenance": {"holdout_disjoint": True}}
    legacy_cached = {"provenance": {}}
    assert _holdout_cache_reusable(disjoint_cached, None) is True
    assert _holdout_cache_reusable(legacy_cached, None) is False
    assert _holdout_cache_reusable(
        legacy_cached, {"holdout-disjoint": False}
    ) is True
    # cross-pairing (active sealed, cached legacy): falls through to the
    # legacy flag comparison here; the eval-split guard refuses the reuse
    # overall regardless — no weakening either way.
    sealed_active = {"eval-split": "sealed_test", "holdout-disjoint": True}
    assert _holdout_cache_reusable(disjoint_cached, sealed_active) is True
    from run_phase4_flower import _eval_split_cache_reusable
    assert _eval_split_cache_reusable(disjoint_cached, sealed_active) is False


@pytest.mark.unit
def test_serving_bundle_sha_null_semantics():
    """Null — not empty string — for arms without the detector; the actual
    plugin's sha for detector arms (never a launch-record value)."""
    from run_phase4_flower import h4_serving_provenance

    no_detector = SimpleNamespace(_plugins=[SimpleNamespace(name="KrumDefense")])
    prov = h4_serving_provenance(no_detector)
    assert prov["serving_bundle_sha256"] is None
    assert prov["serving_bundle_sha256"] != ""

    detector = SimpleNamespace(
        _plugins=[SimpleNamespace(name="H2PrimeDetector",
                                  bundle_sha256="ab" * 32,
                                  cuts_version="v1",
                                  arm_class="krum_family")]
    )
    # erratum B: the cut-table identity rides alongside the bundle sha,
    # read from the same ACTUAL plugin.
    assert h4_serving_provenance(detector) == {
        "serving_bundle_sha256": "ab" * 32,
        "h2p_cuts_version": "v1",
        "h2p_arm_class": "krum_family",
    }
    none_prov = h4_serving_provenance(None)
    assert none_prov["serving_bundle_sha256"] is None
    assert none_prov["h2p_cuts_version"] is None
    assert none_prov["h2p_arm_class"] is None


@pytest.mark.unit
def test_run_uid_provenance_reads_the_actual_signal_logger():
    from run_phase4_flower import run_uid_provenance

    strategy = SimpleNamespace(
        _signal_logger=SimpleNamespace(run_uid="flower_persistent__S3__x__seed42__t0")
    )
    assert run_uid_provenance(strategy) == {
        "run_uid": "flower_persistent__S3__x__seed42__t0"
    }
    # Truthful null when signal logging was disabled / strategy absent.
    assert run_uid_provenance(SimpleNamespace(_signal_logger=None)) == {
        "run_uid": None
    }
    assert run_uid_provenance(None) == {"run_uid": None}


@pytest.mark.unit
def test_universal_run_uid_equals_the_fp_registry_binding():
    """Lane C custody audit: on FP-bearing arms provenance.run_uid and
    fingerprint_registry.run_uid are EQUAL by construction — both read the
    same SignalLogger identity, the registry side via the set_run_id stamp
    ScenarioStrategy applies at construction."""
    import numpy as np
    from flwr.server.strategy import FedAvg
    from flwr.common import ndarrays_to_parameters

    from flowerfl.fingerprint_plugin import FingerprintDefensePlugin
    from flowerfl.fingerprint_registry import (
        FingerprintRegistry, MahalanobisMetric,
    )
    from flowerfl.scenario_strategy import ScenarioStrategy
    from run_phase4_flower import _fingerprint_custody, run_uid_provenance

    fp = FingerprintDefensePlugin(
        registry=FingerprintRegistry(
            tau=1.0, metric=MahalanobisMetric.identity(8), dim=8
        ),
        run_id="",  # must be OVERWRITTEN by the strategy's stamp
        expected_dim=8,
    )
    fake_logger = SimpleNamespace(
        run_uid="flower_persistent__S3__h2pfpkrum__seed42__2026-08-17T00:00:00"
    )
    base = FedAvg(initial_parameters=ndarrays_to_parameters(
        [np.zeros(4, dtype=np.float32)]))
    strategy = ScenarioStrategy(base, plugins=[fp], scenario_path=None,
                                signal_logger=fake_logger)
    prov = run_uid_provenance(strategy)
    custody = _fingerprint_custody(strategy._plugins)
    assert prov["run_uid"] == fake_logger.run_uid
    assert custody["run_uid"] == prov["run_uid"]


# ===========================================================================
# FIX 1 — matrix_doc run_extras.eval_split reachability
# ===========================================================================

import textwrap  # noqa: E402


def _matrix_doc(tmp_path, run_extras_yaml: str):
    template = textwrap.dedent('''\
        ---
        exp_id: EXP-TEST
        slug: h4-composition
        hypothesis: H4
        methodology_version: v1.52
        matrix:
          defenses: [Krum, TrustScore, Krum+TGE+FP, FedAvg]
          scenarios: [C0, S0, S1, S2, S3, S4]
          seeds: [1]
          mode: persistent_optimizer
          max_per_client: 2000000
          rounds: 50
        batch:
          job_queue: praxis-spot-queue
          job_definition: praxis-flowerfl-unit
        @RUN_EXTRAS@---
        H4 sealed-fleet matrix.
        ''')
    p = tmp_path / "doc.md"
    p.write_text(template.replace("@RUN_EXTRAS@", run_extras_yaml))
    return p


@pytest.mark.unit
def test_matrix_doc_accepts_eval_split_sealed_test(tmp_path):
    """Launch-blocking : without this the reused arms can never
    reach the sealed evaluator — the manifest is refused at parse time."""
    from praxis_exp.matrix_doc import parse_matrix

    p = _matrix_doc(tmp_path, "run_extras:\n  eval_split: sealed_test\n")
    doc = parse_matrix(p)
    assert doc.run_extras == {"eval_split": "sealed_test"}


@pytest.mark.unit
def test_matrix_doc_canonicalizes_eval_split_case(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix

    p = _matrix_doc(tmp_path, "run_extras:\n  eval_split: ' Sealed_Test '\n")
    assert parse_matrix(p).run_extras == {"eval_split": "sealed_test"}


@pytest.mark.unit
def test_matrix_doc_accepts_eval_split_legacy_and_absent(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix

    p = _matrix_doc(tmp_path, "run_extras:\n  eval_split: legacy\n")
    assert parse_matrix(p).run_extras == {"eval_split": "legacy"}
    sub = tmp_path / "sub"
    sub.mkdir()
    p2 = _matrix_doc(sub, "")
    assert parse_matrix(p2).run_extras == {}


@pytest.mark.unit
def test_matrix_doc_refuses_unknown_eval_split_value(tmp_path):
    from praxis_exp.matrix_doc import MatrixDocError, parse_matrix

    p = _matrix_doc(tmp_path, "run_extras:\n  eval_split: bogus\n")
    with pytest.raises(MatrixDocError, match="eval_split"):
        parse_matrix(p)


# ===========================================================================
# fleet reachability — entrypoint
# ===========================================================================

@pytest.mark.unit
def test_every_h4_config_has_a_defense_token():
    from docker.entrypoint import defense_token

    for label, (_s, _u, token) in H4_TOKENS.items():
        assert defense_token(label) == token


@pytest.mark.unit
def test_entrypoint_eval_split_argv():
    from docker.entrypoint import eval_split_argv_from_run_extras

    assert eval_split_argv_from_run_extras(None) == []
    assert eval_split_argv_from_run_extras({}) == []
    assert eval_split_argv_from_run_extras(
        {"eval_split": "sealed_test"}
    ) == ["--eval-split", "sealed_test"]
    assert eval_split_argv_from_run_extras(
        {"eval_split": " Sealed_Test "}
    ) == ["--eval-split", "sealed_test"]
    with pytest.raises(ValueError, match="eval_split"):
        eval_split_argv_from_run_extras({"eval_split": "bogus"})


@pytest.mark.unit
def test_runner_argv_carries_eval_split_only_when_declared():
    from docker.entrypoint import runner_argv
    from praxis_exp.units import Unit

    unit = Unit("H2P+FP+Krum", "S3_identity_reset_only",
                "persistent_optimizer", 42, 2_000_000, 50, 0)
    base = runner_argv(unit, scenario_dir="rmc/scenarios",
                       out_dir=Path("/tmp/out"), run_extras=None)
    assert "--eval-split" not in base
    with_split = runner_argv(unit, scenario_dir="rmc/scenarios",
                             out_dir=Path("/tmp/out"),
                             run_extras={"eval_split": "sealed_test"})
    assert with_split[-2:] == ["--eval-split", "sealed_test"]

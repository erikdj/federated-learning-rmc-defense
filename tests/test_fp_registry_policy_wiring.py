"""fp-registry-policy fleet reachability — the corrected-H3 registry mode.

`docs/reproduction/experiments.md`
§ 1 (authorised by methodology v1.49) makes the enroll-everyone, identity-only
candidate pool THE H3 instrument, while the deployed flag-gated pool stays for
H4's composition arms. That choice has to be a DECLARED FACT OF THE RUN,
readable off the result artifact, and it has to be expressible from a design
doc — a mode reachable only in-process is a mode the fleet cannot launch
(image-update checklist item 7, the GWU-59 silent-default class).

Asserts the same chain tests/test_fp_cohort_wiring.py asserts for `fp_cohort`:

  matrix_doc run_extras -> manifest -> entrypoint argv -> REAL runner parser
  -> run-config override -> provenance field -> the registry server_app builds

plus byte-identity of argv / run-config / registry behaviour when the knob is
ABSENT, so every pre-existing experiment is untouched.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ===========================================================================
# 1. entrypoint argv builder
# ===========================================================================

def _unit():
    from praxis_exp.units import Unit
    return Unit("TGE+FP", "S3_identity_reset_only", "persistent_optimizer",
                42, 2_000_000, 50, 0)


def test_policy_argv_absent_is_empty():
    from docker.entrypoint import fp_registry_policy_argv_from_run_extras
    assert fp_registry_policy_argv_from_run_extras(None) == []
    assert fp_registry_policy_argv_from_run_extras({}) == []
    assert fp_registry_policy_argv_from_run_extras({"fp_cohort": "validation"}) == []


@pytest.mark.parametrize("policy", ["flag_gated", "identity_only"])
def test_policy_argv_maps_flag(policy):
    from docker.entrypoint import fp_registry_policy_argv_from_run_extras
    assert fp_registry_policy_argv_from_run_extras(
        {"fp_registry_policy": policy}) == ["--fp-registry-policy", policy]


def test_policy_argv_canonicalizes_case_and_whitespace():
    from docker.entrypoint import fp_registry_policy_argv_from_run_extras
    assert fp_registry_policy_argv_from_run_extras(
        {"fp_registry_policy": "  Identity_Only "}
    ) == ["--fp-registry-policy", "identity_only"]


@pytest.mark.parametrize("bad", ["everyone", "", None, "identity-only", 1])
def test_policy_argv_raises_on_garbage(bad):
    from docker.entrypoint import fp_registry_policy_argv_from_run_extras
    with pytest.raises(ValueError, match="fp_registry_policy"):
        fp_registry_policy_argv_from_run_extras({"fp_registry_policy": bad})


def test_runner_argv_byte_identical_without_the_policy():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios",
                       out_dir=Path("/out"), run_extras=None)
    assert base == runner_argv(u, scenario_dir="rmc/scenarios",
                               out_dir=Path("/out"), run_extras={})
    assert "--fp-registry-policy" not in base


def test_runner_argv_appends_the_policy_with_run_extras():
    from docker.entrypoint import runner_argv
    argv = runner_argv(_unit(), scenario_dir="rmc/scenarios", out_dir=Path("/out"),
                       run_extras={"fp_registry_policy": "identity_only"})
    assert argv[-2:] == ["--fp-registry-policy", "identity_only"]


def test_runner_argv_carries_the_policy_alongside_the_cohort():
    """The corrected H3 declares BOTH: which tau/Sigma, and which pool."""
    from docker.entrypoint import runner_argv
    argv = runner_argv(
        _unit(), scenario_dir="rmc/scenarios", out_dir=Path("/out"),
        run_extras={"fp_cohort": "adjudicating",
                    "fp_registry_policy": "identity_only"},
    )
    assert "--fp-cohort" in argv and "adjudicating" in argv
    assert "--fp-registry-policy" in argv and "identity_only" in argv


# ===========================================================================
# 2. runner CLI -> run-config -> provenance
# ===========================================================================

def test_cli_choices_are_exactly_the_registry_enum():
    """Single-source guard: the runner's accepted values and the enum the
    registry enforces must never drift apart."""
    from flowerfl.fingerprint_registry import RegistryPolicy
    from run_phase4_flower import add_fp_registry_policy_cli_args
    parser = argparse.ArgumentParser()
    add_fp_registry_policy_cli_args(parser)
    action = next(a for a in parser._actions if a.dest == "fp_registry_policy")
    assert list(action.choices) == [p.value for p in RegistryPolicy]
    assert action.default is None


def test_run_config_mapper_absent_is_noop():
    from run_phase4_flower import fp_registry_policy_run_config_from_cli
    assert fp_registry_policy_run_config_from_cli(None) == {}
    assert fp_registry_policy_run_config_from_cli("") == {}


@pytest.mark.parametrize("policy", ["flag_gated", "identity_only"])
def test_run_config_mapper_emits_the_hyphenated_server_key(policy):
    from run_phase4_flower import fp_registry_policy_run_config_from_cli
    assert fp_registry_policy_run_config_from_cli(policy) == {
        "fp-registry-policy": policy
    }


def test_provenance_always_declares_the_policy_field():
    """None must be distinguishable from an explicit flag_gated declaration."""
    from run_phase4_flower import fp_registry_policy_provenance
    assert fp_registry_policy_provenance({}) == {"fp_registry_policy": None}
    assert fp_registry_policy_provenance(
        {"fp-registry-policy": "identity_only"}
    ) == {"fp_registry_policy": "identity_only"}


# ===========================================================================
# 3. matrix_doc parse + the full chain through the REAL parser
# ===========================================================================

_DOC = textwrap.dedent('''\
    ---
    exp_id: EXP-057
    slug: h3-corrected-instrument
    hypothesis: H3
    methodology_version: v1.49
    matrix:
      defenses: [TGE+FP]
      scenarios: [S3_identity_reset_only]
      seeds: [42]
      mode: persistent_optimizer
      max_per_client: 2000000
      rounds: 50
    batch:
      job_queue: praxis-spot-queue
      job_definition: praxis-flowerfl-unit
    {run_extras}---
    fp-registry-policy wiring probe.
    ''')


def _doc_text(block=""):
    return _DOC.format(run_extras=block)


@pytest.mark.parametrize("raw,canon", [
    ("flag_gated", "flag_gated"),
    ("identity_only", "identity_only"),
    ("Identity_Only", "identity_only"),
])
def test_matrix_doc_policy_parsed_and_canonicalized(tmp_path, raw, canon):
    from praxis_exp.matrix_doc import parse_matrix
    p = tmp_path / "d.md"
    p.write_text(_doc_text(f"run_extras:\n  fp_registry_policy: {raw}\n"))
    assert parse_matrix(p).run_extras["fp_registry_policy"] == canon


def test_matrix_doc_policy_garbage_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_registry_policy: everyone\n"))
    with pytest.raises(MatrixDocError, match="fp_registry_policy"):
        parse_matrix(p)


def test_matrix_doc_policy_typo_key_rejected(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_registry_policies: identity_only\n"))
    with pytest.raises(MatrixDocError, match="unknown run_extras"):
        parse_matrix(p)


def test_policy_full_chain_through_real_parser(tmp_path):
    """End-to-end: run_extras -> entrypoint argv -> REAL runner parser ->
    run-config override -> provenance, with nothing dropped along the way."""
    from praxis_exp.matrix_doc import parse_matrix
    from docker.entrypoint import fp_registry_policy_argv_from_run_extras
    from run_phase4_flower import (
        add_fp_registry_policy_cli_args,
        fp_registry_policy_provenance,
        fp_registry_policy_run_config_from_cli,
    )
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_registry_policy: identity_only\n"))
    extras = parse_matrix(p).run_extras
    argv = fp_registry_policy_argv_from_run_extras(extras)
    parser = argparse.ArgumentParser()
    add_fp_registry_policy_cli_args(parser)
    args = parser.parse_args(argv)
    run_config = fp_registry_policy_run_config_from_cli(args.fp_registry_policy)
    assert run_config == {"fp-registry-policy": "identity_only"}
    assert fp_registry_policy_provenance(run_config) == {
        "fp_registry_policy": "identity_only"
    }


# ===========================================================================
# 4. cache-reuse identity: a cached result produced under one candidate pool is
#    NOT a unit of the other — different pool = different instrument.
# ===========================================================================

def _cached(policy=None, omit_field=False):
    prov = {} if omit_field else {"fp_registry_policy": policy}
    return {"trajectory": [1], "return_code": 0, "provenance": prov}


def test_cache_matching_policy_is_reusable():
    from run_phase4_flower import _fp_registry_policy_cache_reusable
    assert _fp_registry_policy_cache_reusable(
        _cached("identity_only"), {"fp-registry-policy": "identity_only"})
    assert _fp_registry_policy_cache_reusable(_cached(None), None)
    assert _fp_registry_policy_cache_reusable(_cached(None), {})


def test_cache_differing_policy_forces_recompute():
    from run_phase4_flower import _fp_registry_policy_cache_reusable
    assert not _fp_registry_policy_cache_reusable(
        _cached("flag_gated"), {"fp-registry-policy": "identity_only"})
    assert not _fp_registry_policy_cache_reusable(
        _cached("identity_only"), {"fp-registry-policy": "flag_gated"})


def test_cache_declared_run_never_reuses_undeclared_result():
    from run_phase4_flower import _fp_registry_policy_cache_reusable
    assert not _fp_registry_policy_cache_reusable(
        _cached(None), {"fp-registry-policy": "identity_only"})
    # pre-wiring result JSON with NO field at all reads as None
    assert not _fp_registry_policy_cache_reusable(
        _cached(omit_field=True), {"fp-registry-policy": "identity_only"})


def test_cache_undeclared_run_may_reuse_incumbent_result():
    from run_phase4_flower import _fp_registry_policy_cache_reusable
    assert _fp_registry_policy_cache_reusable(_cached(omit_field=True), None)


# ===========================================================================
# 5. server_app consumption — the run-config actually changes the instrument
# ===========================================================================

def test_absent_key_builds_the_flag_gated_incumbent():
    from flowerfl.fingerprint_registry import RegistryPolicy
    from flowerfl.server_app import build_fingerprint_registry
    from run_phase4_flower import fp_cohort_run_config_from_cli

    registry = build_fingerprint_registry(fp_cohort_run_config_from_cli("validation"))
    assert registry.policy is RegistryPolicy.FLAG_GATED


@pytest.mark.parametrize("policy", ["flag_gated", "identity_only"])
def test_the_declared_policy_reaches_the_locked_registry(policy):
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import build_fingerprint_registry
    from run_phase4_flower import (
        fp_cohort_run_config_from_cli,
        fp_registry_policy_run_config_from_cli,
    )

    run_config = {
        **fp_cohort_run_config_from_cli("adjudicating"),
        **fp_registry_policy_run_config_from_cli(policy),
    }
    registry = build_fingerprint_registry(run_config)
    assert not getattr(registry, "is_observe_only", False)
    assert registry.tau == fpr.TAU_ADJUDICATING_EVEN_DEVICES
    assert registry.policy is fpr.RegistryPolicy(policy)


def test_an_unknown_policy_in_run_config_is_a_loud_refusal():
    from flowerfl.server_app import build_fingerprint_registry
    with pytest.raises(ValueError, match="fp-registry-policy"):
        build_fingerprint_registry(
            {"fp-cohort": "validation", "fp-registry-policy": "everyone"}
        )


def test_the_prelock_observe_only_registry_carries_the_declared_policy():
    """A pre-lock smoke of the corrected arm must still record what it declared,
    and must still be structurally unable to assert a match under EITHER pool."""
    import numpy as np
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import ObserveOnlyFingerprintRegistry

    registry = ObserveOnlyFingerprintRegistry(
        reason="test", policy=fpr.RegistryPolicy.IDENTITY_ONLY
    )
    assert registry.policy is fpr.RegistryPolicy.IDENTITY_ONLY
    assert registry.is_observe_only
    vector = np.zeros(fpr.FINGERPRINT_DIM)
    registry.observe("client_1", vector, 1, logical_id="client_1")
    result = registry.observe("client_1_new1", vector, 3, logical_id="client_1_new1")
    assert result.assertion.asserted_match is False
    assert result.flagged is False


# ===========================================================================
# 6. custody — the result artifact records which pool produced the events
# ===========================================================================

def test_the_custody_block_records_the_registry_policy():
    import numpy as np
    from flowerfl.fingerprint_plugin import FingerprintDefensePlugin
    from flowerfl.fingerprint_registry import (
        FingerprintRegistry,
        MahalanobisMetric,
        RegistryPolicy,
    )
    from run_phase4_flower import _fingerprint_custody

    for policy in RegistryPolicy:
        registry = FingerprintRegistry(
            tau=1.0, metric=MahalanobisMetric.identity(), policy=policy
        )
        registry.observe(
            "client_1", np.zeros(registry.metric.dim), 1, logical_id="client_1"
        )
        custody = _fingerprint_custody([FingerprintDefensePlugin(registry=registry)])
        assert custody["registry_policy"] == policy.value

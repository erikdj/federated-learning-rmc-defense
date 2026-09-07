"""fp-cohort fleet reachability — the H3 post-τ-lock cohort declaration.

PR #52 locked both τ/Σ pairs and made server_app hard-refuse a post-lock FP-arm
run that does not declare WHICH cohort scores it (`run_config["fp-cohort"]`).
That refusal is only useful if the fleet path can actually make the
declaration: without this wiring, `fp-cohort` was expressible in-process (the
wiring tests drive run_config directly) but NOT from a design doc, so the
post-lock S3 re-entry rehearsal and both H3 cohorts could not launch at all
(image-update checklist item 7 — the GWU-59 silent-default class, except loud).

Asserts the reachability chain end-to-end, mirroring
tests/test_normalization_leak.py section 3-4:

  matrix_doc run_extras -> manifest -> entrypoint argv -> REAL runner parser
  -> run-config override -> provenance field

and byte-identity of argv/run-config/provenance when the knob is absent, so
every pre-lock and non-FP experiment is untouched.
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


def test_fp_cohort_argv_absent_is_empty():
    from docker.entrypoint import fp_cohort_argv_from_run_extras
    assert fp_cohort_argv_from_run_extras(None) == []
    assert fp_cohort_argv_from_run_extras({}) == []
    assert fp_cohort_argv_from_run_extras({"normalize_train_only": True}) == []


@pytest.mark.parametrize("cohort", ["validation", "adjudicating"])
def test_fp_cohort_argv_maps_flag(cohort):
    from docker.entrypoint import fp_cohort_argv_from_run_extras
    assert fp_cohort_argv_from_run_extras(
        {"fp_cohort": cohort}) == ["--fp-cohort", cohort]


def test_fp_cohort_argv_canonicalizes_case_and_whitespace():
    from docker.entrypoint import fp_cohort_argv_from_run_extras
    assert fp_cohort_argv_from_run_extras(
        {"fp_cohort": "  Validation "}) == ["--fp-cohort", "validation"]


@pytest.mark.parametrize("bad", ["whichever", "", None, "both", 1])
def test_fp_cohort_argv_raises_on_garbage(bad):
    from docker.entrypoint import fp_cohort_argv_from_run_extras
    with pytest.raises(ValueError, match="fp_cohort"):
        fp_cohort_argv_from_run_extras({"fp_cohort": bad})


def test_runner_argv_byte_identical_without_fp_cohort():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios",
                       out_dir=Path("results/EXP-052/x"))
    for extras in (None, {}):
        assert runner_argv(u, scenario_dir="rmc/scenarios",
                           out_dir=Path("results/EXP-052/x"),
                           run_extras=extras) == base


def test_runner_argv_appends_fp_cohort_with_run_extras():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios",
                       out_dir=Path("results/EXP-052/x"))
    argv = runner_argv(
        u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-052/x"),
        run_extras={"fp_cohort": "validation"},
    )
    assert argv == base + ["--fp-cohort", "validation"]


# ===========================================================================
# 2. runner CLI + run-config + provenance
# ===========================================================================

def test_cli_choices_are_exactly_the_server_enum():
    """Single-source guard: the runner's accepted values and the enum
    server_app enforces must never drift apart."""
    from flowerfl.fingerprint_registry import CalibrationCohort
    from run_phase4_flower import add_fp_cohort_cli_args
    parser = argparse.ArgumentParser()
    add_fp_cohort_cli_args(parser)
    action = next(a for a in parser._actions if a.dest == "fp_cohort")
    assert list(action.choices) == [c.value for c in CalibrationCohort]
    assert action.default is None


def test_run_config_mapper_absent_is_noop():
    from run_phase4_flower import fp_cohort_run_config_from_cli
    assert fp_cohort_run_config_from_cli(None) == {}
    assert fp_cohort_run_config_from_cli("") == {}


@pytest.mark.parametrize("cohort", ["validation", "adjudicating"])
def test_run_config_mapper_emits_the_hyphenated_server_key(cohort):
    from run_phase4_flower import fp_cohort_run_config_from_cli
    assert fp_cohort_run_config_from_cli(cohort) == {"fp-cohort": cohort}


def test_provenance_always_declares_the_cohort_field():
    """None must be distinguishable from a declared cohort on the artifact."""
    from run_phase4_flower import fp_cohort_provenance
    assert fp_cohort_provenance({}) == {"fp_cohort": None}
    assert fp_cohort_provenance({"fp-cohort": "validation"}) == {
        "fp_cohort": "validation"
    }


# ===========================================================================
# 3. matrix_doc parse + the full chain through the REAL parser
# ===========================================================================

_DOC = textwrap.dedent('''\
    ---
    exp_id: EXP-052
    slug: fp-rehearsal-probe
    hypothesis: H3
    methodology_version: v1.47
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
    fp-cohort wiring probe.
    ''')


def _doc_text(block=""):
    return _DOC.format(run_extras=block)


@pytest.mark.parametrize("raw,canon", [
    ("validation", "validation"),
    ("adjudicating", "adjudicating"),
    ("Validation", "validation"),
])
def test_matrix_doc_fp_cohort_parsed_and_canonicalized(tmp_path, raw, canon):
    from praxis_exp.matrix_doc import parse_matrix
    p = tmp_path / "d.md"
    p.write_text(_doc_text(f"run_extras:\n  fp_cohort: {raw}\n"))
    assert parse_matrix(p).run_extras["fp_cohort"] == canon


def test_matrix_doc_fp_cohort_garbage_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_cohort: whichever\n"))
    with pytest.raises(MatrixDocError, match="fp_cohort"):
        parse_matrix(p)


def test_matrix_doc_fp_cohort_typo_key_rejected(tmp_path):
    """The allowlist catches the un-plumbed-key class loudly at parse."""
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_cohorts: validation\n"))
    with pytest.raises(MatrixDocError, match="unknown run_extras"):
        parse_matrix(p)


def test_fp_cohort_full_chain_through_real_parser(tmp_path):
    """End-to-end: run_extras -> entrypoint argv -> REAL runner parser ->
    run-config override -> provenance, with nothing dropped along the way."""
    from praxis_exp.matrix_doc import parse_matrix
    from docker.entrypoint import fp_cohort_argv_from_run_extras
    from run_phase4_flower import (
        add_fp_cohort_cli_args,
        fp_cohort_provenance,
        fp_cohort_run_config_from_cli,
    )
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  fp_cohort: validation\n"))
    extras = parse_matrix(p).run_extras
    argv = fp_cohort_argv_from_run_extras(extras)
    parser = argparse.ArgumentParser()
    add_fp_cohort_cli_args(parser)
    args = parser.parse_args(argv)
    run_config = fp_cohort_run_config_from_cli(args.fp_cohort)
    assert run_config == {"fp-cohort": "validation"}
    assert fp_cohort_provenance(run_config) == {"fp_cohort": "validation"}


# ===========================================================================
# 4. cache-reuse identity: a cached result scored under one
#    locked cohort must NEVER be returned as a unit of the other — different
#    cohort = different instrument (different τ and Σ).
# ===========================================================================

def _cached(fp_cohort=None, omit_field=False):
    prov = {} if omit_field else {"fp_cohort": fp_cohort}
    return {"trajectory": [1], "return_code": 0, "provenance": prov}


def test_fp_cohort_cache_matching_cohort_is_reusable():
    from run_phase4_flower import _fp_cohort_cache_reusable
    assert _fp_cohort_cache_reusable(
        _cached("validation"), {"fp-cohort": "validation"})
    assert _fp_cohort_cache_reusable(_cached(None), None)
    assert _fp_cohort_cache_reusable(_cached(None), {})


def test_fp_cohort_cache_differing_cohort_forces_recompute():
    from run_phase4_flower import _fp_cohort_cache_reusable
    assert not _fp_cohort_cache_reusable(
        _cached("validation"), {"fp-cohort": "adjudicating"})
    assert not _fp_cohort_cache_reusable(
        _cached("adjudicating"), {"fp-cohort": "validation"})


def test_fp_cohort_cache_declared_run_never_reuses_undeclared_result():
    from run_phase4_flower import _fp_cohort_cache_reusable
    assert not _fp_cohort_cache_reusable(_cached(None), {"fp-cohort": "validation"})
    # pre-wiring result JSON with NO fp_cohort field at all reads as None
    assert not _fp_cohort_cache_reusable(
        _cached(omit_field=True), {"fp-cohort": "adjudicating"})


def test_fp_cohort_cache_undeclared_run_may_reuse_prelock_result():
    from run_phase4_flower import _fp_cohort_cache_reusable
    assert _fp_cohort_cache_reusable(_cached(omit_field=True), None)


def test_fp_cohort_run_config_satisfies_the_locked_server_gate():
    """The declared cohort must actually open server_app's post-lock gate:
    the exact run_config the chain produces builds a LOCKED (non-observe-only)
    registry carrying the pinned validation τ."""
    from flowerfl import fingerprint_registry as fpr
    from flowerfl.server_app import build_fingerprint_registry
    from run_phase4_flower import fp_cohort_run_config_from_cli

    registry = build_fingerprint_registry(
        fp_cohort_run_config_from_cli("validation")
    )
    assert not getattr(registry, "is_observe_only", False)
    assert registry.tau == fpr.TAU_VALIDATION_ALL_DEVICES

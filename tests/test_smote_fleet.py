"""Fleet-path SMOTE pass-through (GWU-59): make the knob reachable from the
Batch chain, not just `flwr run`.

The chain: matrix design-doc frontmatter `run_extras:` -> matrix_launch writes it
into the manifest meta -> docker/entrypoint.py::runner_argv reads meta and appends
the runner CLI flags -> scripts/run_phase4_flower.py injects the run-config keys.
Refills re-run against the ORIGINAL manifest, so run_extras is reproduced; the
refill record also carries it for the audit trail.

Launch-blocking property under test: WITHOUT run_extras every surface is
byte-identical to today (SMOTE off); WITH it the flags/keys flow end to end.
"""
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ===========================================================================
# 1. matrix_doc: optional run_extras frontmatter + loud unknown-key validation
# ===========================================================================

_DOC = textwrap.dedent('''\
    ---
    exp_id: EXP-017
    slug: smote-smoke
    hypothesis: H2
    methodology_version: v1.26
    matrix:
      defenses: [Krum+TGE]
      scenarios: [control_honest, S4]
      seeds: [42]
      mode: persistent_optimizer
      max_per_client: 2000000
      rounds: 50
    batch:
      job_queue: praxis-spot-queue
      job_definition: praxis-flowerfl-unit
    {run_extras}---
    SMOTE Stage-D smoke.
    ''')


def _doc_text(run_extras_block=""):
    return _DOC.format(run_extras=run_extras_block)


def test_matrix_doc_run_extras_absent_defaults_empty(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix
    p = tmp_path / "d.md"; p.write_text(_doc_text())
    assert parse_matrix(p).run_extras == {}


def test_matrix_doc_run_extras_parsed(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix
    block = "run_extras:\n  smote_enabled: true\n  smote_variant: smote\n  smote_target: balanced\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    assert parse_matrix(p).run_extras == {
        "smote_enabled": True, "smote_variant": "smote", "smote_target": "balanced",
    }


def test_matrix_doc_unknown_run_extras_key_raises_loudly(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = "run_extras:\n  smote_enabled: true\n  bogus_key: 1\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="run_extras"):
        parse_matrix(p)


def test_matrix_doc_run_extras_must_be_mapping(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = "run_extras: [smote_enabled]\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="run_extras"):
        parse_matrix(p)


# ===========================================================================
# 2. runner CLI: flag -> run-config injection (default OFF preserved)
# ===========================================================================

def test_smote_run_config_from_cli_off_is_empty():
    from run_phase4_flower import smote_run_config_from_cli
    assert smote_run_config_from_cli(False, "smote", "balanced") == {}


def test_smote_run_config_from_cli_on_injects_hyphenated_keys():
    from run_phase4_flower import smote_run_config_from_cli
    assert smote_run_config_from_cli(True, "random_over", "0.5") == {
        "smote-enabled": True, "smote-variant": "random_over", "smote-target": "0.5",
    }


def test_runner_parser_defaults_smote_off():
    """A bare invocation parses to smote disabled with incumbent defaults."""
    import argparse
    from run_phase4_flower import add_smote_cli_args
    p = argparse.ArgumentParser()
    add_smote_cli_args(p)
    args = p.parse_args([])
    assert args.smote_enabled is False
    assert args.smote_variant == "smote"
    assert args.smote_target == "balanced"


def test_runner_parser_accepts_smote_flags():
    import argparse
    from run_phase4_flower import add_smote_cli_args
    p = argparse.ArgumentParser()
    add_smote_cli_args(p)
    args = p.parse_args(["--smote-enabled", "--smote-variant", "random_over",
                         "--smote-target", "0.5"])
    assert args.smote_enabled is True
    assert args.smote_variant == "random_over"
    assert args.smote_target == "0.5"


# ===========================================================================
# 3. entrypoint: runner_argv byte-identity without run_extras; flags with it
# ===========================================================================

def _unit():
    from praxis_exp.units import Unit
    return Unit("Krum+TGE", "S4", "persistent_optimizer", 42, 2_000_000, 50, 0)


def test_runner_argv_byte_identical_without_run_extras():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-017/x"))
    for extras in (None, {}):
        assert runner_argv(u, scenario_dir="rmc/scenarios",
                           out_dir=Path("results/EXP-017/x"), run_extras=extras) == base


def test_runner_argv_appends_smote_flags_with_run_extras():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-017/x"))
    argv = runner_argv(
        u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-017/x"),
        run_extras={"smote_enabled": True, "smote_variant": "smote", "smote_target": "balanced"},
    )
    assert argv[:len(base)] == base  # existing argv unchanged, flags appended
    assert argv[len(base):] == ["--smote-enabled", "--smote-variant", "smote",
                                "--smote-target", "balanced"]


def test_smote_argv_from_run_extras_disabled_is_empty():
    from docker.entrypoint import smote_argv_from_run_extras
    assert smote_argv_from_run_extras({}) == []
    assert smote_argv_from_run_extras({"smote_enabled": False}) == []
    assert smote_argv_from_run_extras(None) == []


# ===========================================================================
# 4. matrix_launch: run_extras carried into the manifest meta (round-trip)
# ===========================================================================

def _launch_repo(tmp_path, run_extras_block=""):
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "experiments" / "EXP-017-smote-smoke.md").write_text(
        _doc_text(run_extras_block)
    )
    return tmp_path


def _mlflow():
    c = MagicMock()
    c.get_or_create_experiment.return_value = "mlexp-1"
    c.create_run.return_value = "run-1"
    return c


def _git():
    g = MagicMock()
    g.working_tree_clean.return_value = True
    g.head_sha.return_value = "abc1234"
    g.tag_target_sha.return_value = None
    return g


def test_launch_writes_run_extras_into_manifest_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    from praxis_exp.storage import InMemoryObjectStore
    from praxis_exp.batch import FakeBatchSubmitter
    from praxis_exp.manifest import read_manifest
    from praxis_exp.matrix_launch import launch_matrix

    block = "run_extras:\n  smote_enabled: true\n  smote_variant: smote\n  smote_target: balanced\n"
    repo = _launch_repo(tmp_path, block)
    store = InMemoryObjectStore()
    launch_matrix(
        repo, "EXP-017", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(), _git=_git(),
        _no_push=True,
    )
    _, meta, _ = read_manifest(store, "EXP-017")
    assert meta["run_extras"] == {
        "smote_enabled": True, "smote_variant": "smote", "smote_target": "balanced",
    }


def test_launch_absent_run_extras_meta_is_empty_mapping(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    from praxis_exp.storage import InMemoryObjectStore
    from praxis_exp.batch import FakeBatchSubmitter
    from praxis_exp.manifest import read_manifest
    from praxis_exp.matrix_launch import launch_matrix

    repo = _launch_repo(tmp_path)  # no run_extras block
    store = InMemoryObjectStore()
    launch_matrix(
        repo, "EXP-017", image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=FakeBatchSubmitter(), _mlflow=_mlflow(), _git=_git(),
        _no_push=True,
    )
    _, meta, _ = read_manifest(store, "EXP-017")
    assert meta["run_extras"] == {}


# ===========================================================================
# 5. matrix_refill: the refill record carries run_extras from the prior manifest
# ===========================================================================

def test_refill_record_carries_run_extras(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    from praxis_exp.storage import InMemoryObjectStore, marker_key
    from praxis_exp.batch import FakeBatchSubmitter
    from praxis_exp.manifest import write_manifest
    from praxis_exp.matrix_refill import refill_matrix, _refill_record_key
    from praxis_exp.units import expand_matrix

    repo = _launch_repo(tmp_path)  # doc matrix must match the seeded units
    units = expand_matrix(["Krum+TGE"], ["control_honest", "S4"], [42],
                          "persistent_optimizer", 2_000_000, 50)
    store = InMemoryObjectStore()
    run_extras = {"smote_enabled": True, "smote_variant": "smote", "smote_target": "balanced"}
    write_manifest(store, "EXP-017", units, {
        "methodology_version": "v1.26", "image_digest": "sha256:deadbeef",
        "git_sha": "abc1234", "n_units": len(units),
        "launched_at": "2026-07-26T00:00:00Z",
        "parent_run_id": "parent-orig", "array_job_id": "arr-orig",
        "run_extras": run_extras,
    })
    # mark all but array_index 0 done -> exactly one missing cell to refill
    for u in units:
        if u.array_index != 0:
            store.put_bytes(marker_key("EXP-017", u.unit_id), b"")
    missing = [u.unit_id for u in units if u.array_index == 0]

    git = MagicMock()
    git.working_tree_clean.return_value = True
    git.head_sha.return_value = "abc1234"
    git.tag_target_sha.side_effect = lambda _r, name: {"exp/EXP-017": "0ld"}.get(name)
    mlflow = MagicMock(); mlflow.get_or_create_experiment.return_value = "mlexp-1"
    mlflow.create_run.return_value = "refill-parent-1"

    out = refill_matrix(
        repo, "EXP-017", missing, image_digest="sha256:deadbeef",
        container_tracking_uri="http://10.0.0.10:5000",
        _store=store, _batch=FakeBatchSubmitter(), _mlflow=mlflow, _git=git, _no_push=True,
    )
    rec = json.loads(store.get_bytes(_refill_record_key("EXP-017", out["serial"])))
    assert rec["meta"]["run_extras"] == run_extras


# ===========================================================================
# 6. local result cache must not return a SMOTE-off result as
#    the SMOTE arm. Cache reuse is gated on SMOTE provenance identity.
# ===========================================================================

def _cached(enabled=False, variant=None, target=None):
    prov = {"smote_enabled": enabled, "smote_variant": variant, "smote_target": target}
    return {"trajectory": [{"round": 1}], "return_code": 0, "provenance": prov}


def test_cache_reusable_off_result_off_run():
    from run_phase4_flower import _smote_cache_reusable
    assert _smote_cache_reusable(_cached(enabled=False), None) is True


def test_cache_NOT_reusable_off_result_smote_on_run():
    """THE P1: a cached SMOTE-off result must NOT satisfy a --smote-enabled run."""
    from run_phase4_flower import _smote_cache_reusable
    extra = {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    assert _smote_cache_reusable(_cached(enabled=False), extra) is False


def test_cache_reusable_matching_smote_identity():
    from run_phase4_flower import _smote_cache_reusable
    extra = {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    assert _smote_cache_reusable(_cached(True, "smote", "balanced"), extra) is True


def test_cache_NOT_reusable_variant_differs():
    from run_phase4_flower import _smote_cache_reusable
    extra = {"smote-enabled": True, "smote-variant": "random_over", "smote-target": "balanced"}
    assert _smote_cache_reusable(_cached(True, "smote", "balanced"), extra) is False


def test_cache_NOT_reusable_target_differs():
    from run_phase4_flower import _smote_cache_reusable
    extra = {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    # cached target 0.5 (float, as JSON round-trips) vs active balanced
    assert _smote_cache_reusable(_cached(True, "smote", 0.5), extra) is False


def test_cache_reusable_legacy_result_without_smote_provenance():
    """A pre-SMOTE cached result (no smote_* provenance) is reusable for an
    off run, but not for a smote-on run."""
    from run_phase4_flower import _smote_cache_reusable
    legacy = {"trajectory": [{"round": 1}], "return_code": 0, "provenance": {}}
    assert _smote_cache_reusable(legacy, None) is True
    assert _smote_cache_reusable(
        legacy, {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    ) is False


# ===========================================================================
# 7. unrecognized smote_enabled must fail LOUDLY, not coerce
#    to false (which would silently run the whole arm incumbent). Both layers.
# ===========================================================================

@pytest.mark.parametrize("bad_val", ["ture", "1.5", "yes  "])
def test_matrix_doc_bad_smote_enabled_rejected_at_parse(tmp_path, bad_val):
    # QUOTED so YAML yields a genuine string — bareword yes/on/1.5 would be
    # coerced by YAML itself (yes -> bool True) before our validator runs; the
    # danger case this guards is a quoted/typo'd string silently disabling SMOTE.
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = f'run_extras:\n  smote_enabled: "{bad_val}"\n  smote_variant: smote\n'
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="smote_enabled"):
        parse_matrix(p)


@pytest.mark.parametrize("good_block,expected", [
    ("run_extras:\n  smote_enabled: true\n", True),
    ("run_extras:\n  smote_enabled: false\n", False),
])
def test_matrix_doc_canonical_smote_enabled_accepted(tmp_path, good_block, expected):
    from praxis_exp.matrix_doc import parse_matrix
    p = tmp_path / "d.md"; p.write_text(_doc_text(good_block))
    assert parse_matrix(p).run_extras["smote_enabled"] is expected


def test_matrix_doc_bad_smote_variant_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = "run_extras:\n  smote_enabled: true\n  smote_variant: adasyn\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="smote_variant"):
        parse_matrix(p)


def test_matrix_doc_bad_smote_target_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = "run_extras:\n  smote_enabled: true\n  smote_target: 2\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="smote_target"):
        parse_matrix(p)


@pytest.mark.parametrize("bad_val", ["ture", 1.5, "yes  "])
def test_entrypoint_raises_on_garbage_smote_enabled(bad_val):
    """Defense in depth for a hand-edited manifest: entrypoint raises, never
    silently coerces an unrecognized value to disabled."""
    from docker.entrypoint import smote_argv_from_run_extras
    with pytest.raises(ValueError, match="smote_enabled"):
        smote_argv_from_run_extras({"smote_enabled": bad_val})


def test_entrypoint_canonical_string_true_still_works():
    from docker.entrypoint import smote_argv_from_run_extras
    argv = smote_argv_from_run_extras(
        {"smote_enabled": "true", "smote_variant": "smote", "smote_target": "balanced"}
    )
    assert argv == ["--smote-enabled", "--smote-variant", "smote", "--smote-target", "balanced"]


def test_entrypoint_bool_false_is_empty():
    from docker.entrypoint import smote_argv_from_run_extras
    assert smote_argv_from_run_extras({"smote_enabled": False}) == []


# ===========================================================================
# 8. Variant drift canary (v8 change 2): every variant in the single-source
#    registry SUPPORTED_SMOTE_VARIANTS must flow through EVERY allowlist layer
#    (resampler validation, matrix_doc frontmatter parse, entrypoint argv,
#    runner provenance). Adding a variant to the registry without wiring it
#    end-to-end fails HERE, mirroring test_defense_token_covers_every_supported_config.
# ===========================================================================

def test_every_supported_variant_wired_end_to_end(tmp_path):
    from flowerfl.smote_resampler import SUPPORTED_SMOTE_VARIANTS, validate_smote_variant
    from docker.entrypoint import smote_argv_from_run_extras
    from run_phase4_flower import smote_provenance_fields
    from praxis_exp.matrix_doc import parse_matrix

    for v in SUPPORTED_SMOTE_VARIANTS:
        # layer 1: resampler validation accepts it
        assert validate_smote_variant(v) == v
        # layer 2: entrypoint argv passes it through
        argv = smote_argv_from_run_extras(
            {"smote_enabled": True, "smote_variant": v, "smote_target": "balanced"}
        )
        assert ["--smote-variant", v] == argv[1:3]
        # layer 3: runner provenance validates + echoes it
        fields = smote_provenance_fields(
            {"smote-enabled": True, "smote-variant": v, "smote-target": "balanced"}
        )
        assert fields["smote_variant"] == v
        # layer 4: matrix_doc frontmatter parse accepts it
        block = f"run_extras:\n  smote_enabled: true\n  smote_variant: {v}\n"
        p = tmp_path / f"doc_{v}.md"
        p.write_text(_doc_text(block))
        assert parse_matrix(p).run_extras["smote_variant"] == v


def test_random_under_is_registered():
    from flowerfl.smote_resampler import SUPPORTED_SMOTE_VARIANTS
    assert "random_under" in SUPPORTED_SMOTE_VARIANTS

# ===========================================================================
# 9. Stage-F fleet reachability (image-update checklist item 7): the three
#    Stage-F knobs (update_match / weight_mode / smote_semantic_target) must be
#    expressible design-doc -> manifest run_extras -> entrypoint argv -> runner
#    CLI, or Batch units silently run the incumbent defaults (the GWU-59
#    near-miss). Mirrors the SMOTE sections above.
# ===========================================================================

def test_matrix_doc_stage_f_keys_parsed_and_normalized(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix
    block = ("run_extras:\n  smote_enabled: true\n  smote_variant: smote\n"
             "  smote_target: 0.5\n  update_match: true\n"
             "  weight_mode: original\n  smote_semantic_target: true\n")
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    extras = parse_matrix(p).run_extras
    assert extras["update_match"] is True
    assert extras["weight_mode"] == "original"
    assert extras["smote_semantic_target"] is True


@pytest.mark.parametrize("key", ["update_match", "smote_semantic_target"])
def test_matrix_doc_bad_stage_f_bool_rejected_at_parse(tmp_path, key):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = f'run_extras:\n  {key}: "ture"\n'
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match=key):
        parse_matrix(p)


def test_matrix_doc_bad_weight_mode_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = 'run_extras:\n  weight_mode: orginal\n'
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="weight_mode"):
        parse_matrix(p)


def test_stage_f_argv_absent_or_incumbent_is_empty():
    """Byte-identity: no run_extras, or Stage-F keys at incumbent values,
    appends nothing."""
    from docker.entrypoint import stage_f_argv_from_run_extras
    assert stage_f_argv_from_run_extras(None) == []
    assert stage_f_argv_from_run_extras({}) == []
    assert stage_f_argv_from_run_extras(
        {"update_match": False, "weight_mode": "resampled",
         "smote_semantic_target": False}) == []


def test_stage_f_argv_maps_all_three_knobs():
    from docker.entrypoint import stage_f_argv_from_run_extras
    argv = stage_f_argv_from_run_extras(
        {"update_match": True, "weight_mode": "original",
         "smote_semantic_target": True})
    assert argv == ["--update-match", "--weight-mode", "original",
                    "--smote-semantic-target"]


@pytest.mark.parametrize("key", ["update_match", "smote_semantic_target"])
def test_stage_f_argv_raises_on_garbage_bool(key):
    """Hand-edited-manifest defense in depth: a typo'd bool RAISES, never
    silently runs the incumbent (same contract as smote_enabled)."""
    from docker.entrypoint import stage_f_argv_from_run_extras
    with pytest.raises(ValueError, match=key):
        stage_f_argv_from_run_extras({key: "ture"})


def test_stage_f_argv_raises_on_bad_weight_mode():
    from docker.entrypoint import stage_f_argv_from_run_extras
    with pytest.raises(ValueError, match="weight_mode"):
        stage_f_argv_from_run_extras({"weight_mode": "orginal"})


def test_runner_argv_appends_stage_f_flags_after_smote_flags():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-030/x"))
    argv = runner_argv(
        u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-030/x"),
        run_extras={"smote_enabled": True, "smote_variant": "smote",
                    "smote_target": 0.5, "update_match": True,
                    "weight_mode": "original", "smote_semantic_target": True},
    )
    assert argv[:len(base)] == base
    assert argv[len(base):] == [
        "--smote-enabled", "--smote-variant", "smote", "--smote-target", "0.5",
        "--update-match", "--weight-mode", "original", "--smote-semantic-target",
    ]


def test_stage_f_full_arm_argv_parses_through_real_runner_parser():
    """The item-7 closer: a full Stage-F arm's run_extras produces argv the
    REAL runner parser accepts, round-tripping to the run-config the client
    reads — no hop can silently drop a knob."""
    import argparse
    from docker.entrypoint import smote_argv_from_run_extras, stage_f_argv_from_run_extras
    from run_phase4_flower import (
        add_smote_cli_args, add_stage_f_cli_args, stage_f_run_config_from_cli,
    )
    extras = {"smote_enabled": True, "smote_variant": "smote", "smote_target": 0.5,
              "update_match": True, "weight_mode": "original",
              "smote_semantic_target": True}
    argv = smote_argv_from_run_extras(extras) + stage_f_argv_from_run_extras(extras)
    p = argparse.ArgumentParser()
    add_smote_cli_args(p)
    add_stage_f_cli_args(p)
    args = p.parse_args(argv)  # unknown/dropped flags would raise here
    assert stage_f_run_config_from_cli(
        args.update_match, args.weight_mode, args.smote_semantic_target
    ) == {"update-match": True, "weight-mode": "original",
          "smote-semantic-target": True}


# ===========================================================================
# 10. semantic flag + unit target is a fatal in-run combo —
#     over-samplers raise in semantic_attack_target_count (f must be in (0,1)),
#     the under-sampler computes a zero benign target. Must be rejected at
#     pre-registration parse AND at entrypoint argv build (hand-edited
#     manifest), never discovered mid-run on the fleet.
# ===========================================================================

def test_matrix_doc_semantic_with_unit_target_rejected_at_parse(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    block = ("run_extras:\n  smote_enabled: true\n  smote_target: 1\n"
             "  smote_semantic_target: true\n")
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    with pytest.raises(MatrixDocError, match="semantic"):
        parse_matrix(p)


@pytest.mark.parametrize("target", ["0.47", "0.5", "balanced"])
def test_matrix_doc_semantic_with_proper_fraction_accepted(tmp_path, target):
    from praxis_exp.matrix_doc import parse_matrix
    block = (f"run_extras:\n  smote_enabled: true\n  smote_target: {target}\n"
             "  smote_semantic_target: true\n")
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    assert parse_matrix(p).run_extras["smote_semantic_target"] is True


def test_matrix_doc_unit_target_without_semantic_still_accepted(tmp_path):
    """Legacy min/max semantics allow target=1; only the SEMANTIC combo is fatal."""
    from praxis_exp.matrix_doc import parse_matrix
    block = "run_extras:\n  smote_enabled: true\n  smote_target: 1\n"
    p = tmp_path / "d.md"; p.write_text(_doc_text(block))
    assert parse_matrix(p).run_extras["smote_target"] == 1.0


def test_stage_f_argv_semantic_with_unit_target_raises():
    from docker.entrypoint import stage_f_argv_from_run_extras
    with pytest.raises(ValueError, match="semantic"):
        stage_f_argv_from_run_extras(
            {"smote_semantic_target": True, "smote_target": 1})


def test_stage_f_argv_semantic_with_proper_fraction_ok():
    from docker.entrypoint import stage_f_argv_from_run_extras
    argv = stage_f_argv_from_run_extras(
        {"smote_semantic_target": True, "smote_target": 0.47})
    assert argv == ["--smote-semantic-target"]

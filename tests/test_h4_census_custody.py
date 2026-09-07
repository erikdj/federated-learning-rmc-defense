"""Custody refusal matrix (exact-equality, fail-closed), the 9x6x10 census
with the pre-stated drop order, the sealed-seed set gate, and redaction
(no raw seed / run_uid / seed-bearing path in any refusal)."""
from __future__ import annotations

import json

import pytest

from scripts import h4_scoring_lib as lib
from scripts.analyze_h4_composition import assert_census, load_units
from scripts.h4_scoring_lib import ScoringError
from tests.h4_factory import pin_fake_bundle_v2
from tests.h4_factory import (
    CONFIG_LABEL_BY_ARM,
    TEST_SEEDS,
    install_seed_manifest,
    install_split_manifest,
    make_unit,
    mutate,
    write_unit,
)

ALL_ARMS = tuple(CONFIG_LABEL_BY_ARM)


@pytest.fixture(autouse=True)
def _pinned_bundle_v2(monkeypatch):
    """Erratum-B pinned state: swap the TBD_BUNDLE_V2 sentinel for the fake
    bundle_v2 sha so custody tests exercise post-calibration equality
    (sentinel refusal is covered by tests/test_h4_scorer_pin_v2.py)."""
    pin_fake_bundle_v2(monkeypatch)


@pytest.fixture
def env(tmp_path, monkeypatch):
    install_split_manifest(tmp_path, monkeypatch)
    install_seed_manifest(tmp_path, monkeypatch)
    return tmp_path


def _load_one(env, unit, name=None):
    return load_units([write_unit(env / "units", unit, name=name)])


# ===========================================================================
# custody: happy paths
# ===========================================================================

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_happy_unit_loads_for_every_arm(env, arm):
    records = _load_one(env, make_unit(arm, "S3", 90001))
    assert len(records) == 1
    assert records[0]["arm"] == arm
    assert records[0]["scenario"] == "S3"
    assert records[0]["seed"] == 90001
    assert records[0]["acc_final5"] == pytest.approx(0.9)


def test_arm_token_derived_from_config_label(env):
    records = _load_one(env, make_unit("krum_tge_fp", "C0", 90002))
    assert records[0]["arm"] == "krum_tge_fp"


# ===========================================================================
# custody: refusal matrix
# ===========================================================================

def test_unknown_config_label_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001), ["config"], "SuperDefense")
    with pytest.raises(ScoringError, match="arm"):
        _load_one(env, unit)


def test_missing_config_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001), ["config"], None, delete=True)
    with pytest.raises(ScoringError):
        _load_one(env, unit)


def test_missing_provenance_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001), ["provenance"], None,
                  delete=True)
    with pytest.raises(ScoringError, match="provenance"):
        _load_one(env, unit)


def test_unknown_scenario_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "scenario_path"], "rmc/scenarios/S9_who.json")
    with pytest.raises(ScoringError, match="scenario"):
        _load_one(env, unit)


def test_missing_scenario_path_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "scenario_path"], None, delete=True)
    with pytest.raises(ScoringError, match="scenario"):
        _load_one(env, unit)


def test_missing_seed_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001), ["seed"], None, delete=True)
    with pytest.raises(ScoringError, match="seed"):
        _load_one(env, unit)


def test_non_integer_seed_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001), ["seed"], "90001")
    with pytest.raises(ScoringError, match="seed"):
        _load_one(env, unit)


def test_legacy_eval_split_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "eval_split"], "legacy")
    with pytest.raises(ScoringError, match="sealed_test"):
        _load_one(env, unit)


def test_missing_eval_split_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "eval_split"], None, delete=True)
    with pytest.raises(ScoringError, match="eval_split"):
        _load_one(env, unit)


def test_wrong_eval_split_manifest_sha_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "eval_split_manifest_sha256"], "ab" * 32)
    with pytest.raises(ScoringError, match="manifest"):
        _load_one(env, unit)


def test_null_eval_split_manifest_sha_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "eval_split_manifest_sha256"], None)
    with pytest.raises(ScoringError, match="manifest"):
        _load_one(env, unit)


def test_missing_committed_split_manifest_refuses(env, monkeypatch):
    unit = make_unit("krum", "S1", 90001)
    monkeypatch.setattr(lib, "SPLIT_MANIFEST_PATH",
                        env / "does_not_exist.json")
    with pytest.raises(ScoringError, match="manifest"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.DETECTOR_ARMS))
def test_wrong_bundle_sha_on_detector_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "serving_bundle_sha256"], "cd" * 32)
    with pytest.raises(ScoringError, match="bundle"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.DETECTOR_ARMS))
def test_null_bundle_sha_on_detector_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "serving_bundle_sha256"], None)
    with pytest.raises(ScoringError, match="bundle"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.ARM_TOKENS - lib.DETECTOR_ARMS))
def test_nonnull_bundle_sha_on_non_detector_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "serving_bundle_sha256"],
                  lib.SERVING_BUNDLE_SHA256)
    with pytest.raises(ScoringError, match="bundle"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_absent_bundle_sha_key_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "serving_bundle_sha256"], None, delete=True)
    with pytest.raises(ScoringError, match="bundle"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.FP_ARMS))
def test_wrong_fp_policy_on_fp_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "fp_registry_policy"], "identity_only")
    with pytest.raises(ScoringError, match="flag_gated"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.FP_ARMS))
def test_null_fp_policy_on_fp_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "fp_registry_policy"], None)
    with pytest.raises(ScoringError, match="flag_gated"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.ARM_TOKENS - lib.FP_ARMS))
def test_fp_policy_on_non_fp_arm_refuses(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "fp_registry_policy"], "flag_gated")
    with pytest.raises(ScoringError, match="fp_registry_policy"):
        _load_one(env, unit)


@pytest.mark.parametrize("arm", sorted(lib.ARM_TOKENS - lib.FP_ARMS))
def test_absent_fp_policy_key_on_non_fp_arm_is_accepted(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "fp_registry_policy"], None, delete=True)
    assert _load_one(env, unit)[0]["arm"] == arm


def test_fp_arm_without_registry_custody_block_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry"], None, delete=True)
    with pytest.raises(ScoringError, match="fingerprint_registry"):
        _load_one(env, unit)


def test_fp_arm_registry_policy_disagreement_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry", "registry_policy"], "identity_only")
    with pytest.raises(ScoringError, match="registry_policy"):
        _load_one(env, unit)


def test_fp_arm_missing_run_uid_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry", "run_uid"], None, delete=True)
    with pytest.raises(ScoringError, match="run_uid"):
        _load_one(env, unit)


def test_fp_arm_drifted_tau_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry", "tau"], 26.4669)
    with pytest.raises(ScoringError, match="tau"):
        _load_one(env, unit)


def test_fp_arm_wrong_calibration_sha_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry", "calibration_artifact_sha256"],
                  "ef" * 32)
    with pytest.raises(ScoringError, match="calibration"):
        _load_one(env, unit)


def test_fp_arm_wrong_cohort_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["provenance", "fp_cohort"], "adjudicating")
    with pytest.raises(ScoringError, match="cohort"):
        _load_one(env, unit)


# --- universal run identity (director ruling 2026-08-17) -------------------

@pytest.mark.parametrize("arm", ("fedavg", "krum", "trustscore",
                                 "h2p_krum", "h2p_ts", "h2p_fp_krum"))
def test_missing_provenance_run_uid_refuses_on_every_arm_class(env, arm):
    unit = mutate(make_unit(arm, "S1", 90001),
                  ["provenance", "run_uid"], None, delete=True)
    with pytest.raises(ScoringError, match="run_uid"):
        _load_one(env, unit)


def test_empty_provenance_run_uid_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "run_uid"], "")
    with pytest.raises(ScoringError, match="run_uid"):
        _load_one(env, unit)


def test_non_string_provenance_run_uid_refuses(env):
    unit = mutate(make_unit("krum", "S1", 90001),
                  ["provenance", "run_uid"], 12345)
    with pytest.raises(ScoringError, match="run_uid"):
        _load_one(env, unit)


def test_duplicate_run_uid_on_non_fp_arms_refuses_redacted(env):
    a = make_unit("krum", "S1", 90001, run_uid="uid-SHARED-NONFP")
    b = make_unit("fedavg", "S1", 90001, run_uid="uid-SHARED-NONFP")
    paths = [write_unit(env / "units", a), write_unit(env / "units", b)]
    with pytest.raises(ScoringError) as excinfo:
        load_units(paths)
    msg = str(excinfo.value)
    assert "uid-SHARED-NONFP" not in msg   # redacted
    assert "run_uid" in msg and "sha256:" in msg


def test_fp_provenance_vs_registry_run_uid_mismatch_refuses(env):
    unit = mutate(make_unit("h2p_fp_krum", "S1", 90001),
                  ["fingerprint_registry", "run_uid"], "uid-OTHER-COPY")
    with pytest.raises(ScoringError) as excinfo:
        _load_one(env, unit)
    msg = str(excinfo.value)
    assert "uid-OTHER-COPY" not in msg     # redacted
    assert "EQUAL" in msg


def test_fp_matching_run_uid_copies_load(env):
    unit = make_unit("h2p_fp_krum", "S1", 90001, run_uid="uid-match")
    records = _load_one(env, unit)
    assert records[0]["run_uid"] == "uid-match"


def test_non_fp_arm_run_uid_is_recorded(env):
    records = _load_one(env, make_unit("fedavg", "S1", 90001,
                                       run_uid="uid-fedavg-cell"))
    assert records[0]["run_uid"] == "uid-fedavg-cell"


def test_duplicate_run_uid_across_units_refuses(env):
    a = make_unit("h2p_fp_krum", "S1", 90001, run_uid="uid-SHARED")
    b = make_unit("h2p_fp_krum", "S2", 90001, run_uid="uid-SHARED")
    paths = [write_unit(env / "units", a), write_unit(env / "units", b)]
    with pytest.raises(ScoringError) as excinfo:
        load_units(paths)
    assert "uid-SHARED" not in str(excinfo.value)   # redacted
    assert "run_uid" in str(excinfo.value)


def test_duplicate_cell_refuses_without_printing_seed(env):
    a = make_unit("krum", "S1", 90007)
    paths = [write_unit(env / "units", a, name="a.json"),
             write_unit(env / "units", a, name="b.json")]
    with pytest.raises(ScoringError) as excinfo:
        load_units(paths)
    msg = str(excinfo.value)
    assert "cell" in msg
    assert "90007" not in msg


def test_invalid_json_refuses(env):
    path = env / "units" / "broken.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    with pytest.raises(ScoringError, match="JSON"):
        load_units([path])


def test_missing_file_refuses(env):
    with pytest.raises(ScoringError, match="not found"):
        load_units([env / "units" / "absent.json"])


def test_no_units_refuses(env):
    with pytest.raises(ScoringError):
        load_units([])


def test_short_trajectory_refuses_at_load(env):
    unit = make_unit("fedavg", "S1", 90001)
    unit["trajectory"] = unit["trajectory"][:4]
    with pytest.raises(ScoringError, match="fewer than 5"):
        _load_one(env, unit)


def test_refusals_never_echo_a_seed_bearing_path(env):
    """The default fixture filename embeds the seed (mirroring launch
    tooling); a custody refusal must not leak it through the path echo."""
    unit = mutate(make_unit("krum", "S1", 90009),
                  ["provenance", "eval_split"], "legacy")
    path = write_unit(env / "units", unit)   # name embeds seed90009
    assert "90009" in path.name
    with pytest.raises(ScoringError) as excinfo:
        load_units([path])
    msg = str(excinfo.value)
    assert "90009" not in msg
    assert "sha256:" in msg   # the unit is still locatable via the redaction


# ===========================================================================
# census + drop order + seed-set gate
# ===========================================================================

def _records(arms, scenarios=lib.SCENARIOS, seeds=TEST_SEEDS):
    out = []
    for arm in arms:
        for scenario in scenarios:
            for seed in seeds:
                out.append({"arm": arm, "scenario": scenario, "seed": seed,
                            "ref": f"unit[arm={arm},scenario={scenario}]"})
    return out


def test_full_nine_arm_census_passes():
    census = assert_census(_records(ALL_ARMS), list(TEST_SEEDS))
    assert census["arms_dropped"] == []
    assert len(census["arms_present"]) == 9
    assert census["n_units"] == 540


def test_prefix_drop_of_fedavg_passes_with_disclosure():
    arms = [a for a in ALL_ARMS if a != "fedavg"]
    census = assert_census(_records(arms), list(TEST_SEEDS))
    assert census["arms_dropped"] == ["fedavg"]
    assert "fedavg" in census["drop_disclosure"]


def test_prefix_drop_of_three_arms_passes():
    dropped = {"fedavg", "krum_tge_fp", "trustscore"}
    arms = [a for a in ALL_ARMS if a not in dropped]
    census = assert_census(_records(arms), list(TEST_SEEDS))
    assert census["arms_dropped"] == ["fedavg", "krum_tge_fp", "trustscore"]


def test_maximum_drop_leaves_the_adjudicating_pair():
    arms = ["h2p_fp_krum", "krum"]
    census = assert_census(_records(arms), list(TEST_SEEDS))
    assert len(census["arms_dropped"]) == 7
    assert census["n_units"] == 120


def test_non_prefix_drop_refuses():
    # dropping trustscore (3rd in the order) without fedavg/krum_tge_fp
    arms = [a for a in ALL_ARMS if a != "trustscore"]
    with pytest.raises(ScoringError, match="drop order"):
        assert_census(_records(arms), list(TEST_SEEDS))


def test_missing_adjudicating_arm_refuses():
    arms = [a for a in ALL_ARMS if a != "krum"]
    with pytest.raises(ScoringError, match="never"):
        assert_census(_records(arms), list(TEST_SEEDS))


def test_partial_arm_is_not_a_drop_and_refuses():
    records = _records(ALL_ARMS)
    # remove ONE fedavg cell: fedavg is now partial, not dropped
    victim = next(i for i, r in enumerate(records) if r["arm"] == "fedavg")
    del records[victim]
    with pytest.raises(ScoringError, match="all-or-nothing|missing"):
        assert_census(records, list(TEST_SEEDS))


def test_missing_c0_cells_for_surviving_arm_refuses():
    records = [r for r in _records(ALL_ARMS)
               if not (r["arm"] == "h2p_krum" and r["scenario"] == "C0")]
    with pytest.raises(ScoringError, match="C0"):
        assert_census(records, list(TEST_SEEDS))


def test_wrong_seed_set_refuses_count_only():
    bad_seeds = list(TEST_SEEDS[:-1]) + [77777]
    records = _records(ALL_ARMS, seeds=bad_seeds)
    with pytest.raises(ScoringError) as excinfo:
        assert_census(records, list(TEST_SEEDS))
    msg = str(excinfo.value)
    assert "77777" not in msg and "90001" not in msg
    assert "seed" in msg


def test_extra_scenario_refuses():
    records = _records(ALL_ARMS) + [
        {"arm": "krum", "scenario": "S5", "seed": TEST_SEEDS[0], "ref": "x"}]
    with pytest.raises(ScoringError):
        assert_census(records, list(TEST_SEEDS))


def test_manifest_wrong_count_refuses(tmp_path, monkeypatch):
    install_seed_manifest(tmp_path, monkeypatch, seeds=TEST_SEEDS[:9])
    with pytest.raises(ScoringError, match="10"):
        lib_seeds = lib.manifest_seed_list()
        assert_census(_records(ALL_ARMS, seeds=TEST_SEEDS[:9]), lib_seeds)


def test_manifest_missing_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "SEED_MANIFEST_PATH", tmp_path / "nope.json")
    with pytest.raises(ScoringError, match="manifest"):
        lib.manifest_seed_list()


def test_manifest_bad_key_refuses(tmp_path, monkeypatch):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"dev_seeds": [1, 2, 3]}))
    monkeypatch.setattr(lib, "SEED_MANIFEST_PATH", path)
    with pytest.raises(ScoringError, match="confirmatory_seeds"):
        lib.manifest_seed_list()


def test_manifest_non_integer_values_refuse(tmp_path, monkeypatch):
    install_seed_manifest(tmp_path, monkeypatch, seeds=["a", "b"])
    with pytest.raises(ScoringError):
        lib.manifest_seed_list()

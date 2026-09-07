"""Erratum-B build item 2: (scenario, arm_class) cut consumption — bundle v2.

The public workflow is described in `docs/reproduction/experiments.md`.
Historical protocol references: § B2
(RULED, methodology v1.53): the online plugin consumes EITHER the v1
per-scenario `cuts.json` (v1 compatibility) OR a
`cuts_v2.json` keyed {scenario: {arm_class: cut}}. Selection is EXPLICIT via
the pinned cuts-version knob — never auto-detected. C0 has its OWN cut in v2
(the E3-bis alias is retired for v2; it survives only on the v1 path).
Unknown (scenario, arm_class) = refusal. Strict `score > cut` unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowerfl.h2prime_online import (  # noqa: E402
    ARM_CLASSES,
    ARM_CLASS_BY_STRATEGY,
    H2PrimeBundleError,
    OnlineH2PrimeDetectorPlugin,
    load_serving_bundle,
    scenario_token_from_name,
)
from h4_bundle_fixture import (  # noqa: E402
    DEFAULT_CUTS_V2,
    SCENARIO_TOKENS_V2,
    make_synthetic_bundle,
    make_synthetic_bundle_v2,
)


@pytest.fixture
def v2_dir(tmp_path):
    return make_synthetic_bundle_v2(tmp_path / "bundle")


# ---------------------------------------------------------------------------
# explicit version selection — never auto-detect
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unknown_cuts_version_refuses(v2_dir):
    with pytest.raises(H2PrimeBundleError, match="cuts.version|cuts_version"):
        load_serving_bundle(v2_dir, cuts_version="v3")


@pytest.mark.unit
def test_default_is_v1_byte_identical(tmp_path):
    """Absent knob = v1: same cuts, same manifest sha as the pre-erratum call."""
    d = make_synthetic_bundle(tmp_path / "b")
    default = load_serving_bundle(d)
    explicit = load_serving_bundle(d, cuts_version="v1")
    assert default.cuts == explicit.cuts
    assert default.bundle_sha256 == explicit.bundle_sha256
    assert default.cuts_version == "v1"


@pytest.mark.unit
def test_v1_load_ignores_v2_files_even_when_present(v2_dir):
    """Version comes from the pinned knob, not from what exists on disk."""
    bundle = load_serving_bundle(v2_dir, cuts_version="v1")
    assert bundle.cuts_version == "v1"
    assert set(bundle.cuts) == {"S0", "S1", "S2", "S3", "S4"}


# ---------------------------------------------------------------------------
# v2 loading and verification
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_v2_loads_and_pins_manifest_v2_bytes(v2_dir):
    bundle = load_serving_bundle(v2_dir, cuts_version="v2")
    assert bundle.cuts_version == "v2"
    assert set(bundle.cuts) == set(SCENARIO_TOKENS_V2)
    for tok in SCENARIO_TOKENS_V2:
        assert set(bundle.cuts[tok]) == set(ARM_CLASSES)
    expected = hashlib.sha256(
        (v2_dir / "manifest_v2.json").read_bytes()
    ).hexdigest()
    assert bundle.bundle_sha256 == expected


@pytest.mark.unit
def test_v2_missing_manifest_refuses(tmp_path):
    d = make_synthetic_bundle(tmp_path / "b")  # v1 only, no v2 files
    with pytest.raises(H2PrimeBundleError, match="manifest_v2"):
        load_serving_bundle(d, cuts_version="v2")


@pytest.mark.unit
def test_v2_cuts_file_sha_mismatch_refuses(v2_dir):
    path = v2_dir / "cuts_v2.json"
    doc = json.loads(path.read_text())
    doc["S3"]["krum_family"] = 0.999
    path.write_text(json.dumps(doc, indent=1))
    with pytest.raises(H2PrimeBundleError, match="sha256"):
        load_serving_bundle(v2_dir, cuts_version="v2")


@pytest.mark.unit
def test_v2_missing_scenario_refuses(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    del cuts["C0"]
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    with pytest.raises(H2PrimeBundleError, match="cuts_v2"):
        load_serving_bundle(d, cuts_version="v2")


@pytest.mark.unit
def test_v2_missing_arm_class_refuses(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    del cuts["S2"]["ts_family"]
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    with pytest.raises(H2PrimeBundleError, match="cuts_v2"):
        load_serving_bundle(d, cuts_version="v2")


@pytest.mark.unit
def test_v2_non_finite_cut_refuses(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    cuts["S1"]["fedavg_family"] = "nan"
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    with pytest.raises(H2PrimeBundleError, match="finite"):
        load_serving_bundle(d, cuts_version="v2")


@pytest.mark.unit
def test_v2_extra_arm_class_refuses(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    cuts["S0"]["global"] = 0.5
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    with pytest.raises(H2PrimeBundleError, match="cuts_v2"):
        load_serving_bundle(d, cuts_version="v2")


# ---------------------------------------------------------------------------
# scenario token resolution per version
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_v2_c0_resolves_to_its_own_token_not_the_alias():
    assert scenario_token_from_name("C0_clean_no_attack", cuts_version="v2") == "C0"


@pytest.mark.unit
def test_v1_c0_alias_still_resolves_to_s0():
    assert scenario_token_from_name("C0_clean_no_attack") == "S0"
    assert scenario_token_from_name("C0_clean_no_attack", cuts_version="v1") == "S0"


@pytest.mark.unit
@pytest.mark.parametrize("name,token", [
    ("S3_identity_reset_only", "S3"),
    ("s0_clean_baseline", "S0"),
    ("c0_clean_no_attack", "C0"),
])
def test_v2_token_resolution(name, token):
    assert scenario_token_from_name(name, cuts_version="v2") == token


@pytest.mark.unit
def test_v2_unknown_scenario_refuses():
    with pytest.raises(H2PrimeBundleError):
        scenario_token_from_name("X9_mystery", cuts_version="v2")


# ---------------------------------------------------------------------------
# plugin cut selection under v2
# ---------------------------------------------------------------------------

def _v2_plugin(v2_dir, scenario_token="S3", arm_class="krum_family", **kw):
    bundle = load_serving_bundle(v2_dir, cuts_version="v2")
    return OnlineH2PrimeDetectorPlugin(
        bundle=bundle, scenario_token=scenario_token, arm_class=arm_class, **kw
    )


@pytest.mark.unit
def test_v2_plugin_selects_the_pair_cut(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    cuts["S3"]["krum_family"] = 0.625
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    plugin = _v2_plugin(d, "S3", "krum_family")
    assert plugin.cut == 0.625
    assert plugin.arm_class == "krum_family"
    assert plugin.cuts_version == "v2"


@pytest.mark.unit
def test_v2_plugin_c0_uses_c0_cut(tmp_path):
    cuts = {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    cuts["C0"]["fedavg_family"] = 0.311
    cuts["S0"]["fedavg_family"] = 0.999
    d = make_synthetic_bundle_v2(tmp_path / "b", cuts_v2=cuts)
    plugin = _v2_plugin(d, "C0", "fedavg_family")
    assert plugin.cut == 0.311  # NOT the S0 value — alias retired for v2


@pytest.mark.unit
def test_v2_plugin_without_arm_class_refuses(v2_dir):
    bundle = load_serving_bundle(v2_dir, cuts_version="v2")
    with pytest.raises(H2PrimeBundleError, match="arm.class|arm_class"):
        OnlineH2PrimeDetectorPlugin(bundle=bundle, scenario_token="S3")


@pytest.mark.unit
def test_v2_plugin_unknown_arm_class_refuses(v2_dir):
    with pytest.raises(H2PrimeBundleError, match="arm.class|arm_class"):
        _v2_plugin(v2_dir, "S3", "median_family")


@pytest.mark.unit
def test_v2_plugin_unknown_scenario_refuses(v2_dir):
    with pytest.raises(H2PrimeBundleError, match="no serving cut"):
        _v2_plugin(v2_dir, "S9", "krum_family")


@pytest.mark.unit
def test_v1_plugin_behavior_unchanged(tmp_path):
    """The v1 path is byte-identical: same construction, same cut."""
    d = make_synthetic_bundle(tmp_path / "b", cuts={"S0": 0.1, "S1": 0.2,
                                                    "S2": 0.3, "S3": 0.4,
                                                    "S4": 0.5})
    bundle = load_serving_bundle(d)
    plugin = OnlineH2PrimeDetectorPlugin(bundle=bundle, scenario_token="S3")
    assert plugin.cut == 0.4
    assert plugin.cuts_version == "v1"


# ---------------------------------------------------------------------------
# arm-class table (erratum B B1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_arm_class_table_covers_h4_and_observer_base_strategies():
    assert ARM_CLASS_BY_STRATEGY == {
        # H4 composition arms (cut grain per erratum B):
        "ScenarioH2PFPKrum": "krum_family",
        "ScenarioH2PKrum": "krum_family",
        "ScenarioH2PFPTS": "ts_family",
        "ScenarioH2PTS": "ts_family",
        "ScenarioH2PFP": "fedavg_family",
        # observer-attached BASE arms (calibration units):
        "ScenarioKrum": "krum_family",
        "ScenarioTrustScore": "ts_family",
        "ScenarioNone": "fedavg_family",
    }
    assert set(ARM_CLASS_BY_STRATEGY.values()) == set(ARM_CLASSES)

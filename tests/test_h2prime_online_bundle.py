"""H4 serving-bundle loading — hash verification, refusals, flag boundary.

Design authority: spec 2026-08-16 § 7/§ 7.1 (frozen serving instrument,
sha-pinned bundle, no runtime discretion), erratum-A E3 (cut mechanics: flag
= STRICT `score > cut`) and E3-bis (C0 -> S0 cut via an EXPLICIT declared
alias, never a silent fallback), and `docs/reproduction/experiments.md`
(bundle file set + `bundle_sha256` = sha256 of the manifest bytes).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.h2prime_online import (  # noqa: E402
    H2PrimeBundleError,
    SERVING_CUT_ALIASES,
    flag_scores,
    load_serving_bundle,
    scenario_token_from_name,
)
from h4_bundle_fixture import make_synthetic_bundle  # noqa: E402


@pytest.fixture()
def bundle_dir(tmp_path):
    return make_synthetic_bundle(tmp_path / "h4_serving")


# ===========================================================================
# loading + verification
# ===========================================================================

@pytest.mark.unit
def test_loads_a_valid_bundle(bundle_dir):
    bundle = load_serving_bundle(bundle_dir)
    assert set(bundle.cuts) == {"S0", "S1", "S2", "S3", "S4"}
    assert len(bundle.features) == 9
    assert callable(bundle.model.predict_proba)


@pytest.mark.unit
def test_bundle_sha256_is_sha256_of_manifest_bytes(bundle_dir):
    bundle = load_serving_bundle(bundle_dir)
    expected = hashlib.sha256(
        (bundle_dir / "manifest.json").read_bytes()
    ).hexdigest()
    assert bundle.bundle_sha256 == expected


@pytest.mark.unit
def test_missing_manifest_refuses(tmp_path):
    with pytest.raises(H2PrimeBundleError, match="manifest missing"):
        load_serving_bundle(tmp_path / "nowhere")


@pytest.mark.unit
@pytest.mark.parametrize("name", ["model.joblib", "cuts.json", "features.json"])
def test_per_file_sha_mismatch_refuses(bundle_dir, name):
    path = bundle_dir / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(H2PrimeBundleError, match="sha256 verification"):
        load_serving_bundle(bundle_dir)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["model.joblib", "cuts.json", "features.json"])
def test_missing_payload_file_refuses(bundle_dir, name):
    (bundle_dir / name).unlink()
    with pytest.raises(H2PrimeBundleError, match="missing"):
        load_serving_bundle(bundle_dir)


@pytest.mark.unit
def test_manifest_without_files_map_refuses(bundle_dir):
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    del manifest["files"]
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(H2PrimeBundleError, match="'files'"):
        load_serving_bundle(bundle_dir)


@pytest.mark.unit
def test_missing_scenario_cut_refuses(tmp_path):
    """A four-scenario cut table is a corrupt instrument, not a fallback."""
    d = make_synthetic_bundle(
        tmp_path / "b", cuts={"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5}
    )
    with pytest.raises(H2PrimeBundleError, match="cuts.json"):
        load_serving_bundle(d)


@pytest.mark.unit
def test_extra_global_cut_key_refuses(tmp_path):
    """A 'global' cut is the § 2.2-rejected degeneracy — never accepted."""
    cuts = {"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5, "S4": 0.5,
            "global": 0.5}
    d = make_synthetic_bundle(tmp_path / "b", cuts=cuts)
    with pytest.raises(H2PrimeBundleError, match="cuts.json"):
        load_serving_bundle(d)


@pytest.mark.unit
def test_non_finite_cut_refuses(tmp_path):
    d = make_synthetic_bundle(
        tmp_path / "b",
        cuts={"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5, "S4": "nan"},
    )
    with pytest.raises(H2PrimeBundleError, match="finite"):
        load_serving_bundle(d)


@pytest.mark.unit
def test_frozen_feature_order_matches_the_committed_bundle():
    """The module constant is a transcription of data/h4_serving/features.json
    (the model's training column order) — they must never drift."""
    from flowerfl.h2prime_online import FROZEN_FEATURE_ORDER

    committed = json.loads(
        (PROJECT_ROOT / "data" / "h4_serving" / "features.json").read_text()
    )
    assert list(FROZEN_FEATURE_ORDER) == committed


@pytest.mark.unit
def test_exact_frozen_feature_order_passes(bundle_dir):
    """Positive direction of the P2 pin: the exact frozen list loads."""
    from flowerfl.h2prime_online import FROZEN_FEATURE_ORDER

    bundle = load_serving_bundle(bundle_dir)
    assert bundle.features == FROZEN_FEATURE_ORDER


@pytest.mark.unit
def test_reordered_features_refuse(tmp_path):
    """A REORDERED feature list (same names,
    permuted) must refuse — it would silently permute the model's columns."""
    from h4_bundle_fixture import FROZEN_FEATURES

    reordered = list(FROZEN_FEATURES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    d = make_synthetic_bundle(tmp_path / "b", features=reordered)
    with pytest.raises(H2PrimeBundleError, match="frozen feature order"):
        load_serving_bundle(d)


@pytest.mark.unit
def test_renamed_feature_refuses(tmp_path):
    from h4_bundle_fixture import FROZEN_FEATURES

    renamed = list(FROZEN_FEATURES)
    renamed[-1] = "cosine_variance"  # plausible-looking rename
    d = make_synthetic_bundle(tmp_path / "b", features=renamed)
    with pytest.raises(H2PrimeBundleError, match="frozen feature order"):
        load_serving_bundle(d)


@pytest.mark.unit
def test_feature_count_model_mismatch_refuses(tmp_path):
    from h4_bundle_fixture import FROZEN_FEATURES, fit_tiny_gbdt

    d = make_synthetic_bundle(
        tmp_path / "b",
        features=list(FROZEN_FEATURES),
        model=fit_tiny_gbdt(n_features=5),
    )
    with pytest.raises(H2PrimeBundleError, match="features"):
        load_serving_bundle(d)


# ===========================================================================
# flag boundary — erratum-A E3 (STRICT score > cut)
# ===========================================================================

@pytest.mark.unit
def test_score_equal_to_cut_is_not_flagged():
    cut = 0.4375
    flags = flag_scores([cut, np.nextafter(cut, 1.0), cut - 1e-9], cut)
    assert flags.tolist() == [False, True, False]


@pytest.mark.unit
def test_flag_semantics_match_frozen_offline_scorer_on_random_scores():
    """Bit-for-bit against the golden-gated builder's `flagged()`
    (higher_is_trust=False): the frozen inequality is strict `>`."""
    import importlib.util

    builder_path = (
        PROJECT_ROOT / "reproduction" / "protocol" / "h2prime"
        / "revalidate_v115.py"
    )
    spec = importlib.util.spec_from_file_location("rv115_flagtest", builder_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    rng = np.random.default_rng(7)
    scores = rng.random(500)
    cut = float(np.quantile(scores, 1 - 0.10))
    scores[:5] = cut  # force exact-boundary ties
    ours = flag_scores(scores, cut)
    theirs = frozen.flagged(scores, cut, higher_is_trust=False)
    assert np.array_equal(ours, theirs)
    assert not ours[:5].any()  # boundary ties are NOT flagged


# ===========================================================================
# scenario token resolution — E3-bis alias, never a silent fallback
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("name,token", [
    ("S0_clean_baseline", "S0"),
    ("s3_identity_reset_only", "S3"),
    ("S4_full_mix", "S4"),
])
def test_scenario_token_from_name(name, token):
    assert scenario_token_from_name(name) == token


@pytest.mark.unit
def test_c0_resolves_to_the_s0_cut_via_the_declared_alias(bundle_dir):
    assert SERVING_CUT_ALIASES == {"C0_clean_no_attack": "S0"}
    token = scenario_token_from_name("C0_clean_no_attack")
    assert token == "S0"
    bundle = load_serving_bundle(bundle_dir)
    assert bundle.cuts[token] == bundle.cuts["S0"]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [
    "control_honest", "C1_something", "baseline", "", "S_full_mix",
])
def test_undeclared_scenario_refuses(bad):
    with pytest.raises(H2PrimeBundleError, match="alias"):
        scenario_token_from_name(bad)

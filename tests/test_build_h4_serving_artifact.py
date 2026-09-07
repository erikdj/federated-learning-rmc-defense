"""Unit gates for the § 7.1 H4 serving-bundle builder
(scripts/build_h4_serving_artifact.py).

Everything here must hold BEFORE the builder is executed against the exposed
H2′ confirmatory corpus: the cut construction is byte-for-byte the
adjudicator's (same quantile and tie semantics, strict-greater flagging), the
cut structure is per-scenario with no global path, the frozen hyperparameters
and 9-feature order pass through from the adjudication artifact without drift,
the bundle manifest is deterministic and self-hashing, and every refusal path
(missing cells, partial downloads, out-of-band realized FPR) fails loudly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_h4_serving_artifact.py"
ADJUDICATION = REPO / "reproduction" / "evidence" / "h2prime-confirmatory.json"


def _load():
    spec = importlib.util.spec_from_file_location("build_h4_serving_uut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_h4_serving_uut"] = module
    spec.loader.exec_module(module)
    return module


M = _load()

# The frozen references the builder must agree with, imported independently so
# a drift inside the builder cannot vouch for itself.
sys.path.insert(0, str(REPO / "scripts"))
from h2prime_common import FEATS, GBDT_PARAMS, Refusal, R  # noqa: E402
from h2prime_scoring import _cut_at  # noqa: E402

SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]


# --------------------------------------------------------------------------
# helpers — synthetic corpora
# --------------------------------------------------------------------------
def _row(scen: str, seed: int, cid: str, rnd: int, malicious: bool,
         feat_val: float) -> dict:
    r = {f: float(feat_val) for f in FEATS}
    r.update({
        "_scen": scen, "_seed": seed, "logical_cid": cid,
        "scenario_round": rnd, "malicious_gt": bool(malicious),
        "attack_type": "alie" if malicious else "",
    })
    return r


def _synthetic_rows(rng: np.random.Generator, n_honest: int = 40,
                    n_mal: int = 10) -> list[dict]:
    rows = []
    for scen in SCENARIOS:
        for i in range(n_honest):
            rows.append(_row(scen, 42, f"client_{i}", i, False,
                             float(rng.normal(0.0, 1.0))))
        for i in range(n_mal):
            rows.append(_row(scen, 42, f"client_m{i}", i, True,
                             float(rng.normal(3.0, 1.0))))
    return rows


# --------------------------------------------------------------------------
# cut semantics — the adjudicator's construction, not a lookalike
# --------------------------------------------------------------------------
def test_serving_cut_equals_adjudicator_cut_on_random_scores():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=997)
    cut = M.serving_cut(scores)
    assert cut == R.cut_from_calibration(scores, higher_is_trust=False)
    assert cut == _cut_at(scores, 0.10, False)
    assert cut == float(np.quantile(np.asarray(scores, dtype=float), 0.9))


def test_serving_cut_tie_semantics_match_numpy_quantile():
    # Heavy ties around the quantile point: the frozen construction is
    # np.quantile's default linear interpolation, nothing bespoke.
    scores = np.array([0.5] * 18 + [0.9] * 2)
    assert M.serving_cut(scores) == float(np.quantile(scores, 0.9))
    scores2 = np.array([0.25] * 10)
    assert M.serving_cut(scores2) == 0.25


def test_realized_fpr_is_strict_greater_flagging():
    # Exactly-at-cut is NOT flagged (R.flagged is strict `>`); ties at the
    # cut therefore lower the realized FPR, never raise it.
    scores = np.array([0.1] * 90 + [0.9] * 10)
    cut = M.serving_cut(scores)
    fpr = M.realized_fpr(scores, cut)
    assert fpr == float((scores > cut).mean())
    all_equal = np.full(50, 0.3)
    assert M.realized_fpr(all_equal, M.serving_cut(all_equal)) == 0.0


# --------------------------------------------------------------------------
# per-scenario cuts — structure, isolation, refusals; NO global path
# --------------------------------------------------------------------------
def test_per_scenario_cuts_structure_and_isolation():
    rng = np.random.default_rng(1)
    rows = _synthetic_rows(rng)
    # Scenario-shifted honest scores so each cut is provably computed from
    # that scenario's honest rows alone.
    shift = {s: i * 10.0 for i, s in enumerate(SCENARIOS)}
    scores = np.array([
        shift[r["_scen"]] + (5.0 if r["malicious_gt"] else float(rng.uniform(0, 1)))
        for r in rows
    ])
    cuts = M.per_scenario_cuts(rows, scores)
    assert sorted(cuts) == SCENARIOS          # exactly S0..S4, nothing else
    for scen in SCENARIOS:
        honest = np.array([s for r, s in zip(rows, scores)
                           if r["_scen"] == scen and not r["malicious_gt"]])
        assert cuts[scen]["cut"] == float(np.quantile(honest, 0.9))
        assert cuts[scen]["n_honest"] == len(honest)
        assert 0.08 <= cuts[scen]["realized_fpr"] <= 0.12
    # No global-cut code path exists in the module at all (§ 7.1: the global
    # cut is the rejected degeneracy, not a fallback).
    assert not any("global" in name.lower() for name in dir(M)
                   if callable(getattr(M, name)) and not name.startswith("__"))


def test_per_scenario_cuts_refuses_out_of_band_fpr():
    rng = np.random.default_rng(2)
    rows = _synthetic_rows(rng)
    # S3's honest scores all identical -> strict-greater flags nothing ->
    # realized FPR 0.0, outside the closed [0.08, 0.12] band.
    scores = np.array([
        0.5 if (r["_scen"] == "S3" and not r["malicious_gt"])
        else float(rng.uniform(0, 1)) for r in rows
    ])
    with pytest.raises(Refusal, match=r"S3.*0\.0000.*\[0\.08, 0\.12\]"):
        M.per_scenario_cuts(rows, scores)


def test_per_scenario_cuts_refuses_missing_scenario():
    rng = np.random.default_rng(3)
    rows = [r for r in _synthetic_rows(rng) if r["_scen"] != "S4"]
    scores = rng.uniform(size=len(rows))
    with pytest.raises(Refusal, match="S4"):
        M.per_scenario_cuts(rows, scores)


# --------------------------------------------------------------------------
# frozen construction pass-through — no drift
# --------------------------------------------------------------------------
def test_load_frozen_construction_matches_committed_constants():
    fc = M.load_frozen_construction(ADJUDICATION)
    assert fc.gbdt_params == GBDT_PARAMS
    assert list(fc.features) == list(FEATS)
    assert len(fc.features) == 9


def test_load_frozen_construction_refuses_param_drift(tmp_path):
    doc = json.loads(ADJUDICATION.read_text())
    doc["_meta"]["gbdt_params"]["learning_rate"] = 0.2
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(Refusal, match="gbdt_params"):
        M.load_frozen_construction(p)


def test_load_frozen_construction_refuses_feature_drift(tmp_path):
    doc = json.loads(ADJUDICATION.read_text())
    doc["_meta"]["features_frozen_order"] = list(
        reversed(doc["_meta"]["features_frozen_order"]))
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(Refusal, match="features_frozen_order"):
        M.load_frozen_construction(p)


# --------------------------------------------------------------------------
# model fit — ONE classifier, frozen hparams, frozen column order
# --------------------------------------------------------------------------
def test_fit_serving_model_passes_frozen_params_through():
    rng = np.random.default_rng(4)
    rows = _synthetic_rows(rng, n_honest=12, n_mal=4)
    fc = M.load_frozen_construction(ADJUDICATION)
    clf, meta = M.fit_serving_model(rows, fc)
    got = clf.get_params()
    for k, v in GBDT_PARAMS.items():
        assert got[k] == v, f"hyperparameter drift on {k!r}: {got[k]!r} != {v!r}"
    assert meta["n_rows"] == len(rows)
    assert meta["n_positive"] == sum(1 for r in rows if r["malicious_gt"])
    assert clf.predict_proba(np.zeros((1, 9))).shape == (1, 2)


def test_fit_serving_model_refuses_single_class():
    rng = np.random.default_rng(5)
    rows = [r for r in _synthetic_rows(rng) if not r["malicious_gt"]]
    fc = M.load_frozen_construction(ADJUDICATION)
    with pytest.raises(Refusal, match="single class"):
        M.fit_serving_model(rows, fc)


# --------------------------------------------------------------------------
# bundle emission — contract files, deterministic manifest, bundle sha
# --------------------------------------------------------------------------
def _tiny_bundle_inputs():
    rng = np.random.default_rng(6)
    rows = _synthetic_rows(rng, n_honest=12, n_mal=4)
    fc = M.load_frozen_construction(ADJUDICATION)
    clf, fit_meta = M.fit_serving_model(rows, fc)
    scenario_cuts = {
        s: {"cut": 0.5 + i / 10.0, "realized_fpr": 0.1, "n_honest": 120}
        for i, s in enumerate(SCENARIOS)
    }
    corpus_meta = {
        "map_path": "assembly_map.json", "map_sha256": "ab" * 32,
        "n_cells": 50, "cells_by_source": {"EXP-051": 48, "EXP-053": 2},
        "n_rows": len(rows), "seeds_ascending": [1, 2, 3],
        "row_census": M.corpus_census(rows),
    }
    return fc, clf, fit_meta, scenario_cuts, corpus_meta


def test_write_bundle_contract_files_and_hashes(tmp_path):
    fc, clf, fit_meta, cuts, corpus_meta = _tiny_bundle_inputs()
    out = M.write_bundle(tmp_path / "bundle", clf=clf, scenario_cuts=cuts,
                         construction=fc, corpus_meta=corpus_meta,
                         fit_meta=fit_meta, built_at="2026-08-17T00:00:00Z",
                         source_commit="deadbeef")
    d = tmp_path / "bundle"
    assert sorted(p.name for p in d.iterdir()) == [
        "cuts.json", "features.json", "manifest.json", "model.joblib"]
    # cuts.json is the BUILD_CONTRACT shape: plain scenario -> float.
    assert json.loads((d / "cuts.json").read_text()) == {
        s: cuts[s]["cut"] for s in SCENARIOS}
    assert json.loads((d / "features.json").read_text()) == list(fc.features)
    manifest = json.loads((d / "manifest.json").read_text())
    for fn in ("model.joblib", "cuts.json", "features.json"):
        assert manifest["files"][fn] == hashlib.sha256(
            (d / fn).read_bytes()).hexdigest()
    assert manifest["gbdt_params"] == GBDT_PARAMS
    assert manifest["features_frozen_order"] == list(FEATS)
    assert manifest["corpus"]["map_sha256"] == "ab" * 32
    assert manifest["built_at"] == "2026-08-17T00:00:00Z"
    assert manifest["source_commit"] == "deadbeef"
    # bundle_sha256 = sha256 of the manifest bytes, and the manifest itself
    # must not contain it (it hashes those bytes).
    assert out["bundle_sha256"] == hashlib.sha256(
        (d / "manifest.json").read_bytes()).hexdigest()
    assert "bundle_sha256" not in (d / "manifest.json").read_text()


def test_bundle_manifest_is_deterministic(tmp_path):
    fc, clf, fit_meta, cuts, corpus_meta = _tiny_bundle_inputs()
    kw = dict(clf=clf, scenario_cuts=cuts, construction=fc,
              corpus_meta=corpus_meta, fit_meta=fit_meta,
              built_at="2026-08-17T00:00:00Z", source_commit="deadbeef")
    out1 = M.write_bundle(tmp_path / "b1", **kw)
    out2 = M.write_bundle(tmp_path / "b2", **kw)
    assert (tmp_path / "b1" / "manifest.json").read_bytes() == \
           (tmp_path / "b2" / "manifest.json").read_bytes()
    assert out1["bundle_sha256"] == out2["bundle_sha256"]


def test_corpus_census_counts_per_scenario():
    rng = np.random.default_rng(7)
    rows = _synthetic_rows(rng, n_honest=7, n_mal=3)
    census = M.corpus_census(rows)
    assert sorted(census) == SCENARIOS
    for s in SCENARIOS:
        assert census[s] == {"total": 10, "honest": 7, "malicious": 3}


# --------------------------------------------------------------------------
# corpus refusals — never a partial corpus
# --------------------------------------------------------------------------
def test_load_corpus_map_refuses_wrong_cell_count(tmp_path):
    doc = {"cells": [
        {"scenario": "s0_clean_baseline", "seed": 1, "path": "/nonexistent"},
        {"scenario": "s1_benign_churn_only", "seed": 1, "path": "/nonexistent"},
    ]}
    p = tmp_path / "map.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(Refusal, match="50"):
        M.load_corpus_map(p)


def test_rebuild_assembly_map_download_failure_is_loud(tmp_path, monkeypatch):
    def _boom(uri, dest, dry_run):
        raise SystemExit(f"download failed: {uri}")
    monkeypatch.setattr(M.ASSEMBLY_BUILDER, "download", _boom)
    with pytest.raises(Refusal, match="no partial corpus"):
        M.rebuild_assembly_map(tmp_path / "staging", tmp_path / "map.json")


# --------------------------------------------------------------------------
# --reuse-map custody — an external sha256 authority, never self-vouching
# --------------------------------------------------------------------------
def test_verify_reused_map_correct_sha_passes(tmp_path):
    p = tmp_path / "map.json"
    p.write_text('{"cells": []}')
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    M.verify_reused_map(p, sha)                       # no raise
    M.verify_reused_map(p, f"  {sha.upper()}  ")      # normalized, no raise


def test_verify_reused_map_wrong_sha_refuses(tmp_path):
    p = tmp_path / "map.json"
    p.write_text('{"cells": []}')
    with pytest.raises(Refusal, match="sha256"):
        M.verify_reused_map(p, "0" * 64)


def test_verify_reused_map_tampered_map_refuses(tmp_path):
    p = tmp_path / "map.json"
    p.write_text('{"cells": []}')
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text('{"cells": [{"tampered": true}]}')
    with pytest.raises(Refusal, match="sha256"):
        M.verify_reused_map(p, sha)


def test_verify_reused_map_missing_file_refuses(tmp_path):
    with pytest.raises(Refusal, match="not found"):
        M.verify_reused_map(tmp_path / "absent.json", "0" * 64)


def test_reuse_map_without_expected_sha_is_an_argparse_error(tmp_path, capsys):
    p = tmp_path / "map.json"
    p.write_text('{"cells": []}')
    with pytest.raises(SystemExit) as exc:
        M.main(["--reuse-map", str(p), "--built-at", "2026-08-17T00:00:00Z"])
    assert exc.value.code == 2
    assert "--expect-map-sha256" in capsys.readouterr().err


def test_reuse_map_wrong_expected_sha_refuses_before_loading(tmp_path, capsys):
    p = tmp_path / "map.json"
    p.write_text('{"cells": []}')
    rc = M.main(["--reuse-map", str(p), "--expect-map-sha256", "0" * 64,
                 "--built-at", "2026-08-17T00:00:00Z", "--skip-mlflow"])
    assert rc == 2
    assert "sha256" in capsys.readouterr().err


# --------------------------------------------------------------------------
# BUILD_REPORT — states the ACTUAL registration outcome, never a claim
# --------------------------------------------------------------------------
def _report_inputs():
    fc, clf, fit_meta, cuts, corpus_meta = _tiny_bundle_inputs()
    bundle = {"bundle_sha256": "cd" * 32}
    kw = dict(bundle=bundle, scenario_cuts=cuts, corpus_meta=corpus_meta,
              fit_meta=fit_meta, built_at="2026-08-17T00:00:00Z",
              source_commit="deadbeef")
    return kw


def test_report_skip_mlflow_states_skipped_and_claims_nothing(tmp_path):
    path = M.write_report(tmp_path, registration=None, **_report_inputs())
    text = path.read_text()
    assert "SKIPPED (--skip-mlflow)" in text
    assert "offline-only" in text
    assert "registered as" not in text
    assert "champion__H4" not in text


def test_report_registered_states_actual_run_and_version(tmp_path):
    reg = {"run_id": "abc123", "model_name": M.MODEL_NAME,
           "model_version": "7", "alias": M.MODEL_ALIAS}
    path = M.write_report(tmp_path, registration=reg, **_report_inputs())
    text = path.read_text()
    assert "abc123" in text
    assert f"`{M.MODEL_NAME}` v7 @ alias `{M.MODEL_ALIAS}`" in text
    assert "SKIPPED" not in text

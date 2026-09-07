"""Erratum-B build item 3: the observe-log calibration-cut builder.

`scripts/build_h4_serving_cuts_v2.py` consumes the EXP-062 calibration
units' `h2p_observe` blocks (downloaded result JSONs + the fleet manifest),
pools GROUND-TRUTH-HONEST rows per (scenario, arm-class) across the two dev
seeds, and emits serving bundle v2 (cuts_v2.json + manifest_v2.json +
CUTS_V2_REPORT.md) with the frozen E3 cut mechanics. The v1 bundle files
stay byte-untouched.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import build_h4_serving_cuts_v2 as B  # noqa: E402
from h4_bundle_fixture import make_synthetic_bundle  # noqa: E402
from praxis_exp.units import expand_matrix  # noqa: E402

CONFIGS = ["Krum", "TrustScore", "FedAvg"]
SCENARIOS = [
    "C0_clean_no_attack", "S0_clean_baseline", "S1_benign_churn_only",
    "S2_adaptive_switching_only", "S3_identity_reset_only", "S4_full_mix",
]
SEEDS = [42, 137]
ARM_CLASS = {"Krum": "krum_family", "TrustScore": "ts_family",
             "FedAvg": "fedavg_family"}


def _units():
    return expand_matrix(CONFIGS, SCENARIOS, SEEDS, "Flower", 5000, 50)


def _write_manifest(dirpath: Path, units=None) -> Path:
    units = _units() if units is None else units
    payload = {
        "exp_id": "EXP-062",
        "meta": {"run_extras": {"h2p_observe_only": True}},
        "units": [
            {
                "config": u.config, "scenario": u.scenario, "mode": u.mode,
                "seed": u.seed, "max_per_client": u.max_per_client,
                "rounds": u.rounds, "array_index": u.array_index,
                "repeat": u.repeat,
            }
            for u in units
        ],
    }
    path = dirpath / "manifest.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


def _observe_rows(rng, scenario: str, n_honest=300, n_malicious=120,
                  honest_scores=None):
    rows = []
    honest = (honest_scores if honest_scores is not None
              else rng.uniform(0.0, 0.6, size=n_honest))
    for i, s in enumerate(honest):
        rows.append({
            "server_round": 2 + (i % 50), "scenario_round": 1 + (i % 50),
            "logical_cid": f"client_{i % 11}",
            "score": float(s), "would_flag": bool(s > 0.5),
            "malicious_gt": False,
        })
    if not scenario.startswith("C0"):
        for i in range(n_malicious):
            s = float(rng.uniform(0.7, 1.0))
            rows.append({
                "server_round": 2 + (i % 50), "scenario_round": 1 + (i % 50),
                "logical_cid": f"adv_{i % 9}",
                "score": s, "would_flag": True,
                "malicious_gt": True,
            })
    return rows


def _write_results(dirpath: Path, v1_sha: str, mutate=None,
                   honest_by_cell=None):
    """One result JSON per unit; `mutate(unit, payload)` may edit payloads."""
    rng = np.random.default_rng(7)
    dirpath.mkdir(parents=True, exist_ok=True)
    for u in _units():
        cell = (u.scenario, ARM_CLASS[u.config])
        honest_scores = None
        if honest_by_cell and cell in honest_by_cell:
            honest_scores = honest_by_cell[cell]
        rows = _observe_rows(rng, u.scenario, honest_scores=honest_scores)
        payload = {
            "config": u.config,
            "seed": u.seed,
            "return_code": 0,
            "trajectory": [{"round": r, "accuracy": 0.9} for r in range(1, 6)],
            "provenance": {
                "run_uid": f"uid-{u.unit_id}",
                "scenario_path": f"rmc/scenarios/{u.scenario}.json",
                "h2p_observe_only": True,
                "serving_bundle_sha256": v1_sha,
                "h2p_cuts_version": "v1",
                "h2p_arm_class": ARM_CLASS[u.config],
            },
            "h2p_observe": {
                "mode": "observe_only",
                "cuts_version": "v1",
                "scenario_token": "S0" if u.scenario.startswith(("C0", "S0"))
                                  else u.scenario[:2],
                "arm_class": ARM_CLASS[u.config],
                "cut": 0.5,
                "serving_bundle_sha256": v1_sha,
                "n_rows": len(rows),
                "rows": rows,
            },
        }
        if mutate is not None:
            mutate(u, payload)
        (dirpath / f"{u.unit_id}.json").write_text(json.dumps(payload))
    return dirpath


@pytest.fixture
def env(tmp_path):
    """v1 bundle + manifest + 36 observe result JSONs."""
    out_dir = make_synthetic_bundle(tmp_path / "h4_serving")
    v1_sha = hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()
    manifest = _write_manifest(tmp_path)
    results = _write_results(tmp_path / "results", v1_sha)
    return SimpleNamespaceEnv(tmp_path, out_dir, v1_sha, manifest, results)


class SimpleNamespaceEnv:
    def __init__(self, tmp_path, out_dir, v1_sha, manifest, results):
        self.tmp_path = tmp_path
        self.out_dir = out_dir
        self.v1_sha = v1_sha
        self.manifest = manifest
        self.results = results


def _build(env, **overrides):
    argv = [
        "--results-dir", str(env.results),
        "--manifest", str(env.manifest),
        "--out-dir", str(env.out_dir),
        "--built-at", "2026-08-19T00:00:00Z",
    ]
    for k, v in overrides.items():
        argv += [k] if v is True else [k, str(v)]
    return B.main(argv)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_emits_18_cell_cuts_v2(env, capsys):
    assert _build(env) == 0
    cuts = json.loads((env.out_dir / "cuts_v2.json").read_text())
    assert set(cuts) == {"C0", "S0", "S1", "S2", "S3", "S4"}
    for tok in cuts:
        assert set(cuts[tok]) == {"krum_family", "ts_family", "fedavg_family"}
        for v in cuts[tok].values():
            assert np.isfinite(v)
    out = capsys.readouterr().out
    assert "bundle_v2_sha256=" in out


@pytest.mark.unit
def test_bundle_v2_sha_is_sha256_of_manifest_v2_bytes(env, capsys):
    assert _build(env) == 0
    expected = hashlib.sha256(
        (env.out_dir / "manifest_v2.json").read_bytes()).hexdigest()
    assert f"bundle_v2_sha256={expected}" in capsys.readouterr().out


@pytest.mark.unit
def test_v1_files_stay_byte_untouched(env):
    before = {
        name: (env.out_dir / name).read_bytes()
        for name in ("model.joblib", "cuts.json", "features.json",
                     "manifest.json")
    }
    assert _build(env) == 0
    for name, data in before.items():
        assert (env.out_dir / name).read_bytes() == data, name


@pytest.mark.unit
def test_manifest_v2_pins_shared_model_and_features_shas(env):
    assert _build(env) == 0
    v1 = json.loads((env.out_dir / "manifest.json").read_text())
    v2 = json.loads((env.out_dir / "manifest_v2.json").read_text())
    assert v2["files"]["model.joblib"] == v1["files"]["model.joblib"]
    assert v2["files"]["features.json"] == v1["files"]["features.json"]
    assert set(v2["files"]) == {"model.joblib", "cuts_v2.json",
                                "features.json"}
    assert v2["v1_bundle_sha256"] == env.v1_sha
    cal = v2["calibration"]
    assert cal["exp_id"] == "EXP-062"
    assert cal["n_units"] == 36
    # per-cell counts + realized FPRs recorded
    for tok in ("C0", "S0", "S1", "S2", "S3", "S4"):
        for ac in ("krum_family", "ts_family", "fedavg_family"):
            cell = cal["cells"][tok][ac]
            assert cell["n_honest"] >= 400
            assert 0.08 <= cell["realized_fpr"] <= 0.12


@pytest.mark.unit
def test_cut_mechanics_are_the_frozen_e3_quantile(env):
    assert _build(env) == 0
    cuts = json.loads((env.out_dir / "cuts_v2.json").read_text())
    v2 = json.loads((env.out_dir / "manifest_v2.json").read_text())
    # recompute one cell from the raw observe logs
    honest = []
    for u in _units():
        if u.config != "Krum" or not u.scenario.startswith("S3"):
            continue
        payload = json.loads((env.results / f"{u.unit_id}.json").read_text())
        honest += [r["score"] for r in payload["h2p_observe"]["rows"]
                   if not r["malicious_gt"]]
    expected = float(np.quantile(np.asarray(honest, dtype=float), 1 - 0.10))
    assert cuts["S3"]["krum_family"] == pytest.approx(expected)
    # primary mechanics stay the frozen E3 quantile; the recorded semantics
    # must also disclose the erratum-C C1 fallback ( provenance fix)
    assert v2["cut_semantics"].startswith("primary: cut = ")
    assert "Erratum-C C1 fallback" in v2["cut_semantics"]


@pytest.mark.unit
def test_bracket_reported_with_c0_recall_null_with_reason(env):
    assert _build(env) == 0
    v2 = json.loads((env.out_dir / "manifest_v2.json").read_text())
    cell = v2["calibration"]["cells"]["S3"]["krum_family"]
    bracket = cell["bracket"]
    assert set(bracket) == {"0.01", "0.02", "0.05", "0.10"}
    for point in bracket.values():
        assert 0.0 <= point["realized_fpr"] <= 1.0
        assert point["recall"] is not None  # malicious rows present
    c0 = v2["calibration"]["cells"]["C0"]["fedavg_family"]["bracket"]
    for point in c0.values():
        assert point["recall"] is None
        assert "no ground-truth malicious rows" in point["recall_reason"]


@pytest.mark.unit
def test_report_written(env):
    assert _build(env) == 0
    text = (env.out_dir / "CUTS_V2_REPORT.md").read_text()
    assert "bundle_v2_sha256" in text
    assert "| S3 |" in text
    assert "krum_family" in text


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _expect_refusal(env, capsys, needle: str):
    rc = _build(env)
    assert rc == 2
    assert needle in capsys.readouterr().err


@pytest.mark.unit
def test_missing_unit_file_refuses(env, capsys):
    victims = sorted(env.results.glob("*.json"))
    victims[0].unlink()
    _expect_refusal(env, capsys, "missing")


@pytest.mark.unit
def test_census_deviation_refuses(env, capsys):
    units = _units()[:-1]  # drop one cell from the manifest itself
    _write_manifest(env.tmp_path, units)
    _expect_refusal(env, capsys, "census")


@pytest.mark.unit
def test_non_observe_unit_refuses(env, capsys):
    def mutate(u, payload):
        if u.config == "Krum" and u.scenario.startswith("S1") and u.seed == 42:
            payload["provenance"]["h2p_observe_only"] = False
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "observe")


@pytest.mark.unit
def test_wrong_bundle_sha_refuses(env, capsys):
    def mutate(u, payload):
        if u.config == "FedAvg" and u.scenario.startswith("S2") and u.seed == 137:
            payload["h2p_observe"]["serving_bundle_sha256"] = "ff" * 32
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "sha256")


@pytest.mark.unit
def test_wrong_arm_class_refuses(env, capsys):
    def mutate(u, payload):
        if u.config == "TrustScore" and u.scenario.startswith("S4") and u.seed == 42:
            payload["h2p_observe"]["arm_class"] = "krum_family"
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "arm")


@pytest.mark.unit
def test_swapped_result_files_refuse(env, capsys):
    """: a filename-swapped result (S1 payload under the S3 unit's
    filename) must refuse on the scenario binding, never pool into the
    wrong (scenario, arm_class) cell (H3 event<->unit binding lesson)."""
    s1 = s3 = None
    for u in _units():
        if u.config == "Krum" and u.seed == 42:
            if u.scenario.startswith("S1"):
                s1 = env.results / f"{u.unit_id}.json"
            elif u.scenario.startswith("S3"):
                s3 = env.results / f"{u.unit_id}.json"
    s3.write_text(s1.read_text())  # S1 payload under the S3 filename
    _expect_refusal(env, capsys, "scenario")


@pytest.mark.unit
def test_block_scenario_token_mismatch_refuses(env, capsys):
    """: the h2p_observe block's own scenario_token must equal the
    census cell's token (under the block's declared cuts_version)."""
    def mutate(u, payload):
        if u.config == "Krum" and u.scenario.startswith("S3") and u.seed == 42:
            payload["h2p_observe"]["scenario_token"] = "S1"
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "scenario")


@pytest.mark.unit
def test_provenance_scenario_path_mismatch_refuses(env, capsys):
    """: the result's OWN provenance scenario stem must match the
    census cell — a mislabeled result cannot ride a correct block header."""
    def mutate(u, payload):
        if u.config == "FedAvg" and u.scenario.startswith("S2") and u.seed == 42:
            payload["provenance"]["scenario_path"] = (
                "rmc/scenarios/S4_full_mix.json")
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "scenario")


@pytest.mark.unit
def test_c0_unit_binds_via_the_v1_alias_token(env, capsys):
    """Happy-path pin: a C0 unit whose block logged the S0 cut token (the v1
    E3-bis alias the observer actually served) still binds to the C0 census
    cell — the binding compares against the alias-resolved expectation, not
    the raw cell token."""
    assert _build(env) == 0  # fixture C0 blocks carry scenario_token "S0"


@pytest.mark.unit
def test_provenance_arm_class_mismatch_refuses(env, capsys):
    """: the provenance arm-class declaration (when present) must
    match the census cell too — no bypass around the block-level check."""
    def mutate(u, payload):
        if u.config == "Krum" and u.scenario.startswith("S0") and u.seed == 137:
            payload["provenance"]["h2p_arm_class"] = "ts_family"
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "arm")


@pytest.mark.unit
def test_under_400_honest_rows_refuses(env, capsys):
    def mutate(u, payload):
        if u.config == "Krum" and u.scenario.startswith("S1"):
            rows = [r for r in payload["h2p_observe"]["rows"]
                    if r["malicious_gt"]]
            honest = [r for r in payload["h2p_observe"]["rows"]
                      if not r["malicious_gt"]][:150]
            payload["h2p_observe"]["rows"] = honest + rows
            payload["h2p_observe"]["n_rows"] = len(honest + rows)
    _write_results(env.results, env.v1_sha, mutate=mutate)
    _expect_refusal(env, capsys, "400")


@pytest.mark.unit
def test_degenerate_scores_fail_the_fpr_band(env, capsys):
    """All-identical honest scores: strict > flags nothing at the quantile
    cut, realized FPR 0.0 < 0.08 — construction check refuses."""
    ties = {("S1_benign_churn_only", "ts_family"):
            np.full(500, 0.3, dtype=float)}
    _write_results(env.results, env.v1_sha, honest_by_cell=ties)
    _expect_refusal(env, capsys, "0.08")


@pytest.mark.unit
def test_register_without_emitted_bundle_refuses(env, capsys):
    rc = B.main(["--out-dir", str(env.out_dir), "--register",
                 "--mlflow-uri", "http://mlflow.invalid"])
    assert rc == 2
    assert "manifest_v2" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Erratum-C C1: tie-aware step-down fallback
# ---------------------------------------------------------------------------

def test_stepdown_cut_picks_largest_in_band_distinct_value():
    # 1000 honest rows; a 7% tie plateau at 0.9 CONTAINS the 90th-percentile
    # index, so the E3 quantile equals the plateau value and strict-`>`
    # realizes only the 5% above it (< 0.08). Stepping down to the largest
    # distinct value below the plateau realizes 0.12 — in band (inclusive).
    honest = np.concatenate([
        np.linspace(0.0, 0.7, 880),      # 88% below the plateau
        np.full(70, 0.9),                # 7% tie plateau spanning P90
        np.linspace(1.0, 2.0, 50),       # 5% strictly above
    ])
    e3 = B.calibration_cut(honest)
    assert e3 == pytest.approx(0.9)
    assert B.realized_fpr(honest, e3) < 0.08  # the plateau undershoot
    adopted = B.stepdown_cut(honest, 0.08, 0.12)
    assert adopted is not None
    assert adopted < e3
    assert 0.08 <= B.realized_fpr(honest, adopted) <= 0.12


def test_stepdown_cut_returns_none_when_no_value_is_in_band():
    # One giant plateau: below it FPR jumps straight from 0.0 to 0.15.
    honest = np.concatenate([np.full(850, 1.0), np.full(150, 2.0)])
    assert B.stepdown_cut(honest, 0.08, 0.12) is None


def test_in_band_cells_are_byte_identical_to_e3(env, capsys):
    # No cell in the synthetic fixture trips the band, so every cell's cut
    # must equal the frozen E3 quantile exactly and carry a null fallback.
    _build(env)
    cuts = json.loads((env.out_dir / "cuts_v2.json").read_text())
    manifest = json.loads((env.out_dir / "manifest_v2.json").read_text())
    for token, arms in manifest["calibration"]["cells"].items():
        for arm, cell in arms.items():
            assert cell["tie_fallback"] is None
            assert cuts[token][arm] == cell["cut"]


def test_plateau_cell_steps_down_and_discloses(env, capsys):
    # Inject a tie plateau into one cell's honest scores (S0, krum_family):
    # every honest score in the [P86, P97) span collapses onto the P86
    # value, so the E3 quantile lands on the plateau and undershoots the
    # band. The build must SUCCEED, adopt a lower cut for that cell,
    # realize an in-band FPR, and disclose both cuts.
    def mutate(u, payload):
        if u.config == "Krum" and u.scenario.startswith("S0"):
            rows = [r for r in payload["h2p_observe"]["rows"]
                    if not r["malicious_gt"]]
            scores = sorted(r["score"] for r in rows)
            cutoff = scores[int(len(scores) * 0.89)]
            for r in rows:
                if r["score"] >= cutoff:
                    # one shared plateau value across BOTH seeds' units, so
                    # the pooled cell has ~11% tie mass at its top: the E3
                    # quantile lands ON the plateau (strict-`>` FPR 0.0)
                    # and the C1 step-down must fire.
                    r["score"] = 10.0

    _write_results(env.results, env.v1_sha, mutate=mutate)
    _build(env)
    manifest = json.loads((env.out_dir / "manifest_v2.json").read_text())
    cell = manifest["calibration"]["cells"]["S0"]["krum_family"]
    assert cell["tie_fallback"] is not None
    assert cell["tie_fallback"]["primary_cut"] > cell["cut"]
    assert cell["tie_fallback"]["primary_realized_fpr"] < 0.08
    assert 0.08 <= cell["realized_fpr"] <= 0.12

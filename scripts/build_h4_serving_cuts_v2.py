"""Build serving bundle v2 — erratum-B population-matched calibration cuts.

Executes the frozen cut construction summarized in the H4 section of
`docs/reproduction/experiments.md`, ruled as methodology v1.53 and amended by
historical erratum C § C1 (tie-aware
step-down fallback, RULED, methodology v1.54) — verbatim, nothing decided
here:

  1. Read the EXP-062 fleet manifest and enforce the calibration census:
     3 base arms (Krum/TrustScore/FedAvg = krum_family/ts_family/
     fedavg_family) x 6 scenarios (C0, S0-S4) x the 2 dev seeds = 36 units.
     ANY deviation is a refusal — no partial calibration.
  2. Load each unit's downloaded result JSON ({unit_id}.json) and its
     `h2p_observe` block; corroborate custody (observe-only mode, the v1
     serving-bundle sha the observer scored with, the arm-class mapping).
  3. Pool GROUND-TRUTH-HONEST rows per (scenario, arm-class) across seeds;
     REFUSE any cell under 400 honest rows.
  4. Cut per cell with the frozen E3 mechanics:
     `numpy.quantile(honest_scores, 1 - 0.10)` (linear interpolation),
     serving flag STRICT `score > cut`; construction-check the realized
     calibration FPR into the closed band [0.08, 0.12]. Where the E3
     quantile lands on a tie plateau and realizes an out-of-band FPR,
     erratum-C C1 steps the cut DOWN to the largest distinct honest-score
     value with an in-band realized FPR (disclosed per cell as
     `tie_fallback`); refusal stands where no distinct value is in band.
  5. REPORTED bracket per cell at {0.01, 0.02, 0.05, 0.10}: recall vs
     ground-truth-malicious rows where present; C0 recall = null-with-reason.
  6. Emit `cuts_v2.json` + `manifest_v2.json` (same model/features files and
     shas as the v1 manifest — v1 files stay byte-untouched) +
     `CUTS_V2_REPORT.md`; print bundle_v2_sha256 = sha256(manifest_v2 bytes).
  7. `--register` (separate step, hard-fail on error): MLflow
     `praxis-h2prime-detector` v2, alias `champion__H4` moved, artifacts +
     rich tags.

Usage (the executed build):
    conda run -n flowerfl python scripts/build_h4_serving_cuts_v2.py \
        --results-dir /path/to/exp062/results \
        --manifest /path/to/exp062/manifest.json \
        --built-at 2026-08-19T00:00:00Z
    # then, after review:
    AWS_PROFILE=research MLFLOW_TRACKING_URI=http://localhost:5001 \
    conda run -n flowerfl python scripts/build_h4_serving_cuts_v2.py \
        --register
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR_DEFAULT = REPO / "data" / "h4_serving"
SPEC_REF = (
    "docs/superpowers/specs/2026-08-18-h4-preregistration-erratum-b.md "
    "§§ B1-B2 (RULED, methodology v1.53), as amended by "
    "docs/superpowers/specs/2026-08-20-h4-preregistration-erratum-c.md "
    "§ C1 (RULED, methodology v1.54: tie-aware step-down fallback)"
)
MODEL_NAME = "praxis-h2prime-detector"
MODEL_ALIAS = "champion__H4"
MLFLOW_EXPERIMENT = "h4_serving_bundle"
BUNDLE_FORMAT = "h4-serving-v2"
TARGET_FPR = 0.10
FPR_BAND = (0.08, 0.12)
MIN_HONEST_ROWS = 400
BRACKET_FPRS = (0.01, 0.02, 0.05, 0.10)
CUT_SEMANTICS = (
    "primary: cut = float(np.quantile(honest_scores, 1 - 0.10)) per "
    "(scenario, arm_class) on OBSERVED online honest scores (NumPy default "
    "linear interpolation, frozen E3 mechanics); flag = score > cut "
    "(strict). Erratum-C C1 fallback: where the primary cut's realized "
    "strict-> FPR falls outside [0.08, 0.12], the cut steps DOWN to the "
    "largest distinct honest-score value realizing an in-band FPR "
    "(per-cell tie_fallback discloses both cuts); refusal where no "
    "distinct value is in band.")

#: Calibration base arms -> erratum-B arm-classes (§ B1).
CONFIG_ARM_CLASS = {
    "Krum": "krum_family",
    "TrustScore": "ts_family",
    "FedAvg": "fedavg_family",
}
#: The two dev seeds the calibration is pre-registered on (§ B1).
CALIBRATION_SEEDS = frozenset({42, 137})
ARM_CLASSES = ("krum_family", "ts_family", "fedavg_family")
SCENARIO_TOKENS_V2 = ("C0", "S0", "S1", "S2", "S3", "S4")


class Refusal(RuntimeError):
    """The build never emits past one of these."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _scenario_token(name: str) -> str:
    from flowerfl.h2prime_online import scenario_token_from_name

    return scenario_token_from_name(str(name), cuts_version="v2")


# ---------------------------------------------------------------------------
# manifest census + unit loading
# ---------------------------------------------------------------------------

def load_calibration_manifest(path: Path):
    """The EXP-062 manifest, census-enforced: exactly the 36 pre-registered
    (arm-class x scenario x dev-seed) units, no repeats axis."""
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"fleet manifest not found: {path}")
    try:
        data = json.loads(path.read_text())
        exp_id = str(data["exp_id"])
        raw_units = list(data["units"])
    except (ValueError, KeyError, TypeError) as exc:
        raise Refusal(f"fleet manifest unreadable ({path}): {exc}") from exc

    from praxis_exp.units import Unit

    units = [Unit(**u) for u in raw_units]
    seen = set()
    for u in units:
        if u.config not in CONFIG_ARM_CLASS:
            raise Refusal(
                f"census: unit {u.unit_id} config {u.config!r} is not a "
                f"calibration base arm {sorted(CONFIG_ARM_CLASS)} (§ B1)")
        if u.seed not in CALIBRATION_SEEDS:
            raise Refusal(
                f"census: unit {u.unit_id} seed is outside the "
                f"pre-registered dev-seed pair (§ B1)")
        if int(u.repeat) != 0:
            raise Refusal(
                f"census: unit {u.unit_id} has an active repeat axis — the "
                f"calibration matrix is 3x6x2 with no replicates")
        key = (u.config, _scenario_token(u.scenario), u.seed)
        if key in seen:
            raise Refusal(f"census: duplicate cell {key} in the manifest")
        seen.add(key)

    expected = {
        (config, token, seed)
        for config in CONFIG_ARM_CLASS
        for token in SCENARIO_TOKENS_V2
        for seed in CALIBRATION_SEEDS
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise Refusal(
            f"census deviation: manifest covers {len(seen)}/36 calibration "
            f"cells; missing={missing[:6]} extra={extra[:6]} — no partial "
            f"calibration (§ B1)")
    return exp_id, units


def load_unit_observe(results_dir: Path, unit, v1_sha: str) -> dict:
    """One unit's `h2p_observe` block, custody-corroborated."""
    path = Path(results_dir) / f"{unit.unit_id}.json"
    if not path.is_file():
        raise Refusal(f"unit result missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise Refusal(f"unit result unreadable ({path}): {exc}") from exc

    if payload.get("return_code") != 0:
        raise Refusal(f"{unit.unit_id}: return_code != 0 — not a scorable "
                      f"calibration unit")
    prov = payload.get("provenance") or {}
    if prov.get("h2p_observe_only") is not True:
        raise Refusal(
            f"{unit.unit_id}: provenance.h2p_observe_only is not true — the "
            f"unit did not run the observe-only calibration mode (§ B1)")
    block = payload.get("h2p_observe")
    if not isinstance(block, dict) or block.get("mode") != "observe_only":
        raise Refusal(
            f"{unit.unit_id}: no observe_only h2p_observe block — the "
            f"calibration log is absent")

    # --- scenario binding (PR #68 P1 — the H3 event<->unit binding lesson):
    # the result's OWN scenario identity must equal the census cell it is
    # pooled into. A mislabeled or filename-swapped result would otherwise
    # contaminate a neighboring (scenario, arm_class) cell SILENTLY.
    cell_token = _scenario_token(unit.scenario)
    prov_scenario_path = prov.get("scenario_path")
    if prov_scenario_path is not None:
        stem = Path(str(prov_scenario_path)).stem
        try:
            prov_token = _scenario_token(stem)
        except Exception as exc:
            raise Refusal(
                f"{unit.unit_id}: provenance.scenario_path stem {stem!r} "
                f"does not resolve to a calibration scenario token — cannot "
                f"corroborate the census-cell binding: {exc}") from exc
        if prov_token != cell_token:
            raise Refusal(
                f"{unit.unit_id}: provenance scenario {stem!r} (token "
                f"{prov_token!r}) does not EQUAL the census cell's scenario "
                f"token {cell_token!r} — a mislabeled/swapped result must "
                f"never pool into another cell (PR #68 P1)")
    block_cuts_version = block.get("cuts_version")
    if block_cuts_version not in ("v1", "v2"):
        raise Refusal(
            f"{unit.unit_id}: h2p_observe.cuts_version="
            f"{block_cuts_version!r} is not in ('v1', 'v2') — the block's "
            f"scenario token cannot be corroborated without it")
    from flowerfl.h2prime_online import scenario_token_from_name

    expected_block_token = scenario_token_from_name(
        unit.scenario, cuts_version=block_cuts_version)
    if block.get("scenario_token") != expected_block_token:
        raise Refusal(
            f"{unit.unit_id}: h2p_observe.scenario_token="
            f"{block.get('scenario_token')!r} does not EQUAL the census "
            f"cell's expected token {expected_block_token!r} (cell scenario "
            f"{unit.scenario!r} under the block's declared "
            f"{block_cuts_version} cut table) — the observe log belongs to "
            f"a different scenario (PR #68 P1)")

    # --- arm-class binding: block AND (when present) provenance must both
    # name the census cell's arm-class — no bypass path.
    expected_arm = CONFIG_ARM_CLASS[unit.config]
    if block.get("arm_class") != expected_arm:
        raise Refusal(
            f"{unit.unit_id}: h2p_observe.arm_class={block.get('arm_class')!r} "
            f"does not match the config's arm-class {expected_arm!r} (§ B1 "
            f"arm-class table)")
    prov_arm = prov.get("h2p_arm_class")
    if prov_arm is not None and prov_arm != expected_arm:
        raise Refusal(
            f"{unit.unit_id}: provenance.h2p_arm_class={prov_arm!r} does not "
            f"match the config's arm-class {expected_arm!r} — the unit did "
            f"not run the arm it is being pooled as (PR #68 P1)")
    for field, value in (("h2p_observe", block.get("serving_bundle_sha256")),
                         ("provenance", prov.get("serving_bundle_sha256"))):
        if value != v1_sha:
            raise Refusal(
                f"{unit.unit_id}: {field}.serving_bundle_sha256={value!r} "
                f"does not EQUAL the local v1 bundle sha256 {v1_sha!r} — the "
                f"observer scored with a different instrument")
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Refusal(f"{unit.unit_id}: h2p_observe.rows is empty")
    for row in rows:
        score = row.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) \
                or not np.isfinite(float(score)):
            raise Refusal(
                f"{unit.unit_id}: observe row with non-finite score "
                f"{score!r} — refusing to calibrate on it")
        if not isinstance(row.get("malicious_gt"), bool):
            raise Refusal(
                f"{unit.unit_id}: observe row without a boolean "
                f"malicious_gt — ground truth is the calibration's spine")
    return block


# ---------------------------------------------------------------------------
# pooling + cuts (frozen E3 mechanics)
# ---------------------------------------------------------------------------

def pool_cells(units, blocks) -> dict:
    """{(token, arm_class): {"honest": ndarray, "malicious": ndarray}}."""
    pools: dict = {
        (token, ac): {"honest": [], "malicious": []}
        for token in SCENARIO_TOKENS_V2 for ac in ARM_CLASSES
    }
    for unit, block in zip(units, blocks):
        key = (_scenario_token(unit.scenario), CONFIG_ARM_CLASS[unit.config])
        for row in block["rows"]:
            side = "malicious" if row["malicious_gt"] else "honest"
            pools[key][side].append(float(row["score"]))
    return {
        key: {side: np.asarray(vals, dtype=float)
              for side, vals in cell.items()}
        for key, cell in pools.items()
    }


def calibration_cut(honest_scores: np.ndarray) -> float:
    """The frozen E3 cut: linear quantile at 1 - TARGET_FPR."""
    return float(np.quantile(honest_scores, 1.0 - TARGET_FPR))


def realized_fpr(honest_scores: np.ndarray, cut: float) -> float:
    """Strict-greater flagging rate on the honest pool."""
    return float((honest_scores > cut).mean())


def stepdown_cut(honest_scores: np.ndarray, lo: float, hi: float):
    """Erratum-C C1 fallback: the LARGEST distinct honest-score value whose
    strict-`>` realized FPR lies inside [lo, hi], or None if no distinct
    value achieves an in-band FPR.

    Scanning distinct values in descending order, realized FPR is
    non-decreasing; the first value whose FPR reaches the band is the
    largest in-band value, and once the FPR overshoots ``hi`` no smaller
    value can come back down — refusal territory.
    """
    for value in np.sort(np.unique(honest_scores))[::-1]:
        fpr = float((honest_scores > value).mean())
        if lo <= fpr <= hi:
            return float(value)
        if fpr > hi:
            return None
    return None


def bracket_points(honest: np.ndarray, malicious: np.ndarray) -> dict:
    """REPORTED-ONLY recall-vs-FPR at the pre-registered bracket points —
    the online analog of the sealed corpus bracket (§ B1)."""
    out = {}
    for fpr in BRACKET_FPRS:
        cut = float(np.quantile(honest, 1.0 - fpr))
        point = {
            "cut": cut,
            "realized_fpr": float((honest > cut).mean()),
        }
        if malicious.size:
            point["recall"] = float((malicious > cut).mean())
            point["recall_reason"] = None
        else:
            point["recall"] = None
            point["recall_reason"] = (
                "no ground-truth malicious rows in this cell (clean "
                "scenario) — recall is undefined, reported null (§ B1)")
        out[f"{fpr:.2f}"] = point
    return out


def build_cells(pools) -> dict:
    """Per-cell cut + counts + realized FPR + bracket; every guard rail is a
    refusal, not a judgment call (§ B1)."""
    lo, hi = FPR_BAND
    cells: dict = {token: {} for token in SCENARIO_TOKENS_V2}
    for (token, arm_class), pool in sorted(pools.items()):
        honest, malicious = pool["honest"], pool["malicious"]
        if honest.size < MIN_HONEST_ROWS:
            raise Refusal(
                f"cell ({token}, {arm_class}) has {honest.size} honest score "
                f"rows < {MIN_HONEST_ROWS} — refusing to set a cut on a "
                f"starved cell (§ B1 guard rail)")
        cut = calibration_cut(honest)
        fpr = realized_fpr(honest, cut)
        tie_fallback = None
        if not (lo <= fpr <= hi):
            # Erratum-C C1: the E3 quantile landed on a tie plateau of the
            # discrete score distribution — step down to the largest
            # distinct value realizing an in-band FPR; refuse if none does.
            adopted = stepdown_cut(honest, lo, hi)
            if adopted is None:
                raise Refusal(
                    f"cell ({token}, {arm_class}) realized calibration FPR "
                    f"{fpr:.4f} at the E3 cut {cut:.6f} is outside the "
                    f"closed band [{lo:.2f}, {hi:.2f}] and NO distinct "
                    f"honest-score value realizes an in-band FPR — bundle "
                    f"v2 is not emitted (erratum-C C1 refusal)")
            tie_fallback = {
                "primary_cut": cut,
                "primary_realized_fpr": fpr,
                "note": "erratum-C C1 step-down: E3 quantile landed on a "
                        "tie plateau; adopted the largest distinct "
                        "honest-score value with in-band realized FPR",
            }
            cut = adopted
            fpr = realized_fpr(honest, cut)
        cells[token][arm_class] = {
            "cut": cut,
            "realized_fpr": fpr,
            "tie_fallback": tie_fallback,
            "n_honest": int(honest.size),
            "n_malicious": int(malicious.size),
            "bracket": bracket_points(honest, malicious),
        }
    return cells


# ---------------------------------------------------------------------------
# v1 authority + bundle emission (v1 files byte-untouched)
# ---------------------------------------------------------------------------

def read_v1_authority(out_dir: Path) -> dict:
    """The local v1 bundle is the model authority: its manifest sha is what
    every calibration unit must have custody-exported, and its recorded
    model/features shas must still match the bytes on disk (v2 SHARES those
    files — a drifted file would ship a different instrument under v2)."""
    manifest_path = Path(out_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise Refusal(
            f"v1 bundle manifest not found: {manifest_path} — bundle v2 "
            f"shares the v1 model file and cannot be built without it")
    v1_sha = _sha256_bytes(manifest_path.read_bytes())
    v1_manifest = json.loads(manifest_path.read_text())
    files = v1_manifest.get("files") or {}
    for name in ("model.joblib", "features.json"):
        recorded = str(files.get(name, ""))
        path = Path(out_dir) / name
        if not path.is_file():
            raise Refusal(f"v1 bundle file missing: {path}")
        actual = _sha256_file(path)
        if actual != recorded:
            raise Refusal(
                f"v1 bundle file {name} sha256 mismatch (recorded "
                f"{recorded}, actual {actual}) — the shared instrument has "
                f"drifted; refusing to stamp it into bundle v2")
    return {"v1_sha": v1_sha, "files": files}


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()


def write_bundle_v2(out_dir: Path, *, cells: dict, exp_id: str, units,
                    v1_authority: dict, built_at: str,
                    source_commit: str) -> dict:
    """Emit cuts_v2.json + manifest_v2.json. NEVER writes any v1 file; the
    bundle_v2 sha (sha256 of the manifest_v2 bytes) is RETURNED and printed,
    never written into the manifest it hashes."""
    out_dir = Path(out_dir)
    cuts_doc = {
        token: {ac: cells[token][ac]["cut"] for ac in ARM_CLASSES}
        for token in SCENARIO_TOKENS_V2
    }
    (out_dir / "cuts_v2.json").write_text(
        json.dumps(cuts_doc, sort_keys=True, indent=1) + "\n")

    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "construction": SPEC_REF,
        "built_at": built_at,
        "source_commit": source_commit,
        "files": {fn: _sha256_file(out_dir / fn)
                  for fn in ("model.joblib", "cuts_v2.json", "features.json")},
        "v1_bundle_sha256": v1_authority["v1_sha"],
        "target_fpr": TARGET_FPR,
        "fpr_band": list(FPR_BAND),
        "min_honest_rows": MIN_HONEST_ROWS,
        "cut_semantics": CUT_SEMANTICS,
        "calibration": {
            "exp_id": exp_id,
            "design": "3 arm-classes x 6 scenarios (C0, S0-S4) x 2 dev "
                      "seeds, observe-only (erratum B § B1)",
            "n_units": len(units),
            "unit_census": sorted(u.unit_id for u in units),
            "cells": cells,
        },
        "model_registration": {"name": MODEL_NAME, "alias": MODEL_ALIAS,
                               "note": "v2 — alias moved from v1 (§ B2)"},
        "versions": {"numpy": np.__version__},
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    (out_dir / "manifest_v2.json").write_bytes(manifest_bytes)
    return {"bundle_v2_sha256": _sha256_bytes(manifest_bytes),
            "manifest": manifest}


def write_report(out_dir: Path, *, bundle: dict, built_at: str,
                 source_commit: str) -> Path:
    manifest = bundle["manifest"]
    cells = manifest["calibration"]["cells"]
    lines = [
        "# H4 serving bundle v2 — calibration-cut build report",
        "",
        f"Construction: {SPEC_REF}",
        f"Built at: {built_at}  |  source commit: `{source_commit}`",
        f"Calibration fleet: {manifest['calibration']['exp_id']} "
        f"({manifest['calibration']['n_units']} observe-only units)",
        f"v1 bundle (shared model file): `{manifest['v1_bundle_sha256']}`",
        f"**bundle_v2_sha256 (sha256 of manifest_v2.json): "
        f"`{bundle['bundle_v2_sha256']}`**",
        "",
        "| scenario | arm_class | honest | malicious | cut | realized FPR | tie fallback (C1) |",
        "|---|---|---|---|---|---|---|",
    ]
    for token in SCENARIO_TOKENS_V2:
        for ac in ARM_CLASSES:
            c = cells[token][ac]
            tf = c.get("tie_fallback")
            tf_col = (f"E3 cut {tf['primary_cut']:.6f} realized "
                      f"{tf['primary_realized_fpr']:.4f} — stepped down"
                      if tf else "—")
            lines.append(
                f"| {token} | {ac} | {c['n_honest']} | {c['n_malicious']} | "
                f"{c['cut']:.6f} | {c['realized_fpr']:.4f} | {tf_col} |")
    lines += [
        "",
        "## Reported bracket (recall at {0.01, 0.02, 0.05, 0.10} FPR)",
        "",
        "| scenario | arm_class | 0.01 | 0.02 | 0.05 | 0.10 |",
        "|---|---|---|---|---|---|",
    ]
    for token in SCENARIO_TOKENS_V2:
        for ac in ARM_CLASSES:
            pts = cells[token][ac]["bracket"]
            def _fmt(p):
                return ("null (no malicious rows)" if p["recall"] is None
                        else f"{p['recall']:.3f}")
            lines.append(
                f"| {token} | {ac} | " + " | ".join(
                    _fmt(pts[f"{f:.2f}"]) for f in BRACKET_FPRS) + " |")
    lines += [
        "",
        f"- cut semantics: {CUT_SEMANTICS}",
        "- v1 files (model.joblib, cuts.json, features.json, manifest.json) "
        "are BYTE-UNTOUCHED; v2 adds cuts_v2.json + manifest_v2.json only.",
        "- E3-bis C0 alias RETIRED for v2: C0 carries its own calibrated cut.",
        "",
    ]
    path = Path(out_dir) / "CUTS_V2_REPORT.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# MLflow registration (--register, separate step, hard-fail)
# ---------------------------------------------------------------------------

def register_in_mlflow(out_dir: Path, tracking_uri: str) -> dict:
    """Register the EMITTED v2 bundle: new `praxis-h2prime-detector` version,
    alias `champion__H4` MOVED to it, artifacts + rich tags. Any failure is a
    hard error — the executed build must land on the review surface."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest_v2.json"
    if not manifest_path.is_file():
        raise Refusal(
            f"manifest_v2.json not found in {out_dir} — build the bundle "
            f"first; --register registers an EMITTED bundle only")
    manifest_bytes = manifest_path.read_bytes()
    bundle_sha = _sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    for fn, recorded in manifest["files"].items():
        actual = _sha256_file(out_dir / fn)
        if actual != recorded:
            raise Refusal(
                f"bundle v2 file {fn} sha256 mismatch at registration "
                f"(recorded {recorded}, actual {actual}) — refusing to "
                f"register a drifted bundle")

    import joblib
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    clf = joblib.load(out_dir / "model.joblib")
    features = json.loads((out_dir / "features.json").read_text())
    x_sample = np.zeros((2, len(features)), dtype=float)
    signature = infer_signature(x_sample, clf.predict_proba(x_sample))
    cal = manifest["calibration"]
    description = (
        f"H4 serving bundle v2 — erratum-B population-matched calibration "
        f"({SPEC_REF}): the SAME frozen § 7.1 GBDT, cuts recalibrated per "
        f"(scenario, arm-class) on the {cal['exp_id']} observe-only online "
        f"honest scores ({cal['n_units']} dev units; C0 calibrated directly, "
        f"E3-bis alias retired). bundle_v2_sha256={bundle_sha}. No H4 output "
        f"ever touches the model or the cuts.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(
            run_name=f"h4-serving-bundle-v2__{manifest['built_at']}") as run:
        mlflow.set_tags({
            "mlflow.note.content": description,
            "spec": SPEC_REF,
            "bundle_v2_sha256": bundle_sha,
            "v1_bundle_sha256": manifest["v1_bundle_sha256"],
            "source_commit": manifest["source_commit"],
            "built_at": manifest["built_at"],
            "calibration_exp_id": cal["exp_id"],
            "calibration_n_units": str(cal["n_units"]),
        })
        mlflow.log_param("target_fpr", TARGET_FPR)
        mlflow.log_param("min_honest_rows", MIN_HONEST_ROWS)
        for token in SCENARIO_TOKENS_V2:
            for ac in ARM_CLASSES:
                cell = cal["cells"][token][ac]
                mlflow.log_metric(f"cut_{token}_{ac}", cell["cut"])
                mlflow.log_metric(
                    f"realized_fpr_{token}_{ac}", cell["realized_fpr"])
                mlflow.log_metric(f"n_honest_{token}_{ac}", cell["n_honest"])
        for fn in ("cuts_v2.json", "manifest_v2.json", "features.json",
                   "CUTS_V2_REPORT.md"):
            path = out_dir / fn
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="bundle_v2")
        try:
            info = mlflow.sklearn.log_model(
                clf, name="model", registered_model_name=MODEL_NAME,
                signature=signature, input_example=x_sample)
        except TypeError:  # 2.x keyword rename tolerance (v1 precedent)
            info = mlflow.sklearn.log_model(
                clf, artifact_path="model", registered_model_name=MODEL_NAME,
                signature=signature, input_example=x_sample)

    client = MlflowClient(tracking_uri=tracking_uri)
    version = str(getattr(info, "registered_model_version", "") or "")
    if not version:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        ours = [v for v in versions if v.run_id == run.info.run_id]
        if not ours:
            raise Refusal(
                f"model registration produced no version for run "
                f"{run.info.run_id}")
        version = str(max(int(v.version) for v in ours))
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    client.update_model_version(MODEL_NAME, version, description=description)
    for k, v in (("bundle_v2_sha256", bundle_sha),
                 ("source_commit", manifest["source_commit"]),
                 ("spec", SPEC_REF)):
        client.set_model_version_tag(MODEL_NAME, version, k, v)
    return {"run_id": run.info.run_id, "model_name": MODEL_NAME,
            "model_version": version, "alias": MODEL_ALIAS,
            "bundle_v2_sha256": bundle_sha}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="directory of downloaded EXP-062 unit result JSONs "
                         "({unit_id}.json, the S3 results/ prefix synced "
                         "locally)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="the downloaded EXP-062 fleet manifest.json")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--built-at", default=None,
                    help="ISO-8601 build timestamp, recorded verbatim in "
                         "manifest_v2.json (required to build)")
    ap.add_argument("--register", action="store_true",
                    help="register the EMITTED v2 bundle in MLflow (separate "
                         "step; hard-fail on error). No build is performed.")
    ap.add_argument("--mlflow-uri",
                    default=os.environ.get("MLFLOW_TRACKING_URI",
                                           "http://localhost:5001"))
    args = ap.parse_args(argv)

    try:
        if args.register:
            reg = register_in_mlflow(args.out_dir, args.mlflow_uri)
            print(f"[mlflow] run={reg['run_id']} model={reg['model_name']} "
                  f"v{reg['model_version']} alias={reg['alias']} "
                  f"bundle_v2_sha256={reg['bundle_v2_sha256']}")
            return 0

        if args.results_dir is None or args.manifest is None \
                or args.built_at is None:
            ap.error("build mode requires --results-dir, --manifest and "
                     "--built-at (or pass --register)")

        v1_authority = read_v1_authority(args.out_dir)
        exp_id, units = load_calibration_manifest(args.manifest)
        blocks = [
            load_unit_observe(args.results_dir, u, v1_authority["v1_sha"])
            for u in units
        ]
        pools = pool_cells(units, blocks)
        cells = build_cells(pools)
        source_commit = _git_head()
        bundle = write_bundle_v2(
            args.out_dir, cells=cells, exp_id=exp_id, units=units,
            v1_authority=v1_authority, built_at=args.built_at,
            source_commit=source_commit)
        print(f"[bundle_v2] {args.out_dir}")
        for token in SCENARIO_TOKENS_V2:
            for ac in ARM_CLASSES:
                c = cells[token][ac]
                print(f"  {token}/{ac}: cut={c['cut']:.6f} "
                      f"realized_fpr={c['realized_fpr']:.4f} "
                      f"n_honest={c['n_honest']}")
        report = write_report(args.out_dir, bundle=bundle,
                              built_at=args.built_at,
                              source_commit=source_commit)
        print(f"[report] {report}")
        # The sidecar line — the value that becomes the sealed fleet's pin
        # (h4_scoring_lib.SERVING_BUNDLE_SHA256, erratum B closing note).
        print(f"bundle_v2_sha256={bundle['bundle_v2_sha256']}  "
              f"manifest={args.out_dir / 'manifest_v2.json'}")
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

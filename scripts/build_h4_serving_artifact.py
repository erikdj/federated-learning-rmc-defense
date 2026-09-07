"""Build the § 7.1 H4 serving bundle — ONE frozen detector, per-scenario cuts.

Executes the frozen serving construction summarized in the H4 section of
`docs/reproduction/experiments.md` (ratified as methodology v1.51), with
nothing decided here:

  1. Rebuild the 50-cell EXP-051 + EXP-053 assembly map (download-only from
     S3, `scripts/build_exp051_assembly_map.py` precedent). ANY missing cell
     is a refusal — no partial corpus.
  2. Run the § 2.2b golden gate, then build features for the full corpus via
     the frozen golden-gated pipeline (`h2prime_corpus.load_cells`).
  3. Fit ONE GradientBoostingClassifier with the frozen hyperparameters and
     the frozen 9-feature order (both read verbatim from
     `H2PRIME_ADJUDICATION.json` `_meta` and cross-checked against the
     committed constants) on the ENTIRE corpus.
  4. Per-scenario cuts (S0–S4) on each scenario's honest rows at target FPR
     0.10 using the adjudicator's exact cut construction
     (`R.cut_from_calibration`, strict-greater flagging). There is NO global
     cut — § 7.1 names it the rejected degeneracy.
  5. Emit `data/h4_serving/{model.joblib, cuts.json, features.json,
     manifest.json}` + `BUILD_REPORT.md`; `bundle_sha256` = sha256 of the
     manifest.json bytes (printed, never written into manifest.json itself).
  6. Register the bundle in MLflow as `praxis-h2prime-detector`
     @ alias `champion__H4` with signature, artifacts and rich tags.

Usage (the executed build):
    AWS_PROFILE=research MLFLOW_TRACKING_URI=http://localhost:5001 \
    conda run -n flowerfl python scripts/build_h4_serving_artifact.py \
        --staging /path/to/staging --built-at 2026-08-17T00:00:00Z

    # local re-execution against an already-staged map. The reused map must be
    # pinned by an EXTERNAL sha256 authority (the map validates files against
    # digests recorded inside itself, so the map's own identity has to come
    # from outside it). For reproducing THE executed build, the authoritative
    # value is the committed manifest's `corpus.map_sha256`:
    # b794412a75a2bb5340d7296ec92c718f7e908d50b390badfb4f0e5ca0fef2e8f
    ... --reuse-map /path/to/assembly_map.json \
        --expect-map-sha256 b794412a75a2bb5340d7296ec92c718f7e908d50b390badfb4f0e5ca0fef2e8f \
        --skip-mlflow
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_exp051_assembly_map as ASSEMBLY_BUILDER  # noqa: E402
from h2prime_common import (  # noqa: E402
    COMPARABILITY_INTERVAL, CONFIRMATORY, FEATS, GBDT_PARAMS, PINNED_NUMPY,
    PINNED_SKLEARN, R, Refusal, SCENARIOS,
)
from h2prime_corpus import design_matrix, load_assembly_map, load_cells  # noqa: E402
from h2prime_scoring import golden_gate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ADJUDICATION_DEFAULT = (
    REPO / "reproduction" / "evidence" / "h2prime-confirmatory.json")
OUT_DIR_DEFAULT = REPO / "data" / "h4_serving"
SPEC_REF = "docs/superpowers/specs/2026-08-16-h4-composition-preregistration.md § 7.1"
MODEL_NAME = "praxis-h2prime-detector"
MODEL_ALIAS = "champion__H4"
MLFLOW_EXPERIMENT = "h4_serving_bundle"
BUNDLE_FORMAT = "h4-serving-v1"
CUT_SEMANTICS = (
    "cut = float(np.quantile(honest_scores, 1 - 0.10)) per scenario "
    "(R.cut_from_calibration, higher_is_trust=False); flag = score > cut (strict)")


@dataclass(frozen=True)
class FrozenConstruction:
    """The § 7.1 frozen inputs, read verbatim from the adjudication artifact."""

    gbdt_params: dict
    features: tuple[str, ...]


# --------------------------------------------------------------------------
# frozen inputs — verbatim from the adjudication artifact, drift-checked
# --------------------------------------------------------------------------
def load_frozen_construction(path: Path) -> FrozenConstruction:
    """Read `_meta.gbdt_params` / `_meta.features_frozen_order` and refuse on
    ANY drift against the committed constants. Two authorities must agree —
    neither is allowed to vouch for itself."""
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"adjudication artifact not found: {path}")
    try:
        meta = json.loads(path.read_text())["_meta"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise Refusal(f"adjudication artifact unreadable ({path}): {exc}") from exc
    params = meta.get("gbdt_params")
    feats = meta.get("features_frozen_order")
    if not isinstance(params, dict) or not isinstance(feats, list):
        raise Refusal(
            f"adjudication _meta lacks gbdt_params/features_frozen_order: {path}")
    if params != GBDT_PARAMS:
        drift = sorted(set(params.items()) ^ set(GBDT_PARAMS.items()))
        raise Refusal(
            "gbdt_params drift between the adjudication artifact and the "
            f"committed § 2.1a constants: {drift}")
    if list(feats) != list(FEATS):
        raise Refusal(
            "features_frozen_order drift between the adjudication artifact "
            f"and the committed frozen order: artifact={feats} committed={list(FEATS)}")
    return FrozenConstruction(gbdt_params=dict(params), features=tuple(feats))


# --------------------------------------------------------------------------
# corpus — assembly map (download-only) + golden-gated feature build
# --------------------------------------------------------------------------
def rebuild_assembly_map(staging: Path, out: Path) -> Path:
    """Rebuild the 50-cell map via the EXP-051 builder (one `aws s3 cp` per
    cell, digests recorded). A single failed cell aborts the whole build."""
    try:
        rc = ASSEMBLY_BUILDER.main(
            ["--staging", str(staging), "--out", str(out)])
    except SystemExit as exc:
        raise Refusal(
            f"assembly-map rebuild failed — no partial corpus is accepted "
            f"(§ 7.1 requires all 50 cells): {exc}") from exc
    if rc != 0:
        raise Refusal(
            f"assembly-map builder exited {rc} — no partial corpus is accepted")
    return out


def verify_reused_map(map_path: Path, expected_sha256: str) -> None:
    """Pin a REUSED assembly map to an external sha256 authority.

    `load_assembly_map` verifies each staged file against digests recorded in
    the map itself, so a caller-supplied map is self-vouching: a fabricated
    50-cell map whose digests match its own files would pass every check and
    train the detector on the wrong corpus. The fresh `--staging` path derives
    its authority from the S3 download itself; a reused map must instead match
    an EXPECTED sha256 supplied from outside the map (for the executed build,
    the committed manifest's `corpus.map_sha256`)."""
    map_path = Path(map_path)
    if not map_path.is_file():
        raise Refusal(f"reused assembly map not found: {map_path}")
    expected = str(expected_sha256).strip().lower()
    actual = _sha256_file(map_path)
    if actual != expected:
        raise Refusal(
            "reused assembly map sha256 mismatch — the map is self-vouching, "
            "so its identity must be pinned externally (--expect-map-sha256):\n"
            f"  expected {expected}\n  actual   {actual}\n"
            f"  map      {map_path}")


def load_corpus_map(map_path: Path):
    """Validate the map on the CONFIRMATORY profile: exactly 50 cells, content
    digests mandatory, sealed-seed set, registered defense token."""
    return load_assembly_map(Path(map_path), CONFIRMATORY)


def build_corpus_rows(cells) -> tuple[list[dict], dict]:
    """§ 2.2b golden gate first (hard stop on failure), then the frozen
    per-file window-feature derivation."""
    gate = golden_gate()
    rows = load_cells(cells)
    if not rows:
        raise Refusal("corpus produced zero rows")
    missing = [s for s in SCENARIOS if not any(r["_scen"] == s for r in rows)]
    if missing:
        raise Refusal(f"corpus has no rows for scenarios: {missing}")
    return rows, gate


def corpus_census(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per-scenario total/honest/malicious row counts."""
    census = {s: {"total": 0, "honest": 0, "malicious": 0} for s in SCENARIOS}
    for r in rows:
        c = census[r["_scen"]]
        c["total"] += 1
        c["malicious" if r["malicious_gt"] else "honest"] += 1
    return census


# --------------------------------------------------------------------------
# model — ONE classifier on the entire corpus
# --------------------------------------------------------------------------
def fit_serving_model(rows: list[dict],
                      construction: FrozenConstruction) -> tuple:
    """Fit ONE GBDT (frozen hparams, frozen 9-column order) on ALL corpus
    rows, y = malicious_gt. The refit-on-all-adjudicated-data deployment
    model, § 7.1 verbatim."""
    X = design_matrix(rows)
    y = np.array([bool(r["malicious_gt"]) for r in rows])
    if len(np.unique(y)) < 2:
        raise Refusal(
            f"corpus contains a single class ({int(y.sum())} positive of "
            f"{len(y)} rows); a detector cannot be fit")
    clf = GradientBoostingClassifier(**construction.gbdt_params).fit(X, y)
    meta = {
        "n_rows": len(rows),
        "n_positive": int(y.sum()),
        "n_honest": int((~y).sum()),
        "n_features": X.shape[1],
        "classifier": type(clf).__name__,
    }
    return clf, meta


def serving_cut(honest_scores) -> float:
    """The adjudicator's exact § 2.2a cut at the frozen target FPR 0.10 —
    delegated, not transcribed, so the semantics cannot drift."""
    return R.cut_from_calibration(honest_scores, higher_is_trust=False)


def realized_fpr(honest_scores, cut: float) -> float:
    """Fraction of honest rows flagged under the adjudicator's strict-greater
    flagging."""
    return float(R.flagged(honest_scores, cut, False).mean())


def per_scenario_cuts(rows: list[dict], scores: np.ndarray) -> dict[str, dict]:
    """One cut per scenario on THAT scenario's honest rows (§ 7.1). Refuses
    on a missing scenario, an empty honest population, or a realized corpus
    FPR outside the closed comparability band [0.08, 0.12]."""
    if len(rows) != len(scores):
        raise Refusal(
            f"rows/scores misaligned: {len(rows)} rows vs {len(scores)} scores")
    lo, hi = COMPARABILITY_INTERVAL
    out: dict[str, dict] = {}
    for scen in SCENARIOS:
        honest = np.array([s for r, s in zip(rows, scores)
                           if r["_scen"] == scen and not r["malicious_gt"]],
                          dtype=float)
        if honest.size == 0:
            raise Refusal(f"no honest rows for scenario {scen}; no cut can be set")
        cut = serving_cut(honest)
        fpr = realized_fpr(honest, cut)
        if not (lo <= fpr <= hi):
            raise Refusal(
                f"realized corpus FPR for {scen} is {fpr:.4f} at cut "
                f"{cut:.6f}, outside the closed band [{lo:.2f}, {hi:.2f}] — "
                "the bundle is not emitted (§ 7.1 construction check)")
        out[scen] = {"cut": float(cut), "realized_fpr": fpr,
                     "n_honest": int(honest.size)}
    return out


# --------------------------------------------------------------------------
# bundle emission — deterministic manifest, bundle sha over its bytes
# --------------------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bundle(out_dir: Path, *, clf, scenario_cuts: dict,
                 construction: FrozenConstruction, corpus_meta: dict,
                 fit_meta: dict, built_at: str, source_commit: str) -> dict:
    """Emit the BUILD_CONTRACT bundle. `manifest.json` carries per-file
    sha256, corpus provenance, the frozen echoes, source commit and built_at;
    the bundle sha (sha256 of the manifest bytes) is RETURNED, never written
    into the manifest it hashes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(clf, model_path)
    cuts_doc = {s: scenario_cuts[s]["cut"] for s in SCENARIOS}
    (out_dir / "cuts.json").write_text(
        json.dumps(cuts_doc, sort_keys=True, indent=1) + "\n")
    (out_dir / "features.json").write_text(
        json.dumps(list(construction.features), indent=1) + "\n")

    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "construction": SPEC_REF,
        "built_at": built_at,
        "source_commit": source_commit,
        "files": {fn: _sha256_file(out_dir / fn)
                  for fn in ("model.joblib", "cuts.json", "features.json")},
        "corpus": corpus_meta,
        "gbdt_params": construction.gbdt_params,
        "features_frozen_order": list(construction.features),
        "fit": fit_meta,
        "target_fpr": R.TARGET_FPR,
        "fpr_band": list(COMPARABILITY_INTERVAL),
        "cut_semantics": CUT_SEMANTICS,
        "cuts": scenario_cuts,
        "model_registration": {"name": MODEL_NAME, "alias": MODEL_ALIAS},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numpy_pinned": PINNED_NUMPY,
            "sklearn": __import__("sklearn").__version__,
            "sklearn_pinned": PINNED_SKLEARN,
            "joblib": joblib.__version__,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    return {
        "bundle_sha256": _sha256_bytes(manifest_bytes),
        "manifest_path": str(manifest_path),
        "files": dict(manifest["files"]),
    }


def write_report(out_dir: Path, *, bundle: dict, scenario_cuts: dict,
                 corpus_meta: dict, fit_meta: dict, built_at: str,
                 source_commit: str, registration: dict | None) -> Path:
    """Small human-facing BUILD_REPORT.md — counts, cuts, realized FPRs,
    training metadata and the bundle sha sidecar line. `registration` is the
    ACTUAL MLflow outcome (written after that step ran), never a claim: None
    means registration was skipped and the report says so."""
    census = corpus_meta["row_census"]
    lines = [
        "# H4 serving bundle — build report",
        "",
        f"Construction: {SPEC_REF} (RATIFIED, methodology v1.51)",
        f"Built at: {built_at}  |  source commit: `{source_commit}`",
        f"**bundle_sha256 (sha256 of manifest.json): `{bundle['bundle_sha256']}`**",
        "",
        "## Corpus (EXP-051 + EXP-053, 50 cells, golden-gated features)",
        "",
        f"- assembly map sha256: `{corpus_meta['map_sha256']}`",
        f"- cells by source: {corpus_meta['cells_by_source']}",
        f"- rows: {fit_meta['n_rows']} total "
        f"({fit_meta['n_honest']} honest / {fit_meta['n_positive']} malicious)",
        "",
        "| scenario | total | honest | malicious | cut | realized FPR |",
        "|---|---|---|---|---|---|",
    ]
    for s in SCENARIOS:
        c, sc = census[s], scenario_cuts[s]
        lines.append(
            f"| {s} | {c['total']} | {c['honest']} | {c['malicious']} | "
            f"{sc['cut']:.6f} | {sc['realized_fpr']:.4f} |")
    lines += [
        "",
        "## Model",
        "",
        f"- ONE {fit_meta['classifier']} (frozen § 2.1a hparams, "
        f"{fit_meta['n_features']}-feature frozen order), fit on ALL corpus rows",
        f"- cut semantics: {CUT_SEMANTICS}",
    ]
    if registration is not None:
        lines.append(
            f"- MLflow registration: run `{registration['run_id']}`, "
            f"`{registration['model_name']}` v{registration['model_version']} "
            f"@ alias `{registration['alias']}`")
    else:
        lines.append(
            "- MLflow registration: SKIPPED (--skip-mlflow) — bundle is "
            "offline-only")
    lines.append("")
    path = Path(out_dir) / "BUILD_REPORT.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# MLflow registration — the review surface
# --------------------------------------------------------------------------
def register_in_mlflow(*, clf, rows_sample_X: np.ndarray, bundle: dict,
                       scenario_cuts: dict, corpus_meta: dict, fit_meta: dict,
                       out_dir: Path, built_at: str, source_commit: str,
                       tracking_uri: str) -> dict:
    """Register the bundle: sklearn flavor + signature, artifacts, rich tags,
    alias `champion__H4` on the new version. Any failure is a hard error —
    the executed build must land on the review surface."""
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    signature = infer_signature(rows_sample_X, clf.predict_proba(rows_sample_X))
    description = (
        f"H4 serving detector — the § 7.1 pre-registered construction "
        f"({SPEC_REF}): ONE GBDT with the frozen H2′ § 2.1a hyperparameters "
        f"and 9-feature order, refit on the ENTIRE exposed H2′ confirmatory "
        f"corpus (EXP-051+EXP-053, 50 cells, {fit_meta['n_rows']} rows), with "
        f"per-scenario cuts at target FPR 0.10. bundle_sha256="
        f"{bundle['bundle_sha256']}. No H4 output ever touches the model or "
        f"the cuts.")

    with mlflow.start_run(run_name=f"h4-serving-bundle__{built_at}") as run:
        mlflow.set_tags({
            "mlflow.note.content": description,
            "spec": SPEC_REF,
            "bundle_sha256": bundle["bundle_sha256"],
            "source_commit": source_commit,
            "built_at": built_at,
            "corpus_map_sha256": corpus_meta["map_sha256"],
            "corpus_cells_by_source": json.dumps(corpus_meta["cells_by_source"]),
            "corpus_n_cells": str(corpus_meta["n_cells"]),
            "corpus_n_rows": str(fit_meta["n_rows"]),
            "s3_corpus_bucket": corpus_meta.get("bucket", ""),
        })
        mlflow.log_params({f"gbdt.{k}": v for k, v in GBDT_PARAMS.items()})
        mlflow.log_param("features_frozen_order", ",".join(FEATS))
        mlflow.log_param("target_fpr", R.TARGET_FPR)
        for s in SCENARIOS:
            mlflow.log_metric(f"cut_{s}", scenario_cuts[s]["cut"])
            mlflow.log_metric(f"realized_fpr_{s}", scenario_cuts[s]["realized_fpr"])
            mlflow.log_metric(f"n_honest_{s}", scenario_cuts[s]["n_honest"])
        mlflow.log_metric("n_rows", fit_meta["n_rows"])
        mlflow.log_metric("n_malicious_rows", fit_meta["n_positive"])
        # BUILD_REPORT.md is NOT logged here: it is written AFTER this step so
        # it can state the actual registration outcome; main() attaches it to
        # this run once it exists.
        for fn in ("cuts.json", "manifest.json", "features.json"):
            mlflow.log_artifact(str(Path(out_dir) / fn), artifact_path="bundle")
        try:
            info = mlflow.sklearn.log_model(
                clf, name="model", registered_model_name=MODEL_NAME,
                signature=signature, input_example=rows_sample_X)
        except TypeError:  # 2.x keyword rename tolerance (entrypoint precedent)
            info = mlflow.sklearn.log_model(
                clf, artifact_path="model", registered_model_name=MODEL_NAME,
                signature=signature, input_example=rows_sample_X)

    client = MlflowClient(tracking_uri=tracking_uri)
    version = str(getattr(info, "registered_model_version", "") or "")
    if not version:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        ours = [v for v in versions if v.run_id == run.info.run_id]
        if not ours:
            raise Refusal(
                f"model registration produced no version for run {run.info.run_id}")
        version = str(max(int(v.version) for v in ours))
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    client.update_registered_model(MODEL_NAME, description=description)
    client.update_model_version(MODEL_NAME, version, description=description)
    for k, v in (("bundle_sha256", bundle["bundle_sha256"]),
                 ("source_commit", source_commit),
                 ("spec", SPEC_REF)):
        client.set_model_version_tag(MODEL_NAME, version, k, v)
    return {"run_id": run.info.run_id, "model_name": MODEL_NAME,
            "model_version": version, "alias": MODEL_ALIAS}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staging", type=Path, default=None,
                    help="directory to stage the 50 signal logs into (downloads)")
    ap.add_argument("--reuse-map", type=Path, default=None,
                    help="existing assembly map JSON (skips the download step); "
                         "requires --expect-map-sha256")
    ap.add_argument("--expect-map-sha256", default=None,
                    help="external sha256 authority for --reuse-map (the "
                         "committed manifest's corpus.map_sha256 for the "
                         "executed build); the map is self-vouching without it")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--adjudication", type=Path, default=ADJUDICATION_DEFAULT)
    ap.add_argument("--built-at", required=True,
                    help="ISO-8601 build timestamp, recorded verbatim in the manifest")
    ap.add_argument("--skip-mlflow", action="store_true",
                    help="emit the bundle only; no MLflow registration")
    ap.add_argument("--mlflow-uri",
                    default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    args = ap.parse_args(argv)

    if (args.reuse_map is None) == (args.staging is None):
        ap.error("exactly one of --staging (fresh download) or --reuse-map is required")
    if args.reuse_map is not None and args.expect_map_sha256 is None:
        ap.error("--reuse-map requires --expect-map-sha256 (the map verifies "
                 "files against its own recorded digests, so its identity must "
                 "be pinned by an external authority)")

    try:
        if args.reuse_map is not None:
            verify_reused_map(args.reuse_map, args.expect_map_sha256)
            map_path = args.reuse_map
        else:
            map_path = rebuild_assembly_map(
                args.staging, args.staging / "assembly_map.json")

        cells, map_meta = load_corpus_map(map_path)
        rows, gate = build_corpus_rows(cells)
        construction = load_frozen_construction(args.adjudication)
        clf, fit_meta = fit_serving_model(rows, construction)
        scores = clf.predict_proba(design_matrix(rows))[:, 1]
        scenario_cuts = per_scenario_cuts(rows, scores)

        import subprocess
        source_commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()

        corpus_meta = {
            "map_path": str(map_path),
            "map_sha256": map_meta["map_sha256"],
            "n_cells": len(cells),
            "cells_by_source": map_meta["cells_by_source"],
            "seeds_ascending": map_meta["seeds_ascending"],
            "defense_token": map_meta["defense_token"],
            "bucket": map_meta.get("map_meta", {}).get("bucket", ""),
            "n_rows": fit_meta["n_rows"],
            "row_census": corpus_census(rows),
            "golden_gate": gate["status"],
        }
        bundle = write_bundle(
            args.out_dir, clf=clf, scenario_cuts=scenario_cuts,
            construction=construction, corpus_meta=corpus_meta,
            fit_meta=fit_meta, built_at=args.built_at,
            source_commit=source_commit)
        print(f"[bundle] {args.out_dir}")
        for s in SCENARIOS:
            sc = scenario_cuts[s]
            print(f"  {s}: cut={sc['cut']:.6f} realized_fpr={sc['realized_fpr']:.4f} "
                  f"n_honest={sc['n_honest']}")
        # The sidecar line — the single value that rides in custody exports.
        print(f"bundle_sha256={bundle['bundle_sha256']}  manifest={bundle['manifest_path']}")

        reg = None
        if not args.skip_mlflow:
            X_sample = design_matrix(rows[:2])
            reg = register_in_mlflow(
                clf=clf, rows_sample_X=X_sample, bundle=bundle,
                scenario_cuts=scenario_cuts, corpus_meta=corpus_meta,
                fit_meta=fit_meta, out_dir=args.out_dir,
                built_at=args.built_at, source_commit=source_commit,
                tracking_uri=args.mlflow_uri)
            print(f"[mlflow] run={reg['run_id']} model={reg['model_name']} "
                  f"v{reg['model_version']} alias={reg['alias']}")

        # The report is written AFTER the MLflow step so it records the ACTUAL
        # registration outcome (BUILD_REPORT.md is not in the hashed file set,
        # so the bundle_sha256 is unaffected by this ordering).
        report_path = write_report(
            args.out_dir, bundle=bundle, scenario_cuts=scenario_cuts,
            corpus_meta=corpus_meta, fit_meta=fit_meta,
            built_at=args.built_at, source_commit=source_commit,
            registration=reg)
        if reg is not None:
            from mlflow.tracking import MlflowClient
            MlflowClient(tracking_uri=args.mlflow_uri).log_artifact(
                reg["run_id"], str(report_path), "bundle")
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

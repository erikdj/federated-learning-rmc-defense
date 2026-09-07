"""H1 signal-family detector retrain on EXP-014 ramp-3 closed-loop logs (v1.6 §4).

Chain of custody (amendment v1.6 §3/§4; EXP-014 doc; base spec §6.1):
- The v1.6 §3 frozen ramp-selection rule SELECTED r*=3 (open-loop, on the locked
  ramp-8 dev logs). Per §4 (pre-committed, not optional): "H1 detectors train on
  the re-run deployed-config logs." The TGE-family dev cells were RE-RUN with r*=3
  deployed live (closed-loop) = EXP-014. This script performs the pre-committed
  §4 H1 retrain, changing ONLY the input logs (ramp-8 -> EXP-014 ramp-3 closed-
  loop), exactly as the TGE-config threshold re-freeze did (analyze_exp014_closed_loop.py).

- H1 SIGNAL SOURCE = the single named DEPLOYED config Krum+TGE ONLY (base spec §6.1
  "signals from H2's matrix"; h1_signal_family_eval.py docstring F1/F5: "single named
  deployed config (Krum+TGE) only"). Krum+TGE IS ramp-dependent, so its logs are the
  EXP-014 ramp-3 closed-loop Krum+TGE cells (25 = S0-S4 × 5 dev seeds). Krum and
  TrustScore have NO ramp dependence and are never consumed by H1 — no sync needed.

- MODEL / FEATURES / WINDOW reuse the committed H1 pipeline EXACTLY (no re-derivation):
  families S/W/C, sklearn GradientBoostingClassifier(random_state=0), k=3 cold-start
  window, recall@10%FPR via compute_recall_fpr.filter_scope + h2_threshold_pipeline.
  select_threshold. All feature/GBDT/recall functions are imported from
  scripts/h1_signal_family_eval.py; feature math lives in flowerfl.cold_start_detector.

- DEV-STAGE READ ONLY. The committed h1_signal_family_eval.main() evaluates the
  detectors OUT-OF-SAMPLE on CONFIRMATORY-seed logs (F1). Confirmatory has not run
  and its 10 seeds REMAIN SEALED; the pre-registered out-of-sample F1 eval is a
  confirmatory-phase activity (base spec §6.1 signal source = "H2's 150-run matrix";
  design §220/§255 place H1 evaluation after Phase A). What §3.1 (v1.6) pre-commits
  AT THE DEV GATE is H1 detector TRAINING. This script therefore (1) TRAINS+FREEZES
  the S/W/C detectors on the ramp-3 closed-loop Krum+TGE dev logs (the §4.2 artifact),
  and (2) produces a DEV-STAGE leave-one-seed-out (LOSO) cross-validation read for a
  family-composition sanity check (C vs S, C vs W) WITHOUT touching confirmatory.
  Each LOSO fold is the committed h1_signal_family_eval code path run with dev=4 dev
  seeds / eval=1 held-out dev seed. LOSO is the repo's established dev-stage detector
  evaluation (scripts/train_cold_start_detector.py) and reuses the n=5 dev evidential
  floor; it is NOT the pre-registered F1 statistic and makes no significance claim.

- EXTRACTION NOTE (reported, not altered): the committed extract_family_features
  groups rows by logical_cid and keeps each identity's first k=3 OBSERVED rounds, so
  the pooled dev set collapses to a small cold-start matrix (~177 rows: 135 malicious
  / 42 honest for the full 5-seed pool). This is a property of the committed extractor;
  per the "swap inputs only" mandate it is preserved, not fixed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from h1_signal_family_eval import (  # noqa: E402
    _compute_recall_at_fpr,
    _load_jsonl_glob,
    _train_gbdt,
    extract_family_features,
)
from flowerfl.cold_start_detector import Family, feature_names  # noqa: E402

# --- pre-registered scope (do not edit without an amendment) ----------------
DEFENSE = "krum_tge"          # H1 single named deployed config (F5)
ROW_DEFENSE_TOKEN = "krumtge"  # signal-row `defense` field for Krum+TGE
SCENARIOS = ["s0_clean_baseline", "s1_benign_churn_only", "s2_adaptive_switching_only",
             "s3_identity_reset_only", "s4_full_mix"]
SCEN_SHORT = {s: f"S{i}" for i, s in enumerate(SCENARIOS)}
DEV_SEEDS = [42, 137, 256, 314, 500]
MODE = "persistent_optimizer"
FAMILIES = ["S", "W", "C"]
K = 3
TARGET_FPR = 0.10


DEFAULT_READ_LABEL = ("H1 signal-family detector retrain — v1.6 §4 contingency "
                      "(ramp-3 closed-loop)")
DEFAULT_AUTHORITY = ("v1.6 §4 (H1 detectors train on the re-run deployed-config logs)",
                     "base spec §6.1 (families S/W/C, GBDT, k=3, Krum+TGE signal source)",
                     "h1_signal_family_eval.py (committed pipeline; F1/F5)")
DEFAULT_INPUT_SCOPE = ("Krum+TGE ONLY (single named deployed config, F5); ramp-dependent, "
                       "so retrained on EXP-014 ramp-3 closed-loop cells. Krum/TrustScore "
                       "have no ramp dependence and are never consumed by H1 — no sync.")
DEFAULT_TRAINED_ON = ("EXP-014 ramp-3 closed-loop Krum+TGE dev "
                      "(S0-S4 × seeds 42/137/256/314/500)")
DEFAULT_PROVENANCE = ("v1.6 §4 contingency H1 retrain; ramp_rounds=3; "
                      "input logs = results/20260724/exp014_closed_loop/signals")
DEFAULT_PRIOR_RECORD = ("NONE — no prior h1_signal_family_eval output exists in-repo; "
                        "this is the first H1 detector training, at the dev gate on "
                        "ramp-3 per §3.1 order-of-operations")


class H1RetrainError(RuntimeError):
    """Raised when EXP-014 inputs violate an H1-retrain precondition."""


def parse_stem(stem: str) -> tuple[str, str, int]:
    scenario, defense, mode, seed_part = stem.rsplit("__", 3)
    if mode != MODE or not seed_part.startswith("seed"):
        raise H1RetrainError(f"unexpected unit stem: {stem}")
    return scenario, defense, int(seed_part[len("seed"):])


def canonical_stem(scenario: str) -> str:
    """'s4_full_mix' -> 'S4_full_mix' (matches the signal-row `scenario` field)."""
    return scenario[0].upper() + scenario[1:]


def enumerate_units(signals_dir: Path) -> dict[int, Path]:
    """Integrity gate: exactly 25 Krum+TGE files, exact S0-S4 × 5-seed cross-product.
    Returns {seed: {scenario: path}} flattened as {(scenario, seed): path}."""
    files = sorted(signals_dir.glob(f"*__{DEFENSE}__*.jsonl"))
    units: dict[tuple[str, int], Path] = {}
    for f in files:
        scenario, defense, seed = parse_stem(f.stem)
        if defense != DEFENSE:
            continue
        key = (scenario, seed)
        if key in units:
            raise H1RetrainError(f"duplicate cell {key}: {units[key]} and {f}")
        units[key] = f
    expected = {(s, sd) for s in SCENARIOS for sd in DEV_SEEDS}
    got = set(units)
    if got != expected:
        raise H1RetrainError(
            f"matrix incomplete: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    if len(units) != 25:
        raise H1RetrainError(f"expected exactly 25 Krum+TGE units, got {len(units)}")
    return units


def load_seed_rows(units: dict[tuple[str, int], Path], seed: int) -> list[dict]:
    """Pool one dev seed's Krum+TGE rows (all scenarios) with strict per-row custody:
    every row must carry the registered scenario stem, the declared seed, and the
    Krum+TGE defense token. Mirrors analyze_exp014_closed_loop.load_unit_rows."""
    rows: list[dict] = []
    for scenario in SCENARIOS:
        path = units[(scenario, seed)]
        exp_stem = canonical_stem(scenario)
        unit_rows = _load_jsonl_glob(str(path))
        if not unit_rows:
            raise H1RetrainError(f"{path.name}: empty signal log")
        for j, r in enumerate(unit_rows):
            if r.get("scenario") != exp_stem:
                raise H1RetrainError(
                    f"{path.name} row {j}: scenario {r.get('scenario')!r} != {exp_stem!r}")
            if r.get("seed") is None or int(r["seed"]) != seed:
                raise H1RetrainError(
                    f"{path.name} row {j}: seed {r.get('seed')!r} != {seed}")
            if r.get("defense") != ROW_DEFENSE_TOKEN:
                raise H1RetrainError(
                    f"{path.name} row {j}: defense {r.get('defense')!r} != {ROW_DEFENSE_TOKEN!r}")
        rows.extend(unit_rows)
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def loso_read(rows_by_seed: dict[int, list[dict]], family: str) -> dict:
    """Leave-one-seed-out CV: train on 4 dev seeds, eval recall@10%FPR on the held-out
    dev seed, using the committed h1_signal_family_eval code path per fold."""
    folds = []
    for held in DEV_SEEDS:
        train_rows = [r for s in DEV_SEEDS if s != held for r in rows_by_seed[s]]
        eval_rows = rows_by_seed[held]
        X_tr, y_tr = extract_family_features(train_rows, family, K)
        if X_tr.shape[0] == 0 or y_tr.sum() < 1 or (1 - y_tr).sum() < 1:
            folds.append({"held_out_seed": held, "recall_at_fpr": None,
                          "n_train": int(X_tr.shape[0]), "note": "degenerate train fold"})
            continue
        clf = _train_gbdt(X_tr, y_tr)
        res = _compute_recall_at_fpr(clf, eval_rows, family, K, TARGET_FPR)
        res["held_out_seed"] = held
        res["n_train"] = int(X_tr.shape[0])
        res["n_train_pos"] = int(y_tr.sum())
        folds.append(res)
    recalls = [f["recall_at_fpr"] for f in folds if f.get("recall_at_fpr") is not None]
    return {
        "folds": folds,
        "mean_recall_at_10pct_fpr": fmean(recalls) if recalls else None,
        "std_recall_at_10pct_fpr": pstdev(recalls) if len(recalls) > 1 else 0.0,
        "n_folds_scored": len(recalls),
    }


def train_final(all_rows: list[dict], family: str, models_dir: Path, commit: str,
                trained_on: str, provenance: str) -> dict:
    """Train the frozen detector on ALL 5 dev seeds (the §4.2 artifact) and persist it.

    `trained_on` / `provenance` are supplied by the caller so the persisted artifact
    states its OWN input ladder. The fitting itself (features, GBDT, k, recall) is
    input-independent and unchanged.
    """
    X, y = extract_family_features(all_rows, family, K)
    clf = _train_gbdt(X, y)
    importances = sorted(
        zip(feature_names(Family(family)), (float(v) for v in clf.feature_importances_)),
        key=lambda kv: kv[1], reverse=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = models_dir / f"{family}_k{K}_final.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump({
            "model": clf,
            "family": family,
            "k": K,
            "feature_names": feature_names(Family(family)),
            "target_fpr": TARGET_FPR,
            "trained_on": trained_on,
            "provenance": provenance,
            "git_commit": commit,
            "n_train": int(X.shape[0]),
            "n_train_pos": int(y.sum()),
        }, fh)
    return {
        "final_model_path": str(out_pkl.relative_to(REPO)),
        "n_train": int(X.shape[0]),
        "n_train_pos": int(y.sum()),
        "n_train_neg": int((y == 0).sum()),
        "feature_importance": [{"feature": n, "importance": v} for n, v in importances],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals-dir", type=Path,
                    default=REPO / "results/20260724/exp014_closed_loop/signals")
    ap.add_argument("--models-dir", type=Path,
                    default=REPO / "models/h1_signal_family/ramp3_closed_loop")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/20260724/h1_retrain_ramp3/h1_retrain_ramp3_dev_read.json")
    # Provenance strings describe the INPUT LADDER only; defaults reproduce the
    # 2026-07-24 v1.6 §4 ramp-3 read byte-identically. The fitting is unchanged.
    ap.add_argument("--read-label", default=DEFAULT_READ_LABEL)
    ap.add_argument("--authority", action="append", default=None)
    ap.add_argument("--input-scope", default=DEFAULT_INPUT_SCOPE)
    ap.add_argument("--trained-on", default=DEFAULT_TRAINED_ON)
    ap.add_argument("--provenance", default=DEFAULT_PROVENANCE)
    ap.add_argument("--prior-record", default=DEFAULT_PRIOR_RECORD)
    args = ap.parse_args()
    authority = args.authority or list(DEFAULT_AUTHORITY)

    commit = git_commit()

    # --- (1) integrity gate + custody -------------------------------------
    units = enumerate_units(args.signals_dir)
    rows_by_seed = {s: load_seed_rows(units, s) for s in DEV_SEEDS}
    all_rows = [r for s in DEV_SEEDS for r in rows_by_seed[s]]
    integrity = {
        "n_files": len(units),
        "defense": DEFENSE, "row_defense_token": ROW_DEFENSE_TOKEN,
        "cross_product_ok": True,
        "scenarios": [SCEN_SHORT[s] for s in SCENARIOS], "seeds": DEV_SEEDS,
        "raw_rows_per_seed": {str(s): len(rows_by_seed[s]) for s in DEV_SEEDS},
        "total_raw_rows": len(all_rows),
    }

    # --- (2) frozen final detectors (train on all 5 dev seeds) + LOSO read --
    final = {}
    loso = {}
    for fam in FAMILIES:
        final[fam] = train_final(all_rows, fam, args.models_dir, commit,
                                 args.trained_on, args.provenance)
        loso[fam] = loso_read(rows_by_seed, fam)

    # cold-start matrix size (post-extraction), reported once from family C
    Xc, yc = extract_family_features(all_rows, "C", K)
    coldstart = {"n_rows": int(Xc.shape[0]),
                 "n_malicious": int(yc.sum()), "n_honest": int((yc == 0).sum())}

    # --- (3) dev-stage family-composition sanity verdict (NOT adjudicated) --
    mC = loso["C"]["mean_recall_at_10pct_fpr"]
    mS = loso["S"]["mean_recall_at_10pct_fpr"]
    mW = loso["W"]["mean_recall_at_10pct_fpr"]
    verdict = None
    if None not in (mC, mS, mW):
        verdict = {
            "C_ge_S": mC >= mS, "C_ge_W": mC >= mW,
            "C_gt_both": (mC > mS) and (mC > mW),
            "margins": {"C_minus_S": mC - mS, "C_minus_W": mC - mW},
            "note": "DEV-STAGE LOSO sanity only; NOT the pre-registered out-of-sample "
                    "F1 statistic (that is a confirmatory-phase eval, seeds sealed).",
        }

    out = {
        "_meta": {
            "read": args.read_label,
            "authority": authority,
            "input_scope": args.input_scope,
            "signals_dir": str(args.signals_dir),
            "model": "sklearn GradientBoostingClassifier(random_state=0) per family (committed)",
            "k": K, "target_fpr": TARGET_FPR,
            "dev_stage_only": ("out-of-sample F1 eval on confirmatory seeds DEFERRED (sealed); "
                               "LOSO CV is a dev-stage family-composition sanity read, repo "
                               "precedent train_cold_start_detector.py; no significance claim"),
            "extraction_note": ("committed extract_family_features groups by logical_cid and keeps "
                                "first k=3 observed rounds -> small cold-start matrix; preserved, "
                                "not altered (swap-inputs-only mandate)"),
            "prior_h1_record": args.prior_record,
            "git_commit": commit,
        },
        "integrity_gate": integrity,
        "coldstart_matrix": coldstart,
        "final_detectors": final,
        "loso_dev_read": loso,
        "family_composition_verdict_devstage": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False))

    # --- console summary ---------------------------------------------------
    print(f"== {args.read_label} ==")
    print(f"  integrity: {integrity['n_files']} files, cross-product OK, "
          f"cold-start matrix {coldstart['n_rows']} rows "
          f"({coldstart['n_malicious']} mal / {coldstart['n_honest']} honest)")
    print("  LOSO dev recall@10%FPR (mean ± std over 5 held-out dev seeds):")
    for fam in FAMILIES:
        L = loso[fam]
        m = L["mean_recall_at_10pct_fpr"]
        s = L["std_recall_at_10pct_fpr"]
        ms = f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"
        print(f"    {fam}: {ms}  (folds scored {L['n_folds_scored']}/5)  "
              f"-> {final[fam]['final_model_path']}")
    if verdict:
        print(f"  dev-stage family check: C>=S {verdict['C_ge_S']}, C>=W {verdict['C_ge_W']} "
              f"(ΔC-S {verdict['margins']['C_minus_S']:+.4f}, "
              f"ΔC-W {verdict['margins']['C_minus_W']:+.4f}) — NOT the F1 statistic")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

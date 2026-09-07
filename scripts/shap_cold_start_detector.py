"""Track 2 (Phase 4 SHAP) — TreeSHAP on the cold-start detector.

Loads the trained detector at models/cold_start/flower_reset/{S,W,C}_k3_final.pkl
and runs TreeSHAP on held-out signal data to answer:

1. Is `num_examples` actually the SHAP-dominant feature, or is XGBoost gain
   misleading? (gain showed: num_examples=309, update_norm=62, cos_to_median=3.7)
2. Does cos_to_median cleanly separate ALIE from honest, even if its gain is low?
3. Cross-mode check: if we filter to persistent-opt-only signals (Phase 3c real
   ALIE data), does num_examples still dominate? If the dominance disappears,
   it was a data-source artifact (Flower scenario logs vs sweep logs have
   different num_examples distributions).

Output:
    results/20260427/phase3c_shap_analysis.json
    results/20260427/phase3c_shap_summary_{family}.png  (if matplotlib)

Usage:
    conda run -n flowerfl python scripts/shap_cold_start_detector.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.cold_start_detector import (  # noqa: E402
    ColdStartDetector,
    Family,
    FeatureExtractor,
    S_FEATURES,
    W_FEATURES,
    C_FEATURES,
    feature_names,
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_signal_logs(mode: str, sweep_zmax: str = "0.5") -> list[tuple[str, str]]:
    """Return list of (path, source_tag) tuples for held-out signal sources.

    source_tag is one of: 'flower_scenario', 'persistent_dynamic', 'sweep'.
    """
    out = []
    # Flower control scenarios
    for f in sorted(glob.glob(str(PROJECT_ROOT / "signals" / f"{mode}__*__none__seed*.jsonl"))):
        out.append((f, "flower_scenario"))
    # Persistent-opt dynamic RMC (old fake ALIE)
    for f in sorted(glob.glob(str(PROJECT_ROOT / "signals"
                                   / "persistent_optimizer__szelag_dynamic_rmc__*__seed*.jsonl"))):
        out.append((f, "persistent_dynamic"))
    # Real ALIE sweep
    if sweep_zmax:
        for f in sorted(glob.glob(str(PROJECT_ROOT / "signals" / f"sweep_zmax{sweep_zmax}__seed*.jsonl"))):
            out.append((f, "sweep"))
    return out


def build_dataset(
    paths: list[tuple[str, str]],
    family: Family = Family.S,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract cold-start examples (first k rounds per client per run).

    Returns:
        X: (n, d) feature matrix
        y: (n,) binary labels
        attack_types: per-example attack_type strings
        sources: per-example source_tag strings (flower_scenario | persistent_dynamic | sweep)
    """
    extractor = FeatureExtractor()
    all_X: list[np.ndarray] = []
    all_y: list[float] = []
    all_atk: list[str] = []
    all_src: list[str] = []

    for path, source_tag in paths:
        rows = load_jsonl(path)
        if not rows:
            continue
        # Group by client
        by_cid: dict[str, list[dict]] = {}
        for r in rows:
            cid = r.get("logical_cid", "")
            by_cid.setdefault(cid, []).append(r)
        for cid in by_cid:
            by_cid[cid].sort(key=lambda r: r.get("scenario_round", 0))

        for cid, hist in by_cid.items():
            for i, row in enumerate(hist[:k]):
                if family == Family.S:
                    feat = extractor.extract_s(row)
                elif family == Family.W:
                    feat = extractor.extract_w(hist[:i + 1], k)
                else:
                    feat = extractor.extract_c(row, hist[:i + 1], k)
                label = 1.0 if row.get("malicious_gt", False) else 0.0
                all_X.append(feat)
                all_y.append(label)
                all_atk.append(str(row.get("attack_type") or "honest"))
                all_src.append(source_tag)

    if not all_X:
        return (np.empty((0, len(feature_names(family)))),
                np.empty(0), [], [])

    return (np.stack(all_X).astype(np.float32),
            np.asarray(all_y, dtype=np.float32),
            all_atk,
            all_src)


def permutation_importance(
    detector: ColdStartDetector,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Compute permutation importance: drop in PR-AUC when feature shuffled.

    Higher = more important. Computed per subset, so reflects this data's
    actual feature dependencies (unlike global gain).
    """
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base_risk = detector.predict_risk(X)
    if y.sum() == 0 or y.sum() == len(y):
        # Single class — can't compute PR-AUC
        return np.zeros(X.shape[1])
    base_score = float(average_precision_score(y, base_risk))

    importances = np.zeros(X.shape[1])
    n = X.shape[0]
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            perm = rng.permutation(n)
            X_shuffled[:, j] = X_shuffled[perm, j]
            risk_s = detector.predict_risk(X_shuffled)
            score_s = float(average_precision_score(y, risk_s))
            drops.append(base_score - score_s)
        importances[j] = float(np.mean(drops))
    return importances


def marginal_class_stats(
    X: np.ndarray,
    attack_types: list[str],
    feats: list[str],
) -> dict:
    """Per-feature marginal stats split by attack type.

    For each feature, compute mean and std for honest vs ALIE vs gaussian.
    Cohen's d (ALIE vs honest) tells us if the feature itself separates
    classes — independent of the model.
    """
    out: dict = {}
    masks = {
        "honest": np.array([t == "honest" for t in attack_types]),
        "alie": np.array([t == "alie" for t in attack_types]),
        "gaussian_noise": np.array([t == "gaussian_noise" for t in attack_types]),
    }
    for i, fname in enumerate(feats):
        row: dict = {}
        col = X[:, i]
        for class_name, mask in masks.items():
            if mask.sum() == 0:
                row[class_name] = {"n": 0, "mean": 0.0, "std": 0.0}
                continue
            vals = col[mask]
            row[class_name] = {
                "n": int(mask.sum()),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        # Cohen's d for ALIE vs honest
        if row["alie"]["n"] > 0 and row["honest"]["n"] > 0:
            pooled_var = (
                ((row["alie"]["n"] - 1) * row["alie"]["std"] ** 2 +
                 (row["honest"]["n"] - 1) * row["honest"]["std"] ** 2) /
                max(row["alie"]["n"] + row["honest"]["n"] - 2, 1)
            )
            pooled_sd = float(np.sqrt(max(pooled_var, 1e-12)))
            row["cohens_d_alie_vs_honest"] = (
                (row["alie"]["mean"] - row["honest"]["mean"]) / pooled_sd
                if pooled_sd > 1e-12 else 0.0
            )
        else:
            row["cohens_d_alie_vs_honest"] = 0.0
        out[fname] = row
    return out


def shap_analysis(
    detector: ColdStartDetector,
    X: np.ndarray,
    feats: list[str],
    y: np.ndarray | None = None,
) -> tuple[list[tuple[str, float]], np.ndarray | None, str]:
    """Compute feature importance; tries SHAP, falls back to permutation.

    Returns:
        ranked: list of (feature, mean_abs_shap_or_perm) sorted descending
        shap_values: (n, d) array if SHAP succeeded, else None
        method: 'shap' | 'permutation' | 'gain'
    """
    try:
        import shap
        explainer = shap.TreeExplainer(detector._model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        abs_mean = np.abs(vals).mean(axis=0)
        ranked = sorted(zip(feats, abs_mean.tolist()), key=lambda kv: -kv[1])
        return ranked, vals, "shap"
    except Exception as e:
        # SHAP version incompatibility — use permutation importance instead
        if y is not None and len(np.unique(y)) > 1:
            try:
                imp = permutation_importance(detector, X, y)
                ranked = sorted(zip(feats, imp.tolist()), key=lambda kv: -kv[1])
                return ranked, None, "permutation"
            except Exception as e2:
                print(f"  Permutation also failed ({e2}); using global gain")
        # Last resort: global XGBoost gain
        importance = detector._model.get_booster().get_score(importance_type="gain")
        ranked = []
        for i, name in enumerate(feats):
            key = f"f{i}"
            ranked.append((name, importance.get(key, 0.0)))
        ranked.sort(key=lambda kv: -kv[1])
        return ranked, None, "gain"


def per_class_shap(
    shap_values: np.ndarray,
    y: np.ndarray,
    attack_types: list[str],
    feats: list[str],
) -> dict:
    """Compute mean SHAP per feature, split by class and attack type."""
    out: dict = {"per_feature": {}}
    classes = {
        "honest": np.array([t == "honest" for t in attack_types]),
        "alie": np.array([t == "alie" for t in attack_types]),
        "gaussian_noise": np.array([t == "gaussian_noise" for t in attack_types]),
    }
    for i, fname in enumerate(feats):
        per_class = {}
        for class_name, mask in classes.items():
            if mask.sum() == 0:
                per_class[class_name] = {"n": 0, "mean_shap": 0.0,
                                          "mean_abs_shap": 0.0}
                continue
            vals = shap_values[mask, i]
            per_class[class_name] = {
                "n": int(mask.sum()),
                "mean_shap": float(np.mean(vals)),
                "mean_abs_shap": float(np.mean(np.abs(vals))),
            }
        out["per_feature"][fname] = per_class
    return out


def maybe_save_summary_plot(
    shap_values: np.ndarray | None,
    feats: list[str],
    X: np.ndarray,
    out_path: Path,
    title: str = "",
) -> str:
    if shap_values is None:
        return "skipped (no SHAP values)"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, feature_names=feats, show=False)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close("all")
        return str(out_path)
    except Exception as e:
        return f"plot skipped: {e}"


def analyze_subset(
    detector: ColdStartDetector,
    family: Family,
    paths: list[tuple[str, str]],
    label: str,
    k: int = 3,
) -> dict:
    """Run SHAP on a subset of signal sources and return summary."""
    feats = feature_names(family)
    X, y, atk, src = build_dataset(paths, family, k)

    if len(X) == 0:
        return {"label": label, "n": 0, "error": "no data"}

    ranked, shap_vals, method = shap_analysis(detector, X, feats, y=y)
    marginals = marginal_class_stats(X, atk, feats)

    # Predictions
    risk = detector.predict_risk(X)
    if risk.ndim == 0:
        risk = np.array([risk])

    # Per-class breakdown when SHAP available
    per_class_section = (
        per_class_shap(shap_vals, y, atk, feats) if shap_vals is not None else None
    )

    # Source breakdown for sanity (which sources ended up in this subset)
    src_counts: dict[str, int] = {}
    for s in src:
        src_counts[s] = src_counts.get(s, 0) + 1

    return {
        "label": label,
        "family": family.value,
        "method": method,
        "n": int(len(X)),
        "n_pos": int(y.sum()),
        "n_alie": int(sum(1 for t in atk if t == "alie")),
        "n_gaussian": int(sum(1 for t in atk if t == "gaussian_noise")),
        "n_honest": int(sum(1 for t in atk if t == "honest")),
        "source_counts": src_counts,
        "feature_ranking": [
            {"feature": k, "importance": float(v)} for k, v in ranked
        ],
        "marginal_class_stats": marginals,
        "per_class_shap": per_class_section,
        "mean_risk_alie": float(np.mean(risk[np.array([t == "alie" for t in atk])]))
            if any(t == "alie" for t in atk) else None,
        "mean_risk_gaussian": float(np.mean(risk[np.array([t == "gaussian_noise" for t in atk])]))
            if any(t == "gaussian_noise" for t in atk) else None,
        "mean_risk_honest": float(np.mean(risk[np.array([t == "honest" for t in atk])]))
            if any(t == "honest" for t in atk) else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="flower_reset",
                   choices=["flower_reset", "persistent_optimizer"])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--families", default="S,C",
                   help="Comma-separated families to analyze (S, W, C)")
    p.add_argument("--sweep-zmax", default="0.5")
    args = p.parse_args()

    families = [Family(f.strip()) for f in args.families.split(",")]

    print(f"\n{'='*80}")
    print(f"  Phase 4 / Track 2 — SHAP on Cold-Start Detector")
    print(f"  Mode: {args.mode}, Families: {[f.value for f in families]}, k={args.k}")
    print(f"{'='*80}")

    all_paths = find_signal_logs(args.mode, args.sweep_zmax)
    print(f"  Found {len(all_paths)} signal log files:")
    src_counts: dict[str, int] = {}
    for _, src in all_paths:
        src_counts[src] = src_counts.get(src, 0) + 1
    for src, n in src_counts.items():
        print(f"    {src}: {n} files")

    # Build subsets
    sweep_paths = [(p, s) for p, s in all_paths if s == "sweep"]
    persistent_paths = [(p, s) for p, s in all_paths if s == "persistent_dynamic"]
    flower_paths = [(p, s) for p, s in all_paths if s == "flower_scenario"]

    results: dict = {
        "mode": args.mode,
        "k": args.k,
        "subsets_analyzed": [],
        "by_family": {},
    }

    for family in families:
        print(f"\n  --- Family {family.value} ---")
        model_path = (
            PROJECT_ROOT / "models" / "cold_start" / args.mode
            / f"{family.value}_k{args.k}_final.pkl"
        )
        if not model_path.exists():
            print(f"    Model not found: {model_path}")
            continue

        detector = ColdStartDetector.load(str(model_path))
        print(f"    Loaded: {model_path}")

        family_results = {}

        # Subset A: ALL data (matches training corpus)
        family_results["all"] = analyze_subset(
            detector, family, all_paths, "all", args.k
        )
        # Subset B: real ALIE sweep only (Phase 3c data)
        family_results["sweep_only"] = analyze_subset(
            detector, family, sweep_paths, "sweep_only", args.k
        )
        # Subset C: persistent-opt dynamic RMC only (old fake ALIE)
        family_results["persistent_dynamic_only"] = analyze_subset(
            detector, family, persistent_paths,
            "persistent_dynamic_only", args.k
        )
        # Subset D: Flower scenarios only (no attacks — control)
        family_results["flower_scenario_only"] = analyze_subset(
            detector, family, flower_paths,
            "flower_scenario_only", args.k
        )

        # Print summary
        print(f"\n    {'Subset':<28} {'n':>6} {'n_alie':>7} {'n_gauss':>7} "
              f"{'n_honest':>8} {'top feature (importance)':>32}")
        print(f"    {'-'*92}")
        for sub_label, sub in family_results.items():
            if "error" in sub:
                print(f"    {sub_label:<28} {sub['n']:>6} {'(error: ' + sub['error'] + ')':>50}")
                continue
            top = sub["feature_ranking"][0] if sub["feature_ranking"] else None
            top_str = f"{top['feature']} ({top['importance']:.4f})" if top else "(none)"
            print(f"    {sub_label:<28} {sub['n']:>6} {sub['n_alie']:>7} "
                  f"{sub['n_gaussian']:>7} {sub['n_honest']:>8} {top_str:>32}")

        # Detailed feature ranking for all-data subset
        if "all" in family_results and "error" not in family_results["all"]:
            print(f"\n    Family {family.value} mean |SHAP| (all data, n="
                  f"{family_results['all']['n']}):")
            for entry in family_results["all"]["feature_ranking"][:10]:
                print(f"      {entry['feature']:<20} {entry['importance']:.4f}")

        # Per-class signed SHAP (cosine should push UP for malicious)
        if ("all" in family_results
                and family_results["all"].get("per_class_shap") is not None):
            pcs = family_results["all"]["per_class_shap"]["per_feature"]
            print(f"\n    Family {family.value} per-class mean SHAP "
                  f"(positive → pushes prediction toward malicious):")
            print(f"      {'Feature':<20} {'honest':>10} {'alie':>10} {'gaussian':>10}")
            for fname in feature_names(family):
                pc = pcs[fname]
                print(f"      {fname:<20} {pc['honest']['mean_shap']:>+10.4f} "
                      f"{pc['alie']['mean_shap']:>+10.4f} "
                      f"{pc['gaussian_noise']['mean_shap']:>+10.4f}")

        # Marginal class stats — independent of model, shows if features
        # naturally separate ALIE from honest in the data
        for sub_label in ("all", "sweep_only"):
            if sub_label not in family_results or "error" in family_results[sub_label]:
                continue
            mc = family_results[sub_label].get("marginal_class_stats", {})
            if not mc:
                continue
            print(f"\n    Family {family.value} marginal stats — {sub_label} "
                  f"(Cohen's d ALIE vs honest, |d|>0.8 = large effect):")
            print(f"      {'Feature':<20} {'honest_mean':>14} {'alie_mean':>14} "
                  f"{'cohens_d':>10}")
            entries = []
            for fname in feature_names(family):
                if fname not in mc:
                    continue
                row = mc[fname]
                d = row["cohens_d_alie_vs_honest"]
                entries.append((fname, row, d))
            # Sort by abs Cohen's d descending
            entries.sort(key=lambda e: -abs(e[2]))
            for fname, row, d in entries[:8]:
                print(f"      {fname:<20} {row['honest']['mean']:>14.4f} "
                      f"{row['alie']['mean']:>14.4f} {d:>+10.2f}")

        # Save summary plot if SHAP worked
        if "all" in family_results and "error" not in family_results["all"]:
            X, y, atk, _ = build_dataset(all_paths, family, args.k)
            ranked, shap_vals, method = shap_analysis(
                detector, X, feature_names(family)
            )
            if shap_vals is not None:
                out_png = (
                    PROJECT_ROOT / "results" / "20260427"
                    / f"phase3c_shap_summary_{family.value}.png"
                )
                msg = maybe_save_summary_plot(
                    shap_vals, feature_names(family), X, out_png,
                    title=f"Family {family.value} SHAP — {args.mode}"
                )
                print(f"    Plot: {msg}")

        results["by_family"][family.value] = family_results

    # Save full results
    out_path = PROJECT_ROOT / "results" / "20260427" / "phase3c_shap_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_path}")

    # Verdict
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")
    fam_s = results["by_family"].get("S", {})
    if fam_s and "all" in fam_s and "error" not in fam_s["all"]:
        all_method = fam_s["all"].get("method", "?")
        sweep_method = fam_s.get("sweep_only", {}).get("method", "?")
        print(f"  Importance method (all): {all_method}")
        print(f"  Importance method (sweep): {sweep_method}")
        all_top = fam_s["all"]["feature_ranking"][0]["feature"]
        sweep_top = (
            fam_s["sweep_only"]["feature_ranking"][0]["feature"]
            if "sweep_only" in fam_s and "error" not in fam_s["sweep_only"]
            else "?"
        )
        print(f"  Family S top feature on ALL data: {all_top}")
        print(f"  Family S top feature on REAL-ALIE-only (sweep) data: {sweep_top}")

        # Compare marginal Cohen's d for cos_to_median and num_examples on sweep data
        if "sweep_only" in fam_s and "error" not in fam_s["sweep_only"]:
            mc = fam_s["sweep_only"].get("marginal_class_stats", {})
            cos_d = mc.get("cos_to_median", {}).get("cohens_d_alie_vs_honest", 0)
            ne_d = mc.get("num_examples", {}).get("cohens_d_alie_vs_honest", 0)
            norm_d = mc.get("update_norm", {}).get("cohens_d_alie_vs_honest", 0)
            print(f"\n  Real-ALIE-only marginal Cohen's d (data-only, no model):")
            print(f"    cos_to_median: {cos_d:+.2f}")
            print(f"    update_norm:   {norm_d:+.2f}")
            print(f"    num_examples:  {ne_d:+.2f}")
            if abs(cos_d) > abs(ne_d) and abs(cos_d) > 0.5:
                print(f"  → COSINE STORY VALIDATED: cos_to_median has stronger ALIE/honest "
                      f"separation than num_examples in the actual real-ALIE data.")
                print(f"    The high gain on num_examples reflects training-corpus mixture, "
                      f"not real-ALIE detection.")
            elif abs(ne_d) > abs(cos_d) and abs(ne_d) > 0.5:
                print(f"  → CONFOUND CONFIRMED: num_examples separates classes more "
                      f"strongly than cosine even in real-ALIE data.")
            else:
                print(f"  → Mixed: neither feature shows large ALIE/honest separation; "
                      f"detection may rely on subtle multi-feature interactions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

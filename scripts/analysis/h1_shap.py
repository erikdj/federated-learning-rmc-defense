"""Tree SHAP and permutation importance for frozen H1 signal-family detectors.

Rebuilds feature matrices through h1_retrain_ramp3_dev_read and
h1_signal_family_eval, then runs shap.TreeExplainer on the supplied fitted
GBDTs. The optional permutation comparison uses the same model and matrix.

Provide --signals-dir, --models-dir and --out-dir for a new analysis. Default
paths identify the historical ramp-3 development instrument; use the appropriate
corpus and frozen models for the study being reported. This command does not
launch a federated-learning simulation or write to cloud services.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

# Committed extraction / custody path — imported, never re-derived.
from h1_retrain_ramp3_dev_read import (  # noqa: E402
    DEV_SEEDS,
    FAMILIES,
    K,
    enumerate_units,
    extract_family_features,
    load_seed_rows,
)

import pickle  # noqa: E402

DEFAULT_SIGNALS_DIR = REPO / "results/20260724/exp014_closed_loop/signals"
DEFAULT_MODELS_DIR = REPO / "models/h1_signal_family/ramp3_closed_loop"
DEFAULT_OUT_DIR = REPO / "results/20260724/h1_shap"

# Permutation-importance config (apples-to-apples baseline on the SAME matrices).
PERM_N_REPEATS = 30
PERM_SEED = 42
PERM_SCORING = "roc_auc"  # threshold-free; matches the detector's score-based operating point


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def rebuild_dev_matrix(signals_dir: Path, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the EXACT dev feature matrix the H1 retrain trained on, for one family.

    Mirrors ``h1_retrain_ramp3_dev_read.main`` lines that construct ``all_rows``:
    integrity-gate the 25-cell Krum+TGE cross-product, pool all 5 dev seeds with
    strict per-row custody, then delegate to the committed ``extract_family_features``.
    """
    units = enumerate_units(signals_dir)
    rows_by_seed = {s: load_seed_rows(units, s) for s in DEV_SEEDS}
    all_rows = [r for s in DEV_SEEDS for r in rows_by_seed[s]]
    X, y = extract_family_features(all_rows, family, K)
    return X, y


def load_detector(models_dir: Path, family: str) -> dict:
    """Load a frozen §4.2 detector pickle (dict with model / feature_names / …)."""
    with open(models_dir / f"{family}_k{K}_final.pkl", "rb") as fh:
        return pickle.load(fh)


def shap_ranking(clf, X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeExplainer SHAP. Returns (mean_abs_shap[d], shap_values[n, d]).

    For a binary sklearn GBDT, TreeExplainer yields SHAP values in the raw-margin
    (log-odds) space of the positive (malicious) class. Newer shap returns a 3D
    ``(n, d, n_classes)`` array for some estimators; we collapse to the positive
    class when that happens.
    """
    import shap

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, d, n_classes) -> positive class
        sv = sv[:, :, 1]
    elif isinstance(sv, list):  # legacy list-of-arrays API
        sv = np.asarray(sv[1])
    if sv.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP value width {sv.shape[1]} != {len(feature_names)} features"
        )
    mean_abs = np.abs(sv).mean(axis=0)
    return mean_abs, sv


def permutation_ranking(
    clf, X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    """sklearn permutation importance on the SAME matrix/detector (apples-to-apples)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        clf, X, y,
        n_repeats=PERM_N_REPEATS,
        random_state=PERM_SEED,
        scoring=PERM_SCORING,
    )
    return result.importances_mean


def _ranked(names: list[str], values: np.ndarray) -> list[dict]:
    order = np.argsort(-values)
    return [{"feature": names[i], "value": float(values[i])} for i in order]


def plot_family(
    family: str,
    sv: np.ndarray,
    X: np.ndarray,
    mean_abs: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> dict[str, str]:
    """Beeswarm (rich per-example) + mean|SHAP| bar. Clean default styling.

    These are feature-importance panels (no per-defense series), so the locked
    defense palette does not apply; we use neutral styling consistent with the
    repo's committed-figure font/dpi conventions (dpi 150, tight layout).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    dpi = 150
    paths: dict[str, str] = {}

    # --- Beeswarm ---------------------------------------------------------
    explanation = shap.Explanation(
        values=sv, data=X, feature_names=feature_names
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)
    plt.title(f"H1 family {family}: SHAP value distribution (malicious log-odds)")
    plt.tight_layout()
    beeswarm_path = out_dir / f"{family}_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    paths["beeswarm"] = str(beeswarm_path.relative_to(REPO))

    # --- Mean|SHAP| bar ---------------------------------------------------
    order = np.argsort(mean_abs)  # ascending -> largest at top of barh
    names_sorted = [feature_names[i] for i in order]
    vals_sorted = mean_abs[order]
    fig_h = max(2.5, 0.42 * len(feature_names) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    ax.barh(range(len(names_sorted)), vals_sorted, color="#1F497D")
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"H1 family {family}: mean absolute SHAP importance")
    for i, v in enumerate(vals_sorted):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    bar_path = out_dir / f"{family}_shap_bar.png"
    fig.savefig(bar_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    paths["bar"] = str(bar_path.relative_to(REPO))

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals-dir", type=Path, default=DEFAULT_SIGNALS_DIR)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    families_out: dict[str, dict] = {}
    for fam in FAMILIES:
        det = load_detector(args.models_dir, fam)
        clf = det["model"]
        feat_names = list(det["feature_names"])

        X, y = rebuild_dev_matrix(args.signals_dir, fam)
        # Custody cross-check: rebuilt matrix must match the detector's training size.
        if X.shape[0] != det["n_train"] or int(y.sum()) != det["n_train_pos"]:
            raise RuntimeError(
                f"family {fam}: rebuilt matrix {X.shape[0]}x pos={int(y.sum())} "
                f"!= detector n_train={det['n_train']} pos={det['n_train_pos']}"
            )

        mean_abs, sv = shap_ranking(clf, X, feat_names)
        perm = permutation_ranking(clf, X, y, feat_names)
        gbdt_impurity = np.asarray(clf.feature_importances_, dtype=float)

        fig_paths = plot_family(fam, sv, X, mean_abs, feat_names, args.out_dir)

        families_out[fam] = {
            "n_features": len(feat_names),
            "n_rows": int(X.shape[0]),
            "n_malicious": int(y.sum()),
            "n_honest": int((y == 0).sum()),
            "feature_names": feat_names,
            "shap_mean_abs": {n: float(v) for n, v in zip(feat_names, mean_abs)},
            "shap_ranking": _ranked(feat_names, mean_abs),
            "permutation_importance": {n: float(v) for n, v in zip(feat_names, perm)},
            "permutation_ranking": _ranked(feat_names, perm),
            "gbdt_impurity_importance": {n: float(v) for n, v in zip(feat_names, gbdt_impurity)},
            "gbdt_impurity_ranking": _ranked(feat_names, gbdt_impurity),
            "figures": fig_paths,
        }

    out = {
        "_meta": {
            "read": "GWU-26 — genuine TreeExplainer SHAP on the H1 signal-family detectors",
            "method_shap": "shap.TreeExplainer (exact), positive-class (malicious) log-odds margin",
            "method_permutation": (
                f"sklearn.inspection.permutation_importance "
                f"(n_repeats={PERM_N_REPEATS}, random_state={PERM_SEED}, scoring={PERM_SCORING}) "
                "on the SAME H1 matrices/detectors — the apples-to-apples baseline"
            ),
            "detectors": "models/h1_signal_family/ramp3_closed_loop/{S,W,C}_k3_final.pkl (frozen §4.2)",
            "matrix_custody": (
                "rebuilt via committed enumerate_units + load_seed_rows + extract_family_features "
                "(imported from scripts/h1_retrain_ramp3_dev_read.py); identity hard-gated by "
                "tests/test_h1_shap.py"
            ),
            "scope_note": (
                "raw_feature_shap.py's committed record (results/20260515/raw_feature_shap.json) is on "
                "the 45 raw Edge-IIoT features of the torch Net classifier — a different model and "
                "feature space — and is NOT comparable to these H1 detector rankings."
            ),
            "k": K,
            "dev_seeds": DEV_SEEDS,
            "git_commit": commit,
        },
        "families": families_out,
    }
    out_json = args.out_dir / "h1_shap.json"
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False))

    # --- console summary --------------------------------------------------
    print("== GWU-26 H1 SHAP (TreeExplainer, ramp-3 closed-loop Krum+TGE dev) ==")
    for fam in FAMILIES:
        fo = families_out[fam]
        print(f"  family {fam}: {fo['n_rows']} rows "
              f"({fo['n_malicious']} mal / {fo['n_honest']} honest), {fo['n_features']} features")
        top = fo["shap_ranking"][:5]
        print("    top-5 by mean|SHAP|: " +
              ", ".join(f"{e['feature']}={e['value']:.3f}" for e in top))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

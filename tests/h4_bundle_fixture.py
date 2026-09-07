"""Synthetic H4 serving-bundle fixture (BUILD_CONTRACT schema).

Lane A produces the REAL bundle (`data/h4_serving/`); these tests must not
depend on it existing, so they build a tiny schema-exact synthetic bundle:
a real joblib-serialized GradientBoostingClassifier over the frozen
9-feature order, per-scenario cuts, and a manifest with genuine per-file
sha256s. Deterministic (fixed RNG seeds) so hashes are stable within a
test run; committed as a helper module, not a golden artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: The frozen v1.15 § 2.2b 9-feature order (h2prime_common.FEATS).
FROZEN_FEATURES = [
    "update_norm", "train_loss", "num_examples",
    "norm_variance", "loss_slope",
    "cos_to_median", "L2_to_median", "cos_drift", "cos_variance",
]

DEFAULT_CUTS = {"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5, "S4": 0.5}

#: Erratum-B arm-classes (cut grain of bundle v2).
ARM_CLASSES = ("krum_family", "ts_family", "fedavg_family")

#: v2 scenario tokens: C0 has its OWN calibrated cut (E3-bis alias retired).
SCENARIO_TOKENS_V2 = ("C0", "S0", "S1", "S2", "S3", "S4")

DEFAULT_CUTS_V2 = {
    tok: {ac: 0.5 for ac in ARM_CLASSES} for tok in SCENARIO_TOKENS_V2
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_tiny_gbdt(n_features: int = 9, n_estimators: int = 5):
    """A tiny deterministic GradientBoostingClassifier (real sklearn)."""
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, n_features))
    # Separable-ish label so predict_proba spans both sides of 0.5.
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=0
    ).fit(X, y)


def make_synthetic_bundle(
    bundle_dir: Path,
    cuts: "dict | None" = None,
    features: "list | None" = None,
    model=None,
    manifest_extra: "dict | None" = None,
) -> Path:
    """Write a schema-exact serving bundle into `bundle_dir`; returns it."""
    import joblib

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    features = list(features) if features is not None else list(FROZEN_FEATURES)
    cuts = dict(cuts) if cuts is not None else dict(DEFAULT_CUTS)
    model = model if model is not None else fit_tiny_gbdt(len(features))

    joblib.dump(model, bundle_dir / "model.joblib")
    (bundle_dir / "cuts.json").write_text(json.dumps(cuts, indent=1))
    (bundle_dir / "features.json").write_text(json.dumps(features, indent=1))

    manifest = {
        "files": {
            name: _sha256_file(bundle_dir / name)
            for name in ("model.joblib", "cuts.json", "features.json")
        },
        "corpus_manifest_sha256": "0" * 64,
        "gbdt_params": {"n_estimators": getattr(model, "n_estimators", None)},
        "source_commit": "synthetic-test-fixture",
        "built_at": "2026-08-17T00:00:00+00:00",
        **(manifest_extra or {}),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return bundle_dir


def make_synthetic_bundle_v2(
    bundle_dir: Path,
    cuts_v2: "dict | None" = None,
    features: "list | None" = None,
    model=None,
    manifest_extra: "dict | None" = None,
    with_v1: bool = True,
) -> Path:
    """Write a schema-exact erratum-B v2 serving bundle into `bundle_dir`.

    v2 shares `model.joblib` + `features.json` with v1 and adds
    `cuts_v2.json` (keyed {scenario: {arm_class: cut}}) + `manifest_v2.json`
    whose `files` map pins model/cuts_v2/features. `with_v1=True` (the
    deployed layout) writes the full v1 bundle alongside, byte-identical to
    `make_synthetic_bundle`'s output for the same inputs.
    """
    import joblib

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    features = list(features) if features is not None else list(FROZEN_FEATURES)
    model = model if model is not None else fit_tiny_gbdt(len(features))
    if with_v1:
        make_synthetic_bundle(bundle_dir, features=features, model=model)
    else:
        joblib.dump(model, bundle_dir / "model.joblib")
        (bundle_dir / "features.json").write_text(json.dumps(features, indent=1))

    cuts_v2 = (
        {k: dict(v) for k, v in cuts_v2.items()}
        if cuts_v2 is not None
        else {k: dict(v) for k, v in DEFAULT_CUTS_V2.items()}
    )
    (bundle_dir / "cuts_v2.json").write_text(json.dumps(cuts_v2, indent=1))

    manifest_v2 = {
        "bundle_format": "h4-serving-v2",
        "files": {
            name: _sha256_file(bundle_dir / name)
            for name in ("model.joblib", "cuts_v2.json", "features.json")
        },
        "source_commit": "synthetic-test-fixture",
        "built_at": "2026-08-18T00:00:00+00:00",
        **(manifest_extra or {}),
    }
    (bundle_dir / "manifest_v2.json").write_text(json.dumps(manifest_v2, indent=1))
    return bundle_dir

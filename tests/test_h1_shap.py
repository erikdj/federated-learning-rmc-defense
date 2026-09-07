"""— hard-gate that h1_shap's feature-matrix path IS the H1 retrain's.

The SHAP script must attribute importance on exactly the matrix each detector was
trained on. Two guarantees are tested:

1. Function identity — h1_shap reuses the committed extraction primitives
   (extract_family_features, DEV_SEEDS, K) imported from h1_retrain_ramp3_dev_read,
   not a re-derived copy.
2. Same-rows-in == same-matrix-out — on both a synthetic fixture and (when present)
   the real ramp-3 signals, h1_shap.rebuild_dev_matrix produces byte-identical
   (X, y) to the retrain's own all_rows construction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import h1_retrain_ramp3_dev_read as retrain  # noqa: E402
import h1_shap  # noqa: E402


def test_shap_reuses_committed_extraction_path():
    """h1_shap must import the SAME extraction primitives as the retrain (identity)."""
    assert h1_shap.extract_family_features is retrain.extract_family_features
    assert h1_shap.K == retrain.K
    assert h1_shap.DEV_SEEDS == retrain.DEV_SEEDS
    assert h1_shap.FAMILIES == retrain.FAMILIES


def _synthetic_rows() -> list[dict]:
    """Two identities (one malicious, one honest) each with 4 observed rounds."""
    rows = []
    for cid, mal in (("attacker_1", True), ("honest_1", False)):
        for rnd in range(4):
            rows.append({
                "logical_cid": cid,
                "scenario_round": rnd,
                "malicious_gt": mal,
                "update_norm": 1.0 + rnd + (2.0 if mal else 0.0),
                "cos_to_median": 0.9 - 0.1 * rnd - (0.3 if mal else 0.0),
                "L2_to_median": 0.5 + rnd + (1.0 if mal else 0.0),
                "train_loss": 0.4 - 0.05 * rnd,
                "num_examples": 100 + rnd,
                "seed": 42,
                "scenario": "S0_clean_baseline",
                "defense": "krumtge",
            })
    return rows


@pytest.mark.parametrize("family", ["S", "W", "C"])
def test_same_rows_in_same_matrix_out_fixture(family):
    """Same synthetic rows fed through the committed extractor are deterministic and
    the cold-start window (first K rounds/identity) is honored identically."""
    rows = _synthetic_rows()
    X1, y1 = retrain.extract_family_features(rows, family, retrain.K)
    X2, y2 = h1_shap.extract_family_features(rows, family, h1_shap.K)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    # cold-start window: every family emits one row per client-round up to K
    # (W's row is a window over the first obs_round rounds), so 2 identities x K.
    assert X1.shape[0] == 2 * retrain.K


@pytest.mark.parametrize("family", ["S", "W", "C"])
def test_rebuild_matches_retrain_on_real_signals(family):
    """End-to-end: h1_shap.rebuild_dev_matrix == retrain's own all_rows construction."""
    signals_dir = h1_shap.DEFAULT_SIGNALS_DIR
    if not signals_dir.exists() or not list(signals_dir.glob("*__krum_tge__*.jsonl")):
        pytest.skip("ramp-3 closed-loop signals not present")

    # Retrain's construction (h1_retrain_ramp3_dev_read.main, all_rows path).
    units = retrain.enumerate_units(signals_dir)
    rows_by_seed = {s: retrain.load_seed_rows(units, s) for s in retrain.DEV_SEEDS}
    all_rows = [r for s in retrain.DEV_SEEDS for r in rows_by_seed[s]]
    X_ref, y_ref = retrain.extract_family_features(all_rows, family, retrain.K)

    X_shap, y_shap = h1_shap.rebuild_dev_matrix(signals_dir, family)

    assert np.array_equal(X_ref, X_shap)
    assert np.array_equal(y_ref, y_shap)

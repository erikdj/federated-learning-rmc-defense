"""KrumDefensePlugin scores past the certified tolerance (methodology v1.19).

The old guard treated Krum's CERTIFICATION bound (n <= 2f+2) as a
computability bound and returned uniform 1.0 scores at the canonical RMC
parameters (n=20, f=9) — degenerating the calibration Krum arm to
keep-first-9-by-arrival-order and flattening the ``krum_score`` signal-log
channel (no variance -> H2 recall@FPR uncomputable for the Krum arm).

The score is perfectly computable whenever ``num_closest = n - f - 2 >= 1``:
the April Szelag anchor (scripts/reproduce_szelag.py::aggregate_krum) computes
it at exactly n=20/f=9 with no guard. Szelag-faithfulness is the
pre-registered principle; the praxis deliberately studies Krum pushed past its
certified tolerance under RMC. The guard is therefore relaxed to a
computability check: compute whenever ``n - f - 2 >= 1``; WARN once per round
when ``f >= (n-2)/2`` (certified tolerance exceeded); uniform fallback ONLY
when ``n - f - 2 < 1`` (truly uncomputable), at ERROR level.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from flowerfl.byzantine_defense import KrumDefensePlugin

LOGGER_NAME = "flowerfl.byzantine_defense"


class _FakeClient:
    def __init__(self, cid: str):
        self.cid = cid


class _FakeFitRes:
    def __init__(self, vec: np.ndarray):
        self.parameters = ndarrays_to_parameters([vec.astype(np.float32)])
        self.num_examples = 100
        self.metrics = {}


def _results(vectors: list[np.ndarray]):
    return [(_FakeClient(f"cid_{i}"), _FakeFitRes(v)) for i, v in enumerate(vectors)]


def _cluster_with_outlier(n: int, outlier_idx: int, dim: int = 8):
    """n-1 near-identical honest updates + one gross outlier at outlier_idx."""
    rng = np.random.default_rng(42)
    vecs = [rng.normal(0.0, 0.01, size=dim) for _ in range(n)]
    vecs[outlier_idx] = vecs[outlier_idx] + 100.0
    return vecs


@pytest.mark.unit
def test_canonical_n20_f9_scores_nonuniform_and_rank_outlier(caplog):
    """n=20/f=9 (canonical RMC): computable (num_closest=9). Scores must be
    NON-uniform, rank by distance (outlier lowest), and filter_updates must
    exclude the outlier while keeping exactly m=9."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9)
    outlier = 7
    results = _results(_cluster_with_outlier(20, outlier))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        scores = plugin.score_updates(results, server_round=5)

    assert len(set(scores.values())) > 1, "scores must not be uniform at n=20/f=9"
    assert min(scores, key=scores.get) == outlier, "outlier must get the lowest score"

    kept = plugin.filter_updates(results, scores)
    kept_cids = {c.cid for c, _ in kept}
    assert len(kept) == 9
    assert f"cid_{outlier}" not in kept_cids, "outlier must be excluded"

    # Certified tolerance exceeded (f=9 >= (20-2)/2=9): exactly one clear
    # WARNING per round, and it must say the score is still Szelag-faithful.
    warns = [r for r in caplog.records
             if r.levelno == logging.WARNING and "certified bound exceeded" in r.getMessage()]
    assert len(warns) == 1
    assert "Szel" in warns[0].getMessage()

    # The continuous-score channel must carry the variance too (H2 recall@FPR).
    assert len(set(plugin._round_scores[5].values())) > 1


@pytest.mark.unit
def test_churn_round_n15_f9_still_computable_nonuniform():
    """Churn-round cohort n=15 with fixed f=9: num_closest=4 >= 1, so scores
    are computed (non-uniform), not the uniform fallback."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=0)
    results = _results(_cluster_with_outlier(15, outlier_idx=3))

    scores = plugin.score_updates(results, server_round=7)

    assert len(set(scores.values())) > 1, "scores must not be uniform at n=15/f=9"
    assert min(scores, key=scores.get) == 3


@pytest.mark.unit
def test_uncomputable_n12_f11_uniform_fallback_error_log(caplog):
    """n=12/f=11: num_closest = -1 < 1 — truly uncomputable. Uniform fallback
    is kept, but at ERROR level (this state must never occur in registered
    scenarios)."""
    plugin = KrumDefensePlugin(num_malicious=11, num_to_keep=0)
    rng = np.random.default_rng(0)
    results = _results([rng.normal(size=8) for _ in range(12)])

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        scores = plugin.score_updates(results, server_round=2)

    assert set(scores.values()) == {1.0}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "uncomputable case must log at ERROR level"

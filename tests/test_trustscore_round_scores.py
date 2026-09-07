"""Regression (B6, 2026-06-05): TrustScorePlugin must expose per-round scores via
`_round_scores[server_round]`, the attribute ScenarioStrategy reads to write the
`trust_score` field into the signal log.

Without it, every TrustScore run logs trust_score=None for all rows, so the H2
primary metric recall@10%FPR is UNCOMPUTABLE for the entire TrustScore arm (25 of
the 75 runs) — caught by the per-defense instrumentation smoke audit.
KrumDefensePlugin already populates _round_scores (byzantine_defense.py); this
test pins the same contract for TrustScore.
"""
import numbers
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_results(n=5, dim=8, outlier_idx=4, seed=0):
    """Build n (ClientProxy, FitRes) pairs; one client is a clear outlier."""
    from flwr.common import ndarrays_to_parameters

    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, size=dim)
    results = []
    for i in range(n):
        vec = base + rng.normal(0, 0.001, size=dim)
        if i == outlier_idx:
            vec = vec + 10.0  # large deviation from median
        proxy = SimpleNamespace(cid=f"cid_{i}")
        fit_res = SimpleNamespace(
            parameters=ndarrays_to_parameters([vec.astype(np.float32)]),
            num_examples=100,
            metrics={},
        )
        results.append((proxy, fit_res))
    return results


def test_trustscore_populates_round_scores():
    from flowerfl.byzantine_defense import TrustScorePlugin

    plugin = TrustScorePlugin(decay_rate=0.9, outlier_threshold=2.0)
    results = _make_results(n=5)
    plugin.score_updates(results, server_round=1)

    # The attribute the strategy reads must exist and be keyed by server_round.
    assert hasattr(plugin, "_round_scores"), (
        "TrustScorePlugin must expose `_round_scores` (the attribute "
        "ScenarioStrategy reads to log trust_score)."
    )
    assert 1 in plugin._round_scores, "no per-round scores stored for round 1"
    scores = plugin._round_scores[1]
    assert isinstance(scores, dict) and len(scores) == 5, (
        f"expected {{idx: score}} for 5 clients; got {scores!r}"
    )
    # Keyed by client index, values are finite numeric scores (np float ok —
    # they JSON-serialize the same way krum_score does).
    for idx in range(5):
        assert idx in scores, f"missing score for client index {idx}"
        assert isinstance(scores[idx], numbers.Real) and np.isfinite(scores[idx])
        float(scores[idx])  # must be coercible for JSON serialization


def test_trustscore_round_scores_match_returned():
    """The stored per-round scores must equal what score_updates returns."""
    from flowerfl.byzantine_defense import TrustScorePlugin

    plugin = TrustScorePlugin()
    results = _make_results(n=5)
    returned = plugin.score_updates(results, server_round=3)
    assert plugin._round_scores[3] == returned

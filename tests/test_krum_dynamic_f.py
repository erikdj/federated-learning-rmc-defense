"""Dynamic per-round f in KrumDefensePlugin (methodology v1.19, ).

Static peak f=9 breaks S3/S4 disconnect rounds: those rounds schedule only
~11 participants (the 9 adversaries are DISCONNECTED — that is the RMC
pattern), and at n=11 static f=9 gives num_closest = 11-9-2 = 0 < 1, firing
the uncomputable fallback (uniform scores + ERROR log) on a large fraction of
scheduled rounds. The April Szelag anchor solved this with per-round dynamic
f (the original baseline reproduction: ``math.ceil(n/2) - 1``), which
reproduces the documented full-cohort operating point exactly:
n=20 -> f=9, keep=n-f-2=9; n=11 -> f=5, keep=4 (computable, no ERROR).

``dynamic_f=True`` opts a plugin into per-round sizing; the static
num_malicious/num_to_keep are retained for provenance only. Default
(``dynamic_f=False``) is byte-for-byte the static behavior.
"""
from __future__ import annotations

import logging
import math

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


def _results(vectors):
    return [(_FakeClient(f"cid_{i}"), _FakeFitRes(v)) for i, v in enumerate(vectors)]


def _cluster_with_outlier(n: int, outlier_idx: int, dim: int = 8):
    rng = np.random.default_rng(42)
    vecs = [rng.normal(0.0, 0.01, size=dim) for _ in range(n)]
    vecs[outlier_idx] = vecs[outlier_idx] + 100.0
    return vecs


@pytest.mark.unit
def test_effective_f_april_formula():
    """Dynamic f follows the April anchor formula ceil(n/2)-1 per round."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    assert plugin._effective_f(20) == 9   # full cohort — documented 9/20
    assert plugin._effective_f(11) == 5   # S3/S4 disconnect round
    assert plugin._effective_f(15) == 7   # honest-churn round
    assert plugin._effective_f(3) == 1
    # Static plugin ignores n and uses the configured f.
    static = KrumDefensePlugin(num_malicious=9, num_to_keep=9)
    assert static._effective_f(11) == 9


@pytest.mark.unit
def test_dynamic_full_cohort_n20_reproduces_documented_operating_point(caplog):
    """n=20 dynamic: f=ceil(20/2)-1=9, keep=20-9-2=9 — identical to the
    scenario-derived static sizing; real (non-uniform) scores; outlier
    excluded; certified-bound warning still fires (f=9 >= (20-2)/2=9)."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    outlier = 7
    results = _results(_cluster_with_outlier(20, outlier))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        scores = plugin.score_updates(results, server_round=5)

    assert len(set(scores.values())) > 1
    assert min(scores, key=scores.get) == outlier

    kept = plugin.filter_updates(results, scores)
    assert len(kept) == 9
    assert f"cid_{outlier}" not in {c.cid for c, _ in kept}

    warns = [r for r in caplog.records
             if r.levelno == logging.WARNING and "certified bound exceeded" in r.getMessage()]
    assert len(warns) == 1


@pytest.mark.unit
def test_dynamic_disconnect_n11_computable_no_error(caplog):
    """S3/S4 disconnect round n=11: dynamic f=5, keep=4, num_closest=4 —
    computable, non-uniform scores, and NO ERROR-level log (static f=9 would
    have fired the uncomputable fallback here)."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    outlier = 2
    results = _results(_cluster_with_outlier(11, outlier))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        scores = plugin.score_updates(results, server_round=8)
        kept = plugin.filter_updates(results, scores)

    assert len(set(scores.values())) > 1, "n=11 must be computable under dynamic f"
    assert min(scores, key=scores.get) == outlier
    assert len(kept) == 4  # 11 - 5 - 2
    assert f"cid_{outlier}" not in {c.cid for c, _ in kept}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert not errors, f"no ERROR log expected at n=11 dynamic; got {errors}"


@pytest.mark.unit
def test_dynamic_minimal_n3_keeps_at_least_one():
    """n=3: dynamic f=1 -> num_closest=0 (uncomputable -> uniform fallback),
    but keep = max(1, 3-1-2) = 1 — the filter still keeps >= 1 client."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    rng = np.random.default_rng(0)
    results = _results([rng.normal(size=8) for _ in range(3)])

    scores = plugin.score_updates(results, server_round=1)
    kept = plugin.filter_updates(results, scores)

    assert len(kept) >= 1


@pytest.mark.unit
def test_static_path_unchanged_by_default():
    """dynamic_f defaults to False and the static sizing is untouched:
    n=20/f=9/keep=9 behaves exactly as the pre-P1 code."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9)
    assert plugin._dynamic_f is False
    results = _results(_cluster_with_outlier(20, outlier_idx=0))
    scores = plugin.score_updates(results, server_round=3)
    kept = plugin.filter_updates(results, scores)
    assert len(kept) == 9
    assert "cid_0" not in {c.cid for c, _ in kept}


@pytest.mark.unit
def test_dynamic_formula_ties_to_scenario_derived_sizing():
    """Ties the two sources of truth together: at the scenario-derived full
    cohort (20), the April dynamic formula reproduces the scenario-derived
    static f (9) and the documented keep (9)."""
    from scripts.run_phase4_flower import _scenario_defense_sizing
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sizing = _scenario_defense_sizing(str(root / "rmc/scenarios/S4_full_mix.json"))
    assert sizing is not None
    num_malicious, cohort = sizing
    assert math.ceil(cohort / 2) - 1 == num_malicious == 9
    assert cohort - num_malicious - 2 == 9

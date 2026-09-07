"""EXP-061 C6 regression (erratum B build item 4): Krum's uncomputable path
must RECORD its uniform scores.

The smoke diagnosed rounds where an upstream filter (the H4 online detector
during warmup) shrank the cohort below Krum's computability bound
(``num_closest = n - f - 2 < 1``). That path returned uniform scores for
`filter_updates` but never wrote them into ``self._round_scores`` — so the
cid-keyed signal join rendered truthful NULLs for `krum_score` while
``aggregator_rejected > 0``. The fix records the uniform scores before
returning, exactly as the computable path does, so the signal log renders
values (uniform 1.0) instead of nulls.
"""
from __future__ import annotations

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from flowerfl.byzantine_defense import KrumDefensePlugin


class _FakeClient:
    def __init__(self, cid: str):
        self.cid = cid


class _FakeFitRes:
    def __init__(self, vec: np.ndarray):
        self.parameters = ndarrays_to_parameters([vec.astype(np.float32)])
        self.num_examples = 100
        self.metrics = {}


def _results(n: int, dim: int = 4):
    rng = np.random.default_rng(7)
    return [
        (_FakeClient(f"cid_{i}"), _FakeFitRes(rng.normal(size=dim)))
        for i in range(n)
    ]


@pytest.mark.unit
def test_uncomputable_path_records_uniform_scores_into_round_scores():
    """n=3 dynamic cohort: f=ceil(3/2)-1=1 -> num_closest=0 (uncomputable).

    The returned scores must be uniform 1.0 AND recorded into
    ``_round_scores[server_round]`` so the signal join renders values.
    """
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    server_round = 5
    scores = plugin.score_updates(_results(3), server_round)
    assert scores == {0: 1.0, 1: 1.0, 2: 1.0}
    assert plugin._round_scores.get(server_round) == {0: 1.0, 1: 1.0, 2: 1.0}


@pytest.mark.unit
def test_uncomputable_recording_is_a_copy_not_an_alias():
    """Mutating the returned dict must not corrupt the recorded round scores
    (immutability discipline: the record is the signal log's source)."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    scores = plugin.score_updates(_results(3), 1)
    scores[0] = 0.0
    assert plugin._round_scores[1][0] == 1.0


@pytest.mark.unit
def test_computable_path_recording_is_unchanged():
    """Sanity pin: the computable path (n=20) still records per-round scores."""
    plugin = KrumDefensePlugin(num_malicious=9, num_to_keep=9, dynamic_f=True)
    scores = plugin.score_updates(_results(20), 3)
    assert set(scores) == set(range(20))
    assert plugin._round_scores[3] == scores

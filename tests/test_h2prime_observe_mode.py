"""Erratum-B build item 1: observe-only mode of the online H2' detector.

The public workflow is described in `docs/reproduction/experiments.md`.
Historical protocol references: § B1
(RULED, methodology v1.53): in observe-only mode the detector scores every
participating client per round EXACTLY as the enforcing mode does and logs
per-(client, round) rows — raw score, would-flag decision — but
`filter_updates` DROPS NOBODY. OFF/absent = byte-identical enforcing
behavior (pinned here against the incumbent path).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, Status, ndarrays_to_parameters

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowerfl.h2prime_online import (  # noqa: E402
    OnlineH2PrimeDetectorPlugin,
    ServingBundle,
)
from h4_bundle_fixture import FROZEN_FEATURES  # noqa: E402


class _StubModel:
    """predict_proba driven by {update_norm_rounded: prob} (column 0 of the
    frozen feature order), so tests pin probabilities via parameter vectors."""

    def __init__(self, prob_by_norm):
        self._prob_by_norm = dict(prob_by_norm)
        self.n_features_in_ = len(FROZEN_FEATURES)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.array(
            [self._prob_by_norm[round(float(row[0]), 3)] for row in X]
        )
        return np.column_stack([1.0 - probs, probs])


def _bundle(model, cuts=None) -> ServingBundle:
    return ServingBundle(
        bundle_dir=Path("."),
        model=model,
        cuts=dict(cuts or {"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5,
                           "S4": 0.5}),
        features=tuple(FROZEN_FEATURES),
        manifest={},
        bundle_sha256="ab" * 32,
    )


def _results(spec):
    out = []
    for cid, norm, train_loss in spec:
        params = ndarrays_to_parameters([np.array([norm], dtype=np.float64)])
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=100,
            metrics={"train_loss": train_loss},
        )
        out.append((proxy, fit))
    return out


_PROBS = {1.0: 0.4, 2.0: 0.5, 3.0: 0.9}
_SPEC = [("cid_a_0", 1.0, 0.5), ("cid_b_1", 2.0, 0.5), ("cid_c_2", 3.0, 0.5)]
_IDS = {"cid_a_0": "client_0", "cid_b_1": "client_1", "cid_c_2": "client_2"}


def _detector(observe_only=False):
    det = OnlineH2PrimeDetectorPlugin(
        bundle=_bundle(_StubModel(_PROBS)),
        scenario_token="S3",
        observe_only=observe_only,
    )
    det.set_identity_map(dict(_IDS))
    return det


@pytest.mark.unit
def test_observe_mode_scores_everyone_but_drops_nobody():
    det = _detector(observe_only=True)
    results = _results(_SPEC)
    scores = det.score_updates(results, server_round=2)
    # everyone keeps score 1.0 — the drop decision is never made
    assert scores == {0: 1.0, 1: 1.0, 2: 1.0}
    kept = det.filter_updates(results, scores)
    assert [p.cid for p, _ in kept] == ["cid_a_0", "cid_b_1", "cid_c_2"]
    # nothing was ACTUALLY flagged (flagged_identities = enforcement record)
    assert det.flagged_identities(2) == ()


@pytest.mark.unit
def test_observe_rows_carry_score_and_would_flag_per_client_round():
    det = _detector(observe_only=True)
    det.score_updates(_results(_SPEC), server_round=2)
    rows = det.observe_rows
    assert [r["logical_cid"] for r in rows] == [
        "client_0", "client_1", "client_2"]
    by_cid = {r["logical_cid"]: r for r in rows}
    assert by_cid["client_0"]["score"] == pytest.approx(0.4)
    assert by_cid["client_1"]["score"] == pytest.approx(0.5)
    assert by_cid["client_2"]["score"] == pytest.approx(0.9)
    # strict score > cut (cut = 0.5): equal-to-cut is NOT a would-flag
    assert by_cid["client_0"]["would_flag"] is False
    assert by_cid["client_1"]["would_flag"] is False
    assert by_cid["client_2"]["would_flag"] is True
    for r in rows:
        assert r["server_round"] == 2
        assert r["scenario_round"] == 1  # round_offset 1


@pytest.mark.unit
def test_observe_rows_accumulate_across_rounds():
    det = _detector(observe_only=True)
    det.score_updates(_results(_SPEC), server_round=2)
    det.score_updates(_results(_SPEC), server_round=3)
    rows = det.observe_rows
    assert len(rows) == 6
    assert sorted({r["server_round"] for r in rows}) == [2, 3]


@pytest.mark.unit
def test_observe_rows_are_a_defensive_copy():
    det = _detector(observe_only=True)
    det.score_updates(_results(_SPEC), server_round=2)
    rows = det.observe_rows
    rows[0]["score"] = -1.0
    assert det.observe_rows[0]["score"] == pytest.approx(0.4)


@pytest.mark.unit
def test_enforcing_mode_is_byte_identical_and_logs_no_observe_rows():
    """OFF/absent: same drops, same flags, same per-round records as today —
    and the observe channel stays empty."""
    det = _detector(observe_only=False)
    results = _results(_SPEC)
    scores = det.score_updates(results, server_round=2)
    assert scores == {0: 1.0, 1: 1.0, 2: 0.0}
    kept = det.filter_updates(results, scores)
    assert [p.cid for p, _ in kept] == ["cid_a_0", "cid_b_1"]
    assert det.flagged_identities(2) == ("client_2",)
    assert det.observe_rows == []
    assert det.observe_only is False


@pytest.mark.unit
def test_observe_mode_window_state_matches_enforcing_mode():
    """Observe-only must score EXACTLY as today: identical probabilities on
    identical streams round after round (window features accumulate the same
    way — no branch touches feature derivation)."""
    obs, enf = _detector(True), _detector(False)
    for rnd in (2, 3, 4):
        obs.score_updates(_results(_SPEC), server_round=rnd)
        enf.score_updates(_results(_SPEC), server_round=rnd)
        assert obs._probabilities_by_round[rnd] == enf._probabilities_by_round[rnd]


@pytest.mark.unit
def test_observe_mode_default_off():
    det = OnlineH2PrimeDetectorPlugin(
        bundle=_bundle(_StubModel(_PROBS)), scenario_token="S3"
    )
    assert det.observe_only is False

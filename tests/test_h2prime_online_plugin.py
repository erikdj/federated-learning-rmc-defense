"""Online H2' detector plugin — scoring, window state, chain order (§ 7c-bis).

Covers: every identity-mapped client scored on the PRE-filter stream; strict
`> cut` drop decisions; fail-loud on unmapped clients / missing features;
window state accumulating across rounds; the frozen chain order under
PluggableStrategy (detector first, FP hard-drop before the aggregator,
empty-round semantics preserved: (None, {}) => global model unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, Status, ndarrays_to_parameters

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.h2prime_online import (  # noqa: E402
    OnlineH2PrimeDetectorPlugin,
    ServingBundle,
)
from h4_bundle_fixture import FROZEN_FEATURES  # noqa: E402


class _StubModel:
    """predict_proba driven by a {update_norm_rounded: prob} table.

    update_norm is column 0 of the frozen feature order, so a test can pin
    each client's probability through its parameter vector alone.
    """

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
    """spec: list of (cid, norm, train_loss). One 1-param client each, so
    update_norm == |norm| exactly."""
    out = []
    for cid, norm, train_loss in spec:
        params = ndarrays_to_parameters(
            [np.array([norm], dtype=np.float64)]
        )
        metrics = {"partition_id": int(cid.split("_")[-1]),
                   "train_loss": train_loss}
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(
            status=Status(code=Code.OK, message=""),
            parameters=params,
            num_examples=100,
            metrics=metrics,
        )
        out.append((proxy, fit))
    return out


def _detector(prob_by_norm, cuts=None, scenario="S3"):
    plugin = OnlineH2PrimeDetectorPlugin(
        bundle=_bundle(_StubModel(prob_by_norm), cuts=cuts),
        scenario_token=scenario,
    )
    return plugin


@pytest.mark.unit
def test_missing_scenario_cut_at_construction_refuses():
    from flowerfl.h2prime_online import H2PrimeBundleError

    with pytest.raises(H2PrimeBundleError, match="no serving cut"):
        OnlineH2PrimeDetectorPlugin(
            bundle=_bundle(_StubModel({}), cuts={"S0": 0.5, "S1": 0.5,
                                                 "S2": 0.5, "S3": 0.5,
                                                 "S4": 0.5}),
            scenario_token="S9",
        )


@pytest.mark.unit
def test_scores_every_client_and_drops_strictly_above_cut():
    det = _detector({1.0: 0.4, 2.0: 0.5, 3.0: 0.9})  # cut = 0.5
    results = _results([("cid_a_0", 1.0, 0.5), ("cid_b_1", 2.0, 0.5),
                        ("cid_c_2", 3.0, 0.5)])
    det.set_identity_map({"cid_a_0": "client_0", "cid_b_1": "client_1",
                          "cid_c_2": "client_2"})
    scores = det.score_updates(results, server_round=2)
    # prob 0.4 < cut -> keep; prob 0.5 == cut -> NOT flagged (strict >);
    # prob 0.9 > cut -> flagged.
    assert scores == {0: 1.0, 1: 1.0, 2: 0.0}
    kept = det.filter_updates(results, scores)
    assert [p.cid for p, _ in kept] == ["cid_a_0", "cid_b_1"]
    assert det.flagged_identities(2) == ("client_2",)


@pytest.mark.unit
def test_unmapped_client_refuses_loudly():
    det = _detector({1.0: 0.1})
    results = _results([("cid_a_0", 1.0, 0.5)])
    det.set_identity_map({})  # no identity for cid_a_0
    with pytest.raises(RuntimeError, match="no logical identity"):
        det.score_updates(results, server_round=2)


@pytest.mark.unit
def test_missing_train_loss_refuses_loudly():
    det = _detector({1.0: 0.1})
    results = _results([("cid_a_0", 1.0, None)])  # train_loss None
    det.set_identity_map({"cid_a_0": "client_0"})
    with pytest.raises(RuntimeError, match="train_loss"):
        det.score_updates(results, server_round=2)


@pytest.mark.unit
def test_window_state_accumulates_across_rounds():
    """Round 3's norm_variance over the trailing window must be non-zero once
    two rounds of differing norms have been seen — proving the per-client
    trailing state is maintained (and fed to the model)."""
    seen_vectors = []

    class _Recorder(_StubModel):
        def predict_proba(self, X):
            seen_vectors.append(np.asarray(X, dtype=float).copy())
            return super().predict_proba(X)

    det = OnlineH2PrimeDetectorPlugin(
        bundle=_bundle(_Recorder({1.0: 0.1, 2.0: 0.1})),
        scenario_token="S3",
    )
    det.set_identity_map({"cid_a_0": "client_0"})
    det.score_updates(_results([("cid_a_0", 1.0, 0.5)]), server_round=2)
    det.score_updates(_results([("cid_a_0", 2.0, 0.4)]), server_round=3)
    nv_col = FROZEN_FEATURES.index("norm_variance")
    assert seen_vectors[0][0][nv_col] == 0.0          # first round: no window
    assert seen_vectors[1][0][nv_col] == pytest.approx(
        float(np.var([1.0, 2.0]))
    )
    ls_col = FROZEN_FEATURES.index("loss_slope")
    assert seen_vectors[1][0][ls_col] != 0.0          # two losses -> slope


# ===========================================================================
# frozen chain order under PluggableStrategy
# ===========================================================================

class _DropAllPlugin:
    """A stand-in aggregator layer that rejects everything."""

    name = "KrumDefense"

    def on_round_start(self, *a):
        pass

    def observe_cohort(self, *a):
        pass

    def set_identity_map(self, m):
        pass

    def score_updates(self, results, server_round):
        self.scored = [str(p.cid) for p, _ in results]
        return {i: 0.0 for i in range(len(results))}

    def filter_updates(self, results, scores, threshold=0.0):
        return []

    def on_round_end(self, *a):
        pass


@pytest.mark.unit
def test_detector_scores_prefilter_stream_and_empty_round_returns_none():
    """Chain [detector, drop-all aggregator]: the detector sees the FULL
    pre-filter cohort; when zero updates survive, aggregate_fit returns
    (None, {}) — the global model is unchanged that round (measured EXP-052
    semantics, preserved not 'fixed')."""
    from flwr.server.strategy import FedAvg

    from flowerfl.byzantine_defense import PluggableStrategy

    det = _detector({1.0: 0.9, 2.0: 0.1})  # flags the norm-1.0 client
    dropall = _DropAllPlugin()
    strategy = PluggableStrategy(FedAvg(), plugins=[det, dropall])
    det.set_identity_map({"cid_a_0": "client_0", "cid_b_1": "client_1"})

    results = _results([("cid_a_0", 1.0, 0.5), ("cid_b_1", 2.0, 0.5)])
    aggregated, metrics = strategy.aggregate_fit(2, results, [])

    # Detector scored EVERY participant (pre-filter stream)...
    assert set(det._round_scores[2]) == {0, 1}
    # ...the downstream stage saw only the detector's survivors...
    assert dropall.scored == ["cid_b_1"]
    # ...and the empty kept set produced the unchanged-model return.
    assert aggregated is None and metrics == {}
    trace = strategy._h4_chain_trace[2]
    assert trace["kept_cids"] == []
    assert [s["plugin"] for s in trace["stages"]] == [
        "H2PrimeDetector", "KrumDefense"
    ]
    assert trace["stages"][0]["dropped_cids"] == ["cid_a_0"]


@pytest.mark.unit
def test_fp_hard_drop_happens_before_the_aggregator():
    """Chain [detector, FP, aggregator]: a session the FP layer hard-drops
    must never reach the aggregator stage's scoring input."""
    from flwr.server.strategy import FedAvg

    from flowerfl.byzantine_defense import PluggableStrategy

    class _FPHardDrop:
        name = "Fingerprint"

        def on_round_start(self, *a):
            pass

        def observe_cohort(self, results, server_round):
            self.observed = [str(p.cid) for p, _ in results]

        def set_identity_map(self, m):
            pass

        def score_updates(self, results, server_round):
            return {
                i: (0.0 if str(p.cid) == "cid_b_1" else 1.0)
                for i, (p, _) in enumerate(results)
            }

        def filter_updates(self, results, scores, threshold=0.0):
            return [
                (p, f) for i, (p, f) in enumerate(results)
                if scores.get(i, 1.0) > 0.0
            ]

        def on_round_end(self, *a):
            pass

    class _RecordingAggregator(_DropAllPlugin):
        def filter_updates(self, results, scores, threshold=0.0):
            return list(results)  # keep everything; we only record inputs

    det = _detector({1.0: 0.9, 2.0: 0.1, 3.0: 0.1})
    fp = _FPHardDrop()
    agg = _RecordingAggregator()
    strategy = PluggableStrategy(FedAvg(), plugins=[det, fp, agg])
    det.set_identity_map({"cid_a_0": "client_0", "cid_b_1": "client_1",
                          "cid_c_2": "client_2"})

    results = _results([("cid_a_0", 1.0, 0.5), ("cid_b_1", 2.0, 0.5),
                        ("cid_c_2", 3.0, 0.5)])
    aggregated, _ = strategy.aggregate_fit(2, results, [])

    assert fp.observed == ["cid_a_0", "cid_b_1", "cid_c_2"]  # full cohort
    assert agg.scored == ["cid_c_2"]  # detector dropped a, FP dropped b
    assert aggregated is not None
    trace = strategy._h4_chain_trace[2]
    assert trace["stages"][1]["plugin"] == "Fingerprint"
    assert trace["stages"][1]["dropped_cids"] == ["cid_b_1"]
    assert trace["kept_cids"] == ["cid_c_2"]

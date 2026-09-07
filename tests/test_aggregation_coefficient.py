"""Post-defense aggregation-coefficient instrumentation (H3 Step 2, GWU-9).

Why this exists
---------------
`flowerfl/scenario_strategy.py` has always logged
``effective_weight = float(fit_res.num_examples)`` — the **raw, pre-filter,
client-reported** row count. The real aggregation happens downstream in
``PluggableStrategy.aggregate_fit`` → ``self._base.aggregate_fit`` over the
**post-``filter_updates`` survivor set** (with soft-reweight plugins having
possibly rewritten ``num_examples`` along the way).

The locked H3 rejoin-success rule (`data/h3_constants.json`, variant 2B) is
defined on the *aggregation coefficient* — with the outcome class
``blocked: coefficient == 0``. `effective_weight` can never express that: it is
non-zero for every dispatched client, survivor or not. Per amendment v1.10
§ 5.1: *"no H3 result may be computed from `effective_weight=num_examples`
masquerading as a coefficient."*

These tests pin the NEW, additive field ``aggregation_coefficient``:
  * exactly ``0.0`` for a client the defense chain hard-dropped,
  * ``w_i / Σ_j w_j`` over the round's survivors otherwise (so the round's
    coefficients sum to 1.0),
  * ``None`` — never a fabricated number — when the base strategy is not the
    num-examples-weighted FedAvg the formula describes,
and assert that ``effective_weight`` keeps its legacy semantics byte-for-byte
(v4 readers, e.g. `scripts/krum_variance_forensics.py`, must not shift).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedMedian

from flowerfl.byzantine_defense import (
    ByzantineDefensePlugin,
    KrumDefensePlugin,
    PluggableStrategy,
    TGEnsemblePlugin,
    TrustScorePlugin,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _results(cids, num_examples=None, dim=8, seed=0):
    """Fake (client_proxy, fit_res) pairs with real flwr Parameters."""
    rng = np.random.default_rng(seed)
    out = []
    for i, cid in enumerate(cids):
        params = ndarrays_to_parameters([rng.normal(size=dim).astype(np.float32)])
        proxy = SimpleNamespace(cid=cid)
        ne = 10 if num_examples is None else num_examples[i]
        fit = SimpleNamespace(parameters=params, num_examples=ne, metrics={})
        out.append((proxy, fit))
    return out


class SpyPlugin(ByzantineDefensePlugin):
    """Identity filter that records the survivor set it was handed.

    Placed LAST in the chain, its input IS the post-chain survivor set, so the
    test can derive the expected coefficients independently of the strategy's
    own bookkeeping (rather than asserting the implementation against itself).
    """

    def __init__(self):
        self.seen = None

    @property
    def name(self) -> str:
        return "Spy"

    def score_updates(self, results, server_round):
        return {i: 1.0 for i in range(len(results))}

    def filter_updates(self, results, scores, threshold=0.0):
        self.seen = [(str(cp.cid), float(fr.num_examples)) for cp, fr in results]
        return results


class SoftReweightPlugin(ByzantineDefensePlugin):
    """Scales one client's num_examples — the incumbent soft-downweight shape
    (`flowerfl/cold_start_plugin.py:215-231`). The coefficient must follow the
    REWRITTEN weight, since that is what FedAvg actually aggregates on."""

    def __init__(self, target_cid: str, factor: float):
        self._target = target_cid
        self._factor = factor

    @property
    def name(self) -> str:
        return "SoftReweight"

    def score_updates(self, results, server_round):
        return {i: 1.0 for i in range(len(results))}

    def filter_updates(self, results, scores, threshold=0.0):
        out = []
        for cp, fr in results:
            if str(cp.cid) == self._target:
                fr = SimpleNamespace(
                    parameters=fr.parameters,
                    num_examples=max(1, int(fr.num_examples * self._factor)),
                    metrics=fr.metrics,
                )
            out.append((cp, fr))
        return out


class DropAllPlugin(ByzantineDefensePlugin):
    @property
    def name(self) -> str:
        return "DropAll"

    def score_updates(self, results, server_round):
        return {i: 0.0 for i in range(len(results))}

    def filter_updates(self, results, scores, threshold=0.0):
        return []


def _expected_from_spy(all_cids, spy):
    total = sum(w for _cid, w in spy.seen)
    exp = {cid: 0.0 for cid in all_cids}
    if total > 0:
        for cid, w in spy.seen:
            exp[cid] = w / total
    return exp


# ---------------------------------------------------------------------------
# the four defense configurations
# ---------------------------------------------------------------------------


def _config_plugins(name):
    """The four deployed defense configurations (plus the recording spy)."""
    if name == "krum":
        return [KrumDefensePlugin(dynamic_f=True)]
    if name == "trustscore":
        return [TrustScorePlugin()]
    if name == "tge":
        return [TGEnsemblePlugin(num_malicious=2, num_to_keep=5, ramp_rounds=999)]
    if name == "krumtge":
        return [
            KrumDefensePlugin(dynamic_f=True),
            TGEnsemblePlugin(num_malicious=2, num_to_keep=5, ramp_rounds=999),
        ]
    raise AssertionError(name)


@pytest.mark.parametrize("config", ["krum", "trustscore", "tge", "krumtge"])
def test_coefficients_sum_to_one_and_are_zero_for_dropped(config):
    """Across ALL FOUR defense configurations: survivors' coefficients sum to
    1.0 and every hard-dropped client's coefficient is exactly 0.0."""
    cids = [f"raw{i}" for i in range(8)]
    spy = SpyPlugin()
    strategy = PluggableStrategy(
        base_strategy=FedAvg(), plugins=_config_plugins(config) + [spy]
    )

    results = _results(cids, num_examples=[10, 20, 30, 40, 50, 60, 70, 80])
    strategy.aggregate_fit(2, results, [])

    coeffs = strategy.aggregation_coefficients(2)
    assert coeffs is not None, "FedAvg base must yield a computable coefficient"
    assert set(coeffs) == set(cids), (
        "every client that PARTICIPATED must carry a coefficient — a missing key "
        "is an unknown-provenance row and fails integrity gate (b)"
    )

    expected = _expected_from_spy(cids, spy)
    for cid in cids:
        assert coeffs[cid] == pytest.approx(expected[cid]), cid

    survivors = {cid for cid, _w in spy.seen}
    for cid in set(cids) - survivors:
        assert coeffs[cid] == 0.0, f"dropped client {cid} must be exactly 0.0"
    assert sum(coeffs.values()) == pytest.approx(1.0)


def test_krum_actually_drops_someone_so_the_zero_branch_is_exercised():
    """Guard against a vacuous parametrized pass: Multi-Krum with dynamic f on
    n=8 keeps m=max(1, 8-3-2)=3, so five clients MUST land at exactly 0.0."""
    cids = [f"raw{i}" for i in range(8)]
    strategy = PluggableStrategy(
        base_strategy=FedAvg(), plugins=[KrumDefensePlugin(dynamic_f=True)]
    )
    strategy.aggregate_fit(2, _results(cids), [])
    coeffs = strategy.aggregation_coefficients(2)
    zeros = [c for c, v in coeffs.items() if v == 0.0]
    assert len(zeros) == 5, f"expected 5 hard-dropped clients, got {zeros}"


def test_coefficient_follows_soft_reweight_not_the_reported_count():
    """A soft-downweighted client's coefficient must reflect the REWRITTEN
    num_examples FedAvg aggregates on, not the client's reported count."""
    cids = ["a", "b"]
    strategy = PluggableStrategy(
        base_strategy=FedAvg(),
        plugins=[SoftReweightPlugin(target_cid="a", factor=0.1)],
    )
    strategy.aggregate_fit(2, _results(cids, num_examples=[100, 100]), [])
    coeffs = strategy.aggregation_coefficients(2)
    # a: 10, b: 100  ->  10/110, 100/110
    assert coeffs["a"] == pytest.approx(10 / 110)
    assert coeffs["b"] == pytest.approx(100 / 110)


def test_all_filtered_round_records_all_zero_coefficients():
    """When a plugin filters out EVERY client, nothing is aggregated: every
    participant's coefficient is 0.0 (h3_constants.json `blocked`)."""
    cids = ["a", "b", "c"]
    strategy = PluggableStrategy(base_strategy=FedAvg(), plugins=[DropAllPlugin()])
    params, _metrics = strategy.aggregate_fit(2, _results(cids), [])
    assert params is None
    coeffs = strategy.aggregation_coefficients(2)
    assert coeffs == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_zero_total_weight_yields_zero_coefficients_not_nan(monkeypatch):
    """Degenerate Σw = 0: emit 0.0, never NaN/inf (which `_jsonify` would
    silently turn into null and hide)."""
    base = FedAvg()
    monkeypatch.setattr(base, "aggregate_fit", lambda r, res, f: (None, {}))
    strategy = PluggableStrategy(base_strategy=base, plugins=[])
    strategy.aggregate_fit(2, _results(["a", "b"], num_examples=[0, 0]), [])
    coeffs = strategy.aggregation_coefficients(2)
    assert coeffs == {"a": 0.0, "b": 0.0}


def test_non_fedavg_base_yields_none_never_a_fabricated_number():
    """The coefficient formula w_i/Σw is FedAvg's aggregation math. Under a
    base whose math differs (FedMedian/FedTrimmedAvg subclass FedAvg but do
    NOT weight this way), we log null rather than a number that is not the
    coefficient — the exact defect class this change exists to remove."""
    strategy = PluggableStrategy(base_strategy=FedMedian(), plugins=[])
    strategy.aggregate_fit(2, _results(["a", "b"]), [])
    assert strategy.aggregation_coefficients(2) is None


def test_stale_round_guard():
    """Coefficients are keyed to the round they were computed in. A caller
    asking for a different round gets None — never another round's numbers
    (the v1.17 cross-attribution lesson)."""
    strategy = PluggableStrategy(base_strategy=FedAvg(), plugins=[])
    assert strategy.aggregation_coefficients(2) is None  # nothing recorded yet
    strategy.aggregate_fit(2, _results(["a", "b"]), [])
    assert strategy.aggregation_coefficients(2) is not None
    assert strategy.aggregation_coefficients(3) is None


def test_empty_results_records_nothing():
    strategy = PluggableStrategy(base_strategy=FedAvg(), plugins=[])
    assert strategy.aggregate_fit(2, [], []) == (None, {})
    assert strategy.aggregation_coefficients(2) is None


def test_aggregate_fit_return_value_unchanged():
    """Purely additive bookkeeping: what Flower consumes is untouched."""
    strategy = PluggableStrategy(base_strategy=FedAvg(), plugins=[])
    params, _ = strategy.aggregate_fit(2, _results(["a", "b"]), [])
    assert params is not None

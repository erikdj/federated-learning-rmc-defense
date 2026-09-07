"""Identity-keying fidelity for defense plugins (methodology v1.15).

In Flower simulation the raw cid is stable per virtual client for the whole
run, and RMC cycle identities (client_N_newM) map back to the same physical
partition (flowerfl/scenario_strategy.py::LOGICAL_TO_PARTITION). Stateful
defenses must therefore key per-client state on the scenario's LOGICAL
identity supplied by ScenarioStrategy — otherwise an identity-resetting
adversary silently keeps its tenure/reputation across the reset, which
contradicts the RMC threat model: the server cannot link a new identity to
an old client (that linkage is exactly what H3's fingerprint registry is
being evaluated for).
"""
from types import SimpleNamespace

import numpy as np
from flwr.common import ndarrays_to_parameters

from flowerfl.byzantine_defense import TGEnsemblePlugin, TrustScorePlugin


def _results(cids, dim=8, seed=0):
    """Fake (client_proxy, fit_res) pairs with real flwr Parameters."""
    rng = np.random.default_rng(seed)
    out = []
    for cid in cids:
        params = ndarrays_to_parameters([rng.normal(size=dim).astype(np.float32)])
        proxy = SimpleNamespace(cid=cid)
        fit = SimpleNamespace(parameters=params, num_examples=10, metrics={})
        out.append((proxy, fit))
    return out


FOUR = {"rawA": "client_0", "rawB": "client_1", "rawC": "client_2", "rawD": "client_3"}
FOUR_RESET = {**FOUR, "rawA": "client_0_new1"}


class TestTGEIdentityKeying:
    def test_identity_map_overrides_raw_cid(self):
        p = TGEnsemblePlugin()
        p.set_identity_map({"raw1": "client_3"})
        assert p._get_client_id("raw1") == "client_3"

    def test_fallback_without_map_is_legacy_counter(self):
        p = TGEnsemblePlugin()
        assert p._get_client_id("raw1") == "flower_client_0"
        assert p._get_client_id("raw1") == "flower_client_0"
        assert p._get_client_id("raw2") == "flower_client_1"

    def test_tenure_resets_on_identity_reset(self):
        p = TGEnsemblePlugin(warmup_rounds=1)
        results = _results(list(FOUR))
        p.set_identity_map(FOUR)
        for rnd in (1, 2, 3):
            p.score_updates(results, rnd)
        assert p._model.get_tenure("client_0", 3) == 3
        # Identity reset: the same raw cid returns under a new logical identity.
        p.set_identity_map(FOUR_RESET)
        p.score_updates(results, 4)
        assert p._model.get_tenure("client_0_new1", 4) == 1  # cold again


class TestTrustScoreIdentityKeying:
    def test_ema_resets_on_identity_reset(self):
        p = TrustScorePlugin()
        results = _results(list(FOUR), seed=1)
        p.set_identity_map(FOUR)
        for rnd in (1, 2, 3):
            p.score_updates(results, rnd)
        assert "client_0" in p._trust_scores
        p.set_identity_map(FOUR_RESET)
        p.score_updates(results, 4)
        assert "client_0_new1" in p._trust_scores
        # The raw cid must never be a reputation key when a map is present.
        assert "rawA" not in p._trust_scores

    def test_fallback_without_map_keys_raw_cid(self):
        p = TrustScorePlugin()
        results = _results(list(FOUR), seed=2)
        p.score_updates(results, 1)
        assert "rawA" in p._trust_scores


class _StubDetector:
    """Duck-typed stand-in for ColdStartDetector: constant medium risk."""

    def predict_risk(self, features):
        import numpy as np

        return np.full(len(features), 0.5)


class TestColdStartIdentityKeying:
    def _plugin(self):
        from flowerfl.cold_start_plugin import ColdStartDefensePlugin

        return ColdStartDefensePlugin(detector=_StubDetector(), k=3)

    def test_tenure_resets_on_identity_reset(self):
        p = self._plugin()
        results = _results(list(FOUR), dim=16)
        p.set_identity_map(FOUR)
        for rnd in (1, 2, 3, 4, 5):
            p.score_updates(results, rnd)
        # client_0 is past the k=3 window -> established, full trust
        assert p._first_seen["client_0"] == 1
        # Identity reset: same raw cid, new logical identity -> cold again
        p.set_identity_map(FOUR_RESET)
        p.score_updates(results, 6)
        assert p._first_seen["client_0_new1"] == 6
        # The raw cid must never be a tenure key when a map is present
        assert "rawA" not in p._first_seen

    def test_scoring_log_records_logical_identity(self):
        p = self._plugin()
        results = _results(list(FOUR), dim=16)
        p.set_identity_map(FOUR_RESET)
        p.score_updates(results, 1)
        logged_ids = {cid for _, cid, _ in p._scoring_log}
        assert "client_0_new1" in logged_ids
        assert "rawA" not in logged_ids

    def test_fallback_without_map_keys_raw_cid(self):
        p = self._plugin()
        results = _results(list(FOUR), dim=16)
        p.score_updates(results, 1)
        assert "rawA" in p._first_seen


class TestScenarioStrategyIdentityMap:
    def test_build_identity_map_resolves_logical_ids(self):
        from flowerfl.scenario_strategy import ScenarioStrategy

        strat = ScenarioStrategy.__new__(ScenarioStrategy)  # skip heavy __init__
        strat._cid_to_partition = {"rawA": 0, "rawB": 1}
        strat._round_offset = 0
        strat._schedule_cache = {
            5: [
                {"partition_id": 0, "logical_id": "client_0_new1"},
                {"partition_id": 1, "logical_id": "client_1"},
            ]
        }
        results = _results(["rawA", "rawB"])
        m = strat._build_identity_map(5, results)
        assert m == {"rawA": "client_0_new1", "rawB": "client_1"}

    def test_build_identity_map_skips_unmapped_cids(self):
        from flowerfl.scenario_strategy import ScenarioStrategy

        strat = ScenarioStrategy.__new__(ScenarioStrategy)
        strat._cid_to_partition = {"rawA": 0}
        strat._round_offset = 0
        strat._schedule_cache = {1: [{"partition_id": 7, "logical_id": "client_7"}]}
        m = strat._build_identity_map(1, _results(["rawA", "rawZ"]))
        assert m == {}

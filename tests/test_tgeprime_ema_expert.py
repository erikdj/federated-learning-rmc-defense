"""TGE′ two-leg long-memory BANK — unit + model-integration tests (GWU-53).

TGE′ keeps the incumbent LSTM long-memory expert and ADDS a second leg — an EMA
reputation expert over the cold-start (IsolationForest) score s_cs(t) — combined
by the zero-parameter fail-safe min(). These tests pin, TDD-first:

1. The EMA update recurrence incl. absence decay (hand-computed fixture) — the
   RMCDetectionPlugin._temporal_score mechanism with s_cs as the input.
2. Reset re-initialisation: a fresh logical id starts from r_init (0.85, the
   ensemble's neutral no-evidence value), independent of any prior identity.
3. The bank combiner: gate long-memory leg = min(lstm_score, ema_score).
4. The incumbent is bit-identical by default (long_memory_expert="lstm";
   self.ema is None, no new code path runs).
5. ema_alpha / long_memory_expert validation fails loudly.
6. Both raw leg scores are exposed so the combiner is reconstructable offline.
7. Geometric fallback stays re-derivable offline from logged Family-S signals.
8. Regression: honest clients are not mass-filtered while the EMA matures — r_init is the neutral 0.85, so a fresh EMA leg never drags an
   honest client below the 0.7 operational cutoff.

Design notes: ema_alpha is the RETENTION weight on prior reputation (0.9 ADOPTED
from TrustScore); input = s_cs (the fitted IsolationForest's cold-start score),
accumulated only from forest-fit onward (null before); absence gap is tracked
internally from last-seen round, so no scenario_manager is required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rmc.tg_ensemble import (  # noqa: E402
    EMAReputationExpert,
    LSTMTemporalExpert,
    TGEnsembleModel,
)


# ---------------------------------------------------------------------------
# 1. EMA update math incl. absence decay (hand-computed)
# ---------------------------------------------------------------------------

def test_ema_recurrence_with_absence_decay_hand_computed():
    """R = alpha*R_prev + (1-alpha)*s_cs, with R_prev *= absence_decay**gap
    for any missed rounds. alpha=0.5, absence_decay=0.5, r_init=0.5:

      update(A, s=1.0, r=1): R_prev=0.5, gap=0 -> R = 0.5*0.5 + 0.5*1.0 = 0.75
      update(A, s=0.0, r=2): R_prev=0.75, gap=0 -> R = 0.5*0.75 + 0.5*0.0 = 0.375
      update(A, s=1.0, r=5): gap = 5-2-1 = 2 -> R_prev = 0.375*0.5**2 = 0.09375
                             R = 0.5*0.09375 + 0.5*1.0 = 0.546875
    """
    ema = EMAReputationExpert(ema_alpha=0.5, absence_decay=0.5, r_init=0.5)
    assert ema.score("A") is None  # no update yet

    assert ema.update("A", 1.0, 1) == pytest.approx(0.75)
    assert ema.update("A", 0.0, 2) == pytest.approx(0.375)
    assert ema.update("A", 1.0, 5) == pytest.approx(0.546875)
    assert ema.reputation("A") == pytest.approx(0.546875)


def test_first_join_starts_from_r_init_no_decay():
    ema = EMAReputationExpert(ema_alpha=0.9, absence_decay=0.9, r_init=0.5)
    # R = 0.9*0.5 + 0.1*1.0 = 0.55 (memory-heavy: 90% retention)
    assert ema.update("A", 1.0, 7) == pytest.approx(0.55)


def test_ema_alpha_and_absence_decay_validated():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ema_alpha"):
            EMAReputationExpert(ema_alpha=bad)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="absence_decay"):
            EMAReputationExpert(absence_decay=bad)
    # absence_decay == 1.0 is allowed (no decay).
    EMAReputationExpert(absence_decay=1.0)


# ---------------------------------------------------------------------------
# 2. Identity reset -> fresh reputation
# ---------------------------------------------------------------------------

def test_fresh_logical_id_reinitialises_from_r_init():
    ema = EMAReputationExpert(ema_alpha=0.5, absence_decay=0.5, r_init=0.5)
    for r in range(1, 7):
        ema.update("client_3", 1.0, r)  # honest, converges toward 1.0
    established = ema.reputation("client_3")
    assert established > 0.9

    # Reconnect under a fresh logical id -> starts from r_init, no inheritance.
    r_new = ema.update("client_3_new1", 1.0, 7)
    assert r_new == pytest.approx(0.75)  # 0.5*0.5 + 0.5*1.0, one step from r_init
    assert r_new < established
    assert ema.reputation("client_3") == pytest.approx(established)  # untouched


def test_score_is_none_until_first_update():
    ema = EMAReputationExpert()
    assert ema.score("X") is None
    assert ema.has_update("X") is False
    ema.update("X", 0.9, 1)
    assert ema.has_update("X") is True
    assert ema.score("X") == pytest.approx(ema.reputation("X"))


# ---------------------------------------------------------------------------
# 3. Model selection: incumbent untouched, TGE′ opt-in, three modes
# ---------------------------------------------------------------------------

def test_default_model_is_incumbent_lstm_only():
    model = TGEnsembleModel()
    assert model.long_memory_expert == "lstm"
    assert isinstance(model.lstm, LSTMTemporalExpert)
    assert model.ema is None
    assert model.long_memory_combiner == "lstm"


def test_ema_mode_adds_reputation_leg_keeps_lstm():
    model = TGEnsembleModel(long_memory_expert="ema", ema_alpha=0.9)
    assert isinstance(model.lstm, LSTMTemporalExpert)  # LSTM kept
    assert isinstance(model.ema, EMAReputationExpert)
    assert model.ema.ema_alpha == pytest.approx(0.9)
    assert model.long_memory_combiner == "ema"


def test_bank_mode_has_both_legs_and_min_combiner():
    model = TGEnsembleModel(long_memory_expert="bank", ema_alpha=0.9)
    assert isinstance(model.lstm, LSTMTemporalExpert)
    assert isinstance(model.ema, EMAReputationExpert)
    assert model.long_memory_combiner == "min"


def test_model_rejects_unknown_mode_and_bad_alpha():
    with pytest.raises(ValueError, match="long_memory_expert"):
        TGEnsembleModel(long_memory_expert="transformer")
    # alpha validated loudly even in lstm mode
    with pytest.raises(ValueError, match="ema_alpha"):
        TGEnsembleModel(ema_alpha=0.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        TGEnsembleModel(long_memory_expert="bank", ema_alpha=1.0)


# ---------------------------------------------------------------------------
# 4. Bank combiner semantics
# ---------------------------------------------------------------------------

def test_long_memory_score_combiner_per_mode():
    bank = TGEnsembleModel(long_memory_expert="bank")
    assert bank._long_memory_score(0.8, 0.3) == pytest.approx(0.3)  # min
    assert bank._long_memory_score(0.2, 0.9) == pytest.approx(0.2)

    ema = TGEnsembleModel(long_memory_expert="ema")
    assert ema._long_memory_score(0.8, 0.3) == pytest.approx(0.3)  # ema leg

    lstm = TGEnsembleModel()  # incumbent: ignores ema entirely
    assert lstm._long_memory_score(0.8, None) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 5. Incumbent bit-identical when long_memory_expert="lstm" (default)
# ---------------------------------------------------------------------------

def _benign_features(rng: np.random.Generator) -> np.ndarray:
    feats = rng.normal(0.0, 0.05, 12)
    feats[0] = abs(feats[0])
    feats[1] = 0.9 + feats[1] * 0.1
    feats[2] = abs(feats[2])
    return feats.astype(np.float64)


def _run(model: TGEnsembleModel, clients, rounds=8, seed=0):
    rng = np.random.default_rng(seed)
    scores, last_details = [], {}
    for server_round in range(1, rounds + 1):
        feats = {cid: _benign_features(rng) for cid in clients}
        # Cohort observation drives the EMA update (mirrors the plugin hook);
        # score_client below only reads. lstm mode: observe_ema is a no-op.
        for cid in clients:
            model.observe_ema(cid, feats[cid], server_round)
        for cid in clients:
            s, details = model.score_client(cid, feats[cid], server_round)
            model.record_scored(cid, server_round)
            model.record_accepted(cid, server_round)
            scores.append(round(s, 6))
            last_details[cid] = details
        model.on_round_end(server_round)
    return scores, last_details


def test_default_matches_explicit_lstm_bit_identical():
    clients = [f"c{i}" for i in range(6)]
    a, _ = _run(TGEnsembleModel(num_features=12, seed=42), clients)
    b, _ = _run(TGEnsembleModel(num_features=12, long_memory_expert="lstm",
                                ema_alpha=0.9, seed=42), clients)
    assert a == b


# ---------------------------------------------------------------------------
# 6. Bank active-phase: both leg scores logged + final = gate(min) reconstruct
# ---------------------------------------------------------------------------

def _run_capture(model: TGEnsembleModel, clients, rounds, cid_watch, seed=0):
    """Run and return the per-round details for one watched client."""
    rng = np.random.default_rng(seed)
    per_round = {}
    for server_round in range(1, rounds + 1):
        feats = {cid: _benign_features(rng) for cid in clients}
        for cid in clients:
            model.observe_ema(cid, feats[cid], server_round)  # cohort EMA update
        for cid in clients:
            _, details = model.score_client(cid, feats[cid], server_round)
            model.record_scored(cid, server_round)
            model.record_accepted(cid, server_round)
            if cid == cid_watch:
                per_round[server_round] = details
        model.on_round_end(server_round)
    return per_round


def test_bank_active_phase_logs_both_legs_and_final_is_gate_of_min():
    model = TGEnsembleModel(num_features=12, long_memory_expert="bank",
                            ema_alpha=0.9, seed=7)
    clients = [f"client_{i}" for i in range(6)]
    _, last_details = _run(model, clients, rounds=7)
    for cid, d in last_details.items():
        assert d["phase"] == "active", f"{cid}: phase={d['phase']!r}"
        assert d["gbdt_score"] is not None
        assert d["lstm_score"] is not None  # LSTM ready in the both-legs window
        assert d["ema_score"] is not None, f"{cid}: EMA leg score is None in active"
        long_mem = min(d["lstm_score"], d["ema_score"])
        expected = model.gate.compute_score(d["gbdt_score"], long_mem, d["tenure"])
        assert d["final_score"] == pytest.approx(round(expected, 4), abs=1e-3)


def test_ema_accumulates_from_forest_fit_onward_null_before():
    """The EMA leg does not fold in the neutral pre-fit cold-start placeholder:
    tge_ema_score is null during warmup and pre_gbdt (forest not fitted), and
    becomes non-null only from the forest-fit round onward. warmup_rounds=3 ->
    the IsolationForest fits at end of round 3, so the EMA first updates at
    round 4."""
    model = TGEnsembleModel(num_features=12, long_memory_expert="bank",
                            ema_alpha=0.9, seed=1)
    clients = [f"c{i}" for i in range(6)]
    per_round = _run_capture(model, clients, rounds=7, cid_watch="c0", seed=1)
    # rounds 1..3 warmup, gbdt not yet fitted -> EMA null
    for r in (1, 2, 3):
        assert per_round[r]["ema_score"] is None, f"round {r}: EMA should be null pre-fit"
    # from round 4 (forest fitted) the EMA has accumulated
    assert per_round[4]["ema_score"] is not None, "round 4: EMA should update post-fit"


def test_bank_degrades_to_ema_leg_alone_before_lstm_ready():
    """Bank window: forest fitted (round 4) but LSTM not yet fitted (fits end of
    round 5, ready round 6). In rounds 4-5 the gate's long-memory leg is the EMA
    ALONE — phase 'active', lstm_score null, final = gate(gbdt, ema, tenure)."""
    model = TGEnsembleModel(num_features=12, long_memory_expert="bank",
                            ema_alpha=0.9, seed=5)
    clients = [f"c{i}" for i in range(6)]
    per_round = _run_capture(model, clients, rounds=7, cid_watch="c0", seed=5)
    for r in (4, 5):
        d = per_round[r]
        assert d["phase"] == "active", f"round {r}: expected degraded-active; got {d['phase']!r}"
        assert d["lstm_score"] is None, f"round {r}: LSTM not ready, lstm_score must be null"
        assert d["ema_score"] is not None, f"round {r}: EMA leg must carry the long-memory score"
        expected = model.gate.compute_score(d["gbdt_score"], d["ema_score"], d["tenure"])
        assert d["final_score"] == pytest.approx(round(expected, 4), abs=1e-3)
    # by round 6 the LSTM is ready -> both legs present (min combiner)
    assert per_round[6]["lstm_score"] is not None
    assert per_round[6]["ema_score"] is not None


def test_lstm_mode_details_carry_null_ema_score():
    model = TGEnsembleModel(num_features=12, seed=3)
    clients = [f"c{i}" for i in range(6)]
    _, last_details = _run(model, clients, rounds=7)
    for cid, d in last_details.items():
        assert d["ema_score"] is None  # no EMA leg in incumbent mode


# ---------------------------------------------------------------------------
# 8. Regression: honest clients are NOT mass-filtered while the EMA matures
#    (GWU-53). r_init is the neutral no-evidence value (0.85), so a
#    fresh EMA leg never drags an honest client below the 0.7 operational cutoff.
# ---------------------------------------------------------------------------

OPERATIONAL_THRESHOLD = 0.7  # TGEnsemblePlugin._threshold


def test_default_r_init_is_neutral_not_trustscore_half():
    """r_init must be the ensemble's neutral 0.85, not TrustScore's 0.5 —
    a below-cutoff prior would filter honest clients."""
    assert EMAReputationExpert().r_init == pytest.approx(0.85)


def test_bank_ramp3_honest_clients_not_filtered_while_ema_matures():
    """The exact  scenario: bank, ema_alpha=0.9, ramp_rounds=3, honest
    clients. With ramp=3 the gate uses PURE long-memory from tenure 3 (round 4),
    and in rounds 4-5 that is the EMA leg ALONE (LSTM not yet ready). Every
    round's final score for an established honest client must stay >= the 0.7
    operational threshold, so the round is aggregated — no mass-filtering, no
    all-rejected round that would skip on_round_end and stall the LSTM fit.
    """
    model = TGEnsembleModel(num_features=12, long_memory_expert="bank",
                            ema_alpha=0.9, ramp_rounds=3, warmup_rounds=3, seed=42)
    clients = [f"c{i}" for i in range(6)]
    per_round = _run_capture(model, clients, rounds=11, cid_watch="c0", seed=42)

    # rounds 4-11 are the post-warmup scoring rounds; established honest clients
    # must never be filtered (the P1 regression window is rounds 4-9).
    for r in range(4, 12):
        d = per_round[r]
        assert d["phase"] == "active", f"round {r}: phase={d['phase']!r}"
        assert d["final_score"] >= OPERATIONAL_THRESHOLD, (
            f"round {r}: honest client filtered (final={d['final_score']} < "
            f"{OPERATIONAL_THRESHOLD}) — honest-client mass-filter regression"
        )
    # rounds 4-5 exercise the degraded EMA-only window (LSTM not yet ready).
    assert per_round[4]["lstm_score"] is None and per_round[4]["ema_score"] is not None
    # by round 6 the LSTM has fitted (aggregation ran, so on_round_end ran).
    assert model.lstm.is_ready
    assert per_round[6]["lstm_score"] is not None


# ---------------------------------------------------------------------------
# 7. Geometric-EMA remains re-derivable offline from the logged Family-S signals
# ---------------------------------------------------------------------------

def test_geometric_fallback_recomputable_from_logged_signals():
    """The geometric-EMA (rejected as the online input) stays available as an
    offline sensitivity: the per-row Family-S signals the signal log records
    (update_norm, cos_to_median, L2_to_median) suffice to recompute
    geometric_fallback_score, matching the online TGE feature path."""
    from rmc.tg_ensemble import extract_geometric_features, geometric_fallback_score
    from flowerfl.signal_logger import compute_per_client_signals

    rng = np.random.default_rng(11)
    flat_updates = [rng.normal(0, 1, 40).astype(np.float64) for _ in range(8)]
    flat_updates[3] *= 4.0  # one outlier so z-distance / norm-dev are non-trivial

    signals = compute_per_client_signals(
        flat_updates, train_losses=[None] * 8, num_examples_list=[100] * 8
    )
    # Round-level aggregates recomputable from the logged rows.
    l2s = np.array([s["L2_to_median"] for s in signals])
    norms = np.array([s["update_norm"] for s in signals])
    mean_d, std_d = l2s.mean(), l2s.std()
    median_norm = np.median(norms)

    for i in range(8):
        z_distance = (l2s[i] - mean_d) / std_d if std_d > 1e-12 else 0.0
        cos_sim = signals[i]["cos_to_median"]
        norm_dev = abs(norms[i] - median_norm) / median_norm if median_norm > 1e-12 else 0.0
        recomputed = geometric_fallback_score(np.array([z_distance, cos_sim, norm_dev]))

        online = geometric_fallback_score(
            extract_geometric_features(flat_updates[i], flat_updates)
        )
        assert recomputed == pytest.approx(online, abs=1e-4), f"client {i}"


# ---------------------------------------------------------------------------
# 9. EMA absence decay + genuine-gap decay (EMA-level math)
# ---------------------------------------------------------------------------

def test_update_advances_participation_no_decay_when_contiguous():
    """A client updated every round accrues no absence decay (gap 0)."""
    ema = EMAReputationExpert(ema_alpha=0.9, absence_decay=0.5, r_init=0.85)
    ema.update("A", 0.9, 2)               # R2 = 0.9*0.85 + 0.1*0.9 = 0.855
    r3 = ema.update("A", 0.9, 3)          # contiguous -> gap 0
    assert r3 == pytest.approx(0.9 * 0.855 + 0.1 * 0.9)


def test_genuine_absence_still_decays():
    """A client that genuinely did NOT participate for rounds still decays."""
    ema = EMAReputationExpert(ema_alpha=0.9, absence_decay=0.5, r_init=0.85)
    ema.update("A", 0.9, 2)               # R2 = 0.855
    r5 = ema.update("A", 0.9, 5)          # gap = 5-2-1 = 2 -> decay 0.5**2 = 0.25
    assert r5 == pytest.approx(0.9 * (0.855 * 0.25) + 0.1 * 0.9)  # 0.282375


# ---------------------------------------------------------------------------
# 10. Cohort-observation hook: the EMA is a once-per-
#     participating-round reputation, cohort-wide, independent of upstream
#     Krum filtering — with hard isolation from every other expert and the
#     scored-row signal-log population.
# ---------------------------------------------------------------------------

def _model_fitted_gbdt(**kwargs):
    """A model whose IsolationForest is fitted, so observe_ema/score are live."""
    m = TGEnsembleModel(num_features=12, warmup_rounds=1, **kwargs)
    rng = np.random.default_rng(0)
    for _ in range(8):
        m.gbdt.accumulate(rng.normal(0.0, 0.05, 12).astype(np.float64))
    m.gbdt.fit(server_round=1)
    assert m.gbdt.is_ready
    return m


def test_filtered_client_reputation_evolves_with_cohort_evidence():
    """A client observed every round via the cohort hook but NEVER scored by TGE
    (always Krum-filtered) must still have its reputation evolve, not freeze."""
    m = _model_fitted_gbdt(long_memory_expert="bank", ema_alpha=0.9)
    rng = np.random.default_rng(3)
    reps = []
    for r in range(2, 9):
        m.observe_ema("cF", _benign_features(rng), r)  # filtered: only observed
        reps.append(m.ema.reputation("cF"))
    assert all(x is not None for x in reps)               # evolves, never frozen null
    assert len({round(x, 6) for x in reps}) > 1           # and actually moves


def test_survivor_ema_updated_exactly_once_per_round():
    """observe_ema is the sole EMA update; score_client only READS (no double
    count for a survivor observed then scored in the same round)."""
    m = _model_fitted_gbdt(long_memory_expert="bank", ema_alpha=0.9)
    feats = _benign_features(np.random.default_rng(2))
    m.observe_ema("cS", feats, 3)
    n_updates = m.ema._n_updates
    rep = m.ema.reputation("cS")
    m.score_client("cS", feats, 3)          # scoring path must not re-update
    assert m.ema._n_updates == n_updates
    assert m.ema.reputation("cS") == pytest.approx(rep)


def test_observe_ema_isolation_from_other_experts_and_log():
    """observe_ema touches ONLY the EMA — not the IsolationForest training
    buffer, LSTM history, tenure counters, or _last_details."""
    m = _model_fitted_gbdt(long_memory_expert="bank", ema_alpha=0.9)
    gbdt_buf = len(m.gbdt._feature_buffer)
    m.observe_ema("cF", _benign_features(np.random.default_rng(4)), 3)
    assert len(m.gbdt._feature_buffer) == gbdt_buf        # GBDT buffer untouched
    assert "cF" not in m.lstm._history                    # LSTM history untouched
    assert "cF" not in m._first_seen                       # tenure untouched
    assert m.ema.reputation("cF") is not None              # but the EMA did update


def test_observe_ema_noop_before_forest_fit_and_in_lstm_mode():
    # Before the forest fits, s_cs is only the neutral placeholder -> no update.
    m = TGEnsembleModel(num_features=12, long_memory_expert="bank", warmup_rounds=1)
    assert not m.gbdt.is_ready
    m.observe_ema("cF", _benign_features(np.random.default_rng(5)), 2)
    assert m.ema.reputation("cF") is None
    # lstm mode has no EMA leg — observe_ema is a no-op, never raises.
    _model_fitted_gbdt().observe_ema("cF", _benign_features(np.random.default_rng(6)), 3)


def test_observe_cohort_does_not_change_scored_row_population():
    """Population invariance (hard requirement): observing the FULL cohort must
    not inject Krum-filtered clients into the scored-row set — _last_details
    (the signal-log join source) stays exactly the clients TGE scored."""
    from types import SimpleNamespace
    from flwr.common import ndarrays_to_parameters

    def _results(cids, dim=12, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for cid in cids:
            params = ndarrays_to_parameters([rng.normal(0, 0.05, dim).astype(np.float32)])
            out.append((SimpleNamespace(cid=cid),
                        SimpleNamespace(parameters=params, num_examples=10, metrics={})))
        return out

    from flowerfl.byzantine_defense import TGEnsemblePlugin
    plugin = TGEnsemblePlugin(long_memory_expert="bank", ema_alpha=0.9, warmup_rounds=1)
    cohort_cids = ["r0", "r1", "r2", "r3", "r4", "r5"]
    plugin.set_identity_map({c: f"c{i}" for i, c in enumerate(cohort_cids)})
    plugin.on_round_start(5, len(cohort_cids))
    full = _results(cohort_cids, seed=5)
    survivors = full[:4]  # r4, r5 were filtered by the (notional) upstream Krum
    plugin.observe_cohort(full, 5)          # observes ALL six
    plugin.score_updates(survivors, 5)      # scores only the four survivors
    assert set(plugin._last_details.keys()) == {"r0", "r1", "r2", "r3"}, (
        "observe_cohort must not add filtered clients to the scored-row population"
    )


def test_cohort_observation_time_folded_into_tge_timing():
    """observe_cohort's full-cohort work must be counted in the
    reported per-round TGE time, not omitted (it currently runs before the
    score_updates timer)."""
    from types import SimpleNamespace
    from flwr.common import ndarrays_to_parameters
    from flowerfl.byzantine_defense import TGEnsemblePlugin

    def _res(cids, seed=0):
        rng = np.random.default_rng(seed)
        return [
            (SimpleNamespace(cid=c),
             SimpleNamespace(
                 parameters=ndarrays_to_parameters([rng.normal(0, 0.01, 12).astype(np.float32)]),
                 num_examples=10, metrics={}))
            for c in cids
        ]

    plugin = TGEnsemblePlugin(long_memory_expert="bank", ema_alpha=0.9, warmup_rounds=1)
    cohort = ["r0", "r1", "r2", "r3", "r4", "r5"]
    idmap = {c: f"c{i}" for i, c in enumerate(cohort)}

    # Round 1 warmup fits the forest at on_round_end; observe is a pre-fit no-op.
    plugin.on_round_start(1, len(cohort))
    plugin.set_identity_map(idmap)
    full1 = _res(cohort, seed=1)
    plugin.observe_cohort(full1, 1)
    assert plugin._pending_observe_ms == 0.0
    scores1 = plugin.score_updates(full1, 1)
    plugin.filter_updates(full1, scores1)
    plugin.on_round_end(1, None)
    assert plugin._model.gbdt.is_ready

    # Round 2: observe_cohort does real work -> records its time; score_updates
    # folds it into the round's recorded TGE time and resets the pending value.
    plugin.on_round_start(2, len(cohort))
    plugin.set_identity_map(idmap)
    full2 = _res(cohort, seed=2)
    plugin.observe_cohort(full2, 2)
    pending = plugin._pending_observe_ms
    assert pending > 0.0, "observe_cohort must record its wall-clock time"
    plugin.score_updates(full2[:4], 2)      # only survivors scored
    assert plugin._pending_observe_ms == 0.0                 # folded + reset
    assert plugin._timing_per_round[-1] >= pending           # includes observe time

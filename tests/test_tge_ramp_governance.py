"""Amendment v1.6 tenure-ramp governance tests.

Locks three code-level guarantees used by the H2 workflow in
docs/reproduction/experiments.md:

1. The CONFIGURED ramp is authoritative — no silent `max(ramp, 8)` clamp.
   (A dev-gate-selected ramp below 8 must actually deploy, or the historical
   selection protocol is a no-op.)
2. The provisional canonical default is 8 everywhere (model, rule, plugin,
   runner config), and invalid ramps fail loudly instead of being corrected.
3. Active-phase detail dicts carry BOTH raw expert scores non-null — the
   instrumentation contract the § 3.3 offline re-blend depends on.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rmc.tg_ensemble import TenureGatedDecisionRule, TGEnsembleModel


# ---------------------------------------------------------------------------
# 1. Configured ramp is authoritative (clamp removal)
# ---------------------------------------------------------------------------

def test_model_honors_configured_ramp_below_8():
    """A ramp below the old floor must survive construction unchanged —
    otherwise a dev-gate-selected ramp (v1.6 § 3.4) could never deploy."""
    model = TGEnsembleModel(ramp_rounds=5)
    assert model.gate.ramp_rounds == 5


def test_model_default_ramp_is_provisional_canonical_8():
    assert TGEnsembleModel().gate.ramp_rounds == 8


def test_rule_default_ramp_is_provisional_canonical_8():
    assert TenureGatedDecisionRule().ramp_rounds == 8


def test_plugin_passes_configured_ramp_through():
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    plugin = TGEnsemblePlugin(num_malicious=2, num_to_keep=5, ramp_rounds=5)
    assert plugin._model.gate.ramp_rounds == 5


def test_plugin_default_ramp_is_provisional_canonical_8():
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    plugin = TGEnsemblePlugin(num_malicious=2, num_to_keep=5)
    assert plugin._model.gate.ramp_rounds == 8




def test_pyproject_flwr_run_default_ramp_is_provisional_canonical_8():
    """pyproject.toml's tge-ramp-rounds feeds direct `flwr run` invocations (debug-only path, but it must not silently deploy a non-canonical ramp
    now that the model honors the configured value)."""
    import re
    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^tge-ramp-rounds\s*=\s*"(\d+)"', text, re.M)
    assert match is not None, "tge-ramp-rounds missing from pyproject.toml"
    assert match.group(1) == "8", (
        f"pyproject.toml tge-ramp-rounds must be the provisional canonical 8 "
        f"(v1.6 § 2); got {match.group(1)!r}"
    )


# ---------------------------------------------------------------------------
# 2. Invalid ramps fail loudly (replaces the silent correction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_ramp", [2, 1, 0, -1])
def test_ramp_at_or_below_min_tenure_raises(bad_ramp):
    with pytest.raises(ValueError, match="ramp_rounds"):
        TenureGatedDecisionRule(min_tenure=2, ramp_rounds=bad_ramp)


def test_model_propagates_ramp_validation():
    with pytest.raises(ValueError, match="ramp_rounds"):
        TGEnsembleModel(ramp_rounds=1)


# ---------------------------------------------------------------------------
# 3. Runner config + provenance
# ---------------------------------------------------------------------------

def test_runner_tge_configs_pass_gate_selected_3():
    """The runner deploys the GATE-SELECTED ramp (v1.6 § 3 rule, executed at the
    2026-07-23 dev gate: r*=3 — results/20260723/ramp_selection, commit 61ef015).
    Library defaults stay at the provisional canonical 8; the runner config is
    the deploy site and must carry the selected value."""
    from run_phase4_flower import build_strategy_for_config
    for config_name in ("TGE", "Krum+TGE"):
        _, plugin_config = build_strategy_for_config(config_name)
        assert plugin_config["tge_ramp_rounds"] == 3, (
            f"{config_name}: gate-selected ramp is 3 (v1.6 § 3, 2026-07-23); "
            f"got {plugin_config.get('tge_ramp_rounds')!r}"
        )


def test_provenance_reports_true_lstm_state_for_tge():
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields("ScenarioKrumTGE", {"tge-ramp-rounds": 8}, rounds=50)
    assert fields["tge_lstm_state"] == "enabled"
    assert fields["tge_ramp_rounds"] == 8


def test_provenance_gate_settings_track_code_not_copies():
    """ acceptance: result metadata carries expert type, min tenure, and
    operational threshold — sourced from the plugin/rule signatures so the
    provenance cannot drift from the deployed defaults."""
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields("ScenarioKrumTGE", {}, rounds=50)
    assert fields["tge_cold_start_expert"] == "isolation_forest"  # v1.5 § 4
    assert fields["tge_min_tenure"] == 2
    assert fields["tge_operational_threshold"] == pytest.approx(0.7)


def test_provenance_lstm_blend_semantics():
    """tge_lstm_state reflects whether the LSTM can contribute AT ALL: under
    the continuous gate its blend weight is non-zero for tenure > min_tenure
    once it fits (end of round warmup+2), even at ramp=999 — the old
    `ramp >= rounds -> disabled` rule wrote false provenance for
    GBDT-dominant runs. Pure-LSTM reach is a separate field."""
    from run_phase4_flower import tge_provenance_fields
    # legacy 999 on a 50-round run: LSTM still blends (~5% weight at high
    # tenure), only the pure-LSTM regime is unreachable
    f = tge_provenance_fields("ScenarioTGEnsemble", {"tge-ramp-rounds": 999}, rounds=50)
    assert f["tge_lstm_state"] == "enabled"
    assert f["tge_pure_lstm_reach"] is False
    assert f["tge_ramp_rounds"] == 999
    # canonical deployment: enabled, and pure-LSTM regime reachable
    f = tge_provenance_fields("ScenarioKrumTGE", {"tge-ramp-rounds": 8}, rounds=50)
    assert f["tge_lstm_state"] == "enabled"
    assert f["tge_pure_lstm_reach"] is True
    # BOUNDARIES: the runner adds a discovery round
    # (num-server-rounds = rounds + 1, run_phase4_flower.py), plugins first
    # score at server round 2, so max tenure = rounds and the LSTM's first
    # scoring round (server round warmup+3 = 6) exists iff rounds >= warmup+2.
    # rounds=5: server rounds 1..6 — LSTM fits end of round 5, scores round 6
    f = tge_provenance_fields("ScenarioTGEnsemble", {"tge-ramp-rounds": 8}, rounds=5)
    assert f["tge_lstm_state"] == "enabled"
    # rounds=4: server rounds 1..5 — LSTM fits at round 5 but never scores
    f = tge_provenance_fields("ScenarioTGEnsemble", {"tge-ramp-rounds": 8}, rounds=4)
    assert f["tge_lstm_state"] == "disabled"
    assert f["tge_pure_lstm_reach"] is False
    # ramp == rounds: a continuously-present client hits tenure == ramp on
    # the final scored round (tenure >= ramp is inclusive) — reach is True
    f = tge_provenance_fields("ScenarioTGEnsemble", {"tge-ramp-rounds": 8}, rounds=8)
    assert f["tge_pure_lstm_reach"] is True
    f = tge_provenance_fields("ScenarioTGEnsemble", {"tge-ramp-rounds": 8}, rounds=7)
    assert f["tge_pure_lstm_reach"] is False


def test_provenance_defaults_to_8_when_key_missing_for_tge():
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields("ScenarioTGEnsemble", {}, rounds=50)
    assert fields["tge_lstm_state"] == "enabled"
    assert fields["tge_ramp_rounds"] == 8


def test_provenance_na_for_non_tge_strategies():
    from run_phase4_flower import tge_provenance_fields
    for strategy in ("ScenarioKrum", "ScenarioTrustScore", "ScenarioFedMedian"):
        fields = tge_provenance_fields(strategy, {}, rounds=50)
        assert fields == {"tge_lstm_state": "n/a", "tge_ramp_rounds": None}


# ---------------------------------------------------------------------------
# 4. Instrumentation contract: active-phase details carry both expert scores
# ---------------------------------------------------------------------------

def _benign_features(rng: np.random.Generator) -> np.ndarray:
    """12-dim feature vector that passes the warmup geometric gate
    (features[0]=z_dist low, [1]=cos_sim high, [2]=norm_dev low)."""
    feats = rng.normal(0.0, 0.05, 12)
    feats[0] = abs(feats[0])          # small z-distance
    feats[1] = 0.9 + feats[1] * 0.1   # high cosine similarity
    feats[2] = abs(feats[2])          # small norm deviation
    return feats.astype(np.float64)


def test_active_phase_details_carry_both_expert_scores():
    """v1.6 § 5.4: once both experts are ready (phase == 'active'), the details
    dict — which scenario_strategy maps verbatim into the signal-log row fields
    tge_gbdt_score / tge_lstm_score — must carry BOTH raw scores non-null.
    The dev-gate ramp re-blend (§ 3.3) is impossible without them."""
    rng = np.random.default_rng(42)
    model = TGEnsembleModel(
        num_features=12, lstm_train_epochs=1, lstm_hidden_dim=8, seed=42,
    )
    clients = [f"client_{i}" for i in range(6)]
    last_details: dict[str, dict] = {}

    # warmup_rounds=3 -> GBDT fits at round 3; LSTM fits at round 5;
    # both experts are ready from round 6 onward.
    for server_round in range(1, 8):
        for cid in clients:
            _, details = model.score_client(cid, _benign_features(rng), server_round)
            model.record_scored(cid, server_round)
            model.record_accepted(cid, server_round)
            last_details[cid] = details
        model.on_round_end(server_round)

    for cid, details in last_details.items():
        assert details["phase"] == "active", (
            f"{cid}: expected phase 'active' by round 7; got {details['phase']!r}"
        )
        assert details["gbdt_score"] is not None, f"{cid}: gbdt_score is None in active phase"
        assert details["lstm_score"] is not None, f"{cid}: lstm_score is None in active phase"

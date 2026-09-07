"""TGE′ runner config + provenance + plumbing (GWU-53, two-leg bank).

Mirrors tests/test_krumtge_config.py and the provenance section of
tests/test_tge_ramp_governance.py for the TGE′ bank configs. The deployed TGE′
mode is long_memory_expert="bank" (min(LSTM, EMA)); ema_alpha=0.9 is ADOPTED
from TrustScore. Both remain PROVISIONAL pending amendment v1.7 ratification —
these tests pin the plumbing, not a frozen value.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

ADOPTED_EMA_ALPHA = 0.9


# ---------------------------------------------------------------------------
# 1. Supported configs + strategy routing
# ---------------------------------------------------------------------------

def test_tgeprime_configs_supported():
    from run_phase4_flower import SUPPORTED_CONFIGS
    assert "TGEprime" in SUPPORTED_CONFIGS
    assert "Krum+TGEprime" in SUPPORTED_CONFIGS


def test_tgeprime_routes_to_bank_token():
    from run_phase4_flower import build_strategy_for_config, NUM_ROUNDS
    token, cfg = build_strategy_for_config("TGEprime")
    assert token.__name__ == "ScenarioTGEPrime"
    assert cfg["tge_long_memory_expert"] == "bank"
    assert cfg["tge_ema_alpha"] == pytest.approx(ADOPTED_EMA_ALPHA)
    # ramp 3 INHERITED (amendment v1.7): the bank retains the LSTM leg r*=3 was
    # gate-selected on, so EXP-015 stays a single-variable A/B.
    assert cfg["tge_ramp_rounds"] == 3
    assert cfg["tge_ramp_rounds"] < NUM_ROUNDS


def test_krumtgeprime_routes_to_bank_token():
    from run_phase4_flower import build_strategy_for_config, NUM_ROUNDS
    token, cfg = build_strategy_for_config("Krum+TGEprime")
    assert token.__name__ == "ScenarioKrumTGEPrime"
    assert cfg["tge_long_memory_expert"] == "bank"
    assert cfg["tge_ema_alpha"] == pytest.approx(ADOPTED_EMA_ALPHA)
    assert cfg["tge_ramp_rounds"] == 3  # INHERITED (amendment v1.7)
    assert cfg["tge_ramp_rounds"] < NUM_ROUNDS


def test_defense_token_follows_class_name_lowered_convention():
    assert "ScenarioTGEPrime".replace("Scenario", "").lower() == "tgeprime"
    assert "ScenarioKrumTGEPrime".replace("Scenario", "").lower() == "krumtgeprime"


# ---------------------------------------------------------------------------
# 2. Provenance: bank identity, min combiner, alpha
# ---------------------------------------------------------------------------

def test_provenance_reports_bank_min_combiner_and_alpha():
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields(
        "ScenarioTGEPrime",
        {"tge-long-memory-expert": "bank", "tge-ema-alpha": 0.9, "tge-ramp-rounds": 8},
        rounds=50,
    )
    assert fields["tge_long_memory_expert"] == "bank"
    assert fields["tge_long_memory_combiner"] == "min"
    assert fields["tge_ema_alpha"] == pytest.approx(0.9)


def test_provenance_reports_lstm_with_null_alpha_and_lstm_combiner_for_incumbent():
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields("ScenarioKrumTGE", {"tge-ramp-rounds": 8}, rounds=50)
    assert fields["tge_long_memory_expert"] == "lstm"
    assert fields["tge_long_memory_combiner"] == "lstm"
    assert fields["tge_ema_alpha"] is None


def test_provenance_na_dict_unchanged_for_non_tge():
    """The exact non-TGE return is asserted by test_tge_ramp_governance; TGE′
    must not have widened it."""
    from run_phase4_flower import tge_provenance_fields
    fields = tge_provenance_fields("ScenarioKrum", {}, rounds=50)
    assert fields == {"tge_lstm_state": "n/a", "tge_ramp_rounds": None}


def test_provenance_default_expert_is_prime_aware(monkeypatch=None):
    """when the run-config key is absent, provenance must default to
    the SAME mode the server constructs — bank for prime tokens, lstm for the
    incumbent TGE tokens — so provenance can never disagree with execution."""
    from run_phase4_flower import tge_provenance_fields
    prime = tge_provenance_fields("ScenarioTGEPrime", {"tge-ramp-rounds": 3}, rounds=50)
    assert prime["tge_long_memory_expert"] == "bank"
    assert prime["tge_long_memory_combiner"] == "min"
    incumbent = tge_provenance_fields("ScenarioKrumTGE", {}, rounds=50)
    assert incumbent["tge_long_memory_expert"] == "lstm"


def test_provenance_honors_explicit_ema_override():
    """An ema-only isolation override is recorded as ema (not the token default)."""
    from run_phase4_flower import tge_provenance_fields
    f = tge_provenance_fields(
        "ScenarioTGEPrime", {"tge-long-memory-expert": "ema", "tge-ema-alpha": 0.9}, rounds=50
    )
    assert f["tge_long_memory_expert"] == "ema"
    assert f["tge_long_memory_combiner"] == "ema"


def test_provenance_ema_only_disables_legacy_lstm_fields():
    """In ema-only component isolation the
    LSTM never feeds the gate, so tge_lstm_state/tge_pure_lstm_reach must not
    report it enabled/reachable — that would group ema runs as LSTM-active."""
    from run_phase4_flower import tge_provenance_fields
    f = tge_provenance_fields(
        "ScenarioTGEPrime",
        {"tge-long-memory-expert": "ema", "tge-ema-alpha": 0.9, "tge-ramp-rounds": 3},
        rounds=50,
    )
    assert f["tge_long_memory_expert"] == "ema"
    assert f["tge_lstm_state"] == "disabled"
    assert f["tge_pure_lstm_reach"] is False


def test_provenance_bank_keeps_lstm_active():
    """bank uses the LSTM leg, so its legacy LSTM fields stay run-length/ramp
    derived (a long run enables it, ramp<=rounds makes pure-LSTM reachable)."""
    from run_phase4_flower import tge_provenance_fields
    f = tge_provenance_fields(
        "ScenarioTGEPrime",
        {"tge-long-memory-expert": "bank", "tge-ema-alpha": 0.9, "tge-ramp-rounds": 3},
        rounds=50,
    )
    assert f["tge_lstm_state"] == "enabled"
    assert f["tge_pure_lstm_reach"] is True


def test_provenance_records_effective_default_ema_alpha_when_key_absent():
    """A prime run that omits
    tge-ema-alpha still constructs the EMA with the plugin default, so
    provenance must record that default (0.9), not None. lstm stays None."""
    import inspect
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    from run_phase4_flower import tge_provenance_fields
    plugin_default = inspect.signature(
        TGEnsemblePlugin.__init__).parameters["ema_alpha"].default
    f = tge_provenance_fields("ScenarioTGEPrime", {"tge-ramp-rounds": 3}, rounds=50)
    assert f["tge_ema_alpha"] == pytest.approx(plugin_default)  # not None
    incumbent = tge_provenance_fields("ScenarioKrumTGE", {}, rounds=50)
    assert incumbent["tge_ema_alpha"] is None


def test_server_app_honors_configured_long_memory_mode():
    """the prime server branches must READ tge-long-memory-expert
    (default bank) and pass it, not hard-code "bank" — otherwise an ema
    override would execute the bank combiner while provenance records ema."""
    src = (PROJECT_ROOT / "flowerfl" / "server_app.py").read_text()
    assert src.count('run_config.get("tge-long-memory-expert", "bank")') >= 2
    assert "long_memory_expert=tge_long_memory" in src
    # the old hard-coded form must be gone from the prime branches
    assert 'long_memory_expert="bank"' not in src


# ---------------------------------------------------------------------------
# 3. Plugin / standalone plumbing across the three modes
# ---------------------------------------------------------------------------

def test_plugin_bank_mode_builds_both_legs():
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    from rmc.tg_ensemble import EMAReputationExpert, LSTMTemporalExpert
    plugin = TGEnsemblePlugin(
        num_malicious=2, num_to_keep=5, long_memory_expert="bank", ema_alpha=0.9
    )
    assert plugin._model.long_memory_expert == "bank"
    assert isinstance(plugin._model.lstm, LSTMTemporalExpert)
    assert isinstance(plugin._model.ema, EMAReputationExpert)
    assert plugin._model.ema.ema_alpha == pytest.approx(0.9)


def test_plugin_defaults_to_lstm_incumbent():
    from flowerfl.byzantine_defense import TGEnsemblePlugin
    from rmc.tg_ensemble import LSTMTemporalExpert
    plugin = TGEnsemblePlugin(num_malicious=2, num_to_keep=5)
    assert plugin._model.long_memory_expert == "lstm"
    assert isinstance(plugin._model.lstm, LSTMTemporalExpert)
    assert plugin._model.ema is None




# ---------------------------------------------------------------------------
# 4. Runner extra-key hyphenation covers the new keys
# ---------------------------------------------------------------------------

def test_extra_key_hyphenation_maps_tgeprime_keys():
    from run_phase4_flower import _hyphenate_tge_extra
    mapped = _hyphenate_tge_extra(
        {"tge_ramp_rounds": 8, "tge_long_memory_expert": "bank", "tge_ema_alpha": 0.9}
    )
    assert mapped == {
        "tge-ramp-rounds": 8,
        "tge-long-memory-expert": "bank",
        "tge-ema-alpha": 0.9,
    }


# ---------------------------------------------------------------------------
# 5. Signal-log schema carries the second leg
# ---------------------------------------------------------------------------

def test_signal_record_includes_tge_ema_score_field():
    """scenario_strategy.py must emit tge_ema_score in the per-client record so
    the bank min() is reconstructable offline."""
    src = (PROJECT_ROOT / "flowerfl" / "scenario_strategy.py").read_text()
    assert '"tge_ema_score": tge_details.get("ema_score")' in src


# ---------------------------------------------------------------------------
# 6. pyproject defaults keep a bare `flwr run` on the incumbent LSTM
# ---------------------------------------------------------------------------

def test_pyproject_long_memory_default_is_lstm():
    import re
    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^tge-long-memory-expert\s*=\s*"(\w+)"', text, re.M)
    assert m is not None, "tge-long-memory-expert missing from pyproject.toml"
    assert m.group(1) == "lstm", (
        f"a bare `flwr run` must stay on the incumbent LSTM; got {m.group(1)!r}"
    )

from praxis_exp.units import Unit
from praxis_exp.runner_paths import model_filename, result_filename, signal_filename, exec_mode_token


def _unit(config, scenario, mode, seed):
    return Unit(config, scenario, mode, seed, 2_000_000, 50, 0)


def test_exec_mode_token_covers_runner_mode_vocabulary():
    # Locks the mapping against a wrong "fix" to == "persistent" (see docstring).
    assert exec_mode_token("persistent_optimizer") == "flower_persistent"
    assert exec_mode_token("persistent") == "flower_persistent"
    assert exec_mode_token("Flower") == "flower_reset"
    assert exec_mode_token("flower_reset") == "flower_reset"
    assert exec_mode_token("reset") == "flower_reset"


def test_result_filename_matches_runner_convention():
    assert result_filename(_unit("Krum+TGE", "S4", "persistent_optimizer", 42)) == \
        "phase4_flower__krum_tge__seed42.json"


def test_signal_filename_matches_runner_convention():
    assert signal_filename(_unit("Krum", "S0_clean", "persistent_optimizer", 42),
                           defense_token="krum") == \
        "flower_persistent__S0_clean__krum__seed42.jsonl"


def test_model_filename_matches_runner_convention():
    """req 6 (models): mirrors run_phase4_flower.py's exp_name + '__model.pt'
    (the same naming this module already uses for result_filename)."""
    assert model_filename(_unit("Krum+TGE", "S4", "persistent_optimizer", 42)) == \
        "phase4_flower__krum_tge__seed42__model.pt"

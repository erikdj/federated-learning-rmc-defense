"""Endpoint arithmetic (acc_final5, final-round, f1), degradation sign
conventions, gate boundary semantics, and the reference-anomaly window."""
from __future__ import annotations

import pytest

from scripts.h4_scoring_lib import (
    ANOMALY_GAP,
    MEDIAN_BAR,
    P_BAR,
    ScoringError,
    acc_final5,
    exact_median,
    f1_final5,
    final_round_accuracy,
    is_reference_anomaly,
    rolling5_peak,
    scenario_gate,
)
from tests.h4_factory import make_trajectory


# ---------------------------------------------------------------------------
# acc_final5
# ---------------------------------------------------------------------------

def test_acc_final5_is_mean_of_last_five_rounds():
    traj = make_trajectory(n_rounds=8, base=0.1,
                           final5=[0.80, 0.85, 0.90, 0.95, 1.00])
    assert acc_final5(traj, "unit") == pytest.approx(0.90)


def test_acc_final5_orders_by_round_number_not_list_order():
    traj = make_trajectory(n_rounds=7, base=0.1, final5=[0.5, 0.6, 0.7, 0.8, 0.9])
    shuffled = [traj[3], traj[6], traj[0], traj[5], traj[1], traj[4], traj[2]]
    assert acc_final5(shuffled, "unit") == pytest.approx(acc_final5(traj, "unit"))


def test_acc_final5_exactly_five_rounds():
    traj = make_trajectory(n_rounds=5, final5=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert acc_final5(traj, "unit") == pytest.approx(0.3)


def test_acc_final5_refuses_short_trajectory():
    traj = make_trajectory(n_rounds=5, final5=[0.1, 0.2, 0.3, 0.4, 0.5])[:4]
    with pytest.raises(ScoringError, match="fewer than 5"):
        acc_final5(traj, "unit")


def test_acc_final5_refuses_empty_trajectory():
    with pytest.raises(ScoringError):
        acc_final5([], "unit")


def test_acc_final5_refuses_duplicate_round_numbers():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[0]["round"] = traj[1]["round"]
    with pytest.raises(ScoringError, match="duplicate round"):
        acc_final5(traj, "unit")


def test_acc_final5_refuses_missing_accuracy_field():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    del traj[-1]["accuracy"]
    with pytest.raises(ScoringError, match="accuracy"):
        acc_final5(traj, "unit")


def test_acc_final5_refuses_non_numeric_accuracy():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[-1]["accuracy"] = "0.9"
    with pytest.raises(ScoringError):
        acc_final5(traj, "unit")


# --- finite/in-range metric validation -------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                 -0.1, 1.1, True])
def test_acc_final5_refuses_non_finite_or_out_of_range(bad):
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[-1]["accuracy"] = bad
    with pytest.raises(ScoringError, match="accuracy"):
        acc_final5(traj, "unit")


def test_acc_final5_boundary_values_zero_and_one_pass():
    traj = make_trajectory(n_rounds=5, final5=[0.0, 1.0, 0.0, 1.0, 0.0])
    assert acc_final5(traj, "unit") == pytest.approx(0.4)


def test_nan_in_pre_window_rounds_also_refuses():
    # NaN anywhere in the trajectory is a broken unit, not just in the
    # final-5 window: rolling5_peak consumes every round.
    traj = make_trajectory(n_rounds=10, base=0.2, final5=0.9)
    traj[0]["accuracy"] = float("nan")
    with pytest.raises(ScoringError, match="accuracy"):
        rolling5_peak(traj, "unit")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.2,
                                 "0.9", True])
def test_f1_final5_present_but_malformed_refuses(bad):
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[-1]["f1"] = bad
    with pytest.raises(ScoringError, match="f1"):
        f1_final5(traj, "unit")


def test_f1_final5_boundary_values_pass():
    traj = make_trajectory(n_rounds=5, final5=0.9)
    for entry in traj:
        entry["f1"] = 1.0
    traj[-1]["f1"] = 0.0
    assert f1_final5(traj, "unit") == pytest.approx(0.8)


# --- round validation, never coercion --------------------

def test_fractional_round_refuses():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[2]["round"] = 1.9
    with pytest.raises(ScoringError, match="round"):
        acc_final5(traj, "unit")


def test_bool_round_refuses():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[0]["round"] = True
    with pytest.raises(ScoringError, match="round"):
        acc_final5(traj, "unit")


def test_string_round_refuses():
    # the Lane-B emitter writes ints (parse_eval_trajectory); a string
    # round is a malformed unit, not a format to accommodate.
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[1]["round"] = "7"
    with pytest.raises(ScoringError, match="round"):
        acc_final5(traj, "unit")


def test_integral_float_round_accepted_as_int():
    # JSON round-trip artifact: 7.0 is round 7.
    traj = make_trajectory(n_rounds=7, base=0.1,
                           final5=[0.5, 0.6, 0.7, 0.8, 0.9])
    for entry in traj:
        entry["round"] = float(entry["round"])
    assert acc_final5(traj, "unit") == pytest.approx(0.7)


def test_nan_round_refuses():
    traj = make_trajectory(n_rounds=6, final5=0.9)
    traj[3]["round"] = float("nan")
    with pytest.raises(ScoringError, match="round"):
        acc_final5(traj, "unit")


# ---------------------------------------------------------------------------
# other endpoints
# ---------------------------------------------------------------------------

def test_final_round_accuracy_is_last_round():
    traj = make_trajectory(n_rounds=8, final5=[0.80, 0.85, 0.90, 0.95, 0.72])
    assert final_round_accuracy(traj, "unit") == pytest.approx(0.72)


def test_f1_final5_mean_of_last_five_f1():
    traj = make_trajectory(n_rounds=8, final5=[0.80, 0.85, 0.90, 0.95, 1.00])
    # factory writes f1 = accuracy - 0.05
    assert f1_final5(traj, "unit") == pytest.approx(0.85)


def test_f1_final5_none_when_f1_absent():
    traj = make_trajectory(n_rounds=8, final5=0.9, with_f1=False)
    assert f1_final5(traj, "unit") is None


def test_f1_final5_none_when_any_last5_entry_lacks_f1():
    traj = make_trajectory(n_rounds=8, final5=0.9)
    del traj[-2]["f1"]
    assert f1_final5(traj, "unit") is None


# ---------------------------------------------------------------------------
# degradation / reduction sign conventions (arithmetic-level)
# ---------------------------------------------------------------------------

def test_degradation_reduction_sign_convention():
    # krum: C0 0.90 -> S3 0.60 = degradation 0.30
    # h2p_fp_krum: C0 0.90 -> S3 0.85 = degradation 0.05
    # reduction = deg(comparator) - deg(treatment) = +0.25 (treatment better)
    deg_comparator = 0.90 - 0.60
    deg_treatment = 0.90 - 0.85
    assert deg_comparator - deg_treatment == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# gate boundaries (pre-registered: median >= 0.05, p <= 0.05)
# ---------------------------------------------------------------------------

def test_gate_median_exactly_at_bar_passes():
    gate = scenario_gate(MEDIAN_BAR, 0.01)
    assert gate["median_pass"] and gate["p_pass"] and gate["passed"]


def test_gate_p_exactly_at_bar_passes():
    gate = scenario_gate(0.20, P_BAR)
    assert gate["passed"]


def test_gate_median_just_below_bar_fails():
    gate = scenario_gate(MEDIAN_BAR - 1e-9, 0.01)
    assert not gate["median_pass"] and not gate["passed"]


def test_gate_p_just_above_bar_fails():
    gate = scenario_gate(0.20, P_BAR + 1e-9)
    assert not gate["p_pass"] and not gate["passed"]


def test_gate_both_fail():
    gate = scenario_gate(-0.10, 0.90)
    assert not gate["passed"]


def test_exact_median_even_count_is_mean_of_middle_two():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert exact_median(values) == pytest.approx(0.55)
    assert exact_median([3.0, 1.0, 2.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# reference anomaly (S6 item 4): acc_final5 > 5pp below own rolling-5 peak
# ---------------------------------------------------------------------------

def _dipping_trajectory(peak: float, final: float):
    """5 rounds at `peak`, then 5 rounds at `final`."""
    entries = []
    for r in range(1, 11):
        acc = peak if r <= 5 else final
        entries.append({"round": r, "accuracy": acc, "f1": acc, "loss": 0.1})
    return entries


def test_rolling5_peak_finds_early_plateau():
    traj = _dipping_trajectory(0.95, 0.70)
    assert rolling5_peak(traj, "unit") == pytest.approx(0.95)


def test_rolling5_peak_on_rising_trajectory_is_final_window():
    traj = make_trajectory(n_rounds=10, base=0.2, final5=0.9)
    assert rolling5_peak(traj, "unit") == pytest.approx(0.9)


def test_reference_anomaly_flags_absorbed_cell():
    traj = _dipping_trajectory(0.95, 0.70)   # gap 0.25 > 0.05
    assert is_reference_anomaly(traj, "unit")["flagged"] is True


def test_reference_anomaly_boundary_exactly_5pp_not_flagged():
    traj = _dipping_trajectory(0.95, 0.95 - ANOMALY_GAP)
    assert is_reference_anomaly(traj, "unit")["flagged"] is False


def test_reference_anomaly_healthy_cell_not_flagged():
    traj = make_trajectory(n_rounds=10, base=0.2, final5=0.9)
    out = is_reference_anomaly(traj, "unit")
    assert out["flagged"] is False
    assert out["acc_final5"] == pytest.approx(0.9)
    assert out["peak_rolling5"] == pytest.approx(0.9)

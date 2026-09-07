#!/usr/bin/env python3
"""Verify published verdict arithmetic from the curated machine-readable evidence."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from h4_scoring_lib import exact_median, exact_wilcoxon_two_sided

EVIDENCE = REPO / "reproduction" / "evidence"


def read(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)


def verify_h1() -> None:
    doc = read("h1-heldout.json")
    margins = doc["per_scenario"]
    a = all(margins[s]["C_minus_S"] >= 0.05 for s in ("S2", "S3", "S4"))
    b = all(margins[s]["C_minus_W"] >= 0.05 for s in ("S2", "S3", "S4"))
    c = all(doc["wilcoxon"][s]["pass"] for s in ("S2", "S3", "S4"))
    assert doc["verdict"]["a_C_gt_S_5pp_all_S2S3S4"] is a
    assert doc["verdict"]["b_C_gt_W_5pp_all_S2S3S4"] is b
    assert doc["verdict"]["c_wilcoxon_all"] is c
    assert not (a and b and c)


def verify_h2() -> None:
    doc = read("h2-heldout.json")
    criteria = doc["criteria"]
    a = criteria["a_s4_recall"]["tge_s4_mean"] >= 0.85
    b = all(item["margin"] >= 0.05 for item in criteria["b_dominance"].values())
    alpha = 1 / 32
    c = all(
        families[item["best_baseline"]]["p_one_sided_exact"] <= alpha
        for scenario, item in criteria["b_dominance"].items()
        for families in [criteria["c_wilcoxon"][scenario]]
    )
    verdict = doc["verdict"]
    assert verdict["a_pass"] is a
    assert verdict["b_pass_all_scenarios"] is b
    assert verdict["c_pass_vs_best_baseline_all_scenarios"] is c
    assert not (a and b and c)


def verify_h2prime() -> None:
    doc = read("h2prime-confirmatory.json")
    p1 = doc["P1"]
    p1_pass = (
        p1["mean_recall"] >= p1["floor"]
        and 0.08 <= p1["realized_blended_fpr"] <= 0.12
    )
    p2 = doc["P2"]
    sign = p2["sign_test"]
    comp = p2["readout_grain_comparability"]
    p2_pass = (
        sign["positive"] >= p2["required_strictly_positive"]
        and comp["detector_in_interval"]
        and comp["baseline_in_interval"]
    )
    assert (p1["verdict"] == "PASS") is p1_pass
    assert (p2["verdict"] == "PASS") is p2_pass
    assert doc["verdict"]["p1"] == "PASS"
    assert doc["verdict"]["p2"] == "PASS"


def verify_h3() -> None:
    doc = read("h3-confirmatory.json")
    components = doc["components"]
    checks = []
    for key in ("rank1_S3", "rank1_S4"):
        checks.append(components[key]["value"] >= components[key]["bar"])
    for key in ("wrong_device_link_rate_malicious", "wrong_device_link_rate_honest"):
        checks.append(components[key]["value"] <= components[key]["bar"])
    assert all(checks)
    assert doc["verdict"] == "PASS"
    assert doc["counts_match_design"] is True


def verify_h4() -> None:
    doc = read("h4-confirmatory.json")
    passes = []
    for scenario, component in doc["primary"]["components"].items():
        diffs = [pair["reduction"] for pair in component["pairs"]]
        median = exact_median(diffs)
        test = exact_wilcoxon_two_sided(diffs)
        assert close(median, component["median_reduction"]), scenario
        assert close(test["p_two_sided"], component["wilcoxon"]["p_two_sided"]), scenario
        passed = median >= 0.05 and test["p_two_sided"] <= 0.05
        assert component["gate"]["passed"] is passed, scenario
        passes.append(passed)
    assert not all(passes)
    assert doc["verdict"] == "FALSIFIED"


def main() -> int:
    checks = [verify_h1, verify_h2, verify_h2prime, verify_h3, verify_h4]
    for check in checks:
        check()
        print(f"PASS {check.__name__.removeprefix('verify_')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

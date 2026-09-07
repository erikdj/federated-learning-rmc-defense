#!/usr/bin/env python3
"""Verify published verdict arithmetic from the curated machine-readable evidence."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from h4_scoring_lib import exact_median, exact_wilcoxon_two_sided
from analyze_h2_dev_read import exact_wilcoxon_onesided

EVIDENCE = REPO / "reproduction" / "evidence"


def read(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)


def verify_h1() -> None:
    doc = read("h1-heldout.json")
    cuts = read("h1-thresholds-dev-frozen.json")
    assert doc["dev_thresholds"] == cuts["thresholds"]
    seeds = doc["_meta"]["heldout_seeds"]
    assert len(seeds) == len(set(seeds)) == 10
    expected = {f"S{s}__seed{seed}" for s in range(5) for seed in seeds}
    for arm in ("primary", "secondary"):
        block = doc[arm]
        cells, margins = block["per_cell"], block["per_scenario"]
        assert set(cells) == expected
        for scenario, summary in margins.items():
            units = [cells[f"{scenario}__seed{seed}"] for seed in seeds]
            for family in ("S", "W", "C"):
                assert close(summary[family], mean(u[family]["recall_at_fpr"] for u in units))
                assert close(summary[f"{family}_actual_fpr"], mean(u[family]["actual_fpr"] for u in units))
                if arm == "primary":
                    assert all(u[family]["threshold"] == cuts["thresholds"][family]["threshold"] for u in units)
            for family in ("S", "W"):
                assert close(summary[f"C_minus_{family}"], summary["C"] - summary[family])
                if scenario in ("S2", "S3", "S4"):
                    # Count-equivalent recall differences share ranks in the read.
                    diffs = [round(u["C"]["recall_at_fpr"] - u[family]["recall_at_fpr"], 9) for u in units]
                    w_plus, p_value = exact_wilcoxon_onesided(diffs)
                    recorded = block["wilcoxon"][scenario]
                    assert close(recorded[f"p_C_gt_{family}"], p_value)
                    assert close(recorded[f"W_plus_C_gt_{family}"], w_plus)
        a = all(margins[s]["C_minus_S"] >= 0.05 for s in ("S2", "S3", "S4"))
        b = all(margins[s]["C_minus_W"] >= 0.05 for s in ("S2", "S3", "S4"))
        c = True
        for scenario in ("S2", "S3", "S4"):
            test = block["wilcoxon"][scenario]
            passed = test["p_C_gt_S"] <= 1 / 32 and test["p_C_gt_W"] <= 1 / 32
            assert test["pass"] is passed
            c = c and passed
        assert block["verdict"]["a_C_gt_S_5pp_all_S2S3S4"] is a
        assert block["verdict"]["b_C_gt_W_5pp_all_S2S3S4"] is b
        assert block["verdict"]["c_wilcoxon_all"] is c
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
    window = doc["secondaries"]["window_aware_loao_sensitivity"]
    assert window["status"] == "COMPUTED"
    q1, q2 = window["p1_quantity"], window["p2_quantity"]
    for key, value in q1["primary"].items():
        assert value == p1[key], key
    assert q2["primary_sign_test"] == sign
    assert q2["primary_per_seed_margin"] == p2["per_seed_diff"]
    assert close(q1["delta_mean_recall"], q1["window_aware"]["mean_recall"] - p1["mean_recall"])
    assert close(q1["delta_realized_blended_fpr"], q1["window_aware"]["realized_blended_fpr"] - p1["realized_blended_fpr"])
    assert close(q1["window_aware"]["mean_recall"], mean(q1["window_aware"]["per_seed_recall"].values()))
    assert q2["window_aware_sign_test"]["positive"] == sum(v > 0 for v in q2["window_aware_per_seed_margin"].values())


def verify_provenance() -> None:
    """Check published hash claims against the distributed artifact bytes."""
    doc = read("provenance.json")
    entries = doc["artifacts"] + doc["frozen_protocol"] + [doc["calibration_artifact"]]
    for entry in entries:
        if not entry.get("public_path"):
            continue
        expected = entry.get("public_sha256", entry.get("sha256"))
        assert expected, entry["public_path"]
        actual = hashlib.sha256((REPO / entry["public_path"]).read_bytes()).hexdigest()
        assert actual == expected, f"stale provenance hash: {entry['public_path']}"


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
    checks = [verify_h1, verify_h2, verify_h2prime, verify_h3, verify_h4, verify_provenance]
    for check in checks:
        check()
        print(f"PASS {check.__name__.removeprefix('verify_')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

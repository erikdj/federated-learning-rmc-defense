"""H2 CONFIRMATORY (held-out test) read at the FROZEN LEAK-FREE cuts.

This is the one-shot, post-exposure-immutable thesis test of H2 on the EXP-048
held-out logs (200 units = 4 defenses x 5 scenarios x 10 SEALED test seeds).

It is the SAME code path as the dev read (scripts/analyze_h2_dev_read.py):
identical metric (recall@10%FPR, all-rounds scope), identical bands (a/b/c),
identical exact one-sided paired Wilcoxon, identical reduce_unit() reduction
with per-unit scenario/seed/defense identity gates. Only three PRE-REGISTERED
inputs change, exactly as the confirmatory read is defined (v1.10 § 4, v1.14):

  1. Data      : EXP-048 held-out logs (10 sealed seeds), not EXP-011/012/013.
  2. Thresholds: the FROZEN LEAK-FREE F8 10% cuts
                 (results/20260806/leakfree_reseal/bracket_cuts_leakfree.json
                  sha 744af38b, commit 36d32df), applied UNCHANGED — never
                  re-derived, never re-selected (v1.10 § 4.3 gate 2).
  3. Seeds     : the 10 sealed h2_confirm seeds (data/h2_confirm_seeds.json).

Bands (base spec § 6.2, byte-unchanged; EXP-048 doc § 4):
  (a) TGE recall@10%FPR >= 0.85 in S4;
  (b) TGE > best baseline by >= 5 pp recall@10%FPR in S2/S3/S4;
  (c) one-sided paired Wilcoxon (TGE > baseline), p <= 0.0312 (= 1/32).
Falsification: ANY one of (a)/(b)/(c) fails.

NO parameter, cut, scope, estimator, eligibility, or band is changed here.
Re-running this deterministic scoring is permitted; the frozen inputs are fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_variance_envelope import (  # noqa: E402
    DEFAULT_COLDSTART_K,
    DEFENSE_SCORE_SPECS,
    UnitSpec,
    load_thresholds,
    reduce_unit,
)
from analyze_h2_dev_read import (  # noqa: E402  reuse the EXACT criteria machinery
    CRITERION_ALPHA,
    CRITERION_MARGIN_PP,
    CRITERION_S4_RECALL,
    DOMINANCE_SCENARIOS,
    SCENARIOS,
    SCEN_SHORT,
    DEFENSES,
    BASELINES,
    exact_wilcoxon_onesided,
    parse_stem,
)

# The 10 SEALED confirmatory seeds (data/h2_confirm_seeds.json).
SEEDS = [1009, 1733, 2521, 3299, 4127, 5051, 6079, 7177, 8231, 9337]


def enumerate_units(data_root: Path) -> list[tuple[str, str, int, Path, Path]]:
    units = []
    for f in sorted((data_root / "EXP-048" / "results").glob("*.json")):
        scenario, defense, seed = parse_stem(f.stem)
        units.append((scenario, defense, seed, f,
                      data_root / "EXP-048" / "signals" / f"{f.stem}.jsonl"))
    cells = {(s, d, sd) for s, d, sd, _, _ in units}
    expected = {(s, d, sd) for s in SCENARIOS for d in DEFENSES for sd in SEEDS}
    if cells != expected:
        raise SystemExit(f"matrix incomplete: missing={sorted(expected - cells)} "
                         f"extra={sorted(cells - expected)}")
    if len(units) != 200:
        raise SystemExit(f"expected exactly 200 held-out units, got {len(units)}")
    return units


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--thresholds", required=True, type=Path,
                    help="FROZEN leak-free 10%% cuts artifact (load_thresholds shape)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds)
    units = enumerate_units(args.data_root)

    per_cell: dict[tuple[str, str, int], dict] = {}
    for scenario, defense, seed, result_path, signal_path in units:
        spec = DEFENSE_SCORE_SPECS[defense]
        unit = UnitSpec(arm="confirm", defense=defense, seed=seed, replicate=1,
                        result_path=result_path, signal_path=signal_path)
        canonical_stem = scenario[0].upper() + scenario[1:]
        rep = reduce_unit(unit, spec, thresholds[defense],
                          k=DEFAULT_COLDSTART_K, expected_scenario_stem=canonical_stem)
        result = json.loads(result_path.read_text())
        m = dict(rep.metrics)
        m["mean_accuracy"] = result.get("mean_accuracy")
        m["mean_f1"] = result.get("mean_f1")
        m["fpr_allrounds"] = rep.diagnostics.get("fpr_allrounds")
        per_cell[(scenario, defense, seed)] = m

    def agg(scenario: str, defense: str, metric: str) -> dict:
        vals = [per_cell[(scenario, defense, s)][metric] for s in SEEDS]
        vals = [v for v in vals if v is not None]
        return {"mean": mean(vals) if vals else None,
                "sd": stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "per_seed": {s: per_cell[(scenario, defense, s)][metric] for s in SEEDS}}

    matrix = {SCEN_SHORT[sc]: {d: {met: agg(sc, d, met)
                                   for met in ("recall_allrounds", "recall_coldstart",
                                               "auc_allrounds", "auc_coldstart",
                                               "mean_accuracy", "fpr_allrounds")}
                               for d in DEFENSES}
              for sc in SCENARIOS}

    crit = {}
    s4_tge = [per_cell[("s4_full_mix", "tge", s)]["recall_allrounds"] for s in SEEDS]
    crit["a_s4_recall"] = {
        "criterion": f"TGE recall@10%FPR >= {CRITERION_S4_RECALL} in S4",
        "tge_s4_mean": mean(s4_tge), "tge_s4_per_seed": dict(zip(SEEDS, s4_tge)),
        "pass": mean(s4_tge) >= CRITERION_S4_RECALL,
    }
    crit["b_dominance"] = {}
    crit["c_wilcoxon"] = {}
    for sc in DOMINANCE_SCENARIOS:
        short = SCEN_SHORT[sc]
        base_means = {b: matrix[short][b]["recall_allrounds"]["mean"] for b in BASELINES}
        best_b = max(base_means, key=lambda b: base_means[b] if base_means[b] is not None else -1)
        tge_mean = matrix[short]["tge"]["recall_allrounds"]["mean"]
        crit["b_dominance"][short] = {
            "criterion": f"TGE > best baseline by >= {CRITERION_MARGIN_PP} recall@10%FPR",
            "best_baseline": best_b, "best_baseline_mean": base_means[best_b],
            "tge_mean": tge_mean,
            "margin": (tge_mean - base_means[best_b])
            if None not in (tge_mean, base_means[best_b]) else None,
            "pass": (tge_mean is not None and base_means[best_b] is not None
                     and tge_mean - base_means[best_b] >= CRITERION_MARGIN_PP),
        }
        crit["c_wilcoxon"][short] = {}
        for b in BASELINES:
            diffs = [per_cell[(sc, "tge", s)]["recall_allrounds"]
                     - per_cell[(sc, b, s)]["recall_allrounds"] for s in SEEDS]
            w, p = exact_wilcoxon_onesided(diffs)
            crit["c_wilcoxon"][short][b] = {
                "diffs_tge_minus_baseline": dict(zip(SEEDS, diffs)),
                "w_plus": w, "p_one_sided_exact": p,
                "pass_alpha_0312": p <= CRITERION_ALPHA,
            }

    all_b = all(v["pass"] for v in crit["b_dominance"].values())
    best_c = all(crit["c_wilcoxon"][SCEN_SHORT[sc]][crit["b_dominance"][SCEN_SHORT[sc]]
                 ["best_baseline"]]["pass_alpha_0312"] for sc in DOMINANCE_SCENARIOS)
    a_pass = crit["a_s4_recall"]["pass"]
    verdict = {
        "a_pass": a_pass,
        "b_pass_all_scenarios": all_b,
        "c_pass_vs_best_baseline_all_scenarios": best_c,
        "confirmatory": ("H2 CONFIRMED (held-out test) — all three bands MET" if
                         (a_pass and all_b and best_c)
                         else "H2 FALSIFIED (held-out test) — falsification condition fired "
                              "(any one of a/b/c failed)"),
    }

    out = {
        "_meta": {
            "read": "H2 CONFIRMATORY held-out-test read at FROZEN LEAK-FREE 10% cuts",
            "thresholds": {d: thresholds[d] for d in DEFENSES},
            "thresholds_source": str(args.thresholds),
            "coldstart_k": DEFAULT_COLDSTART_K,
            "adjudication_scope": "recall_allrounds (base spec § 6.2 unscoped primary)",
            "units": 200, "seeds": SEEDS,
            "immutability": "post-exposure immutability IN FORCE (EXP-039 unseal); "
                            "cuts/scope/estimator/bands byte-unchanged from pre-registration",
            "s0_naming_caveat": "s0_clean_baseline filename is LEGACY — content = 9/20 "
                                "sustained ALIE static baseline",
        },
        "matrix": matrix,
        "criteria": crit,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

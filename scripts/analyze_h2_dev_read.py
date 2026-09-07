"""H2 development-stage read at the FROZEN operating points (dev gate).

Pre-registered basis (chain of custody):
- Criteria: experimental-design spec sec 6.2 (as amended v1.4: online defense set
  {krum, trustscore, tge, krum_tge}) —
    (a) TGE >= 0.85 recall@10%FPR in S4;
    (b) TGE > best baseline by >= 5 pp recall@10%FPR in S2/S3/S4;
    (c) one-sided paired Wilcoxon (TGE > baseline), n=5 dev seeds, p <= 0.0312.
  Falsification: ANY one of (a)/(b)/(c) fails. This is the DEV-stage read of
  the locked 100-cell sweep; the confirmatory read (10 sealed seeds) is the
  thesis test and is NOT performed here.
- Thresholds: results/20260719/variance_envelope/thresholds_dev_honest.json
  (frozen F8 cuts; applied UNCHANGED, never re-derived — same discipline and
  same code path as the variance envelope: reduce_unit() from
  scripts/analyze_variance_envelope.py, which enforces per-unit scenario/seed/
  defense identity gates).
- Metric scope: sec 6.2 defines the H2 primary as recall@10%FPR over the
  defense's per-round detection decisions (unscoped => all rounds).
  recall_coldstart (tenure in [1,k], the variance study's primary) is reported
  alongside for the H1/attribution context; adjudication here uses ALLROUNDS.
- Data: EXP-011 (98 units) + the two storm-refill cells per each refill doc's
  pre-stated analysis rule — EXP-012 contributes ONLY Krum x S3 x seed137,
  EXP-013 contributes ONLY TrustScore x S1 x seed42. Padding units excluded.
- v1.20: accuracy context reported as trajectory means (result mean_accuracy),
  not final-round snapshots.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
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

SCENARIOS = ["s0_clean_baseline", "s1_benign_churn_only", "s2_adaptive_switching_only",
             "s3_identity_reset_only", "s4_full_mix"]
SCEN_SHORT = {s: f"S{i}" for i, s in enumerate(SCENARIOS)}
DEFENSES = ["krum", "trustscore", "tge", "krum_tge"]
BASELINES = ["krum", "trustscore", "krum_tge"]
SEEDS = [42, 137, 256, 314, 500]
REFILL_RULE = {  # exp dir -> the single analysis-bearing unit stem
    "EXP-012": "s3_identity_reset_only__krum__persistent_optimizer__seed137",
    "EXP-013": "s1_benign_churn_only__trustscore__persistent_optimizer__seed42",
}
CRITERION_MARGIN_PP = 0.05
CRITERION_S4_RECALL = 0.85
# Spec sec 6.2: "n=5 dev seeds floors one-sided at p=0.0312" and criterion (c)
# "Wilcoxon p <= 0.0312" — the stated floor IS the exact 1/32 = 0.03125 written
# to 4 dp, so the all-signs-positive case is by construction attainable and
# passing. Adjudicate against the exact floor, not the rounded literal.
CRITERION_ALPHA = 1 / 32
DOMINANCE_SCENARIOS = ["s2_adaptive_switching_only", "s3_identity_reset_only", "s4_full_mix"]


def parse_stem(stem: str) -> tuple[str, str, int]:
    scenario, defense, mode, seed_part = stem.rsplit("__", 3)
    if mode != "persistent_optimizer" or not seed_part.startswith("seed"):
        raise ValueError(f"unexpected unit stem: {stem}")
    return scenario, defense, int(seed_part[4:])


def enumerate_units(data_root: Path) -> list[tuple[str, str, int, Path, Path]]:
    units = []
    for f in sorted((data_root / "EXP-011" / "results").glob("*.json")):
        scenario, defense, seed = parse_stem(f.stem)
        units.append((scenario, defense, seed, f,
                      data_root / "EXP-011" / "signals" / f"{f.stem}.jsonl"))
    for exp, stem in REFILL_RULE.items():
        f = data_root / exp / "results" / f"{stem}.json"
        if not f.exists():
            raise FileNotFoundError(f"refill analysis cell missing: {f}")
        scenario, defense, seed = parse_stem(stem)
        units.append((scenario, defense, seed, f,
                      data_root / exp / "signals" / f"{stem}.jsonl"))
    cells = {(s, d, sd) for s, d, sd, _, _ in units}
    expected = {(s, d, sd) for s in SCENARIOS for d in DEFENSES for sd in SEEDS}
    if cells != expected:
        raise SystemExit(f"matrix incomplete: missing={sorted(expected - cells)} "
                         f"extra={sorted(cells - expected)}")
    if len(units) != 100:
        raise SystemExit(f"expected exactly 100 analysis-bearing units, got {len(units)}")
    return units


def exact_wilcoxon_onesided(diffs: list[float]) -> tuple[float, float]:
    """Exact one-sided paired Wilcoxon signed-rank: H1 = median(diff) > 0.
    Zeros dropped (zero_method='wilcox'); ties get mid-ranks; p = P(W+ >= obs)
    under the 2^m sign-flip null. Returns (W_plus, p). All-zero -> (0, 1.0)."""
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 0.0, 1.0
    abs_sorted = sorted((abs(d), i) for i, d in enumerate(nz))
    ranks = [0.0] * len(nz)
    j = 0
    while j < len(abs_sorted):
        k = j
        while k + 1 < len(abs_sorted) and abs_sorted[k + 1][0] == abs_sorted[j][0]:
            k += 1
        midrank = (j + k) / 2 + 1
        for m in range(j, k + 1):
            ranks[abs_sorted[m][1]] = midrank
        j = k + 1
    w_obs = sum(r for r, d in zip(ranks, nz) if d > 0)
    n = len(nz)
    ge = sum(1 for signs in itertools.product((0, 1), repeat=n)
             if sum(r for r, s in zip(ranks, signs) if s) >= w_obs)
    return w_obs, ge / (2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--thresholds", type=Path,
                    default=REPO / "results/20260719/variance_envelope/thresholds_dev_honest.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds)
    units = enumerate_units(args.data_root)

    per_cell: dict[tuple[str, str, int], dict] = {}
    for scenario, defense, seed, result_path, signal_path in units:
        spec = DEFENSE_SCORE_SPECS[defense]
        unit = UnitSpec(arm="dev", defense=defense, seed=seed, replicate=1,
                        result_path=result_path, signal_path=signal_path)
        # unit filenames carry the lowercase stem; the scenario JSON (and the
        # integrity gate's registered form) is canonical 'S<n>_...'
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

    # --- pre-registered criteria (dev-stage read) --------------------------
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
    verdict = {
        "a_pass": crit["a_s4_recall"]["pass"],
        "b_pass_all_scenarios": all_b,
        "c_pass_vs_best_baseline_all_scenarios": best_c,
        "dev_read": ("H2 criteria MET at dev stage" if
                     (crit["a_s4_recall"]["pass"] and all_b and best_c)
                     else "H2 criteria NOT MET at dev stage — falsification condition fired "
                          "(dev-stage read; gate decision on the fork belongs to Erik)"),
    }

    out = {
        "_meta": {
            "read": "H2 development-stage read at frozen thresholds (sec 6.2 criteria)",
            "thresholds": {d: thresholds[d] for d in DEFENSES},
            "coldstart_k": DEFAULT_COLDSTART_K,
            "adjudication_scope": "recall_allrounds (sec 6.2 unscoped primary)",
            "units": 100,
            "refill_rule_applied": REFILL_RULE,
            "s0_naming_caveat": "s0_clean_baseline filename is LEGACY - content = 9/20 "
                                "sustained ALIE static baseline (audited vs spec v1.4 pre-launch)",
        },
        "matrix": matrix,
        "criteria": crit,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps(verdict, indent=1))
    for sc in SCENARIOS:
        short = SCEN_SHORT[sc]
        row = "  ".join(f"{d}={matrix[short][d]['recall_allrounds']['mean']:.3f}"
                        if matrix[short][d]['recall_allrounds']['mean'] is not None else f"{d}=NA"
                        for d in DEFENSES)
        print(f"{short}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

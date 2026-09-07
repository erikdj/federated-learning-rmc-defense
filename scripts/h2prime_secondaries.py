"""Mandatory REPORTED secondaries for the H2′ confirmatory read.

Every function here is NON-ADJUDICATING. They exist so that v1.15 § 4's
"Secondaries — REPORTED, with NO pass/fail attached" are produced by the SAME
single pass that adjudicates P1 ∧ P2: the sealed corpus is opened exactly once
(EXP-051 § 5.1), so a secondary that needed a second read could never be
computed at all.

Covered here:
  § 4 secondary 4  — AUC per scenario (threshold-free)
  § 4 secondary 5  — pooled-row ALIE recall (computed in the adjudicator)
  § 4 secondary 7  — the three fixed-baseline contrasts on the P2 population
  § 4 secondary 8  — global-cut calibration sensitivity, and the G2-EXTENDED
                     matched TGE-vs-H2′ contrast on `tge_score` non-null rows
  § 4 secondary 9  — the full-coverage descriptive TGE contrast on the EXP-048
                     standalone-TGE arm (EXPOSED data, zero fleet compute)

The per-cell inputs are collected by `adjudicate_h2prime.score_corpus`; this
module only reduces them.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from statistics import mean

# G2 coverage rule (EXP-051 § 2, v1.15 § 1): a fold/slice with fewer than this
# many scored malicious rows is reported NOT COMPARABLE, with no statistic.
G2_MIN_SCORED_MALICIOUS = 30


def _frac(scored, total) -> float | None:
    return (scored / total) if total else None


def _census(cell: dict) -> dict:
    """The registered TGE coverage census for one fold.

    v1.15 § 1 DECISION G item 3: "the TGE coverage fraction (non-null
    `tge_score` rows / eligible rows, SPLIT HONEST / MALICIOUS and by attack
    family) is reported for every fold and slice". The floor (item 4) is scored
    on the MALICIOUS population; the honest split is reported alongside it.
    """
    return {
        "n_mal_scored": cell.get("n_mal_scored", 0),
        "n_mal_total": cell.get("n_mal_total"),
        "n_honest_scored": cell.get("n_honest_scored"),
        "n_honest_total": cell.get("n_honest_total"),
        "coverage_mal": _frac(cell.get("n_mal_scored", 0),
                              cell.get("n_mal_total") or 0),
        "coverage_honest": _frac(cell.get("n_honest_scored") or 0,
                                 cell.get("n_honest_total") or 0),
    }


def _by_seed(cells: dict, scen: str, field: str) -> dict:
    return {sd: v[field] for (sc, sd), v in sorted(cells.items())
            if sc == scen and v.get(field) is not None}


def per_scenario_auc(scored: dict, scenarios: list[str], t_crit: float, df: int,
                     ci_fn) -> dict:
    """§ 4 secondary 4 — threshold-free Mann-Whitney AUC, ALIE vs honest.

    Dev reference (`S4_ALIE_COLLAPSE.md` § 1.3): 0.82 – 0.92 across scenarios.
    Reported as the scale-stable view; explicitly NOT the adjudicating metric.
    """
    out = {"_note": "REPORTED, NON-ADJUDICATING (§ 4 secondary 4)"}
    for scen in scenarios:
        vals = _by_seed(scored["alie"], scen, "auc")
        if not vals:
            continue
        series = [vals[s] for s in sorted(vals)]
        out[scen] = {
            "per_seed": {str(s): vals[s] for s in sorted(vals)},
            "mean": mean(series),
            "ci95_student_t": ci_fn(series, t_crit, df),
        }
    return out


def bracket_recall_by_fpr(scored: dict, scenarios: list[str], targets,
                          adjudicating_target: float, interval, ci_fn,
                          t_crit: float, df: int) -> dict:
    """§ 6 (ratification addition) — LOAO recall across the 1/2/5/10 % bracket.

    REPORTED ONLY, NO pass/fail. The § 4 bands adjudicate at the 10 % point
    exactly as ratified; this block exists so the praxis is not hostage to one
    design-time operating point.

    Comparability is annotated per point against THAT POINT'S OWN target ± the
    proportional interval — the frozen [0.08, 0.12] is ±20 % of 0.10, so the
    ratio is taken from the frozen constants rather than restated, and a change
    to the interval propagates here instead of drifting. It is EVIDENCE ONLY:
    nothing in this block halts, at any point, including 10 %.
    """
    lo_rel, hi_rel = (interval[0] / adjudicating_target,
                      interval[1] / adjudicating_target)
    cells = scored.get("bracket") or {}
    out = {
        "_note": ("REPORTED ONLY, NO pass/fail (§ 6). The § 4 bands adjudicate "
                  f"at {adjudicating_target} and are untouched by this block."),
        "targets": list(targets),
        "adjudicating_target": adjudicating_target,
        "interval_rule": (f"each point's own target x [{lo_rel:g}, {hi_rel:g}] — "
                          f"the frozen {list(interval)} at the "
                          f"{adjudicating_target} point"),
        "comparability_is_evidence_only": True,
        "points": {},
    }
    for t in targets:
        lo, hi = t * lo_rel, t * hi_rel
        point = {"target_fpr": t, "interval": [lo, hi],
                 "is_adjudicating_point": t == adjudicating_target,
                 "scenarios": {}}
        for scen in scenarios:
            rec = {sd: v["recall"] for (sc, sd, tt), v in sorted(cells.items())
                   if sc == scen and tt == t}
            fpr = {sd: v["realized_fpr"] for (sc, sd, tt), v in sorted(cells.items())
                   if sc == scen and tt == t}
            if not rec:
                continue
            seeds = sorted(rec)
            vals = [rec[s] for s in seeds]
            point["scenarios"][scen] = {
                "n_folds": len(seeds),
                "per_seed_recall": {str(s): rec[s] for s in seeds},
                "mean_recall": mean(vals),
                "ci95_student_t": ci_fn(vals, t_crit, df),
                "per_seed_realized_fpr": {str(s): fpr[s] for s in seeds},
                "mean_realized_fpr": mean(fpr[s] for s in seeds),
                "per_seed_in_interval": {str(s): lo <= fpr[s] <= hi for s in seeds},
                "in_interval": lo <= mean(fpr[s] for s in seeds) <= hi,
            }
        out["points"][f"{t:g}"] = point
    return out


def _fmt_fpr(x: float | None) -> str:
    return "unmeasured (no honest rows)" if x is None else f"{x:.4f}"


def _pooled_fpr(cells) -> tuple[float | None, int, int]:
    """Row-pooled realized FPR over a cohort of ALIE-population cells.

    v1.15b § 1.2 item 1 freezes the aggregation as the ROW-POOLED rate over the
    population's honest rows — exact integer arithmetic over flag counts, never
    the unweighted mean of per-cell rates. Same construction as the adjudicating
    guard in `h2prime_bands`, so the reported contrasts are guarded at the same
    grain the verdict is.
    """
    flagged = sum(c["n_flagged_honest"] for c in cells)
    honest = sum(c["n_honest"] for c in cells)
    return (flagged / honest if honest else None), flagged, honest


def alie_fixed_baseline_contrasts(scored: dict, baselines, scenarios: list[str],
                                  sign_fn, comparable_fn) -> dict:
    """§ 4 secondary 7 — detector vs EACH enumerated instrument on the P2
    population, separately from the oracle-maximum comparator.

    Same estimand as P2 (equal-weight macro-average of the ALIE-bearing
    scenario-cell recalls within each seed), so the numbers are directly
    comparable to the adjudicated contrast — but these adjudicate nothing.

    **The § 3.2 comparability discipline binds here too, at the v1.15b § 1.2
    readout grain.** Sharing P2's estimand means sharing P2's guard: if either
    side's COHORT realized FPR — row-pooled over the ALIE-bearing scored
    population of the paired seeds — falls outside the closed interval, the
    contrast is NOT COMPARABLE and publishes no statistic. Both realized FPRs
    are printed as the exclusion's evidence (§ 3.2's "reported as INCONCLUSIVE
    with the realized FPR printed"); census and label ride every branch, since
    suppression is for contrast STATISTICS only.

    `comparable_fn` is REQUIRED, not defaulted: a guard a caller can omit by
    forgetting an argument is not a guard.
    """
    det_cells, base_cells = scored["alie"], scored["base_alie"]
    det_macro: dict[int, list[float]] = {}
    for (_sc, sd), v in sorted(det_cells.items()):
        det_macro.setdefault(sd, []).append(v["recall"])
    det_seed = {s: mean(v) for s, v in sorted(det_macro.items())}

    out = {"_note": "REPORTED, NON-ADJUDICATING (§ 4 secondary 7)",
           # The detector's own per-seed readout, published ONCE for the block.
           # It is not attributable to any one instrument, so it is not a
           # statistic OF a contrast; suppression binds at the contrast node.
           "detector_per_seed_macro": {str(s): v for s, v in det_seed.items()},
           "comparability_grain": ("COHORT — the row-pooled realized FPR of each "
                                   "side over the ALIE-bearing scored population "
                                   "of the paired seeds (v1.15b § 1.2)"),
           "contrasts": {}}
    for b, _trust, label in baselines:
        macro: dict[int, list[float]] = {}
        for (_sc, sd), v in sorted(base_cells[b].items()):
            macro.setdefault(sd, []).append(v["recall"])
        b_seed = {s: mean(v) for s, v in sorted(macro.items())}
        common = sorted(set(det_seed) & set(b_seed))
        # An instrument that pairs with NOTHING used to vanish here via
        # `continue` — the same silent-omission species this block's § 3.2 guard
        # closed, at the pairing grain instead of the operating-point grain. It
        # now leaves an exclusion node carrying its census and its named cause,
        # so "no contrast" and "no instrument" cannot print identically.
        # Each side's cohort is the population ITS contribution to THIS contrast
        # was scored on — the paired seeds only, per side. `common` can differ
        # between instruments, so the detector cohort is re-pooled per contrast
        # rather than computed once and reused.
        seeds_in = set(common)
        det_fpr, det_flagged, det_honest = _pooled_fpr(
            [v for (_sc, sd), v in sorted(det_cells.items()) if sd in seeds_in])
        b_fpr, b_flagged, b_honest = _pooled_fpr(
            [v for (_sc, sd), v in sorted(base_cells[b].items()) if sd in seeds_in])
        entry = {
            "label": label,
            # Census — rides BOTH branches. Two paired seeds and ten are
            # different stories and must not print identically.
            "n_paired_seeds": len(common),
            "paired_seeds": [str(s) for s in common],
            # The two side counts make an UNPAIRED exclusion legible: "5 vs 5
            # with zero overlap" and "5 vs 0" are different corpus defects and
            # must not print identically.
            "n_detector_seeds": len(det_seed),
            "n_baseline_seeds": len(b_seed),
            "detector_flagged_honest": det_flagged,
            "detector_honest_rows": det_honest,
            "baseline_flagged_honest": b_flagged,
            "baseline_honest_rows": b_honest,
            # Calibration evidence — rides BOTH branches. On the halting branch
            # these two ARE the evidence for the halt, so suppressing them would
            # delete the reason for the exclusion (§ 3.2).
            "detector_pooled_fpr": det_fpr,
            "baseline_pooled_fpr": b_fpr,
            "detector_in_interval": comparable_fn(det_fpr),
            "baseline_in_interval": comparable_fn(b_fpr),
        }
        if not common:
            # Cause precedence, exactly as `_exclusion_cause` orders its own:
            # an unpaired instrument has no paired cohort to realize an
            # operating point ON, so it is reported as unpaired rather than as
            # an FPR failure. The pooled FPRs are None here BECAUSE the cohort
            # is empty — naming the cause is what keeps that from reading as
            # "out of interval".
            entry["status"] = "NOT COMPARABLE"
            entry.update(exclusion_fields("unpaired", (
                f"{b}: {len(b_seed)} baseline seed(s) and {len(det_seed)} "
                f"detector seed(s) share none")))
        elif not (entry["detector_in_interval"] and entry["baseline_in_interval"]):
            outside = []
            if not entry["detector_in_interval"]:
                outside.append(f"detector {_fmt_fpr(det_fpr)}")
            if not entry["baseline_in_interval"]:
                outside.append(f"{b} {_fmt_fpr(b_fpr)}")
            entry["status"] = "NOT COMPARABLE"
            entry.update(exclusion_fields("fpr_interval", " and ".join(outside)))
        else:
            diffs = [det_seed[s] - b_seed[s] for s in common]
            entry["status"] = "COMPUTED"
            entry["baseline_per_seed_macro"] = {str(s): b_seed[s] for s in common}
            entry["per_seed_diff"] = {str(s): det_seed[s] - b_seed[s] for s in common}
            entry["margin"] = (mean(det_seed[s] for s in common)
                               - mean(b_seed[s] for s in common))
            entry["sign_test"] = sign_fn(diffs)
        out["contrasts"][b] = entry
    return out


def global_cut_sensitivity(scored: dict, scenarios: list[str], p1_slice: str,
                           comparable_fn) -> dict:
    """§ 2.2 / § 4 secondary 8 — the GLOBAL-cut arm, reported as a sensitivity.

    The pre-registered adjudicating granularity is PER-SCENARIO (§ 2.2). The
    global cut pools the calibration seed's honest rows across all scenarios;
    on dev it realized ~0.169 FPR against the per-scenario arm's 0.101 and
    inflated held-out-ALIE S4 recall from 0.206 to 0.409 by spending false
    positives. It is reported here so that contrast is visible, never used.
    """
    blend_g = scored.get("blend_global") or {}
    alie_g = scored.get("alie_global") or {}
    out = {"_note": ("REPORTED, NON-ADJUDICATING (§ 2.2 disclosure / § 4 "
                     "secondary 8) — the per-scenario arm is the adjudicating one")}

    rec = _by_seed(blend_g, p1_slice, "recall")
    fpr = _by_seed(blend_g, p1_slice, "fpr")
    if rec:
        out[f"{p1_slice}_blend_global_cut"] = {
            "per_seed": {str(s): rec[s] for s in sorted(rec)},
            "mean_recall": mean(rec.values()),
            "mean_realized_fpr": mean(fpr.values()) if fpr else None,
            "would_be_comparable": comparable_fn(mean(fpr.values())) if fpr else False,
        }

    macro: dict[int, list[float]] = {}
    fprs: dict[int, list[float]] = {}
    for (_sc, sd), v in sorted(alie_g.items()):
        macro.setdefault(sd, []).append(v["recall"])
        fprs.setdefault(sd, []).append(v["fpr"])
    if macro:
        out["alie_macro_global_cut"] = {
            "per_seed": {str(s): mean(v) for s, v in sorted(macro.items())},
            "mean": mean(mean(v) for v in macro.values()),
            "mean_realized_fpr": mean(mean(v) for v in fprs.values()),
        }
    return out


#: The complete set of reasons a fold may be excluded from a contrast. Both
#: reducers classify against THIS set, so a guard that exists on one side and
#: not the other is visible as a missing cause rather than as silent inclusion.
EXCLUSION_CAUSES = ("census_only", "coverage_floor", "fpr_interval", "unpaired")
_CAUSE_PHRASE = {
    "census_only": "census-only: uncovered honest side",
    "coverage_floor": f"below the {G2_MIN_SCORED_MALICIOUS}-row coverage floor",
    "fpr_interval": "out-of-interval realized FPR (§ 3.2 comparability)",
    "unpaired": "no seed is present on both sides, so nothing is paired",
}
_EXCLUSION_NOTE = {
    "census_only": "census only — uncovered honest side, no operating point",
    "coverage_floor": "no statistic computed (pre-registered coverage rule)",
    "fpr_interval": ("realized FPR outside the comparability interval; "
                     "no statistic computed"),
    "unpaired": ("no statistic computed — the two sides share no seed, so the "
                 "paired estimand is undefined (this is a CENSUS fact about "
                 "the corpus, not an operating-point failure)"),
}

#: Which reducers can REALIZE which cause, and why the others cannot.
#:
#: The cause VOCABULARY is shared — one universe, so a guard that exists on one
#: side and not the other shows up as a missing cause rather than as silent
#: inclusion (rounds 14/16). REALIZABILITY is not shared, and round 29 is where
#: that first mattered: `unpaired` is structurally impossible at the fold grain.
#: Stating the impossibility here, with its structural reason, is what keeps the
#: two sides from quietly diverging — the invariant test asserts BOTH halves,
#: that the realizable causes fire and that the unrealizable ones cannot.
CAUSE_REALIZABILITY = {
    # `g2_scored_rows_contrast` and `exp048_full_coverage_contrast`
    "fold_grain": {
        "realizable": ("census_only", "coverage_floor", "fpr_interval"),
        "unrealizable": {
            "unpaired": ("both arms are scored on the SAME (scenario, seed) "
                         "cell, so pairing is by construction and a unit "
                         "cannot exist on one side only"),
        },
    },
    # `alie_fixed_baseline_contrasts` — the § 4 secondary 7 seed-macro grain
    "seed_macro": {
        "realizable": ("fpr_interval", "unpaired"),
        "unrealizable": {
            "census_only": ("the population IS the ALIE-bearing scored cells; a "
                            "cell with no operating point never enters the "
                            "macro, so there is no census-only unit to exclude"),
            "coverage_floor": ("the 30-row floor is the G2 scored-rows rule "
                               "(§ 1 DECISION G) and binds on the `tge_score` "
                               "non-null population; secondary 7 scores the "
                               "full population, so no floor binds"),
        },
    },
}


def exclusion_fields(cause: str, detail: str = "") -> dict:
    """The three fields every excluded node publishes, built exactly ONE way.

    Round 20 found the same six facts carrying two vocabularies on two sides of
    one branch. Every consumer — this module and the blend-margin reducer in
    `adjudicate_h2prime` — builds its halt prose here, so a cause cannot acquire
    a second phrasing by being written out a second time.
    """
    if cause not in EXCLUSION_CAUSES:
        raise KeyError(f"{cause!r} is not a registered exclusion cause; "
                       f"add it to EXCLUSION_CAUSES and CAUSE_REALIZABILITY")
    return {
        "exclusion_cause": cause,
        "reason": ((f"{detail} — " if detail else "")
                   + _CAUSE_PHRASE[cause] + "; no statistic computed"),
        "note": _EXCLUSION_NOTE[cause],
    }


def _exclusion_cause(cell: dict, comparable_fn) -> str | None:
    """Why this fold cannot enter a contrast, or None if it can.

    Order matters only for reporting: a census-only cell has no operating point
    at all, so it is reported as such rather than as an FPR failure.
    """
    if cell.get("status") == "CENSUS ONLY":
        return "census_only"
    if cell.get("n_mal_scored", 0) < G2_MIN_SCORED_MALICIOUS:
        return "coverage_floor"
    if comparable_fn is not None and not (comparable_fn(cell.get("det_fpr"))
                                          and comparable_fn(cell.get("tge_fpr"))):
        return "fpr_interval"
    return None


def g2_scored_rows_contrast(scored: dict, scenarios: list[str],
                            expected_seeds: list[int] | None = None,
                            attacks: list[str] | None = None,
                            sign_fn=None, ci_fn=None,
                            ci_retained_fn=None, comparable_fn=None) -> dict:
    """§ 1 DECISION G2-EXTENDED, first half — matched TGE-vs-H2′ contrast on the
    CONFIRMATORY logs, restricted to `tge_score` non-null rows.

    Krum-filtered rows never reach the TGE stage, so `tge_score` is null on a
    large fraction of rows (dev: 55 % overall). Per the pre-registered mechanics
    the contrast runs ONLY on non-null rows, BOTH detectors are re-scored on
    that reduced population, the coverage fraction is reported for every slice,
    and any slice with fewer than 30 scored malicious rows is reported
    NOT COMPARABLE with no statistic computed. Dropped rows are never imputed
    as detected or missed.

    **The 30-row floor binds PER FOLD, not on the seed-summed aggregate.** Each
    (scenario, seed) cell is one fold's scoring pass, and the registered rule
    disqualifies a fold that scored too few malicious rows. Summing first would
    let ten 4-row folds masquerade as one comparable n = 40 slice — the exact
    reading this note exists to forbid. Disqualified folds contribute NOTHING:
    not to the mean, not to the count.
    """
    cells = scored.get("g2") or {}
    out = {"_note": ("REPORTED, NON-ADJUDICATING (§ 1 DECISION G2-EXTENDED). "
                     "Scored-rows-only contrast; dropped rows are NEVER imputed."),
           "min_scored_malicious_for_comparability": G2_MIN_SCORED_MALICIOUS,
           "comparability_grain": "PER FOLD — one (scenario, seed) cell",
           "scenarios": {}}
    n_expected = len(expected_seeds) if expected_seeds is not None else None
    for scen in scenarios:
        sc_cells = {k: v for k, v in sorted(cells.items()) if k[0] == scen}
        if not sc_cells:
            continue
        # A CENSUS ONLY cell carries census facts and no contrast: its honest
        # side is uncovered, so there is no operating point and no det/tge
        # recall. It can still clear the 30-row malicious floor, so the floor
        # alone would admit it into a blend it cannot participate in.
        causes = {k: _exclusion_cause(v, comparable_fn) for k, v in sc_cells.items()}
        comparable_cells = {k: v for k, v in sc_cells.items() if causes[k] is None}
        disqualified = {k: v["n_mal_scored"] for k, v in sc_cells.items()
                        if k not in comparable_cells}
        census_only = sorted(f"seed {k[1]}" for k, v in sc_cells.items()
                             if causes[k] == "census_only")
        by_cause = {c: sorted(f"seed {k[1]}" for k in causes if causes[k] == c)
                    for c in EXCLUSION_CAUSES}
        n_total = sum(v["n_mal_total"] for v in sc_cells.values())
        n_honest_total = sum(v["n_honest_total"] for v in sc_cells.values())
        entry = {
            "n_folds": len(sc_cells),
            # The expected fold count comes from the REGISTERED seed set, not
            # from the cells present, so a wholly missing fold is visible.
            "n_folds_expected": n_expected,
            "missing_folds": (
                [f"seed {sd}" for sd in expected_seeds
                 if (scen, sd) not in sc_cells]
                if expected_seeds is not None else None),
            "n_folds_comparable": len(comparable_cells),
            "disqualified_folds": {f"seed {k[1]}": n for k, n in sorted(disqualified.items())},
            "census_only_folds_excluded_from_contrast": census_only,
            "exclusion_causes": {c: len(v) for c, v in by_cause.items()},
            "excluded_folds_by_cause": by_cause,
            "coverage_fraction_malicious": (
                sum(v["n_mal_scored"] for v in sc_cells.values()) / n_total
                if n_total else None),
            "n_scored_malicious_all_folds": sum(
                v["n_mal_scored"] for v in sc_cells.values()),
            "n_malicious_total": n_total,
            "coverage_fraction_honest": (
                sum(v["n_honest_scored"] for v in sc_cells.values()) / n_honest_total
                if n_honest_total else None),
            # A disqualified fold publishes NO contrast statistic — the
            # registered rule is "no statistic computed" for that unit, so the
            # entry carries its status and row count and nothing else. Printing
            # a recall beside NOT COMPARABLE invites exactly the reading the
            # rule forbids.
            # Census facts (scored count, eligible total, coverage) ride on
            # EVERY branch; only contrast STATISTICS are suppressed. 18/18 and
            # 18/180 are different stories and must not print identically.
            "per_fold": {
                f"{k[0]}x{k[1]}": (
                    {**_census(v), "status": "COMPUTED",
                     "det_recall": v["det_recall"], "tge_recall": v["tge_recall"],
                     "det_fpr": v["det_fpr"], "tge_fpr": v["tge_fpr"]}
                    if k in comparable_cells else
                    {**_census(v), "status": "NOT COMPARABLE",
                     **({"detector_realized_fpr": v.get("det_fpr"),
                         "tge_realized_fpr": v.get("tge_fpr"),
                         "detector_comparable": comparable_fn(v.get("det_fpr")),
                         "tge_comparable": comparable_fn(v.get("tge_fpr"))}
                        if causes[k] == "fpr_interval" else {}),
                     "note": _EXCLUSION_NOTE[causes[k]]}
                ) for k, v in sc_cells.items()},
        }
        if not comparable_cells:
            entry["status"] = "NOT COMPARABLE"
            # Halt reasons name their ACTUAL cause, not the floor by default.
            present = [c for c in EXCLUSION_CAUSES if by_cause[c]]
            entry["reason"] = (
                f"all {len(sc_cells)} folds excluded — "
                + ", ".join(f"{len(by_cause[c])} {_CAUSE_PHRASE[c]}"
                            for c in present)
                + "; no statistic computed")
        else:
            entry["status"] = "COMPUTED"
            entry["computed_over_n_folds"] = len(comparable_cells)
            entry["detector"] = {
                "mean_recall": mean(v["det_recall"] for v in comparable_cells.values()),
                "mean_realized_fpr": mean(v["det_fpr"] for v in comparable_cells.values()),
                "per_seed": {str(k[1]): v["det_recall"]
                             for k, v in comparable_cells.items()},
            }
            entry["tge"] = {
                "mean_recall": mean(v["tge_recall"] for v in comparable_cells.values()),
                "mean_realized_fpr": mean(v["tge_fpr"] for v in comparable_cells.values()),
                "per_seed": {str(k[1]): v["tge_recall"]
                             for k, v in comparable_cells.items()},
            }
            entry["margin_detector_minus_tge"] = (
                entry["detector"]["mean_recall"] - entry["tge"]["mean_recall"])
            # The registered paired form, same as the EXP-048 contrast: per-seed
            # differences over the COMPARABLE folds only, with the frozen
            # Student-t CI. Reported-only, like everything in this block.
            seeds_ok = sorted(k[1] for k in comparable_cells)
            diffs = [comparable_cells[(scen, sd)]["det_recall"]
                     - comparable_cells[(scen, sd)]["tge_recall"] for sd in seeds_ok]
            entry["per_seed_diff"] = {str(sd): d for sd, d in zip(seeds_ok, diffs)}
            if sign_fn is not None:
                entry["sign_test"] = sign_fn(diffs)
            if ci_retained_fn is not None and ci_fn is not None:
                entry["ci95_student_t_on_paired_diff"] = ci_retained_fn(diffs, ci_fn)

        # ---- PER-ATTACK-FAMILY slices: the floor binds here too ------------
        # A fold can clear 30 pooled rows while one family contributes only a
        # handful; that family's contrast is not comparable even though the
        # fold's blend is. Statistics are suppressed for the family, not for
        # the slice, and the scorer's per-family census is kept either way.
        fam_entries: dict = {}
        for A in (attacks or []):
            fam_cells = {k: v["per_family"][A] for k, v in sc_cells.items()
                         if (v.get("per_family") or {}).get(A) is not None}
            # EVERY fold is classified for EVERY registered family, not just
            # the folds the scorer materialized an entry for. A zero-covered
            # family and a family under a census-only parent are excluded for
            # REASONS, and iterating only the emitted entries would keep those
            # reasons out of the cause counts and out of the symmetry
            # invariant — the folds would be censused but uncaused.
            fam_causes, fam_census = {}, {}
            for k in sorted(sc_cells):
                parent_census_only = sc_cells[k].get("status") == "CENSUS ONLY"
                cell = fam_cells.get(k)
                if cell is not None:
                    # § 3.2 binds at the FAMILY-SLICE grain too: a family whose
                    # OWN operating point is out of interval is not comparable
                    # even when the pooled fold's is inside.
                    cause = _exclusion_cause(
                        {**cell, "status": ("CENSUS ONLY" if parent_census_only
                                            else None)},
                        comparable_fn)
                    base = _census(cell)
                    evidence = ({"detector_realized_fpr": cell.get("det_fpr"),
                                 "tge_realized_fpr": cell.get("tge_fpr"),
                                 "detector_comparable": comparable_fn(cell.get("det_fpr")),
                                 "tge_comparable": comparable_fn(cell.get("tge_fpr"))}
                                if cause == "fpr_interval" else {})
                    note = _EXCLUSION_NOTE[cause] if cause else None
                else:
                    # Synthesized fold: no scorer entry for this family.
                    eligible = (sc_cells[k].get("family_eligible_totals") or {}
                                ).get(A, 0)
                    covered = (sc_cells[k].get("family_covered_totals") or {}
                               ).get(A, 0)
                    cause = "census_only" if parent_census_only else "coverage_floor"
                    base = _census({
                        "n_mal_scored": covered, "n_mal_total": eligible,
                        "n_honest_scored": sc_cells[k].get("n_honest_scored"),
                        "n_honest_total": sc_cells[k].get("n_honest_total")})
                    evidence = {}
                    note = ("census only — uncovered honest side, no operating point"
                            if parent_census_only else
                            ("no contrast for this family in this fold" if covered
                             else ("family had eligible malicious rows but TGE "
                                   "scored none of them" if eligible
                                   else "family not scheduled in this fold")))
                fam_causes[k] = cause
                if cause is None:
                    # A RETAINED family fold serializes BOTH SIDES' readouts,
                    # not just the slice mean and the paired difference. The
                    # registered output is a MATCHED contrast: a difference
                    # that cannot be audited back to the two recalls and the
                    # two realized operating points that produced it is not a
                    # matched contrast, it is a number. This mirrors what the
                    # EXP-048 retained folds serialize (round 13) — same key
                    # names, so the two sides of the study read with one
                    # vocabulary rather than two.
                    fam_census[f"{k[0]}x{k[1]}"] = {
                        **base, "status": "COMPUTED",
                        "detector_recall": cell["det_recall"],
                        "tge_recall": cell["tge_recall"],
                        # Realized FPRs are MEASUREMENTS and ride unconditionally.
                        "detector_realized_fpr": cell.get("det_fpr"),
                        "tge_realized_fpr": cell.get("tge_fpr"),
                        # The comparability annotations ride on the RETAINED
                        # rows too: "in interval" is a claim this row makes,
                        # and a claim only present on the rows that fail is not
                        # a claim a reader can check. They are the GATE's
                        # verdict, so like `sign_fn` and `ci_fn` in this same
                        # reducer they appear only when the gate was supplied —
                        # asserting `True` with no gate behind it would be the
                        # more dangerous output.
                        **({"detector_comparable": comparable_fn(cell.get("det_fpr")),
                            "tge_comparable": comparable_fn(cell.get("tge_fpr"))}
                           if comparable_fn is not None else {}),
                    }
                else:
                    fam_census[f"{k[0]}x{k[1]}"] = {
                        **base, "status": "NOT COMPARABLE", **evidence,
                        "note": note}
            ok = {k: v for k, v in fam_cells.items() if fam_causes.get(k) is None}
            census = fam_census
            fe = {
                "n_folds": len(sc_cells),
                # Counted from the COVERED-ROW tally, not from whether the
                # scorer materialized an entry: a census-only parent leaves
                # per_family empty while its families may well have covered
                # malicious rows, and counting entries would report those
                # families as zero-coverage.
                "n_folds_with_covered_rows": sum(
                    1 for k in sc_cells
                    if (k in fam_cells
                        or (sc_cells[k].get("family_covered_totals") or {}
                            ).get(A, 0) > 0)),
                "n_folds_comparable": len(ok),
                "exclusion_causes": {
                    c: sum(1 for x in fam_causes.values() if x == c)
                    for c in EXCLUSION_CAUSES},
                # AGGREGATE COVERAGE for the family slice, summed over EVERY
                # fold — partially covered and disqualified included. A fold
                # count says how many folds were excluded; it does not say how
                # much of the family this slice actually scored, and those are
                # different questions with different answers. Suppression is
                # for contrast statistics; coverage is a census fact and rides
                # on every branch. Split honest/malicious per DECISION G item 3.
                "coverage_population": (
                    "every fold in this family slice, including disqualified"),
                "n_scored_malicious_all_folds": sum(
                    r["n_mal_scored"] or 0 for r in census.values()),
                "n_malicious_total_all_folds": sum(
                    r["n_mal_total"] or 0 for r in census.values()),
                "coverage_fraction_malicious": _frac(
                    sum(r["n_mal_scored"] or 0 for r in census.values()),
                    sum(r["n_mal_total"] or 0 for r in census.values())),
                "n_scored_honest_all_folds": sum(
                    r["n_honest_scored"] or 0 for r in census.values()),
                "n_honest_total_all_folds": sum(
                    r["n_honest_total"] or 0 for r in census.values()),
                "coverage_fraction_honest": _frac(
                    sum(r["n_honest_scored"] or 0 for r in census.values()),
                    sum(r["n_honest_total"] or 0 for r in census.values())),
                "census_per_fold": census,
            }
            if not ok:
                fe["status"] = "NOT COMPARABLE"
                present = [c for c in EXCLUSION_CAUSES
                           if fe["exclusion_causes"][c]]
                fe["reason"] = (
                    f"every {A} fold excluded — "
                    + ", ".join(f"{fe['exclusion_causes'][c]} {_CAUSE_PHRASE[c]}"
                                for c in present)
                    + "; no statistic computed for this family")
            else:
                seeds_f = sorted(k[1] for k in ok)
                dif = [ok[(scen, sd)]["det_recall"] - ok[(scen, sd)]["tge_recall"]
                       for sd in seeds_f]
                fe["status"] = "COMPUTED"
                fe["detector_mean_recall"] = mean(ok[(scen, sd)]["det_recall"]
                                                  for sd in seeds_f)
                fe["tge_mean_recall"] = mean(ok[(scen, sd)]["tge_recall"]
                                             for sd in seeds_f)
                fe["margin_detector_minus_tge"] = mean(dif)
                fe["per_seed_diff"] = {str(sd): d for sd, d in zip(seeds_f, dif)}
                if sign_fn is not None:
                    fe["sign_test"] = sign_fn(dif)
                if ci_retained_fn is not None and ci_fn is not None:
                    fe["ci95_student_t_on_paired_diff"] = ci_retained_fn(dif, ci_fn)
            fam_entries[A] = fe
        if fam_entries:
            entry["per_attack_family"] = fam_entries
        out["scenarios"][scen] = entry
    return out



# The EXP-048 standalone-TGE arm (§ 4 secondary 9) lives in
# `h2prime_exp048.py` — a different corpus with its own registered seed
# universe and its own pre-read gate.

"""Band adjudication for H2′ — P1, P2, their gates, and the conjunction.

This module is the adjudicating core: every rule here is transcribed from the
ratified pre-registration (v1.15 § 3.2 / § 4, as amended by v1.15b § 1.2), and
none of it is selectable at analysis time. The reported-only sensitivities live
in `h2prime_secondaries.py`; the scoring engine lives in the driver.

Split out of `adjudicate_h2prime.py` (2026-08-12) for file size; behaviour
unchanged, verified by a byte-identical dev-smoke output across the split.
"""
from __future__ import annotations

from statistics import mean, pstdev

from h2prime_common import (
    ALPHA, BASELINES, COMPARABILITY_INTERVAL, HardStop, P1_FLOOR, P1_SLICE,
    P2_POPULATION_FAMILY, Profile, R, SCENARIOS, _fmt, canonical_device_id,
    comparable, exact_sign_p, student_t_ci,
)

def _slice_by_seed(cells: dict[tuple[str, int], dict], scen: str, field: str) -> dict[int, float]:
    return {sd: v[field] for (sc, sd), v in sorted(cells.items()) if sc == scen}


def adjudicate_p1(scored: dict, profile: Profile,
                  registered_seeds: list[int] | None = None) -> dict:
    """§ 4 (P1): S4 blended-LOAO recall mean ≥ 0.35, comparability-gated.

    Like P2, the seed universe comes from the REGISTERED set when supplied. A
    missing S4 unit is a corpus defect, so it HARD STOPS rather than returning
    INCONCLUSIVE — INCONCLUSIVE is terminal for the cohort under § 3.2, and a
    clerically fixable gap must never be recorded as a scientific halt.
    """
    if registered_seeds is not None:
        missing = [f"seed {sd} missing {P1_SLICE}" for sd in sorted(registered_seeds)
                   if (P1_SLICE, sd) not in scored["blend"]]
        extra = sorted({sd for (sc, sd) in scored["blend"] if sc == P1_SLICE}
                       - set(registered_seeds))
        if missing or extra:
            raise HardStop(
                "P1 STRUCTURAL POPULATION GATE: the "
                f"{P1_SLICE} blended population is not the registered seed set. "
                "Nothing is adjudicated."
                + ("\n  " + "\n  ".join(missing) if missing else "")
                + (f"\n  unregistered seeds present: {extra}" if extra else "")
            )
    recall_by_seed = _slice_by_seed(scored["blend"], P1_SLICE, "recall")
    fpr_by_seed = _slice_by_seed(scored["blend"], P1_SLICE, "fpr")
    seeds = sorted(recall_by_seed)
    vals = [recall_by_seed[s] for s in seeds]
    fprs = [fpr_by_seed[s] for s in seeds]

    result = {
        "band": "P1 — blended-LOAO recall@10%FPR on the S4 switching surface ≥ 0.35",
        "slice": P1_SLICE,
        "floor": P1_FLOOR,
        "n_units": len(vals),
        "per_seed_recall": {str(s): recall_by_seed[s] for s in seeds},
        "per_seed_realized_blended_fpr": {str(s): fpr_by_seed[s] for s in seeds},
        "mean_recall": mean(vals) if vals else None,
        "sd_population": pstdev(vals) if len(vals) > 1 else 0.0,
        "realized_blended_fpr": mean(fprs) if fprs else None,
        "comparability_interval": list(COMPARABILITY_INTERVAL),
        "comparability_basis": (
            "the adjudicated readout is the mean over the per-seed S4 blended "
            "recalls; its realized FPR is the mean over the per-seed S4 blended "
            "FPRs (§ 3.2 'its realized FPR')"
        ),
        "ci95_student_t": student_t_ci(vals, profile.t_crit, profile.df),
        "ci_is_reported_not_adjudicating": True,
    }
    if len(vals) != profile.n_seeds:
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            f"expected {profile.n_seeds} out-of-fold S4 units, got {len(vals)}"
        )
        return result

    realized = result["realized_blended_fpr"]
    result["per_seed_comparability"] = {str(s): comparable(fpr_by_seed[s]) for s in seeds}
    result["all_per_seed_comparable"] = all(result["per_seed_comparability"].values())
    if not comparable(realized):
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            f"§ 3.2 calibration-integrity halt: realized blended FPR {realized:.4f} "
            "is outside the frozen comparability interval "
            f"[{COMPARABILITY_INTERVAL[0]}, {COMPARABILITY_INTERVAL[1]}]"
        )
        return result
    result["verdict"] = "PASS" if result["mean_recall"] >= P1_FLOOR else "FAIL"
    result["reason"] = (
        f"mean blended recall {result['mean_recall']:.4f} "
        f"{'≥' if result['verdict'] == 'PASS' else '<'} floor {P1_FLOOR:.2f} "
        f"at realized blended FPR {realized:.4f}"
    )
    return result


def _p2_population_gate(det_cells: dict, base_cells: dict, seeds: list[int],
                        derived: bool = False) -> None:
    """Structural gates on P2's population — enforced by the executor itself.

    Custody is a separate layer and this must not depend on it: a corpus defect
    has to halt loudly here rather than be averaged over silently.

    `seeds` MUST be the registered (sealed) seed set, not one derived from the
    scored cells. A derived universe is defined by the very data it is meant to
    check: a seed whose five cells are all absent simply disappears from it, the
    cross-product shrinks to 9 × 5, and every completeness test still passes.

    Gate 1 — FROZEN POPULATION (§ 4.1 cell table). `alie` is scheduled in ALL
    five scenarios, so P2's estimand is a macro-average over exactly S0…S4 for
    every seed. A seed missing a cell would contribute a macro-average over
    fewer scenarios while still counting toward the ≥ 9-of-10 sign rule — a
    different estimand wearing the same name.

    Gate 2 — ENUMERATED-BASELINE COMPLETENESS (§ 3.1). The comparator is the
    maximum over all THREE instruments. Building it from a subset would drop a
    potentially stronger arm and inflate the detector's margin. § 3.1 chose
    these three precisely because they exist on Krum+TGE logs, so on a clean
    corpus this can never fire; it exists so a dirty one halts.
    """
    missing_cells = [f"seed {sd} missing {scen}"
                     for sd in seeds for scen in SCENARIOS
                     if (scen, sd) not in det_cells]
    unregistered = sorted({sd for (_sc, sd) in det_cells} - set(seeds))
    if unregistered:
        raise HardStop(
            "P2 STRUCTURAL POPULATION GATE: scored cells carry seeds that are "
            f"not in the registered seed set: {unregistered}. Nothing is "
            "adjudicated."
        )
    if missing_cells:
        raise HardStop(
            "P2 STRUCTURAL POPULATION GATE: the ALIE-bearing population is not "
            "the frozen S0–S4 × seed cross product, so the macro-average "
            "estimand is undefined. Nothing is adjudicated."
            + ("\n  (seed universe was DERIVED from the scored cells, which "
               "cannot detect a wholly absent seed — pass registered_seeds)"
               if derived else "")
            + "\n  " + "\n  ".join(missing_cells)
        )

    incomplete = []
    for key in sorted(det_cells):
        for b, _t, _l in BASELINES:
            cell = base_cells[b].get(key)
            if cell is None:
                incomplete.append(f"{key[0]}×{key[1]} — {b} absent entirely")
                continue
            # ROW-LEVEL completeness, not key-level. A cell that scored the
            # instrument on a SUBSET of rows is a population mismatch: the
            # detector's recall is over every ALIE row in the cell, the
            # baseline's over fewer, and the two are then compared as if paired.
            n_mal_null = cell.get("mal_null_dropped", 0)
            n_hon_null = cell.get("honest_null_dropped", 0)
            # Calibration nulls matter as much as scored-row nulls: the cut
            # comes off the calibration honest population, so an instrument
            # null there takes its operating point from a DIFFERENT honest
            # population than the detector's, and the "matched operating point"
            # the margin claims is no longer matched.
            n_cal_null = cell.get("cal_null_dropped", 0)
            if n_mal_null or n_hon_null or n_cal_null:
                incomplete.append(
                    f"{key[0]}×{key[1]} — {b} null on {n_mal_null} ALIE row(s), "
                    f"{n_hon_null} scored honest row(s) and {n_cal_null} "
                    f"CALIBRATION row(s); the detector scored the full cell "
                    f"({cell.get('n_mal')} ALIE / {cell.get('n_honest')} honest "
                    "scored for this instrument)")
    if incomplete:
        raise HardStop(
            "P2 ENUMERATED-BASELINE COMPLETENESS GATE: the oracle maximum must "
            "be taken over all three instruments "
            f"({', '.join(b for b, _t, _l in BASELINES)}) on the SAME rows the "
            "detector scored; an absent instrument omits a potentially stronger "
            "arm, and a partially-null one compares populations that are not "
            "the same population. Nothing is adjudicated.\n  "
            + "\n  ".join(incomplete)
        )


def _tie_census(units: list[dict]) -> dict:
    """Tie structure at the arg-max, split three ways.

    The tie-break's verdict-relevance differs per case and a single count hides
    that: with NO tie the selection is forced by the data; with a FULL three-way
    tie any instrument could have been guarded; with a PARTIAL tie exactly two
    instruments were candidates and the frozen order chose between those two.
    The partial case is the one a reader cannot reconstruct without being told
    which instrument won, so it is named per cell.
    """
    none_, partial, full = 0, {}, 0
    for u in units:
        recalls = u.get("baseline_recalls") or {}
        if not recalls:
            continue
        top = max(recalls.values())
        tied = sorted(b for b, v in recalls.items() if round(v, 12) == round(top, 12))
        if len(tied) == 1:
            none_ += 1
        elif len(tied) == len(recalls):
            full += 1
        else:
            partial[f"{u['scenario']}x{u['seed']}"] = {
                "tied_instruments": tied,
                "tie_break_selected": u.get("argmax_baseline")}
    return {
        "n_cells_no_tie": none_,
        "n_cells_partial_tie": len(partial),
        "n_cells_full_tie": full,
        "n_cells_where_argmax_was_a_tie": len(partial) + full,
        "partial_tie_cells": partial,
    }


def adjudicate_p2(scored: dict, profile: Profile,
                  registered_seeds: list[int] | None = None) -> dict:
    """§ 4 (P2): detector > per-unit oracle max over the three baselines,
    equal-weight macro-averaged over ALIE-bearing scenarios, ≥ 9 of 10 strictly
    positive. Zeros carry no sign.

    `registered_seeds` is the SEALED seed set from the corpus layer. It is the
    seed universe the gate checks against, because a universe derived from the
    scored cells cannot detect a seed that is missing entirely — the
    cross-product would silently shrink to 9 × 5 and still look complete.
    """
    det_cells = scored["alie"]
    base_cells = scored["base_alie"]
    seeds = (sorted(registered_seeds) if registered_seeds is not None
             else sorted({sd for (_sc, sd) in det_cells}))
    _p2_population_gate(det_cells, base_cells, seeds,
                        derived=registered_seeds is None)

    units: list[dict] = []
    for (sc, sd), dv in sorted(det_cells.items()):
        candidates = {}
        for b, _t, _l in BASELINES:
            bv = base_cells[b].get((sc, sd))
            if bv is not None:
                candidates[b] = bv
        if not candidates:
            units.append({"scenario": sc, "seed": sd, "detector_recall": dv["recall"],
                          "detector_fpr": dv["fpr"], "oracle_max": None,
                          "argmax_baseline": None, "baseline_fpr": None,
                          "diff": None, "comparable_diagnostic": False,
                          "note": "no baseline computable on this unit"})
            continue
        # § 3.1 — the per-unit ORACLE MAXIMUM (a switching oracle).
        # v1.15b § 1.2 item 3 — the tie-break is FROZEN IN SPEC TEXT to the first
        # entry of the enumerated tuple (krum_score, L2_to_median, cos_to_median),
        # because § 4 (P2) expects exact-0.000 ties as the normal regime and the
        # arg-max identity under a tie decides WHICH instrument's FPR is guarded.
        # The recall value is tie-break-invariant; the guarded identity is not.
        omax = max(c["recall"] for c in candidates.values())
        argmax = next(b for b, _t, _l in BASELINES
                      if b in candidates and candidates[b]["recall"] == omax)
        units.append({
            "scenario": sc, "seed": sd,
            "detector_recall": dv["recall"], "detector_fpr": dv["fpr"],
            "detector_n_mal": dv["n_mal"],
            "oracle_max": omax, "argmax_baseline": argmax,
            "baseline_fpr": candidates[argmax]["fpr"],
            "baseline_recalls": {b: v["recall"] for b, v in sorted(candidates.items())},
            "diff": dv["recall"] - omax,
            # DIAGNOSTIC ONLY from v1.15b § 1.2: the guard binds at the
            # adjudicated-readout grain, not per unit. Retained as the
            # calibration-integrity evidence trail.
            "comparable_diagnostic": comparable(dv["fpr"])
            and comparable(candidates[argmax]["fpr"]),
        })

    by_seed_det: dict[int, list[float]] = {}
    by_seed_orc: dict[int, list[float]] = {}
    by_seed_tp: dict[int, list[int]] = {}
    by_seed_nm: dict[int, list[int]] = {}
    for u in units:
        if u["oracle_max"] is None:
            continue
        by_seed_det.setdefault(u["seed"], []).append(u["detector_recall"])
        by_seed_orc.setdefault(u["seed"], []).append(u["oracle_max"])
    for (sc, sd), dv in sorted(det_cells.items()):
        by_seed_tp.setdefault(sd, []).append(dv["n_tp"])
        by_seed_nm.setdefault(sd, []).append(dv["n_mal"])

    macro_det = {s: mean(v) for s, v in sorted(by_seed_det.items())}
    macro_orc = {s: mean(v) for s, v in sorted(by_seed_orc.items())}
    diffs = [macro_det[s] - macro_orc[s] for s in sorted(macro_det)]
    sign = exact_sign_p(diffs)
    w_plus, wilcoxon_p = R.exact_wilcoxon_onesided(diffs)

    result = {
        "band": (
            "P2 — detector recall@10%FPR strictly exceeds the per-unit oracle "
            f"maximum over {[b for b, _, _ in BASELINES]} in ≥ "
            f"{profile.p2_min_positive} of {profile.n_seeds} seeds"
        ),
        "population": "ALIE-bearing rows, all scenarios that schedule alie",
        "estimand": "equal-weight macro-average of ALIE-bearing scenario-cell recalls per seed",
        "required_strictly_positive": profile.p2_min_positive,
        "n_units_cells": len(units),
        "per_unit": units,
        "per_seed_detector_macro": {str(s): v for s, v in macro_det.items()},
        "per_seed_oracle_max_macro": {str(s): v for s, v in macro_orc.items()},
        "per_seed_diff": {str(s): macro_det[s] - macro_orc[s] for s in sorted(macro_det)},
        "sign_test": sign,
        "alpha": ALPHA,
        "wilcoxon_reported_not_adjudicating": {
            "w_plus": w_plus, "p_one_sided_exact": wilcoxon_p},
        "pooled_row_alie_recall_per_seed_sensitivity": {
            str(s): (sum(by_seed_tp[s]) / sum(by_seed_nm[s]) if sum(by_seed_nm[s]) else None)
            for s in sorted(by_seed_tp)
        },
        "comparability_interval": list(COMPARABILITY_INTERVAL),
    }

    # ---- v1.15b § 1.2 — THE ADJUDICATING COMPARABILITY GUARD ----------------
    # The [0.08, 0.12] guard binds at the grain of P2's ADJUDICATED READOUT:
    # the ROW-POOLED realized FPR of each side of the adjudicating contrast over
    # the ALIE-bearing scored population (item 1). The baseline side is the
    # per-cell ARG-MAX instrument — the one that actually composes the oracle
    # maximum — with its flag decisions pooled across cells (item 2), under the
    # frozen tuple tie-break (item 3). Either side outside ⇒ INCONCLUSIVE
    # (item 4). Pooling is exact integer arithmetic over flag counts, never a
    # mean of per-cell rates.
    scored_units = [u for u in units if u["baseline_fpr"] is not None]
    det_flagged = sum(det_cells[(u["scenario"], u["seed"])]["n_flagged_honest"]
                      for u in scored_units)
    det_honest = sum(det_cells[(u["scenario"], u["seed"])]["n_honest"]
                     for u in scored_units)
    base_flagged = sum(base_cells[u["argmax_baseline"]][(u["scenario"], u["seed"])]
                       ["n_flagged_honest"] for u in scored_units)
    base_honest = sum(base_cells[u["argmax_baseline"]][(u["scenario"], u["seed"])]
                      ["n_honest"] for u in scored_units)
    det_pooled = det_flagged / det_honest if det_honest else None
    base_pooled = base_flagged / base_honest if base_honest else None

    result["readout_grain_comparability"] = {
        "_authority": "v1.15b § 1.2 (row-pooled, arg-max baseline side, frozen tie-break)",
        "detector_pooled_fpr": det_pooled,
        "detector_flagged_honest": det_flagged, "detector_honest_rows": det_honest,
        "baseline_pooled_fpr": base_pooled,
        "baseline_flagged_honest": base_flagged, "baseline_honest_rows": base_honest,
        "detector_in_interval": comparable(det_pooled),
        "baseline_in_interval": comparable(base_pooled),
        "argmax_instrument_counts": {
            b: sum(1 for u in scored_units if u["argmax_baseline"] == b)
            for b, _t, _l in BASELINES},
        **_tie_census(scored_units),
    }

    # Sub-readout FPRs — REPORTED, never adjudicating (v1.15b § 1.2 final bullet).
    by_seed_dfpr: dict[int, list[float]] = {}
    by_seed_bfpr: dict[int, list[float]] = {}
    for u in scored_units:
        by_seed_dfpr.setdefault(u["seed"], []).append(u["detector_fpr"])
        by_seed_bfpr.setdefault(u["seed"], []).append(u["baseline_fpr"])
    per_instrument = {}
    for b, _t, _l in BASELINES:
        cells_b = [base_cells[b][k] for k in sorted(det_cells) if k in base_cells[b]]
        fl = sum(c["n_flagged_honest"] for c in cells_b)
        hn = sum(c["n_honest"] for c in cells_b)
        per_instrument[b] = {"pooled_fpr": (fl / hn if hn else None),
                             "in_interval": comparable(fl / hn) if hn else False}
    result["realized_fpr_diagnostic"] = {
        "_note": ("DIAGNOSTIC, NON-ADJUDICATING — the guard binds at the "
                  "readout grain above (v1.15b § 1.2)"),
        "per_unit_detector_in_interval": sum(
            1 for u in scored_units if comparable(u["detector_fpr"])),
        "per_unit_baseline_in_interval": sum(
            1 for u in scored_units if comparable(u["baseline_fpr"])),
        "per_unit_total": len(scored_units),
        "per_seed_macro_detector": {str(s): mean(v) for s, v in sorted(by_seed_dfpr.items())},
        "per_seed_macro_baseline": {str(s): mean(v) for s, v in sorted(by_seed_bfpr.items())},
        "per_instrument_pooled": per_instrument,
    }

    if len(diffs) != profile.n_seeds:
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            f"expected {profile.n_seeds} paired seed differences, got {len(diffs)}"
        )
        return result

    if not (comparable(det_pooled) and comparable(base_pooled)):
        outside = []
        if not comparable(det_pooled):
            outside.append(f"detector {_fmt(det_pooled)}")
        if not comparable(base_pooled):
            outside.append(f"arg-max baseline {_fmt(base_pooled)}")
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            "§ 3.2 calibration-integrity halt at the v1.15b § 1.2 readout grain: "
            + " and ".join(outside)
            + f" outside [{COMPARABILITY_INTERVAL[0]}, {COMPARABILITY_INTERVAL[1]}]"
        )
        return result

    result["verdict"] = "PASS" if sign["positive"] >= profile.p2_min_positive else "FAIL"
    result["reason"] = (
        f"{sign['positive']} strictly positive / {sign['negative']} negative / "
        f"{sign['zero']} zero paired differences; rule requires ≥ "
        f"{profile.p2_min_positive} strictly positive "
        f"(attained exact one-sided p = {sign['p_one_sided_exact']:.5f})"
    )
    return result



def strict_identity_block(rows: list[dict], primary: dict, strict: dict,
                          devices: dict[str, set[str]], profile: Profile,
                          registered_seeds: list[int] | None = None) -> dict:
    """v1.15b § 3 strict-identity-LOAO sensitivity — REPORTED, adjudicates nothing."""
    census = {}
    for A, banned in sorted(devices.items()):
        aliases = sorted({r["logical_cid"] for r in rows
                          if canonical_device_id(r["logical_cid"]) in banned})
        raw_named = sorted({r["logical_cid"] for r in rows
                            if r["malicious_gt"] and r.get("attack_type") == A})
        excluded_rows = sum(1 for r in rows
                            if canonical_device_id(r["logical_cid"]) in banned)
        census[A] = {
            "canonical_devices_exposed": sorted(banned),
            "n_canonical_devices": len(banned),
            "all_aliases_covered": aliases,
            "n_aliases_covered": len(aliases),
            "n_aliases_a_raw_logical_cid_key_would_have_MISSED":
                len([a for a in aliases if a not in raw_named]),
            "aliases_a_raw_logical_cid_key_would_have_MISSED":
                [a for a in aliases if a not in raw_named],
            "n_rows_excluded_from_fit_and_calibration": excluded_rows,
        }

    def s4(scored):
        rec = _slice_by_seed(scored["blend"], P1_SLICE, "recall")
        fpr = _slice_by_seed(scored["blend"], P1_SLICE, "fpr")
        return {"per_seed": {str(s): rec[s] for s in sorted(rec)},
                "mean": mean(rec.values()) if rec else None,
                "mean_realized_fpr": mean(fpr.values()) if fpr else None}

    p_s4 = s4(primary)
    block = {
        "_note": ("REPORTED, NON-ADJUDICATING (v1.15 § 3.2 mandatory sensitivity; "
                  "key FROZEN by v1.15b § 3 as the canonical base device lineage). "
                  "The exclusion applies to the DETECTOR's fit and calibration "
                  "populations; the unsupervised baselines are unchanged per "
                  "§ 3.2's 'everything else unchanged'."),
        "identity_key": "flowerfl.signal_logger.canonical_device_id (imported, not re-derived)",
        "exclusion_census": census,
        "s4_blend_primary": p_s4,
    }

    degen = strict.get("degenerate") or {}
    if degen:
        block["status"] = "UNDEFINED — structurally degenerate under RMC"
        block["degenerate_folds"] = degen
        block["finding"] = (
            "The RMC adversary SWITCHES strategy: the same device lineages carry "
            "every attack family, so excluding all devices that carry family F "
            "anywhere in the corpus removes the ENTIRE positive class and no "
            "detector can be fit. This is not a divergence measurement — it is "
            "the finding, recorded pre-data as the determination in v1.15b § 3: "
            "device-strict LOAO and RMC's switching adversary are mutually "
            "exclusive by construction, which is the structural reason the "
            "behavior-row grain is the only LOAO grain RMC admits. No fallback "
            "exclusion grain is invented here."
        )
        block["primary_arm_unaffected"] = True
        return block

    s_s4 = s4(strict)
    p2_strict = adjudicate_p2(strict, profile, registered_seeds)
    block.update({
        "status": "COMPUTED",
        "s4_blend_strict_identity": s_s4,
        "s4_blend_divergence": (
            None if p_s4["mean"] is None or s_s4["mean"] is None
            else s_s4["mean"] - p_s4["mean"]),
        "p2_sign_test_strict_identity": p2_strict["sign_test"],
        "p2_per_seed_diff_strict_identity": p2_strict["per_seed_diff"],
        "divergence_is_itself_the_finding": (
            "§ 3.2: 'If the two arms diverge materially, that divergence is itself "
            "the finding and is reported as such.'"),
    })
    return block


def overall_verdict(p1: dict, p2: dict) -> dict:
    """P1 ∧ P2. A failing band falsifies (§ 4.1); INCONCLUSIVE blocks a verdict."""
    verdicts = [p1["verdict"], p2["verdict"]]
    if "FAIL" in verdicts:
        overall = "FAIL — H2′ generalization claim FALSIFIED"
    elif "INCONCLUSIVE" in verdicts:
        overall = "INCONCLUSIVE — calibration-integrity halt (TERMINAL for this cohort)"
    else:
        overall = "PASS — P1 ∧ P2 both hold"
    return {"p1": p1["verdict"], "p2": p2["verdict"], "conjunction": overall,
            "terminal_protocol": (
                "§ 3.2: an INCONCLUSIVE band stays INCONCLUSIVE for this sealed "
                "cohort. The only permitted post-hoc actions are the enumerated "
                "clerical corrections (a) and (b); anything else requires a new "
                "dated pre-registration AND a freshly sealed cohort."
            )}


"""Human-readable rendering of the H2′ read — the verdict block and dry-run plan.

Presentation only: every number printed here is already in the results JSON, and
nothing in this module computes or decides anything.

Split out of `adjudicate_h2prime.py` (2026-08-12) for file size; behaviour
unchanged, verified by a byte-identical dev-smoke output across the split.
"""
from __future__ import annotations

from h2prime_common import (
    ALPHA, ATTACKS, BASELINES, COMPARABILITY_INTERVAL, P1_FLOOR, P1_SLICE,
    PINNED_NUMPY, PINNED_SKLEARN, SCENARIOS, _fmt,
)

def render_verdict_block(report: dict) -> str:
    prof = report["_meta"]["profile"]
    p1, p2 = report["P1"], report["P2"]
    v = report["verdict"]
    L: list[str] = []
    L.append("=" * 78)
    L.append(f"H2′ CONFIRMATORY ADJUDICATION — profile {prof['name']}"
             + ("" if prof["adjudicating"] else "  [NON-ADJUDICATING SELF-TEST]"))
    L.append("=" * 78)
    L.append(f"corpus       : {report['_meta']['n_cells']} cells, "
             f"{report['_meta']['n_seeds']} seeds, defense "
             f"{report['_meta']['defense_token']}, sources "
             f"{report['_meta']['cells_by_source']}")
    L.append(f"golden gate  : {report['_meta']['golden_gate']['status']} "
             f"({report['_meta']['golden_gate']['frozen_commit']})")
    L.append(f"rows loaded  : {report['_meta']['n_rows']}  "
             f"(census {report['_meta']['row_census']})")
    drops = {b: d for b, d in report["_meta"]["baseline_null_drop_summary"].items()
             if d["mal_rows_dropped_null"] or d["honest_rows_dropped_null"]
             or d["calibration_rows_dropped_null"]}
    L.append(f"baseline null-drops : {drops if drops else 'none'}")
    L.append("")
    L.append("--- P1 — blended-LOAO recall@10%FPR on S4 ≥ 0.35 ------------------")
    L.append(f"  mean blended recall     : {_fmt(p1['mean_recall'])}   (floor {P1_FLOOR})")
    ci = p1["ci95_student_t"]
    L.append(f"  95% Student-t CI        : [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]"
             f"  (df={ci['df']}, t={ci['t_crit']}, SE={_fmt(ci['se'])}) — REPORTED, not adjudicating")
    L.append(f"  realized blended FPR    : {_fmt(p1['realized_blended_fpr'])}   "
             f"(comparability [{COMPARABILITY_INTERVAL[0]}, {COMPARABILITY_INTERVAL[1]}])")
    L.append(f"  per-seed recall         : "
             + ", ".join(f"{s}={_fmt(v, 3)}" for s, v in p1["per_seed_recall"].items()))
    L.append(f"  VERDICT                 : {p1['verdict']} — {p1['reason']}")
    L.append("")
    L.append("--- P2 — ALIE-superiority over the per-unit oracle maximum ---------")
    st = p2["sign_test"]
    L.append(f"  paired seed differences : {st['positive']} positive / "
             f"{st['negative']} negative / {st['zero']} zero "
             f"(required ≥ {p2['required_strictly_positive']} strictly positive)")
    L.append(f"  exact one-sided p       : {st['p_one_sided_exact']:.5f} "
             f"(α = {ALPHA}; the COUNT governs, p is a readout)")
    L.append("  per-seed detector vs oracle-max (macro-average over ALIE scenarios):")
    for s in p2["per_seed_diff"]:
        L.append(f"    seed {s:>6} : detector {_fmt(p2['per_seed_detector_macro'][s], 3)} "
                 f"vs oracle {_fmt(p2['per_seed_oracle_max_macro'][s], 3)} "
                 f"→ diff {p2['per_seed_diff'][s]:+.4f}")
    rg = p2.get("readout_grain_comparability")
    if rg:
        L.append("  ADJUDICATING comparability guard (v1.15b § 1.2, row-pooled):")
        L.append(f"    detector       : {_fmt(rg['detector_pooled_fpr'])} "
                 f"({rg['detector_flagged_honest']}/{rg['detector_honest_rows']} honest "
                 f"rows flagged) — {'INSIDE' if rg['detector_in_interval'] else 'OUTSIDE'}")
        L.append(f"    arg-max baseline: {_fmt(rg['baseline_pooled_fpr'])} "
                 f"({rg['baseline_flagged_honest']}/{rg['baseline_honest_rows']}) — "
                 f"{'INSIDE' if rg['baseline_in_interval'] else 'OUTSIDE'}")
        L.append(f"    arg-max selections {rg['argmax_instrument_counts']}; "
                 f"ties {rg['n_cells_where_argmax_was_a_tie']}/"
                 f"{p2['n_units_cells']} cells (tie-break = frozen tuple order)")
    diag = p2.get("realized_fpr_diagnostic")
    if diag:
        L.append("  sub-readout FPRs (REPORTED, non-adjudicating):")
        L.append(f"    per-unit inside interval : detector "
                 f"{diag['per_unit_detector_in_interval']}/{diag['per_unit_total']}, "
                 f"arg-max baseline {diag['per_unit_baseline_in_interval']}"
                 f"/{diag['per_unit_total']}")
        L.append("    per-instrument pooled    : " + ", ".join(
            f"{b} {_fmt(v['pooled_fpr'])}" for b, v in diag["per_instrument_pooled"].items()))
    L.append(f"  VERDICT                 : {p2['verdict']} — {p2['reason']}")
    L.append("")
    sec = report["secondaries"]
    for scen in (P1_SLICE, "S3"):
        e = sec.get(f"{scen}_blend_margin")
        if not e or "oracle_max" not in e:
            continue
        L.append(f"--- SECONDARY (reported, no pass/fail): {scen} blend vs oracle max ---")
        L.append(f"  detector {_fmt(e['detector_blended_mean'])} vs oracle "
                 f"{_fmt(e['oracle_max']['mean'])} → margin "
                 f"{e['oracle_max']['margin_detector_minus_oracle']:+.4f}, "
                 f"{e['oracle_max']['sign_test']['positive']}/"
                 f"{len(e['oracle_max']['per_seed'])} seeds favour the detector, "
                 f"exact p = {e['oracle_max']['sign_test']['p_one_sided_exact']:.5f}")
        L.append(f"  detector realized FPR {_fmt(e['detector_mean_realized_fpr'])}")
    auc = sec.get("auc_per_scenario") or {}
    auc_scens = [s for s in SCENARIOS if s in auc]
    if auc_scens:
        L.append("--- SECONDARY (reported, no pass/fail): AUC per scenario -----------")
        L.append("  " + ", ".join(f"{s}={_fmt(auc[s]['mean'], 3)}" for s in auc_scens))
    gc = sec.get("global_cut_sensitivity") or {}
    gk = gc.get(f"{P1_SLICE}_blend_global_cut")
    if gk:
        L.append("--- SECONDARY (reported, no pass/fail): global-cut sensitivity ------")
        L.append(f"  {P1_SLICE} blend under the GLOBAL cut: "
                 f"{_fmt(gk['mean_recall'])} at realized FPR "
                 f"{_fmt(gk['mean_realized_fpr'])} "
                 f"(per-scenario arm adjudicates; global would "
                 f"{'be' if gk['would_be_comparable'] else 'NOT be'} comparable)")
    g2 = sec.get("g2_scored_rows_contrast") or {}
    if g2.get("scenarios"):
        L.append("--- SECONDARY (reported, no pass/fail): G2 matched TGE contrast -----")
        for s, e in g2["scenarios"].items():
            if e["status"] == "NOT COMPARABLE":
                L.append(f"  {s}: NOT COMPARABLE — {e['reason']}")
            else:
                L.append(f"  {s}: detector {_fmt(e['detector']['mean_recall'], 3)} vs "
                         f"TGE {_fmt(e['tge']['mean_recall'], 3)} "
                         f"(margin {e['margin_detector_minus_tge']:+.3f}, "
                         f"malicious coverage "
                         f"{_fmt(e['coverage_fraction_malicious'], 3)})")
    ex = sec.get("exp048_standalone_tge_full_coverage") or {}
    if ex.get("status") == "NOT RUN":
        L.append("--- SECONDARY: EXP-048 full-coverage TGE contrast — NOT RUN --------")
        L.append(f"  {ex['reason']}")
    si = sec.get("strict_identity_loao_sensitivity")
    if si:
        L.append("--- SECONDARY (reported, no pass/fail): strict-identity LOAO -------")
        L.append(f"  identity key : canonical base device lineage (v1.15b § 3)")
        for A, c in si["exclusion_census"].items():
            L.append(f"    {A:<15} {c['n_canonical_devices']} devices → "
                     f"{c['n_aliases_covered']} aliases, "
                     f"{c['n_rows_excluded_from_fit_and_calibration']} rows excluded "
                     f"({c['n_aliases_a_raw_logical_cid_key_would_have_MISSED']} alias(es) a "
                     f"raw logical_cid key would have leaked)")
        if si.get("status", "").startswith("UNDEFINED"):
            L.append(f"  STATUS       : {si['status']}")
            for A_, why in si["degenerate_folds"].items():
                L.append(f"    fold {A_:<15}: {why}")
            L.append("  " + si["finding"].replace("\n", " "))
            L.append("  The PRIMARY adjudication above is unaffected.")
        else:
            L.append(f"  S4 blend     : primary {_fmt(si['s4_blend_primary']['mean'])} vs "
                     f"strict-identity {_fmt(si['s4_blend_strict_identity']['mean'])} "
                     f"(divergence {si['s4_blend_divergence']:+.4f})")
            st2 = si["p2_sign_test_strict_identity"]
            L.append(f"  P2 sign count under strict identity: {st2['positive']} positive / "
                     f"{st2['negative']} negative / {st2['zero']} zero")
    wa = sec.get("window_aware_loao_sensitivity")
    if wa:
        L.append("--- SECONDARY (reported, no pass/fail): window-aware LOAO ---------")
        L.append("  rule         : fit also drops rows whose derived window saw "
                 "the held-out family")
        for s, c in wa["corpus_census"].items():
            share = c["share_of_malicious_rows_with_a_foreign_family"]
            L.append(f"    {s:<3} {c['n_malicious_rows_with_a_FOREIGN_family_in_window']}"
                     f"/{c['n_malicious_rows']} malicious rows carry a foreign "
                     f"family in-window ({_fmt(share, 3) if share is not None else 'n/a'})")
        if wa.get("status", "").startswith("UNDEFINED"):
            L.append(f"  STATUS       : {wa['status']}")
            for A_, why in wa.get("degenerate_folds", {}).items():
                L.append(f"    fold {A_:<15}: {why}")
            L.append("  " + wa["finding"].replace("\n", " "))
            L.append("  The PRIMARY adjudication above is unaffected.")
        else:
            q1 = wa["p1_quantity"]
            L.append(f"  rows dropped : {wa['n_window_excluded_rows_total']} beyond the "
                     "current-label rule, summed over folds")
            L.append(f"  {P1_SLICE} blend     : primary "
                     f"{_fmt(q1['primary']['mean_recall'])} vs window-aware "
                     f"{_fmt(q1['window_aware']['mean_recall'])} "
                     f"(delta {q1['delta_mean_recall']:+.4f}) at realized FPR "
                     f"{_fmt(q1['window_aware']['realized_blended_fpr'])}")
            q2 = wa["p2_quantity"]
            L.append(f"  P2 sign count: primary {q2['primary_sign_test']['positive']} "
                     f"positive vs window-aware "
                     f"{q2['window_aware_sign_test']['positive']} positive "
                     f"(delta {q2['delta_n_strictly_positive']:+d})")
            deltas = wa["per_scenario_blended_recall"]
            L.append("  per-scenario blend delta (window − primary): " + ", ".join(
                f"{s}={v['delta_window_minus_primary']:+.3f}"
                for s, v in deltas.items()
                if v["delta_window_minus_primary"] is not None))
    L.append("")
    L.append("=" * 78)
    L.append(f"OVERALL (P1 ∧ P2) : {v['conjunction']}")
    L.append(f"                    P1 = {v['p1']}   P2 = {v['p2']}")
    if not prof["adjudicating"]:
        L.append("NOTE: DEV-SMOKE profile — these numbers are an executor self-test on "
                 "already-disclosed dev data and adjudicate NOTHING.")
    L.append("=" * 78)
    return "\n".join(L)


def render_dry_run(plan_report: dict) -> str:
    L = ["=" * 78, "H2′ ADJUDICATION — DRY RUN (no row is scored)", "=" * 78]
    m = plan_report["_meta"]
    L.append(f"profile      : {m['profile']['name']}")
    L.append(f"assembly map : {m['map_path']}")
    L.append(f"             : sha256 {m['map_sha256']}")
    L.append(f"cells        : {m['n_cells']} ({m['cells_by_source']}), "
             f"defense {m['defense_token']}")
    L.append(f"seeds (asc)  : {m['seeds_ascending']}")
    L.append(f"golden gate  : {m['golden_gate']['status']}")
    L.append(f"sklearn      : {m['versions']['sklearn']} (pinned {PINNED_SKLEARN})"
             + ("" if m["versions"]["sklearn"] == PINNED_SKLEARN
                else "  ← DISCLOSE the version difference (§ 2.1a)"))
    L.append(f"numpy        : {m['versions']['numpy']} (pinned {PINNED_NUMPY})"
             + ("" if m["versions"]["numpy"] == PINNED_NUMPY
                else "  ← DISCLOSE the version difference (§ 2.2a item 7)"))
    L.append("")
    L.append("§ 2.2a rotation plan (1-based, ascending seed order):")
    L.append("  i | test   | calibration | fit (3 seeds)")
    for r in plan_report["rotation_plan"]:
        L.append(f"  {r['i']:>2}| {r['test']:<6} | {r['calibration']:<11} | "
                 + ", ".join(str(x) for x in r["fit"]))
    L.append("")
    L.append(f"planned fits : {len(plan_report['rotation_plan'])} rotations × "
             f"{len(ATTACKS)} LOAO folds = "
             f"{len(plan_report['rotation_plan']) * len(ATTACKS)} GBDT fits")
    L.append(f"planned cuts : {len(plan_report['rotation_plan'])} × {len(ATTACKS)} folds × "
             f"{len(SCENARIOS)} scenarios detector cuts, plus "
             f"{len(plan_report['rotation_plan'])} × {len(BASELINES)} × "
             f"{len(SCENARIOS)} baseline cuts")
    L.append("=" * 78)
    L.append("DRY RUN COMPLETE — no signal-log row was opened, no score computed.")
    return "\n".join(L)


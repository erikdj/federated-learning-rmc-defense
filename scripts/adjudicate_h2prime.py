"""ONE-SHOT P1 ∧ P2 adjudication executor for the H2′ confirmatory (EXP-051 + EXP-053).

Authority — every construction here is transcribed from, or imported from, the
frozen pre-registration; nothing in this file is a modeling choice:

  guide docs/reproduction/experiments.md
        detector, enumerated verbatim (GBDT hyperparameters, no preprocessing)
        § 2.2   per-scenario nested-independent 10 % FPR operating point
        § 2.2a  rotating fit / calibration / test partition (n = 10), quantile +
                strict-inequality conventions, held-out-family exclusion on the
                fit AND calibration populations
        § 2.2b  window-feature derivation (the committed builder + golden-hash gate)
        § 3.1   statistical unit, the three enumerated baselines, the per-unit
                oracle maximum
        § 3.2   blended-LOAO construction, the [0.08, 0.12] comparability
                interval, INCONCLUSIVE-is-terminal
        § 4     P1 (blended floor on S4) and P2 (ALIE-superiority sign count)
  exp   docs/experiments/EXP-051-h2prime-confirmatory.md § 3 (rotation table), § 5.1 (seal)

Correspondence to the dev pipeline (§ 2.1a "Training entry point"): the loader,
the window-feature builder, the cut and the flag rules are IMPORTED from
`reproduction/protocol/h2prime/revalidate_v115.py` (commit
dcef0f7) — the harness that produced every dev number the § 4 bands are grounded
on. The per-cell blend and the baseline treatment mirror
`reproduction/protocol/h2prime/blended_loao.py::main` statement
for statement; its `BASELINES` tuple is imported rather than retyped.

The corpus is selected by an explicit per-cell assembly map (never a prefix sync
or wildcard merge — the EXP-011 overwrite trap, EXP-053 § 2.3).

Usage — single invocation, no variant flags:

    python scripts/adjudicate_h2prime.py MAP.json --out RESULTS.json
    python scripts/adjudicate_h2prime.py MAP.json --dry-run      # plan only, scores nothing
    python scripts/adjudicate_h2prime.py MAP.json --out DEV.json --dev-smoke

`--dev-smoke` is an executor self-test on the already-disclosed EXP-011 dev
corpus. It REFUSES to run on any sealed seed, and its verdicts are labelled
non-adjudicating. There is no flag that changes how the sealed corpus is scored.

Module layout (split 2026-08-12 for file size; behaviour unchanged, verified by
a byte-identical dev-smoke output across the split):
  h2prime_common.py      frozen constants, profiles, frozen-harness imports, stats
  h2prime_corpus.py      assembly map, sealed seeds, § 2.2a rotation, loading
  h2prime_bands.py       P1, P2, their structural gates, the conjunction
  h2prime_secondaries.py the REPORTED-only secondaries (§ 4) + EXP-048 loader
  h2prime_report.py      the human-readable verdict block
  this file              the golden gate, the scoring pass, and the CLI
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2prime_common import (  # noqa: E402
    ALPHA, ATTACKS, AUC, BASELINES, COMPARABILITY_INTERVAL, CONFIRMATORY,
    DEV_SEEDS, DEV_SMOKE, FEATS, GBDT_PARAMS, GOLDEN, HardStop, P1_FLOOR,
    P1_SLICE, P2_POPULATION_FAMILY, PINNED_NUMPY, PINNED_SKLEARN, Profile, R,
    REGISTERED_DEFENSE_TOKEN,
    EXP048_REQUIRED_KEYS, Refusal, SCENARIOS, SCORING_INPUT_KEYS,
    SCORING_BASELINE_KEYS, SCORING_ENUM_CONTRACT, SCORING_VALUE_CONTRACT,
    SEALED_SEED_SHA256, TARGET_FPR, _fmt,
    canonical_device_id, ci_on_retained, comparable, exact_sign_p,
    student_t_ci,
)
from h2prime_corpus import (  # noqa: E402
    Cell, Rotation, design_matrix, exposed_devices, h2_confirm_seeds,
    load_assembly_map, load_cells, rotation_plan, sealed_seeds,
)
from h2prime_bands import (  # noqa: E402
    _slice_by_seed, adjudicate_p1, adjudicate_p2, overall_verdict,
    strict_identity_block,
)
from h2prime_report import render_dry_run, render_verdict_block  # noqa: E402
import h2prime_secondaries as SEC  # noqa: E402
import h2prime_exp048 as E48  # noqa: E402
import h2prime_schema as SCHEMA  # noqa: E402
from h2prime_scoring import (  # noqa: E402,F401
    BRACKET_TARGETS, _blended_at_cuts, _cut_at, golden_gate,
)

# --------------------------------------------------------------------------
# gate 0 — the § 2.2b golden-hash pre-condition
# --------------------------------------------------------------------------
def score_corpus(rows: list[dict], plan: list[Rotation],
                 exclude_devices: dict[str, set[str]] | None = None) -> dict:
    """Execute § 2.2a mechanically; return every per-cell quantity P1/P2 need.

    Structure and statement order mirror `blended_loao.py::main`: per rotation,
    fit one detector per held-out family on the three fit seeds (family-F
    attack-active rows removed), take per-scenario cuts on the calibration
    seed's honest rows, then score the test seed's cells.

    `exclude_devices` selects the v1.15b § 3 STRICT-IDENTITY arm: for fold F,
    every row of a device lineage that carries family F anywhere in the corpus
    is removed from the fit AND calibration populations. Scored test rows are
    untouched. Reported-only; it adjudicates nothing.
    """
    strict = exclude_devices is not None
    degenerate: dict[str, str] = {}
    blend: dict[tuple[str, int], dict] = {}
    alie: dict[tuple[str, int], dict] = {}
    per_family: dict[tuple[str, str, int], dict] = {}
    base_blend: dict[str, dict[tuple[str, int], dict]] = {b: {} for b, _, _ in BASELINES}
    base_alie: dict[str, dict[tuple[str, int], dict]] = {b: {} for b, _, _ in BASELINES}
    bracket: dict[tuple[str, int, float], dict] = {}   # § 6 reported bracket
    fit_census: list[dict] = []
    # Reported-secondary collectors (§ 4 secondaries 4 / 8 / 9). Gathered in
    # THIS pass because the sealed corpus is opened exactly once.
    blend_global: dict[tuple[str, int], dict] = {}
    alie_global: dict[tuple[str, int], dict] = {}
    g2: dict[tuple[str, int], dict] = {}

    by_seed_scen: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        by_seed_scen.setdefault((r["_seed"], r["_scen"]), []).append(r)

    for rot in plan:
        fit_rows_all = [r for r in rows if r["_seed"] in rot.fit]
        clfs, census = {}, {}
        for A in ATTACKS:
            # § 2.2a item 3 — held-out family excluded from the FIT population.
            tr = [r for r in fit_rows_all
                  if not (r["malicious_gt"] and r.get("attack_type") == A)]
            if strict:
                banned = exclude_devices[A]
                tr = [r for r in tr
                      if canonical_device_id(r["logical_cid"]) not in banned]
            X = design_matrix(tr)
            y = np.array([bool(r["malicious_gt"]) for r in tr])
            if strict and len(np.unique(y)) < 2:
                # The exclusion removed the entire positive class. Under RMC the
                # same devices carry every family, so a device-level hold-out of
                # family F removes ALL attackers. Recorded as structurally
                # UNDEFINED rather than fitted on one class or silently relaxed;
                # the primary arm is unaffected. See the report's
                # strict_identity_loao_sensitivity block.
                degenerate[A] = (
                    f"positive class empty after the v1.15b § 3 device-lineage "
                    f"exclusion ({len(tr)} fit rows, {int(y.sum())} positive)")
                continue
            clfs[A] = GradientBoostingClassifier(**GBDT_PARAMS).fit(X, y)
            census[A] = {"n_fit_rows": len(tr),
                         "n_fit_positive": int(y.sum()),
                         "n_excluded_family_rows": len(fit_rows_all) - len(tr)}
        fit_census.append({"rotation": rot.i, "test_seed": rot.test,
                           "calibration_seed": rot.calibration,
                           "fit_seeds": list(rot.fit), "per_fold": census})
        if strict and degenerate:
            # The blend needs all three fold detectors; with any fold undefined
            # the strict arm produces no comparable quantity at all. Stop here
            # and report, rather than emit a partial blend that looks like one.
            return {"blend": {}, "alie": {}, "per_family": {},
                    "base_blend": {b: {} for b, _t, _l in BASELINES},
                    "base_alie": {b: {} for b, _t, _l in BASELINES},
                    "fit_census": fit_census, "degenerate": degenerate,
                    "blend_global": {}, "alie_global": {}, "g2": {}}

        def pm(A: str, rs: list[dict]) -> np.ndarray:
            if not rs:
                return np.empty(0)
            return clfs[A].predict_proba(design_matrix(rs))[:, 1]

        # § 2.2 GLOBAL-cut arm (reported sensitivity): one cut per fold from the
        # calibration seed's honest rows POOLED across scenarios, mirroring
        # blended_loao.py's `cut_global`. The per-scenario arm adjudicates.
        cal_pooled = [r for r in rows
                      if r["_seed"] == rot.calibration and not r["malicious_gt"]]
        cut_global = {A: R.cut_from_calibration(pm(A, cal_pooled), higher_is_trust=False)
                      for A in ATTACKS} if cal_pooled else {}

        for scen in SCENARIOS:
            cal_all = by_seed_scen.get((rot.calibration, scen), [])
            # § 2.2a item 3 — the calibration population is honest-only, so the
            # held-out family is absent from it by construction.
            cal = [r for r in cal_all if not r["malicious_gt"]]
            # Strict-identity arm: the exclusion applies to the DETECTOR's
            # calibration population per fold. The baselines are untouched —
            # § 3.2 specifies "everything else unchanged", and they are
            # unsupervised instruments with no identity exposure to remove.
            cal_fold = {A: cal for A in ATTACKS}
            if strict:
                cal_fold = {
                    A: [r for r in cal
                        if canonical_device_id(r["logical_cid"]) not in exclude_devices[A]]
                    for A in ATTACKS
                }
            te = by_seed_scen.get((rot.test, scen), [])
            te_h = [r for r in te if not r["malicious_gt"]]
            fam = {A: [r for r in te if r["malicious_gt"] and r.get("attack_type") == A]
                   for A in ATTACKS}
            n_mal = sum(len(v) for v in fam.values())
            if not cal or not te_h or n_mal == 0:
                continue
            if any(not v for v in cal_fold.values()):
                raise HardStop(
                    "strict-identity arm emptied a calibration population at "
                    f"(scenario={scen}, calibration seed={rot.calibration}); no "
                    "cut can be taken. Reported as an error rather than filled in."
                )

            # § 2.2 / § 2.2a item 6 — one cut per (fold, scenario), risk direction.
            cuts = {A: R.cut_from_calibration(pm(A, cal_fold[A]), higher_is_trust=False)
                    for A in ATTACKS}
            honest_scores = {A: pm(A, te_h) for A in ATTACKS}
            honest_flag = {A: R.flagged(honest_scores[A], cuts[A], False)
                           for A in ATTACKS}
            honest_fpr = {A: float(honest_flag[A].mean()) for A in ATTACKS}
            honest_n_flagged = {A: int(honest_flag[A].sum()) for A in ATTACKS}

            # ---- P1 surface: blended-LOAO (§ 3.2) -------------------------
            b = _blended_at_cuts(fam, cuts, te_h, pm, n_mal, ATTACKS)
            blend[(scen, rot.test)] = {
                "recall": b["recall"], "fpr": b["fpr"],
                "n_mal": n_mal, "n_honest": len(te_h), "family_mix": b["family_mix"],
                "families_present": sorted(b["family_mix"]), "rotation": rot.i,
            }

            # ---- § 6 BRACKET — REPORTED ONLY, adjudicates nothing ---------
            # Same populations, same fold, same § 2.2a construction; only the
            # target FPR moves. 0.10 runs through this loop like the others, so
            # the equality with the adjudicated readout is a CHECKED fact.
            for t in BRACKET_TARGETS:
                t_cuts = {A: _cut_at(pm(A, cal_fold[A]), t, False) for A in ATTACKS}
                bt = _blended_at_cuts(fam, t_cuts, te_h, pm, n_mal, ATTACKS)
                bracket[(scen, rot.test, t)] = {
                    "recall": bt["recall"], "realized_fpr": bt["fpr"],
                    "n_mal": n_mal, "n_honest": len(te_h),
                }

            # ---- per-family fold recalls (mandatory secondary, § 4 item 3) --
            for A in ATTACKS:
                if not fam[A]:
                    continue
                f = R.flagged(pm(A, fam[A]), cuts[A], False)
                per_family[(A, scen, rot.test)] = {
                    "recall": float(f.mean()), "fpr": honest_fpr[A],
                    "n_mal": len(fam[A]), "n_honest": len(te_h), "cut": cuts[A],
                }

            # ---- P2 population: ALIE-bearing rows (§ 4) --------------------
            A = P2_POPULATION_FAMILY
            if fam[A]:
                alie_scores = pm(A, fam[A])
                f = R.flagged(alie_scores, cuts[A], False)
                alie[(scen, rot.test)] = {
                    "recall": float(f.mean()), "fpr": honest_fpr[A],
                    "n_mal": len(fam[A]), "n_tp": int(f.sum()),
                    "n_honest": len(te_h), "n_flagged_honest": honest_n_flagged[A],
                    "cut": cuts[A], "rotation": rot.i,
                    # § 4 secondary 4 — threshold-free AUC, ALIE vs honest, using
                    # the committed dev implementation's tie handling.
                    "auc": AUC(alie_scores, honest_scores[A]),
                }

            # ---- baselines, identical calibration treatment (§ 3.1) --------
            mal_pooled = [r for A2 in ATTACKS for r in fam[A2]]
            for b, trust, _label in BASELINES:
                # Null baseline scores are dropped per row, as
                # revalidate_v115.baseline_arm does. The drop counts are carried
                # into the report so a subset can never be scored silently.
                cv = [r[b] for r in cal if r.get(b) is not None]
                if not cv:
                    continue
                c = R.cut_from_calibration(cv, higher_is_trust=trust)
                sh = np.array([r[b] for r in te_h if r.get(b) is not None], dtype=float)
                if sh.size == 0:
                    continue
                b_hflag = R.flagged(sh, c, trust)
                b_fpr = float(b_hflag.mean())
                b_n_flagged = int(b_hflag.sum())
                nulls = {"cal_null_dropped": len(cal) - len(cv),
                         "honest_null_dropped": len(te_h) - int(sh.size)}
                sm = np.array([r[b] for r in mal_pooled if r.get(b) is not None],
                              dtype=float)
                if sm.size:
                    base_blend[b][(scen, rot.test)] = {
                        "recall": float(R.flagged(sm, c, trust).mean()),
                        "fpr": b_fpr, "n_mal": int(sm.size), "n_honest": int(sh.size),
                        "cut": c, "mal_null_dropped": len(mal_pooled) - int(sm.size),
                        **nulls,
                    }
                alie_rows = fam[P2_POPULATION_FAMILY]
                sa = np.array([r[b] for r in alie_rows if r.get(b) is not None],
                              dtype=float)
                if sa.size:
                    fa = R.flagged(sa, c, trust)
                    base_alie[b][(scen, rot.test)] = {
                        "recall": float(fa.mean()), "fpr": b_fpr,
                        "n_mal": int(sa.size), "n_tp": int(fa.sum()),
                        "n_honest": int(sh.size),
                        "n_flagged_honest": b_n_flagged, "cut": c,
                        "mal_null_dropped": len(alie_rows) - int(sa.size), **nulls,
                    }

            # ---- REPORTED secondary: the GLOBAL-cut arm (§ 2.2 / § 4 s8) ----
            if cut_global:
                g_tp, g_parts = 0, []
                for A2 in ATTACKS:
                    if not fam[A2]:
                        continue
                    g_tp += int(R.flagged(pm(A2, fam[A2]), cut_global[A2], False).sum())
                    g_parts.append((len(fam[A2]) / n_mal)
                                   * float(R.flagged(honest_scores[A2],
                                                     cut_global[A2], False).mean()))
                blend_global[(scen, rot.test)] = {
                    "recall": g_tp / n_mal, "fpr": sum(g_parts),
                    "n_mal": n_mal, "n_honest": len(te_h)}
                A2 = P2_POPULATION_FAMILY
                if fam[A2]:
                    gf = R.flagged(pm(A2, fam[A2]), cut_global[A2], False)
                    alie_global[(scen, rot.test)] = {
                        "recall": float(gf.mean()),
                        "fpr": float(R.flagged(honest_scores[A2],
                                               cut_global[A2], False).mean()),
                        "n_mal": len(fam[A2])}

            # ---- REPORTED secondary: G2-EXTENDED matched TGE contrast -------
            # § 1 DECISION G2-EXTENDED: `tge_score` non-null rows ONLY, BOTH
            # detectors re-scored on that reduced population, coverage reported,
            # dropped rows never imputed as detected or missed.
            cal_tge = [r for r in cal if r.get("tge_score") is not None]
            te_h_tge = [r for r in te_h if r.get("tge_score") is not None]
            mal_tge = {A2: [r for r in fam[A2] if r.get("tge_score") is not None]
                       for A2 in ATTACKS}
            n_mal_tge = sum(len(v) for v in mal_tge.values())
            eligible_totals = {A2: len(fam[A2]) for A2 in ATTACKS}
            if n_mal and not (cal_tge and te_h_tge and n_mal_tge):
                uncovered = sorted(
                    side for side, ok in (("calibration_honest", bool(cal_tge)),
                                          ("test_honest", bool(te_h_tge)),
                                          ("malicious", bool(n_mal_tge))) if not ok)
                # Eligible malicious rows exist but TGE scored none of them (or
                # the calibration/honest side is uncovered). No contrast is
                # possible, but the fold MUST still appear in the census —
                # otherwise "the family existed and TGE never scored it" is
                # indistinguishable from "the fold wasn't there".
                g2[(scen, rot.test)] = {
                    "status": "CENSUS ONLY",
                    "note": ("no contrast computable — uncovered side(s): "
                             + ", ".join(uncovered)),
                    "uncovered_sides": uncovered,
                    # The REAL malicious coverage is preserved. A missing honest
                    # or calibration side is its own census fact and is never a
                    # reason to zero a malicious population that WAS scored.
                    "n_mal_scored": n_mal_tge, "n_mal_total": n_mal,
                    "n_honest_scored": len(te_h_tge), "n_honest_total": len(te_h),
                    "det_fpr_family_mix": {},
                    "families_absent_zero_weight": sorted(ATTACKS),
                    "family_eligible_totals": eligible_totals,
                    # Preserve the PER-FAMILY covered counts too, not just the
                    # pooled one: the reducer builds a per-family census row for
                    # each family, and zeroing them here would report "TGE
                    # scored none of this family" for families it did score.
                    "family_covered_totals": {A2: len(mal_tge[A2]) for A2 in ATTACKS},
                    "per_family": {},
                }
            elif cal_tge and te_h_tge and n_mal_tge:
                # detector re-scored on the reduced population, per-scenario cut
                d_cuts = {A2: R.cut_from_calibration(pm(A2, cal_tge),
                                                     higher_is_trust=False)
                          for A2 in ATTACKS}
                d_tp = sum(int(R.flagged(pm(A2, mal_tge[A2]), d_cuts[A2], False).sum())
                           for A2 in ATTACKS if mal_tge[A2])
                # § 3.2 matched honest-side cost: the blend's realized FPR is the
                # FAMILY-MIX-WEIGHTED mean of the component detectors' honest
                # FPRs, w_F = n_mal_F / Σ n_mal_F, on the covered rows. A family
                # absent from this fold carries ZERO weight — averaging it in at
                # an equal share would price a detector that dispatched no rows.
                d_mix = {A2: len(mal_tge[A2]) / n_mal_tge for A2 in ATTACKS
                         if mal_tge[A2]}
                d_fpr = sum(
                    w * float(R.flagged(pm(A2, te_h_tge), d_cuts[A2], False).mean())
                    for A2, w in d_mix.items())
                # TGE re-scored on the same rows; tge_score is TRUST-direction
                t_cut = R.cut_from_calibration(
                    [r["tge_score"] for r in cal_tge], higher_is_trust=True)
                t_mal = np.array([r["tge_score"] for A2 in ATTACKS
                                  for r in mal_tge[A2]], dtype=float)
                t_hon = np.array([r["tge_score"] for r in te_h_tge], dtype=float)
                # Per-family detail on the SAME covered rows, so a paired
                # per-fold contrast never mixes populations.
                per_fam = {}
                for A2 in ATTACKS:
                    if not mal_tge[A2]:
                        continue
                    per_fam[A2] = {
                        "det_recall": float(R.flagged(
                            pm(A2, mal_tge[A2]), d_cuts[A2], False).mean()),
                        # The A2-fold detector's OWN honest FPR on the covered
                        # rows. A per-family reduction must be annotated with
                        # the family's operating point, not the blend's.
                        "det_fpr": float(R.flagged(
                            pm(A2, te_h_tge), d_cuts[A2], False).mean()),
                        "tge_fpr": float(R.flagged(t_hon, t_cut, True).mean()),
                        "n_honest_scored": len(te_h_tge),
                        "n_honest_total": len(te_h),
                        "tge_recall": float(R.flagged(
                            [r["tge_score"] for r in mal_tge[A2]], t_cut, True).mean()),
                        "n_mal_scored": len(mal_tge[A2]),
                        "n_mal_total": len(fam[A2]),
                    }
                g2[(scen, rot.test)] = {
                    "det_recall": d_tp / n_mal_tge,
                    "det_fpr": d_fpr,
                    "tge_recall": float(R.flagged(t_mal, t_cut, True).mean()),
                    "tge_fpr": float(R.flagged(t_hon, t_cut, True).mean()),
                    "n_mal_scored": n_mal_tge, "n_mal_total": n_mal,
                    "n_honest_scored": len(te_h_tge), "n_honest_total": len(te_h),
                    "det_fpr_family_mix": d_mix,
                    # Eligible malicious rows per family REGARDLESS of TGE
                    # coverage, so a family with zero covered rows can still be
                    # censused as "N eligible / 0 covered" rather than "0 / 0".
                    "family_eligible_totals": eligible_totals,
                    "families_absent_zero_weight": sorted(
                        A2 for A2 in ATTACKS if A2 not in d_mix),
                    "per_family": per_fam,
                }

    return {"blend": blend, "alie": alie, "per_family": per_family,
            "base_blend": base_blend, "base_alie": base_alie,
            "bracket": bracket,
            "fit_census": fit_census, "degenerate": degenerate,
            "blend_global": blend_global, "alie_global": alie_global, "g2": g2}




def _fmt_blend_fpr(x: float | None) -> str:
    return "unmeasured (no scored units)" if x is None else f"{x:.4f}"


def secondaries(scored: dict, profile: Profile,
                registered_seeds: list[int] | None = None) -> dict:
    """§ 4 Secondaries — REPORTED, no pass/fail attached to any of them."""
    out: dict = {"_note": "REPORTED, NON-ADJUDICATING (v1.15 § 4 Secondaries)"}

    for scen in (P1_SLICE, "S3"):
        det = _slice_by_seed(scored["blend"], scen, "recall")
        det_fpr = _slice_by_seed(scored["blend"], scen, "fpr")
        if not det:
            continue
        entry = {
            "detector_blended_mean": mean(det.values()),
            "detector_per_seed": {str(s): det[s] for s in sorted(det)},
            "detector_mean_realized_fpr": mean(det_fpr.values()),
            "detector_ci95_student_t": student_t_ci(
                [det[s] for s in sorted(det)], profile.t_crit, profile.df),
            "baselines": {},
        }
        orc_by_seed: dict[int, float] = {}
        argmax_by_seed: dict[int, str] = {}
        for b, _t, label in BASELINES:
            bs = _slice_by_seed(scored["base_blend"][b], scen, "recall")
            bf = _slice_by_seed(scored["base_blend"][b], scen, "fpr")
            if not bs:
                continue
            common = sorted(set(det) & set(bs))
            fixed_diffs = [det[s] - bs[s] for s in common]
            # § 3.2 at the READOUT GRAIN, the mirror of the § 4 secondary-7
            # guard. This block's readout is a MEAN OVER PER-SEED BLENDED
            # values, so its realized FPR is the mean over the per-seed blended
            # FPRs — the construction v1.15b § 1.2 froze for P1 on this same
            # surface ("comparability_basis" in `adjudicate_p1`), NOT the
            # row-pooled rate secondary 7 uses. The detector's blended FPR is a
            # family-mix-weighted mixture, not a row rate; row-pooling it would
            # redefine the quantity P1 adjudicates on.
            b_mean_fpr = mean(bf.values()) if bf else None
            node = {
                "label": label,
                "n_paired_seeds": len(common),
                "paired_seeds": [str(s) for s in common],
                # Calibration evidence — rides BOTH branches; on a halt these
                # two ARE the evidence for it (§ 3.2 "with the realized FPR
                # printed").
                "mean_realized_fpr": b_mean_fpr,
                "detector_mean_realized_fpr": entry["detector_mean_realized_fpr"],
                "detector_in_interval": comparable(
                    entry["detector_mean_realized_fpr"]),
                "baseline_in_interval": comparable(b_mean_fpr),
            }
            if node["detector_in_interval"] and node["baseline_in_interval"]:
                node["status"] = "COMPUTED"
                node["mean"] = mean(bs.values())
                node["per_seed"] = {str(s): bs[s] for s in sorted(bs)}
                node["fixed_contrast_margin"] = (
                    mean(det.values()) - mean(bs.values()))
                node["fixed_contrast_sign_test"] = exact_sign_p(fixed_diffs)
            else:
                outside = []
                if not node["detector_in_interval"]:
                    outside.append(
                        f"detector {_fmt_blend_fpr(entry['detector_mean_realized_fpr'])}")
                if not node["baseline_in_interval"]:
                    outside.append(f"{b} {_fmt_blend_fpr(b_mean_fpr)}")
                node["status"] = "NOT COMPARABLE"
                node.update(SEC.exclusion_fields(
                    "fpr_interval", " and ".join(outside)))
            entry["baselines"][b] = node
            for s in common:
                if s not in orc_by_seed or bs[s] > orc_by_seed[s]:
                    orc_by_seed[s] = bs[s]
                    argmax_by_seed[s] = b
        if orc_by_seed:
            common = sorted(set(det) & set(orc_by_seed))
            diffs = [det[s] - orc_by_seed[s] for s in common]
            entry["oracle_max"] = {
                "mean": mean(orc_by_seed[s] for s in common),
                "per_seed": {str(s): orc_by_seed[s] for s in common},
                "argmax_baseline_per_seed": {str(s): argmax_by_seed[s] for s in common},
                "margin_detector_minus_oracle": mean(det[s] for s in common)
                - mean(orc_by_seed[s] for s in common),
                "per_seed_diff": {str(s): det[s] - orc_by_seed[s] for s in common},
                "sign_test": exact_sign_p(diffs),
                # DISCLOSURE, not a change of estimand. The oracle maximum is
                # composed from the per-seed baseline recalls BEFORE the § 3.2
                # guard is applied to the fixed contrasts, so an instrument
                # whose own contrast is suppressed can still compose the oracle.
                # Changing that would change what the oracle IS, which this
                # amendment's scope discipline forbids — so it is named instead.
                "contributing_instruments_not_comparable": sorted(
                    b for b, n in entry["baselines"].items()
                    if n["status"] == "NOT COMPARABLE"
                    and b in argmax_by_seed.values()),
            }
        out[f"{scen}_blend_margin"] = entry

    fam: dict[str, dict] = {}
    for (A, scen, sd), v in sorted(scored["per_family"].items()):
        fam.setdefault(A, {}).setdefault(scen, {})[str(sd)] = {
            "recall": v["recall"], "realized_fpr": v["fpr"], "n_mal": v["n_mal"]}
    for A, per_scen in fam.items():
        for scen, per_seed in per_scen.items():
            vals = [x["recall"] for x in per_seed.values()]
            per_scen[scen] = {
                "per_seed": per_seed,
                "mean_recall": mean(vals),
                "mean_realized_fpr": mean(x["realized_fpr"] for x in per_seed.values()),
                "ci95_student_t": student_t_ci(vals, profile.t_crit, profile.df),
            }
    out["per_family_fold_recalls"] = fam
    out["declared_vacuous_cells"] = ["label_flip × S3 — structurally zero rows (§ 4.1 table)"]

    # The remaining mandatory secondaries, all from THIS pass (§ 5.1: the sealed
    # corpus is opened exactly once, so there is no second read to defer to).
    out["auc_per_scenario"] = SEC.per_scenario_auc(
        scored, SCENARIOS, profile.t_crit, profile.df, student_t_ci)
    out["alie_fixed_baseline_contrasts"] = SEC.alie_fixed_baseline_contrasts(
        scored, BASELINES, SCENARIOS, exact_sign_p, comparable)
    # § 6 (ratification) — the reported FPR bracket. Same pass, same rows.
    out["bracket_recall_by_fpr"] = SEC.bracket_recall_by_fpr(
        scored, SCENARIOS, BRACKET_TARGETS, TARGET_FPR, COMPARABILITY_INTERVAL,
        student_t_ci, profile.t_crit, profile.df)
    out["global_cut_sensitivity"] = SEC.global_cut_sensitivity(
        scored, SCENARIOS, P1_SLICE, comparable)
    out["g2_scored_rows_contrast"] = SEC.g2_scored_rows_contrast(
        scored, SCENARIOS, registered_seeds, ATTACKS, exact_sign_p,
        student_t_ci, ci_on_retained, comparable)

    exp048 = E48.resolve_exp048_dir()
    if exp048:
        # § 4 secondary 9 — the REGISTERED mechanics on the exposed arm: the
        # § 2.2a rotation, LOAO fits, nested-independent per-scenario
        # calibration, and a row-matched TGE comparison. Descriptive only.
        rows_048 = E48.load_standalone_tge_rows(
            exp048, R.SCEN_SHORT, R.derive_window_feats)
        # The rotation runs over the REGISTERED h2_confirm universe, not over
        # whatever seeds happen to be staged.
        seeds_048 = h2_confirm_seeds()
        plan_048 = rotation_plan(seeds_048)
        scored_048 = score_corpus(rows_048, plan_048)
        out["exp048_standalone_tge_full_coverage"] = E48.exp048_full_coverage_contrast(
            rows_048, plan_048, scored_048, SCENARIOS, ATTACKS, exact_sign_p,
            comparable, student_t_ci, ci_on_retained, seeds_048)
    else:
        out["exp048_standalone_tge_full_coverage"] = {
            "status": "NOT RUN",
            "reason": ("H2PRIME_EXP048_SIG_DIR is unset or absent, so the § 4 "
                       "secondary 9 descriptive contrast was not computed. This "
                       "is a scope disclosure, not a result — EXP-048 is EXPOSED "
                       "data (unsealed 2026-08-09) and carries no seal obligation."),
        }
    out["not_computed_by_this_executor"] = []
    return out



# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _versions() -> dict:
    import sklearn
    return {"python": ".".join(str(v) for v in sys.version_info[:3]),
            "numpy": np.__version__, "sklearn": sklearn.__version__}


def exp048_input_gate(profile: Profile) -> str | None:
    """§ 4 secondary 9's input must be staged BEFORE the single sealed pass.

    The same-pass rule (EXP-051 § 5.1) means every mandatory output has to come
    out of the one execution, so a missing input is a read-prep defect to fix
    beforehand — not a disclosure to write afterwards. On a real invocation an
    unset `H2PRIME_EXP048_SIG_DIR` is therefore a HARD STOP, deliberately
    firing in `--dry-run` too, which is where read prep is meant to catch it.

    EXP-048 was unsealed 2026-08-09: staging it carries no seal implication.
    """
    d = E48.resolve_exp048_dir()
    if not profile.adjudicating:
        return d
    if not d:
        raise HardStop(
            "EXP-048 INPUT GATE: H2PRIME_EXP048_SIG_DIR is unset or does not "
            "exist, so the § 4 secondary 9 full-coverage TGE contrast could not "
            "be produced by the single sealed pass. The sealed corpus is opened "
            "EXACTLY ONCE (EXP-051 § 5.1), so every mandatory reported output "
            "must come out of that pass — a missing input is a read-prep defect "
            "to fix BEFORE the read, not a disclosure to write after it. Stage "
            "the EXP-048 standalone-TGE logs (EXPOSED data, unsealed "
            "2026-08-09) and point the variable at them, then re-run the dry-run."
        )
    # Existence is not staging. Validate CONTENT here, because a gate that
    # accepts an empty directory only fails AFTER sealed rows have been opened,
    # and the one-shot read cannot be retried.
    try:
        E48.validate_exp048_dir(d, R.SCEN_SHORT, SCENARIOS, h2_confirm_seeds(),
                                EXP048_REQUIRED_KEYS, SCORING_VALUE_CONTRACT,
                                SCORING_ENUM_CONTRACT)
    except (ValueError, OSError) as exc:
        raise HardStop(
            f"EXP-048 INPUT GATE: the staged directory {d} does not contain a "
            f"usable standalone-TGE arm — {exc}. The § 4 secondary 9 contrast "
            "runs the registered mechanics (rotation, LOAO fits, per-scenario "
            "calibration, row-matched TGE comparison), so it needs a complete "
            "scenario × seed grid of parseable `__tge__` unit files. Fix the "
            "staging and re-run the dry-run BEFORE the sealed read."
        ) from exc
    return d


def build_report(map_path: Path, profile: Profile, dry_run: bool) -> tuple[dict, str]:
    gate = golden_gate()
    exp048_input_gate(profile)
    cells, map_meta = load_assembly_map(map_path, profile)
    plan = rotation_plan(map_meta["seeds_ascending"])

    meta = {
        "executor": "scripts/adjudicate_h2prime.py",
        "spec": ("docs/superpowers/specs/"
                 "2026-08-09-praxis-experimental-design-v1.15-h2prime.md"),
        "experiment": "EXP-051 (48 cells) + EXP-053 refill (2 cells)",
        "profile": {"name": profile.name, "n_seeds": profile.n_seeds,
                    "n_cells": profile.n_cells, "df": profile.df,
                    "t_crit": profile.t_crit,
                    "p2_min_positive": profile.p2_min_positive,
                    "adjudicating": profile.adjudicating},
        "golden_gate": gate,
        "versions": _versions(),
        "pinned_versions": {"sklearn": PINNED_SKLEARN, "numpy": PINNED_NUMPY},
        "gbdt_params": GBDT_PARAMS,
        "features_frozen_order": FEATS,
        "baselines": [{"field": b, "higher_is_trust": t, "label": l}
                      for b, t, l in BASELINES],
        "n_cells": len(cells),
        "n_seeds": len(map_meta["seeds_ascending"]),
        **map_meta,
    }
    plan_rows = [{"i": r.i, "test": r.test, "calibration": r.calibration,
                  "fit": list(r.fit)} for r in plan]

    if dry_run:
        report = {"_meta": meta, "rotation_plan": plan_rows,
                  "dry_run": True,
                  "cells": [{"scenario": c.scenario, "scen": c.scen, "seed": c.seed,
                             "source": c.source, "path": c.path} for c in
                            sorted(cells, key=lambda c: (c.scen, c.seed))]}
        return report, render_dry_run({**report, "_meta": meta})

    rows = load_cells(cells)
    meta["n_rows"] = len(rows)
    meta["row_census"] = {
        "honest": sum(1 for r in rows if not r["malicious_gt"]),
        "malicious": sum(1 for r in rows if r["malicious_gt"]),
        "malicious_blank_family_excluded_from_blend": sum(
            1 for r in rows if r["malicious_gt"] and not r.get("attack_type")),
    }
    scored = score_corpus(rows, plan)
    meta["baseline_null_drop_summary"] = {
        b: {
            "cells": len(scored["base_blend"][b]),
            "mal_rows_dropped_null": sum(v.get("mal_null_dropped", 0)
                                         for v in scored["base_blend"][b].values()),
            "honest_rows_dropped_null": sum(v.get("honest_null_dropped", 0)
                                            for v in scored["base_blend"][b].values()),
            "calibration_rows_dropped_null": sum(v.get("cal_null_dropped", 0)
                                                 for v in scored["base_blend"][b].values()),
        }
        for b, _t, _l in BASELINES
    }
    p1 = adjudicate_p1(scored, profile, map_meta["seeds_ascending"])
    p2 = adjudicate_p2(scored, profile, map_meta["seeds_ascending"])

    # v1.15b § 3 mandatory sensitivity, computed in the SAME invocation so the
    # sealed corpus is opened exactly once (EXP-051 § 5.1).
    devices = exposed_devices(rows)
    scored_strict = score_corpus(rows, plan, exclude_devices=devices)

    sec = secondaries(scored, profile, map_meta["seeds_ascending"])
    sec["strict_identity_loao_sensitivity"] = strict_identity_block(
        rows, scored, scored_strict, devices, profile,
        map_meta["seeds_ascending"])
    report = {
        "_meta": meta,
        "rotation_plan": plan_rows,
        "fit_census": scored["fit_census"],
        "fit_census_strict_identity": scored_strict["fit_census"],
        "P1": p1,
        "P2": p2,
        "secondaries": sec,
        "verdict": overall_verdict(p1, p2),
    }
    # The whole document is validated against the frozen output contract before
    # it is returned — on EVERY run including dev-smoke. A malformed result is
    # not a result, so this is a hard stop rather than a warning.
    try:
        meta["output_schema_gate"] = SCHEMA.check(report)
    except SCHEMA.SchemaViolation as exc:
        raise HardStop(str(exc)) from exc
    return report, render_verdict_block(report)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One-shot P1 ∧ P2 adjudication of the H2′ confirmatory corpus.")
    ap.add_argument("assembly_map", type=Path,
                    help="cell → local-path map produced by build_exp051_assembly_map.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="results JSON path (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate map + rotation + golden gate; score nothing")
    ap.add_argument("--dev-smoke", action="store_true",
                    help="executor self-test on the disclosed EXP-011 dev corpus; "
                         "refuses any sealed seed and adjudicates nothing")
    args = ap.parse_args(argv)

    profile = DEV_SMOKE if args.dev_smoke else CONFIRMATORY
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")

    try:
        report, text = build_report(args.assembly_map, profile, args.dry_run)
    except (Refusal, HardStop) as exc:
        kind = "REFUSED" if isinstance(exc, Refusal) else "HARD STOP"
        print(f"\n*** {kind} — nothing was scored ***\n{exc}\n", file=sys.stderr)
        return 2

    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**report, "verdict_block": text}
        args.out.write_text(
            json.dumps(payload, sort_keys=True, indent=1, allow_nan=False) + "\n",
            encoding="utf-8")
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

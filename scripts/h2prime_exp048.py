"""§ 4 secondary 9 — the EXP-048 standalone-TGE arm (EXPOSED data, descriptive).

Staging validation, loading, and the full-coverage matched TGE-vs-H2′ contrast
for the arm EXP-048 published on 2026-08-09. Separated from
`h2prime_secondaries.py` (2026-08-12) because it is a distinct concern — a
different corpus, its own registered seed universe, and a pre-read gate the
confirmatory side has no equivalent of — and because that module had grown past
the project's 800-line ceiling.

DESCRIPTIVE ONLY. Nothing here adjudicates.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from statistics import mean

from h2prime_common import (EXP048_REQUIRED_KEYS, HardStop, Profile, R,
                            SCENARIOS, SCORING_ENUM_CONTRACT,
                            SCORING_VALUE_CONTRACT)
from h2prime_corpus import h2_confirm_seeds, rotation_plan
from h2prime_secondaries import (EXCLUSION_CAUSES, G2_MIN_SCORED_MALICIOUS,
                                 _census, _frac)
# The staging gate lives in its own module (round 30, 800-line ceiling) and is
# RE-EXPORTED here so `E48.validate_exp048_dir` remains the one public name.
from h2prime_exp048_gate import validate_exp048_dir  # noqa: F401


def _fmt_fpr(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"

def load_standalone_tge_rows(sig_dir: str | os.PathLike, scen_short: dict,
                             derive_window_feats) -> list[dict]:
    """Load the EXP-048 standalone-TGE arm's signal logs.

    EXP-048 was unsealed on 2026-08-09, so this is EXPOSED data — it carries no
    seal obligation and its contrast is DESCRIPTIVE ONLY (§ 4 secondary 9).
    The loader mirrors the frozen per-file discipline: one
    `derive_window_feats()` invocation per (scenario, seed, defense) file, with
    the same filename↔row identity assertions, files consumed in ascending
    unit-id order.

    Only `tge`-configuration files are taken: on a standalone-TGE arm every row
    reaches the TGE stage, which is what makes this the FULL-COVERAGE half of
    G2-EXTENDED.
    """
    rows: list[dict] = []
    seen_units: dict[tuple[str, int], str] = {}
    for fn in sorted(glob.glob(os.path.join(str(sig_dir), "*.jsonl"))):
        stem = os.path.basename(fn)[: -len(".jsonl")]
        parts = stem.split("__")
        if len(parts) != 4:
            raise ValueError(f"unit-id stem is not scenario__defense__exec__seed: {stem}")
        scen, defense, _exec, seedtok = parts
        if defense != "tge":
            continue                      # standalone-TGE arm only
        seed = int(seedtok.replace("seed", ""))
        unit_key = (scen_short[scen], seed)
        if unit_key in seen_units:
            raise ValueError(
                f"duplicate staged unit for {unit_key[0]}x{unit_key[1]} — "
                f"{os.path.basename(seen_units[unit_key])} and "
                f"{os.path.basename(fn)}; refusing to load both")
        seen_units[unit_key] = fn
        with open(fn, "r", encoding="utf-8") as fh:
            frows = [json.loads(line) for line in fh if line.strip()]
        for r in frows:
            if r["seed"] != seed:
                raise ValueError(f"row seed {r['seed']} != filename seed {seed}: {fn}")
            if scen_short[r["scenario"].lower()] != scen_short[scen]:
                raise ValueError(f"row scenario != filename scenario: {fn}")
            r["_scen"] = scen_short[scen]
            r["_seed"] = seed
            r["_defense"] = defense
        derive_window_feats(frows)
        rows.extend(frows)
    return rows


def standalone_tge_coverage(rows: list[dict], scenarios: list[str]) -> dict:
    """Full-coverage census of the EXP-048 standalone-TGE arm.

    Reports what fraction of rows carry a non-null `tge_score` per scenario —
    the property that makes this arm the full-coverage complement to the
    Krum+TGE scored-rows contrast. DESCRIPTIVE ONLY; adjudicates nothing.
    """
    out = {"_note": ("REPORTED, DESCRIPTIVE ONLY on EXPOSED data "
                     "(EXP-048 unsealed 2026-08-09; § 4 secondary 9)"),
           "n_rows": len(rows), "scenarios": {}}
    for scen in scenarios:
        sc = [r for r in rows if r["_scen"] == scen]
        if not sc:
            continue
        nonnull = [r for r in sc if r.get("tge_score") is not None]
        mal = [r for r in nonnull if r["malicious_gt"]]
        out["scenarios"][scen] = {
            "n_rows": len(sc),
            "tge_score_coverage": len(nonnull) / len(sc),
            "n_scored_malicious": len(mal),
            "seeds": sorted({r["_seed"] for r in sc}),
        }
    return out

def resolve_exp048_dir() -> str | None:
    """Where the EXP-048 standalone-TGE logs live, if they are staged locally."""
    d = os.environ.get("H2PRIME_EXP048_SIG_DIR")
    if d and Path(d).is_dir():
        return d
    return None


def exp048_full_coverage_contrast(rows: list[dict], plan, scored: dict,
                                  scenarios: list[str], attacks: list[str],
                                  sign_fn, comparable_fn, ci_fn, ci_retained_fn,
                                  expected_seeds: list[int] | None = None) -> dict:
    """§ 4 secondary 9 — the FULL-COVERAGE matched TGE-vs-H2\u2032 contrast.

    This runs the REGISTERED mechanics, not a census: the § 2.2a rotation over
    the EXP-048 seeds, LOAO detector fits, nested-independent per-scenario
    calibration, and a ROW-MATCHED TGE comparison — read off the scorer\u2019s own
    `g2` cells, which rescore BOTH sides on exactly the `tge_score`-covered
    rows. That matters even here: pairing full-population detector cells against
    a TGE dictionary built on covered rows only would compare two different
    populations the moment any `tge_score` is null. The intersection\u2019s coverage
    is reported beside every readout.

    It is FULL COVERAGE because the standalone-TGE arm never Krum-filters, so
    coverage is 1.0 in the ordinary case — the complement to the Krum+TGE
    scored-rows contrast, which is a sub-population by construction. The
    machinery does not assume that, it measures it.

    DESCRIPTIVE ONLY, on EXPOSED data (EXP-048 unsealed 2026-08-09). It
    adjudicates nothing; every readout carries its realized FPR, the § 3.2
    comparability annotation, and the frozen Student-t CI on the paired
    differences.
    """
    cells = scored.get("g2") or {}

    def paired(get, scen, cov_key=None):
        # ONE floor, ONE discipline, both sides: the same 30-row rule that
        # governs the confirmatory G2 side is applied here BEFORE anything
        # enters the paired vectors. An under-floor fold is disqualified, not
        # averaged in, and the census records it either way.
        d, t, cov, census, causes = {}, {}, {}, {}, {}
        for (sc, sd), v in sorted(cells.items()):
            if sc != scen:
                continue
            probe = get(v)
            parent_census_only = v.get("status") == "CENSUS ONLY"
            unscheduled = bool(probe is not None and probe.get("unscheduled"))
            if parent_census_only or unscheduled:
                # BRANCH ORDER IS THE CLASSIFICATION. A family absent from a
                # CENSUS-ONLY fold was excluded by the parent's missing
                # operating point — the coverage floor never got to run on it.
                # Testing `unscheduled` first recorded `coverage_floor` and
                # blamed a guard that never fired. The parent's status is
                # decided FIRST, exactly as the confirmatory side decides it
                # (`parent_census_only` in h2prime_secondaries).
                causes[sd] = "census_only" if parent_census_only else "coverage_floor"
                # Ask the reduction's own accessor for the census so a
                # per-family reduction reports the FAMILY's covered/eligible
                # counts, not the pooled cell's (round-11 #2, remaining branch).
                # `is not None`, not truthiness: an all-zero family census is a
                # FACT about the family and must not fall back to the blend.
                src = (probe["census"] if probe is not None
                       and probe.get("census") is not None else v)
                census[sd] = {
                    **_census(src),
                    "status": "NOT COMPARABLE",
                    # Both facts survive when both are true. Naming only the
                    # parent would erase that the family was never scheduled.
                    "note": (("census only — uncovered honest side, no operating "
                              "point and therefore no contrast"
                              + ("; the family is also not scheduled in this fold"
                                 if unscheduled else ""))
                             if parent_census_only else
                             "family not scheduled in this fold")}
                continue
            got = probe
            if got is None:
                causes[sd] = "coverage_floor"
                census[sd] = {**_census({}), "status": "NOT COMPARABLE",
                              "note": "no covered rows for this slice"}
                continue
            n_scored = got["n_mal_scored"]
            if n_scored < G2_MIN_SCORED_MALICIOUS:
                # Suppression applies to contrast STATISTICS only. Coverage and
                # census facts are what a disqualified fold is FOR — they are
                # the evidence that it was disqualified and why.
                causes[sd] = "coverage_floor"
                census[sd] = {**_census(got.get("census") or {}),
                              "status": "NOT COMPARABLE",
                              "note": "no statistic computed "
                                      "(pre-registered coverage rule)"}
                continue
            # § 3.2 COMPARABILITY, applied here exactly as everywhere else: a
            # fold that clears the row floor but realizes an out-of-interval FPR
            # is not comparable. Annotating it false while still letting it into
            # the contrast is the one thing the interval exists to prevent.
            d_fpr = got.get("det_fpr")
            t_fpr = got.get("tge_fpr")
            outside = []
            if not comparable_fn(d_fpr):
                outside.append(f"detector {_fmt_fpr(d_fpr)}")
            if not comparable_fn(t_fpr):
                outside.append(f"TGE {_fmt_fpr(t_fpr)}")
            if outside:
                causes[sd] = "fpr_interval"
                census[sd] = {
                    **_census(got.get("census") or {}),
                    "status": "NOT COMPARABLE",
                    # The realized FPRs are the EVIDENCE for the exclusion and
                    # are retained; only contrast statistics are suppressed.
                    "detector_realized_fpr": d_fpr, "tge_realized_fpr": t_fpr,
                    "detector_comparable": comparable_fn(d_fpr),
                    "tge_comparable": comparable_fn(t_fpr),
                    "note": ("realized FPR outside the comparability interval: "
                             + " and ".join(outside)
                             + "; no statistic computed")}
                continue
            cell = cells[(scen, sd)]
            census[sd] = {
                **_census(got.get("census") or {}), "status": "COMPUTED",
                # A retained fold serializes BOTH per-seed recalls, not just
                # their difference, plus its realized FPRs and comparability —
                # the difference alone cannot be audited back to its sides.
                "detector_recall": got["det"], "tge_recall": got["tge"],
                # The annotation source MUST match the reduction's population:
                # a per-family reduction supplies its own FPRs, and only a
                # blended reduction falls back to the cell's blended pair.
                "detector_realized_fpr": got.get("det_fpr", cell.get("det_fpr")),
                "tge_realized_fpr": got.get("tge_fpr", cell.get("tge_fpr")),
                "detector_comparable": comparable_fn(
                    got.get("det_fpr", cell.get("det_fpr"))),
                "tge_comparable": comparable_fn(
                    got.get("tge_fpr", cell.get("tge_fpr"))),
            }
            d[sd], t[sd] = got["det"], got["tge"]
            cov[sd] = got["coverage"]

        missing = ([s for s in sorted(expected_seeds) if s not in census]
                   if expected_seeds is not None else None)
        # Zero-coverage folds stay in the census rather than vanishing from the
        # denominator (registered-universe discipline).
        if expected_seeds is not None:
            for s in missing or []:
                causes[s] = "coverage_floor"
                census[s] = {**_census({}), "status": "NOT COMPARABLE",
                             "note": "fold produced no cell"}
        base = {
            "n_folds": len(census),
            "n_folds_comparable": len(d),
            "n_seeds_expected": (len(expected_seeds)
                                 if expected_seeds is not None else None),
            "seeds_missing_from_pairing": missing,
            "disqualified_folds": {
                f"seed {sd}": c["n_mal_scored"]
                for sd, c in sorted(census.items()) if c["status"] != "COMPUTED"},
            "census_per_fold": {f"seed {sd}": c for sd, c in sorted(census.items())},
            "min_scored_malicious_for_comparability": G2_MIN_SCORED_MALICIOUS,
            "row_matched": True,
            # AGGREGATE COVERAGE over EVERY fold, disqualified included:
            # summing only retained folds reports the coverage of the folds
            # that SURVIVED, systematically higher than the slice's. Split
            # honest/malicious per DECISION G item 3.
            "coverage_all_folds_population": (
                "every fold in the census, including disqualified — census "
                "facts are not suppressed with statistics"),
            "n_scored_malicious_all_folds": sum(
                c["n_mal_scored"] or 0 for c in census.values()),
            "n_malicious_total_all_folds": sum(
                c["n_mal_total"] or 0 for c in census.values()),
            "coverage_malicious_all_folds": _frac(
                sum(c["n_mal_scored"] or 0 for c in census.values()),
                sum(c["n_mal_total"] or 0 for c in census.values())),
            "n_scored_honest_all_folds": sum(
                c["n_honest_scored"] or 0 for c in census.values()),
            "n_honest_total_all_folds": sum(
                c["n_honest_total"] or 0 for c in census.values()),
            "coverage_honest_all_folds": _frac(
                sum(c["n_honest_scored"] or 0 for c in census.values()),
                sum(c["n_honest_total"] or 0 for c in census.values())),
            # causes + comparable must account for EVERY fold (round-18's
            # accounting invariant), mirrored onto this side.
            # Round 29: this was a RESTATED literal, so the 4th cause landed on
            # the confirmatory side only. Enumerate from the vocabulary itself.
            "exclusion_causes": {
                c: sum(1 for x in causes.values() if x == c)
                for c in EXCLUSION_CAUSES},
        }
        if not d:
            # Halt reasons name their ACTUAL cause. Folds excluded by the
            # § 3.2 interval did clear the row floor; saying otherwise would
            # misattribute a calibration failure to a coverage failure.
            by_floor = sum(1 for c in census.values()
                           if "coverage rule" in c.get("note", ""))
            by_fpr = sum(1 for c in census.values()
                         if "comparability interval" in c.get("note", ""))
            if by_fpr and not by_floor:
                cause = ("all folds excluded: out-of-interval realized FPR "
                         "(§ 3.2 comparability)")
            elif by_floor and not by_fpr:
                cause = (f"no fold cleared the {G2_MIN_SCORED_MALICIOUS}-row "
                         "coverage floor")
            elif by_fpr and by_floor:
                cause = (f"all folds excluded: {by_floor} below the "
                         f"{G2_MIN_SCORED_MALICIOUS}-row coverage floor, "
                         f"{by_fpr} on out-of-interval realized FPR")
            else:
                cause = "no fold produced a scorable contrast"
            base["status"] = "NOT COMPARABLE"
            base["reason"] = f"{cause}; no statistic computed for this slice"
            # NOTE: exclusion_causes is already set on `base` from the
            # per-fold cause map above, in the SAME three-cause taxonomy both
            # reducers share. Re-deriving it here from note text would give a
            # second, differently-keyed answer to the same question.
            return base

        seeds = sorted(d)
        diffs = [d[s] - t[s] for s in seeds]
        base.update({
            "status": "COMPUTED",
            "n_paired_seeds": len(seeds),
            "detector_mean": mean(d[s] for s in seeds),
            "tge_mean": mean(t[s] for s in seeds),
            "margin_detector_minus_tge": mean(diffs),
            "per_seed_diff": {str(s): d[s] - t[s] for s in seeds},
            "sign_test": sign_fn(diffs),
            # df comes from the RETAINED folds, never the full profile.
            "ci95_student_t_on_paired_diff": ci_retained_fn(diffs, ci_fn),
            # SCOPE STATED, not widened: these annotate the PAIRED statistic,
            # whose population IS the retained folds, so averaging excluded
            # folds in would misdescribe it. `base` carries the slice-wide
            # census figures for the complete-coverage question.
            "intersection_coverage_population": (
                "comparable folds only — this annotates the paired statistic, "
                "whose population IS the retained folds; see "
                "coverage_malicious_all_folds for the whole slice"),
            "intersection_coverage_per_seed": {str(s): cov[s] for s in seeds},
            "intersection_coverage_mean": mean(cov[s] for s in seeds),
        })
        if cov_key == "blend":
            fd = [cells[(scen, s)]["det_fpr"] for s in seeds]
            ft = [cells[(scen, s)]["tge_fpr"] for s in seeds]
            base["detector_realized_fpr"] = mean(fd)
            base["tge_realized_fpr"] = mean(ft)
            base["detector_comparable"] = comparable_fn(mean(fd))
            base["tge_comparable"] = comparable_fn(mean(ft))
        return base

    out = {
        "_note": ("REPORTED, DESCRIPTIVE ONLY on EXPOSED data (EXP-048 unsealed "
                  "2026-08-09; § 4 secondary 9). Registered mechanics: § 2.2a "
                  "rotation, LOAO fits, nested-independent per-scenario "
                  "calibration, ROW-MATCHED TGE comparison on the covered "
                  "intersection. Adjudicates nothing."),
        "ci_method": ("two-sided 95 % Student-t on the paired per-seed "
                      "differences, df = n_retained - 1 with the matching "
                      "critical t (reported-only; no CI below 2 retained folds)"),
        "coverage": standalone_tge_coverage(rows, scenarios),
        "rotation": [{"i": r.i, "test": r.test, "calibration": r.calibration,
                      "fit": list(r.fit)} for r in plan],
        "blended": {}, "per_family": {},
    }
    for scen in scenarios:
        b = paired(lambda v: {"det": v.get("det_recall"), "tge": v.get("tge_recall"),
                              "n_mal_scored": v["n_mal_scored"],
                              "census": v,
                              "det_fpr": v.get("det_fpr"),
                              "tge_fpr": v.get("tge_fpr"),
                              "coverage": _frac(v["n_mal_scored"],
                                                v.get("n_mal_total") or 0)},
                   scen, cov_key="blend")
        if b:
            out["blended"][scen] = b
    for A in attacks:
        for scen in scenarios:
            def get(v, _A=A):
                pf = (v.get("per_family") or {}).get(_A)
                if pf is None:
                    # Zero-covered family: the scorer preserves the eligible
                    # total, so the fold still yields a 0/N census row instead
                    # of vanishing (the confirmatory side's fix, mirrored).
                    eligible = (v.get("family_eligible_totals") or {}).get(_A)
                    covered = (v.get("family_covered_totals") or {}).get(_A, 0)
                    if not eligible:
                        # UNSCHEDULED in this fold. Round-7's distinction says
                        # 0/0 "not scheduled" and 0/N "scheduled but unscored"
                        # are different facts; returning None here would drop
                        # the row entirely and erase both.
                        return {"det": None, "tge": None, "n_mal_scored": 0,
                                "coverage": None, "unscheduled": True,
                                "census": {
                                    "n_mal_scored": 0, "n_mal_total": 0,
                                    "n_honest_scored": v.get("n_honest_scored"),
                                    "n_honest_total": v.get("n_honest_total")}}
                    return {"det": None, "tge": None, "n_mal_scored": covered,
                            "coverage": _frac(covered, eligible),
                            "census": {"n_mal_scored": covered,
                                       "n_mal_total": eligible,
                                       "n_honest_scored": v.get("n_honest_scored"),
                                       "n_honest_total": v.get("n_honest_total")}}
                return {"det": pf["det_recall"], "tge": pf["tge_recall"],
                        "n_mal_scored": pf["n_mal_scored"],
                        # finding 4: a per-family reduction is annotated with
                        # the FAMILY's operating point, never the blend's.
                        "det_fpr": pf.get("det_fpr"), "tge_fpr": pf.get("tge_fpr"),
                        "census": {"n_mal_scored": pf["n_mal_scored"],
                                   "n_mal_total": pf.get("n_mal_total"),
                                   "n_honest_scored": pf.get("n_honest_scored"),
                                   "n_honest_total": pf.get("n_honest_total")},
                        "coverage": _frac(pf["n_mal_scored"],
                                          pf.get("n_mal_total") or 0)}
            pfe = paired(get, scen)
            if pfe:
                out["per_family"].setdefault(A, {})[scen] = pfe
    return out


def exp048_input_gate(profile: Profile) -> str | None:
    """§ 4 secondary 9's input must be staged BEFORE the single sealed pass.

    The same-pass rule (EXP-051 § 5.1) means every mandatory output has to come
    out of the one execution, so a missing input is a read-prep defect to fix
    beforehand — not a disclosure to write afterwards. On a real invocation an
    unset `H2PRIME_EXP048_SIG_DIR` is therefore a HARD STOP, deliberately
    firing in `--dry-run` too, which is where read prep is meant to catch it.

    EXP-048 was unsealed 2026-08-09: staging it carries no seal implication.
    """
    d = resolve_exp048_dir()
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
        validate_exp048_dir(d, R.SCEN_SHORT, SCENARIOS, h2_confirm_seeds(),
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

"""§ 4 secondary 9 — the EXP-048 STAGING GATE (content validation of the arm).

Split out of `h2prime_exp048.py` (2026-08-12, round 30) because that module had
reached the project's 800-line ceiling and the pre-read gate is a distinct
concern from the contrast that consumes it: the gate runs BEFORE the sealed
corpus is opened and its whole job is to refuse a corpus, while everything left
behind reduces one that was already accepted.

Pure move — no behaviour changed in the split itself; the digest is invariant
across it and that is proved separately from the fix that follows.

`h2prime_exp048` re-exports `validate_exp048_dir`, so every existing caller
(`E48.validate_exp048_dir`) keeps working unchanged.
"""
from __future__ import annotations

import glob
import json
import os

from h2prime_corpus import rotation_plan


def _parse_unit_rows(fn: str, scen: str, seed: int, scen_short: dict,
                     required_keys, value_contract=None,
                     enum_contract=None) -> dict:
    """Parse EVERY row of a staged unit; return its CLASS CENSUS or raise.

    First-line-only validation lets a malformed row 900 lines in burn the sealed
    pass, which cannot be retried. EXP-048 is exposed data of modest size, so a
    full parse before the read costs nothing worth saving.

    The census (rows / honest / malicious / family-labelled malicious) is
    counted here because the parse already visits every row: the caller decides
    whether the POPULATIONS are scorable, which the syntactic grid cannot say.
    """
    # The registered family roster, taken from the enum contract the caller
    # already supplies — never from the staged rows. The empty string is the
    # frozen loader's discovery-row marker, not a family.
    registered = tuple(a for a in (enum_contract or {}).get("attack_type", ()) if a)
    n = honest = malicious = fam_labelled = 0
    by_family = {a: 0 for a in registered}
    with open(fn, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{os.path.basename(fn)} line {lineno} is not valid JSON: {exc}"
                ) from exc
            missing = [k for k in required_keys if k not in row]
            if missing:
                raise ValueError(
                    f"{os.path.basename(fn)} line {lineno} lacks "
                    + ", ".join(f"'{k}'" for k in missing))
            # VALUES, not just keys. A null or non-finite raw feature becomes
            # NaN in the design matrix and the § 2.1a hard stop fires INSIDE the
            # sealed pass, which cannot be retried. It must fail at dry-run.
            # `tge_score` is deliberately NOT in this set: its null is a
            # measured coverage fact, handled by the coverage machinery.
            for k, kind in (value_contract or {}).items():
                if k not in row:
                    continue                     # presence handled above
                v = row[k]
                where = f"{os.path.basename(fn)} line {lineno}"
                if v is None:
                    if kind.endswith("_or_null"):
                        continue
                    raise ValueError(
                        f"{where} has a NULL '{k}' — the scoring path consumes "
                        f"it as {kind}")
                if kind.startswith("numeric"):
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise ValueError(f"{where} has a non-numeric '{k}' "
                                         f"({type(v).__name__})")
                    if v != v or v in (float("inf"), float("-inf")):
                        raise ValueError(f"{where} has a non-finite '{k}' ({v!r})")
                elif kind.startswith("int"):
                    if isinstance(v, bool) or not isinstance(v, int):
                        raise ValueError(f"{where} has a non-integer '{k}' "
                                         f"({type(v).__name__})")
                elif kind == "bool":
                    if not isinstance(v, bool):
                        raise ValueError(f"{where} has a non-boolean '{k}' "
                                         f"({type(v).__name__})")
                elif kind == "str":
                    if not isinstance(v, str):
                        raise ValueError(f"{where} has a non-string '{k}' "
                                         f"({type(v).__name__})")
            for k, allowed in (enum_contract or {}).items():
                if k in row and row[k] not in allowed:
                    raise ValueError(
                        f"{os.path.basename(fn)} line {lineno} has an "
                        f"UNREGISTERED '{k}' value {row[k]!r} — the scorer "
                        f"partitions on this field, so a value outside "
                        f"{list(allowed)} would create a phantom family slice "
                        "no LOAO fold was fit for")
            if row["seed"] != seed:
                raise ValueError(
                    f"{os.path.basename(fn)} line {lineno} row seed "
                    f"{row['seed']} != filename seed {seed}")
            # ENUM, not merely a type. A string-valued but UNREGISTERED
            # scenario satisfies the value contract's `str` check and then
            # KeyErrors on the very next lookup — the round-13 attack_type
            # defect, one field over. A KeyError escaping a gate reads as an
            # executor crash; a corpus that is wrong has to SAY it is wrong.
            row_scen = str(row["scenario"]).lower()
            if row_scen not in scen_short:
                raise ValueError(
                    f"{os.path.basename(fn)} line {lineno} has an UNREGISTERED "
                    f"'scenario' value {row['scenario']!r} — the gate maps this "
                    f"field onto the registered roster {sorted(scen_short)}, so "
                    "a value outside it belongs to no LOAO fold")
            if scen_short[row_scen] != scen_short[scen]:
                raise ValueError(
                    f"{os.path.basename(fn)} line {lineno} row scenario "
                    f"{row['scenario']!r} != filename scenario")
            n += 1
            if row.get("malicious_gt"):
                malicious += 1
                fam = row.get("attack_type")
                if fam in registered:
                    fam_labelled += 1
                    by_family[fam] += 1
            else:
                honest += 1
    if n == 0:
        raise ValueError(f"{os.path.basename(fn)} has no rows")
    return {"rows": n, "honest": honest, "malicious": malicious,
            "family_labelled_malicious": fam_labelled, "by_family": by_family}


def validate_exp048_dir(sig_dir: str | os.PathLike, scen_short: dict,
                        scenarios: list[str], expected_seeds: list[int],
                        required_keys, value_contract=None,
                        enum_contract=None) -> dict:
    """CONTENT validation of the staged EXP-048 arm — returns a census or raises.

    A directory that merely EXISTS is not a staged input. If the gate accepts an
    empty directory, the failure surfaces only after the sealed corpus has been
    opened, and the one-shot read cannot be retried — so the check has to look at
    what is actually in the directory, before any sealed row is touched:

      * at least one `tge`-configuration file is present;
      * the grid matches the REGISTERED scenario × seed universe EXACTLY —
        `expected_seeds` comes from `data/h2_confirm_seeds.json` (EXP-048 ran
        the ten h2_confirm seeds), never from the staged files themselves, so a
        wholly-omitted seed halts instead of silently shrinking the universe;
      * every ROW of every file parses, carries the required fields, and its
        identity matches the filename.

    Raises ValueError with the specific defect; the caller converts that into
    the executor's HardStop.
    """
    expected = sorted(expected_seeds)
    files = sorted(glob.glob(os.path.join(str(sig_dir), "*.jsonl")))
    units: dict[tuple[str, int], str] = {}
    rows_seen: dict[tuple[str, int], int] = {}
    for fn in files:
        stem = os.path.basename(fn)[: -len(".jsonl")]
        parts = stem.split("__")
        if len(parts) != 4:
            # STRICTLY STRONGER THAN THE LOADER: the loader raises on any
            # `.jsonl` whose stem is not a four-part unit id, so a gate that
            # skipped it would pass a directory the loader then dies on —
            # after the sealed corpus is already open.
            raise ValueError(
                f"staged file is not a four-part unit id: {os.path.basename(fn)}")
        scen, defense, _exec, seedtok = parts
        if defense != "tge":
            continue
        if scen not in scen_short:
            raise ValueError(f"unknown scenario in staged EXP-048 file: {stem}")
        try:
            seed = int(seedtok.replace("seed", ""))
        except ValueError as exc:
            raise ValueError(f"unparseable seed in staged EXP-048 file: {stem}") from exc
        key = (scen_short[scen], seed)
        # DUPLICATE UNITS: the same (scenario, seed) staged twice under
        # different execution tokens. Silent last-wins would score whichever
        # file sorted later and hide the other entirely.
        if key in units:
            raise ValueError(
                f"duplicate staged unit for {key[0]}x{key[1]} — "
                f"{os.path.basename(units[key])} and {os.path.basename(fn)}. "
                "The contrast scores one file per cell; which one is not a "
                "choice this executor may make.")
        units[key] = fn
        rows_seen[key] = _parse_unit_rows(fn, scen, seed, scen_short,
                                          required_keys, value_contract,
                                          enum_contract)

    if not units:
        raise ValueError(
            "no standalone-TGE (`__tge__`) unit files found. The full-coverage "
            "contrast reads the STANDALONE-TGE arm; a directory of Krum+TGE logs "
            "cannot supply it.")
    missing = [f"{sc}x{sd}" for sc in scenarios for sd in expected
               if (sc, sd) not in units]
    if missing:
        raise ValueError(
            "staged EXP-048 arm does not cover the REGISTERED grid "
            f"({len(scenarios)} scenarios x {len(expected)} h2_confirm seeds). "
            f"Missing: {missing}")
    unregistered = sorted({(sc, sd) for (sc, sd) in units
                           if sd not in set(expected) or sc not in set(scenarios)})
    if unregistered:
        raise ValueError(
            "staged EXP-048 arm carries units outside the registered grid: "
            f"{[f'{sc}x{sd}' for sc, sd in unregistered]}")

    # ======================================================================
    # POPULATION ENUMERATION — every sub-population the registered mechanics
    # fit, calibrate, or take a rate from, with its disposition. Rounds 22-25
    # found the same defect at four grains one at a time; this list is the
    # artifact that covers the CLASS, and a new population must be added here
    # with a disposition rather than discovered by the next review round.
    #
    #  1. UNIT (scenario x seed) — honest >=1, malicious >=1, >=1 registered
    #     family. CHECKED below.
    #  2. ARM-WIDE held-out-family complement. CHECKED below. Subsumed by (3),
    #     kept ahead of it for the coarser, clearer diagnostic.
    #  3. ROTATION FIT (rot.fit's three seeds x each held-out family).
    #     CHECKED below. `fit_rows_all` filters on SEED ONLY, so diversity has
    #     to reach every rotation, not merely the arm.
    #  4. ROTATION CALIBRATION (honest rows per calibration seed x scenario).
    #     NOT separately checked — IMPLIED: grid completeness requires every
    #     (scenario, seed) unit, (1) requires honest >=1 on each, and every
    #     seed serves as calibration exactly once (`calib = s(1 + (i mod n))`
    #     is a permutation). Pinned by a test, not left as prose.
    #  5. ROTATION TEST per-family malicious. DELIBERATELY NOT REQUIRED — an
    #     absent family is the § 3.2 census's 0/0 "not scheduled in this fold"
    #     row, not a defect. Requiring it would contradict this module.
    #  6. BASELINE CUT population (`cv`, non-null instrument on calibration
    #     honest rows). MUST NOT be required — on the standalone-TGE arm
    #     `krum_score` is null on 100% of rows by construction, so `cv` is
    #     legitimately empty and the arm skips that instrument.
    #  7. G2 COVERED sub-populations (`cal_tge`, `te_h_tge`, `mal_tge`) — the
    #     tge_score-covered rows that set BOTH sides' G2 cuts. ABSORBED by
    #     census semantics: the producer routes an uncovered side to a
    #     CENSUS ONLY cell naming it in `uncovered_sides`, and the `elif`
    #     requires all three non-empty before any cut is taken.
    #  8. GLOBAL-CUT population (`cal_pooled`) — the calibration seed's honest
    #     rows POOLED ACROSS SCENARIOS for § 4 secondary 8. A different
    #     population from (4), so (4) says nothing about it. ABSORBED:
    #     `cut_global = {...} if cal_pooled else {}` plus `if cut_global:` at
    #     every consumer omits the block rather than cutting on nothing.
    #
    # AGGREGATION GRAIN is a SEPARATE AXIS from this list. The eight entries
    # are populations the mechanics FIT or CALIBRATE on; a reported census
    # figure additionally has to declare which folds it SUMS over. Round 26
    # found coverage summed over comparable folds only, which describes the
    # survivors rather than the slice. The rule, stated once: a coverage or
    # census aggregate sums over EVERY fold including disqualified ones, and
    # any figure scoped to the retained folds says so in a sibling
    # `*_population` key. Suppression is for contrast statistics; coverage is
    # never one.
    #
    # (7) and (8) are absorbed, not merely tolerated: `cut_from_calibration`
    # on an empty population raises IndexError, so those guards are what stand
    # between an uncovered fold and an opaque crash inside the sealed pass.
    # Both dispositions are proved by fixtures, not asserted.
    # ======================================================================

    # ---- CLASS POPULATIONS, not just the syntactic grid -------------------
    # A unit can satisfy the grid, parse cleanly, and still be unscorable. The
    # registered mechanics consume POPULATIONS, not rows: per-scenario
    # calibration takes its cut from the HONEST rows of the calibration seed,
    # the realized FPR is the honest flag rate, and every recall on both sides
    # is computed over MALICIOUS rows. A unit missing either class yields no
    # operating point and no readout — and says so only after the sealed
    # corpus is open, which is the one thing this gate exists to prevent.
    #
    # The third check is the subtle one. `score_corpus` partitions malicious
    # rows by family and pools only the REGISTERED families
    # (`mal_pooled = [r for A2 in ATTACKS for r in fam[A2]]`), so a unit whose
    # entire malicious population carries the discovery-row marker ("") falls
    # out of the pooled contrast in silence: malicious rows present, no cell.
    #
    # DELIBERATELY NOT CHECKED — per-family presence. The frozen harness
    # registers a flat family roster and a scenario roster but NO
    # scenario-to-family schedule, so any per-family expectation would have to
    # be read off the staged arm itself, which is deriving the expectation from
    # the data under check. It would also contradict the § 3.2 census semantics
    # this very module implements, where a family absent from a fold is a 0/0
    # "family not scheduled in this fold" row and not a defect. Requiring it
    # would hard-stop legitimate arms: in the disclosed EXP-011 dev corpus S0
    # and S1 schedule `alie` alone and S3 carries no `label_flip`.
    registered_families = tuple(
        a for a in (enum_contract or {}).get("attack_type", ()) if a)
    deficient = []
    for key in sorted(units):
        c = rows_seen[key]
        unit = f"{key[0]}x{key[1]}"
        if not c["honest"]:
            deficient.append(
                f"{unit}: {c['rows']} rows, NO HONEST rows — the calibration "
                "cut and the realized FPR are both undefined without them")
        if not c["malicious"]:
            deficient.append(
                f"{unit}: {c['rows']} rows, NO MALICIOUS rows — neither side "
                "of the matched contrast has a recall population")
        elif registered_families and not c["family_labelled_malicious"]:
            # The roster GUARD, not an optimisation: without a registered
            # family roster this check has nothing to check against, and
            # asserting "no registered family" from an empty roster would
            # condemn every unit. The executor's own call site always passes
            # SCORING_ENUM_CONTRACT, so the production path is fully covered —
            # a test pinning that is the companion to this branch.
            deficient.append(
                f"{unit}: {c['malicious']} malicious rows but NONE carries a "
                f"registered family label {list(registered_families)} — the "
                "pooled contrast reads registered families only, so this unit "
                "would contribute no cell at all")
    if deficient:
        raise ValueError(
            "staged EXP-048 arm satisfies the grid but carries units with no "
            "scorable class population: " + "; ".join(deficient))

    # ---- LOAO FIT POPULATIONS, at the ARM grain --------------------------
    # The § 2.2a item 3 fit holds out one family at a time:
    #     fit_rows_all = [r for r in rows if r["_seed"] in rot.fit]
    #     tr = [r for r in fit_rows_all
    #           if not (r["malicious_gt"] and r.get("attack_type") == A)]
    #     y  = [bool(r["malicious_gt"]) for r in tr]
    # so the positive class that survives holding out family A is every
    # malicious row whose family is NOT A — pooled across the fit seeds and,
    # crucially, across ALL SCENARIOS (`fit_rows_all` filters on seed only).
    #
    # That pooling is exactly why a per-SCENARIO one-family arm is LEGAL: S0
    # carrying only `alie` is fine because S2 in the same fit seeds supplies the
    # other families. It is also why an ARM-WIDE single-family grid is not —
    # holding out that one family empties the positive class everywhere at once,
    # `y` becomes single-class, and the GBDT fit dies inside the sealed pass.
    #
    # The primary arm has NO runtime guard for this: the single-class check at
    # the fit site is `if strict and len(np.unique(y)) < 2`, armed only for the
    # strict-identity sensitivity arm. Unguarded at fit time means it has to be
    # caught here, before the sealed corpus is opened.
    #
    # The complement counts EVERY malicious row that is not family A, including
    # discovery rows (""): `y` is `malicious_gt`, and `tr` drops rows only by
    # family match, so an unlabelled attacker still carries the positive class.
    if registered_families:
        arm_total_malicious = sum(c["malicious"] for c in rows_seen.values())
        arm_by_family = {
            A: sum(c["by_family"].get(A, 0) for c in rows_seen.values())
            for A in registered_families}
        unfittable = []
        for A in registered_families:
            if not arm_by_family[A]:
                continue                 # not scored anywhere; nothing to fit
            complement = arm_total_malicious - arm_by_family[A]
            if complement == 0:
                unfittable.append(
                    f"{A} ({arm_by_family[A]} malicious rows arm-wide, and NO "
                    "malicious rows of any other family or discovery rows "
                    "remain once it is held out)")
        if unfittable:
            raise ValueError(
                "staged EXP-048 arm cannot support the § 2.2a LOAO fits — "
                "holding out these families empties the positive class across "
                "the whole arm, so the fit is single-class: "
                + "; ".join(unfittable)
                + ". The fit pools across the fit seeds' scenarios, so a "
                "per-SCENARIO one-family arm is fine; an ARM-WIDE one is not")

        # ---- ...and again at the ROTATION grain --------------------------
        # The arm-wide check above is the COARSE one and cannot see this: an
        # arm diverse overall can still contain a rotation whose three fit
        # seeds happen to carry a single family. `fit_rows_all` selects on
        # `r["_seed"] in rot.fit`, so the fit population is those three seeds
        # ONLY — diversity living in the other two seeds does not reach it.
        #
        # The rotations are the REGISTERED § 2.2a construction imported from
        # h2prime_corpus, built over the same expected seed universe the
        # reducer uses. Re-deriving the index formula here would be a second
        # copy free to drift from the one that actually runs.
        by_seed_total: dict[int, int] = {}
        by_seed_family: dict[int, dict] = {}
        for (_sc, sd), c in rows_seen.items():
            by_seed_total[sd] = by_seed_total.get(sd, 0) + c["malicious"]
            d = by_seed_family.setdefault(sd, {a: 0 for a in registered_families})
            for a in registered_families:
                d[a] += c["by_family"].get(a, 0)
        rot_bad = []
        for rot in rotation_plan(expected):
            fit_seeds = list(rot.fit)
            fit_total = sum(by_seed_total.get(sd, 0) for sd in fit_seeds)
            if fit_total == 0:
                rot_bad.append(
                    f"rotation {rot.i} (fit seeds {fit_seeds}): NO malicious "
                    "rows at all across the fit seeds, so every held-out-family "
                    "fit is single-class")
                continue
            for A in registered_families:
                fit_A = sum(by_seed_family.get(sd, {}).get(A, 0)
                            for sd in fit_seeds)
                if fit_A and fit_total - fit_A == 0:
                    rot_bad.append(
                        f"rotation {rot.i} (fit seeds {fit_seeds}): holding out "
                        f"{A} leaves 0 of {fit_total} malicious rows — the fit "
                        "seeds carry that family alone")
        if rot_bad:
            raise ValueError(
                "staged EXP-048 arm is diverse overall but cannot support the "
                "§ 2.2a LOAO fits ROTATION BY ROTATION: " + "; ".join(rot_bad)
                + ". The fit population is `rot.fit`'s three seeds only, so "
                "family diversity has to reach every rotation, not just the arm")

    return {"n_files": len(units), "scenarios": sorted(scenarios),
            "seeds": expected,
            "rows_parsed": sum(c["rows"] for c in rows_seen.values()),
            "honest_rows": sum(c["honest"] for c in rows_seen.values()),
            "malicious_rows": sum(c["malicious"] for c in rows_seen.values())}

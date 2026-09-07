#!/usr/bin/env python3
"""H3 PRIMARY scorer (D7) — detector-independent re-link recall / false-link rate.

Design authority — v1.10 § 5.1, the FROZEN scorer contract
----------------------------------------------------------
* **ARM RULE:** "the H3 Boolean is scored on the `TGE+FP` arm ONLY". The
  `TGE-only` arm has no registry, hence no re-link assertions, and contributes
  **nothing** to the primary Boolean. The arm token is read from the event row's
  **`h3_arm`** field — see the naming note below.
* **VALIDATION COHORT NEVER ADJUDICATES:** § 5.1 — the validation run "reports
  the metric, but no H3 pass/fail is recorded from it", and its performance "can
  neither block nor tune the sealed run". Its report therefore carries
  `verdict: null`, `informs_only: true` and
  `verdict_status: "REPORTED (NOT ADJUDICATING)"`; every metric VALUE is still
  reported in full, only the pass/fail CLAIM is withheld. The ADJUDICATING
  cohort's verdict shape is unchanged.
* **CLASSIFICATION:** "A re-entry event is a **correct re-link** iff
  `asserted_match ∧ (asserted_parent_logical_id == gt_logical_id)`; a **false
  link** iff `asserted_match ∧ (asserted_parent_logical_id != gt_logical_id)`
  **or** an honest (`gt_is_malicious=false`) event with `asserted_match=true`."
* **RECALL** = correct re-links / `gt_is_malicious` events, **pooled** across the
  5 seeds as a single ratio per scenario (**not** a per-seed mean), required
  separately in S3 AND S4, each ≥ 0.85.
* **FLR** = false links / `gt_is_malicious=false` events, **S4 only** (S3 has
  zero honest CID-change events by construction), pooled, ≤ 0.10.
* **VERDICT:** PASS iff `recall_S3 ≥ 0.85` AND `recall_S4 ≥ 0.85` AND
  `FLR_S4 ≤ 0.10`. Any component failing is falsification.
* **ZERO-DENOMINATOR RULE:** a component with a zero denominator is
  INCONCLUSIVE and the verdict is INCONCLUSIVE-gated — never auto-pass, never
  auto-fail.
* **DEDUP** on `reentry_event_key` (one row per event, not per round
  participation).
* **DEVICE HOLD-OUT (D9 axis ii, ADJUDICATING cohort):** keep only events whose
  `gt_logical_id` maps to an **odd** base partition.
* **PRECISION** is reported for context only and is never a bar (F3).

The scorer runs on the LOGGED REGISTRY ASSERTIONS, never on enforcement
outcomes. That is what makes it detector-independent (§ 5.1 integrity
assertion), and it is why nothing in this file reads an aggregation coefficient.

WHY THE ARM FIELD IS `h3_arm` AND NOT `arm` (DRYRUN F4)
------------------------------------------------------
`arm` is ALREADY TAKEN in this codebase: `flowerfl/resampling_manifest.py` and
`flowerfl/client_app.py` use it for the resampling manifest's variant label
(`smote@balanced`, `off`, …), and that label travels in the same signal-log row
as the H3 defense token. Two different meanings under one key is how a control
row silently ends up in an adjudicating denominator. The event corpus therefore
carries the defense token as **`h3_arm`** (emitted by
`scripts/extract_h3_events.py`), the rule itself is unchanged (TGE+FP only), and
a row that carries `arm` but no `h3_arm` is refused BY NAME rather than read as
if the two were the same field.

>>> ADDENDUM B — TWO CORRECTIONS TO THE FROZEN DECISION STATISTIC <<<
**Status: RATIFIED 2026-08-08 (Erik Jones; methodology v1.46) — Addendum B of
`docs/reproduction/experiments.md` (STATUS: RATIFIED —
IN FORCE); this scorer implements the ratified semantics.** Correcting a
structurally unsatisfiable frozen statistic still *changes* it, so neither
correction below is a mere reading — both are recorded here and surfaced in
every report rather than left implicit.

**B1 — identity comparison, implemented as `base_partition`.** The contract
compares `asserted_parent_logical_id == gt_logical_id`. Read as raw string
equality that comparison can never be satisfied: `gt_logical_id` is the
RE-ENTRANT's identity (e.g. `client_2_new2`) while the asserted parent is a
PRIOR identity of the same device (`client_2`, or `client_2_new1` after a
chained reset), so recall would be structurally 0 for every event. This scorer
compares at **base-partition granularity** via `partition_of()` — the same
`LOGICAL_TO_PARTITION` map § 5.1 names as the truth key, and the same derivation
§ 5.1 already uses for the odd/even parity filter ("parity derived from
`gt_logical_id`; no extra field"). Emitted as `identity_comparison`.

**B2 — false-link population, implemented as `honest_only`.** The contract's
prose labels a matched malicious event with the wrong parent a "false link",
but defines FLR's denominator as the honest population only — so the stated
numerator and denominator are drawn from different populations. FLR could
exceed 1.0, and a purely positive-population failure could falsify H3 on the
false-link bar with zero honest devices mislinked. This scorer separates
`wrong_malicious_link` (depresses RECALL) from `false_link` (matched honest
events — the FLR numerator). Emitted as `false_link_population`. See
`classify_event`.
"""
from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.fingerprint_registry import is_holdout_partition, partition_of  # noqa: E402

# ===========================================================================
# FROZEN CONSTANTS (v1.10 § 5.1)
# ===========================================================================

RECALL_BAR: float = 0.85
FLR_BAR: float = 0.10

#: The `TGE+FP` arm token. The arm rule scores this arm and no other.
FP_ARM_DEFAULT: str = "tgefp"

#: The event-row field carrying the H3 arm token. DELIBERATELY NOT `arm` — that
#: name is the resampling manifest's variant label (see the module docstring).
H3_ARM_FIELD: str = "h3_arm"

#: The resampling manifest's field name. Its presence WITHOUT `h3_arm` is a
#: named refusal, never a fallback: the two fields mean different things.
RESAMPLING_ARM_FIELD: str = "arm"

#: Scenario keys as they appear in the event rows.
SCENARIO_S3 = "S3_identity_reset_only"
SCENARIO_S4 = "S4_full_mix"

#: Addendum B semantics, surfaced in every report — see module docstring. The
#: estimator-addenda spec (§ "On ratification") directs that this string be
#: flipped from `AMENDMENT-REQUIRED` to a citation of that file once ratified;
#: that landed 2026-08-08 at methodology v1.46. The declared constants below do
#: not change — the scorer already implemented Addendum B as ratified.
IDENTITY_COMPARISON = "base_partition"          # B1
FALSE_LINK_POPULATION = "honest_only"           # B2
ADDENDUM_STATUS = (
    "RATIFIED 2026-08-08 (Erik Jones; methodology v1.46) — Addendum B of "
    "docs/superpowers/specs/2026-08-08-h3-estimator-addenda.md: B1 identity "
    "comparison at base-partition granularity; B2 false-link numerator restricted "
    "to the honest population, malicious wrong-parent links depress recall instead."
)

REQUIRED_FIELDS = (
    "reentry_event_key",
    "server_round",
    "current_cid",
    "gt_logical_id",
    "gt_is_malicious",
    "asserted_match",
    "asserted_parent_entry_id",
    "asserted_parent_logical_id",
    "min_d",
    "tau",
    "generation",
)

#: Values that mean "provenance is missing" — gate (b) requires zero of these.
UNUSABLE_VALUES = {"UNKNOWN", "unknown", ""}


class Cohort(str, Enum):
    VALIDATION = "validation"
    ADJUDICATING = "adjudicating"


#: What the validation cohort's `verdict_status` says instead of PASS/FAIL.
VALIDATION_VERDICT_STATUS = "REPORTED (NOT ADJUDICATING)"

#: Why the validation cohort's `verdict` is null (F5 — the anti-discretion core
#: of D9). Written into every validation artifact so the withholding travels
#: with the numbers rather than living only in this file.
VALIDATION_VERDICT_WITHHELD_REASON = (
    "v1.10 § 5.1: the VALIDATION run 'reports the metric, but no H3 pass/fail is "
    "recorded from it', and its performance 'can neither block nor tune the "
    "sealed run'. Every metric value below is reported in full; only the "
    "pass/fail claim is withheld. The per-component `status` fields are "
    "metric-versus-bar comparisons reported for information and constitute NO "
    "H3 verdict — the H3 Boolean comes from the ADJUDICATING cohort alone."
)


#: § 5.1 expected event counts. VALIDATION scores all devices; ADJUDICATING
#: scores only the odd-partition hold-out.
DESIGN_COUNTS: Dict[Cohort, Dict[str, int]] = {
    Cohort.VALIDATION: {"recall_S3": 180, "recall_S4": 180, "flr_S4": 15},
    Cohort.ADJUDICATING: {"recall_S3": 80, "recall_S4": 80, "flr_S4": 15},
}


class ScoringError(RuntimeError):
    """Raised when the event corpus cannot be scored under the frozen contract."""


# ===========================================================================
# Classification
# ===========================================================================

def _same_device(asserted_parent_logical_id: str, gt_logical_id: str) -> bool:
    """Identity comparison at base-partition granularity (see module docstring)."""
    return partition_of(asserted_parent_logical_id) == partition_of(gt_logical_id)


def classify_event(event: Mapping[str, Any]) -> str:
    """`correct_relink` | `wrong_malicious_link` | `false_link` | `miss` | `true_negative`.

    **Why five classes and not the contract's four.** § 5.1's prose labels BOTH
    "matched malicious event whose parent is wrong" and "matched honest event"
    as a *false link*, but defines FLR's denominator as the honest
    (`gt_is_malicious=false`) population only. Pooling the two into one
    numerator divides a count drawn from the malicious ∪ honest populations by
    an honest-only denominator: FLR could exceed 1.0, and — worse — a purely
    POSITIVE-population failure (the matcher linking a malicious re-entrant to
    the wrong device) could fail the false-link bar without a single honest
    device ever being mislinked.

    The two are therefore separated:
      * `wrong_malicious_link` — matched, malicious, wrong parent. Not a correct
        re-link, so it depresses RECALL. It never touches FLR.
      * `false_link` — matched honest event. The FLR numerator, drawn from
        exactly the population FLR's denominator counts.

    This is part of the same "the frozen statistic is not self-consistent"
    family as the identity-comparison reading below; both are covered by the
    pending scorer addendum (see the module docstring).
    """
    matched = bool(event["asserted_match"])
    malicious = bool(event["gt_is_malicious"])

    if not matched:
        return "miss" if malicious else "true_negative"

    parent = event.get("asserted_parent_logical_id")
    if not parent:
        raise ScoringError(
            f"event {event.get('reentry_event_key')!r} asserts a match but carries "
            "no asserted_parent_logical_id — the row is not scorable"
        )
    if not malicious:
        # An honest reconnection linked to a flagged entry is a false link,
        # whichever entry it was linked to.
        return "false_link"
    if _same_device(str(parent), str(event["gt_logical_id"])):
        return "correct_relink"
    return "wrong_malicious_link"


# ===========================================================================
# Loading
# ===========================================================================

def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ScoringError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return rows


def load_events(
    paths: Sequence[Path],
    arm: str = FP_ARM_DEFAULT,
    arm_field: str = H3_ARM_FIELD,
) -> List[Dict[str, Any]]:
    """Load schema-v5 re-entry rows: apply the ARM RULE, then deduplicate.

    Rows from any other arm are dropped by design (the arm rule), but a row that
    is malformed, carries unknown provenance, or conflicts with another row
    under the same `reentry_event_key` is a hard error — silently tolerating
    either would corrupt a denominator.

    The arm token is read from `h3_arm`. A row carrying only the resampling
    manifest's `arm` is refused by name (DRYRUN F4): the two fields are
    different quantities and reading one as the other would let a control row
    into an adjudicating denominator.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ScoringError(f"events file not found: {path}")
        for row in _iter_rows(path):
            row_arm = row.get(arm_field)
            if row_arm is None:
                if arm_field == H3_ARM_FIELD and RESAMPLING_ARM_FIELD in row:
                    raise ScoringError(
                        f"{path}: row {row.get('reentry_event_key')!r} has no "
                        f"{H3_ARM_FIELD!r} field but does carry "
                        f"{RESAMPLING_ARM_FIELD!r}={row[RESAMPLING_ARM_FIELD]!r} — "
                        "that is the RESAMPLING manifest's variant label "
                        "(flowerfl/resampling_manifest.py), NOT the H3 arm, and "
                        "it is never read as one. Re-extract the corpus with "
                        "scripts/extract_h3_events.py, which emits the defense "
                        f"token as {H3_ARM_FIELD!r}"
                    )
                raise ScoringError(
                    f"{path}: row has no {arm_field!r} field — the arm rule "
                    "(v1.10 § 5.1) cannot be applied"
                )
            if str(row_arm) != str(arm):
                continue  # the TGE-only control arm contributes nothing (arm rule)

            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ScoringError(
                    f"{path}: re-entry row is missing required schema-v5 field(s): "
                    f"{missing}"
                )
            for field in ("gt_logical_id", "current_cid", "reentry_event_key"):
                if str(row[field]) in UNUSABLE_VALUES or row[field] is None:
                    raise ScoringError(
                        f"{path}: row {row.get('reentry_event_key')!r} has "
                        f"UNKNOWN/missing provenance in {field!r} — gate (b) "
                        "requires zero such rows"
                    )

            key = str(row["reentry_event_key"])
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(row)
                continue
            conflicting = [
                f for f in REQUIRED_FIELDS if existing.get(f) != row.get(f)
            ]
            if conflicting:
                raise ScoringError(
                    f"conflict under reentry_event_key {key!r}: rows disagree on "
                    f"{conflicting}"
                )
    return list(by_key.values())


# ===========================================================================
# Scoring
# ===========================================================================

def _component(
    label: str,
    scenario: str,
    numerator: int,
    denominator: int,
    bar: float,
    direction: str,
) -> Dict[str, Any]:
    if denominator == 0:
        return {
            "label": label,
            "scenario": scenario,
            "numerator": numerator,
            "denominator": 0,
            "value": None,
            "bar": bar,
            "direction": direction,
            "status": "INCONCLUSIVE",
        }
    value = numerator / denominator
    passed = value >= bar if direction == "at_least" else value <= bar
    return {
        "label": label,
        "scenario": scenario,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "bar": bar,
        "direction": direction,
        "status": "PASS" if passed else "FAIL",
    }


def score_events(
    events: Sequence[Mapping[str, Any]],
    cohort: Cohort,
    require_design_counts: bool = True,
) -> Dict[str, Any]:
    """Apply the frozen decision statistic and return the full report.

    Args:
        events: deduplicated, arm-filtered re-entry rows.
        cohort: ADJUDICATING applies the D9 odd-partition hold-out filter.
        require_design_counts: gate (d) — refuse when the realised event census
            differs from the design counts. Set False only for diagnostics on a
            deliberately partial corpus; the verdict path never sets it False.
    """
    cohort = Cohort(cohort)
    scored = [
        event
        for event in events
        if cohort is Cohort.VALIDATION or is_holdout_partition(str(event["gt_logical_id"]))
    ]

    empty_tally = {
        "correct_relink": 0,
        "wrong_malicious_link": 0,
        "false_link": 0,
        "miss": 0,
        "true_negative": 0,
    }
    tallies = {
        SCENARIO_S3: dict(empty_tally),
        SCENARIO_S4: dict(empty_tally),
    }
    populations = {
        SCENARIO_S3: {"malicious": 0, "honest": 0},
        SCENARIO_S4: {"malicious": 0, "honest": 0},
    }
    for event in scored:
        scenario = str(event["scenario"])
        if scenario not in tallies:
            raise ScoringError(
                f"unexpected scenario {scenario!r} — H3 is scored on "
                f"{SCENARIO_S3} and {SCENARIO_S4} only"
            )
        tallies[scenario][classify_event(event)] += 1
        populations[scenario][
            "malicious" if bool(event["gt_is_malicious"]) else "honest"
        ] += 1

    components = {
        "recall_S3": _component(
            "re-link recall (S3)", SCENARIO_S3,
            tallies[SCENARIO_S3]["correct_relink"],
            populations[SCENARIO_S3]["malicious"],
            RECALL_BAR, "at_least",
        ),
        "recall_S4": _component(
            "re-link recall (S4)", SCENARIO_S4,
            tallies[SCENARIO_S4]["correct_relink"],
            populations[SCENARIO_S4]["malicious"],
            RECALL_BAR, "at_least",
        ),
        # S3 is excluded from FLR by construction: identity-reset-only has zero
        # honest CID-change events (v1.10 § 5.1).
        # NUMERATOR IS HONEST-ONLY: `false_link` counts matched HONEST events
        # exclusively. Malicious wrong-parent matches are `wrong_malicious_link`
        # and depress recall instead — pooling them here would divide a
        # malicious-population count by an honest-only denominator.
        "flr_S4": _component(
            "false-link rate (S4)", SCENARIO_S4,
            tallies[SCENARIO_S4]["false_link"],
            populations[SCENARIO_S4]["honest"],
            FLR_BAR, "at_most",
        ),
    }

    realised = {name: comp["denominator"] for name, comp in components.items()}
    design = DESIGN_COUNTS[cohort]
    deltas = {
        name: realised[name] - design[name]
        for name in design
        if realised[name] != design[name]
    }
    if deltas and require_design_counts:
        raise ScoringError(
            f"realised event census does not equal the design counts for cohort "
            f"'{cohort.value}': realised {realised} vs design {design} "
            f"(deltas {deltas}). Gate (d) requires an exact match."
        )

    statuses = {comp["status"] for comp in components.values()}
    if "INCONCLUSIVE" in statuses:
        verdict = "INCONCLUSIVE"
    elif "FAIL" in statuses:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    # F5 — THE VALIDATION COHORT RECORDS NO PASS/FAIL CLAIM.
    # § 5.1: it "reports the metric, but no H3 pass/fail is recorded from it".
    # A bare `verdict: "PASS"` on the informing cohort is shape-indistinguishable
    # from the adjudicating verdict and is exactly the artifact that gets quoted
    # later as though H3 had passed, so the claim is withheld at the source —
    # not merely annotated. The numbers are all still here; only the claim is
    # gone. Withholding is unconditional: a validation FAIL is withheld too, so
    # this can never operate as a PASS-only filter.
    informs_only = cohort is Cohort.VALIDATION
    if informs_only:
        reported_verdict: Any = None
        verdict_status = VALIDATION_VERDICT_STATUS
        verdict_withheld_reason: Any = VALIDATION_VERDICT_WITHHELD_REASON
    else:
        reported_verdict = verdict
        verdict_status = verdict
        verdict_withheld_reason = None

    return {
        "cohort": cohort.value,
        "identity_comparison": IDENTITY_COMPARISON,
        "false_link_population": FALSE_LINK_POPULATION,
        "addendum_status": ADDENDUM_STATUS,
        "arm_rule": f"TGE+FP arm only, read from {H3_ARM_FIELD!r} (v1.10 § 5.1)",
        "n_events_scored": len(scored),
        "n_events_loaded": len(events),
        "components": components,
        "informs_only": informs_only,
        "verdict": reported_verdict,
        "verdict_status": verdict_status,
        "verdict_withheld_reason": verdict_withheld_reason,
        "verdict_rule": (
            "PASS iff recall_S3 >= 0.85 AND recall_S4 >= 0.85 AND FLR_S4 <= 0.10; "
            "any zero denominator makes that component INCONCLUSIVE and gates the "
            "verdict — never an auto-pass"
        ),
        "design_counts": design,
        "realised_counts": realised,
        "counts_match_design": not deltas,
        "design_count_deltas": deltas,
        "event_tallies": tallies,
        "populations": populations,
        "reported_only": _reported_only(components, tallies, populations),
    }


def _wrong_malicious_link_rate(tallies, populations, scenario: str) -> Dict[str, Any]:
    """Addendum B: the wrong-malicious-link COUNT, and its RATE on the same
    denominator recall uses.

    Separating `wrong_malicious_link` from FLR (B2) must not bury it. Eight
    mislinks read very differently at a denominator of 80 than at 180, and the
    absolute count alone hides that, so the proportion is reported beside it.
    REPORTED CONTEXT, NEVER A BAR (F3) — nothing here enters any verdict
    condition.

    Zero denominator follows the file's INCONCLUSIVE convention (`_component`):
    value `None` with an explicit status, never a NaN and never a 0.0 that would
    read as "no mislinks".
    """
    count = tallies[scenario]["wrong_malicious_link"]
    denominator = populations[scenario]["malicious"]
    inconclusive = denominator == 0
    suffix = scenario[:2]
    return {
        f"wrong_malicious_links_{suffix}": count,
        f"wrong_malicious_link_rate_{suffix}": (
            None if inconclusive else count / denominator
        ),
        f"wrong_malicious_link_rate_denominator_{suffix}": denominator,
        f"wrong_malicious_link_rate_status_{suffix}": (
            "INCONCLUSIVE" if inconclusive else "REPORTED"
        ),
    }


def _reported_only(components, tallies, populations) -> Dict[str, Any]:
    """Context numbers that are REPORTED and never gate (F3: FPR != precision)."""
    correct = tallies[SCENARIO_S4]["correct_relink"]
    false_links = tallies[SCENARIO_S4]["false_link"]
    wrong_malicious = tallies[SCENARIO_S4]["wrong_malicious_link"]
    linked = correct + false_links + wrong_malicious
    return {
        "precision_S4": (correct / linked) if linked else None,
        "precision_note": (
            "implied precision, reported for context only — never a threshold "
            "(v1.10 § 5.1 F3)"
        ),
        # Reported for BOTH scenarios: recall is a bar in S3 and S4 alike, so a
        # matcher that links malicious re-entrants to the wrong device must be
        # visible as a proportion in both.
        **_wrong_malicious_link_rate(tallies, populations, SCENARIO_S3),
        **_wrong_malicious_link_rate(tallies, populations, SCENARIO_S4),
        "wrong_malicious_link_note": (
            "matched malicious re-entrants whose asserted parent is a DIFFERENT "
            "device: these depress recall and are excluded from FLR, whose "
            "denominator counts honest events only. The rate is the count over "
            "the SAME denominator recall uses (the scenario's malicious event "
            "population) — reported context, NEVER a bar; a zero denominator "
            "reads INCONCLUSIVE, never 0.0"
        ),
        "flr_granularity_note": (
            "at a denominator of "
            f"{populations[SCENARIO_S4]['honest']}, FLR resolves in steps of "
            "1/n; the bar means 'at most one false link in fifteen' on the "
            "adjudicating cohort — a documented limitation"
        ),
    }


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the H3 PRIMARY metric (D7) from schema-v5 re-entry events. "
            "Runs on the logged registry assertions only — never on enforcement "
            "outcomes."
        )
    )
    parser.add_argument("--events", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--cohort", required=True, choices=[c.value for c in Cohort],
        help="validation (all devices, informs only) or adjudicating (odd hold-out)",
    )
    parser.add_argument("--arm", default=FP_ARM_DEFAULT, help="the +FP arm token")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-partial-census",
        action="store_true",
        help=(
            "DIAGNOSTIC ONLY: score a corpus whose event census differs from the "
            "design counts. Never use this for a verdict — gate (d) requires an "
            "exact match."
        ),
    )
    args = parser.parse_args(argv)

    try:
        events = load_events(args.events, arm=args.arm)
        report = score_events(
            events,
            Cohort(args.cohort),
            require_design_counts=not args.allow_partial_census,
        )
    except ScoringError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    for name, component in report["components"].items():
        value = component["value"]
        rendered = "n/a" if value is None else f"{value:.4f}"
        print(
            f"{name:>10}  {rendered:>8}  "
            f"({component['numerator']}/{component['denominator']})  "
            f"bar {component['direction']} {component['bar']}  "
            f"{component['status']}"
        )
    # The validation cohort prints its status, never a bare verdict — the
    # terminal transcript is quoted as readily as the JSON (F5).
    print(f"\nVERDICT ({report['cohort']}): {report['verdict_status']}")
    if report["informs_only"]:
        print(report["verdict_withheld_reason"])
    if args.allow_partial_census:
        print("WARNING: --allow-partial-census was set; this is NOT a verdict run.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

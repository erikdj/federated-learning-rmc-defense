#!/usr/bin/env python3
"""H3 CORRECTED-INSTRUMENT scorer — identity-only rank-1 / wrong-device rate.

STATUS: **RATIFIED — rev-5, Erik Jones 2026-08-14 ("approve"), methodology v1.50.**
-----------------------------------------------------------------------------------
This implements the H3 scoring workflow in
`docs/reproduction/experiments.md`. The bars below — P1 ≥ 0.85 rank-1 on the MALICIOUS re-entry
population per scenario, and a wrong-device link guard ≤ 0.10 scored SEPARATELY
on the malicious and honest populations — are **FROZEN**. This scorer's ONE
execution on the sealed adjudicating cohort adjudicates H3; validation-cohort
reads inform and never adjudicate.

Why a second scorer instead of an edit
--------------------------------------
`scripts/analyze_h3_relink.py` is the FROZEN D7 scorer of the as-built
instrument, and its FAIL verdict stands as a result about that instrument. It is
not edited, not imported for its decision statistic, and not superseded. This
file scores a DIFFERENT instrument asking a DIFFERENT question.

* **The as-built instrument (D7)** gated the re-entry candidate pool on upstream
  detector flags, so what it measured was P(flagged by the detector) ×
  P(re-linked | flagged) — RQ2's detection performance multiplied into RQ3's
  identity question. Methodology v1.49 ruled that an invalid test of H3.
* **The corrected instrument** enrolls every session and considers ALL sessions
  first seen strictly before the re-entrant. The question becomes purely
  identification: *is the nearest candidate the correct base device?*

THE REDEFINITION THAT FOLLOWS (§ 2)
-----------------------------------
**An honest device correctly re-identified as ITSELF is a CORRECT decision.**
Every one of EXP-056's so-called "false links" was a self-re-identification
(`client_13 → client_13`, `client_15 → client_15`); under an identity-only
instrument that outcome is the instrument working. The enforcement consequences
of linking honest devices belong to H4's utility metrics, not here.

The guard therefore counts WRONG-DEVICE links only, and counts them SEPARATELY
on the malicious and honest populations. Pooling them hides the honest
population inside the malicious one: with 15 honest events in 175, every honest
device could be linked to the WRONG device and a pooled guard would still read
15/175 = 0.086 and PASS.

RANK-1 AND THE NEAREST-CANDIDATE COLUMN  (read this before quoting P1)
----------------------------------------------------------------------
P1 is defined threshold-FREE: the nearest candidate is the correct base device,
whatever the distance. The frozen re-entry contract records
`asserted_parent_logical_id` **only when the match fired** (`min_d ≤ τ`), so on
its own it cannot answer that for an event whose nearest candidate sat beyond τ.

The registry therefore records the nearest candidate UNCONDITIONALLY, in the
additive `nearest_entry_id` / `nearest_logical_id` pair
(`signal_logger.REENTRY_NEAREST_FIELDS`; the frozen v1.10 § 5.1 table is
untouched and the D7 scorer's `REQUIRED_FIELDS` does not move — a standing
regression guard asserts D7 output is byte-identical with and without the pair).

On a corpus carrying the column, rank-1 is fully determinate:

* `nearest_logical_id` **non-null** — that is the nearest candidate; compare it
  to the truth at base-partition grain.
* `nearest_logical_id` **null** — the candidate pool was genuinely EMPTY, so
  rank-1 cannot have been correct: a determinate `rank1_no_candidate`.

On a corpus written BEFORE the extension the key is absent entirely, and the
honest reading is kept rather than deleted: an unmatched event with a finite
`min_d` had a nearest candidate whose identity was discarded, so it is
`rank1_indeterminate` and makes that scenario's P1 **INCONCLUSIVE** — never an
auto-pass, never an auto-fail, exactly as a zero denominator is treated.
`nearest_candidate_coverage` in every report says which corpus this is.

Note what the extension does NOT give: it records the nearest candidate, not the
ordered candidate list, so ranks 2 and beyond remain unrecorded and the CMC
curve stays in `not_derivable`.

Frozen mechanics reused, not re-implemented
-------------------------------------------
The arm rule (`TGE+FP` only, read from `h3_arm`), the `reentry_event_key` dedup,
the malformed/unknown-provenance refusals, base-partition identity comparison
and the odd-partition hold-out filter are imported from the D7 scorer and the
registry so the two cannot drift. Only the decision statistic is new.

THE INSTRUMENT ASSERTION
------------------------
The event corpus does not carry the registry policy — `extract_h3_events.py`
projects the schema-v5 row onto a fixed field list, and the policy lives in the
RESULT JSON (`provenance.fp_registry_policy` and the custody block's
`fingerprint_registry.registry_policy`). This scorer therefore REQUIRES the unit
result JSONs and refuses unless every one of them declares — and observed —
`identity_only`. Scoring flag-gated events under this statistic would produce a
rank-1 number for the very instrument v1.49 ruled invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.fingerprint_registry import (  # noqa: E402
    CALIBRATION_ARTIFACT_SHA256,
    CalibrationCohort,
    RegistryPolicy,
    is_holdout_partition,
    locked_metric,
    locked_tau,
    partition_of,
)
from flowerfl.signal_logger import build_reentry_event_key  # noqa: E402

# The D7 scorer is imported READ-ONLY, for the machinery both instruments share.
# Its decision statistic (RECALL_BAR / FLR_BAR / classify_event / score_events)
# is deliberately not used — a test asserts those names never appear here.
from scripts.analyze_h3_relink import (  # noqa: E402
    FP_ARM_DEFAULT,
    H3_ARM_FIELD,
    SCENARIO_S3,
    SCENARIO_S4,
    VALIDATION_VERDICT_STATUS,
    VALIDATION_VERDICT_WITHHELD_REASON,
    Cohort,
    ScoringError,
    load_events,
)

# ===========================================================================
# RATIFIED constants (amendment § 2 — FROZEN at rev-5, methodology v1.50)
# ===========================================================================

STATUS = (
    "RATIFIED — amendment 2026-08-14 § 2, rev-5 (Erik Jones, 2026-08-14 "
    "'approve'; methodology v1.50). The bars below are FROZEN; the ONE "
    "execution on the sealed adjudicating cohort adjudicates H3."
)

#: P1 (primary): rank-1 identification rate on the MALICIOUS re-entry
#: population, per scenario, pooled across seeds. Honest rank-1 is reported
#: separately and never enters a P1 denominator.
P1_RANK1_BAR: float = 0.85

#: P2 (guard): wrong-device link rate at the locked τ. Applied TWICE, once per
#: population — see DESIGN_COUNTS for why pooling them hides the honest half.
P2_WRONG_DEVICE_BAR: float = 0.10

#: The instrument this statistic is valid for. Anything else is refused.
REQUIRED_REGISTRY_POLICY: str = RegistryPolicy.IDENTITY_ONLY.value

#: § 2's redefinition, surfaced in every report.
WRONG_DEVICE_GUARD_DEFINITION = (
    "An honest device correctly re-identified as ITSELF is a CORRECT decision. "
    "EXP-056's so-called 'false links' were all self-re-identifications; under "
    "an identity-only instrument that outcome is the instrument working. The "
    "guard therefore counts WRONG-DEVICE links only — a link asserted at the "
    "locked tau to a device that is not the re-entrant's — and is scored "
    "SEPARATELY on the malicious and honest populations, which differ by an "
    "order of magnitude in size. Enforcement consequences of linking honest "
    "devices belong to H4's utility metrics, not here."
)

#: Identity is compared at base-partition granularity (Addendum B1), the same
#: reading the frozen D7 scorer uses.
IDENTITY_COMPARISON = "base_partition"

#: The additive optional column carrying the nearest candidate's identity
#: (`signal_logger.REENTRY_NEAREST_FIELDS`). Its PRESENCE is what makes rank-1
#: determinate on unmatched events; its absence marks a pre-extension corpus.
NEAREST_FIELD = "nearest_logical_id"

#: v1.10 D9 cohort shape: {TGE-only, TGE+FP} x {S3, S4} x 5 seeds.
#:
#: The four denominators are POPULATION-ALIGNED. P1 is scored on
#: the MALICIOUS re-entry population per scenario, in continuity with v1.10 D7's
#: recall population; honest rank-1 is reported separately and never enters a
#: P1 denominator. The wrong-device guard is likewise split, because the two
#: populations differ by an order of magnitude in size: pooled, all 15 honest
#: events could be linked to the WRONG device and the guard would still read
#: 15/175 = 0.086 <= 0.10 and PASS.
#:
#: S3 carries no honest CID-change events by construction, so every honest event
#: is an S4 event; the honest guard's denominator is the same 15 in both cohorts
#: (the three benign-churn reconnects x 5 seeds all sit on ODD partitions).
DESIGN_COUNTS: Dict[Cohort, Dict[str, int]] = {
    Cohort.VALIDATION: {
        "rank1_S3": 180,
        "rank1_S4": 180,
        "wrong_device_link_rate_malicious": 360,
        "wrong_device_link_rate_honest": 15,
    },
    Cohort.ADJUDICATING: {
        "rank1_S3": 80,
        "rank1_S4": 80,
        "wrong_device_link_rate_malicious": 160,
        "wrong_device_link_rate_honest": 15,
    },
}

_TALLY_CLASSES = (
    "rank1_correct",
    "rank1_wrong_device",
    "rank1_no_candidate",
    "rank1_indeterminate",
)


# ===========================================================================
# The instrument assertion
# ===========================================================================

def _require_locked_instrument(path: Path, registry_block: Mapping[str, Any],
                               declared_cohort: str) -> None:
    """FAIL-CLOSED corroboration that the unit scored under THE locked
    instrument for its declared cohort.

    Three exact-equality checks against the hash-verified lock in
    `flowerfl.fingerprint_registry` — not substring or parse-based reads, which
    skip when the custody value is absent or malformed. Absent, `identity`,
    `restored`, wrong-cohort and wrong-estimator values all refuse alike: an
    uncorroborated unit is exactly the boundary where a raw-180 or non-locked
    metric would slip into scoring behind a correct policy declaration. No unresolved custody value survives.
    """
    try:
        cohort = CalibrationCohort(declared_cohort)
    except ValueError as exc:
        raise ScoringError(
            f"{path}: provenance.fp_cohort={declared_cohort!r} is not a "
            f"calibration cohort (expected one of "
            f"{sorted(c.value for c in CalibrationCohort)})."
        ) from exc

    expected_provenance = locked_metric(cohort).provenance
    unit_provenance = registry_block.get("metric_provenance")
    if unit_provenance != expected_provenance:
        raise ScoringError(
            f"{path}: fingerprint_registry.metric_provenance="
            f"{unit_provenance!r} does not EQUAL the locked "
            f"{declared_cohort!r} metric's provenance "
            f"{expected_provenance!r}. The unit cannot be corroborated to "
            "have scored under the locked instrument, and an uncorroborated "
            "unit is refused, never waved through."
        )

    expected_tau = locked_tau(cohort)
    unit_tau = registry_block.get("tau")
    if not isinstance(unit_tau, (int, float)) or float(unit_tau) != expected_tau:
        raise ScoringError(
            f"{path}: fingerprint_registry.tau={unit_tau!r} does not EQUAL "
            f"the locked {declared_cohort!r} tau {expected_tau!r}. A drifted "
            "or absent tau is a different instrument."
        )

    unit_sha = registry_block.get("calibration_artifact_sha256")
    if unit_sha != CALIBRATION_ARTIFACT_SHA256:
        raise ScoringError(
            f"{path}: fingerprint_registry.calibration_artifact_sha256="
            f"{unit_sha!r} does not EQUAL the pinned lock-artifact hash "
            f"{CALIBRATION_ARTIFACT_SHA256!r} (absent counts as a mismatch). "
            "The run must prove it loaded the exact locked calibration "
            "artifact; units from images predating this custody field are "
            "not valid corrected-instrument units."
        )


def load_provenance(paths: Sequence[Path]) -> Dict[str, Any]:
    """Read unit result JSONs and REFUSE anything the statistic does not describe.

    Three things are checked, each against TWO independent records so a
    declaration alone can never carry a unit:

    * **registry policy** — `provenance.fp_registry_policy` (declared) against
      `fingerprint_registry.registry_policy` (what the registry reported at run
      end). A unit that declared the corrected instrument but ran the deployed
      one would otherwise be scored under a statistic that does not describe it.
    * **locked instrument** — `provenance.fp_cohort` must name a calibration
      cohort, and the custody block's `metric_provenance`, `tau` and
      `calibration_artifact_sha256` must EQUAL the locked instrument's values
      for that cohort exactly (`_require_locked_instrument`). The two cohorts
      are different instruments (different τ AND Σ, D9 axis (ii)), and an
      absent or malformed custody value refuses rather than skips.
    * **run identity** — `fingerprint_registry.run_uid`, which is what binds
      this result to the events it produced. Required, and required UNIQUE:
      two units under one run_uid make the binding ambiguous, and an ambiguous
      binding is indistinguishable from a wrong one.
    """
    units: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ScoringError(f"provenance file not found: {path}")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ScoringError(f"{path} is not valid JSON: {exc}") from exc

        provenance = payload.get("provenance") or {}
        registry_block = payload.get("fingerprint_registry") or {}
        declared = provenance.get("fp_registry_policy")
        custody = registry_block.get("registry_policy")

        if declared != REQUIRED_REGISTRY_POLICY:
            raise ScoringError(
                f"{path}: provenance.fp_registry_policy={declared!r} — this "
                f"statistic is valid ONLY for {REQUIRED_REGISTRY_POLICY!r} runs. "
                "The flag-gated pool measures detection multiplied into "
                "identity, which methodology v1.49 ruled an invalid test of H3; "
                "scoring it here would produce a rank-1 number for exactly that "
                "instrument. Score flag-gated runs with the frozen D7 scorer."
            )
        if custody is not None and custody != declared:
            raise ScoringError(
                f"{path}: the declaration and the run disagree — "
                f"provenance.fp_registry_policy={declared!r} but "
                f"fingerprint_registry.registry_policy={custody!r}. The unit did "
                "not execute the instrument it declared."
            )
        if custody is None:
            raise ScoringError(
                f"{path}: no fingerprint_registry.registry_policy in the custody "
                "block — the declaration cannot be corroborated against what the "
                "registry actually ran, and a declaration alone is not evidence."
            )

        declared_cohort = provenance.get("fp_cohort")
        if not declared_cohort:
            raise ScoringError(
                f"{path}: provenance.fp_cohort is {declared_cohort!r} — which "
                "LOCKED tau/Sigma pair scored this unit must be a declared fact "
                "of the run. The validation and adjudicating cohorts are "
                "different instruments (v1.10 D9 axis (ii))."
            )
        _require_locked_instrument(path, registry_block, declared_cohort)

        run_uid = registry_block.get("run_uid")
        if not run_uid:
            raise ScoringError(
                f"{path}: no fingerprint_registry.run_uid — that is the run "
                "identity every reentry_event_key is built from "
                "(`{run_uid}:{round}:{cid}`), and without it this result cannot "
                "be bound to the events it produced. A (scenario, seed) join is "
                "not a run identity: a different run of the SAME design cell "
                "would satisfy it."
            )

        scenario = Path(str(provenance.get("scenario_path", ""))).stem
        if not scenario:
            raise ScoringError(f"{path}: provenance carries no scenario_path")
        units.append({
            "path": str(path),
            "run_uid": str(run_uid),
            "scenario": scenario,
            "seed": payload.get("seed"),
            "fp_cohort": declared_cohort,
            "registry_policy": declared,
            # The custody tau, already verified == the locked constant by
            # _require_locked_instrument; kept on the unit so every EVENT row
            # can be corroborated against it too.
            "tau": float(registry_block["tau"]),
        })

    if not units:
        raise ScoringError("no provenance files were supplied")

    seen: Dict[str, str] = {}
    for unit in units:
        previous = seen.get(unit["run_uid"])
        if previous is not None:
            raise ScoringError(
                f"ambiguous provenance: {previous} and {unit['path']} both "
                f"declare run_uid {_redacted(unit['run_uid'])} (redacted — "
                "run_uids embed launch seeds). One run identity must map to "
                "exactly one result; a duplicate makes every binding through "
                "it unverifiable."
            )
        seen[unit["run_uid"]] = unit["path"]

    return {
        "registry_policy": REQUIRED_REGISTRY_POLICY,
        "n_units": len(units),
        "units": units,
        "run_uids": sorted(seen),
        "covered_cells": sorted(
            {(u["scenario"], u["seed"]) for u in units}, key=repr
        ),
    }


def _redacted(value: Optional[str]) -> str:
    """A stable, non-reversible fingerprint of a run-identifying string.

    run_uids — and the event keys that embed them — carry the launch seed in
    clear text, so on the sealed cohort a refusal that echoed them verbatim
    would print sealed values into logs. Refusals therefore identify rows by
    SHA-256 prefix: enough to locate the offending row offline by hashing
    candidates, never enough to reveal a seed.
    """
    if value is None:
        return "<absent>"
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _assert_provenance_covers(events: Sequence[Mapping[str, Any]],
                              provenance: Mapping[str, Any],
                              cohort: Cohort) -> None:
    """Bind every event to a provenance record by EXACT run_uid, and check the
    cohort matches the one being scored.

    Binding on (scenario, seed) is NOT a run identity. An identity_only result
    JSON and a flag_gated event log from the same design cell would satisfy it,
    and the corrected-instrument verdict would then be computed over the wrong
    instrument's events. `run_uid` is what the events themselves are keyed on,
    so an exact match is the only join that cannot be satisfied by a different
    run.

    The cohort check closes the second substitution: a VALIDATION unit's events,
    filtered to odd partitions, can satisfy the ADJUDICATING design counts
    exactly, and would then be adjudicated under the wrong locked τ and Σ.
    """
    by_run = {u["run_uid"]: u for u in provenance["units"]}

    unbound = sorted({
        str(event.get("run_uid")) for event in events
        if not event.get("run_uid") or str(event["run_uid"]) not in by_run
    })
    if unbound:
        raise ScoringError(
            f"{len(unbound)} distinct run_uid value(s) "
            f"({[_redacted(u) for u in unbound]}) have no matching provenance "
            f"record among the {len(by_run)} supplied. Every event must bind "
            "to the result JSON of the run that emitted it — a (scenario, "
            "seed) match is not a run identity, and an unbindable event is "
            "not a bound one. (run_uids embed launch seeds and are printed "
            "redacted.)"
        )

    mismatched = sorted({
        (unit["path"], unit["fp_cohort"])
        for unit in by_run.values()
        if str(unit["fp_cohort"]) != cohort.value
    })
    if mismatched:
        raise ScoringError(
            f"cohort mismatch: scoring as {cohort.value!r} but unit(s) declare a "
            f"different fp_cohort {mismatched}. The validation and adjudicating "
            "cohorts carry DIFFERENT locked tau/Sigma pairs (v1.10 § 5.1 / D9 "
            "axis (ii)); a validation run's events filtered to the odd hold-out "
            "can satisfy the adjudicating census exactly and would be "
            "adjudicated under the wrong instrument."
        )

    # Binding is more than run_uid presence: the event's own scenario, seed and
    # tau must AGREE with the corroborated unit it binds to. Otherwise a row
    # with a valid run_uid but a swapped scenario/seed reallocates S3/S4
    # outcomes while every matrix count still balances, and a row scored under
    # a different tau than the unit's verified one slips past the unit-level
    # corroboration.
    for event in events:
        unit = by_run[str(event["run_uid"])]
        key = event.get("reentry_event_key")
        # The key CONTRACT is f"{run_uid}:{server_round}:{current_cid}"
        # (flowerfl/signal_logger.build_reentry_event_key). Rebuilding it from
        # the row's own fields catches a relabeled run_uid column: the embedded
        # UID still names the run that actually emitted the row, so a
        # wrong-run corpus cannot be re-badged onto a supplied unit

        try:
            expected_key = build_reentry_event_key(
                str(event["run_uid"]), int(event["server_round"]),
                str(event["current_cid"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScoringError(
                f"event {_redacted(key)} lacks the fields the key contract is "
                f"built from ({type(exc).__name__}) — an event whose identity "
                "cannot be reconstructed cannot be bound."
            ) from exc
        if key != expected_key:
            raise ScoringError(
                f"event key {_redacted(key)} does not equal the contract "
                f"rebuild {_redacted(expected_key)} from the row's own "
                "run_uid/server_round/current_cid. A relabeled run_uid column "
                "with the original UID still embedded in the key is a "
                "wrong-run row wearing a supplied unit's badge; refused. "
                "(keys embed launch seeds and are printed redacted.)"
            )
        if str(event.get("scenario")) != unit["scenario"]:
            raise ScoringError(
                f"event {_redacted(key)} carries scenario "
                f"{event.get('scenario')!r} but "
                f"binds to unit {unit['path']} which ran "
                f"{unit['scenario']!r}. A scenario-swapped row reallocates "
                "per-scenario outcomes while the census still balances; the "
                "corpus is broken and is refused."
            )
        if event.get("seed") != unit["seed"]:
            raise ScoringError(
                f"event {_redacted(key)} does not carry its bound unit's seed "
                f"({unit['path']}). A seed-swapped row is a row from a run "
                "this unit does not describe."
            )
        event_tau = event.get("tau")
        if not isinstance(event_tau, (int, float)) or float(event_tau) != unit["tau"]:
            raise ScoringError(
                f"event {_redacted(key)} records tau={event_tau!r} but its "
                "bound unit "
                f"{unit['path']} scored under the verified locked tau "
                f"{unit['tau']!r}. A row decided at a different tau than the "
                "unit's locked instrument cannot be scored."
            )


#: The v1.10 D9 unit matrix: {S3, S4} × 5 seeds, one FP unit per cell.
_UNITS_PER_SCENARIO: int = 5
_HONEST_EVENTS_PER_S4_UNIT: int = 3
_DESIGN_KEY_BY_SCENARIO = {SCENARIO_S3: "rank1_S3", SCENARIO_S4: "rank1_S4"}

#: The registered seed manifest per cohort: (file, key). The verdict path
#: requires the unit matrix's seed set to EQUAL the manifest's exactly — an
#: internally-consistent matrix on the WRONG seeds (a transcription slip, a
#: stray dev cell) would otherwise satisfy every structural check. Values are compared as sets and NEVER printed:
#: refusal messages carry counts only.
SEED_MANIFESTS: Dict[Cohort, tuple] = {
    Cohort.VALIDATION: (PROJECT_ROOT / "data" / "seeds.json", "dev_seeds"),
    Cohort.ADJUDICATING: (
        PROJECT_ROOT / "data" / "h3_eval_seeds_v2.json", "h3_eval_seeds",
    ),
}


def _manifest_seed_set(cohort: Cohort) -> frozenset:
    path, key = SEED_MANIFESTS[cohort]
    if not Path(path).exists():
        raise ScoringError(
            f"registered seed manifest for cohort {cohort.value!r} not found "
            f"at {path} — the matrix cannot be verified without it."
        )
    try:
        values = json.loads(Path(path).read_text())[key]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ScoringError(
            f"registered seed manifest {path} is unreadable or lacks "
            f"{key!r}: {type(exc).__name__}"
        ) from exc
    if not values or not all(isinstance(v, int) for v in values):
        raise ScoringError(
            f"registered seed manifest {path} key {key!r} is empty or "
            "non-integer — refusing to verify against a malformed manifest."
        )
    return frozenset(values)


def _assert_unit_matrix(provenance: Mapping[str, Any], cohort: Cohort,
                        unit_census: Mapping[str, Mapping[str, int]]) -> None:
    """The verdict-path completeness gate BEYOND the aggregate census.

    The four aggregate denominators alone do not prove the design matrix is
    complete: a missing or partial FP unit can be masked by excess rows from
    another unit while 80/80/160/15 still balances. This gate therefore requires the exact matrix — 5 distinct
    seeds per scenario, the SAME seed set in both scenarios, one unit per
    (scenario, seed) cell — and the exact per-unit scored-event census
    (design/5 malicious events per unit; 3 honest events per S4 unit, 0
    elsewhere). It runs only with `require_design_counts` — diagnostic partial
    reads stay possible and stay loudly labeled.
    """
    units = provenance["units"]
    per_scenario: Dict[str, List[Mapping[str, Any]]] = {
        SCENARIO_S3: [], SCENARIO_S4: [],
    }
    for unit in units:
        if unit["scenario"] not in per_scenario:
            raise ScoringError(
                f"{unit['path']}: scenario {unit['scenario']!r} is not part of "
                f"the H3 design matrix ({sorted(per_scenario)})."
            )
        per_scenario[unit["scenario"]].append(unit)

    cells = {(u["scenario"], u["seed"]) for u in units}
    if len(cells) != len(units):
        raise ScoringError(
            "duplicate (scenario, seed) cell in the supplied provenance: "
            f"{len(units)} units cover only {len(cells)} distinct cells. Two "
            "runs of one design cell make the census ambiguous."
        )
    for scenario, scenario_units in per_scenario.items():
        if len(scenario_units) != _UNITS_PER_SCENARIO:
            raise ScoringError(
                f"unit matrix incomplete: {len(scenario_units)} unit(s) for "
                f"{scenario!r}, design requires exactly {_UNITS_PER_SCENARIO} "
                "(one per seed)."
            )
    seed_sets = {
        scenario: {u["seed"] for u in scenario_units}
        for scenario, scenario_units in per_scenario.items()
    }
    if seed_sets[SCENARIO_S3] != seed_sets[SCENARIO_S4]:
        raise ScoringError(
            "the two scenarios were not run on the same seed set "
            f"({len(seed_sets[SCENARIO_S3] ^ seed_sets[SCENARIO_S4])} seed(s) "
            "differ) — the design matrix is {S3, S4} x the SAME 5 seeds."
        )

    manifest_path, manifest_key = SEED_MANIFESTS[cohort]
    expected_seeds = _manifest_seed_set(cohort)
    unit_seeds = frozenset(seed_sets[SCENARIO_S3])
    if unit_seeds != expected_seeds:
        raise ScoringError(
            f"the unit matrix's seed set does not EQUAL the registered "
            f"manifest {manifest_path} [{manifest_key!r}]: "
            f"{len(unit_seeds ^ expected_seeds)} seed(s) differ "
            f"({len(unit_seeds)} in units, {len(expected_seeds)} in the "
            "manifest; values not printed). An internally-consistent matrix "
            "on the wrong seeds is not the registered design."
        )

    design = DESIGN_COUNTS[cohort]
    for unit in units:
        design_total = design[_DESIGN_KEY_BY_SCENARIO[unit["scenario"]]]
        if design_total % _UNITS_PER_SCENARIO:
            raise ScoringError(
                f"design count {design_total} for {unit['scenario']!r} is not "
                f"divisible by {_UNITS_PER_SCENARIO} units — the per-unit "
                "census expectation is ill-defined; refusing."
            )
        expected = {
            "malicious": design_total // _UNITS_PER_SCENARIO,
            "honest": (
                _HONEST_EVENTS_PER_S4_UNIT
                if unit["scenario"] == SCENARIO_S4 else 0
            ),
        }
        got = unit_census.get(unit["run_uid"], {"malicious": 0, "honest": 0})
        if dict(got) != expected:
            raise ScoringError(
                f"{unit['path']}: scored-event census for this unit is {got} "
                f"but the design requires exactly {expected}. An underfull "
                "unit compensated by an overfull one balances the aggregate "
                "census while the matrix is broken; per-unit counts are what "
                "make that loud."
            )


# ===========================================================================
# Classification
# ===========================================================================

def classify_event(event: Mapping[str, Any]) -> str:
    """`rank1_correct` | `rank1_wrong_device` | `rank1_no_candidate` |
    `rank1_indeterminate`.

    On a MATCHED event the frozen `asserted_parent_logical_id` stays
    authoritative. The two records agree by construction, and reading
    `nearest_*` instead would silently move which field the verdict depends on.

    On an UNMATCHED event the additive `nearest_logical_id`
    (`signal_logger.REENTRY_NEAREST_FIELDS`) makes rank-1 fully determinate:
    present and non-null identifies the nearest candidate, present and null
    means the pool was empty. A corpus written before that extension carries the
    key not at all, and those events stay `rank1_indeterminate` — the honest
    reading, and the reason the legacy path is kept rather than deleted.

    NOTE this classification is about RANK-1 only. `rank1_wrong_device` on an
    unmatched event is NOT a link at τ and must never reach P2 — see
    `_is_wrong_device_link`.
    """
    if bool(event["asserted_match"]):
        parent = event.get("asserted_parent_logical_id")
        if not parent:
            raise ScoringError(
                f"event {_redacted(event.get('reentry_event_key'))} asserts a "
                "match but carries no asserted_parent_logical_id — the row is "
                "not scorable"
            )
        same = partition_of(str(parent)) == partition_of(str(event["gt_logical_id"]))
        return "rank1_correct" if same else "rank1_wrong_device"

    if NEAREST_FIELD in event:
        nearest = event.get(NEAREST_FIELD)
        if not nearest:
            return "rank1_no_candidate"
        same = partition_of(str(nearest)) == partition_of(str(event["gt_logical_id"]))
        return "rank1_correct" if same else "rank1_wrong_device"

    min_d = event.get("min_d")
    if min_d is None:
        return "rank1_no_candidate"
    return "rank1_indeterminate"


def _is_wrong_device_link(event: Mapping[str, Any]) -> bool:
    """P2's numerator: a LINK at the locked τ to the wrong device.

    Deliberately independent of `classify_event`. Since the nearest-candidate
    extension, an unmatched event can be `rank1_wrong_device` — its nearest
    candidate was another device — but it produced NO link, because the distance
    never cleared τ. Counting it here would manufacture guard failures out of
    events the instrument correctly declined to link.
    """
    if not bool(event["asserted_match"]):
        return False
    parent = event.get("asserted_parent_logical_id")
    return partition_of(str(parent)) != partition_of(str(event["gt_logical_id"]))


# ===========================================================================
# Scoring
# ===========================================================================

CI_ALPHA: float = 0.05


def _clopper_pearson(numerator: int, denominator: int, side: str,
                     alpha: float = CI_ALPHA) -> float:
    """One-sided exact (Clopper-Pearson) bound on a binomial rate.

    Exact rather than normal-approximate because the denominators here are
    small — the honest guard is scored on FIFTEEN events, where a Wald interval
    is simply wrong. The bound answers "given k of n, how extreme could the true
    rate be and still have produced this?", which is what a bar of 0.10 on n=15
    needs stated alongside it.

    The closed forms at the edges are exact and are asserted in the tests:
    k = 0 gives an upper bound of 1 - alpha**(1/n); k = n gives a lower bound of
    alpha**(1/n).
    """
    from scipy.stats import beta

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if side == "upper":
        if numerator >= denominator:
            return 1.0
        return float(beta.ppf(1.0 - alpha, numerator + 1, denominator - numerator))
    if side == "lower":
        if numerator <= 0:
            return 0.0
        return float(beta.ppf(alpha, numerator, denominator - numerator + 1))
    raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")


def _confidence(numerator: int, denominator: int, direction: str) -> Dict[str, Any]:
    """The one-sided 95% bound in the CONSERVATIVE direction for this component.

    A guard (`at_most`) fails by being HIGH, so it is bounded ABOVE. A rank-1
    bar (`at_least`) fails by being LOW, so it is bounded BELOW — an upper bound
    on a rank-1 rate would be the anti-conservative direction and would read as
    though the instrument were better than the data supports.

    For the `at_least` components the same statement is ALSO given as an upper
    bound, on the quantity where "upper" is the conservative side: the miss
    rate, 1 - rank1. The two are the same number expressed either way.
    """
    side = "upper" if direction == "at_most" else "lower"
    bound = _clopper_pearson(numerator, denominator, side=side)
    payload: Dict[str, Any] = {
        "method": "clopper_pearson_one_sided_95",
        "alpha": CI_ALPHA,
        "bound_side": side,
        "bound": bound,
        "note": (
            "Exact one-sided 95% bound, taken on the side a failure would come "
            "from: guards are bounded above, rank-1 bars below. Exact rather "
            "than normal-approximate because the denominators are small."
        ),
    }
    if side == "lower":
        payload["error_rate_upper"] = 1.0 - bound
        payload["error_rate_note"] = (
            "the same bound as an UPPER bound on the miss rate (1 - rank1)"
        )
    return payload


def _component(label: str, scenario: Optional[str], numerator: int,
               denominator: int, indeterminate: int, bar: float,
               direction: str, resolution_note: Optional[str] = None) -> Dict[str, Any]:
    """One component, with the file's INCONCLUSIVE conventions.

    A component is INCONCLUSIVE when its denominator is zero OR when any of its
    events are rank-1 indeterminate. Both mean the same thing operationally: the
    corpus does not support a pass/fail claim, and neither direction may be
    assumed.
    """
    if numerator < 0 or numerator > max(denominator, 0):
        raise ScoringError(
            f"component {label!r}: numerator {numerator} outside "
            f"[0, {denominator}] — the population accounting upstream is "
            "broken, and an impossible rate must refuse, not report."
        )
    if denominator == 0 or indeterminate:
        return {
            "label": label,
            "scenario": scenario,
            "numerator": numerator,
            "denominator": denominator,
            "indeterminate": indeterminate,
            "value": None,
            "bar": bar,
            "direction": direction,
            "ci95": {
                "method": "clopper_pearson_one_sided_95",
                "alpha": CI_ALPHA,
                "bound_side": "upper" if direction == "at_most" else "lower",
                "bound": None,
                "note": "not computable: the component is INCONCLUSIVE",
            },
            "resolution_note": resolution_note,
            "status": "INCONCLUSIVE",
            "inconclusive_reason": (
                "zero denominator" if denominator == 0
                else f"{indeterminate} event(s) whose nearest candidate is not "
                     "recorded (see the module docstring: rank-1 is threshold-"
                     "free, the row records the parent only when the match fired)"
            ),
        }
    value = numerator / denominator
    passed = value >= bar if direction == "at_least" else value <= bar
    return {
        "label": label,
        "scenario": scenario,
        "numerator": numerator,
        "denominator": denominator,
        "indeterminate": 0,
        "value": value,
        "bar": bar,
        "direction": direction,
        "ci95": _confidence(numerator, denominator, direction),
        "resolution_note": resolution_note,
        "status": "PASS" if passed else "FAIL",
    }


def score_events(
    events: Sequence[Mapping[str, Any]],
    cohort: Cohort,
    provenance: Mapping[str, Any],
    require_design_counts: bool = True,
) -> Dict[str, Any]:
    """Apply the RATIFIED § 2 statistic and return the full report."""
    cohort = Cohort(cohort)
    _assert_provenance_covers(events, provenance, cohort)

    scored = [
        event for event in events
        if cohort is Cohort.VALIDATION
        or is_holdout_partition(str(event["gt_logical_id"]))
    ]

    tallies = {
        SCENARIO_S3: {name: 0 for name in _TALLY_CLASSES},
        SCENARIO_S4: {name: 0 for name in _TALLY_CLASSES},
    }
    # P1 is scored on the MALICIOUS population per scenario; honest rank-1 is
    # tallied separately and reported, never pooled into a P1 denominator.
    # Both populations are counted DIRECTLY in this loop — deriving the
    # malicious counts by subtracting honest totals from a pooled tally lets a
    # misplaced honest row inflate a malicious numerator past its denominator

    populations = {SCENARIO_S3: 0, SCENARIO_S4: 0}
    malicious_correct = {SCENARIO_S3: 0, SCENARIO_S4: 0}
    malicious_indeterminate = {SCENARIO_S3: 0, SCENARIO_S4: 0}
    honest_population = 0
    honest_rank1_correct = 0
    honest_rank1_indeterminate = 0
    honest_wrong_device_links = 0
    per_device: Dict[int, Dict[str, int]] = {}
    distances: List[float] = []
    wrong_device_links = 0
    with_nearest_field = 0
    unit_census: Dict[str, Dict[str, int]] = {}

    for event in scored:
        scenario = str(event["scenario"])
        if scenario not in tallies:
            raise ScoringError(
                f"unexpected scenario {scenario!r} — H3 is scored on "
                f"{SCENARIO_S3} and {SCENARIO_S4} only"
            )
        outcome = classify_event(event)
        tallies[scenario][outcome] += 1
        malicious = bool(event["gt_is_malicious"])
        unit_bucket = unit_census.setdefault(
            str(event.get("run_uid")), {"malicious": 0, "honest": 0}
        )
        if malicious:
            populations[scenario] += 1
            unit_bucket["malicious"] += 1
            if outcome == "rank1_correct":
                malicious_correct[scenario] += 1
            elif outcome == "rank1_indeterminate":
                malicious_indeterminate[scenario] += 1
        else:
            if scenario != SCENARIO_S4:
                raise ScoringError(
                    f"honest re-entry event "
                    f"{_redacted(event.get('reentry_event_key'))} carries scenario "
                    f"{scenario!r} — S3 carries no honest CID-change events by "
                    "construction (every honest event is an S4 benign-churn "
                    "reconnect), so an honest row outside S4 means the corpus "
                    "labels are broken. Refusing rather than tallying it."
                )
            honest_population += 1
            unit_bucket["honest"] += 1
            if outcome == "rank1_correct":
                honest_rank1_correct += 1
            elif outcome == "rank1_indeterminate":
                honest_rank1_indeterminate += 1
        if _is_wrong_device_link(event):
            if malicious:
                wrong_device_links += 1
            else:
                honest_wrong_device_links += 1
        if NEAREST_FIELD in event:
            with_nearest_field += 1

        device = partition_of(str(event["gt_logical_id"]))
        bucket = per_device.setdefault(
            device, {"correct": 0, "events": 0, "indeterminate": 0}
        )
        bucket["events"] += 1
        if outcome == "rank1_correct":
            bucket["correct"] += 1
        elif outcome == "rank1_indeterminate":
            bucket["indeterminate"] += 1

        min_d = event.get("min_d")
        if min_d is not None:
            distances.append(float(min_d))

    malicious_events = sum(populations.values())
    total_indeterminate = sum(t["rank1_indeterminate"] for t in tallies.values())

    if require_design_counts:
        _assert_unit_matrix(provenance, cohort, unit_census)

    components = {
        "rank1_S3": _component(
            "rank-1 identification, malicious re-entries (S3)", SCENARIO_S3,
            malicious_correct[SCENARIO_S3], populations[SCENARIO_S3],
            malicious_indeterminate[SCENARIO_S3],
            P1_RANK1_BAR, "at_least",
        ),
        "rank1_S4": _component(
            "rank-1 identification, malicious re-entries (S4)", SCENARIO_S4,
            malicious_correct[SCENARIO_S4], populations[SCENARIO_S4],
            malicious_indeterminate[SCENARIO_S4],
            P1_RANK1_BAR, "at_least",
        ),
        # Both guards are threshold-BASED (links at the locked tau), so an
        # unrecorded nearest candidate does not make them indeterminate: an
        # event that did not link simply did not link.
        "wrong_device_link_rate_malicious": _component(
            "wrong-device link rate at the locked tau (malicious events)", None,
            wrong_device_links, malicious_events, 0,
            P2_WRONG_DEVICE_BAR, "at_most",
        ),
        "wrong_device_link_rate_honest": _component(
            "wrong-device link rate at the locked tau (honest events)", None,
            honest_wrong_device_links, honest_population, 0,
            P2_WRONG_DEVICE_BAR, "at_most",
            resolution_note=(
                f"scored on {honest_population} honest event(s); at the design "
                "count of 15 the rate resolves only in steps of 1/15 = 0.0667, "
                f"so a bar of {P2_WRONG_DEVICE_BAR} admits at most 1 wrong-device "
                "link. The exact one-sided bound in `ci95` is what makes that "
                "resolution honest — 0/15 does NOT mean the true rate is zero."
            ),
        ),
    }

    realised = {name: comp["denominator"] for name, comp in components.items()}
    design = DESIGN_COUNTS[cohort]
    deltas = {
        name: realised[name] - design[name]
        for name in design if realised[name] != design[name]
    }
    if deltas and require_design_counts:
        raise ScoringError(
            f"realised event census does not equal the design counts for cohort "
            f"'{cohort.value}': realised {realised} vs design {design} "
            f"(deltas {deltas}). The census gate requires an exact match."
        )

    statuses = {comp["status"] for comp in components.values()}
    if "INCONCLUSIVE" in statuses:
        conjunction = "INCONCLUSIVE"
    elif "FAIL" in statuses:
        conjunction = "FAIL"
    else:
        conjunction = "PASS"

    # A VERDICT exists only on the adjudicating cohort WITH the census +
    # matrix gates enforced. A diagnostic run (--allow-partial-census) still
    # reports the component conjunction, but its artifact must be structurally
    # DISTINGUISHABLE from a sealed verdict — a terminal-only warning is not
    # custody. The conjunction is therefore
    # always present as `component_conjunction`, and `verdict` is non-null
    # only on the fully-gated adjudicating path.
    informs_only = cohort is Cohort.VALIDATION
    diagnostic = not require_design_counts
    if informs_only:
        verdict = None
        verdict_status = VALIDATION_VERDICT_STATUS
        verdict_withheld_reason = VALIDATION_VERDICT_WITHHELD_REASON
        if diagnostic:
            verdict_status += " — DIAGNOSTIC (census gate disabled)"
            verdict_withheld_reason += (
                " ADDITIONALLY the census and unit-matrix gates were "
                "DISABLED (--allow-partial-census): this artifact is a "
                "diagnostic read even as validation material."
            )
    elif diagnostic:
        verdict = None
        verdict_status = "DIAGNOSTIC — VERDICT WITHHELD"
        verdict_withheld_reason = (
            "the census and unit-matrix gates were DISABLED "
            "(--allow-partial-census); a partial or malformed corpus can "
            "satisfy every remaining check, so no pass/fail claim is made. "
            "The component conjunction is reported for diagnosis only."
        )
    else:
        verdict = conjunction
        verdict_status = conjunction
        verdict_withheld_reason = None
    return {
        "instrument": "identity_only",
        "status": STATUS,
        "registry_policy": provenance["registry_policy"],
        "provenance_units": provenance["units"],
        "cohort": cohort.value,
        "identity_comparison": IDENTITY_COMPARISON,
        "wrong_device_guard_definition": WRONG_DEVICE_GUARD_DEFINITION,
        "arm_rule": f"TGE+FP arm only, read from {H3_ARM_FIELD!r} (v1.10 § 5.1)",
        "n_events_loaded": len(events),
        "n_events_scored": len(scored),
        "components": components,
        "informs_only": informs_only,
        "census_gate_enforced": bool(require_design_counts),
        "diagnostic_partial_census": diagnostic,
        "component_conjunction": conjunction,
        "verdict": verdict,
        "verdict_status": verdict_status,
        "verdict_withheld_reason": verdict_withheld_reason,
        "verdict_rule": (
            f"RATIFIED (rev-5, methodology v1.50): PASS iff rank1_S3 >= "
            f"{P1_RANK1_BAR} AND rank1_S4 >= "
            f"{P1_RANK1_BAR} AND wrong_device_link_rate_malicious <= "
            f"{P2_WRONG_DEVICE_BAR} AND wrong_device_link_rate_honest <= "
            f"{P2_WRONG_DEVICE_BAR} — the conjunction of all FOUR. The two "
            "guards are separate because the populations differ by an order of "
            "magnitude: pooled, every honest event could be linked to the wrong "
            "device and the guard would still read 15/175 = 0.086 and pass. "
            "A zero denominator OR any rank-1 indeterminate event makes that "
            "component INCONCLUSIVE and gates the verdict — never an auto-pass. "
            "Bars FROZEN at ratification (rev-5, 2026-08-14)."
        ),
        "design_counts": design,
        "realised_counts": realised,
        "counts_match_design": not deltas,
        "design_count_deltas": deltas,
        "event_tallies": tallies,
        "populations": {
            "malicious_per_scenario": populations,
            "malicious_total": malicious_events,
            "honest_total": honest_population,
        },
        "n_indeterminate": total_indeterminate,
        "nearest_candidate_coverage": {
            "field": NEAREST_FIELD,
            "n_with_nearest_field": with_nearest_field,
            "n_events_scored": len(scored),
            "complete": bool(scored) and with_nearest_field == len(scored),
            "note": (
                "Rank-1 is threshold-free, so it is fully determinate only on "
                "events carrying the nearest-candidate column. A corpus written "
                "before that extension reports complete=false, and any of its "
                "unmatched events with a finite min_d are INCONCLUSIVE rather "
                "than assumed either way."
            ),
        },
        "reported_only": _reported_only(
            per_device, distances,
            honest_rank1_correct, honest_population, honest_rank1_indeterminate,
        ),
        "not_derivable": _not_derivable(),
    }


def _percentiles(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "status": "INCONCLUSIVE"}
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "note": (
            "over every event carrying a finite min_d; events whose candidate "
            "pool was empty record a null distance and are excluded"
        ),
    }


def _reported_only(per_device: Mapping[int, Mapping[str, int]],
                   distances: Sequence[float],
                   honest_correct: int, honest_total: int,
                   honest_indeterminate: int) -> Dict[str, Any]:
    """Context blocks. REPORTED, never a bar."""
    return {
        # Honest rank-1 is a REPORTED line, never a P1 component: P1 is scored
        # on the malicious population, in continuity with v1.10 D7's recall
        # population, and pooling 15 honest events into an 80-event denominator
        # would let either population mask the other.
        "honest_rank1": {
            "numerator": honest_correct,
            "denominator": honest_total,
            "indeterminate": honest_indeterminate,
            "rate": (
                None if honest_indeterminate or not honest_total
                else honest_correct / honest_total
            ),
            "note": (
                "an honest device re-identified as ITSELF is a CORRECT "
                "decision (§ 2). Reported for context; it gates nothing."
            ),
        },
        "per_device_rank1": {
            str(device): {
                "correct": counts["correct"],
                "events": counts["events"],
                "indeterminate": counts["indeterminate"],
                "rate": (
                    None if counts["indeterminate"] or not counts["events"]
                    else counts["correct"] / counts["events"]
                ),
            }
            for device, counts in sorted(per_device.items())
        },
        "per_device_note": (
            "reported context, never a bar. A device with any indeterminate "
            "event reports rate=None rather than a rate computed over the "
            "answerable subset, which would flatter the instrument."
        ),
        "min_d_percentiles": _percentiles(distances),
    }


def _not_derivable() -> Dict[str, Any]:
    """Quantities § 2 lists that this corpus CANNOT support.

    Emitted explicitly, with the reason, rather than omitted or approximated.
    An absent key reads as an oversight; a fabricated one is worse.
    """
    unrecorded = (
        "the schema-v5 re-entry row records only the NEAREST candidate, and only "
        "when the match fired (min_d <= tau). Ranks beyond the first are never "
        "written, so no CMC curve past rank 1 exists in this corpus."
    )
    return {
        "cmc_rank_2_3": {
            "value": None,
            "reason": unrecorded,
            "remedy": (
                "would require the registry to log the ordered candidate list "
                "(or at least the top-3) per re-entry event — a registry and "
                "schema change, not a scoring change."
            ),
        },
        "full_cmc_curve": {
            "value": None,
            "reason": unrecorded,
            "remedy": "as above",
        },
        "distance_separability_census": {
            "value": None,
            "reason": (
                "requires the distance to the nearest SAME-device candidate and "
                "to the nearest OTHER-device candidate separately; the row "
                "carries a single min_d and no per-candidate breakdown."
            ),
            "remedy": (
                "computable offline from the custody observation_log by "
                "re-running the matcher, which is a replay, not a read of the "
                "logged assertions — deliberately out of scope for a scorer "
                "whose integrity claim is that it scores what the registry "
                "decided."
            ),
        },
    }


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the CORRECTED H3 identity instrument (RATIFIED, amendment "
            "2026-08-14 § 2 rev-5, methodology v1.50): rank-1 identification "
            "per scenario and the "
            "wrong-device link rate at the locked tau. Requires identity_only "
            "runs and refuses anything else."
        )
    )
    parser.add_argument("--events", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--provenance", nargs="+", required=True, type=Path,
        help=(
            "unit result JSONs. Every (scenario, seed) present in the events "
            "must be covered, and every unit must declare AND have run "
            "fp_registry_policy=identity_only."
        ),
    )
    parser.add_argument(
        "--cohort", required=True, choices=[c.value for c in Cohort],
        help="validation (all devices, informs only) or adjudicating (odd hold-out)",
    )
    parser.add_argument("--arm", default=FP_ARM_DEFAULT, help="the +FP arm token")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-partial-census", action="store_true",
        help=(
            "DIAGNOSTIC ONLY: score a corpus whose event census differs from the "
            "design counts. Never use this for a verdict."
        ),
    )
    args = parser.parse_args(argv)

    try:
        provenance = load_provenance(args.provenance)
        events = load_events(args.events, arm=args.arm)
        report = score_events(
            events,
            Cohort(args.cohort),
            provenance,
            require_design_counts=not args.allow_partial_census,
        )
    except ScoringError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"instrument: identity_only ({report['n_events_scored']} events scored)")
    for name, component in report["components"].items():
        value = component["value"]
        rendered = "n/a" if value is None else f"{value:.4f}"
        bound = component["ci95"]["bound"]
        side = component["ci95"]["bound_side"]
        ci = "n/a" if bound is None else f"{side[0]}95={bound:.4f}"
        print(
            f"{name:>34}  {rendered:>8}  "
            f"({component['numerator']}/{component['denominator']})  "
            f"bar {component['direction']} {component['bar']}  "
            f"{ci:>12}  {component['status']}"
        )
    if report["n_indeterminate"]:
        print(
            f"\n{report['n_indeterminate']} event(s) are rank-1 INDETERMINATE: a "
            "nearest candidate existed but the row records its identity only "
            "when the match fired. Rank-1 is threshold-free and cannot be read "
            "for those events."
        )
    honest_guard = report["components"]["wrong_device_link_rate_honest"]
    if honest_guard.get("resolution_note"):
        print(f"\n{honest_guard['resolution_note']}")
    print(f"\nVERDICT ({report['cohort']}): {report['verdict_status']}")
    if report["verdict_withheld_reason"]:
        print(report["verdict_withheld_reason"])
    if report["diagnostic_partial_census"]:
        print(
            f"component conjunction (diagnosis only, NOT a verdict): "
            f"{report['component_conjunction']}"
        )
    print(f"\nSTATUS: {STATUS}")
    if args.allow_partial_census:
        print("WARNING: --allow-partial-census was set; this is NOT a verdict run.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

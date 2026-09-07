#!/usr/bin/env python3
"""THE frozen one-shot H4 composition scorer.

STATUS: FROZEN pre-data — pre-registration 2026-08-16 (RATIFIED, methodology
v1.51) as amended by erratum A 2026-08-17 (RULED, v1.52; E5 fixes this
scorer's freeze point: committed + reviewed BEFORE the acceptance
smoke's EXP doc seals and before any sealed-fleet unit launches).

WHAT ADJUDICATES (§ 4.1 as amended by E1)
-----------------------------------------
Per (arm, scenario, seed): `acc_final5` = mean sealed-test accuracy over the
final 5 rounds. Then

    degradation(arm, SX, seed) = acc_final5(arm, C0, seed)
                                 - acc_final5(arm, SX, seed)      SX in S1..S4
    degradation_reduction(SX, seed) = degradation(krum, SX, seed)
                                 - degradation(h2p_fp_krum, SX, seed)

GATE per scenario: median over the 10 sealed seed-pairs >= 0.05 (5 pp,
inclusive) AND exact paired two-sided Wilcoxon p <= 0.05 (inclusive).
Verdict = CONFIRMED iff ALL FOUR of S1-S4 pass both; FALSIFIED if any
scenario fails either; INCONCLUSIVE only if a component cannot be computed
at all. Custody or census failure is a REFUSAL (exit 2) — never a silent
pass, never a partial read. The component table is always emitted.

The primary contrast is arm 1 (`h2p_fp_krum`) vs arm 2 (`krum`) and NOTHING
ELSE: every other block in the output is REPORTED, NON-GATING, and cannot
rescue, overturn, soften or strengthen the primary (§ 2 no-rescue rule,
extended verbatim to arm 9 by erratum-A E2).

FAIL-CLOSED ARCHITECTURE (mirrors `analyze_h3_identity.py`)
-----------------------------------------------------------
* **Locked-instrument corroboration by EXACT equality** — never substring,
  never parse-and-skip. Absent custody = refusal, not skip.
* **Census gate**: 9 arms x 6 scenarios (C0, S0-S4) x the 10 sealed seeds =
  540 cells, one unit per cell — or a matrix reduced ONLY by the pre-stated
  drop order 8-4-7-9-6-5-3 (all-or-nothing per arm, prefix-only, disclosed;
  arms 1 and 2 never drop; C0 required for every surviving arm).
* **Seed-set gate** vs `data/seeds.json` `confirmatory_seeds`, count-only
  refusal messages.
* **Redaction, stricter than H3**: the sealed seeds are consumed for the
  first time by this fleet, so NO seed value appears in any log, refusal or
  output. run_uids embed launch seeds -> SHA-256 prefix redaction; launch
  tooling embeds seeds in unit FILENAMES -> paths are redacted the same way;
  seeds appear in outputs only as `seed_ordinal` (index into the sorted
  sealed manifest).
* **Diagnostic verdict withholding**: `--allow-partial-census` (and any
  `--force-diagnostic` re-read) reports the component conjunction for
  diagnosis but sets `verdict` null with status
  "DIAGNOSTIC — VERDICT WITHHELD".
* **ONE execution, inviolable terminal artifact**: the adjudicating read
  runs once on the completed census. A second run against an existing
  output path is refused; `--force-diagnostic` permits a re-read but only
  as a diagnostic (withheld-verdict) artifact, and it NEVER writes over
  the terminal artifact — a protected path routes the diagnostic to a
  clearly-marked `.diagnostic` sibling, and if that too is protected the
  run refuses. Only a file provably holding a previous diagnostic
  artifact of this scorer is ever overwritten.

CUSTODY MATRIX (per unit; erratum-A E4 + spec § 5 + BUILD_CONTRACT)
-------------------------------------------------------------------
* `provenance.eval_split == "sealed_test"` and
  `provenance.eval_split_manifest_sha256` == sha256 of the committed
  `data/val_test_split_manifest.json` BYTES, equal across all units.
* Detector-bearing arms (the five `h2p_*` arms):
  `provenance.serving_bundle_sha256` == the ACTIVE serving-bundle pin
  (`h4_scoring_lib.required_serving_bundle_sha256()` — the erratum-B
  bundle_v2 sha, source `data/h4_serving/manifest_v2.json`; while the pin
  is the TBD_BUNDLE_V2 sentinel the scorer REFUSES). All other arms: null
  (never a string, never absent — a missing key is a refusal).
* `provenance.run_uid` present, non-empty, and globally UNIQUE across the
  census on EVERY arm (director ruling 2026-08-17: universal run-identity
  export; FP-bearing arms additionally keep the registry-block copy and
  the two must be EQUAL). Missing, empty, duplicated, or disagreeing
  run identity = refusal, always echoed redacted.
* FP-bearing arms (`h2p_fp_krum`, `h2p_fp`, `h2p_fp_ts`, `krum_tge_fp`):
  `provenance.fp_registry_policy == "flag_gated"`, corroborated against the
  run's own `fingerprint_registry.registry_policy` (a declaration alone is
  not evidence), plus the § 5 locked-instrument equalities —
  `fp_cohort == "validation"`, the locked validation tau, metric provenance
  and calibration-artifact sha — and a unique `run_uid`. Non-FP arms:
  policy null/absent.

§ 6 DIAGNOSTICS are aggregated REPORTED and NON-GATING, with null-vs-zero
preserved (null = layer absent from the arm; nulls are not findings).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.fingerprint_registry import (  # noqa: E402
    CALIBRATION_ARTIFACT_SHA256,
    CalibrationCohort,
    locked_metric,
    locked_tau,
)
from scripts import h4_scoring_lib as lib  # noqa: E402
from scripts.h4_scoring_lib import (  # noqa: E402
    ARM_BY_NUMBER,
    ARM_TOKENS,
    DETECTOR_ARMS,
    DROP_ORDER,
    FP_ARMS,
    N_SEEDS,
    NEVER_DROP,
    REQUIRED_EVAL_SPLIT,
    SCENARIOS,
    STATUS,
    ScoringError,
    acc_final5,
    f1_final5,
    final_round_accuracy,
    redacted,
)
from scripts.h4_report_lib import (  # noqa: E402
    aggregate_diagnostics,
    artifact_class,
    build_primary,
    build_secondaries,
    diagnostic_sibling,
    write_memo,
)

#: § 5 froze the v2 VALIDATION-cohort lock for H4 (full device population).
REQUIRED_FP_COHORT = CalibrationCohort.VALIDATION.value
REQUIRED_FP_POLICY = "flag_gated"

_ARM_NUMBER = {token: number for number, token in ARM_BY_NUMBER.items()}

VERDICT_WITHHELD_STATUS = "DIAGNOSTIC — VERDICT WITHHELD"

VERDICT_RULE = (
    "FROZEN (v1.51 § 4.1 as amended by erratum-A E1, v1.52): per scenario "
    "SX in {S1..S4}, degradation_reduction(SX, seed) = "
    "[acc_final5(krum, C0) - acc_final5(krum, SX)] - "
    "[acc_final5(h2p_fp_krum, C0) - acc_final5(h2p_fp_krum, SX)]; the "
    "scenario passes iff median over the 10 sealed seed-pairs >= 0.05 "
    "(inclusive) AND exact paired two-sided Wilcoxon p <= 0.05 (inclusive). "
    "CONFIRMED iff ALL FOUR scenarios pass both; FALSIFIED if any scenario "
    "fails either; INCONCLUSIVE only when a component cannot be computed. "
    "Custody or census failure is a refusal, never a silent pass. No other "
    "arm or block can rescue, overturn, soften or strengthen this contrast."
)

ONE_EXECUTION_WORDING = (
    "THE ONE adjudicating execution has already written this artifact. The "
    "H4 read runs ONCE on the completed census (§ 8): its output is "
    "terminal, and rerunning against the same path would re-roll a sealed "
    "verdict. If you need a diagnostic re-read, pass --force-diagnostic — "
    "the re-read's verdict is WITHHELD and its output is ROUTED to a "
    "'.diagnostic' sibling path; the terminal artifact itself is inviolable "
    "and is never overwritten by any invocation of this scorer."
)


# ===========================================================================
# Unit loading + custody corroboration
# ===========================================================================

def _unit_ref(arm: str, scenario: str, path: Path) -> str:
    """A refusal-safe unit label: arm and scenario are design coordinates,
    the path is redacted because launch tooling embeds the seed in
    filenames. Hash candidate path strings to locate the file offline."""
    return f"unit[arm={arm}, scenario={scenario}, path={redacted(str(path))}]"


def _validate_unit(path: Path, payload: Mapping[str, Any],
                   expected_split_sha: str) -> Dict[str, Any]:
    """One unit's full custody corroboration. Every check is an EXACT
    equality; every absence is a refusal, never a skip."""
    config = payload.get("config")
    if not isinstance(config, str) or not config:
        raise ScoringError(
            f"unit {redacted(str(path))}: no 'config' label — the arm token "
            "cannot be derived, and an unidentifiable unit is refused."
        )
    arm = config.replace("+", "_").lower()
    if arm not in ARM_TOKENS:
        raise ScoringError(
            f"unit {redacted(str(path))}: config {config!r} does not map to "
            f"a pre-registered arm token (the § 2 matrix as amended: "
            f"{sorted(ARM_TOKENS)}). An off-matrix unit cannot be scored."
        )

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ScoringError(
            f"unit {redacted(str(path))} (arm {arm}): no provenance block — "
            "custody cannot be corroborated, and an uncorroborated unit is "
            "refused, never waved through."
        )

    scenario_path = provenance.get("scenario_path")
    if not scenario_path:
        raise ScoringError(
            f"unit {redacted(str(path))} (arm {arm}): provenance carries no "
            "scenario_path — the scenario coordinate cannot be identified."
        )
    stem = Path(str(scenario_path)).stem
    scenario = stem.split("_")[0]
    if scenario not in SCENARIOS:
        raise ScoringError(
            f"unit {redacted(str(path))} (arm {arm}): scenario_path stem "
            f"{stem!r} does not begin with a pre-registered scenario code "
            f"({list(SCENARIOS)})."
        )
    ref = _unit_ref(arm, scenario, path)

    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ScoringError(
            f"{ref}: 'seed' is absent or non-integer (value not printed) — "
            "the census is seed-anchored and a seedless unit cannot be "
            "placed in it."
        )

    # --- universal run identity (director ruling 2026-08-17) ---------------
    run_uid = provenance.get("run_uid")
    if not isinstance(run_uid, str) or not run_uid:
        raise ScoringError(
            f"{ref}: provenance.run_uid is absent or empty — EVERY census "
            "unit must carry its run identity (director ruling 2026-08-17: "
            "universal run_uid export on all arms); without it the unit "
            "cannot be bound to the run that produced it. (run_uids embed "
            "launch seeds and are never echoed; the unit is identified by "
            "its redacted ref.)"
        )

    # --- erratum-A E4: sealed-test evaluation population -------------------
    if "eval_split" not in provenance:
        raise ScoringError(
            f"{ref}: provenance.eval_split is MISSING — which evaluation "
            "population scored this unit must be a declared fact of the run "
            "(erratum-A E4). A missing field is a refusal, not a skip."
        )
    if provenance["eval_split"] != REQUIRED_EVAL_SPLIT:
        raise ScoringError(
            f"{ref}: provenance.eval_split={provenance['eval_split']!r} — "
            f"H4 units evaluate on the sealed manifest's test indices "
            f"({REQUIRED_EVAL_SPLIT!r}, erratum-A E4). A legacy-holdout "
            "trajectory is a different evaluation population and is refused."
        )
    if "eval_split_manifest_sha256" not in provenance:
        raise ScoringError(
            f"{ref}: provenance.eval_split_manifest_sha256 is MISSING — the "
            "unit cannot prove WHICH split manifest its evaluator read "
            "(erratum-A E4)."
        )
    unit_split_sha = provenance["eval_split_manifest_sha256"]
    if unit_split_sha != expected_split_sha:
        raise ScoringError(
            f"{ref}: eval_split_manifest_sha256={unit_split_sha!r} does not "
            f"EQUAL the committed manifest's byte sha256 "
            f"{expected_split_sha!r} (data/val_test_split_manifest.json, "
            "locked 2026-05-14, never re-cut). A drifted or absent split is "
            "a different sealed test."
        )

    # --- § 7 item 1: the frozen serving bundle -----------------------------
    if "serving_bundle_sha256" not in provenance:
        raise ScoringError(
            f"{ref}: provenance.serving_bundle_sha256 key is MISSING — the "
            "custody contract requires the field on EVERY arm (null is the "
            "declared value for non-detector arms; a missing field is a "
            "refusal, not a skip)."
        )
    bundle = provenance["serving_bundle_sha256"]
    if arm in DETECTOR_ARMS:
        # Erratum-B pin swap (v1.53): the ACTIVE pin is resolved through the
        # sentinel-guarded accessor — while bundle v2 is unpinned this refuses
        # BEFORE any equality, so a fleet can never be custody-checked against
        # a placeholder (dynamic lib. lookup, monkeypatch-visible in tests).
        expected_bundle = lib.required_serving_bundle_sha256()
        if bundle != expected_bundle:
            raise ScoringError(
                f"{ref}: serving_bundle_sha256={bundle!r} does not EQUAL "
                f"the pinned serving-bundle sha {expected_bundle!r} "
                "(erratum B: sha256 of data/h4_serving/manifest_v2.json "
                "bytes). A drifted or absent bundle is a different "
                "instrument; the retired v1 pin interprets EXP-061 only."
            )
    elif bundle is not None:
        raise ScoringError(
            f"{ref}: arm {arm!r} carries no online H2' detector but "
            f"declares serving_bundle_sha256={bundle!r} — the custody "
            "contract requires null (not empty string, not a value) on "
            "non-detector arms; the unit did not run the arm it declares."
        )

    # --- § 5: the FLAG_GATED identity layer + locked instrument ------------
    if arm in FP_ARMS:
        declared_policy = provenance.get("fp_registry_policy")
        if "fp_registry_policy" not in provenance or declared_policy != REQUIRED_FP_POLICY:
            raise ScoringError(
                f"{ref}: provenance.fp_registry_policy="
                f"{declared_policy!r} — the H4 identity layer is frozen at "
                f"{REQUIRED_FP_POLICY!r} (§ 5); anything else (or a missing "
                "field) is a different design point and is refused."
            )
        registry = payload.get("fingerprint_registry")
        if not isinstance(registry, Mapping) or not registry:
            raise ScoringError(
                f"{ref}: no fingerprint_registry custody block — the policy "
                "declaration cannot be corroborated against what the "
                "registry actually ran, and a declaration alone is not "
                "evidence (H3 discipline; spec § 5)."
            )
        custody_policy = registry.get("registry_policy")
        if custody_policy != REQUIRED_FP_POLICY:
            raise ScoringError(
                f"{ref}: the declaration and the run disagree — provenance "
                f"declares {REQUIRED_FP_POLICY!r} but "
                f"fingerprint_registry.registry_policy={custody_policy!r}. "
                "The unit did not execute the instrument it declared."
            )
        declared_cohort = provenance.get("fp_cohort")
        if declared_cohort != REQUIRED_FP_COHORT:
            raise ScoringError(
                f"{ref}: provenance.fp_cohort={declared_cohort!r} — H4 § 5 "
                f"froze the v2 {REQUIRED_FP_COHORT!r}-cohort lock (the full "
                "device population is that cohort's calibration population); "
                "a different cohort is a different locked tau/Sigma."
            )
        cohort = CalibrationCohort(REQUIRED_FP_COHORT)
        expected_provenance = locked_metric(cohort).provenance
        unit_metric = registry.get("metric_provenance")
        if unit_metric != expected_provenance:
            raise ScoringError(
                f"{ref}: fingerprint_registry.metric_provenance="
                f"{unit_metric!r} does not EQUAL the locked "
                f"{REQUIRED_FP_COHORT!r} metric's provenance "
                f"{expected_provenance!r} — the unit cannot be corroborated "
                "to have scored under the locked instrument."
            )
        expected_tau = locked_tau(cohort)
        unit_tau = registry.get("tau")
        if (not isinstance(unit_tau, (int, float))
                or isinstance(unit_tau, bool)
                or float(unit_tau) != expected_tau):
            raise ScoringError(
                f"{ref}: fingerprint_registry.tau={unit_tau!r} does not "
                f"EQUAL the locked {REQUIRED_FP_COHORT!r} tau "
                f"{expected_tau!r}. A drifted or absent tau is a different "
                "instrument."
            )
        unit_sha = registry.get("calibration_artifact_sha256")
        if unit_sha != CALIBRATION_ARTIFACT_SHA256:
            raise ScoringError(
                f"{ref}: fingerprint_registry.calibration_artifact_sha256="
                f"{unit_sha!r} does not EQUAL the pinned lock-artifact hash "
                f"{CALIBRATION_ARTIFACT_SHA256!r} (absent counts as a "
                "mismatch). The run must prove it loaded the exact locked "
                "calibration artifact."
            )
        run_uid_value = registry.get("run_uid")
        if not run_uid_value:
            raise ScoringError(
                f"{ref}: no fingerprint_registry.run_uid — without the run "
                "identity this result cannot be bound to the identity "
                "events it produced."
            )
        if str(run_uid_value) != run_uid:
            raise ScoringError(
                f"{ref}: provenance.run_uid {redacted(run_uid)} does not "
                f"EQUAL fingerprint_registry.run_uid "
                f"{redacted(str(run_uid_value))} (both redacted — run_uids "
                "embed launch seeds). The declaration and the registry must "
                "corroborate; a unit whose two run-identity records "
                "disagree did not run what it declares."
            )
    else:
        stray_policy = provenance.get("fp_registry_policy")
        if stray_policy is not None:
            raise ScoringError(
                f"{ref}: arm {arm!r} carries no identity layer but declares "
                f"fp_registry_policy={stray_policy!r} — the custody "
                "contract requires null/absent on non-FP arms; the unit did "
                "not run the arm it declares."
            )

    trajectory = payload.get("trajectory")
    return {
        "path": str(path),
        "ref": ref,
        "arm": arm,
        "scenario": scenario,
        "seed": seed,
        "run_uid": run_uid,
        "trajectory": trajectory,
        "diagnostics": payload.get("h4_diagnostics"),
        # Endpoints computed eagerly: the <5-round refusal must hold for
        # EVERY census unit, not only the arms a contrast happens to touch.
        "acc_final5": acc_final5(trajectory, ref),
        "final_round": final_round_accuracy(trajectory, ref),
        "f1_final5": f1_final5(trajectory, ref),
    }


def load_units(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Read + custody-corroborate every unit result JSON; refuse anything
    the statistic does not describe. Cross-unit: duplicate cells and
    duplicate run_uids make the census ambiguous and refuse."""
    if not paths:
        raise ScoringError("no unit result JSONs were supplied")
    expected_split_sha = lib.split_manifest_sha256()
    records: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ScoringError(
                f"unit result not found: {redacted(str(path))} (paths are "
                "printed redacted — launch tooling embeds seeds in "
                "filenames; hash your candidate path strings to locate it)."
            )
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ScoringError(
                f"unit {redacted(str(path))} is not valid JSON "
                f"({exc.msg} at line {exc.lineno})."
            ) from exc
        if not isinstance(payload, dict):
            raise ScoringError(
                f"unit {redacted(str(path))} is not a JSON object."
            )
        records.append(_validate_unit(path, payload, expected_split_sha))

    cells: Dict[Any, Dict[str, Any]] = {}
    for record in records:
        key = (record["arm"], record["scenario"], record["seed"])
        previous = cells.get(key)
        if previous is not None:
            raise ScoringError(
                f"two units cover the same (arm, scenario, seed) cell: "
                f"{previous['ref']} and {record['ref']} (seed values not "
                "printed). One unit per cell; a duplicate makes the census "
                "ambiguous."
            )
        cells[key] = record

    uids: Dict[str, Dict[str, Any]] = {}
    for record in records:
        uid = record["run_uid"]
        if uid is None:
            continue
        previous = uids.get(uid)
        if previous is not None:
            raise ScoringError(
                f"ambiguous provenance: {previous['ref']} and "
                f"{record['ref']} both declare run_uid {redacted(uid)} "
                "(redacted — run_uids embed launch seeds). One run identity "
                "must map to exactly one unit."
            )
        uids[uid] = record
    return records


# ===========================================================================
# Census
# ===========================================================================

def assert_census(units: Sequence[Mapping[str, Any]],
                  manifest_seeds: Sequence[int]) -> Dict[str, Any]:
    """The 9 x 6 x 10 census, tolerating ONLY the pre-stated drop order.

    Arms are dropped all-or-nothing, absent arms must form a prefix of
    8-4-7-9-6-5-3, arms 1 and 2 never drop, and every surviving arm needs
    its full C0+S0-S4 x sealed-seed block (C0 cells are exempt from any
    drop: they are the degradation reference). Refusal messages carry seed
    COUNTS only, never values.
    """
    if len(manifest_seeds) != N_SEEDS:
        raise ScoringError(
            f"the registered seed manifest carries {len(manifest_seeds)} "
            f"value(s); the sealed H4 design registers exactly {N_SEEDS} "
            "(values not printed). A resized manifest is not the registered "
            "design."
        )
    manifest_set = frozenset(manifest_seeds)

    observed_arms = {u["arm"] for u in units}
    missing_adjudicating = [a for a in NEVER_DROP if a not in observed_arms]
    if missing_adjudicating:
        raise ScoringError(
            f"adjudicating arm(s) {missing_adjudicating} are absent from "
            "the corpus — arms 1 and 2 never drop (§ 2); without them there "
            "is no primary contrast and no census."
        )
    absent = ARM_TOKENS - observed_arms
    if absent != set(DROP_ORDER[:len(absent)]):
        raise ScoringError(
            f"absent arm(s) {sorted(absent)} do not form a prefix of the "
            f"pre-stated drop order {list(DROP_ORDER)} (erratum-A: "
            "8-4-7-9-6-5-3 by arm number). An out-of-order drop is a "
            "discretionary drop, and discretionary drops are refused."
        )
    dropped = [a for a in DROP_ORDER if a in absent]

    seed_union = {u["seed"] for u in units}
    if seed_union != manifest_set:
        raise ScoringError(
            "the corpus seed set does not EQUAL the registered manifest "
            f"({len(seed_union ^ manifest_set)} seed(s) differ; "
            f"{len(seed_union)} in units, {len(manifest_set)} in the "
            "manifest; values not printed). An internally-consistent matrix "
            "on the wrong seeds is not the registered design."
        )

    expected_cells = {(s, seed) for s in SCENARIOS for seed in manifest_set}
    for arm in sorted(observed_arms, key=_ARM_NUMBER.__getitem__):
        got = {(u["scenario"], u["seed"]) for u in units if u["arm"] == arm}
        extra = got - expected_cells
        if extra:
            scenarios = sorted({s for s, _ in extra})
            raise ScoringError(
                f"arm {arm!r}: {len(extra)} cell(s) outside the registered "
                f"6x{N_SEEDS} design (scenario(s) {scenarios}; seed values "
                "not printed)."
            )
        missing = expected_cells - got
        if missing:
            by_scenario: Dict[str, int] = {}
            for s, _ in missing:
                by_scenario[s] = by_scenario.get(s, 0) + 1
            raise ScoringError(
                f"arm {arm!r}: {len(missing)} cell(s) missing vs the sealed "
                f"6x{N_SEEDS} design (by scenario: "
                f"{dict(sorted(by_scenario.items()))}; seed values not "
                "printed). An arm leaves the matrix ALL-OR-NOTHING per the "
                "pre-stated drop order — a partial arm is a broken census, "
                "not a drop — and C0 cells are required for every surviving "
                "arm (they are the § 4.1 degradation reference)."
            )

    present = sorted(observed_arms, key=_ARM_NUMBER.__getitem__)
    disclosure = (
        "no arms dropped — the full 9-arm matrix is present" if not dropped
        else (
            f"arm(s) dropped per the pre-stated order 8-4-7-9-6-5-3: "
            f"{dropped} (a prefix, disclosed per § 2; arms 1 and 2 present)"
        )
    )
    return {
        "arms_present": present,
        "arms_dropped": dropped,
        "drop_disclosure": disclosure,
        "n_units": len(units),
        "scenarios": list(SCENARIOS),
        "n_seeds": N_SEEDS,
        "seed_note": (
            "seed values are sealed and never printed; seed_ordinal in this "
            "report = index into the SORTED registered manifest"
        ),
    }


# ===========================================================================
# The report
# ===========================================================================

def score_units(units: Sequence[Mapping[str, Any]], *,
                require_census: bool = True,
                diagnostic: bool = False) -> Dict[str, Any]:
    """Assemble the full report. `require_census=False` and/or
    `diagnostic=True` both WITHHOLD the verdict; only the fully-gated path
    emits a terminal CONFIRMED/FALSIFIED/INCONCLUSIVE."""
    if require_census:
        manifest_seeds = lib.manifest_seed_list()
        census = assert_census(units, manifest_seeds)
        census["census_gate_enforced"] = True
        seeds_sorted: Sequence[int] = manifest_seeds
    else:
        observed = sorted({u["arm"] for u in units},
                          key=lambda a: _ARM_NUMBER.get(a, 99))
        seeds_sorted = sorted({u["seed"] for u in units})
        census = {
            "census_gate_enforced": False,
            "arms_present": observed,
            "arms_dropped": None,
            "drop_disclosure": (
                "census gate DISABLED (--allow-partial-census): arm "
                "presence is observed, not verified; no drop-order claim "
                "is made"
            ),
            "n_units": len(units),
            "scenarios": sorted({u["scenario"] for u in units},
                                key=SCENARIOS.index),
            "n_seeds": len(seeds_sorted),
            "seed_note": (
                "seed values never printed; seed_ordinal = index into the "
                "sorted OBSERVED seed set (diagnostic mode)"
            ),
        }

    lookup = {(u["arm"], u["scenario"], u["seed"]): u for u in units}
    observed_arms = {u["arm"] for u in units}

    primary = build_primary(lookup, seeds_sorted)
    conjunction = primary["conjunction"]
    secondaries = build_secondaries(lookup, seeds_sorted, observed_arms)
    diagnostics = aggregate_diagnostics(units, seeds_sorted)

    withheld = diagnostic or not require_census
    if withheld:
        verdict = None
        verdict_status = VERDICT_WITHHELD_STATUS
        reasons = []
        if not require_census:
            reasons.append(
                "the census and seed-set gates were DISABLED "
                "(--allow-partial-census): a partial or malformed corpus "
                "can satisfy every remaining check"
            )
        if diagnostic and require_census:
            reasons.append(
                "this is a diagnostic re-read (--force-diagnostic): the ONE "
                "adjudicating execution is the only run that may emit a "
                "terminal verdict"
            )
        verdict_withheld_reason = (
            "; ".join(reasons) + ". The component conjunction is reported "
            "for diagnosis only and is NOT a verdict."
        )
    else:
        verdict = conjunction
        verdict_status = conjunction
        verdict_withheld_reason = None

    unit_rows = sorted(
        (
            {
                "ref": u["ref"],
                "arm": u["arm"],
                "scenario": u["scenario"],
                "seed_ordinal": (
                    list(seeds_sorted).index(u["seed"])
                    if u["seed"] in set(seeds_sorted) else None),
                "run_uid_redacted": (
                    redacted(u["run_uid"]) if u["run_uid"] else None),
            }
            for u in units
        ),
        key=lambda r: (_ARM_NUMBER.get(r["arm"], 99), r["scenario"],
                       r["seed_ordinal"] if r["seed_ordinal"] is not None
                       else -1),
    )

    return {
        "instrument": "h4_composition",
        "status": STATUS,
        "spec": {
            "preregistration": (
                "docs/superpowers/specs/"
                "2026-08-16-h4-composition-preregistration.md (RATIFIED, "
                "methodology v1.51)"
            ),
            "erratum": (
                "docs/superpowers/specs/"
                "2026-08-17-h4-preregistration-erratum-a.md (RULED, v1.52)"
            ),
        },
        "arm_matrix": {str(n): token for n, token in ARM_BY_NUMBER.items()},
        "custody": {
            "eval_split": REQUIRED_EVAL_SPLIT,
            "eval_split_manifest_sha256": lib.split_manifest_sha256(),
            "serving_bundle_sha256": lib.required_serving_bundle_sha256(),
            "serving_bundle_source": "data/h4_serving/manifest_v2.json",
            "detector_arms": sorted(DETECTOR_ARMS),
            "fp_arms": sorted(FP_ARMS),
            "fp_registry_policy": REQUIRED_FP_POLICY,
            "fp_cohort": REQUIRED_FP_COHORT,
            "n_units": len(units),
        },
        "census": census,
        "primary": primary,
        "component_conjunction": conjunction,
        "verdict": verdict,
        "verdict_status": verdict_status,
        "verdict_withheld_reason": verdict_withheld_reason,
        "verdict_rule": VERDICT_RULE,
        "diagnostic_mode": withheld,
        "secondaries": secondaries,
        "diagnostics": diagnostics,
        "units": unit_rows,
    }


# ===========================================================================
# Memo + CLI
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "THE frozen one-shot H4 composition scorer (v1.51 + erratum-A "
            "v1.52). Requires the completed 9x6x10 sealed census (or a "
            "pre-stated-drop-order-reduced matrix) with exact custody; "
            "refuses anything else. Runs ONCE."
        )
    )
    parser.add_argument(
        "--units", nargs="+", required=True, type=Path,
        help="unit result JSONs — one per (arm, scenario, seed) cell",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="verdict JSON path (refused if it already exists — ONE "
             "execution — unless --force-diagnostic)",
    )
    parser.add_argument(
        "--memo", type=Path, default=None,
        help="human memo path (default: --out with a .md suffix)",
    )
    parser.add_argument(
        "--allow-partial-census", action="store_true",
        help="DIAGNOSTIC ONLY: disable the census/seed-set gates. The "
             "verdict is WITHHELD. Never use this for a verdict.",
    )
    parser.add_argument(
        "--force-diagnostic", action="store_true",
        help="allow re-running against an existing output path. The "
             "re-read is a DIAGNOSTIC: its verdict is WITHHELD, so the "
             "terminal verdict can never be re-rolled.",
    )
    args = parser.parse_args(argv)

    out_path: Path = args.out
    routed_note: Optional[str] = None
    if not args.force_diagnostic:
        if out_path.exists():
            print(f"REFUSED: output path {out_path} already exists — "
                  f"{ONE_EXECUTION_WORDING}", file=sys.stderr)
            return 2
    else:
        # The terminal artifact is INVIOLABLE. A diagnostic re-read may
        # only ever land on a path that is absent or provably holds a
        # previous diagnostic artifact; anything else routes to the
        # '.diagnostic' sibling, and if that is protected too, refuse.
        if artifact_class(out_path) == "protected":
            sibling = diagnostic_sibling(out_path)
            if artifact_class(sibling) == "protected":
                print(
                    f"REFUSED: {out_path} holds an artifact this scorer "
                    "may not overwrite (an adjudicating verdict, or a file "
                    "it cannot prove is a diagnostic), and the diagnostic "
                    f"sibling {sibling} is protected as well. The terminal "
                    "artifact is inviolable — point --out somewhere fresh. "
                    f"{ONE_EXECUTION_WORDING}", file=sys.stderr)
                return 2
            routed_note = (
                f"terminal artifact at {out_path} preserved untouched; "
                f"diagnostic output routed to {sibling}")
            out_path = sibling
    memo_path: Path = args.memo or out_path.with_suffix(".md")

    try:
        units = load_units(args.units)
        report = score_units(
            units,
            require_census=not args.allow_partial_census,
            diagnostic=args.allow_partial_census or args.force_diagnostic,
        )
    except ScoringError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"instrument: h4_composition ({report['custody']['n_units']} "
          f"units; census: {report['census']['drop_disclosure']})")
    print(f"{'scenario':>9}  {'median':>9}  {'p(2-sided)':>11}  "
          f"{'n':>3}  status")
    for scenario, component in report["primary"]["components"].items():
        if component["computed"]:
            wilcoxon = component["wilcoxon"]
            print(f"{scenario:>9}  {component['median_reduction']:>+9.4f}  "
                  f"{wilcoxon['p_two_sided']:>11.4g}  "
                  f"{component['n_pairs']:>3}  {component['status']}")
        else:
            print(f"{scenario:>9}  {'n/a':>9}  {'n/a':>11}  "
                  f"{component['n_pairs']:>3}  {component['status']}")
    print(f"\ncomponent conjunction: {report['component_conjunction']}")
    print(f"VERDICT: {report['verdict_status']}")
    if report["verdict_withheld_reason"]:
        print(report["verdict_withheld_reason"])
    print(f"\nSTATUS: {STATUS}")
    if args.allow_partial_census:
        print("WARNING: --allow-partial-census was set; this is NOT a "
              "verdict run.")
    if args.force_diagnostic:
        print("WARNING: --force-diagnostic was set; this artifact is a "
              "diagnostic, not the adjudicating read.")
    if routed_note:
        print(routed_note)

    # Defense in depth: never let ANY write land on a protected artifact,
    # whatever path got us here.
    if artifact_class(out_path) == "protected":
        print(f"REFUSED: {out_path} became protected between the gate and "
              f"the write — {ONE_EXECUTION_WORDING}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    write_memo(report, memo_path)
    print(f"written: {out_path}")
    print(f"written: {memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

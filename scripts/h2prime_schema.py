"""The frozen OUTPUT CONTRACT for the H2′ results JSON, in one declarative place.

Rounds 4 through 9 of adversarial review found the same two defects over and
over, one branch at a time: a node that lost a mandatory census field, or a
disqualified node that leaked a contrast statistic it was not entitled to
publish. Each was fixed where it was found. This module retires the class:
every branch of the serialized output is described ONCE, as data, and the whole
document is validated at the end of every run — including dev-smoke — with a
HARD STOP on violation.

The contract is declarative on purpose. A reviewer should be able to read the
entire output guarantee here, in one screen of rules, without tracing asserts
scattered through five modules.

Selector language (deliberately tiny — it only has to address this document):
  "P1"                          the P1 block
  "secondaries.auc_per_scenario.*"   every value of that dict
  "*.per_fold.*"                the per-fold rows under any parent
A rule that selects NOTHING is a FAILURE, not a pass. Vacuous satisfaction is
how a schema comes to read as coverage while checking nothing — this module's
own first draft shipped with a rule that silently matched zero nodes. Blocks the
document may legitimately omit (an EXP-048 contrast that did not run) carry an
explicit `optional_when` stating the condition, so the omission is declared
rather than inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# v1.15 § 1 DECISION G item 3 — the TGE coverage census, split honest/malicious.
CENSUS_KEYS = ("n_mal_scored", "n_mal_total", "n_honest_scored",
               "n_honest_total", "coverage_mal", "coverage_honest", "status")
# Contrast statistics a disqualified node may NEVER publish (item 4: "no
# contrast statistic is computed on it").
CONTRAST_STATISTICS = (
    # per-fold readouts (the CONTRAST outcome; realized FPRs are classified
    # as calibration evidence below, not as outcomes)
    "det_recall", "tge_recall",
    "detector_recall",
    # slice-level readouts
    "detector", "tge", "detector_mean", "tge_mean",
    "detector_mean_recall", "tge_mean_recall",
    "margin_detector_minus_tge",
    # the fixed-baseline contrast's own two names for the same two objects
    # (§ 4 secondary 7): the baseline side's readout and the slice margin. They
    # were unguarded until round 28, when the § 3.2 discipline was extended to
    # that block — the mirror of the leak the rest of this tuple already names.
    "baseline_per_seed_macro", "margin",
    # paired inference
    "per_seed_diff", "sign_test", "ci95_student_t_on_paired_diff",
)

# The § 4 blend-margin block's statistics. Held in its OWN tuple rather than
# folded into CONTRAST_STATISTICS because two of its names — `mean` and
# `per_seed` — are generic, and every other guarded block would start forbidding
# them by accident. Round 29: this block was the last unguarded baseline
# comparison.
BLEND_MARGIN_STATISTICS = (
    "mean", "per_seed", "fixed_contrast_margin", "fixed_contrast_sign_test",
)

#: Every enumerated statistic, whatever block it belongs to. The classification
#: meta-test builds its known-key universe from THIS, so a new block's statistic
#: tuple is covered the moment it is registered here — the round-28 lesson about
#: enumerations that live only in a fixture.
ALL_STATISTIC_KEYS = tuple(
    dict.fromkeys(CONTRAST_STATISTICS + BLEND_MARGIN_STATISTICS))

# Keys that appear on a retained contrast node and are NOT statistics. Each
# carries the reason it is census-or-structural, so "unclassified" can never be
# the silent default: a new key that is neither census, statistic, nor listed
# here fails the classification meta-test.
NON_STATISTIC_KEYS = {
    "note": "prose explaining a status",
    "uncovered_sides": "census fact — which side of the fold had no coverage",
    "reason": "prose explaining a NOT COMPARABLE / NOT RUN verdict",
    "n_folds": "structural count of folds in the slice",
    "n_folds_expected": "structural count from the REGISTERED seed set",
    "n_folds_with_covered_rows": "structural count of folds with any coverage",
    "n_folds_comparable": "structural count of folds clearing the floor",
    "computed_over_n_folds": "structural count — which folds the mean used",
    "n_paired_seeds": "structural count of retained pairs",
    "n_seeds_expected": "structural count from the REGISTERED seed set",
    "seeds_missing_from_pairing": "census — seeds that produced no cell",
    "missing_folds": "census — folds absent from the slice",
    "disqualified_folds": "census — folds below the floor, with their counts",
    "census_only_folds_excluded_from_contrast": "census — folds with no operating point",
    "exclusion_causes": "census — how many folds each guard excluded",
    "excluded_folds_by_cause": "census — which folds each guard excluded, by name",
    "per_fold": "the per-fold census map itself",
    "census_per_fold": "the per-fold census map itself",
    "per_attack_family": "the per-family sub-slices",
    "coverage_fraction_malicious": "census — coverage, split malicious",
    "coverage_fraction_honest": "census — coverage, split honest",
    "n_scored_malicious_all_folds": "census — summed scored rows",
    "n_malicious_total": "census — summed eligible rows",
    "intersection_coverage_per_seed": "census — coverage of the row-matched intersection",
    "intersection_coverage_mean": "census — coverage of the row-matched intersection",
    "min_scored_malicious_for_comparability": "the frozen floor value, a constant",
    "comparability_grain": "prose naming the grain the floor binds at",
    "row_matched": "structural flag — the populations were matched by construction",
    "coverage": "census — covered fraction of the fold",
    # AGGREGATE coverage (round 26). Census facts at the SLICE grain: a fold
    # count says how many were excluded, not how much of the population was
    # scored. Summed over every fold including disqualified ones, because
    # suppression is for contrast statistics and coverage is not one.
    "coverage_population": "census — states which folds the coverage sums over",
    "coverage_all_folds_population": "census — states which folds the coverage sums over",
    "intersection_coverage_population": "census — states the scope of the intersection coverage",
    "n_scored_malicious_all_folds": "census — summed scored rows, every fold",
    "n_malicious_total_all_folds": "census — summed eligible rows, every fold",
    "coverage_malicious_all_folds": "census — coverage over every fold, malicious",
    "n_scored_honest_all_folds": "census — summed scored honest rows, every fold",
    "n_honest_total_all_folds": "census — summed eligible honest rows, every fold",
    "coverage_honest_all_folds": "census — coverage over every fold, honest",
    # CALIBRATION EVIDENCE, not contrast outcome. § 3.2 requires a halted
    # readout to be "reported as INCONCLUSIVE with the realized FPR printed":
    # the FPR is the EVIDENCE FOR the halt, so suppressing it on a
    # non-comparable node would delete the reason for the exclusion. Same
    # distinction round 10 drew for coverage.
    "det_fpr": "calibration evidence — the realized operating point",
    "tge_fpr": "calibration evidence — the realized operating point",
    "detector_realized_fpr": "calibration evidence — the realized operating point",
    "tge_realized_fpr": "calibration evidence — the realized operating point",
    "detector_comparable": "calibration evidence — whether that FPR is in interval",
    "tge_comparable": "calibration evidence — whether that FPR is in interval",
    # § 4 secondary 7, guarded at the v1.15b § 1.2 readout grain. The two pooled
    # FPRs are the EVIDENCE FOR a halt, so they ride the halting branch too; the
    # flag counts they were pooled from ride with them, because a rate without
    # its denominator cannot be audited.
    "label": "structural — the instrument's registered display name",
    "paired_seeds": "census — which seeds the contrast pairs over",
    "exclusion_cause": "census — the named cause, from the shared EXCLUSION_CAUSES set",
    "detector_pooled_fpr": "calibration evidence — the cohort operating point",
    "baseline_pooled_fpr": "calibration evidence — the cohort operating point",
    "detector_in_interval": "calibration evidence — whether that FPR is in interval",
    "baseline_in_interval": "calibration evidence — whether that FPR is in interval",
    "detector_flagged_honest": "census — flagged honest rows behind the pooled rate",
    "detector_honest_rows": "census — honest rows behind the pooled rate",
    "baseline_flagged_honest": "census — flagged honest rows behind the pooled rate",
    "baseline_honest_rows": "census — honest rows behind the pooled rate",
    "n_detector_seeds": "census — seeds the detector side has, for unpaired exclusions",
    "n_baseline_seeds": "census — seeds the baseline side has, for unpaired exclusions",
    # § 4 blend-margin block (round 29), guarded at the P1 readout grain: the
    # mean over per-seed blended FPRs, not the row-pooled rate — see the
    # `comparability_basis` P1 publishes for this same surface.
    "mean_realized_fpr": "calibration evidence — this side's cohort operating point",
    "detector_mean_realized_fpr": "calibration evidence — the detector's cohort operating point",
    "contributing_instruments_not_comparable": (
        "census — non-comparable instruments that still compose the oracle max"),
}

# The mirror of CONTRAST_STATISTICS. Rounds 4-9 chased statistics LEAKING onto
# rows not entitled to publish them; round 20 found the opposite hole on the
# other side of the same branch — a RETAINED row that published its slice mean
# and paired difference while dropping the two recalls and the two operating
# points those were computed from. Suppression has a contract; retention did
# not, so an omission read as compliance. Both directions are stated now.
#
# Two vocabularies exist for the same six facts and both are checked, each
# against the rows that use it: the pooled confirmatory per-fold row publishes
# the blend's readouts as `det_*`, while every per-family and EXP-048 census
# row publishes them as `detector_*`/`*_realized_fpr`.
# The emitted SECONDARY BLOCKS, registered so the rule-coverage meta-test has a
# universe that does not depend on a hand-built fixture remembering to include
# every block. Round 28: three blocks were unguarded precisely because the
# fixture omitted them, so the enumeration never saw them and the coverage test
# passed while checking nothing about them. A new secondary must be added here,
# which forces both a rule and fixture representation.
SECONDARY_BLOCKS = (
    "auc_per_scenario", "alie_fixed_baseline_contrasts",
    "bracket_recall_by_fpr",
    "g2_scored_rows_contrast", "global_cut_sensitivity",
    "exp048_standalone_tge_full_coverage", "strict_identity_loao_sensitivity",
    "declared_vacuous_cells", "not_computed_by_this_executor",
    "S3_blend_margin", "S4_blend_margin", "per_family_fold_recalls",
)

RETAINED_READOUT_KEYS = (
    "detector_recall", "tge_recall",
    "detector_realized_fpr", "tge_realized_fpr",
    "detector_comparable", "tge_comparable",
)
POOLED_RETAINED_READOUT_KEYS = (
    "det_recall", "tge_recall", "det_fpr", "tge_fpr",
)


@dataclass(frozen=True)
class Condition:
    """A declarative predicate over the document: `selector` equals `value`."""

    selector: str
    equals: object
    because: str = ""

    def holds(self, report: dict) -> bool:
        parts = [p for p in self.selector.split(".") if p]
        node = report
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return node == self.equals


@dataclass(frozen=True)
class Rule:
    """One clause of the output contract."""

    name: str
    select: str
    required: tuple = ()
    forbidden_when_status: tuple = ()      # (status_value, forbidden_keys)
    # The mirror clause: (status_value, required_keys). `required` alone cannot
    # express this — a census row is legitimately without readouts when it is
    # NOT COMPARABLE, so the obligation is conditional on the status the row
    # itself declares.
    required_when_status: tuple = ()       # (status_value, required_keys)
    authority: str = ""
    note: str = ""
    # A rule whose selector resolves to nothing FAILS unless a Condition over
    # the document itself says the block is legitimately absent. A bare string
    # would be unconditionally truthy — the block could vanish from a report
    # that CLAIMS to contain it and the gate would shrug.
    optional_when: Condition | None = None


SCHEMA: tuple[Rule, ...] = (
    Rule(
        name="document root",
        select="",
        required=("_meta", "rotation_plan", "P1", "P2", "secondaries", "verdict"),
        authority="v1.15 § 4 — the adjudicated bands plus the reported secondaries",
    ),
    Rule(
        name="provenance block",
        select="_meta",
        required=("executor", "spec", "profile", "golden_gate", "versions",
                  "gbdt_params", "features_frozen_order", "baselines",
                  "map_sha256", "defense_token", "seeds_ascending",
                  "n_cells", "n_seeds"),
        authority="§ 2.1a — the estimator and pipeline must be readable off the output",
    ),
    Rule(
        name="P1 band",
        select="P1",
        required=("band", "slice", "floor", "n_units", "per_seed_recall",
                  "per_seed_realized_blended_fpr", "mean_recall",
                  "realized_blended_fpr", "comparability_interval",
                  "ci95_student_t", "ci_is_reported_not_adjudicating",
                  "verdict", "reason"),
        authority="§ 4 (P1) — floor, realized FPR, CI reported-not-adjudicating",
    ),
    Rule(
        name="P2 band",
        select="P2",
        required=("band", "population", "estimand", "required_strictly_positive",
                  "per_unit", "per_seed_detector_macro", "per_seed_oracle_max_macro",
                  "per_seed_diff", "sign_test", "alpha",
                  "readout_grain_comparability", "realized_fpr_diagnostic",
                  "verdict", "reason"),
        authority="§ 4 (P2) — sign count, oracle maximum, v1.15b § 1.2 guard",
    ),
    Rule(
        name="P2 adjudicating comparability guard",
        select="P2.readout_grain_comparability",
        required=("_authority", "detector_pooled_fpr", "baseline_pooled_fpr",
                  "detector_flagged_honest", "detector_honest_rows",
                  "baseline_flagged_honest", "baseline_honest_rows",
                  "detector_in_interval", "baseline_in_interval",
                  "argmax_instrument_counts", "n_cells_where_argmax_was_a_tie",
                  "n_cells_no_tie", "n_cells_partial_tie", "n_cells_full_tie",
                  "partial_tie_cells"),
        authority="v1.15b § 1.2 — row-pooled, arg-max side, frozen tie-break",
    ),
    Rule(
        name="P2 per-unit rows",
        select="P2.per_unit.*",
        required=("scenario", "seed", "detector_recall", "detector_fpr",
                  "comparable_diagnostic"),
        authority="§ 3.1 — the (scenario, seed) statistical unit",
    ),
    Rule(
        name="conjunction",
        select="verdict",
        required=("p1", "p2", "conjunction", "terminal_protocol"),
        authority="§ 4.1 / § 3.2 — P1 ∧ P2 and the terminal protocol",
    ),
    # ---- reported secondaries ------------------------------------------
    Rule(
        name="AUC per scenario",
        select="secondaries.auc_per_scenario.*",
        required=("per_seed", "mean", "ci95_student_t"),
        authority="§ 4 secondary 4",
    ),
    Rule(
        name="G2 slice",
        select="secondaries.g2_scored_rows_contrast.scenarios.*",
        required=("status", "n_folds", "n_folds_expected", "n_folds_comparable",
                  "per_fold", "coverage_fraction_malicious",
                  "coverage_fraction_honest", "disqualified_folds"),
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        authority="§ 1 DECISION G items 3–4",
    ),
    Rule(
        name="G2 per-fold census",
        select="secondaries.g2_scored_rows_contrast.scenarios.*.per_fold.*",
        required=CENSUS_KEYS,
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        required_when_status=("COMPUTED", POOLED_RETAINED_READOUT_KEYS),
        authority="§ 1 DECISION G item 3 (census) / item 4 (no statistic)",
        note="the round 4-9 defect class, stated once for every fold row",
    ),
    Rule(
        name="G2 per-family slice",
        select="secondaries.g2_scored_rows_contrast.scenarios.*.per_attack_family.*",
        required=("status", "n_folds", "n_folds_comparable", "census_per_fold"),
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        authority="§ 1 DECISION G item 3 — split by attack family",
    ),
    Rule(
        name="G2 per-family census rows",
        select=("secondaries.g2_scored_rows_contrast.scenarios.*"
                ".per_attack_family.*.census_per_fold.*"),
        required=CENSUS_KEYS,
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        required_when_status=("COMPUTED", RETAINED_READOUT_KEYS),
        authority="§ 1 DECISION G item 3 (census) / item 4 (no statistic)",
        note=("round 20: a comparable family fold cleared every guard and "
              "serialized census + status only, so the slice kept a paired "
              "difference no reader could audit back to its two sides"),
    ),
    Rule(
        name="blend margin slice (S4)",
        select="secondaries.S4_blend_margin",
        required=("detector_blended_mean", "detector_per_seed",
                  "detector_mean_realized_fpr", "detector_ci95_student_t",
                  "oracle_max", "baselines"),
        authority="§ 4 — reported blended margin against the § 3.1 baselines",
    ),
    Rule(
        name="blend margin slice (S3)",
        select="secondaries.S3_blend_margin",
        required=("detector_blended_mean", "detector_per_seed",
                  "detector_mean_realized_fpr", "detector_ci95_student_t",
                  "oracle_max", "baselines"),
        authority="§ 4 — reported blended margin against the § 3.1 baselines",
    ),
    # Round 29: the S3/S4 fixed-baseline margins were the last baseline
    # comparison emitting statistics with no comparability guard. One rule
    # covers both scenarios' nodes.
    Rule(
        name="blend margin fixed-baseline contrast (S4)",
        select="secondaries.S4_blend_margin.baselines.*",
        required=("label", "status", "n_paired_seeds", "paired_seeds",
                  "mean_realized_fpr", "detector_mean_realized_fpr",
                  "detector_in_interval", "baseline_in_interval"),
        forbidden_when_status=("NOT COMPARABLE", BLEND_MARGIN_STATISTICS),
        required_when_status=("COMPUTED", BLEND_MARGIN_STATISTICS),
        authority=("§ 4 blended margin, guarded at the P1 readout grain "
                   "(§ 3.2: every baseline comparison, symmetrically)"),
    ),
    Rule(
        name="blend margin fixed-baseline contrast (S3)",
        select="secondaries.S3_blend_margin.baselines.*",
        required=("label", "status", "n_paired_seeds", "paired_seeds",
                  "mean_realized_fpr", "detector_mean_realized_fpr",
                  "detector_in_interval", "baseline_in_interval"),
        forbidden_when_status=("NOT COMPARABLE", BLEND_MARGIN_STATISTICS),
        required_when_status=("COMPUTED", BLEND_MARGIN_STATISTICS),
        authority=("§ 4 blended margin, guarded at the P1 readout grain "
                   "(§ 3.2: every baseline comparison, symmetrically)"),
    ),
    # Round 30: these two replace a SELECTOR-ONLY rule. Selecting a node and
    # requiring nothing of it is the vacuous-coverage failure this module's own
    # docstring warns about — the block could have dropped every readout and
    # passed. The family level genuinely has no fixed key set (its keys ARE the
    # scenarios), so the requirements bind one and two levels down, where the
    # reducer does have a fixed shape. The block emits ONE branch — there is no
    # status and no suppression path — so a retention contract would be
    # vacuous here; `required` is unconditional on purpose.
    Rule(
        name="per-family fold recalls (scenario slice)",
        select="secondaries.per_family_fold_recalls.*.*",
        required=("per_seed", "mean_recall", "mean_realized_fpr",
                  "ci95_student_t"),
        authority="§ 4 item 3 — per-family per-fold recall readouts",
    ),
    Rule(
        name="per-family fold recalls (per-seed rows)",
        select="secondaries.per_family_fold_recalls.*.*.per_seed.*",
        required=("recall", "realized_fpr", "n_mal"),
        authority=("§ 4 item 3 — the fold row behind every family mean, with "
                   "its operating point and its malicious row count"),
    ),
    # § 6 (ratification addition) — REPORTED ONLY. Guarded like every other
    # emitted block: a reported number that no rule checks is a number nobody
    # is accountable for. There is no status and no suppression path here —
    # the block halts at no point, by ratified design — so `required` is
    # unconditional, as on the other single-branch blocks.
    Rule(
        name="bracket recall (block)",
        select="secondaries.bracket_recall_by_fpr",
        required=("targets", "adjudicating_target", "interval_rule",
                  "comparability_is_evidence_only", "points"),
        authority="§ 6 — reported 1/2/5/10 % FPR bracket, adjudicates nothing",
    ),
    Rule(
        name="bracket recall (point)",
        select="secondaries.bracket_recall_by_fpr.points.*",
        required=("target_fpr", "interval", "is_adjudicating_point", "scenarios"),
        authority="§ 6 — each point carries its own proportional interval",
    ),
    Rule(
        name="bracket recall (scenario slice)",
        select="secondaries.bracket_recall_by_fpr.points.*.scenarios.*",
        required=("n_folds", "per_seed_recall", "mean_recall", "ci95_student_t",
                  "per_seed_realized_fpr", "mean_realized_fpr",
                  "per_seed_in_interval", "in_interval"),
        authority=("§ 6 — recall AND its realized FPR at every point; a recall "
                   "without its operating point is not a readout"),
    ),
    Rule(
        name="global-cut sensitivity",
        select="secondaries.global_cut_sensitivity.S4_blend_global_cut",
        required=("per_seed", "mean_recall", "mean_realized_fpr",
                  "would_be_comparable"),
        authority="§ 2.2 disclosure / § 4 secondary 8",
    ),
    Rule(
        name="ALIE fixed-baseline contrasts",
        select="secondaries.alie_fixed_baseline_contrasts.contrasts.*",
        # The four statistics moved OUT of `required` and into the retention
        # contract: a NOT COMPARABLE contrast owes none of them, and demanding
        # them there would fail the very node the § 3.2 halt produces. What is
        # owed unconditionally is the label, the census, and BOTH pooled FPRs —
        # the evidence for whichever branch the node took.
        required=("label", "status", "n_paired_seeds", "paired_seeds",
                  "n_detector_seeds", "n_baseline_seeds",
                  "detector_pooled_fpr", "baseline_pooled_fpr",
                  "detector_in_interval", "baseline_in_interval",
                  "detector_flagged_honest", "detector_honest_rows",
                  "baseline_flagged_honest", "baseline_honest_rows"),
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        required_when_status=("COMPUTED", ("baseline_per_seed_macro",
                                           "per_seed_diff", "margin",
                                           "sign_test")),
        authority=("§ 4 secondary 7, guarded at the v1.15b § 1.2 readout grain "
                   "(§ 3.2 applies to every baseline comparison, symmetrically)"),
    ),
    Rule(
        name="EXP-048 blended slice",
        select="secondaries.exp048_standalone_tge_full_coverage.blended.*",
        required=("status", "n_folds", "n_folds_comparable", "census_per_fold",
                  "row_matched", "min_scored_malicious_for_comparability"),
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        authority="§ 4 secondary 9 — registered mechanics, one floor both sides",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because=("the arm was not staged; the block reports NOT RUN as a "
                     "scope disclosure. If the report claims anything else, "
                     "these results are mandatory.")),
    ),
    Rule(
        name="EXP-048 per-fold census",
        select=("secondaries.exp048_standalone_tge_full_coverage.blended.*"
                ".census_per_fold.*"),
        required=CENSUS_KEYS,
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        required_when_status=("COMPUTED", RETAINED_READOUT_KEYS),
        authority="§ 4 secondary 9 — the mirror of the confirmatory census",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because="the arm was not staged (see the rule above)"),
    ),
    Rule(
        name="EXP-048 per-family slice",
        select="secondaries.exp048_standalone_tge_full_coverage.per_family.*.*",
        required=("status", "n_folds", "n_folds_comparable", "census_per_fold",
                  "row_matched"),
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        authority="§ 4 secondary 9 — per-family paired output on the exposed arm",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because="the arm was not staged"),
    ),
    Rule(
        name="EXP-048 per-family census rows",
        select=("secondaries.exp048_standalone_tge_full_coverage.per_family.*.*"
                ".census_per_fold.*"),
        required=CENSUS_KEYS,
        forbidden_when_status=("NOT COMPARABLE", CONTRAST_STATISTICS),
        required_when_status=("COMPUTED", RETAINED_READOUT_KEYS),
        authority="§ 1 DECISION G item 3 — the census, on every family branch",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because="the arm was not staged"),
    ),
    Rule(
        name="EXP-048 coverage census",
        select="secondaries.exp048_standalone_tge_full_coverage.coverage",
        required=("n_rows", "scenarios"),
        authority="§ 1 DECISION G item 3 — the full-coverage census of the arm",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because="the arm was not staged"),
    ),
    Rule(
        name="EXP-048 coverage census per scenario",
        select="secondaries.exp048_standalone_tge_full_coverage.coverage.scenarios.*",
        required=("n_rows", "tge_score_coverage", "n_scored_malicious", "seeds"),
        authority="§ 1 DECISION G item 3 — coverage reported for every slice",
        optional_when=Condition(
            "secondaries.exp048_standalone_tge_full_coverage.status", "NOT RUN",
            because="the arm was not staged"),
    ),
    Rule(
        name="strict-identity sensitivity",
        select="secondaries.strict_identity_loao_sensitivity",
        required=("_note", "identity_key", "exclusion_census", "status"),
        authority="v1.15b § 3 — canonical device lineage, reported-only",
    ),
)


class SchemaViolation(RuntimeError):
    """The serialized output does not satisfy the frozen contract."""


def _resolve(node, parts: list[str]) -> list:
    """Walk the selector, fanning out on `*`. Missing paths yield nothing."""
    if not parts:
        return [node]
    head, rest = parts[0], parts[1:]
    if head == "*":
        # Fans out over dict VALUES and over list ELEMENTS: several blocks in
        # this document are lists of rows (P2.per_unit), and a wildcard that
        # silently skipped them would make its rule vacuous — a schema that
        # checks nothing is worse than no schema, because it reads as coverage.
        if isinstance(node, list):
            out = []
            for child in node:
                out.extend(_resolve(child, rest))
            return out
        if not isinstance(node, dict):
            return []
        out = []
        for key, child in node.items():
            if str(key).startswith("_"):
                continue                      # `_note`/`_meta` are prose, not rows
            out.extend(_resolve(child, rest))
        return out
    if not isinstance(node, dict) or head not in node:
        return []
    return _resolve(node[head], rest)


def check(report: dict) -> dict:
    """Validate the whole document. Returns a receipt or raises SchemaViolation."""
    violations: list[str] = []
    checked = 0
    for rule in SCHEMA:
        parts = [p for p in rule.select.split(".") if p]
        nodes = [n for n in _resolve(report, parts) if isinstance(n, dict)]
        if not nodes and not (rule.optional_when
                              and rule.optional_when.holds(report)):
            violations.append(
                f"[{rule.name}] selector '{rule.select or '<root>'}' matched NO "
                f"nodes — a mandatory block is missing or the rule is dead "
                f"({rule.authority})"
                + (f"; the rule is optional only when "
                   f"{rule.optional_when.selector} == "
                   f"{rule.optional_when.equals!r}, which this document does "
                   f"not satisfy" if rule.optional_when else ""))
        for node in _resolve(report, parts):
            if not isinstance(node, dict):
                continue
            checked += 1
            missing = [k for k in rule.required if k not in node]
            if missing:
                violations.append(
                    f"[{rule.name}] missing {missing} "
                    f"(selector '{rule.select or '<root>'}'; {rule.authority})")
            if rule.required_when_status:
                status, needed = rule.required_when_status
                if node.get("status") == status:
                    absent = [k for k in needed if k not in node]
                    if absent:
                        violations.append(
                            f"[{rule.name}] status={status!r} dropped retained "
                            f"readouts {absent} — a matched contrast must "
                            f"serialize BOTH sides (selector '{rule.select}'; "
                            f"{rule.authority})")
            if rule.forbidden_when_status:
                status, forbidden = rule.forbidden_when_status
                if node.get("status") == status:
                    leaked = [k for k in forbidden if k in node]
                    if leaked:
                        violations.append(
                            f"[{rule.name}] status={status!r} leaked contrast "
                            f"statistics {leaked} (selector '{rule.select}'; "
                            f"{rule.authority})")
    if violations:
        raise SchemaViolation(
            "OUTPUT SCHEMA VIOLATION — the results document does not satisfy the "
            "frozen contract in scripts/h2prime_schema.py. This is a hard stop: "
            "a malformed result is not a result.\n  " + "\n  ".join(violations))
    return {"schema": "scripts/h2prime_schema.py",
            "rules": len(SCHEMA), "nodes_checked": checked,
            "census_keys": list(CENSUS_KEYS),
            "contrast_statistics_guarded": list(CONTRAST_STATISTICS),
            "status": "PASS"}

#!/usr/bin/env python3
"""Report assembly for the frozen H4 scorer: paired contrasts, the § 4.1
primary block, the § 4.2/4.3 secondaries, the § 6 diagnostics aggregation
and the human memo.

Split out of `scripts/analyze_h4_composition.py` purely for file-size
discipline; the design authority and the freeze status are that module's
docstring. Everything here is deterministic assembly over already-validated
unit records — no custody decision is made in this module, and nothing
here can print a seed value (units arrive with pre-redacted refs and
seeds are referenced by ordinal only).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts import h4_scoring_lib as lib
from scripts.h4_scoring_lib import (
    ATTACK_SCENARIOS,
    ATTRIBUTIVE_CONTRASTS,
    COMPARATOR_ARM,
    REFERENCE_SCENARIO,
    S0_SCENARIO,
    TREATMENT_ARM,
    exact_median,
    exact_wilcoxon_two_sided,
    is_reference_anomaly,
    scenario_gate,
)

#: The § 6 per-layer tally classes (BUILD_CONTRACT / flowerfl.h4_diagnostics).
_TALLY_KEYS = (
    "detector_dropped_honest",
    "detector_dropped_malicious",
    "fp_hard_dropped",
    "aggregator_rejected",
)


# ===========================================================================
# Paired contrasts
# ===========================================================================

def _paired_component(lookup: Mapping[Any, Mapping[str, Any]],
                      treatment: str, comparator: str, reference: str,
                      scenario: str, seeds_sorted: Sequence[int],
                      endpoint: str) -> Dict[str, Any]:
    """degradation/reduction over seed-pairs for one (contrast, scenario).

    reduction(seed) = degradation(comparator) - degradation(treatment);
    positive = the treatment degrades less. Pairs whose cells are missing
    (diagnostic mode) or whose endpoint is null (F1 on a legacy trajectory)
    are skipped AND counted, never fabricated.
    """
    pairs: List[Dict[str, Any]] = []
    n_missing_cells = 0
    n_endpoint_null = 0
    for ordinal, seed in enumerate(seeds_sorted):
        needed = (
            lookup.get((treatment, reference, seed)),
            lookup.get((treatment, scenario, seed)),
            lookup.get((comparator, reference, seed)),
            lookup.get((comparator, scenario, seed)),
        )
        if any(cell is None for cell in needed):
            n_missing_cells += 1
            continue
        values = [cell[endpoint] for cell in needed]
        if any(v is None for v in values):
            n_endpoint_null += 1
            continue
        deg_treatment = values[0] - values[1]
        deg_comparator = values[2] - values[3]
        pairs.append({
            "seed_ordinal": ordinal,
            "degradation_treatment": deg_treatment,
            "degradation_comparator": deg_comparator,
            "reduction": deg_comparator - deg_treatment,
        })
    base = {
        "scenario": scenario,
        "reference_scenario": reference,
        "endpoint": endpoint,
        "treatment_arm": treatment,
        "comparator_arm": comparator,
        "n_pairs": len(pairs),
        "n_pairs_missing_cells": n_missing_cells,
        "n_pairs_endpoint_null": n_endpoint_null,
    }
    if not pairs:
        base.update({
            "computed": False,
            "reason": (
                "no complete seed-pair could be formed "
                f"({n_missing_cells} with missing cells, "
                f"{n_endpoint_null} with a null endpoint)"
            ),
            "median_reduction": None,
            "wilcoxon": None,
            "pairs": [],
        })
        return base
    diffs = [p["reduction"] for p in pairs]
    base.update({
        "computed": True,
        "median_reduction": exact_median(diffs),
        "wilcoxon": exact_wilcoxon_two_sided(diffs),
        "pairs": pairs,
    })
    return base


def _attack_components(lookup, treatment, comparator, reference,
                       seeds_sorted, endpoint) -> Dict[str, Any]:
    return {
        scenario: _paired_component(
            lookup, treatment, comparator, reference, scenario,
            seeds_sorted, endpoint)
        for scenario in ATTACK_SCENARIOS
    }


def build_primary(lookup, seeds_sorted) -> Dict[str, Any]:
    components = _attack_components(
        lookup, TREATMENT_ARM, COMPARATOR_ARM, REFERENCE_SCENARIO,
        seeds_sorted, "acc_final5")
    for component in components.values():
        if component["computed"]:
            component["gate"] = scenario_gate(
                component["median_reduction"],
                component["wilcoxon"]["p_two_sided"])
            component["status"] = (
                "PASS" if component["gate"]["passed"] else "FAIL")
        else:
            component["gate"] = None
            component["status"] = "INCONCLUSIVE"
    statuses = {c["status"] for c in components.values()}
    if "INCONCLUSIVE" in statuses:
        conjunction = "INCONCLUSIVE"
    elif "FAIL" in statuses:
        conjunction = "FALSIFIED"
    else:
        conjunction = "CONFIRMED"
    return {
        "endpoint": "acc_final5",
        "reference_scenario": REFERENCE_SCENARIO,
        "treatment_arm": TREATMENT_ARM,
        "comparator_arm": COMPARATOR_ARM,
        "components": components,
        "conjunction": conjunction,
    }


def build_secondaries(lookup, seeds_sorted,
                       observed_arms) -> Dict[str, Any]:
    """§ 4.1 variants, the S0-referenced contrast (E1 secondary) and the
    § 2 attributive contrasts. ALL REPORTED, NON-GATING."""
    contrasts: Dict[str, Any] = {}
    for label, arm_x, arm_y in ATTRIBUTIVE_CONTRASTS:
        entry: Dict[str, Any] = {
            "treatment_arm": arm_x,
            "comparator_arm": arm_y,
            "delta_definition": (
                "per (scenario, seed): degradation(comparator) - "
                "degradation(treatment); positive = treatment degrades less"
            ),
        }
        absent = [a for a in (arm_x, arm_y) if a not in observed_arms]
        if absent:
            entry.update({
                "computed": False,
                "reason": (
                    f"arm(s) {absent} not in the census (dropped per the "
                    "pre-stated drop order); the contrast is not computable"
                ),
                "components": None,
            })
        else:
            entry["computed"] = True
            entry["components"] = _attack_components(
                lookup, arm_x, arm_y, REFERENCE_SCENARIO, seeds_sorted,
                "acc_final5")
        contrasts[label] = entry
    return {
        "note": (
            "REPORTED, NON-GATING (§ 2 no-rescue rule, extended verbatim to "
            "arm 9 by erratum-A E2): nothing here can rescue, overturn, "
            "soften or strengthen the primary contrast."
        ),
        "single_final_round": {
            "label": "single-final-round accuracy variant "
                     "(direct Szelag comparability)",
            "components": _attack_components(
                lookup, TREATMENT_ARM, COMPARATOR_ARM, REFERENCE_SCENARIO,
                seeds_sorted, "final_round"),
        },
        "f1_final5": {
            "label": "F1-based § 4.1 analog (continuity with the original "
                     "F1 registration; final-5-round mean F1)",
            "components": _attack_components(
                lookup, TREATMENT_ARM, COMPARATOR_ARM, REFERENCE_SCENARIO,
                seeds_sorted, "f1_final5"),
        },
        "s0_referenced": {
            "label": "S0-referenced contrast (pre-registered secondary, "
                     "erratum-A E1): marginal utility harm of the RMC "
                     "confounders under constant attack",
            "components": _attack_components(
                lookup, TREATMENT_ARM, COMPARATOR_ARM, S0_SCENARIO,
                seeds_sorted, "acc_final5"),
        },
        "attributive_contrasts": contrasts,
    }


# ===========================================================================
# § 6 diagnostics (REPORTED, NON-GATING)
# ===========================================================================

def _aggregate_layer_totals(values: List[Any]) -> Dict[str, Any]:
    """null-preserving sum: all-null -> null (layer absent, not a finding);
    any null/int mix -> null + inconsistency flag (never coerced to 0)."""
    nulls = [v for v in values if v is None]
    numbers = [v for v in values if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    if len(nulls) == len(values):
        return {"total": None, "inconsistent": False}
    if len(numbers) == len(values):
        return {"total": sum(numbers), "inconsistent": False}
    return {"total": None, "inconsistent": True}


def aggregate_diagnostics(units: Sequence[Mapping[str, Any]],
                          seeds_sorted: Sequence[int]) -> Dict[str, Any]:
    """§ 6 items 1-4 aggregated per (arm, scenario). Null-vs-zero semantics
    preserved end to end; a unit without an `h4_diagnostics` block is a
    REPORTED anomaly (diagnostics never gate), not a refusal."""
    ordinal = {seed: i for i, seed in enumerate(seeds_sorted)}
    missing_refs: List[str] = []
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for unit in units:
        diag = unit.get("diagnostics")
        if not isinstance(diag, Mapping):
            missing_refs.append(unit["ref"])
            continue
        bucket = buckets.setdefault(unit["arm"], {}).setdefault(
            unit["scenario"], {
                "n_units": 0,
                "empty_total": 0,
                "units_with_empty": 0,
                "frac_values": [],
                "rounds_total": 0,
                "rounds_null_blackout": 0,
                "layer_values": {key: [] for key in _TALLY_KEYS},
                "detector": {"present": False, "flagged_malicious": 0,
                             "flagged_honest": 0, "kept_malicious": 0,
                             "kept_honest": 0},
            })
        bucket["n_units"] += 1
        empty = ((diag.get("empty_aggregate_rounds") or {}).get("count")) or 0
        bucket["empty_total"] += int(empty)
        if empty:
            bucket["units_with_empty"] += 1
        fractions = diag.get("kept_set_malicious_fraction") or []
        sizes = diag.get("kept_set_size") or []
        for fraction, size in zip(fractions, sizes):
            bucket["rounds_total"] += 1
            if fraction is None:
                bucket["rounds_null_blackout"] += 1
                continue
            bucket["frac_values"].append(float(fraction))
            kept_malicious = int(round(float(fraction) * int(size)))
            bucket["detector"]["kept_malicious"] += kept_malicious
            bucket["detector"]["kept_honest"] += int(size) - kept_malicious
        totals = diag.get("per_layer_removal_totals") or {}
        for key in _TALLY_KEYS:
            bucket["layer_values"][key].append(totals.get(key))
        if (diag.get("layers_present") or {}).get("detector"):
            bucket["detector"]["present"] = True
            bucket["detector"]["flagged_malicious"] += (
                totals.get("detector_dropped_malicious") or 0)
            bucket["detector"]["flagged_honest"] += (
                totals.get("detector_dropped_honest") or 0)

    per_arm_scenario: Dict[str, Dict[str, Any]] = {}
    detector_proxies: Dict[str, Dict[str, Any]] = {}
    for arm, scenarios in buckets.items():
        for scenario, bucket in scenarios.items():
            layer_totals = {}
            inconsistent = []
            for key, values in bucket["layer_values"].items():
                agg = _aggregate_layer_totals(values)
                layer_totals[key] = agg["total"]
                if agg["inconsistent"]:
                    inconsistent.append(key)
            fracs = bucket["frac_values"]
            per_arm_scenario.setdefault(arm, {})[scenario] = {
                "n_units": bucket["n_units"],
                "empty_aggregate_rounds_total": bucket["empty_total"],
                "units_with_empty_rounds": bucket["units_with_empty"],
                "kept_set_malicious_fraction": {
                    "rounds_total": bucket["rounds_total"],
                    "rounds_null_blackout": bucket["rounds_null_blackout"],
                    "mean": (sum(fracs) / len(fracs)) if fracs else None,
                    "max": max(fracs) if fracs else None,
                },
                "per_layer_removal_totals": layer_totals,
                "layer_presence_inconsistent": inconsistent,
            }
            det = bucket["detector"]
            if det["present"]:
                recall_den = det["flagged_malicious"] + det["kept_malicious"]
                fpr_den = det["flagged_honest"] + det["kept_honest"]
                detector_proxies.setdefault(arm, {})[scenario] = {
                    "flagged_malicious": det["flagged_malicious"],
                    "flagged_honest": det["flagged_honest"],
                    "kept_malicious": det["kept_malicious"],
                    "kept_honest": det["kept_honest"],
                    "recall_proxy": (
                        det["flagged_malicious"] / recall_den
                        if recall_den else None),
                    "fpr_proxy": (
                        det["flagged_honest"] / fpr_den if fpr_den else None),
                }

    anomalies: List[Dict[str, Any]] = []
    n_reference_cells = 0
    for unit in units:
        if unit["scenario"] not in (REFERENCE_SCENARIO, S0_SCENARIO):
            continue
        n_reference_cells += 1
        check = is_reference_anomaly(unit["trajectory"], unit["ref"])
        if check["flagged"]:
            anomalies.append({
                "arm": unit["arm"],
                "scenario": unit["scenario"],
                "seed_ordinal": ordinal.get(unit["seed"]),
                "acc_final5": check["acc_final5"],
                "peak_rolling5": check["peak_rolling5"],
                "gap": check["gap"],
            })

    return {
        "note": (
            "§ 6 blackout/capture diagnostics — REPORTED, NON-GATING. A "
            "null tally means the layer is absent from the arm; nulls are "
            "not findings and are never coerced to zero. These diagnostics "
            "explain the primary endpoint; they do not gate it."
        ),
        "units_missing_diagnostics": {
            "count": len(missing_refs),
            "refs": missing_refs,
            "note": (
                "reported anomaly only — diagnostics never gate; the § 8 "
                "smoke is the VALUE-level check that the block populates"
            ),
        },
        "per_arm_scenario": per_arm_scenario,
        "detector_proxies": {
            "caveat": (
                "PROXIES from the § 6 tallies, not the § 4.2 flag-stream "
                "recall: denominators exclude clients removed by the FP and "
                "aggregator layers (their tallies are not class-split), so "
                "recall_proxy is exact only for detector-first arms and "
                "kept-set counts are reconstructed from "
                "kept_set_malicious_fraction x kept_set_size. Consistency "
                "evidence only; NON-GATING (§ 4.2 registers no bar)."
            ),
            "per_arm": detector_proxies,
        },
        "reference_anomaly": {
            "rule": (
                "a C0 or S0 cell whose acc_final5 sits STRICTLY more than "
                "5 pp below that cell's own peak rolling-5-round mean "
                "accuracy (GWU-51 absorbing-state signature); NON-GATING — "
                "the § 4.1 formula is computed as registered regardless"
            ),
            "n_cells_checked": n_reference_cells,
            "flagged_cells": anomalies,
        },
    }


def _component_line(name: str, component: Mapping[str, Any]) -> str:
    if not component.get("computed"):
        return (f"| {name} | n/a | n/a | n/a | "
                f"{component.get('status', 'not computed')} |")
    wilcoxon = component["wilcoxon"]
    gate = component.get("gate")
    status = component.get("status", "reported")
    return (
        f"| {name} | {component['median_reduction']:+.4f} | "
        f"{wilcoxon['p_two_sided']:.4g} | {component['n_pairs']} | "
        f"{status if gate is not None else 'reported'} |"
    )


def write_memo(report: Mapping[str, Any], memo_path: Path) -> None:
    """The human memo. Contains NO seed values and no raw run_uids."""
    lines = [
        "# H4 composition — scorer output memo",
        "",
        f"**Verdict: {report['verdict_status']}**",
        "",
    ]
    if report["verdict_withheld_reason"]:
        lines += [f"> {report['verdict_withheld_reason']}", ""]
    lines += [
        f"- Instrument: `{report['instrument']}` — {report['status']}",
        f"- Units: {report['custody']['n_units']}",
        f"- Census: {report['census']['drop_disclosure']}",
        f"- Serving bundle sha256: "
        f"`{report['custody']['serving_bundle_sha256']}`",
        f"- Sealed-split manifest sha256: "
        f"`{report['custody']['eval_split_manifest_sha256']}`",
        "",
        "## Primary (§ 4.1 as amended by E1) — "
        "h2p_fp_krum vs krum, C0-referenced, acc_final5",
        "",
        "| scenario | median reduction | Wilcoxon p (two-sided) | n pairs "
        "| status |",
        "|---|---|---|---|---|",
    ]
    for scenario, component in report["primary"]["components"].items():
        lines.append(_component_line(scenario, component))
    lines += [
        "",
        f"Gate: median >= {lib.MEDIAN_BAR} AND p <= {lib.P_BAR}, per "
        "scenario, ALL FOUR conjunctive. "
        f"Component conjunction: **{report['component_conjunction']}**.",
        "",
        "## Secondaries (REPORTED, NON-GATING)",
        "",
    ]
    secondaries = report["secondaries"]
    for key in ("single_final_round", "f1_final5", "s0_referenced"):
        block = secondaries[key]
        lines.append(f"### {block['label']}")
        lines.append("")
        lines.append("| scenario | median reduction | p | n | note |")
        lines.append("|---|---|---|---|---|")
        for scenario, component in block["components"].items():
            lines.append(_component_line(scenario, component))
        lines.append("")
    lines.append("### Attributive contrasts (per § 2; non-gating)")
    lines.append("")
    for label, entry in secondaries["attributive_contrasts"].items():
        if not entry["computed"]:
            lines.append(f"- **{label}** ({entry['treatment_arm']} vs "
                         f"{entry['comparator_arm']}): {entry['reason']}")
            continue
        medians = {
            scenario: (f"{c['median_reduction']:+.4f}"
                       if c.get("computed") else "n/a")
            for scenario, c in entry["components"].items()
        }
        lines.append(f"- **{label}** ({entry['treatment_arm']} vs "
                     f"{entry['comparator_arm']}): median reduction "
                     f"{medians}")
    diagnostics = report["diagnostics"]
    anomalies = diagnostics["reference_anomaly"]
    lines += [
        "",
        "## § 6 diagnostics (REPORTED, NON-GATING)",
        "",
        f"- Units missing an h4_diagnostics block: "
        f"{diagnostics['units_missing_diagnostics']['count']}",
        f"- Reference-anomaly flags (C0/S0 cells, GWU-51 signature): "
        f"{len(anomalies['flagged_cells'])} of "
        f"{anomalies['n_cells_checked']} cells",
    ]
    for arm, scenarios in sorted(diagnostics["per_arm_scenario"].items()):
        empties = {s: b["empty_aggregate_rounds_total"]
                   for s, b in sorted(scenarios.items())
                   if b["empty_aggregate_rounds_total"]}
        if empties:
            lines.append(f"- Empty-aggregate (blackout) rounds, arm "
                         f"`{arm}`: {empties}")
    lines += [
        "",
        "---",
        "",
        f"_Rule: {report['verdict_rule']}_",
        "",
        "_Seed values are sealed and appear nowhere in this memo or the "
        "JSON; cells are identified by seed_ordinal. Paths and run_uids "
        "are redacted (sha256 prefixes) because both can embed launch "
        "seeds._",
    ]
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text("\n".join(lines) + "\n")



# ===========================================================================
# Output-artifact protection (one-execution discipline)
# ===========================================================================

def artifact_class(path: Path) -> str:
    """'absent' | 'diagnostic' | 'protected' — what may be written at `path`.

    Only a file we can PROVE is a withheld-verdict diagnostic artifact of
    this scorer is overwritable. An adjudicating (verdict non-null)
    artifact is the sealed ONE-execution output and is inviolable; an
    unparseable or foreign file might BE that artifact in a corrupted or
    hand-edited state, so it is protected too — fail closed.
    """
    if not path.exists():
        return "absent"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "protected"
    if (isinstance(payload, dict)
            and payload.get("instrument") == "h4_composition"
            and payload.get("verdict") is None
            and payload.get("diagnostic_mode") is True):
        return "diagnostic"
    return "protected"


def diagnostic_sibling(path: Path) -> Path:
    """`verdict.json` -> `verdict.diagnostic.json` — clearly marked, never
    the sealed path."""
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}.diagnostic{suffix}")

"""H4 § 6 blackout/capture diagnostics — per-unit, UNIFORM across all 9 arms.

Design authority: `2026-08-16-h4-composition-preregistration.md` § 6 (as
amended by erratum-A) and `docs/reproduction/experiments.md`:

    Per unit, an `h4_diagnostics` block with per-round series + summaries:
    `kept_set_size[]`, `empty_aggregate_rounds` (count + round list),
    `kept_set_malicious_fraction[]` (ground truth from the scenario
    schedule), per-layer removal tallies {detector_dropped_honest,
    detector_dropped_malicious, fp_hard_dropped, aggregator_rejected} —
    uniform across all arms; a tally whose layer is absent in the arm is
    **null, never zero**, and nulls are not findings.

The raw evidence is `PluggableStrategy`'s per-round chain trace (which
plugin dropped which cids, and what survived to the base aggregation);
this module joins it against the scenario's ground truth (partition ->
schedule entry -> adversarial identity) and emits the block the frozen H4
scorer consumes. Reported, NON-GATING — these diagnostics explain the
primary endpoint; they never gate it.

Null semantics beyond layers: `kept_set_malicious_fraction` for a round
whose kept set is EMPTY is null (0/0 is undefined), and the round appears
in `empty_aggregate_rounds` — the blackout signature, recorded not
"fixed" (the global model is unchanged on such a round; that is measured
EXP-052 behavior and a diagnostic, not a bug).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Chain-layer classification by plugin name. Arm 4 (krum_tge_fp) keeps its
#: legacy Krum-first order — there TGE is the detector layer and Krum the
#: aggregator layer; position in the chain does not change the layer a plugin
#: belongs to.
DETECTOR_PLUGIN_NAMES = ("H2PrimeDetector", "TGEnsemble")
FP_PLUGIN_NAMES = ("Fingerprint",)
AGGREGATOR_PLUGIN_NAMES = ("KrumDefense", "TrustScore")

_TALLY_KEYS = (
    "detector_dropped_honest",
    "detector_dropped_malicious",
    "fp_hard_dropped",
    "aggregator_rejected",
)


def _layer_of(plugin_name: str) -> Optional[str]:
    if plugin_name in DETECTOR_PLUGIN_NAMES:
        return "detector"
    if plugin_name in FP_PLUGIN_NAMES:
        return "fp"
    if plugin_name in AGGREGATOR_PLUGIN_NAMES:
        return "aggregator"
    return None


def build_h4_diagnostics(strategy_obj: Any) -> Optional[Dict[str, Any]]:
    """The per-unit `h4_diagnostics` block, or None for non-scenario runs.

    Requires a ScenarioStrategy-shaped object exposing the chain trace
    (`_h4_chain_trace`, recorded by `PluggableStrategy.aggregate_fit`), the
    cid->partition map, the per-round schedule cache, the adversarial
    identity set, and the discovery-round offset. Discovery rounds (which
    bypass the plugin path by design) are excluded, exactly as the signal
    log excludes them.

    Ground-truth joins are fail-loud (v1.17 misattribution family): a kept
    or dropped cid that cannot be resolved to a scheduled logical identity
    raises rather than being silently classified.
    """
    trace_map = getattr(strategy_obj, "_h4_chain_trace", None)
    schedule_cache = getattr(strategy_obj, "_schedule_cache", None)
    if not trace_map or not schedule_cache:
        return None
    cid_to_partition = dict(getattr(strategy_obj, "_cid_to_partition", {}) or {})
    adv_ids = set(getattr(strategy_obj, "_adv_ids", set()) or set())
    round_offset = int(getattr(strategy_obj, "_round_offset", 1))
    plugin_names = [
        getattr(p, "name", "") for p in getattr(strategy_obj, "_plugins", []) or []
    ]

    detector = next((n for n in plugin_names if n in DETECTOR_PLUGIN_NAMES), None)
    fp = next((n for n in plugin_names if n in FP_PLUGIN_NAMES), None)
    aggregator = next(
        (n for n in plugin_names if n in AGGREGATOR_PLUGIN_NAMES), None
    )

    def _is_malicious(cid: str, scenario_round: int) -> bool:
        partition = cid_to_partition.get(str(cid))
        entries = schedule_cache.get(scenario_round) or []
        entry = next(
            (e for e in entries if e["partition_id"] == partition), None
        )
        if partition is None or entry is None:
            raise RuntimeError(
                f"h4_diagnostics: cid {cid!r} at scenario round "
                f"{scenario_round} cannot be resolved to a scheduled logical "
                f"identity (partition={partition}) — refusing to classify it "
                f"by guess (v1.17 misattribution family)."
            )
        return entry["logical_id"] in adv_ids

    # Scored rounds only: server rounds whose scenario round has a schedule.
    scored = sorted(
        server_round
        for server_round in trace_map
        if schedule_cache.get(int(server_round) - round_offset)
    )
    if not scored:
        return None

    server_rounds: List[int] = []
    scenario_rounds: List[int] = []
    kept_set_size: List[int] = []
    kept_set_malicious_fraction: List[Optional[float]] = []
    empty_rounds_server: List[int] = []
    per_round: Dict[str, Optional[List[int]]] = {
        "detector_dropped_honest": [] if detector else None,
        "detector_dropped_malicious": [] if detector else None,
        "fp_hard_dropped": [] if fp else None,
        "aggregator_rejected": [] if aggregator else None,
    }

    for server_round in scored:
        scenario_round = int(server_round) - round_offset
        trace = trace_map[server_round]
        kept = list(trace.get("kept_cids") or [])
        server_rounds.append(int(server_round))
        scenario_rounds.append(scenario_round)
        kept_set_size.append(len(kept))
        if kept:
            n_mal = sum(1 for cid in kept if _is_malicious(cid, scenario_round))
            kept_set_malicious_fraction.append(n_mal / len(kept))
        else:
            # 0/0 — null, never zero; the round is a blackout, not a finding
            # of "no capture".
            kept_set_malicious_fraction.append(None)
            empty_rounds_server.append(int(server_round))

        stage_drops: Dict[str, List[str]] = {"detector": [], "fp": [], "aggregator": []}
        for stage in trace.get("stages", []):
            layer = _layer_of(stage.get("plugin", ""))
            if layer is not None:
                stage_drops[layer].extend(stage.get("dropped_cids", []))

        if detector:
            det_honest = 0
            det_malicious = 0
            for cid in stage_drops["detector"]:
                if _is_malicious(cid, scenario_round):
                    det_malicious += 1
                else:
                    det_honest += 1
            per_round["detector_dropped_honest"].append(det_honest)
            per_round["detector_dropped_malicious"].append(det_malicious)
        if fp:
            per_round["fp_hard_dropped"].append(len(stage_drops["fp"]))
        if aggregator:
            per_round["aggregator_rejected"].append(len(stage_drops["aggregator"]))

    totals: Dict[str, Optional[int]] = {
        key: (sum(series) if series is not None else None)
        for key, series in per_round.items()
    }

    return {
        # Which chain layer each tally class maps to in THIS arm (null =
        # layer absent from the arm; its tallies below are null, not zero).
        "layers_present": {
            "detector": detector,
            "fp": fp,
            "aggregator": aggregator,
        },
        "rounds": {"server": server_rounds, "scenario": scenario_rounds},
        "kept_set_size": kept_set_size,
        "kept_set_malicious_fraction": kept_set_malicious_fraction,
        "empty_aggregate_rounds": {
            "count": len(empty_rounds_server),
            "server_rounds": empty_rounds_server,
            "scenario_rounds": [r - round_offset for r in empty_rounds_server],
        },
        "per_layer_removals_per_round": per_round,
        "per_layer_removal_totals": totals,
    }

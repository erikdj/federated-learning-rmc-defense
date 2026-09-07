"""Post-hoc metric helpers for Phase 4 result JSON.

Implements §9.3 of the Step 2 Amendment 1 spec:
  - compute_convergence_metrics(trajectory)
  - compute_confounder_control_metrics(scenario, trajectory, scoring_log)
"""
from __future__ import annotations

FLAG_THRESHOLD = 0.5  # CS trust below this is "flagged"


def compute_convergence_metrics(trajectory: list[dict]) -> dict:
    """Compute rounds-to-converge + stability metrics from an F1 trajectory.

    Args:
        trajectory: list of {"round": int, "f1": float, ...} entries, in
            chronological order. Must be non-empty.

    Returns:
        Dict with rounds_to_reach_90pct_final_F1, rounds_to_reach_50pct_final_F1,
        stability_last10 (std of f1 across last 10 rounds), monotonicity_score
        (fraction of round-pair transitions where f1 did not decrease).
    """
    if not trajectory:
        return {
            "rounds_to_reach_90pct_final_F1": -1,
            "rounds_to_reach_50pct_final_F1": -1,
            "stability_last10": 0.0,
            "monotonicity_score": 0.0,
        }

    f1s = [float(t.get("f1", 0.0)) for t in trajectory]
    final_f1 = f1s[-1]

    def first_round_at(threshold: float) -> int:
        for i, v in enumerate(f1s):
            if v >= threshold:
                return int(trajectory[i].get("round", i))
        return -1  # never reached

    target_90 = 0.9 * final_f1
    target_50 = 0.5 * final_f1

    # Standard deviation of last 10 rounds
    last = f1s[-10:] if len(f1s) >= 10 else f1s
    mean = sum(last) / len(last)
    variance = sum((v - mean) ** 2 for v in last) / len(last)
    stability = variance ** 0.5

    # Monotonicity: fraction of (i, i+1) pairs where f1[i+1] >= f1[i]
    pairs = len(f1s) - 1
    non_decreasing = sum(1 for i in range(pairs) if f1s[i + 1] >= f1s[i])
    monotonicity = non_decreasing / pairs if pairs > 0 else 1.0

    return {
        "rounds_to_reach_90pct_final_F1": first_round_at(target_90),
        "rounds_to_reach_50pct_final_F1": first_round_at(target_50),
        "stability_last10": float(stability),
        "monotonicity_score": float(monotonicity),
    }


def compute_confounder_control_metrics(
    scenario: dict,
    trajectory: list[dict],
    scoring_log: list[tuple],
) -> dict | None:
    """Compute discrimination ratio and per-identity-type breakdowns.

    Args:
        scenario: parsed scenario dict; must contain "honest_events" (list of
            dicts with offline_id, rejoin_round, victim_original_id) and
            "clients" map. honest_events may be empty.
        trajectory: per-round metric trajectory (unused for confounder control;
            kept for signature symmetry with convergence helper).
        scoring_log: list of (round, logical_cid, cs_trust_score) tuples
            emitted by ColdStartDefensePlugin's score_updates. May be empty
            if CS plugin not active.

    Returns:
        Dict with TPR/FPR/discrimination ratio + per-identity breakdowns, or
        None if scenario has no honest events (confounder-control not
        applicable for this run).

        Note: ``discrimination_ratio`` is ``None`` when ``fpr_honest_reconnect``
        is 0 but ``tpr_adversary_reconnect`` is positive (perfect discrimination
        — ratio is mathematically undefined). We serialize as ``None``/``null``
        rather than ``float('inf')`` because ``Infinity`` is not valid per
        RFC 8259 JSON. When both TPR and FPR are zero, the ratio is 0.0
        (no signal either way).
    """
    honest_events = scenario.get("honest_events", [])
    if not honest_events:
        return None

    honest_offline_ids = {ev["offline_id"] for ev in honest_events}
    n_honest = len(honest_events)

    # Adversary reconnect identities — derived from client roster naming pattern.
    # Convention from generate_continuous_reconnect_scenario: client_K_newN.
    clients = scenario.get("clients", {})
    adversary_offline_ids = {cid for cid in clients
                             if "_new" in cid and cid.startswith("client_")}
    # Also pick up adversary ids that appear in the scoring log but aren't yet
    # in the client roster (matches naming convention client_K_newN). This
    # keeps the count consistent with logs from synthetic / runtime scenarios.
    for _, cid, _ in scoring_log:
        if isinstance(cid, str) and cid.startswith("client_") and "_new" in cid:
            adversary_offline_ids.add(cid)
    n_adversary = len(adversary_offline_ids)

    # Tally flag rates and per-identity distributions.
    def _empty_dist() -> dict:
        return {"n_events": 0, "flag_rate": 0.0, "trust_scores": []}

    per_honest = {oid: _empty_dist() for oid in honest_offline_ids}
    per_adv = {oid: _empty_dist() for oid in adversary_offline_ids}

    flagged_honest = 0
    flagged_adv = 0
    seen_honest = 0
    seen_adv = 0

    for round_, cid, trust in scoring_log:
        flagged = trust < FLAG_THRESHOLD
        if cid in honest_offline_ids:
            seen_honest += 1
            per_honest[cid]["n_events"] += 1
            per_honest[cid]["trust_scores"].append(float(trust))
            if flagged:
                flagged_honest += 1
        elif cid in adversary_offline_ids:
            seen_adv += 1
            per_adv[cid]["n_events"] += 1
            per_adv[cid]["trust_scores"].append(float(trust))
            if flagged:
                flagged_adv += 1

    fpr = flagged_honest / seen_honest if seen_honest > 0 else 0.0
    tpr = flagged_adv / seen_adv if seen_adv > 0 else 0.0
    if fpr > 0:
        discrimination = tpr / fpr
    elif tpr > 0:
        # Undefined: positive detection rate but no honest events to compute
        # FPR against. Serialize as None (JSON-safe null) rather than
        # float('inf') which is not valid per RFC 8259.
        discrimination = None
    else:
        discrimination = 0.0  # no signal: no events flagged in either group

    def _finalize(per: dict) -> dict:
        out = {}
        for cid, d in per.items():
            ts = d.get("trust_scores", [])
            new_d = {k: v for k, v in d.items() if k != "trust_scores"}
            if ts:
                new_d["flag_rate"] = sum(1 for t in ts if t < FLAG_THRESHOLD) / len(ts)
                mean_ts = sum(ts) / len(ts)
                new_d["cs_trust_distribution"] = {
                    "mean": mean_ts,
                    "std": (sum((t - mean_ts) ** 2 for t in ts) / len(ts)) ** 0.5,
                    "min": min(ts),
                    "max": max(ts),
                }
            else:
                new_d["cs_trust_distribution"] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            out[cid] = new_d
        return out

    return {
        "tpr_adversary_reconnect": tpr,
        "fpr_honest_reconnect": fpr,
        "discrimination_ratio": discrimination,
        "n_adversary_reconnect_events": n_adversary,
        "n_honest_reconnect_events": n_honest,
        "per_adversary_identity": _finalize(per_adv),
        "per_honest_identity": _finalize(per_honest),
    }

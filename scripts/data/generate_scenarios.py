"""Parameterized scenario generator for FlowerFL RMC experiments.

Emits scenario JSON files consumed by flowerfl/scenario_strategy.py
(ScenarioStrategy) to drive an RMC run: which clients participate each
round, which logical identities are actively attacking, and what attack
type/params they use. There are THREE generation eras/families in this
file, oldest first:

    v1 (generate_intensity_scenario)         — single reconnect cycle, legacy.
    v2 (generate_continuous_reconnect_scenario) — multi-cycle continuous
        reconnection, closer to Szelag's paper; the `--heterogeneous` CLI
        flag layers 3 honest reconnect cycles on top of this as "v3".
    Design D (generate_clean_baseline_scenario / generate_benign_churn_only_scenario /
        generate_adaptive_switching_only_scenario / generate_identity_reset_only_scenario /
        generate_full_mix_scenario) — the CURRENT, spec-locked scenario family
        (S0-S4) used for the praxis's confirmatory H1/H2 runs. See
        docs/reproduction/experiments.md
        for the authoritative semantics; summarized here for orientation:

    S0 (clean_baseline)         — static-attack baseline: 9/20 clients run
        sustained ALIE from their ORIGINAL identities for the whole run
        (post-discovery). No benign churn, no identity reset, no strategy
        switching. NOT a "no-attack" control — it is an attack-bearing
        control that isolates the detection-capacity ceiling with every
        confounder OFF.
    S1 (benign_churn_only)      — S0's sustained-ALIE attack PLUS 3 honest
        clients disconnecting/reconnecting under new identities. Isolates
        false-positive sensitivity to legitimate (non-adversarial) churn.
    S2 (adaptive_switching_only) — 9 adversaries keep their ORIGINAL
        identities (no reconnect) but rotate attack TYPE across cycles
        (gaussian_noise -> ALIE -> label_flip -> ...). Isolates
        within-identity attack-mode variation.
    S3 (identity_reset_only)    — pure RMC: 9 adversaries attack, disconnect,
        and reconnect under `_newK` identities across 4 cycles. No benign
        churn, no strategy switching — identity reset is the SOLE active
        confounder. This is the defining RMC mechanic.
    S4 (full_mix)               — all three confounders simultaneously
        (benign churn + strategy switching + identity reset); the
        deployment-realistic, thesis-target scenario (H2 pre-registered
        threshold references S4 specifically).

    Every S0-S4 generator uses the LOWER-slot adversary convention
    (client_0..client_{n_adversaries-1} = the 9 adversaries; see
    flowerfl/task.py DATASET_CONFIGS `malicious_order`) for consistency
    across the family, and is parameter-locked (raises ValueError if called
    with anything other than the spec's n_adversaries=9/n_clients=20/
    n_rounds=50/cycle-count values) since these are pre-registered scenarios,
    not free-form generators.

Logical-identity / physical-client mapping and the substitution rule:
    Every scenario dict's top-level `clients` map assigns each logical
    identity (e.g. "client_3", "client_3_new1", "honest_11_offline2") a
    `physical_id` — an index into the underlying data partition (see
    flowerfl/task.py's `_rmc` dataset variants, which each add ONE extra
    duplicate partition to back a single reconnect identity, or in the
    Design D generators below, a distinct physical_id slot per cycle).
    Reconnect identities SUBSTITUTE for their original in the `participants`
    list for the rounds they're active — the original and its `_newK`
    replacement are never both listed as participants in the same round;
    that would double-count one physical client as two simultaneous
    connections, which the RMC threat model does not represent (a
    reconnecting client is the SAME physical device claiming a new logical
    identity, not a clone of itself).

    The consuming side (flowerfl/scenario_strategy.py::ScenarioStrategy.
    LOGICAL_TO_PARTITION) is a fixed lookup table, not auto-derived from
    each scenario's `clients` map — every logical_id a generator here emits
    into a `participants` list MUST already have a LOGICAL_TO_PARTITION
    entry, or ScenarioStrategy raises ValueError at schedule-build time
    (hard error, not a silent skip — a 2026-05-26 finding showed that
    silently dropping an unrecognized logical_id silently drops scheduled
    adversaries and corrupts the threat model). When adding a new reconnect
    identity naming convention here, LOGICAL_TO_PARTITION must be extended
    to match.

The two older generator families (v1, v2) remain below for
backward-compatibility / provenance of earlier (pre-Design-D) results; new
scenario work should use the Design D family or the spec-driven
generate_scenario_from_spec() path (see "Spec-driven generation" section
near the bottom of this file).

Two generators (v1, v2 — see Design D summary above for the current family):

1. `generate_intensity_scenario` (v1 — SINGLE-CYCLE; methodologically incomplete)
   Schedule (single warmup-attack-disconnect-rejoin cycle in 50 rounds):
     Rounds  1-10  : warmup, all clients honest
     Rounds 11-20  : initial gaussian-noise attack from all malicious clients
     Rounds 21-30  : forced disconnect of N adversaries
     Rounds 31-50  : post-rejoin ALIE attack from N _new identities

   NOTE: This is what plan v2 Task 5 generated. Szelag 2504.03077v1 (§V Results)
   actually uses CONTINUOUS reconnection ("reconnections on separate malicious
   clients to occur immediately after forcible disconnections"). The single-cycle
   pattern below is a methodological simplification that produces fewer
   cumulative adversary-session-rounds than Szelag's continuous design.
   Retained for backward-compatibility; prefer v2 below for new work.

2. `generate_continuous_reconnect_scenario` (v2 — MULTI-CYCLE; closer to Szelag)
   Schedule (warmup → initial Gaussian → N reconnect cycles in 50 rounds):
     Rounds  1-5   : warmup, all clients honest
     Rounds  6-10  : initial gaussian-noise attack from N adversaries
     Cycle 1: rounds 11-12 disconnect, 13-20 ALIE from `_new1` identities
     Cycle 2: rounds 21-22 disconnect, 23-30 ALIE from `_new2` identities
     Cycle 3: rounds 31-32 disconnect, 33-40 ALIE from `_new3` identities
     Cycle 4: rounds 41-42 disconnect, 43-50 ALIE from `_new4` identities

   For N=4 cycles (default): 9 initial + 9*4 = 45 unique malicious identities
   over the simulation, matching Szelag's "many adversaries via continuous
   reconnection" cumulative-count framing. Each `client_X_newK` maps to a
   distinct physical_id slot so Flower treats reconnections as new sessions.

Usage:
    from generate_scenarios import generate_intensity_scenario, generate_continuous_reconnect_scenario
    # v1 single-cycle
    sc1 = generate_intensity_scenario(n_adversaries=9, n_clients=20, n_rounds=50)
    # v2 continuous reconnect
    sc2 = generate_continuous_reconnect_scenario(n_adversaries=9, n_clients=20, n_rounds=50, reconnect_cycles=4)

CLI:
    # v1 (legacy)
    conda run -n flowerfl python scripts/data/generate_scenarios.py \\
        --intensities 1,3,5,7,9 --n-clients 20 --n-rounds 50 \\
        --out-dir rmc/scenarios

    # v2 continuous reconnect (recommended for new work)
    conda run -n flowerfl python scripts/data/generate_scenarios.py \\
        --intensities 1,3,5,7,9 --n-clients 20 --n-rounds 50 \\
        --reconnect-cycles 4 --out-dir rmc/scenarios --schema v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random as _random
import sys
from pathlib import Path
from typing import Any

# M1: named module-level constants replacing magic numbers (spec § 4.4).
DEFAULT_DISCOVERY_ROUNDS = 2  # spec § 4.4 — leading honest rounds for client→partition discovery


def generate_intensity_scenario(
    n_adversaries: int,
    n_clients: int = 20,
    n_rounds: int = 50,
    scenario_name: str | None = None,
    alie_z_max: float = 0.9,
) -> dict[str, Any]:
    """Generate a Szelag-faithful scenario at a given adversary intensity.

    Args:
        n_adversaries: Number of clients that disconnect-and-rejoin with ALIE.
        n_clients: Total number of clients (honest + adversaries).
        n_rounds: Total simulation rounds.
        scenario_name: Optional name; defaults to "rmc_intensity_{N}_v1".
        alie_z_max: ALIE attack strength parameter. Canonical = 0.9 (Szelag-exact, spec § 4.2).
            Sensitivity arm = 0.5 (opt-in only).

    Returns:
        Scenario dict ready for json.dump with the standard schema.
    """
    if n_adversaries < 0 or n_adversaries > n_clients:
        raise ValueError(f"n_adversaries={n_adversaries} out of [0, {n_clients}]")
    if n_rounds < 50:
        raise ValueError("Szelag schedule requires at least 50 rounds")

    name = scenario_name or f"rmc_intensity_{n_adversaries}_v1"
    n_honest = n_clients - n_adversaries

    # Client roster
    clients: dict[str, dict[str, int]] = {}
    for i in range(n_clients):
        clients[f"client_{i}"] = {"physical_id": i}
    # Add _new identities for each adversary (occupy physical_ids n_clients..n_clients+n_adversaries-1)
    for i, adv_idx in enumerate(range(n_honest, n_clients)):
        clients[f"client_{adv_idx}_new"] = {"physical_id": n_clients + i}

    all_clients = [f"client_{i}" for i in range(n_clients)]
    honest_clients = [f"client_{i}" for i in range(n_honest)]
    adv_clients = [f"client_{i}" for i in range(n_honest, n_clients)]
    adv_new_clients = [f"client_{i}_new" for i in range(n_honest, n_clients)]

    # Schedule per Szelag structure
    schedule: list[dict[str, Any]] = []

    # Phase 1: Warmup rounds 1-10 — all honest
    schedule.append({
        "rounds": [1, 10],
        "participants": all_clients,
        "attacks": {},
        "comment": f"Warmup: all {n_clients} clients honest"
    })

    # Phase 2: Initial gaussian-noise attack rounds 11-20
    schedule.append({
        "rounds": [11, 20],
        "participants": all_clients,
        "attacks": {
            cid: {"type": "gaussian_noise", "params": {"mean": 0.0, "std": 2.0}}
            for cid in adv_clients
        },
        "comment": f"Initial loud Gaussian attack from {n_adversaries} adversaries"
    })

    # Phase 3: Forced disconnect rounds 21-30 — adversaries absent
    schedule.append({
        "rounds": [21, 30],
        "participants": honest_clients,
        "attacks": {},
        "comment": f"Forced disconnect of {n_adversaries} adversaries"
    })

    # Phase 4: Post-rejoin ALIE rounds 31-50 — adversaries return with new identities
    schedule.append({
        "rounds": [31, n_rounds],
        "participants": honest_clients + adv_new_clients,
        "attacks": {
            cid: {"type": "alie", "params": {"z_max": alie_z_max}}
            for cid in adv_new_clients
        },
        "comment": f"Post-rejoin ALIE attack from {n_adversaries} _new identities (z_max={alie_z_max})"
    })

    return {
        "name": name,
        "description": (
            f"Generated by scripts/data/generate_scenarios.py at intensity={n_adversaries}/"
            f"{n_clients} = {100 * n_adversaries / n_clients:.0f}% malicious. "
            f"Szelag-style schedule: warmup → initial Gaussian → forced disconnect → ALIE rejoin. "
            f"SINGLE-CYCLE pattern; Szelag's actual paper uses CONTINUOUS reconnection — see "
            f"generate_continuous_reconnect_scenario for the v2 multi-cycle generator."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": 42,
        "clients": clients,
        "schedule": schedule,
    }


def generate_continuous_reconnect_scenario(
    n_adversaries: int,
    n_clients: int = 20,
    n_rounds: int = 50,
    reconnect_cycles: int = 4,
    scenario_name: str | None = None,
    alie_z_max: float = 0.9,
    warmup_rounds: int = 5,
    initial_gaussian_rounds: int = 5,
    disconnect_rounds: int = 2,
    honest_disconnect_cycles: int = 0,
    honest_reconnect_window: tuple[int, int] = (10, 45),
    honest_cycle_size: int = 1,
) -> dict[str, Any]:
    """Generate a Szelag-faithful CONTINUOUS-RECONNECT scenario.

    Closer to Szelag 2504.03077v1 §V: "we allow for reconnections on separate
    malicious clients to occur immediately after forcible disconnections."

    Schedule (for default n_rounds=50, reconnect_cycles=4):
        Rounds  1- 5: warmup, all clients honest
        Rounds  6-10: initial Gaussian noise attack from N adversaries (`client_X` IDs)
        Per cycle K in {1..reconnect_cycles}:
            disconnect: 2 rounds, adversaries absent
            ALIE attack: remainder rounds, adversaries return as `client_X_newK`

    Each cycle uses a fresh logical identity suffix (`_new1`, `_new2`, ...);
    each suffix maps to a distinct physical_id slot so Flower treats every
    reconnection as a new session. The data partition is the same physical
    device across all suffix variants (model the RMC "same device, new identity"
    threat). Cumulative malicious-identity sessions = n_adversaries × (1 + reconnect_cycles).

    Args:
        n_adversaries: Number of physical adversaries.
        n_clients: Total clients including the n_adversaries.
        n_rounds: Total simulation rounds.
        reconnect_cycles: Number of disconnect-reconnect cycles after the initial attack.
        scenario_name: Optional name; defaults to "rmc_intensity_{N}_continuous_v2".
        alie_z_max: ALIE attack strength. Canonical = 0.9 (Szelag-exact, spec § 4.2).
            Sensitivity arm = 0.5 (opt-in only).
        warmup_rounds: Rounds where all clients honest (default 5).
        initial_gaussian_rounds: Rounds for the initial Gaussian wave (default 5).
        disconnect_rounds: Rounds the adversaries are absent each cycle (default 2).
        honest_disconnect_cycles: When > 0, add this many honest disconnect/
            reconnect cycles into the schedule. Each honest cycle: an honest
            client randomly disconnects (1-2 rounds absent) and rejoins under a
            new logical identity `honest_K_offlineN`. Stochastic schedule
            sampled at generation time; reproducible given the same parameters
            and `scenario_name` (used as the RNG seed via hash).
        honest_reconnect_window: (round_lo, round_hi) range within which honest
            reconnect events may be scheduled. Default (10, 45) avoids the
            warmup + initial-attack phases.
        honest_cycle_size: Number of distinct honest clients reconnecting per
            cycle (default 1). Same physical client may be selected in multiple
            cycles — each cycle creates a NEW `_offlineN` identity (e.g.,
            `honest_10_offline2` and `honest_10_offline3` are different sessions
            of the same physical device).

    Returns:
        Scenario dict ready for json.dump with the same top-level schema as v1.
    """
    if n_adversaries < 0 or n_adversaries > n_clients:
        raise ValueError(f"n_adversaries={n_adversaries} out of [0, {n_clients}]")
    if reconnect_cycles < 1:
        raise ValueError("reconnect_cycles must be >= 1")

    fixed_overhead = warmup_rounds + initial_gaussian_rounds
    cycle_budget_total = n_rounds - fixed_overhead
    if cycle_budget_total < reconnect_cycles * (disconnect_rounds + 1):
        raise ValueError(
            f"n_rounds={n_rounds} too small for {reconnect_cycles} cycles "
            f"with disconnect_rounds={disconnect_rounds} each "
            f"(need at least {fixed_overhead + reconnect_cycles * (disconnect_rounds + 1)})"
        )

    name = scenario_name or f"rmc_intensity_{n_adversaries}_continuous_v2"
    n_honest = n_clients - n_adversaries

    # Build client roster: originals + N copies of each adversary identity per cycle
    clients: dict[str, dict[str, int]] = {}
    for i in range(n_clients):
        clients[f"client_{i}"] = {"physical_id": i}
    # Assign physical_id slots for each cycle's `_newK` identities
    # Slot layout: originals 0..n_clients-1, then per-cycle blocks of n_adversaries
    next_slot = n_clients
    new_identities_by_cycle: list[list[str]] = []
    for cycle_idx in range(1, reconnect_cycles + 1):
        cycle_new_ids: list[str] = []
        for adv_idx in range(n_honest, n_clients):
            new_id = f"client_{adv_idx}_new{cycle_idx}"
            clients[new_id] = {"physical_id": next_slot}
            cycle_new_ids.append(new_id)
            next_slot += 1
        new_identities_by_cycle.append(cycle_new_ids)

    # Seed a local RNG deterministically from scenario_name (when provided) so
    # multiple invocations with the same name produce identical honest schedules.
    rng_name = scenario_name or f"rmc_intensity_{n_adversaries}_continuous_v2"
    rng_seed = int(hashlib.sha256(rng_name.encode()).hexdigest()[:8], 16)
    rng = _random.Random(rng_seed)

    honest_events: list[dict[str, Any]] = []
    if honest_disconnect_cycles > 0:
        if honest_cycle_size < 1 or honest_cycle_size > n_honest:
            raise ValueError(
                f"honest_cycle_size={honest_cycle_size} must be in [1, n_honest={n_honest}]; "
                f"got n_honest = n_clients - n_adversaries = {n_clients} - {n_adversaries}"
            )
        honest_indices = list(range(n_honest))
        win_lo, win_hi = honest_reconnect_window
        if win_lo < warmup_rounds + initial_gaussian_rounds + 1:
            win_lo = warmup_rounds + initial_gaussian_rounds + 1
        if win_hi > n_rounds:
            win_hi = n_rounds
        if win_lo >= win_hi:
            raise ValueError(
                f"honest_reconnect_window {honest_reconnect_window} has no rounds "
                f"after warmup+initial; window must include rounds > {warmup_rounds + initial_gaussian_rounds}"
            )
        if win_hi - win_lo < 2:
            raise ValueError(
                f"honest_reconnect_window {honest_reconnect_window} (after clamping to "
                f"[{win_lo}, {win_hi}]) is too narrow; need win_hi - win_lo >= 2 to "
                f"allow at least one offline round."
            )

        # Pick rounds and victims
        for cycle_idx in range(honest_disconnect_cycles):
            # Choose victims for this cycle (uniform without replacement from the
            # not-already-rotated honest pool; with replacement across cycles is OK
            # since each cycle creates a NEW _offlineN identity).
            victims = rng.sample(honest_indices, k=honest_cycle_size)
            offline_round = rng.randint(win_lo, win_hi - 2)  # leave 2 rounds for offline+rejoin
            rejoin_round = offline_round + rng.randint(1, 2)  # 1 or 2 rounds offline
            if rejoin_round > n_rounds:
                rejoin_round = n_rounds

            for victim in victims:
                offline_id = f"honest_{victim}_offline{cycle_idx + 1}"
                clients[offline_id] = {"physical_id": next_slot}
                next_slot += 1
                honest_events.append({
                    "victim_original_id": f"client_{victim}",
                    "offline_id": offline_id,
                    "offline_round": offline_round,
                    "rejoin_round": rejoin_round,
                })

    all_clients = [f"client_{i}" for i in range(n_clients)]
    honest_clients = [f"client_{i}" for i in range(n_honest)]
    adv_clients = [f"client_{i}" for i in range(n_honest, n_clients)]

    schedule: list[dict[str, Any]] = []

    # Phase 1: Warmup
    schedule.append({
        "rounds": [1, warmup_rounds],
        "participants": all_clients,
        "attacks": {},
        "comment": f"Warmup: all {n_clients} clients honest"
    })

    # Phase 2: Initial Gaussian-noise wave
    init_start = warmup_rounds + 1
    init_end = warmup_rounds + initial_gaussian_rounds
    schedule.append({
        "rounds": [init_start, init_end],
        "participants": all_clients,
        "attacks": {
            cid: {"type": "gaussian_noise", "params": {"mean": 0.0, "std": 2.0}}
            for cid in adv_clients
        },
        "comment": f"Initial Gaussian attack from {n_adversaries} adversaries (original identities)"
    })

    # Phase 3+: N reconnect cycles. Last cycle absorbs the remainder.
    cursor = init_end + 1
    per_cycle_total = cycle_budget_total // reconnect_cycles
    remainder = cycle_budget_total - per_cycle_total * reconnect_cycles
    for cycle_idx in range(reconnect_cycles):
        cycle_total = per_cycle_total + (remainder if cycle_idx == reconnect_cycles - 1 else 0)
        alie_rounds = cycle_total - disconnect_rounds

        # Disconnect sub-phase: adversaries absent
        disc_start = cursor
        disc_end = cursor + disconnect_rounds - 1
        schedule.append({
            "rounds": [disc_start, disc_end],
            "participants": honest_clients,
            "attacks": {},
            "comment": f"Cycle {cycle_idx + 1} disconnect: {n_adversaries} adversaries absent"
        })

        # ALIE sub-phase: adversaries return with this cycle's `_newK` identities
        alie_start = disc_end + 1
        alie_end = alie_start + alie_rounds - 1
        cycle_ids = new_identities_by_cycle[cycle_idx]
        schedule.append({
            "rounds": [alie_start, alie_end],
            "participants": honest_clients + cycle_ids,
            "attacks": {
                cid: {"type": "alie", "params": {"z_max": alie_z_max}}
                for cid in cycle_ids
            },
            "comment": (
                f"Cycle {cycle_idx + 1} ALIE: {n_adversaries} adversaries return "
                f"as `_new{cycle_idx + 1}` identities (z_max={alie_z_max})"
            )
        })

        cursor = alie_end + 1

    # Honest disconnect/reconnect events. These are sparse METADATA-ONLY entries
    # that downstream consumers (Task 4c metric extractor) read from the separate
    # `honest_events` field. They MUST NOT participate in scheduling — the
    # adversarial schedule already covers these rounds, and ScenarioStrategy's
    # `_build_schedule` REPLACES the per-round cache for each entry. The
    # `skip_scheduling: True` flag tells ScenarioStrategy to ignore these entries
    # when populating its round → entries cache (otherwise an empty
    # `participants: []` would boot all clients for the offline window).
    for ev in honest_events:
        schedule.append({
            "rounds": [ev["offline_round"], ev["rejoin_round"] - 1],
            "participants": [],  # metadata-only; see skip_scheduling below
            "attacks": {},
            "comment": f"Honest disconnect: {ev['victim_original_id']} offline rounds {ev['offline_round']}..{ev['rejoin_round'] - 1}",
            "honest_reconnect": True,
            "skip_scheduling": True,
            "honest_offline_round": ev["offline_round"],
            "honest_rejoin_round": ev["rejoin_round"],
            "honest_victim_original": ev["victim_original_id"],
            "honest_offline_id": ev["offline_id"],
        })
        schedule.append({
            "rounds": [ev["rejoin_round"], n_rounds],
            "participants": [ev["offline_id"]],  # metadata-only; see skip_scheduling below
            "attacks": {},
            "comment": f"Honest rejoin: {ev['victim_original_id']} returns as {ev['offline_id']} at round {ev['rejoin_round']}",
            "honest_reconnect": True,
            "skip_scheduling": True,
            "honest_rejoin_round": ev["rejoin_round"],
            "honest_victim_original": ev["victim_original_id"],
            "honest_offline_id": ev["offline_id"],
        })

    return {
        "name": name,
        "description": (
            f"Continuous-reconnect scenario at intensity={n_adversaries}/{n_clients} = "
            f"{100 * n_adversaries / n_clients:.0f}% per-round malicious. "
            f"{reconnect_cycles} reconnect cycles in {n_rounds} rounds → "
            f"{n_adversaries * (1 + reconnect_cycles)} cumulative malicious-identity sessions "
            f"(1 initial + {reconnect_cycles} reconnect waves). "
            f"Closer to Szelag 2504.03077v1 §V continuous-reconnection threat model."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": 42,
        "clients": clients,
        "schedule": schedule,
        "honest_events": honest_events,
    }


def generate_clean_baseline_scenario(
    n_adversaries: int = 9,
    n_clients: int = 20,
    n_rounds: int = 50,
    alie_z_max: float = 0.9,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """Design D Scenario S0 — clean-baseline.

    No benign churn, no strategy switching, no identity reset. Single sustained
    ALIE attack stream from the original adversary identities. Yields the
    detection-capacity ceiling per spec § 5.

    Schedule:
        - rounds 1..discovery_rounds: all honest (discovery)
        - rounds discovery_rounds+1..n_rounds: n_adversaries clients
          (client_0..client_{n_adversaries-1}) run ALIE z_max=alie_z_max
    """
    if n_adversaries < 0 or n_adversaries > n_clients:
        raise ValueError(
            f"n_adversaries={n_adversaries} out of [0, {n_clients}]"
        )
    name = scenario_name or "S0_clean_baseline"

    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    all_cids = [f"client_{i}" for i in range(n_clients)]
    mal_cids = [f"client_{i}" for i in range(n_adversaries)]

    schedule = [
        {
            "rounds": [1, discovery_rounds],
            "participants": all_cids,
            "attacks": {},
            "comment": f"Discovery rounds 1..{discovery_rounds} — all honest",
        },
        {
            "rounds": [discovery_rounds + 1, n_rounds],
            "participants": all_cids,
            "attacks": {
                cid: {"type": "alie", "params": {"z_max": alie_z_max}}
                for cid in mal_cids
            },
            "comment": (
                f"S0 sustained ALIE attack from {n_adversaries} adversaries "
                f"(z_max={alie_z_max}); no reconnect, no strategy switch"
            ),
        },
    ]

    return {
        "name": name,
        "description": (
            "Design D Scenario S0 (clean baseline). 45% adversaries via sustained "
            f"ALIE (z_max={alie_z_max}). No benign churn, no strategy switching, "
            "no identity reset. Detection-capacity ceiling per spec § 5."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


def generate_clean_no_attack_scenario(
    n_clients: int = 20,
    n_rounds: int = 50,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """C0 — clean NO-ATTACK utility reference (v1.10 § 5.0 D1, erratum-A E1).

    S0's EXACT population/rounds/schedule shape with ZERO malicious clients:
    the same 20 original identities, the same two-block schedule (discovery +
    one sustained block to n_rounds), full participation every round, and NO
    attack config anywhere — no churn, no switching, no identity reset. This
    is the genuine no-attack utility ceiling the D1 degradation formula
    references (`degradation = acc_final5(C0) − acc_final5(SX)`); S0 is NOT a
    no-attack control (it carries 45% sustained ALIE), which is why C0 exists.

    Parameter-locked like the rest of the Design-D family: raises on anything
    other than the spec's 20-client/50-round shape.
    """
    if n_clients != 20:
        raise ValueError(
            f"C0 is locked at n_clients=20 (S0's population shape); "
            f"got n_clients={n_clients}"
        )
    if n_rounds != 50:
        raise ValueError(f"C0 is locked at 50 rounds; got n_rounds={n_rounds}")

    name = scenario_name or "C0_clean_no_attack"
    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    all_cids = [f"client_{i}" for i in range(n_clients)]

    schedule = [
        {
            "rounds": [1, discovery_rounds],
            "participants": all_cids,
            "attacks": {},
            "comment": f"Discovery rounds 1..{discovery_rounds} — all honest",
        },
        {
            "rounds": [discovery_rounds + 1, n_rounds],
            "participants": all_cids,
            "attacks": {},
            "comment": (
                "C0 clean reference: every client honest for the whole run — "
                "no attack, no churn, no switching, no identity reset"
            ),
        },
    ]

    return {
        "name": name,
        "description": (
            "C0 clean no-attack utility reference (v1.10 § 5.0 D1; H4 "
            "erratum-A E1). S0's exact population/rounds/schedule shape with "
            "ZERO malicious clients: 20 honest clients, full participation, "
            "no attack config, no confounders. The D1 degradation formula's "
            "reference cell."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


def generate_benign_churn_only_scenario(
    n_adversaries: int = 9,
    n_clients: int = 20,
    n_rounds: int = 50,
    n_honest_reconnect_cycles: int = 3,
    alie_z_max: float = 0.9,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """Design D Scenario S1 — benign-churn-only.

    9 malicious clients (client_0..client_8) run sustained ALIE (z_max=alie_z_max)
    from original identities throughout the post-discovery run.

    3 honest clients each disconnect for ~5 rounds, then rejoin as
    `client_{N}_new1` — adding 3 honest reconnect identities to the roster.

    Isolates false-positive sensitivity to legitimate honest churn (per spec § 5).

    Args:
        n_adversaries: locked at 9 per spec § 4.1 (45% adversary fraction)
        n_clients: locked at 20 per spec § 4.1
        n_rounds: locked at 50 per spec § 4.4
        n_honest_reconnect_cycles: locked at 3 per spec § 4.3 (v3 confounder design)
        alie_z_max: canonical 0.9 per spec § 4.2
        discovery_rounds: leading honest rounds for client→partition discovery
        seed: scenario RNG seed
        scenario_name: override default \"S1_benign_churn_only\" filename stem
    """
    if n_adversaries != 9 or n_clients != 20:
        raise ValueError(
            "S1 is locked at n_adversaries=9, n_clients=20 per spec § 4.1; "
            f"got n_adversaries={n_adversaries}, n_clients={n_clients}"
        )
    if n_honest_reconnect_cycles != 3:
        raise ValueError(
            "S1 is locked at 3 honest reconnect cycles per spec § 4.3"
        )
    if n_rounds != 50:
        raise ValueError("S1 is locked at 50 rounds per spec § 4.4")

    name = scenario_name or "S1_benign_churn_only"

    # Roster: 20 originals + 3 honest reconnect identities (1 per cycle)
    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    reconnect_pairs = [
        # (honest_base_cid, new_cid, disconnect_round, rejoin_round)
        ("client_11", "client_11_new1", 11, 16),
        ("client_13", "client_13_new1", 23, 28),
        ("client_15", "client_15_new1", 35, 40),
    ]
    physical_slot = n_clients
    for _, new_cid, _, _ in reconnect_pairs:
        clients[new_cid] = {"physical_id": physical_slot}
        physical_slot += 1

    mal_cids = [f"client_{i}" for i in range(n_adversaries)]
    alie_attacks = {
        cid: {"type": "alie", "params": {"z_max": alie_z_max}} for cid in mal_cids
    }

    all_originals = [f"client_{i}" for i in range(n_clients)]

    # Helper: compute the participants list for a given (left, rejoined) state.
    def _active(left: set[str], rejoined: dict[str, str]) -> list[str]:
        """left = originals currently absent; rejoined = {base_cid: new_cid} replacements."""
        result = []
        for cid in all_originals:
            if cid in left:
                # If a replacement has rejoined, it's now active under the new cid.
                if cid in rejoined:
                    result.append(rejoined[cid])
                # else: this original is currently absent (between disconnect and rejoin)
            else:
                result.append(cid)
        return result

    # Build schedule by walking through the 3 reconnect cycles.
    schedule = []

    # Block 0: discovery
    schedule.append(
        {
            "rounds": [1, discovery_rounds],
            "participants": all_originals,
            "attacks": {},
            "comment": f"Discovery rounds 1..{discovery_rounds} — all originals honest",
        }
    )

    # Track state as we move through rounds
    left: set[str] = set()
    rejoined: dict[str, str] = {}
    current_start = discovery_rounds + 1

    # Sort reconnect events by round number to build sequential blocks.
    events = []
    for base, new_cid, disc, rejoin in reconnect_pairs:
        events.append((disc, "disconnect", base, new_cid))
        events.append((rejoin, "rejoin", base, new_cid))
    events.sort()

    for round_no, kind, base, new_cid in events:
        # Block before this transition
        end_of_prev = round_no - 1
        if current_start <= end_of_prev:
            schedule.append(
                {
                    "rounds": [current_start, end_of_prev],
                    "participants": _active(left, rejoined),
                    "attacks": dict(alie_attacks),
                    "comment": (
                        f"S1 stable window: 9 adversaries running ALIE z_max="
                        f"{alie_z_max}, honest churn state: left={sorted(left - set(rejoined.keys()))}, "
                        f"rejoined={sorted(rejoined.values())}"
                    ),
                }
            )

        if kind == "disconnect":
            left.add(base)
        else:  # rejoin
            rejoined[base] = new_cid

        current_start = round_no

    # Final tail block to n_rounds
    if current_start <= n_rounds:
        schedule.append(
            {
                "rounds": [current_start, n_rounds],
                "participants": _active(left, rejoined),
                "attacks": dict(alie_attacks),
                "comment": (
                    f"S1 final window: 9 adversaries ALIE, honest state: "
                    f"left={sorted(left - set(rejoined.keys()))}, "
                    f"rejoined={sorted(rejoined.values())}"
                ),
            }
        )

    return {
        "name": name,
        "description": (
            "Design D Scenario S1 (benign-churn-only). 9 adversaries running "
            f"sustained ALIE (z_max={alie_z_max}) from original identities. "
            f"{n_honest_reconnect_cycles} honest reconnect cycles introduce "
            "client_11_new1, client_13_new1, client_15_new1. No malicious "
            "reconnect, no strategy switching. Isolates false-positive sensitivity "
            "to legitimate churn per spec § 5."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


def generate_adaptive_switching_only_scenario(
    n_adversaries: int = 9,
    n_clients: int = 20,
    n_rounds: int = 50,
    n_switch_cycles: int = 4,
    alie_z_max: float = 0.9,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """Design D Scenario S2 — adaptive-switching-only.

    Malicious clients (client_0..client_{n_adversaries-1}) keep their original
    identities (no `_newK` reconnect) and rotate attack TYPE across cycles:

        cycle 1: gaussian_noise (mean=2.0, std=2.0)
        cycle 2: alie (z_max=alie_z_max)
        cycle 3: label_flip
        cycle 4: gaussian_noise (rotation wraps)
        ...

    Isolates within-identity attack-mode variation — the praxis 'adaptive
    strategy switching' confounder.

    No benign churn, no identity reset.

    Args:
        n_adversaries: locked at 9 per spec § 4.1
        n_clients: locked at 20 per spec § 4.1
        n_rounds: locked at 50 per spec § 4.4
        n_switch_cycles: locked at 4 per spec § 4.2 strategy-switching variant
        alie_z_max: canonical 0.9 per spec § 4.2
        discovery_rounds: leading honest rounds for client→partition discovery
        seed: scenario RNG seed
        scenario_name: override default \"S2_adaptive_switching_only\" filename stem
    """
    if n_adversaries != 9 or n_clients != 20:
        raise ValueError(
            "S2 is locked at n_adversaries=9, n_clients=20 per spec § 4.1; "
            f"got n_adversaries={n_adversaries}, n_clients={n_clients}"
        )
    if n_rounds != 50:
        raise ValueError("S2 is locked at 50 rounds per spec § 4.4")
    if n_switch_cycles < 3:
        raise ValueError(
            "S2 requires at least 3 switch cycles to exercise the full rotation "
            f"(gaussian → ALIE → label_flip); got {n_switch_cycles}"
        )

    name = scenario_name or "S2_adaptive_switching_only"

    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    all_cids = [f"client_{i}" for i in range(n_clients)]
    mal_cids = [f"client_{i}" for i in range(n_adversaries)]

    rotation: list[tuple[str, dict]] = [
        ("gaussian_noise", {"mean": 2.0, "std": 2.0}),
        ("alie", {"z_max": alie_z_max}),
        ("label_flip", {}),
    ]

    schedule = [
        {
            "rounds": [1, discovery_rounds],
            "participants": all_cids,
            "attacks": {},
            "comment": f"Discovery rounds 1..{discovery_rounds} — all honest",
        }
    ]

    # Evenly distribute attack rounds across n_switch_cycles cycles
    attack_rounds_start = discovery_rounds + 1
    attack_rounds_total = n_rounds - discovery_rounds
    per_cycle = attack_rounds_total // n_switch_cycles
    remainder = attack_rounds_total % n_switch_cycles

    cursor = attack_rounds_start
    for cycle in range(n_switch_cycles):
        atype, params = rotation[cycle % len(rotation)]
        # Distribute remainder rounds across the first `remainder` cycles
        block_len = per_cycle + (1 if cycle < remainder else 0)
        end = min(n_rounds, cursor + block_len - 1)
        if cursor > n_rounds:
            break
        schedule.append(
            {
                "rounds": [cursor, end],
                "participants": all_cids,
                "attacks": {
                    cid: {"type": atype, "params": params} for cid in mal_cids
                },
                "comment": (
                    f"S2 cycle {cycle + 1}: 9 adversaries rotate to attack_type="
                    f"{atype} (rounds {cursor}..{end})"
                ),
            }
        )
        cursor = end + 1

    return {
        "name": name,
        "description": (
            "Design D Scenario S2 (adaptive-switching-only). 45% adversaries "
            "keep original identities; rotate Gaussian noise → ALIE (z_max="
            f"{alie_z_max}) → label_flip across {n_switch_cycles} cycles. "
            "No benign churn, no identity reset. Isolates within-identity "
            "attack variation per spec § 5."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


def generate_identity_reset_only_scenario(
    n_adversaries: int = 9,
    n_clients: int = 20,
    n_rounds: int = 50,
    reconnect_cycles: int = 4,
    alie_z_max: float = 0.9,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """Design D Scenario S3 — identity-reset-only (pure RMC).

    9 malicious clients (client_0..client_{n_adversaries-1}) attack with the
    standard Gaussian-then-ALIE pattern, then reconnect with `_new{K}` identities
    across `reconnect_cycles` cycles. Honest clients stay attached throughout.

    No benign churn, no strategy switching. Identity reset is the SOLE active
    confounder — the defining feature of RMC.

    Uses the LOWER-slot adversary convention (client_0..client_{n_adversaries-1})
    for consistency with S0, S1, S2.

    Args:
        n_adversaries: locked at 9 per spec § 4.1
        n_clients: locked at 20 per spec § 4.1
        n_rounds: locked at 50 per spec § 4.4
        reconnect_cycles: locked at 4 per spec § 4.3 (RMC v1.0 cycle count)
        alie_z_max: canonical 0.9 per spec § 4.2
        discovery_rounds: leading honest rounds for client→partition discovery
        seed: scenario RNG seed
        scenario_name: override default "S3_identity_reset_only" filename stem
    """
    if n_adversaries != 9 or n_clients != 20:
        raise ValueError(
            "S3 is locked at n_adversaries=9, n_clients=20 per spec § 4.1; "
            f"got n_adversaries={n_adversaries}, n_clients={n_clients}"
        )
    if n_rounds != 50:
        raise ValueError("S3 is locked at 50 rounds per spec § 4.4")
    if reconnect_cycles != 4:
        raise ValueError(
            f"S3 is locked at 4 reconnect cycles per spec § 4.3 RMC v1.0; "
            f"got {reconnect_cycles}"
        )

    name = scenario_name or "S3_identity_reset_only"

    # LOWER-slot convention: adversaries client_0..client_{n_adversaries-1}
    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    mal_cids = [f"client_{i}" for i in range(n_adversaries)]
    honest_cids = [f"client_{i}" for i in range(n_adversaries, n_clients)]

    # Add malicious reconnect identities (client_0_new1..client_8_new{K})
    physical_slot = n_clients
    cycle_new_ids: list[list[str]] = []
    for cycle in range(1, reconnect_cycles + 1):
        ids_this_cycle = []
        for adv_idx in range(n_adversaries):
            new_id = f"client_{adv_idx}_new{cycle}"
            clients[new_id] = {"physical_id": physical_slot}
            ids_this_cycle.append(new_id)
            physical_slot += 1
        cycle_new_ids.append(ids_this_cycle)

    # Schedule layout:
    #   Block 0: discovery
    #   Block 1: original Gaussian preface
    #   For each cycle k = 1..reconnect_cycles:
    #     Block (2k): disconnect (adversaries absent for ~1 round)
    #     Block (2k+1): _newK ALIE attack
    #
    # Divide post-discovery + post-Gaussian rounds across reconnect_cycles cycles.

    schedule: list[dict] = []

    # Block 0: discovery rounds
    schedule.append({
        "rounds": [1, discovery_rounds],
        "participants": list(clients.keys())[:n_clients],  # only originals
        "attacks": {},
        "comment": f"Discovery rounds 1..{discovery_rounds} — all originals honest",
    })

    # Define per-cycle round bands.
    # Allocate: gaussian preface = 5 rounds, then 4 cycles of equal length.
    gaussian_start = discovery_rounds + 1
    gaussian_end = gaussian_start + 4  # 5 rounds of gaussian preface
    post_gaussian_rounds = n_rounds - gaussian_end
    per_cycle = post_gaussian_rounds // reconnect_cycles
    if per_cycle < 4:
        raise ValueError("not enough rounds to allocate cycles; check arithmetic")

    # Block 1: gaussian preface from ORIGINAL adversary identities
    schedule.append({
        "rounds": [gaussian_start, gaussian_end],
        "participants": list(clients.keys())[:n_clients],
        "attacks": {
            cid: {"type": "gaussian_noise", "params": {"mean": 2.0, "std": 2.0}}
            for cid in mal_cids
        },
        "comment": (
            f"Initial Gaussian attack from {n_adversaries} original adversaries"
        ),
    })

    # State-machine active set: starts with original 20 clients.
    # For each cycle:
    #   - Disconnect block: remove the LEAVING ids (mal_originals for cycle 0,
    #     previous cycle's _newK for cycle > 0). mal_originals NEVER re-enter.
    #   - ALIE block: add the new cycle's _newK ids.
    # This mirrors the explicit state-machine in generate_full_mix_scenario so
    # that mal_originals are absent from ALL disconnect blocks after cycle 1.
    active: set[str] = set(list(clients.keys())[:n_clients])  # original 20

    def _sorted_active(s: set[str]) -> list[str]:
        """Numeric-aware sort: client_0, client_1, ..., client_0_new1, ..."""
        return sorted(s, key=lambda c: (int(c.split("_")[1]), c.split("_", 2)[2] if "_new" in c else ""))

    cursor = gaussian_end + 1
    for cycle_idx in range(reconnect_cycles):
        # Disconnect block: remove the adversaries that are LEAVING this cycle.
        if cycle_idx == 0:
            leaving = set(mal_cids)  # original adversaries leave permanently
        else:
            leaving = set(cycle_new_ids[cycle_idx - 1])  # previous _newK leave
        for cid in leaving:
            active.discard(cid)

        # Disconnect block (1 round): adversaries absent, no attacks
        schedule.append({
            "rounds": [cursor, cursor],
            "participants": _sorted_active(active),
            "attacks": {},
            "comment": (
                f"Cycle {cycle_idx + 1} disconnect: "
                f"{n_adversaries} adversaries absent"
            ),
        })
        cursor += 1

        # ALIE block: new cycle's _newK adversaries join
        new_adversaries = cycle_new_ids[cycle_idx]
        for cid in new_adversaries:
            active.add(cid)

        end = cursor + per_cycle - 2 if cycle_idx < reconnect_cycles - 1 else n_rounds
        if cycle_idx == reconnect_cycles - 1:
            end = n_rounds  # last cycle stretches to n_rounds

        schedule.append({
            "rounds": [cursor, end],
            "participants": _sorted_active(active),
            "attacks": {
                cid: {"type": "alie", "params": {"z_max": alie_z_max}}
                for cid in new_adversaries
            },
            "comment": (
                f"Cycle {cycle_idx + 1} ALIE: "
                f"{n_adversaries} adversaries return as `_new{cycle_idx + 1}` "
                f"identities (z_max={alie_z_max})"
            ),
        })
        cursor = end + 1

    return {
        "name": name,
        "description": (
            "Design D Scenario S3 (identity-reset-only). 9 adversaries "
            "(client_0..client_8) attack with Gaussian-then-ALIE, then reconnect "
            f"under `client_N_new{{K}}` identities across {reconnect_cycles} "
            f"cycles (z_max={alie_z_max}). No benign churn, no strategy switching. "
            "Identity reset is the SOLE active confounder per spec § 5."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


def generate_full_mix_scenario(
    n_adversaries: int = 9,
    n_clients: int = 20,
    n_rounds: int = 50,
    malicious_reconnect_cycles: int = 4,
    n_honest_reconnect_cycles: int = 3,
    alie_z_max: float = 0.9,
    discovery_rounds: int = DEFAULT_DISCOVERY_ROUNDS,
    seed: int = 42,
    scenario_name: str | None = None,
) -> dict:
    """Design D Scenario S4 — full-mix deployment-realistic.

    ALL THREE confounders active:
      * Benign churn — 3 honest reconnect cycles (client_11, client_13, client_15
        leave and rejoin as client_{N}_new1 each)
      * Strategy switching — malicious cycles rotate attack types:
          preface=gaussian_noise, cycle1=ALIE, cycle2=label_flip,
          cycle3=gaussian_noise, cycle4=ALIE
      * Identity reset — 4 malicious reconnect cycles (mal_new1..mal_new4)

    This is the thesis-target scenario per spec § 6.2 (H2: TGE >= 0.85 recall
    at FPR <= 0.10 in S4).

    Uses the LOWER-slot adversary convention (client_0..client_{n_adversaries-1})
    consistent with S0/S1/S2/S3.

    Args:
        n_adversaries: locked at 9 per spec § 4.1
        n_clients: locked at 20 per spec § 4.1
        n_rounds: locked at 50 per spec § 4.4
        malicious_reconnect_cycles: locked at 4 per spec § 4.3
        n_honest_reconnect_cycles: locked at 3 per spec § 4.3
        alie_z_max: canonical 0.9 per spec § 4.2
        discovery_rounds: leading honest rounds for client→partition discovery
        seed: scenario RNG seed
        scenario_name: override default "S4_full_mix" filename stem
    """
    if (
        n_adversaries != 9 or n_clients != 20 or n_rounds != 50
        or malicious_reconnect_cycles != 4 or n_honest_reconnect_cycles != 3
    ):
        raise ValueError(
            "S4 is locked at n_adversaries=9, n_clients=20, n_rounds=50, "
            "malicious_reconnect_cycles=4, n_honest_reconnect_cycles=3 per spec § 4"
        )

    name = scenario_name or "S4_full_mix"

    # Roster: 20 originals + 36 mal_newK + 3 honest_newK
    clients = {f"client_{i}": {"physical_id": i} for i in range(n_clients)}
    mal_originals = [f"client_{i}" for i in range(n_adversaries)]
    honest_originals = [f"client_{i}" for i in range(n_adversaries, n_clients)]

    physical_slot = n_clients

    # 4 malicious reconnect cycles
    mal_new_by_cycle: list[list[str]] = []
    for cycle in range(1, malicious_reconnect_cycles + 1):
        cycle_ids = []
        for adv_idx in range(n_adversaries):
            new_id = f"client_{adv_idx}_new{cycle}"
            clients[new_id] = {"physical_id": physical_slot}
            cycle_ids.append(new_id)
            physical_slot += 1
        mal_new_by_cycle.append(cycle_ids)

    # 3 honest reconnect identities — client_11, client_13, client_15 each → _new1
    honest_churn_originals = ["client_11", "client_13", "client_15"]
    honest_new_for_base: dict[str, str] = {}
    for base in honest_churn_originals:
        new_id = f"{base}_new1"
        clients[new_id] = {"physical_id": physical_slot}
        honest_new_for_base[base] = new_id
        physical_slot += 1

    # Rotation lookup (cycle 1 = first index of rotation after the preface)
    rotation: list[tuple[str, dict]] = [
        ("alie", {"z_max": alie_z_max}),
        ("label_flip", {}),
        ("gaussian_noise", {"mean": 2.0, "std": 2.0}),
    ]

    schedule: list[dict] = []

    # State tracking: active set is mutated in-place as blocks are built
    active: set[str] = set(f"client_{i}" for i in range(n_clients))  # originals only

    def _make_block(
        start: int, end: int, attackers_dict: dict, comment: str
    ) -> dict:
        return {
            "rounds": [start, end],
            "participants": sorted(active),
            "attacks": dict(attackers_dict),
            "comment": comment,
        }

    # ───── PHASE: DISCOVERY ─────
    schedule.append(_make_block(1, discovery_rounds, {},
        f"Discovery rounds 1..{discovery_rounds} — all originals honest"))

    # ───── PHASE: GAUSSIAN PREFACE (originals attacking) ─────
    gaussian_attacks = {
        cid: {"type": "gaussian_noise", "params": {"mean": 2.0, "std": 2.0}}
        for cid in mal_originals
    }
    schedule.append(_make_block(3, 6, gaussian_attacks,
        "Gaussian preface from 9 original adversaries"))

    # ───── PHASE: MAL CYCLE 1 (ALIE) ─────
    # r=7: mal_originals disconnect
    for cid in mal_originals:
        active.discard(cid)
    schedule.append(_make_block(7, 7, {},
        "Mal cycle 1 disconnect: original adversaries leave"))

    # r=8..13: mal_new1 (ALIE)
    for cid in mal_new_by_cycle[0]:
        active.add(cid)
    atype, params = rotation[0]
    new1_attacks = {cid: {"type": atype, "params": params} for cid in mal_new_by_cycle[0]}
    schedule.append(_make_block(8, 13, new1_attacks,
        f"Mal cycle 1 ALIE: 9 _new1 adversaries (z_max={alie_z_max})"))

    # ───── PHASE: HONEST CHURN 1 ─────
    # r=14: client_11 leaves
    active.discard("client_11")
    schedule.append(_make_block(14, 14, new1_attacks,
        "Honest churn 1 disconnect: client_11 leaves; mal_new1 ALIE continues"))
    # r=15: still absent
    schedule.append(_make_block(15, 15, new1_attacks,
        "Honest churn 1 absent: client_11 still gone; mal_new1 ALIE continues"))
    # r=16..18: client_11_new1 joins
    active.add(honest_new_for_base["client_11"])
    schedule.append(_make_block(16, 18, new1_attacks,
        "Honest churn 1 rejoin: client_11_new1 active; mal_new1 ALIE continues"))

    # ───── PHASE: MAL CYCLE 2 (label_flip) ─────
    # r=19: mal_new1 leave
    for cid in mal_new_by_cycle[0]:
        active.discard(cid)
    schedule.append(_make_block(19, 19, {},
        "Mal cycle 2 disconnect: _new1 adversaries leave"))

    # r=20..26: mal_new2 (label_flip)
    for cid in mal_new_by_cycle[1]:
        active.add(cid)
    atype, params = rotation[1]
    new2_attacks = {cid: {"type": atype, "params": params} for cid in mal_new_by_cycle[1]}
    schedule.append(_make_block(20, 26, new2_attacks,
        "Mal cycle 2 label_flip: 9 _new2 adversaries"))

    # ───── PHASE: HONEST CHURN 2 ─────
    # r=27: client_13 leaves
    active.discard("client_13")
    schedule.append(_make_block(27, 27, new2_attacks,
        "Honest churn 2 disconnect: client_13 leaves"))
    # r=28..29: absent
    schedule.append(_make_block(28, 29, new2_attacks,
        "Honest churn 2 absent: mal_new2 label_flip continues"))
    # r=30: client_13_new1 joins (briefly before mal cycle 3 starts)
    active.add(honest_new_for_base["client_13"])
    schedule.append(_make_block(30, 30, new2_attacks,
        "Honest churn 2 rejoin: client_13_new1 active; mal_new2 label_flip"))

    # ───── PHASE: MAL CYCLE 3 (gaussian_noise) ─────
    # r=31: mal_new2 leave
    for cid in mal_new_by_cycle[1]:
        active.discard(cid)
    schedule.append(_make_block(31, 31, {},
        "Mal cycle 3 disconnect: _new2 adversaries leave"))

    # r=32..36: mal_new3 (gaussian_noise)
    for cid in mal_new_by_cycle[2]:
        active.add(cid)
    atype, params = rotation[2]
    new3_attacks = {cid: {"type": atype, "params": params} for cid in mal_new_by_cycle[2]}
    schedule.append(_make_block(32, 36, new3_attacks,
        "Mal cycle 3 gaussian_noise: 9 _new3 adversaries"))

    # ───── PHASE: HONEST CHURN 3 ─────
    # r=37: client_15 leaves
    active.discard("client_15")
    schedule.append(_make_block(37, 37, new3_attacks,
        "Honest churn 3 disconnect: client_15 leaves"))
    # r=38..39: absent
    schedule.append(_make_block(38, 39, new3_attacks,
        "Honest churn 3 absent: mal_new3 gaussian_noise continues"))
    # r=40: client_15_new1 joins
    active.add(honest_new_for_base["client_15"])
    schedule.append(_make_block(40, 40, new3_attacks,
        "Honest churn 3 rejoin: client_15_new1 active; mal_new3 gaussian_noise"))

    # ───── PHASE: MAL CYCLE 4 (ALIE) ─────
    # r=41: mal_new3 leave
    for cid in mal_new_by_cycle[2]:
        active.discard(cid)
    schedule.append(_make_block(41, 41, {},
        "Mal cycle 4 disconnect: _new3 adversaries leave"))

    # r=42..50: mal_new4 (ALIE z=0.9); rotation[3 % 3 = 0] = ALIE
    for cid in mal_new_by_cycle[3]:
        active.add(cid)
    atype, params = rotation[0]
    new4_attacks = {cid: {"type": atype, "params": params} for cid in mal_new_by_cycle[3]}
    schedule.append(_make_block(42, 50, new4_attacks,
        f"Mal cycle 4 ALIE: 9 _new4 adversaries (z_max={alie_z_max})"))

    return {
        "name": name,
        "description": (
            "Design D Scenario S4 (full-mix deployment-realistic). All three "
            f"confounders enabled: 4 malicious reconnect cycles (_new1..4), "
            f"3 honest reconnect cycles (client_11_new1, client_13_new1, "
            f"client_15_new1), and attack-type rotation across malicious cycles "
            f"(gaussian_noise → ALIE → label_flip → gaussian_noise → ALIE; "
            f"ALIE z_max={alie_z_max}). Thesis target per spec § 6.2."
        ),
        "dataset": "edge_full_20_rmc",
        "num_rounds": n_rounds,
        "seed": seed,
        "clients": clients,
        "schedule": schedule,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spec-driven generation (Sweep 1)
# ─────────────────────────────────────────────────────────────────────────────

def _spec_to_clean_baseline(spec: dict, params: dict) -> dict:
    """Translate a clean_baseline spec to the generator call."""
    return generate_clean_baseline_scenario(
        n_adversaries=params["n_adversaries"],
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        alie_z_max=params["alie_z_max"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )


def _spec_to_benign_churn_only(spec: dict, params: dict) -> dict:
    """Translate a benign_churn_only spec to the generator call.

    The generator hardcodes the 3 honest churn pairs internally; the spec's
    `honest_churn` array documents them for audit traceability and must match
    what the generator produces in cycle count.  Full per-pair validation is a
    known limitation: the spec records the pairs as declared intent; the
    generator's internal pairs are the authoritative schedule.  The test suite
    cross-checks behavioural equivalence (participants + attacks per block).
    """
    sc = generate_benign_churn_only_scenario(
        n_adversaries=params["n_adversaries"],
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        n_honest_reconnect_cycles=len(params["honest_churn"]),
        alie_z_max=params["alie_z_max"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )
    return sc


def _spec_to_adaptive_switching_only(spec: dict, params: dict) -> dict:
    """Translate an adaptive_switching_only spec to the generator call.

    The spec's `attack_rotation` array documents the rotation for audit
    traceability.  The generator uses its own internal rotation list, which
    must match the spec's entries.  The test suite cross-checks behavioural
    equivalence per block.
    """
    return generate_adaptive_switching_only_scenario(
        n_adversaries=params["n_adversaries"],
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        n_switch_cycles=params["n_switch_cycles"],
        alie_z_max=params["alie_z_max"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )


def _spec_to_identity_reset_only(spec: dict, params: dict) -> dict:
    """Translate an identity_reset_only spec to the generator call."""
    return generate_identity_reset_only_scenario(
        n_adversaries=params["n_adversaries"],
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        reconnect_cycles=params["reconnect_cycles"],
        alie_z_max=params["alie_z_max"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )


def _spec_to_full_mix(spec: dict, params: dict) -> dict:
    """Translate a full_mix spec to the generator call.

    The spec's `schedule_milestones` and `honest_churn` arrays document the
    explicit round layout for audit traceability.  The generator hardcodes the
    same values internally.  The test suite cross-checks behavioural equivalence
    per block rather than relying on the spec values at runtime.
    """
    return generate_full_mix_scenario(
        n_adversaries=params["n_adversaries"],
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        malicious_reconnect_cycles=params["malicious_reconnect_cycles"],
        n_honest_reconnect_cycles=params["n_honest_reconnect_cycles"],
        alie_z_max=params["alie_z_max"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )


def _spec_to_clean_no_attack(spec: dict, params: dict) -> dict:
    """Translate a clean_no_attack spec (C0, erratum-A E1) to the generator."""
    n_adversaries = params.get("n_adversaries", 0)
    if n_adversaries != 0:
        raise ValueError(
            f"clean_no_attack spec declares n_adversaries={n_adversaries}; "
            f"C0 is the ZERO-malicious reference (v1.10 D1)"
        )
    return generate_clean_no_attack_scenario(
        n_clients=params["n_clients"],
        n_rounds=params["n_rounds"],
        discovery_rounds=params["discovery_rounds"],
        seed=params["seed"],
        scenario_name=spec["name"],
    )


_SPEC_DISPATCH: dict[str, Any] = {
    "clean_no_attack": _spec_to_clean_no_attack,
    "clean_baseline": _spec_to_clean_baseline,
    "benign_churn_only": _spec_to_benign_churn_only,
    "adaptive_switching_only": _spec_to_adaptive_switching_only,
    "identity_reset_only": _spec_to_identity_reset_only,
    "full_mix": _spec_to_full_mix,
}


def generate_scenario_from_spec(spec_path: Path) -> dict:
    """Read a scenario spec JSON and emit the expanded scenario dict.

    Dispatches on spec['type'] to the appropriate generator function:
      - clean_baseline        → generate_clean_baseline_scenario
      - benign_churn_only     → generate_benign_churn_only_scenario
      - adaptive_switching_only → generate_adaptive_switching_only_scenario
      - identity_reset_only   → generate_identity_reset_only_scenario
      - full_mix              → generate_full_mix_scenario

    Args:
        spec_path: Path to a scenario spec JSON file
            (e.g. rmc/scenario_specs/S0_clean_baseline.spec.json).

    Returns:
        Expanded scenario dict ready for json.dump.

    Raises:
        ValueError: If the spec's `type` field is not recognised.
        KeyError:   If required keys are missing from `params`.
    """
    spec = json.loads(Path(spec_path).read_text())
    spec_type = spec["type"]
    params = spec["params"]

    fn = _SPEC_DISPATCH.get(spec_type)
    if fn is None:
        raise ValueError(
            f"Unknown scenario spec type: {spec_type!r}. "
            f"Supported types: {sorted(_SPEC_DISPATCH)}"
        )
    return fn(spec, params)


def main() -> int:
    """CLI entry point with two independent generation paths.

    1. `--from-spec SPEC_JSON`: the CURRENT path for Design D (S0-S4)
       scenarios. Reads a spec JSON (see rmc/scenario_specs/ and
       generate_scenario_from_spec() above), dispatches to the matching
       generate_*_scenario() function, and writes the expanded scenario.
       All other CLI args below are ignored when --from-spec is given.
    2. Everything else (--intensities/--schema/--heterogeneous/--n-clients/
       --n-rounds/--reconnect-cycles): the legacy v1/v2/v3 intensity-sweep
       path, sweeping generate_intensity_scenario /
       generate_continuous_reconnect_scenario across multiple adversary
       counts in one invocation. Retained for the older scenario families;
       Design D scenarios are not produced by this path.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intensities", default="1,3,5,7,9",
                   help="Comma-separated adversary counts (default: 1,3,5,7,9)")
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--out-dir", type=Path,
                   default=Path("rmc/scenarios"))
    p.add_argument("--schema", choices=["v1", "v2"], default="v1",
                   help="v1 = single-cycle (legacy); v2 = continuous reconnect (recommended)")
    p.add_argument("--reconnect-cycles", type=int, default=4,
                   help="Number of reconnect cycles for v2 schema (default: 4)")
    p.add_argument("--heterogeneous", action="store_true",
                   help="Generate v3 heterogeneous-reconnect scenarios "
                        "(adds 3 honest reconnect cycles + the existing v2 backbone). "
                        "Filename suffix becomes _heterogeneous_v3.")
    p.add_argument(
        "--from-spec",
        type=Path,
        default=None,
        metavar="SPEC_JSON",
        help="Read a scenario spec JSON and emit the expanded scenario. "
             "Skips all other generation paths.  Use --output to set the "
             "destination path; otherwise the scenario's 'name' field is used.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="OUT_JSON",
        help="Output path for --from-spec (default: <name>.json in cwd).",
    )
    args = p.parse_args()

    if args.from_spec:
        sc = generate_scenario_from_spec(args.from_spec)
        out_path = args.output if args.output else Path(f"{sc['name']}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sc, indent=2))
        print(f"wrote {out_path}")
        return 0

    intensities = [int(x) for x in args.intensities.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.heterogeneous:
        # TODO: expose as CLI args if heterogeneous variants are needed for
        # future ablations (honest_disconnect_cycles, honest_cycle_size,
        # honest_reconnect_window are currently hardcoded below).
        for n in intensities:
            sc = generate_continuous_reconnect_scenario(
                n_adversaries=n,
                n_clients=args.n_clients,
                n_rounds=args.n_rounds,
                reconnect_cycles=args.reconnect_cycles,
                honest_disconnect_cycles=3,
                honest_cycle_size=1,
                honest_reconnect_window=(10, 45),
                scenario_name=f"rmc_intensity_{n}_heterogeneous_v3",
            )
            out_path = args.out_dir / f"rmc_intensity_{n}_heterogeneous_v3.json"
            out_path.write_text(json.dumps(sc, indent=2))
            cumulative = n * (1 + args.reconnect_cycles)
            n_honest_events = sum(1 for e in sc["schedule"] if e.get("honest_reconnect"))
            print(
                f"  {out_path}: {n} per-round × {1 + args.reconnect_cycles} waves = "
                f"{cumulative} adversarial identities + {n_honest_events} honest-reconnect schedule entries"
            )
        return 0

    if args.schema == "v2":
        for n in intensities:
            sc = generate_continuous_reconnect_scenario(
                n_adversaries=n,
                n_clients=args.n_clients,
                n_rounds=args.n_rounds,
                reconnect_cycles=args.reconnect_cycles,
            )
            out_path = args.out_dir / f"{sc['name']}.json"
            out_path.write_text(json.dumps(sc, indent=2))
            cumulative = n * (1 + args.reconnect_cycles)
            print(f"  {out_path}: {n} per-round × {1 + args.reconnect_cycles} waves = {cumulative} cumulative identities")
        return 0

    for n in intensities:
        sc = generate_intensity_scenario(
            n_adversaries=n,
            n_clients=args.n_clients,
            n_rounds=args.n_rounds,
        )
        out_path = args.out_dir / f"{sc['name']}.json"
        out_path.write_text(json.dumps(sc, indent=2))
        n_alie = sum(
            1
            for block in sc["schedule"]
            for cid, atk in (block.get("attacks") or {}).items()
            if atk.get("type") == "alie"
        )
        print(f"  {out_path}: {n_alie} unique ALIE adversaries")

    return 0


if __name__ == "__main__":
    sys.exit(main())

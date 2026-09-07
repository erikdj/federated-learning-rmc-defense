"""
Scenario Strategy for Flower-Native RMC Experiments.

Wraps PluggableStrategy to add scenario-driven client selection and
per-round attack injection via FitIns.config. Loads the same JSON
scenario files used by the standalone rmc/ simulation.

Architecture:
    ScenarioStrategy
        - Extends PluggableStrategy
        - Overrides configure_fit to filter clients per schedule
        - Injects attack_type/attack_sigma into FitIns.config
        - Overrides evaluate for fixed holdout evaluation

CID Mapping:
    Flower 1.25 simulation assigns random 64-bit node IDs as cids.
    We discover the cid→partition-id mapping by requesting all clients
    to report their partition-id in FitRes metrics during round 1.
    From round 2 onwards, we use this mapping for scenario filtering.
"""

import json
import os
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

import numpy as np

from flowerfl.byzantine_defense import (
    ByzantineDefensePlugin,
    PluggableStrategy,
    canonicalize_result_order,
)
from flowerfl.mlflow_live import LiveRoundMetricLogger
from flowerfl.signal_logger import (
    REENTRY_EVENT_FIELDS,
    SignalLogger,
    build_reentry_event_key,
    canonical_device_id,
    compute_per_client_signals,
    flatten_parameters,
    is_reentry_identity,
)

logger = logging.getLogger(__name__)

# Sentinel distinguishing an ABSENT manifest key from a present None in the
# static-row repeat comparison.
_MISSING = object()

# The REGISTRY half of the schema-v5 re-entry contract (v1.10 § 5.1). The
# fingerprint plugin is authoritative for these six; the ground-truth half
# (`reentry_event_key`, `current_cid`, `gt_logical_id`, `gt_is_malicious`) is
# this module's. Both halves are named in `signal_logger.REENTRY_EVENT_FIELDS`,
# which stays the single source — nothing is added to the schema here.
REGISTRY_HALF_FIELDS = (
    "asserted_match",
    "asserted_parent_entry_id",
    "asserted_parent_logical_id",
    "min_d",
    "tau",
    "generation",
)

#: The registry fields the join must carry BEYOND the frozen v1.10 table: the
#: additive nearest-candidate pair (signal_logger.REENTRY_NEAREST_FIELDS,
#: ). Kept as a separate tuple so REGISTRY_HALF_FIELDS stays the frozen
#: six verbatim. EXP-057 smoke finding (2026-08-15): the registry recorded the
#: pair unconditionally and the plugin row carried it, but this module's
#: projection dropped it — every fleet row logged nearest_* as null, and an
#: unmatched event would misread as "empty candidate pool" under the corrected
#: instrument's rank-1 statistic.
REGISTRY_NEAREST_FIELDS = (
    "nearest_entry_id",
    "nearest_logical_id",
)


def adversarial_identities(schedule: list) -> set:
    """Logical ids EVER assigned an attack anywhere in the schedule. Per-identity
    ground truth (F7): such an id is malicious for every round it participates,
    including dormant rounds before it (re)starts attacking."""
    adv = set()
    for block in schedule:
        if block.get("skip_scheduling", False):
            continue
        for cid, spec in (block.get("attacks") or {}).items():
            if spec and spec.get("type"):
                adv.add(cid)
    return adv


def compute_tenure(logical_cid: str, server_round: int, first_seen: dict) -> int:
    """Rounds since this logical identity first participated (>=1). `first_seen`
    is a mutable {logical_cid: first_round} cache owned by the strategy instance.
    A reconnecting identity (new logical id) restarts at tenure 1."""
    if logical_cid not in first_seen:
        first_seen[logical_cid] = server_round
    return server_round - first_seen[logical_cid] + 1


class ScenarioStrategy(PluggableStrategy):
    """
    Strategy that drives Flower-native RMC experiments.

    Loads a scenario JSON file and uses it to:
    1. Select which clients participate each round (configure_fit override)
    2. Inject per-client attack instructions into FitIns.config
    3. Run fixed holdout evaluation after aggregation (evaluate override)

    The scenario JSON maps logical IDs to partition IDs. In Flower simulation,
    each supernode has a partition-id in its node_config. The strategy discovers
    the cid→partition-id mapping at runtime from client-reported metrics.

    Identity remapping: client_9_new maps to partition 10 (a copy of partition 9's
    data), giving it a genuinely different Flower cid for tenure tracking. For
    RMC scenarios using `client_N_newM` cycle identities (v2+), the rejoining
    client maps back to its original partition N — same physical client returning
    under a new identity.
    """

    # Maps scenario logical IDs to Flower partition IDs.
    # Supports:
    #   - 10-client (edge_full_rmc): client_0..client_9 + client_9_new (partition 10)
    #   - 20-client (edge_full_20_rmc): client_0..client_19 + client_19_new (partition 20)
    #   - RMC v2+ cycle identities: client_N_newM (M=1..10) → partition N
    LOGICAL_TO_PARTITION = {
        **{f"client_{i}": i for i in range(21)},
        "client_9_new": 10,
        "client_19_new": 20,
        # RMC cycle identities: same physical partition, new identity.
        **{
            f"client_{n}_new{m}": n
            for n in range(21)
            for m in range(1, 11)
        },
    }

    def __init__(
        self,
        base_strategy: Strategy,
        plugins: List[ByzantineDefensePlugin] = None,
        scenario_path: str = None,
        eval_manager=None,
        signal_logger: Optional[SignalLogger] = None,
        live_metric_logger: Optional[LiveRoundMetricLogger] = None,
    ):
        super().__init__(base_strategy, plugins)
        self._scenario = None
        self._schedule_cache: Dict[int, List[dict]] = {}
        self._eval_manager = eval_manager
        self._signal_logger = signal_logger
        # Live per-round MLflow streaming (instrumentation only; spec Lane C
        # item 10). None unless the unit was launched via `praxis exp launch`.
        # server_app.server_fn wires the env-derived logger via
        # set_live_metric_logger; tests inject a fake through this constructor.
        self._live_metric_logger = live_metric_logger
        # Mapping discovered at runtime: cid -> partition_id
        self._cid_to_partition: Dict[str, int] = {}
        # Reverse: partition_id -> cid
        self._partition_to_cid: Dict[int, str] = {}
        # Reverse: partition_id -> logical_cid (filled from schedule entries)
        self._partition_to_logical: Dict[int, str] = {}
        self._mapping_ready = False

        # Discovery round offset: round 1 is used for cid discovery,
        # so the scenario schedule starts at round 2 (offset=1)
        self._round_offset = 1

        # Defense overhead instrumentation (Task 4c.4): wall-clock per round
        # for the base aggregation call (typically Krum). Measures only the
        # super.aggregate_fit call, not plugin scoring (which lives in
        # each plugin's own _timing_per_round list).
        self._krum_timing_per_round: list[float] = []
        self._tenure_first_seen: dict = {}
        # Stage-F §5: durable server-side collection of the
        # per-client resampling manifest fit-metric, keyed by partition_id. Rows
        # are static per unit, so we keep the first appearance; the runner lifts
        # this dict into the result JSON so completed Stage-F units carry per-
        # client resampling/step evidence (worker stdout + fit metrics alone do
        # NOT persist on the fleet — log_to_driver=False, no other consumer).
        self._resampling_manifest: Dict[int, dict] = {}
        # Stage-F BLOCKER fix: the AUTHORITATIVE set of partition_ids this
        # strategy dispatched for fit across ALL rounds (discovery included).
        # The runner asserts the collected manifest EXACTLY covers this set
        # before persisting a unit result — so a post-dispatch client failure
        # that drops a manifest row FAILS the unit instead of silently
        # producing incomplete per-client evidence. Derived from the dispatch
        # path (not get_num_clients): participation is scenario-driven.
        self._dispatched_partitions: set[int] = set()
        # cids dispatched BEFORE the cid->partition mapping is ready
        # (the discovery round). Flower cids are opaque until fit metrics
        # arrive, so a discovery-round client that FAILS never resolves to a
        # partition and would silently vanish from BOTH the collected manifest
        # and the expected partition set. Recording the dispatched cids lets
        # the gate flag any that never resolved (see unresolved_dispatched_cids).
        self._pre_mapping_dispatched_cids: set[str] = set()

        # H3 (v1.10 § 5.1): the registry half of the re-entry contract joins
        # onto the ground-truth half by `reentry_event_key`, which BOTH sides
        # build as `{run_id}:{round}:{cid}`. The plugin therefore has to stamp
        # the SAME run id the signal logger writes, or the two halves would
        # never join and the D7 metric would have an empty numerator. Guarded by
        # the attribute check: no incumbent plugin exposes `set_run_id`, so this
        # is a no-op for every non-H3 arm.
        run_uid = getattr(signal_logger, "run_uid", None)
        if run_uid is not None:
            for plugin in self._plugins:
                stamp = getattr(plugin, "set_run_id", None)
                if callable(stamp):
                    stamp(run_uid)

        if scenario_path and os.path.exists(scenario_path):
            with open(scenario_path, "r") as f:
                self._scenario = json.load(f)
            self._build_schedule()
            self._adv_ids = adversarial_identities(self._scenario["schedule"])
            print(f"[ScenarioStrategy] Loaded scenario: {self._scenario['name']} "
                  f"({self._scenario['num_rounds']} rounds, "
                  f"Flower rounds 2-{self._scenario['num_rounds'] + 1})")
        else:
            print(f"[ScenarioStrategy] No scenario loaded — all clients participate every round")

    def _build_schedule(self):
        """Pre-compute per-round participant lists with attack configs."""
        num_rounds = self._scenario["num_rounds"]
        for r in range(1, num_rounds + 1):
            self._schedule_cache[r] = []

        # M5: Track which rounds have been covered by non-skip_scheduling blocks.
        # Two blocks may not cover the same round — overlap would silently
        # overwrite, breaking the per-round n_malicious invariant A4 checks.
        # Spec § 4.9 / Bug #1 defense-in-depth.
        covered_rounds: set[int] = set()

        for block in self._scenario["schedule"]:
            # Metadata-only entries (e.g., honest_reconnect annotations from the
            # v3 heterogeneous scenario) carry skip_scheduling: True. Skip them
            # so they don't overwrite the executable adversary schedule for the
            # same rounds — downstream consumers read honest events from the
            # separate `honest_events` field instead.
            if block.get("skip_scheduling", False):
                continue

            start, end = block["rounds"]
            participants = block["participants"]
            attacks = block.get("attacks", {})

            for r in range(start, end + 1):
                if r in covered_rounds:
                    raise ValueError(
                        f"Scenario schedule overlap at round {r}: two "
                        f"non-skip_scheduling blocks cover the same round. "
                        f"Block: rounds=[{start},{end}], comment="
                        f"{block.get('comment', '')!r}. Spec § 4.9 / Bug #1."
                    )
                covered_rounds.add(r)

                entries = []
                for logical_id in participants:
                    partition_id = self.LOGICAL_TO_PARTITION.get(logical_id)
                    if partition_id is None:
                        raise ValueError(
                            f"Unknown logical_id in scenario: {logical_id!r}. "
                            f"Add it to LOGICAL_TO_PARTITION. A silent skip here "
                            f"drops scheduled clients (e.g., RMC adversaries) and "
                            f"corrupts the threat model — see 2026-05-26 finding."
                        )

                    attack_info = attacks.get(logical_id)
                    entry = {
                        "logical_id": logical_id,
                        "partition_id": partition_id,
                        "attack_type": attack_info["type"] if attack_info else None,
                        "attack_params": attack_info.get("params", {}) if attack_info else {},
                    }
                    entries.append(entry)
                    # Populate reverse map opportunistically so confounder-control
                    # post-processing (Task 4c.5) can translate flower_cid →
                    # logical_cid via partition_id.
                    self._partition_to_logical[partition_id] = logical_id
                self._schedule_cache[r] = entries

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Override to filter clients per scenario schedule and inject attack config."""
        self._current_round = server_round

        # Save global parameters for computing client updates in aggregate_fit
        # (needed for real ALIE: update = client_params - global_params)
        self._global_params_ndarrays = parameters_to_ndarrays(parameters)

        # Notify plugins
        num_clients = client_manager.num_available()
        for plugin in self._plugins:
            plugin.on_round_start(server_round, num_clients)

        # Get base configs (all clients)
        base_configs = self._base.configure_fit(server_round, parameters, client_manager)

        # thread the current server round into every client's FitIns.config
        # so the client can derive a per-(client, round) RNG seed in fit. Done at
        # this single point so all downstream branches (discovery, filtered,
        # no-schedule) inherit it via their `dict(fit_ins.config)` copies.
        base_configs = [
            (client_proxy, FitIns(
                parameters=fit_ins.parameters,
                config={**(dict(fit_ins.config) if fit_ins.config else {}),
                        "server_round": server_round},
            ))
            for client_proxy, fit_ins in base_configs
        ]

        # while the cid->partition mapping is not yet ready (the
        # discovery round + any pre-scenario path), EVERY client in base_configs
        # is dispatched but its partition is still unknowable. Record the
        # dispatched cids so the gate can later flag any that never resolved to
        # a partition (a discovery-round failure). Once mapping is ready the
        # partition-level tracking below is authoritative.
        if not self._mapping_ready:
            for client_proxy, _fit_ins in base_configs:
                self._pre_mapping_dispatched_cids.add(client_proxy.cid)

        if not self._scenario:
            return base_configs

        # Round 1: send all clients to discover cid→partition mapping
        if not self._mapping_ready:
            print(f"[ScenarioStrategy] Round {server_round}: discovery round — "
                  f"all {len(base_configs)} clients participate")
            # Inject "report_partition" flag so clients include partition-id in metrics
            configs = []
            for client_proxy, fit_ins in base_configs:
                new_config = dict(fit_ins.config) if fit_ins.config else {}
                new_config["report_partition"] = "1"
                new_config["attack_type"] = ""  # honest for discovery round
                configs.append((client_proxy, FitIns(parameters=fit_ins.parameters, config=new_config)))
            return configs

        # Map Flower round to scenario round (offset by discovery round)
        scenario_round = server_round - self._round_offset
        schedule = self._schedule_cache.get(scenario_round, [])
        if not schedule:
            print(f"[ScenarioStrategy] Round {server_round} (scenario R{scenario_round}): "
                  f"no schedule entry, using all clients")
            # BLOCKER fix: all clients are dispatched here — record every
            # partition whose cid is already mapped so the completeness gate
            # accounts for it.
            for client_proxy, _fit_ins in base_configs:
                pid = self._cid_to_partition.get(client_proxy.cid)
                if pid is not None:
                    self._dispatched_partitions.add(int(pid))
            return base_configs

        # Build set of partition IDs that should participate
        scheduled_partitions = {}
        for entry in schedule:
            scheduled_partitions[entry["partition_id"]] = entry

        # Filter and inject attack config
        filtered_configs = []
        for client_proxy, fit_ins in base_configs:
            partition_id = self._cid_to_partition.get(client_proxy.cid)
            if partition_id is None:
                continue  # unknown client

            if partition_id not in scheduled_partitions:
                continue  # not scheduled this round

            entry = scheduled_partitions[partition_id]

            # Clone the config dict and inject attack instructions
            new_config = dict(fit_ins.config) if fit_ins.config else {}
            if entry["attack_type"]:
                new_config["attack_type"] = entry["attack_type"]
                for k, v in entry["attack_params"].items():
                    new_config[f"attack_{k}"] = v
            else:
                new_config["attack_type"] = ""  # explicit honest marker

            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=new_config)
            filtered_configs.append((client_proxy, new_fit_ins))
            # BLOCKER fix: record this dispatched partition so the unit-level
            # completeness gate can demand exactly one manifest row for it.
            self._dispatched_partitions.add(partition_id)

        attacking = [e["logical_id"] for e in schedule if e["attack_type"]]
        print(f"[ScenarioStrategy] Round {server_round} (scenario R{scenario_round}): "
              f"{len(filtered_configs)}/{len(base_configs)} clients selected "
              f"(attacking: {attacking or 'none'})")

        # Spec § 4.9 integrity marker (A3 + A4 dispatch-time check): emit count
        # of CLIENTS DISPATCHED to training for this round + ground-truth malicious
        # count, so the runner can assert participants and n_malicious match the
        # scenario declaration. This is a dispatch-time check; a post-dispatch
        # Ray failure would not be caught here. Round value is scenario_round
        # (the same indexing used by the scenario JSON's schedule[*].rounds field),
        # so A4 can cross-reference directly. Skip discovery rounds
        # (scenario_round < 1) — no schedule entries to compare.
        if scenario_round >= 1:
            n_malicious_round = sum(
                1 for e in schedule if e.get("attack_type")
            )
            print(
                f"[Integrity] round={scenario_round} "
                f"participants={len(filtered_configs)} "
                f"n_malicious={n_malicious_round}",
                flush=True,
            )
            n_alie_round = sum(
                1 for e in schedule if e.get("attack_type") == "alie"
            )
            if n_alie_round > 0:
                print(
                    f"[Integrity] round={scenario_round} alie_active=1 "
                    f"n_alie={n_alie_round}",
                    flush=True,
                )

        return filtered_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Override to discover cid→partition mapping from round 1 results.

        During the discovery round, we bypass plugin scoring entirely to avoid
        poisoning the tenure tracker — the discovery round includes ALL clients
        (even those not scheduled), which would prematurely set first_seen.
        """
        # Canonicalize arrival order ONCE at this OUTER entry, BEFORE any branching
        # (drift-investigation scoped fix). This is the single consumption point:
        # the discovery-round base aggregation (below), the plugin path via
        # super.aggregate_fit, the ALIE server-side replacement, the identity
        # map, AND _maybe_log_signals all read this SAME ordered list. If the pin
        # lived only in super (PluggableStrategy), the plugins' positional
        # _round_scores (in sorted order) would be joined against _maybe_log_signals
        # iterating the arrival-ordered caller list — silent cross-client
        # misattribution (the v1.17 bug class), corrupting recall@FPR. super's own
        # canonicalize is then an idempotent no-op; empty results pass through.
        if results:
            results = canonicalize_result_order(results, server_round)

        # Stage-F §5 (P1-3): lift the per-client resampling manifest from the fit
        # metrics into a durable server-side dict. Runs on EVERY round (incl.
        # discovery) so a row is captured the first time each partition reports.
        self._collect_resampling_manifest(server_round, results)

        if not self._mapping_ready and results:
            # Extract partition-id from client metrics
            for client_proxy, fit_res in results:
                partition_id = fit_res.metrics.get("partition_id")
                if partition_id is not None:
                    pid = int(partition_id)
                    self._cid_to_partition[client_proxy.cid] = pid
                    self._partition_to_cid[pid] = client_proxy.cid
                    # BLOCKER fix: the discovery round dispatches every client;
                    # record each partition as it is discovered so the gate's
                    # expected set includes discovery participants too.
                    self._dispatched_partitions.add(pid)

            if self._cid_to_partition:
                self._mapping_ready = True
                print(f"[ScenarioStrategy] Discovered cid→partition mapping: "
                      f"{len(self._cid_to_partition)} clients")
                for cid, pid in sorted(self._cid_to_partition.items(), key=lambda x: x[1]):
                    print(f"  partition {pid} → cid {cid}")

            # During discovery round: aggregate WITHOUT plugin scoring
            # to avoid setting first_seen for clients not in the schedule
            print(f"[ScenarioStrategy] Discovery round: bypassing defense plugins")
            return self._base.aggregate_fit(server_round, results, failures)

        # Phase 3c: Real ALIE (Baruch et al. 2019) — server-side replacement
        # For ALIE attacks, the server computes the malicious update from
        # cross-client statistics, since clients can't access other clients'
        # updates. This replaces the client's local result BEFORE aggregation.
        results = self._maybe_apply_real_alie(server_round, results)

        # RMC identity fidelity (methodology v1.15): give every plugin this
        # round's {raw cid -> logical identity} map BEFORE scoring, so
        # stateful defenses key tenure/reputation on the logical identity.
        # Raw simulation cids are stable per virtual client, and cycle
        # identities (client_N_newM) reuse the same partition — without this
        # map an identity reset would NOT reset the defense's per-client
        # state, granting the detector identity linkage the threat model
        # forbids (and decoupling it from the signal log's ground-truth
        # tenure, which is computed from logical_cid).
        identity_map = self._build_identity_map(server_round, results)
        for plugin in self._plugins:
            plugin.set_identity_map(identity_map)

        # Defense overhead bookkeeping (Task 4c.4): time the full plugin
        # pipeline + base aggregation. This is the closest analogue to
        # "Krum aggregation time" for the PluggableStrategy stack.
        _t_start = time.perf_counter()
        # Delegate to parent (PluggableStrategy) for plugin hooks + base aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        _t_elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
        self._krum_timing_per_round.append(_t_elapsed_ms)

        # Emit per-client signal log row (Phase 1 P1.2)
        self._maybe_log_signals(server_round, results)

        return aggregated

    def unresolved_dispatched_cids(self) -> "set[str]":
        """Cids dispatched pre-mapping that never resolved to a partition.

        These are discovery-round clients that were dispatched but never
        returned a successful fit (so their partition was never learned). The
        unit-level completeness gate treats a nonempty result as a violation —
        such a client would otherwise vanish from both the collected manifest
        and the expected partition set, letting a unit succeed with missing
        per-client evidence.
        """
        return set(self._pre_mapping_dispatched_cids) - set(self._cid_to_partition)

    def _collect_resampling_manifest(self, server_round, results) -> None:
        """Collect the per-client resampling manifest fit-metric server-side (§5).

        Rows are static per unit, so the FIRST appearance per partition_id is
        kept. The ``[MANIFEST]`` line is printed DRIVER-side here at collection
        so it reaches the captured log — worker stdout is hidden by
        ``log_to_driver=False`` on the fleet, so the client's own print never
        surfaces.

        BLOCKER fix — the collector is now LOUD. A client that emits a manifest
        metric which is malformed JSON, or a row lacking ``partition_id``, is a
        code bug and evidence loss: it raises ``ValueError`` naming the round
        and client rather than being silently skipped. A REPEAT row for an
        already-collected partition MUST be IDENTICAL (rows are static per
        unit) — a differing repeat raises, naming the pid and the differing
        keys. An IDENTICAL repeat stays an idempotent no-op (no reprint). A
        client that emits NO manifest metric is left to the unit-level
        completeness gate (missing coverage), not raised here.

        the static-row equality is regime-aware on the natural step
        count. When the incoming row's own ``update_match`` field is falsey
        (False or absent) AND ``actual_steps`` is present in BOTH rows, its
        VALUE is excluded from the comparison; presence mismatches ALWAYS
        raise in both regimes — only present-in-both value drift on
        ``actual_steps`` is tolerated under OFF.
        Every other field stays strictly compared — including ``max_steps``,
        which OFF rows carry as None (no cap), so a numeric value there is
        corruption, not legitimate drift. Absent keys are
        distinguished from present-None via the ``_MISSING`` sentinel so a
        sparse repeat row still raises. Rationale:
        under update-match OFF, natural step counts are legitimate per-round
        quantities, not per-unit invariants — ``train_label_flip`` runs 1
        natural pass vs 5 epochs on every other training path ( tracks
        that asymmetry as a separate decision), so an S4 partition whose
        lineage rotates into label_flip legitimately reports a different
        natural ``actual_steps`` than its first-recorded row.
        ``assert_arm_compliance`` only ascribes meaning to
        ``actual_steps == max_steps`` when update-match is ON — this gate must
        not guard a field the OFF regime does not define as static. When
        ``update_match`` is True the FULL comparison applies, unchanged."""
        if not results:
            return
        for _proxy, fit_res in results:
            cid = getattr(_proxy, "cid", "?")
            metrics = getattr(fit_res, "metrics", None) or {}
            raw = metrics.get("resampling_manifest")
            if raw is None:
                continue
            try:
                row = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"[MANIFEST] malformed resampling_manifest at round "
                    f"{server_round} client cid={cid}: {exc}"
                ) from exc
            pid = row.get("partition_id")
            if pid is None:
                raise ValueError(
                    f"[MANIFEST] resampling_manifest at round {server_round} "
                    f"client cid={cid} is missing partition_id: {row!r}"
                )
            pid = int(pid)
            prior = self._resampling_manifest.get(pid)
            if prior is not None:
                compared = set(prior) | set(row)
                if (not row.get("update_match")
                        and "actual_steps" in prior and "actual_steps" in row):
                    # update-match OFF (row-declared) — the natural
                    # step count is per-round, not static; tolerate VALUE
                    # drift only when the field is present in BOTH rows. A
                    # presence mismatch stays a custody error (round-3
                    # residual). max_steps stays compared: OFF rows carry
                    # max_steps=None (no cap), so a numeric value is
                    # corruption, not drift.
                    compared -= {"actual_steps"}
                differing = sorted(
                    k for k in compared
                    # _MISSING keeps an absent key distinct from a present
                    # None, matching dict equality.
                    if prior.get(k, _MISSING) != row.get(k, _MISSING)
                )
                if differing:
                    raise ValueError(
                        f"[MANIFEST] conflicting repeat row for partition {pid} "
                        f"at round {server_round} (cid={cid}); manifest rows are "
                        f"static per unit but these keys differ: {differing}"
                    )
                continue  # identical repeat: idempotent no-op, no reprint
            self._resampling_manifest[pid] = row
            print(f"[MANIFEST] {json.dumps(row)}")

    def _build_identity_map(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Dict[str, str]:
        """This round's {raw Flower cid -> scenario logical identity}.

        Uses the same cid -> partition -> schedule-entry resolution as
        `_maybe_log_signals`, so the identity the defenses key state on is
        exactly the identity the signal log records as ground truth. Raw
        cids with no partition mapping or no schedule entry are omitted
        (plugins fall back to legacy raw-cid keying for those).
        """
        scenario_round = server_round - self._round_offset
        schedule = self._schedule_cache.get(scenario_round, [])
        if not schedule:
            return {}
        partition_to_entry = {e["partition_id"]: e for e in schedule}
        mapping: Dict[str, str] = {}
        for client_proxy, _fit_res in results:
            partition_id = self._cid_to_partition.get(client_proxy.cid)
            if partition_id is None:
                continue
            entry = partition_to_entry.get(partition_id)
            if entry is None:
                continue
            mapping[client_proxy.cid] = entry["logical_id"]
        return mapping

    def _maybe_apply_real_alie(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Replace ALIE-attacking clients' updates with the real Baruch et al. attack.

        Real ALIE (Baruch et al. 2019, Algorithm 3):
        1. Compute coordinate-wise mean/std of honest clients' updates
        2. Set malicious update = mean_honest - z_max * std_honest
        3. All malicious clients send the SAME crafted update

        This is a server-side operation because it requires cross-client
        statistics that no single client can access in Flower's architecture.
        """
        scenario_round = server_round - self._round_offset
        schedule = self._schedule_cache.get(scenario_round, [])
        if not schedule:
            return results

        # Identify ALIE-attacking partition IDs and their z_max
        alie_partitions: Dict[int, float] = {}
        for entry in schedule:
            if entry.get("attack_type") == "alie":
                z_max = float(entry.get("attack_params", {}).get("z_max", 1.0))
                alie_partitions[entry["partition_id"]] = z_max

        if not alie_partitions:
            return results

        # Classify results into honest / malicious indices
        honest_indices = []
        malicious_indices = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            partition_id = self._cid_to_partition.get(client_proxy.cid)
            if partition_id in alie_partitions:
                malicious_indices.append(idx)
            else:
                honest_indices.append(idx)

        if not honest_indices:
            logger.warning("No honest clients to compute ALIE stats from")
            return results

        global_params = self._global_params_ndarrays

        # Compute honest updates (params - global_params) and flatten
        honest_updates_flat = []
        for idx in honest_indices:
            _, fit_res = results[idx]
            client_params = parameters_to_ndarrays(fit_res.parameters)
            update = np.concatenate([
                (cp - gp).ravel() for cp, gp in zip(client_params, global_params)
            ])
            honest_updates_flat.append(update)

        honest_stack = np.stack(honest_updates_flat)  # (n_honest, d)
        honest_mean = honest_stack.mean(axis=0)
        # Population std (ddof=0) to avoid NaN when n_honest=1.
        # With 1 honest client, std=0 → ALIE update equals the honest mean.
        honest_std = honest_stack.std(axis=0, ddof=0)

        # Get z_max (use the first ALIE client's z_max — all should match)
        z_max = next(iter(alie_partitions.values()))

        # Construct ALIE update: mean - z_max * std (component-wise)
        alie_update_flat = honest_mean - z_max * honest_std

        # Reshape back into per-layer arrays matching global_params shapes
        alie_params = []
        offset = 0
        for gp in global_params:
            size = gp.size
            layer_update = alie_update_flat[offset:offset + size].reshape(gp.shape)
            alie_params.append(gp + layer_update)
            offset += size

        alie_parameters = ndarrays_to_parameters(alie_params)

        # Compute diagnostic: norm ratio of ALIE update vs honest mean norm
        alie_norm = float(np.linalg.norm(alie_update_flat))
        honest_norms = [float(np.linalg.norm(u)) for u in honest_updates_flat]
        honest_mean_norm = float(np.mean(honest_norms))
        norm_ratio = alie_norm / max(honest_mean_norm, 1e-12)

        print(f"[ScenarioStrategy] Real ALIE (z_max={z_max}): "
              f"replacing {len(malicious_indices)} clients | "
              f"norm_ratio={norm_ratio:.2f}x | "
              f"honest_mean_norm={honest_mean_norm:.2f} | "
              f"alie_norm={alie_norm:.2f}")

        # Replace malicious clients' parameters
        new_results = list(results)
        for idx in malicious_indices:
            client_proxy, fit_res = results[idx]
            new_fit_res = FitRes(
                status=fit_res.status,
                parameters=alie_parameters,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            new_results[idx] = (client_proxy, new_fit_res)

        return new_results

    def _registry_event_index(self) -> Dict[str, Dict[str, Any]]:
        """`{reentry_event_key: registry-half row}` from the fingerprint plugin.

        Read from the plugin's append-only `reentry_events`, which it writes in
        `observe_cohort` — i.e. from the COMPLETE unfiltered cohort, before any
        plugin filters and before any enforcement action. That ordering is what
        makes the § 5.1 INTEGRITY ASSERTION mechanical rather than aspirational:
        the rows this index carries cannot depend on what the defense did with
        the update, because they are written before it does anything.

        Duck-typed on `reentry_events` so no incumbent plugin is touched.
        """
        index: Dict[str, Dict[str, Any]] = {}
        for plugin in self._plugins:
            rows = getattr(plugin, "reentry_events", None)
            if not rows:
                continue
            for row in rows:
                key = row.get("reentry_event_key")
                if key is None:
                    continue
                index[key] = row
        return index

    @staticmethod
    def _registry_half(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The six registry fields of one assertion row, JSON-safe.

        Non-finite floats become explicit nulls: JSON has no `inf`/`NaN`
        literal, and `signal_logger._jsonify` already applies exactly this
        policy to non-finite numpy floats. `min_d = inf` is the ordinary
        "no flagged candidate existed" reading and `tau` is NaN under the
        pre-τ-lock observe-only posture; in both cases `asserted_match` carries
        the decision, so nothing is lost by logging the number as null.
        """
        carried = REGISTRY_HALF_FIELDS + REGISTRY_NEAREST_FIELDS
        if row is None:
            return {field: None for field in carried}
        fields: Dict[str, Any] = {}
        for field in carried:
            value = row.get(field)
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            fields[field] = value
        return fields

    def _reentry_event_fields(
        self,
        server_round: int,
        logical_cid: str,
        flower_cid: str,
        partition_id: int,
        tenure: int,
        run_uid: Optional[str],
        registry_index: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """The schema-v5 re-entry event contract for one row (v1.10 § 5.1).

        A **re-entry event** is one row per CID-appearance that is a NEW cid for
        a returning logical device — operationally, the first round in which an
        RMC cycle identity (`client_N_newM`) participates, i.e. ``tenure == 1``.
        Every other row carries the contract's fields as explicit nulls.

        **`current_cid` is the FLOWER CID, never the logical identity.** § 5.1
        defines it as *"the new Flower CID the device reappeared under"*, and
        the two must stay distinct: `current_cid` is the OBSERVABLE the re-link
        metric has to recover the identity FROM, while `gt_logical_id` is the
        truth it is scored AGAINST. Writing the logical id into `current_cid`
        would collapse the very distinction H3 exists to test. The frozen event
        key `{run_id}:{round}:{cid}` (§ 5.1 line 776) takes the same CID — it is
        the key OF that CID-appearance; `data/h3_constants.json` carries no key
        spec, so § 5.1 is the sole authority.

        **Harness note for the fingerprint lane.** In Flower *simulation* the
        raw cid is stable per virtual client for the whole run (v1.15), so the
        emitted `current_cid` is in fact the SAME value before and after an
        identity reset — it is "new" only in the design's device-level sense.
        The event key stays unique because the round differs. The registry must
        therefore NEVER key or match on `current_cid`: doing so would hand it
        the identity linkage the RMC threat model forbids and H3 exists to test.
        Defenses key on the logical identity via `set_identity_map` for exactly
        this reason.

        **Both halves are written here.** The ground-truth half is this
        module's; the REGISTRY half (`asserted_match`, `asserted_parent_*`,
        `min_d`, `tau`, `generation`) is joined in from the fingerprint
        plugin's append-only `reentry_events` on `reentry_event_key`, the
        frozen dedup unit both sides build identically.

        Per the § 5.1 INTEGRITY ASSERTION the registry half is written
        **regardless of any downstream enforcement action** (block, downweight
        or accept), so the re-link metric is computable from the signal log
        independent of what the defense did with the update. That property is
        structural, not conventional: the plugin appends those rows from
        `observe_cohort`, which runs on the complete unfiltered cohort before
        any filtering — see `_registry_event_index`. An arm with no fingerprint
        plugin (every incumbent) simply finds nothing to join and logs the six
        fields as explicit nulls, exactly as before.
        """
        fields: Dict[str, Any] = {field: None for field in REENTRY_EVENT_FIELDS}
        if tenure != 1 or not is_reentry_identity(logical_cid):
            return fields

        gt_logical_id = canonical_device_id(logical_cid)
        # Truth-key integrity: the returning identity MUST resolve to its
        # parent's partition (that is what makes it the same physical device).
        # A mismatch would silently corrupt the H3 primary metric's truth key,
        # so it fails loudly — the v1.17 misattribution-family rule.
        expected_partition = self.LOGICAL_TO_PARTITION.get(gt_logical_id)
        if expected_partition != partition_id:
            raise RuntimeError(
                f"Re-entry truth-key mismatch: {logical_cid!r} canonicalizes to "
                f"{gt_logical_id!r} (partition {expected_partition}) but the row's "
                f"physical_partition_id is {partition_id}. The H3 re-link metric "
                f"is scored on this key — refusing to emit a corrupt event."
            )

        if run_uid is None:
            # Only reachable with a logger test-double: the real SignalLogger
            # always carries a run_uid. Say so loudly — the scorer DEDUPS on
            # this key, so a null key silently drops the event from the
            # denominator, the one failure mode that would bias the metric.
            print(
                f"[Integrity] round={server_round} re-entry event for "
                f"{logical_cid!r} has NO run_uid — reentry_event_key logged "
                f"null; this event is not scorer-computable",
                flush=True,
            )
        event_key = (
            build_reentry_event_key(run_uid, server_round, flower_cid)
            if run_uid is not None else None
        )
        fields["reentry_event_key"] = event_key
        fields["current_cid"] = flower_cid
        fields["gt_logical_id"] = gt_logical_id
        fields["gt_is_malicious"] = bool(logical_cid in self._adv_ids)

        # The registry half, joined by key. A null key cannot join (it is
        # already reported loudly above), and an arm with no fingerprint plugin
        # finds no rows — both leave the six fields as explicit nulls.
        index = self._registry_event_index() if registry_index is None else registry_index
        fields.update(
            self._registry_half(index.get(event_key) if event_key is not None else None)
        )
        return fields

    def _maybe_log_signals(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """Emit one SignalLogger row per (client, round) after aggregation."""
        if self._signal_logger is None or not results:
            return

        scenario_round = server_round - self._round_offset
        schedule = self._schedule_cache.get(scenario_round, [])
        if not schedule:
            return

        partition_to_entry = {e["partition_id"]: e for e in schedule}

        # Flatten all updates for Family S feature computation
        flat_updates: List[Any] = []
        num_examples_list: List[int] = []
        train_losses: List[Optional[float]] = []
        ordering = []  # (client_proxy, fit_res, logical_cid, partition_id, entry)
        for client_proxy, fit_res in results:
            partition_id = self._cid_to_partition.get(client_proxy.cid)
            if partition_id is None:
                continue
            entry = partition_to_entry.get(partition_id)
            if entry is None:
                continue
            logical_cid = entry["logical_id"]

            flat = flatten_parameters(fit_res.parameters)
            flat_updates.append(flat)
            num_examples_list.append(int(fit_res.num_examples))
            tl = fit_res.metrics.get("train_loss")
            try:
                train_losses.append(float(tl) if tl is not None else None)
            except (TypeError, ValueError):
                train_losses.append(None)
            ordering.append((client_proxy, fit_res, logical_cid, partition_id, entry))

        if not ordering:
            return

        signals = compute_per_client_signals(
            flat_updates, train_losses, num_examples_list
        )

        # Collect plugin scores for this round if plugins carry them.
        krum_scores = {}
        trust_scores = {}
        tge_details_by_cid: dict = {}
        tge_threshold_seen: float | None = None

        for plugin in self._plugins:
            # Plugins that populate _round_scores: KrumDefense, TrustScore.
            round_scores = getattr(plugin, "_round_scores", None)
            if round_scores:
                scores_this_round = round_scores.get(server_round)
                if scores_this_round:
                    if plugin.name == "KrumDefense":
                        krum_scores = scores_this_round
                    elif plugin.name == "TrustScore":
                        trust_scores = scores_this_round
            # TGEnsemble uses _last_details (per-call), not _round_scores. Must
            # be checked outside the round_scores guard or it gets skipped — see
            # 2026-05-26 regression on commit 89a1882.
            if plugin.name == "TGEnsemble":
                tge_details_by_cid = getattr(plugin, "_last_details", {})
                raw_threshold = getattr(plugin, "_threshold", None)
                if raw_threshold is not None:
                    tge_threshold_seen = float(raw_threshold)

        # H4 § 7c-bis chains place Krum/TrustScore DOWNSTREAM of a filtering
        # detector, so their positional `_round_scores` (keyed by index into
        # the SUBSET they scored) can no longer be joined by position against
        # this method's full-round `ordering`. Re-key them by CID using the
        # chain trace's per-stage input order (PluggableStrategy records it
        # for every plugin round). For every incumbent arm — where the scored
        # list IS the full canonicalized round — the cid-keyed join produces
        # byte-identical values to the old positional join; for a downstream
        # stage it is the only correct join (v1.17 misattribution family). A
        # client the stage never scored gets a truthful null. Falls back to
        # the legacy positional join only when no trace exists (direct test
        # harnesses that bypass PluggableStrategy.aggregate_fit).
        _round_trace = getattr(self, "_h4_chain_trace", {}).get(int(server_round))

        def _cid_keyed_scores(plugin_name: str, positional: dict) -> "dict | None":
            if not positional or _round_trace is None:
                return None
            stage = next(
                (s for s in _round_trace.get("stages", [])
                 if s.get("plugin") == plugin_name),
                None,
            )
            if stage is None:
                return None
            stage_cids = stage["input_cids"]
            bad = [i for i in positional if not 0 <= int(i) < len(stage_cids)]
            if bad:
                raise RuntimeError(
                    f"{plugin_name} _round_scores carry indices {sorted(bad)} "
                    f"outside its scored stage of {len(stage_cids)} clients — "
                    f"signal-log join would misattribute scores "
                    f"(methodology v1.17)"
                )
            return {stage_cids[int(i)]: v for i, v in positional.items()}

        krum_cid_scores = _cid_keyed_scores("KrumDefense", krum_scores)
        trust_cid_scores = _cid_keyed_scores("TrustScore", trust_scores)

        # v1.17 integrity assertion (§ 4.9 loud-fail family): TGE details are
        # keyed by raw cid; every key must belong to a client in THIS round's
        # results, else the join below would misattribute scores. (Before
        # v1.17 the join was positional, so in the composed Krum+TGE chain —
        # where TGE only scores Krum's survivors — survivor scores landed on
        # the wrong clients' rows whenever Krum filtered anyone.)
        if tge_details_by_cid:
            _result_cids = {str(cp.cid) for cp, _fr in results}
            _unknown = set(tge_details_by_cid) - _result_cids
            if _unknown:
                raise RuntimeError(
                    f"TGE details keyed by cids not present in this round's "
                    f"results: {sorted(_unknown)[:5]} — signal-log join would "
                    f"misattribute scores (methodology v1.17)"
                )

        # H3 schema v5 : this round's POST-`filter_updates`, round-
        # normalized aggregation coefficients, keyed by RAW cid. `None` when the
        # round bypassed the plugin path or the base is not FedAvg — in which
        # case the field is logged null, never a value that is not the
        # coefficient. See PluggableStrategy.aggregation_coefficients.
        coefficients = self.aggregation_coefficients(server_round)
        run_uid = getattr(self._signal_logger, "run_uid", None)
        missing_coefficient_cids: list[str] = []
        # H3: build the registry-half index ONCE per round rather than per row.
        registry_index = self._registry_event_index()
        joined_event_keys: set[str] = set()

        records = []
        for idx, (client_proxy, fit_res, logical_cid, partition_id, entry) in enumerate(
            ordering
        ):
            tge_details = tge_details_by_cid.get(str(client_proxy.cid), {})
            tge_score = tge_details.get("final_score")
            tge_decision: bool | None = None
            if tge_score is not None and tge_threshold_seen is not None:
                tge_decision = bool(tge_score >= tge_threshold_seen)

            raw_cid = str(client_proxy.cid)
            tenure = compute_tenure(
                logical_cid, server_round, self._tenure_first_seen
            )
            reentry_fields = self._reentry_event_fields(
                server_round, logical_cid, raw_cid, int(partition_id),
                tenure, run_uid, registry_index=registry_index,
            )
            if reentry_fields["reentry_event_key"] is not None:
                joined_event_keys.add(reentry_fields["reentry_event_key"])

            if coefficients is None:
                aggregation_coefficient = None
            else:
                aggregation_coefficient = coefficients.get(raw_cid)
                if aggregation_coefficient is None:
                    # Fail-SAFE, not fail-silent: a participant with no recorded
                    # coefficient is a code defect. Log null (which integrity
                    # gate (b) rejects as unknown-provenance) and say so loudly,
                    # rather than substitute a number.
                    missing_coefficient_cids.append(raw_cid)

            rec = {
                "logical_cid": logical_cid,
                "tenure": tenure,
                "flower_cid": raw_cid,
                "physical_partition_id": int(partition_id),
                "malicious_gt": (logical_cid in self._adv_ids),
                "attack_type": entry.get("attack_type") or "",
                "krum_score": (
                    krum_cid_scores.get(raw_cid)
                    if krum_cid_scores is not None
                    else (krum_scores.get(idx) if krum_scores else None)
                ),
                "trust_score": (
                    trust_cid_scores.get(raw_cid)
                    if trust_cid_scores is not None
                    else (trust_scores.get(idx) if trust_scores else None)
                ),
                # LEGACY, semantics UNCHANGED at v5: the RAW, pre-filter,
                # client-reported num_examples. It is non-zero for every
                # dispatched client, survivor or not — never a keep-flag — and
                # v4 readers depend on exactly this. The post-defense quantity
                # H3 scores on is `aggregation_coefficient` below.
                "effective_weight": float(fit_res.num_examples),
                # Schema v5 (v1.10 § 5.1): the post-`filter_updates`,
                # round-normalized FedAvg coefficient — 0.0 for a hard-dropped
                # client, summing to 1.0 over the round's survivors. This is
                # what data/h3_constants.json's rejoin-success rule is defined
                # on (`blocked: coefficient == 0`).
                "aggregation_coefficient": aggregation_coefficient,
                **reentry_fields,
                "tge_score": tge_score,
                "tge_gbdt_score": tge_details.get("gbdt_score"),
                "tge_lstm_score": tge_details.get("lstm_score"),
                # TGE′ second long-memory leg : EMA reputation. Null in
                # incumbent LSTM runs and until the EMA has ≥1 update; in bank
                # mode the gate's long-memory leg = min(tge_lstm_score,
                # tge_ema_score), reconstructable offline from these two fields.
                "tge_ema_score": tge_details.get("ema_score"),
                "tge_tenure": tge_details.get("tenure"),
                "tge_phase": tge_details.get("phase"),
                "tge_gate": tge_details.get("gate"),
                "tge_threshold": tge_threshold_seen,
                "tge_decision": tge_decision,
                **signals[idx],
            }
            records.append(rec)

        if missing_coefficient_cids:
            print(
                f"[Integrity] round={server_round} "
                f"aggregation_coefficient MISSING for "
                f"{sorted(missing_coefficient_cids)} — these rows are logged "
                f"null and will fail the H3 schema-v5 computability gate",
                flush=True,
            )

        # H3: a registry assertion for THIS round that found no ground-truth row
        # to land on. The two halves are meant to be in one-to-one correspondence
        # (the plugin emits one row per new claimed identity after enrolment; the
        # ground-truth layer emits one per tenure-1 cycle identity), so an orphan
        # means the correspondence broke and the D7 numerator would silently lose
        # the event. Never silent — the scorer dedups on this key.
        orphan_keys = sorted(
            key for key, row in registry_index.items()
            if int(row.get("server_round", -1)) == int(server_round)
            and key not in joined_event_keys
        )
        if orphan_keys:
            print(
                f"[Integrity] round={server_round} registry re-link assertions "
                f"with NO ground-truth event row: {orphan_keys} — the registry "
                f"half of these events is NOT in the signal log and they are "
                f"absent from the H3 re-link population",
                flush=True,
            )

        try:
            self._signal_logger.log_round(
                server_round=server_round,
                scenario_round=scenario_round,
                per_client_records=records,
            )
        except Exception as e:
            logger.warning(f"SignalLogger failed to log round {server_round}: {e}")

    def set_live_metric_logger(
        self, logger_obj: Optional[LiveRoundMetricLogger]
    ) -> None:
        """Attach (or clear) the live per-round MLflow logger.

        Single wiring point used by ``server_app.server_fn`` so the nine
        ScenarioStrategy construction sites stay unchanged. Passing ``None``
        (the env-derived default when not launched via ``praxis exp launch``)
        leaves live logging disabled.
        """
        self._live_metric_logger = logger_obj

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation using FixedEvalManager."""
        if self._eval_manager is None:
            return self._base.evaluate(server_round, parameters)

        weights = parameters_to_ndarrays(parameters)
        metrics = self._eval_manager.evaluate(weights)

        # Prec/Rec appended so the
        # parsed trajectory carries all five live-contract metrics; the existing
        # F1=/Acc=/Loss= prefix regex in the runner parsers is unaffected.
        eval_line = (f"[ScenarioStrategy] Round {server_round} eval: "
                     f"F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} "
                     f"Loss={metrics['loss']:.4f} "
                     f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")
        # Per-class benign/attack metrics appended when the eval manager supplies
        # them (Stage-F). Append-only: the suffix is a separate optional
        # regex group in parse_eval_trajectory, so a manager that omits them (or an
        # older log) parses exactly as before.
        _per_class = ("attack_precision", "attack_recall", "attack_f1",
                      "benign_precision", "benign_recall", "benign_f1")
        if all(k in metrics for k in _per_class):
            eval_line += (
                f" AttP={metrics['attack_precision']:.4f} "
                f"AttR={metrics['attack_recall']:.4f} "
                f"AttF1={metrics['attack_f1']:.4f} "
                f"BenP={metrics['benign_precision']:.4f} "
                f"BenR={metrics['benign_recall']:.4f} "
                f"BenF1={metrics['benign_f1']:.4f}")
        print(eval_line)

        # Live per-round MLflow streaming (Lane C item 10). This is the
        # authoritative per-round metric path: it logs each round to MLflow as
        # the round completes (step=server_round), so a RUNNING unit shows a
        # live trajectory and a mid-run crash/reclaim keeps every completed
        # round. Best-effort — the logger swallows all tracking errors, so this
        # never affects the returned metrics, the parsed trajectory, or anything
        # persisted to disk/S3. No-op when no logger is attached.
        if self._live_metric_logger is not None:
            try:
                self._live_metric_logger.log_round(
                    server_round,
                    {
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "loss": metrics["loss"],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - never fail the round
                # Defense-in-depth simulation boundary: the logger is already
                # best-effort internally, but a tracking failure must NEVER
                # affect the round, the returned metrics, or the trajectory.
                logger.warning(
                    "live MLflow round-%s logging failed: %s", server_round, exc
                )

        return metrics["loss"], {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

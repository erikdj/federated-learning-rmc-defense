"""
Byzantine Defense Plugin Architecture for FlowerFL.

Provides a pluggable defense hook system that wraps Flower aggregation
strategies with custom Byzantine-robust logic. This is the module that
implements the defense arms compared in the praxis's H2 experiment (RQ2:
does TGE achieve higher recall at 10% FPR than its paired baseline under
RMC?) — see the H2 section of docs/reproduction/experiments.md for the locked
defense matrix and thresholds.

Architecture:
    ByzantineDefensePlugin (ABC)
        └── KrumDefensePlugin (wraps Multi-Krum with custom scoring)
        └── TrustScorePlugin (reputation-based filtering)
        └── TGEnsemblePlugin (Tenure-Gated Ensemble: GBDT + LSTM + tenure gate)

    PluggableStrategy (Flower Strategy wrapper)
        - Wraps any base Flower strategy (FedAvg, etc.)
        - Calls registered plugins before/after aggregation
        - Provides hook points: pre_aggregate, score_updates, post_aggregate

Two distinct notions of "threshold" appear below and should not be conflated:
    1. Each plugin's own accept/reject cutoff used in filter_updates (e.g.
       TGEnsemblePlugin's self._threshold=0.7, Krum's top-m selection) — this
       governs which updates are actually included in THIS round's
       aggregation. It is an operational decision, not an evaluation metric.
    2. The recall@10%FPR evaluation metric (RQ2/H2) — computed OFFLINE from
       the continuous per-client scores each plugin exposes via
       `_round_scores` (KrumDefensePlugin, TrustScorePlugin) or
       `_last_details` (TGEnsemblePlugin), which flowerfl/scenario_strategy.py
       reads after each score_updates call and writes into the signal log
       (flowerfl/signal_logger.py). The evaluation threshold is selected
       per-defense on the 5 dev seeds and then frozen before scoring the 10
       confirmatory seeds (spec v1.3 § 2, "frozen-threshold-from-dev"
       protocol) — it is independent of each plugin's own operational
       threshold above.

Usage:
    # In server_app.py:
    from flowerfl.byzantine_defense import PluggableStrategy, KrumDefensePlugin

    base_strategy = FedAvg(...)
    plugin = KrumDefensePlugin(num_malicious=2)
    strategy = PluggableStrategy(base_strategy, plugins=[plugin])
"""

import math

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import logging

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy

logger = logging.getLogger(__name__)


# ============================================================================
# PLUGIN INTERFACE
# ============================================================================

class ByzantineDefensePlugin(ABC):
    """
    Abstract base class for Byzantine defense plugins.

    Plugins are hooks that execute at specific points during FL aggregation.
    They can filter, score, or modify client updates before aggregation.

    To create a novel defense, subclass this and implement the methods.
    Concrete subclasses that want their continuous scores logged for
    recall@FPR evaluation (see module docstring) should also populate a
    `_round_scores: Dict[int, Dict[int, float]]` attribute keyed by
    server_round — flowerfl/scenario_strategy.py looks for this attribute
    name by convention (duck-typed, not enforced by this ABC).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    def on_round_start(self, server_round: int, num_clients: int) -> None:
        """Called at the start of each FL round. Optional hook."""
        pass

    def observe_cohort(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> None:
        """Observe the FULL round cohort BEFORE any plugin filtering. Optional.

        PluggableStrategy calls this on every plugin with the complete,
        unfiltered results at the start of aggregate_fit — so a plugin composed
        AFTER an upstream filter (e.g. TGE behind Krum) can still see every
        participant's update. Default no-op: only stateful defenses that must
        observe the whole cohort regardless of downstream filtering override it
        (TGE′'s EMA reputation leg). Must not mutate the results.
        """
        pass

    def set_identity_map(self, cid_to_identity: Dict[str, str]) -> None:
        """Install the server-side identity resolution for the CURRENT round.

        In Flower simulation the raw `cid` is stable per virtual client for
        the whole run, and RMC cycle identities (`client_N_newM`) map back to
        the same physical partition — so raw-cid keying would let stateful
        defenses silently link a "new" identity to its previous life, which
        the RMC threat model forbids (identity linkage is exactly what H3's
        fingerprint registry is being evaluated for). ScenarioStrategy calls
        this every scored round with {raw cid -> logical identity}; stateful
        plugins key their per-client state through `resolve_identity`.
        Methodology v1.15.
        """
        self._identity_map = dict(cid_to_identity)

    def resolve_identity(self, cid: str) -> Optional[str]:
        """Logical identity for a raw cid this round, or None if unmapped
        (e.g. plain strategies with no scenario schedule)."""
        return getattr(self, "_identity_map", {}).get(cid)

    @abstractmethod
    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """
        Score each client update. Higher score = more trustworthy.

        Args:
            results: List of (client_proxy, fit_result) from clients
            server_round: Current FL round number

        Returns:
            Dict mapping client index -> trust score [0.0, 1.0]
        """
        ...

    def filter_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        scores: Dict[int, float],
        threshold: float = 0.0,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """
        Filter out untrusted updates based on scores.
        Default: keep all updates with score > threshold.

        This default (score > 0.0 keeps everything, since scores are
        normalized to [0.0, 1.0]) is a no-op filter; subclasses override
        this to apply their own operational accept/reject rule (see module
        docstring for how this differs from the recall@FPR evaluation
        threshold). PluggableStrategy.aggregate_fit calls this once per
        plugin, chaining each plugin's output into the next plugin's input.
        """
        filtered = []
        for idx, (client, fit_res) in enumerate(results):
            if scores.get(idx, 0.0) > threshold:
                filtered.append((client, fit_res))
            else:
                logger.info(f"[{self.name}] Filtered client {idx} (score={scores.get(idx, 0.0):.3f})")
        return filtered

    def on_round_end(self, server_round: int, aggregated_params: Parameters) -> None:
        """Called after aggregation completes. Optional hook."""
        pass


# ============================================================================
# KRUM DEFENSE PLUGIN
# ============================================================================

class KrumDefensePlugin(ByzantineDefensePlugin):
    """
    Krum-based defense plugin.

    Implements Multi-Krum scoring as a plugin that can be composed with
    any base aggregation strategy. Scores clients based on pairwise
    distance to other updates — outlier updates get low scores.

    This wraps the well-known Krum algorithm (Blanchard et al., 2017)
    as a reusable plugin rather than a monolithic strategy replacement.

    Reference:
        Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).
        Machine learning with adversaries: Byzantine tolerant gradient descent.
        NeurIPS 2017.
    """

    def __init__(
        self,
        num_malicious: int = 0,
        num_to_keep: int = 0,
        dynamic_f: bool = False,
    ):
        """
        Args:
            num_malicious: Expected number of Byzantine clients (f). With
                ``dynamic_f=True`` this is retained for PROVENANCE only —
                per-round sizing ignores it.
            num_to_keep: How many top-scoring clients to keep (Multi-Krum m).
                         0 = auto-compute as max(1, n - f - 2). With
                         ``dynamic_f=True``, provenance only (see above).
            dynamic_f: When True, f is recomputed EVERY round from that
                round's participant count n as ``ceil(n/2) - 1`` and the keep
                count as ``max(1, n - f - 2)`` — the April Szeląg-anchor
                formula (the original baseline reproduction). Adopted for
                scenario deployments (methodology v1.19, ) because
                S3/S4 disconnect rounds schedule only ~11 participants (the 9
                adversaries are disconnected — the RMC pattern) and a static
                f=9 there makes num_closest = 11-9-2 = 0 (uncomputable).
                Dynamic sizing reproduces the documented full-cohort operating
                point exactly: n=20 -> f=9, keep=9; n=11 -> f=5, keep=4.
        """
        self._num_malicious = num_malicious
        self._num_to_keep = num_to_keep
        self._dynamic_f = dynamic_f
        self._round_scores: Dict[int, Dict[int, float]] = {}

    def _effective_f(self, n: int) -> int:
        """Round-effective f: April formula ``ceil(n/2) - 1`` when dynamic
        (reproduce_szelag.py:388,739), else the configured static f."""
        if self._dynamic_f:
            return math.ceil(n / 2) - 1
        return self._num_malicious

    @property
    def name(self) -> str:
        return "KrumDefense"

    def _flatten_parameters(self, parameters: Parameters) -> np.ndarray:
        """Convert Parameters to a single flat numpy vector."""
        ndarrays = parameters_to_ndarrays(parameters)
        return np.concatenate([arr.flatten() for arr in ndarrays])

    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """Score clients using Krum's distance-based criterion.

        Records the resulting per-client scores into `self._round_scores`
        (keyed by server_round) for the recall@FPR evaluation pipeline (see
        module docstring) — this is in addition to returning them for
        filter_updates's own accept/reject decision this round.
        """
        n = len(results)
        f = self._effective_f(n)  # per-round ceil(n/2)-1 when dynamic (v1.19)
        num_closest = n - f - 2

        # COMPUTABILITY check, not certification (methodology v1.19). The old
        # guard `n <= 2f+2` was Krum's CERTIFICATION bound; treating it as a
        # computability bound returned uniform 1.0 scores at the canonical RMC
        # parameters (n=20, f=9), degenerating the Krum arm to
        # keep-first-m-by-arrival-order and flattening the krum_score
        # signal-log channel (no variance -> H2 recall@FPR uncomputable for
        # the Krum arm). The score is computable whenever num_closest >= 1 —
        # the April Szelag anchor (the original baseline reproduction)
        # computes it at exactly n=20/f=9 with no guard, and
        # Szelag-faithfulness is the pre-registered principle: the praxis
        # deliberately studies Krum PUSHED PAST its certified tolerance
        # under RMC.
        if num_closest < 1:
            # Truly uncomputable: no closest-neighbor distances to sum. This
            # state must never occur in a registered scenario (canonical RMC:
            # num_closest = 20-9-2 = 9; worst S3/S4 churn cohort 15-9-2 = 4).
            # It IS reachable in H4 chains when an upstream filter shrinks the
            # cohort (detector warmup mass-flags — EXP-061 C6): the uniform
            # scores must be RECORDED into _round_scores like every computable
            # round, or the cid-keyed signal join renders nulls for a round
            # that genuinely scored (erratum B, methodology v1.53).
            logger.error(
                f"[{self.name}] Krum score UNCOMPUTABLE: n={n}, f={f} -> "
                f"num_closest={num_closest} < 1. Returning uniform scores. "
                f"This must never occur in a registered scenario — check the "
                f"scenario's declared adversary count against its cohort."
            )
            scores = {i: 1.0 for i in range(n)}
            self._round_scores[server_round] = dict(scores)
            return scores

        if f >= (n - 2) / 2:
            # Certified tolerance exceeded — still compute, Szelag-faithfully.
            # One clear warning per round (score_updates runs once per round).
            logger.warning(
                f"[{self.name}] Krum certified bound exceeded (n={n}, f={f}, "
                f"tolerance f<(n-2)/2): scores computed Szeląg-faithfully "
                f"beyond certification"
            )

        # Flatten all client updates
        flat_updates = []
        for _, fit_res in results:
            flat_updates.append(self._flatten_parameters(fit_res.parameters))

        # Compute pairwise squared distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2
                distances[i][j] = d
                distances[j][i] = d

        # For each client, sum the n-f-2 closest distances (num_closest
        # computed above, alongside the computability check)
        krum_scores_raw = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(distances[i])  # includes self (0)
            # Skip self (index 0), take next num_closest
            krum_scores_raw[i] = np.sum(sorted_dists[1:num_closest + 1])

        # Convert to trust scores: lower Krum score = more trustworthy.
        # Min-max normalize the raw sum-of-distances to [0, 1] and invert,
        # so the closest-to-consensus client scores near 1.0 and the most
        # distant scores near 0.0. Degenerates to all-1.0 when every client's
        # raw score is identical (max_score == min_score) rather than
        # dividing by zero.
        max_score = krum_scores_raw.max()
        min_score = krum_scores_raw.min()
        if max_score > min_score:
            normalized = 1.0 - (krum_scores_raw - min_score) / (max_score - min_score)
        else:
            normalized = np.ones(n)

        scores = {i: float(normalized[i]) for i in range(n)}
        self._round_scores[server_round] = scores

        # Log top and bottom clients
        sorted_clients = sorted(scores.items(), key=lambda x: -x[1])
        logger.info(f"[{self.name}] Round {server_round} scores: "
                   f"best={sorted_clients[0]}, worst={sorted_clients[-1]}")

        return scores

    def filter_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        scores: Dict[int, float],
        threshold: float = 0.0,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Keep only the top-m scoring clients (Multi-Krum selection).

        This is Krum's OPERATIONAL exclusion decision for this round's
        aggregation — the `threshold` parameter is unused (Krum selects by
        rank, not by a score cutoff); it is accepted only to satisfy the
        base class's filter_updates signature.
        """
        n = len(results)
        f = self._effective_f(n)  # per-round ceil(n/2)-1 when dynamic (v1.19)
        if self._dynamic_f:
            # Dynamic sizing (April formula): keep tracks this round's n, so
            # S3/S4 disconnect rounds (n~11 -> f=5, keep=4) stay meaningful
            # instead of keeping a full-cohort-sized 9 of 11.
            m = max(1, n - f - 2)
        else:
            m = self._num_to_keep if self._num_to_keep > 0 else max(1, n - f - 2)

        # Sort by score descending, keep top m
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        keep_indices = set(idx for idx, _ in ranked[:m])

        filtered = []
        for idx, (client, fit_res) in enumerate(results):
            if idx in keep_indices:
                filtered.append((client, fit_res))
            else:
                logger.info(f"[{self.name}] Excluded client {idx} "
                          f"(score={scores.get(idx, 0.0):.3f})")

        print(f"[{self.name}] Kept {len(filtered)}/{n} clients "
              f"(m={m}, f={f})")

        return filtered


# ============================================================================
# TRUST SCORE PLUGIN (Reputation-based)
# ============================================================================

class TrustScorePlugin(ByzantineDefensePlugin):
    """
    Reputation-based defense plugin.

    Maintains a running trust score for each client across rounds.
    Clients whose updates are consistent with the consensus get higher
    trust scores; outlier clients get penalized.

    This demonstrates a stateful plugin that tracks client behavior
    over time — the foundation for detecting reconnecting malicious clients.

    Does NOT override filter_updates: it relies on the ABC's default
    (keep score > 0.0). Because trust scores start at 0.5 (neutral) and move
    via an exponential moving average rather than a hard cutoff, this is a
    much weaker operational exclusion rule than KrumDefensePlugin's top-m
    selection or TGEnsemblePlugin's 0.7 threshold — most rounds, few or no
    clients are actually excluded from aggregation even when their trust
    score has dropped substantially. Recall@10%FPR for this defense is
    therefore evaluated primarily from the continuous `_round_scores`
    exposed below, not from what filter_updates actually excludes.
    """

    def __init__(self, decay_rate: float = 0.9, outlier_threshold: float = 2.0):
        """
        Args:
            decay_rate: How quickly old trust scores decay (0-1).
            outlier_threshold: Z-score threshold for flagging outliers.
        """
        self._decay_rate = decay_rate
        self._outlier_threshold = outlier_threshold
        self._trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)  # Start neutral
        self._round_history: List[Dict] = []
        # Per-round {idx: score}, read by ScenarioStrategy to log `trust_score`
        # into the signal records (mirrors KrumDefensePlugin._round_scores).
        # Without this, trust_score is never logged and recall@10%FPR is
        # uncomputable for the TrustScore arm (B6, 2026-06-05).
        self._round_scores: Dict[int, Dict[int, float]] = {}

    @property
    def name(self) -> str:
        return "TrustScore"

    def _flatten_parameters(self, parameters: Parameters) -> np.ndarray:
        ndarrays = parameters_to_ndarrays(parameters)
        return np.concatenate([arr.flatten() for arr in ndarrays])

    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """Score based on deviation from median update + historical trust.

        Per-round outlier penalty (z-score of L2 distance to the coordinate-
        wise median update) is blended into each client's persistent trust
        score via an EMA (`decay_rate`), keyed by the scenario's logical
        identity when ScenarioStrategy supplies one (v1.15; falls back to the
        raw Flower `cid` otherwise) so reputation survives across rounds but
        NOT across an identity reset. Clients seen for the first time get 0.5
        (neutral) trust before this round's contribution is blended in — see
        `self._trust_scores` default in __init__.
        """
        n = len(results)

        # Flatten updates
        flat_updates = []
        client_ids = []
        for client_proxy, fit_res in results:
            flat_updates.append(self._flatten_parameters(fit_res.parameters))
            # Key reputation by the scenario's logical identity when available
            # (v1.15): a reconnect under a new identity must start from
            # neutral trust, exactly as a real server would treat it.
            client_ids.append(self.resolve_identity(client_proxy.cid) or client_proxy.cid)

        flat_updates = np.array(flat_updates)

        # Compute median update
        median_update = np.median(flat_updates, axis=0)

        # Compute distances from median
        distances = np.array([
            np.linalg.norm(u - median_update)
            for u in flat_updates
        ])

        # Z-score the distances
        mean_dist = distances.mean()
        std_dist = distances.std()
        if std_dist > 0:
            z_scores = (distances - mean_dist) / std_dist
        else:
            z_scores = np.zeros(n)

        # Convert to round score: 1.0 for close to median, low for outliers
        round_scores = {}
        for i in range(n):
            cid = client_ids[i]

            # Current round contribution: outlier penalty
            if z_scores[i] > self._outlier_threshold:
                round_contribution = 0.0  # Flagged as outlier
            else:
                # Linear scale: z=0 -> 1.0, z=threshold -> 0.0
                round_contribution = max(0.0, 1.0 - z_scores[i] / self._outlier_threshold)

            # Combine with historical trust (exponential moving average)
            old_trust = self._trust_scores[cid]
            new_trust = self._decay_rate * old_trust + (1 - self._decay_rate) * round_contribution
            self._trust_scores[cid] = new_trust

            round_scores[i] = new_trust

        # Log
        flagged = sum(1 for z in z_scores if z > self._outlier_threshold)
        print(f"[{self.name}] Round {server_round}: {flagged}/{n} clients flagged as outliers")

        self._round_history.append({
            "round": server_round,
            "scores": dict(round_scores),
            "flagged": flagged,
        })

        # Expose per-round scores for signal logging (see __init__ note, B6).
        self._round_scores[server_round] = round_scores

        return round_scores


# ============================================================================
# TG-ENSEMBLE DEFENSE PLUGIN (Tenure-Gated Ensemble: GBDT + LSTM)
# ============================================================================

class TGEnsemblePlugin(ByzantineDefensePlugin):
    """
    Tenure-Gated Ensemble (TGE) defense plugin — the praxis's thesis defense.

    Combines a GBDT cold-start expert — despite the class name, this is an
    IsolationForest anomaly detector on geometric update-distance features
    (see rmc/tg_ensemble.py::GBDTColdStartExpert; the class name is a legacy
    holdover from an earlier XGBoost-based design and is NOT a GBDT model)
    — with an LSTM temporal expert (autoencoder on per-client feature
    sequences) via a tenure-gated decision rule that routes scoring based on
    client tenure. The signal-log field this plugin populates is named
    `gbdt_score` for schema-stability reasons but always carries the
    IsolationForest cold-start expert's score, never an XGBoost score. (The
    XGBoost-based cold-start classifier lives separately in
    flowerfl/cold_start_detector.py — that one is the OFFLINE H1
    signal-family classifier trained from accumulated signal logs, not an
    online per-round ensemble member; do not conflate the two.)

    During warmup (first few rounds), falls back to simple geometric scoring.
    After warmup, both experts are trained on accumulated data from accepted
    clients and used for anomaly detection. The tenure gate (rmc/tg_ensemble.py
    TenureGate) blends the two experts' scores by client tenure: pure
    IsolationForest for tenure < 2, a linear blend for tenure in [2, ramp_rounds),
    pure LSTM for tenure >= ramp_rounds. The configured ramp is authoritative
    (the old silent `max(ramp, 8)` floor was removed per amendment v1.6 § 2,
    closing ): the provisional canonical value is 8, and the FINAL value
    is selected empirically at the H2 dev gate described in
    `docs/reproduction/experiments.md` — do not cite the ramp as locked until
    that selection has run.

    Threshold: 0.7 (below = filtered). This is the plugin's OPERATIONAL
    accept/reject cutoff (see module docstring) — distinct from the
    recall@10%FPR evaluation threshold computed offline from `_last_details`.
    """

    def __init__(
        self,
        num_malicious: int = 0,
        num_to_keep: int = 0,
        threshold: float = 0.7,
        warmup_rounds: int = 3,
        ramp_rounds: int = 8,
        long_memory_expert: str = "lstm",
        ema_alpha: float = 0.9,
    ):
        """
        Args:
            num_malicious: Expected number of Byzantine clients (used for
                fallback multi-krum keep count if all are filtered).
            num_to_keep: Minimum clients to keep (0 = auto).
            threshold: Trust score threshold for accept/reject.
            warmup_rounds: Rounds before experts are trained.
            ramp_rounds: Rounds over which to blend cold-start -> long memory.
            long_memory_expert: "lstm" (incumbent TGE — bit-identical), "ema"
                (EMA reputation leg only, component isolation), or "bank" (TGE′:
                min(LSTM, EMA), ). Selects how the tenure gate's
                long-memory leg is formed.
            ema_alpha: EMA retention weight for the reputation expert (ema/bank
                only; inert for "lstm"). ADOPTED at 0.9 from TrustScore's
                validated constant; the amendment's dev gate may revise it.
        """
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from rmc.tg_ensemble import TGEnsembleModel

        self._num_malicious = num_malicious
        self._num_to_keep = num_to_keep
        self._threshold = threshold
        # keyed by RAW CID, not positional index (methodology v1.17): in a
        # composed chain this plugin may receive only upstream survivors, and
        # a positional join against the strategy's full results list would
        # misattribute scores across clients.
        self._last_details: Dict[str, Dict[str, Any]] = {}

        self._model = TGEnsembleModel(
            warmup_rounds=warmup_rounds,
            threshold=threshold,
            ramp_rounds=ramp_rounds,
            long_memory_expert=long_memory_expert,
            ema_alpha=ema_alpha,
        )
        self._boundaries_discovered = False

        # Map Flower cid -> stable internal client ID
        # In Flower simulation, cids are random 64-bit node IDs.
        # We use a counter to assign stable sequential IDs.
        self._cid_map: Dict[str, str] = {}
        self._cid_counter = 0
        # Defense overhead instrumentation (Task 4c.4): wall-clock per round (ms)
        self._timing_per_round: list[float] = []
        # Cohort-observation time for the CURRENT round, folded
        # into the round's recorded TGE time by score_updates so
        # tge_score_time_per_round_ms includes the observe_cohort pass.
        self._pending_observe_ms: float = 0.0

    @property
    def name(self) -> str:
        return "TGEnsemble"

    def _get_client_id(self, cid: str) -> str:
        """Resolve the identity key the tenure gate tracks state against.

        When ScenarioStrategy has installed this round's identity map
        (v1.15), the key is the scenario's LOGICAL identity (`client_N` /
        `client_N_newM`) — so an identity reset genuinely resets tenure and
        LSTM history, matching the signal log's ground-truth
        `compute_tenure(logical_cid,...)` and the RMC threat model (the
        server cannot link a new identity to an old client without H3's
        fingerprint registry).

        Fallback (no map, e.g. plain strategies without a scenario
        schedule): the legacy per-run sequential mapping from the raw
        Flower cid, cached for the lifetime of this plugin instance.
        """
        mapped = self.resolve_identity(cid)
        if mapped is not None:
            return mapped
        if cid not in self._cid_map:
            self._cid_map[cid] = f"flower_client_{self._cid_counter}"
            self._cid_counter += 1
        return self._cid_map[cid]

    def _flatten_parameters(self, parameters: Parameters) -> np.ndarray:
        """Convert Parameters to a single flat numpy vector."""
        ndarrays = parameters_to_ndarrays(parameters)
        return np.concatenate([arr.flatten() for arr in ndarrays])

    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """Score clients using the Tenure-Gated Ensemble.

        Populates `self._last_details` (keyed by RAW CID — not positional
        index, methodology v1.17 — of the raw expert scores, e.g.
        `gbdt_score`; see class docstring for why that key name does not
        imply an XGBoost model) which flowerfl/scenario_strategy.py reads
        after this call to write the signal-log row, joining by each row's
        own cid. Clients filtered by an upstream plugin (e.g. Krum in the
        composed Krum+TGE chain) never reach this call and therefore have no
        entry — their rows truthfully carry null TGE fields. `_last_details`
        is overwritten every call (per-round scope), unlike
        KrumDefensePlugin/TrustScorePlugin's `_round_scores` which accumulate
        history keyed by server_round.
        """
        _t_start = time.perf_counter()
        n = len(results)

        # Discover layer boundaries from first client's parameters (once)
        if not self._boundaries_discovered and results:
            ndarrays = parameters_to_ndarrays(results[0][1].parameters)
            param_shapes = [arr.shape for arr in ndarrays]
            self._model.discover_layer_boundaries(param_shapes)
            self._boundaries_discovered = True
            # Update dimensions
            from rmc.tg_ensemble import LSTMTemporalExpert
            num_layers = len(param_shapes)
            # Geometric feature layout (rmc/tg_ensemble.py::extract_geometric_features):
            # 3 global scalars (z-distance from median, cosine similarity to
            # median, L2 norm deviation) + one per-layer norm-ratio feature
            # per model layer + 3 summary stats (mean/std/max of |update|).
            # This must match the LSTM's input_dim exactly, hence the rebuild
            # below whenever the discovered architecture changes num_layers. The
            # TGE′ EMA leg (self._model.ema) is dimension-agnostic (it consumes
            # the scalar cold-start score) so it needs no rebuild.
            actual_features = 3 + num_layers + 3
            self._model.num_features = actual_features
            if actual_features != self._model.lstm.input_dim:
                self._model.lstm = LSTMTemporalExpert(
                    input_dim=actual_features,
                    hidden_dim=32,
                    num_layers=1,
                    max_seq_len=10,
                    warmup_rounds=self._model.warmup_rounds + 2,
                    refit_interval=5,
                    train_epochs=20,
                    seed=self._model.seed,
                )

        # Flatten all updates
        flat_updates = []
        client_ids = []
        for client_proxy, fit_res in results:
            flat_updates.append(self._flatten_parameters(fit_res.parameters))
            client_ids.append(self._get_client_id(client_proxy.cid))

        # Extract features and score each client
        scores = {}
        details_per_client: Dict[str, Dict[str, Any]] = {}
        for i in range(n):
            features = self._model.extract_features(flat_updates[i], flat_updates)
            trust_score, details = self._model.score_client(
                client_ids[i], features, server_round,
            )
            scores[i] = trust_score
            # keyed by raw cid (v1.17) — scores stays index-keyed for the
            # filter_updates contract; only the logging handoff is cid-keyed
            details_per_client[str(results[i][0].cid)] = details

        self._last_details = details_per_client  # expose to strategy (cid-keyed)

        # Log summary
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        logger.info(
            f"[{self.name}] Round {server_round} scores: "
            f"best=({sorted_scores[0][0]}, {sorted_scores[0][1]:.3f}), "
            f"worst=({sorted_scores[-1][0]}, {sorted_scores[-1][1]:.3f})"
        )

        # Defense overhead bookkeeping (Task 4c.4). Fold in this round's
        # observe_cohort time so the reported per-round TGE
        # cost includes the full-cohort observation pass, then clear it.
        _t_elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
        _t_elapsed_ms += self._pending_observe_ms
        self._pending_observe_ms = 0.0
        self._timing_per_round.append(_t_elapsed_ms)

        return scores

    def filter_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        scores: Dict[int, float],
        threshold: float = 0.0,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Filter clients based on TGE trust scores (operational threshold).

        Uses self._threshold (0.7, not the `threshold` parameter) for
        accept/reject decisions — the parameter exists only to satisfy the
        base class signature. Also records BOTH accepted and rejected
        clients back into the ensemble's training buffers so both experts
        keep learning from the full population, not just the clients this
        defense chose to trust.
        """
        n = len(results)
        filtered = []
        accepted_ids = []
        rejected_ids = []

        for idx, (client, fit_res) in enumerate(results):
            client_id = self._get_client_id(client.cid)
            if scores.get(idx, 0.0) >= self._threshold:
                filtered.append((client, fit_res))
                accepted_ids.append(client_id)
            else:
                rejected_ids.append(client_id)
                logger.info(
                    f"[{self.name}] Filtered client {idx} "
                    f"(score={scores.get(idx, 0.0):.3f})"
                )

        # Record ALL scored clients (accepted + rejected) for the
        # IsolationForest cold-start expert's training buffer — training on
        # every scored client, not just accepted ones, prevents confirmation
        # bias (training the anomaly model only on updates it already
        # trusted, which would make it progressively blind to anything it
        # once rejected).
        server_round = getattr(self, '_current_round', 0)
        for idx, (client, fit_res) in enumerate(results):
            client_id = self._get_client_id(client.cid)
            self._model.record_scored(client_id, server_round)

        # Record accepted clients for LSTM temporal training
        for cid in accepted_ids:
            self._model.record_accepted(cid, server_round)

        print(
            f"[{self.name}] Kept {len(filtered)}/{n} clients "
            f"(threshold={self._threshold}, rejected={rejected_ids or 'none'})"
        )

        return filtered

    def on_round_start(self, server_round: int, num_clients: int) -> None:
        """Cache the current round on self so filter_updates (which doesn't
        receive server_round directly) can pass it to record_scored/record_accepted."""
        self._current_round = server_round

    def observe_cohort(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> None:
        """Update the EMA reputation leg for the FULL round cohort.

        The bank's EMA is a once-per-participating-round reputation, cohort-wide
        and independent of upstream filtering (amendment v1.7 §2.1). In the
        Krum+TGE′ cascade this plugin only SCORES Krum's survivors, but
        PluggableStrategy calls this hook with the COMPLETE cohort before the
        chain — so every participant's reputation evolves each round it
        participates, not only the rounds it survives Krum. Delegates to
        TGEnsembleModel.observe_ema, which touches ONLY the EMA (no
        IsolationForest buffer, LSTM history, tenure, or _last_details) so the
        scored-row signal-log population is byte-identical. No-op in lstm mode
        and before the forest fits (guarded here to skip feature extraction).

        Timing : this hook's full-cohort flattening + feature
        extraction + forest scoring is substantial (and roughly duplicates the
        subsequent scoring pass), so its wall-clock is captured here and folded
        into the round's reported TGE overhead by score_updates — otherwise
        tge_score_time_per_round_ms would understate the true cost by ~half."""
        self._pending_observe_ms = 0.0
        if self._model.ema is None or not self._model.gbdt.is_ready or not results:
            return
        _t_start = time.perf_counter()
        flat_updates = [self._flatten_parameters(fr.parameters) for _, fr in results]
        for i, (client_proxy, _fr) in enumerate(results):
            features = self._model.extract_features(flat_updates[i], flat_updates)
            self._model.observe_ema(
                self._get_client_id(client_proxy.cid), features, server_round
            )
        self._pending_observe_ms = (time.perf_counter() - _t_start) * 1000.0

    def on_round_end(self, server_round: int, aggregated_params: Parameters) -> None:
        """End-of-round: trigger periodic expert refitting.

        Delegates to TGEnsembleModel.on_round_end, which refits the
        IsolationForest / retrains the LSTM once accumulated data crosses
        each expert's warmup/refit_interval thresholds (see TGEnsembleModel
        in rmc/tg_ensemble.py). No-op most rounds.
        """
        self._model.on_round_end(server_round)


# ============================================================================
# PLUGGABLE STRATEGY WRAPPER
# ============================================================================

def _fit_res_partition_id(client_result):
    """partition_id metric of a (ClientProxy, FitRes) pair, or None if absent."""
    _client_proxy, fit_res = client_result
    metrics = getattr(fit_res, "metrics", None) or {}
    return metrics.get("partition_id")


def canonicalize_result_order(results, server_round=None):
    """Return `results` sorted into the canonical ascending-partition_id order.

    Drift investigation (2026-07-27): Flower/Ray's async simulation scheduler does
    not pin client-arrival order across runs even at a fixed seed, and two
    order-sensitivities followed from consuming results in arrival order — (1)
    KrumDefensePlugin.filter_updates breaks EXACT score ties with a STABLE sort
    over the positionally-indexed list, so a tie at the keep/drop cutoff flips the
    keep-set by arrival order (RMC duplicate-partition identities produce
    byte-identical updates); (2) FedAvg's weighted-sum reduction is not
    float-summation-order-invariant.

    Sorting on the client's OWN reported partition_id pins both. partition_id is on
    EVERY FitRes — the single return in FlowerClient.fit (flowerfl/client_app.py:
    149) sets metrics["partition_id"] after every honest AND attack branch — so the
    fallback below never fires in production; a missing key falls back to a STABLE
    cid order (never arrival order) with a loud warning. NOT keyed on flower cid
    (an ephemeral per-run node id, not a stable client identity).

    IDEMPOTENT: the key is deterministic and the sort stable, so re-sorting an
    already-canonical list is a no-op. This lets ScenarioStrategy canonicalize
    ONCE at its entry (so super.aggregate_fit AND _maybe_log_signals consume the
    SAME ordered list, avoiding cross-client signal-log misattribution) while
    PluggableStrategy still canonicalizes for direct (non-scenario) callers,
    harmlessly.
    """
    if any(_fit_res_partition_id(cr) is None for cr in results):
        n_missing = sum(1 for cr in results if _fit_res_partition_id(cr) is None)
        logger.warning(
            "[canonicalize_result_order] Round %s: %d/%d results missing "
            "partition_id metric; falling back to stable cid ordering (arrival "
            "order is NOT used). Every FlowerClient.fit() should report "
            "partition_id (flowerfl/client_app.py:149) — investigate the source "
            "of these updates.",
            server_round, n_missing, len(results),
        )

    def _key(client_result):
        client_proxy, _fit_res = client_result
        pid = _fit_res_partition_id(client_result)
        # (0, pid) sorts real partition_ids ascending first; (1, cid) puts any
        # missing-pid client after them in a STABLE, arrival-order-free order.
        return (0, float(pid)) if pid is not None else (1, str(client_proxy.cid))

    return sorted(results, key=_key)


class PluggableStrategy(Strategy):
    """
    Strategy wrapper that integrates Byzantine defense plugins.

    Wraps any Flower Strategy and injects plugin hooks at aggregation time.
    This is the main integration point between Flower's FL pipeline and
    custom defense logic.

    Hook execution order:
    1. on_round_start - plugins prepare for the round
    2. score_updates - plugins score each client's update
    3. filter_updates - plugins filter out low-scoring clients
    4. base_strategy.aggregate_fit - actual aggregation on filtered updates
    5. on_round_end - plugins process the result

    Multi-plugin composition (e.g. Krum + TGE, or the pre-correction-era
    Krum + ColdStartDefensePlugin) is SEQUENTIAL, not independent: each
    plugin's score_updates/filter_updates pair only sees the subset of
    results that survived every prior plugin in `self._plugins` (see
    aggregate_fit below). Plugin order therefore matters — it determines
    which defense gets first refusal on excluding a client.
    """

    def __init__(self, base_strategy: Strategy, plugins: List[ByzantineDefensePlugin] = None):
        """
        Args:
            base_strategy: The Flower strategy to wrap (e.g., FedAvg)
            plugins: List of defense plugins to apply
        """
        self._base = base_strategy
        self._plugins = plugins or []
        self._current_round = 0
        # Additive bookkeeping only (req 6, models): remembers the most
        # recently aggregated Parameters so the runner can persist the final
        # global model after the simulation ends. Never read by the FL loop
        # itself — aggregate_fit's return value (what Flower actually
        # consumes) is unaffected. Deliberately NOT overwritten by a None
        # aggregation (a round where every update was filtered out) so this
        # always holds the last successfully aggregated model.
        self._last_aggregated_parameters = None
        # H3 / (schema v5): the round's POST-`filter_updates` aggregation
        # coefficients, {raw cid -> a_i}. Recorded here because this is the only
        # place that sees BOTH the full participant list and the survivor set
        # the base strategy actually aggregates. Keyed by the round it was
        # computed in so a consumer can never be handed another round's numbers
        # (the v1.17 cross-attribution lesson). See aggregation_coefficients.
        self._round_coefficients: Optional[Dict[str, float]] = None
        self._round_coefficients_round: Optional[int] = None
        self._coefficient_base_warned = False
        # H4 § 6 chain trace (additive bookkeeping, spec 2026-08-16 + erratum
        # A): per round, the canonicalized input cid order, each plugin
        # stage's input/dropped cids, and the kept set handed to the base
        # aggregation. Two consumers: (1) flowerfl/h4_diagnostics.py joins it
        # with scenario ground truth for the per-unit `h4_diagnostics` block;
        # (2) ScenarioStrategy._maybe_log_signals uses each stage's input
        # order to key positional plugin `_round_scores` by CID — without
        # that, a Krum/TrustScore stage placed DOWNSTREAM of a filtering
        # plugin (the H4 § 7c-bis chains) would be positionally joined
        # against the FULL round ordering and silently misattribute scores
        # across clients (the v1.17 bug class). Never read by the FL loop.
        self._h4_chain_trace: Dict[int, Dict[str, Any]] = {}

        plugin_names = [p.name for p in self._plugins]
        print(f"[PluggableStrategy] Wrapping {type(base_strategy).__name__} "
              f"with plugins: {plugin_names}")

    # --- Delegate configuration methods to base strategy ---

    def initialize_parameters(self, client_manager):
        return self._base.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        self._current_round = server_round
        # Notify plugins
        num_clients = client_manager.num_available()
        for plugin in self._plugins:
            plugin.on_round_start(server_round, num_clients)
        return self._base.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._base.configure_evaluate(server_round, parameters, client_manager)

    # --- H3 aggregation-coefficient bookkeeping (schema v5, ) ---

    def aggregation_coefficients(self, server_round: int) -> Optional[Dict[str, float]]:
        """This round's post-defense aggregation coefficients, or ``None``.

        ``{raw cid -> a_i}`` over EVERY client that participated in
        `server_round`: ``a_i = w_i / Σ_j w_j`` across the survivors of the
        plugin chain (where ``w`` is the possibly-rewritten ``num_examples``
        that FedAvg actually weights on), and exactly ``0.0`` for a client the
        chain hard-dropped. The values sum to 1.0 whenever anything was
        aggregated.

        Returns ``None`` — never a fabricated number — when (a) nothing was
        recorded for `server_round` (a round that bypassed the plugin path, so
        another round's numbers must not leak into it), or (b) the base
        strategy is not the num-examples-weighted FedAvg this formula
        describes.

        This is the quantity the locked `data/h3_constants.json` rejoin-success
        rule is defined on (`blocked: coefficient == 0`). The signal log's
        legacy `effective_weight` is the RAW pre-filter count and cannot
        express it (v1.10 § 5.1).
        """
        if self._round_coefficients_round != int(server_round):
            return None
        return self._round_coefficients

    def _record_aggregation_coefficients(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        filtered_results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """Compute and stash this round's coefficients (see the accessor)."""
        self._round_coefficients_round = int(server_round)

        # The formula IS FedAvg's aggregation math. FedMedian/FedTrimmedAvg
        # subclass FedAvg but do not weight this way, so an exact type check —
        # not isinstance — is what keeps us from logging a number that is not
        # the coefficient. Every praxis scenario deployment wraps a plain
        # FedAvg (flowerfl/server_app.py:348).
        if type(self._base) is not FedAvg:
            if not self._coefficient_base_warned:
                self._coefficient_base_warned = True
                logger.warning(
                    "[PluggableStrategy] base strategy is %s, not FedAvg — "
                    "aggregation_coefficient will be logged as null rather "
                    "than a value that is not the aggregation coefficient",
                    type(self._base).__name__,
                )
            self._round_coefficients = None
            return

        # Sum (rather than overwrite) per cid: a duplicated cid would otherwise
        # silently drop weight out of the normalizer.
        survivor_weights: Dict[str, float] = defaultdict(float)
        for client_proxy, fit_res in filtered_results:
            survivor_weights[str(client_proxy.cid)] += float(fit_res.num_examples)

        total = float(sum(survivor_weights.values()))
        coefficients: Dict[str, float] = {
            str(client_proxy.cid): 0.0 for client_proxy, _fit_res in results
        }
        if total > 0.0:
            for cid, weight in survivor_weights.items():
                # A survivor cid absent from `results` cannot happen (the chain
                # only ever narrows), but writing it in keeps the round's
                # coefficients summing to 1.0 if it ever did.
                coefficients[cid] = weight / total
        self._round_coefficients = coefficients

    # --- Core aggregation with plugin hooks ---

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with Byzantine defense plugins.

        Runs every plugin's score_updates/filter_updates pair in
        sequence (see class docstring for why order matters), then delegates
        the surviving updates to the wrapped base_strategy for the actual
        FedAvg/etc. aggregation math. Returns (None, {}) — Flower's
        convention for "nothing to aggregate this round" — if there were no
        results to begin with, or if any single plugin filters out every
        remaining client.
        """

        if not results:
            # H4 § 6: a round with ZERO incoming FitRes must
            # still appear in the chain trace, or build_h4_diagnostics would
            # silently omit it instead of reporting it as aggregating nothing.
            # Recorded as input [], stages [], kept [] — kept_set_size 0 with
            # zero per-layer drops, truthfully DISTINGUISHABLE from an
            # all-filtered blackout round (whose stages carry the drops).
            self._h4_chain_trace[int(server_round)] = {
                "input_cids": [], "stages": [], "kept_cids": [],
            }
            return None, {}

        # Canonicalize arrival order BEFORE any plugin sees the results (drift
        # investigation, 2026-07-27). ScenarioStrategy.aggregate_fit already
        # canonicalizes at its OUTER entry (so its _maybe_log_signals consumes the
        # same ordering the plugins scored — see that method) and calls
        # super.aggregate_fit == this method; the helper is idempotent, so this
        # re-sort is a harmless no-op for scenario runs and the sole pin for direct
        # (non-scenario) PluggableStrategy callers.
        results = canonicalize_result_order(results, server_round)

        print(f"\n[PluggableStrategy] Round {server_round}: "
              f"{len(results)} updates received")

        # Run each plugin's scoring and filtering. filtered_results is
        # reassigned after each plugin, so subsequent plugins in the list
        # only score/filter the subset that already survived — this is the
        # SEQUENTIAL composition described in the class docstring.
        # Cohort observation : give every plugin the COMPLETE, unfiltered
        # cohort before any filtering, so a plugin composed after an upstream
        # filter (TGE′'s EMA leg behind Krum) can observe all participants. Base
        # class no-op — incumbent plugins are unaffected.
        for plugin in self._plugins:
            plugin.observe_cohort(results, server_round)

        # H4 § 6 chain trace (see __init__): record the canonicalized input
        # order, each stage's input/dropped cids, and the final kept set.
        trace: Dict[str, Any] = {
            "input_cids": [str(cp.cid) for cp, _fr in results],
            "stages": [],
            "kept_cids": [],
        }
        self._h4_chain_trace[int(server_round)] = trace

        filtered_results = results
        for plugin in self._plugins:
            stage_input_cids = [str(cp.cid) for cp, _fr in filtered_results]
            scores = plugin.score_updates(filtered_results, server_round)
            filtered_results = plugin.filter_updates(filtered_results, scores)
            surviving_cids = {str(cp.cid) for cp, _fr in filtered_results}
            trace["stages"].append({
                "plugin": plugin.name,
                "input_cids": stage_input_cids,
                "dropped_cids": [
                    cid for cid in stage_input_cids if cid not in surviving_cids
                ],
            })

            if not filtered_results:
                print(f"[PluggableStrategy] WARNING: All updates filtered out by {plugin.name}")
                # Nothing is aggregated this round, so EVERY participant's
                # aggregation coefficient is 0.0 — the h3_constants.json
                # `blocked` outcome class, recorded rather than left stale.
                # EMPTY-ROUND SEMANTICS (H4 spec § 6, measured EXP-052
                # behavior, preserved deliberately): returning (None, {})
                # leaves the global model UNCHANGED this round; the trace's
                # empty kept set is the blackout diagnostic, not a bug.
                self._record_aggregation_coefficients(server_round, results, [])
                return None, {}

        trace["kept_cids"] = [str(cp.cid) for cp, _fr in filtered_results]
        print(f"[PluggableStrategy] After filtering: {len(filtered_results)}/{len(results)} updates")

        # H3 schema v5 : record the post-filter coefficients BEFORE the
        # base aggregation, from the exact survivor list handed to it.
        self._record_aggregation_coefficients(server_round, results, filtered_results)

        # Delegate to base strategy for actual aggregation
        aggregated_params, metrics = self._base.aggregate_fit(
            server_round, filtered_results, failures
        )

        # Post-aggregation hooks
        if aggregated_params is not None:
            for plugin in self._plugins:
                plugin.on_round_end(server_round, aggregated_params)
            # req 6 (models): remember the last successfully aggregated
            # model (see __init__ docstring note). Purely additive.
            self._last_aggregated_parameters = aggregated_params

        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return self._base.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self._base.evaluate(server_round, parameters)

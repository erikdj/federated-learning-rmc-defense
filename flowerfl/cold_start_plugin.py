"""
Cold-Start Detector Plugin for Flower Byzantine Defense.

Phase 4 (P4.1/P4.2): Wraps the XGBoost cold-start detector as a
ByzantineDefensePlugin, enabling composition with baseline defenses
(Krum, TrustScore) via PluggableStrategy.

Integration mode: soft weighting, not hard rejection.
    effective_weight = baseline_weight × (1 - g(risk))
where g is a configurable weighting function (linear, squared, sigmoid).

The plugin tracks client tenure (rounds since first seen) and only
applies the detector during the cold-start window (first k rounds).
After k rounds, the client is considered established and the plugin
assigns full trust (risk=0).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from flowerfl.byzantine_defense import ByzantineDefensePlugin
from flowerfl.cold_start_detector import ColdStartDetector, FeatureExtractor, Family
from flowerfl.signal_logger import compute_per_client_signals, flatten_parameters

logger = logging.getLogger(__name__)


def _weighting_linear(risk: float) -> float:
    """Linear: trust = 1 - risk."""
    return max(0.0, 1.0 - risk)


def _weighting_squared(risk: float) -> float:
    """Squared: trust = (1 - risk)^2.  Penalizes high-risk more aggressively."""
    return max(0.0, (1.0 - risk) ** 2)


def _weighting_sigmoid(risk: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
    """Sigmoid: smooth step from 1 (low risk) to 0 (high risk)."""
    x = steepness * (risk - midpoint)
    return 1.0 / (1.0 + math.exp(x))


WEIGHTING_FUNCTIONS = {
    "linear": _weighting_linear,
    "squared": _weighting_squared,
    "sigmoid": _weighting_sigmoid,
}


class ColdStartDefensePlugin(ByzantineDefensePlugin):
    """Byzantine defense plugin using the XGBoost cold-start detector.

    During the first k rounds after a client is first seen, the plugin
    computes a risk score using the trained detector and applies a
    soft weight: trust = g(risk). After k rounds, trust = 1.0.

    Unlike Krum/TrustScore which filter (remove) untrusted clients,
    this plugin REWEIGHTS them — no client is dropped entirely.
    """

    def __init__(
        self,
        detector: ColdStartDetector,
        k: int = 3,
        weighting: str = "linear",
    ):
        """
        Args:
            detector: Trained ColdStartDetector instance.
            k: Cold-start window size (rounds after first seen).
            weighting: Weighting function name ("linear", "squared", "sigmoid").
        """
        self._detector = detector
        self._k = k
        self._weighting_name = weighting
        self._weighting_fn = WEIGHTING_FUNCTIONS[weighting]
        self._extractor = FeatureExtractor()

        # Tenure tracking: identity -> first_seen_round. Keyed by the
        # scenario's LOGICAL identity when ScenarioStrategy installs the
        # per-round identity map (v1.15) — so a malicious identity reset
        # (client_N_newM on the same raw cid) genuinely re-enters the
        # cold-start window instead of inheriting the old tenure. Falls
        # back to the raw flower cid when no map exists.
        self._first_seen: Dict[str, int] = {}
        # Per-round scores for signal logging
        self._round_scores: Dict[int, Dict[int, float]] = {}
        # Defense overhead instrumentation (Task 4c.4): wall-clock per round (ms)
        self._timing_per_round: list[float] = []
        # Confounder-control scoring log (Task 4c.5): (round, identity, trust).
        # With an identity map installed this records the logical identity at
        # scoring time (strictly better than the runner's post-hoc
        # partition->logical translation, which is lossy after an identity
        # reset); the runner's translator passes unknown/logical ids through
        # unchanged, so both keying modes remain compatible with it.
        self._scoring_log: list[tuple[int, str, float]] = []

    @property
    def name(self) -> str:
        return "ColdStartDetector"

    def on_round_start(self, server_round: int, num_clients: int) -> None:
        pass

    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """Score each client: 1.0 = fully trusted, 0.0 = fully untrusted.

        Clients in their cold-start window (first k rounds) get scored
        by the XGBoost detector. Established clients get 1.0.
        """
        _t_start = time.perf_counter()
        if not results:
            self._timing_per_round.append(0.0)
            return {}

        # Flatten parameters for signal computation
        flat_updates = []
        train_losses = []
        num_examples_list = []
        for _, fit_res in results:
            flat_updates.append(flatten_parameters(fit_res.parameters))
            tl = fit_res.metrics.get("train_loss")
            try:
                train_losses.append(float(tl) if tl is not None else None)
            except (TypeError, ValueError):
                train_losses.append(None)
            num_examples_list.append(int(fit_res.num_examples))

        # Compute Family S signals for this round
        signals = compute_per_client_signals(
            flat_updates, train_losses, num_examples_list
        )

        scores = {}
        resolved_ids = []  # per-index identity keys, reused by the log below
        for idx, (client_proxy, fit_res) in enumerate(results):
            # Key tenure by logical identity when available (v1.15); an
            # identity reset must re-enter the cold-start window.
            cid = self.resolve_identity(str(client_proxy.cid)) or str(client_proxy.cid)
            resolved_ids.append(cid)

            # Track tenure
            if cid not in self._first_seen:
                self._first_seen[cid] = server_round

            rounds_since_join = server_round - self._first_seen[cid] + 1

            if rounds_since_join > self._k:
                # Established client — full trust
                scores[idx] = 1.0
            else:
                # Cold-start window — run detector
                features = self._extractor.extract_s(signals[idx])
                risk = float(self._detector.predict_risk(
                    features.reshape(1, -1)
                )[0])
                trust = self._weighting_fn(risk)
                scores[idx] = trust

                logger.debug(
                    f"[ColdStart] Client {cid} round {server_round} "
                    f"(tenure {rounds_since_join}/{self._k}): "
                    f"risk={risk:.3f} trust={trust:.3f}"
                )

            # Confounder-control: record (round, cid, trust) for every scored
            # client (cold-window OR established). Runner translates flower_cid
            # to logical_cid post-hoc using the strategy's partition map.
            self._scoring_log.append((int(server_round), cid, float(scores[idx])))

        self._round_scores[server_round] = scores

        # Per-round observability for sweep diagnostics (added 2026-05-15).
        # Emits one line per round, regex-parseable by downstream tools.
        # Reports cold-start window scoring only (clients with tenure <= k);
        # established clients always get trust=1.0 and aren't informative.
        cold_scores = [s for idx, s in scores.items()
                       if (server_round - self._first_seen[resolved_ids[idx]] + 1) <= self._k]
        if cold_scores:
            n_flagged = sum(1 for s in cold_scores if s < 0.5)
            print(
                f"[ColdStart] round={server_round} "
                f"n_cold={len(cold_scores)} "
                f"n_flagged={n_flagged} "
                f"score_min={min(cold_scores):.3f} "
                f"score_mean={sum(cold_scores)/len(cold_scores):.3f} "
                f"score_max={max(cold_scores):.3f}",
                flush=True,
            )

        # Defense overhead bookkeeping (Task 4c.4).
        _t_elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
        self._timing_per_round.append(_t_elapsed_ms)

        return scores

    def filter_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        scores: Dict[int, float],
        threshold: float = 0.0,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Soft reweighting: scale num_examples by trust score.

        Unlike hard filtering (Krum/TrustScore), this keeps ALL clients
        but modulates their effective weight in FedAvg aggregation.
        FedAvg weights by num_examples, so scaling num_examples by
        trust score implements soft reweighting.
        """
        reweighted = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            trust = scores.get(idx, 1.0)
            if trust < 1.0:
                # Scale num_examples to reduce this client's weight
                scaled_examples = max(1, int(fit_res.num_examples * trust))
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=fit_res.parameters,
                    num_examples=scaled_examples,
                    metrics=fit_res.metrics,
                )
                reweighted.append((client_proxy, new_fit_res))
            else:
                reweighted.append((client_proxy, fit_res))
        return reweighted

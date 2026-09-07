"""Fingerprint defense plugin — identity-bound re-entry prevention (H3).

Design authority
----------------
* v1.10 § 5.0 **D4** — "**Hard-drop (coefficient = 0).** The FP plugin sits
  **after** the soft-reweight incumbent; on a Mahalanobis-matched flagged
  re-entrant its action is a true **coeff = 0** hard-block. The incumbent
  `max(1,·)` floor would leave tiny non-zero mass → `re_entry_block_rate`
  structurally 0; hard-drop makes it measurable and honors
  `h3_constants.json` `blocked: coefficient == 0` **as written**."
* v1.10 § 5.1 **INTEGRITY ASSERTION** — the registry's re-link decisions must be
  "computable from the signal log **independent of enforcement**", written
  "**regardless of any downstream action** (block, downweight, or accept)".
* v1.10 § 5.1 gate **(e)** — `FitRes.metrics["fingerprint"]` present for 100%
  of client-rounds.
* `docs/harness/architecture.md` "Plugin integration"; chain order
  `[Krum, TGE, Fingerprint]` — this plugin is always **last**.

Why observation lives in `observe_cohort`, not `score_updates`
--------------------------------------------------------------
`PluggableStrategy.aggregate_fit` calls `observe_cohort` on every plugin with
the **complete, unfiltered** cohort before any plugin filters
(`byzantine_defense.py`, the hook), then chains
`score_updates`/`filter_updates` over the *shrinking* survivor set. Because
this plugin is last, anything an upstream detector dropped would never reach its
`score_updates` — and a dropped re-entrant is exactly the event H3 must score.
Registering fingerprints and emitting re-link assertions from `observe_cohort`
therefore makes the assertions independent of **both** upstream filtering and
this plugin's own enforcement action, which is what turns
"detector-independent" into a mechanical property.

Deliberate difference from the PHASE7_DESIGN sketch
---------------------------------------------------
PHASE7 sketched `filter_updates` as soft reweighting "matches
ColdStartDefensePlugin pattern", with `num_examples = max(1, ·)`. **D4
supersedes that**: a `max(1, ·)` floor leaves non-zero aggregation mass, which
makes the secondary `re_entry_block_rate` structurally zero. The default here is
a true hard drop. `downweight` and `accept` exist only so the pre-registered
integrity test can flip the enforcement action; they are never the H3 arm.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from flowerfl.byzantine_defense import ByzantineDefensePlugin
from flowerfl.fingerprint import FINGERPRINT_DIM, FeatureSpecError, decode_fingerprint
from flowerfl.fingerprint_registry import FingerprintRegistry, MatchAssertion

logger = logging.getLogger(__name__)

#: The metrics key the client transports its 180-dim fingerprint under.
FINGERPRINT_METRIC_KEY = "fingerprint"


class EnforcementMode(str, Enum):
    """What the plugin DOES with a matched flagged re-entrant.

    HARD_DROP  — D4, the pre-registered H3 action: the update is removed, so its
                 aggregation coefficient is exactly 0.
    DOWNWEIGHT — the incumbent `max(1, ·)` soft floor. **Not** the H3 arm; kept
                 so the § 5.1 integrity test can flip enforcement.
    ACCEPT     — no action at all. Same purpose.
    """

    HARD_DROP = "hard_drop"
    DOWNWEIGHT = "downweight"
    ACCEPT = "accept"


ENFORCEMENT_MODES: Tuple[str, ...] = tuple(mode.value for mode in EnforcementMode)


class FingerprintDefensePlugin(ByzantineDefensePlugin):
    """Links a returning device to its prior flagged identity and blocks it."""

    def __init__(
        self,
        registry: FingerprintRegistry,
        run_id: str = "",
        enforcement_mode: str = EnforcementMode.HARD_DROP.value,
        expected_dim: int = FINGERPRINT_DIM,
    ):
        """
        Args:
            registry: An already-constructed registry carrying the LOCKED τ and
                Mahalanobis metric for the cohort being run.
            run_id: Stamped into `reentry_event_key` (`{run_id}:{round}:{cid}`).
            enforcement_mode: `hard_drop` (D4 default) | `downweight` | `accept`.
            expected_dim: Fingerprint dimension to validate against.
        """
        try:
            mode = EnforcementMode(str(enforcement_mode))
        except ValueError as exc:
            raise ValueError(
                f"unknown enforcement mode {enforcement_mode!r}; "
                f"expected one of {ENFORCEMENT_MODES}"
            ) from exc

        self._registry = registry
        self._run_id = str(run_id)
        self._mode = mode
        self._expected_dim = int(expected_dim)

        # Re-link assertions, one row per NEW-CID appearance. Written from
        # observe_cohort and NEVER touched by enforcement (§ 5.1 integrity).
        self._reentry_events: List[Dict[str, Any]] = []
        # THIS ROUND's cohort, keyed by CLAIMED IDENTITY, never by Flower CID
        # (see _session_key). Round-scoped on purpose: a run-long set would
        # make every client that merely sat out a sampling round look like an
        # upstream rejection. See score_updates.
        self._round_sessions: set[str] = set()
        self._cohort_round: Optional[int] = None
        # The round this plugin first observed anything — the enrollment round.
        # Sessions first seen then are INITIAL ENROLLMENTS, not re-entries.
        self._enrollment_round: Optional[int] = None
        self._initial_enrollments: set[str] = set()
        self._warned_unmapped = False
        # Per-round continuous scores, logged by convention (see the
        # ByzantineDefensePlugin docstring: scenario_strategy duck-types this).
        self._round_scores: Dict[int, Dict[int, float]] = {}
        self._blocked_by_round: Dict[int, Tuple[str, ...]] = {}
        self._last_scored_round = 0
        # gate (e) telemetry — denominator is PARTICIPATING client-rounds
        # (excludes discovery-round fits and scenario-dropped clients; see
        # `participating_client_round_count` and EMISSION_CONTRACT § 4.5).
        self._participating_client_round_count = 0
        self._participating_missing_fingerprint_count = 0

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "Fingerprint"

    @property
    def registry(self) -> FingerprintRegistry:
        return self._registry

    @property
    def enforcement_mode(self) -> EnforcementMode:
        return self._mode

    @property
    def reentry_events(self) -> List[Dict[str, Any]]:
        """Schema-v5 re-entry rows for the signal logger to drain.

        Returned as the live list so the logger can consume incrementally; it
        is append-only and never rewritten.
        """
        return self._reentry_events

    @property
    def initial_enrollments(self) -> Tuple[str, ...]:
        """Sessions enrolled in the first observed cohort — NOT re-entry events."""
        return tuple(sorted(self._initial_enrollments))

    @property
    def enrollment_round(self) -> Optional[int]:
        """The first round this plugin observed a cohort."""
        return self._enrollment_round

    @property
    def participating_client_round_count(self) -> int:
        """Gate (e) denominator: client-rounds this plugin actually observed.

        The `participating_` prefix is load-bearing, not decoration. Two classes
        of client-round are NOT in it:

        1. **Discovery-round fits.** `ScenarioStrategy.aggregate_fit` returns
           before `super` on the discovery round, so `observe_cohort` never
           sees them. That bypass is deliberate and documented there: the
           discovery round dispatches ALL clients including unscheduled ones,
           and routing them here would enrol devices the scenario never
           scheduled and set `_enrollment_round` off a non-scenario round —
           corrupting `_is_initial_enrollment`, the logic that keeps initial
           enrolments out of the re-entry numerator.
        2. **Clients the scenario dropped.** `control_benign_churn` deliberately
           excuses 3–5 clients a round; its census is 959, not 1 000.

        EMISSION_CONTRACT_20260808 § 4.5 fixes the reading for both: gate (e)'s
        "100 % of client-rounds" is "100 % of **participating** client-rounds".
        The name says so because a bare `client_round_count` in a custody record
        reads as a whole-run rate, and no reader should have to find a docstring
        to learn that it is not one.
        """
        return self._participating_client_round_count

    @property
    def participating_missing_fingerprint_count(self) -> int:
        """Participating client-rounds that carried no usable fingerprint."""
        return self._participating_missing_fingerprint_count

    def participating_fingerprint_emission_rate(self) -> float:
        """Gate (e): fraction of PARTICIPATING client-rounds with a fingerprint.

        See :attr:`participating_client_round_count` for what the denominator
        excludes and why (EMISSION_CONTRACT_20260808 § 4.5).
        """
        if self._participating_client_round_count == 0:
            return 1.0
        return 1.0 - (
            self._participating_missing_fingerprint_count
            / self._participating_client_round_count
        )

    def blocked_cids(self, server_round: int) -> Tuple[str, ...]:
        return self._blocked_by_round.get(int(server_round), ())

    @property
    def run_id(self) -> str:
        """The run identity this plugin stamps into every `reentry_event_key`.

        Exposed because it is the ONLY value that binds a unit's result JSON to
        the event rows it produced: the key is literally
        `{run_id}:{round}:{cid}`, so an offline scorer that joins on it cannot
        be handed events from a different run of the same design cell.
        """
        return self._run_id

    def set_run_id(self, run_id: str) -> None:
        """Stamp the run id used in `reentry_event_key` (before round 1)."""
        self._run_id = str(run_id)

    # -- observation (enforcement-independent) ------------------------------

    def _extract_fingerprint(self, fit_res) -> Optional[np.ndarray]:
        payload = (getattr(fit_res, "metrics", None) or {}).get(FINGERPRINT_METRIC_KEY)
        if payload is None or payload == "":
            return None
        try:
            return decode_fingerprint(payload, expected_dim=self._expected_dim)
        except FeatureSpecError as exc:
            logger.warning("[Fingerprint] unusable fingerprint payload: %s", exc)
            return None

    def _session_key(self, cid: str) -> str:
        """The identity the client CLAIMS this round — the registry's only key.

        **Never the Flower CID.** In Flower *simulation* the raw cid is stable
        per virtual client for the whole run (v1.15), so it does NOT change at
        an identity reset: keying the registry on it would (a) hand the defense
        the cross-reset linkage the RMC threat model forbids and H3 exists to
        measure, and (b) make a reset produce no new-identity event at all, so
        the primary metric would have an empty numerator by construction.
        `ScenarioStrategy` installs {raw cid -> logical identity} every scored
        round precisely so stateful defenses key through it.

        The logical identity is treated as an OPAQUE session label here; nothing
        in the registry or the matcher parses it. Parsing it back to a partition
        happens only in ground-truth machinery (the scorer, cohort selection).
        """
        identity = self.resolve_identity(str(cid))
        if identity is None:
            if not self._warned_unmapped:
                logger.warning(
                    "[Fingerprint] no identity map installed — falling back to the "
                    "raw Flower cid as the session key. In a scenario run this is a "
                    "DEFECT: the raw cid is stable across identity resets, so no "
                    "re-entry event would ever be produced."
                )
                self._warned_unmapped = True
            return str(cid)
        return str(identity)

    def observe_cohort(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> None:
        """Register every participant's fingerprint and emit re-link assertions.

        Called by `PluggableStrategy` with the COMPLETE unfiltered cohort before
        any plugin filters. Performs no enforcement and does not mutate
        `results`.

        The observed-session set is reset per round. It records who was in THIS
        round's cohort, which is what `score_updates` needs to tell an upstream
        rejection from a non-participant.
        """
        if self._cohort_round != int(server_round):
            self._cohort_round = int(server_round)
            self._round_sessions = set()
        if self._enrollment_round is None:
            self._enrollment_round = int(server_round)

        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            session_key = self._session_key(cid)
            logical_id = self.resolve_identity(cid)
            self._participating_client_round_count += 1

            fingerprint = self._extract_fingerprint(fit_res)
            if fingerprint is None:
                self._participating_missing_fingerprint_count += 1
                logger.warning(
                    "[Fingerprint] round %s cid %s emitted no usable fingerprint "
                    "(gate (e) requires 100%% emission)",
                    server_round, cid,
                )
                continue

            self._round_sessions.add(session_key)
            result = self._registry.observe(
                session_key,
                fingerprint,
                server_round=server_round,
                logical_id=logical_id,
                flower_cid=cid,
            )
            if result.assertion is None:
                continue
            if self._is_initial_enrollment(session_key, server_round):
                # The device's FIRST enrollment in this run is not a re-entry:
                # nothing "returned". Emitting a row here would put an event in
                # the numerator's population that the ground-truth layer never
                # emits, corrupting the join and the denominators.
                self._initial_enrollments.add(session_key)
                continue
            self._reentry_events.append(
                self._event_row(cid, logical_id, server_round, result.assertion)
            )

    def _is_initial_enrollment(self, session_key: str, server_round: int) -> bool:
        """Is this new session an initial enrollment rather than a re-entry?

        v1.10 § 5.1 defines a re-entry event as "one row per client CID-appearance
        that is a **new CID for a RETURNING logical device**". At the start of a
        run every device enrolls for the first time; those appearances are not
        returns and must not enter the metric's populations.

        The registry cannot use ground truth to tell the two apart — inferring
        "returning" is the whole thing H3 measures. The observable that *is*
        legitimately available is timing: every identity present in the first
        cohort this plugin sees is an initial enrollment; a claimed identity
        that first appears in a LATER round is a returning-device candidate.
        That is exactly the design's device-level notion of "new", and it uses
        no information the server is not entitled to.

        Note this costs nothing in matching power: at the enrollment round no
        entry has been flagged yet (flagging happens in `score_updates`, after
        `observe_cohort`), so an enrollment-round assertion could never be a
        match anyway — `min_d` is `inf` by construction.
        """
        return self._enrollment_round is None or int(server_round) == self._enrollment_round

    def _event_row(
        self,
        cid: str,
        logical_id: Optional[str],
        server_round: int,
        assertion: MatchAssertion,
    ) -> Dict[str, Any]:
        """One schema-v5 re-entry row (v1.10 § 5.1 field contract).

        `gt_is_malicious` is deliberately NOT set here — the plugin has no
        ground truth. The scenario layer owns that field; the scorer joins on
        `reentry_event_key`.
        """
        row: Dict[str, Any] = {
            "reentry_event_key": f"{self._run_id}:{int(server_round)}:{cid}",
            "server_round": int(server_round),
            "current_cid": cid,
            "gt_logical_id": logical_id,
        }
        row.update(assertion.as_log_fields())
        return row

    # -- scoring ------------------------------------------------------------

    def score_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[int, float]:
        """0.0 for a matched flagged re-entrant, 1.0 for everyone else.

        Also infers this round's upstream verdict: this plugin is last in the
        chain, so a claimed identity that was in **this round's cohort** but is
        no longer in `results` was rejected by an upstream detector — i.e.
        flagged malicious. That flag is what makes the device a match candidate
        for its future re-entries.

        The comparison is strictly round-scoped. Against a run-long observed set
        every client that merely sat out a sampling round — or that the scenario
        schedule legitimately excused, which `control_benign_churn` does by
        design for 3-5 clients a round — would be mistaken for an upstream
        rejection and permanently flagged, manufacturing match candidates out of
        honest devices and inflating the false-link rate.
        """
        surviving = {self._session_key(str(proxy.cid)) for proxy, _ in results}
        for session_key in sorted(self._round_sessions - surviving):
            entry = self._registry.entry_for_session(session_key)
            # A.6: record the rejection UNCONDITIONALLY, before the flag gate.
            # The gate below is exactly what loses the signal for an entry that
            # was already inherited-flagged, and the A.5(a) Euclidean
            # counterfactual needs those rounds. Passive: it cannot flag, cannot
            # raise, and changes no re-link decision.
            self._registry.record_upstream_rejection(session_key, server_round)
            if not entry.flag_status:
                self._registry.flag(
                    session_key, reason="upstream_filter", server_round=server_round
                )

        scores: Dict[int, float] = {}
        for idx, (client_proxy, _fit_res) in enumerate(results):
            try:
                entry = self._registry.entry_for_session(
                    self._session_key(str(client_proxy.cid))
                )
            except KeyError:
                # Never observed (no fingerprint emitted, or scored without a
                # preceding observe_cohort): full trust, never blocked.
                scores[idx] = 1.0
                continue
            # Block only a re-entrant whose flag was INHERITED from a
            # fingerprint match — not a client an upstream detector flagged
            # directly, whose enforcement is that detector's business.
            matched_reentrant = entry.flag_status and entry.flag_reason == "inherited"
            scores[idx] = 0.0 if matched_reentrant else 1.0

        self._round_scores[int(server_round)] = dict(scores)
        self._last_scored_round = int(server_round)
        return scores

    # -- enforcement (D4) ---------------------------------------------------

    def filter_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        scores: Dict[int, float],
        threshold: float = 0.0,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """D4 hard-drop: a matched re-entrant's aggregation coefficient is 0.

        `threshold` exists only to satisfy the ABC signature — the decision is
        the registry's binary match, not a tunable cut.
        """
        survivors: List[Tuple[ClientProxy, FitRes]] = []
        blocked: List[str] = []

        for idx, (client_proxy, fit_res) in enumerate(results):
            cid = str(client_proxy.cid)
            if scores.get(idx, 1.0) > 0.0:
                survivors.append((client_proxy, fit_res))
                continue

            blocked.append(cid)
            if self._mode is EnforcementMode.HARD_DROP:
                logger.info(
                    "[Fingerprint] HARD-DROP cid %s (matched flagged re-entrant; "
                    "aggregation coefficient = 0)", cid,
                )
                continue
            if self._mode is EnforcementMode.DOWNWEIGHT:
                survivors.append(
                    (
                        client_proxy,
                        FitRes(
                            status=fit_res.status,
                            parameters=fit_res.parameters,
                            num_examples=max(1, int(fit_res.num_examples * 0.0)),
                            metrics=fit_res.metrics,
                        ),
                    )
                )
                continue
            survivors.append((client_proxy, fit_res))  # ACCEPT

        # Recorded for the SECONDARY enforcement metric; never read by the
        # primary D7 scorer, which runs on the logged assertions alone.
        # The ABC does not pass the round to filter_updates, so it is carried
        # from the score_updates call PluggableStrategy always pairs it with.
        self._blocked_by_round[self._last_scored_round] = tuple(blocked)
        return survivors

    def on_round_end(self, server_round: int, aggregated_params: Parameters) -> None:
        pass

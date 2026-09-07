"""Server-side fingerprint registry: Mahalanobis matching, EMA, generations.

Design authority
----------------
* v1.10 § 5.1 — "the 180-dim fingerprint (D8 proxy construct), Mahalanobis
  matching, the EMA α=0.1 registry update, identity binding, and the two-stage
  τ-calibration *procedure* (dev-FPR=1%, **LOCKED in
  `flowerfl/fingerprint_registry.py` before any eval run**) — as pre-registered."
* v1.10 § 5.1 / **D9 axis (ii)** — device hold-out: "τ **and** the Mahalanobis
  covariance are calibrated on the **even**-numbered base partitions only
  (`{0,2,…,18}`) ... and the adjudicating re-link metrics are computed **only**
  over re-entry events of the held-out ODD-partition devices."
* v1.10 § 5.1 **INTEGRITY ASSERTION** — "the registry's re-link decisions must
  be computable from the signal log independent of enforcement". This module is
  therefore *decision-only*: it computes and returns `MatchAssertion`s and never
  touches aggregation weights. Enforcement lives in `fingerprint_plugin.py`.
* `docs/PHASE7_DESIGN.md` — the `RegistryEntry` schema and the registry's three
  operations (new CID / returning CID / flagged).

PENDING ADDENDUM A — covariance estimator selection
---------------------------------------------------
**Status: AMENDMENT-REQUIRED, implemented as `within_device`.** The calibration
path estimates Σ from the **pooled within-device scatter**
(`MahalanobisMetric.from_within_device_population`), not the total covariance.

Adjudication: the within-device (within-class) scatter is the scientifically
correct estimator for an identity-linking metric — the total covariance is
dominated by the between-device spread, which is the signal the matcher needs,
so whitening by it divides the signal away (measured: within-device link rate at
the 1%-FPR τ collapses from ~1.0 to 0.04 on separable synthetic data). But
PHASE7_DESIGN says only "estimated from the honest sub-population", which does
not uniquely entail either estimator. This is therefore a **substantive
estimator selection, not a reading of the design text**, and it is being carried
into a dated addendum at the lane gate. `from_population` (total covariance) is
retained for ungrouped use and is NOT used by calibration.

Deliberate deviation from PHASE7_DESIGN, recorded
-------------------------------------------------
PHASE7 sketched an *online-refined* covariance ("as the registry accumulates
honest fingerprints, Σ is refined"). v1.10 § 5.1 supersedes that: τ **and** the
covariance are calibration constants, locked before any eval run and never
re-derived afterwards. An online-refined Σ would silently move the decision
boundary during evaluation and make the locked τ meaningless. The registry
therefore takes an already-locked `MahalanobisMetric` and never adapts it.

A.6 — per-observation log (PASSIVE)
-----------------------------------
The historical protocol's A.5(a), summarized by the H3 workflow in
`docs/reproduction/experiments.md`, pre-registers a
NON-GATING naive-Euclidean baseline comparator on the UN-WHITENED 180-dim
vector, and A.6 makes persisting that vector a blocking build prerequisite: the
schema-v5 re-entry row carries `asserted_match` / `asserted_parent_logical_id` /
`min_d` / `tau` but not the vector, so without a sibling artifact the comparator
"is unrecoverable without a re-run". The registry already sees every observed
vector, so it retains them here and `scripts/run_phase4_flower.py` exports them
in the run-end custody record. The signal-log schema stays v5 and is untouched.

The log is **write-only within this module**: nothing reads it back, so no value
in it can reach τ, Σ, a re-link decision or any gate. It is custody/analysis
evidence only.

Nothing here reads sealed material or prints seed values.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from flowerfl.fingerprint import FINGERPRINT_DIM

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Pre-registered EMA smoothing factor for the registry fingerprint update.
EMA_ALPHA: float = 0.1

#: A.6 (THRESHOLD_GROUNDING_20260808 § 5): hard bound on the per-observation
#: log so a pathological run cannot grow it without limit. The designed shape is
#: ~20 devices × ~50 rounds ≈ 1 000 rows (measured: ~2.0 MB of `indent=2` JSON);
#: this leaves 20× headroom, so the implied worst case is ~41 MB — chosen
#: deliberately, because losing the corpus costs a re-run while an oversized
#: result JSON only costs storage. Exceeding the bound DROPS rows and says so
#: loudly (see `FingerprintRegistry._record_observation`) — a silently truncated
#: corpus would bias the A.5(a) comparator.
OBSERVATION_LOG_MAX_ROWS: int = 20_000

#: A.6 bound on the upstream-rejection log. One row per rejected client-round,
#: so the designed shape is well under the observation log's; same 20× headroom
#: on the same reasoning, and the rows are tiny (two scalars).
UPSTREAM_REJECTION_LOG_MAX_ROWS: int = 20_000

#: D9 axis (ii): the even base partitions τ/Σ are calibrated on for the
#: ADJUDICATING cohort, and the odd base partitions the verdict is scored on.
ADJUDICATING_CALIBRATION_PARTITIONS: Tuple[int, ...] = tuple(range(0, 20, 2))
ODD_HOLDOUT_PARTITIONS: Tuple[int, ...] = tuple(range(1, 20, 2))


class RegistryPolicy(str, Enum):
    """Which prior sessions a NEW claimed identity is matched against.

    FLAG_GATED    — the DEPLOYED incumbent, and the default. Candidates are the
                    entries an upstream detector (or an earlier inherited
                    match) has FLAGGED. This is the pool `_nearest_flagged`
                    has always used, and it stays for H4's composition arms.

    IDENTITY_ONLY — the corrected H3 instrument described in
                    `docs/reproduction/experiments.md`. Every session
                    enrolls at first emission and the candidate pool is ALL
                    sessions first seen STRICTLY before the re-entrant, with no
                    flag gating at all.

    Why the second mode exists: under FLAG_GATED the realised re-link
    instrument measures P(flagged by the detector) x P(re-linked | flagged) —
    RQ2's detection performance multiplied into RQ3's identity question, which
    is not what v1.10 D7 designates as "detector-independent re-link
    identification". The director ruling of 2026-08-14 makes the identity-only
    instrument THE H3 test.

    In BOTH modes enforcement stays keyed on FLAG STATUS, never on the identity
    link: an honest device correctly re-identified as itself inherits nothing,
    because identity linking is not an enforcement decision. See
    :meth:`FingerprintRegistry.observe`.
    """

    FLAG_GATED = "flag_gated"
    IDENTITY_ONLY = "identity_only"


#: Absent declaration == the deployed incumbent, byte for byte.
DEFAULT_REGISTRY_POLICY: RegistryPolicy = RegistryPolicy.FLAG_GATED


class CalibrationCohort(str, Enum):
    """The two independently-calibrated τ/Σ pairs (v1.10 § 5.1).

    VALIDATION      — dev seeds, ALL 20 base partitions calibrated and scored.
                      Informs; never adjudicates.
    ADJUDICATING    — sealed `h3_eval` seeds; τ/Σ from the EVEN partitions only,
                      metrics scored only on the held-out ODD partitions.
    """

    VALIDATION = "validation"
    ADJUDICATING = "adjudicating"


# ===========================================================================
# LOCKED τ CONSTANTS — pre-registration gate (c)
# ===========================================================================
# v1.10 § 5.1: "BOTH τ values locked in code (all-device validation-τ AND
# even-device adjudicating-τ, both derived from the calibration logs BEFORE any
# eval scenario runs, gate (c))" and "τ is NEVER re-derived after any eval
# scenario runs."
#
# They are `None` until `scripts/calibrate_fp_threshold.py` has run on the two
# attack-free control scenarios (D2) and its output has been reviewed and
# committed here. Every accessor raises until then: a silent default τ would be
# an un-pre-registered threshold, which is exactly what gate (c) forbids.
#
# The 180×180 covariance cannot sensibly live inline, so the locked precision
# matrices live in the calibration artifact below and the artifact's SHA-256 is
# pinned *in this module* — the hash is the thing that is locked in code.
# ---------------------------------------------------------------------------
# The v2 lock (2026-08-14) PROSPECTIVELY REPLACES the 2026-08-10 v1 lock under
# the corrected-instrument protocol summarized in
# docs/reproduction/experiments.md: same EXP-050 calibration corpus, R5 58-dim
# metric-level feature mask, fully-evaluated screens. The v1 artifact
# (`data/fingerprint_tau_locked_v1.json`, τ 24.378021816195496 /
# 28.01139899170691) stays committed untouched as the historical instrument of
# record for EXP-052/054/055/056.
TAU_VALIDATION_ALL_DEVICES: Optional[float] = 18.639429816855873
TAU_ADJUDICATING_EVEN_DEVICES: Optional[float] = 26.466874982783164

#: SHA-256 of the public metadata-redacted calibration artifact (per cohort's
#: precision matrix + provenance). Pinned here so the artifact cannot drift.
CALIBRATION_ARTIFACT_SHA256: Optional[str] = "034ace2b1b85074d44e39c663f3345612996269a5f104b674a5d1a1f376f847f"
CALIBRATION_ARTIFACT_PATH = PROJECT_ROOT / "data" / "fingerprint_tau_locked_v2.json"

#: Filled in alongside the τ values: calibration run ids, script SHA, date,
#: commit — the provenance a reviewer needs to verify gate (c) precedence.
TAU_LOCK_RECORD: Dict[str, Any] = {
    "status": "LOCKED",
    "locked_on": "2026-08-14",
    # The calibrator that WROTE the committed artifact: the post-PR-#59
    # zero-warning v3 re-lock run (R5 feature mask, fully-evaluated screens,
    # homogeneity gates PASS both cohorts) —
    # results/20260814/h3_relock_r5_v3/calibration_stdout.txt.
    "script_sha256": "93be1f51cceade0c522cdfee424e8b27eb4ce29a034c0266f363b8509c4a7132",
    "fpr_target": 0.01,
    "observed_scenarios": ["control_benign_churn", "control_honest"],
    "observed_seeds": [42, 137, 256, 314, 500],
    # The ten durable MLflow child run ids the 9795 calibration rows came from,
    # each paired with the fleet unit that produced it, so an auditor can map
    # the corpus back to custody records rather than to a scratch path. Same
    # pairs are stamped in the artifact's `_meta.observed_units`; the manifest
    # they were resolved from is committed at
    # results/20260810/exp050_tau_calibration/mlflow_run_ids.json.
    "mlflow_experiment_id": "501",
    "mlflow_parent_run_id": "da4b4a6b766045f187a904f12328bc61",
    "observed_run_ids": [
        "2c5883cabdd640648dc54dc5eb99acdd",
        "34a13853b2a44251941845b6202c56ab",
        "600a459db2b74bae877bd62fc5231ef6",
        "75640a3268a947d7a75ad2a057f921d7",
        "876065efe14547e6b090d88ffd8edd9c",
        "87fcfb2cbb954dd095503396389897a6",
        "89cc7b200d454ec4b1113729cbcbc979",
        "b448253b57594ba09a26a8edcec7d357",
        "b9ea17a568cb4ec899e2698225cb44db",
        "cf783d7c4b7f4889965ff2126e602390",
    ],
    "observed_units": {
        "control_benign_churn__tge_fp__persistent_optimizer__seed42.json": "87fcfb2cbb954dd095503396389897a6",
        "control_benign_churn__tge_fp__persistent_optimizer__seed137.json": "34a13853b2a44251941845b6202c56ab",
        "control_benign_churn__tge_fp__persistent_optimizer__seed256.json": "b448253b57594ba09a26a8edcec7d357",
        "control_benign_churn__tge_fp__persistent_optimizer__seed314.json": "75640a3268a947d7a75ad2a057f921d7",
        "control_benign_churn__tge_fp__persistent_optimizer__seed500.json": "2c5883cabdd640648dc54dc5eb99acdd",
        "control_honest__tge_fp__persistent_optimizer__seed42.json": "876065efe14547e6b090d88ffd8edd9c",
        "control_honest__tge_fp__persistent_optimizer__seed137.json": "cf783d7c4b7f4889965ff2126e602390",
        "control_honest__tge_fp__persistent_optimizer__seed256.json": "600a459db2b74bae877bd62fc5231ef6",
        "control_honest__tge_fp__persistent_optimizer__seed314.json": "89cc7b200d454ec4b1113729cbcbc979",
        "control_honest__tge_fp__persistent_optimizer__seed500.json": "b9ea17a568cb4ec899e2698225cb44db",
    },
    # Addendum A's predeclared rule re-run on the R5-masked corpus:
    # validation selected outright (within_link_rate 0.9953 vs 0.9750,
    # delta > 0.01); adjudicating was a TIE (delta 0.000043 ≤ 0.01) which the
    # rule resolves to the SIMPLER metric. The v1 lock's adjudicating pick was
    # pooled_within_ledoit_wolf; the change is a product of the pre-stated rule
    # on the masked corpus, not a re-decision.
    "selected_metric": {
        "validation": "shrinkage_to_identity",
        "adjudicating": "shrinkage_to_identity",
    },
    # The code state that produced the v3 re-lock: master at the PR #59 merge
    # (het-screen fail-open fix; no-unresolved-screen-value-survives invariant).
    "git_commit": "1ebab1a",
    "calibration_source": "EXP-050 array 06764fe9 (10/10 SUCCEEDED, census 9795/9795)",
    "feature_selection": {
        "rule": "R5",
        "n_surviving": 58,
        "n_total": 180,
        "note": (
            "metric-level mask (dead ∪ pool-flagged ∪ lattice-flagged ∪ "
            "het>100 removed); per-dim outcomes recorded in the artifact's "
            "feature_selection block"
        ),
    },
    "supersedes": {
        "artifact": "data/fingerprint_tau_locked_v1.json",
        "locked_on": "2026-08-10",
        "authority": (
            "docs/superpowers/specs/2026-08-14-h3-corrected-instrument-"
            "preregistration.md § 3.1 (RATIFIED rev-5, methodology v1.50); "
            "prospective only — EXP-052/054/055/056 remain scored under v1"
        ),
    },
}


class TauNotLockedError(RuntimeError):
    """Raised when a locked τ / covariance is requested before calibration."""


class TauLockIntegrityError(RuntimeError):
    """Raised when the LOCKED calibration artifact fails verification at access.

    Deliberately NOT a subclass of `TauNotLockedError`. "Not locked yet" is a
    legitimate pre-lock state that `server_app.build_fingerprint_registry`
    downgrades to an observe-only registry; a lock that FAILS verification —
    artifact missing, bytes off the SHA-256 pin, cohort block absent — is a
    post-lock integrity failure. If it were catchable as "not locked", an H3
    evaluation with a damaged artifact would silently run with no re-link
    matching and produce invalid results. It must propagate and stop the run.
    """


def _tau_constant(cohort: "CalibrationCohort") -> Optional[float]:
    return {
        CalibrationCohort.VALIDATION: TAU_VALIDATION_ALL_DEVICES,
        CalibrationCohort.ADJUDICATING: TAU_ADJUDICATING_EVEN_DEVICES,
    }[CalibrationCohort(cohort)]


def _verified_calibration_payload(cohort: "CalibrationCohort") -> Dict[str, Any]:
    """The locked artifact's cohort block, hash-verified against the in-code pin.

    THE single verified read path. τ and Σ are two halves of one calibration —
    a τ paired with a drifted Σ changes every Mahalanobis decision just as much
    as a drifted τ does — so both accessors route through here rather than
    hashing only on the covariance path. Refusals are loud and of two distinct
    families: `TauNotLockedError` for the genuinely pre-lock state (downgradable
    to observe-only), `TauLockIntegrityError` for a post-lock artifact that
    fails verification (never downgradable — must stop the run). Callers must
    never silently fall back to a default in either case.
    """
    cohort = CalibrationCohort(cohort)
    if CALIBRATION_ARTIFACT_SHA256 is None:
        raise TauNotLockedError(
            f"the calibration artifact for cohort '{cohort.value}' has not been "
            "calibrated and locked. Run scripts/calibrate_fp_threshold.py on the "
            "two attack-free control scenarios and commit the result before any "
            "H3 evaluation scenario runs (v1.10 § 5.1 gate (c))."
        )
    if not CALIBRATION_ARTIFACT_PATH.exists():
        raise TauLockIntegrityError(
            f"locked calibration artifact missing: {CALIBRATION_ARTIFACT_PATH}"
        )
    raw = CALIBRATION_ARTIFACT_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != CALIBRATION_ARTIFACT_SHA256:
        raise TauLockIntegrityError(
            "locked calibration artifact sha256 mismatch: expected "
            f"{CALIBRATION_ARTIFACT_SHA256}, found {actual}"
        )
    payload = json.loads(raw)
    try:
        return payload["cohorts"][cohort.value]
    except KeyError as exc:
        raise TauLockIntegrityError(
            f"locked calibration artifact has no cohort '{cohort.value}'"
        ) from exc


def locked_tau(cohort: "CalibrationCohort") -> float:
    """The locked τ for a cohort, or a loud refusal if it is not calibrated yet.

    Verifies the calibration artifact's SHA-256 at access, exactly as
    `locked_metric` does. Before this went through `_verified_calibration_payload`
    the integrity guarantee was only true for callers that asked for the
    covariance: a caller wanting just the threshold got it back even with the
    artifact missing or its bytes drifted off the pin.

    It deliberately does NOT cross-check the returned τ against the artifact's
    own τ at runtime. That agreement is a property of the committed tree, and
    `tests/test_fingerprint_registry.py` asserts it directly; enforcing it here
    instead would make it impossible to exercise the post-lock code path with a
    simulated τ, which the wiring tests legitimately do.
    """
    value = _tau_constant(cohort)
    if value is None:
        raise TauNotLockedError(
            f"tau for cohort '{CalibrationCohort(cohort).value}' has not been "
            "calibrated and locked. Run scripts/calibrate_fp_threshold.py on the "
            "two attack-free control scenarios and commit the result before any "
            "H3 evaluation scenario runs (v1.10 § 5.1 gate (c))."
        )
    _verified_calibration_payload(cohort)
    return float(value)


def locked_metric(cohort: "CalibrationCohort") -> "MahalanobisMetric":
    """The locked Mahalanobis metric for a cohort (hash-verified on load)."""
    return MahalanobisMetric.from_dict(_verified_calibration_payload(cohort)["metric"])


# ===========================================================================
# Ground-truth partition parity (D9 axis (ii))
# ===========================================================================
_LOGICAL_ID_RE = re.compile(r"^client_(\d+)(?:_new(\d+)?)?$")


def partition_of(logical_id: str) -> int:
    """Base partition for a logical identity — the ground-truth key.

    Delegates to `ScenarioStrategy.LOGICAL_TO_PARTITION`
    (`flowerfl/scenario_strategy.py:107-111`), which is the single source of
    truth and is NEVER modified from here; a regex fallback covers identities
    the map does not enumerate. `tests/test_fingerprint_registry.py` asserts the
    two never diverge.

    NOTE the map's two legacy aliases: `client_9_new` → 10 and `client_19_new`
    → 20 (the duplicate partitions), which is why the map is consulted first.
    """
    try:  # local import: keep this module free of the flwr/torch import cost
        from flowerfl.scenario_strategy import ScenarioStrategy

        mapping = ScenarioStrategy.LOGICAL_TO_PARTITION
    except Exception:  # pragma: no cover - defensive
        mapping = {}
    if logical_id in mapping:
        return int(mapping[logical_id])
    match = _LOGICAL_ID_RE.match(str(logical_id))
    if not match:
        raise KeyError(f"unmappable logical identity: {logical_id!r}")
    return int(match.group(1))


def is_calibration_partition(logical_id: str, cohort: "CalibrationCohort") -> bool:
    """May this device's fingerprints enter the cohort's τ/Σ calibration?"""
    if CalibrationCohort(cohort) is CalibrationCohort.VALIDATION:
        return True
    return partition_of(logical_id) in ADJUDICATING_CALIBRATION_PARTITIONS


def is_holdout_partition(logical_id: str) -> bool:
    """Is this device in the ADJUDICATING scoring hold-out (odd partitions)?"""
    return partition_of(logical_id) in ODD_HOLDOUT_PARTITIONS


def select_calibration_vectors(
    records: Iterable[Mapping[str, Any]],
    cohort: "CalibrationCohort",
    logical_id_key: str = "logical_id",
    fingerprint_key: str = "fingerprint",
) -> np.ndarray:
    """Stack the fingerprints a cohort is allowed to calibrate on.

    The parity filter is applied HERE, at the point the population is built, so
    an odd-partition fingerprint cannot reach the ADJUDICATING covariance by any
    path — "asserted by construction, not by comment".
    """
    selected: List[np.ndarray] = []
    for record in records:
        logical_id = record[logical_id_key]
        if not is_calibration_partition(logical_id, cohort):
            continue
        selected.append(np.asarray(record[fingerprint_key], dtype=np.float64))
    if not selected:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(selected)


# ===========================================================================
# Mahalanobis metric
# ===========================================================================

def _conditioning_scale(residuals: np.ndarray) -> np.ndarray:
    """Per-dimension divisor that conditions the covariance without overflowing.

    A plain `np.std` is not usable here. Fingerprint dimensions reach ~1e150
    (the `tcp.payload` moments), so `std` squares them into float64 overflow and
    returns `inf`; a naive "replace non-finite with 1.0" guard would then leave
    the residual at 1e150 and poison every downstream estimate with NaN.

    The standard deviation is therefore computed on a max-abs-normalised copy
    and rescaled — exact algebra, bounded intermediates. A near-zero standard
    deviation is floored relative to the column's magnitude so a single outlier
    cannot blow the conditioned residual up.

    Any diagonal pre-scaling is absorbed by the precision estimated in the same
    basis, so this choice conditions the arithmetic without changing the
    Mahalanobis form.
    """
    max_abs = np.max(np.abs(residuals), axis=0)
    max_abs = np.where(np.isfinite(max_abs) & (max_abs > 0.0), max_abs, 1.0)
    std = np.std(residuals / max_abs, axis=0) * max_abs
    std = np.where(np.isfinite(std) & (std > 0.0), std, max_abs)
    return np.maximum(std, max_abs * 1e-6)


@dataclass(frozen=True)
class MahalanobisMetric:
    """d(x, μ) = √((x-μ)ᵀ Σ⁻¹ (x-μ)), with per-dimension pre-scaling.

    The 180 fingerprint dimensions span ~25 orders of magnitude (packet-flag
    means near 0, `tcp.payload` moments near 1e150), so the raw covariance is
    numerically hopeless. Each residual is divided by the population's
    per-dimension scale first and the (shrunk) correlation-scale precision is
    applied to the result — algebraically the same Mahalanobis form, expressed
    in a conditioned basis.

    `dim`, `scale` and `precision` are expressed in the SUBSPACE the metric was
    fitted on. When `mask` is set they describe the surviving dimensions only,
    and `input_dim` is the full width callers still hand in — see `mask`.
    """

    dim: int
    scale: np.ndarray
    precision: np.ndarray
    shrinkage: float = 0.0
    provenance: str = "identity"
    #: Ascending indices of the surviving dimensions (the R5 feature-selection
    #: rule), or None for the full-width metric. `EDA_MEMO.md` § 6.2 puts the
    #: mask HERE rather than in the emission contract: the raw 180-dim emission
    #: is unchanged, and every caller — registry, custody export, offline replay
    #: — keeps passing full-width vectors without knowing a subspace exists.
    mask: Optional[np.ndarray] = None
    #: Full vector width `mask` indexes into. Meaningless without `mask`.
    input_dim: Optional[int] = None

    @property
    def expected_input_dim(self) -> int:
        """The width of the vectors `distance` accepts (NOT `dim` when masked)."""
        return int(self.input_dim) if self.mask is not None else int(self.dim)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Restrict full-width vectors to the surviving dimensions.

        THE single place the mask is applied, so a caller can never fit on the
        subspace and then measure in the full space (or the reverse).
        """
        array = np.asarray(vectors, dtype=np.float64)
        if self.mask is None:
            return array
        return array[..., self.mask]

    @staticmethod
    def _validated_mask(mask: Any, input_dim: int) -> np.ndarray:
        indices = np.asarray(mask, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("mask selects no dimensions — there is nothing to fit")
        if indices.size != np.unique(indices).size:
            raise ValueError(f"mask contains duplicate indices: {indices.tolist()}")
        if indices.min() < 0 or indices.max() >= int(input_dim):
            raise ValueError(
                f"mask indices must lie in [0, {input_dim}); got "
                f"[{indices.min()}, {indices.max()}]"
            )
        return np.sort(indices)

    @classmethod
    def identity(cls, dim: int = FINGERPRINT_DIM) -> "MahalanobisMetric":
        """Σ = I — the PHASE7 cold-start metric, before any calibration."""
        return cls(
            dim=int(dim),
            scale=np.ones(int(dim), dtype=np.float64),
            precision=np.eye(int(dim), dtype=np.float64),
            shrinkage=0.0,
            provenance="identity",
        )

    @classmethod
    def from_population(
        cls,
        vectors: np.ndarray,
        shrinkage: Optional[float] = None,
        provenance: str = "population",
    ) -> "MahalanobisMetric":
        """Estimate Σ⁻¹ from an undifferentiated fingerprint population.

        This uses the TOTAL covariance and is appropriate only when the rows
        carry no device grouping. For identity linking use
        `from_within_device_population` — see its docstring for why the total
        covariance is the wrong scatter for that job.

        Args:
            vectors: (n_samples, dim) fingerprints.
            shrinkage: explicit convex shrinkage toward the identity in the
                scaled basis. ``None`` uses Ledoit-Wolf's analytically optimal
                coefficient (deterministic; sklearn is already a dependency).
                Shrinkage is not optional in spirit — with 180 dimensions and
                O(10²–10³) samples the sample covariance is singular.
        """
        matrix = np.asarray(vectors, dtype=np.float64)
        cls._check_population(matrix)
        residuals = matrix - matrix.mean(axis=0)
        return cls._from_residuals(residuals, shrinkage, provenance)

    @classmethod
    def from_within_device_population(
        cls,
        vectors: np.ndarray,
        groups: Sequence[Any],
        shrinkage: Optional[float] = None,
        provenance: str = "within_device_population",
        mask: Optional[Any] = None,
        input_dim: Optional[int] = None,
    ) -> "MahalanobisMetric":
        """Estimate Σ⁻¹ from the POOLED WITHIN-DEVICE scatter.

        PHASE7_DESIGN specifies "Σ ... estimated from the honest sub-population
        of the registry". For an identity-linking metric that must be read as
        the **within-device** scatter — how much one device's fingerprint moves
        between rounds — not the total scatter of the pooled population.

        The distinction is not cosmetic. The total covariance is dominated by
        the *between-device* spread, which is precisely the signal the matcher
        needs. Whitening by it divides that signal away: on separable synthetic
        data the total-covariance metric collapses the within-device link rate at
        the 1%-FPR τ from ~1.0 to ~0.04. Pooling each device's residuals about
        its own mean whitens the *noise* and leaves the between-device
        separation intact — the standard within-class formulation for
        verification metrics.

        Args:
            vectors: (n_samples, dim) honest fingerprints, FULL width.
            groups: per-row device key (base partition), same length as vectors.
            mask: surviving-dimension indices (the R5 rule). The projection
                happens HERE, so the fit and the recorded mask cannot disagree —
                a caller cannot fit the full space and then label it masked.
            input_dim: full width `mask` indexes; defaults to `vectors`' width.
        """
        matrix = np.asarray(vectors, dtype=np.float64)
        cls._check_population(matrix)
        selected: Optional[np.ndarray] = None
        if mask is not None:
            width = int(input_dim) if input_dim is not None else int(matrix.shape[1])
            if matrix.shape[1] != width:
                raise ValueError(
                    f"population width {matrix.shape[1]} != declared input_dim {width}"
                )
            selected = cls._validated_mask(mask, width)
            matrix = matrix[:, selected]
            input_dim = width
        keys = list(groups)
        if len(keys) != matrix.shape[0]:
            raise ValueError(
                f"groups length {len(keys)} != population rows {matrix.shape[0]}"
            )

        residual_blocks: List[np.ndarray] = []
        for key in sorted(set(keys), key=repr):
            index = [i for i, k in enumerate(keys) if k == key]
            if len(index) < 2:
                continue  # a single observation carries no within-device scatter
            block = matrix[index]
            residual_blocks.append(block - block.mean(axis=0))
        if not residual_blocks:
            raise ValueError(
                "no device has 2+ observations — within-device scatter is undefined"
            )
        residuals = np.vstack(residual_blocks)
        return cls._from_residuals(
            residuals, shrinkage, provenance, mask=selected, input_dim=input_dim
        )

    @staticmethod
    def _check_population(matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"population must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] < 2:
            raise ValueError(
                "need at least 2 fingerprints to estimate a covariance, got "
                f"{matrix.shape[0]}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("calibration population contains non-finite values")

    @classmethod
    def _from_residuals(
        cls,
        residuals: np.ndarray,
        shrinkage: Optional[float],
        provenance: str,
        mask: Optional[np.ndarray] = None,
        input_dim: Optional[int] = None,
    ) -> "MahalanobisMetric":
        """Shared estimator: scale, shrink, invert. Residuals are pre-centred.

        The pre-centring subtraction upstream can itself overflow for extreme
        fingerprints, so the residuals are re-validated here rather than trusted
        — a non-finite residual would silently poison the whole precision matrix
        and therefore every distance computed against it.
        """
        if not np.all(np.isfinite(residuals)):
            raise ValueError(
                "covariance residuals are non-finite — the centring subtraction "
                "overflowed. The population contains fingerprints too extreme to "
                "estimate a covariance from; fix the construct or the input data "
                "rather than proceeding with a poisoned precision matrix."
            )
        n_samples, dim = residuals.shape
        scale = _conditioning_scale(residuals)
        scaled = residuals / scale
        if not np.all(np.isfinite(scaled)):  # pragma: no cover - defensive
            raise ValueError("conditioned residuals are non-finite")

        if shrinkage is None:
            from sklearn.covariance import LedoitWolf

            estimator = LedoitWolf(assume_centered=True).fit(scaled)
            precision = np.asarray(estimator.precision_, dtype=np.float64)
            used_shrinkage = float(estimator.shrinkage_)
        else:
            used_shrinkage = float(shrinkage)
            if not 0.0 <= used_shrinkage <= 1.0:
                raise ValueError(f"shrinkage must be in [0, 1], got {used_shrinkage}")
            covariance = (scaled.T @ scaled) / n_samples
            mu = float(np.trace(covariance) / dim)
            covariance = (1.0 - used_shrinkage) * covariance + used_shrinkage * mu * np.eye(dim)
            precision = np.linalg.pinv(covariance)

        precision = 0.5 * (precision + precision.T)  # enforce exact symmetry
        return cls(
            dim=int(dim),
            scale=np.asarray(scale, dtype=np.float64),
            precision=precision,
            shrinkage=used_shrinkage,
            provenance=provenance,
            mask=mask,
            input_dim=int(input_dim) if mask is not None else None,
        )

    def distance(self, x: np.ndarray, mu: np.ndarray) -> float:
        """Mahalanobis distance between a fingerprint and a registry mean.

        **Overflow policy (stated, and tested).** Both inputs are validated
        finite before they reach the registry, but the raw subtraction ``a - b``
        can still overflow when two finite fingerprints sit at opposite ends of
        the float64 range, and the quadratic form can overflow for an extreme
        residual. Either way the result is ``inf``, which the matcher reads as
        "no link" — the SAFE direction: an overflow can only ever cause a MISS,
        never a false link, so it can depress recall but can never manufacture
        the false-link result the H3 verdict is most sensitive to. The event is
        logged at WARNING so it is never silent.
        """
        a = np.asarray(x, dtype=np.float64)
        b = np.asarray(mu, dtype=np.float64)
        width = self.expected_input_dim
        if a.shape != (width,) or b.shape != (width,):
            raise ValueError(
                f"fingerprint dim mismatch: metric expects ({width},), got "
                f"{a.shape} and {b.shape}"
            )
        # The mask is applied BEFORE the residual, so a dropped dimension can
        # never contribute — including via an overflow in the subtraction.
        a = self.project(a)
        b = self.project(b)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (a - b) / self.scale
            quadratic = float(residual @ self.precision @ residual)
        if not np.isfinite(quadratic):
            logger.warning(
                "[Fingerprint] Mahalanobis distance overflowed to non-finite; "
                "treating as NO MATCH (the safe direction — an overflow can "
                "only cause a miss, never a false link)"
            )
            return float("inf")
        return float(np.sqrt(max(quadratic, 0.0)))

    def to_dict(self) -> Dict[str, Any]:
        """Artifact form. `mask`/`input_dim` are emitted only when masked, so a
        full-width metric serialises byte-identically to the committed v1 lock."""
        payload: Dict[str, Any] = {
            "dim": int(self.dim),
            "scale": [float(v) for v in self.scale],
            "precision": [[float(v) for v in row] for row in self.precision],
            "shrinkage": float(self.shrinkage),
            "provenance": self.provenance,
        }
        if self.mask is not None:
            payload["mask"] = [int(i) for i in self.mask]
            payload["input_dim"] = int(self.expected_input_dim)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MahalanobisMetric":
        raw_mask = payload.get("mask")
        mask = None
        input_dim = None
        if raw_mask is not None:
            input_dim = int(payload["input_dim"])
            mask = cls._validated_mask(raw_mask, input_dim)
        return cls(
            dim=int(payload["dim"]),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            precision=np.asarray(payload["precision"], dtype=np.float64),
            shrinkage=float(payload.get("shrinkage", 0.0)),
            provenance=str(payload.get("provenance", "restored")),
            mask=mask,
            input_dim=input_dim,
        )


# ===========================================================================
# Registry entries and assertions
# ===========================================================================

@dataclass(frozen=True)
class RegistryEntry:
    """One claimed-identity session's fingerprint state (per PHASE7_DESIGN).

    **`session_key` is the identity the client CLAIMS, never its Flower CID.**
    In Flower *simulation* the raw cid is stable per virtual client for the whole
    run (v1.15), so it does NOT change at an identity reset. Keying the registry
    on it would hand the defense exactly the cross-reset identity linkage the RMC
    threat model forbids and H3 exists to measure. `flower_cid` is carried for
    audit output only and is never a lookup key or a match feature.
    """

    entry_id: str
    session_key: str
    flower_cid: Optional[str]
    logical_id: Optional[str]
    fingerprint_vec: np.ndarray
    first_seen_round: int
    last_seen_round: int
    flag_status: bool = False
    flag_reason: Optional[str] = None
    flag_round: Optional[int] = None
    generation: int = 0
    parent_entry_id: Optional[str] = None


@dataclass(frozen=True)
class MatchAssertion:
    """The registry's re-link decision for one NEW-CID appearance.

    Field names are the schema-v5 re-entry contract from v1.10 § 5.1, so the
    signal logger can write them straight through. Written **regardless of any
    downstream enforcement action** — that is the pre-registered integrity
    assertion that makes the D7 metric detector-independent.
    """

    asserted_match: bool
    asserted_parent_entry_id: Optional[str]
    asserted_parent_logical_id: Optional[str]
    min_d: float
    tau: float
    generation: int
    #: ADDITIVE (not part of the frozen v1.10 § 5.1 table): the nearest
    #: candidate considered, recorded UNCONDITIONALLY — including when the
    #: distance exceeded τ and no match was asserted.
    #:
    #: Why: the corrected instrument's P1 is rank-1 identification, which is
    #: threshold-FREE. With only the `asserted_*` pair, an event whose nearest
    #: candidate sat beyond τ discarded that candidate's identity, and `min_d`
    #: alone does not say who it was — so rank-1 was unanswerable for exactly
    #: the events that decide whether the instrument works. `None` here means
    #: the candidate pool was genuinely EMPTY, which is a determinate outcome,
    #: not a missing record.
    #:
    #: The `asserted_*` fields above keep their frozen matched-only semantics
    #: untouched: on a match the two pairs agree; without a match `asserted_*`
    #: stays null exactly as before, so no D7 quantity moves.
    nearest_entry_id: Optional[str] = None
    nearest_logical_id: Optional[str] = None

    def as_log_fields(self) -> Dict[str, Any]:
        """The subset of the schema-v5 row this module is authoritative for.

        The first six are the frozen v1.10 § 5.1 registry half, unchanged in
        name, meaning and value. The `nearest_*` pair is an additive extension —
        see the field comments above and `signal_logger.REENTRY_NEAREST_FIELDS`.
        """
        return {
            "asserted_match": bool(self.asserted_match),
            "asserted_parent_entry_id": self.asserted_parent_entry_id,
            "asserted_parent_logical_id": self.asserted_parent_logical_id,
            "min_d": float(self.min_d),
            "tau": float(self.tau),
            "generation": int(self.generation),
            "nearest_entry_id": self.nearest_entry_id,
            "nearest_logical_id": self.nearest_logical_id,
        }


@dataclass(frozen=True)
class FingerprintObservation:
    """One AS-OBSERVED (round, claimed identity, vector) draw — A.5(a) / A.6.

    Retained purely so the naive-Euclidean baseline comparator pre-registered in
    THRESHOLD_GROUNDING_20260808 § 5 A.5(a) is computable offline from the run's
    custody record instead of requiring a re-run (A.6). It is **evidence, never
    an input**: nothing in this module reads the log back, so no value here can
    reach τ, Σ or a re-link decision.

    `vector` is the vector exactly as `observe()` received it — NOT the EMA
    registry state, which by construction differs from the second observation of
    a device onward. The comparator has to see the same per-round draws the
    matcher saw. It is a read-only defensive copy, so a caller reusing its input
    buffer cannot retro-edit the corpus.

    `session_key` is carried through as the same OPAQUE label the registry uses
    everywhere else. No partition id, no ground truth: the scorer owns that.
    """

    server_round: int
    session_key: str
    vector: np.ndarray

    def as_custody_row(self) -> Dict[str, Any]:
        """JSON row for the custody export.

        `fingerprint_vec_b64` is the raw little-endian float64 buffer, base64
        encoded. Two reasons over a decimal list:

        * **Exactness is unconditional.** The bytes ARE the doubles; nothing
          depends on CPython's `repr` being shortest-round-trip, or on a reader
          parsing decimals with a correctly-rounded strtod.
        * **Size.** At the designed 180-dim × ~1 000-row shape a decimal list
          costs ~5.7 MB of `indent=2` JSON per unit; this costs ~2.0 MB, and it
          is immune to the per-line bloat `indent=2` inflicts on float arrays.

        Read it offline with exactly one line — the same line the artifact
        advertises in its own `read_offline` field::

            np.frombuffer(base64.b64decode(row["fingerprint_vec_b64"]), dtype="<f8")

        The surrounding provenance (round, session key, per-entry lifecycle)
        stays plain-text readable; only the 180 opaque floats are packed.
        """
        buffer = np.asarray(self.vector, dtype=np.float64).astype("<f8", copy=False)
        return {
            "server_round": int(self.server_round),
            "session_key": str(self.session_key),
            "fingerprint_vec_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
        }


@dataclass(frozen=True)
class UpstreamRejectionEvent:
    """One round-scoped upstream rejection of a claimed identity — A.6.

    Recorded UNCONDITIONALLY at the call site, which is the whole point. The
    registry's flag lifecycle records only the FIRST flag: the plugin calls
    :meth:`FingerprintRegistry.flag` under ``if not entry.flag_status``, and
    ``flag()`` itself preserves an existing ``flag_round``. So once an entry has
    been INHERITED-flagged by the Mahalanobis matcher, every later upstream
    rejection of it vanishes from custody.

    That is exactly the evidence the A.5(a) Euclidean counterfactual needs. An
    entry this run inherited-flagged might not be flagged at all under τ′, and
    then "when did the detector reject it?" has no answer — for precisely the
    entries the comparator is most interesting on. This log answers it.

    Like the observation log it is write-only within this module and carries the
    OPAQUE session key and nothing else. No ground truth.
    """

    server_round: int
    session_key: str

    def as_custody_row(self) -> Dict[str, Any]:
        return {
            "server_round": int(self.server_round),
            "session_key": str(self.session_key),
        }


@dataclass(frozen=True)
class ObservationResult:
    """What one `observe()` call did."""

    entry_id: str
    is_first_appearance: bool
    flagged: bool
    assertion: Optional[MatchAssertion]


# ===========================================================================
# The registry
# ===========================================================================

class FingerprintRegistry:
    """Bounded, in-memory registry of per-CID fingerprints.

    Storage is bounded by the number of distinct Flower CIDs seen (≤ ~200
    across a 50-round simulation, per PHASE7_DESIGN).
    """

    def __init__(
        self,
        tau: float,
        metric: Optional[MahalanobisMetric] = None,
        ema_alpha: float = EMA_ALPHA,
        dim: int = FINGERPRINT_DIM,
        max_observation_rows: int = OBSERVATION_LOG_MAX_ROWS,
        max_upstream_rejection_rows: int = UPSTREAM_REJECTION_LOG_MAX_ROWS,
        policy: "RegistryPolicy | str" = DEFAULT_REGISTRY_POLICY,
    ):
        try:
            self._policy = RegistryPolicy(policy)
        except ValueError as exc:
            raise ValueError(
                f"unknown registry policy {policy!r}; expected one of "
                f"{[p.value for p in RegistryPolicy]}"
            ) from exc
        tau_value = float(tau)
        if not np.isfinite(tau_value) or tau_value <= 0.0:
            raise ValueError(f"tau must be a positive finite float, got {tau!r}")
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha!r}")
        self._tau = tau_value
        self._dim = int(dim)
        self._metric = metric if metric is not None else MahalanobisMetric.identity(self._dim)
        # The registry always handles FULL-width fingerprints; a masked metric
        # projects internally (EDA_MEMO § 6.2), so the width that must agree is
        # the metric's expected INPUT width, not its fitted subspace `dim`.
        if self._metric.expected_input_dim != self._dim:
            raise ValueError(
                f"metric expects {self._metric.expected_input_dim}-dim vectors "
                f"!= registry dim {self._dim}"
            )
        self._alpha = float(ema_alpha)
        self._entries: Dict[str, RegistryEntry] = {}
        self._next_entry_index = 0

        # A.5(a)/A.6 observation log — passive custody, never read back here.
        max_rows = int(max_observation_rows)
        if max_rows <= 0:
            raise ValueError(
                f"max_observation_rows must be a positive int, got "
                f"{max_observation_rows!r}"
            )
        self._max_observation_rows = max_rows
        self._observations: List[FingerprintObservation] = []
        self._observations_dropped = 0
        self._observation_truncation_announced = False

        max_rejection_rows = int(max_upstream_rejection_rows)
        if max_rejection_rows <= 0:
            raise ValueError(
                f"max_upstream_rejection_rows must be a positive int, got "
                f"{max_upstream_rejection_rows!r}"
            )
        self._max_upstream_rejection_rows = max_rejection_rows
        self._upstream_rejections: List[UpstreamRejectionEvent] = []
        self._upstream_rejections_dropped = 0
        self._rejection_truncation_announced = False

    # -- introspection ------------------------------------------------------

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def policy(self) -> RegistryPolicy:
        """Which prior sessions a new claimed identity is matched against."""
        return self._policy

    @property
    def metric(self) -> MahalanobisMetric:
        return self._metric

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> Tuple[RegistryEntry, ...]:
        """Snapshot of every entry (entries are frozen; the tuple is a copy)."""
        return tuple(self._entries.values())

    def entry_for_session(self, session_key: str) -> RegistryEntry:
        return self._entries[str(session_key)]

    def flagged_entries(self) -> Tuple[RegistryEntry, ...]:
        return tuple(e for e in self._entries.values() if e.flag_status)

    # -- A.5(a)/A.6 observation log (passive; export-only) -------------------

    def observations(self) -> Tuple[FingerprintObservation, ...]:
        """Every observed draw, in observation order (see `A.6` above)."""
        return tuple(self._observations)

    @property
    def observation_log_truncated(self) -> bool:
        """True when the bound was hit and rows were DROPPED."""
        return self._observations_dropped > 0

    @property
    def observation_log_dropped_count(self) -> int:
        return int(self._observations_dropped)

    @property
    def observation_log_max_rows(self) -> int:
        return int(self._max_observation_rows)

    def upstream_rejections(self) -> Tuple[UpstreamRejectionEvent, ...]:
        """Every recorded upstream rejection, in call order (see A.6 above)."""
        return tuple(self._upstream_rejections)

    @property
    def upstream_rejection_log_truncated(self) -> bool:
        return self._upstream_rejections_dropped > 0

    @property
    def upstream_rejection_log_dropped_count(self) -> int:
        return int(self._upstream_rejections_dropped)

    @property
    def upstream_rejection_log_max_rows(self) -> int:
        return int(self._max_upstream_rejection_rows)

    def record_upstream_rejection(self, session_key: str, server_round: int) -> None:
        """Record that the upstream detector rejected this identity this round.

        PURELY PASSIVE. It does not flag, does not look the session up, and does
        not raise on an unknown key — unlike :meth:`flag`, which raises by
        design. This runs on the aggregation path for every rejected client
        every round, so it must not be able to introduce a new failure there.

        Call it UNCONDITIONALLY, *before* the `if not entry.flag_status` gate
        around :meth:`flag`. That gate is precisely what loses the signal for an
        already-flagged entry, and the A.5(a) counterfactual needs it.
        """
        if len(self._upstream_rejections) >= self._max_upstream_rejection_rows:
            self._upstream_rejections_dropped += 1
            if not self._rejection_truncation_announced:
                self._rejection_truncation_announced = True
                message = (
                    "[Fingerprint] UPSTREAM REJECTION LOG TRUNCATED at %d rows "
                    "(round %s, session %s). Further rejections are DROPPED. "
                    "The A.5(a) inherited-flag counterfactual for this run is "
                    "INCOMPLETE and must not be replayed as if it were whole."
                )
                logger.error(message, self._max_upstream_rejection_rows,
                             server_round, session_key)
                print(
                    message % (self._max_upstream_rejection_rows, server_round,
                               session_key),
                    flush=True,
                )
            return
        self._upstream_rejections.append(
            UpstreamRejectionEvent(
                server_round=int(server_round), session_key=str(session_key)
            )
        )

    def _record_observation(
        self, session_key: str, vector: np.ndarray, server_round: int
    ) -> None:
        """Append one AS-OBSERVED draw. Called before any EMA refresh.

        Bounded by `max_observation_rows`. Overflow drops the row and is
        announced at ERROR the first time plus counted in the custody export —
        never silent, because a truncated corpus that looked complete would bias
        the A.5(a) comparator.

        This method must stay side-effect-free with respect to matching: it
        touches only the log, so the log can be capped to a single row without
        moving a single re-link decision (asserted in
        `tests/test_fingerprint_registry.py`).
        """
        if len(self._observations) >= self._max_observation_rows:
            self._observations_dropped += 1
            if not self._observation_truncation_announced:
                self._observation_truncation_announced = True
                message = (
                    "[Fingerprint] OBSERVATION LOG TRUNCATED at %d rows "
                    "(round %s, session %s). Further draws are DROPPED. The "
                    "A.5(a) baseline comparator corpus for this run is "
                    "INCOMPLETE and must not be scored as if it were whole."
                )
                logger.error(message, self._max_observation_rows, server_round,
                             session_key)
                print(
                    message % (self._max_observation_rows, server_round, session_key),
                    flush=True,
                )
            return
        frozen = np.array(vector, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        self._observations.append(
            FingerprintObservation(
                server_round=int(server_round),
                session_key=str(session_key),
                vector=frozen,
            )
        )

    # -- mutation (always replace, never mutate an entry in place) ----------

    def _validate(self, fingerprint: np.ndarray) -> np.ndarray:
        vector = np.asarray(fingerprint, dtype=np.float64)
        if vector.shape != (self._dim,):
            raise ValueError(
                f"fingerprint dim mismatch: expected ({self._dim},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("fingerprint must be finite")
        return vector

    def observe(
        self,
        session_key: str,
        fingerprint: np.ndarray,
        server_round: int,
        logical_id: Optional[str] = None,
        flower_cid: Optional[str] = None,
    ) -> ObservationResult:
        """Register one claimed identity's fingerprint for this round.

        NEW session  → compare against every FLAGGED entry; ``min_d ≤ τ``
                       inherits the parent's flag and ``generation + 1``,
                       otherwise a fresh generation-0 entry. Returns the
                       `MatchAssertion` for the event.
        RETURNING    → EMA-refresh the stored vector and bump
                       ``last_seen_round``. Returns ``assertion=None``: a
                       re-entry event is one row per NEW claimed identity, so
                       repeat participation must not manufacture extra events.

        `session_key` MUST be the identity the client claims (the logical
        identity the scenario installs via `set_identity_map`), NOT the Flower
        CID — see `RegistryEntry`. `flower_cid` is stored for audit only and
        never participates in matching. The decision is a function of the
        fingerprint and the registry's own internal entry ids alone; a test
        asserts the assertions are invariant to relabelling either key.

        This method performs **no enforcement**. It is safe (and required) to
        call it for the full unfiltered cohort.
        """
        key = str(session_key)
        vector = self._validate(fingerprint)
        round_index = int(server_round)
        # A.5(a)/A.6: record the draw AS OBSERVED, before any EMA refresh. This
        # is the only statement in `observe()` that the log participates in; it
        # reads nothing and returns nothing, so the decision below is unchanged.
        self._record_observation(key, vector, round_index)

        existing = self._entries.get(key)
        if existing is not None:
            blended = (1.0 - self._alpha) * existing.fingerprint_vec + self._alpha * vector
            self._entries[key] = replace(
                existing,
                fingerprint_vec=blended,
                last_seen_round=round_index,
                logical_id=logical_id if logical_id is not None else existing.logical_id,
                flower_cid=flower_cid if flower_cid is not None else existing.flower_cid,
            )
            return ObservationResult(
                entry_id=existing.entry_id,
                is_first_appearance=False,
                flagged=existing.flag_status,
                assertion=None,
            )

        if self._policy is RegistryPolicy.IDENTITY_ONLY:
            parent, min_d = self._nearest_enrolled(vector, round_index)
        else:
            parent, min_d = self._nearest_flagged(vector)
        matched = parent is not None and min_d <= self._tau
        generation = (parent.generation + 1) if matched else 0
        # Identity linking is NOT an enforcement decision. Under FLAG_GATED
        # every candidate is flagged by construction, so this is exactly the
        # incumbent's `flag_status=matched`; under IDENTITY_ONLY it is what
        # keeps the D4 hard-drop scoped to FLAGGED devices, so an honest device
        # re-identified as itself is recorded as linked and left alone.
        inherits_flag = bool(matched and parent.flag_status)

        entry_id = f"fp-{self._next_entry_index:04d}"
        self._next_entry_index += 1
        entry = RegistryEntry(
            entry_id=entry_id,
            session_key=key,
            flower_cid=flower_cid,
            logical_id=logical_id,
            fingerprint_vec=vector,
            first_seen_round=round_index,
            last_seen_round=round_index,
            flag_status=inherits_flag,
            flag_reason="inherited" if inherits_flag else None,
            flag_round=round_index if inherits_flag else None,
            generation=generation,
            parent_entry_id=parent.entry_id if matched else None,
        )
        self._entries[key] = entry

        assertion = MatchAssertion(
            asserted_match=bool(matched),
            asserted_parent_entry_id=parent.entry_id if matched else None,
            asserted_parent_logical_id=parent.logical_id if matched else None,
            min_d=float(min_d),
            tau=self._tau,
            generation=generation,
            # Unconditional: `parent` is the nearest candidate whether or not it
            # cleared tau. None only when the pool was empty.
            nearest_entry_id=parent.entry_id if parent is not None else None,
            nearest_logical_id=parent.logical_id if parent is not None else None,
        )
        if matched:
            logger.info(
                "[Fingerprint] session=%s linked to entry %s (logical %s) "
                "d=%.4f <= tau=%.4f generation=%d policy=%s flag_inherited=%s",
                key, parent.entry_id, parent.logical_id, min_d, self._tau,
                generation, self._policy.value, inherits_flag,
            )
        return ObservationResult(
            entry_id=entry_id,
            is_first_appearance=True,
            flagged=inherits_flag,
            assertion=assertion,
        )

    def _nearest_flagged(
        self, vector: np.ndarray
    ) -> Tuple[Optional[RegistryEntry], float]:
        """Nearest FLAGGED entry and its distance (inf when there are none).

        Exact distance ties are real here: an identity reset copies the device's
        partition byte-for-byte, so a device's root entry and its earlier
        respawns can sit at d = 0 from the new fingerprint simultaneously. The
        tie-break is pre-declared and deterministic — **highest generation
        first, then `entry_id`** (assigned in first-appearance order) — so the
        immediate predecessor is the asserted parent and `generation` chains.
        A non-deterministic tie-break here would reproduce the aggregate_fit
        arrival-order defect of GWU-51 / EXP-019 inside the H3 metric itself.
        """
        candidates = sorted(
            self.flagged_entries(), key=lambda e: (-e.generation, e.entry_id)
        )
        return self._nearest_of(vector, candidates)

    def _nearest_enrolled(
        self, vector: np.ndarray, server_round: int
    ) -> Tuple[Optional[RegistryEntry], float]:
        """Nearest of EVERY session enrolled strictly before `server_round`.

        The IDENTITY_ONLY pool (corrected-instrument § 1). Two differences from
        :meth:`_nearest_flagged`, both pre-declared:

        * **no flag gating** — every enrolled session is a candidate, which is
          what makes the re-link decision detector-independent;
        * **strictly-earlier enrollment** — a session first seen in the SAME
          round is not a candidate. Within a round the observe order is the
          arrival order, and letting it decide candidacy would put the
          arrival-order non-determinism of GWU-51 / EXP-019 straight inside the
          H3 metric.

        The tie-break is the replay's disclosed Policy-B rule: candidates
        ordered by ``(-first_seen_round, entry_id)`` and replaced only on a
        strictly smaller distance, so the MOST RECENT enrollment wins an exact
        tie. It is the analogue of the flagged pool's "highest generation
        first" — the immediate predecessor is the asserted parent — and it is
        mirrored byte-for-byte from
        `results/20260814/h3_replay_study/replay_engine.py`.
        """
        candidates = sorted(
            (e for e in self._entries.values() if e.first_seen_round < int(server_round)),
            key=lambda e: (-e.first_seen_round, e.entry_id),
        )
        return self._nearest_of(vector, candidates)

    def _nearest_of(
        self, vector: np.ndarray, candidates: Sequence[RegistryEntry]
    ) -> Tuple[Optional[RegistryEntry], float]:
        """Nearest candidate in a PRE-ORDERED pool (inf when the pool is empty).

        Shared by both policies so the two differ only in who is a candidate
        and in what order, never in how the winner is picked: the pool is
        visited in the caller's order and a later entry replaces the best only
        on a STRICTLY smaller distance, so the first candidate in that order
        wins an exact tie.
        """
        best: Optional[RegistryEntry] = None
        best_d = float("inf")
        for entry in candidates:
            distance = self._metric.distance(vector, entry.fingerprint_vec)
            if distance < best_d:
                best, best_d = entry, distance
        return best, best_d

    def flag(self, session_key: str, reason: str, server_round: int) -> RegistryEntry:
        """Mark a claimed identity as flagged (upstream detector said malicious)."""
        key = str(session_key)
        entry = self._entries[key]  # KeyError on unknown session — loud by design
        flagged = replace(
            entry,
            flag_status=True,
            flag_reason=str(reason),
            flag_round=int(server_round) if entry.flag_round is None else entry.flag_round,
        )
        self._entries[key] = flagged
        return flagged

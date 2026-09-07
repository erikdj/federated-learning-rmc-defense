#!/usr/bin/env python3
"""Calibrate the H3 fingerprint-match threshold τ — attack-free controls ONLY.

Design authority
----------------
* v1.10 § 5.0 **D2** — "H3-A — existing `rmc/scenarios/control_honest.json` +
  `control_benign_churn.json`, **after a named freshness/integrity verify**
  (current-runner-compatible; zero malicious clients — G13). **S0/S1 are
  attack-bearing and are forbidden for calibration.**"
* v1.10 § 5.1 — "**BOTH τ values locked in code** (all-device validation-τ AND
  even-device adjudicating-τ, both derived from the calibration logs **before
  any eval scenario runs**, gate (c))"; "**τ is NEVER re-derived after any eval
  scenario has run.**"; D9 axis (ii) device hold-out — τ **and** the covariance
  are calibrated on the **even** base partitions for the adjudicating cohort.
* `docs/PHASE7_DESIGN.md` "Threshold τ calibration" — within-vs-across
  Mahalanobis distributions, τ at the dev-FPR = 1% point, then LOCKED.

The allowlist is a PRE-REGISTRATION PROPERTY
--------------------------------------------
`ALLOWED_CALIBRATION_SCENARIOS` is a hard-coded module constant with **no CLI
override of any kind**. A single row from any other scenario refuses the whole
input. Calibrating on an attack-bearing scenario would move the decision
boundary onto data the metric is later scored against, which is exactly what D2
forbids — so this is a refusal, not a warning.

Status
------
Runnable as soon as fingerprint observations exist. The input contract below is
deliberately narrow and schema-agnostic so it can be fed either by a purpose-made
dump or by the schema-v5 signal log once the instrumentation lane lands.

Input contract (JSONL, one object per line, or a JSON array):

    {"run_id": "...", "source_unit": "...", "scenario": "control_honest",
     "seed": 42, "server_round": 7, "logical_id": "client_3",
     "fingerprint": [ ... 180 floats ... ]}

`gt_logical_id` is accepted as an alias for `logical_id`, and `fingerprint` may
be a JSON-encoded string (the `FitRes.metrics` transport form).

Rows carrying `source_unit` but no `run_id` (the fleet extractor's output form)
get their durable run id from `--run-id-manifest`, so the locked artifact cites
custody records rather than the ephemeral path the rows were staged at.

Usage:
    python scripts/calibrate_fp_threshold.py \
        --observations signals/control_honest_*.jsonl signals/control_benign_churn_*.jsonl \
        --run-id-manifest results/20260810/exp050_tau_calibration/mlflow_run_ids.json \
        --git-commit "$(git rev-parse --short HEAD)" \
        --out data/fingerprint_tau_locked_v1.json

Then paste the printed snippet into `flowerfl/fingerprint_registry.py`. That
paste is the gate-(c) lock and must be reviewed and committed BEFORE any H3
evaluation scenario runs.

What the artifact carries beyond τ
----------------------------------
* **Addendum A's predeclared metric comparison.** Both candidate metrics are
  calibrated inside EACH cohort's own population, each at its own realised-FPR
  ≤ 0.01 τ, and the winner is chosen by a rule fixed before the data was seen
  (`METRIC_SELECTION_RULE`). Both summaries are recorded whichever wins; the
  loser is recorded, never deleted. See `select_metric_by_rule`.
* **A size-independent degeneracy refusal** on the WITHIN-device population
  (`_refuse_constant_devices`, `_refuse_zero_within_median`), because the
  across-pair tie count is cohort-size dependent and lets a cached-fingerprint
  contract through on the 20-device cohort — EMISSION_CONTRACT § 3.7.
* **A homogeneity refusal on the devices the cohort will SCORE**
  (`_refuse_heterogeneous_scoreable_devices`), which is the only guard that can
  see a hold-out device — see its docstring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.fingerprint_registry import (  # noqa: E402
    ADJUDICATING_CALIBRATION_PARTITIONS,
    ODD_HOLDOUT_PARTITIONS,
    CalibrationCohort,
    MahalanobisMetric,
    is_calibration_partition,
    partition_of,
)

# ===========================================================================
# PRE-REGISTERED CONSTANTS — no CLI may change any of these
# ===========================================================================

#: D2: the ONLY scenarios τ may be calibrated on. S0/S1 and every RMC scenario
#: are attack-bearing and forbidden.
ALLOWED_CALIBRATION_SCENARIOS: Tuple[str, ...] = (
    "control_benign_churn",
    "control_honest",
)

#: PHASE7_DESIGN: "Choose τ at the dev-FPR=1% point".
FPR_TARGET: float = 0.01

#: Pair subsampling is seeded so the calibration is reproducible byte-for-byte.
PAIR_SAMPLING_SEED: int = 20260805
MAX_PAIRS_PER_POPULATION: int = 2_000_000

#: Largest QUALIFYING pair population that is still enumerated exactly before
#: subsampling. Above it, pair indices are sampled in rank space and the pair set
#: is never built at all. The pre-registered EXP-050 corpus is 9 795
#: observations = 47 966 115 pairs; enumerating those costs ~768 MB of index
#: array (and a further ~69 GB if their 180-dim deltas were taken in one block),
#: so the across-device population necessarily takes the sampled path.
EXACT_PAIR_ENUMERATION_LIMIT: int = 8_000_000

#: Blocking constants. These bound peak memory; they change no result.
_PAIR_ENUMERATION_ROW_BLOCK: int = 512
_DISTANCE_BLOCK_PAIRS: int = 16_384
_PAIR_SAMPLING_BATCH: int = 1_000_000

#: Per-device within-pair cap for the scoreable-homogeneity gate. The gate reads
#: a MEDIAN, which a large uniform subsample estimates to far better precision
#: than the order-of-magnitude margins it adjudicates; the cap only bounds the
#: cost on a corpus far larger than the pre-registered one (at EXP-050's ~490
#: observations per device the exact population is ~120 000 pairs, well under
#: it, so the real lock takes the exact path). Subsampling is seeded, so the
#: gate's verdict is reproducible byte-for-byte either way.
MAX_WITHIN_PAIRS_PER_SCOREABLE_DEVICE: int = 200_000

#: The devices each cohort's locked (τ, Σ) will be used to SCORE — NOT the
#: devices it calibrates on. For ADJUDICATING the two sets are DISJOINT (D9 axis
#: (ii)), which is the whole reason `_refuse_heterogeneous_scoreable_devices`
#: has to exist: every other guard in this file runs on the calibration
#: population and therefore cannot see a hold-out device at all.
SCOREABLE_PARTITIONS: Dict[str, Tuple[int, ...]] = {
    CalibrationCohort.VALIDATION.value: tuple(range(20)),
    CalibrationCohort.ADJUDICATING.value: ODD_HOLDOUT_PARTITIONS,
}

SCENARIO_DIR = PROJECT_ROOT / "rmc" / "scenarios"
SEEDS_MANIFEST = PROJECT_ROOT / "data" / "seeds.json"

# ---------------------------------------------------------------------------
# The FROZEN R5 feature-selection rule (draft amendment § 3.1(b))
# ---------------------------------------------------------------------------
# `results/20260814/h3_feature_eda/EDA_MEMO.md` § 6.1 freezes the rule:
#
#   surviving dims = NOT dead
#                  ∧ column NOT pool-flagged (top-1 second-moment share > 0.5
#                    in ANY device's raw pool)
#                  ∧ NOT lattice-flagged
#                  ∧ within-device het ratio ≤ 100
#
# Why it is legitimately pre-registerable: **nothing in the rule reads a
# distance, a re-link outcome, an eval scenario, or any scored quantity.** The
# pool screen is computed from the raw partition parquets alone — the memo's
# "distance-free, fingerprint-free" property, and it caught BOTH known-
# pathological columns blind, one of them via a device that never failed. The
# three control-side screens are degeneracy tests on attack-free control
# fingerprints, the same population τ is calibrated from.
#
# The screens run across EVERY device in the corpus, not only a cohort's
# calibration partitions — mirroring `step1_scores.py`, and for the same reason
# the § 5(1) homogeneity gate reads the scored devices: a degeneracy that is
# invisible to the fitted half is exactly the one that breaks the instrument.
# This is a disclosed property of the rule, not an accident.
#
# The mask applies at the METRIC level (§ 6.2). The 180-dim emission contract is
# untouched, so `features_sha256` does not move; only the CALIBRATION artifact
# hash changes, which the § 5(1) gate re-verifies anyway.

FEATURE_SELECTION_RULE_ID: str = "R5"

#: § 3 of the memo: one pool row carrying more than half a column's second
#: moment means the bootstrapped moment IS that row's capture count. STRUCTURAL,
#: not tuned — beyond 0.5 no other row can outweigh the one row.
POOL_FLAG_TOP1_SHARE_THRESHOLD: float = 0.5

#: § 5 honesty box: the one screen parameter with tuning flavour, chosen as an
#: order-of-magnitude bound over the typical 1.5–50 range BEFORE rule
#: sensitivities were seen. Disclosed; R4 (the pool screen alone) carries no
#: such parameter and is recorded as the ablation on every artifact.
HET_RATIO_MAX: float = 100.0

#: Capture-lottery lattice flag: few distinct levels with a relative spread that
#: dwarfs ordinary jitter, in ANY device (`step1_scores.py`).
LATTICE_MAX_DISTINCT_LEVELS: int = 8
LATTICE_MIN_RELATIVE_SPREAD: float = 0.5

#: Fewer surviving dimensions than this cannot support a covariance at all.
MIN_SURVIVING_DIMS: int = 2

#: The raw partition pools the a-priori screen reads. Partitions 0–19 are the FL
#: devices; `client_20.parquet` exists but is NOT a device partition (EDA § 5).
POOL_PARQUET_DIR = PROJECT_ROOT / "data" / "edge_full_20"
POOL_PARTITIONS: Tuple[int, ...] = tuple(range(20))

FEATURE_SELECTION_RULE = (
    "R5 (FROZEN, draft amendment § 3.1(b) / EDA_MEMO § 6.1): a fingerprint "
    "dimension survives iff it is NOT dead (constant in every device), its "
    "COLUMN is not pool-flagged (max over devices of the top-1 second-moment "
    f"share > {POOL_FLAG_TOP1_SHARE_THRESHOLD} in the raw partition pools), it "
    "is not lattice-flagged (<= "
    f"{LATTICE_MAX_DISTINCT_LEVELS} distinct levels with relative spread > "
    f"{LATTICE_MIN_RELATIVE_SPREAD} in any device), and its within-device "
    f"heteroscedasticity ratio is <= {HET_RATIO_MAX}. Distance-free and "
    "outcome-free by construction: no screen reads a distance, a re-link "
    "decision, or any eval scenario. The mask is applied at the METRIC level, "
    "so the 180-dim emission contract and its features_sha256 are unchanged. "
    "R4 (dead + pool screen only, the parameter-free ablation) is recorded "
    "beside it on every artifact and never selected automatically."
)


def _dev_seeds() -> Tuple[int, ...]:
    return tuple(int(s) for s in json.loads(SEEDS_MANIFEST.read_text())["dev_seeds"])


#: v1.10 § 5.1 — calibration runs on the 5 public dev seeds.
DEV_SEEDS: Tuple[int, ...] = _dev_seeds()


#: Pending-addendum A, surfaced in the locked artifact — see fingerprint_registry.
COVARIANCE_ESTIMATOR = "pooled_within_device"

# ---------------------------------------------------------------------------
# Addendum A — the PREDECLARED, CALIBRATION-ONLY metric comparison
# ---------------------------------------------------------------------------
# The H3 workflow in `docs/reproduction/experiments.md` settles the
# covariance-estimator choice empirically rather than on its author's reasoning,
# under a rule fixed before the calibration data was seen. Both candidates are
# estimated per cohort, each gets ITS OWN τ at realised FPR ≤ 0.01 through the
# unchanged `select_tau`, and both summaries are written into the τ-lock
# artifact whichever one wins. The loser is recorded, never deleted.
#
# Neither candidate needs a new estimator: at `shrinkage=1.0`
# `MahalanobisMetric.from_within_device_population` collapses the precision to an
# exact multiple of the identity in a basis already scaled by the per-dimension
# within-device residual spread, and that multiple is absorbed by τ — i.e. it
# IS the globally shared shrinkage-to-identity / diagonal within-scale metric.
# (Verified, not assumed: see `test_the_alternative_metric_is_shrinkage_to_identity`.)

#: (i) the implemented pooled-within-device scatter + Ledoit-Wolf precision.
INCUMBENT_METRIC = "pooled_within_ledoit_wolf"
#: (ii) the globally shared shrinkage-to-identity / diagonal within-scale metric.
ALTERNATIVE_METRIC = "shrinkage_to_identity"
CANDIDATE_METRICS: Tuple[str, ...] = (INCUMBENT_METRIC, ALTERNATIVE_METRIC)

#: Explicit shrinkage per candidate. ``None`` = Ledoit-Wolf's analytic optimum.
METRIC_SHRINKAGE: Dict[str, Any] = {
    INCUMBENT_METRIC: None,
    ALTERNATIVE_METRIC: 1.0,
}

#: The tie-break target named in the addendum: the SIMPLER of the two.
SIMPLER_METRIC = ALTERNATIVE_METRIC
METRIC_TIE_MARGIN: float = 0.01

#: float64 slack ONLY, so that a difference of *exactly* 0.01 reads as "within
#: 0.01" as the rule says (0.90 - 0.89 == 0.010000000000000009 in binary
#: floating point). It is not a widening of the margin.
_TIE_MARGIN_EPSILON: float = 1e-9

METRIC_SELECTION_RULE = (
    "Addendum A predeclared comparison — the metric with the HIGHER calibration "
    "within_link_rate at ITS OWN realised-FPR <= 0.01 operating point wins; ANY "
    "TIE WITHIN 0.01 GOES TO THE SIMPLER METRIC "
    f"({SIMPLER_METRIC}). The rule is total: one pre-named number decides it and "
    "there is no second criterion. It is fixed in the addendum, applies inside "
    "each cohort's own calibration population (so no odd-partition fingerprint "
    "enters the adjudicating selection — D9 axis (ii)), and has NO command-line "
    "override — a flag able to move the selected metric would reintroduce exactly "
    "the post-hoc discretion the predeclaration exists to remove."
)
#: Addendum A's authorisation status, stamped into every artifact this script
#: writes. The estimator-addenda spec (§ "On ratification") directs that this
#: string be flipped from `AMENDMENT-REQUIRED` to a citation of that file once
#: Erik ratifies; that ratification landed 2026-08-08 at methodology v1.46, so
#: the artifact now records the ratified status instead of self-reporting the
#: selected estimator as unauthorised. The constant is kept (rather than
#: deleted) because the artifact must still name the substantive choice it
#: encodes and the authority that permits it.
ADDENDUM_STATUS = (
    "RATIFIED 2026-08-08 (Erik Jones; methodology v1.46) — Addendum A of "
    "docs/superpowers/specs/2026-08-08-h3-estimator-addenda.md: tau and Sigma "
    "are estimated from the POOLED WITHIN-DEVICE scatter, not the total "
    "covariance. Scientifically correct for identity linking, and a substantive "
    "estimator selection not uniquely entailed by PHASE7_DESIGN's 'estimated "
    "from the honest sub-population' — which is why it was carried as a "
    "self-declared AMENDMENT-REQUIRED marker until it was ratified at "
    "methodology v1.46. COVARIANCE_ESTIMATOR is unchanged by ratification."
)


class CalibrationRefusal(RuntimeError):
    """Raised when calibration would violate a pre-registered constraint."""


# ===========================================================================
# D2 freshness / integrity verify
# ===========================================================================

def _count_attack_entries(scenario: Mapping[str, Any]) -> int:
    total = 0
    for block in scenario.get("schedule", []) or []:
        attacks = block.get("attacks") or {}
        total += len(attacks)
    return total


def verify_scenarios_attack_free(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """D2's *named* freshness/integrity verify. Refuses on anything unexpected.

    Checks, per scenario: it is on the allowlist; the file exists; it declares
    ZERO attack entries anywhere in its schedule (G13); and records the file's
    SHA-256 plus its shape, so the calibration artifact carries proof of exactly
    which scenario bytes it was calibrated against.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for name in names:
        if name not in ALLOWED_CALIBRATION_SCENARIOS:
            raise CalibrationRefusal(
                f"scenario {name!r} is not in the allowlist "
                f"{ALLOWED_CALIBRATION_SCENARIOS} — D2 forbids calibrating τ on "
                "any attack-bearing scenario"
            )
        path = SCENARIO_DIR / f"{name}.json"
        if not path.exists():
            raise CalibrationRefusal(f"calibration scenario file missing: {path}")
        raw = path.read_bytes()
        scenario = json.loads(raw)
        num_attacks = _count_attack_entries(scenario)
        if num_attacks != 0:
            raise CalibrationRefusal(
                f"scenario {name!r} declares {num_attacks} attack entries — "
                "D2 requires zero malicious clients in the calibration population"
            )
        report[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "attack_free": True,
            "num_attack_entries": 0,
            "num_rounds": int(scenario.get("num_rounds", 0)),
            "num_clients": len(scenario.get("clients", {}) or {}),
            "dataset": scenario.get("dataset"),
        }
    return report


# ===========================================================================
# Observation loading
# ===========================================================================

@dataclass(frozen=True)
class Observations:
    """A validated corpus of honest fingerprints, one row per client-round."""

    vectors: np.ndarray
    partitions: Tuple[int, ...]
    logical_ids: Tuple[str, ...]
    scenarios: Tuple[str, ...]
    seeds: Tuple[int, ...]
    run_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    #: The per-row calibration unit (`source_unit`), i.e. the fleet result file
    #: the fingerprint came from. Paired with `run_ids` it is what lets an
    #: auditor map a row back to a durable MLflow run.
    source_units: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def _iter_records(path: Path) -> Iterable[Mapping[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)

    def _lines():
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationRefusal(
                    f"{path}:{line_number} is not valid JSON: {exc}"
                ) from exc

    return _lines()


def _coerce_fingerprint(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise CalibrationRefusal(f"fingerprint must be 1-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationRefusal("fingerprint contains non-finite values")
    return array


def load_run_id_manifest(path: Path) -> Dict[str, str]:
    """Load a calibration-unit → MLflow run-id manifest.

    The extractor stamps each observation with its `source_unit` (the fleet
    result file) but not with the MLflow run id, so without this map the locked
    artifact could only cite an ephemeral scratch path — an auditor could not
    tie the rows to durable custody records. Accepts either a bare
    `{unit: run_id}` object or one nested under a `run_ids` key.
    """
    path = Path(path)
    if not path.exists():
        raise CalibrationRefusal(f"run-id manifest not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationRefusal(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(raw, Mapping) and "run_ids" in raw:
        raw = raw["run_ids"]
    if not isinstance(raw, Mapping):
        raise CalibrationRefusal(
            f"{path}: expected an object mapping source_unit -> run_id"
        )
    mapping: Dict[str, str] = {}
    for unit, run_id in raw.items():
        if not isinstance(run_id, str) or not run_id.strip():
            raise CalibrationRefusal(f"{path}: unit {unit!r} has no usable run id")
        mapping[str(unit)] = run_id.strip()
    if not mapping:
        raise CalibrationRefusal(f"{path}: run-id manifest is empty")
    return mapping


def load_observations(
    paths: Sequence[Path],
    run_id_manifest: Optional[Mapping[str, str]] = None,
) -> Observations:
    """Load and VALIDATE fingerprint observations. Refuses, never filters.

    Silently dropping a forbidden row would be indistinguishable from having
    calibrated on it, so a single out-of-allowlist scenario or non-dev seed
    refuses the entire input.

    `run_id_manifest` resolves each row's `source_unit` to a durable MLflow run
    id when the extractor did not stamp `run_id` on the row itself. It is
    strict on purpose: a unit missing from a supplied manifest refuses, because
    a partially-attributed corpus is worse than an unattributed one.
    """
    vectors: List[np.ndarray] = []
    partitions: List[int] = []
    logical_ids: List[str] = []
    scenarios: List[str] = []
    seeds: List[int] = []
    run_ids: List[str] = []
    sources: List[str] = []
    source_units: List[str] = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            raise CalibrationRefusal(f"observations file not found: {path}")
        for record in _iter_records(path):
            scenario = str(record.get("scenario", ""))
            if scenario not in ALLOWED_CALIBRATION_SCENARIOS:
                raise CalibrationRefusal(
                    f"{path}: refusing scenario {scenario!r} — D2 permits τ "
                    f"calibration ONLY on {ALLOWED_CALIBRATION_SCENARIOS}. "
                    "This allowlist is pre-registration evidence and has no override."
                )
            try:
                seed = int(record["seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationRefusal(f"{path}: record has no usable seed") from exc
            if seed not in DEV_SEEDS:
                raise CalibrationRefusal(
                    f"{path}: refusing seed {seed} — τ is calibrated on the public "
                    f"dev seeds {DEV_SEEDS} only; confirmatory and sealed seeds are "
                    "never opened for calibration"
                )
            logical_id = record.get("logical_id") or record.get("gt_logical_id")
            if not logical_id:
                raise CalibrationRefusal(f"{path}: record has no logical identity")
            if "fingerprint" not in record:
                raise CalibrationRefusal(f"{path}: record has no fingerprint")

            source_unit = str(record.get("source_unit", ""))
            run_id = str(record.get("run_id", "") or "")
            if not run_id and run_id_manifest is not None:
                if source_unit not in run_id_manifest:
                    raise CalibrationRefusal(
                        f"{path}: no run id for source_unit {source_unit!r} — the "
                        "supplied run-id manifest must cover every calibration "
                        "unit, otherwise the locked artifact would carry a "
                        "partial custody record"
                    )
                run_id = run_id_manifest[source_unit]

            vectors.append(_coerce_fingerprint(record["fingerprint"]))
            partitions.append(partition_of(str(logical_id)))
            logical_ids.append(str(logical_id))
            scenarios.append(scenario)
            seeds.append(seed)
            run_ids.append(run_id)
            sources.append(str(path))
            source_units.append(source_unit)

    if not vectors:
        raise CalibrationRefusal("no fingerprint observations were loaded")
    widths = {v.shape[0] for v in vectors}
    if len(widths) != 1:
        raise CalibrationRefusal(f"inconsistent fingerprint dimensions: {sorted(widths)}")

    return Observations(
        vectors=np.vstack(vectors),
        partitions=tuple(partitions),
        logical_ids=tuple(logical_ids),
        scenarios=tuple(scenarios),
        seeds=tuple(seeds),
        run_ids=tuple(run_ids),
        sources=tuple(sources),
        source_units=tuple(source_units),
    )


# ===========================================================================
# τ selection
# ===========================================================================

def select_tau(across_distances: np.ndarray, fpr_target: float = FPR_TARGET) -> float:
    """τ at the dev-FPR = `fpr_target` point of the ACROSS-device distribution.

    A false positive is an across-device pair that would be *linked*, i.e. whose
    Mahalanobis distance is ≤ τ. τ is the **largest observed across-device
    distance whose cumulative count does not exceed** ``floor(fpr_target · n)``.

    That is stricter than a plain quantile and deliberately so: with ties (and
    identity resets produce exact ties) an interpolated quantile can realise an
    FPR *above* the target. This rule guarantees ``realised_fpr ≤ fpr_target``
    exactly, and is fully deterministic.
    """
    distances = np.asarray(across_distances, dtype=np.float64)
    if distances.size == 0:
        raise CalibrationRefusal(
            "cannot select τ: the across-device distance distribution is empty"
        )
    budget = int(np.floor(float(fpr_target) * distances.size))
    if budget < 1:
        raise CalibrationRefusal(
            f"only {distances.size} across-device pairs — too few to resolve an "
            f"FPR of {fpr_target} (need at least {int(np.ceil(1 / fpr_target))})"
        )
    values, counts = np.unique(distances, return_counts=True)
    cumulative = np.cumsum(counts)
    admissible = np.nonzero(cumulative <= budget)[0]
    if admissible.size == 0:
        raise CalibrationRefusal(
            "cannot select τ: the smallest across-device distance already occurs "
            f"more than {budget} times — the calibration population is degenerate"
        )
    tau = float(values[admissible[-1]])
    if not np.isfinite(tau) or tau <= 0.0:
        raise CalibrationRefusal(
            f"selected τ is not a usable positive distance: {tau!r} — the "
            "calibration population is degenerate"
        )
    return tau


# ===========================================================================
# Cohort calibration
# ===========================================================================

def _whiten(vectors: np.ndarray, metric: MahalanobisMetric) -> np.ndarray:
    """Map into the basis where Mahalanobis distance is plain Euclidean.

    Full-width vectors are projected through the metric's mask FIRST, via the
    metric's own `project` — the single application point (EDA_MEMO § 6.2), so
    the whitened basis and the fitted precision can never disagree about which
    dimensions exist.
    """
    scaled = metric.project(vectors) / metric.scale
    precision = metric.precision
    for jitter in (0.0, 1e-12, 1e-9, 1e-6):
        try:
            factor = np.linalg.cholesky(
                precision + jitter * np.eye(precision.shape[0])
            )
            break
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            factor = None
    if factor is None:  # pragma: no cover - defensive
        raise CalibrationRefusal("precision matrix is not positive definite")
    return scaled @ factor


def _qualifying_pair_count(codes: np.ndarray, same_group: bool) -> int:
    """How many within- (or across-) device pairs exist — WITHOUT building them."""
    n = int(codes.size)
    total = n * (n - 1) // 2
    counts = np.bincount(codes) if n else np.zeros(0, dtype=np.int64)
    within = int(sum(int(c) * (int(c) - 1) // 2 for c in counts))
    return within if same_group else total - within


def _enumerate_within_pairs(codes: np.ndarray) -> np.ndarray:
    """Every within-device pair, in lexicographic (i, j) order.

    Built group by group — the enumeration never touches a pair whose members
    sit in different devices, so its cost is the size of the ANSWER, not of the
    full n(n-1)/2 combination space.
    """
    blocks: List[np.ndarray] = []
    for key in np.unique(codes):
        members = np.nonzero(codes == key)[0]
        if members.size < 2:
            continue
        left, right = np.triu_indices(members.size, k=1)
        blocks.append(np.stack([members[left], members[right]], axis=1))
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.vstack(blocks).astype(np.int64, copy=False)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _enumerate_across_pairs(codes: np.ndarray) -> np.ndarray:
    """Every across-device pair, in lexicographic (i, j) order, in row blocks.

    Peak cost is one `block x n` boolean mask rather than the whole pair space.
    """
    n = int(codes.size)
    blocks: List[np.ndarray] = []
    for start in range(0, n, _PAIR_ENUMERATION_ROW_BLOCK):
        stop = min(start + _PAIR_ENUMERATION_ROW_BLOCK, n)
        rows = np.arange(start, stop)
        columns = np.arange(start + 1, n)
        if columns.size == 0:
            continue
        keep = (columns[None, :] > rows[:, None]) & (
            codes[columns][None, :] != codes[rows][:, None]
        )
        row_index, column_index = np.nonzero(keep)
        if row_index.size:
            blocks.append(
                np.stack([rows[row_index], columns[column_index]], axis=1)
            )
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(blocks).astype(np.int64, copy=False)


def _row_starts(n: int) -> np.ndarray:
    """Rank of the first pair (i, i+1) for every i, plus the total at the end.

    Lets a pair rank be unranked to (i, j) in exact integer arithmetic with O(n)
    memory, which is what makes sampling possible without a pair array.
    """
    counts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])


def _unrank_pairs(
    ranks: np.ndarray, row_start: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair rank -> (i, j), i < j, in exact integer arithmetic.

    Returned as two 1-D arrays rather than an (n, 2) stack so the sampling loop
    can test the group condition without paying for the stacked copy.
    """
    first = np.searchsorted(row_start, ranks, side="right") - 1
    second = first + 1 + (ranks - row_start[first])
    return first, second


def _distinct_ranks(
    total: int,
    size: int,
    rng: np.random.Generator,
    acceptance: float,
    keep: Any = None,
    label: str = "pair",
) -> np.ndarray:
    """`size` distinct ranks, uniform over the population `keep` admits.

    Ranks are drawn i.i.d. over `[0, total)` in bounded batches; when `keep` is
    given, the survivors are i.i.d. uniform over the ADMITTED sub-population. The
    distinct values of an i.i.d. uniform draw are exchangeable, so a uniformly
    chosen `size`-subset of them is a uniform sample WITHOUT replacement — the
    same distribution `rng.choice(..., replace=False)` gives, obtained without
    ever building the population it would have chosen from.
    """
    collected = np.empty(0, dtype=np.int64)
    drawn = 0
    draw_budget = 64 * size + 1_000_000
    while collected.size < size:
        need = size - collected.size
        batch = int(min(_PAIR_SAMPLING_BATCH, need / acceptance * 1.3 + 1024))
        ranks = rng.integers(0, total, size=batch, dtype=np.int64)
        if keep is not None:
            ranks = ranks[keep(ranks)]
        collected = np.union1d(collected, ranks)
        del ranks
        drawn += batch
        if drawn > draw_budget:  # pragma: no cover - unreachable by design
            raise CalibrationRefusal(
                f"{label} sampling did not converge: {collected.size} of {size} "
                f"distinct ranks after {drawn} draws"
            )
    chosen = collected[rng.permutation(collected.size)[:size]]
    del collected
    chosen.sort()
    return chosen


def _device_members(codes: np.ndarray) -> List[np.ndarray]:
    """Ascending row indices per device, devices with < 2 observations dropped."""
    members: List[np.ndarray] = []
    for key in np.unique(codes):
        index = np.nonzero(codes == key)[0]
        if index.size >= 2:
            members.append(index)
    return members


def _sample_within_pair_indices(
    codes: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform sample of within-device pairs, drawn in the WITHIN rank space.

    Rejection-sampling these out of the full pair space would be hopeless — at
    the calibration corpus the within-device pairs are ~5 % of all pairs — so
    the rank space is the concatenation of each device's own triangular index.
    Every draw is admissible by construction.
    """
    members = _device_members(codes)
    sizes = np.array([m.size for m in members], dtype=np.int64)
    counts = sizes * (sizes - 1) // 2
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    ranks = _distinct_ranks(
        int(offsets[-1]), size, rng, acceptance=1.0, label="within-device pair"
    )

    device = np.searchsorted(offsets, ranks, side="right") - 1
    pairs = np.empty((ranks.size, 2), dtype=np.int64)
    for position, index in enumerate(members):
        selected = np.nonzero(device == position)[0]
        if selected.size == 0:
            continue
        local = ranks[selected] - offsets[position]
        left, right = _unrank_pairs(local, _row_starts(int(index.size)))
        pairs[selected, 0] = index[left]
        pairs[selected, 1] = index[right]
    return pairs


def _sample_across_pair_indices(
    codes: np.ndarray, qualifying: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform sample of across-device pairs, drawn in the FULL pair rank space.

    Same-device draws are rejected. That is efficient here and only here: the
    across-device pairs dominate any multi-device population, so the acceptance
    rate is high, and no pair array is materialised at any point.
    """
    row_start = _row_starts(int(codes.size))
    total = int(row_start[-1])

    def _keep(ranks: np.ndarray) -> np.ndarray:
        first, second = _unrank_pairs(ranks, row_start)
        return codes[first] != codes[second]

    chosen = _distinct_ranks(
        total,
        size,
        rng,
        acceptance=max(qualifying / total, 1e-6),
        keep=_keep,
        label="across-device pair",
    )
    first, second = _unrank_pairs(chosen, row_start)
    return np.stack([first, second], axis=1)


def _blocked_distances(whitened: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Euclidean distances over `pairs`, in blocks.

    Elementwise identical to the single-shot form; blocked only so the
    (n_pairs x dim) delta matrix never exists in full — at the pre-registered
    corpus that matrix alone would be 2e6 x 180 x 8 B = 2.9 GB.
    """
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for start in range(0, pairs.shape[0], _DISTANCE_BLOCK_PAIRS):
        stop = min(start + _DISTANCE_BLOCK_PAIRS, pairs.shape[0])
        block = pairs[start:stop]
        deltas = whitened[block[:, 0]] - whitened[block[:, 1]]
        out[start:stop] = np.sqrt(
            np.maximum(np.einsum("ij,ij->i", deltas, deltas), 0.0)
        )
    return out


def _pair_distances(
    whitened: np.ndarray,
    groups: Sequence[int],
    same_group: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distances over within-device (same_group) or across-device pairs.

    At or below `MAX_PAIRS_PER_POPULATION` this is the full pair population, in
    the same lexicographic order the old `itertools.combinations` enumeration
    produced, and NO randomness is consumed. Above it the population is
    subsampled exactly as before.

    The pair set is never materialised in full: the old implementation built
    `np.array(list(itertools.combinations(range(n), 2)))` before filtering, which
    at the 9 795-observation calibration corpus is 47 966 115 rows — ~768 MB per
    call, twice per metric, before its 180-dim deltas.
    """
    codes = np.unique(np.asarray(groups), return_inverse=True)[1].astype(
        np.int64, copy=False
    )
    if codes.size != whitened.shape[0]:
        raise CalibrationRefusal(
            f"group labels ({codes.size}) do not match observations "
            f"({whitened.shape[0]})"
        )

    qualifying = _qualifying_pair_count(codes, same_group)
    if qualifying == 0:
        return np.empty(0, dtype=np.float64)

    cap = MAX_PAIRS_PER_POPULATION
    if qualifying <= max(EXACT_PAIR_ENUMERATION_LIMIT, cap):
        pairs = (
            _enumerate_within_pairs(codes)
            if same_group
            else _enumerate_across_pairs(codes)
        )
        if qualifying > cap:
            # The pre-existing subsample, byte-for-byte, wherever the qualifying
            # population is still small enough to enumerate.
            chosen = rng.choice(pairs.shape[0], size=cap, replace=False)
            pairs = pairs[np.sort(chosen)]
    elif same_group:
        pairs = _sample_within_pair_indices(codes, cap, rng)
    else:
        pairs = _sample_across_pair_indices(codes, qualifying, cap, rng)

    return _blocked_distances(whitened, pairs)


def _summary(distances: np.ndarray) -> Dict[str, float]:
    if distances.size == 0:
        return {}
    return {
        "n": int(distances.size),
        "min": float(np.min(distances)),
        "p01": float(np.quantile(distances, 0.01)),
        "median": float(np.median(distances)),
        "p99": float(np.quantile(distances, 0.99)),
        "max": float(np.max(distances)),
    }


def _refuse_constant_devices(
    vectors: np.ndarray,
    partitions: Sequence[int],
    cohort: CalibrationCohort,
) -> None:
    """EMISSION_CONTRACT § 3.7 — explicit, SIZE-INDEPENDENT degeneracy refusal.

    The across-pair TIE COUNT in `select_tau` is not a reliable detector of a
    degenerate calibration population: it fires on `floor(0.01 * n_pairs)`, which
    is a function of COHORT SIZE. Measured, under a cached (non-varying)
    fingerprint contract: the 10-device adjudicating cohort refused correctly
    while the 20-device validation cohort produced a LOCKABLE τ with no error and
    a headline `within_link_rate` of 1.0000 — a *perfect* instrument, the most
    attractive possible wrong answer, with the only tells in fields nobody gates
    on (`within_median == 0.0`, τ of order 1e16).

    A device whose observations are all identical also poisons the pooled
    within-device scatter directly — it contributes exactly-zero residuals to Σ
    and trivially-linked within-pairs that inflate the headline — so it is
    checked on the RAW vectors, before any metric is estimated, and refuses
    whichever cohort it appears in.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    keys = list(partitions)
    constant: List[int] = []
    for key in sorted(set(keys)):
        index = [i for i, k in enumerate(keys) if k == key]
        if len(index) < 2:
            continue  # a single observation is not evidence of a constant device
        block = matrix[index]
        if bool(np.all(block == block[0])):
            constant.append(key)
    if constant:
        raise CalibrationRefusal(
            f"cohort '{cohort.value}': degenerate within-device calibration "
            f"population — device(s) {constant} have identical observations on "
            "every round, so their within-device scatter is exactly zero. A "
            "within-client distance distribution that is a point mass at zero is "
            "not a distribution the pre-registered τ procedure can operate on "
            "(EMISSION_CONTRACT § 3.7). Fix the fingerprint emission contract; "
            "do NOT lock a τ from this population."
        )


def _refuse_zero_within_median(
    within: np.ndarray,
    cohort: CalibrationCohort,
    metric_name: str,
) -> None:
    """The second § 3.7 condition, independent of cohort size and of any device
    being wholly constant: a within-device MEDIAN distance of exactly zero."""
    if within.size == 0:
        return
    median = float(np.median(within))
    if median == 0.0:
        raise CalibrationRefusal(
            f"cohort '{cohort.value}' / metric '{metric_name}': degenerate "
            "within-device calibration population — within_median == 0.0, i.e. "
            "at least half of all same-device pairs sit at exactly zero "
            "distance. τ selected against this population is meaningless "
            "however healthy the across-device side looks "
            "(EMISSION_CONTRACT § 3.7)."
        )


def fingerprint_dim_names() -> Tuple[str, ...]:
    """The 180 dimension names, `column__moment`, in emission order."""
    from flowerfl.fingerprint import FINGERPRINT_MOMENTS, load_feature_spec

    spec = load_feature_spec()
    return tuple(
        f"{column}__{moment}"
        for column in spec.features
        for moment in FINGERPRINT_MOMENTS
    )


def _normalised_squares(values: np.ndarray) -> np.ndarray:
    """Squares of `values` rescaled so the largest is exactly 1.0.

    `top1_share` is max(x^2) / sum(x^2), a RATIO of squares, so it is invariant
    under any positive rescaling of the column. Dividing by max |x| BEFORE
    squaring is therefore EXACT, not an approximation — and it is the only way
    to compute the share at all for the raw-scale `tcp.payload` family, whose
    values pass ~1.3e154 and square straight past float64's 1.8e308 into `inf`.

    That overflow was not merely imprecise, it was SILENT: `inf/inf` is NaN, and
    `NaN > 0.0` is False, so the affected column never displaced the initialised
    0.0 and was recorded in the artifact as `pool_top1_share = 0.0` —
    indistinguishable from a healthy column. The only tell was a RuntimeWarning
    in the calibration log (the overflow finding).
    """
    array = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(array))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(array)
    return (array / scale) ** 2


def pool_screen(
    features: Sequence[str],
    parquet_dir: Path = POOL_PARQUET_DIR,
    partitions: Sequence[int] = POOL_PARTITIONS,
) -> Dict[str, Any]:
    """The A-PRIORI pool-side screen — distance-free and fingerprint-free.

    Mirrors `results/20260814/h3_feature_eda/step3_pool_rule.py`::

        top1_share(column, device) = max(x^2) / sum(x^2) over the device's pool
        FLAG column iff max over devices top1_share > 0.5

    A column where ONE pool row dominates the second moment makes the emitted
    moment an integer capture count — the partitions-7/11 mechanism. Reading the
    RAW pools rather than the fingerprints is what makes this screen a-priori:
    it cannot see which device failed, and in the EDA it caught
    `http.content_length` through partition 3's pool, a device that never failed.

    A missing partition parquet REFUSES. A screen that silently skipped a device
    would be a screen that cannot claim "in ANY device".
    """
    import pandas as pd

    parquet_dir = Path(parquet_dir)
    columns = list(features)
    top1: Dict[str, float] = {column: 0.0 for column in columns}
    worst: Dict[str, Optional[int]] = {column: None for column in columns}

    for partition in partitions:
        path = parquet_dir / f"client_{int(partition)}.parquet"
        if not path.exists():
            raise CalibrationRefusal(
                f"pool parquet missing for partition {partition}: {path}. The "
                "a-priori pool screen flags a column when ONE device's pool is "
                "dominated by a single row, so it cannot be computed with a "
                "device unread."
            )
        frame = pd.read_parquet(path, columns=columns)
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            squares = _normalised_squares(values)
            total = float(squares.sum())
            if total <= 0.0:
                continue
            share = float(squares.max() / total)
            if not np.isfinite(share):
                # FAIL CLOSED. An unresolvable share must never be read as "this
                # column is fine" — that is exactly how the overflow hid.
                raise CalibrationRefusal(
                    f"pool screen: column {column!r} on partition {partition} "
                    f"produced a non-finite top-1 second-moment share "
                    f"({share!r}). The screen cannot say whether one pool row "
                    "dominates this column, and an unscreened column is not a "
                    "passed one. Do NOT lock a tau from this pool."
                )
            if share > top1[column]:
                top1[column] = share
                worst[column] = int(partition)

    flagged = sorted(
        column for column in columns
        if top1[column] > POOL_FLAG_TOP1_SHARE_THRESHOLD
    )
    return {
        "threshold": POOL_FLAG_TOP1_SHARE_THRESHOLD,
        "partitions": [int(p) for p in partitions],
        "parquet_dir": str(parquet_dir),
        "top1_share": top1,
        "worst_device": worst,
        "pool_flagged": flagged,
    }


def control_screens(
    vectors: np.ndarray, partitions: Sequence[int]
) -> Dict[str, List[Any]]:
    """The three control-side degeneracy screens, per dimension.

    Mirrors `step1_scores.py`: `dead` (constant in EVERY device), `lattice_flag`
    (few distinct levels with a relative spread that dwarfs ordinary jitter, in
    ANY device), and `het_ratio` (max/min within-device std across devices).

    Computed over every device present in the corpus — see the module-level rule
    note. A device with a single observation contributes no std and is skipped
    for the het ratio, but still counts for `dead`.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    keys = [int(p) for p in partitions]
    if matrix.shape[0] != len(keys):
        raise CalibrationRefusal(
            f"group labels ({len(keys)}) do not match observations "
            f"({matrix.shape[0]})"
        )
    devices = sorted(set(keys))
    blocks = {device: matrix[[i for i, k in enumerate(keys) if k == device]]
              for device in devices}
    n_dims = int(matrix.shape[1])

    dead: List[bool] = []
    lattice_flag: List[bool] = []
    lattice_levels: List[int] = []
    for index in range(n_dims):
        constant = True
        flagged = False
        worst_levels = 0
        for block in blocks.values():
            column = block[:, index]
            levels = np.unique(np.round(column, 10))
            if levels.size <= 1:
                continue
            constant = False
            worst_levels = levels.size if worst_levels == 0 else min(
                worst_levels, levels.size
            )
            spread = (levels.max() - levels.min()) / (abs(np.median(column)) + 1e-12)
            if (levels.size <= LATTICE_MAX_DISTINCT_LEVELS
                    and spread > LATTICE_MIN_RELATIVE_SPREAD):
                flagged = True
        dead.append(bool(constant))
        lattice_flag.append(bool(flagged))
        lattice_levels.append(int(worst_levels))

    # Per-dimension pre-scaling by the corpus-wide max |value|. The std is
    # scale-EQUIVARIANT and the het ratio (max std / min std) therefore
    # scale-INVARIANT, so dividing first is EXACT — and it is the only way to
    # compute a variance at all for the raw-scale `tcp.payload` family, whose
    # values pass ~1.3e154 and square past float64's 1.8e308 inside `var`.
    # Unscaled, the overflow was SILENT AND FAIL-OPEN: the NaN het ratio failed
    # the `> HET_RATIO_MAX` comparison and the dim SURVIVED a screen that never
    # evaluated it (the r5_v2 candidate run's RuntimeWarnings; same defect
    # class as the pool screen's overflow finding, one screen over).
    dim_scale = np.max(np.abs(matrix), axis=0)
    dim_scale = np.where(dim_scale > 0, dim_scale, 1.0)
    stds = np.vstack([
        np.sqrt((block / dim_scale).var(axis=0, ddof=1))
        for block in blocks.values()
        if block.shape[0] >= 2
    ]) if any(b.shape[0] >= 2 for b in blocks.values()) else np.zeros((1, n_dims))
    if not np.all(np.isfinite(stds)):
        measured_devices = [d for d in devices if blocks[d].shape[0] >= 2]
        rows_bad, cols_bad = np.nonzero(~np.isfinite(stds))
        pairs = sorted({
            f"dim {int(col)} on device {measured_devices[int(row)]}"
            for row, col in zip(rows_bad, cols_bad)
        })
        raise CalibrationRefusal(
            "control screens: non-finite within-device std after per-dimension "
            f"scaling ({'; '.join(pairs)}) — the het screen cannot evaluate "
            "these and MUST NOT pass them unevaluated. Fix the input or the "
            "construct; an unevaluated screen is not a pass."
        )
    # A dim that never varies in any device has no finite ratio; it is caught by
    # the `dead` screen, so it is given inf here rather than left to produce an
    # all-NaN slice warning from `nanmin`.
    het_ratio: List[float] = []
    for index in range(n_dims):
        column = stds[:, index]
        positive = column[column > 0]
        if positive.size == 0:
            het_ratio.append(float("inf"))
            continue
        het_ratio.append(float(column.max() / positive.min()))

    return {
        "dead": dead,
        "lattice_flag": lattice_flag,
        "lattice_worst_levels": lattice_levels,
        "het_ratio": het_ratio,
        "devices": devices,
    }


def _finite_or_none(value: float) -> Optional[float]:
    """A JSON-safe float. Non-finite becomes null.

    `json.dumps` writes bare `Infinity` / `NaN`, which are NOT JSON: a strict
    parser doing custody or lock verification rejects the whole artifact. A dead
    dimension legitimately has an infinite het ratio (it never varies, so the
    min within-device std is zero), and null is the truthful encoding of
    "undefined" — the dim's separate `dead` outcome is what records WHY it was
    dropped, and is untouched by this.
    """
    number = float(value)
    return number if np.isfinite(number) else None


def build_feature_mask(
    vectors: np.ndarray,
    partitions: Sequence[int],
    pool_report: Mapping[str, Any],
    dim_names: Sequence[str],
) -> Dict[str, Any]:
    """Apply the frozen R5 rule and return the mask plus every screen's outcome.

    Every dimension gets a recorded outcome for every screen, whether or not it
    survived, so the lock artifact answers "why was this dim dropped?" without a
    re-run. `dropped_by` lists the screens that excluded it — a dim can fail
    several, and recording only the first would misrepresent the rule.
    """
    names = [str(name) for name in dim_names]
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape[1] != len(names):
        raise CalibrationRefusal(
            f"corpus width {matrix.shape[1]} != {len(names)} declared dimension "
            "names — the mask would be recorded against the wrong features"
        )
    screens = control_screens(matrix, partitions)
    flagged_columns = set(pool_report["pool_flagged"])
    top1 = pool_report["top1_share"]

    records: List[Dict[str, Any]] = []
    mask: List[int] = []
    ablation: List[int] = []
    for index, name in enumerate(names):
        column = name.rsplit("__", 1)[0]
        is_dead = bool(screens["dead"][index])
        is_pool = column in flagged_columns
        is_lattice = bool(screens["lattice_flag"][index])
        het = float(screens["het_ratio"][index])
        dropped = [
            label
            for label, failed in (
                ("dead", is_dead),
                ("pool_flagged", is_pool),
                ("lattice_flagged", is_lattice),
                # A dead dim has no scatter at all, so its het ratio is
                # undefined (inf) rather than "too heteroscedastic". Reporting
                # it under this screen too would misdescribe why it was
                # dropped; `dead` already excludes it.
                ("het_ratio_above_cap", not is_dead and het > HET_RATIO_MAX),
            )
            if failed
        ]
        if not dropped:
            mask.append(index)
        if not is_dead and not is_pool:
            ablation.append(index)
        records.append({
            "dim": index,
            "name": name,
            "column": column,
            "dead": is_dead,
            "lattice_flag": is_lattice,
            "lattice_worst_levels": int(screens["lattice_worst_levels"][index]),
            "het_ratio": _finite_or_none(het),
            "pool_flagged": is_pool,
            "pool_top1_share": float(top1.get(column, 0.0)),
            "surviving": not dropped,
            "dropped_by": dropped,
        })

    if len(mask) < MIN_SURVIVING_DIMS:
        raise CalibrationRefusal(
            f"the frozen {FEATURE_SELECTION_RULE_ID} rule leaves {len(mask)} "
            f"surviving dimension(s) of {len(names)} — fewer than the "
            f"{MIN_SURVIVING_DIMS} a covariance needs. Either the corpus is not "
            "the pre-registered control population or the screens are being fed "
            "the wrong pool; do NOT lock a tau from this mask."
        )

    return {
        "rule": FEATURE_SELECTION_RULE_ID,
        "rule_description": FEATURE_SELECTION_RULE,
        "mask": mask,
        "mask_names": [names[i] for i in mask],
        "n_surviving": len(mask),
        "n_total": len(names),
        "ablation_R4_mask": ablation,
        "ablation_R4_n_surviving": len(ablation),
        "ablation_note": (
            "R4 = dead + pool screen only, the parameter-free ablation "
            "(EDA_MEMO § 6.1). Recorded for comparison; never selected here."
        ),
        "parameters": {
            "pool_flag_top1_share_threshold": POOL_FLAG_TOP1_SHARE_THRESHOLD,
            "het_ratio_max": HET_RATIO_MAX,
            "lattice_max_distinct_levels": LATTICE_MAX_DISTINCT_LEVELS,
            "lattice_min_relative_spread": LATTICE_MIN_RELATIVE_SPREAD,
        },
        "pool_screen": {
            "threshold": pool_report["threshold"],
            "parquet_dir": pool_report.get("parquet_dir"),
            "partitions": pool_report.get("partitions"),
            "pool_flagged": sorted(flagged_columns),
            "top1_share": dict(top1),
            "worst_device": dict(pool_report.get("worst_device", {})),
        },
        "screened_devices": screens["devices"],
        "dims": records,
    }


def _device_within_medians(
    whitened: np.ndarray,
    partitions: Sequence[int],
    devices: Sequence[int],
    rng: np.random.Generator,
) -> Dict[int, Tuple[float, int]]:
    """Median within-device distance per device, in the whitened basis.

    Returns ``{device: (median, n_pairs)}``, omitting any device with fewer than
    two observations — one draw carries no within-device distance at all, so
    there is nothing to take a median of. Callers must treat an omitted scored
    device as a refusal, not as a pass.

    Above `MAX_WITHIN_PAIRS_PER_SCOREABLE_DEVICE` the capped sample is drawn in
    the device's own pair RANK space (`_row_starts` / `_unrank_pairs`, the same
    machinery `_sample_within_pair_indices` uses) and the full triangle is never
    built. Enumerating first and subsampling after would make the cap bound
    nothing: 20 000 observations of one device is ~200 M pairs allocated to keep
    200 k of them.
    """
    codes = np.asarray(partitions)
    medians: Dict[int, Tuple[float, int]] = {}
    cap = MAX_WITHIN_PAIRS_PER_SCOREABLE_DEVICE
    for device in devices:
        members = np.nonzero(codes == device)[0]
        if members.size < 2:
            continue
        total = int(members.size) * (int(members.size) - 1) // 2
        if total > cap:
            row_start = _row_starts(int(members.size))
            ranks = _distinct_ranks(
                total, cap, rng, acceptance=1.0,
                label=f"within-device pair (partition {device})",
            )
            left, right = _unrank_pairs(ranks, row_start)
        else:
            left, right = np.triu_indices(members.size, k=1)
        pairs = np.stack([members[left], members[right]], axis=1)
        distances = _blocked_distances(whitened, pairs)
        medians[int(device)] = (float(np.median(distances)), int(pairs.shape[0]))
    return medians


def scoreable_homogeneity_report(
    vectors: np.ndarray,
    partitions: Sequence[int],
    cohort: CalibrationCohort,
    metric: MahalanobisMetric,
    metric_name: str,
    tau: float,
    *,
    scoreable_partitions: Sequence[int],
    calibration_partitions: Sequence[int],
) -> Dict[str, Any]:
    """ROOTCAUSE_MEMO § 5(1) — can the SCORED devices live under this (τ, Σ)?

    A locked (τ, Σ) is an instrument applied to the devices the cohort SCORES,
    but for the ADJUDICATING cohort those devices are precisely the ones D9 axis
    (ii) holds OUT of the fit. Every other guard here reads the calibration
    population, so a hold-out device's noise is structurally invisible to all of
    them — which is exactly what happened at the EXP-050 lock: the even-device
    fit reported a within-link rate of 0.9939 while partitions 7 and 11 sat at
    6.8 τ and 98 000 τ **against their own next-round selves**, unread.

    The condition is the one the memo names: a device whose MEDIAN within-device
    distance exceeds τ cannot be re-identified as itself even in the attack-free
    control data, so no threshold-based verdict computed over it means anything.
    The comparison uses the CANDIDATE metric and that candidate's OWN τ, so each
    Addendum-A candidate is judged at its own operating point, never the other's.

    `scoreable_partitions` is an explicit PARAMETER, never inferred from the
    corpus: the set of devices a cohort will score is a pre-registered design
    fact (`SCOREABLE_PARTITIONS`), and deriving it from whatever happened to be
    in the input would make the gate silently weaker on a thin corpus.

    Inputs are the attack-free control fingerprints already loaded for
    calibration. The gate reads no eval scenario and no scored quantity, so it
    is legitimately pre-registerable.

    The gate is FAIL-CLOSED on a scored device it cannot measure. A device with
    fewer than two observations has no within-device distance at all, so there
    is nothing to compare against τ — and `_require_complete_calibration_corpus`
    checks (scenario, seed) cells, NOT partition coverage, so nothing else in
    this file would notice. Recording such a device and returning "not refused"
    would let a corpus carrying zero rows for partition 7 produce a lock while
    silently skipping precisely the device class this gate exists for. Both
    failure modes therefore refuse, and the message distinguishes them.

    This function is PURE: it computes and returns, never raises on the
    condition. `_refuse_heterogeneous_scoreable_devices` turns the report into a
    refusal, and is applied to the SELECTED metric only — see there.
    """
    whitened = _whiten(np.asarray(vectors, dtype=np.float64), metric)
    rng = np.random.default_rng(PAIR_SAMPLING_SEED)

    scoreable = sorted({int(p) for p in scoreable_partitions})
    calibration = sorted({int(p) for p in calibration_partitions})
    measured = _device_within_medians(
        whitened, partitions, sorted(set(scoreable) | set(calibration)), rng
    )

    calibration_medians = [measured[d][0] for d in calibration if d in measured]
    max_calibration_median = max(calibration_medians) if calibration_medians else 0.0

    observation_counts = {
        int(device): int(np.count_nonzero(np.asarray(partitions) == device))
        for device in scoreable
    }
    devices: List[Dict[str, Any]] = []
    unmeasurable: List[Dict[str, Any]] = []
    for device in scoreable:
        if device not in measured:
            unmeasurable.append(
                {"partition": device, "n_observations": observation_counts[device]}
            )
            continue
        median, n_pairs = measured[device]
        devices.append(
            {
                "partition": device,
                "within_median": median,
                "n_within_pairs": n_pairs,
                "within_median_over_tau": (
                    _finite_or_none(median / tau) if tau else None
                ),
                "ratio_to_calibration_max": (
                    _finite_or_none(median / max_calibration_median)
                    if max_calibration_median > 0.0
                    else None
                ),
                "exceeds_tau": bool(median > tau),
            }
        )

    if unmeasurable:
        print(
            f"REFUSING: cohort '{cohort.value}' / metric '{metric_name}': the "
            f"scoreable-device homogeneity gate cannot measure device(s) "
            f"{[d['partition'] for d in unmeasurable]} — fewer than two "
            "observations each. A scored device the gate cannot see is the "
            "failure mode the gate exists to prevent.",
            file=sys.stderr,
        )

    offenders = [d for d in devices if d["exceeds_tau"]]
    return {
        "metric_name": metric_name,
        "tau": float(tau),
        "cohort": CalibrationCohort(cohort).value,
        "scoreable_partitions": scoreable,
        "calibration_partitions": calibration,
        "n_devices_checked": len(devices),
        "scoreable_devices": devices,
        "scoreable_devices_unmeasurable": unmeasurable,
        "max_scoreable_median": max((d["within_median"] for d in devices), default=0.0),
        "max_calibration_median": float(max_calibration_median),
        "offending_partitions": [d["partition"] for d in offenders],
        "unmeasurable_partitions": [d["partition"] for d in unmeasurable],
        "refused": bool(offenders) or bool(unmeasurable),
    }


def _refuse_heterogeneous_scoreable_devices(report: Mapping[str, Any]) -> None:
    """Turn a failing homogeneity report into the § 5(1) refusal.

    Applied to the SELECTED metric ONLY, and deliberately so. Addendum A
    calibrates both candidates and discards one under a predeclared rule; the
    loser is recorded as evidence but is not an instrument, and no (τ, Σ) is
    ever locked from it. Refusing on a candidate that lost the comparison would
    let a rejected estimator veto a lock whose actual metric is healthy — a
    different gate from the one the memo specifies, and a stricter one than the
    evidence supports. Both candidates' reports are still written into the
    artifact, so a loser that would have failed is visible, never silent.
    """
    if not report.get("refused"):
        return

    unmeasurable = report.get("scoreable_devices_unmeasurable") or []
    if unmeasurable:
        counts = "; ".join(
            f"partition {d['partition']}: {d['n_observations']} observation"
            f"{'' if d['n_observations'] == 1 else 's'}"
            for d in unmeasurable
        )
        raise CalibrationRefusal(
            f"cohort '{report['cohort']}' / metric '{report['metric_name']}': the "
            f"within-device scatter of scored device(s) "
            f"{report['unmeasurable_partitions']} CANNOT BE MEASURED — {counts}, "
            "and two are needed for a single within-device distance. The "
            "(scenario, seed) completeness check does not cover partition "
            "coverage, so nothing else would notice: this lock would be written "
            "having silently skipped exactly the scored devices the gate exists "
            "to check (ROOTCAUSE_MEMO 2026-08-14 § 5(1)). Supply the missing "
            "device(s)' control-run fingerprints; do NOT lock a tau that was "
            "never tested against every device it will score."
        )

    offenders = [d for d in report["scoreable_devices"] if d["exceeds_tau"]]
    max_calibration_median = float(report["max_calibration_median"])
    def _ratio(value: Any) -> str:
        return "undefined" if value is None else f"{float(value):.3g}"

    detail = "; ".join(
        f"partition {d['partition']}: within-device median {d['within_median']:.6g} "
        f"= {_ratio(d['within_median_over_tau'])} x tau, and "
        f"{_ratio(d['ratio_to_calibration_max'])} x the calibration cohort's "
        f"worst device ({max_calibration_median:.6g})"
        for d in offenders
    )
    raise CalibrationRefusal(
        f"cohort '{report['cohort']}' / metric '{report['metric_name']}': the "
        f"within-device scatter of device(s) {report['offending_partitions']} "
        f"EXCEEDS the locked tau ({float(report['tau']):.6g}) — {detail}. These "
        "devices are SCORED under this lock; for the adjudicating cohort they are "
        "also held OUT of it, so their noise never reached the covariance that has "
        "to whiten it. A device whose own next emission is further from it than "
        "tau cannot be re-identified as itself even on attack-free control data, "
        "and any threshold-based verdict computed over it measures the instrument, "
        "not the device (ROOTCAUSE_MEMO 2026-08-14 § 5(1)). Fix the fingerprint "
        "moment construct; do NOT lock a tau from this population."
    )


def select_metric_by_rule(summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply Addendum A's PREDECLARED selection rule. No discretion, no override.

    Higher calibration `within_link_rate` at each metric's OWN realised-FPR ≤ 0.01
    operating point wins; any tie within `METRIC_TIE_MARGIN` goes to the SIMPLER
    metric. Pure and total — one pre-named number decides it.
    """
    missing = [name for name in CANDIDATE_METRICS if name not in summaries]
    if missing:  # pragma: no cover - defensive
        raise CalibrationRefusal(
            f"the predeclared comparison needs both candidate metrics; missing {missing}"
        )
    incumbent = float(summaries[INCUMBENT_METRIC]["within_link_rate"])
    alternative = float(summaries[ALTERNATIVE_METRIC]["within_link_rate"])
    delta = incumbent - alternative

    if abs(delta) <= METRIC_TIE_MARGIN + _TIE_MARGIN_EPSILON:
        selected = SIMPLER_METRIC
        reason = (
            f"TIE: |within_link_rate delta| = {abs(delta):.6f} <= tie margin "
            f"{METRIC_TIE_MARGIN} — the tie goes to the SIMPLER metric "
            f"({SIMPLER_METRIC})"
        )
    elif delta > 0:
        selected = INCUMBENT_METRIC
        reason = (
            f"{INCUMBENT_METRIC} wins outright: within_link_rate {incumbent:.6f} "
            f"vs {alternative:.6f} (delta {delta:.6f} > {METRIC_TIE_MARGIN})"
        )
    else:
        selected = ALTERNATIVE_METRIC
        reason = (
            f"{ALTERNATIVE_METRIC} wins outright: within_link_rate "
            f"{alternative:.6f} vs {incumbent:.6f} (delta {-delta:.6f} > "
            f"{METRIC_TIE_MARGIN})"
        )

    return {
        "selected_metric": selected,
        "rule": METRIC_SELECTION_RULE,
        "reason": reason,
        "within_link_rate": {
            INCUMBENT_METRIC: incumbent,
            ALTERNATIVE_METRIC: alternative,
        },
        "within_link_rate_delta": delta,
        "tie_margin": METRIC_TIE_MARGIN,
        "simpler_metric": SIMPLER_METRIC,
        "candidate_metrics": list(CANDIDATE_METRICS),
    }


def _calibrate_metric(
    vectors: np.ndarray,
    partitions: Sequence[int],
    cohort: CalibrationCohort,
    metric_name: str,
    *,
    corpus_vectors: np.ndarray,
    corpus_partitions: Sequence[int],
    scoreable_partitions: Sequence[int],
    mask: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """One candidate metric's complete calibration summary, on ONE cohort.

    Each candidate is estimated on, and scored against, the same cohort-local
    population, and receives **its own** τ through the unchanged `select_tau` —
    neither metric is ever scored at the other's threshold. The pair sample is
    drawn from a freshly seeded generator, so both candidates see the identical
    pair population and the comparison is not confounded by the sample.

    `corpus_*` is the WHOLE loaded corpus, not the cohort-filtered population:
    the scoreable-homogeneity gate has to read devices this cohort deliberately
    excludes from its fit. Those rows reach the gate and nothing else — every
    estimate below is still computed from `vectors` / `partitions` alone, so no
    hold-out fingerprint can touch τ or Σ (D9 axis (ii)).
    """
    # The R5 mask is applied HERE, before the fit: Sigma and tau are estimated
    # on the surviving dimensions only (draft amendment § 3.1(b)). Everything
    # below — whitening, pair distances, tau selection, and the § 5(1)
    # homogeneity gate — therefore operates in the masked geometry.
    metric = MahalanobisMetric.from_within_device_population(
        vectors,
        partitions,
        shrinkage=METRIC_SHRINKAGE[metric_name],
        provenance=f"h3-calibration/{cohort.value}/{metric_name}",
        mask=mask,
        input_dim=int(np.asarray(vectors).shape[1]) if mask is not None else None,
    )
    whitened = _whiten(vectors, metric)
    rng = np.random.default_rng(PAIR_SAMPLING_SEED)

    within = _pair_distances(whitened, partitions, same_group=True, rng=rng)
    _refuse_zero_within_median(within, cohort, metric_name)
    across = _pair_distances(whitened, partitions, same_group=False, rng=rng)
    tau = select_tau(across, FPR_TARGET)

    # ROOTCAUSE_MEMO § 5(1) diagnostics, per candidate. The refusal itself is
    # applied post-selection in `build_cohort_calibration` — still before any
    # (τ, Σ) can reach the artifact writer.
    homogeneity = scoreable_homogeneity_report(
        corpus_vectors,
        corpus_partitions,
        cohort,
        metric,
        metric_name,
        tau,
        scoreable_partitions=scoreable_partitions,
        calibration_partitions=partitions,
    )

    return {
        "metric_name": metric_name,
        "tau": tau,
        "fpr_target": FPR_TARGET,
        "realised_fpr": float(np.mean(across <= tau)),
        "within_link_rate": float(np.mean(within <= tau)),
        "shrinkage": float(metric.shrinkage),
        "n_within_pairs": int(within.size),
        "n_across_pairs": int(across.size),
        "within_distance_summary": _summary(within),
        "across_distance_summary": _summary(across),
        "scoreable_homogeneity": homogeneity,
        "metric": metric.to_dict(),
    }


#: Accepted values for the feature-selection declaration. `None` is the
#: UNMASKED incumbent (what the committed v1 lock was fitted under); "R5" is the
#: frozen rule. The CLI makes the choice REQUIRED — see `main` — so a re-lock
#: can never acquire or lose the mask by forgetting a flag.
FEATURE_SELECTION_CHOICES: Tuple[str, ...] = ("none", FEATURE_SELECTION_RULE_ID)


def _resolve_feature_selection(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).lower() == "none":
        return None
    if str(value) != FEATURE_SELECTION_RULE_ID:
        raise CalibrationRefusal(
            f"unknown feature selection {value!r}; expected one of "
            f"{list(FEATURE_SELECTION_CHOICES)}"
        )
    return FEATURE_SELECTION_RULE_ID


def build_cohort_calibration(
    observations: Observations,
    cohort: CalibrationCohort,
    feature_selection: Optional[str] = None,
    pool_dir: Path = POOL_PARQUET_DIR,
    dim_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Estimate Σ and select τ for one cohort's calibration population.

    The parity filter is applied FIRST, so for the ADJUDICATING cohort no
    odd-partition fingerprint can reach either the covariance or τ — D9 axis
    (ii)'s device hold-out, enforced by construction.

    BOTH Addendum A candidate metrics are calibrated here, inside this cohort's
    own population, and the predeclared rule selects between them. The selected
    metric's numbers are promoted to the top level (so the gate-(c) lock reads
    them unchanged); both are recorded under `metric_comparison`.

    The last thing this function does before returning a lockable (τ, Σ) is the
    ROOTCAUSE_MEMO § 5(1) scoreable-device homogeneity refusal — the only guard
    here that reads devices outside the calibration population.
    """
    cohort = CalibrationCohort(cohort)
    keep = [
        index
        for index, logical_id in enumerate(observations.logical_ids)
        if is_calibration_partition(logical_id, cohort)
    ]
    if not keep:
        raise CalibrationRefusal(
            f"cohort '{cohort.value}' has no eligible calibration observations"
        )

    vectors = observations.vectors[keep]
    partitions = [observations.partitions[i] for i in keep]
    distinct = sorted(set(partitions))
    if len(distinct) < 2:
        raise CalibrationRefusal(
            f"cohort '{cohort.value}' has {len(distinct)} device(s); at least 2 are "
            "needed to form across-device pairs"
        )

    if not any(partitions.count(p) >= 2 for p in distinct):
        raise CalibrationRefusal(
            f"cohort '{cohort.value}' has no within-device pairs — each device "
            "needs at least two observations for the within-device scatter"
        )

    _refuse_constant_devices(vectors, partitions, cohort)

    # THE PIPELINE ORDER (draft amendment § 3.1(b)-(c)):
    #   screens -> mask -> fit candidates -> Addendum-A selection ->
    #   § 5(1) homogeneity gate -> lock or refuse.
    # The screens read the WHOLE corpus (every device, as `step1_scores.py`
    # does) and the raw pools; the fit that follows still sees only this
    # cohort's calibration partitions.
    # NOT named `selection`: that name is already the Addendum-A METRIC
    # selection below, and the two are different decisions.
    rule = _resolve_feature_selection(feature_selection)
    if rule is None:
        mask = None
        feature_mask = {
            "rule": None,
            "note": (
                "UNMASKED — no feature-selection rule declared. Sigma and tau "
                "are fitted on every emitted dimension, which is what the "
                "committed v1 lock was produced under."
            ),
        }
    else:
        names = (
            list(dim_names) if dim_names is not None
            else list(fingerprint_dim_names())
        )
        columns = sorted({name.rsplit("__", 1)[0] for name in names})
        pool_report = pool_screen(columns, pool_dir)
        feature_mask = build_feature_mask(
            observations.vectors, observations.partitions, pool_report, names
        )
        mask = feature_mask["mask"]

    comparison = {
        name: _calibrate_metric(
            vectors,
            partitions,
            cohort,
            name,
            corpus_vectors=observations.vectors,
            corpus_partitions=observations.partitions,
            scoreable_partitions=SCOREABLE_PARTITIONS[cohort.value],
            mask=mask,
        )
        for name in CANDIDATE_METRICS
    }
    selection = select_metric_by_rule(comparison)
    selected = comparison[selection["selected_metric"]]

    # ROOTCAUSE_MEMO § 5(1): the last gate before a (τ, Σ) becomes lockable.
    _refuse_heterogeneous_scoreable_devices(selected["scoreable_homogeneity"])

    return {
        "cohort": cohort.value,
        "tau": selected["tau"],
        "feature_selection": feature_mask,
        "scoreable_partitions": list(SCOREABLE_PARTITIONS[cohort.value]),
        "scoreable_homogeneity": selected["scoreable_homogeneity"],
        "fpr_target": FPR_TARGET,
        "realised_fpr": selected["realised_fpr"],
        "within_link_rate": selected["within_link_rate"],
        "n_calibration_vectors": int(vectors.shape[0]),
        "calibration_partitions": distinct,
        "n_within_pairs": selected["n_within_pairs"],
        "n_across_pairs": selected["n_across_pairs"],
        "within_distance_summary": selected["within_distance_summary"],
        "across_distance_summary": selected["across_distance_summary"],
        "metric": selected["metric"],
        "selected_metric": selection["selected_metric"],
        "metric_selection": selection,
        "metric_comparison": comparison,
    }


# ===========================================================================
# Artifact
# ===========================================================================

def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_complete_calibration_corpus(observations: "Observations") -> None:
    """The lock requires the WHOLE pre-registered calibration design, exactly.

    `load_observations` refuses anything *outside* the allowlist, which stops
    forbidden data getting in. It does not stop a lock being produced from a
    thin *subset* — one control, one seed — and that subset is not the design
    D2/§ 5.1 pre-register. A τ derived from `control_honest` alone would never
    have seen benign churn; a τ from a single seed would be one draw of the
    run-to-run variance the 5-seed design exists to average over. Both are
    silent under a nonempty-subset check and both change the locked threshold.

    Every (scenario, seed) cell of the design must also be non-empty, so a file
    that merely mentions a seed in one row cannot satisfy the check.
    """
    observed_scenarios = set(observations.scenarios)
    observed_seeds = set(observations.seeds)
    required_scenarios = set(ALLOWED_CALIBRATION_SCENARIOS)
    required_seeds = set(DEV_SEEDS)

    problems: List[str] = []
    if observed_scenarios != required_scenarios:
        missing = sorted(required_scenarios - observed_scenarios)
        if missing:
            problems.append(f"missing calibration scenario(s): {missing}")
    if observed_seeds != required_seeds:
        missing_seeds = sorted(required_seeds - observed_seeds)
        if missing_seeds:
            problems.append(f"missing dev seed(s): {missing_seeds}")

    present_cells = set(zip(observations.scenarios, observations.seeds))
    missing_cells = sorted(
        (scenario, seed)
        for scenario in required_scenarios
        for seed in required_seeds
        if (scenario, seed) not in present_cells
    )
    if missing_cells and not problems:
        problems.append(f"missing (scenario, seed) cell(s): {missing_cells}")

    if problems:
        raise CalibrationRefusal(
            "refusing to lock τ from an incomplete calibration corpus — "
            + "; ".join(problems)
            + f". The pre-registered design is every one of "
            f"{list(ALLOWED_CALIBRATION_SCENARIOS)} x {list(DEV_SEEDS)} "
            f"({len(required_scenarios) * len(required_seeds)} cells); "
            f"observed scenarios {sorted(observed_scenarios)}, "
            f"seeds {sorted(observed_seeds)}."
        )


def calibrate(
    observation_paths: Sequence[Path],
    out_path: Path,
    run_id_manifest_path: Optional[Path] = None,
    git_commit: str = "pending",
    feature_selection: Optional[str] = None,
    pool_dir: Path = POOL_PARQUET_DIR,
    dim_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run both cohort calibrations and write the locked artifact."""
    out_path = Path(out_path)
    if out_path.exists():
        raise CalibrationRefusal(
            f"{out_path} already exists. τ is NEVER re-derived once locked "
            "(v1.10 § 5.1). Move the existing artifact aside deliberately, with a "
            "recorded reason, if a genuine re-calibration is authorised."
        )

    run_id_manifest = (
        load_run_id_manifest(Path(run_id_manifest_path))
        if run_id_manifest_path is not None
        else None
    )
    observations = load_observations(
        [Path(p) for p in observation_paths], run_id_manifest=run_id_manifest
    )
    observed_scenarios = sorted(set(observations.scenarios))
    _require_complete_calibration_corpus(observations)
    verification = verify_scenarios_attack_free(observed_scenarios)

    # Unit → run-id pairs actually witnessed in the corpus. Recorded as pairs
    # (not two parallel lists) so the artifact alone answers "which run produced
    # which unit"; sorted for byte-stable output.
    observed_units = [
        {"source_unit": unit, "run_id": run_id}
        for unit, run_id in sorted(
            {
                (u, r)
                for u, r in zip(observations.source_units, observations.run_ids)
                if u or r
            }
        )
    ]
    observed_run_ids = sorted({r for r in observations.run_ids if r})
    if not observed_run_ids:
        print(
            "WARNING: no calibration run ids resolved — the artifact will not "
            "carry a durable custody record. Pass --run-id-manifest.",
            file=sys.stderr,
        )

    payload: Dict[str, Any] = {
        "_meta": {
            "artifact": "fingerprint_tau_locked",
            "version": "v1",
            "description": (
                "LOCKED H3 fingerprint-match thresholds and Mahalanobis "
                "covariances. Two cohorts: 'validation' (all 20 base partitions) "
                "and 'adjudicating' (even partitions only, D9 axis (ii) device "
                "hold-out)."
            ),
            "authority": [
                "v1.10 § 5.0 D2 (calibration reference), § 5.1 gate (c) "
                "(both τ locked in code before any eval scenario runs)",
                "docs/PHASE7_DESIGN.md 'Threshold τ calibration'",
            ],
            "generated_by": "scripts/calibrate_fp_threshold.py",
            "script_sha256": _script_sha256(),
            "generated_on": date.today().isoformat(),
            "git_commit": git_commit,
            "fpr_target": FPR_TARGET,
            "pair_sampling_seed": PAIR_SAMPLING_SEED,
            "covariance_estimator": COVARIANCE_ESTIMATOR,
            "addendum_status": ADDENDUM_STATUS,
            "candidate_metrics": list(CANDIDATE_METRICS),
            "incumbent_metric": INCUMBENT_METRIC,
            "alternative_metric": ALTERNATIVE_METRIC,
            "simpler_metric": SIMPLER_METRIC,
            "metric_tie_margin": METRIC_TIE_MARGIN,
            "metric_selection_rule": METRIC_SELECTION_RULE,
            "metric_comparison_note": (
                "Addendum A's predeclared comparison. Both candidate metrics are "
                "calibrated inside EACH cohort's own population, each at its own "
                "realised-FPR <= 0.01 τ, and BOTH summaries are recorded here "
                "BEFORE any eval run. The losing metric is recorded, never "
                "deleted. Per-cohort outcome: cohorts[*].metric_selection."
            ),
            "feature_selection_rule": _resolve_feature_selection(feature_selection),
            "feature_selection_rule_description": FEATURE_SELECTION_RULE,
            "feature_selection_parameters": {
                "pool_flag_top1_share_threshold": POOL_FLAG_TOP1_SHARE_THRESHOLD,
                "het_ratio_max": HET_RATIO_MAX,
                "lattice_max_distinct_levels": LATTICE_MAX_DISTINCT_LEVELS,
                "lattice_min_relative_spread": LATTICE_MIN_RELATIVE_SPREAD,
                "min_surviving_dims": MIN_SURVIVING_DIMS,
            },
            "feature_selection_note": (
                "The surviving-dim mask is applied at the METRIC level: the "
                "180-dim emission contract and its features_sha256 are "
                "UNCHANGED, and only this calibration artifact moves "
                "(EDA_MEMO § 6.2). Per-cohort mask, per-dim screen outcomes and "
                "the R4 ablation are at cohorts[*].feature_selection."
            ),
            "pool_screen_dir": (
                str(pool_dir) if _resolve_feature_selection(feature_selection)
                else None
            ),
            "scoreable_partitions": {
                name: list(value) for name, value in SCOREABLE_PARTITIONS.items()
            },
            "scoreable_homogeneity_note": (
                "ROOTCAUSE_MEMO 2026-08-14 § 5(1). Before a (tau, Sigma) is "
                "lockable, every device the cohort will SCORE must have a median "
                "within-device distance <= tau under the selected metric, measured "
                "on this attack-free control corpus. For the adjudicating cohort "
                "those devices are held OUT of the fit (D9 axis (ii)), so no other "
                "guard in the calibrator can see them. Both candidate metrics' "
                "reports are recorded at cohorts[*].metric_comparison[*]."
                "scoreable_homogeneity; the SELECTED metric's report is the one "
                "that gates, and is promoted to cohorts[*].scoreable_homogeneity."
            ),
            "allowed_scenarios": list(ALLOWED_CALIBRATION_SCENARIOS),
            "observed_scenarios": observed_scenarios,
            "dev_seeds": list(DEV_SEEDS),
            "observed_seeds": sorted(set(observations.seeds)),
            "observed_run_ids": observed_run_ids,
            "observed_units": observed_units,
            "run_id_manifest": (
                {
                    "path": str(run_id_manifest_path),
                    "sha256": hashlib.sha256(
                        Path(run_id_manifest_path).read_bytes()
                    ).hexdigest(),
                }
                if run_id_manifest_path is not None
                else None
            ),
            "observation_sources": sorted(set(observations.sources)),
            "n_observations": len(observations),
            "fingerprint_dim": observations.dim,
            "scenario_verification": verification,
            "adjudicating_calibration_partitions": list(
                ADJUDICATING_CALIBRATION_PARTITIONS
            ),
        },
        "cohorts": {
            cohort.value: build_cohort_calibration(
                observations, cohort, feature_selection=feature_selection,
                pool_dir=pool_dir, dim_names=dim_names,
            )
            for cohort in CalibrationCohort
        },
        "artifact_sha256_note": (
            "The SHA-256 of THIS FILE's bytes is pinned as "
            "CALIBRATION_ARTIFACT_SHA256 in flowerfl/fingerprint_registry.py; "
            "recompute it after any reformatting."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False is the BACKSTOP for the whole non-finite class: rather
    # than emitting `Infinity`/`NaN` (which json.dumps does by default and no
    # strict parser accepts), the write fails loudly and the lock is not
    # produced at all. Individual known-infinite quantities are already
    # normalised to null at the point they are recorded (`_finite_or_none`).
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _lock_snippet(payload: Mapping[str, Any], artifact_path: Path) -> str:
    digest = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
    validation = payload["cohorts"][CalibrationCohort.VALIDATION.value]
    adjudicating = payload["cohorts"][CalibrationCohort.ADJUDICATING.value]
    return "\n".join(
        [
            "",
            "=" * 78,
            "GATE (c) LOCK SNIPPET — paste into flowerfl/fingerprint_registry.py,",
            "review it, and COMMIT IT BEFORE ANY H3 EVALUATION SCENARIO RUNS.",
            "=" * 78,
            f"TAU_VALIDATION_ALL_DEVICES: Optional[float] = {validation['tau']!r}",
            f"TAU_ADJUDICATING_EVEN_DEVICES: Optional[float] = {adjudicating['tau']!r}",
            f'CALIBRATION_ARTIFACT_SHA256: Optional[str] = "{digest}"',
            "TAU_LOCK_RECORD = {",
            '    "status": "LOCKED",',
            f'    "locked_on": "{payload["_meta"]["generated_on"]}",',
            f'    "script_sha256": "{payload["_meta"]["script_sha256"]}",',
            f'    "fpr_target": {payload["_meta"]["fpr_target"]!r},',
            f'    "observed_scenarios": {payload["_meta"]["observed_scenarios"]!r},',
            f'    "observed_seeds": {payload["_meta"]["observed_seeds"]!r},',
            f'    "observed_run_ids": {payload["_meta"]["observed_run_ids"]!r},',
            f'    "observed_units": {payload["_meta"]["observed_units"]!r},',
            f'    "selected_metric": {{"validation": "{validation["selected_metric"]}",'
            f' "adjudicating": "{adjudicating["selected_metric"]}"}},',
            f'    "git_commit": "{payload["_meta"]["git_commit"]}",',
            "}",
            "=" * 78,
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the H3 fingerprint threshold τ on the two attack-free "
            "control scenarios ONLY (v1.10 § 5.0 D2). The scenario allowlist is "
            "pre-registration evidence and has NO command-line override."
        )
    )
    parser.add_argument(
        "--observations",
        nargs="+",
        required=True,
        type=Path,
        help="fingerprint observation files (JSONL or JSON array)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "fingerprint_tau_locked_v1.json",
        help="where to write the locked calibration artifact",
    )
    parser.add_argument(
        "--run-id-manifest",
        type=Path,
        default=None,
        help=(
            "JSON map of source_unit -> MLflow run id, stamped into the "
            "artifact so the calibration corpus is traceable to durable custody "
            "records. Strict: every observed unit must appear in it."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        required=True,
        choices=list(FEATURE_SELECTION_CHOICES),
        help=(
            "REQUIRED declaration of the feature-selection rule: 'none' fits "
            "Sigma/tau on every emitted dimension (the committed v1 lock's "
            "posture), 'R5' applies the frozen rule of the corrected-instrument "
            "amendment. There is no default: acquiring or losing the mask by "
            "forgetting a flag would change the instrument silently."
        ),
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=POOL_PARQUET_DIR,
        help="raw partition parquets the a-priori pool screen reads (R5 only)",
    )
    parser.add_argument(
        "--git-commit",
        default="pending",
        help=(
            "the code-and-corpus revision that produced this calibration, "
            "recorded in _meta.git_commit (same semantics as TAU_LOCK_RECORD's)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        payload = calibrate(
            args.observations,
            args.out,
            run_id_manifest_path=args.run_id_manifest,
            git_commit=args.git_commit,
            feature_selection=args.feature_selection,
            pool_dir=args.pool_dir,
        )
    except CalibrationRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    for name, cohort in payload["cohorts"].items():
        print(
            f"[{name}] tau={cohort['tau']:.6f} "
            f"realised_fpr={cohort['realised_fpr']:.4f} "
            f"within_link_rate={cohort['within_link_rate']:.4f} "
            f"n_vectors={cohort['n_calibration_vectors']} "
            f"partitions={cohort['calibration_partitions']}"
        )
        for candidate, summary in cohort["metric_comparison"].items():
            marker = "SELECTED" if candidate == cohort["selected_metric"] else "        "
            print(
                f"    {marker} {candidate}: tau={summary['tau']:.6f} "
                f"realised_fpr={summary['realised_fpr']:.4f} "
                f"within_link_rate={summary['within_link_rate']:.4f} "
                f"shrinkage={summary['shrinkage']:.4f} "
                f"pairs(within/across)={summary['n_within_pairs']}/"
                f"{summary['n_across_pairs']}"
            )
        print(f"    rule outcome: {cohort['metric_selection']['reason']}")
        homogeneity = cohort["scoreable_homogeneity"]
        print(
            f"    scoreable homogeneity (§ 5(1)): PASS — "
            f"{homogeneity['n_devices_checked']} scored device(s) checked, worst "
            f"within-device median {homogeneity['max_scoreable_median']:.6f} vs "
            f"tau {homogeneity['tau']:.6f} "
            f"(calibration worst {homogeneity['max_calibration_median']:.6f})"
        )
    print(f"\nwritten: {args.out}")
    print(_lock_snippet(payload, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

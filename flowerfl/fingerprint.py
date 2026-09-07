"""The 180-dimensional transmission-fingerprint construct (H3).

Design authority
----------------
* The H3 workflow in `docs/reproduction/experiments.md` uses the **D8 proxy
  construct**: the static
  protocol/traffic-feature distribution used as a *software-only* identity
  proxy. A synthetic per-logical-device timing model is an optional,
  separately pre-registered robustness arm and is **not** in scope.
* § 5.1 — the 180-dim fingerprint, Mahalanobis matching and the EMA registry
  are unchanged from base spec § 6.3; the 45-feature set (G4) "must be verified
  recoverable/reproducible from the current parquet, including the 14-column
  transmission-timing subset, as the top silent-failure risk."
* `docs/harness/architecture.md` — 45 protocol features × 4 statistical moments
  (mean, std, skew, kurtosis) = 180 dimensions, computed CLIENT-SIDE over the
  client's full local data and transported in `FitRes.metrics["fingerprint"]`.

Reused seed
-----------
The moment computation is a port of `compute_client_signature` in
`scripts/fingerprint_feasibility.py` (the Phase-0 research script) — the same
per-column ``mean/std/skew/kurtosis`` in the same order.

Relationship to the Phase-0 / scipy reference, stated precisely
---------------------------------------------------------------
The two agree to floating-point tolerance on well-conditioned, fully-finite
columns (asserted to 1e-10 in `tests/test_fingerprint_features.py`). They are
**not** identical in general, and the differences are deliberate:

1. **Shape moments are computed on the standardised column.** Skewness and
   excess kurtosis are scale-invariant, so ``skew(x) == skew((x-mu)/sigma)``
   exactly; computing them on the standardised copy avoids the float64
   overflow that `tcp.options` (~1e118) and `tcp.payload` (large enough that
   ``sum(x**2)`` overflows) provoke inside ``numpy.std`` / ``scipy.stats.skew``,
   which then return ``inf``/``nan`` for **every** partition.
2. **NaN handling matches ``nan_policy="omit"``; INFINITIES DO NOT.**
   ``scipy``'s ``nan_policy="omit"`` omits NaN but does **not** omit ``±inf`` —
   an infinity propagates and poisons the moment. This module omits NaN (and
   records how many, per feature) and **REFUSES** on any infinite input value
   rather than silently computing a moment from the remaining subset, which
   would fabricate a number the data does not support. Measured on the live
   `data/edge_full_20` parquet: **zero** NaN and **zero** inf across partitions
   0/1/9/19/20, so the refusal costs nothing on real data and exists to make a
   future data defect loud.
3. **Degenerate moments are pinned to 0.0.** A constant column has no shape
   information (``std == 0`` ⇒ skew/kurtosis undefined); the convention is 0.0.

Every substitution and omission is recorded in `SanitizationReport` with a
per-feature count and kind — see that class for the exhaustive list of what is
counted. `compute_fingerprint` always returns a finite vector, a hard
requirement of the Mahalanobis matcher.

Nothing in this module reads sealed material, touches the signal-log schema,
or performs enforcement; it is a pure construct library.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The four statistical moments, in the order they occupy each feature's slot.
FINGERPRINT_MOMENTS: Tuple[str, ...] = ("mean", "std", "skew", "kurtosis")
MOMENTS_PER_FEATURE: int = len(FINGERPRINT_MOMENTS)
#: 45 protocol features × 4 moments (PHASE7_DESIGN "Fingerprint vector").
FINGERPRINT_NUM_FEATURES: int = 45
FINGERPRINT_DIM: int = FINGERPRINT_NUM_FEATURES * MOMENTS_PER_FEATURE

#: Locked, hash-stamped feature artifact (H3 execution plan Step 1).
FEATURE_SPEC_FILENAME = "fingerprint_features_v1.json"


class FeatureSpecError(ValueError):
    """Raised when the locked feature set is missing, drifted, or unusable."""


def feature_spec_path() -> Path:
    """Canonical on-disk location of the locked feature artifact."""
    return PROJECT_ROOT / "data" / FEATURE_SPEC_FILENAME


def hash_feature_list(features: Sequence[str]) -> str:
    """SHA-256 of a feature list under the artifact's stamped hashing rule.

    Rule (also recorded in the artifact's ``hashes._rule``): newline-join the
    names in order, append a trailing newline, hash the UTF-8 bytes.
    """
    payload = "\n".join(str(f) for f in features) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    """The locked 45-feature set the 180-dim fingerprint is built from."""

    version: str
    features: Tuple[str, ...]
    timing_features: Tuple[str, ...]
    moments: Tuple[str, ...]
    dim: int
    features_sha256: str
    timing_subset_sha256: str
    source_features_json_sha256: str
    source: str

    def labels(self) -> Tuple[str, ...]:
        """The 180 dimension labels, ``{feature}__{moment}``, in vector order."""
        return tuple(
            f"{col}__{moment}" for col in self.features for moment in self.moments
        )

    def moment_index(self, feature: str, moment: str) -> int:
        """Index of one (feature, moment) slot inside the 180-dim vector."""
        if feature not in self.features:
            raise FeatureSpecError(f"unknown feature: {feature!r}")
        if moment not in self.moments:
            raise FeatureSpecError(f"unknown moment: {moment!r}")
        return self.features.index(feature) * len(self.moments) + self.moments.index(
            moment
        )

    def timing_indices(self) -> Tuple[int, ...]:
        """Vector indices spanned by the transmission-timing subset (D8 anchor)."""
        return tuple(
            self.moment_index(col, moment)
            for col in self.timing_features
            for moment in self.moments
        )


def load_feature_spec(path: Optional[Path] = None, verify: bool = True) -> FeatureSpec:
    """Load (and by default verify) the locked feature artifact.

    Verification is not decoration: an un-noticed change to the feature list
    silently changes the meaning of every fingerprint and of the locked τ. Any
    mismatch is a hard error.
    """
    spec_path = Path(path) if path is not None else feature_spec_path()
    if not spec_path.exists():
        raise FeatureSpecError(f"locked feature artifact not found: {spec_path}")
    try:
        payload: Dict[str, Any] = json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise FeatureSpecError(f"feature artifact is not valid JSON: {spec_path}") from exc

    try:
        features = tuple(str(f) for f in payload["features"])
        moments = tuple(str(m) for m in payload["moments"])
        subset = payload["transmission_timing_subset"]
        timing = tuple(str(f) for f in subset["features"])
        hashes = payload["hashes"]
        dim = int(payload["fingerprint_dim"])
        meta = payload["_meta"]
    except (KeyError, TypeError) as exc:
        raise FeatureSpecError(f"feature artifact is malformed: {spec_path}") from exc

    if len(set(features)) != len(features):
        raise FeatureSpecError("feature artifact contains duplicate feature names")
    if moments != FINGERPRINT_MOMENTS:
        raise FeatureSpecError(
            f"moment order drifted: {moments!r} != {FINGERPRINT_MOMENTS!r}"
        )
    if int(payload.get("num_features", len(features))) != len(features):
        raise FeatureSpecError("num_features disagrees with the feature list length")
    if dim != len(features) * len(moments):
        raise FeatureSpecError(
            f"fingerprint dim {dim} != {len(features)} features x {len(moments)} moments"
        )
    if int(subset.get("num_features", len(timing))) != len(timing):
        raise FeatureSpecError("timing subset num_features disagrees with its list")
    missing_timing = [c for c in timing if c not in features]
    if missing_timing:
        raise FeatureSpecError(
            f"transmission-timing columns are not in the feature set: {missing_timing}"
        )

    if verify:
        actual = hash_feature_list(features)
        if actual != hashes.get("features_sha256"):
            raise FeatureSpecError(
                "feature list sha256 mismatch — the locked 45-feature set has drifted "
                f"(stamped {hashes.get('features_sha256')}, computed {actual})"
            )
        actual_timing = hash_feature_list(timing)
        if actual_timing != hashes.get("timing_subset_sha256"):
            raise FeatureSpecError(
                "transmission-timing subset sha256 mismatch "
                f"(stamped {hashes.get('timing_subset_sha256')}, computed {actual_timing})"
            )

    return FeatureSpec(
        version=str(meta.get("version", "")),
        features=features,
        timing_features=timing,
        moments=moments,
        dim=dim,
        features_sha256=str(hashes.get("features_sha256", "")),
        timing_subset_sha256=str(hashes.get("timing_subset_sha256", "")),
        source_features_json_sha256=str(hashes.get("source_features_json_sha256", "")),
        source=str(meta.get("source", "")),
    )


# ---------------------------------------------------------------------------
# The construct itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SanitizationReport:
    """Exhaustive account of every value this module omitted or substituted.

    Four independent channels, each keyed so a reviewer can see *which* feature
    and *how many* values were involved:

    ``omitted_nan`` — ``{feature: count}`` of NaN cells dropped before the
        moments were computed. This matches ``scipy``'s ``nan_policy="omit"``.
        Dropping cells changes the effective sample size, so it is counted, not
        merely flagged.
    ``all_nan`` — features whose every cell was NaN, so no moment exists at all.
        All four moments are pinned to 0.0.
    ``degenerate`` — ``(feature, moment)`` pairs undefined because the column is
        constant (``std == 0`` ⇒ skew/kurtosis undefined); pinned to 0.0 by
        convention.
    ``sanitized`` — ``(feature, moment)`` pairs that came out non-finite for any
        *other* reason and were pinned to 0.0. On real data this must be empty;
        a non-empty list is a defect signal, asserted as such in the tests.

    Infinities are NOT a channel here: an infinite input value is REFUSED
    (`FeatureSpecError`), never omitted — see the module docstring.
    """

    degenerate: Tuple[Tuple[str, str], ...] = ()
    sanitized: Tuple[Tuple[str, str], ...] = ()
    omitted_nan: Mapping[str, int] = field(default_factory=dict)
    all_nan: Tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """True when nothing at all was omitted, pinned, or substituted."""
        return (
            not self.degenerate
            and not self.sanitized
            and not self.omitted_nan
            and not self.all_nan
        )

    @property
    def total_omitted_cells(self) -> int:
        return int(sum(self.omitted_nan.values()))


def _column_values(df, column: str) -> np.ndarray:
    try:
        series = df[column]
    except KeyError as exc:
        raise FeatureSpecError(f"missing feature column in frame: {column!r}") from exc
    values = np.asarray(series.to_numpy(), dtype=np.float64)
    if values.size == 0:
        raise FeatureSpecError(f"empty column: {column!r} (no rows to fingerprint)")
    return values


def _column_moments(values: np.ndarray, column: str) -> Tuple[List[float], bool, int]:
    """(moments, degenerate flag, NaN cells omitted) for one column.

    Every moment is computed on a **scale-normalised copy** ``z = x / max|x|``
    and rescaled analytically afterwards. This is exact algebra, not an
    approximation: ``mean(x) = s·mean(z)``, ``std(x) = s·std(z)``, and the two
    shape moments are scale-invariant so they are read straight off ``z``.

    It matters on real data. Edge-IIoT's `tcp.payload` column carries values
    large enough that ``sum(x²)`` overflows float64, so the naive
    ``np.std``/``scipy.stats.skew`` path returns ``inf``/``nan`` for **every**
    partition — three of that feature's four moments would be silently dead.
    `tcp.options` (~1e118) overflows the same way inside ``scipy.stats.skew``.
    Normalising first keeps every intermediate bounded by 1.

    NaN cells are omitted (and counted). An INFINITE cell is refused: unlike a
    NaN it is not "missing", it is a value the moment cannot represent, and
    computing the moment from the surviving cells would fabricate a number the
    data does not support. ``scipy``'s ``nan_policy="omit"`` does not omit
    infinities either — it propagates them — so refusing is the honest reading.
    """
    infinite = np.isinf(values)
    n_infinite = int(infinite.sum())
    if n_infinite:
        raise FeatureSpecError(
            f"feature column {column!r} contains {n_infinite} infinite value(s); "
            "refusing to fingerprint. An infinity is not missing data — computing "
            "the moment from the remaining cells would fabricate a value. Fix the "
            "upstream data or add an explicit, pre-registered policy."
        )

    nan_mask = np.isnan(values)
    n_omitted = int(nan_mask.sum())
    if n_omitted == values.size:
        return [float("nan")] * MOMENTS_PER_FEATURE, False, n_omitted
    clean = values[~nan_mask] if n_omitted else values

    scale = float(np.max(np.abs(clean)))
    if not np.isfinite(scale) or scale == 0.0:
        # All-zero column (or an unusable scale): constant, no shape information.
        mean = 0.0 if scale == 0.0 else float("nan")
        std = 0.0 if scale == 0.0 else float("nan")
        return [mean, std, 0.0, 0.0], True, n_omitted

    z = clean / scale
    z_mean = float(np.mean(z))
    centred = z - z_mean
    m2 = float(np.mean(centred**2))

    mean = scale * z_mean
    if not np.isfinite(m2) or m2 <= 0.0:
        # Constant non-zero column: std == 0, shape moments undefined.
        return [mean, 0.0, 0.0, 0.0], True, n_omitted

    z_std = float(np.sqrt(m2))
    std = scale * z_std
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    skewness = m3 / (m2**1.5)
    excess_kurtosis = m4 / (m2**2) - 3.0
    return [mean, std, skewness, excess_kurtosis], False, n_omitted


def compute_fingerprint_with_report(
    df,
    spec_or_features,
) -> Tuple[np.ndarray, SanitizationReport]:
    """Compute the fingerprint vector and report every substitution made.

    Args:
        df: A pandas DataFrame holding (at least) the locked feature columns.
        spec_or_features: A `FeatureSpec` or an explicit ordered feature list.

    Returns:
        ``(vector, report)`` — ``vector`` is float64, ``len(features) * 4``
        long, and always finite.
    """
    if isinstance(spec_or_features, FeatureSpec):
        features: Sequence[str] = spec_or_features.features
    else:
        features = tuple(str(f) for f in spec_or_features)
    if not features:
        raise FeatureSpecError("empty feature list — nothing to fingerprint")

    stats: List[float] = []
    degenerate: List[Tuple[str, str]] = []
    sanitized: List[Tuple[str, str]] = []
    omitted_nan: Dict[str, int] = {}
    all_nan: List[str] = []

    for column in features:
        values = _column_values(df, column)
        moments, is_degenerate, n_omitted = _column_moments(values, column)
        if n_omitted:
            omitted_nan[column] = n_omitted
            if n_omitted == values.size:
                all_nan.append(column)
        for moment_name, value in zip(FINGERPRINT_MOMENTS, moments):
            if not np.isfinite(value):
                sanitized.append((column, moment_name))
                value = 0.0
            elif is_degenerate and moment_name in ("skew", "kurtosis"):
                degenerate.append((column, moment_name))
            stats.append(float(value))

    if omitted_nan:
        logger.warning(
            "[Fingerprint] omitted %d NaN cell(s) across %d feature(s) "
            "(nan_policy='omit' semantics); per-feature counts: %s",
            sum(omitted_nan.values()),
            len(omitted_nan),
            dict(list(omitted_nan.items())[:8]),
        )
    if sanitized:
        logger.warning(
            "[Fingerprint] %d non-finite moment(s) pinned to 0.0: %s",
            len(sanitized),
            sanitized[:8],
        )

    vector = np.asarray(stats, dtype=np.float64)
    if not np.all(np.isfinite(vector)):  # pragma: no cover - defensive
        raise FeatureSpecError("fingerprint vector is non-finite after sanitisation")
    return vector, SanitizationReport(
        degenerate=tuple(degenerate),
        sanitized=tuple(sanitized),
        omitted_nan=dict(omitted_nan),
        all_nan=tuple(all_nan),
    )


def compute_fingerprint(df, spec_or_features) -> np.ndarray:
    """The 180-dim fingerprint vector (see `compute_fingerprint_with_report`)."""
    vector, _ = compute_fingerprint_with_report(df, spec_or_features)
    return vector


# ---------------------------------------------------------------------------
# FitRes.metrics transport
# ---------------------------------------------------------------------------

def encode_fingerprint(vector: np.ndarray) -> str:
    """JSON-encode a fingerprint for `FitRes.metrics["fingerprint"]`.

    Python's float repr round-trips exactly, so encode→decode is bit-identical
    (asserted in tests) — the registry must not see a perturbed vector.
    """
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise FeatureSpecError(f"fingerprint must be 1-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FeatureSpecError("refusing to encode a non-finite fingerprint")
    return json.dumps([float(x) for x in array], separators=(",", ":"))


def decode_fingerprint(
    payload: str, expected_dim: Optional[int] = None
) -> np.ndarray:
    """Decode a `FitRes.metrics["fingerprint"]` payload back to float64."""
    if payload is None or payload == "":
        raise FeatureSpecError("empty fingerprint payload")
    try:
        values = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FeatureSpecError("fingerprint payload is not valid JSON") from exc
    if not isinstance(values, list):
        raise FeatureSpecError("fingerprint payload is not a JSON array")
    array = np.asarray(values, dtype=np.float64)
    if expected_dim is not None and array.shape != (expected_dim,):
        raise FeatureSpecError(
            f"fingerprint dim mismatch: expected {expected_dim}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FeatureSpecError("decoded fingerprint is non-finite")
    return array

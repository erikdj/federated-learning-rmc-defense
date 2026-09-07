"""Client-side fingerprint EMISSION — the pool, the per-round draw, the vector.

Design authority
----------------
`docs/reproduction/experiments.md` describes the public H3 workflow. This
module implements the historical emission contract, and nothing else:

* **§ 4.1 — when, and over what.** Each round, *before* training, the client
  computes the 180-dim vector over an **independent bootstrap draw of
  ``M_FP = 100,000`` rows, WITH REPLACEMENT**, from a fixed raw pool of its own
  training rows.
* **§ 4.2 — the pool.** The device's training-split rows, in the locked
  45-feature order, as **raw float64** — read *before* the label drop, the
  float32 cast, ``nan_to_num`` and Z-score normalisation. Bounded at
  ``P = min(n_train, 100_000)``, taken as the **first P** of the existing
  seed-42 train permutation. Keyed on ``partition_id`` **only**.
* **§ 4.3 — determinism, arm invariance, RNG isolation.** The emission is a pure
  function of ``(partition_id, base_seed, server_round)``: no model, no server
  state, no defense config, no enforcement outcome. The draw uses its **own**
  ``np.random.Generator``, domain-separated from the training stream via a
  ``SeedSequence`` on ``[derive_seed(...), FP_STREAM_ID]``.
* **§ 4.4 — transport.** ``encode_fingerprint`` into
  ``FitRes.metrics["fingerprint"]``.

Why the pool must be RAW float64 (§ 2.3, BUILD-BLOCKING)
-------------------------------------------------------
The fit-time training tensor is **not** a valid fingerprint source. Two
independent defects:

1. It is float32, and ``tcp.payload`` reaches ``5.859e239`` on partition 3 —
   past float32's ``3.4e38`` ceiling — so the production
   ``astype(float32) -> nan_to_num`` silently zeroes **52,331 cells** on that
   partition alone. Eight of the 180 dimensions would be computed from a column
   whose largest ~52,000 values had been replaced by zeros: a different
   construct from the one ``data/fingerprint_features_v1.json`` defines.
2. It is Z-scored **per client on that client's own data**, so ``mean ~ 0`` and
   ``std ~ 1`` for *every* device by construction — 90 of the 180 dimensions
   would carry almost no identity signal.

Why per-round redrawing at all (§ 2.2, § 3.3)
---------------------------------------------
The client loads its partition once per run and trains on all of it every round,
and all four moments are permutation-invariant — so a fingerprint over "the rows
the client trained on this round" is **byte-identical every round**, i.e. the
same degeneracy as caching, reached by a longer route. Under any cached contract
a re-entrant presents a byte-identical vector to its own parent entry, so
re-link recall would be 1.0 as an *arithmetic identity* and the pooled
within-device covariance the matcher needs would be undefined. The bootstrap
draw is what makes the within-device distance distribution exist.

RNG isolation is the highest-risk detail in the whole contract
--------------------------------------------------------------
A single draw from the **global** NumPy/torch RNG would shift every subsequent
dropout mask and DataLoader shuffle and silently break byte-reproducibility
against every sealed H1/H2 run. Nothing in this module ever touches a global
RNG; ``tests/test_client_fingerprint_emission.py`` carries the dedicated
regression the contract demands by name.

This module reads no sealed material, no model, and no server state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from flowerfl.fingerprint import (
    FINGERPRINT_DIM,
    FeatureSpec,
    FeatureSpecError,
    compute_fingerprint,
    encode_fingerprint,
    load_feature_spec,
)
from flowerfl.seeding import derive_seed

logger = logging.getLogger(__name__)

#: Per-round bootstrap draw size (§ 3.5, § 4.1). LOCKED before the acceptance
#: smoke and selected on a **compute-budget** rule — "the largest round value
#: whose fingerprint computation stays within ~10 % of unit wall-clock" — never
#: on a measured H3 quantity (§ 7.2). Changing it changes the instrument.
M_FP: int = 100_000

#: Pool bound P = min(n_train, POOL_MAX_ROWS) (§ 4.2). ~36 MB/client at float64.
POOL_MAX_ROWS: int = 100_000

#: Domain-separation constant for the fingerprint RNG stream (§ 4.3(2)).
#: ``int.from_bytes(b"FPv1", "big")`` — a fixed, arbitrary, never-reused tag so
#: the fingerprint stream cannot become correlated with the training stream if
#: the training RNG is ever re-plumbed.
FP_STREAM_ID: int = 0x46507631

#: The metrics key the FP plugin already reads (§ 4.4). Deliberately duplicated
#: rather than imported: the client must not import a server-side plugin. A test
#: asserts the two constants agree.
FINGERPRINT_METRIC_KEY: str = "fingerprint"


class FingerprintEmissionError(RuntimeError):
    """Raised when a fingerprint cannot be built or computed.

    Gate (e) requires ``FitRes.metrics["fingerprint"]`` for **100 %** of
    participating client-rounds, so every failure here is loud: the unit fails
    rather than quietly emitting nothing.
    """


@lru_cache(maxsize=1)
def _locked_spec() -> FeatureSpec:
    """The hash-verified locked 45-feature spec, loaded once per process.

    Verification is not decoration — an un-noticed change to the feature list
    silently changes the meaning of every fingerprint and of the locked tau.
    """
    return load_feature_spec(verify=True)


@dataclass(frozen=True, eq=False)
class FingerprintPool:
    """A device's fixed raw fingerprint pool (§ 4.2).

    Attributes:
        partition_id: The device this pool belongs to. It is the ONLY identity
            input to the pool — no base_seed, by design.
        features: The locked 45 feature names, in vector order. ``values``
            columns are aligned to this tuple.
        values: ``(P, 45)`` **raw float64**, read-only.
        n_train: The device's full training-split size *before* the ``P`` bound,
            kept for provenance (P < n_train means the pool was truncated).
    """

    partition_id: int
    features: Tuple[str, ...]
    values: np.ndarray
    n_train: int

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def size(self) -> int:
        """Number of rows in the pool (``P``)."""
        return len(self)


def build_fingerprint_pool(
    df,
    *,
    train_indices: Sequence[int],
    partition_id: int,
    spec: Optional[FeatureSpec] = None,
    max_rows: int = POOL_MAX_ROWS,
) -> FingerprintPool:
    """Build a device's fixed raw fingerprint pool (§ 4.2).

    Args:
        df: The partition frame **as read** — raw dtypes, label column still
            present, before any cast/normalisation. Rows are addressed
            POSITIONALLY (``iloc``), matching how ``TensorDataset`` indexes the
            same frame downstream.
        train_indices: The existing seed-42 train permutation for this device
            (``random_split(...).indices``). The pool takes its **first P**
            entries — deterministic, no new RNG, no extra parquet pass.
        partition_id: The device id. The pool's only identity input.
        spec: Feature spec override (tests); defaults to the locked artifact.
        max_rows: The ``P`` bound.

    Raises:
        FingerprintEmissionError: on an empty train split, a frame missing any
            locked feature column, or a column that is not numeric.
    """
    resolved = spec if spec is not None else _locked_spec()
    features = tuple(resolved.features)

    if train_indices is None or len(train_indices) == 0:
        raise FingerprintEmissionError(
            f"client {partition_id}: cannot build a fingerprint pool from an "
            "empty training split"
        )
    if max_rows <= 0:
        raise FingerprintEmissionError(
            f"client {partition_id}: pool bound must be positive, got {max_rows}"
        )

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise FingerprintEmissionError(
            f"client {partition_id}: partition frame is missing "
            f"{len(missing)} locked fingerprint feature column(s): {missing[:8]}"
        )

    n_train = int(len(train_indices))
    bound = min(n_train, int(max_rows))
    rows = np.asarray(train_indices, dtype=np.int64)[:bound]

    try:
        # Columns first (drops the label and anything else), then the rows, then
        # ONE float64 materialisation. Raw: no cast to float32, no nan_to_num,
        # no Z-score — see the module docstring, § 2.3.
        values = (
            df.loc[:, list(features)]
            .iloc[rows]
            .to_numpy(dtype=np.float64, copy=True)
        )
    except (ValueError, TypeError) as exc:
        raise FingerprintEmissionError(
            f"client {partition_id}: fingerprint feature columns are not "
            f"numerically readable as float64: {exc}"
        ) from exc

    values.setflags(write=False)
    if bound < n_train:
        logger.info(
            "[Fingerprint] client %s: pool bounded to %d of %d training rows "
            "(P = min(n_train, %d))",
            partition_id, bound, n_train, max_rows,
        )
    return FingerprintPool(
        partition_id=int(partition_id),
        features=features,
        values=values,
        n_train=n_train,
    )


def fingerprint_rng(
    *, base_seed: int, partition_id: int, server_round: int
) -> np.random.Generator:
    """A dedicated, domain-separated Generator for one client-round (§ 4.3).

    Seeded through a ``SeedSequence`` on
    ``[derive_seed(base_seed, partition_id, server_round), FP_STREAM_ID]`` —
    the existing training key **plus** the fingerprint domain tag, so the
    two streams cannot become correlated. It is a fresh instance every call and
    NEVER touches the global NumPy or torch RNG.
    """
    seed_sequence = np.random.SeedSequence(
        [int(derive_seed(base_seed, partition_id, server_round)), FP_STREAM_ID]
    )
    return np.random.default_rng(seed_sequence)


def draw_round_indices(
    pool_size: int,
    *,
    base_seed: int,
    partition_id: int,
    server_round: int,
    m_fp: int = M_FP,
) -> np.ndarray:
    """``m_fp`` row indices drawn WITH REPLACEMENT from a pool of ``pool_size``.

    With replacement, and at a fixed **size** rather than a fixed fraction, for
    two reasons the contract derives from the ratified estimator (§ 3.2):

    * the pooled within-device Sigma (Addendum A) is coherent only if the
      per-round noise is homogeneous across devices, and corpora here span
      21k-2M rows — a fixed *fraction* would give the smallest device ~10x the
      noise of the largest;
    * a without-replacement draw at ``m = N`` reproduces the pool exactly and
      returns the degenerate cached contract through the back door (measured:
      partition 8 silently did exactly this at m = 50,000).
    """
    if pool_size <= 0:
        raise FingerprintEmissionError(
            f"client {partition_id}: cannot draw from an empty fingerprint pool"
        )
    if m_fp <= 0:
        raise FingerprintEmissionError(
            f"client {partition_id}: draw size must be positive, got {m_fp}"
        )
    rng = fingerprint_rng(
        base_seed=base_seed, partition_id=partition_id, server_round=server_round
    )
    return rng.integers(0, int(pool_size), size=int(m_fp))


def _resolve_partition_id(pool: FingerprintPool, partition_id: Optional[int]) -> int:
    if partition_id is None:
        return int(pool.partition_id)
    if int(partition_id) != int(pool.partition_id):
        raise FingerprintEmissionError(
            "fingerprint pool partition mismatch: pool was built for partition "
            f"{pool.partition_id} but emission was keyed to {partition_id}. The "
            "pool is partition-keyed (contract § 4.2); a mis-keyed pool would "
            "fingerprint the wrong device."
        )
    return int(partition_id)


def compute_round_fingerprint(
    pool: FingerprintPool,
    *,
    base_seed: int,
    server_round: int,
    partition_id: Optional[int] = None,
    m_fp: int = M_FP,
) -> np.ndarray:
    """The 180-dim fingerprint for one client-round (§ 4.1).

    A pure function of ``(pool, base_seed, partition_id, server_round)`` — it
    reads no model, no server state, no defense config and no enforcement
    outcome, which is what makes the emission **arm-invariant by construction**.

    Raises:
        FingerprintEmissionError: on any failure at all. Gate (e) requires 100 %
            emission, so a fingerprint that cannot be computed fails the unit.
    """
    pid = _resolve_partition_id(pool, partition_id)
    indices = draw_round_indices(
        len(pool),
        base_seed=base_seed,
        partition_id=pid,
        server_round=server_round,
        m_fp=m_fp,
    )
    # ``copy=False`` keeps this a view over the freshly drawn block; the block
    # itself is a new array, so the read-only pool is never mutated.
    frame = pd.DataFrame(
        pool.values[indices], columns=list(pool.features), copy=False
    )
    try:
        vector = compute_fingerprint(frame, pool.features)
    except (FeatureSpecError, ValueError, TypeError) as exc:
        logger.error(
            "[Fingerprint] client %s round %s: FAILED to compute the fingerprint "
            "over %d drawn rows — %s",
            pid, server_round, len(indices), exc,
        )
        raise FingerprintEmissionError(
            f"client {pid} round {server_round}: fingerprint computation failed: {exc}"
        ) from exc

    if vector.shape != (FINGERPRINT_DIM,):  # pragma: no cover - defensive
        raise FingerprintEmissionError(
            f"client {pid} round {server_round}: fingerprint dim "
            f"{vector.shape} != ({FINGERPRINT_DIM},)"
        )
    return vector


def compute_round_fingerprint_payload(
    pool: FingerprintPool,
    *,
    base_seed: int,
    server_round: int,
    partition_id: Optional[int] = None,
    m_fp: int = M_FP,
) -> str:
    """The encoded ``FitRes.metrics["fingerprint"]`` payload for one round (§ 4.4)."""
    vector = compute_round_fingerprint(
        pool,
        base_seed=base_seed,
        server_round=server_round,
        partition_id=partition_id,
        m_fp=m_fp,
    )
    try:
        return encode_fingerprint(vector)
    except FeatureSpecError as exc:
        raise FingerprintEmissionError(
            f"client {pool.partition_id} round {server_round}: "
            f"fingerprint encoding failed: {exc}"
        ) from exc

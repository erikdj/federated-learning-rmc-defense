"""Client-side fingerprint EMISSION — the pool, the per-round draw, the vector.

Design authority
----------------
The public workflow is described in `docs/reproduction/experiments.md`.
Historical protocol references: § 2.3 (raw float64 pool — the
BUILD-BLOCKING finding), § 4.1 (per-round bootstrap draw at M_FP), § 4.2 (the
bounded, seed-invariant, partition-keyed pool), § 4.3 (determinism, arm
invariance and **RNG isolation** — "the single highest-risk implementation
detail in this contract"), § 4.4 (transport).

These are the module-level tests. The client-wiring half (off-by-default,
arm invariance through `FlowerClient.fit`, and the H1/H2-protecting RNG
isolation regression) lives in `tests/test_client_fingerprint_emission.py`.

No FL training, no parquet from `data/`, no cloud: every fixture here is a small
synthetic frame carrying the locked 45 feature columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowerfl.fingerprint import (
    FINGERPRINT_DIM,
    decode_fingerprint,
    load_feature_spec,
)
from flowerfl.fingerprint_emission import (
    FINGERPRINT_METRIC_KEY,
    FP_STREAM_ID,
    M_FP,
    POOL_MAX_ROWS,
    FingerprintEmissionError,
    FingerprintPool,
    build_fingerprint_pool,
    compute_round_fingerprint,
    compute_round_fingerprint_payload,
    draw_round_indices,
    fingerprint_rng,
)
from flowerfl.seeding import derive_seed

SPEC = load_feature_spec(verify=True)
FEATURES = SPEC.features


def _frame(n_rows: int, *, seed: int = 0, extra_cols: bool = True) -> pd.DataFrame:
    """A synthetic partition frame: the locked 45 features + noise columns."""
    rng = np.random.default_rng(seed)
    cols = {
        name: rng.normal(loc=float(i), scale=1.0 + i, size=n_rows)
        for i, name in enumerate(FEATURES)
    }
    if extra_cols:
        # Real partitions carry the label column (and it must never leak into
        # the pool), so keep one here.
        cols["Attack_label"] = rng.integers(0, 2, size=n_rows)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

def test_m_fp_is_the_locked_contract_value():
    # § 3.5 / § 7.2: M_FP is locked BEFORE the acceptance smoke and selected on a
    # compute-budget rule. A drift here silently changes the instrument.
    assert M_FP == 100_000


def test_pool_bound_is_the_locked_contract_value():
    # § 4.2: P = min(n_train, 100_000).
    assert POOL_MAX_ROWS == 100_000


def test_fp_stream_id_is_a_fixed_domain_separation_constant():
    # § 4.3(2): a fixed constant, not a value that can drift per run.
    assert FP_STREAM_ID == int.from_bytes(b"FPv1", "big")
    assert isinstance(FP_STREAM_ID, int)


def test_metric_key_matches_the_plugin_contract():
    # § 4.4: the key the FP plugin already reads.
    from flowerfl.fingerprint_plugin import FINGERPRINT_METRIC_KEY as PLUGIN_KEY

    assert FINGERPRINT_METRIC_KEY == PLUGIN_KEY == "fingerprint"


# ---------------------------------------------------------------------------
# § 4.2 — the pool
# ---------------------------------------------------------------------------

def test_pool_is_raw_float64_in_the_locked_feature_order():
    df = _frame(50)
    pool = build_fingerprint_pool(df, train_indices=list(range(40)), partition_id=3)

    assert isinstance(pool, FingerprintPool)
    assert pool.features == FEATURES
    assert pool.values.dtype == np.float64
    assert pool.values.shape == (40, len(FEATURES))
    assert pool.partition_id == 3
    # Column j of the pool IS feature j of the locked spec, raw.
    for j, name in enumerate(FEATURES):
        np.testing.assert_array_equal(pool.values[:, j], df[name].to_numpy()[:40])


def test_pool_excludes_the_label_column():
    df = _frame(20)
    pool = build_fingerprint_pool(df, train_indices=list(range(16)), partition_id=0)
    assert "Attack_label" not in pool.features
    assert pool.values.shape[1] == 45


def test_pool_preserves_magnitudes_that_the_float32_path_destroys():
    """§ 2.3, the BUILD-BLOCKING finding.

    ``tcp.payload`` carries values up to 5.859e239 on partition 3; the
    production ``astype(float32) -> nan_to_num`` path zeroes them. A pool built
    from that tensor would compute a DIFFERENT construct from the locked one.
    """
    df = _frame(20)
    df.loc[0, "tcp.payload"] = 5.859e239
    pool = build_fingerprint_pool(df, train_indices=list(range(16)), partition_id=3)

    col = pool.features.index("tcp.payload")
    assert pool.values[0, col] == 5.859e239
    # The production path would have destroyed it — this is what we must avoid.
    with np.errstate(over="ignore"):
        destroyed = np.nan_to_num(np.float32(5.859e239), posinf=0.0, neginf=0.0)
    assert destroyed == 0.0


def test_pool_is_bounded_at_the_contract_maximum():
    df = _frame(60)
    pool = build_fingerprint_pool(
        df, train_indices=list(range(60)), partition_id=0, max_rows=25
    )
    assert len(pool) == 25
    assert pool.n_train == 60


def test_pool_takes_the_first_P_of_the_given_train_permutation():
    """§ 4.2: the FIRST P of the EXISTING seed-42 permutation — no new RNG."""
    df = _frame(60)
    perm = [17, 3, 41, 8, 22, 0, 55, 31]
    pool = build_fingerprint_pool(
        df, train_indices=perm, partition_id=0, max_rows=5
    )
    assert len(pool) == 5
    expected = df.iloc[perm[:5]].loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(pool.values, expected)


def test_pool_keeps_every_row_when_the_train_split_is_smaller_than_the_bound():
    df = _frame(30)
    pool = build_fingerprint_pool(
        df, train_indices=list(range(24)), partition_id=8, max_rows=POOL_MAX_ROWS
    )
    assert len(pool) == 24 == pool.n_train


def test_pool_build_takes_no_seed_argument():
    """§ 4.2 keying: the pool depends on partition_id ONLY, never on base_seed.

    A seed-dependent pool would add a between-run component to the within-device
    scatter that the within-run parent<->re-entrant comparison does not contain,
    inflating Sigma and systematically OVER-LINKING. The strongest structural
    guarantee is that no seed can even be passed.
    """
    import inspect

    params = set(inspect.signature(build_fingerprint_pool).parameters)
    assert not {p for p in params if "seed" in p}, params


def test_pool_values_are_read_only():
    df = _frame(20)
    pool = build_fingerprint_pool(df, train_indices=list(range(16)), partition_id=0)
    with pytest.raises(ValueError):
        pool.values[0, 0] = 1.0


def test_pool_refuses_a_frame_missing_a_locked_feature():
    df = _frame(20).drop(columns=[FEATURES[7]])
    with pytest.raises(FingerprintEmissionError, match=FEATURES[7]):
        build_fingerprint_pool(df, train_indices=list(range(16)), partition_id=0)


def test_pool_refuses_an_empty_train_split():
    df = _frame(20)
    with pytest.raises(FingerprintEmissionError, match="empty"):
        build_fingerprint_pool(df, train_indices=[], partition_id=0)


# ---------------------------------------------------------------------------
# § 4.1 / § 4.3 — the per-round draw
# ---------------------------------------------------------------------------

def test_draw_is_deterministic_for_the_same_key():
    a = draw_round_indices(1000, base_seed=42, partition_id=3, server_round=7, m_fp=250)
    b = draw_round_indices(1000, base_seed=42, partition_id=3, server_round=7, m_fp=250)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (250,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_seed": 43, "partition_id": 3, "server_round": 7},
        {"base_seed": 42, "partition_id": 4, "server_round": 7},
        {"base_seed": 42, "partition_id": 3, "server_round": 8},
    ],
)
def test_draw_changes_with_every_key_component(kwargs):
    base = draw_round_indices(1000, base_seed=42, partition_id=3, server_round=7, m_fp=250)
    other = draw_round_indices(1000, m_fp=250, **kwargs)
    assert not np.array_equal(base, other)


def test_draw_is_with_replacement_so_m_may_exceed_the_pool():
    """§ 3.2 'why with replacement': a fixed-size bootstrap is defined for EVERY
    device regardless of corpus size (partition 8 holds only 21,264 rows) and
    cannot collapse into the degenerate cached contract."""
    idx = draw_round_indices(50, base_seed=42, partition_id=8, server_round=1, m_fp=500)
    assert idx.shape == (500,)
    assert idx.min() >= 0 and idx.max() < 50
    assert len(np.unique(idx)) < 500  # necessarily, by pigeonhole


def test_draw_is_domain_separated_from_the_training_stream():
    """§ 4.3(2): seeded through a SeedSequence on [derive_seed(...), FP_STREAM_ID],
    NOT on the training key alone."""
    key = derive_seed(42, 3, 7)
    naive = np.random.default_rng(key).integers(0, 1000, size=250)
    ours = draw_round_indices(1000, base_seed=42, partition_id=3, server_round=7, m_fp=250)
    assert not np.array_equal(naive, ours)


def test_fingerprint_rng_is_an_isolated_generator():
    rng = fingerprint_rng(base_seed=42, partition_id=3, server_round=7)
    assert isinstance(rng, np.random.Generator)
    assert rng is not np.random.default_rng  # not the global legacy state


def test_drawing_never_consumes_the_global_numpy_or_torch_rng():
    """§ 4.3(1) at module level — the H1/H2-protecting property.

    A single global draw shifts every subsequent dropout mask and shuffle order
    and would silently break byte-reproducibility against every sealed run.
    """
    import torch

    np.random.seed(1234)
    torch.manual_seed(1234)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    draw_round_indices(1000, base_seed=42, partition_id=3, server_round=7, m_fp=500)

    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


# ---------------------------------------------------------------------------
# § 4.1 / § 4.4 — the vector and its transport
# ---------------------------------------------------------------------------

def test_round_fingerprint_is_180_dim_finite_float64():
    pool = build_fingerprint_pool(_frame(400), train_indices=list(range(320)), partition_id=1)
    vec = compute_round_fingerprint(pool, base_seed=42, server_round=1, m_fp=500)
    assert vec.shape == (FINGERPRINT_DIM,) == (180,)
    assert vec.dtype == np.float64
    assert np.all(np.isfinite(vec))


def test_round_fingerprint_is_reproducible_for_the_same_key():
    pool = build_fingerprint_pool(_frame(400), train_indices=list(range(320)), partition_id=1)
    a = compute_round_fingerprint(pool, base_seed=42, server_round=3, m_fp=500)
    b = compute_round_fingerprint(pool, base_seed=42, server_round=3, m_fp=500)
    np.testing.assert_array_equal(a, b)


def test_round_fingerprints_differ_across_rounds():
    """§ 3.3 / § 1.2: the WITHIN-device distribution must not be a point mass at
    zero — that is exactly the degeneracy the cached contract produced."""
    pool = build_fingerprint_pool(_frame(400), train_indices=list(range(320)), partition_id=1)
    vectors = [
        compute_round_fingerprint(pool, base_seed=42, server_round=r, m_fp=500)
        for r in range(1, 6)
    ]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert not np.array_equal(vectors[i], vectors[j])


def test_round_fingerprint_uses_the_pools_partition_id_as_the_draw_key():
    """Two pools with identical CONTENT but different partition ids must draw
    different rows — the key is (partition_id, base_seed, server_round)."""
    df = _frame(400)
    p1 = build_fingerprint_pool(df, train_indices=list(range(320)), partition_id=1)
    p2 = build_fingerprint_pool(df, train_indices=list(range(320)), partition_id=2)
    a = compute_round_fingerprint(p1, base_seed=42, server_round=1, m_fp=500)
    b = compute_round_fingerprint(p2, base_seed=42, server_round=1, m_fp=500)
    assert not np.array_equal(a, b)


def test_round_fingerprint_rejects_a_mismatched_partition_id():
    pool = build_fingerprint_pool(_frame(100), train_indices=list(range(80)), partition_id=1)
    with pytest.raises(FingerprintEmissionError, match="partition"):
        compute_round_fingerprint(
            pool, base_seed=42, partition_id=9, server_round=1, m_fp=100
        )


def test_payload_round_trips_through_the_plugin_decode_path():
    pool = build_fingerprint_pool(_frame(400), train_indices=list(range(320)), partition_id=1)
    vec = compute_round_fingerprint(pool, base_seed=42, server_round=2, m_fp=500)
    payload = compute_round_fingerprint_payload(pool, base_seed=42, server_round=2, m_fp=500)

    assert isinstance(payload, str)
    decoded = decode_fingerprint(payload, expected_dim=FINGERPRINT_DIM)
    np.testing.assert_array_equal(decoded, vec)


def test_computing_the_vector_never_consumes_the_global_rng():
    import torch

    pool = build_fingerprint_pool(_frame(400), train_indices=list(range(320)), partition_id=1)
    np.random.seed(99)
    torch.manual_seed(99)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    compute_round_fingerprint_payload(pool, base_seed=42, server_round=4, m_fp=500)

    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


def test_emission_failure_is_loud_never_silent():
    """Gate (e) demands 100% emission — a fingerprint that cannot be computed
    must raise, never return None or an empty payload."""
    pool = build_fingerprint_pool(_frame(100), train_indices=list(range(80)), partition_id=0)
    broken = FingerprintPool(
        partition_id=0,
        features=pool.features,
        values=np.full((4, 45), np.inf),
        n_train=4,
    )
    with pytest.raises(FingerprintEmissionError):
        compute_round_fingerprint_payload(broken, base_seed=42, server_round=1, m_fp=10)

"""H3 Step 1 — the 45-feature set is LOCKED, hash-stamped, and reproducible.

Design authority: v1.10 § 5.0 D8 (fingerprint construct = proxy path (a), the
static protocol/traffic-feature distribution) + § 5.1 ("the 45-feature set (G4)
must be verified recoverable/reproducible from the current parquet, including
the 14-column transmission-timing subset, as the top silent-failure risk").

These tests are the Step-1 validation gate from
`docs/reproduction/experiments.md`:

    "Test loads the current parquet, computes the 180-dim vector for two
     partitions, asserts dimension = 180, asserts zero NaN/inf, and asserts
     the vector is byte-reproducible across two invocations."

The parquet-backed tests skip when `data/` is not materialised (it is
gitignored); the artifact/hash tests always run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flowerfl.fingerprint import (
    FINGERPRINT_DIM,
    FINGERPRINT_MOMENTS,
    MOMENTS_PER_FEATURE,
    FeatureSpecError,
    compute_fingerprint,
    compute_fingerprint_with_report,
    decode_fingerprint,
    encode_fingerprint,
    feature_spec_path,
    hash_feature_list,
    load_feature_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "data" / "fingerprint_features_v1.json"
SOURCE_FEATURES = PROJECT_ROOT / "data" / "edge_full" / "features.json"
# The H3 scenarios (control_honest / control_benign_churn / S3 / S4) all declare
# dataset `edge_full_20_rmc`, which task.py resolves to data/edge_full_20.
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "edge_full_20"
EVAL_LABEL_COLUMN = "Attack_label"

_PARQUET_PRESENT = (EVAL_DATA_DIR / "client_0.parquet").exists()
_SKIP_PARQUET = pytest.mark.skipif(
    not _PARQUET_PRESENT,
    reason="data/edge_full_20 parquet not materialised (gitignored)",
)


# ---------------------------------------------------------------------------
# The locked artifact
# ---------------------------------------------------------------------------

def test_artifact_exists_at_the_canonical_path():
    assert ARTIFACT.exists(), f"locked feature artifact missing: {ARTIFACT}"
    assert feature_spec_path() == ARTIFACT


def test_artifact_declares_45_features_and_180_dimensions():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["num_features"] == 45
    assert len(payload["features"]) == 45
    assert len(set(payload["features"])) == 45, "duplicate feature name"
    assert payload["moments"] == list(FINGERPRINT_MOMENTS)
    assert len(payload["moments"]) == MOMENTS_PER_FEATURE
    assert payload["fingerprint_dim"] == FINGERPRINT_DIM == 45 * 4 == 180


def test_artifact_hashes_are_self_consistent():
    """The stamped SHA-256s must recompute from the stamped lists."""
    payload = json.loads(ARTIFACT.read_text())
    hashes = payload["hashes"]
    assert hashes["features_sha256"] == hash_feature_list(payload["features"])
    assert hashes["timing_subset_sha256"] == hash_feature_list(
        payload["transmission_timing_subset"]["features"]
    )


def test_artifact_carries_the_14_column_transmission_timing_subset():
    """D8's construct-validity anchor — the transmission-timing columns."""
    payload = json.loads(ARTIFACT.read_text())
    subset = payload["transmission_timing_subset"]
    assert subset["num_features"] == 14
    assert len(subset["features"]) == 14
    assert set(subset["features"]).issubset(set(payload["features"]))


def test_artifact_records_provenance():
    payload = json.loads(ARTIFACT.read_text())
    meta = payload["_meta"]
    for key in ("description", "authority", "source", "locked_at", "version"):
        assert meta.get(key), f"provenance field missing/empty: {key}"


# ---------------------------------------------------------------------------
# load_feature_spec()
# ---------------------------------------------------------------------------

def test_load_feature_spec_returns_a_frozen_verified_spec():
    spec = load_feature_spec()
    assert spec.dim == 180
    assert len(spec.features) == 45
    assert len(spec.timing_features) == 14
    assert isinstance(spec.features, tuple)
    with pytest.raises((AttributeError, TypeError)):
        spec.features = ()  # type: ignore[misc]


def test_load_feature_spec_labels_are_feature_by_moment_in_order():
    spec = load_feature_spec()
    labels = spec.labels()
    assert len(labels) == 180
    assert labels[0] == f"{spec.features[0]}__mean"
    assert labels[3] == f"{spec.features[0]}__kurtosis"
    assert labels[4] == f"{spec.features[1]}__mean"
    assert spec.moment_index(spec.features[1], "std") == 5


def test_load_feature_spec_rejects_a_tampered_feature_list(tmp_path):
    payload = json.loads(ARTIFACT.read_text())
    payload["features"][0] = "tampered.column"
    bad = tmp_path / "fingerprint_features_v1.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(FeatureSpecError, match="sha256"):
        load_feature_spec(bad)


def test_load_feature_spec_rejects_a_wrong_dimension(tmp_path):
    payload = json.loads(ARTIFACT.read_text())
    payload["fingerprint_dim"] = 176
    bad = tmp_path / "fingerprint_features_v1.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(FeatureSpecError, match="dim"):
        load_feature_spec(bad)


def test_load_feature_spec_rejects_timing_columns_outside_the_feature_set(tmp_path):
    payload = json.loads(ARTIFACT.read_text())
    subset = payload["transmission_timing_subset"]
    subset["features"][0] = "not.a.feature"
    subset_hash = hashlib.sha256(
        ("\n".join(subset["features"]) + "\n").encode("utf-8")
    ).hexdigest()
    payload["hashes"]["timing_subset_sha256"] = subset_hash
    bad = tmp_path / "fingerprint_features_v1.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(FeatureSpecError, match="timing"):
        load_feature_spec(bad)


# ---------------------------------------------------------------------------
# Drift against the upstream source (data/edge_full/features.json)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SOURCE_FEATURES.exists(), reason="data/edge_full not materialised")
def test_locked_list_still_matches_the_upstream_features_json():
    """Silent-failure guard: the dataset's own feature list must not have drifted."""
    payload = json.loads(ARTIFACT.read_text())
    raw = SOURCE_FEATURES.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == payload["hashes"]["source_features_json_sha256"]
    ), "data/edge_full/features.json changed since the fingerprint feature set was locked"
    upstream = json.loads(raw)["features"]
    assert upstream == payload["features"], "upstream feature ORDER drifted"


@_SKIP_PARQUET
def test_locked_list_matches_the_evaluation_parquet_column_order():
    """The 45 columns must be recoverable, in order, from the H3 parquet."""
    import pyarrow.parquet as pq

    spec = load_feature_spec()
    columns = list(pq.read_schema(EVAL_DATA_DIR / "client_0.parquet").names)
    assert [c for c in columns if c != EVAL_LABEL_COLUMN] == list(spec.features)


# ---------------------------------------------------------------------------
# The 180-dim vector reproduces from the current parquet (the Step-1 gate)
# ---------------------------------------------------------------------------

@_SKIP_PARQUET
def test_fingerprint_reproduces_from_two_partitions_of_the_current_parquet():
    import pandas as pd

    spec = load_feature_spec()
    vectors = {}
    for partition in (0, 1):
        df = pd.read_parquet(
            EVAL_DATA_DIR / f"client_{partition}.parquet", columns=list(spec.features)
        )
        first = compute_fingerprint(df, spec)
        second = compute_fingerprint(df, spec)

        assert first.shape == (180,)
        assert first.dtype == np.float64
        assert np.all(np.isfinite(first)), "fingerprint contains NaN/inf"
        # Byte-reproducible across invocations.
        assert first.tobytes() == second.tobytes()
        vectors[partition] = first

    assert vectors[0].tobytes() != vectors[1].tobytes(), (
        "two distinct partitions produced an identical fingerprint"
    )


@_SKIP_PARQUET
@pytest.mark.parametrize("partition", [0, 9, 19, 20])
def test_no_moment_is_silently_lost_on_the_real_partitions(partition):
    """Regression gate for the `tcp.payload` float64 variance overflow.

    Before the scale-normalised moment computation, `std(tcp.payload)`
    overflowed to inf on EVERY partition, so three of that feature's four
    moments were dead weight in the 180-dim vector. A non-empty `sanitized`
    list on real data is a defect, not a data property.
    """
    import pandas as pd

    spec = load_feature_spec()
    df = pd.read_parquet(
        EVAL_DATA_DIR / f"client_{partition}.parquet", columns=list(spec.features)
    )
    vector, report = compute_fingerprint_with_report(df, spec)
    assert np.all(np.isfinite(vector))
    assert report.sanitized == (), (
        f"moments lost to numerical failure on partition {partition}: {report.sanitized}"
    )
    # Measured: the live parquet carries zero NaN and zero inf. If that ever
    # changes the effective sample size has silently moved and we want to know.
    assert report.omitted_nan == {}, (
        f"partition {partition} now carries NaN cells: {report.omitted_nan}"
    )
    assert report.all_nan == ()


@_SKIP_PARQUET
def test_byte_duplicate_partitions_produce_identical_fingerprints():
    """client_20 is a byte-dup of client_19 (task.py `duplicate_partitions`).

    This is the identity-reset mechanism H3 must link, so the construct MUST
    map them to the same point.
    """
    import pandas as pd

    spec = load_feature_spec()
    vectors = []
    for partition in (19, 20):
        df = pd.read_parquet(
            EVAL_DATA_DIR / f"client_{partition}.parquet", columns=list(spec.features)
        )
        vectors.append(compute_fingerprint(df, spec))
    assert vectors[0].tobytes() == vectors[1].tobytes()


@_SKIP_PARQUET
def test_fingerprint_round_trips_through_the_metrics_encoding():
    import pandas as pd

    spec = load_feature_spec()
    df = pd.read_parquet(
        EVAL_DATA_DIR / "client_0.parquet", columns=list(spec.features)
    )
    vec = compute_fingerprint(df, spec)
    restored = decode_fingerprint(encode_fingerprint(vec))
    assert restored.tobytes() == vec.tobytes()


# ---------------------------------------------------------------------------
# Numerical contract (synthetic — always runs)
# ---------------------------------------------------------------------------

def _frame(**columns):
    import pandas as pd

    return pd.DataFrame(columns)


def test_moments_are_row_order_invariant():
    rng = np.random.default_rng(7)
    values = rng.normal(size=500)
    features = ("a",)
    direct = compute_fingerprint(_frame(a=values), features)
    shuffled = compute_fingerprint(_frame(a=rng.permutation(values)), features)
    assert np.allclose(direct, shuffled, rtol=0, atol=1e-12)


def test_constant_column_yields_zero_shape_moments_not_nan():
    vec, report = compute_fingerprint_with_report(
        _frame(a=np.zeros(100)), ("a",)
    )
    assert vec.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert np.all(np.isfinite(vec))
    assert ("a", "skew") in report.degenerate
    assert ("a", "kurtosis") in report.degenerate


@pytest.mark.parametrize("scale", [1e118, 1e200])
def test_extreme_magnitude_column_does_not_overflow_into_nan(scale):
    """Edge-IIoT reality: `tcp.options` ~1e118 (scipy.skew overflows) and
    `tcp.payload` large enough that ``sum(x**2)`` overflows float64 outright."""
    base = np.array([1.0, 2.0, 3.0, 40.0] * 25, dtype=np.float64)
    values = base * scale
    vec, report = compute_fingerprint_with_report(_frame(a=values), ("a",))
    assert np.all(np.isfinite(vec)), vec
    assert report.sanitized == ()
    # Every moment rescales exactly: mean/std are linear, shape moments invariant.
    unit = compute_fingerprint(_frame(a=base), ("a",))
    assert np.isclose(vec[0], unit[0] * scale, rtol=1e-12)
    assert np.isclose(vec[1], unit[1] * scale, rtol=1e-12)
    assert np.allclose(vec[2:], unit[2:], rtol=1e-9, atol=1e-9)


def test_moments_agree_with_the_phase0_reference_implementation():
    """The construct must stay byte-comparable with `compute_client_signature()`
    (scripts/fingerprint_feasibility.py:32-42) on well-conditioned data."""
    from scipy.stats import kurtosis, skew

    rng = np.random.default_rng(3)
    values = rng.gamma(shape=2.0, scale=3.0, size=2000)
    reference = [
        float(np.nanmean(values)),
        float(np.nanstd(values)),
        float(skew(values, nan_policy="omit")),
        float(kurtosis(values, nan_policy="omit")),
    ]
    vec = compute_fingerprint(_frame(a=values), ("a",))
    assert np.allclose(vec, reference, rtol=1e-10, atol=1e-10)


def test_all_nan_column_is_sanitised_and_reported():
    vec, report = compute_fingerprint_with_report(
        _frame(a=np.full(10, np.nan)), ("a",)
    )
    assert np.all(np.isfinite(vec))
    assert vec.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert report.sanitized, "non-finite moments must be reported, not silently zeroed"
    assert report.omitted_nan == {"a": 10}
    assert report.all_nan == ("a",)
    assert report.clean is False


def test_partial_nan_cells_are_omitted_and_COUNTED_per_feature():
    """`nan_policy='omit'` semantics — but the dropped cells change the
    effective sample size, so the count is recorded rather than merely flagged."""
    values = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    vec, report = compute_fingerprint_with_report(_frame(a=values), ("a",))
    assert report.omitted_nan == {"a": 2}
    assert report.total_omitted_cells == 2
    assert report.all_nan == ()
    assert report.clean is False
    # The moments are those of the surviving cells, exactly as scipy would give.
    survivors = np.array([1.0, 2.0, 4.0, 6.0])
    assert np.isclose(vec[0], survivors.mean())
    assert np.isclose(vec[1], survivors.std())


def test_a_clean_column_reports_nothing_at_all():
    rng = np.random.default_rng(17)
    _, report = compute_fingerprint_with_report(
        _frame(a=rng.normal(size=200)), ("a",)
    )
    assert report.clean is True
    assert report.omitted_nan == {}
    assert report.total_omitted_cells == 0


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_an_infinite_cell_is_REFUSED_never_silently_omitted(bad):
    """scipy's nan_policy='omit' does NOT omit infinities — it propagates them.

    An infinity is not missing data; computing the moment from the surviving
    cells would fabricate a value the data does not support. Measured on the
    live parquet there are zero infinities, so this refusal costs nothing real
    and exists to make a future data defect loud.
    """
    values = np.array([1.0, 2.0, bad, 4.0])
    with pytest.raises(FeatureSpecError, match="infinite"):
        compute_fingerprint(_frame(a=values), ("a",))


def test_the_infinity_refusal_names_the_offending_feature_and_count():
    values = np.array([1.0, np.inf, np.inf, 4.0])
    with pytest.raises(FeatureSpecError) as excinfo:
        compute_fingerprint(_frame(good=np.arange(4.0), bad=values), ("good", "bad"))
    message = str(excinfo.value)
    assert "'bad'" in message and "2 infinite" in message


def test_missing_column_is_a_loud_error():
    with pytest.raises(FeatureSpecError, match="missing"):
        compute_fingerprint(_frame(a=np.arange(10.0)), ("a", "b"))


def test_empty_frame_is_a_loud_error():
    with pytest.raises(FeatureSpecError, match="empty"):
        compute_fingerprint(_frame(a=np.array([], dtype=np.float64)), ("a",))


def test_encoding_is_stable_and_dimension_checked():
    rng = np.random.default_rng(11)
    vec = rng.normal(size=180)
    payload = encode_fingerprint(vec)
    assert isinstance(payload, str)
    assert encode_fingerprint(vec) == payload  # deterministic
    assert decode_fingerprint(payload).tobytes() == vec.tobytes()
    with pytest.raises(FeatureSpecError, match="dim"):
        decode_fingerprint(json.dumps([1.0, 2.0]), expected_dim=180)

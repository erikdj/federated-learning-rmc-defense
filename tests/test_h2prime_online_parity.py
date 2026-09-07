"""GOLDEN PARITY GATE — online window-feature builder vs the frozen offline
builder, BIT-EXACT on the recorded golden fixture (spec 2026-08-16 § 7 item 2).

The online builder (`flowerfl/h2prime_online.OnlineWindowFeatureBuilder`) is a
transcription of the frozen `derive_window_feats` (frozen at commit dcef0f7,
under the committed golden-hash gate `tests/test_window_feats_golden.py`) —
kept local to the serving module so runtime does not depend on an analysis
helper. This test replays the same committed golden
signal-log fixture through both builders and asserts every derived feature is
bit-identical (`==` on floats, no tolerance), in both the frozen consumption
order and the online per-round arrival order. **Parity failure is
launch-blocking** — it means the served features are not the features the
H2' instrument was calibrated on.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.h2prime_online import (  # noqa: E402
    DERIVED_FEATS,
    OnlineWindowFeatureBuilder,
    WINDOW,
)

BUILDER_PATH = (
    PROJECT_ROOT / "reproduction" / "protocol" / "h2prime"
    / "revalidate_v115.py"
)
GOLDEN_INPUT = (
    PROJECT_ROOT / "reproduction" / "protocol" / "h2prime"
    / "golden" / "window_feats_golden_input.jsonl"
)


def _load_frozen_builder():
    spec = importlib.util.spec_from_file_location(
        "revalidate_v115_frozen_parity", BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden_rows() -> list[dict]:
    with open(GOLDEN_INPUT, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _frozen_output(rows: list[dict]) -> list[dict]:
    frozen = _load_frozen_builder()
    out = copy.deepcopy(rows)
    frozen.derive_window_feats(out)
    return out


def _frozen_consumption_order(rows: list[dict]) -> list[int]:
    """Row indices in the frozen builder's per-cid consumption order:
    groups in first-appearance order, each sorted ascending by
    scenario_round with ties broken by input order (CPython stable sort) —
    transcribed from tests/test_window_feats_golden.py::canonicalize."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["logical_cid"], []).append(i)
    order: list[int] = []
    for members in groups.values():
        order.extend(sorted(members, key=lambda i: rows[i]["scenario_round"]))
    return order


def _online_arrival_order(rows: list[dict]) -> list[int]:
    """Global round order (stable) — how rounds actually arrive online."""
    return sorted(range(len(rows)), key=lambda i: rows[i]["scenario_round"])


def _assert_bit_exact(order: list[int], rows: list[dict], frozen_rows: list[dict]):
    builder = OnlineWindowFeatureBuilder()
    for i in order:
        online_row = builder.add_row(rows[i])
        for feat in DERIVED_FEATS:
            assert online_row[feat] == frozen_rows[i][feat], (
                f"PARITY FAILURE (launch-blocking, spec 2026-08-16 § 7 item "
                f"2): row {i} cid={rows[i]['logical_cid']!r} "
                f"round={rows[i]['scenario_round']} feature {feat!r}: "
                f"online={online_row[feat]!r} frozen={frozen_rows[i][feat]!r}"
            )


@pytest.mark.unit
def test_window_constant_matches_frozen():
    assert WINDOW == _load_frozen_builder().WINDOW


@pytest.mark.unit
def test_derived_feature_names_match_frozen():
    frozen = _load_frozen_builder()
    assert set(DERIVED_FEATS) <= set(frozen.FEATS_V115)
    assert set(frozen.FEATS_V115) - set(frozen.RAW) == set(DERIVED_FEATS)


@pytest.mark.unit
def test_golden_parity_in_frozen_consumption_order():
    rows = _golden_rows()
    frozen_rows = _frozen_output(rows)
    _assert_bit_exact(_frozen_consumption_order(rows), rows, frozen_rows)


@pytest.mark.unit
def test_golden_parity_in_online_arrival_order():
    """Cross-client interleaving must not change any per-client feature —
    the online builder consumes rounds as they arrive, not grouped by cid."""
    rows = _golden_rows()
    frozen_rows = _frozen_output(rows)
    _assert_bit_exact(_online_arrival_order(rows), rows, frozen_rows)


@pytest.mark.unit
def test_online_builder_does_not_mutate_input_rows():
    rows = _golden_rows()
    snapshot = copy.deepcopy(rows)
    builder = OnlineWindowFeatureBuilder()
    for i in _online_arrival_order(rows):
        builder.add_row(rows[i])
    assert rows == snapshot


@pytest.mark.unit
def test_tenure_reset_starts_a_fresh_episode_bit_exact():
    """A synthetic RMC rejoin (tenure resets to 1 under the same logical id)
    must reproduce the frozen episode-reset behaviour exactly."""
    rows = []
    for rnd in range(1, 9):
        tenure = rnd if rnd <= 4 else rnd - 4      # reset at round 5
        rows.append({
            "logical_cid": "client_3",
            "scenario_round": rnd,
            "tenure": tenure,
            "update_norm": 1.0 + 0.31 * rnd,
            "train_loss": 0.9 - 0.05 * rnd,
            "cos_to_median": 0.99 - 0.01 * rnd,
            "L2_to_median": 0.1 * rnd,
            "num_examples": 100,
        })
    frozen_rows = _frozen_output(rows)
    _assert_bit_exact(list(range(len(rows))), rows, frozen_rows)
    # And the reset row itself must be an all-zeros window row (fresh episode).
    builder = OnlineWindowFeatureBuilder()
    outputs = [builder.add_row(r) for r in rows]
    for feat in DERIVED_FEATS:
        assert outputs[4][feat] == 0.0

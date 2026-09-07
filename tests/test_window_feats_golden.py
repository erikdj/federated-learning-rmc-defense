"""Golden window-feature fixture gate — spec v1.15 rev-6 s2.2b (pre-ratification).

Runs the FROZEN `derive_window_feats()` construction (the versioned feature
builder in `reproduction/protocol/h2prime/revalidate_v115.py`,
frozen at commit dcef0f7) on the committed golden input and asserts:

  1. the golden input file's SHA-256 matches GOLDEN_HASHES.md;
  2. the canonical serialization of the derived output is BYTE-equal to the
     committed golden expected file;
  3. the SHA-256 of that canonical serialization matches GOLDEN_HASHES.md.

Passing this test is a PRE-CONDITION of the H2' confirmatory scoring pass.
Per the spec's hard-stop rule (s2.2b): "the confirmatory window-feature
builder -- the cited function itself or any re-implementation -- must
reproduce the recorded hash before any confirmatory row is scored. A
mismatch is a hard stop, not a diff to eyeball: it means the pipeline being
scored is not the pipeline this document's dev grounding was measured on."
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO / "reproduction" / "protocol" / "h2prime" / "revalidate_v115.py"
)
GOLDEN_DIR = (
    REPO / "reproduction" / "protocol" / "h2prime" / "golden"
)
GOLDEN_INPUT = GOLDEN_DIR / "window_feats_golden_input.jsonl"
GOLDEN_EXPECTED = GOLDEN_DIR / "window_feats_golden_expected.json"
GOLDEN_HASHES = GOLDEN_DIR / "GOLDEN_HASHES.md"

FROZEN_COMMIT = "dcef0f7"
DERIVED_FEATS = ["norm_variance", "loss_slope", "cos_drift", "cos_variance"]

HARD_STOP = (
    "HARD STOP (spec v1.15 rev-6 s2.2b): \"the confirmatory window-feature "
    "builder -- the cited function itself or any re-implementation -- must "
    "reproduce the recorded hash before any confirmatory row is scored. A "
    "mismatch is a hard stop, not a diff to eyeball: it means the pipeline "
    "being scored is not the pipeline this document's dev grounding was "
    "measured on.\""
)


def _load_frozen_builder():
    """Import the frozen feature builder from `reproduction/protocol/h2prime/`.

    revalidate_v115.py is not on any package path (it lives under `reproduction/protocol/h2prime/`),
    so it is loaded by absolute file location, repo-root-relative, which works
    from any pytest invocation cwd.
    """
    spec = importlib.util.spec_from_file_location(
        "revalidate_v115_frozen", BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_golden_input_rows() -> list[dict]:
    with open(GOLDEN_INPUT, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def canonicalize(rows: list[dict]) -> bytes:
    """Canonical serialization of the derived features, in the FROZEN total
    row order of spec s2.2b: rows grouped per logical_cid (groups in order of
    first appearance in the input file, matching the builder's defaultdict
    insertion order), each group sorted ascending by scenario_round with ties
    broken by original input-file row index (CPython stable sort — the
    builder's actual consumption order, transcribed not invented).

    One output entry per input row — no row dropped, none duplicated.
    """
    indexed = list(enumerate(rows))
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, r in indexed:
        groups.setdefault(r["logical_cid"], []).append((i, r))
    entries = []
    for cid, members in groups.items():
        ordered = sorted(members, key=lambda ir: ir[1]["scenario_round"])
        for i, r in ordered:
            entry = {
                "row_index": i,
                "logical_cid": cid,
                "scenario_round": r["scenario_round"],
                "tenure": r.get("tenure"),
            }
            for f in DERIVED_FEATS:
                entry[f] = r[f]
            entries.append(entry)
    doc = {
        "_meta": {
            "builder": ".planning/h2prime/protocol_exact_revalidation/"
                       "revalidate_v115.py::derive_window_feats",
            "frozen_commit": FROZEN_COMMIT,
            "spec": "docs/superpowers/specs/"
                    "2026-08-09-praxis-experimental-design-v1.15-h2prime.md s2.2b",
            "n_rows": len(rows),
            "derived_feats": DERIVED_FEATS,
        },
        "rows": entries,
    }
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _recorded_hashes() -> dict[str, str]:
    text = GOLDEN_HASHES.read_text(encoding="utf-8")
    pat = re.compile(r"^\s*(\S+)\s+sha256\s*=\s*([0-9a-f]{64})\s*$", re.M)
    found = {m.group(1): m.group(2) for m in pat.finditer(text)}
    assert found, f"no sha256 records parsed from {GOLDEN_HASHES}"
    return found


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# fixture integrity
# ---------------------------------------------------------------------------
def test_golden_artifacts_exist():
    for p in (BUILDER_PATH, GOLDEN_INPUT, GOLDEN_EXPECTED, GOLDEN_HASHES):
        assert p.is_file(), f"missing golden-gate artifact: {p}"


def test_golden_input_hash_matches_recorded():
    recorded = _recorded_hashes()[GOLDEN_INPUT.name]
    actual = _sha256(GOLDEN_INPUT.read_bytes())
    assert actual == recorded, (
        f"{HARD_STOP}\n  golden INPUT file hash drifted:\n"
        f"  recorded {recorded}\n  actual   {actual}"
    )


def test_golden_expected_file_hash_matches_recorded():
    recorded = _recorded_hashes()[GOLDEN_EXPECTED.name]
    actual = _sha256(GOLDEN_EXPECTED.read_bytes())
    assert actual == recorded, (
        f"{HARD_STOP}\n  golden EXPECTED file hash drifted:\n"
        f"  recorded {recorded}\n  actual   {actual}"
    )


# ---------------------------------------------------------------------------
# the gate: frozen construction must reproduce the recorded hash
# ---------------------------------------------------------------------------
@pytest.mark.golden
def test_frozen_builder_reproduces_golden_hash():
    mod = _load_frozen_builder()
    rows = load_golden_input_rows()
    mod.derive_window_feats(rows)
    canon = canonicalize(rows)

    expected_bytes = GOLDEN_EXPECTED.read_bytes()
    assert canon == expected_bytes, (
        f"{HARD_STOP}\n  canonical serialization is not byte-equal to "
        f"{GOLDEN_EXPECTED.name}"
    )

    recorded = _recorded_hashes()[GOLDEN_EXPECTED.name]
    actual = _sha256(canon)
    assert actual == recorded, (
        f"{HARD_STOP}\n  recorded {recorded}\n  actual   {actual}"
    )


# ---------------------------------------------------------------------------
# structural coverage the spec's fixture table demands
# ---------------------------------------------------------------------------
def test_output_row_count_equals_input_row_count():
    """s2.2b stride rule: exactly one feature row per consumed input row."""
    rows = load_golden_input_rows()
    expected = json.loads(GOLDEN_EXPECTED.read_text(encoding="utf-8"))
    assert len(expected["rows"]) == len(rows)
    assert sorted(e["row_index"] for e in expected["rows"]) == list(range(len(rows)))


def test_fixture_covers_required_structural_cases():
    rows = load_golden_input_rows()
    counts = Counter(r["logical_cid"] for r in rows)
    # window saturation: at least one client with > 3 rounds
    assert any(n > 3 for n in counts.values()), "no window-saturated client"
    # minimum-period: at least one client with <= 1 row
    assert any(n <= 1 for n in counts.values()), "no single-row client"
    # multi-client
    assert len(counts) >= 3, "fixture is not multi-client"
    # RMC rejoin lineage present (fresh-cid rejoin identities, e.g. *_newN)
    assert any("_new" in c for c in counts), "no RMC rejoin-lineage identity"


def test_single_row_client_gets_all_zero_window_feats():
    """s2.2b minimum-period rule: <=1 usable observation -> exactly 0.0 for
    all four derived features (no NaN, no back-fill)."""
    mod = _load_frozen_builder()
    rows = load_golden_input_rows()
    mod.derive_window_feats(rows)
    counts = Counter(r["logical_cid"] for r in rows)
    singles = [c for c, n in counts.items() if n == 1]
    assert singles
    for r in rows:
        if r["logical_cid"] in singles:
            for f in DERIVED_FEATS:
                assert r[f] == 0.0, (r["logical_cid"], f, r[f])


def test_every_episode_first_round_is_all_zeros():
    """The first round of every episode is a defined all-zeros window row."""
    mod = _load_frozen_builder()
    rows = load_golden_input_rows()
    mod.derive_window_feats(rows)
    firsts: dict[str, dict] = {}
    for r in rows:
        cid = r["logical_cid"]
        if cid not in firsts or r["scenario_round"] < firsts[cid]["scenario_round"]:
            firsts[cid] = r
    for cid, r in firsts.items():
        for f in DERIVED_FEATS:
            assert r[f] == 0.0, (cid, f, r[f])


def test_no_derived_feature_is_nan_or_null():
    mod = _load_frozen_builder()
    rows = load_golden_input_rows()
    mod.derive_window_feats(rows)
    for r in rows:
        for f in DERIVED_FEATS:
            v = r[f]
            assert v is not None and v == v, (r["logical_cid"], f, v)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

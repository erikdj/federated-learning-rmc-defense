"""Unit tests for scripts/compute_bracket_cuts.py (amendment v1.8 §4 bracket cuts).

Covers the two things the pre-unblinding lock rests on:
  1. the quantile math is the frozen F8 procedure at each bracket FPR, and
  2. the anchor gate fires LOUDLY when a pool fails to reproduce its frozen 10%
     cut (so 1%/5% numbers can never be emitted from a wrong pool).
Uses synthetic honest pools written to tmp jsonl — no network, no real logs.
"""
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import compute_bracket_cuts as cbc  # noqa: E402
from h2_threshold_pipeline import select_threshold  # noqa: E402


def _write_pool(path: Path, honest_scores, *, defense_token="tgensemble",
                score_field="tge_score", scenario="S4_full_mix", seed=42,
                n_malicious=3):
    """Write a synthetic signal log: honest rows with the given scores plus a few
    malicious rows (excluded from the honest pool) and one honest row with a null
    score (also excluded), all carrying valid identity tokens."""
    lines = []
    for s in honest_scores:
        lines.append({"malicious_gt": False, score_field: s,
                      "defense": defense_token, "scenario": scenario, "seed": seed})
    for _ in range(n_malicious):
        lines.append({"malicious_gt": True, score_field: 0.999,
                      "defense": defense_token, "scenario": scenario, "seed": seed})
    lines.append({"malicious_gt": False, score_field: None,
                  "defense": defense_token, "scenario": scenario, "seed": seed})
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")


def test_quantile_math_matches_frozen_procedure():
    """Bracket cuts equal select_threshold at each FPR = sorted[floor(fpr*n)]."""
    pool = [i / 100 for i in range(100)]  # 0.00.. 0.99, n=100
    for fpr, expected_idx in [(0.01, 1), (0.05, 5), (0.10, 10)]:
        assert select_threshold(pool, fpr) == pool[expected_idx]


def test_load_honest_pool_excludes_malicious_and_nulls(tmp_path):
    scores = [0.1 * i for i in range(1, 11)]
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    pool, inv = cbc.load_honest_pool("tge", ["S4"], [f])
    assert sorted(pool) == pytest.approx(sorted(scores))
    assert len(pool) == 10  # malicious + null-score honest excluded
    assert inv[0]["honest_scored_in_scope"] == 10
    assert inv[0]["rows"] == 14  # 10 honest + 3 malicious + 1 null


def test_strict_identity_token_mismatch_is_loud(tmp_path):
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, [0.5, 0.6], defense_token="krumtge")  # wrong token for 'tge'
    with pytest.raises(cbc.AnchorError, match="defense token"):
        cbc.load_honest_pool("tge", ["S4"], [f])


def test_compute_pool_anchor_pass(tmp_path, monkeypatch):
    scores = [i / 1000 for i in range(1000)]  # n=1000, 10% => sorted[100] = 0.1
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    monkeypatch.setitem(cbc.FROZEN_10PCT, ("tge", "primary_s4"), select_threshold(scores, 0.10))
    meta = {"scope": ["S4"], "unanchored": False, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("tge", "primary_s4", meta)
    assert res["anchor"]["match"] is True
    assert res["n_honest"] == 1000
    assert res["bracket_cuts"]["0.01"]["cut"] == scores[10]
    assert res["bracket_cuts"]["0.05"]["cut"] == scores[50]


def test_compute_pool_anchor_failure_is_loud(tmp_path, monkeypatch):
    """A pool whose 10% cut does NOT match the frozen record must abort — never
    silently emit 1%/5% numbers."""
    scores = [i / 1000 for i in range(1000)]
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    monkeypatch.setitem(cbc.FROZEN_10PCT, ("tge", "primary_s4"), 0.42424242)  # wrong
    meta = {"scope": ["S4"], "unanchored": False, "files": [f], "provenance": "synthetic"}
    with pytest.raises(cbc.AnchorError, match="ANCHOR FAILURE"):
        cbc.compute_pool("tge", "primary_s4", meta)


def test_unanchored_pool_skips_anchor_check(tmp_path):
    """A pool with no frozen record (unanchored diagnostic) computes without an
    anchor assertion but is flagged unanchored."""
    scores = [i / 100 for i in range(100)]
    f = tmp_path / "s0_clean_baseline__krum__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores, defense_token="krum", score_field="krum_score",
                scenario="S0_clean_baseline")
    meta = {"scope": ["S0"], "unanchored": True, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("krum", "sensitivity_s0s4", meta)
    assert res["anchor"]["has_frozen_record"] is False
    assert res["anchor"]["unanchored"] is True
    assert "match" not in res["anchor"]


def test_effective_support_points_reported(tmp_path):
    scores = [i / 100 for i in range(100)]
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    monkeypatch_free_meta = {"scope": ["S4"], "unanchored": True, "files": [f],
                             "provenance": "synthetic"}
    res = cbc.compute_pool("tge", "primary_s4_unanchored_probe", monkeypatch_free_meta)
    # 1% of n=100 rests on ~1 support point — the honesty note the memo carries.
    assert res["bracket_cuts"]["0.01"]["effective_support_points"] == 1.0


def test_custom_quantiles_adds_2pct_without_disturbing_frozen_cuts(tmp_path):
    """Passing an extended quantile set (adds 2%) recomputes each cut as
    sorted[floor(fpr*n)] and leaves the frozen 1/5/10 cuts value-identical."""
    scores = [i / 1000 for i in range(1000)]  # n=1000
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    meta = {"scope": ["S4"], "unanchored": True, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("tge", "primary_s4_probe", meta,
                           quantiles=(0.01, 0.02, 0.05, 0.10))
    c = res["bracket_cuts"]
    assert set(c) == {"0.01", "0.02", "0.05", "0.10"}
    assert c["0.02"]["cut"] == scores[20]   # floor(0.02*1000) = 20
    # frozen points unchanged vs the default 3-quantile bracket:
    assert c["0.01"]["cut"] == scores[10]
    assert c["0.05"]["cut"] == scores[50]
    assert c["0.10"]["cut"] == scores[100]


def test_anchored_pool_without_10pct_quantile_is_loud(tmp_path):
    """A frozen-record pool computed with a quantile set that omits 0.10 must
    abort — the anchor point can't be checked, so no cuts are emitted."""
    scores = [i / 1000 for i in range(1000)]
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    meta = {"scope": ["S4"], "unanchored": False, "files": [f], "provenance": "synthetic"}
    with pytest.raises(cbc.AnchorError, match="0.10 anchor"):
        cbc.compute_pool("tge", "primary_s4", meta, quantiles=(0.01, 0.02, 0.05))


def _frozen_json(path: Path, defense, scope, cuts: dict):
    path.write_text(json.dumps({"configs": {defense: {scope: {
        "bracket_cuts": {q: {"cut": v} for q, v in cuts.items()}}}}}))


def test_verify_against_frozen_json_all_match(tmp_path):
    frozen = tmp_path / "bracket_cuts.json"
    _frozen_json(frozen, "tge", "primary_s4",
                 {"0.01": 0.1, "0.05": 0.2, "0.10": 0.3})
    results = {"tge": {"primary_s4": {"bracket_cuts": {
        "0.01": {"cut": 0.1}, "0.02": {"cut": 0.15},
        "0.05": {"cut": 0.2}, "0.10": {"cut": 0.3}}}}}
    chk = cbc.verify_against_frozen_json(results, frozen)
    assert chk["all_match"] is True
    assert chk["shared_quantiles_checked"] == 3  # 0.01, 0.05, 0.10 (not the new 0.02)


def test_verify_against_frozen_json_mismatch_is_loud(tmp_path):
    frozen = tmp_path / "bracket_cuts.json"
    _frozen_json(frozen, "tge", "primary_s4", {"0.10": 0.3})
    results = {"tge": {"primary_s4": {"bracket_cuts": {
        "0.02": {"cut": 0.15}, "0.10": {"cut": 0.31}}}}}  # 0.10 drifted
    with pytest.raises(cbc.AnchorError, match="FROZEN-JSON ANCHOR FAILURE"):
        cbc.verify_against_frozen_json(results, frozen)


# =============================================================================
# v1.10 §3.3 parameterization : --source-root / per-pool overrides,
# frozen-snapshot write guard, and the value-identity regression gate.
# =============================================================================

FROZEN_SNAPSHOT_JSON = REPO / "reproduction/protocol/h2-bracket/bracket_cuts.json"


def test_default_source_paths_are_the_frozen_hardcoded_ones():
    """With everything unset, resolution returns EXACTLY the paths the frozen
    leak-on record was derived from — the parameterization must not move the
    defaults by a single character."""
    dev, sig, overridden = cbc.resolve_source_paths(None, None, None)
    assert overridden is False
    assert dev == REPO / "results/20260726/_dev_honest_sources"
    assert sig == REPO / "results/20260724/exp014_closed_loop/signals"
    # and the default registry resolves its files under exactly those dirs
    src = cbc.build_sources(False)
    krum_first = src[("krum", "primary_s4")]["files"][0]
    assert krum_first == dev / "EXP-005c/s4_full_mix__krum__persistent_optimizer__seed42.jsonl"
    assert all(f.parent == sig for f in src[("tge", "primary_s4")]["files"])
    assert all(f.parent == sig for f in src[("krum_tge", "sensitivity_s0s4")]["files"])


def _mk_layout(root: Path) -> tuple[Path, Path]:
    dev = root / "_dev_honest_sources"
    sig = root / "exp014_closed_loop" / "signals"
    dev.mkdir(parents=True)
    sig.mkdir(parents=True)
    return dev, sig


def test_source_root_overrides_pool_paths(tmp_path):
    _mk_layout(tmp_path)
    exp_dev, exp_sig = tmp_path / "_dev_honest_sources", tmp_path / "exp014_closed_loop/signals"
    got_dev, got_sig, got_overridden = cbc.resolve_source_paths(tmp_path, None, None)
    assert got_overridden is True
    assert got_dev == exp_dev
    assert got_sig == exp_sig
    # the registry follows the override for EVERY pool
    src = cbc.build_sources(True, dev_src=got_dev, exp014_signals=got_sig)
    for meta in src.values():
        for f in meta["files"]:
            assert str(f).startswith(str(tmp_path)), f


def test_per_pool_override_beats_source_root(tmp_path):
    _mk_layout(tmp_path / "root")
    other_dev = tmp_path / "other_dev"
    other_dev.mkdir()
    dev, sig, overridden = cbc.resolve_source_paths(tmp_path / "root", other_dev, None)
    assert overridden is True
    assert dev == other_dev  # per-pool wins
    assert sig == tmp_path / "root" / "exp014_closed_loop" / "signals"


def test_missing_source_root_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="--source-root"):
        cbc.resolve_source_paths(tmp_path / "does_not_exist", None, None)


def test_source_root_missing_expected_layout_fails_loudly(tmp_path):
    (tmp_path / "_dev_honest_sources").mkdir()  # signals subdir missing
    with pytest.raises(FileNotFoundError, match="signals"):
        cbc.resolve_source_paths(tmp_path, None, None)


def test_out_dir_guard_blocks_frozen_snapshot_when_inputs_overridden(tmp_path):
    """v1.10 §3.3 item 4: overridden-input outputs must NEVER land in the frozen
    leak-on snapshot dir (byte-for-byte preservation)."""
    frozen_dir = cbc.FROZEN_SNAPSHOT_DIR
    with pytest.raises(cbc.AnchorError, match="frozen"):
        cbc.ensure_out_dir_allowed(frozen_dir, "anything.json", inputs_overridden=True)
    with pytest.raises(cbc.AnchorError, match="frozen"):
        cbc.ensure_out_dir_allowed(frozen_dir / "sub", "x.json", inputs_overridden=True)
    # default (no overrides) keeps the existing CLI behavior untouched
    cbc.ensure_out_dir_allowed(frozen_dir, "bracket_cuts.json", inputs_overridden=False)
    # overridden inputs with a separate out dir are fine
    cbc.ensure_out_dir_allowed(tmp_path, "leakfree.json", inputs_overridden=True)


_DEFAULT_SOURCES_PRESENT = (
    (REPO / "results/20260726/_dev_honest_sources").is_dir()
    and (REPO / "results/20260724/exp014_closed_loop/signals").is_dir())

_SOURCES_ABSENT_REASON = (
    "VALUE-IDENTITY GATE TEST SKIPPED (test only — the RUNTIME gate in "
    "compute_bracket_cuts.py has NO skip path and hard-aborts any overridden-"
    "input run on such a checkout): the frozen dev honest pools (gitignored, "
    "~83MB: results/20260726/_dev_honest_sources + "
    "results/20260724/exp014_closed_loop/signals) are not present here.")


def _split(vals: list, n_chunks: int) -> list[list]:
    chunks: list[list] = [[] for _ in range(n_chunks)]
    for i, v in enumerate(vals):
        chunks[i % n_chunks].append(v)
    return chunks


def _engineer_pools(v_primary: float, v_sens: float | None = None):
    """Engineer honest-score pools that reproduce the REAL frozen anchors:
    a 30-score primary pool whose 10% cut (sorted[floor(0.1*30)] = sorted[3])
    is EXACTLY v_primary, plus (if v_sens is given) a 70-score extension so the
    combined 100-score pool's 10% cut (sorted[10]) is EXACTLY v_sens. All
    values distinct; works for v_sens on either side of v_primary."""
    if v_sens is None:
        return ([v_primary - (3 - i) * 1e-4 for i in range(3)] + [v_primary]
                + [v_primary + (i + 1) * 1e-4 for i in range(26)]), None
    lo, hi = sorted((v_primary, v_sens))
    step = (hi - lo) / 32
    if v_primary > v_sens:
        below = [v_sens - (10 - i) * step for i in range(10)]   # 10 < v_sens
        mid = [v_sens + (i + 1) * step for i in range(3)]       # 3 in (v_sens, v_primary)
        above = [v_primary + (i + 1) * step for i in range(85)]
        primary = mid + [v_primary] + above[:26]                # sorted[3] = v_primary
        ext = below + [v_sens] + above[26:]                     # combined sorted[10] = v_sens
    else:
        below = [v_primary - (3 - i) * step for i in range(3)]  # 3 < v_primary
        mid = [v_primary + (i + 1) * step for i in range(6)]    # 6 in (v_primary, v_sens)
        above = [v_sens + (i + 1) * step for i in range(89)]
        primary = below + [v_primary] + above[:26]              # sorted[3] = v_primary
        ext = mid + [v_sens] + above[26:]                       # combined sorted[10] = v_sens
    return primary, ext


def _write_synthetic_tree(root: Path) -> None:
    """Build an anchor-CONSISTENT synthetic source tree in the --source-root
    layout: every pool is engineered so its 10% cut equals the REAL frozen
    record, letting the end-to-end override test run the REAL identity gate and
    REAL anchor checks with no stubbing."""
    dev, sig = _mk_layout(root)
    seeds = (42, 137, 256, 314, 500)
    other_stems = ["s0_clean_baseline", "s1_benign_churn_only",
                   "s2_adaptive_switching_only", "s3_identity_reset_only"]

    def write(path, defense, stem, seed, vals):
        spec = cbc.DEFENSE_SCORE_SPECS[defense]
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_pool(path, vals, defense_token=spec.row_defense_token,
                    score_field=spec.score_field,
                    scenario=stem[0].upper() + stem[1:], seed=seed)

    for d in ("krum", "trustscore"):
        primary, _ = _engineer_pools(cbc.FROZEN_10PCT[(d, "primary_s4")])
        for exp, chunk in zip(("EXP-005c", "EXP-005e"), _split(primary, 2)):
            write(dev / exp / f"s4_full_mix__{d}__persistent_optimizer__seed42.jsonl",
                  d, "s4_full_mix", 42, chunk)
    for d in ("tge", "krum_tge"):
        primary, ext = _engineer_pools(cbc.FROZEN_10PCT[(d, "primary_s4")],
                                       cbc.FROZEN_10PCT[(d, "sensitivity_s0s4")])
        for seed, chunk in zip(seeds, _split(primary, 5)):
            write(sig / f"s4_full_mix__{d}__persistent_optimizer__seed{seed}.jsonl",
                  d, "s4_full_mix", seed, chunk)
        cells = [(s, sd) for s in other_stems for sd in seeds]
        for (s, sd), chunk in zip(cells, _split(ext, len(cells))):
            write(sig / f"{s}__{d}__persistent_optimizer__seed{sd}.jsonl",
                  d, s, sd, chunk)


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_main_end_to_end_with_source_root(tmp_path, monkeypatch):
    """--source-root wires through main with the REAL runtime identity gate —
    NO stubbing: the gate first verifies the frozen
    snapshot from the default pools, then the overridden pools (engineered to
    reproduce the real frozen anchors) are read from the override tree and the
    output lands OUTSIDE the frozen snapshot dir with the override and the
    gate's real attestation recorded in _meta."""
    root = tmp_path / "leakfree_sources"
    _write_synthetic_tree(root)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["compute_bracket_cuts.py",
                                      "--source-root", str(root),
                                      "--out-dir", str(out_dir)])
    assert cbc.main() == 0
    out = json.loads((out_dir / "bracket_cuts.json").read_text())
    assert out["_meta"]["source_override"] == {
        "dev_src": str(root / "_dev_honest_sources"),
        "exp014_signals": str(root / "exp014_closed_loop" / "signals")}
    ig = out["_meta"]["identity_gate"]
    assert ig["passed"] is True
    assert ig["compared_against"] == "reproduction/protocol/h2-bracket/bracket_cuts.json"
    assert ig["pools_checked"] == 6 and ig["cuts_checked"] == 18
    got = {(d, sc) for d in out["configs"] for sc in out["configs"][d]}
    assert got == {("krum", "primary_s4"), ("trustscore", "primary_s4"),
                   ("tge", "primary_s4"), ("krum_tge", "primary_s4"),
                   ("tge", "sensitivity_s0s4"), ("krum_tge", "sensitivity_s0s4")}
    for d in out["configs"]:
        for sc in out["configs"][d]:
            assert out["configs"][d][sc]["anchor"]["match"] is True


def test_main_refuses_frozen_out_dir_with_source_root(tmp_path, monkeypatch):
    root = tmp_path / "leakfree_sources"
    _write_synthetic_tree(root)
    monkeypatch.setattr(sys, "argv", ["compute_bracket_cuts.py",
                                      "--source-root", str(root)])  # default out-dir = frozen
    with pytest.raises(cbc.AnchorError, match="refusing to write"):
        cbc.main()


# --- runtime identity-gate enforcement (: MUST-NOT-PROCEED is
# structural — a checkout that cannot prove default-input value identity
# CANNOT derive from overridden inputs; no skip path, no escape flag) --------

def test_deep_diff_reports_nested_paths_and_missing_keys():
    a = {"x": {"y": 1, "z": [1, 2]}, "only_frozen": 1}
    b = {"x": {"y": 2, "z": [1, 3]}, "extra": 5}
    joined = "\n".join(cbc.deep_diff(a, b))
    assert "x.y" in joined and "x.z[1]" in joined
    assert "only_frozen" in joined and "extra" in joined


def test_deep_diff_skip_allowlist_is_exact_paths():
    a = {"_meta": {"keep": 1}}
    b = {"_meta": {"keep": 1, "anchor_verified": True}}
    assert cbc.deep_diff(a, b, skip=("_meta.anchor_verified",)) == []
    assert cbc.deep_diff(a, b) != []
    # the allowlist is exact: skipping one volatile key hides nothing else
    b2 = {"_meta": {"keep": 2, "anchor_verified": True}}
    assert cbc.deep_diff(a, b2, skip=("_meta.anchor_verified",)) != []


def test_identity_gate_missing_default_pools_aborts(tmp_path, monkeypatch):
    """A checkout WITHOUT the frozen default pools cannot pass the gate —
    hard AnchorError, never a skip."""
    monkeypatch.setattr(cbc, "DEV_SRC", tmp_path / "absent_dev")
    monkeypatch.setattr(cbc, "EXP014_SIGNALS", tmp_path / "absent_sig")
    with pytest.raises(cbc.AnchorError, match="IDENTITY GATE"):
        cbc.run_identity_gate()


def test_identity_gate_missing_frozen_snapshot_aborts(tmp_path):
    with pytest.raises(cbc.AnchorError, match="IDENTITY GATE"):
        cbc.run_identity_gate(frozen_path=tmp_path / "no_such_snapshot.json")


def test_main_with_overrides_hard_aborts_without_default_pools(tmp_path, monkeypatch):
    """Overridden inputs on a checkout without the frozen default pools must
    abort BEFORE any leak-free computation — nothing is written."""
    root = tmp_path / "leakfree_sources"
    _write_synthetic_tree(root)
    monkeypatch.setattr(cbc, "DEV_SRC", tmp_path / "absent_dev")
    monkeypatch.setattr(cbc, "EXP014_SIGNALS", tmp_path / "absent_sig")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["compute_bracket_cuts.py",
                                      "--source-root", str(root),
                                      "--out-dir", str(out_dir)])
    with pytest.raises(cbc.AnchorError, match="IDENTITY GATE"):
        cbc.main()
    assert not out_dir.exists()


def test_identity_skip_fields_is_minimal():
    """the ONLY structural exclusion from the identity
    comparison is the post-freeze default-None cross-check key. Everything
    else is either strictly compared or held to a pinned expected value."""
    assert cbc.IDENTITY_SKIP_FIELDS == ("_meta.frozen_json_cross_check",)


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_value_identity_default_inputs_reproduce_frozen_snapshot():
    """v1.10 §3.3 item 2 — regression over the SAME shared verification function
    the runtime gate uses: with the DEFAULT (unset) inputs, the COMPLETE output
    object reproduces the committed frozen leak-on snapshot value-identically,
    with field-specific expected-value assertions instead of a blanket skip. The frozen JSON is read-only here — never regenerated."""
    att = cbc.run_identity_gate()
    assert att["passed"] is True
    assert att["compared_against"] == "reproduction/protocol/h2-bracket/bracket_cuts.json"
    assert att["pools_checked"] == 6
    assert att["cuts_checked"] == 18  # 6 pools x {1,5,10}%
    assert att["fields_excluded"] == list(cbc.IDENTITY_SKIP_FIELDS)
    # Every _meta key that exists ONLY under input overrides is asserted ABSENT
    # from a default build. `leakfree_provenance` joined that set with the
    # leak-free mode — the enumeration grows as the guard grows; it is never
    # relaxed.
    assert set(att["field_expectations"]) == {
        "_meta.anchor_verified", "_meta.purpose",
        "_meta.source_override", "_meta.identity_gate",
        "_meta.leakfree_provenance"}


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
@pytest.mark.parametrize("mutate, expect_path", [
    # a drifted cut value...
    (lambda f: f["configs"]["krum"]["primary_s4"]["bracket_cuts"]["0.01"]
        .__setitem__("cut", 0.123456789),
     "configs.krum.primary_s4.bracket_cuts.0.01.cut"),
    #...a non-cut field: the comparison is the COMPLETE object, not a
    # spot-check of cut values...
    (lambda f: f["configs"]["tge"]["primary_s4"]["source_files"][0]
        .__setitem__("rows", 1),
     "configs.tge.primary_s4.source_files[0].rows"),
    #...and a strictly-compared _meta field: no blanket _meta allowlist
    (lambda f: f["_meta"].__setitem__("procedure", "tampered"),
     "_meta.procedure"),
])
def test_identity_gate_tampered_frozen_json_aborts(tmp_path, mutate, expect_path):
    frozen = json.loads(FROZEN_SNAPSHOT_JSON.read_text())
    mutate(frozen)
    tampered = tmp_path / "bracket_cuts.json"
    tampered.write_text(json.dumps(frozen))
    with pytest.raises(cbc.AnchorError) as ei:
        cbc.run_identity_gate(frozen_path=tampered)
    msg = str(ei.value)
    assert "IDENTITY GATE FAILURE" in msg
    assert expect_path in msg


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_main_with_real_override_paths_runs_gate_and_attests(tmp_path, monkeypatch):
    """Full integration, no stubs: explicitly supplied per-pool overrides (equal
    to the default dirs) trip the override path, the runtime gate verifies the
    frozen snapshot for real, and the output carries the attestation."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py",
        "--dev-src", str(cbc.DEV_SRC),
        "--exp014-signals", str(cbc.EXP014_SIGNALS),
        "--out-dir", str(out_dir)])
    assert cbc.main() == 0
    out = json.loads((out_dir / "bracket_cuts.json").read_text())
    ig = out["_meta"]["identity_gate"]
    assert ig["passed"] is True
    assert ig["pools_checked"] == 6 and ig["cuts_checked"] == 18
    assert ig["compared_against"] == "reproduction/protocol/h2-bracket/bracket_cuts.json"
    assert out["_meta"]["source_override"] == {
        "dev_src": str(cbc.DEV_SRC), "exp014_signals": str(cbc.EXP014_SIGNALS)}


# =============================================================================
# v1.14 §3.1 leak-free re-freeze mode (--leakfree-signals).
#
# Executes the v1.10 §3.1-3.5 re-freeze mechanics (citation per v1.14 rev-3)
# under the never-mix-regimes rule (v1.10 §2.4 / v1.11 §3). Each test below
# pins one of the four blockers found in the reproduction/protocol/h2-bracket/ preparation:
#   A. the leak-ON 10% anchor must NOT fire against a leak-FREE pool
#   B. leak-free mode must NEVER leave krum/trustscore on the leak-on pool
#   C. the S0-S4 sensitivity pools must NOT be registered (v1.10 §3.2)
#   + the two artifact-contract fields v1.10 §3.3 requires but the frozen
#     script never emitted (degeneracy flag, Krum+TGE survivor coverage).
# =============================================================================

LEAKFREE_SEEDS = (42, 137, 256, 314, 500)


_FAKE_DIGEST = "sha256:d7fbee5b19c542e630e058c2a0d349a72adc0d4c50e3f0a58c0b898b5e488310"


def _write_leakfree_provenance(sig: Path, *, normalize=True, digest=_FAKE_DIGEST,
                               omit: set | None = None) -> None:
    """Stage the regime-provenance manifest the leak-free mode requires. The
    signal.jsonl rows carry NO regime field (verified against the real EXP-041
    logs), so provenance must come from a manifest built at staging time from the
    result JSONs."""
    omit = omit or set()
    cells = []
    for d in ("krum", "trustscore", "tge", "krum_tge"):
        for seed in LEAKFREE_SEEDS:
            name = f"s4_full_mix__{d}__persistent_optimizer__seed{seed}.jsonl"
            if name in omit:
                continue
            cells.append({
                "file": name, "source_experiment": "EXP-041",
                "normalize_train_only": (normalize(name) if callable(normalize)
                                         else normalize),
                "image_digest": digest(name) if callable(digest) else digest,
            })
    (sig / cbc.LEAKFREE_PROVENANCE_NAME).write_text(
        json.dumps({"_meta": {"regime": "leak-free"}, "cells": cells}, indent=2))


def _write_leakfree_tree(sig: Path, *, scores=None, provenance=True) -> None:
    """Four configs x five dev seeds of s4_full_mix leak-free cells, with values
    deliberately UNEQUAL to any frozen leak-on anchor (blocker A evidence)."""
    sig.mkdir(parents=True, exist_ok=True)
    vals = scores if scores is not None else [i / 1000 for i in range(1000)]
    for d in ("krum", "trustscore", "tge", "krum_tge"):
        spec = cbc.DEFENSE_SCORE_SPECS[d]
        for seed, chunk in zip(LEAKFREE_SEEDS, _split(list(vals), 5)):
            _write_pool(sig / f"s4_full_mix__{d}__persistent_optimizer__seed{seed}.jsonl",
                        chunk, defense_token=spec.row_defense_token,
                        score_field=spec.score_field, scenario="S4_full_mix", seed=seed)
    if provenance:
        _write_leakfree_provenance(sig)


# --- blocker A: regime-scoped anchors ---------------------------------------

def test_leakfree_scope_key_has_no_frozen_anchor():
    """The leak-free scope key is absent from FROZEN_10PCT by construction, so
    compute_pool takes its frozen-is-None branch. The leak-ON keys keep theirs."""
    for d in ("krum", "trustscore", "tge", "krum_tge"):
        assert (d, cbc.LEAKFREE_SCOPE) not in cbc.FROZEN_10PCT
        assert (d, "primary_s4") in cbc.FROZEN_10PCT


def test_leakfree_pool_does_not_fire_leakon_anchor(tmp_path):
    """Blocker A: a leak-free pool whose 10% cut differs from the frozen leak-on
    record computes cleanly instead of aborting with ANCHOR FAILURE."""
    scores = [i / 1000 for i in range(1000)]  # 10% cut = 0.1, vs frozen tge 0.5754
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    meta = {"scope": ["S4"], "unanchored": True, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("tge", cbc.LEAKFREE_SCOPE, meta)
    assert res["anchor"]["has_frozen_record"] is False
    assert res["anchor"]["unanchored"] is True
    assert res["bracket_cuts"]["0.10"]["cut"] == scores[100]
    assert res["bracket_cuts"]["0.10"]["cut"] != cbc.FROZEN_10PCT[("tge", "primary_s4")]


# --- blockers B + C: what the leak-free registry contains --------------------

def test_leakfree_sources_register_only_four_primary_pools(tmp_path):
    """Blocker B (all four configs redirected together — no leak-on residue) and
    blocker C (no S0-S4 sensitivity pools; v1.10 §3.2 does not reproduce them)."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    src = cbc.build_leakfree_sources(sig)
    assert set(src) == {(d, cbc.LEAKFREE_SCOPE)
                        for d in ("krum", "trustscore", "tge", "krum_tge")}
    assert not any(sc.startswith("sensitivity") for _, sc in src)
    for (d, _sc), meta in src.items():
        assert meta["unanchored"] is True
        assert meta["scope"] == ["S4"]
        assert len(meta["files"]) == 5
        # every file — INCLUDING krum/trustscore — comes from the leak-free dir
        assert all(f.parent == sig for f in meta["files"])
        assert {int(f.stem.split("seed")[-1]) for f in meta["files"]} == set(LEAKFREE_SEEDS)


def test_leakfree_never_reads_the_leakon_dev_sources(tmp_path):
    """Blocker B, stated as the property that failed: no leak-free pool may
    resolve a path under the frozen leak-on EXP-005c/e tree."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    src = cbc.build_leakfree_sources(sig)
    for meta in src.values():
        for f in meta["files"]:
            assert "EXP-005c" not in str(f) and "EXP-005e" not in str(f)
            assert not str(f).startswith(str(cbc.DEV_SRC))


@pytest.mark.parametrize("extra", [
    ["--source-root", "SOME"], ["--dev-src", "SOME"], ["--exp014-signals", "SOME"]])
def test_leakfree_mode_refuses_leakon_input_overrides(tmp_path, monkeypatch, extra):
    """Blocker B as a guard: combining the leak-free source with any leak-on
    input override is exactly the silent cross-regime mix v1.10 §2.4 forbids."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    other = tmp_path / "other"
    (other / "_dev_honest_sources").mkdir(parents=True)
    (other / "exp014_closed_loop" / "signals").mkdir(parents=True)
    argv = ["compute_bracket_cuts.py", "--leakfree-signals", str(sig),
            "--out-dir", str(tmp_path / "out")]
    argv += [extra[0], str(other)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(cbc.AnchorError, match="cross-regime"):
        cbc.main()


def test_leakfree_mode_refuses_krumts_sensitivity(tmp_path, monkeypatch):
    """Blocker C as a guard: v1.10 §3.2 does not reproduce ANY pooled S0-S4
    sensitivity in the leak-free regime."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--krumts-sensitivity", "--out-dir", str(tmp_path / "out")])
    with pytest.raises(cbc.AnchorError, match="sensitivity"):
        cbc.main()


def test_leakfree_missing_signals_dir_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(tmp_path / "nope"),
        "--out-dir", str(tmp_path / "out")])
    with pytest.raises(FileNotFoundError, match="--leakfree-signals"):
        cbc.main()


# --- the two artifact-contract fields (v1.10 §3.3) --------------------------

def test_survivor_coverage_counts_null_scored_honest_rows(tmp_path):
    """v1.10 §3.3 'survivor coverage reported'. _write_pool adds exactly one
    null-score honest row, which the frozen eligibility filter drops."""
    f = tmp_path / "s4_full_mix__krum_tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, [0.1, 0.2, 0.3], defense_token="krumtge", score_field="tge_score")
    cov = cbc.survivor_coverage([f], "krum_tge")
    assert cov["score_field"] == "tge_score"
    assert cov["honest_rows_total"] == 4      # 3 scored + 1 null-score honest
    assert cov["honest_rows_scored"] == 3     # malicious rows never counted
    assert cov["rows_dropped_null_score"] == 1
    assert cov["survivor_coverage"] == pytest.approx(0.75)


def test_degeneracy_flag_fires_on_score_mass_point(tmp_path):
    """v1.10 §3.3 degeneracy warning: Krum's known score-mass-at-zero makes the
    low quantiles unable to realize their nominal FPR. No cut is emitted without
    this flag."""
    scores = [0.0] * 90 + [0.5 + i / 100 for i in range(10)]  # n=100, mass at 0
    f = tmp_path / "s4_full_mix__krum__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores, defense_token="krum", score_field="krum_score")
    meta = {"scope": ["S4"], "unanchored": True, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("krum", cbc.LEAKFREE_SCOPE, meta,
                           quantiles=(0.01, 0.05, 0.10))
    flags = cbc.degeneracy(res)
    assert flags["0.01"]["degenerate"] is True
    assert flags["0.05"]["degenerate"] is True
    assert "mass point" in flags["0.01"]["reason"]
    # a non-degenerate quantile carries the flag too, set False with no reason
    assert set(flags) == {"0.01", "0.05", "0.10"}
    assert flags["0.01"]["reason"] is not None


def test_degeneracy_flag_absent_on_well_spread_pool(tmp_path):
    scores = [i / 1000 for i in range(1000)]
    f = tmp_path / "s4_full_mix__tge__persistent_optimizer__seed42.jsonl"
    _write_pool(f, scores)
    meta = {"scope": ["S4"], "unanchored": True, "files": [f], "provenance": "synthetic"}
    res = cbc.compute_pool("tge", cbc.LEAKFREE_SCOPE, meta)
    flags = cbc.degeneracy(res)
    assert all(f["degenerate"] is False for f in flags.values())
    assert all(f["reason"] is None for f in flags.values())


# --- end-to-end through main, real identity gate, no stubbing -------------

@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_main_leakfree_end_to_end(tmp_path, monkeypatch):
    """The full re-freeze path: the REAL runtime identity gate proves default-
    input value identity FIRST, then exactly four unanchored leak-free pools are
    emitted, each carrying both artifact-contract fields, with the leak-free
    source recorded in _meta."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--quantiles", "0.01", "0.02", "0.05", "0.10",
        "--out-dir", str(out_dir), "--out-name", "bracket_cuts_leakfree.json"])
    assert cbc.main() == 0
    out = json.loads((out_dir / "bracket_cuts_leakfree.json").read_text())

    ig = out["_meta"]["identity_gate"]
    assert ig["passed"] is True
    assert ig["pools_checked"] == 6 and ig["cuts_checked"] == 18
    assert out["_meta"]["source_override"]["leakfree_signals"] == str(sig)
    assert "leak-free" in out["_meta"]["source_override"]["regime"]

    got = {(d, sc) for d in out["configs"] for sc in out["configs"][d]}
    assert got == {(d, cbc.LEAKFREE_SCOPE)
                   for d in ("krum", "trustscore", "tge", "krum_tge")}
    assert out["_meta"]["anchors_checked"] == []  # nothing leak-on was asserted
    for d in out["configs"]:
        pool = out["configs"][d][cbc.LEAKFREE_SCOPE]
        assert pool["anchor"]["has_frozen_record"] is False
        assert set(pool["bracket_cuts"]) == {"0.01", "0.02", "0.05", "0.10"}
        assert set(pool["degeneracy"]) == {"0.01", "0.02", "0.05", "0.10"}
        assert pool["survivor_coverage"]["honest_rows_scored"] == pool["n_honest"]
        assert len(pool["source_files"]) == 5


def test_main_leakfree_refuses_frozen_out_dir(tmp_path, monkeypatch):
    """The leak-free run is an overridden-input run, so the frozen leak-on
    snapshot dir stays byte-for-byte protected (v1.10 §3.3 item 4)."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig)])  # default out-dir
    with pytest.raises(cbc.AnchorError, match="refusing to write"):
        cbc.main()


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_leakfree_meta_keys_stay_absent_from_default_build():
    """The leak-free mode must add NOTHING to a default-input build — that is
    what keeps the value-identity gate byte-for-byte."""
    att = cbc.run_identity_gate()
    assert att["passed"] is True


def test_main_leakfree_hard_aborts_without_default_pools(tmp_path, monkeypatch):
    """Leak-free derivation on a checkout that cannot prove default-input
    identity must abort BEFORE computing anything — nothing is written."""
    sig = tmp_path / "leakfree_signals"
    _write_leakfree_tree(sig)
    monkeypatch.setattr(cbc, "DEV_SRC", tmp_path / "absent_dev")
    monkeypatch.setattr(cbc, "EXP014_SIGNALS", tmp_path / "absent_sig")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--out-dir", str(out_dir)])
    with pytest.raises(cbc.AnchorError, match="IDENTITY GATE"):
        cbc.main()
    assert not out_dir.exists()


# =============================================================================
# auto-review fixes. Three findings, all of the silent-mislabeling class
# this tooling exists to kill:
#   R1 (P1) regime-provenance validation of the leak-free inputs
#   R2 (P1) the complete {1,2,5,10}% bracket is mandatory in leak-free mode
#   R3 (P2) no vacuous anchor_verified:true on an unanchored leak-free record
# =============================================================================

# --- R1: provenance is validated, not assumed from filenames ----------------

def test_signal_rows_carry_no_regime_field_so_manifest_is_required(tmp_path):
    """The premise of the manifest requirement, pinned: a signal row has no
    normalize_train_only (verified against the real EXP-041 logs, 1604 rows
    sampled). is_dir + a filename match is NOT provenance."""
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig)
    row = json.loads(next(iter(
        (sig / "s4_full_mix__krum__persistent_optimizer__seed42.jsonl")
        .read_text().splitlines())))
    assert "normalize_train_only" not in row
    assert cbc.LEAKFREE_PROVENANCE_NAME == "provenance.json"


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_leakfree_missing_provenance_manifest_halts(tmp_path, monkeypatch):
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig, provenance=False)
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--out-dir", str(tmp_path / "out")])
    with pytest.raises(cbc.AnchorError, match="provenance manifest"):
        cbc.main()
    assert not (tmp_path / "out").exists()


def test_leakfree_cell_absent_from_manifest_halts_naming_the_cell(tmp_path):
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig, provenance=False)
    missing = "s4_full_mix__tge__persistent_optimizer__seed256.jsonl"
    _write_leakfree_provenance(sig, omit={missing})
    src = cbc.build_leakfree_sources(sig)
    files = [f for m in src.values() for f in m["files"]]
    with pytest.raises(cbc.AnchorError, match=re.escape(missing)):
        cbc.validate_leakfree_provenance(sig, files)


def test_leakfree_normalize_train_only_false_halts_naming_the_cell(tmp_path):
    """A leak-ON cell staged into the leak-free dir is exactly the silent
    cross-regime mislabeling this gate exists to catch."""
    sig = tmp_path / "sig"
    bad = "s4_full_mix__krum__persistent_optimizer__seed137.jsonl"
    _write_leakfree_tree(sig, provenance=False)
    _write_leakfree_provenance(sig, normalize=lambda n: n != bad)
    src = cbc.build_leakfree_sources(sig)
    files = [f for m in src.values() for f in m["files"]]
    with pytest.raises(cbc.AnchorError) as ei:
        cbc.validate_leakfree_provenance(sig, files)
    msg = str(ei.value)
    assert bad in msg and "normalize_train_only" in msg


@pytest.mark.parametrize("bad_value", [None, "true", 1, 0])
def test_leakfree_non_boolean_true_normalize_flag_halts(tmp_path, bad_value):
    """Only a real boolean True admits a cell — a truthy string or 1 must not
    slip through (absence is treated as failure, never as default-ok)."""
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig, provenance=False)
    _write_leakfree_provenance(sig, normalize=bad_value)
    src = cbc.build_leakfree_sources(sig)
    files = [f for m in src.values() for f in m["files"]]
    with pytest.raises(cbc.AnchorError, match="normalize_train_only"):
        cbc.validate_leakfree_provenance(sig, files)


def test_leakfree_mixed_image_digest_halts(tmp_path):
    """The adjudicating S4 pool is v13-only (v1.11 § 4.2); a pool spanning two
    image digests is a v1.11 § 3 no-pooling violation."""
    sig = tmp_path / "sig"
    other = "s4_full_mix__krum_tge__persistent_optimizer__seed500.jsonl"
    _write_leakfree_tree(sig, provenance=False)
    _write_leakfree_provenance(
        sig, digest=lambda n: "sha256:0ther" if n == other else _FAKE_DIGEST)
    src = cbc.build_leakfree_sources(sig)
    files = [f for m in src.values() for f in m["files"]]
    with pytest.raises(cbc.AnchorError, match="digest"):
        cbc.validate_leakfree_provenance(sig, files)


def test_leakfree_valid_provenance_returns_attestation(tmp_path):
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig)
    src = cbc.build_leakfree_sources(sig)
    files = [f for m in src.values() for f in m["files"]]
    att = cbc.validate_leakfree_provenance(sig, files)
    assert att["cells_validated"] == 20
    assert att["normalize_train_only"] is True
    assert att["image_digest"] == _FAKE_DIGEST
    assert att["source_experiments"] == ["EXP-041"]


# --- R2: the complete bracket is mandatory ----------------------------------

@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_leakfree_defaults_to_the_complete_bracket(tmp_path, monkeypatch):
    """With --quantiles unset, leak-free mode computes all of {1,2,5,10}% —
    a partial freeze must not be reachable by omission."""
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--out-dir", str(out_dir)])
    assert cbc.main() == 0
    out = json.loads((out_dir / "bracket_cuts.json").read_text())
    assert out["_meta"]["bracket_fprs"] == [0.01, 0.02, 0.05, 0.10]
    for d in out["configs"]:
        assert set(out["configs"][d][cbc.LEAKFREE_SCOPE]["bracket_cuts"]) == {
            "0.01", "0.02", "0.05", "0.10"}


@pytest.mark.parametrize("quantiles", [
    ["0.01", "0.10"], ["0.10"], ["0.01", "0.02", "0.05"],
    ["0.01", "0.02", "0.05", "0.10", "0.20"]])
def test_leakfree_partial_bracket_halts(tmp_path, monkeypatch, quantiles):
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig)
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--quantiles", *quantiles, "--out-dir", str(tmp_path / "out")])
    with pytest.raises(cbc.AnchorError, match="complete"):
        cbc.main()


def test_leakon_default_bracket_is_unchanged_by_the_leakfree_requirement():
    """The leak-free requirement must not move the leak-on default bracket."""
    assert cbc.BRACKET_FPRS == (0.01, 0.05, 0.10)
    assert cbc.LEAKFREE_BRACKET == (0.01, 0.02, 0.05, 0.10)


# --- R3: honest anchor metadata on an unanchored record ---------------------

@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_leakfree_anchor_verified_is_null_with_reason(tmp_path, monkeypatch):
    """anchor_verified must NOT be vacuously true just because zero anchors were
    checked. The identity gate stays recorded separately as what actually ran."""
    sig = tmp_path / "sig"
    _write_leakfree_tree(sig)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "compute_bracket_cuts.py", "--leakfree-signals", str(sig),
        "--out-dir", str(out_dir)])
    assert cbc.main() == 0
    meta = json.loads((out_dir / "bracket_cuts.json").read_text())["_meta"]
    assert meta["anchor_verified"] is None
    assert meta["anchors_checked"] == []
    assert "unanchored by construction" in meta["anchor_gate"].lower()
    assert "leak-on" in meta["anchor_gate"]
    # the verification that DID run is recorded, separately and truthfully
    assert meta["identity_gate"]["passed"] is True
    assert meta["leakfree_provenance"]["cells_validated"] == 20


@pytest.mark.skipif(not _DEFAULT_SOURCES_PRESENT, reason=_SOURCES_ABSENT_REASON)
def test_leakon_default_build_still_claims_anchor_verified_true():
    """The honest-metadata change must not touch the leak-on record, whose
    anchor_verified:true is EARNED (6 pools reproduce their frozen 10% cuts)."""
    att = cbc.run_identity_gate()
    assert att["passed"] is True

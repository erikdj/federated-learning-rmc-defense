"""Gates for the WINDOW-AWARE LOAO sensitivity arm (reported-only).

Four of the nine frozen features are window statistics: the committed builder
derives them over a trailing window of up to three rows per (logical_cid,
tenure-episode). The § 2.2a fit exclusion, by contrast, is a ROW-LABEL rule. In
S2 the same identity switches attack family WITHOUT a tenure reset, so the first
rows of a retained family carry values computed partly from held-out-family
rows. This module gates the arm that measures what the fit looks like once those
rows are held out too.

Two properties are load-bearing and are tested here as such:

  * the annotation reproduces the FROZEN builder's episode logic exactly — same
    grouping, same sort, same reset condition, same trailing slice — so the arm
    excludes the rows whose features really did see the held-out family;
  * the PRIMARY path is untouched. `score_corpus` with the flag at its default
    must be behaviourally identical to the pre-arm executor, because the arm is
    a sensitivity and the pre-registered read is the one that adjudicates.

No golden-hash gate is exercised: nothing here scores a sealed row, and the
annotation is asserted against hand-computed windows rather than a recorded
hash.
"""
from __future__ import annotations

import json

import pytest

import test_adjudicate_h2prime as BASE

A = BASE.A                      # the executor module, loaded once by BASE
import h2prime_corpus as C      # noqa: E402  (scripts/ is on sys.path via A)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row(cid, rnd, tenure, family=None, seed=1, scen_long="s2_adaptive_switching_only"):
    """One signal-log row in the shape `derive_window_feats` consumes."""
    mal = family is not None
    tge = (0.2 + BASE._jitter(cid, rnd, seed, salt=7) * 0.5 if mal
           else 0.55 + BASE._jitter(cid, rnd, seed, salt=7) * 0.45)
    return BASE._tge_row(cid, rnd, seed, scen_long, scen_long, tge,
                         tenure=tenure, mal=mal, family=family or "alie",
                         noisy=True)


def _derive(rows):
    """The load_cells discipline, per unit: frozen builder, then annotation."""
    A.R.derive_window_feats(rows)
    C.annotate_window_families(rows)
    return rows


def _families(rows):
    return [r["_window_families"] for r in rows]


# The S2 shape the arm exists for: one identity carries gaussian_noise, then
# switches to label_flip mid-run with NO tenure reset.
SWITCH_AT = 5
ROUNDS = 8


def _family_for(cid, rnd):
    if cid == "client_1":
        return "alie"
    if cid == "client_2":
        return "gaussian_noise" if rnd < SWITCH_AT else "label_flip"
    if cid == "client_3":
        return "label_flip"
    return None                                   # client_0 / honest_1


def _synthetic_corpus(seeds=(1, 2, 3, 4, 5)):
    """A five-seed × five-scenario corpus with ONE switching identity per unit.

    Built in memory rather than staged on disk because the arm under test reads
    only rows; the per-unit discipline that matters (derive, then annotate, one
    unit at a time) is reproduced exactly, so no window ever spans two units.
    """
    rows = []
    for scen_long in BASE.SCENARIOS_LONG:
        scen = A.R.SCEN_SHORT[scen_long]
        for seed in seeds:
            unit = [_row(cid, rnd, rnd, _family_for(cid, rnd), seed, scen_long)
                    for cid in ("client_0", "honest_1",
                                "client_1", "client_2", "client_3")
                    for rnd in range(1, ROUNDS + 1)]
            for r in unit:
                r["_scen"], r["_seed"] = scen, seed
                r["_defense"], r["_source"] = "tge", "SYNTHETIC"
            rows.extend(_derive(unit))
    return rows


def _canonical(obj):
    """A JSON-comparable projection of a `score_corpus` result.

    The result is keyed by tuples, so it cannot be serialized directly; keys are
    rendered with `repr` and everything is sorted, which makes the comparison
    total rather than a spot check of a few fields.
    """
    if isinstance(obj, dict):
        return {repr(k): _canonical(v)
                for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def _dump(scored):
    return json.dumps(_canonical(scored), sort_keys=True, default=repr)


# ---------------------------------------------------------------------------
# (a) the annotation reproduces the builder's window, family by family
# ---------------------------------------------------------------------------
def test_switch_without_a_tenure_reset_marks_exactly_the_first_two_rows():
    """S2: tenure runs 1..8, the family switches at round 5, no reset.

    WINDOW = 3 (past + current), so rounds 5 and 6 are the only retained-family
    rows whose window still reaches back into the held-out family. Round 7's
    window is rounds 5-7, which is entirely the new family.
    """
    rows = _derive([_row("client_2", rnd, rnd, _family_for("client_2", rnd))
                    for rnd in range(1, ROUNDS + 1)])
    assert _families(rows) == [
        ["gaussian_noise"],                       # r1
        ["gaussian_noise"],                       # r2
        ["gaussian_noise"],                       # r3
        ["gaussian_noise"],                       # r4
        ["gaussian_noise", "label_flip"],         # r5 — window r3,r4,r5
        ["gaussian_noise", "label_flip"],         # r6 — window r4,r5,r6
        ["label_flip"],                           # r7 — window r5,r6,r7
        ["label_flip"],                           # r8
    ]
    assert A.R.WINDOW == 3, "the expectation above is computed for WINDOW = 3"


def test_annotation_uses_the_same_window_length_as_the_frozen_builder():
    """The length is READ from the builder, never retyped.

    A row whose episode is longer than the window must forget the family that
    fell out of it — which is only true if both sides agree on the length.
    """
    rows = _derive([_row("client_2", rnd, rnd, "gaussian_noise" if rnd == 1
                         else "label_flip")
                    for rnd in range(1, 6)])
    # round 1 is gaussian_noise; it leaves the window after round 1 + WINDOW - 1
    carrying = [i + 1 for i, f in enumerate(_families(rows))
                if "gaussian_noise" in f]
    assert carrying == list(range(1, 1 + A.R.WINDOW))


# ---------------------------------------------------------------------------
# (b) a tenure reset starts a fresh window
# ---------------------------------------------------------------------------
def test_a_tenure_reset_starts_a_fresh_window():
    """S3/S4: the family switch coincides with a reconnect.

    A rejoining identity never inherits its previous episode's statistics, so
    it must never inherit its previous episode's FAMILIES either — otherwise
    the arm would exclude rows whose features could not have seen the family.
    """
    rows = _derive(
        [_row("client_2", rnd, rnd, "gaussian_noise") for rnd in range(1, 5)]
        + [_row("client_2", rnd, rnd - 4, "label_flip") for rnd in range(5, 9)]
    )
    assert _families(rows) == [["gaussian_noise"]] * 4 + [["label_flip"]] * 4
    after = _families(rows)[4:]
    assert all("gaussian_noise" not in f for f in after), (
        "a row after the rejoin carried the pre-rejoin family")


def test_a_reset_is_detected_on_a_NON_INCREASING_tenure_not_only_on_tenure_1():
    """The builder's condition is `tenure <= prev_tenure`, transcribed as-is."""
    rows = _derive(
        [_row("client_2", 1, 3, "gaussian_noise"),
         _row("client_2", 2, 3, "label_flip")]        # equal tenure = new episode
    )
    assert _families(rows) == [["gaussian_noise"], ["label_flip"]]


# ---------------------------------------------------------------------------
# (c) honest rows
# ---------------------------------------------------------------------------
def test_honest_rows_carry_an_empty_family_list():
    rows = _derive([_row("client_0", rnd, rnd) for rnd in range(1, 6)])
    assert _families(rows) == [[]] * 5


def test_every_honest_row_of_the_corpus_carries_an_empty_family_list():
    rows = _synthetic_corpus()
    honest = [r for r in rows if not r["malicious_gt"]]
    assert honest, "fixture produced no honest rows"
    assert all(r["_window_families"] == [] for r in honest)
    # ...and the annotation is TOTAL: every row has one, malicious or not.
    assert all("_window_families" in r for r in rows)


# ---------------------------------------------------------------------------
# (d) the fit exclusion, counted
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored_arms():
    """Both arms over one synthetic corpus. Module-scoped: 30 GBDT fits each."""
    rows = _synthetic_corpus()
    plan = A.rotation_plan(sorted({r["_seed"] for r in rows}))
    return rows, plan, A.score_corpus(rows, plan), A.score_corpus(
        rows, plan, window_aware=True)


def test_window_arm_excludes_exactly_the_rows_whose_window_saw_the_family(scored_arms):
    """CENSUS, hand-computed.

    Only `client_2` switches, and only forward (gaussian_noise → label_flip at
    round 5). So beyond the current-label rule the window rule can remove
    exactly its rounds 5 and 6, and only on the gaussian_noise fold:

      3 fit seeds × 5 scenarios × 2 rows = 30

    The label_flip fold removes nothing extra — those two rows are already
    dropped by their own label — and the alie fold removes nothing at all,
    because no window ever reaches an alie row from another identity.
    """
    _rows, plan, primary, window = scored_arms
    assert len(window["fit_census"]) == len(plan)
    for rot in window["fit_census"]:
        per_fold = rot["per_fold"]
        assert set(per_fold) == set(A.ATTACKS)
        assert per_fold["gaussian_noise"]["n_window_excluded_rows"] == 30
        assert per_fold["label_flip"]["n_window_excluded_rows"] == 0
        assert per_fold["alie"]["n_window_excluded_rows"] == 0
    # and the fit population really is smaller by exactly that count
    for w_rot, p_rot in zip(window["fit_census"], primary["fit_census"]):
        for fam in A.ATTACKS:
            w, p = w_rot["per_fold"][fam], p_rot["per_fold"][fam]
            assert p["n_fit_rows"] - w["n_fit_rows"] == w["n_window_excluded_rows"]


def test_window_arm_leaves_the_calibration_population_untouched(scored_arms):
    """The baselines are cut on the CALIBRATION honest rows and never see the
    detector, so an identical baseline cut in every cell is proof that the arm
    did not touch that population — a stronger check than re-counting rows."""
    _rows, _plan, primary, window = scored_arms
    for b, _t, _l in A.BASELINES:
        assert set(window["base_blend"][b]) == set(primary["base_blend"][b])
        for key, cell in primary["base_blend"][b].items():
            assert window["base_blend"][b][key] == cell
        assert window["base_alie"][b] == primary["base_alie"][b]


def test_window_arm_leaves_the_scored_test_population_untouched(scored_arms):
    """Same cells, same row counts, same family mix — only the fit moved."""
    _rows, _plan, primary, window = scored_arms
    assert set(window["blend"]) == set(primary["blend"])
    for key, cell in primary["blend"].items():
        w = window["blend"][key]
        for field in ("n_mal", "n_honest", "family_mix", "families_present",
                      "rotation"):
            assert w[field] == cell[field], f"{field} moved on cell {key}"
    for key, cell in primary["alie"].items():
        assert window["alie"][key]["n_mal"] == cell["n_mal"]
        assert window["alie"][key]["n_honest"] == cell["n_honest"]


def test_window_arm_actually_changes_the_detector(scored_arms):
    """ANTI-VACUITY. A sensitivity that cannot move is not a sensitivity: if
    the exclusion changed no score, every equality above would hold trivially
    and the arm would be reporting the primary read under another name."""
    _rows, _plan, primary, window = scored_arms
    cuts_p = {k: v["cut"] for k, v in primary["per_family"].items()}
    cuts_w = {k: v["cut"] for k, v in window["per_family"].items()}
    assert set(cuts_p) == set(cuts_w)
    assert any(cuts_p[k] != cuts_w[k] for k in cuts_p if k[0] == "gaussian_noise"), (
        "the gaussian_noise fold's detector is identical under an exclusion "
        "that removed 30 fit rows from it")


def test_window_arm_refuses_an_unannotated_corpus():
    """A missing annotation would silently reproduce the primary fit."""
    rows = [_row("client_1", rnd, rnd, "alie") for rnd in range(1, 4)]
    A.R.derive_window_feats(rows)                 # derived, but NOT annotated
    for r in rows:
        r["_scen"], r["_seed"] = "S4", 1
    with pytest.raises(A.HardStop, match="_window_families"):
        A.score_corpus(rows, A.rotation_plan([1, 2, 3, 4, 5]), window_aware=True)


# ---------------------------------------------------------------------------
# (e) the regression: the primary path did not move
# ---------------------------------------------------------------------------
def test_default_call_is_identical_to_an_explicit_window_aware_false(scored_arms):
    """The whole result document, not a spot check."""
    rows, plan, primary, _window = scored_arms
    explicit = A.score_corpus(rows, plan, window_aware=False)
    assert _dump(explicit) == _dump(primary)


def test_the_primary_census_carries_no_window_key(scored_arms):
    """PRE-CHANGE BEHAVIOUR. The default path emits exactly the census it
    emitted before the arm existed: no new key, absent rather than zero, so a
    reader cannot mistake the primary fit for a window-aware one."""
    _rows, _plan, primary, _window = scored_arms
    for rot in primary["fit_census"]:
        for fam, fold in rot["per_fold"].items():
            assert "n_window_excluded_rows" not in fold, (
                f"the primary arm emitted a window census key on fold {fam}")
            assert set(fold) == {"n_fit_rows", "n_fit_positive",
                                 "n_excluded_family_rows"}
    assert not primary.get("degenerate")


def test_the_primary_arm_never_reads_the_annotation(scored_arms):
    """Strip the annotation and the default path must produce the same result.

    The equality above compares two calls that both saw annotated rows; this
    one proves the primary fit does not depend on the annotation AT ALL.
    """
    rows, plan, primary, _window = scored_arms
    stripped = [{k: v for k, v in r.items() if k != "_window_families"}
                for r in rows]
    assert _dump(A.score_corpus(stripped, plan)) == _dump(primary)


# ---------------------------------------------------------------------------
# the reported block
# ---------------------------------------------------------------------------
def test_window_aware_block_reports_both_arms_and_satisfies_the_schema(scored_arms):
    rows, _plan, primary, window = scored_arms
    block = A.window_aware_block(rows, primary, window, A.DEV_SMOKE)
    assert block["status"] == "COMPUTED"
    assert block["n_window_excluded_rows_total"] == 30 * 5      # 5 rotations

    # the corpus census sees the switching identity in every scenario of this
    # fixture: 2 of its rows per unit carry a foreign family in-window
    for scen, c in block["corpus_census"].items():
        assert c["n_malicious_rows_with_a_FOREIGN_family_in_window"] == 2 * 5
        assert c["foreign_family_pairs"] == {"label_flip<-gaussian_noise": 10}
        assert c["share_of_malicious_rows_with_a_foreign_family"] == pytest.approx(
            10 / c["n_malicious_rows"])

    # both arms' P1 and P2 quantities are present, and NO verdict is copied
    q1, q2 = block["p1_quantity"], block["p2_quantity"]
    assert set(q1["primary"]) == set(q1["window_aware"])
    assert "verdict" not in q1["primary"] and "verdict" not in q1["window_aware"]
    assert q1["delta_mean_recall"] == pytest.approx(
        q1["window_aware"]["mean_recall"] - q1["primary"]["mean_recall"])
    assert q2["delta_n_strictly_positive"] == (
        q2["window_aware_sign_test"]["positive"]
        - q2["primary_sign_test"]["positive"])
    assert set(block["per_scenario_blended_recall"]) == set(A.SCENARIOS)

    # the primary quantity the block republishes IS the adjudicated one
    p1 = A.adjudicate_p1(primary, A.DEV_SMOKE)
    assert q1["primary"]["mean_recall"] == p1["mean_recall"]
    assert q1["primary"]["realized_blended_fpr"] == p1["realized_blended_fpr"]

    report = {"secondaries": {"window_aware_loao_sensitivity": block}}
    rule = next(r for r in A.SCHEMA.SCHEMA if r.name == "window-aware sensitivity")
    assert not [k for k in rule.required if k not in block]
    assert A.SCHEMA._resolve(report, rule.select.split(".")) == [block]


def test_window_aware_block_reports_a_degenerate_arm_instead_of_a_partial_one():
    """If the exclusion empties a fit population the arm has no quantity at all.

    It must say so — and must not publish a blend assembled from the folds that
    survived, which would read as a comparable number.
    """
    degenerate = {"degenerate": {"alie": "positive class empty after the "
                                         "window-aware exclusion (0 positive)"},
                  "blend": {}, "fit_census": []}
    block = A.window_aware_block([], {"blend": {}}, degenerate, A.DEV_SMOKE)
    assert block["status"].startswith("UNDEFINED")
    assert block["primary_arm_unaffected"] is True
    assert "p1_quantity" not in block and "p2_quantity" not in block
    for key in ("_note", "window_rule", "corpus_census", "status"):
        assert key in block

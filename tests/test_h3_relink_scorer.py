"""H3 Step 7 — the D7 primary scorer (re-link recall / false-link rate).

Design authority — the FROZEN scorer contract, v1.10 § 5.1:

    "A re-entry event is a **correct re-link** iff `asserted_match ∧
     (asserted_parent_logical_id == gt_logical_id)`; a **false link** iff
     `asserted_match ∧ (asserted_parent_logical_id != gt_logical_id)` **or** an
     honest (`gt_is_malicious=false`) event with `asserted_match=true`. Then
     **recall** = correct-re-links over `gt_is_malicious` events; **FLR** =
     false-links over `gt_is_malicious=false` events. Deduplicate on
     `reentry_event_key` ... an event with `asserted_match=false` counts as a
     miss (recall) and contributes nothing to FLR."

Plus: the ARM RULE (`TGE+FP` only), pooling as a single ratio per scenario (not
a per-seed mean), the S3 FLR exclusion by construction, the zero-denominator
INCONCLUSIVE rule, and the D9 odd-partition hold-out filter (80 / 80 / 15).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_h3_relink import (
    DESIGN_COUNTS,
    FLR_BAR,
    FP_ARM_DEFAULT,
    RECALL_BAR,
    Cohort,
    ScoringError,
    classify_event,
    load_events,
    main,
    score_events,
)

MALICIOUS_PARTITIONS = tuple(range(0, 9))       # client_0..client_8
HONEST_CHURN_PARENTS = (11, 13, 15)            # S4 benign reconnects
CYCLES = 4
SEEDS = (42, 137, 256, 314, 500)


def _event(
    *,
    seed=42,
    scenario="S4_full_mix",
    partition=1,
    cycle=1,
    server_round=10,
    malicious=True,
    matched=True,
    parent_partition=None,
    arm=FP_ARM_DEFAULT,
    key=None,
):
    parent = partition if parent_partition is None else parent_partition
    return {
        "reentry_event_key": key or f"run-{scenario}-{seed}:{server_round}:cid-{partition}-{cycle}",
        # `h3_arm`, NOT `arm`: `arm` is the resampling manifest's variant label
        # elsewhere in this codebase (flowerfl/resampling_manifest.py) and the
        # two meanings meet in the same signal-log row (DRYRUN F4).
        "h3_arm": arm,
        "scenario": scenario,
        "seed": seed,
        "server_round": server_round,
        "current_cid": f"cid-{partition}-{cycle}",
        "gt_logical_id": f"client_{partition}_new{cycle}",
        "gt_is_malicious": malicious,
        "asserted_match": matched,
        "asserted_parent_entry_id": f"fp-{parent}" if matched else None,
        "asserted_parent_logical_id": f"client_{parent}" if matched else None,
        "min_d": 0.5 if matched else 42.0,
        "tau": 1.0,
        "generation": 1 if matched else 0,
    }


def _full_design():
    """The complete pre-registered event census: 5 seeds x {S3, S4}."""
    events = []
    for seed in SEEDS:
        for scenario in ("S3_identity_reset_only", "S4_full_mix"):
            for partition in MALICIOUS_PARTITIONS:
                for cycle in range(1, CYCLES + 1):
                    events.append(
                        _event(
                            seed=seed,
                            scenario=scenario,
                            partition=partition,
                            cycle=cycle,
                            server_round=10 * cycle,
                        )
                    )
            if scenario == "S4_full_mix":
                for partition in HONEST_CHURN_PARENTS:
                    events.append(
                        _event(
                            seed=seed,
                            scenario=scenario,
                            partition=partition,
                            cycle=1,
                            server_round=25,
                            malicious=False,
                            matched=False,
                        )
                    )
    return events


def _write(tmp_path: Path, events, name="events.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


# ---------------------------------------------------------------------------
# Frozen bars and design counts
# ---------------------------------------------------------------------------

def test_bars_are_the_pre_registered_values():
    assert RECALL_BAR == 0.85
    assert FLR_BAR == 0.10


def test_design_counts_match_the_amendment_tables():
    assert DESIGN_COUNTS[Cohort.VALIDATION] == {
        "recall_S3": 180, "recall_S4": 180, "flr_S4": 15,
    }
    assert DESIGN_COUNTS[Cohort.ADJUDICATING] == {
        "recall_S3": 80, "recall_S4": 80, "flr_S4": 15,
    }


# ---------------------------------------------------------------------------
# The frozen classification rule
# ---------------------------------------------------------------------------

def test_matched_malicious_event_with_the_right_parent_is_a_correct_relink():
    assert classify_event(_event(partition=3, parent_partition=3)) == "correct_relink"


def test_matched_malicious_event_with_the_wrong_parent_is_its_own_class():
    """NOT `false_link`: it is drawn from the malicious population, and FLR's
    denominator counts honest events only."""
    assert classify_event(_event(partition=3, parent_partition=5)) == "wrong_malicious_link"


def test_unmatched_malicious_event_is_a_miss():
    assert classify_event(_event(matched=False)) == "miss"


def test_any_matched_honest_event_is_a_false_link():
    """Even a link to the honest device's OWN prior identity is a false link:
    an honest reconnection must never be linked to a flagged entry."""
    assert classify_event(_event(malicious=False, partition=11, parent_partition=11)) == "false_link"
    assert classify_event(_event(malicious=False, partition=11, parent_partition=2)) == "false_link"


def test_unmatched_honest_event_is_a_true_negative():
    assert classify_event(_event(malicious=False, matched=False)) == "true_negative"


def test_a_matched_event_without_a_parent_identity_is_a_loud_error():
    bad = _event()
    bad["asserted_parent_logical_id"] = None
    with pytest.raises(ScoringError, match="parent"):
        classify_event(bad)


# ---------------------------------------------------------------------------
# Loading: arm rule, dedup, validation
# ---------------------------------------------------------------------------

def test_arm_rule_keeps_only_the_fp_arm(tmp_path):
    events = [
        _event(partition=1, arm=FP_ARM_DEFAULT),
        _event(partition=3, arm="tge", key="control-row"),
    ]
    kept = load_events([_write(tmp_path, events)])
    assert [e["h3_arm"] for e in kept] == [FP_ARM_DEFAULT]


def test_the_arm_field_is_h3_arm_and_a_row_without_it_is_refused(tmp_path):
    """DRYRUN F4: the scorer's refusal to score an armless row is load-bearing."""
    armless = _event()
    del armless["h3_arm"]
    with pytest.raises(ScoringError, match="h3_arm"):
        load_events([_write(tmp_path, [armless])])


def test_the_resampling_manifests_arm_cannot_pass_for_the_h3_arm(tmp_path):
    """A row carrying only `arm` (the resampling manifest's variant label) must
    be refused BY NAME — never read as the H3 arm, never silently dropped."""
    mislabelled = _event()
    del mislabelled["h3_arm"]
    mislabelled["arm"] = "smote@balanced"
    with pytest.raises(ScoringError, match="resampling"):
        load_events([_write(tmp_path, [mislabelled])])


def test_a_resampling_arm_alongside_the_h3_arm_is_ignored(tmp_path):
    """Both meanings in one row is the expected steady state; the H3 arm rule
    reads `h3_arm` and nothing else."""
    both = _event(arm=FP_ARM_DEFAULT)
    both["arm"] = "off"
    kept = load_events([_write(tmp_path, [both])])
    assert len(kept) == 1 and kept[0]["h3_arm"] == FP_ARM_DEFAULT


def test_events_are_deduplicated_on_the_event_key(tmp_path):
    duplicated = _event()
    kept = load_events([_write(tmp_path, [duplicated, dict(duplicated)])])
    assert len(kept) == 1


def test_conflicting_rows_under_one_event_key_are_a_loud_error(tmp_path):
    first = _event()
    second = dict(first, asserted_match=False)
    with pytest.raises(ScoringError, match="conflict"):
        load_events([_write(tmp_path, [first, second])])


def test_a_missing_required_field_is_a_loud_error(tmp_path):
    incomplete = _event()
    del incomplete["gt_is_malicious"]
    with pytest.raises(ScoringError, match="gt_is_malicious"):
        load_events([_write(tmp_path, [incomplete])])


def test_an_unusable_provenance_row_is_a_loud_error(tmp_path):
    """Gate (b): zero UNKNOWN / missing-provenance rows."""
    unknown = _event()
    unknown["gt_logical_id"] = "UNKNOWN"
    with pytest.raises(ScoringError, match="UNKNOWN"):
        load_events([_write(tmp_path, [unknown])])


# ---------------------------------------------------------------------------
# Scoring: denominators, pooling, hold-out
# ---------------------------------------------------------------------------

def test_full_design_yields_the_validation_denominators():
    report = score_events(_full_design(), Cohort.VALIDATION)
    assert report["components"]["recall_S3"]["denominator"] == 180
    assert report["components"]["recall_S4"]["denominator"] == 180
    assert report["components"]["flr_S4"]["denominator"] == 15
    assert report["counts_match_design"] is True


def test_odd_partition_holdout_yields_exactly_80_80_15():
    """D9 axis (ii): calibration never saw these devices."""
    report = score_events(_full_design(), Cohort.ADJUDICATING)
    assert report["components"]["recall_S3"]["denominator"] == 80
    assert report["components"]["recall_S4"]["denominator"] == 80
    assert report["components"]["flr_S4"]["denominator"] == 15
    assert report["counts_match_design"] is True


def test_s3_is_excluded_from_the_false_link_rate_by_construction():
    report = score_events(_full_design(), Cohort.VALIDATION)
    assert "flr_S3" not in report["components"]
    assert report["components"]["flr_S4"]["scenario"] == "S4_full_mix"


def test_pooling_is_a_single_ratio_per_scenario_not_a_per_seed_mean():
    """Unequal per-seed denominators make the two definitions differ."""
    events = []
    # seed 42: 1 event, correct.
    events.append(_event(seed=42, scenario="S3_identity_reset_only", partition=1, cycle=1))
    # seed 137: 3 events, all misses.
    for cycle in (1, 2, 3):
        events.append(
            _event(
                seed=137, scenario="S3_identity_reset_only", partition=3,
                cycle=cycle, server_round=10 * cycle, matched=False,
            )
        )
    report = score_events(events, Cohort.VALIDATION, require_design_counts=False)
    pooled = report["components"]["recall_S3"]["value"]
    assert pooled == pytest.approx(0.25)          # 1/4 pooled
    assert pooled != pytest.approx(0.5)           # per-seed mean would be (1 + 0)/2


def test_a_zero_denominator_is_inconclusive_never_a_pass_or_fail():
    s3_only = [e for e in _full_design() if e["scenario"] == "S3_identity_reset_only"]
    report = score_events(s3_only, Cohort.ADJUDICATING, require_design_counts=False)
    assert report["components"]["recall_S4"]["status"] == "INCONCLUSIVE"
    assert report["components"]["flr_S4"]["status"] == "INCONCLUSIVE"
    assert report["verdict"] == "INCONCLUSIVE"


def test_the_validation_cohort_withholds_even_an_inconclusive_verdict():
    """Withholding is UNCONDITIONAL: `verdict` is null on the validation cohort
    whatever the components say, so the field's shape never varies with the
    outcome. The component statuses still carry the INCONCLUSIVE information."""
    s3_only = [e for e in _full_design() if e["scenario"] == "S3_identity_reset_only"]
    report = score_events(s3_only, Cohort.VALIDATION, require_design_counts=False)
    assert report["components"]["recall_S4"]["status"] == "INCONCLUSIVE"
    assert report["verdict"] is None
    assert report["verdict_status"] == "REPORTED (NOT ADJUDICATING)"


def test_realised_counts_below_design_are_reported_not_hidden():
    events = _full_design()[:-1]
    report = score_events(events, Cohort.VALIDATION, require_design_counts=False)
    assert report["counts_match_design"] is False
    assert report["design_count_deltas"]


def test_design_count_mismatch_refuses_by_default():
    """Gate (d): realised counts must equal the design counts exactly."""
    with pytest.raises(ScoringError, match="design"):
        score_events(_full_design()[:-1], Cohort.VALIDATION)


# ---------------------------------------------------------------------------
# The verdict Boolean
# ---------------------------------------------------------------------------

def test_a_perfect_run_passes():
    report = score_events(_full_design(), Cohort.ADJUDICATING)
    assert report["components"]["recall_S3"]["value"] == 1.0
    assert report["components"]["recall_S4"]["value"] == 1.0
    assert report["components"]["flr_S4"]["value"] == 0.0
    assert report["verdict"] == "PASS"


def test_recall_below_the_bar_in_either_scenario_falsifies():
    for target in ("S3_identity_reset_only", "S4_full_mix"):
        events = []
        for event in _full_design():
            odd = int(event["gt_logical_id"].split("_")[1]) % 2 == 1
            if event["scenario"] == target and event["gt_is_malicious"] and odd:
                # Miss 20% of the held-out malicious events (recall 0.8 < 0.85).
                if event["server_round"] == 40:
                    event = dict(event, asserted_match=False,
                                 asserted_parent_logical_id=None,
                                 asserted_parent_entry_id=None)
            events.append(event)
        report = score_events(events, Cohort.ADJUDICATING)
        assert report["components"][f"recall_{target[:2]}"]["value"] == pytest.approx(0.75)
        assert report["verdict"] == "FAIL", target


def test_two_false_links_in_fifteen_falsifies_on_flr():
    events = []
    flipped = 0
    for event in _full_design():
        if not event["gt_is_malicious"] and flipped < 2:
            event = dict(
                event,
                asserted_match=True,
                asserted_parent_logical_id="client_1",
                asserted_parent_entry_id="fp-1",
            )
            flipped += 1
        events.append(event)
    report = score_events(events, Cohort.ADJUDICATING)
    assert report["components"]["flr_S4"]["value"] == pytest.approx(2 / 15)
    assert report["verdict"] == "FAIL"


def test_one_false_link_in_fifteen_still_passes_the_bar():
    events = []
    flipped = 0
    for event in _full_design():
        if not event["gt_is_malicious"] and flipped < 1:
            event = dict(
                event,
                asserted_match=True,
                asserted_parent_logical_id="client_1",
                asserted_parent_entry_id="fp-1",
            )
            flipped += 1
        events.append(event)
    report = score_events(events, Cohort.ADJUDICATING)
    assert report["components"]["flr_S4"]["value"] == pytest.approx(1 / 15)
    assert report["verdict"] == "PASS"


def test_malicious_wrong_parent_links_depress_recall_and_LEAVE_FLR_UNTOUCHED():
    """Regression: the FLR numerator must be drawn from the FLR denominator's
    population. Mixing malicious wrong-parent links in could push FLR above 1.0
    and fail the false-link bar with zero honest devices ever mislinked."""
    events = []
    rewired = 0
    for event in _full_design():
        odd = int(event["gt_logical_id"].split("_")[1]) % 2 == 1
        if (
            event["scenario"] == "S4_full_mix"
            and event["gt_is_malicious"]
            and odd
            and rewired < 8
        ):
            # Matched, but to a DIFFERENT device.
            event = dict(
                event,
                asserted_parent_logical_id="client_99",
                asserted_parent_entry_id="fp-99",
            )
            rewired += 1
        events.append(event)

    report = score_events(events, Cohort.ADJUDICATING)
    flr = report["components"]["flr_S4"]
    assert flr["numerator"] == 0, "a malicious mislink leaked into the FLR numerator"
    assert flr["value"] == 0.0
    assert flr["denominator"] == 15
    # ...and they DO count against recall.
    assert report["components"]["recall_S4"]["numerator"] == 72
    assert report["components"]["recall_S4"]["value"] == pytest.approx(72 / 80)
    assert report["event_tallies"]["S4_full_mix"]["wrong_malicious_link"] == 8
    # Recall 72/80 = 0.90 still clears its 0.85 bar, so the run PASSES — which is
    # exactly the point. Under the old pooled numerator these same 8 malicious
    # mislinks would have made FLR 8/15 = 0.53 > 0.10 and falsified H3 on the
    # false-link bar without a single honest device ever being mislinked.
    assert report["verdict"] == "PASS"


def test_flr_can_never_exceed_one():
    """Every malicious event mislinked AND every honest event mislinked."""
    events = []
    for event in _full_design():
        events.append(
            dict(
                event,
                asserted_match=True,
                asserted_parent_logical_id="client_99",
                asserted_parent_entry_id="fp-99",
            )
        )
    report = score_events(events, Cohort.ADJUDICATING)
    flr = report["components"]["flr_S4"]
    assert flr["numerator"] == 15 and flr["denominator"] == 15
    assert flr["value"] == 1.0
    assert report["components"]["recall_S3"]["value"] == 0.0


def test_precision_is_reported_but_never_gates():
    report = score_events(_full_design(), Cohort.ADJUDICATING)
    assert "precision_S4" in report["reported_only"]
    assert "precision" not in json.dumps(report["components"])


# ---------------------------------------------------------------------------
# F5 — the VALIDATION cohort informs; it never adjudicates
# ---------------------------------------------------------------------------

def _mislink(events, n, scenario="S4_full_mix"):
    """Rewire `n` held-out malicious S4 events to a DIFFERENT device."""
    out, rewired = [], 0
    for event in events:
        odd = int(event["gt_logical_id"].split("_")[1]) % 2 == 1
        if (
            event["scenario"] == scenario
            and event["gt_is_malicious"]
            and odd
            and rewired < n
        ):
            event = dict(event, asserted_parent_logical_id="client_99",
                         asserted_parent_entry_id="fp-99")
            rewired += 1
        out.append(event)
    return out


def test_the_validation_cohort_records_no_pass_fail_claim():
    """v1.10 § 5.1: the validation run 'reports the metric, but no H3 pass/fail
    is recorded from it'. A bare `verdict: PASS` is exactly the artifact someone
    quotes six weeks later as though H3 had passed."""
    report = score_events(_full_design(), Cohort.VALIDATION)
    assert report["verdict"] is None
    assert report["informs_only"] is True
    assert report["verdict_status"] == "REPORTED (NOT ADJUDICATING)"
    assert report["verdict_withheld_reason"]
    # No top-level field ANYWHERE in the artifact carries the bare claim.
    assert "PASS" not in [v for v in report.values() if isinstance(v, str)]


def test_the_validation_cohort_still_reports_every_metric_value():
    """Only the pass/fail CLAIM is withheld — never the numbers."""
    report = score_events(_full_design(), Cohort.VALIDATION)
    for name in ("recall_S3", "recall_S4", "flr_S4"):
        component = report["components"][name]
        assert component["value"] is not None
        assert component["numerator"] is not None
        assert component["denominator"] > 0
    assert report["reported_only"]["precision_S4"] is not None


def test_the_adjudicating_verdict_shape_is_completely_unchanged():
    report = score_events(_full_design(), Cohort.ADJUDICATING)
    assert report["verdict"] == "PASS"
    assert isinstance(report["verdict"], str)
    assert report["informs_only"] is False
    assert report["verdict_status"] == "PASS"
    assert report["verdict_withheld_reason"] is None


def test_a_failing_validation_cohort_also_records_no_verdict():
    """Withholding must not be a disguised PASS-only rule: FAIL is withheld too."""
    report = score_events(_mislink(_full_design(), 60), Cohort.VALIDATION)
    assert report["components"]["recall_S4"]["status"] == "FAIL"
    assert report["verdict"] is None
    assert report["informs_only"] is True


def test_the_two_cohorts_reports_are_never_shape_confusable():
    validation = score_events(_full_design(), Cohort.VALIDATION)
    adjudicating = score_events(_full_design(), Cohort.ADJUDICATING)
    assert set(validation) == set(adjudicating)          # same keys...
    assert validation["verdict"] != adjudicating["verdict"]   # ...different claim
    assert validation["informs_only"] is not adjudicating["informs_only"]


def test_the_validation_cli_never_prints_a_bare_verdict(tmp_path, capsys):
    path = _write(tmp_path, _full_design())
    out = tmp_path / "h3_relink.json"
    assert main(["--events", str(path), "--cohort", "validation", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "REPORTED (NOT ADJUDICATING)" in printed
    assert "VERDICT (validation): PASS" not in printed
    assert json.loads(out.read_text())["verdict"] is None


# ---------------------------------------------------------------------------
# Addendum B build-lane task — `wrong_malicious_link_rate`
# ---------------------------------------------------------------------------

def test_wrong_malicious_links_are_reported_as_a_rate_on_the_recall_denominator():
    """Addendum B: separating malicious wrong-parent links from FLR must not
    bury them. 8/80 reads differently from 8/180 — the count alone hides that."""
    report = score_events(_mislink(_full_design(), 8), Cohort.ADJUDICATING)
    reported = report["reported_only"]
    assert reported["wrong_malicious_links_S4"] == 8
    assert reported["wrong_malicious_link_rate_S4"] == pytest.approx(8 / 80)
    # ...on exactly the denominator recall uses.
    assert report["components"]["recall_S4"]["denominator"] == 80
    assert reported["wrong_malicious_link_rate_status_S4"] == "REPORTED"


def test_the_rate_uses_the_cohorts_own_denominator():
    """The same 8 mislinks read 8/180 on the validation cohort."""
    events = _mislink(_full_design(), 8)
    validation = score_events(events, Cohort.VALIDATION)
    assert validation["reported_only"]["wrong_malicious_link_rate_S4"] == pytest.approx(8 / 180)


def test_the_rate_is_reported_for_s3_too_because_recall_is_barred_there_too():
    report = score_events(_mislink(_full_design(), 8, scenario="S3_identity_reset_only"),
                          Cohort.ADJUDICATING)
    assert report["reported_only"]["wrong_malicious_links_S3"] == 8
    assert report["reported_only"]["wrong_malicious_link_rate_S3"] == pytest.approx(8 / 80)


def test_a_zero_malicious_denominator_makes_the_rate_inconclusive_not_nan():
    """The file's zero-denominator convention: value None, status INCONCLUSIVE —
    never a NaN, never a 0.0 that reads as 'no mislinks'."""
    honest_only = [e for e in _full_design() if not e["gt_is_malicious"]]
    report = score_events(honest_only, Cohort.VALIDATION, require_design_counts=False)
    reported = report["reported_only"]
    assert report["components"]["recall_S4"]["denominator"] == 0
    assert reported["wrong_malicious_link_rate_S4"] is None
    assert reported["wrong_malicious_link_rate_status_S4"] == "INCONCLUSIVE"
    assert "nan" not in json.dumps(reported).lower()


def test_the_rate_never_gates_any_verdict():
    """F3: reported context, never a bar. 8/80 = 0.10 sits on the FLR bar's
    numeric value and must still not touch the verdict."""
    report = score_events(_mislink(_full_design(), 8), Cohort.ADJUDICATING)
    assert report["components"]["recall_S4"]["value"] == pytest.approx(72 / 80)
    assert report["verdict"] == "PASS"
    assert "wrong_malicious_link" not in json.dumps(report["components"])
    # Nothing in the reported-only block carries a pass/fail status of its own.
    statuses = [
        v for k, v in report["reported_only"].items()
        if k.startswith("wrong_malicious_link_rate_status")
    ]
    assert statuses and not ({"PASS", "FAIL"} & set(statuses))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_writes_a_report(tmp_path, capsys):
    path = _write(tmp_path, _full_design())
    out = tmp_path / "h3_relink.json"
    assert main(["--events", str(path), "--cohort", "adjudicating", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PASS"
    assert "recall_S3" in capsys.readouterr().out


def test_main_reports_a_refusal_without_a_traceback(tmp_path, capsys):
    path = _write(tmp_path, _full_design()[:-1])
    out = tmp_path / "h3_relink.json"
    assert main(["--events", str(path), "--cohort", "validation", "--out", str(out)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()

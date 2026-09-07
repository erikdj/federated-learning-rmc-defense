"""The corrected-H3 IDENTITY scorer (amendment § 2, RATIFIED rev-5, v1.50).

The instrument under test is the identity-only registry: every session enrolls,
and a re-entry considers ALL sessions first seen strictly before it. What the
scorer measures is therefore P1 = rank-1 identification (is the nearest
candidate the correct base device?) and P2 = wrong-device link rate at the
locked tau. An honest device correctly re-identified as ITSELF is a CORRECT
decision, never a false link — that redefinition is the whole point of § 2.

These tests pin the mechanics the D7 scorer already froze (base-partition
identity, the odd-partition adjudicating filter, dedup, zero-denominator
INCONCLUSIVE, the exact census gate, validation-never-adjudicates) plus the two
things that are new: the identity_only provenance refusal, and the honest
handling of what the schema-v5 event row CANNOT answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_h3_identity import (
    P1_RANK1_BAR,
    P2_WRONG_DEVICE_BAR,
    REQUIRED_REGISTRY_POLICY,
    Cohort,
    ScoringError,
    classify_event,
    load_provenance,
    main,
    score_events,
)

S3 = "S3_identity_reset_only"
S4 = "S4_full_mix"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _locked_custody(cohort):
    """Custody fields matching THE locked instrument for `cohort`.

    Built from the hash-verified lock itself rather than hardcoded strings, so
    these tests stay green across a re-lock (v1 -> v2): the scorer requires
    EXACT equality with whatever instrument is locked at scoring time.
    An invalid cohort yields null fields — the declared-cohort refusal fires
    before any of them are read.
    """
    from flowerfl.fingerprint_registry import (
        CALIBRATION_ARTIFACT_SHA256,
        CalibrationCohort,
        locked_metric,
        locked_tau,
    )

    try:
        cohort_enum = CalibrationCohort(cohort)
    except (TypeError, ValueError):
        return {
            "metric_provenance": None,
            "tau": None,
            "calibration_artifact_sha256": None,
        }
    return {
        "metric_provenance": locked_metric(cohort_enum).provenance,
        "tau": locked_tau(cohort_enum),
        "calibration_artifact_sha256": CALIBRATION_ARTIFACT_SHA256,
    }

def _uid(scenario=S4, seed=42):
    """The run identity every event and its result JSON must agree on."""
    return f"sim__{scenario}__tgefp__seed{seed}__20260814T101500Z"


def _locked_tau_value(cohort="adjudicating"):
    from flowerfl.fingerprint_registry import CalibrationCohort, locked_tau

    return locked_tau(CalibrationCohort(cohort))


def _event(key, gt, *, scenario=S4, matched=True, parent=None, min_d=5.0,
           malicious=True, tau=None, seed=42, run_uid=None):
    # Event rows must carry their bound unit's verified locked tau (the scorer
    # refuses a row decided at any other tau). Default = the adjudicating lock;
    # validation-cohort corpora pass tau explicitly.
    if tau is None:
        tau = _locked_tau_value("adjudicating")
    uid = run_uid or _uid(scenario, seed)
    return {
        "run_uid": uid,
        # the frozen contract: f"{run_uid}:{server_round}:{current_cid}"
        "reentry_event_key": f"{uid}:7:cid-{key}",
        "server_round": 7,
        "current_cid": f"cid-{key}",
        "gt_logical_id": gt,
        "gt_is_malicious": malicious,
        "asserted_match": matched,
        "asserted_parent_entry_id": "fp-0001" if matched else None,
        "asserted_parent_logical_id": parent,
        "min_d": min_d,
        "tau": tau,
        "generation": 1,
        "scenario": scenario,
        "h3_arm": "tgefp",
        "seed": seed,
    }


def _rebind(event, run_uid, seed=None):
    """Relabel an event onto another run COMPLETELY: run_uid, the contract key
    rebuilt from it, and optionally the seed. Tests that want the DEEPER gates
    (census, matrix, scenario/seed binding) to fire must forge all of it —
    a partial relabel is what the key-integrity check now catches."""
    event["run_uid"] = run_uid
    event["reentry_event_key"] = (
        f"{run_uid}:{event['server_round']}:{event['current_cid']}"
    )
    if seed is not None:
        event["seed"] = seed
    return event


def _result_json(tmp_path, name, *, policy="identity_only", scenario=S4, seed=42,
                 custody_policy=None, cohort="adjudicating", run_uid=None):
    payload = {
        "config": "TGE+FP",
        "seed": seed,
        "provenance": {
            "scenario_path": f"rmc/scenarios/{scenario}.json",
            "fp_cohort": cohort,
            "fp_registry_policy": policy,
        },
        "fingerprint_registry": {
            "registry_policy": custody_policy if custody_policy else policy,
            "tau_posture": "locked",
            "run_uid": run_uid or _uid(scenario, seed),
            **_locked_custody(cohort),
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# The RATIFIED bars and the policy requirement
# ---------------------------------------------------------------------------

def test_the_ratified_bars_are_the_amendment_values():
    assert P1_RANK1_BAR == 0.85
    assert P2_WRONG_DEVICE_BAR == 0.10


def test_the_scorer_requires_the_identity_only_registry():
    assert REQUIRED_REGISTRY_POLICY == "identity_only"


def test_provenance_refuses_a_flag_gated_run(tmp_path):
    path = _result_json(tmp_path, "unit.json", policy="flag_gated")
    with pytest.raises(ScoringError, match="identity_only"):
        load_provenance([path])


def test_provenance_refuses_an_undeclared_run(tmp_path):
    path = _result_json(tmp_path, "unit.json", policy=None)
    with pytest.raises(ScoringError, match="identity_only"):
        load_provenance([path])


def test_provenance_refuses_when_custody_contradicts_the_declaration(tmp_path):
    """The run-config DECLARED identity_only but the registry actually ran
    flag_gated. Reading only the declaration would score the wrong instrument."""
    path = _result_json(tmp_path, "unit.json", policy="identity_only",
                        custody_policy="flag_gated")
    with pytest.raises(ScoringError, match="disagree"):
        load_provenance([path])


def test_provenance_accepts_an_identity_only_run(tmp_path):
    path = _result_json(tmp_path, "unit.json")
    report = load_provenance([path])
    assert report["registry_policy"] == "identity_only"
    assert report["units"][0]["scenario"] == S4
    assert report["units"][0]["seed"] == 42


def test_scoring_refuses_events_whose_units_have_no_provenance(tmp_path):
    """Every event must bind to a supplied result — otherwise the policy
    assertion silently applies to a subset of the corpus."""
    provenance = load_provenance([_result_json(tmp_path, "u.json", seed=42)])
    events = [_event("e1", "client_1_new1", seed=137)]
    with pytest.raises(ScoringError, match="run_uid"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


# ---------------------------------------------------------------------------
# Classification — including what the schema cannot answer
# ---------------------------------------------------------------------------

def test_a_matched_event_on_the_same_base_device_is_rank1_correct():
    event = _event("e", "client_1_new1", parent="client_1")
    assert classify_event(event) == "rank1_correct"


def test_identity_is_compared_at_base_partition_granularity():
    """`client_1_new2` and `client_1_new1` are the same device (B1)."""
    event = _event("e", "client_1_new2", parent="client_1_new1")
    assert classify_event(event) == "rank1_correct"


def test_a_matched_event_on_another_device_is_a_wrong_device_link():
    event = _event("e", "client_1_new1", parent="client_3")
    assert classify_event(event) == "rank1_wrong_device"


def test_an_honest_device_re_identified_as_itself_is_CORRECT():
    """§ 2's redefinition: the EXP-056 'false links' were all
    self-re-identifications, which under an identity-only instrument is the
    instrument working."""
    event = _event("e", "client_13_new1", parent="client_13", malicious=False)
    assert classify_event(event) == "rank1_correct"


def test_an_event_with_no_candidate_at_all_is_a_determinate_miss():
    """`min_d` null means the pool was empty: there was no nearest candidate, so
    rank-1 cannot have been correct."""
    event = _event("e", "client_1_new1", matched=False, parent=None, min_d=None)
    assert classify_event(event) == "rank1_no_candidate"


def test_an_unmatched_event_with_a_finite_distance_is_INDETERMINATE():
    """THE SCHEMA LIMIT. A nearest candidate existed (min_d is finite) but the
    row records its identity only when the match fired, so rank-1 — which is
    threshold-FREE — cannot be read off this row either way."""
    event = _event("e", "client_1_new1", matched=False, parent=None, min_d=310.0)
    assert classify_event(event) == "rank1_indeterminate"


def test_a_matched_event_with_no_parent_is_a_hard_error():
    event = _event("e", "client_1_new1", matched=True, parent=None)
    with pytest.raises(ScoringError, match="asserted_parent_logical_id"):
        classify_event(event)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _clean_corpus(tmp_path, n_s3=4, n_s4=4, honest=2, cohort="adjudicating"):
    """A small all-correct corpus with matching provenance."""
    tau = _locked_tau_value(cohort)
    events = []
    for index in range(n_s3):
        events.append(_event(f"s3-{index}", f"client_{2*index+1}_new1",
                             scenario=S3, parent=f"client_{2*index+1}", tau=tau))
    for index in range(n_s4):
        events.append(_event(f"s4-{index}", f"client_{2*index+1}_new1",
                             scenario=S4, parent=f"client_{2*index+1}", tau=tau))
    for index in range(honest):
        events.append(_event(f"h-{index}", f"client_{2*index+1}_new1",
                             scenario=S4, parent=f"client_{2*index+1}",
                             malicious=False, tau=tau))
    provenance = load_provenance([
        _result_json(tmp_path, "s3.json", scenario=S3, cohort=cohort),
        _result_json(tmp_path, "s4.json", scenario=S4, cohort=cohort),
    ])
    return events, provenance


def test_a_perfect_corpus_passes_every_component(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S3"]["value"] == 1.0
    assert report["components"]["rank1_S4"]["value"] == 1.0
    assert report["components"]["wrong_device_link_rate_malicious"]["value"] == 0.0
    assert report["components"]["wrong_device_link_rate_honest"]["value"] == 0.0
    assert report["component_conjunction"] == "PASS"
    assert report["verdict"] is None       # diagnostic runs never carry a verdict


def test_the_adjudicating_cohort_keeps_only_odd_partitions(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("even", "client_2_new1", parent="client_2"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["n_events_scored"] == len(events) - 1


def test_the_validation_cohort_records_no_pass_fail_claim(tmp_path):
    events, provenance = _clean_corpus(tmp_path, cohort="validation")
    report = score_events(events, Cohort.VALIDATION, provenance,
                          require_design_counts=False)
    assert report["verdict"] is None
    assert report["informs_only"] is True
    assert "NOT ADJUDICATING" in report["verdict_status"]
    # the VALUES are all still reported
    assert report["components"]["rank1_S3"]["value"] == 1.0


def test_an_indeterminate_event_makes_rank1_INCONCLUSIVE(tmp_path):
    """Never an auto-pass and never an auto-fail: if the corpus cannot answer
    rank-1 for an event, the component cannot be adjudicated."""
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("ind", "client_5_new1", scenario=S4,
                         matched=False, parent=None, min_d=310.0))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    component = report["components"]["rank1_S4"]
    assert component["status"] == "INCONCLUSIVE"
    assert component["indeterminate"] == 1
    assert report["component_conjunction"] == "INCONCLUSIVE"
    assert report["verdict"] is None


def test_a_no_candidate_event_counts_against_rank1_without_blocking_it(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("nc", "client_5_new1", scenario=S4,
                         matched=False, parent=None, min_d=None))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    component = report["components"]["rank1_S4"]
    assert component["status"] in ("PASS", "FAIL")
    assert component["indeterminate"] == 0
    # malicious-only: 4 correct + this no-candidate miss; the 2 honest events
    # are reported separately and are NOT in this denominator
    assert component["denominator"] == 5
    assert component["numerator"] == 4


def test_a_wrong_device_link_hits_both_components(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("wd", "client_5_new1", scenario=S4, parent="client_7"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S4"]["value"] < 1.0
    assert report["components"]["wrong_device_link_rate_malicious"]["value"] > 0.0


def test_honest_self_relinks_never_enter_the_wrong_device_numerator(tmp_path):
    events, provenance = _clean_corpus(tmp_path, honest=5)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["wrong_device_link_rate_honest"]["numerator"] == 0
    assert report["components"]["wrong_device_link_rate_malicious"]["numerator"] == 0
    assert report["event_tallies"][S4]["rank1_correct"] == 9


def test_a_zero_denominator_is_INCONCLUSIVE_never_a_pass(tmp_path):
    provenance = load_provenance([_result_json(tmp_path, "s4.json", scenario=S4)])
    events = [_event("s4-0", "client_1_new1", scenario=S4, parent="client_1")]
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S3"]["status"] == "INCONCLUSIVE"
    assert report["components"]["rank1_S3"]["value"] is None
    assert report["component_conjunction"] == "INCONCLUSIVE"
    assert report["verdict"] is None


def test_the_census_gate_requires_the_exact_design_counts(tmp_path):
    # The unit-matrix gate now fires FIRST on this
    # 2-unit corpus; the aggregate-census gate still backstops it.
    events, provenance = _clean_corpus(tmp_path)
    with pytest.raises(ScoringError, match="census|unit matrix"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_an_unexpected_scenario_is_refused(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("x", "client_1_new1", scenario="S0_clean_baseline",
                         parent="client_1"))
    with pytest.raises(ScoringError, match="scenario"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


# ---------------------------------------------------------------------------
# What the schema CANNOT support — declared, never fabricated
# ---------------------------------------------------------------------------

def test_the_report_declares_the_non_derivable_blocks(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    absent = report["not_derivable"]
    assert "cmc_rank_2_3" in absent
    for entry in absent.values():
        assert entry["value"] is None
        assert entry["reason"]


def test_distance_percentiles_ARE_derivable_and_reported(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    distances = report["reported_only"]["min_d_percentiles"]
    assert distances["n"] == 10
    assert distances["p50"] == pytest.approx(5.0)


def test_per_device_rank1_is_reported_with_its_indeterminate_count(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("ind", "client_5_new1", scenario=S4,
                         matched=False, parent=None, min_d=310.0))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    per_device = report["reported_only"]["per_device_rank1"]
    assert per_device["5"]["indeterminate"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_scores_a_corpus_and_writes_the_report(tmp_path, capsys):
    events, _ = _clean_corpus(tmp_path)
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    out = tmp_path / "report.json"
    code = main([
        "--events", str(events_path),
        "--provenance", str(tmp_path / "s3.json"), str(tmp_path / "s4.json"),
        "--cohort", "adjudicating",
        "--out", str(out),
        "--allow-partial-census",
    ])
    assert code == 0
    captured = capsys.readouterr().out
    assert "rank1_S3" in captured
    assert "VERDICT" in captured
    report = json.loads(out.read_text())
    assert report["registry_policy"] == "identity_only"


def test_main_refuses_a_flag_gated_corpus(tmp_path, capsys):
    events, _ = _clean_corpus(tmp_path)
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    bad = _result_json(tmp_path, "bad.json", policy="flag_gated")
    code = main([
        "--events", str(events_path),
        "--provenance", str(bad),
        "--cohort", "adjudicating",
        "--allow-partial-census",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err


def test_the_frozen_d7_scorer_is_not_imported_for_its_decision_statistic():
    """This scorer REUSES the D7 loader (arm rule, dedup, provenance refusals)
    but must never inherit its recall/FLR statistic — different instrument,
    different question."""
    import scripts.analyze_h3_identity as identity

    # the D7 decision statistic is absent from this module's namespace
    for name in ("RECALL_BAR", "FLR_BAR", "FALSE_LINK_POPULATION"):
        assert not hasattr(identity, name), name
    # and the shared machinery it DOES import is only the loader/cohort/filter
    assert identity.classify_event.__module__ == identity.__name__
    assert identity.score_events.__module__ == identity.__name__
    assert identity.load_events.__module__ == "scripts.analyze_h3_relink"


# ---------------------------------------------------------------------------
# The nearest-candidate extension: rank-1 becomes fully determinate
# ---------------------------------------------------------------------------

def _nearest(key, gt, nearest, *, scenario=S4, matched=False, min_d=310.0,
             malicious=True, seed=42):
    event = _event(key, gt, scenario=scenario, matched=matched, min_d=min_d,
                   malicious=malicious, seed=seed,
                   parent=nearest if matched else None)
    event["nearest_entry_id"] = "fp-0007" if nearest else None
    event["nearest_logical_id"] = nearest
    return event


def test_an_unmatched_event_with_a_recorded_nearest_is_DETERMINATE():
    event = _nearest("e", "client_1_new1", "client_1")
    assert classify_event(event) == "rank1_correct"


def test_an_unmatched_event_whose_nearest_is_another_device_is_wrong_device():
    event = _nearest("e", "client_1_new1", "client_3")
    assert classify_event(event) == "rank1_wrong_device"


def test_an_explicit_null_nearest_still_means_no_candidate():
    event = _nearest("e", "client_1_new1", None, min_d=None)
    assert classify_event(event) == "rank1_no_candidate"


def test_a_legacy_row_without_the_field_keeps_the_indeterminate_path():
    """Pre-extension corpora must still be scoreable, and still honest."""
    event = _event("e", "client_1_new1", matched=False, parent=None, min_d=310.0)
    assert "nearest_logical_id" not in event
    assert classify_event(event) == "rank1_indeterminate"


def test_a_matched_event_prefers_the_frozen_asserted_parent():
    """`asserted_parent_*` remains authoritative when a match fired — the two
    agree by construction, and reading `nearest_*` instead would silently
    change which field the verdict depends on."""
    event = _nearest("e", "client_1_new1", "client_1", matched=True, min_d=5.0)
    assert event["asserted_parent_logical_id"] == "client_1"
    assert classify_event(event) == "rank1_correct"


def test_an_enriched_corpus_makes_P1_conclusive(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_nearest("far", "client_5_new1", "client_5"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    component = report["components"]["rank1_S4"]
    assert component["indeterminate"] == 0
    assert component["status"] == "PASS"
    assert report["component_conjunction"] == "PASS"


def test_an_UNMATCHED_wrong_device_nearest_does_NOT_count_as_a_link(tmp_path):
    """P2 is a rate of LINKS at the locked tau. A nearest candidate that never
    cleared tau produced no link, so it must depress P1 without touching P2 —
    otherwise the extension would manufacture guard failures out of events the
    instrument correctly declined to link."""
    events, provenance = _clean_corpus(tmp_path)
    events.append(_nearest("far-wrong", "client_5_new1", "client_7"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S4"]["value"] < 1.0
    assert report["components"]["wrong_device_link_rate_malicious"]["numerator"] == 0
    assert report["components"]["wrong_device_link_rate_malicious"]["value"] == 0.0


def test_a_MATCHED_wrong_device_link_still_counts_against_the_guard(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_nearest("near-wrong", "client_5_new1", "client_7",
                           matched=True, min_d=5.0))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["wrong_device_link_rate_malicious"]["numerator"] == 1


def test_the_report_declares_whether_the_corpus_carries_the_nearest_pair(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    plain = score_events(events, Cohort.ADJUDICATING, provenance,
                         require_design_counts=False)
    assert plain["nearest_candidate_coverage"]["n_with_nearest_field"] == 0
    assert plain["nearest_candidate_coverage"]["complete"] is False

    enriched = [
        _nearest(f"n-{i}", f"client_{2 * i + 1}_new1", f"client_{2 * i + 1}",
                 matched=True, min_d=5.0)
        for i in range(4)
    ]
    report = score_events(enriched, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["nearest_candidate_coverage"]["complete"] is True


def test_cmc_stays_non_derivable_even_with_the_nearest_field(tmp_path):
    """The extension records the nearest candidate, not the ordered list —
    rank 2 and beyond are still unrecorded."""
    events, provenance = _clean_corpus(tmp_path)
    events.append(_nearest("far", "client_5_new1", "client_5"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["not_derivable"]["cmc_rank_2_3"]["value"] is None


# ---------------------------------------------------------------------------
# Provenance binds to events by exact run_uid
# ---------------------------------------------------------------------------
# A (scenario, seed) join is not a run identity. An identity_only result JSON
# and a flag_gated event log from the SAME design cell would satisfy it, and the
# corrected-instrument verdict would be computed over the wrong instrument's
# events. `run_uid` is the run identity the events themselves are keyed on:
# `reentry_event_key` is literally f"{run_uid}:{round}:{cid}".

RUN_A = "sim__S4_full_mix__tgefp__seed42__20260814T101500Z"
RUN_B = "sim__S4_full_mix__tgefp__seed42__20260814T2359ZZ"


def _result_with_run(tmp_path, name, run_uid, *, policy="identity_only",
                     scenario=S4, seed=42, cohort="adjudicating"):
    payload = {
        "config": "TGE+FP",
        "seed": seed,
        "provenance": {
            "scenario_path": f"rmc/scenarios/{scenario}.json",
            "fp_cohort": cohort,
            "fp_registry_policy": policy,
        },
        "fingerprint_registry": {
            "registry_policy": policy,
            "tau_posture": "locked",
            "run_uid": run_uid,
            **_locked_custody(cohort),
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _run_event(key, gt, run_uid, **over):
    return _event(key, gt, run_uid=run_uid, **over)


def test_provenance_records_the_run_uid(tmp_path):
    report = load_provenance([_result_with_run(tmp_path, "u.json", RUN_A)])
    assert report["units"][0]["run_uid"] == RUN_A


def test_provenance_refuses_a_unit_with_no_run_uid(tmp_path):
    payload = json.loads(_result_json(tmp_path, "u.json").read_text())
    payload["fingerprint_registry"].pop("run_uid")   # a pre-binding custody block
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ScoringError, match="run_uid"):
        load_provenance([path])


def test_provenance_refuses_two_units_sharing_a_run_uid(tmp_path):
    with pytest.raises(ScoringError, match="ambiguous|duplicate") as excinfo:
        load_provenance([
            _result_with_run(tmp_path, "a.json", RUN_A),
            _result_with_run(tmp_path, "b.json", RUN_A),
        ])
    # the refusal must not echo the run_uid (it embeds the launch seed)
    assert RUN_A not in str(excinfo.value)
    assert "redacted" in str(excinfo.value)


def test_events_from_an_unbound_run_are_REFUSED(tmp_path):
    """THE P1-a CASE: identity_only provenance and events from a DIFFERENT run
    of the same (scenario, seed) cell. The old join passed this."""
    provenance = load_provenance([_result_with_run(tmp_path, "u.json", RUN_A)])
    events = [_run_event("e1", "client_1_new1", RUN_B, parent="client_1")]
    with pytest.raises(ScoringError, match="run_uid"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_events_bound_to_the_declared_run_are_scored(tmp_path):
    provenance = load_provenance([_result_with_run(tmp_path, "u.json", RUN_A)])
    events = [_run_event("e1", "client_1_new1", RUN_A, parent="client_1")]
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["n_events_scored"] == 1


def test_an_event_with_no_run_uid_is_REFUSED(tmp_path):
    """Unbindable is not the same as bound; it must never be scored."""
    provenance = load_provenance([_result_with_run(tmp_path, "u.json", RUN_A)])
    event = _event("e1", "client_1_new1", parent="client_1")
    event.pop("run_uid")           # a corpus extracted before the binding landed
    with pytest.raises(ScoringError, match="run_uid"):
        score_events([event], Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


# ---------------------------------------------------------------------------
# The scoring cohort must match the unit's locked tau/Sigma
# ---------------------------------------------------------------------------
# A validation-cohort unit's events, filtered to odd partitions, can satisfy the
# adjudicating design counts exactly — and would then be adjudicated under the
# WRONG tau/Sigma. The two cohorts are different instruments (D9 axis ii).

def test_a_validation_unit_scored_as_adjudicating_is_REFUSED(tmp_path):
    provenance = load_provenance([
        _result_with_run(tmp_path, "u.json", RUN_A, cohort="validation")
    ])
    events = [_run_event("e1", "client_1_new1", RUN_A, parent="client_1")]
    with pytest.raises(ScoringError, match="fp_cohort"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_an_adjudicating_unit_scored_as_validation_is_REFUSED(tmp_path):
    provenance = load_provenance([
        _result_with_run(tmp_path, "u.json", RUN_A, cohort="adjudicating")
    ])
    events = [_run_event("e1", "client_1_new1", RUN_A, parent="client_1")]
    with pytest.raises(ScoringError, match="fp_cohort"):
        score_events(events, Cohort.VALIDATION, provenance,
                     require_design_counts=False)


def test_a_matching_cohort_is_accepted(tmp_path):
    provenance = load_provenance([
        _result_with_run(tmp_path, "u.json", RUN_A, cohort="adjudicating")
    ])
    events = [_run_event("e1", "client_1_new1", RUN_A, parent="client_1")]
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["cohort"] == "adjudicating"


def test_a_cohort_the_custody_metric_contradicts_is_REFUSED(tmp_path):
    """`metric_provenance` records which locked Sigma the run ACTUALLY loaded,
    so a declaration that disagrees with it means the unit did not run the
    cohort it claims."""
    payload = json.loads(
        _result_with_run(tmp_path, "u.json", RUN_A, cohort="adjudicating").read_text()
    )
    payload["fingerprint_registry"]["metric_provenance"] = (
        "h3-calibration/validation/pooled_within_ledoit_wolf"
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ScoringError, match="does not EQUAL the locked"):
        load_provenance([path])


def test_a_unit_with_no_declared_cohort_is_REFUSED(tmp_path):
    payload = json.loads(
        _result_with_run(tmp_path, "u.json", RUN_A).read_text()
    )
    payload["provenance"]["fp_cohort"] = None
    path = tmp_path / "nocohort.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ScoringError, match="fp_cohort"):
        load_provenance([path])


# ===========================================================================
# Population-aligned P1/P2 denominators
# ===========================================================================
# P1 pooled malicious and honest events into one denominator, and P2 pooled
# both populations into one guard. The failure mode: with 15 honest events
# in 175, EVERY honest device could be linked to the WRONG device and the guard
# still read 15/175 = 0.086 <= 0.10 — a PASS. The two populations are different
# sizes and different questions, so they get different denominators.

def test_P1_denominators_are_MALICIOUS_ONLY(tmp_path):
    events, provenance = _clean_corpus(tmp_path, n_s3=4, n_s4=4, honest=2)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S3"]["denominator"] == 4
    assert report["components"]["rank1_S4"]["denominator"] == 4  # NOT 4 + 2


def test_honest_rank1_is_reported_separately_never_in_P1(tmp_path):
    events, provenance = _clean_corpus(tmp_path, n_s3=4, n_s4=4, honest=2)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    honest = report["reported_only"]["honest_rank1"]
    assert honest["denominator"] == 2
    assert honest["numerator"] == 2
    assert "bar" not in honest      # reported, never a component


def test_the_design_counts_are_the_aligned_populations():
    from scripts.analyze_h3_identity import DESIGN_COUNTS

    adjudicating = DESIGN_COUNTS[Cohort.ADJUDICATING]
    assert adjudicating["rank1_S3"] == 80
    assert adjudicating["rank1_S4"] == 80
    assert adjudicating["wrong_device_link_rate_malicious"] == 160
    assert adjudicating["wrong_device_link_rate_honest"] == 15

    validation = DESIGN_COUNTS[Cohort.VALIDATION]
    assert validation["rank1_S3"] == 180
    assert validation["rank1_S4"] == 180
    assert validation["wrong_device_link_rate_malicious"] == 360
    assert validation["wrong_device_link_rate_honest"] == 15


def test_P2_is_TWO_components_each_at_the_same_bar(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    components = report["components"]
    assert set(components) == {
        "rank1_S3", "rank1_S4",
        "wrong_device_link_rate_malicious", "wrong_device_link_rate_honest",
    }
    for name in ("wrong_device_link_rate_malicious",
                 "wrong_device_link_rate_honest"):
        assert components[name]["bar"] == P2_WRONG_DEVICE_BAR
        assert components[name]["direction"] == "at_most"


def test_THE_HOLE_all_honest_wrong_linked_now_FAILS(tmp_path):
    """Regression case: every honest event linked to the WRONG device.
    Pooled, that was 15/175 = 0.086 and PASSED. Split, it is 15/15 = 1.0."""
    events, provenance = _clean_corpus(tmp_path, n_s3=4, n_s4=4, honest=0)
    for index in range(5):
        events.append(_event(f"h-{index}", f"client_{2 * index + 1}_new1",
                             scenario=S4, parent="client_17", malicious=False))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    honest_guard = report["components"]["wrong_device_link_rate_honest"]
    assert honest_guard["numerator"] == 5
    assert honest_guard["denominator"] == 5
    assert honest_guard["value"] == 1.0
    assert honest_guard["status"] == "FAIL"
    assert report["component_conjunction"] == "FAIL"
    # and the malicious guard is untouched by it
    assert report["components"]["wrong_device_link_rate_malicious"]["value"] == 0.0


def test_the_verdict_is_the_conjunction_of_all_four(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["component_conjunction"] == "PASS"
    assert "rank1_S3" in report["verdict_rule"]
    assert "wrong_device_link_rate_honest" in report["verdict_rule"]


def test_a_malicious_wrong_link_hits_only_the_malicious_guard(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("mw", "client_5_new1", scenario=S4, parent="client_7"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["wrong_device_link_rate_malicious"]["numerator"] == 1
    assert report["components"]["wrong_device_link_rate_honest"]["numerator"] == 0


def test_an_unmatched_wrong_nearest_still_depresses_P1_ONLY(tmp_path):
    """The _is_wrong_device_link decoupling stands after the split."""
    events, provenance = _clean_corpus(tmp_path)
    events.append(_nearest("far-wrong", "client_5_new1", "client_7"))
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S4"]["value"] < 1.0
    assert report["components"]["wrong_device_link_rate_malicious"]["numerator"] == 0
    assert report["components"]["wrong_device_link_rate_honest"]["numerator"] == 0


# ---------------------------------------------------------------------------
# Clopper-Pearson bounds, per component
# ---------------------------------------------------------------------------

def test_every_component_carries_a_one_sided_95_clopper_pearson_bound(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    for name, component in report["components"].items():
        bound = component["ci95"]
        assert bound["method"] == "clopper_pearson_one_sided_95"
        assert bound["bound"] is not None, name
        assert 0.0 <= bound["bound"] <= 1.0


def test_the_bound_side_follows_the_components_direction(tmp_path):
    """A guard is bounded ABOVE (its failure is a high rate); a rank-1 bar is
    bounded BELOW (its failure is a low rate). An upper bound on P1 would be
    the anti-conservative direction."""
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["components"]["rank1_S3"]["ci95"]["bound_side"] == "lower"
    assert (report["components"]["wrong_device_link_rate_honest"]["ci95"]
            ["bound_side"] == "upper")


def test_a_rank1_component_also_reports_the_error_rate_upper_bound(tmp_path):
    """Stated as an UPPER bound too, on the quantity where 'upper' is the
    conservative direction: the miss rate."""
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    component = report["components"]["rank1_S3"]
    assert component["ci95"]["error_rate_upper"] == pytest.approx(
        1.0 - component["ci95"]["bound"]
    )


def test_a_zero_numerator_guard_has_a_nonzero_upper_bound(tmp_path):
    """0/15 does not mean the true rate is 0 — the exact bound is what makes
    the N=15 resolution honest."""
    events, provenance = _clean_corpus(tmp_path, honest=15)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    guard = report["components"]["wrong_device_link_rate_honest"]
    assert guard["numerator"] == 0
    assert guard["value"] == 0.0
    assert guard["ci95"]["bound"] > 0.15   # 1 - 0.05**(1/15) ~= 0.181


def test_the_clopper_pearson_upper_bound_matches_the_closed_form(tmp_path):
    from scripts.analyze_h3_identity import _clopper_pearson

    # k = 0: the exact one-sided upper bound is 1 - alpha**(1/n)
    assert _clopper_pearson(0, 15, side="upper") == pytest.approx(
        1.0 - 0.05 ** (1 / 15)
    )
    # k = n: the rate is bounded above by 1.0
    assert _clopper_pearson(15, 15, side="upper") == 1.0
    # k = n: the exact one-sided LOWER bound is alpha**(1/n)
    assert _clopper_pearson(15, 15, side="lower") == pytest.approx(
        0.05 ** (1 / 15)
    )
    assert _clopper_pearson(0, 15, side="lower") == 0.0


def test_the_honest_guard_discloses_its_exact_count_resolution(tmp_path):
    events, provenance = _clean_corpus(tmp_path, honest=15)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    note = report["components"]["wrong_device_link_rate_honest"]["resolution_note"]
    assert "15" in note
    assert "at most 1" in note


def test_no_false_accept_or_FMR_language_survives():
    """The guard counts WRONG-DEVICE links. 'False accept' / FMR is a
    verification-task term and describes a different quantity."""
    import scripts.analyze_h3_identity as identity

    source = Path(identity.__file__).read_text().lower()
    for term in ("false accept", "false-accept", "fmr", "false match rate"):
        assert term not in source, term


# ===========================================================================
# locked-instrument corroboration is FAIL-CLOSED
# ===========================================================================
# The old check parsed `h3-calibration/{cohort}/...` and SKIPPED when the
# custody value was absent or malformed — the exact boundary where a raw-180
# or non-locked metric slips into scoring behind a correct policy declaration.
# Corroboration is now exact equality against the hash-verified lock, on THREE
# custody fields, and an unresolved value refuses.

def _mutated_unit(tmp_path, name, registry_overrides=None, drop=(), cohort=None):
    path = _result_with_run(tmp_path, name + ".base.json", RUN_A)
    payload = json.loads(path.read_text())
    for key in drop:
        payload["fingerprint_registry"].pop(key, None)
    for key, value in (registry_overrides or {}).items():
        payload["fingerprint_registry"][key] = value
    if cohort is not None:
        payload["provenance"]["fp_cohort"] = cohort
    out = tmp_path / name
    out.write_text(json.dumps(payload))
    return out


def test_a_unit_with_NO_metric_provenance_is_REFUSED(tmp_path):
    """The old parse-based check returned None here and WAVED THE UNIT THROUGH."""
    path = _mutated_unit(tmp_path, "nomp.json", drop=("metric_provenance",))
    with pytest.raises(ScoringError, match="metric_provenance"):
        load_provenance([path])


@pytest.mark.parametrize("bad", [
    "identity",                                   # unlocked identity metric
    "restored",                                   # from_dict fallback marker
    "",                                           # empty
    "h3-calibration/adjudicating",                # truncated — no metric segment
    "h3-calibration/adjudicating/wrong_metric",   # right cohort, WRONG estimator
])
def test_malformed_or_wrong_metric_provenance_is_REFUSED(tmp_path, bad):
    """`h3-calibration/adjudicating/wrong_metric` parsed to the RIGHT cohort
    under the old check and passed — the estimator segment was never compared."""
    path = _mutated_unit(tmp_path, "badmp.json",
                         registry_overrides={"metric_provenance": bad})
    with pytest.raises(ScoringError, match="does not EQUAL the locked"):
        load_provenance([path])


def test_a_unit_with_a_drifted_tau_is_REFUSED(tmp_path):
    path = _mutated_unit(tmp_path, "badtau.json",
                         registry_overrides={"tau": 999.0})
    with pytest.raises(ScoringError, match="tau"):
        load_provenance([path])


def test_a_unit_with_NO_tau_is_REFUSED(tmp_path):
    path = _mutated_unit(tmp_path, "notau.json", drop=("tau",))
    with pytest.raises(ScoringError, match="tau"):
        load_provenance([path])


def test_a_unit_with_a_wrong_artifact_hash_is_REFUSED(tmp_path):
    path = _mutated_unit(
        tmp_path, "badsha.json",
        registry_overrides={"calibration_artifact_sha256": "0" * 64},
    )
    with pytest.raises(ScoringError, match="calibration_artifact_sha256"):
        load_provenance([path])


def test_a_unit_with_NO_artifact_hash_is_REFUSED(tmp_path):
    """Units from images predating the custody field are not valid
    corrected-instrument units — absence counts as a mismatch."""
    path = _mutated_unit(tmp_path, "nosha.json",
                         drop=("calibration_artifact_sha256",))
    with pytest.raises(ScoringError, match="calibration_artifact_sha256"):
        load_provenance([path])


def test_a_garbage_cohort_is_REFUSED_as_not_a_cohort(tmp_path):
    path = _mutated_unit(tmp_path, "badcohort.json", cohort="dev")
    with pytest.raises(ScoringError, match="not a calibration cohort"):
        load_provenance([path])


# ===========================================================================
# malicious P1 cannot be inflated by honest rows
# ===========================================================================
# The old implementation derived malicious rank-1 counts by subtracting honest
# totals from a pooled tally, and subtracted them from S4 ONLY — an honest row
# misplaced into S3 kept its correct outcome inside malicious_correct[S3] while
# the denominator stayed malicious-only. Counting is now direct, an honest row
# outside S4 refuses, and an impossible numerator refuses at the component.

def test_an_honest_S3_event_is_REFUSED_not_tallied(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    events.append(_event("h-s3", "client_9_new1", scenario=S3,
                         parent="client_9", malicious=False))
    with pytest.raises(ScoringError, match="honest re-entry event"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_an_impossible_numerator_refuses_at_the_component():
    from scripts.analyze_h3_identity import _component

    with pytest.raises(ScoringError, match="outside"):
        _component("x", None, 6, 5, 0, 0.85, "at_least")
    with pytest.raises(ScoringError, match="outside"):
        _component("x", None, -1, 5, 0, 0.85, "at_least")


# ===========================================================================
# the unit MATRIX, not just the aggregate census
# ===========================================================================
# 80/80/160/15 can balance while a unit is missing, duplicated, or partial —
# excess rows from one unit masking the deficit of another. The verdict path
# now requires the exact {S3, S4} x 5-seed matrix, one unit per cell, and the
# exact per-unit scored-event census.

MATRIX_SEEDS = (41, 42, 43, 44, 45)
_ODD = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


@pytest.fixture
def matrix_manifest(monkeypatch, tmp_path):
    """Point the adjudicating seed-manifest check at THESE tests' seeds.

    The real manifest is the sealed h3_eval_v2 file, whose values tests must
    neither read nor embed; the check itself is exercised against a stand-in."""
    import scripts.analyze_h3_identity as identity

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    path = manifest_dir / "matrix_manifest.json"
    path.write_text(json.dumps({"h3_eval_seeds": list(MATRIX_SEEDS)}))
    monkeypatch.setitem(identity.SEED_MANIFESTS, Cohort.ADJUDICATING,
                        (path, "h3_eval_seeds"))


def _design_exact_corpus(tmp_path):
    """A full adjudicating-design corpus: 10 units, 16 malicious each,
    3 honest per S4 unit, all rank-1 correct, all on odd partitions."""
    events, paths = [], []
    for scenario in (S3, S4):
        for seed in MATRIX_SEEDS:
            uid = _uid(scenario, seed)
            paths.append(_result_json(
                tmp_path, f"{scenario}_{seed}.json", scenario=scenario,
                seed=seed, cohort="adjudicating", run_uid=uid,
            ))
            for index in range(16):
                device = _ODD[index % len(_ODD)]
                events.append(_event(
                    f"{scenario}-{seed}-m{index}", f"client_{device}_new{index}",
                    scenario=scenario, parent=f"client_{device}", seed=seed,
                    run_uid=uid,
                ))
            if scenario == S4:
                for index in range(3):
                    device = _ODD[index]
                    events.append(_event(
                        f"{scenario}-{seed}-h{index}",
                        f"client_{device}_new9{index}", scenario=scenario,
                        parent=f"client_{device}", seed=seed,
                        malicious=False, run_uid=uid,
                    ))
    return events, load_provenance(paths)


def test_a_design_exact_corpus_passes_the_full_verdict_path(tmp_path, matrix_manifest):
    events, provenance = _design_exact_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance)
    assert report["counts_match_design"] is True
    assert report["census_gate_enforced"] is True
    assert report["verdict"] == "PASS"     # the ONLY path that carries one


def test_a_missing_unit_masked_by_an_overfull_one_is_REFUSED(tmp_path, matrix_manifest):
    """Aggregate census stays EXACT (160/15) — the old gate passed this."""
    events, _ = _design_exact_corpus(tmp_path)
    paths = sorted(p for p in Path(tmp_path).glob("*.json")
                   if ".base." not in p.name and p.name != f"{S3}_45.json")
    provenance = load_provenance(paths)
    donor, recipient = _uid(S3, 45), _uid(S3, 44)
    for event in events:
        if event["run_uid"] == donor:
            _rebind(event, recipient, seed=44)   # a COMPLETE forgery
    with pytest.raises(ScoringError, match="unit matrix incomplete"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_a_duplicate_cell_under_different_run_uids_is_REFUSED(tmp_path, matrix_manifest):
    events, _ = _design_exact_corpus(tmp_path)
    paths = sorted(p for p in Path(tmp_path).glob("*.json")
                   if ".base." not in p.name)
    dup = _result_json(tmp_path, "dup_cell.json", scenario=S3, seed=45,
                       cohort="adjudicating", run_uid="sim__dup__run")
    provenance = load_provenance(list(paths) + [dup])
    with pytest.raises(ScoringError, match="duplicate .scenario, seed. cell"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_an_underfull_unit_compensated_by_an_overfull_one_is_REFUSED(tmp_path, matrix_manifest):
    """Both units exist, census balances at 160 — per-unit counts catch it."""
    events, provenance = _design_exact_corpus(tmp_path)
    donor, recipient = _uid(S3, 45), _uid(S3, 44)
    moved = 0
    for event in events:
        if event["run_uid"] == donor and moved < 4:
            _rebind(event, recipient, seed=44)   # complete forgery
            moved += 1
    with pytest.raises(ScoringError, match="scored-event census for this unit"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_a_zero_event_provenance_unit_is_REFUSED(tmp_path, matrix_manifest):
    events, provenance = _design_exact_corpus(tmp_path)
    donor, recipient = _uid(S3, 45), _uid(S3, 44)
    for event in events:
        if event["run_uid"] == donor:
            _rebind(event, recipient, seed=44)   # complete forgery
    with pytest.raises(ScoringError, match="scored-event census for this unit"):
        score_events(events, Cohort.ADJUDICATING, provenance)


# ===========================================================================
# the scorer's own record says RATIFIED
# ===========================================================================

def test_the_scorer_reports_its_bars_as_RATIFIED():
    from scripts.analyze_h3_identity import STATUS

    assert "RATIFIED" in STATUS
    assert "v1.50" in STATUS
    assert "PROPOSED" not in STATUS


# ===========================================================================
# events must AGREE with their bound unit
# ===========================================================================
# run_uid presence alone is not binding: a row with a valid run_uid but a
# swapped scenario/seed reallocates S3/S4 outcomes while every matrix count
# still balances, and a row decided at a different tau slips past the
# unit-level corroboration.

def test_a_scenario_swapped_event_is_REFUSED(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("swap", "client_5_new1", scenario=S3, parent="client_5")
    _rebind(bad, _uid(S4, 42))         # binds to the S4 unit, claims S3
    events.append(bad)
    with pytest.raises(ScoringError, match="scenario"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_a_seed_swapped_event_is_REFUSED(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("seedswap", "client_5_new1", scenario=S4, parent="client_5")
    _rebind(bad, _uid(S4, 42), seed=137)   # not the bound unit's seed
    events.append(bad)
    with pytest.raises(ScoringError, match="seed"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_an_event_decided_at_a_drifted_tau_is_REFUSED(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("badtau", "client_5_new1", scenario=S4, parent="client_5",
                 tau=999.0)
    events.append(bad)
    with pytest.raises(ScoringError, match="tau"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_an_event_with_NO_tau_is_REFUSED(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("notau", "client_5_new1", scenario=S4, parent="client_5")
    bad["tau"] = None
    events.append(bad)
    with pytest.raises(ScoringError, match="tau"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


# ===========================================================================
# the matrix seed set must EQUAL the manifest
# ===========================================================================

def test_a_matrix_on_the_wrong_seeds_is_REFUSED(tmp_path, monkeypatch):
    """Internally consistent (5+5 units, same seed set) but NOT the registered
    seeds — the pre-fix gate passed this."""
    import scripts.analyze_h3_identity as identity

    manifest = tmp_path / "other_manifest.json"
    manifest.write_text(json.dumps({"h3_eval_seeds": [901, 902, 903, 904, 905]}))
    monkeypatch.setitem(identity.SEED_MANIFESTS, Cohort.ADJUDICATING,
                        (manifest, "h3_eval_seeds"))
    events, provenance = _design_exact_corpus(tmp_path)
    with pytest.raises(ScoringError, match="values not printed"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_a_missing_seed_manifest_is_REFUSED(tmp_path, monkeypatch):
    import scripts.analyze_h3_identity as identity

    monkeypatch.setitem(identity.SEED_MANIFESTS, Cohort.ADJUDICATING,
                        (tmp_path / "absent.json", "h3_eval_seeds"))
    events, provenance = _design_exact_corpus(tmp_path)
    with pytest.raises(ScoringError, match="manifest"):
        score_events(events, Cohort.ADJUDICATING, provenance)


def test_the_default_manifests_point_at_the_registered_files():
    # Full RESOLVED paths, not basenames — a same-named file in another
    # directory must not satisfy this pin.
    import scripts.analyze_h3_identity as identity

    val_path, val_key = identity.SEED_MANIFESTS[Cohort.VALIDATION]
    adj_path, adj_key = identity.SEED_MANIFESTS[Cohort.ADJUDICATING]
    data_dir = identity.PROJECT_ROOT / "data"
    assert Path(val_path).resolve() == (data_dir / "seeds.json").resolve()
    assert val_key == "dev_seeds"
    assert Path(adj_path).resolve() == (data_dir / "h3_eval_seeds_v2.json").resolve()
    assert adj_key == "h3_eval_seeds"


# ===========================================================================
# key integrity + diagnostic distinguishability
# ===========================================================================

def test_a_relabeled_run_uid_with_the_original_key_is_REFUSED(tmp_path):
    """The key contract embeds the EMITTING run's UID. Relabeling the run_uid
    column onto a supplied unit — with scenario/seed/tau all aligned — must
    still refuse, because the key betrays the origin."""
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("relabel", "client_5_new1", scenario=S4, parent="client_5")
    bad["reentry_event_key"] = f"{RUN_B}:7:cid-relabel"   # emitted by RUN_B
    events.append(bad)
    with pytest.raises(ScoringError, match="contract rebuild"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_an_event_missing_key_fields_is_REFUSED(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("nofields", "client_5_new1", scenario=S4, parent="client_5")
    del bad["server_round"]
    events.append(bad)
    with pytest.raises(ScoringError, match="cannot be reconstructed"):
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)


def test_a_diagnostic_run_withholds_the_verdict(tmp_path):
    """--allow-partial-census artifacts must be structurally DISTINGUISHABLE
    from a sealed verdict — not just a terminal warning."""
    events, provenance = _clean_corpus(tmp_path)
    report = score_events(events, Cohort.ADJUDICATING, provenance,
                          require_design_counts=False)
    assert report["verdict"] is None
    assert report["verdict_status"] == "DIAGNOSTIC — VERDICT WITHHELD"
    assert report["diagnostic_partial_census"] is True
    assert report["census_gate_enforced"] is False
    assert report["component_conjunction"] == "PASS"
    assert "no pass/fail claim" in report["verdict_withheld_reason"]


# ===========================================================================
# refusals never echo run-identifying strings
# ===========================================================================
# run_uids (and the event keys embedding them) carry the launch seed in clear
# text; on the sealed cohort a refusal that echoed them would print sealed
# values into logs.

def test_binding_refusals_never_echo_run_identifying_strings(tmp_path):
    events, provenance = _clean_corpus(tmp_path)
    bad = _event("relabel2", "client_5_new1", scenario=S4, parent="client_5")
    bad["reentry_event_key"] = f"{RUN_B}:7:cid-relabel2"
    events.append(bad)
    with pytest.raises(ScoringError) as excinfo:
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)
    message = str(excinfo.value)
    assert RUN_B not in message
    assert _uid(S4, 42) not in message
    assert "sha256:" in message


def test_unbound_run_refusals_never_echo_run_uids(tmp_path):
    provenance = load_provenance([_result_json(tmp_path, "u.json", seed=42)])
    events = [_event("e1", "client_1_new1", seed=137)]
    with pytest.raises(ScoringError) as excinfo:
        score_events(events, Cohort.ADJUDICATING, provenance,
                     require_design_counts=False)
    message = str(excinfo.value)
    assert _uid(S4, 137) not in message
    assert _uid(S4, 42) not in message
    assert "redacted" in message


def test_a_validation_partial_census_artifact_is_marked_diagnostic(tmp_path):
    """Even informs-only material must be distinguishable from a gated read."""
    events, provenance = _clean_corpus(tmp_path, cohort="validation")
    report = score_events(events, Cohort.VALIDATION, provenance,
                          require_design_counts=False)
    assert report["verdict"] is None
    assert "NOT ADJUDICATING" in report["verdict_status"]
    assert "DIAGNOSTIC" in report["verdict_status"]
    assert report["diagnostic_partial_census"] is True
    assert "census and unit-matrix gates" in report["verdict_withheld_reason"]

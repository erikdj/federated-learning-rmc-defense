"""H3 — schema-v5 signal log → primary-scorer event corpus (DRYRUN F4).

The adapter between `flowerfl/signal_logger.py`'s schema-v5 rows and the frozen
primary scorer `scripts/analyze_h3_relink.py`. Two things it must get right, and
both are integrity properties rather than conveniences:

* **Never drop an event silently.** A dropped event shrinks a denominator, and a
  silently shrunk denominator is the one failure mode that BIASES the metric
  rather than merely degrading it. Every refusal is loud and names the row.
* **The arm field is `h3_arm`, never `arm`.** `arm` is already taken in this
  codebase for the resampling manifest's variant label
  (`flowerfl/resampling_manifest.py`), whose values look like `smote@balanced` /
  `off`. Both meanings meet in the same signal-log row, so the H3 arm carries a
  distinct name and a row carrying only the resampling `arm` can never satisfy
  the v1.10 § 5.1 arm rule.

v4 logs are refused **by name**: the dry-run (§ 1.2/1.3) established that they
structurally cannot feed this pipeline — the fingerprint was never on the wire
and all 12 v5 fields, `aggregation_coefficient` included, are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowerfl.signal_logger import (
    REENTRY_EVENT_FIELDS,
    SCHEMA_V5_ADDED_FIELDS,
    SIGNAL_LOG_SCHEMA_VERSION,
)
from scripts.analyze_h3_relink import (
    Cohort,
    ScoringError,
    load_events,
    score_events,
)
from scripts.extract_h3_events import (
    EVENT_CORPUS_FIELDS,
    SUPPORTED_SCHEMA_VERSION,
    ExtractionError,
    extract_events,
    main,
)

MALICIOUS_PARTITIONS = tuple(range(0, 9))     # client_0..client_8
HONEST_CHURN_PARENTS = (11, 13, 15)          # S4 benign reconnects
CYCLES = 4
SEEDS = (42, 137, 256, 314, 500)
SCENARIOS = ("S3_identity_reset_only", "S4_full_mix")


def _run_uid(scenario: str, seed: int, defense: str = "tgefp") -> str:
    return f"flower_reset__{scenario}__{defense}__seed{seed}__2026-08-08T00:00:00Z"


def _base_row(**over):
    """A schema-v5 signal-log row with every v5 field explicitly null."""
    row = {
        "signal_log_schema_version": SIGNAL_LOG_SCHEMA_VERSION,
        "seed": 42,
        "scenario": "S4_full_mix",
        "exec_mode": "flower_reset",
        "dataset": "edge_full_20_rmc",
        "defense": "tgefp",
        "git_commit": "deadbeef",
        "run_started_at": "2026-08-08T00:00:00Z",
        "run_uid": _run_uid("S4_full_mix", 42),
        "server_round": 10,
        "scenario_round": 9,
        "logical_cid": "client_1_new1",
        "flower_cid": "cid-1-1",
        "physical_partition_id": 1,
        "malicious_gt": True,
        "attack_type": "alie",
        "num_examples": 100,
        "train_loss": 0.5,
        "update_norm": 1.0,
        "cos_to_median": 0.9,
        "L2_to_median": 0.5,
        "krum_score": None,
        "trust_score": None,
        "effective_weight": 100.0,
        "tenure": 1,
    }
    row.update({field: None for field in SCHEMA_V5_ADDED_FIELDS})
    row.update(over)
    return row


def _event_row(
    *,
    seed=42,
    scenario="S4_full_mix",
    partition=1,
    cycle=1,
    server_round=10,
    malicious=True,
    matched=True,
    parent_partition=None,
    defense="tgefp",
    key=None,
    **over,
):
    """A schema-v5 row that IS a re-entry event (all ten fields populated)."""
    parent = partition if parent_partition is None else parent_partition
    cid = f"cid-{partition}-{cycle}"
    row = _base_row(
        seed=seed,
        scenario=scenario,
        defense=defense,
        run_uid=_run_uid(scenario, seed, defense),
        server_round=server_round,
        scenario_round=server_round - 1,
        logical_cid=f"client_{partition}_new{cycle}",
        flower_cid=cid,
        physical_partition_id=partition,
        malicious_gt=malicious,
        tenure=1,
        reentry_event_key=key or f"{_run_uid(scenario, seed, defense)}:{server_round}:{cid}",
        current_cid=cid,
        gt_logical_id=f"client_{partition}_new{cycle}",
        gt_is_malicious=malicious,
        asserted_match=matched,
        asserted_parent_entry_id=f"fp-{parent}" if matched else None,
        asserted_parent_logical_id=f"client_{parent}" if matched else None,
        min_d=0.5 if matched else 42.0,
        tau=1.0,
        generation=1 if matched else 0,
    )
    row.update(over)
    return row


def _non_event_row(**over):
    """An ordinary participation row: every re-entry field is an explicit null."""
    return _base_row(logical_cid="client_4", flower_cid="cid-4", tenure=7,
                     physical_partition_id=4, **over)


def _write_log(tmp_path: Path, rows, name="signals.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _design_log_rows():
    """The full pre-registered census as schema-v5 signal-log rows.

    9 malicious devices x 4 cycles x 5 seeds x {S3, S4}, plus the 3 benign-churn
    reconnects (parents 11/13/15) in S4 — v1.10 § 5.1's 180/180/15.
    """
    rows = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for partition in MALICIOUS_PARTITIONS:
                for cycle in range(1, CYCLES + 1):
                    rows.append(
                        _event_row(
                            seed=seed, scenario=scenario, partition=partition,
                            cycle=cycle, server_round=10 * cycle,
                        )
                    )
            if scenario == "S4_full_mix":
                for partition in HONEST_CHURN_PARENTS:
                    rows.append(
                        _event_row(
                            seed=seed, scenario=scenario, partition=partition,
                            cycle=1, server_round=25, malicious=False, matched=False,
                        )
                    )
            # Ordinary participation rows must not become events.
            rows.append(_non_event_row(seed=seed, scenario=scenario,
                                       run_uid=_run_uid(scenario, seed)))
    return rows


# ---------------------------------------------------------------------------
# Schema-version gating (dry-run § 1.2: v4 cannot feed this pipeline)
# ---------------------------------------------------------------------------

def test_the_supported_schema_version_is_five():
    assert SUPPORTED_SCHEMA_VERSION == SIGNAL_LOG_SCHEMA_VERSION == 5


def test_a_v4_log_is_refused_by_name(tmp_path):
    v4 = _event_row()
    v4["signal_log_schema_version"] = 4
    for field in SCHEMA_V5_ADDED_FIELDS:
        v4.pop(field, None)
    with pytest.raises(ExtractionError, match=r"schema v4"):
        extract_events([_write_log(tmp_path, [v4])])


def test_a_v4_refusal_names_the_missing_v5_instrument(tmp_path):
    v4 = _non_event_row()
    v4["signal_log_schema_version"] = 4
    for field in SCHEMA_V5_ADDED_FIELDS:
        v4.pop(field, None)
    with pytest.raises(ExtractionError, match="aggregation_coefficient"):
        extract_events([_write_log(tmp_path, [v4])])


def test_a_row_without_a_schema_version_is_refused(tmp_path):
    row = _event_row()
    del row["signal_log_schema_version"]
    with pytest.raises(ExtractionError, match="signal_log_schema_version"):
        extract_events([_write_log(tmp_path, [row])])


def test_a_mixed_version_log_is_refused(tmp_path):
    good = _event_row()
    bad = _event_row(partition=3, key="other")
    bad["signal_log_schema_version"] = 4
    with pytest.raises(ExtractionError, match=r"schema v4"):
        extract_events([_write_log(tmp_path, [good, bad])])


# ---------------------------------------------------------------------------
# Event selection — never drop an event silently
# ---------------------------------------------------------------------------

def test_only_reentry_rows_become_events(tmp_path):
    rows = [_non_event_row(), _event_row(), _non_event_row(server_round=11)]
    events = extract_events([_write_log(tmp_path, rows)])
    assert len(events) == 1
    assert events[0]["gt_logical_id"] == "client_1_new1"


def test_a_partially_populated_event_row_is_a_loud_error(tmp_path):
    """A null key with a populated truth half is the null-`run_uid` case
    `scenario_strategy` warns about: it would silently shrink a denominator."""
    row = _event_row()
    row["reentry_event_key"] = None
    with pytest.raises(ExtractionError, match="reentry_event_key"):
        extract_events([_write_log(tmp_path, [row])])


def test_an_event_row_missing_a_v5_key_entirely_is_a_loud_error(tmp_path):
    row = _event_row()
    del row["generation"]
    with pytest.raises(ExtractionError, match="generation"):
        extract_events([_write_log(tmp_path, [row])])


def test_a_matched_event_with_no_asserted_parent_is_a_loud_error(tmp_path):
    row = _event_row(matched=True)
    row["asserted_parent_logical_id"] = None
    with pytest.raises(ExtractionError, match="asserted_parent_logical_id"):
        extract_events([_write_log(tmp_path, [row])])


def test_an_event_with_a_null_registry_decision_is_a_loud_error(tmp_path):
    """§ 5.1 integrity assertion: the registry half is written for every event,
    regardless of what the defense did with the update."""
    row = _event_row()
    row["asserted_match"] = None
    with pytest.raises(ExtractionError, match="asserted_match"):
        extract_events([_write_log(tmp_path, [row])])


def test_a_null_tau_is_a_loud_error(tmp_path):
    """Gate (c): a null tau means the registry ran without a locked threshold."""
    row = _event_row()
    row["tau"] = None
    with pytest.raises(ExtractionError, match="tau"):
        extract_events([_write_log(tmp_path, [row])])


def test_the_observe_only_pre_lock_shape_is_refused_by_name(tmp_path):
    """`min_d = inf`, no τ — a legitimate LOG shape (non-finite floats are
    JSON-nulled) and an illegitimate SCORING shape: every event is an unmatched
    miss, so recall is 0 by construction and H3 would falsify by arithmetic."""
    row = _event_row(matched=False)
    row["min_d"] = None
    row["tau"] = None
    with pytest.raises(ExtractionError, match="PRE-LOCK"):
        extract_events([_write_log(tmp_path, [row])])


def test_an_unmatched_event_may_carry_a_null_distance(tmp_path):
    """An overflowed / absent Mahalanobis distance reads as no-match and is
    JSON-nulled by `_jsonify`; it must not cost the corpus an event."""
    row = _event_row(matched=False)
    row["min_d"] = None
    events = extract_events([_write_log(tmp_path, [row])])
    assert len(events) == 1
    assert events[0]["min_d"] is None
    assert events[0]["asserted_match"] is False


def test_unknown_provenance_is_refused(tmp_path):
    """Gate (b): zero UNKNOWN / missing-provenance rows."""
    row = _event_row()
    row["gt_logical_id"] = "UNKNOWN"
    with pytest.raises(ExtractionError, match="UNKNOWN"):
        extract_events([_write_log(tmp_path, [row])])


def test_a_missing_scenario_is_refused(tmp_path):
    row = _event_row()
    row["scenario"] = ""
    with pytest.raises(ExtractionError, match="scenario"):
        extract_events([_write_log(tmp_path, [row])])


def test_malformed_json_is_refused_with_the_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(_event_row()) + "\n{not json}\n")
    with pytest.raises(ExtractionError, match=r":2"):
        extract_events([path])


def test_a_missing_log_file_is_refused(tmp_path):
    with pytest.raises(ExtractionError, match="not found"):
        extract_events([tmp_path / "nope.jsonl"])


# ---------------------------------------------------------------------------
# The `arm` / `h3_arm` collision
# ---------------------------------------------------------------------------

def test_the_defense_token_becomes_the_h3_arm(tmp_path):
    events = extract_events([_write_log(tmp_path, [_event_row(defense="tgefp")])])
    assert events[0]["h3_arm"] == "tgefp"


def test_the_control_arm_is_extracted_too_and_filtered_by_the_scorer(tmp_path):
    """The extractor is not the place the arm rule is applied — the scorer is."""
    rows = [
        _event_row(defense="tgefp"),
        _event_row(defense="tge", partition=3, key="control-row"),
    ]
    events = extract_events([_write_log(tmp_path, rows)])
    assert sorted(e["h3_arm"] for e in events) == ["tge", "tgefp"]


def test_the_resampling_arm_is_never_the_h3_arm(tmp_path):
    """`arm` in a signal-log row is the resampling manifest's variant label."""
    row = _event_row(defense="tgefp", arm="smote@balanced")
    events = extract_events([_write_log(tmp_path, [row])])
    assert events[0]["h3_arm"] == "tgefp"
    assert events[0]["resampling_arm"] == "smote@balanced"
    assert "arm" not in events[0], "the bare `arm` key must not reach the corpus"


def test_a_row_carrying_only_the_resampling_arm_cannot_satisfy_the_h3_arm_rule(tmp_path):
    """The scorer must refuse — not silently read the resampling label as the
    H3 arm, and not silently drop the row out of a denominator either."""
    corpus_row = dict(
        extract_events([_write_log(tmp_path, [_event_row()])])[0]
    )
    del corpus_row["h3_arm"]
    corpus_row["arm"] = "smote@balanced"
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(corpus_row) + "\n")
    with pytest.raises(ScoringError, match="resampling"):
        load_events([path])


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_events_are_deduplicated_on_the_event_key(tmp_path):
    row = _event_row()
    events = extract_events([_write_log(tmp_path, [row, dict(row)])])
    assert len(events) == 1


def test_dedup_spans_multiple_logs(tmp_path):
    row = _event_row()
    events = extract_events([
        _write_log(tmp_path, [row], name="a.jsonl"),
        _write_log(tmp_path, [dict(row)], name="b.jsonl"),
    ])
    assert len(events) == 1


def test_conflicting_rows_under_one_event_key_are_a_loud_error(tmp_path):
    first = _event_row()
    second = _event_row()
    second["asserted_match"] = False
    second["asserted_parent_logical_id"] = None
    second["asserted_parent_entry_id"] = None
    with pytest.raises(ExtractionError, match="conflict"):
        extract_events([_write_log(tmp_path, [first, second])])


# ---------------------------------------------------------------------------
# The output contract, and the end-to-end run through the REAL scorer
# ---------------------------------------------------------------------------

def test_emitted_rows_carry_exactly_the_declared_corpus_fields(tmp_path):
    events = extract_events([_write_log(tmp_path, [_event_row()])])
    assert set(events[0]) == set(EVENT_CORPUS_FIELDS)


def test_the_corpus_carries_every_field_the_frozen_scorer_requires(tmp_path):
    from scripts.analyze_h3_relink import REQUIRED_FIELDS
    events = extract_events([_write_log(tmp_path, [_event_row()])])
    assert set(REQUIRED_FIELDS) <= set(events[0])
    assert "scenario" in events[0] and "h3_arm" in events[0]


def test_extractor_output_scores_end_to_end_on_the_real_scorer(tmp_path):
    """The whole point of the adapter: v5 logs in, a scorable corpus out."""
    log = _write_log(tmp_path, _design_log_rows())
    corpus = tmp_path / "events.jsonl"
    assert main(["--logs", str(log), "--out", str(corpus)]) == 0

    events = load_events([corpus])
    report = score_events(events, Cohort.ADJUDICATING)
    assert report["components"]["recall_S3"]["denominator"] == 80
    assert report["components"]["recall_S4"]["denominator"] == 80
    assert report["components"]["flr_S4"]["denominator"] == 15
    assert report["counts_match_design"] is True
    assert report["verdict"] == "PASS"


def test_extractor_output_also_yields_the_validation_denominators(tmp_path):
    log = _write_log(tmp_path, _design_log_rows())
    events = load_events([_write_corpus(tmp_path, extract_events([log]))])
    report = score_events(events, Cohort.VALIDATION)
    assert report["components"]["recall_S3"]["denominator"] == 180
    assert report["components"]["recall_S4"]["denominator"] == 180
    assert report["components"]["flr_S4"]["denominator"] == 15


def _write_corpus(tmp_path: Path, events, name="corpus.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_writes_a_jsonl_corpus_and_a_census(tmp_path, capsys):
    log = _write_log(tmp_path, [_event_row(), _non_event_row()])
    out = tmp_path / "events.jsonl"
    assert main(["--logs", str(log), "--out", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    printed = capsys.readouterr().out
    assert "rows read" in printed and "events" in printed


def test_main_refuses_a_v4_log_without_a_traceback(tmp_path, capsys):
    v4 = _event_row()
    v4["signal_log_schema_version"] = 4
    log = _write_log(tmp_path, [v4])
    out = tmp_path / "events.jsonl"
    assert main(["--logs", str(log), "--out", str(out)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()


def test_the_reentry_field_contract_is_read_from_the_logger_not_duplicated():
    """Binding the extractor to `REENTRY_EVENT_FIELDS` means a future contract
    change cannot silently desynchronise the two ends."""
    from scripts.extract_h3_events import REENTRY_EVENT_FIELDS as extractor_fields
    assert tuple(extractor_fields) == tuple(REENTRY_EVENT_FIELDS)


# ===========================================================================
# The nearest-candidate pair — ADDITIVE and OPTIONAL
# ===========================================================================

def test_the_nearest_pair_is_emitted_when_the_log_carries_it(tmp_path):
    from scripts.extract_h3_events import extract_events

    row = _event_row()
    row["nearest_entry_id"] = "fp-0002"
    row["nearest_logical_id"] = "client_3"
    path = _write_log(tmp_path, [row], name="log.jsonl")
    corpus = extract_events([path])
    assert corpus[0]["nearest_entry_id"] == "fp-0002"
    assert corpus[0]["nearest_logical_id"] == "client_3"


def test_the_nearest_pair_is_ABSENT_not_null_on_a_pre_extension_log(tmp_path):
    """Absent means 'this corpus cannot answer rank-1 threshold-free'; an
    explicit null would read as 'the pool was empty', which is a different and
    determinate fact."""
    from scripts.extract_h3_events import extract_events

    row = _event_row()
    # `_base_row` now emits the pair (schema extension); strip it to reproduce a
    # log written before the extension landed.
    row.pop("nearest_entry_id", None)
    row.pop("nearest_logical_id", None)
    path = _write_log(tmp_path, [row], name="log.jsonl")
    corpus = extract_events([path])
    assert "nearest_entry_id" not in corpus[0]
    assert "nearest_logical_id" not in corpus[0]


def test_an_explicit_null_nearest_pair_is_carried_through(tmp_path):
    """A post-extension log whose pool was empty writes explicit nulls, and the
    corpus must preserve that — it is a real observation, not a gap."""
    from scripts.extract_h3_events import extract_events

    row = _event_row()
    row["nearest_entry_id"] = None
    row["nearest_logical_id"] = None
    path = _write_log(tmp_path, [row], name="log.jsonl")
    corpus = extract_events([path])
    assert corpus[0]["nearest_entry_id"] is None
    assert corpus[0]["nearest_logical_id"] is None


def test_the_frozen_d7_required_fields_are_unchanged():
    from scripts.analyze_h3_relink import REQUIRED_FIELDS

    assert REQUIRED_FIELDS == (
        "reentry_event_key",
        "server_round",
        "current_cid",
        "gt_logical_id",
        "gt_is_malicious",
        "asserted_match",
        "asserted_parent_entry_id",
        "asserted_parent_logical_id",
        "min_d",
        "tau",
        "generation",
    )
    assert "nearest_entry_id" not in REQUIRED_FIELDS
    assert "nearest_logical_id" not in REQUIRED_FIELDS


def test_D7_REGRESSION_GUARD_the_new_fields_move_no_frozen_number(tmp_path):
    """STANDING INVARIANT, not a one-time read.

    The frozen D7 scorer must produce byte-identical output over a corpus that
    carries the nearest-candidate pair and the same corpus without it. This is
    what makes the addition safe: `classify_event` returns miss/true_negative
    before it ever reads a parent on an unmatched event, so a populated
    `nearest_*` cannot reach any D7 quantity.
    """
    import json

    from scripts.analyze_h3_relink import Cohort, load_events, score_events
    from scripts.extract_h3_events import extract_events

    plain = []
    for index in range(6):
        row = _event_row()
        row["reentry_event_key"] = f"run:{index}:cid-{index}"
        row["current_cid"] = f"cid-{index}"
        row["gt_logical_id"] = f"client_{2 * index + 1}_new1"
        # a mix of matched and unmatched events, so both classification paths
        # are exercised under both corpora
        if index % 2:
            row["asserted_match"] = False
            row["asserted_parent_entry_id"] = None
            row["asserted_parent_logical_id"] = None
            row["min_d"] = 310.0
        else:
            row["asserted_match"] = True
            row["asserted_parent_logical_id"] = f"client_{2 * index + 1}"
        # the pre-extension corpus carries no nearest pair at all
        row.pop("nearest_entry_id", None)
        row.pop("nearest_logical_id", None)
        plain.append(row)

    enriched = []
    for index, row in enumerate(plain):
        copy = dict(row)
        # populated on EVERY row, including the unmatched ones — the whole point
        copy["nearest_entry_id"] = "fp-0007"
        copy["nearest_logical_id"] = f"client_{2 * index + 1}"
        enriched.append(copy)

    plain_path = _write_log(tmp_path, plain, name="plain.jsonl")
    enriched_path = _write_log(tmp_path, enriched, name="enriched.jsonl")

    reports = []
    for index, path in enumerate((plain_path, enriched_path)):
        corpus_path = tmp_path / f"corpus-{index}.jsonl"
        corpus_path.write_text(
            "\n".join(json.dumps(r) for r in extract_events([path])) + "\n"
        )
        events = load_events([corpus_path])
        reports.append(
            score_events(events, Cohort.ADJUDICATING, require_design_counts=False)
        )
    assert json.dumps(reports[0], sort_keys=True) == json.dumps(
        reports[1], sort_keys=True
    )

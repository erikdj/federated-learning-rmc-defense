"""Signal-log schema v4 → v5 (H3 Step 2).

v5 adds, additively and non-destructively:

  * ``aggregation_coefficient`` — the post-``filter_updates``, round-normalized
    FedAvg coefficient (0.0 for a hard-dropped client). The locked
    `data/h3_constants.json` rejoin-success rule is defined on this quantity;
    `effective_weight` (raw pre-filter ``num_examples``) cannot express it and
    keeps its **exact legacy semantics** here — v4 rows are never reinterpreted.
  * the **re-entry event contract** of amendment v1.10 § 5.1 — the named fields
    a scorer needs to decide, per re-entry event, whether the registry's
    asserted link was correct.
  * ``client_timing_observation`` — the v1.10-mandated "(D8 optional-arm-ready)
    per-client timing field". Structurally null in the launch design; the
    synthetic per-device timing model is a separately pre-registered optional
    arm (D8), explicitly NOT in H3's launch scope.

The registry half of the re-entry contract (``asserted_match``, ``min_d``,
``tau``, ``generation``, the parent ids) is null until the fingerprint plugin
lands; the **ground-truth half** is populated here, because it is pure
instrumentation and integrity gate (d) — "realized event counts match design
exactly" — is scored off it.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from flowerfl.byzantine_defense import KrumDefensePlugin
from flowerfl.scenario_strategy import ScenarioStrategy
from flowerfl.signal_logger import (
    REENTRY_EVENT_FIELDS,
    SCHEMA_V5_ADDED_FIELDS,
    SIGNAL_LOG_SCHEMA_VERSION,
    SignalLogger,
    canonical_device_id,
    is_reentry_identity,
)

MINIMAL_RECORD = {
    "logical_cid": "client_0", "flower_cid": "abc",
    "physical_partition_id": 0, "malicious_gt": False,
    "attack_type": "", "num_examples": 100,
    "train_loss": 0.5, "update_norm": 1.0,
    "cos_to_median": 0.95, "L2_to_median": 0.5,
    "krum_score": None, "trust_score": None,
    "effective_weight": 100.0,
}


def _logger(tmp_path, **meta):
    return SignalLogger(
        path=str(tmp_path / "test.jsonl"),
        run_metadata={
            "seed": 42, "scenario": "test", "exec_mode": "flower_reset",
            "dataset": "test_dataset", "defense": "krum", **meta,
        },
    )


def _first_row(tmp_path):
    with open(tmp_path / "test.jsonl") as f:
        return json.loads(f.readline())


# ---------------------------------------------------------------------------
# schema version + field contract
# ---------------------------------------------------------------------------


def test_schema_version_is_five(tmp_path):
    assert SIGNAL_LOG_SCHEMA_VERSION == 5
    lg = _logger(tmp_path)
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[dict(MINIMAL_RECORD)])
    lg.close()
    assert _first_row(tmp_path)["signal_log_schema_version"] == 5


def test_schema_version_resists_caller_override(tmp_path):
    lg = _logger(tmp_path, signal_log_schema_version=99)
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[dict(MINIMAL_RECORD)])
    lg.close()
    assert _first_row(tmp_path)["signal_log_schema_version"] == 5


def test_every_v5_field_is_present_and_defaults_to_null(tmp_path):
    """Integrity gate (b) requires the FULL v5 field set on every row: a
    *missing* key is a defect, an explicit null is a truthful 'not applicable'.
    A caller that supplies nothing must still produce the complete key set."""
    lg = _logger(tmp_path)
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[dict(MINIMAL_RECORD)])
    lg.close()
    row = _first_row(tmp_path)
    for field in SCHEMA_V5_ADDED_FIELDS:
        assert field in row, f"v5 field missing from row: {field}"
        assert row[field] is None, f"v5 field {field} should default to null"


def test_caller_values_win_over_the_v5_defaults(tmp_path):
    lg = _logger(tmp_path)
    rec = {**MINIMAL_RECORD, "aggregation_coefficient": 0.25, "asserted_match": False}
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[rec])
    lg.close()
    row = _first_row(tmp_path)
    assert row["aggregation_coefficient"] == 0.25
    assert row["asserted_match"] is False


def test_reentry_contract_is_exactly_the_v1_10_field_list():
    """Frozen by amendment v1.10 § 5.1. `server_round` is the eleventh contract
    field and is already a row-level field emitted by log_round()."""
    assert REENTRY_EVENT_FIELDS == (
        "reentry_event_key",
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


def test_legacy_v4_fields_are_untouched(tmp_path):
    """`effective_weight` keeps its exact v4 meaning (raw pre-filter
    num_examples). v4 readers must not shift under them."""
    lg = _logger(tmp_path)
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[dict(MINIMAL_RECORD)])
    lg.close()
    row = _first_row(tmp_path)
    assert row["effective_weight"] == 100.0
    assert row["num_examples"] == 100
    for legacy in ("krum_score", "trust_score", "cos_to_median", "L2_to_median"):
        assert legacy in row


def test_run_uid_is_stable_within_a_run_and_carries_the_run_identity(tmp_path):
    lg = _logger(tmp_path)
    lg.log_round(server_round=1, scenario_round=0, per_client_records=[dict(MINIMAL_RECORD)])
    lg.log_round(server_round=2, scenario_round=1, per_client_records=[dict(MINIMAL_RECORD)])
    lg.close()
    rows = [json.loads(x) for x in open(tmp_path / "test.jsonl")]
    uids = {r["run_uid"] for r in rows}
    assert len(uids) == 1
    uid = uids.pop()
    assert "flower_reset" in uid and "seed42" in uid and rows[0]["run_started_at"] in uid


# ---------------------------------------------------------------------------
# identity helpers (shared with the fingerprint lane's scorer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("logical,expected", [
    ("client_5_new2", "client_5"),
    ("client_0_new1", "client_0"),
    ("client_11", "client_11"),
    ("client_19_new", "client_19_new"),  # legacy alias -> partition 20, NOT a cycle id
])
def test_canonical_device_id(logical, expected):
    assert canonical_device_id(logical) == expected


@pytest.mark.parametrize("logical,expected", [
    ("client_5_new2", True),
    ("client_0_new1", True),
    ("client_11", False),
    ("client_9_new", False),   # legacy: maps to partition 10, a DIFFERENT device
    ("client_19_new", False),
])
def test_is_reentry_identity(logical, expected):
    assert is_reentry_identity(logical) is expected


# ---------------------------------------------------------------------------
# realized-vs-design event census (integrity gate (d) pre-check, offline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario,exp_malicious,exp_honest", [
    ("rmc/scenarios/S3_identity_reset_only.json", 36, 0),
    ("rmc/scenarios/S4_full_mix.json", 36, 3),
])
def test_scenario_reentry_census_matches_the_design_counts(scenario, exp_malicious, exp_honest):
    """v1.10 § 5.1 design counts: 9 malicious devices × 4 reset cycles = 36
    malicious re-entry events per run in BOTH scenarios; S4 adds exactly 3
    honest CID-change events (parents 11, 13, 15) and S3 has ZERO by
    construction (hence FLR is S4-only)."""
    from pathlib import Path
    from flowerfl.scenario_strategy import adversarial_identities

    doc = json.loads(Path(scenario).read_text())
    adv = adversarial_identities(doc["schedule"])
    seen, malicious, honest = set(), 0, 0
    for block in doc["schedule"]:
        if block.get("skip_scheduling", False):
            continue
        start, end = block["rounds"]
        for _r in range(start, end + 1):
            for logical in block["participants"]:
                if logical in seen:
                    continue
                seen.add(logical)
                if is_reentry_identity(logical):
                    if logical in adv:
                        malicious += 1
                    else:
                        honest += 1
    assert (malicious, honest) == (exp_malicious, exp_honest)


# ---------------------------------------------------------------------------
# end-to-end through ScenarioStrategy
# ---------------------------------------------------------------------------


TEST_SCENARIO = {
    "name": "t_reentry",
    "description": "two honest rounds, then client_0 returns as client_0_new1",
    "dataset": "edge_full_20_rmc",
    "num_rounds": 4,
    "seed": 42,
    "schedule": [
        {"rounds": [1, 2],
         "participants": ["client_0", "client_1", "client_2", "client_3"],
         "attacks": {}},
        {"rounds": [3, 4],
         "participants": ["client_0_new1", "client_1", "client_2", "client_3"],
         "attacks": {"client_0_new1": {"type": "gaussian_noise", "params": {"sigma": 0.5}}}},
    ],
}


def _scenario_results(partitions, cid_of, num_examples, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, pid in enumerate(partitions):
        params = ndarrays_to_parameters([rng.normal(size=dim).astype(np.float32)])
        proxy = SimpleNamespace(cid=cid_of[pid])
        fit = SimpleNamespace(
            parameters=params, num_examples=num_examples[i], metrics={}
        )
        out.append((proxy, fit))
    return out


def _drive(tmp_path, plugins, cid_of=None):
    """Run the 4 scenario rounds (Flower rounds 2..5) and return the rows.

    `cid_of` maps partition -> Flower CID; override it to make the Flower CIDs
    unmistakably distinct from the logical identities.
    """
    path = tmp_path / "scn.json"
    path.write_text(json.dumps(TEST_SCENARIO))
    logger = SignalLogger(
        path=str(tmp_path / "sig.jsonl"),
        run_metadata={"seed": 42, "scenario": "t_reentry",
                      "exec_mode": "flower_reset", "dataset": "edge_full_20_rmc",
                      "defense": "krum"},
    )
    strategy = ScenarioStrategy(
        base_strategy=FedAvg(), plugins=plugins,
        scenario_path=str(path), signal_logger=logger,
    )
    cid_of = cid_of if cid_of is not None else {p: f"raw{p}" for p in range(4)}
    strategy._cid_to_partition = {c: p for p, c in cid_of.items()}
    strategy._partition_to_cid = dict(cid_of)
    strategy._mapping_ready = True

    for server_round in (2, 3, 4, 5):
        results = _scenario_results(
            [0, 1, 2, 3], cid_of, [10, 20, 30, 40], seed=server_round
        )
        strategy.aggregate_fit(server_round, results, [])
    logger.close()
    return [json.loads(x) for x in open(tmp_path / "sig.jsonl")]


def test_rows_carry_the_post_filter_coefficient_and_keep_effective_weight(tmp_path):
    rows = _drive(tmp_path, plugins=[])
    assert rows, "no signal rows emitted"
    by_round = {}
    for r in rows:
        by_round.setdefault(r["server_round"], []).append(r)

    for server_round, group in by_round.items():
        coeffs = [r["aggregation_coefficient"] for r in group]
        assert all(c is not None for c in coeffs), f"round {server_round}: null coefficient"
        assert sum(coeffs) == pytest.approx(1.0)
        # No filtering plugin -> coefficient is the FedAvg share of the raw counts.
        for r in group:
            assert r["aggregation_coefficient"] == pytest.approx(
                r["effective_weight"] / 100.0
            )
        # Legacy semantics preserved exactly: raw, pre-filter num_examples.
        assert sorted(r["effective_weight"] for r in group) == [10.0, 20.0, 30.0, 40.0]


def test_hard_dropped_clients_log_coefficient_exactly_zero(tmp_path):
    """Multi-Krum on n=4: f=1, m=max(1, 4-1-2)=1 -> three clients at 0.0."""
    rows = _drive(tmp_path, plugins=[KrumDefensePlugin(dynamic_f=True)])
    for server_round in {r["server_round"] for r in rows}:
        group = [r for r in rows if r["server_round"] == server_round]
        zeros = [r for r in group if r["aggregation_coefficient"] == 0.0]
        assert len(zeros) == 3, f"round {server_round}: {[r['aggregation_coefficient'] for r in group]}"
        assert sum(r["aggregation_coefficient"] for r in group) == pytest.approx(1.0)
        # ... while effective_weight stays non-zero for the dropped clients:
        # the exact reason it cannot serve as the H3 coefficient.
        assert all(r["effective_weight"] > 0 for r in zeros)


def test_reentry_event_ground_truth_is_emitted_once_per_event(tmp_path):
    rows = _drive(tmp_path, plugins=[])
    events = [r for r in rows if r["reentry_event_key"] is not None]
    assert len(events) == 1, f"expected exactly one re-entry event, got {len(events)}"
    ev = events[0]
    assert ev["logical_cid"] == "client_0_new1"
    # § 5.1: current_cid is "the new FLOWER CID the device reappeared under" —
    # NOT the logical identity. See the dedicated regression test below.
    assert ev["current_cid"] == "raw0"
    assert ev["gt_logical_id"] == "client_0"
    assert ev["gt_is_malicious"] is True
    assert ev["server_round"] == 4  # scenario round 3 = Flower round 4
    assert ev["reentry_event_key"] == f"{ev['run_uid']}:4:raw0"
    # The registry half stays null until the fingerprint plugin lands.
    for field in ("asserted_match", "asserted_parent_entry_id",
                  "asserted_parent_logical_id", "min_d", "tau", "generation"):
        assert ev[field] is None


def test_current_cid_is_the_flower_cid_never_the_logical_identity(tmp_path):
    """§ 5.1 field table, line 778: `current_cid` = "The new Flower CID the
    device reappeared under"; line 776: the event key is `{run_id}:{round}:{cid}`
    — the key OF that CID-appearance, so the same CID. (`data/h3_constants.json`
    carries no key spec; § 5.1 is the sole authority.)

    This is not pedantry about field names. `current_cid` is the OBSERVABLE the
    re-link metric must recover the identity FROM; `gt_logical_id` is the truth
    it is scored AGAINST. Aliasing one to the other collapses the distinction
    the entire H3 evaluation rests on — a scorer reading the logical identity
    out of `current_cid` would "re-link" perfectly by construction.

    Driven with opaque Flower CIDs so logical id, canonical device id, and
    Flower CID are three DISTINCT values in the same row.
    """
    opaque = {p: f"9f3c11d{p}e70b4a82" for p in range(4)}
    rows = _drive(tmp_path, plugins=[], cid_of=opaque)
    events = [r for r in rows if r["reentry_event_key"] is not None]
    assert len(events) == 1
    ev = events[0]

    assert ev["current_cid"] == opaque[0]
    assert ev["current_cid"] == ev["flower_cid"]
    assert ev["logical_cid"] == "client_0_new1"
    assert ev["gt_logical_id"] == "client_0"
    assert ev["reentry_event_key"] == f"{ev['run_uid']}:4:{opaque[0]}"

    # All three are genuinely distinct — no aliasing, in either direction.
    assert len({ev["current_cid"], ev["logical_cid"], ev["gt_logical_id"]}) == 3
    assert ev["current_cid"] != ev["gt_logical_id"]
    assert ev["current_cid"] != ev["logical_cid"]
    assert "client_" not in ev["current_cid"]
    assert ev["gt_logical_id"] not in ev["reentry_event_key"].rsplit(":", 1)[1]


def test_non_reentry_rows_carry_null_event_fields(tmp_path):
    rows = _drive(tmp_path, plugins=[])
    for r in rows:
        if r["logical_cid"] == "client_0_new1" and r["server_round"] == 4:
            continue
        for field in REENTRY_EVENT_FIELDS:
            assert r[field] is None, (r["logical_cid"], r["server_round"], field)


def test_timing_slot_is_structurally_null(tmp_path):
    """D8's synthetic per-device timing arm is out of H3's launch scope; the
    slot exists so the optional arm needs no second schema bump."""
    rows = _drive(tmp_path, plugins=[])
    assert all(r["client_timing_observation"] is None for r in rows)


# ---------------------------------------------------------------------------
# back-compat: v4 readers must parse v5, and v4 logs must keep parsing
# ---------------------------------------------------------------------------


def _gate_lib():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import _calibration_gate_lib as gl
    return gl


@pytest.mark.parametrize("version", [3, 4, 5])
def test_calibration_gate_readers_parse_v3_v4_and_v5(version):
    """The full signal-hygiene reader — not just its version check — must pass
    on every accepted schema. v5 rows additionally carry the new fields, and a
    v4 row that has none of them must parse byte-identically to before."""
    from calibration_gate_fixtures import ROUNDS, make_signal_rows_for_round, make_unit_ref

    gl = _gate_lib()
    unit = make_unit_ref("Krum")
    rows = []
    for server_round in range(2, ROUNDS + 2):
        for row in make_signal_rows_for_round("Krum", server_round, schema_version=version):
            if version == 5:
                row = {
                    **row,
                    "run_uid": "flower_reset__s__krum__seed42__2026-08-05T00:00:00Z",
                    **{f: None for f in SCHEMA_V5_ADDED_FIELDS},
                    "aggregation_coefficient": 1.0 / 20,
                }
            rows.append(row)

    failures = [
        r for r in gl.check_signal_hygiene(unit, rows) if r.required and not r.passed
    ]
    assert not failures, [(f.name, f.detail) for f in failures]


def test_v4_row_semantics_are_not_reinterpreted():
    """A v4 row has NO aggregation_coefficient. Readers must see its absence
    (None), never silently fall back to `effective_weight` — the raw pre-filter
    count — as a stand-in coefficient."""
    from calibration_gate_fixtures import make_signal_rows_for_round

    row = make_signal_rows_for_round("Krum", 2, schema_version=4)[0]
    assert row["effective_weight"] == 100.0
    assert row.get("aggregation_coefficient") is None

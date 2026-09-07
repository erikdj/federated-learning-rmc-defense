"""SMOTE provenance + governance-canary tests.

Mirrors the tge-ramp governance pattern (tests/test_tge_ramp_governance.py):
the runner emits a provenance field for every run declaring its SMOTE status,
and the pyproject default is OFF so a bare `flwr run` is the incumbent.

Per-client application records (director follow-up, closes DESIGN.md Stage-D
provenance list): load_data emits ONE structured `[SMOTE]...` line per client
at data prep; the runner parses them into per-client counts + the run-level
skip flag + the reproducibility seed component.
"""
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4_flower import (
    smote_provenance_fields,
    _smote_mlflow_params,
    parse_smote_records,
)


# Sample structured per-client record lines (the exact format load_data emits).
_APPLIED_3 = ("[SMOTE] client=3 status=applied reason=none variant=smote "
              "target=balanced k=5 n_before=80 n_after=160 synthetic=80 seed_component=111")
_APPLIED_4 = ("[SMOTE] client=4 status=applied reason=none variant=smote "
              "target=balanced k=5 n_before=90 n_after=180 synthetic=90 seed_component=222")
_SKIP_SINGLE_1 = ("[SMOTE] WARNING client=1 status=skipped reason=single_class variant=smote "
                  "target=balanced k=0 n_before=80 n_after=80 synthetic=0 seed_component=333")
_SKIP_STARVED_2 = ("[SMOTE] WARNING client=2 status=skipped reason=minority_starved variant=smote "
                   "target=balanced k=0 n_before=70 n_after=70 synthetic=0 seed_component=444")


# ---------------------------------------------------------------------------
# Record parser
# ---------------------------------------------------------------------------

def test_parse_records_none_when_clean():
    assert parse_smote_records("[Dataset] Loading edge client 3\n") == []


def test_parse_records_single_line():
    recs = parse_smote_records(_APPLIED_3 + "\n")
    assert len(recs) == 1
    r = recs[0]
    assert r["client"] == 3
    assert r["status"] == "applied"
    assert r["reason"] == "none"
    assert r["variant"] == "smote"
    assert r["k"] == 5
    assert r["n_before"] == 80
    assert r["n_after"] == 160
    assert r["synthetic"] == 80
    assert r["seed_component"] == 111


def test_parse_records_all_clients_mixed():
    out = "\n".join([_APPLIED_3, _SKIP_SINGLE_1, _SKIP_STARVED_2, _APPLIED_4]) + "\n"
    recs = parse_smote_records(out)
    assert len(recs) == 4
    assert {r["client"] for r in recs} == {1, 2, 3, 4}
    assert sum(1 for r in recs if r["status"] == "applied") == 2
    assert sum(1 for r in recs if r["status"] == "skipped") == 2


# ---------------------------------------------------------------------------
# OFF path: incumbent, no new fields, no records consulted
# ---------------------------------------------------------------------------

def test_provenance_absent_key_is_incumbent():
    fields = smote_provenance_fields({})
    assert fields == {
        "smote_enabled": False,
        "smote_variant": None,
        "smote_target": None,
        "smote_skipped_reason": None,
    }


def test_provenance_off_ignores_records():
    """A stray record on an OFF run is never recorded — off = incumbent, and the
    per-client aggregation fields are absent entirely."""
    fields = smote_provenance_fields({"smote-enabled": False}, records=parse_smote_records(_APPLIED_3))
    assert fields["smote_enabled"] is False
    assert fields["smote_skipped_reason"] is None
    for k in ("smote_applied_count", "smote_skipped_count", "smote_skip_reasons", "smote_seed_component"):
        assert k not in fields


# ---------------------------------------------------------------------------
# ON path: config echo + per-client aggregation
# ---------------------------------------------------------------------------

def test_provenance_enabled_echoes_normalized_config():
    fields = smote_provenance_fields(
        {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}
    )
    assert fields["smote_enabled"] is True
    assert fields["smote_variant"] == "smote"
    assert fields["smote_target"] == "balanced"


def test_provenance_enabled_float_target_normalized():
    fields = smote_provenance_fields(
        {"smote-enabled": True, "smote-variant": "random_over", "smote-target": "0.5"}
    )
    assert fields["smote_variant"] == "random_over"
    assert fields["smote_target"] == pytest.approx(0.5)


def test_provenance_enabled_validates_loudly():
    with pytest.raises(ValueError, match="smote_variant"):
        smote_provenance_fields({"smote-enabled": True, "smote-variant": "adasyn"})


def test_provenance_enabled_no_records_zero_counts():
    fields = smote_provenance_fields({"smote-enabled": True}, records=[])
    assert fields["smote_applied_count"] == 0
    assert fields["smote_skipped_count"] == 0
    assert fields["smote_skip_reasons"] == {}
    assert fields["smote_skipped_reason"] is None


def test_provenance_enabled_aggregates_records():
    out = "\n".join([_APPLIED_3, _SKIP_SINGLE_1, _SKIP_STARVED_2, _APPLIED_4])
    fields = smote_provenance_fields({"smote-enabled": True}, records=parse_smote_records(out))
    assert fields["smote_applied_count"] == 2
    assert fields["smote_skipped_count"] == 2
    assert fields["smote_skip_reasons"] == {"single_class": 1, "minority_starved": 1}
    # run-level flag: sorted, comma-joined distinct reasons (backward-compatible)
    assert fields["smote_skipped_reason"] == "minority_starved,single_class"


def test_provenance_seed_component_is_run_base_seed():
    fields = smote_provenance_fields({"smote-enabled": True, "seed": 137}, records=[])
    assert fields["smote_seed_component"] == 137


def test_provenance_seed_component_defaults_to_42():
    fields = smote_provenance_fields({"smote-enabled": True}, records=[])
    assert fields["smote_seed_component"] == 42


# ---------------------------------------------------------------------------
# parser-side dedup so smote_applied_count can never exceed
# the client count. LRU eviction / multi-worker caches can re-emit a [SMOTE] line
# for the SAME logical client; counting LINES over-reports. Dedup by
# (client, status, variant, target, seed_component); a SAME client with
# CONFLICTING status/config is flagged in smote_record_conflicts, not collapsed.
# ---------------------------------------------------------------------------

# client=3 re-emitted BYTE-IDENTICALLY except the cache_hits running counter
# (LRU eviction re-running the SAME deterministic resample) — must count once.
_APPLIED_3_DUP = _APPLIED_3 + " cache_hits=7"
# client=3 with a CONFLICTING status — a real anomaly to surface loudly.
_CONFLICT_3 = ("[SMOTE] WARNING client=3 status=skipped reason=single_class variant=smote "
               "target=balanced k=0 n_before=80 n_after=80 synthetic=0 "
               "seed_component=111")
# client=3, SAME status/config but a DIFFERENT OUTCOME (n_after/synthetic) — P2-4:
# must NOT collapse as an exact re-emission; must be flagged as a conflict.
_APPLIED_3_DIFF_OUTCOME = ("[SMOTE] client=3 status=applied reason=none variant=smote "
                          "target=balanced k=5 n_before=80 n_after=200 synthetic=120 "
                          "seed_component=111 cache_hits=9")


def test_provenance_dedupes_duplicate_client_records():
    """A client re-emitted BYTE-IDENTICALLY except cache_hits (LRU eviction /
    multi-worker) is counted ONCE — applied_count never exceeds the client
    count."""
    out = "\n".join([_APPLIED_3, _APPLIED_3_DUP, _APPLIED_4])
    fields = smote_provenance_fields({"smote-enabled": True}, records=parse_smote_records(out))
    assert fields["smote_applied_count"] == 2   # clients {3, 4}, not 3 lines
    assert fields["smote_skipped_count"] == 0
    assert fields["smote_record_conflicts"] == 0


def test_provenance_flags_conflicting_client_records():
    """SAME client with conflicting status is surfaced in smote_record_conflicts
    (nonzero = investigate), not silently collapsed, and excluded from counts."""
    out = "\n".join([_APPLIED_3, _CONFLICT_3, _APPLIED_4])
    fields = smote_provenance_fields({"smote-enabled": True}, records=parse_smote_records(out))
    assert fields["smote_record_conflicts"] == 1
    assert fields["smote_applied_count"] == 1   # only client 4 is unambiguous
    assert fields["smote_skipped_count"] == 0


def test_provenance_flags_conflicting_outcome_same_config():
    """P2-4: two records for one client with the SAME status/config but DIFFERENT
    outcomes must NOT collapse — the differing outcome is a conflict, not an exact
    re-emission."""
    out = "\n".join([_APPLIED_3, _APPLIED_3_DIFF_OUTCOME, _APPLIED_4])
    fields = smote_provenance_fields({"smote-enabled": True}, records=parse_smote_records(out))
    assert fields["smote_record_conflicts"] == 1   # client 3 has conflicting outcomes
    assert fields["smote_applied_count"] == 1      # only client 4 unambiguous


def test_provenance_conflicts_zero_on_clean_run():
    out = "\n".join([_APPLIED_3, _SKIP_SINGLE_1, _APPLIED_4])
    fields = smote_provenance_fields({"smote-enabled": True}, records=parse_smote_records(out))
    assert fields["smote_record_conflicts"] == 0
    assert fields["smote_applied_count"] == 2
    assert fields["smote_skipped_count"] == 1


def test_mlflow_params_include_conflicts_when_nonzero():
    out = "\n".join([_APPLIED_3, _CONFLICT_3])
    params = _smote_mlflow_params(
        {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"},
        records=parse_smote_records(out),
    )
    assert params["smote_record_conflicts"] == "1"


# ---------------------------------------------------------------------------
# MLflow params
# ---------------------------------------------------------------------------

def test_mlflow_params_empty_when_off():
    assert _smote_mlflow_params({}) == {}
    assert _smote_mlflow_params({"smote-enabled": False}) == {}


def test_mlflow_params_present_when_on():
    params = _smote_mlflow_params(
        {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"}, records=[]
    )
    assert params["smote_enabled"] == "True"
    assert params["smote_variant"] == "smote"
    assert params["smote_target"] == "balanced"
    assert params["smote_applied_count"] == "0"
    assert params["smote_skipped_count"] == "0"
    assert params["smote_seed_component"] == "42"


def test_mlflow_params_include_skip_reason_when_present():
    out = "\n".join([_SKIP_SINGLE_1, _APPLIED_3])
    params = _smote_mlflow_params(
        {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"},
        records=parse_smote_records(out),
    )
    assert params["smote_skipped_reason"] == "single_class"
    assert params["smote_applied_count"] == "1"
    assert params["smote_skipped_count"] == "1"


def test_mlflow_params_omit_skip_reason_when_absent():
    params = _smote_mlflow_params(
        {"smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced"},
        records=parse_smote_records(_APPLIED_3),
    )
    assert "smote_skipped_reason" not in params


# ---------------------------------------------------------------------------
# Governance canary: bare `flwr run` stays incumbent (default OFF)
# ---------------------------------------------------------------------------

def test_pyproject_smote_default_is_off():
    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^smote-enabled\s*=\s*(\w+)', text, re.M)
    assert match is not None, "smote-enabled missing from pyproject.toml"
    assert match.group(1) == "false", (
        f"pyproject.toml smote-enabled must default to false (study knob, "
        f"experiment-relevant path must be inert by default); got {match.group(1)!r}"
    )

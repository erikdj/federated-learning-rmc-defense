"""Tests for praxis_exp/round_trace.py — per-round trace reconstruction.

Focus on the PURE core (build_round_spans / score_config): the corrected
top-k-by-trust selection, the bypass flag, and the soft-ensemble carve-out."""
from praxis_exp.round_trace import (
    build_round_spans, build_round_table, emit_round_table, cleanup_traces,
    score_config, _delete_prior_traces,
)


def _row(sr, cid, mal, score, attack, scorekey="krum_score"):
    return {
        "server_round": sr, "logical_cid": cid, "malicious_gt": mal,
        scorekey: score, "attack_type": (attack if mal else ""),
        "effective_weight": 1000,  # present for everyone — must NOT be used as a keep-flag
    }


def test_score_config():
    assert score_config("krum") == ("krum_score", "hard_topk")
    assert score_config("krumtge") == ("krum_score", "hard_topk")
    assert score_config("trustscore") == ("trust_score", "hard_topk")
    assert score_config("tgensemble")[1] == "soft_ensemble"
    # TGE′ tokens (GWU-53; EXP-016 postmortem — unknown tokens silently lose the
    # timeline selection lens, so prime configs must be mapped explicitly).
    assert score_config("krumtgeprime") == ("krum_score", "hard_topk")
    assert score_config("tgeprime") == ("tge_score", "soft_ensemble")
    assert score_config("fedavg") == (None, "none")
    assert score_config("unknown-x")[1] == "none"


def test_build_round_spans_krum_topk_and_bypass():
    rows = []
    # gaussian (sr5): malicious look like outliers (low trust 0.2), honest high (0.9)
    rows += [_row(5, f"m{i}", True, 0.2, "gaussian_noise") for i in range(9)]
    rows += [_row(5, f"h{i}", False, 0.9, None) for i in range(11)]
    # ALIE (sr11): malicious SATURATE trust to 1.0 (above honest 0.9) -> capture kept set
    rows += [_row(11, f"m{i}", True, 1.0, "alie") for i in range(9)]
    rows += [_row(11, f"h{i}", False, 0.9, None) for i in range(11)]
    # honest-only disconnect (sr8): 11 honest, no malicious present
    rows += [_row(8, f"h{i}", False, 0.9, None) for i in range(11)]
    traj = [
        {"round": 0, "f1": 0.35, "accuracy": 0.53, "loss": 0.71},
        {"round": 5, "f1": 0.92, "accuracy": 0.92, "loss": 0.50},
        {"round": 8, "f1": 0.97, "accuracy": 0.97, "loss": 0.10},
        {"round": 11, "f1": 0.63, "accuracy": 0.61, "loss": 7.70},
    ]
    spans = build_round_spans(rows, traj, defense_token="krum")
    by = {s["server_round"]: s for s in spans}

    # gaussian: outliers filtered, no bypass, healthy
    assert by[5]["attack_type"] == "gaussian_noise"
    assert by[5]["kept_malicious"] == 0
    assert by[5]["keep_count"] == 9  # max(1, 20-9-2)
    assert by[5]["filter_bypassed"] is False
    assert by[5]["is_error"] is False
    assert by[5]["name"] == "r05_gaussian_noise"

    # ALIE: attackers win the top-9, filter bypassed, degraded -> ERROR span
    assert by[11]["attack_type"] == "alie"
    assert by[11]["kept_malicious"] == 9
    assert by[11]["filter_bypassed"] is True
    assert by[11]["is_error"] is True
    assert by[11]["name"] == "r11_alie_BYPASS"
    assert by[11]["malicious_score_max"] == 1.0
    assert by[11]["honest_score_max"] == 0.9

    # honest-only disconnect: no malicious, never bypassed
    assert by[8]["malicious_present"] == 0
    assert by[8]["kept_malicious"] == 0
    assert by[8]["filter_bypassed"] is False

    # round 0 (initial eval): no signal rows -> zero participants, no crash
    assert by[0]["participants"] == 0
    assert by[0]["scenario_round"] == -1
    assert "kept_malicious" not in by[0]  # no scored clients


def test_build_round_spans_tge_soft_ensemble_has_no_hard_count():
    rows = [_row(11, "m0", True, 0.9, "alie", scorekey="tge_score")]
    traj = [{"round": 11, "f1": 0.81, "accuracy": 0.81, "loss": 1.0}]
    spans = build_round_spans(rows, traj, defense_token="tgensemble")
    s = spans[0]
    assert s["selection_mode"] == "soft_ensemble"
    assert "kept_malicious" not in s     # Krum-shaped metric must NOT apply to TGE
    assert "filter_bypassed" not in s
    assert "note" in s
    assert s["name"] == "r11_alie"       # no _BYPASS suffix for soft ensemble


def test_build_round_spans_server_round_offset():
    rows = [_row(20, "h0", False, 0.9, None)]
    traj = [{"round": 20, "f1": 0.98, "accuracy": 0.98, "loss": 0.05}]
    spans = build_round_spans(rows, traj, defense_token="krum")
    assert spans[0]["scenario_round"] == 19  # server_round = scenario_round + 1


# --- idempotency: the delete must match (exp_id, unit_id) and honour the search cap ---

class _TraceStub:
    def __init__(self, tid, name, exp, unit):
        self.info = type("I", (), {
            "trace_id": tid,
            "tags": {"mlflow.traceName": name, "praxis.exp_id": exp, "praxis.unit_id": unit},
        })()


class _FakeTraceClient:
    def __init__(self, traces):
        self._traces = traces
        self.deleted = []
        self.searched_max = None

    def search_traces(self, experiment_ids, max_results=500):
        # guards the exact bug that broke idempotency: max_results=1000 is
        # rejected by SearchTracesV3, so the search must stay within the cap.
        self.searched_max = max_results
        assert max_results <= 500
        return self._traces

    def delete_traces(self, experiment_id, trace_ids):
        self.deleted.extend(trace_ids)


def test_delete_prior_traces_matches_exp_and_unit_only():
    traces = [
        _TraceStub("t1", "fl_training__krum", "EXP-005c", "U"),   # match
        _TraceStub("t2", "fl_training__krum", "EXP-005e", "U"),   # same unit, other launch -> NO
        _TraceStub("t3", "fl_training__krum", "EXP-005c", "V"),   # other unit -> NO
        _TraceStub("t4", "persist_unit",       "EXP-005c", "U"),  # not a round trace -> NO
    ]
    c = _FakeTraceClient(traces)
    n = _delete_prior_traces(c, "40", "EXP-005c", "U")
    assert n == 1
    assert c.deleted == ["t1"]
    assert c.searched_max <= 500


def test_delete_prior_traces_search_failure_is_non_fatal():
    class _Boom:
        def search_traces(self, experiment_ids, max_results=500):
            raise RuntimeError("SearchTracesV3 rejected max_results")
    # a failed cleanup search must return 0 (and warn), never raise into the caller
    assert _delete_prior_traces(_Boom(), "40", "EXP-005c", "U") == 0


# --- GWU-47 Lane B: per-round timeline as a log_table (column-oriented + refresh) ---

def test_build_round_table_is_column_oriented_and_pandas_shaped():
    import pandas as pd
    rows = [_row(5, f"m{i}", True, 0.2, "gaussian_noise") for i in range(9)]
    rows += [_row(5, f"h{i}", False, 0.9, None) for i in range(11)]
    rows += [_row(11, f"m{i}", True, 1.0, "alie") for i in range(9)]
    rows += [_row(11, f"h{i}", False, 0.9, None) for i in range(11)]
    traj = [
        {"round": 5, "f1": 0.92, "accuracy": 0.92, "loss": 0.50},
        {"round": 11, "f1": 0.63, "accuracy": 0.61, "loss": 7.70},
    ]
    spans = build_round_spans(rows, traj, defense_token="krum")
    table = build_round_table(spans)
    # column-oriented dict {col: [values]} — the shape log_table feeds pd.DataFrame,
    # NOT the split-orient {"columns":..,"data":..} on-disk form, and NOT the derived name.
    assert isinstance(table, dict)
    assert "columns" not in table and "data" not in table
    assert "name" not in table
    assert set(table) >= {"server_round", "attack_type", "kept_malicious", "is_error"}
    df = pd.DataFrame(table)
    assert len(df) == len(spans) == 2                 # one row per round
    assert list(df["server_round"]) == [5, 11]
    assert table["note"] == [None, None]              # missing key -> None (krum has no soft note)


def test_build_round_table_empty_returns_none():
    assert build_round_table([]) is None


class _FakeTableClient:
    def __init__(self, existing=()):
        self._existing = [type("A", (), {"path": p})() for p in existing]
        self.deleted = []
        self.logged = []

    def list_artifacts(self, run_id):
        return self._existing

    def delete_artifact(self, run_id, path):
        self.deleted.append(path)

    def log_table(self, run_id, data, artifact_file):
        self.logged.append((run_id, artifact_file))


def test_emit_round_table_refreshes_when_table_exists():
    """log_table APPENDS -> refresh (delete before log) so re-runs don't duplicate rows."""
    c = _FakeTableClient(existing=["round_timeline.json", "other.json"])
    emit_round_table(c, "run-1", {"server_round": [1]})
    assert c.deleted == ["round_timeline.json"]
    assert c.logged == [("run-1", "round_timeline.json")]


def test_emit_round_table_logs_without_delete_when_absent():
    c = _FakeTableClient(existing=["other.json"])
    emit_round_table(c, "run-1", {"server_round": [1]})
    assert c.deleted == []
    assert c.logged == [("run-1", "round_timeline.json")]


def test_emit_round_table_skips_none_table():
    c = _FakeTableClient()
    emit_round_table(c, "run-1", None)
    assert c.logged == [] and c.deleted == []


# --- GWU-47 Lane B: one-time cleanup of retired traces (round + coarse) ---

class _FakeCleanupClient:
    def __init__(self, traces):
        self._traces = traces
        self.deleted = []

    def get_or_create_experiment(self, name):
        return "40"

    def search_traces(self, experiment_ids, max_results=500):
        assert max_results <= 500          # SearchTracesV3 cap
        return self._traces

    def delete_traces(self, experiment_id, trace_ids):
        self.deleted.extend(trace_ids)


def test_cleanup_traces_removes_round_and_coarse_but_scopes_round_to_sweep(monkeypatch, tmp_path):
    import praxis_exp.round_trace as rt
    monkeypatch.setattr(rt, "_experiment_name", lambda repo, eid: "calibration")
    traces = [
        _TraceStub("t1", "fl_training__krum", "EXP-005e", "U1"),  # this sweep -> delete
        _TraceStub("t2", "run_phase4_flower", None, None),         # coarse (untagged) -> delete
        _TraceStub("t3", "persist_unit",       None, None),        # coarse (untagged) -> delete
        _TraceStub("t4", "fl_training__krum", "EXP-005c", "U1"),  # OTHER sweep -> keep
    ]
    c = _FakeCleanupClient(traces)
    out = cleanup_traces(tmp_path, "EXP-005e", _client=c)
    assert out["traces_deleted"] == 3
    assert set(c.deleted) == {"t1", "t2", "t3"}


def test_cleanup_traces_idempotent_no_traces(monkeypatch, tmp_path):
    import praxis_exp.round_trace as rt
    monkeypatch.setattr(rt, "_experiment_name", lambda repo, eid: "calibration")
    c = _FakeCleanupClient([])
    assert cleanup_traces(tmp_path, "EXP-005e", _client=c)["traces_deleted"] == 0
    assert c.deleted == []

import json

import pytest

from praxis_exp.storage import InMemoryObjectStore
from praxis_exp.units import expand_matrix
from praxis_exp.manifest import write_manifest
from docker.entrypoint import resolve_unit, should_skip


def _seed_manifest(store):
    units = expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42], "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, "EXP-005", units, meta={"methodology_version": "v1.9"})
    return units


def test_resolve_unit_by_array_index():
    store = InMemoryObjectStore(); _seed_manifest(store)
    u = resolve_unit(store, "EXP-005", 1)
    assert u.array_index == 1


def test_should_skip_reflects_done_marker(tmp_path):
    from praxis_exp.integrity import persist_unit
    store = InMemoryObjectStore(); units = _seed_manifest(store)
    u = units[0]
    assert should_skip(store, "EXP-005", u) is False
    r = tmp_path/"r.json"; r.write_text("{}"); s = tmp_path/"s.jsonl"; s.write_text("{}\n")
    persist_unit(store, "EXP-005", u.unit_id, r, s)
    assert should_skip(store, "EXP-005", u) is True


def test_defense_token_known_and_unknown():
    from docker.entrypoint import defense_token
    assert defense_token("Krum") == "krum"
    assert defense_token("Krum+TGE") == "krumtge"
    assert defense_token("TGE") == "tgensemble"
    # TGE′ : strategy-class-lowered convention (server_app.py:78) —
    # ScenarioTGEPrime -> "tgeprime", ScenarioKrumTGEPrime -> "krumtgeprime".
    # EXP-016 postmortem: these were missing from _DEFENSE_TOKEN, so both smoke
    # units trained 50 rounds then crashed in finalization (exit 1, 2026-07-25).
    assert defense_token("TGEprime") == "tgeprime"
    assert defense_token("Krum+TGEprime") == "krumtgeprime"
    import pytest
    with pytest.raises(KeyError):
        defense_token("NoSuchConfig")


def test_defense_token_covers_every_supported_config():
    """Drift canary (EXP-016 postmortem): every runner config MUST have a signal-
    log token, so adding a config to SUPPORTED_CONFIGS without updating
    _DEFENSE_TOKEN fails HERE instead of in post-training finalization on Batch."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_phase4_flower import SUPPORTED_CONFIGS
    from docker.entrypoint import defense_token
    for config in SUPPORTED_CONFIGS:
        assert defense_token(config)  # raises KeyError on any unmapped config


def _unit(config="Krum", scenario="S0", mode="persistent_optimizer", seed=42):
    from praxis_exp.units import Unit
    return Unit(config, scenario, mode, seed, 2_000_000, 50, 0)


def test_build_unit_tags_are_metadata_only():
    """redesign § 2: tags carry identity/provenance/dataset_name ONLY — the
    experiment inputs (config/scenario/seed/mode) are PARAMS, not tags."""
    from docker.entrypoint import build_unit_tags
    tags = build_unit_tags(_unit(), methodology_version="v1.17", image_digest="sha256:abc")
    assert tags == {
        "unit_id": _unit().unit_id,
        "defense_token": "krum",
        "methodology_version": "v1.17",
        "image_digest": "sha256:abc",
        "dataset_name": "edge_full_20_rmc",
    }


def test_build_unit_params_are_inputs_only():
    """redesign § 2: params are the experiment INPUTS. ``defense`` is the config
    label (``Krum``); the signal-log token lives in the ``defense_token`` tag."""
    from docker.entrypoint import build_unit_params
    params = build_unit_params(_unit())
    assert params == {
        "defense": "Krum",
        "scenario": "S0",
        "seed": "42",
        "mode": "persistent_optimizer",
        "rounds": "50",
        "max_per_client": "2000000",
        "reporting_split": "val",
    }


def test_params_and_tags_are_disjoint():
    """redesign § 3 rule: a key appears in EXACTLY ONE of params/tags."""
    from docker.entrypoint import build_unit_params, build_unit_tags
    params = build_unit_params(_unit())
    tags = build_unit_tags(_unit(), methodology_version="v1.17", image_digest="sha256:abc")
    assert set(params) & set(tags) == set()


def test_s3_uris_for_unit_matches_persist_unit_key_layout():
    """req 5: no duplicated string logic — same key functions persist_unit uses."""
    from docker.entrypoint import s3_uris_for_unit
    from praxis_exp import storage
    u = _unit()
    tags = s3_uris_for_unit("praxis-bucket", "EXP-005", u.unit_id)
    assert tags["s3_result_uri"] == f"s3://praxis-bucket/{storage.result_key('EXP-005', u.unit_id)}"
    assert tags["s3_signal_uri"] == f"s3://praxis-bucket/{storage.signal_key('EXP-005', u.unit_id)}"
    assert tags["s3_done_uri"] == f"s3://praxis-bucket/{storage.marker_key('EXP-005', u.unit_id)}"
    assert "s3_console_url" in tags


def test_trajectory_metrics_returns_step_keyed_triples():
    """req C metrics: per-round accuracy/f1/loss as step metrics (step=round)."""
    from docker.entrypoint import trajectory_metrics
    result = {
        "trajectory": [
            {"round": 1, "accuracy": 0.5, "f1": 0.4, "loss": 1.2},
            {"round": 2, "accuracy": 0.6, "f1": 0.5, "loss": 1.0},
        ]
    }
    points = trajectory_metrics(result)
    assert ("accuracy", 0.5, 1) in points
    assert ("f1", 0.4, 1) in points
    assert ("loss", 1.2, 1) in points
    assert ("accuracy", 0.6, 2) in points
    assert len(points) == 6


def test_trajectory_metrics_empty_when_no_trajectory():
    from docker.entrypoint import trajectory_metrics
    assert trajectory_metrics({}) == []
    assert trajectory_metrics({"trajectory": []}) == []


def test_trajectory_metrics_skips_missing_keys():
    from docker.entrypoint import trajectory_metrics
    result = {"trajectory": [{"round": 1, "accuracy": 0.5}]}
    points = trajectory_metrics(result)
    assert points == [("accuracy", 0.5, 1)]


def test_trajectory_metrics_includes_precision_recall_when_present():
    """New-image trajectories carry precision/recall too — all five live-contract
    metrics get logged per round."""
    from docker.entrypoint import trajectory_metrics
    result = {"trajectory": [
        {"round": 1, "accuracy": 0.5, "precision": 0.55, "recall": 0.45, "f1": 0.4, "loss": 1.2},
    ]}
    points = trajectory_metrics(result)
    assert ("precision", 0.55, 1) in points
    assert ("recall", 0.45, 1) in points
    assert len(points) == 5


def test_trajectory_metrics_includes_per_class_when_present():
    """Per-class benign/attack metrics parity with the direct-log path in
    run_phase4_flower.py — self-heal/backfill-replayed units get the same MLflow
    series."""
    from docker.entrypoint import trajectory_metrics
    result = {"trajectory": [{
        "round": 1, "accuracy": 0.5, "f1": 0.4, "loss": 1.2,
        "attack_precision": 0.7, "attack_recall": 0.6, "attack_f1": 0.65,
        "benign_precision": 0.8, "benign_recall": 0.9, "benign_f1": 0.85,
    }]}
    points = trajectory_metrics(result)
    for key in ("attack_precision", "attack_recall", "attack_f1",
                "benign_precision", "benign_recall", "benign_f1"):
        assert (key, result["trajectory"][0][key], 1) in points


def test_trajectory_metrics_skips_per_class_when_absent():
    """A legacy trajectory (no per-class fields) emits none of them — never
    fabricated."""
    from docker.entrypoint import trajectory_metrics
    result = {"trajectory": [{"round": 1, "accuracy": 0.5, "f1": 0.4, "loss": 1.2}]}
    points = trajectory_metrics(result)
    emitted = {key for key, _, _ in points}
    assert emitted == {"accuracy", "f1", "loss"}


def test_final_metrics_present_subset():
    """req C: final_accuracy, final_f1, mean_accuracy where present."""
    from docker.entrypoint import final_metrics
    result = {"final_accuracy": 0.79, "final_f1": 0.75, "mean_accuracy": 0.80, "other": 1}
    assert final_metrics(result) == {
        "final_accuracy": 0.79, "final_f1": 0.75, "mean_accuracy": 0.80,
    }


def test_final_metrics_omits_missing_or_none():
    from docker.entrypoint import final_metrics
    result = {"final_accuracy": 0.79, "final_f1": None}
    assert final_metrics(result) == {"final_accuracy": 0.79}


def test_final_metrics_adds_wall_clock_rounds_and_final_loss():
    """redesign § 2 item 7: wall_clock_sec (elapsed_seconds), rounds_completed
    (trajectory length) and final_loss (last trajectory row) are derived when
    present."""
    from docker.entrypoint import final_metrics
    result = {
        "final_accuracy": 0.79,
        "final_f1": 0.75,
        "mean_accuracy": 0.80,
        "elapsed_seconds": 123.5,
        "trajectory": [
            {"round": 1, "accuracy": 0.5, "f1": 0.4, "loss": 1.2},
            {"round": 2, "accuracy": 0.6, "f1": 0.5, "loss": 0.9},
        ],
    }
    metrics = final_metrics(result)
    assert metrics["wall_clock_sec"] == 123.5
    assert metrics["rounds_completed"] == 2.0
    assert metrics["final_loss"] == 0.9
    assert metrics["final_accuracy"] == 0.79


def test_cold_start_model_tag_present_for_cs_configs():
    """req 6 (models): tag TGE/Krum+TGE (and any CS-using config) with the CS
    pkl path/name actually used, read from the result's own provenance block
    (grounded — no re-derivation of runner internals)."""
    from docker.entrypoint import cold_start_model_tag
    result = {"provenance": {"cs_model_path": "models/cold_start/flower_reset/S_k3_final.pkl"}}
    assert cold_start_model_tag(result) == "models/cold_start/flower_reset/S_k3_final.pkl"


def test_cold_start_model_tag_none_when_not_used():
    from docker.entrypoint import cold_start_model_tag
    assert cold_start_model_tag({"provenance": {"cs_model_path": ""}}) is None
    assert cold_start_model_tag({"provenance": {}}) is None
    assert cold_start_model_tag({}) is None


def test_build_unit_note_contains_summary_and_console_link():
    from docker.entrypoint import build_unit_note
    note = build_unit_note(
        _unit(),
        result={"final_accuracy": 0.79, "final_f1": 0.75},
        console_url="https://us-east-1.console.aws.amazon.com/s3/buckets/b?prefix=x/",
        result_filename="phase4_flower__krum__seed42.json",
    )
    assert _unit().unit_id in note
    assert "phase4_flower__krum__seed42.json" in note
    assert "https://us-east-1.console.aws.amazon.com/s3/buckets/b?prefix=x/" in note
    assert "0.79" in note


def test_main_skip_path_never_touches_mlflow(monkeypatch):
    """ : the enrichment moved MLflow contact
    (set_tracking_uri / set_experiment / tracing) BEFORE the should_skip
    check — so a Batch retry of an already-committed unit would exit nonzero
    on a transient MLflow outage and retry pointlessly, even though its
    artifacts are already durable. The skip path must resolve the unit,
    check the done-marker, and return 0 before ANY mlflow attribute is
    touched."""
    from docker import entrypoint
    from praxis_exp import storage

    store = InMemoryObjectStore()
    units = expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42],
                          "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, "EXP-005", units, meta={"methodology_version": "v1.9"})
    store.put_bytes(storage.marker_key("EXP-005", units[0].unit_id), b"")  # unit 0 committed

    class _ExplodingMlflow:
        def __getattr__(self, name):
            raise AssertionError(
                f"tracking server touched via mlflow.{name} on the skip path"
            )

    monkeypatch.setenv("PRAXIS_EXP_ID", "EXP-005")
    monkeypatch.setenv("PRAXIS_BUCKET", "b")
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "0")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unreachable:5000")
    monkeypatch.setenv("PRAXIS_MLFLOW_EXPERIMENT", "h2-dev-sweep")
    assert entrypoint.main(_store=store, _mlflow=_ExplodingMlflow()) == 0


def test_runner_argv_includes_rounds_and_all_unit_fields():
    """ : the runner defaults --rounds to 50
    (run_phase4_flower.py:1180), and the entrypoint's subprocess argv never
    passed it — a PRE-EXISTING omission (the pre-branch argv also lacked it)
    surfaced by the enrichment now logging unit.rounds as the run param:
    any matrix declaring rounds != 50 would have run 50 rounds while the
    manifest/MLflow advertised the declared value. The argv must carry every
    unit field the manifest records."""
    import sys as _sys
    from pathlib import Path as _Path
    from docker.entrypoint import runner_argv
    from praxis_exp.units import Unit
    u = Unit("Krum", "S4", "persistent_optimizer", 42, 2_000_000, 40, 0)  # rounds=40 != default
    argv = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=_Path("results/EXP-005/x"))

    def arg_after(flag):
        return argv[argv.index(flag) + 1]

    assert argv[0] == _sys.executable
    assert argv[1] == "scripts/run_phase4_flower.py"
    assert arg_after("--rounds") == "40"
    assert arg_after("--configs") == "Krum"
    assert arg_after("--modes") == "persistent_optimizer"
    assert arg_after("--seeds") == "42"
    assert arg_after("--scenario") == "rmc/scenarios/S4.json"
    assert arg_after("--max-per-client") == "2000000"
    assert arg_after("--reporting-split") == "val"
    assert arg_after("--output-dir") == "results/EXP-005/x"


class _FakeFluentMlflow:
    """Fake for the fluent-mlflow surface post_persist_enrichment uses
    (set_tag / log_artifact). Records calls; raises on configured tag keys."""

    def __init__(self, fail_on_tags=()):
        self.calls = []
        self._fail_on = set(fail_on_tags)

    def set_tag(self, key, value):
        self.calls.append(("set_tag", key, value))
        if key in self._fail_on:
            raise RuntimeError(f"transient mlflow error on {key}")

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def log_input(self, dataset, context="training"):
        self.calls.append(("log_input", getattr(dataset, "name", None), context))


def test_post_persist_enrichment_swallows_mlflow_errors(tmp_path, capsys):
    """ : after persist_unit commits the
    done-marker, MLflow decoration is best-effort — a transient MLflow
    failure must NOT raise (a nonzero exit triggers a Batch retry that
    no-ops on the done-marker, leaving a spurious failed attempt)."""
    from docker.entrypoint import post_persist_enrichment
    fake = _FakeFluentMlflow(fail_on_tags={"s3_result_uri"})
    ok = post_persist_enrichment(
        fake, _unit(), bucket="b", exp_id="EXP-005",
        result={}, model_path=tmp_path / "absent.pt",
    )  # must not raise
    assert ok is False  # a failed decoration block reports False (never raises)
    assert "WARN" in capsys.readouterr().out


def test_post_persist_enrichment_sets_unit_status_done_last(tmp_path):
    """unit_status=done is attempted LAST (inside the guarded block) so a
    fully decorated run is the only one marked done."""
    from docker.entrypoint import post_persist_enrichment
    fake = _FakeFluentMlflow()
    ok = post_persist_enrichment(
        fake, _unit(), bucket="b", exp_id="EXP-005",
        result={"final_accuracy": 0.7}, model_path=tmp_path / "absent.pt",
    )
    assert ok is True  # a fully-completed decoration block reports True
    set_tags = [c for c in fake.calls if c[0] == "set_tag"]
    assert set_tags[-1][1] == "unit_status" and set_tags[-1][2] == "done"
    # S3 links + note landed before it
    keys = [c[1] for c in set_tags]
    assert {"s3_result_uri", "s3_signal_uri", "s3_done_uri",
            "s3_console_url", "mlflow.note.content"} <= set(keys)


def test_post_persist_enrichment_sets_criteria_ok_before_done(tmp_path):
    """criteria_ok is written on the live success path (before unit_status=done)
    with the SAME clean-completion logic as the Lane D backfill, so live and
    backfilled runs tag identically."""
    from docker.entrypoint import post_persist_enrichment
    ok = _FakeFluentMlflow()
    post_persist_enrichment(
        ok, _unit(), bucket="b", exp_id="EXP-005",
        result={"return_code": 0, "trajectory": [{"round": 1}]},
        model_path=tmp_path / "absent.pt",
    )
    ok_tags = {c[1]: c[2] for c in ok.calls if c[0] == "set_tag"}
    assert ok_tags["criteria_ok"] == "true"
    keys = [c[1] for c in ok.calls if c[0] == "set_tag"]
    assert keys.index("criteria_ok") < keys.index("unit_status")  # done stays last

    bad = _FakeFluentMlflow()
    post_persist_enrichment(
        bad, _unit(), bucket="b", exp_id="EXP-005",
        result={"return_code": 0, "trajectory": []},  # empty trajectory -> unclean
        model_path=tmp_path / "absent.pt",
    )
    bad_tags = {c[1]: c[2] for c in bad.calls if c[0] == "set_tag"}
    assert bad_tags["criteria_ok"] == "false"


def test_post_persist_enrichment_logs_model_when_present(tmp_path):
    from docker.entrypoint import post_persist_enrichment
    model = tmp_path / "phase4_flower__krum__seed42__model.pt"
    model.write_bytes(b"x")
    fake = _FakeFluentMlflow()
    post_persist_enrichment(
        fake, _unit(), bucket="b", exp_id="EXP-005", result={}, model_path=model,
    )
    assert ("log_artifact", str(model)) in fake.calls
    assert ("set_tag", "model_file", model.name) in fake.calls


def test_build_unit_note_includes_rmc_params_and_verdict():
    """: the note carries RMC params (mode/rounds/max) + criteria_ok."""
    from docker.entrypoint import build_unit_note
    note = build_unit_note(
        _unit(mode="persistent_optimizer"),
        result={"return_code": 0, "trajectory": [{"round": 1}], "final_f1": 0.9},
        console_url="https://s3", result_filename="r.json",
    )
    assert "Mode: `persistent_optimizer`" in note
    assert "Rounds:" in note and "Max/client:" in note
    assert "criteria_ok: `True`" in note


def test_post_persist_enrichment_logs_signal_dataset(tmp_path):
    """: the live container references the signal log as a
    dataset-by-source (context "signal") — parity with the backfill path, no
    byte copy. unit_status=done still stays last."""
    from docker.entrypoint import post_persist_enrichment
    fake = _FakeFluentMlflow()
    post_persist_enrichment(
        fake, _unit(), bucket="b", exp_id="EXP-005", result={},
        model_path=tmp_path / "absent.pt",
    )
    sig = [c for c in fake.calls if c[0] == "log_input"]
    assert sig and sig[0][1].startswith("signal_") and sig[0][2] == "signal"
    set_tags = [c for c in fake.calls if c[0] == "set_tag"]
    assert set_tags[-1][1] == "unit_status"  # done remains last


# ---------------------------------------------------------------------------
# req 1/3/4/5/6/8 — lifecycle + start-metadata + enrichment helpers
# ---------------------------------------------------------------------------

class _SysMetrics:
    def __init__(self, fail=False):
        self.enabled = False
        self._fail = fail

    def enable_system_metrics_logging(self):
        if self._fail:
            raise RuntimeError("psutil unavailable")
        self.enabled = True


class _Pytorch:
    def __init__(self, fail=False):
        self.logged = []
        self._fail = fail

    def log_model(self, model, name=None, artifact_path=None, registered_model_name=None,
                  signature=None, input_example=None):
        if self._fail:
            raise RuntimeError("log_model needs a 3.x server (2.18 404s)")
        self.logged.append(
            {"model": model, "name": name or artifact_path, "registered": registered_model_name,
             "signature": signature, "input_example": input_example}
        )
        return type("_ModelInfo", (), {"model_id": "m-fake-123"})()


class _Run:
    def __init__(self, run_id):
        self.info = type("_Info", (), {"run_id": run_id})()


class _FullFakeMlflow:
    """Full fluent-mlflow surface main + the start-metadata helpers use."""

    def __init__(self, *, sysmetrics_fail=False, pytorch_fail=False, run_id="run-xyz",
                 experiment_id="exp-fake-1"):
        self.calls = []
        self.tags = {}
        self.params = {}
        self.metrics = []
        self.model_metrics = []
        self.ended = []
        self.inputs = []
        self.system_metrics = _SysMetrics(sysmetrics_fail)
        self.pytorch = _Pytorch(pytorch_fail)
        self._run_id = run_id
        self._experiment_id = experiment_id

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        # main now reads experiment.experiment_id off the return
        # value to scope the resume-by-unit lookup, so the fake returns an object
        # exposing it (fluent mlflow.set_experiment returns an Experiment).
        self.calls.append(("set_experiment", name))
        return type("_Exp", (), {"experiment_id": self._experiment_id})()

    def start_run(self, run_name=None, run_id=None):
        # main resumes via start_run(run_id=...) or creates via
        # start_run(run_name=...) — the fake accepts and records both.
        self.calls.append(("start_run", run_name, run_id))
        return _Run(run_id or self._run_id)

    def set_tag(self, key, value):
        self.calls.append(("set_tag", key, value))
        self.tags[key] = value

    def log_param(self, key, value):
        self.calls.append(("log_param", key, value))
        self.params[key] = value

    def log_metric(self, key, value, step=0, model_id=None):
        self.calls.append(("log_metric", key, value, step))
        self.metrics.append((key, value, step))
        if model_id is not None:
            self.model_metrics.append((key, value, model_id))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def log_input(self, dataset, context=None):
        self.calls.append(("log_input", getattr(dataset, "name", None), context))
        self.inputs.append((dataset, context))

    def end_run(self, status="FINISHED"):
        self.calls.append(("end_run", status))
        self.ended.append(status)


def test_run_controller_terminate_is_idempotent_first_status_wins():
    from docker.entrypoint import _RunController
    fake = _FullFakeMlflow()
    ctrl = _RunController(fake, "run-1")
    ctrl.terminate("FINISHED")
    ctrl.terminate("FAILED")  # must be a no-op
    assert fake.ended == ["FINISHED"]


def test_run_controller_sigterm_marks_killed_and_exits():
    from docker.entrypoint import _RunController
    fake = _FullFakeMlflow()
    ctrl = _RunController(fake, "run-1")
    with pytest.raises(SystemExit):
        ctrl._on_sigterm(15, None)
    assert fake.tags["unit_status"] == "killed"
    assert "reclaim_reason" in fake.tags
    assert fake.ended == ["KILLED"]


def test_run_controller_sigterm_after_commit_ends_finished():
    """A SIGTERM during the slow post-commit decoration must NOT flip a durably
    committed unit to KILLED — it ends FINISHED."""
    from docker.entrypoint import _RunController
    fake = _FullFakeMlflow()
    ctrl = _RunController(fake, "run-1")
    ctrl.mark_committed(criteria_ok=True)
    with pytest.raises(SystemExit) as ei:
        ctrl._on_sigterm(15, None)
    assert ei.value.code == 0
    assert fake.ended == ["FINISHED"]
    assert fake.tags["unit_status"] == "done"
    assert fake.tags["criteria_ok"] == "true"  # captured criteria_ok tagged on the SIGTERM path
    assert "reclaim_reason" not in fake.tags


def test_run_controller_mark_error_never_raises():
    from docker.entrypoint import _RunController

    class _Boom:
        def set_tag(self, *a):
            raise RuntimeError("mlflow down")

    _RunController(_Boom(), "run-1").mark_error()  # must not raise


def test_enable_system_metrics_enables_and_swallows_errors():
    from docker.entrypoint import _enable_system_metrics
    ok = _FullFakeMlflow()
    _enable_system_metrics(ok)
    assert ok.system_metrics.enabled is True
    _enable_system_metrics(_FullFakeMlflow(sysmetrics_fail=True))  # must not raise


def test_log_dataset_input_logs_native_dataset(monkeypatch, tmp_path):
    from docker.entrypoint import log_dataset_input
    monkeypatch.chdir(tmp_path)  # no dataset metadata on disk -> digest None, still logs
    fake = _FullFakeMlflow()
    ok = log_dataset_input(fake, bucket="praxis-bucket")
    assert ok is True  # reports True when the input logged
    assert ("log_input", "edge_full_20_rmc", "training") in fake.calls


def test_log_dataset_input_swallows_build_errors():
    from docker.entrypoint import log_dataset_input

    class _Boom:
        def log_input(self, *a, **k):
            raise RuntimeError("boom")

    ok = log_dataset_input(_Boom(), bucket="b")  # must not raise
    assert ok is False  # reports False on a swallowed failure


def test_set_start_metadata_sets_params_tags_links_and_dataset(monkeypatch, tmp_path):
    from docker.entrypoint import _set_start_metadata
    monkeypatch.chdir(tmp_path)
    fake = _FullFakeMlflow()
    ok = _set_start_metadata(
        fake, _unit(), bucket="praxis-bucket", exp_id="EXP-005",
        methodology_version="v1.17", image_digest="sha256:abc", parent_run_id="parent-1",
    )
    assert ok is True  # returns the dataset-input outcome (True on full success)
    assert fake.tags["mlflow.parentRunId"] == "parent-1"
    assert fake.tags["defense_token"] == "krum"
    assert fake.tags["dataset_name"] == "edge_full_20_rmc"
    assert fake.tags["unit_status"] == "running"
    assert fake.tags["s3_result_uri"].startswith("s3://praxis-bucket/")
    assert "cloudwatch_log_url" in fake.tags
    assert fake.params["defense"] == "Krum" and fake.params["reporting_split"] == "val"
    # params vs tags disjoint at run start
    tag_keys = {c[1] for c in fake.calls if c[0] == "set_tag"}
    assert set(fake.params) & tag_keys == set()
    assert any(c[0] == "log_input" for c in fake.calls)


class _FakeTensor:
    """Minimal tensor stand-in for the signature forward-pass chain
    (torch.from_numpy(x) -> model(x) ->.detach.numpy)."""
    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def numpy(self):
        import numpy as np
        return np.asarray(self._arr)


def _fake_torch(loaded):
    class _T:
        def load(self, path, map_location=None):
            return loaded

        def from_numpy(self, arr):
            return _FakeTensor(arr)
    return _T()


class _FakeNet:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, sd):
        self.loaded = sd

    def eval(self):
        return self

    def __call__(self, x):  # forward pass for signature inference
        return _FakeTensor(x.numpy() if hasattr(x, "numpy") else x)


def test_log_native_model_reconstructs_state_dict_and_registers(tmp_path):
    from docker.entrypoint import log_native_model
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    fake = _FullFakeMlflow()
    net = _FakeNet()
    log_native_model(
        fake, model_path=model, defense_token="krum", model_dataset="edge_full",
        _torch=_fake_torch({"w": 1}), _create_model=lambda ds: net,
    )
    assert fake.pytorch.logged[0]["registered"] == "praxis-krum"
    assert net.loaded == {"w": 1}  # state_dict rehydrated into the module


def test_log_native_model_attaches_signature_and_links_model_id(tmp_path, monkeypatch):
    """: the logged model carries a signature + deterministic
    input_example, and final metrics link to it via model_id. The dataset
    metadata is created in the test so no machine-local data checkout is used."""
    from docker.entrypoint import log_native_model
    dataset_dir = tmp_path / "data" / "edge_full_20"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"_meta": {"num_features": 45}})
    )
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    fake = _FullFakeMlflow()
    net = _FakeNet()
    log_native_model(
        fake, model_path=model, defense_token="krum", model_dataset="edge_full_20",
        result={"final_f1": 0.9, "final_accuracy": 0.9, "trajectory": []},
        _torch=_fake_torch({"w": 1}), _create_model=lambda ds: net,
    )
    logged = fake.pytorch.logged[0]
    assert logged["registered"] == "praxis-krum"
    assert logged["signature"] is not None       # signature attached
    assert logged["input_example"] is not None   # deterministic zeros sample
    # final metrics linked to the model version via model_id
    assert fake.model_metrics and all(m[2] == "m-fake-123" for m in fake.model_metrics)
    assert {m[0] for m in fake.model_metrics} >= {"final_f1", "final_accuracy"}


def test_log_native_model_logs_module_directly(tmp_path):
    from docker.entrypoint import log_native_model
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    fake = _FullFakeMlflow()
    module = object()  # a non-dict = already a Module
    log_native_model(
        fake, model_path=model, defense_token="tgensemble", model_dataset="edge_full",
        _torch=_fake_torch(module), _create_model=lambda ds: _FakeNet(),
    )
    assert fake.pytorch.logged[0]["model"] is module
    assert fake.pytorch.logged[0]["registered"] == "praxis-tgensemble"


def test_log_native_model_noop_when_model_absent(tmp_path):
    from docker.entrypoint import log_native_model
    fake = _FullFakeMlflow()
    log_native_model(
        fake, model_path=tmp_path / "absent.pt", defense_token="krum",
        model_dataset="edge_full", _torch=_fake_torch({}), _create_model=lambda ds: _FakeNet(),
    )
    assert fake.pytorch.logged == []


def test_log_native_model_swallows_log_model_errors(tmp_path):
    from docker.entrypoint import log_native_model
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    fake = _FullFakeMlflow(pytorch_fail=True)
    log_native_model(  # must not raise even though log_model raises
        fake, model_path=model, defense_token="krum", model_dataset="edge_full",
        _torch=_fake_torch(object()), _create_model=lambda ds: _FakeNet(),
    )


def _setup_main(monkeypatch, tmp_path, *, rc=0, write_signal=True):
    """Wire main for an integration test: manifest in an in-memory store, env
    set, and subprocess.call faked to emulate the runner writing result+signal."""
    import docker.entrypoint as ep
    from praxis_exp.runner_paths import result_filename, signal_filename
    from docker.entrypoint import defense_token as _dt

    store = InMemoryObjectStore()
    units = expand_matrix(["Krum", "TrustScore"], ["S0", "S4"], [42],
                          "persistent_optimizer", 2_000_000, 50)
    write_manifest(store, "EXP-005", units, meta={"methodology_version": "v1.9"})
    unit = units[0]  # array_index 0 = Krum/S0
    monkeypatch.chdir(tmp_path)
    for k, v in {
        "PRAXIS_EXP_ID": "EXP-005", "PRAXIS_BUCKET": "b",
        "AWS_BATCH_JOB_ARRAY_INDEX": "0", "MLFLOW_TRACKING_URI": "http://unreachable:5000",
        "PRAXIS_MLFLOW_EXPERIMENT": "h2-dev-sweep", "PRAXIS_PARENT_RUN_ID": "parent-1",
    }.items():
        monkeypatch.setenv(k, v)

    out_dir = tmp_path / "results" / "EXP-005" / unit.unit_id
    sig_dir = tmp_path / "signals"

    def fake_call(argv, env=None):
        if rc != 0:
            return rc
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / result_filename(unit)).write_text(json.dumps({
            "return_code": 0, "elapsed_seconds": 12.3,
            "final_accuracy": 0.79, "final_f1": 0.75, "mean_accuracy": 0.80,
            "trajectory": [{"round": 1, "accuracy": 0.5, "f1": 0.4, "loss": 1.2}],
            "provenance": {"cs_model_path": ""},
        }))
        if write_signal:
            sig_dir.mkdir(parents=True, exist_ok=True)
            (sig_dir / signal_filename(unit, defense_token=_dt(unit.config))).write_text("{}\n")
        return 0

    monkeypatch.setattr(ep.subprocess, "call", fake_call)
    # keep the main integration tests hermetic — default the
    # resume-by-unit seam to "no prior run" (the create path) so they never reach
    # a real mlflow.tracking.MlflowClient / tracking server. Tests that exercise
    # the resume path override this after calling _setup_main. raising=False so
    # this helper is usable during the RED phase before the seam exists.
    monkeypatch.setattr(ep, "_find_resumable_run_id", lambda *a, **k: None, raising=False)
    return store, unit


def test_main_success_ends_finished_and_progresses_status(monkeypatch, tmp_path):
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert fake.ended == ["FINISHED"]
    status = [c[2] for c in fake.calls if c[0] == "set_tag" and c[1] == "unit_status"]
    assert status[0] == "running" and status[-1] == "done"
    # params vs tags disjoint across the whole run
    tag_keys = {c[1] for c in fake.calls if c[0] == "set_tag"}
    assert set(fake.params) & tag_keys == set()
    metric_keys = {m[0] for m in fake.metrics}
    assert {"final_accuracy", "wall_clock_sec", "rounds_completed"} <= metric_keys
    assert any(c[0] == "log_input" for c in fake.calls)
    assert fake.system_metrics.enabled is True


def test_main_runner_failure_ends_failed(monkeypatch, tmp_path):
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=7)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 7
    assert fake.tags["unit_status"] == "runner_failed"
    assert fake.ended == ["FAILED"]


def test_main_persist_failure_ends_failed_and_reraises(monkeypatch, tmp_path):
    import docker.entrypoint as ep
    from praxis_exp.integrity import IntegrityError
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0, write_signal=False)
    fake = _FullFakeMlflow()
    with pytest.raises(IntegrityError):
        ep.main(_store=store, _mlflow=fake)
    assert fake.ended == ["FAILED"]
    assert fake.tags.get("unit_status") == "errored"


# ---------------------------------------------------------------------------
# resume-by-unit: _find_resumable_run_id seam + main resume
# ---------------------------------------------------------------------------

class _FakeResumeClient:
    """Bare-client stand-in driven through _find_resumable_run_id's
    ``_client_factory`` seam: records the search and returns preconfigured runs
    (oldest-first, matching MlflowClient.search_runs order_by start_time ASC)."""

    def __init__(self, runs=None, *, raise_on_search=False):
        self._runs = list(runs or [])
        self._raise = raise_on_search
        self.searches = []

    def search_runs(self, experiment_ids, filter_string="", order_by=None,
                    max_results=None, **kwargs):
        self.searches.append({
            "experiment_ids": experiment_ids, "filter_string": filter_string,
            "order_by": order_by, "max_results": max_results,
        })
        if self._raise:
            raise RuntimeError("mlflow search_runs failed (server down)")
        return self._runs


def test_find_resumable_run_id_returns_newest_for_parent_unit():
    """Two prior runs for (parent, unit), oldest-first -> the NEWEST run id (last
    in start_time-ASC order) is returned, and the filter/order match
    PraxisMlflowClient.find_runs_by_unit exactly (parent-scoped clause included)."""
    from docker.entrypoint import _find_resumable_run_id
    client = _FakeResumeClient(runs=[_Run("attempt-old"), _Run("attempt-new")])
    got = _find_resumable_run_id(
        "exp-1", "Krum__S0__seed42", "parent-7", _client_factory=lambda: client,
    )
    assert got == "attempt-new"
    s = client.searches[0]
    assert s["experiment_ids"] == ["exp-1"]
    assert s["filter_string"] == (
        "tags.unit_id = 'Krum__S0__seed42' "
        "and tags.`mlflow.parentRunId` = 'parent-7'"
    )
    assert s["order_by"] == ["attributes.start_time ASC"]


def test_find_resumable_run_id_requires_parent_scope():
    """FIX-4 (L4): resume is only safe PARENT-SCOPED. With no parent_run_id the
    lookup returns None immediately and issues NO search — an unparented
    ``tags.unit_id`` match spans the WHOLE experiment and could cross-resume a
    DIFFERENT launch's run for the same unit slug. Production always injects
    PRAXIS_PARENT_RUN_ID (matrix_launch.py); a direct/manual submit must mint a
    fresh run, never cross-resume."""
    from docker.entrypoint import _find_resumable_run_id
    client = _FakeResumeClient(runs=[_Run("would-cross-resume")])
    got = _find_resumable_run_id("exp-1", "u-1", None, _client_factory=lambda: client)
    assert got is None
    assert client.searches == []  # no search issued at all


def test_find_resumable_run_id_none_when_absent():
    """No prior run for (parent, unit) -> None (caller mints a fresh run)."""
    from docker.entrypoint import _find_resumable_run_id
    client = _FakeResumeClient(runs=[])
    assert _find_resumable_run_id(
        "exp-1", "u-1", "p-1", _client_factory=lambda: client) is None


def test_find_resumable_run_id_none_on_lookup_error():
    """Strictly best-effort: a client error returns None and never propagates —
    a resume-lookup failure must never fail the unit (design § 5.1)."""
    from docker.entrypoint import _find_resumable_run_id
    client = _FakeResumeClient(raise_on_search=True)
    assert _find_resumable_run_id(
        "exp-1", "u-1", "p-1", _client_factory=lambda: client) is None


def test_find_resumable_run_id_default_client_path(monkeypatch):
    """The DEFAULT path (no _client_factory) must construct a real
    mlflow.tracking.MlflowClient — exercised here by monkeypatching that class.

    Guards the load-bearing trap (design § 5.1): entrypoint.py has NO
    module-level ``import mlflow``, so if the helper referenced
    mlflow.tracking.MlflowClient without a LOCAL ``import mlflow.tracking``, the
    default path would raise NameError -> be swallowed by the best-effort except
    -> silently return None, leaving resume-by-unit dead in production while the
    factory-injected tests above stayed green."""
    import mlflow.tracking
    from docker.entrypoint import _find_resumable_run_id
    client = _FakeResumeClient(runs=[_Run("resumed-via-default")])
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *a, **k: client)
    got = _find_resumable_run_id("exp-1", "u-9", "parent-9")
    assert got == "resumed-via-default"
    assert client.searches  # the real default construction path actually ran


def test_main_resumes_existing_run_for_uncommitted_unit(monkeypatch, tmp_path):
    """: when a prior (parent, unit) run exists (a reclaimed
    attempt), main RESUMES it via start_run(run_id=...) instead of minting a
    new child with run_name=... — retries collapse into ONE run (no zombie)."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    seam_calls = []

    def fake_seam(experiment_id, unit_id, parent_run_id):
        seam_calls.append((experiment_id, unit_id, parent_run_id))
        return "prior-run-9"

    monkeypatch.setattr(ep, "_find_resumable_run_id", fake_seam)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    # the seam was consulted with the resolved experiment id, unit id, injected parent
    assert seam_calls == [("exp-fake-1", unit.unit_id, "parent-1")]
    starts = [c for c in fake.calls if c[0] == "start_run"]
    assert starts == [("start_run", None, "prior-run-9")]  # run_id=, NOT run_name=
    assert fake.ended == ["FINISHED"]


def test_main_creates_run_when_none_exists(monkeypatch, tmp_path):
    """Seam returns None (no prior run) -> unchanged behaviour: a fresh child via
    start_run(run_name=unit_id)."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    monkeypatch.setattr(ep, "_find_resumable_run_id", lambda *a, **k: None)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    starts = [c for c in fake.calls if c[0] == "start_run"]
    assert starts == [("start_run", unit.unit_id, None)]  # run_name=, NOT run_id=
    assert fake.ended == ["FINISHED"]


# ---------------------------------------------------------------------------
# live_enrichment=complete marker gates the skip-complete path
# ---------------------------------------------------------------------------

def test_main_sets_live_enrichment_complete_on_full_success(monkeypatch, tmp_path):
    """On a fully-enriched success (start-metadata OK AND the full post-persist
    decoration block OK) main sets ``live_enrichment=complete`` — the marker the
    finalizer's skip-complete fast path requires before it may skip re-logging a
    run. unit_status=done alone is insufficient (it is guaranteed even on partial
    decoration), so the marker is the real proof of a fully-logged run."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert fake.tags.get("live_enrichment") == "complete"
    assert fake.tags["unit_status"] == "done"
    assert fake.ended == ["FINISHED"]


def test_main_omits_live_enrichment_marker_when_decoration_fails(monkeypatch, tmp_path):
    """If the post-persist decoration block did NOT fully complete
    (post_persist_enrichment returns False), unit_status=done is STILL guaranteed
     but ``live_enrichment=complete`` must be OMITTED — such a run is missing
    decoration, so the finalizer must RE-LOG it, never skip it."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    # simulate a partially-failed decoration block (returns False, never raises)
    monkeypatch.setattr(ep, "post_persist_enrichment", lambda *a, **k: False)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert "live_enrichment" not in fake.tags       # marker OMITTED
    assert fake.tags["unit_status"] == "done"       # done-guarantee unchanged
    assert fake.ended == ["FINISHED"]


def test_main_omits_live_enrichment_marker_when_metrics_fail(monkeypatch, tmp_path):
    """the metric-logging block must gate the marker too. If it
    fails partway AFTER final_f1 is logged, the marker + final_f1 predicate would
    falsely prove completeness and strand a run missing mean_accuracy/wall_clock_sec/
    rounds_completed/trajectory points. unit_status=done still set, NO marker."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)

    class _MetricBoom(_FullFakeMlflow):
        def log_metric(self, key, value, step=0, model_id=None):
            if key == "mean_accuracy":  # a later key — final_f1 has already been logged
                raise RuntimeError("transient log_metric failure")
            super().log_metric(key, value, step=step, model_id=model_id)

    fake = _MetricBoom()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert ("log_metric", "final_f1", 0.75, 0) in fake.calls  # final_f1 WAS logged first
    assert "live_enrichment" not in fake.tags                 # metric block failed -> no marker
    assert fake.tags["unit_status"] == "done"                 # done still guaranteed
    assert fake.ended == ["FINISHED"]


def test_main_omits_live_enrichment_marker_when_dataset_input_fails(monkeypatch, tmp_path):
    """The native dataset input is part of the repairable
    set (_enrich_completed_unit -> _log_dataset_input), so a failed mlflow.log_input for
    the training dataset must clear start_ok and OMIT the marker — even though
    log_dataset_input swallows the error internally. unit_status=done still set."""
    import docker.entrypoint as ep
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)

    class _DatasetBoom(_FullFakeMlflow):
        def log_input(self, dataset, context=None):
            if context == "training":  # the native dataset input from _set_start_metadata
                raise RuntimeError("dataset input log failed")
            super().log_input(dataset, context=context)

    fake = _DatasetBoom()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert "live_enrichment" not in fake.tags       # dataset input failed -> no marker
    assert fake.tags["unit_status"] == "done"       # done still guaranteed
    assert fake.ended == ["FINISHED"]


def test_main_omits_live_enrichment_marker_when_artifact_log_fails(monkeypatch, tmp_path):
    """the result-artifact upload (the ONE artifact the backfill
    _enrich_completed_unit repairs) must gate the marker. If it raises, unit_status=done
    is still set but the marker is OMITTED so the finalizer re-logs the run."""
    import docker.entrypoint as ep
    from praxis_exp.runner_paths import result_filename
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)

    class _ArtifactBoom(_FullFakeMlflow):
        def log_artifact(self, path):
            if path.endswith(result_filename(unit)):
                raise RuntimeError("result artifact upload failed")
            super().log_artifact(path)

    fake = _ArtifactBoom()
    assert ep.main(_store=store, _mlflow=fake) == 0
    assert "live_enrichment" not in fake.tags       # result-artifact upload failed -> no marker
    assert fake.tags["unit_status"] == "done"       # done still guaranteed
    assert fake.ended == ["FINISHED"]


def test_main_logs_result_artifact_for_backfill_parity(monkeypatch, tmp_path):
    """the live path now uploads result.json as an artifact —
    the ONE artifact the backfill (_enrich_completed_unit -> _log_artifacts) repairs —
    so a skip-complete run matches a backfilled run (live==backfill parity)."""
    import docker.entrypoint as ep
    from praxis_exp.runner_paths import result_filename
    store, unit = _setup_main(monkeypatch, tmp_path, rc=0)
    fake = _FullFakeMlflow()
    assert ep.main(_store=store, _mlflow=fake) == 0
    logged = [c[1] for c in fake.calls if c[0] == "log_artifact"]
    assert any(p.endswith(result_filename(unit)) for p in logged)
    assert fake.tags.get("live_enrichment") == "complete"

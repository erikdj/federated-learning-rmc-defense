"""Shared synthetic-fixture builders for tests/test_audit_calibration_gate.py.

Not a test module (no test_ prefix): holds the EXP-005c result-JSON /
signal-log factories so the test module stays within the repo's file-size
guidance. All schema decisions here cite the producing code -- see the
docstrings in scripts/_calibration_gate_lib.py for the full ground-truth map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from praxis_exp.units import unit_id  # noqa: E402

import _calibration_gate_lib as gl  # noqa: E402
from _calibration_gate_types import UnitRef  # noqa: E402

EXP_ID = "EXP-005c"
SCENARIO = "S4_full_mix"
SEED = 42
MODE = "persistent_optimizer"
ROUNDS = 3  # small fixture round count; trajectory covers 0..ROUNDS+1
# The audit-time ACTIVE methodology version (top entry of the real log) --
# fixtures use it so they stay hermetic across methodology bumps; tests that
# exercise the stale-version failure hardcode an old version deliberately.
ACTIVE_METHODOLOGY = gl.active_methodology_version(
    PROJECT_ROOT / "docs" / "METHODOLOGY_LOG.md"
)
LOCKED = json.loads((PROJECT_ROOT / "data" / "hparams_locked.json").read_text())
LOCKED_LR = LOCKED["persistent_optimizer"]["lr"]
LOCKED_EPOCHS = LOCKED["persistent_optimizer"]["local_epochs"]

# S4_full_mix scenario-derived defense sizing, verified against
# run_phase4_flower.py::_scenario_defense_sizing on this repo's scenario JSON
# (max declared malicious per round = 9; max declared participants = 20).
S4_DECLARED_ADVERSARIES = 9
S4_COHORT = 20
KRUM_F_POLICY = "dynamic ceil(n/2)-1"  # run_phase4_flower.py::KRUM_F_POLICY L686

CONFIGS = ["Krum", "TrustScore", "Krum+TGE", "TGE"]


def _tge_defaults():
    return gl.tge_runtime_defaults()


def _uid(config: str) -> str:
    return unit_id(config, SCENARIO, MODE, SEED)


def make_unit_ref(config: str) -> UnitRef:
    return UnitRef(
        unit_id=_uid(config), config=config, scenario=SCENARIO, seed=SEED,
        expected_optimizer_state="persistent",
        defense_token=gl.defense_token_for(config), rounds=ROUNDS,
    )


def make_result(
    config: str,
    *,
    seed: int = SEED,
    final_accuracy: float = 0.9791,
    elapsed_seconds: float = 111.0,
    optimizer_state: str = "persistent",
    return_code: int = 0,
    provenance_overrides: dict | None = None,
    include_hparams: bool = True,
    hparams_lr: float | None = None,
    rounds: int = ROUNDS,
    trajectory_rounds: list[int] | None = None,
) -> dict:
    is_tge = "TGE" in config
    has_krum = "Krum" in config
    prov = {
        "runner_version": "unified-v1.0",
        "scenario_path": f"rmc/scenarios/{SCENARIO}.json",
        "optimizer_state": optimizer_state,
        "flwr_version": "1.29.0",
        # defense-sizing provenance (v1.19 / PR #12 round-2 P2):
        # run_phase4_flower.py::_defense_provenance_fields L689-710
        "krum_f_policy": KRUM_F_POLICY if has_krum else "n/a",
        "scenario_declared_adversaries": S4_DECLARED_ADVERSARIES,
        "defense_cohort_size": S4_COHORT,
    }
    if is_tge:
        d = _tge_defaults()
        prov.update({
            "tge_lstm_state": "enabled",
            "tge_pure_lstm_reach": True,
            "tge_ramp_rounds": d["ramp_rounds"],
            "tge_cold_start_expert": "isolation_forest",
            "tge_min_tenure": d["min_tenure"],
            "tge_operational_threshold": d["threshold"],
        })
    else:
        prov.update({"tge_lstm_state": "n/a", "tge_ramp_rounds": None})
    if provenance_overrides:
        prov.update(provenance_overrides)

    # Trajectory covers server rounds 0..rounds+1: num-server-rounds = rounds+1
    # (discovery round, run_phase4_flower.py L437) and Flower's Server.fit
    # evaluates round 0 (initial) plus every round 1..num_rounds.
    traj_rounds = trajectory_rounds if trajectory_rounds is not None else list(range(0, rounds + 2))
    result = {
        "config": config,
        "strategy": f"Scenario{config.replace('+', '')}",
        "seed": seed,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "trajectory": [
            {"round": r, "accuracy": final_accuracy, "f1": 0.95, "loss": 0.1}
            for r in traj_rounds
        ],
        "convergence": {"rounds_to_converge": 3},
        "defense_overhead": {"mean_seconds": 0.02},
        "final_accuracy": final_accuracy,
        "mean_accuracy": final_accuracy,
        "provenance": prov,
    }
    if include_hparams:
        result["hparams"] = {"lr": hparams_lr if hparams_lr is not None else LOCKED_LR,
                              "local_epochs": LOCKED_EPOCHS}
    return result


def make_signal_row(
    config: str,
    server_round: int,
    logical_cid: str,
    *,
    run_started_at: str = "2026-07-10T00:00:00Z",
    schema_version: int = 3,
    tge_score: float | None = None,
    malicious: bool = False,
    krum_score: float | None = None,
) -> dict:
    row = {
        "seed": SEED, "scenario": SCENARIO, "exec_mode": "flower_persistent",
        "dataset": "edge_full_20_rmc", "defense": gl.defense_token_for(config),
        "git_commit": "deadbeef", "run_started_at": run_started_at,
        "signal_log_schema_version": schema_version,
        "server_round": server_round, "scenario_round": server_round - 1,
        "logical_cid": logical_cid, "flower_cid": f"cid-{logical_cid}",
        "physical_partition_id": 0, "malicious_gt": malicious, "attack_type": "",
        "num_examples": 100, "train_loss": 0.1, "update_norm": 1.0,
        "cos_to_median": 0.9, "L2_to_median": 0.2,
        "krum_score": krum_score if "Krum" in config else None,
        "trust_score": 0.5 if config == "TrustScore" else None,
        "effective_weight": 100.0,
    }
    if "TGE" in config:
        d = _tge_defaults()
        if tge_score is not None:
            row.update({
                "tge_score": tge_score, "tge_gbdt_score": 0.3, "tge_lstm_score": 0.4,
                "tge_tenure": 3, "tge_phase": "active", "tge_gate": "blend",
                "tge_threshold": d["threshold"], "tge_decision": tge_score >= d["threshold"],
            })
        else:
            row.update({
                "tge_score": None, "tge_gbdt_score": None, "tge_lstm_score": None,
                "tge_tenure": None, "tge_phase": None, "tge_gate": None,
                # tge_threshold is a ROUND-LEVEL constant even for unscored rows
                # (flowerfl/scenario_strategy.py::_maybe_log_signals L616-620/641).
                "tge_threshold": d["threshold"], "tge_decision": None,
            })
    return row


def make_signal_rows_for_round(
    config: str, server_round: int, *, n_participants: int = 20,
    n_scored: int | None = None, run_started_at: str = "2026-07-10T00:00:00Z",
    schema_version: int = 3, flat_krum_scores: bool = False,
) -> list[dict]:
    """n_scored only matters for Krum+TGE/TGE configs: the first n_scored
    logical_cids get a non-null tge_score, the rest are unscored (simulating
    upstream Krum filtering in the composed chain). For Krum+TGE the default
    is the DYNAMIC Multi-Krum survivor count for this round's cohort
    (f=ceil(n/2)-1, keep=max(1,n-f-2) -- methodology v1.19); for TGE-only,
    every participant is scored. krum_score varies per client unless
    flat_krum_scores (the pre-v1.19 certification-bound flattening
    signature: uniform 1.0)."""
    if n_scored is None:
        if config == "Krum+TGE":
            n_scored = gl.krum_dynamic_keep(n_participants)
        else:
            n_scored = n_participants if "TGE" in config else 0
    rows = []
    for i in range(n_participants):
        cid = f"client_{i}"
        scored = "TGE" in config and i < n_scored
        # Mark a couple of clients malicious every round so
        # audit_run_instrumentation.audit()'s "malicious_gt never True" /
        # "scored malicious=0" gaps don't fire on an otherwise-valid fixture.
        malicious = i < 2
        krum_score = None
        if "Krum" in config:
            krum_score = 1.0 if flat_krum_scores else round(0.3 + 0.02 * i, 4)
        rows.append(make_signal_row(
            config, server_round, cid, run_started_at=run_started_at,
            schema_version=schema_version, tge_score=(0.8 if scored else None),
            malicious=malicious, krum_score=krum_score,
        ))
    return rows


def write_signal_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


def signal_rounds() -> range:
    """Server rounds that carry signal rows: {2..ROUNDS+1}. Ground truth:
    ScenarioStrategy._round_offset = 1 (flowerfl/scenario_strategy.py:135 --
    server round 1 is cid discovery; startup print says "Flower rounds
    2-{num_rounds+1}") and _maybe_log_signals returns early unless
    _schedule_cache[server_round - 1] is non-empty (scenario_strategy.py:
    533-536); generated scenarios declare every scenario round 1..num_rounds,
    so signal rows exist exactly for server rounds 2..rounds+1."""
    return range(2, ROUNDS + 2)


def build_passing_local_fixture(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results"
    signals_dir = tmp_path / "signals"
    for config in CONFIGS:
        uid = _uid(config)
        result = make_result(config)
        write_result(results_dir / f"{uid}.json", result)
        rows = []
        for rnd in signal_rounds():
            rows.extend(make_signal_rows_for_round(config, rnd))
        write_signal_log(signals_dir / f"{uid}.jsonl", rows)
    return results_dir, signals_dir


MAX_PER_CLIENT = 2_000_000  # EXP-005c design matrix (docs/experiments/EXP-005c-calibration.md)


def expected_units():
    """The fixture's expected unit set (EXP-005c defenses x S4 x seed 42, at
    the fixture's small ROUNDS) -- the same praxis_exp.units.expand_matrix
    expansion the manifest / design-doc resolution paths produce."""
    from praxis_exp.units import expand_matrix

    return expand_matrix(CONFIGS, [SCENARIO], [SEED], MODE, MAX_PER_CLIENT, ROUNDS)


def write_local_manifest(tmp_path: Path, *, meta: dict | None = None,
                         exp_id: str = EXP_ID) -> Path:
    """A downloaded-manifest.json fixture with the same payload shape
    praxis_exp.manifest.write_manifest produces (exp_id / meta / units)."""
    from dataclasses import asdict

    payload = {
        "exp_id": exp_id,
        "meta": meta if meta is not None else {
            "methodology_version": ACTIVE_METHODOLOGY, "image_digest": "sha256:abc",
            "git_sha": "deadbeef", "n_units": len(CONFIGS),
        },
        "units": [asdict(u) for u in expected_units()],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path

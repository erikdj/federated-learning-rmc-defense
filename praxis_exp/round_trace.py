"""``praxis exp timeline EXP-NNN`` — reconstruct the per-round timeline TABLE from S3.

A per-round attack/defense timeline for a completed sweep, built post-hoc from the
durable signal log + result trajectory (never from the Flower/Ray hot path) and logged
as ``round_timeline.json`` via ``mlflow.log_table`` (renders in the 3.14 artifact
browser as a table). One row per server round carrying the attack type,
participant/malicious counts, the defense's trust-score extremes, and — for hard top-k
defenses — how many malicious clients landed inside the kept set (``kept_malicious``),
the "filter bypassed" signal. (Per-round MLflow *traces* were retired in GWU-47: an FL
training run has no call tree, so this timeline is tabular data — a table, not a trace.
The ``search_traces``/``delete_traces`` passthroughs survive only for one-time cleanup.)

``mlflow.log_table`` APPENDS, so ``emit_round_table`` refreshes (list -> delete -> log)
before logging — an idempotent re-enrich never duplicates rows.

Selection semantics (verified 2026-07-13, see reference-signal-log-semantics):
  - ``krum_score`` / ``trust_score`` are TRUST scores — higher = kept; the kept set
    is the top ``keep = max(1, n - num_malicious - 2)`` by score. (NOT the lowest;
    and ``effective_weight`` is a sample-count, NOT a keep flag.)
  - Krum / Krum+TGE / TrustScore are hard top-k → ``kept_malicious`` is meaningful.
  - TGE is a soft tenure-gated ensemble (no hard top-k) → ``kept_malicious`` N/A.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from praxis_exp import storage
from praxis_exp.integrity import is_done
from praxis_exp.manifest import read_manifest
from praxis_exp.matrix_doc import parse_matrix
from praxis_exp.mlflow_client import PraxisMlflowClient
from praxis_exp.storage import ObjectNotFoundError, ObjectStore
from praxis_exp.units import Unit

# defense_token -> (signal score field, selection mode)
_HARD, _SOFT, _NONE = "hard_topk", "soft_ensemble", "none"
_DEFENSE_SCORE: dict[str, tuple[Optional[str], str]] = {
    "krum": ("krum_score", _HARD),
    "krumtge": ("krum_score", _HARD),      # Krum arm drives selection; TGE gates after
    "krumtgeprime": ("krum_score", _HARD),  # same chain shape with the TGE′ bank (GWU-53)
    "trustscore": ("trust_score", _HARD),
    "tgensemble": ("tge_score", _SOFT),
    "tgeprime": ("tge_score", _SOFT),      # FedAvg + TGE′ bank (GWU-53)
    "tge": ("tge_score", _SOFT),
    "fedavg": (None, _NONE),
    # H3 fingerprint arms (image-checklist item 6 sweep, 2026-08-17): FP adds
    # no selection score of its own — the aggregating layer's lens applies.
    "tgefp": ("tge_score", _SOFT),         # ScenarioTGEFP lowered
    "krumtgefp": ("krum_score", _HARD),    # Krum-first legacy chain
    # H4 composition arms (spec 2026-08-16 § 2 + erratum-A). The detector's
    # score lives in h4_diagnostics, not signal rows; the round-trace lens is
    # the downstream aggregator's. krum/trust scores are truthfully null for
    # detector-dropped clients (cid-keyed join) — the lens tolerates nulls.
    "h2pfpkrum": ("krum_score", _HARD),    # ScenarioH2PFPKrum lowered
    "h2pkrum": ("krum_score", _HARD),      # ScenarioH2PKrum lowered
    "h2pfpts": ("trust_score", _HARD),     # ScenarioH2PFPTS lowered
    "h2pts": ("trust_score", _HARD),       # ScenarioH2PTS lowered (arm 9)
    "h2pfp": (None, _NONE),                # detector+FP over FedAvg — no lens
    "none": (None, _NONE),                 # ScenarioNone lowered — H4 floor arm
}
_F1_ERROR = 0.75  # spans below this are flagged ERROR (attack-degraded)


def score_config(defense_token: str) -> tuple[Optional[str], str]:
    """(score_field, selection_mode) for a defense; unknown -> no selection lens."""
    return _DEFENSE_SCORE.get((defense_token or "").lower(), (None, _NONE))


def _round_attack(rows: list[dict]) -> str:
    """Dominant ground-truth attack among the round's malicious clients."""
    c = Counter(r.get("attack_type") for r in rows if r.get("malicious_gt") and r.get("attack_type"))
    return c.most_common(1)[0][0] if c else "none"


def build_round_spans(
    signal_rows: list[dict], trajectory: list[dict], *, defense_token: str,
) -> list[dict]:
    """Per-round span descriptors for one run. PURE (no MLflow/IO).

    ``signal_rows`` = parsed signal jsonl (one per (server_round, client));
    ``trajectory`` = result JSON's per-round [{round, f1, accuracy, loss}, ...].
    """
    score_key, mode = score_config(defense_token)
    by_round: dict[int, list[dict]] = defaultdict(list)
    for r in signal_rows:
        sr = r.get("server_round")
        if sr is not None:
            by_round[sr].append(r)
    traj = {t["round"]: t for t in trajectory if "round" in t}

    spans: list[dict] = []
    for sr in sorted(traj):
        t = traj[sr]
        g = by_round.get(sr, [])
        n = len(g)
        mal = [x for x in g if x.get("malicious_gt")]
        hon = [x for x in g if not x.get("malicious_gt")]
        n_mal = len(mal)
        attack = _round_attack(g) if g else "none"
        ms = [x.get(score_key) for x in mal if score_key and x.get(score_key) is not None]
        hs = [x.get(score_key) for x in hon if score_key and x.get(score_key) is not None]
        span: dict[str, Any] = {
            "server_round": sr,
            "scenario_round": sr - 1,
            "attack_type": attack,
            "participants": n,
            "malicious_present": n_mal,
            "selection_mode": mode,
            "f1": round(t["f1"], 4),
            "accuracy": round(t["accuracy"], 4),
            "loss": round(t["loss"], 4),
            "is_error": t["f1"] < _F1_ERROR,
        }
        if ms:
            span["malicious_score_max"] = round(max(ms), 4)
        if hs:
            span["honest_score_max"] = round(max(hs), 4)
        bypass = False
        if mode == _HARD and score_key:
            scored = [x for x in g if x.get(score_key) is not None]
            kept_mal = None
            if scored:
                keep = max(1, n - n_mal - 2)
                top = sorted(scored, key=lambda x: -(x.get(score_key) or 0))[:keep]
                kept_mal = sum(1 for x in top if x.get("malicious_gt"))
                span["kept_malicious"] = kept_mal
                span["keep_count"] = keep
                bypass = n_mal > 0 and kept_mal > keep / 2
                span["filter_bypassed"] = bypass
        elif mode == _SOFT:
            span["note"] = "soft tenure-gated ensemble: no hard top-k; kept_malicious N/A"
        span["name"] = f"r{sr:02d}_{attack}{'_BYPASS' if bypass else ''}"
        spans.append(span)
    return spans


# Column order for the timeline table (drops the trace-only derived ``name``;
# precision/recall are final-only, not per-round). A span missing a key
# contributes ``None`` in that column's list.
_TIMELINE_COLUMNS = [
    "server_round", "scenario_round", "attack_type", "participants",
    "malicious_present", "selection_mode", "f1", "accuracy", "loss",
    "malicious_score_max", "honest_score_max", "keep_count", "kept_malicious",
    "filter_bypassed", "note", "is_error",
]


def build_round_table(spans: list[dict]) -> Optional[dict]:
    """Column-oriented dict ``{col: [per-round values]}`` for ``mlflow.log_table``.
    PURE (no MLflow/IO, pandas-free). This IS the shape ``log_table`` feeds to
    ``pd.DataFrame(data)`` — NOT the split-orient ``{"columns":..,"data":..}``
    on-disk form (which would yield a 2-column table named columns/data). A span
    missing a key contributes ``None`` for that round. ``None`` for empty spans."""
    if not spans:
        return None
    return {col: [sp.get(col) for sp in spans] for col in _TIMELINE_COLUMNS}


def emit_round_table(client: Any, run_id: str, table: Optional[dict]) -> None:
    """Log the timeline ``table`` to ``run_id`` as ``round_timeline.json``.
    ``mlflow.log_table`` APPENDS, so REFRESH first (list -> delete -> log) — else a
    re-run duplicates every row. Skips when ``table is None`` (empty spans)."""
    if table is None:
        return
    existing = {getattr(a, "path", a) for a in client.list_artifacts(run_id)}
    if "round_timeline.json" in existing:
        client.delete_artifact(run_id, "round_timeline.json")
    client.log_table(run_id, table, artifact_file="round_timeline.json")


class RetraceError(RuntimeError):
    """Raised when a trace-rebuild precondition (missing manifest/doc) fails."""


def _find_design_doc(repo_root: Path, exp_id: str) -> Path:
    exp_dir = repo_root / "docs" / "experiments"
    for f in exp_dir.iterdir():
        if f.is_file() and f.name.startswith(f"{exp_id}-") and not f.name.endswith("-result.md"):
            return f
    raise RetraceError(f"no design doc for {exp_id} in {exp_dir}")


def _experiment_name(repo_root: Path, exp_id: str) -> str:
    return parse_matrix(_find_design_doc(repo_root, exp_id)).slug


def _read_json(store: ObjectStore, key: str) -> Optional[Any]:
    try:
        return json.loads(store.get_bytes(key))
    except ObjectNotFoundError:
        return None
    except Exception as e:  # pragma: no cover - corrupt artifact
        print(f"[trace] WARN: {key} not valid JSON ({e}); skipping")
        return None


def _read_signal_rows(store: ObjectStore, exp_id: str, unit_id: str) -> list[dict]:
    try:
        raw = store.get_bytes(storage.signal_key(exp_id, unit_id))
    except ObjectNotFoundError:
        return []
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _delete_prior_traces(client: Any, experiment_id: str, exp_id: str, unit_id: str) -> int:
    """Delete any prior fl_training trace tagged with this (exp_id, unit_id) so a
    re-trace is idempotent. Matching on BOTH tags is required because a design-family
    experiment (e.g. ``calibration``) holds several launches whose unit_ids repeat —
    unit_id alone would delete another launch's trace. Best-effort: trace search/delete
    varies across MLflow builds."""
    try:
        # NOTE: SearchTracesV3 caps max_results (1000 is rejected as
        # INVALID_PARAMETER_VALUE) — keep <= 500. A design-family experiment
        # accumulates a few traces per launch, so 500 covers the realistic depth.
        traces = client.search_traces(experiment_ids=[experiment_id], max_results=500)
    except Exception as e:
        # Do NOT swallow silently — a failed cleanup search leaves duplicate
        # traces on every re-run (this exact bug: max_results=1000 threw and was
        # eaten, so idempotency never fired).
        print(f"[trace] WARN: idempotency trace-search failed ({e}); may leave a duplicate trace")
        return 0
    stale = []
    for t in traces:
        info = getattr(t, "info", t)
        tags = getattr(info, "tags", {}) or {}
        name = tags.get("mlflow.traceName", "")
        if (name.startswith("fl_training")
                and tags.get("praxis.exp_id") == exp_id
                and tags.get("praxis.unit_id") == unit_id):
            stale.append(info.trace_id)
    if stale:
        try:
            client.delete_traces(experiment_id=experiment_id, trace_ids=stale)
        except Exception as e:
            print(f"[trace] WARN: could not delete prior traces for {unit_id}: {e}")
            return 0
    return len(stale)


def tabulate_experiment(
    repo_root: Path, exp_id: str, *, _store: ObjectStore,
    _client: Optional[Any] = None,
) -> dict[str, Any]:
    """(Re)build the per-round timeline TABLE (``round_timeline.json`` via
    ``log_table``) for every completed unit of a sweep from S3.

    Idempotent: ``emit_round_table`` refreshes (list -> delete -> log) before
    logging, so re-running does NOT duplicate rows (``log_table`` appends).
    Cleanup of any stale ``fl_training__*`` traces is a separate one-time pass
    (``praxis exp cleanup-traces``), not per-run here."""
    repo_root = Path(repo_root)
    _, _meta, units = read_manifest(_store, exp_id)
    client = _client or PraxisMlflowClient()
    experiment_id = client.get_or_create_experiment(_experiment_name(repo_root, exp_id))
    parent_run_id = client.find_parent_run(experiment_id, exp_id)

    tabulated = skipped = failed = 0
    for unit in units:
        try:
            if not is_done(_store, exp_id, unit.unit_id):
                skipped += 1
                continue
            result = _read_json(_store, storage.result_key(exp_id, unit.unit_id))
            if not result or not result.get("trajectory"):
                skipped += 1
                continue
            runs = client.find_runs_by_unit(
                experiment_id, unit.unit_id, parent_run_id=parent_run_id,
            )
            if not runs:
                skipped += 1
                continue
            run_id = runs[-1]
            signal_rows = _read_signal_rows(_store, exp_id, unit.unit_id)
            dtoken = _defense_token(unit)
            spans = build_round_spans(
                signal_rows, result["trajectory"], defense_token=dtoken,
            )
            emit_round_table(client, run_id, build_round_table(spans))
            tabulated += 1
        except Exception as e:
            failed += 1
            print(f"[timeline] WARN: unit {unit.unit_id} table failed ({e}); continuing")
    return {
        "experiment_id": experiment_id, "units_tabulated": tabulated,
        "units_skipped": skipped, "units_failed": failed,
    }


_COARSE_TRACE_NAMES = ("run_phase4_flower", "persist_unit")


def cleanup_traces(
    repo_root: Path, exp_id: str, *, _client: Optional[Any] = None,
) -> dict[str, Any]:
    """One-time removal of RETIRED traces for a sweep (GWU-47). Deletes BOTH:
    this sweep's per-round ``fl_training__*`` traces (tagged ``praxis.exp_id``,
    so scoped to the sweep) AND the coarse ``run_phase4_flower``/``persist_unit``
    spans the old ``traced_span`` wrapper emitted (untagged, no run linkage — all
    off-label, so removed experiment-wide). Idempotent: no matching traces -> 0
    deleted, never raises into the caller."""
    repo_root = Path(repo_root)
    client = _client or PraxisMlflowClient()
    experiment_id = client.get_or_create_experiment(_experiment_name(repo_root, exp_id))
    try:
        traces = client.search_traces(experiment_ids=[experiment_id], max_results=500)
    except Exception as e:
        print(f"[cleanup-traces] WARN: trace search failed ({e}); nothing deleted")
        return {"experiment_id": experiment_id, "traces_deleted": 0}
    stale = []
    for t in traces:
        info = getattr(t, "info", t)
        tags = getattr(info, "tags", {}) or {}
        name = tags.get("mlflow.traceName", "")
        is_round = name.startswith("fl_training") and tags.get("praxis.exp_id") == exp_id
        is_coarse = name in _COARSE_TRACE_NAMES
        if is_round or is_coarse:
            stale.append(info.trace_id)
    if stale:
        try:
            client.delete_traces(experiment_id=experiment_id, trace_ids=stale)
        except Exception as e:
            print(f"[cleanup-traces] WARN: could not delete traces ({e})")
            return {"experiment_id": experiment_id, "traces_deleted": 0}
    return {"experiment_id": experiment_id, "traces_deleted": len(stale)}


def _defense_token(unit: Unit) -> str:
    """The signal/tag defense_token for a unit's config (reuses the entrypoint's map)."""
    from praxis_exp.enrich import defense_token  # loaded-by-file-path entrypoint helper
    return defense_token(unit.config)

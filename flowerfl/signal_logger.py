"""
Phase 1 P1.2 — Server-side signal logger for Track B cold-start detector training.

Emits one JSONL row per (client, round) during FL runs, capturing the
server-observable signals that a cold-start detector could use:

    seed                          : experiment RNG seed
    scenario                      : scenario name
    exec_mode                     : "flower_reset" | "persistent_optimizer"
    dataset                       : dataset name (e.g., edge_full_20_rmc)
    defense                       : defense under evaluation (e.g., fedavg, krum)
    git_commit                    : commit hash at run time
    run_started_at                : ISO timestamp of run start
    server_round                  : Flower round number (1-indexed)
    scenario_round                : scenario-relative round (after discovery offset)
    logical_cid                   : scenario logical id (e.g., client_19, client_19_new)
    flower_cid                    : Flower's random 64-bit cid
    physical_partition_id         : partition id in the dataset (0..20)
    malicious_gt                  : True if client is under attack this round
    attack_type                   : attack kind if malicious (else empty)
    num_examples                  : num training samples reported by client
    train_loss                    : client-reported training loss
    update_norm                   : L2 norm of the client update vector
    cos_to_median                 : cosine similarity between client update and median
    L2_to_median                  : L2 distance between client update and median
    krum_score                    : Krum trust score in [0,1] (None if no Krum plugin)
    trust_score                   : TrustScore EMA in [0,1] (None if no TrustScore plugin)
    effective_weight              : RAW, PRE-FILTER client-reported num_examples.
                                    NOT an aggregation coefficient — see below.

Schema v5 (2026-08-05) additions
--------------------------------
    aggregation_coefficient       : the POST-`filter_updates`, round-normalized
                                    FedAvg coefficient a_i = w_i / Σ_j w_j over
                                    the round's survivors; exactly 0.0 for a
                                    hard-dropped client; None when the base
                                    strategy is not num-examples-weighted FedAvg
                                    (never a fabricated number).
    client_timing_observation     : reserved, structurally null. The v1.10-
                                    mandated "(D8 optional-arm-ready) per-client
                                    timing field"; the synthetic per-device
                                    timing model is a separately pre-registered
                                    OPTIONAL arm, not in H3's launch scope.
    <re-entry event contract>     : the ten row-level fields of amendment v1.10
                                    § 5.1 (see REENTRY_EVENT_FIELDS). Non-null
                                    only on a re-entry event row; `server_round`
                                    is the contract's eleventh field and is
                                    already a row-level field.
    run_uid                       : stable identity of THIS run, used as the
                                    `run_id` component of `reentry_event_key`.

`effective_weight` vs `aggregation_coefficient` (do not conflate).
`effective_weight` is the client's self-reported row count *before* any defense
filtering — non-zero for every dispatched client, survivor or not. It has never
been a keep-flag and its meaning is UNCHANGED at v5 (v4 readers keep working and
no v4 row is reinterpreted). The locked `data/h3_constants.json` rejoin-success
rule is defined on the aggregation coefficient, whose `blocked` outcome class is
`coefficient == 0` — inexpressible in `effective_weight`. Per amendment v1.10
§ 5.1: *no H3 result may be computed from `effective_weight=num_examples`
masquerading as a coefficient.*

The file is append-only JSONL; a file per run.

Usage from ScenarioStrategy (Flower):
    logger = SignalLogger(
        path="signals/flower_reset__rmc_szelag_20__krum__seed42.jsonl",
        run_metadata={
            "seed": 42,
            "scenario": "rmc_szelag_20",
            "exec_mode": "flower_reset",
            "dataset": "edge_full_20_rmc",
            "defense": "krum",
        },
    )
    logger.log_round(server_round=5, scenario_round=4, per_client_records=[...])

Usage from single-process sim (persistent_optimizer mode):
    Same API; pass exec_mode="persistent_optimizer".
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Schema v5 contract (frozen by amendment v1.10 § 5.1 — do not reorder/rename)
# --------------------------------------------------------------------------

SIGNAL_LOG_SCHEMA_VERSION = 5

#: The re-entry event contract, verbatim from v1.10 § 5.1's field table. The
#: table's eleventh field, `server_round`, is already a row-level field emitted
#: by :meth:`SignalLogger.log_round`, so it is not repeated here.
REENTRY_EVENT_FIELDS = (
    "reentry_event_key",            # str  — "{run_uid}:{server_round}:{current_cid}"
    "current_cid",                  # str  — the new FLOWER CID the device reappeared
                                    #        under (§ 5.1). The OBSERVABLE, never the
                                    #        logical identity: it is what the re-link
                                    #        metric must recover `gt_logical_id` FROM.
    "gt_logical_id",                # str  — ground-truth device identity (truth key)
    "gt_is_malicious",              # bool — sets the positive vs negative population
    "asserted_match",               # bool — did the registry assert a link (min_d <= tau)?
    "asserted_parent_entry_id",     # str|None — the registry entry the match asserted
    "asserted_parent_logical_id",   # str|None — compared to gt_logical_id to score it
    "min_d",                        # float — Mahalanobis distance to nearest flagged entry
    "tau",                          # float — the locked tau in force
    "generation",                   # int   — inherited generation (0 if new entry)
)

#: ADDITIVE extension to the re-entry contract — deliberately NOT folded into
#: `REENTRY_EVENT_FIELDS`, which is v1.10 § 5.1's frozen table and stays
#: byte-identical in name, order and meaning.
#:
#: The nearest candidate the registry considered, recorded on every re-entry
#: decision INCLUDING one that asserted no match. The corrected instrument's
#: primary metric is rank-1 identification, which is threshold-free; with only
#: the `asserted_*` pair, an event whose nearest candidate sat beyond τ threw
#: that candidate's identity away and left rank-1 unanswerable. Nothing here
#: feeds the D7 statistic: `analyze_h3_relink` reads by name from its own frozen
#: REQUIRED_FIELDS and never sees these.
REENTRY_NEAREST_FIELDS = (
    "nearest_entry_id",             # str|None — nearest candidate's registry entry
    "nearest_logical_id",           # str|None — nearest candidate's identity;
                                    #        None ONLY when the pool was empty
)

#: Every field v5 adds to a row. Emitted on EVERY row with a null default, so a
#: *missing* key is a defect while an explicit null is a truthful "not
#: applicable" — the distinction integrity gate (b) ("zero unknown-provenance
#: rows") is scored on.
#:
#: The schema VERSION is unchanged. `REENTRY_NEAREST_FIELDS` is purely additive:
#: no existing field is reordered, renamed or given a new meaning, and every
#: consumer reads by name. A version bump would instead invalidate every
#: existing v5 log and corpus for a diagnostic addition — the extractor treats
#: the pair as optional so pre-extension logs stay readable.
SCHEMA_V5_ADDED_FIELDS = (
    "aggregation_coefficient",
    "client_timing_observation",
) + REENTRY_EVENT_FIELDS + REENTRY_NEAREST_FIELDS

#: RMC cycle identity: `client_<N>_new<M>` — the SAME physical device returning
#: under a new identity (`ScenarioStrategy.LOGICAL_TO_PARTITION` maps it back to
#: partition N). The legacy aliases `client_9_new` / `client_19_new` carry no
#: cycle number and map to DIFFERENT partitions (10 / 20) — a different device,
#: deliberately excluded.
_CYCLE_IDENTITY_RE = re.compile(r"^(client_\d+)_new\d+$")


def is_reentry_identity(logical_cid: str) -> bool:
    """True iff `logical_cid` is an RMC cycle identity (a returning device)."""
    return bool(_CYCLE_IDENTITY_RE.match(str(logical_cid)))


def canonical_device_id(logical_cid: str) -> str:
    """The ground-truth device identity behind a logical id.

    `client_5_new2` -> `client_5`; anything else is returned unchanged. This is
    the single canonicalization both sides of the re-link comparison must use:
    a link is CORRECT iff `asserted_parent_logical_id == gt_logical_id`
    (v1.10 § 5.1 scorer contract), and the adjudicating cohort's odd/even device
    hold-out parity is derived from this id.
    """
    m = _CYCLE_IDENTITY_RE.match(str(logical_cid))
    return m.group(1) if m else str(logical_cid)


def build_reentry_event_key(run_uid: str, server_round: int, current_cid: str) -> str:
    """The frozen dedup unit: `{run_id}:{round}:{cid}` (v1.10 § 5.1)."""
    return f"{run_uid}:{int(server_round)}:{current_cid}"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


class SignalLogger:
    """Append-only JSONL logger for per-client per-round server-side signals."""

    def __init__(self, path: str, run_metadata: Dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.run_metadata = {
            "git_commit": _git_commit(),
            "run_started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **run_metadata,
            # v3 (2026-06-06): adds defense-agnostic `tenure`; `malicious_gt` is now per-identity (F6/F7).
            # v4 (2026-07-25): adds `tge_ema_score` — the TGE′ bank's EMA-reputation
            #   leg (null when the row's client was unscored; GWU-53). Bumping the
            #   version keeps pre-TGE′ v3 logs (which cannot carry the field)
            #   distinguishable from TGE′ logs, so a bank log missing its EMA leg
            #   can't be mistaken for legitimately null.
            # v5 (2026-08-05): adds `aggregation_coefficient` (the post-filter,
            #   round-normalized FedAvg coefficient the locked h3_constants.json
            #   rejoin rule is defined on — `effective_weight` is the raw
            #   pre-filter count and is UNCHANGED), the v1.10 § 5.1 re-entry
            #   event contract, the D8-ready timing slot, and `run_uid`. Readers
            #   are version-GATED, never migrated: v4 logs keep parsing and no
            #   v4 row is reinterpreted.
            "signal_log_schema_version": SIGNAL_LOG_SCHEMA_VERSION,  # must come last to be non-overridable
        }
        required = {"seed", "scenario", "exec_mode", "dataset", "defense"}
        missing = required - set(self.run_metadata.keys())
        if missing:
            raise ValueError(f"SignalLogger run_metadata missing keys: {missing}")

        # v5: a stable identity for THIS run, needed as the `run_id` component
        # of `reentry_event_key`. The file is one-run-per-file by construction
        # (the calibration gate asserts a single `run_started_at` per log), so
        # the naming tuple plus the start stamp identifies it uniquely. Computed
        # AFTER the required-key check so a malformed call fails on that first.
        self.run_metadata["run_uid"] = (
            f"{self.run_metadata['exec_mode']}__{self.run_metadata['scenario']}"
            f"__{self.run_metadata['defense']}__seed{self.run_metadata['seed']}"
            f"__{self.run_metadata['run_started_at']}"
        )

        self._file = open(self.path, "a", buffering=1)  # line-buffered

    @property
    def run_uid(self) -> str:
        """Stable identity of this run (the `run_id` in `reentry_event_key`)."""
        return self.run_metadata["run_uid"]

    def log_round(
        self,
        server_round: int,
        scenario_round: int,
        per_client_records: List[Dict[str, Any]],
    ) -> None:
        """Emit one JSONL row per entry in `per_client_records`.

        Each record must contain at minimum:
            logical_cid, flower_cid, physical_partition_id, malicious_gt,
            attack_type, num_examples, train_loss, update_norm,
            cos_to_median, L2_to_median, krum_score, trust_score,
            effective_weight

        Every schema-v5 field (``SCHEMA_V5_ADDED_FIELDS``) is emitted on every
        row, defaulting to null when the caller supplies nothing — so a MISSING
        key is unambiguously a defect and an explicit null is a truthful "not
        applicable" (integrity gate (b): zero unknown-provenance rows).
        """
        v5_defaults = {field: None for field in SCHEMA_V5_ADDED_FIELDS}
        for rec in per_client_records:
            row = {
                **self.run_metadata,
                "server_round": int(server_round),
                "scenario_round": int(scenario_round),
                **v5_defaults,
                **rec,
            }
            # JSON-serializable cleanup
            row = _jsonify(row)
            self._file.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SignalLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _jsonify(obj: Any) -> Any:
    """Convert numpy types and None-like to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [_jsonify(x) for x in obj.tolist()]
    if obj is None:
        return None
    return obj


# ============================================================================
# Server-side feature extractors — library of reusable per-round computations
# ============================================================================


def flatten_parameters(parameters) -> np.ndarray:
    """Flatten Flower Parameters (or list of ndarrays) to a 1-D vector."""
    from flwr.common import parameters_to_ndarrays, Parameters
    if isinstance(parameters, Parameters):
        arrays = parameters_to_ndarrays(parameters)
    elif isinstance(parameters, (list, tuple)):
        arrays = parameters
    else:
        raise TypeError(f"Unsupported parameters type: {type(parameters)}")
    return np.concatenate([np.asarray(a).ravel() for a in arrays])


def compute_per_client_signals(
    flat_updates: List[np.ndarray],
    train_losses: List[float],
    num_examples_list: List[int],
) -> List[Dict[str, float]]:
    """Compute Family S signals for one round of client updates.

    Returns a list of dicts aligned with `flat_updates` index:
        update_norm, cos_to_median, L2_to_median
    Median is computed element-wise across all flat_updates.
    """
    n = len(flat_updates)
    if n == 0:
        return []

    stacked = np.stack(flat_updates, axis=0)  # (n, d)
    median = np.median(stacked, axis=0)  # (d,)
    median_norm = float(np.linalg.norm(median)) or 1e-12

    out = []
    for i, u in enumerate(flat_updates):
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-12:
            cos = 0.0
        else:
            cos = float(np.dot(u, median) / (u_norm * median_norm))
        l2 = float(np.linalg.norm(u - median))
        out.append(
            {
                "update_norm": u_norm,
                "cos_to_median": cos,
                "L2_to_median": l2,
                "train_loss": (
                    float(train_losses[i])
                    if train_losses[i] is not None
                    and np.isfinite(train_losses[i])
                    else None
                ),
                "num_examples": int(num_examples_list[i]),
            }
        )
    return out


def malicious_gt_from_schedule(
    scenario_round_entries: List[Dict[str, Any]], logical_cid: str
) -> tuple[bool, str]:
    """Given the scenario_round entries list (as built in ScenarioStrategy), return
    (is_malicious, attack_type) for a given logical_cid.
    Returns (False, "") if the client is honest or not in the schedule.
    """
    for entry in scenario_round_entries:
        if entry["logical_id"] == logical_cid:
            at = entry.get("attack_type") or ""
            return bool(at), at
    return False, ""

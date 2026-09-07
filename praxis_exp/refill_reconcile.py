"""Reconcile a sweep parent run's §5.6 completeness tags after a refill.

GWU-41 required addition (director ruling 2026-07-24): when a refill completes
previously-missing cells, the ORIGINAL parent run's §5.6 seal tags
(``sweep_incomplete`` / ``missing_cells``, set by the self-heal finalizer) must
be updated to reflect the post-refill state — otherwise the parent claims
incomplete forever. This must be automatic (not operator-remembered) and must
NEVER delete the historical record: a ``refill_history`` tag records what was
refilled and when, built from the durable refill contract records.

The single reconciliation implementation ``reconcile_sweep_tags`` is shared by
both paths that evaluate sweep completeness:
  - the self-heal finalizer/reaper (``selfheal_lambda._annotate_incomplete_sweep``)
  - the CLI enrich seal path (``enrich.enrich_experiment``, the standard
    post-sweep step the operator already runs to ingest the refilled children)

so a refilled sweep reconciles on whichever runs next.

Invariant preserved from the pre-GWU-41 behavior: a PRISTINE complete sweep (one
that never had a refill) still gets ``done_count`` / ``n_units`` only —
``sweep_incomplete`` / ``missing_cells`` stay ABSENT (never ``"false"``). Only a
sweep that actually has refill records reconciles its stale incompleteness tags.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from praxis_exp.integrity import is_done
from praxis_exp.manifest import read_manifest
from praxis_exp.storage import ObjectStore, sweep_prefix

# Fields lifted from each refill contract record into the parent's refill_history.
_REFILL_HISTORY_FIELDS = (
    "serial", "refilled_cells", "launched_at", "git_sha", "image_digest",
    "provenance_changed",
)


def _serial_sort_key(key: str) -> tuple[int, int, str]:
    """Order refill record keys by NUMERIC serial so r10 sorts after r2, not
    before it (lexicographic ``sorted`` would misorder at r10+).
    ``sweeps/{exp}/refills/<serial>/manifest.json`` -> the ``rN`` segment; a
    non-``rN`` segment sorts last, then lexicographically, for determinism."""
    seg = key.split("/")[-2] if "/" in key else key
    m = re.fullmatch(r"r(\d+)", seg)
    return (0, int(m.group(1)), "") if m else (1, 0, seg)


def read_refill_records(store: ObjectStore, exp_id: str) -> list[dict[str, Any]]:
    """The meta of every refill contract record under
    ``sweeps/{exp_id}/refills/<serial>/manifest.json``, oldest serial first (by
    NUMERIC serial). Absent/unreadable records are skipped (best-effort history,
    never fatal)."""
    prefix = f"{sweep_prefix(exp_id)}/refills/"
    keys = [k for k in store.find_keys(prefix, limit=1000) if k.endswith("/manifest.json")]
    records: list[dict[str, Any]] = []
    for key in sorted(keys, key=_serial_sort_key):
        try:
            meta = json.loads(store.get_bytes(key)).get("meta", {})
        except Exception:
            continue
        records.append(meta)
    return records


def _refill_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {f: rec[f] for f in _REFILL_HISTORY_FIELDS if f in rec}
        for rec in records
    ]


def reconcile_sweep_tags(
    client: Any, store: ObjectStore, exp_id: str, parent_run_id: str,
) -> None:
    """Recompute sweep completeness from live S3 and (re)write the parent's
    §5.6 tags. May raise; callers wrap best-effort.

    - ``done_count`` / ``n_units``: always refreshed.
    - incomplete (cells still missing): ``missing_cells`` + ``sweep_incomplete=true``.
    - complete AND this sweep had refills: clear the stale incompleteness —
      ``sweep_incomplete=false``, ``missing_cells=""`` — and stamp
      ``sweep_reconciled_at``. The history is preserved in ``refill_history``.
    - complete with NO refills: pristine — leave ``sweep_incomplete`` /
      ``missing_cells`` ABSENT (unchanged legacy behavior).
    - ``refill_history``: written whenever any refill record exists, so the
      record survives the status flipping to complete.
    """
    _, _, units = read_manifest(store, exp_id)
    missing = [u.unit_id for u in units if not is_done(store, exp_id, u.unit_id)]
    n_units = len(units)
    client.set_tag(parent_run_id, "done_count", str(n_units - len(missing)))
    client.set_tag(parent_run_id, "n_units", str(n_units))

    records = read_refill_records(store, exp_id)

    if missing:
        client.set_tag(parent_run_id, "missing_cells", ",".join(missing))
        client.set_tag(parent_run_id, "sweep_incomplete", "true")
    elif records:
        # A refilled sweep that is now complete: clear the finalizer's stale
        # incompleteness rather than leave the parent claiming incomplete forever.
        client.set_tag(parent_run_id, "sweep_incomplete", "false")
        client.set_tag(parent_run_id, "missing_cells", "")
        client.set_tag(parent_run_id, "sweep_reconciled_at", datetime.utcnow().isoformat() + "Z")
    # else: pristine complete sweep -> incompleteness tags stay ABSENT.

    if records:
        client.set_tag(parent_run_id, "refill_history", json.dumps(_refill_history(records)))

"""Per-run durable persistence with a done-marker commit point + idempotent skip."""
from __future__ import annotations

from pathlib import Path

from praxis_exp import storage
from praxis_exp.storage import ObjectExistsError, ObjectStore


class IntegrityError(RuntimeError):
    """Raised when a persist precondition or verify-after-write check fails."""


def persist_unit(
    store: ObjectStore,
    exp_id: str,
    unit_id: str,
    result_path: Path,
    signal_path: Path,
) -> None:
    """Durably record one finished unit: payload first, done-marker last.

    Ordering is load-bearing. The done-marker is the commit point: a reader that
    sees the marker is guaranteed the result + signal already landed. The marker
    is written conditionally (if_none_match) so a concurrent/duplicate run is a
    harmless no-op (units are deterministic by seed).

    Callers should check is_done() first; persist_unit unconditionally re-uploads
    the payload (deterministic content) and is a no-op only on the marker.
    """
    # pre-flight: refuse to commit if a source artifact is missing
    for p, label in ((result_path, "result_path"), (signal_path, "signal_path")):
        if not Path(p).is_file():
            raise IntegrityError(f"{label} does not exist or is not a file: {p}")

    result_k = storage.result_key(exp_id, unit_id)
    signal_k = storage.signal_key(exp_id, unit_id)

    store.put_file(result_k, result_path)
    store.put_file(signal_k, signal_path)

    # verify-after-write: each artifact must be present AND byte-complete (a
    # multipart upload can fail mid-flight while head() still reports it present)
    for key, local_path in ((result_k, result_path), (signal_k, signal_path)):
        actual = store.size(key)
        if actual is None:
            raise IntegrityError(f"verify-after-write failed: {key} absent after upload")
        expected = Path(local_path).stat().st_size
        if actual != expected:
            raise IntegrityError(
                f"verify-after-write size mismatch on {key}: expected {expected} bytes, got {actual}"
            )

    try:
        store.put_bytes(storage.marker_key(exp_id, unit_id), b"", if_none_match=True)
    except ObjectExistsError:
        pass  # already committed by a prior/concurrent identical run


def is_done(store: ObjectStore, exp_id: str, unit_id: str) -> bool:
    """True iff the unit's done-marker exists (the idempotent-skip predicate)."""
    return store.head(storage.marker_key(exp_id, unit_id))

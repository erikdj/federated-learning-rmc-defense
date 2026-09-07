"""Pure, DI-friendly helpers for MLflow experiment/run enrichment.

Shared by ``matrix_launch.py`` (parent-run + experiment metadata),
``docker/entrypoint.py`` (child-run tags/params/metrics), and ``ingest.py``
(S3 link backfill). Kept pure — no ``mlflow``/``boto3`` imports — so callers
can unit test the string/dict building without network or AWS access. All
S3 key strings are built exclusively from ``praxis_exp.storage``'s key
functions so the layout is defined in exactly one place.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from praxis_exp.storage import (
    manifest_key,
    marker_key,
    result_key,
    s3_console_url,
    s3_uri,
    signal_key,
    sweep_prefix,
)

__all__ = [
    "current_amendment_filename",
    "build_experiment_description",
    "build_parent_run_params",
    "parent_run_s3_tags",
    "unit_s3_tags",
    "dataset_source_uri",
    "dataset_digest_from_metadata",
    "cloudwatch_log_url",
]

# Matches both the unversioned base spec (2026-05-27-praxis-experimental-design.md)
# and dated amendments (2026-07-11-praxis-experimental-design-v1.6.md).
_SPEC_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-praxis-experimental-design(?:-v(?P<ver>\d+\.\d+))?\.md$"
)

_BASE_SPEC_NAME = "2026-05-27-praxis-experimental-design.md"


def current_amendment_filename(repo_root: Path) -> str:
    """The filename of the most recent dated praxis-experimental-design spec
    from the legacy amendment directory inspected by this function. Falls back to the
    unversioned base spec name if no dated amendment file exists on disk
    (e.g. a fresh checkout before any amendment landed).

    Selection is by filename date (YYYY-MM-DD prefix), which sorts
    lexicographically — NOT by the embedded ``-vN.M`` suffix, since the spec's
    own amendment protocol dates every successor file.
    """
    specs_dir = Path(repo_root) / "docs" / "superpowers" / "specs"
    candidates: list[tuple[str, str]] = []
    if specs_dir.is_dir():
        for f in specs_dir.iterdir():
            m = _SPEC_RE.match(f.name)
            if m:
                candidates.append((m.group("date"), f.name))
    if not candidates:
        return _BASE_SPEC_NAME
    candidates.sort()
    return candidates[-1][1]


def _doc_title(doc: Any) -> str:
    for line in doc.body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return f"{doc.exp_id} — {doc.slug}"


def _first_prose_paragraph(body: str) -> str:
    for para in body.strip().split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#"):
            return " ".join(line.strip() for line in para.splitlines())
    return ""


def build_experiment_description(doc: Any, repo_root: Path, doc_path: Path) -> str:
    """Markdown body for the experiment-level ``mlflow.note.content`` tag.

    Combines the design doc's title, its first prose paragraph, the
    repo-relative path to the design doc itself, the methodology version
    (per ``docs/METHODOLOGY_LOG.md``), and the current experimental-design
    amendment filename — so a reader in the MLflow UI never has to leave it
    to find the governing spec.
    """
    repo_root = Path(repo_root)
    doc_path = Path(doc_path)
    rel_path = doc_path.relative_to(repo_root) if doc_path.is_absolute() else doc_path
    amendment = current_amendment_filename(repo_root)
    lines = [f"### {_doc_title(doc)}", ""]
    paragraph = _first_prose_paragraph(doc.body)
    if paragraph:
        lines += [paragraph, ""]
    lines += [
        f"- Design doc: `{rel_path}`",
        f"- Methodology version: `{doc.methodology_version}` per `docs/METHODOLOGY_LOG.md`",
        f"- Experimental design spec (current amendment): `docs/superpowers/specs/{amendment}`",
    ]
    return "\n".join(lines)


def build_parent_run_params(doc: Any, n_units: int) -> dict[str, str]:
    """Matrix-sweep PARAMS (not tags — these describe the design, not the
    launch instance) for the parent run: what was actually swept."""
    return {
        "defenses": ",".join(doc.defenses),
        "scenarios": ",".join(doc.scenarios),
        "seeds": ",".join(str(s) for s in doc.seeds),
        "mode": doc.mode,
        "rounds": str(doc.rounds),
        "max_per_client": str(doc.max_per_client),
        # Always present so the parent run reconstructs the full swept matrix
        # from params alone; repeats=1 explicitly records the replicate axis
        # was inactive rather than omitting it ambiguously.
        "repeats": str(doc.repeats),
        "n_units": str(n_units),
    }


def parent_run_s3_tags(bucket: str, exp_id: str, *, region: str = "us-east-1") -> dict[str, str]:
    """S3 tags for the parent (sweep) run: the manifest URI plus a console
    deep-link to the whole sweep's S3 prefix (results/signals/done/manifest)."""
    return {
        "s3_manifest_uri": s3_uri(bucket, manifest_key(exp_id)),
        "s3_console_url": s3_console_url(bucket, sweep_prefix(exp_id), region=region),
    }


def dataset_source_uri(bucket: str, dataset_dir: str = "edge_full_20") -> str:
    """The ``s3://`` URI of the (shared) EdgeIIoT dataset directory.

    Single source for the native-dataset ``log_input`` source (redesign § 2,
    item 4): every unit trains on the same dataset directory, so the URI is a
    pure function of bucket + directory. Replaces the old ``dataset_dir_s3``
    decoration tag — the S3 location now lives on the run's native Dataset
    input instead of being duplicated as a tag.
    """
    return s3_uri(bucket, f"data/{dataset_dir}/")


def dataset_digest_from_metadata(metadata: dict | None) -> str | None:
    """A short, stable digest for the dataset input, derived from the dataset's
    ``metadata.json`` ``_meta`` block (total_rows, num_features, partition
    count) when available — so re-running the same dataset yields the same
    digest and MLflow dedupes the input. Returns ``None`` when no usable
    metadata is present (MLflow then computes its own digest from the source).
    """
    if not metadata:
        return None
    meta = metadata.get("_meta") or {}
    parts = [
        meta.get("total_rows"),
        meta.get("num_features"),
        (meta.get("partitioning") or {}).get("num_partitions"),
    ]
    material = "|".join(str(p) for p in parts if p is not None)
    if not material:
        return None
    return hashlib.sha1(material.encode()).hexdigest()[:12]


def _cw_encode(value: str) -> str:
    """CloudWatch console fragment encoding: the console double-encodes path
    segments (``/`` -> ``$252F``), i.e. percent-encode then swap ``%`` -> ``$25``.
    """
    return quote(value, safe="").replace("%", "$25")


def cloudwatch_log_url(
    *,
    region: str = "us-east-1",
    log_group: str = "/aws/batch/job",
    log_stream: str | None = None,
) -> str:
    """A clickable CloudWatch Logs console deep-link for a Batch unit
    (redesign § 2, item 6). Deep-links straight to the unit's log *stream*
    when its name is known; otherwise to the log *group* (the operator lands
    where they can find the stream). Best-effort by design — the stream name
    is frequently unknown at run start.
    """
    base = (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{_cw_encode(log_group)}"
    )
    if log_stream:
        return f"{base}/log-events/{_cw_encode(log_stream)}"
    return base


def unit_s3_tags(bucket: str, exp_id: str, unit_id: str, *, region: str = "us-east-1") -> dict[str, str]:
    """S3 tags for one matrix unit's child run: the three artifacts
    ``persist_unit`` durably writes (result/signal/done-marker), built from
    the SAME key functions ``persist_unit``/``is_done`` use (no duplicated
    string logic), plus a console deep-link scoped to this unit's result key.
    """
    return {
        "s3_result_uri": s3_uri(bucket, result_key(exp_id, unit_id)),
        "s3_signal_uri": s3_uri(bucket, signal_key(exp_id, unit_id)),
        "s3_done_uri": s3_uri(bucket, marker_key(exp_id, unit_id)),
        # exact=True: the console filter is a plain string-prefix over keys,
        # and the result is an OBJECT (`{unit_id}.json`), not a folder — a
        # trailing slash would match nothing.
        "s3_console_url": s3_console_url(
            bucket, f"{sweep_prefix(exp_id)}/results/{unit_id}", region=region, exact=True
        ),
    }

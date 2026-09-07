"""``praxis exp ingest EXP-NNN`` implementation."""
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from praxis_exp.design_doc import parse
from praxis_exp.mlflow_client import PraxisMlflowClient
from praxis_exp.storage import s3_console_url, s3_uri


class IngestError(RuntimeError):
    pass


def _find_design_doc(repo_root: Path, exp_id: str) -> Path:
    exp_dir = repo_root / "docs" / "experiments"
    for f in exp_dir.iterdir():
        if f.is_file() and f.name.startswith(f"{exp_id}-") and not f.name.endswith("-result.md"):
            return f
    raise IngestError(f"no design doc for {exp_id}")


def _check_criteria(predictions: dict[str, Any], result: dict[str, Any]) -> bool:
    """Compare predictions against actual result. Return True if criteria met."""
    final_acc = result.get("final_accuracy", result.get("final_acc"))
    if final_acc is None:
        return False
    lo = predictions.get("final_accuracy_min")
    hi = predictions.get("final_accuracy_max")
    if lo is not None and final_acc < float(lo):
        return False
    if hi is not None and final_acc > float(hi):
        return False
    return True


class _AttrClient:
    """Minimal interface PraxisMlflowClient needs to expose to ingest.

    ``artifact_uri_for_run`` is OPTIONAL (checked via hasattr below) — a
    client that doesn't implement it just skips the S3-link tags (req G),
    same as any other optional-enrichment code path in this project."""

    def find_run_id_for_exp(self, exp_id: str) -> str: ...
    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None: ...
    def set_tag(self, run_id: str, key: str, value: str) -> None: ...
    def log_artifact(self, run_id: str, path: str) -> None: ...
    def artifact_uri_for_run(self, run_id: str) -> str: ...


def _set_s3_link_tags(client: Any, run_id: str, result_path: Path) -> None:
    """req G: tag the ingested run with s3_result_uri/s3_console_url,
    derived from the run's real MLflow artifact_uri (the SAME s3:// location
    ``log_artifact`` just uploaded ``result_path`` to) — not a hand-rolled
    guess at the artifact-location convention. Best-effort: a client that
    doesn't support artifact_uri_for_run, or a server error resolving it,
    must never fail ingestion (mirrors the graceful-degradation pattern used
    throughout the MLflow enrichment work)."""
    if not hasattr(client, "artifact_uri_for_run"):
        return
    try:
        base_uri = client.artifact_uri_for_run(run_id)
        parsed = urlsplit(base_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            return
        bucket = parsed.netloc
        base_key = parsed.path.lstrip("/")
        result_uri = s3_uri(bucket, f"{base_key}/{Path(result_path).name}")
        client.set_tag(run_id, "s3_result_uri", result_uri)
        client.set_tag(run_id, "s3_console_url", s3_console_url(bucket, base_key))
    except Exception as e:
        print(f"[ingest] WARN: failed to set S3 link tags for run {run_id}: {e}")


def ingest_experiment(
    repo_root: Path,
    exp_id: str,
    result_path: Optional[Path] = None,
    _client: Optional[_AttrClient] = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    doc = parse(_find_design_doc(repo_root, exp_id))
    if result_path is None:
        # default location matches launch.py _default_runner --output-dir (`results/{exp_id}-aws/`)
        result_path = repo_root / "results" / f"{exp_id}-aws" / f"phase4_flower__{doc.params['defense'].lower()}__seed{doc.params['seed']}.json"
    result = json.loads(Path(result_path).read_text())

    client = _client or PraxisMlflowClient()
    # If the production client doesn't have find_run_id_for_exp we patch by reading git tag
    if hasattr(client, "find_run_id_for_exp"):
        run_id = client.find_run_id_for_exp(exp_id)
    else:
        run_id = _read_run_id_from_tag(repo_root, exp_id)

    # Log final metrics
    for key in ("final_f1", "final_accuracy", "mean_accuracy"):
        if key in result and result[key] is not None:
            client.log_metric(run_id, key, float(result[key]))

    # Log signal log + scenario + result.json as artifacts
    sig_dir = repo_root / "signals"
    if sig_dir.exists():
        for sig in sig_dir.glob(f"*{doc.params['defense'].lower()}*seed{doc.params.get('seed', 42)}.jsonl"):
            client.log_artifact(run_id, str(sig))
    client.log_artifact(run_id, str(result_path))
    _set_s3_link_tags(client, run_id, result_path)

    # Evaluate pre-registered predictions
    ok = _check_criteria(doc.predictions, result)
    client.set_tag(run_id, "criteria_ok", "true" if ok else "false")

    # Write result.md companion
    result_md = repo_root / "docs" / "experiments" / f"{exp_id}-result.md"
    result_md.write_text(_render_result_md(doc, result, ok))

    return {"run_id": run_id, "criteria_ok": ok}


def _read_run_id_from_tag(repo_root: Path, exp_id: str) -> str:
    import subprocess
    out = subprocess.check_output(["git", "tag", "-n99", f"exp/{exp_id}"], cwd=str(repo_root), text=True)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("mlflow_run_id:"):
            return line.split(":", 1)[1].strip()
    raise IngestError(f"could not find mlflow_run_id in tag exp/{exp_id}")


def _render_result_md(doc: Any, result: dict[str, Any], ok: bool) -> str:
    return f"""# {doc.exp_id} — {doc.slug} — Result

**Status:** {'PASS' if ok else 'FAIL'}
**Hypothesis:** {doc.hypothesis}

## Summary metrics

| Metric | Value |
|---|---|
| final_f1 | {result.get('final_f1', 'n/a')} |
| final_accuracy | {result.get('final_accuracy', result.get('final_acc', 'n/a'))} |
| mean_accuracy | {result.get('mean_accuracy', result.get('mean_acc', 'n/a'))} |

## Criteria evaluation

Pre-registered predictions: `{doc.predictions}`
Met: **{ok}**

## Trajectory

See MLflow run for per-round metrics.
"""

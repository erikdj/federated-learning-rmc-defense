"""``praxis exp launch EXP-NNN`` implementation."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from praxis_exp.design_doc import DesignDocError, parse
from praxis_exp import git_helper
from praxis_exp.mlflow_client import PraxisMlflowClient


class LaunchError(RuntimeError):
    """Raised when a launch precondition fails."""


def _read_ignore(repo_root: Path) -> list[str]:
    f = repo_root / ".experiment-ignore"
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip() and not line.startswith("#")]


def _find_design_doc(repo_root: Path, exp_id: str) -> Path:
    exp_dir = repo_root / "docs" / "experiments"
    for f in exp_dir.iterdir():
        if f.is_file() and f.name.startswith(f"{exp_id}-") and not f.name.endswith("-result.md"):
            return f
    raise LaunchError(f"no design doc found for {exp_id} in {exp_dir}")


def _default_runner(repo: Path, run_id: str, params: dict[str, Any]) -> int:
    """Default runner: shell out to scripts/run_phase4_flower.py."""
    cmd = [
        sys.executable,
        "scripts/run_phase4_flower.py",
        "--configs", str(params["defense"]),
        "--modes", str(params["mode"]),
        "--seeds", str(params["seed"]),
        "--scenario", str(params["scenario"]),
        "--max-per-client", str(params["max_per_client"]),
        "--reporting-split", "val",
        "--output-dir", str(repo / "results" / f"{params['exp_id']}-aws/"),
    ]
    env = dict(os.environ, PRAXIS_MLFLOW_RUN_ID=run_id, PRAXIS_RAY_CPUS=os.environ.get("PRAXIS_RAY_CPUS", "8"))
    return subprocess.call(cmd, cwd=str(repo), env=env)


def _preflight_data(repo: Path, params: dict[str, Any]) -> None:
    """v1.3 discipline: refuse to launch if required data isn't on this host.

    EXP-003 exposed the silent-failure mode where data/edge_full_20/client_*.parquet
    was missing from a fresh git clone (334MB of parquet data isn't tracked in git).
    Raise loudly here so the experimenter sees the gap *before* the MLflow run is
    created and the git tag is pushed.
    """
    scenario = Path(params.get("scenario", ""))
    if scenario.parts and not (repo / scenario).exists():
        raise LaunchError(f"scenario file missing: {scenario}")

    # The dataset name is determined by the runner (DATASET constant + _rmc suffix
    # stripping for eval), so we check the most common eval-side dependency:
    # data/edge_full_20/client_*.parquet. If a future experiment uses a different
    # dataset, add an entry here.
    edge_data_dir = repo / "data" / "edge_full_20"
    if not edge_data_dir.exists():
        raise LaunchError(
            f"data dir missing: {edge_data_dir}. The runner's FixedEvalManager "
            f"requires per-client parquet files here. On a fresh git clone these "
            f"must be synced separately (the data is ~334MB and not tracked in git)."
        )
    client_files = list(edge_data_dir.glob("client_*.parquet"))
    if len(client_files) < 20:
        raise LaunchError(
            f"only {len(client_files)} client parquet files in {edge_data_dir}, "
            f"expected >= 20. Data sync incomplete."
        )


def launch_experiment(
    repo_root: Path,
    exp_id: str,
    _client: Optional[Any] = None,
    _runner: Optional[Callable[[Path, str, dict[str, Any]], int]] = None,
    _no_push: bool = False,
    _skip_preflight: bool = False,
) -> dict[str, Any]:
    """Atomically: validate, tag, push, create MLflow run, invoke runner."""
    repo_root = Path(repo_root)

    # 1. Load + validate design doc
    try:
        doc_path = _find_design_doc(repo_root, exp_id)
    except LaunchError:
        raise
    try:
        doc = parse(doc_path)
    except DesignDocError as e:
        raise LaunchError(f"design doc invalid: {e}") from e

    # 2. Working tree clean
    if not git_helper.working_tree_clean(repo_root, ignore=_read_ignore(repo_root)):
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True)
        raise LaunchError(f"working tree has uncommitted changes:\n{dirty}")

    # 2b. Data preflight (v1.3 discipline). Tests opt out via _skip_preflight.
    if not _skip_preflight:
        _preflight_data(repo_root, doc.params)

    sha = git_helper.head_sha(repo_root)

    # 3. Create MLflow run
    client = _client or PraxisMlflowClient()
    exp_name = f"{exp_id}__{doc.slug}"
    mlflow_exp_id = client.get_or_create_experiment(exp_name)
    run_id = client.create_run(
        mlflow_exp_id,
        tags={
            "mlflow.runName": exp_name,
            "exp_id": exp_id,
            "slug": doc.slug,
            "hypothesis": doc.hypothesis,
            "methodology_version": doc.methodology_version,
            "design_doc_path": str(doc_path.relative_to(repo_root)),
            "git_sha": sha,
            "launched_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    params = dict(doc.params, exp_id=exp_id)
    client.log_params(run_id, params)
    for k, v in doc.predictions.items():
        client.set_tag(run_id, f"pred.{k}", str(v))

    # 4. Create + push annotated tag
    tag_msg = f"""exp/{exp_id} - {doc.slug}

mlflow_run_id: {run_id}
mlflow_experiment_name: {exp_name}
git_sha: {sha}
launched_at: {datetime.utcnow().isoformat()}Z
hypothesis: {doc.hypothesis}
design_doc: {doc_path.relative_to(repo_root)}
"""
    try:
        git_helper.create_annotated_tag(repo_root, f"exp/{exp_id}", tag_msg)
        if not _no_push:
            git_helper.push_with_tags(repo_root)
    except subprocess.CalledProcessError as e:
        # rollback
        git_helper.delete_local_tag(repo_root, f"exp/{exp_id}")
        client.set_terminated(run_id, "FAILED")
        raise LaunchError(f"git tag/push failed: {e}") from e

    # 5. Invoke runner
    runner = _runner or _default_runner
    rc = runner(repo_root, run_id, params)
    if rc != 0:
        client.set_terminated(run_id, "FAILED")
        raise LaunchError(f"runner exited non-zero ({rc})")

    client.set_terminated(run_id, "FINISHED")
    return {"run_id": run_id, "experiment_name": exp_name, "git_sha": sha}

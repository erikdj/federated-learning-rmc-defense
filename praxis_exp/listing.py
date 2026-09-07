"""``praxis exp list`` + INDEX.md regenerator."""
from pathlib import Path
from typing import Optional, Any

from praxis_exp.mlflow_client import PraxisMlflowClient


class _ListClient:
    def search_runs(self, **filters) -> list[dict[str, Any]]: ...


def list_experiments(
    repo_root: Path,
    status: Optional[str] = None,
    methodology: Optional[str] = None,
    hypothesis: Optional[str] = None,
    limit: int = 200,
    _client: Optional[_ListClient] = None,
) -> list[dict[str, Any]]:
    """Print a table to stdout and rewrite INDEX.md."""
    client = _client or _DefaultListClient()
    runs = client.search_runs(status=status, methodology=methodology, hypothesis=hypothesis, limit=limit)

    # Render INDEX.md
    repo_root = Path(repo_root)
    idx = repo_root / "docs" / "experiments" / "INDEX.md"
    idx.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiments index", "", "| ID | Slug | Status | Methodology | Criteria OK |", "|---|---|---|---|---|"]
    for r in runs:
        lines.append(
            f"| {r.get('exp_id','?')} | {r.get('slug','?')} | {r.get('status','?')} | "
            f"{r.get('methodology_version','?')} | {r.get('criteria_ok','?')} |"
        )
    idx.write_text("\n".join(lines) + "\n")
    return runs


class _DefaultListClient:
    """Real MLflow-backed search."""

    def __init__(self) -> None:
        self._pc = PraxisMlflowClient()

    def search_runs(self, **filters) -> list[dict[str, Any]]:
        import mlflow
        all_exp = self._pc._client.search_experiments()
        rows: list[dict[str, Any]] = []
        for exp in all_exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=filters.get("limit", 200))
            for _, r in runs.iterrows():
                rows.append({
                    "exp_id": r.get("tags.exp_id", "?"),
                    "slug": r.get("tags.slug", "?"),
                    "status": r["status"],
                    "methodology_version": r.get("tags.methodology_version", "?"),
                    "criteria_ok": r.get("tags.criteria_ok", "?"),
                })
        return rows

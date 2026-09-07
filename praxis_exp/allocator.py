"""Allocates the next EXP-NNN sequential id."""
import re
from pathlib import Path

EXP_PATTERN = re.compile(r"^EXP-(\d{3})-(?!result)[a-z0-9-]+\.md$")


def next_experiment_id(repo_root: Path) -> str:
    """Return the next sequential EXP-NNN.

    Scans ``<repo_root>/docs/experiments/`` for files matching
    ``EXP-NNN-<slug>.md`` (excluding result files). The next id is one
    greater than the max observed; 'EXP-001' when none exist.
    """
    exp_dir = Path(repo_root) / "docs" / "experiments"
    if not exp_dir.exists():
        return "EXP-001"

    used: set[int] = set()
    for f in exp_dir.iterdir():
        if not f.is_file():
            continue
        m = EXP_PATTERN.match(f.name)
        if m:
            used.add(int(m.group(1)))

    next_n = max(used, default=0) + 1
    return f"EXP-{next_n:03d}"

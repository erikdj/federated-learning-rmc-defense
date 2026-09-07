"""``praxis exp new <slug>`` implementation."""
from datetime import date
from pathlib import Path
import re

from praxis_exp.allocator import next_experiment_id

SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{2,80}$")


class ScaffoldError(ValueError):
    """Raised when scaffolding fails."""


def _current_methodology_version(repo_root: Path) -> str:
    log = repo_root / "docs" / "METHODOLOGY_LOG.md"
    if not log.exists():
        return "v0.1"
    text = log.read_text()
    matches = re.findall(r"^## (v\d+\.\d+) ", text, flags=re.M)
    if not matches:
        return "v0.1"
    return matches[0]  # most recent (top of file)


def scaffold_experiment(repo_root: Path, slug: str) -> Path:
    """Create ``docs/experiments/EXP-NNN-<slug>.md`` from template; return its path."""
    if not SLUG_RE.match(slug):
        raise ScaffoldError(
            f"invalid slug {slug!r}: must match {SLUG_RE.pattern}"
        )

    exp_dir = Path(repo_root) / "docs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for existing in exp_dir.iterdir():
        if existing.is_file() and existing.stem.endswith(f"-{slug}"):
            raise ScaffoldError(f"a design doc for slug {slug!r} already exists: {existing.name}")

    exp_id = next_experiment_id(repo_root)
    methodology_version = _current_methodology_version(repo_root)

    template = (exp_dir / "_TEMPLATE.md").read_text()
    out = template.replace("{{EXP_ID}}", exp_id)
    out = out.replace("{{SLUG}}", slug)
    out = out.replace("{{DATE}}", date.today().isoformat())
    out = out.replace("{{METHODOLOGY_VERSION}}", methodology_version)

    path = exp_dir / f"{exp_id}-{slug}.md"
    path.write_text(out)
    return path

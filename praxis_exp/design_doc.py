"""Parses ``docs/experiments/EXP-NNN-<slug>.md`` files."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import yaml


REQUIRED_FIELDS = ("exp_id", "slug", "hypothesis", "methodology_version", "params")
PLACEHOLDER_PATTERN = re.compile(r"\b(TBD|TODO|<FILL>)\b")


class DesignDocError(ValueError):
    """Raised when a design doc fails validation."""


@dataclass
class DesignDoc:
    exp_id: str
    slug: str
    hypothesis: str
    methodology_version: str
    params: dict[str, Any]
    predictions: dict[str, Any]
    body: str
    path: Path


def parse(path: Path) -> DesignDoc:
    """Parse a design doc file. Raises DesignDocError if invalid."""
    text = Path(path).read_text()
    if not text.startswith("---\n"):
        raise DesignDocError(f"{path}: missing YAML front-matter")
    _, frontmatter, body = text.split("---\n", 2)
    data = yaml.safe_load(frontmatter)
    if not isinstance(data, dict):
        raise DesignDocError(f"{path}: front-matter must be a YAML mapping")
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise DesignDocError(f"{path}: missing required field '{field}'")
    if PLACEHOLDER_PATTERN.search(body):
        raise DesignDocError(f"{path}: body contains placeholder (TBD/TODO/<FILL>) — design doc incomplete")
    return DesignDoc(
        exp_id=str(data["exp_id"]),
        slug=str(data["slug"]),
        hypothesis=str(data["hypothesis"]),
        methodology_version=str(data["methodology_version"]),
        params=dict(data["params"]),
        predictions=dict(data.get("predictions", {})),
        body=body,
        path=Path(path),
    )

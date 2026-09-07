"""Public test-command defaults."""
from pathlib import Path
import re


def test_default_pytest_selection_excludes_special_protocol_gates():
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^addopts\s*=\s*"([^"]*)"', text, flags=re.MULTILINE)
    assert match is not None
    addopts = match.group(1)

    for marker in ("slow", "anchor", "ray", "golden"):
        assert f"not {marker}" in addopts

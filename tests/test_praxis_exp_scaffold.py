"""Scaffold command tests."""
from pathlib import Path
import pytest


@pytest.fixture
def tmp_repo(tmp_path):
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    tpl = tmp_path / "docs" / "experiments" / "_TEMPLATE.md"
    tpl.write_text("---\nexp_id: {{EXP_ID}}\nslug: {{SLUG}}\n---\nbody {{DATE}} {{METHODOLOGY_VERSION}}\n")
    (tmp_path / "docs" / "METHODOLOGY_LOG.md").write_text(
        "## v1.2 — 2026-05-27 — Framework adopted\n"
    )
    return tmp_path


def test_scaffold_creates_file_with_substitution(tmp_repo):
    from praxis_exp.scaffold import scaffold_experiment
    path = scaffold_experiment(tmp_repo, slug="my-slug")
    assert path.exists()
    text = path.read_text()
    assert "EXP-001" in text
    assert "my-slug" in text
    assert "v1.2" in text


def test_scaffold_rejects_bad_slug(tmp_repo):
    from praxis_exp.scaffold import scaffold_experiment, ScaffoldError
    with pytest.raises(ScaffoldError, match="slug"):
        scaffold_experiment(tmp_repo, slug="BAD_SLUG")


def test_scaffold_warns_on_duplicate_slug(tmp_repo):
    from praxis_exp.scaffold import scaffold_experiment, ScaffoldError
    scaffold_experiment(tmp_repo, slug="reused")
    with pytest.raises(ScaffoldError, match="exists"):
        scaffold_experiment(tmp_repo, slug="reused")

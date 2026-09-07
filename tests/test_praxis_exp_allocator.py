"""Allocator unit tests."""
from pathlib import Path
import pytest


def test_next_id_empty_returns_001(tmp_path):
    """No prior experiments → next is EXP-001."""
    from praxis_exp.allocator import next_experiment_id
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    nid = next_experiment_id(tmp_path)
    assert nid == "EXP-001"


def test_next_id_skips_existing(tmp_path):
    """If EXP-001 + EXP-002 exist, next is EXP-003."""
    from praxis_exp.allocator import next_experiment_id
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-001-foo.md").touch()
    (exp_dir / "EXP-002-bar.md").touch()
    assert next_experiment_id(tmp_path) == "EXP-003"


def test_next_id_zero_padding_to_3(tmp_path):
    """Zero-pads to 3 digits up through EXP-999."""
    from praxis_exp.allocator import next_experiment_id
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-009-z.md").touch()
    assert next_experiment_id(tmp_path) == "EXP-010"


def test_next_id_ignores_non_exp_files(tmp_path):
    """Non-EXP-prefixed files are ignored."""
    from praxis_exp.allocator import next_experiment_id
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "_TEMPLATE.md").touch()
    (exp_dir / "INDEX.md").touch()
    (exp_dir / "EXP-005-real.md").touch()
    assert next_experiment_id(tmp_path) == "EXP-006"


def test_next_id_ignores_result_files(tmp_path):
    """EXP-NNN-result.md doesn't count as a separate experiment."""
    from praxis_exp.allocator import next_experiment_id
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "EXP-001-foo.md").touch()
    (exp_dir / "EXP-001-result.md").touch()
    assert next_experiment_id(tmp_path) == "EXP-002"

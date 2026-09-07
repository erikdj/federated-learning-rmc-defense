"""git_helper module tests using a real temporary git repo."""
import subprocess
from pathlib import Path
import pytest


@pytest.fixture
def tmp_git_repo(tmp_path):
    """Initialise a real git repo in tmp_path with one commit."""
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_working_tree_clean_true(tmp_git_repo):
    from praxis_exp.git_helper import working_tree_clean
    assert working_tree_clean(tmp_git_repo, ignore=[]) is True


def test_working_tree_dirty_with_unignored_file(tmp_git_repo):
    from praxis_exp.git_helper import working_tree_clean
    (tmp_git_repo / "dirty.txt").write_text("x")
    assert working_tree_clean(tmp_git_repo, ignore=[]) is False


def test_working_tree_clean_when_only_ignored_file(tmp_git_repo):
    from praxis_exp.git_helper import working_tree_clean
    (tmp_git_repo / ".omc-state.json").write_text("x")
    assert working_tree_clean(tmp_git_repo, ignore=[".omc-state.json"]) is True


def test_working_tree_clean_with_glob_pattern(tmp_git_repo):
    """v1.3 — `results/EXP-*-aws/` matches `results/EXP-003-aws/`."""
    from praxis_exp.git_helper import working_tree_clean
    (tmp_git_repo / "results").mkdir()
    (tmp_git_repo / "results" / "EXP-003-aws").mkdir()
    (tmp_git_repo / "results" / "EXP-003-aws" / "x.json").write_text("{}")
    assert working_tree_clean(tmp_git_repo, ignore=["results/EXP-*-aws/"]) is True


def test_working_tree_clean_dir_pattern_matches_nested_files(tmp_git_repo):
    """2026-07-10 — dir patterns must survive porcelain's untracked-dir collapse."""
    from praxis_exp.git_helper import working_tree_clean
    nested = tmp_git_repo / "results" / "EXP-003-aws" / "sub"
    nested.mkdir(parents=True)
    (nested / "deep.json").write_text("{}")
    assert working_tree_clean(tmp_git_repo, ignore=["results/EXP-*-aws/"]) is True
    # ...but an unrelated untracked file alongside still fails the gate.
    (tmp_git_repo / "stray.txt").write_text("x")
    assert working_tree_clean(tmp_git_repo, ignore=["results/EXP-*-aws/"]) is False


def test_working_tree_clean_literal_dir_pattern(tmp_git_repo):
    """Literal (non-glob) dir entries like `results/aws_validation/` match nested files."""
    from praxis_exp.git_helper import working_tree_clean
    d = tmp_git_repo / "results" / "aws_validation"
    d.mkdir(parents=True)
    (d / "cmp.json").write_text("{}")
    assert working_tree_clean(tmp_git_repo, ignore=["results/aws_validation/"]) is True


def test_create_annotated_tag(tmp_git_repo):
    from praxis_exp.git_helper import create_annotated_tag
    create_annotated_tag(tmp_git_repo, "exp/EXP-001", "Test message body")
    out = subprocess.check_output(["git", "tag", "-l", "exp/EXP-001"], cwd=tmp_git_repo, text=True)
    assert "exp/EXP-001" in out


def test_create_annotated_tag_message_preserved(tmp_git_repo):
    from praxis_exp.git_helper import create_annotated_tag
    create_annotated_tag(tmp_git_repo, "exp/EXP-001", "Multi\nline\nmessage")
    out = subprocess.check_output(["git", "show", "exp/EXP-001"], cwd=tmp_git_repo, text=True)
    assert "Multi" in out and "line" in out


def test_head_sha(tmp_git_repo):
    from praxis_exp.git_helper import head_sha
    sha = head_sha(tmp_git_repo)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_tag_target_sha_none_when_tag_missing(tmp_git_repo):
    """PR #13 P2 (comment 3566953649): relaunch support needs to inspect
    existing exp/ tags without shelling git tag -f (custody anchors are
    immutable)."""
    from praxis_exp.git_helper import tag_target_sha
    assert tag_target_sha(tmp_git_repo, "exp/EXP-999") is None


def test_tag_target_sha_returns_commit_sha_for_annotated_tag(tmp_git_repo):
    """An annotated tag's own object sha differs from the commit it points
    at; tag_target_sha must dereference to the COMMIT sha (what head_sha
    returns) so equality checks against HEAD work."""
    from praxis_exp.git_helper import create_annotated_tag, head_sha, tag_target_sha
    create_annotated_tag(tmp_git_repo, "exp/EXP-001", "msg")
    assert tag_target_sha(tmp_git_repo, "exp/EXP-001") == head_sha(tmp_git_repo)

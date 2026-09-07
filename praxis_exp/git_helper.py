"""Thin shell wrapper for git operations used by praxis_exp."""
from pathlib import Path
import subprocess
from typing import Iterable


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def working_tree_clean(repo: Path, ignore: Iterable[str]) -> bool:
    """Return True iff `git status` only mentions paths matched by ``ignore``.

    Each entry in ``ignore`` is an exact path, an fnmatch glob pattern
    (e.g. ``results/EXP-*-aws/``), or a directory pattern (trailing ``/``)
    that matches everything beneath it. Glob support added 2026-05-27 after
    EXP-004 hit the exact-match limitation; ``--untracked-files=all`` added
    2026-07-10 because plain porcelain collapses a fully-untracked directory
    to ``dir/``, which silently defeated glob dir patterns like
    ``results/EXP-*-aws/`` when the parent directory itself was untracked.
    """
    import fnmatch
    out = _run(["git", "status", "--porcelain", "--untracked-files=all"], repo)
    if not out:
        return True
    patterns = [p.strip() for p in ignore if p.strip()]

    def _matches(path: str, pattern: str) -> bool:
        if path == pattern:
            return True
        if pattern.endswith("/"):
            # Directory pattern: match the dir itself or anything beneath it
            # (fnmatch `*` crosses `/`, so this covers nested files too).
            return fnmatch.fnmatchcase(path, pattern + "*") or fnmatch.fnmatchcase(
                path.rstrip("/") + "/", pattern
            )
        return fnmatch.fnmatchcase(path, pattern)

    for line in out.splitlines():
        path = line[3:].strip()
        if not any(_matches(path, p) for p in patterns):
            return False
    return True


def head_sha(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], repo)


def current_branch(repo: Path) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)


def tag_target_sha(repo: Path, tag_name: str) -> str | None:
    """The COMMIT sha a tag points at (dereferenced through the annotated-tag
    object via ``^{commit}``), or None if the tag does not exist.

    Added for PR #13 P2 (comment 3566953649): relaunches of the same EXP-NNN
    need to inspect existing ``exp/`` tags — which are immutable
    chain-of-custody anchors and must never be force-moved — to decide
    whether to reuse the tag (same sha) or mint a serial-suffixed one
    (different sha).
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag_name}^{{commit}}"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def create_annotated_tag(repo: Path, tag_name: str, message: str) -> None:
    subprocess.check_call(
        ["git", "tag", "-a", tag_name, "-m", message],
        cwd=str(repo),
    )


def push_with_tags(repo: Path, remote: str = "origin", branch: str = "master") -> None:
    subprocess.check_call(
        ["git", "push", "--follow-tags", remote, branch],
        cwd=str(repo),
    )


def delete_local_tag(repo: Path, tag_name: str) -> None:
    subprocess.call(
        ["git", "tag", "-d", tag_name],
        cwd=str(repo),
    )


def fetch(repo: Path, remote: str = "origin") -> None:
    subprocess.check_call(["git", "fetch", remote], cwd=str(repo))


def local_matches_remote(repo: Path, remote: str = "origin", branch: str = "master") -> bool:
    """Return True iff HEAD == origin/branch after a fetch."""
    fetch(repo, remote)
    local = head_sha(repo)
    remote_sha = _run(["git", "rev-parse", f"{remote}/{branch}"], repo)
    return local == remote_sha

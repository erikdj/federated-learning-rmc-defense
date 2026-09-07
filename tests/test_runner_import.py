"""Importing the experiment runner must not create output directories."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_runner_import_does_not_create_result_directories(tmp_path):
    """A relocated module import is isolated from existing repository outputs."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    runner = scripts_dir / "run_phase4_flower.py"
    shutil.copy2(REPO / "scripts" / "run_phase4_flower.py", runner)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO), env.get("PYTHONPATH", "")))
    )
    subprocess.run(
        [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1])", str(runner)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (tmp_path / "results").exists()

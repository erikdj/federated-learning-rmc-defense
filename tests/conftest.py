"""Pytest scaffolding: ensure project root is on sys.path so `flowerfl`
package imports work even though the package is not pip-installed in the
conda env. This file is a minor deviation from the plan — required because
`flowerfl` is not installed in editable mode in the conda env, only on
the runtime sys.path via CWD-as-blank.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

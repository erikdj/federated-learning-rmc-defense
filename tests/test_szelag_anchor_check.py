import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

def test_within_tolerance_passes_and_outside_fails():
    from szelag_anchor_check import within_tolerance
    assert within_tolerance(final_acc=0.978, anchor=0.9791, tol=0.01) is True
    assert within_tolerance(final_acc=0.95, anchor=0.9791, tol=0.01) is False

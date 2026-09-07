import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

def test_paired_diff_ci_excludes_zero_for_clear_effect():
    from effect_size import paired_diff_bootstrap_ci
    tge =  [0.90,0.91,0.89,0.92,0.90,0.93,0.88,0.91,0.90,0.92]
    base = [0.80,0.81,0.79,0.82,0.80,0.83,0.78,0.81,0.80,0.82]
    est, lo, hi = paired_diff_bootstrap_ci(tge, base, n_boot=2000, seed=0)
    assert est > 0 and lo > 0

"""Paired effect size with bootstrap CI (F3) — primary quantitative summary."""
import numpy as np


def paired_diff_bootstrap_ci(a, b, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Mean paired difference (a - b) with a percentile bootstrap CI over seeds.
    Returns (estimate, lo, hi). `a` and `b` are paired per-seed metric values."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(d.mean()), float(lo), float(hi)

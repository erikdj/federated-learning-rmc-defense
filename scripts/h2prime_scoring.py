"""Frozen-construction scoring primitives and the § 2.2b golden gate.

Split out of `adjudicate_h2prime.py` (2026-08-13) at the 800-line ceiling. What
lives here is everything that TRANSCRIBES a frozen construction — the golden
gate that proves the window-feature builder is unmodified, the § 2.2a cut at an
arbitrary target, and the § 2.2/§ 3.2 blended-LOAO readout — as opposed to the
pass that applies them and the CLI that drives it.

Pure move plus the § 6 additions; the dev-smoke digest is invariant across the
split itself and that is proved separately from the feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2prime_common import GOLDEN, HardStop, R  # noqa: E402


def golden_gate() -> dict:
    """Run the committed golden-hash gate. A failure is a hard stop (§ 2.2b)."""
    checks = [
        GOLDEN.test_golden_artifacts_exist,
        GOLDEN.test_golden_input_hash_matches_recorded,
        GOLDEN.test_golden_expected_file_hash_matches_recorded,
        GOLDEN.test_frozen_builder_reproduces_golden_hash,
        GOLDEN.test_output_row_count_equals_input_row_count,
        GOLDEN.test_single_row_client_gets_all_zero_window_feats,
        GOLDEN.test_every_episode_first_round_is_all_zeros,
        GOLDEN.test_no_derived_feature_is_nan_or_null,
    ]
    for check in checks:
        try:
            check()
        except AssertionError as exc:                       # pragma: no cover
            raise HardStop(
                f"§ 2.2b golden-hash gate FAILED at {check.__name__}: {exc}"
            ) from exc
    return {
        "gate": "tests/test_window_feats_golden.py (committed, unmodified)",
        "checks_run": [c.__name__ for c in checks],
        "recorded_hashes": GOLDEN._recorded_hashes(),
        "frozen_commit": GOLDEN.FROZEN_COMMIT,
        "status": "PASS",
    }


#: § 6 (ratification addition) — the reported FPR bracket. 0.10 is the
#: ADJUDICATING point and is listed here too, so the bracket's own 10 % entry is
#: produced by the same code path as the other three and can be checked against
#: the band's number instead of being assumed equal to it.
BRACKET_TARGETS = (0.01, 0.02, 0.05, 0.10)


def _cut_at(honest_scores, target: float, higher_is_trust: bool) -> float:
    """The § 2.2a cut construction at an ARBITRARY target FPR.

    `R.cut_from_calibration` hard-codes the frozen `TARGET_FPR`, and the builder
    is under the golden gate — it must not be edited to take a parameter. This
    mirrors its body exactly with the target lifted out, and a test asserts the
    two agree bit-for-bit at 0.10 on random input, so the bracket is the same
    construction rather than a lookalike.
    """
    a = np.asarray(honest_scores, dtype=float)
    return float(np.quantile(a, target if higher_is_trust else 1 - target))


def _blended_at_cuts(fam, cuts, honest_rows, pm, n_mal, attacks) -> dict:
    """The § 2.2/§ 3.2 blended-LOAO readout under a given cut set.

    ONE construction, called by the adjudicating P1 surface AND by the § 6
    bracket. The bracket's 10 % point therefore cannot drift from the number the
    band adjudicates: they are the same code, not two copies that happen to
    agree today. The mixture weights are malicious-row shares and the blended
    FPR is the weight-matched mixture of per-family honest FPRs, exactly as
    ratified.
    """
    tp, mix, fpr_parts = 0, {}, []
    for A in attacks:
        if not fam[A]:
            continue
        tp += int(R.flagged(pm(A, fam[A]), cuts[A], False).sum())
        w = len(fam[A]) / n_mal
        mix[A] = w
        fpr_parts.append(
            w * float(R.flagged(pm(A, honest_rows), cuts[A], False).mean()))
    return {"recall": tp / n_mal, "fpr": sum(fpr_parts), "family_mix": mix}

"""The § 2.1a-named confirmatory scoring driver for H2′ — entry point.

Spec v1.15 § 2.1a ("Training entry point — NAMED") designates
`scripts/analyze_h2prime_confirm_read.py` as **the only program that scores
confirmatory rows**, mirroring `revalidate_v115.py`'s construction (the § 2.2a
rotation, the § 2.2b feature builder, the § 2.2a item 7 quantile / equality
conventions, and the § 2.1a estimator) — the same relationship
`scripts/analyze_h2_confirm_read.py` bears to the H2 dev read.

**Disclosure (2026-08-12):** this file did not exist at ratification (`bada1b8`)
despite § 2.1a describing it as committed then. The implementation lives in
`scripts/adjudicate_h2prime.py`, written before any sealed row was opened and
before the golden gate was run against the confirmatory corpus; this module is
the pre-registered NAME bound to that implementation, so the § 2.1a "only
program" constraint holds in fact. The gap between the spec's description and
the repository state is recorded here rather than papered over.

Correspondence to `revalidate_v115.py` (§ 2.1a requirement): the adjudicator
IMPORTS `derive_window_feats`, `cut_from_calibration`, `flagged`, `FEATS_V115`,
`ATTACKS`, `SCEN_SHORT` and `exact_wilcoxon_onesided` from that module at its
committed state, and imports `BASELINES` from `blended_loao.py`. No construction
is re-derived.

Usage is identical to the implementation module:

    python scripts/analyze_h2prime_confirm_read.py MAP.json --dry-run
    python scripts/analyze_h2prime_confirm_read.py MAP.json --out RESULTS.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjudicate_h2prime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the LOCKED, hash-stamped H3 fingerprint feature artifact.

H3 execution plan Step 1 (`docs/reproduction/experiments.md`):

    "Lock the 45-feature set and prove it reproduces from the current parquet,
     including the 14-column transmission-timing subset. Write the list into a
     versioned, hash-stamped artifact rather than leaving it implicit in a
     research script."

Design authority: v1.10 § 5.0 **D8** (construct = proxy path (a)) and § 5.1
(the 45-feature set G4 "must be verified recoverable/reproducible from the
current parquet... as the top silent-failure risk"); `docs/harness/architecture.md`
"Fingerprint vector (180 dimensions)".

The artifact this writes (`data/fingerprint_features_v1.json`) is
**pre-registration evidence**: it is written once, before τ calibration, and
`tests/test_fingerprint_features.py` re-verifies every stamped hash on each
test run. Re-running this script is only appropriate to (re)generate the
artifact from an unchanged source — a *changed* feature set is a construct
change and requires a dated amendment, not a regeneration.

Usage:
    python scripts/lock_fingerprint_features.py [--check]

    --check Recompute and compare against the committed artifact; exit 1 on
              any difference. Writes nothing. (CI / pre-launch drift gate.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FEATURES = PROJECT_ROOT / "data" / "edge_full" / "features.json"
ARTIFACT = PROJECT_ROOT / "data" / "fingerprint_features_v1.json"

EXPECTED_NUM_FEATURES = 45

# The 14-column transmission-timing subset — the construct-validity anchor for
# the "transmission fingerprint" framing. Verbatim from the Phase-0 research
# script `scripts/fingerprint_feasibility.py:272-275` (`edge_timing`).
TRANSMISSION_TIMING_FEATURES = [
    "udp.time_delta",
    "tcp.len",
    "tcp.seq",
    "tcp.ack",
    "tcp.ack_raw",
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
]

LOCKED_AT = "2026-08-05"


def hash_feature_list(names) -> str:
    """sha256 of ('\\n'.join(names) + '\\n'), utf-8 — the stamped hashing rule."""
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def build_payload() -> "OrderedDict[str, object]":
    if not SOURCE_FEATURES.exists():
        raise SystemExit(
            f"source feature list not found: {SOURCE_FEATURES} "
            "(data/ is gitignored — materialise the dataset first)"
        )
    raw = SOURCE_FEATURES.read_bytes()
    features = list(json.loads(raw)["features"])

    if len(features) != EXPECTED_NUM_FEATURES:
        raise SystemExit(
            f"expected {EXPECTED_NUM_FEATURES} features, found {len(features)} — "
            "this is a construct change, not a regeneration"
        )
    if len(set(features)) != len(features):
        raise SystemExit("source feature list contains duplicates")
    missing = [c for c in TRANSMISSION_TIMING_FEATURES if c not in features]
    if missing:
        raise SystemExit(f"transmission-timing columns absent from the source: {missing}")

    payload: "OrderedDict[str, object]" = OrderedDict()
    payload["_meta"] = OrderedDict(
        [
            ("artifact", "fingerprint_features"),
            ("version", "v1"),
            (
                "description",
                "LOCKED 45-column protocol-feature set the 180-dim H3 transmission "
                "fingerprint is built from (45 features x 4 moments: mean, std, "
                "skew, kurtosis). Hash-stamped so feature drift is a loud failure "
                "rather than a silent change in the meaning of every fingerprint "
                "and of the locked tau.",
            ),
            (
                "authority",
                [
                    "docs/superpowers/specs/2026-08-02-praxis-experimental-design-"
                    "v1.10.md section 5.0 D8 (construct = proxy path (a)) + section "
                    "5.1 (the 45-feature set G4 must be verified recoverable/"
                    "reproducible from the current parquet, including the 14-column "
                    "transmission-timing subset)",
                    "docs/PHASE7_DESIGN.md 'Fingerprint vector (180 dimensions)'",
                    ".planning/h3h4/H3_EXECUTION_PLAN_20260805.md Step 1",
                ],
            ),
            (
                "source",
                "data/edge_full/features.json (byte-identical feature list to "
                "data/edge_full_20/features.json, the dataset the H3 scenarios "
                "resolve to via task.py 'edge_full_20_rmc')",
            ),
            (
                "evaluation_dataset",
                "edge_full_20_rmc -> data/edge_full_20 (label column "
                "'Attack_label'; the locked feature order equals the parquet "
                "column order minus the label)",
            ),
            ("locked_at", LOCKED_AT),
            ("git_commit", "pending"),
            (
                "pre_registration_note",
                "Locked before any H3 tau-calibration or evaluation run. Changing "
                "this artifact changes the construct and requires a dated "
                "amendment per the base spec's section 12 protocol.",
            ),
            ("generated_by", "scripts/lock_fingerprint_features.py"),
        ]
    )
    payload["num_features"] = len(features)
    payload["features"] = features
    payload["moments"] = ["mean", "std", "skew", "kurtosis"]
    payload["fingerprint_dim"] = len(features) * 4
    payload["transmission_timing_subset"] = OrderedDict(
        [
            (
                "description",
                "Transmission/timing-bearing columns — the construct-validity "
                "anchor for the 'transmission fingerprint' framing (v1.10 section "
                "1.0 construct note). Reported alongside the full vector; the H3 "
                "metric uses all 45 features.",
            ),
            ("provenance", "scripts/fingerprint_feasibility.py:272-275 (edge_timing)"),
            ("num_features", len(TRANSMISSION_TIMING_FEATURES)),
            ("features", list(TRANSMISSION_TIMING_FEATURES)),
        ]
    )
    payload["hashes"] = OrderedDict(
        [
            (
                "_rule",
                "sha256 of ('\\n'.join(names) + '\\n') encoded utf-8, names in the "
                "order listed in this artifact",
            ),
            ("features_sha256", hash_feature_list(features)),
            ("timing_subset_sha256", hash_feature_list(TRANSMISSION_TIMING_FEATURES)),
            ("source_features_json_sha256", hashlib.sha256(raw).hexdigest()),
        ]
    )
    return payload


def _render(payload) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed artifact instead of writing it",
    )
    args = parser.parse_args(argv)

    payload = build_payload()
    rendered = _render(payload)

    if args.check:
        if not ARTIFACT.exists():
            print(f"FAIL: artifact missing: {ARTIFACT}", file=sys.stderr)
            return 1
        committed = ARTIFACT.read_text()
        # Compare the load-bearing content, not incidental metadata formatting.
        want = json.loads(rendered)
        have = json.loads(committed)
        for key in ("features", "moments", "fingerprint_dim", "num_features",
                    "transmission_timing_subset", "hashes"):
            if want[key] != have[key]:
                print(f"FAIL: '{key}' differs from the committed artifact", file=sys.stderr)
                return 1
        print("OK: committed feature artifact matches the current source")
        return 0

    ARTIFACT.write_text(rendered)
    print(f"written: {ARTIFACT}")
    for key, value in payload["hashes"].items():
        if key != "_rule":
            print(f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

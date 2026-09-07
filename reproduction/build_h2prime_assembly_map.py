#!/usr/bin/env python3
"""Build a hash-pinned H2-prime assembly map from local custody directories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCENARIOS = [
    "s0_clean_baseline", "s1_benign_churn_only", "s2_adaptive_switching_only",
    "s3_identity_reset_only", "s4_full_mix",
]
SEEDS = [90369, 29387, 98362, 45013, 39477, 70402, 47599, 39375, 17869, 96540]
UNIT_TEMPLATE = "{scenario}__krum_tge__persistent_optimizer__seed{seed}.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, directory = value.partition("=")
    if not separator or not label or not directory:
        raise argparse.ArgumentTypeError("source must have the form LABEL=DIRECTORY")
    return label, Path(directory)


def load_decisions(path: Path | None, labels: set[str]) -> dict[tuple[str, int], str]:
    expected = {(scenario, seed) for scenario in SCENARIOS for seed in SEEDS}
    if path is None:
        if len(labels) != 1:
            raise SystemExit("multiple --source values require --custody-sidecar")
        label = next(iter(labels))
        return {cell: label for cell in expected}

    document = json.loads(path.read_text(encoding="utf-8"))
    cells = document["cells"] if isinstance(document, dict) else document
    decisions: dict[tuple[str, int], str] = {}
    for entry in cells:
        cell = (str(entry["scenario"]).lower(), int(entry["seed"]))
        source = str(entry["source"])
        if cell in decisions:
            raise SystemExit(f"duplicate custody decision for {cell}")
        if source not in labels:
            raise SystemExit(f"unknown source label {source!r} for {cell}")
        decisions[cell] = source
    if set(decisions) != expected:
        raise SystemExit(
            f"custody sidecar is not the exact 5 x 10 matrix: "
            f"missing={sorted(expected - set(decisions))}, "
            f"extra={sorted(set(decisions) - expected)}"
        )
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", type=parse_source, required=True,
        metavar="LABEL=DIRECTORY",
        help="directory containing signal JSONL files; repeat for refill sources",
    )
    parser.add_argument(
        "--custody-sidecar", type=Path,
        help="JSON cells array assigning each (scenario, seed) to a source label",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sources = dict(args.source)
    if len(sources) != len(args.source):
        raise SystemExit("source labels must be unique")
    decisions = load_decisions(args.custody_sidecar, set(sources))
    if args.out.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.out}; use --force to replace it")

    entries = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            source = decisions[(scenario, seed)]
            filename = UNIT_TEMPLATE.format(scenario=scenario, seed=seed)
            path = sources[source] / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"missing or empty custody file for {(scenario, seed)}: {path}")
            entries.append({
                "scenario": scenario,
                "seed": seed,
                "source": source,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    metadata = {
        "builder": "reproduction/build_h2prime_assembly_map.py",
        "purpose": "cell-to-local-signal-log map for H2-prime adjudication",
        "n_cells": len(entries),
        "cells_by_source": {
            label: sum(1 for entry in entries if entry["source"] == label)
            for label in sorted(sources)
        },
        "content_read": "bytes hashed for custody; JSONL rows not parsed",
    }
    if args.custody_sidecar is not None:
        metadata["custody_sidecar_sha256"] = sha256_file(args.custody_sidecar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"_meta": metadata, "cells": entries}, indent=1) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

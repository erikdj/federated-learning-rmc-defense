"""Build the EXP-051 + EXP-053 assembly map for the H2′ adjudication.

DOWNLOAD-ONLY. This program stages the 50 confirmatory signal logs and records
where each cell came from. It never opens a log, never parses a row, and never
computes anything from log content — the § 5.1 DO-NOT-READ seal stays intact
until `scripts/adjudicate_h2prime.py` runs.

Cell → serving-prefix decisions come from the custody step, in one of two forms:

  * a JSON sidecar (preferred if the custody agent left one), shaped
    `{"cells": [{"scenario": ..., "seed": ..., "source": "EXP-051"|"EXP-053"}]}`;
  * otherwise the 50-cell assembly table in
    `results/20260812/exp051_custody/CUSTODY_REPORT.md`, parsed mechanically.

A prefix sync or wildcard merge is forbidden (EXP-053 § 2.3; the EXP-011
overwrite trap): every object is fetched from the prefix the custody report
assigns to that cell, one `aws s3 cp` per cell.

Usage:
    python scripts/build_exp051_assembly_map.py \
        --staging /path/to/staging --out /path/to/assembly_map.json
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "results" / "20260812" / "exp051_custody" / "CUSTODY_REPORT.md"
BUCKET = os.environ.get("PRAXIS_ARTIFACT_BUCKET", "")
UNIT_TEMPLATE = "{scenario}__krum_tge__persistent_optimizer__seed{seed}"
EXPECTED_CELLS = 50
SOURCES = {"EXP-051", "EXP-053"}

# | # | scenario | seed | EXP-051 array idx | serving prefix | results | signals | done |
ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([a-z0-9_]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|"
)


def parse_custody_report(path: Path) -> list[dict]:
    """Extract the 50-cell assembly table. Structure only — no metric content."""
    cells = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        _idx, scenario, seed, array_idx, prefix_cell = m.groups()
        prefix = prefix_cell.replace("*", "").strip()
        source = next((s for s in SOURCES if s in prefix), None)
        if source is None:
            raise SystemExit(f"unrecognized serving prefix in custody table: {prefix_cell!r}")
        cells.append({"scenario": scenario, "seed": int(seed),
                      "source": source, "array_idx": int(array_idx)})
    return cells


def parse_sidecar(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    cells = doc["cells"] if isinstance(doc, dict) else doc
    out = []
    for c in cells:
        source = str(c["source"]).replace("*", "").strip()
        source = next((s for s in SOURCES if s in source), None)
        if source is None:
            raise SystemExit(f"unrecognized source in sidecar: {c}")
        out.append({"scenario": str(c["scenario"]).lower(), "seed": int(c["seed"]),
                    "source": source, "array_idx": c.get("array_idx")})
    return out


def validate(cells: list[dict]) -> None:
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"custody decisions cover {len(cells)} cells, expected {EXPECTED_CELLS}")
    keys = [(c["scenario"], c["seed"]) for c in cells]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise SystemExit(f"duplicate (scenario, seed) decisions: {dupes}")
    scenarios = sorted({c["scenario"] for c in cells})
    seeds = sorted({c["seed"] for c in cells})
    if len(scenarios) != 5 or len(seeds) != 10 or len(cells) != 50:
        raise SystemExit(
            f"not the 5 × 10 cross product: {len(scenarios)} scenarios, {len(seeds)} seeds")


def s3_uri(source: str, unit: str) -> str:
    if not BUCKET:
        raise SystemExit("Set PRAXIS_ARTIFACT_BUCKET to the bucket holding the custody-selected corpus")
    return f"s3://{BUCKET}/sweeps/{source}/signals/{unit}.jsonl"


def download(uri: str, dest: Path, dry_run: bool) -> None:
    cmd = ["aws", "s3", "cp", uri, str(dest), "--only-show-errors"]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"download failed: {uri}\n{proc.stderr.strip()}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staging", type=Path, required=True,
                    help="local directory the 50 signal logs are staged into")
    ap.add_argument("--out", type=Path, required=True, help="assembly map JSON path")
    ap.add_argument("--custody-report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="JSON sidecar of cell→source decisions (takes precedence)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the plan without downloading")
    args = ap.parse_args(argv)

    if args.sidecar is not None:
        if not args.sidecar.is_file():
            raise SystemExit(f"sidecar not found: {args.sidecar}")
        cells = parse_sidecar(args.sidecar)
        decision_source = str(args.sidecar)
    else:
        if not args.custody_report.is_file():
            raise SystemExit(f"custody report not found: {args.custody_report}")
        cells = parse_custody_report(args.custody_report)
        decision_source = str(args.custody_report)
    validate(cells)

    args.staging.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in sorted(cells, key=lambda c: (c["scenario"], c["seed"])):
        unit = UNIT_TEMPLATE.format(scenario=c["scenario"], seed=c["seed"])
        uri = s3_uri(c["source"], unit)
        dest = args.staging / f"{unit}.jsonl"
        print(f"[{c['source']}] {unit}")
        download(uri, dest, args.dry_run)
        entry = {"scenario": c["scenario"], "seed": c["seed"], "source": c["source"],
                 "array_idx": c.get("array_idx"), "s3_uri": uri, "path": str(dest)}
        # A DIGEST MAY ONLY DESCRIBE BYTES THIS RUN FETCHED.
        # Round 27 made the builder emit digests in --dry-run so a planning map
        # could not masquerade as a custody map. That fix over-reached: it
        # hashed whatever happened to sit at the destination, and in --dry-run
        # `download()` never ran, so a stale file from an earlier build or a
        # foreign file dropped at the path would be digested AS IF it were the
        # registered S3 object. That is worse than no digest — it manufactures
        # custody evidence for bytes nobody fetched.
        #
        # So: digest ONLY on a real run, where download() just wrote the file.
        # A dry-run entry always declares the digest unavailable, whether or
        # not something is sitting at the path.
        if args.dry_run:
            entry["sha256"] = None
            entry["digest_unavailable"] = (
                "dry-run: download did not run, so any bytes at this path were "
                "not fetched by this build and cannot be vouched for — this map "
                "cannot be used for the sealed read")
        else:
            if not dest.is_file() or dest.stat().st_size == 0:
                raise SystemExit(f"staged file missing or empty: {dest}")
            entry["bytes"] = dest.stat().st_size
            entry["sha256"] = sha256_file(dest)
        entries.append(entry)

    doc = {
        "_meta": {
            "builder": "scripts/build_exp051_assembly_map.py",
            "purpose": "cell → local signal-log path for the H2′ P1∧P2 adjudication",
            "decision_source": decision_source,
            "bucket": BUCKET,
            "unit_template": UNIT_TEMPLATE,
            "n_cells": len(entries),
            "cells_by_source": {s: sum(1 for e in entries if e["source"] == s)
                                for s in sorted({e["source"] for e in entries})},
            "content_read": "NONE — download only; no signal-log row was parsed",
            "dry_run": bool(args.dry_run),
        },
        "cells": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(f"\n[written] {args.out}  ({len(entries)} cells, "
          f"{doc['_meta']['cells_by_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

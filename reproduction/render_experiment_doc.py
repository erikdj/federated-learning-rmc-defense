#!/usr/bin/env python3
"""Render a reproduction matrix JSON as a praxis launch document."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EXP_ID = re.compile(r"^EXP-[0-9]{3,}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--job-queue", required=True)
    parser.add_argument("--job-definition", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not EXP_ID.fullmatch(args.exp_id):
        raise SystemExit("--exp-id must have the form EXP-101")
    if not args.job_queue.strip() or any(c.isspace() for c in args.job_queue):
        raise SystemExit("--job-queue must be one non-empty resource identifier")
    if not args.job_definition.strip() or any(c.isspace() for c in args.job_definition):
        raise SystemExit("--job-definition must be one non-empty resource identifier")
    if not IMAGE_DIGEST.fullmatch(args.image_digest):
        raise SystemExit("--image-digest must be sha256 followed by 64 lowercase hex digits")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("launchable") is False:
        raise SystemExit(
            f"{args.config} is marked launchable=false ({config.get('status')}); "
            "a ratified successor config is required"
        )
    matrix = config["matrix"]
    expected = (
        len(matrix["defenses"]) * len(matrix["scenarios"]) *
        len(matrix["seeds"]) * int(matrix.get("repeats", 1))
    )
    if expected != int(config["expected_units"]):
        raise SystemExit(
            f"matrix expands to {expected} units, expected {config['expected_units']}"
        )

    slug = args.config.stem.replace("_", "-")
    frontmatter = {
        "exp_id": args.exp_id,
        "slug": slug,
        "hypothesis": config["claim"],
        "methodology_version": "public-reproduction-v1",
        "matrix": matrix,
        "run_extras": config.get("run_extras", {}),
        "batch": {
            "job_queue": args.job_queue,
            "job_definition": args.job_definition,
            "image_digest": args.image_digest,
        },
    }
    try:
        config_label = args.config.resolve().relative_to(REPO).as_posix()
    except ValueError:
        config_label = args.config.name
    body = (
        f"# {args.exp_id}: {config['claim']}\n\n"
        f"Rendered from `{config_label}` "
        f"(sha256 `{sha256_file(args.config)}`).\n"
    )
    rendered = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + body

    # Validate the exact bytes before placing them in docs/experiments. This
    # exercises the launch parser, deterministic matrix expansion, and every
    # requested strategy mapping without constructing a server or training.
    from praxis_exp.matrix_doc import parse_matrix
    from praxis_exp.units import expand_matrix
    from scripts.run_phase4_flower import build_strategy_for_config

    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / f"{args.exp_id}.md"
        candidate.write_text(rendered, encoding="utf-8")
        parsed = parse_matrix(candidate)
        units = expand_matrix(
            parsed.defenses, parsed.scenarios, parsed.seeds, parsed.mode,
            parsed.max_per_client, parsed.rounds, parsed.repeats,
        )
        if len(units) != expected:
            raise SystemExit(f"validated expansion produced {len(units)} units, expected {expected}")
        for defense in parsed.defenses:
            build_strategy_for_config(defense)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

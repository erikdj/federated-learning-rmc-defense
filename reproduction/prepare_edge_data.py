#!/usr/bin/env python3
"""Run the historical Edge-IIoT CSV-to-ten-client preprocessing with explicit paths."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Edge-IIoTset directory containing 'Normal traffic' and 'Attack traffic'",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/edge_full"),
        help="destination for client_0.parquet through client_9.parquet",
    )
    args = parser.parse_args()

    normal = args.dataset_root / "Normal traffic"
    attack = args.dataset_root / "Attack traffic"
    if not normal.is_dir() or not attack.is_dir():
        raise SystemExit(
            "dataset root must contain the exact 'Normal traffic' and "
            "'Attack traffic' directories"
        )

    import scripts.process_edge_full as pipeline

    pipeline.DATASET_ROOT = str(args.dataset_root.resolve())
    pipeline.NORMAL_DIR = str(normal.resolve())
    pipeline.ATTACK_DIR = str(attack.resolve())
    pipeline.OUTPUT_DIR = str(args.output_dir.resolve())
    pipeline.process()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

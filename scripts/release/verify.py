#!/usr/bin/env python3
"""Verify release payload checksums and publication boundaries, without dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

PRIVATE_PARTS = {'.release-work', '.superpowers', '.planning', '.claude', '.codex', '.git', '.venv', '__pycache__'}
PRIVATE_NAMES = {'AGENTS.md', 'CLAUDE.md', 'SKILL.md', 'local_defaults.py'}
OFFICE_SUFFIXES = {'.docx', '.doc', '.pptx', '.xlsx', '.pdf'}


def forbidden(path: str) -> bool:
    relative = PurePosixPath(path)
    return (
        not path or relative.is_absolute() or '..' in relative.parts
        or '\\' in path or bool(set(relative.parts) & PRIVATE_PARTS)
        or relative.name in PRIVATE_NAMES or relative.suffix.lower() in OFFICE_SUFFIXES
        or relative.name.startswith('.env') and relative.name != '.env.example'
    )


def verify(root: Path) -> list[str]:
    root = root.resolve()
    errors = []
    try:
        manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
        if manifest.get('format') != 1 or not isinstance(manifest.get('files'), list):
            return ['unsupported release manifest format']
    except (OSError, ValueError) as exc:
        return [f'cannot read release manifest: {exc}']
    seen = set()
    for item in manifest['files']:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            errors.append('malformed manifest entry')
            continue
        name = item['path']
        if forbidden(name):
            errors.append(f'forbidden release path: {name}')
            continue
        if name in seen:
            errors.append(f'duplicate release path: {name}')
            continue
        seen.add(name)
        path = root / name
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root):
            errors.append(f'symlink in release path: {name}')
        elif not path.is_file():
            errors.append(f'missing release file: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item.get('sha256'):
            errors.append(f'checksum mismatch: {name}')
    if (root / '.git').exists():
        try:
            tracked = set(subprocess.check_output(
                ['git', '-C', str(root), 'ls-files', '-z'], text=True
            ).split('\0')) - {''}
            for extra in sorted(tracked - seen - {'RELEASE_MANIFEST.json'}):
                errors.append(f'unlisted tracked file: {extra}')
        except subprocess.CalledProcessError:
            errors.append('cannot enumerate tracked files')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = verify(args.root)
    if errors:
        for error in errors:
            print(f'FAIL: {error}')
        return 1
    manifest = json.loads((args.root / 'RELEASE_MANIFEST.json').read_text())
    print(f"Verified {len(manifest['files'])} release files; checksums and path boundaries match.")
    print('This verifies the distributed payload, not raw-data experiment reproduction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

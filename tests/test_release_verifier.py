"""The release verifier must reject altered payloads and unsafe manifests."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).resolve().parents[1] / 'scripts/release/verify.py'
    spec = importlib.util.spec_from_file_location('release_verify', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_for(root, content=b'original\n'):
    (root / 'payload.txt').write_bytes(content)
    manifest = {'format': 1, 'files': [{'path': 'payload.txt', 'sha256': hashlib.sha256(content).hexdigest()}]}
    (root / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest))


def test_clean_release_and_tampered_payload(tmp_path):
    verify = load_verifier().verify
    manifest_for(tmp_path)
    assert verify(tmp_path) == []
    (tmp_path / 'payload.txt').write_bytes(b'changed\n')
    assert any('checksum' in error for error in verify(tmp_path))


def test_missing_payload_is_rejected(tmp_path):
    verify = load_verifier().verify
    manifest_for(tmp_path)
    (tmp_path / 'payload.txt').unlink()
    assert any('missing' in error for error in verify(tmp_path))


@pytest.mark.parametrize('path', ['../outside', '/absolute', '.release-work/private.md', '.claude/session.json', 'docs/draft.docx'])
def test_unsafe_or_private_manifest_path_is_rejected(tmp_path, path):
    verify = load_verifier().verify
    (tmp_path / 'RELEASE_MANIFEST.json').write_text(json.dumps({'format': 1, 'files': [{'path': path, 'sha256': '0' * 64}]}))
    assert any('forbidden' in error for error in verify(tmp_path))


def test_symlink_escape_is_rejected(tmp_path):
    verify = load_verifier().verify
    manifest_for(tmp_path)
    target = tmp_path / 'payload.txt'
    target.unlink()
    target.symlink_to('/etc/hosts')
    assert any('symlink' in error for error in verify(tmp_path))


def test_extra_tracked_file_is_rejected(tmp_path):
    import subprocess
    verify = load_verifier().verify
    manifest_for(tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'payload.txt', 'RELEASE_MANIFEST.json'], check=True)
    assert verify(tmp_path) == []
    (tmp_path / 'unreviewed.txt').write_text('extra')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'unreviewed.txt'], check=True)
    assert any('unlisted tracked' in error for error in verify(tmp_path))

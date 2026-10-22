"""Verify published source, then require separately supplied model assets for deployment."""
import argparse
import hashlib
import json
from pathlib import Path


def checked_path(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Unsafe package member: ' + rel)
    return path


def check_file(path, expected, rel):
    if not path.is_file():
        raise ValueError('Missing package member: ' + rel)
    if (hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']
            or path.stat().st_size != expected['bytes']):
        raise ValueError('Package checksum mismatch: ' + rel)


def verify(root=None, require_assets=True):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    manifest = json.loads((root / 'PACKAGE.json').read_text(encoding='utf-8'))
    files = dict(manifest['files'])
    for rel, expected in files.items():
        check_file(checked_path(root, rel), expected, rel)
    missing = []
    for rel, expected in manifest['assets'].items():
        path = checked_path(root, rel)
        if path.exists():
            check_file(path, expected, rel)
            files[rel] = expected
        else:
            missing.append(rel)
    if require_assets and missing:
        raise ValueError('Model assets are supplied separately. Run prepare_assets.py --source LOCAL_C2_BUNDLE first. Missing: '
                         + ', '.join(missing))
    actual = {p.relative_to(root / 'payload').as_posix() for p in (root / 'payload').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    expected = {rel.removeprefix('payload/') for rel in files if rel.startswith('payload/')}
    if actual != expected:
        raise ValueError('Unexpected or missing files inside payload')
    result = dict(manifest, files=files)
    print(('BUNDLE' if not missing else 'SOURCE') + ' CHECK PASSED: ' + str(len(files)) + ' files')
    return result


def verify_installation(task, manifest):
    task = Path(task).resolve()
    count = 0
    for rel, expected in manifest['files'].items():
        if not rel.startswith('payload/participant/'):
            continue
        dest = checked_path(task, rel.removeprefix('payload/'))
        if not dest.is_relative_to(task / 'participant'):
            raise ValueError('Unsafe installed adapter file: ' + rel)
        check_file(dest, expected, rel)
        count += 1
    print('INSTALLED ADAPTER CHECK PASSED: ' + str(count) + ' files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true', help='Verify Git source before separately providing model assets')
    args = parser.parse_args()
    verify(require_assets=not args.source_only)

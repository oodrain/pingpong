"""Copy hash-checked assets from an existing authorized local C2 bundle; never download weights."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from verify_bundle import checked_path, check_file, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing complete C2 adapter, its participant directory, or its weights directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = verify(root, require_assets=False)
    source = args.source.resolve()
    if not source.is_dir():
        parser.error('The local source directory does not exist')
    copies = []
    # Validate every source and existing destination before copying anything.
    for rel, expected in manifest['assets'].items():
        name = Path(rel).name
        candidates = (source / rel, source / 'weights' / name, source / name)
        matches = [p for p in candidates if p.is_file()]
        if not matches:
            parser.error('Missing authorized local asset: ' + name)
        origin = matches[0]
        check_file(origin, expected, name)
        destination = checked_path(root.resolve(), rel)
        if destination.exists():
            check_file(destination, expected, rel)
        copies.append((origin, destination))
    for origin, destination in copies:
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            shutil.copyfile(origin, temp_path)
            temp_path.replace(destination)
        finally:
            temp_path.unlink(missing_ok=True)
    verify(root)
    print('LOCAL MODEL ASSETS READY; no training or model parameter changes performed.')


if __name__ == '__main__':
    main()

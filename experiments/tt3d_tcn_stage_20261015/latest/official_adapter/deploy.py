"""Back up affected files, then install only the participant algorithm payload."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from verify_bundle import verify, verify_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path, help='Existing official pingpang directory')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = verify(package)
    task = args.task_dir.resolve()
    for required in ('core/types.py', 'run.py', 'integrity.py', '.integrity.json', 'participant/__init__.py'):
        if not (task / required).is_file():
            parser.error('Not an existing official task directory: missing ' + required)
    subprocess.run([sys.executable, str(task / 'integrity.py'), '--verify'], cwd=task, check=True)
    participant = (task / 'participant').resolve()
    if participant.parent != task:
        parser.error('participant must be a normal directory inside the official task')
    payload = package / 'payload/participant'
    files = []
    for rel in manifest['files']:
        if not rel.startswith('payload/participant/'):
            continue
        item = Path(rel.removeprefix('payload/participant/'))
        dest = (participant / item).resolve()
        if not dest.is_relative_to(participant):
            parser.error('Unsafe installation path')
        if dest.exists() and not dest.is_file():
            parser.error('Expected an algorithm file, found a directory: ' + str(item))
        if item.parts[0] not in ('solution.py', 'tcn_runtime', 'weights', 'configs'):
            parser.error('Unexpected algorithm payload path')
        files.append((item, payload / item, dest))
    backup = task.parent / 'outputs/tcn_adapter_backups' / (
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    records = []
    # Complete all backups before replacing any algorithm file.
    for rel, source, dest in files:
        same = dest.is_file() and hashlib.sha256(dest.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
        existed = dest.is_file()
        if existed and not same:
            target = backup / 'participant' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, target)
        records.append({'path': ('participant' / rel).as_posix(), 'existed': existed,
                        'unchanged': same, 'backed_up': existed and not same})
    (backup / 'backup_manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    for (_, source, dest), record in zip(files, records):
        if not record['unchanged']:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    subprocess.run([sys.executable, str(task / 'integrity.py'), '--verify'], cwd=task, check=True)
    verify_installation(task, manifest)
    print('DEPLOY PASSED')
    print('Algorithm backup: ' + str(backup))
    print('No protected framework files were modified.')


if __name__ == '__main__':
    main()

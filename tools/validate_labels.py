"""Tool-independent label validation and export. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {'A', 'B', 'C', 'D'}
ROUNDS = {'initial', 'review'}
NON_TABLE = {'net_contact', 'floor_contact', 'racket_contact', 'other'}
FILES = {'metadata.json', 'events.jsonl', 'uncertain.jsonl', 'non_table_events.jsonl', 'README.md'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def label_digest(path):
    # Git/Windows may convert CRLF to LF. Bind reviews to canonical LF bytes.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_json(text):
    def invalid_constant(value):
        raise ValueError(f'nonstandard JSON number: {value}')
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def read_json(path):
    return parse_json(Path(path).read_text(encoding='utf-8'))


def read_records(path, required=True):
    if not path.exists():
        require(not required, f'missing file: {path.name}')
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = parse_json(line)
            require(isinstance(row, dict), 'each line must be a JSON object')
            rows.append(row)
        except (ValueError, TypeError) as error:
            raise ValueError(f'{path.name}:{line_no}: {error}') from error
    return rows


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text_value(value):
    return isinstance(value, str) and bool(value.strip())


def validate_data(meta, events, uncertain, non_table):
    require(isinstance(meta, dict), 'metadata must be an object')
    require(type(meta.get('schema_version')) is int and meta['schema_version'] in (1, 2), 'schema_version must be integer 1 or 2')
    video_id, member, phase = meta.get('video_id'), meta.get('annotator'), meta.get('round')
    require(isinstance(video_id, str) and re.fullmatch(r'\d{2}_\d{3}', video_id), 'invalid video_id')
    require(member in MEMBERS, 'annotator must be A/B/C/D')
    require(phase in ROUNDS, 'round must be initial/review')
    for key in ('width', 'height', 'n_frames', 'label_version'):
        require(type(meta.get(key)) is int and meta[key] > 0, f'{key} must be a positive integer')
    require(finite_number(meta.get('fps')) and meta['fps'] > 0, 'fps must be finite and positive')
    for key in ('video_sha256', 'annotation_sha256'):
        require(isinstance(meta.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', meta[key]), f'{key} must be lowercase SHA256')
    require(type(meta.get('completed')) is bool, 'completed must be boolean')
    require(meta.get('source_type') in ('manual', 'imported'), 'source_type must be manual/imported')
    require(text_value(meta.get('tool')), 'tool is required')
    require(isinstance(meta.get('no_events_reason'), str), 'no_events_reason must be a string')
    try:
        updated = datetime.fromisoformat(meta['updated_at'].replace('Z', '+00:00'))
        require(updated.tzinfo is not None, 'updated_at must include a timezone')
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError('updated_at must be an ISO8601 time with timezone') from error
    n = meta['n_frames']
    ranges = meta.get('checked_ranges')
    require(isinstance(ranges, list), 'checked_ranges must be an array')
    previous_end = -2
    for interval in ranges:
        require(isinstance(interval, list) and len(interval) == 2 and all(type(v) is int for v in interval), 'range must contain two integer endpoints')
        start, end = interval
        require(0 <= start <= end < n and start > previous_end + 1, 'ranges must be in bounds, sorted, disjoint and merged')
        previous_end = end
    if meta['completed']:
        require(ranges == [[0, n - 1]], 'completed requires checked_ranges covering every frame')
        require(bool(events) or text_value(meta['no_events_reason']), 'completed empty events require no_events_reason')
    if meta['schema_version'] == 2:
        require(meta.get('completion_basis') == ('manual_attestation' if meta['completed'] else None),
                'completion_basis must match the manual completion declaration')
        require('displayed_ranges' in meta, 'v2 requires displayed_ranges (null if unavailable)')
        displayed = meta['displayed_ranges']
        if displayed is not None:
            require(isinstance(displayed, list), 'displayed_ranges must be an array or null')
            end_before = -2
            for interval in displayed:
                require(isinstance(interval, list) and len(interval) == 2
                        and all(type(v) is int for v in interval), 'displayed range needs integer endpoints')
                start, end = interval
                require(0 <= start <= end < n and start > end_before + 1, 'invalid displayed_ranges')
                end_before = end
    reference = meta.get('review_of')
    if phase == 'initial':
        require(reference is None, 'initial review_of must be null')
    else:
        require(isinstance(reference, dict) and reference.get('annotator') in MEMBERS and reference['annotator'] != member, 'review must reference a different initial annotator')
        require(type(reference.get('label_version')) is int and reference['label_version'] > 0, 'review_of label_version is required')
        for key in ('metadata_sha256', 'events_sha256', 'uncertain_sha256'):
            require(isinstance(reference.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', reference[key]), f'review_of.{key} must be SHA256')
    ids, seen_events, seen_non_table, seen_ranges = set(), set(), set(), set()

    def common(row):
        require(isinstance(row, dict), 'record must be an object')
        require(text_value(row.get('event_id')) and row['event_id'] not in ids, 'event_id must be unique within this submission')
        ids.add(row['event_id'])
        for key, expected in (('video_id', video_id), ('annotator', member), ('round', phase)):
            require(row.get(key) == expected, f'record {key} disagrees with metadata')

    def frame(row):
        require(type(row.get('frame_id')) is int and 0 <= row['frame_id'] < n, 'frame_id must be a 0-based in-bounds integer')

    def coordinates(row, optional=False):
        require('x' in row and 'y' in row, 'x and y are required')
        x, y = row['x'], row['y']
        if optional and x is None and y is None:
            return
        require(finite_number(x) and finite_number(y) and 0 <= x < meta['width'] and 0 <= y < meta['height'], 'coordinates must be finite original-image pixels in bounds')

    for row in events:
        common(row)
        frame(row)
        coordinates(row)
        require(row.get('event_type') == 'table_bounce' and row.get('review_state') == 'provisional', 'table events must be table_bounce/provisional')
        require(row['frame_id'] not in seen_events, 'duplicate table event frame')
        seen_events.add(row['frame_id'])
    require([r['frame_id'] for r in events] == sorted(seen_events), 'events must be sorted by frame_id')
    for row in uncertain:
        common(row)
        start, end = row.get('frame_start'), row.get('frame_end')
        require(type(start) is int and type(end) is int and 0 <= start <= end < n, 'uncertain endpoints must be in bounds, inclusive integers')
        require(text_value(row.get('reason')) and row.get('review_state') == 'uncertain', 'uncertain records require reason and uncertain state')
        require('frame_id' not in row and 'x' not in row and 'y' not in row, 'uncertain records use intervals, not invented exact coordinates')
        require((start, end) not in seen_ranges, 'duplicate uncertainty interval')
        seen_ranges.add((start, end))
    require([(r['frame_start'], r['frame_end']) for r in uncertain] == sorted(seen_ranges), 'uncertain records must be sorted by interval')
    for row in non_table:
        common(row)
        frame(row)
        coordinates(row, optional=True)
        require(row.get('event_type') in NON_TABLE and row.get('review_state') == 'provisional' and text_value(row.get('reason')), 'non-table records require allowed type, provisional state and reason')
        key = row['frame_id'], row['event_type']
        require(key not in seen_non_table, 'duplicate non-table frame/type')
        seen_non_table.add(key)
    require([(r['frame_id'], r['event_type']) for r in non_table] == sorted(seen_non_table), 'non-table records must be sorted by frame/type')
    return meta


def validate_folder(folder):
    require(folder.is_dir() and not folder.is_symlink(), f'not a regular directory: {folder}')
    for path in folder.iterdir():
        require(path.is_file() and not path.is_symlink() and path.name in FILES, f'unexpected submission file: {path.name}')
    meta = read_json(folder / 'metadata.json')
    validate_data(meta, read_records(folder / 'events.jsonl'), read_records(folder / 'uncertain.jsonl'), read_records(folder / 'non_table_events.jsonl', False))
    members = {meta['annotator']}
    if meta['round'] == 'review':
        # Exchanges are reviewer-grouped; the repository may group by initial owner.
        # The actual reviewer always remains metadata.annotator and must differ.
        members.add(meta['review_of']['annotator'])
    require(folder.name == meta['round'] and folder.parent.name == meta['video_id']
            and folder.parent.parent.name in members,
            'directory must be MEMBER/VIDEO/ROUND; review MEMBER must be reviewer or initial source annotator')
    return meta


def review_reference(source):
    meta = validate_folder(source)
    require(meta['round'] == 'initial' and meta['completed'], 'reference requires a completed initial submission')
    return {'annotator': meta['annotator'], 'label_version': meta['label_version'],
            **{key: label_digest(source / name) for key, name in (
                ('metadata_sha256', 'metadata.json'), ('events_sha256', 'events.jsonl'), ('uncertain_sha256', 'uncertain.jsonl'))}}


def validate_tree(root, tasks_path=None, data_root=None):
    require(root.is_dir(), f'missing labels directory: {root}')
    require(not root.is_symlink(), 'labels root must not be a symlink')
    tasks = {}
    require(data_root is None or tasks_path is not None, '--verify-inputs requires an explicit --tasks input mapping')
    if tasks_path:
        tasks = {t['video_id']: t for t in read_json(tasks_path)['tasks']}
    folders = []
    for member in root.iterdir():
        if member.name == 'README.md':
            require(member.is_file() and not member.is_symlink(), 'labels README must be a regular file')
            continue
        require(member.name in MEMBERS and member.is_dir() and not member.is_symlink(), f'unexpected labels entry: {member.name}')
        for video in member.iterdir():
            require(video.is_dir() and not video.is_symlink(), f'unexpected member entry: {video.name}')
            for phase in video.iterdir():
                require(phase.is_dir() and not phase.is_symlink() and phase.name in ROUNDS, f'unexpected video entry: {phase.name}')
                folders.append(phase)
    metas = {}
    warnings = []
    for folder in folders:
        try:
            meta = validate_folder(folder)
            if tasks_path:
                task = tasks.get(meta['video_id'])
                require(task is not None, f'video not in task list: {meta["video_id"]}')
                require(meta['annotator'] == task['annotator' if meta['round'] == 'initial' else 'reviewer'], 'member disagrees with fixed task assignment')
                if meta['round'] == 'review':
                    require(meta['review_of']['annotator'] == task['annotator'], 'review references wrong initial member')
                if data_root is not None:
                    for path_key, hash_key in (('video', 'video_sha256'), ('annotation', 'annotation_sha256')):
                        input_path = Path(task[path_key])
                        if not input_path.is_absolute():
                            input_path = data_root / input_path
                        require(digest(input_path) == meta[hash_key], f'{path_key} hash disagrees with local input')
            metas[folder] = meta
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise ValueError(f'{folder}: {error}') from error
    for folder, meta in metas.items():
        if meta['round'] != 'review':
            continue
        ref = meta['review_of']
        source = root / ref['annotator'] / meta['video_id'] / 'initial'
        require(source in metas, f'review initial source is not included: {source}')
        original = metas[source]
        require(original['completed'], 'review initial source is not complete')
        for key in ('video_sha256', 'annotation_sha256', 'width', 'height', 'n_frames', 'fps'):
            require(meta[key] == original[key], f'review source {key} differs')
        same_version = ref['label_version'] == original['label_version']
        same_files = all(ref[key] == label_digest(source / filename) for key, filename in (
            ('metadata_sha256', 'metadata.json'), ('events_sha256', 'events.jsonl'), ('uncertain_sha256', 'uncertain.jsonl')))
        if not same_version or not same_files:
            warnings.append(f'{folder}: STALE_REVIEW; initial changed. Keep history, re-review before comparing.')
    return {'submissions': len(metas), 'completed': sum(m['completed'] for m in metas.values()), 'warnings': warnings}


def merged_ranges(frames, n):
    values = sorted(set(frames))
    require(all(type(v) is int and 0 <= v < n for v in values), 'invalid checked frame in session')
    result = []
    for value in values:
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def move_folder(source, target):
    """Retry only transient Windows sharing/permission failures, never other I/O errors."""
    for attempt in range(5):
        try:
            return Path(source).rename(target)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.03 * (attempt + 1))


@contextmanager
def export_lock(output_root, member, video_id, phase):
    """Keep concurrent exports serialized without adding files to the labels tree."""
    key = hashlib.sha256(str((Path(output_root) / member / video_id / phase).resolve()).encode()).hexdigest()
    folder = Path(output_root).resolve().parent / '.annotation-export-locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (key + '.lock')).open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('另一个进程正在导出同一份结果，请稍后重试。') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def export_session(workspace, output_root, phase, version=None, review_source=None, *, schema_version=1, backup_root=None):
    state = read_json(Path(workspace) / 'session.json')
    member = state.get('participants', {}).get(phase)
    require(member in MEMBERS and phase in ROUNDS, 'invalid export member/round')
    require(re.fullmatch(r'\d{2}_\d{3}', state['video']['video_id']) is not None, 'invalid video_id')
    with export_lock(output_root, member, state['video']['video_id'], phase):
        return _export_session(workspace, output_root, phase, version, review_source,
                               schema_version=schema_version, backup_root=backup_root)


def _export_session(workspace, output_root, phase, version=None, review_source=None, *, schema_version=1, backup_root=None):
    """Export a complete folder transaction; None version enables idempotent auto-versioning."""
    workspace, output_root = Path(workspace), Path(output_root)
    require(version is None or (type(version) is int and version > 0), 'label_version must be a positive integer')
    state = read_json(workspace / 'session.json')
    member = state.get('participants', {}).get(phase)
    require(member in MEMBERS, 'session participant must be registered as A/B/C/D')
    progress = state['rounds'][phase]
    video = state['video']
    inspected_all = progress.get('inspected_all_attested') is True or len(set(progress.get('reviewed_frames', []))) == video['n_frames']
    completed = progress.get('completed_at') is not None and (inspected_all or (
        schema_version == 2 and state.get('completion_policy') == 'manual_attestation'))
    # Playing through alone does not count as manual inspection.
    checked = [[0, video['n_frames'] - 1]] if completed else merged_ranges(progress.get('reviewed_frames', []), video['n_frames'])
    meta = {
        'schema_version': schema_version, 'video_id': video['video_id'], 'annotator': member, 'round': phase,
        **{key: video[key] for key in ('width', 'height', 'n_frames', 'fps')},
        **{key: state['inputs'][key] for key in ('video_sha256', 'annotation_sha256')},
        'completed': completed, 'checked_ranges': checked,
        'no_events_reason': progress.get('no_events_reason', ''),
        'label_version': version or 1, 'updated_at': datetime.now(timezone.utc).isoformat(),
        'tool': 'team_annotation/session-export-v1',
        'source_type': 'imported' if phase == 'initial' and state.get('imported_from') else 'manual',
        'review_of': None,
    }
    imported_meta = state.get('standard_metadata', {}).get(phase)
    if imported_meta:
        meta = {**imported_meta, **meta}
        meta['source_type'] = imported_meta['source_type']
        meta['tool'] = imported_meta['tool']
        if 'note' in imported_meta:
            meta['note'] = imported_meta['note']
    elif schema_version == 2:
        meta['tool'] = 'local_annotation/v2'
    if schema_version == 2:
        meta['completion_basis'] = 'manual_attestation' if completed else None
        meta['displayed_ranges'] = (None if progress.get('displayed_unavailable') else
                                    merged_ranges(progress.get('seen_frames', []), video['n_frames']))
    if state.get('imported_from'):
        meta['note'] = 'Contains imported pilot evidence; current member is responsible for this submission, not necessarily the original observer.'
    if phase == 'review':
        require(review_source is not None, 'review export requires --review-of pointing to standard initial folder')
        original = validate_folder(review_source)
        require(original['round'] == 'initial' and original['completed'] and original['video_id'] == video['video_id'], 'review-of must be completed initial for same video')
        require(original['annotator'] != member, 'reviewer must differ from initial annotator')
        for key in ('video_sha256', 'annotation_sha256', 'width', 'height', 'n_frames', 'fps'):
            require(meta[key] == original[key], f'review source {key} differs')
        snapshot = state.get('events', {}).get('initial', [])
        standard_events = read_records(review_source / 'events.jsonl')
        core_keys = ('frame_id', 'x', 'y')
        require([{k: e[k] for k in core_keys} for e in snapshot] == [{k: e[k] for k in core_keys} for e in standard_events], 'session initial snapshot differs from requested review source')
        uncertainty_keys = ('frame_start', 'frame_end', 'reason')
        snapshot_uncertain = sorted([{k: e[k] for k in uncertainty_keys} for e in state.get('uncertain', []) if e['round'] == 'initial'], key=lambda e: (e['frame_start'], e['frame_end']))
        standard_uncertain = [{k: e[k] for k in uncertainty_keys} for e in read_records(review_source / 'uncertain.jsonl')]
        require(snapshot_uncertain == standard_uncertain, 'session uncertainty snapshot differs from requested review source')
        meta['review_of'] = review_reference(review_source)
        bound = state.get('standard_review_reference')
        require(bound is None or bound == meta['review_of'], 'review source changed; keep the bound initial snapshot')

    def normalize(row, kind):
        record = dict(row)
        old_member = record.get('annotator')
        if old_member and old_member != member:
            record['source_annotator'] = old_member
        record.update(annotator=member, round=phase, video_id=video['video_id'])
        new_state = 'uncertain' if kind == 'uncertain' else 'provisional'
        if record.get('review_state') and record['review_state'] != new_state:
            record['source_review_state'] = record['review_state']
        record['review_state'] = new_state
        if kind == 'non_table':
            record.setdefault('x', None)
            record.setdefault('y', None)
            record['reason'] = record.get('reason') or record.get('notes') or 'Legacy interference record; check original provenance.'
        return record

    events = sorted([normalize(row, 'event') for row in state['events'][phase]], key=lambda row: row['frame_id'])
    uncertain = sorted([normalize(row, 'uncertain') for row in state.get('uncertain', []) if row['round'] == phase], key=lambda row: (row['frame_start'], row['frame_end']))
    non_table = sorted([normalize(row, 'non_table') for row in state.get('non_table', []) if row['round'] == phase], key=lambda row: (row['frame_id'], row['event_type']))
    target = output_root / member / video['video_id'] / phase
    require(target.resolve().is_relative_to(output_root.resolve()), 'export target escapes output root')
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if target.exists():
        previous = validate_folder(target)
        for key in ('video_sha256', 'annotation_sha256'):
            require(previous[key] == meta[key], 'export target belongs to different input; select another output root')
        if version is None:
            ignore = {'label_version', 'updated_at'}
            same = ({k: v for k, v in previous.items() if k not in ignore}
                    == {k: v for k, v in meta.items() if k not in ignore})
            same = same and all(read_records(target / f'{name}.jsonl', name != 'non_table_events') == rows
                                for name, rows in (('events', events), ('uncertain', uncertain), ('non_table_events', non_table)))
            baseline_version = (imported_meta or {}).get('label_version', 0)
            if same and previous['label_version'] >= baseline_version:
                return target
            meta['label_version'] = (baseline_version if same else
                                     max(previous['label_version'], baseline_version) + 1)
        else:
            require(version > previous['label_version'], 'export exists: increase --label-version for an update')
    elif version is None and imported_meta:
        ignore = {'label_version', 'updated_at'}
        original_rows = state.get('standard_records', {}).get(phase, {})
        same = ({k: v for k, v in imported_meta.items() if k not in ignore}
                == {k: v for k, v in meta.items() if k not in ignore}
                and original_rows == {'events': events, 'uncertain': uncertain, 'non_table_events': non_table})
        meta['label_version'] = imported_meta['label_version'] + (0 if same else 1)
        if same:
            meta['updated_at'] = imported_meta['updated_at']
    validate_data(meta, events, uncertain, non_table)
    staging = Path(tempfile.mkdtemp(prefix='submission-', dir=target.parent))
    rollback = None
    try:
        (staging / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        for name, rows in (('events', events), ('uncertain', uncertain), ('non_table_events', non_table)):
            (staging / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
        if target.exists():
            backup = Path(backup_root or ROOT / 'outputs/submission_backups') / member / video['video_id'] / phase / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, backup)
            # Swap whole folders, rather than risking mixed-version files on failure.
            rollback = output_root.resolve().parent / '.annotation-export-locks' / (staging.name + '-previous')
            move_folder(target, rollback)
            try:
                move_folder(staging, target)
            except OSError:
                move_folder(rollback, target)
                raise
            try:
                shutil.rmtree(rollback)
            except OSError:
                pass  # Outside the labels tree; the permanent version backup is also retained.
            rollback = None
        else:
            move_folder(staging, target)
        return target
    finally:
        if staging.exists():
            require(staging.resolve().parent == target.parent.resolve() and staging.name.startswith('submission-'), 'unsafe staging cleanup path')
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('validate')
    check.add_argument('root', nargs='?', type=Path, default=ROOT / 'labels')
    check.add_argument('--tasks', type=Path, help='Optional fixed task assignments; required with --verify-inputs')
    check.add_argument('--require-current-reviews', action='store_true')
    check.add_argument('--verify-inputs', action='store_true', help='Also hash local input video and object JSON; needs assigned data')
    check.add_argument('--data-root', type=Path, default=ROOT)
    export = sub.add_parser('export')
    export.add_argument('--workspace', required=True, type=Path)
    export.add_argument('--round', choices=sorted(ROUNDS), default='initial')
    export.add_argument('--output-root', type=Path, default=ROOT / 'labels')
    export.add_argument('--label-version', type=int, default=1)
    export.add_argument('--review-of', type=Path)
    export.add_argument('--schema-version', type=int, choices=(1, 2), default=1)
    export.add_argument('--auto-version', action='store_true', help='Do not rewrite unchanged results; increment changed versions')
    reference = sub.add_parser('reference', help='Print review_of object for a custom annotation tool')
    reference.add_argument('source', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            report = validate_tree(args.root, args.tasks, args.data_root if args.verify_inputs else None)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            require(not (args.require_current_reviews and report['warnings']), 'stale reviews exist; re-review before using final labels')
            print('PASS: format checked' + (' and assignments checked' if args.tasks else '') + '; annotation accuracy still requires human review.')
        elif args.command == 'reference':
            print(json.dumps(review_reference(args.source), ensure_ascii=True, indent=2))
        else:
            path = export_session(args.workspace, args.output_root, args.round,
                                  None if args.auto_version else args.label_version, args.review_of,
                                  schema_version=args.schema_version)
            print(str(path))
            print('Exported. Run validate before committing.')
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

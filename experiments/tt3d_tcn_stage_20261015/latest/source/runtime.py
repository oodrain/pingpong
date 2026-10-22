"""Portable inference for the retained control; no annotation reads."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .model import load, predict
from .observation_features import augment
from .decoder import original_control
from .frame_features import decode_frame_events

SCHEMA = 'retained_tcn_progress_v1'


def export_control(source, output):
    """Copy tensors/configuration only, excluding local provenance metadata."""
    value = torch.load(source, weights_only=True, map_location='cpu')
    while 'checkpoints' not in value:
        for key in ('selected', 'selected_variant'):
            if key in value and value[key] != 'control':
                raise ValueError('Only the retained control can be exported')
        value = value['control']
    if value.get('policy') != 'fps_frame_agreement':
        raise ValueError('Unexpected frozen admission policy')
    states = {}
    for branch in ('calibrated_event', 'fps_context'):
        states[branch] = []
        for checkpoint in value['checkpoints'][branch]:
            states[branch].append({
                'architecture': dict(checkpoint['architecture']),
                'state_dict': checkpoint['state_dict'],
                'normalization': {k: checkpoint['normalization'][k] for k in ('mean', 'scale')},
            })
        if len(states[branch]) != 3:
            raise ValueError('Expected the frozen three-seed ensemble')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Choose a new model output file')
    torch.save({'schema': SCHEMA, 'checkpoints': states}, output)


def load_asset(path):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if checkpoint['schema'] != SCHEMA:
        raise ValueError('Unsupported model schema')
    return {branch: [(load(s), s['normalization']) for s in states]
            for branch, states in checkpoint['checkpoints'].items()}


def infer(models, features, tracks):
    features = np.asarray(features, np.float32)
    if features.shape != (len(tracks), 51) or not np.isfinite(features).all():
        raise ValueError('Expected the unchanged automatic 51-feature contract')
    if [r['frame_id'] for r in tracks] != list(range(len(tracks))):
        raise ValueError('Track IDs must be contiguous decoded frame IDs')
    scores = {branch: np.mean([predict(model, augment(features, tracks, branch), stats)
              for model, stats in states], axis=0) for branch, states in models.items()}
    points, admission = original_control(scores['calibrated_event'], scores['fps_context'], tracks)
    hits = decode_frame_events(scores['calibrated_event'][:, 1], tracks, .5, .12)[0]
    return points, hits, scores, admission


def infer_with_table_range(models, features, tracks, raw_table_proposals, image_size):
    """Unchanged neural inference followed by the whole-video spatial gate."""
    from .table_range import constrain_landings
    points, hits, scores, admission = infer(models, features, tracks)
    points, table_range = constrain_landings(points, raw_table_proposals, image_size)
    return points, hits, scores, {'admission': admission, 'table_range': table_range}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')

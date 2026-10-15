"""Enforce the frozen source split for future training and evaluation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'outputs/data_split_audit_20261014'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_protocol(folder=DEFAULT):
    folder=Path(folder)
    freeze=json.loads((folder/'freeze.json').read_text(encoding='utf-8-sig'))
    for name,expected in freeze['files'].items():
        if sha(folder/name)!=expected:raise ValueError('Frozen dataset protocol changed: '+name)
    manifest=json.loads((folder/'split_manifest.json').read_text(encoding='utf-8-sig'))
    ids=[r['video_id'] for r in manifest['videos']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate video identity')
    groups={split:{r['source_group'] for r in manifest['videos'] if r['split']==split}
            for split in ('train','validation_reserved')}
    if groups['train']&groups['validation_reserved']:raise ValueError('Source recordings cross the split')
    return manifest


def select(manifest,video_ids,purpose):
    allowed={'training':{'train'},'threshold_selection':{'train'},
             'validation':{'validation_reserved'},'public_regression':{'public_regression'}}
    if purpose not in allowed:raise ValueError('Unknown data-use purpose')
    if not video_ids or len(video_ids)!=len(set(video_ids)):raise ValueError('Empty/duplicate selection')
    records={r['video_id']:r for r in manifest['videos']};result=[]
    for vid in video_ids:
        if vid not in records:raise ValueError('Unknown video: '+vid)
        record=records[vid]
        if record['split'] not in allowed[purpose]:raise ValueError('Forbidden '+purpose+' input: '+vid)
        result.append(record)
    return result


def verify_inputs(records,include_raw_cache=False):
    for record in records:
        keys=[('video_path','video_sha256')]
        if 'object_annotation_path' in record:keys.append(('object_annotation_path','object_annotation_sha256'))
        if include_raw_cache and 'raw_cache_path' in record:keys.append(('raw_cache_path','raw_cache_sha256'))
        for path,key in keys:
            if sha(record[path])!=record[key]:raise ValueError('Frozen input changed: '+record['video_id']+' '+path)


def check_model_for_formal_validation(model,manifest):
    provenance=model.get('training_provenance',{})
    ids=provenance.get('training_video_ids')
    if provenance.get('split_id')!=manifest['split_id'] or not ids:
        raise ValueError('Model has no verified train-only provenance for this split')
    select(manifest,ids,'training')
    return True

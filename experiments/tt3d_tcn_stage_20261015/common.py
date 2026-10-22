"""Portable readers for the stage's derived arrays, never original videos."""
import hashlib
import os
import json
from pathlib import Path
import sys
import numpy as np

CODE=Path(__file__).resolve().parent
HERE=Path(os.environ.get('TT3D_TCN_BUNDLE',str(CODE/'local_bundle'))).resolve()
REPO=CODE.parents[1]
sys.path.insert(0,str(REPO))

def read(path):
    if not Path(path).is_file():raise FileNotFoundError('Local capsule is required: run prepare_local_bundle.py --source <your-local-capsule>. Missing: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def manifest(kind):
    data=read(HERE/'data'/f'{kind}_manifest.json');split=read(HERE/'data/frozen_split.json')
    allowed={r['video_id']:r['source_group'] for r in split['sets']['train']}
    held={r['video_id'] for r in split['sets']['validation_reserved']}
    public={r['video_id'] for r in split['sets']['public_regression']}
    train_groups=set(allowed.values());held_groups={r['source_group'] for r in split['sets']['validation_reserved']}
    if train_groups&held_groups or set(allowed)&(held|public):raise ValueError('Frozen split overlaps')
    ids=[r['video_id'] for r in data['sequences']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate video identity')
    for row in data['sequences']:
        if kind=='train' and (row['video_id'] not in allowed or row['source_group']!=allowed[row['video_id']]):
            raise ValueError('Non-training source in training bundle')
        if kind=='eval' and row['video_id'] not in held|public:raise ValueError('Invalid evaluation video')
        if sha(HERE/row['path'])!=row['sha256']:raise ValueError('Derived input hash mismatch')
    return data
def tracks(arrays):
    result=[]
    for fid,timestamp in enumerate(arrays['timestamps']):
        xy=arrays['xy'][fid];confidence=float(arrays['confidence'][fid])
        result.append({'frame_id':fid,'timestamp_s':float(timestamp),'detected':bool(arrays['detected'][fid]),
            'in_image':bool(arrays['in_image'][fid]),'x':float(xy[0]) if np.isfinite(xy[0]) else None,
            'y':float(xy[1]) if np.isfinite(xy[1]) else None,'confidence':confidence if np.isfinite(confidence) else None})
    return result
def sequences(kind):
    data=manifest(kind);result=[]
    for row in data['sequences']:
        with np.load(HERE/row['path'],allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
        result.append(dict(row,**arrays,tracks=tracks(arrays)))
    return data,result

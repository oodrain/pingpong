"""Retry one failed video without changing model/rules or re-encoding frames."""
import argparse
import json
from pathlib import Path
import sys

import cv2

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import sha,load_protocol,select
from development.detection.video_contact_pipeline import run
from development.detection.startup_pts_capture import StartupPTSCapture


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--video-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.evaluation/'evaluation_plan.json').read_text(encoding='utf-8'))
    manifest=load_protocol();record=next(r for r in manifest['videos'] if r['video_id']==a.video_id)
    purpose='validation' if record['split']=='validation_reserved' else 'public_regression'
    select(manifest,[record['video_id']],purpose)
    if record['video_id'] not in plan['video_ids'] or sha(record['video_path'])!=plan['video_sha256'][record['video_id']]:
        raise ValueError('Retry video is not the frozen evaluation input')
    if sha(plan['model_path'])!=plan['model_sha256']:raise ValueError('Frozen model changed')
    for path,expected in plan['pipeline_source_hashes'].items():
        if sha(path)!=expected:raise ValueError('Frozen inference rules/source changed')
    original=cv2.VideoCapture;cv2.VideoCapture=StartupPTSCapture
    import evaluate_frozen_full_videos as batch
    batch.initialise(2)
    try:report=run(Path(record['video_path']),a.output,batch.MODELS,classifier_path=Path(plan['model_path']),threads=2)
    finally:cv2.VideoCapture=original
    if report['video_sha256']!=record['video_sha256'] or report['n_frames']!=record['n_frames']:
        raise ValueError('Retry inference input/frame count differs')
    adapter=PROJECT/'development/detection/startup_pts_capture.py'
    report['decoder_compatibility_adapter']={'source_sha256':sha(adapter),'source_path':str(adapter),
        'repairs':StartupPTSCapture.repairs,'model_and_policy_unchanged':True,'evaluation_labels_used':False}
    (a.output/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'source_startup_pts_capture.py').write_bytes(adapter.read_bytes())
    (a.output/'source_retry_frozen_video_pts.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'video_id':record['video_id'],'frames':report['n_frames'],'repairs':StartupPTSCapture.repairs},ensure_ascii=False))


if __name__=='__main__':main()

"""Add three distinct stable frames for each of five six-point pilot videos."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,rows,sha,write
from development.detection.auto_table_geometry import order_quad


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--geometry-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();cv2.setNumThreads(4)
    manifest=read(args.pack/'manifest.json');geo=read(args.geometry_run/'summary.json')
    if manifest['source_sha256'][str((args.geometry_run/'geometry_samples.jsonl').resolve())]!=sha(args.geometry_run/'geometry_samples.jsonl'):raise ValueError('Geometry changed')
    pilot=[('public','v000038'),('public','v000085'),('public','v000117'),('public','v000028'),('train','23_001')]
    available={(c['group'],c['video_id']):c for c in manifest['cases']}
    if any(k not in available for k in pilot):raise ValueError('Pilot video unavailable')
    groups=defaultdict(list)
    for s in rows(args.geometry_run/'geometry_samples.jsonl'):
        if s['state']=='stable' and s['usable']:groups[s['group'],s['video_id'],s['camera_epoch']].append(s)
    data=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])));public=data/'participant/participant/pingpang/public_data'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    metas={('public',m['video_id']):m for m in read(public/'manifest.json')['videos']}
    metas.update({('train',m['video_id']):m for m in read(train/'selected_manifest.json')['videos']})
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();cases=[]
    for old in manifest['cases']:
        source=args.pack/old['image_url']
        if sha(source)!=old['image_sha256']:raise ValueError('Pack frame changed')
        shutil.copy2(source,out/old['image_url'])
        cases.append(dict(old,priority='pilot' if (old['group'],old['video_id']) in pilot else 'remaining'))
    for key in pilot:
        c=available[key];ss=groups[*key,c['camera_epoch']]
        if len(ss)<3:raise ValueError('Fewer than three stable frames for pilot')
        selected=[s for s in ss if s['frame_id']!=c['frame_id']]
        # Spread samples in time, keeping the verified original representative.
        additions=[selected[0],selected[-1]]
        if additions[0]['frame_id']==additions[1]['frame_id']:raise ValueError('Pilot frame duplication')
        m=metas[key];g,vid=key
        path=public/m['path'] if g=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(path)!=geo['input_sha256'][str(path.resolve())]:raise ValueError('Video changed')
        wanted={s['frame_id']:s for s in additions};cap=cv2.VideoCapture(str(path))
        try:
            for f in range(max(wanted)+1):
                if not cap.grab():raise ValueError('Truncated video')
                if f not in wanted:continue
                ok,im=cap.retrieve();s=wanted[f]
                if not ok or not np.array_equal(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,36)),np.array(s['scene_thumbnail'])):raise ValueError('Frame identity mismatch')
                case_id=f'{g}_{vid}_camera{c["camera_epoch"]}_{f:06d}'
                frame=out/'frames'/(case_id+'.png')
                if not cv2.imwrite(str(frame),im):raise ValueError('Image write failed')
                cases.append(dict(c,case_id=case_id,frame_id=f,image_url='frames/'+frame.name,
                    image_sha256=sha(frame),proposed_corners=order_quad(s['corners']).tolist(),priority='pilot'))
        finally:cap.release()
    cases.sort(key=lambda c:(pilot.index((c['group'],c['video_id'])) if (c['group'],c['video_id']) in pilot else len(pilot),c['group'],c['video_id'],c['frame_id']))
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    manifest.update(cases=cases,pilot_videos=[dict(group=g,video_id=v) for g,v in pilot],pilot_frames_per_video=3,
        prior_pack=str(args.pack.resolve()),prior_manifest_sha256=sha(args.pack/'manifest.json'))
    manifest['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    write(out/'manifest.json',manifest)
    print(dict(cases=len(cases),pilot_cases=sum(c['priority']=='pilot' for c in cases),pilot_videos=len(pilot),human_submissions=0),flush=True)


if __name__=='__main__':main()

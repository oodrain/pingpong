"""Prepare stable-frame six-point review; generate no human submissions."""
import argparse
import os
from pathlib import Path
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,rows,sha,write
from development.detection.auto_table_geometry import order_quad


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-run',type=Path,required=True)
    parser.add_argument('--geometry-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cv2.setNumThreads(4)
    cam=read(args.camera_run/'summary.json');geo=read(args.geometry_run/'summary.json')
    if cam['geometry_sha256']!=sha(args.geometry_run/'geometry_samples.jsonl'):raise ValueError('Geometry/calibration mismatch')
    samples={(s['group'],s['video_id'],s['frame_id']):s for s in rows(args.geometry_run/'geometry_samples.jsonl')}
    data=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])));public=data/'participant/participant/pingpang/public_data'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    metas={('public',m['video_id']):m for m in read(public/'manifest.json')['videos']}
    metas.update({('train',m['video_id']):m for m in read(train/'selected_manifest.json')['videos']})
    pilot=[('public','v000038'),('public','v000085'),('public','v000117'),('train','19_001'),('train','23_001')]
    accepted=[r for r in read(args.camera_run/'cameras.json') if r['usable']]
    accepted.sort(key=lambda r:(pilot.index((r['group'],r['video_id'])) if (r['group'],r['video_id']) in pilot else len(pilot),r['group'],r['video_id'],r['camera_epoch']))
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    cases=[]
    for r in accepted:
        group,vid=r['group'],r['video_id'];m=metas[group,vid]
        chosen=min((h for h in r['hypotheses'] if h['usable']),key=lambda h:h['representative_camera']['reprojection_rmse_px'])
        f=chosen['representative_frame'];s=samples[group,vid,f]
        if not s['usable'] or s['state']!='stable' or s['camera_epoch']!=r['camera_epoch']:raise ValueError('Unstable representative')
        path=public/m['path'] if group=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(path)!=geo['input_sha256'][str(path.resolve())]:raise ValueError('Video changed')
        cap=cv2.VideoCapture(str(path))
        try:
            for _ in range(f+1):
                if not cap.grab():raise ValueError('Video truncated before representative frame')
            ok,im=cap.retrieve()
        finally:cap.release()
        if not ok or im.shape[:2]!=(m['height'],m['width']):raise ValueError('Frame decode failed')
        thumb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,36))
        if not np.array_equal(thumb,np.array(s['scene_thumbnail'])):raise ValueError('Frame identity mismatch')
        case_id=f'{group}_{vid}_camera{r["camera_epoch"]}_{f:06d}'
        frame=out/'frames'/(case_id+'.png')
        if not cv2.imwrite(str(frame),im):raise ValueError('Frame save failed')
        cases.append(dict(case_id=case_id,group=group,video_id=vid,frame_id=f,width=m['width'],height=m['height'],
            image_url='frames/'+frame.name,image_sha256=sha(frame),video_sha256=sha(path),camera_epoch=r['camera_epoch'],
            proposed_corners=order_quad(s['corners']).tolist(),corner_source='automatic_proposal_requires_confirmation',
            priority='pilot' if (group,vid) in pilot else 'remaining',
            input_axis_status=r['physical_axes']))
    write(out/'manifest.json',dict(schema_version=1,purpose='manual net/axis and camera review',cases=cases,
        camera_source=str(args.camera_run.resolve()),geometry_source=str(args.geometry_run.resolve()),
        source_sha256={str(p.resolve()):sha(p) for p in [args.camera_run/'cameras.json',args.geometry_run/'geometry_samples.jsonl',Path(__file__)]},
        human_submissions_generated=False,default_corners='automatic proposal, explicit human confirmation required'))
    print(dict(cases=len(cases),pilot_cases=sum(c['priority']=='pilot' for c in cases),human_submissions=0,pack=str(out)),flush=True)


if __name__=='__main__':main()

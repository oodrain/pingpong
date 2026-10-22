"""Offline TT3D-model camera estimates after geometry refinement.

Only freshly stable, image-supported samples enter estimation. Held or unknown
geometry cannot be an input. Multiple frames check focal consistency; this is
not independent physical calibration validation.
"""
import argparse
from collections import defaultdict,Counter
import os
from pathlib import Path
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.detection.tt3d_camera_estimation import estimate_camera,project_table_grid,ASSUMPTIONS
from development.evaluation.run_auto_table_v2 import read,rows,write,sha


def summarize_hypothesis(estimates,axis):
    accepted=[(s,next(c for c in est['candidates'] if c['edge01_axis']==axis))
              for s,est in estimates if any(c['usable'] and c['edge01_axis']==axis for c in est['candidates'])]
    if len(accepted)<3:
        return dict(edge01_axis=axis,usable=False,reason='fewer_than_three_supported_estimates',supported_frames=len(accepted))
    focal=np.array([c['focal_px'] for s,c in accepted]);median=float(np.median(focal))
    deviations=abs(focal-median)/median
    consistent=[pair for pair,dev in zip(accepted,deviations) if dev<=.15]
    fraction=len(consistent)/len(estimates)
    spread=float(np.max(deviations))
    if len(consistent)<3 or fraction<.6 or spread>.30:
        return dict(edge01_axis=axis,usable=False,reason='focal_inconsistent_between_frames',
                    supported_frames=len(accepted),focal_estimates_px=focal.tolist(),maximum_relative_deviation=spread)
    representative=min(consistent,key=lambda pair:abs(pair[1]['focal_px']-median))
    s,c=representative
    return dict(edge01_axis=axis,usable=True,reason='estimated_under_assumptions',
        supported_frames=len(accepted),consistent_frames=len(consistent),median_focal_px=median,
        focal_estimates_px=focal.tolist(),maximum_relative_deviation=spread,
        representative_frame=s['frame_id'],representative_geometry_epoch=s['epoch'],
        representative_camera=c,representative_corners=s['corners'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();cv2.setNumThreads(4)
    provenance=read(args.run/'summary.json')
    if provenance['completed_videos']!=169:raise ValueError('Geometry regression incomplete')
    if provenance['source_sha256'][str((PROJECT/'development/detection/auto_table_geometry.py').resolve())]!=sha(PROJECT/'development/detection/auto_table_geometry.py'):
        raise ValueError('Geometry source differs')
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'overlays').mkdir()
    groups=defaultdict(list);all_keys=set()
    for s in rows(args.run/'geometry_samples.jsonl'):
        key=s['group'],s['video_id'],s['camera_epoch'];all_keys.add(key)
        if s['state']!='stable' or not s['usable']:continue
        ev=s['raw_proposal'].get('image_perimeter',{})
        if len(ev.get('straight_span_by_side',[]))!=4 or len(ev.get('neutral_border_support_by_side',[]))!=4:continue
        if not all(span>=.25 or neutral>=.75 for span,neutral in zip(ev.get('straight_span_by_side',[]),ev.get('neutral_border_support_by_side',[]))):continue
        groups[key].append(s)
    data=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])));public=data/'participant/participant/pingpang/public_data'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    metas={('public',m['video_id']):m for m in read(public/'manifest.json')['videos']}
    metas.update({('train',m['video_id']):m for m in read(train/'selected_manifest.json')['videos']})
    results=[]
    for index,key in enumerate(sorted(all_keys)):
        group,vid,epoch=key;m=metas[group,vid];samples=groups[key]
        # Selection uses time and geometry status, never event labels or corners.
        indices=np.unique(np.rint(np.linspace(0,len(samples)-1,min(7,len(samples)))).astype(int)) if samples else []
        chosen=[samples[i] for i in indices]
        estimates=[(s,estimate_camera(s['corners'],(m['width'],m['height']))) for s in chosen]
        hypotheses=[summarize_hypothesis(estimates,axis) for axis in ('length','width')]
        valid=[h for h in hypotheses if h['usable']]
        result=dict(group=group,video_id=vid,camera_epoch=epoch,fresh_stable_samples=len(samples),
            evaluated_frames=[s['frame_id'] for s in chosen],usable=bool(valid),hypotheses=hypotheses,
            physical_axes='ambiguous' if len(valid)>1 else 'conditional_metric_hypothesis' if valid else 'unknown',
            unique_physical_direction_verified=False,frame_estimates=[dict(frame_id=s['frame_id'],estimate=est) for s,est in estimates])
        if valid:
            path=public/m['path'] if group=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
            if sha(path)!=provenance['input_sha256'][str(path.resolve())]:raise ValueError('Video changed')
            h=min(valid,key=lambda p:p['representative_camera']['reprojection_rmse_px'])
            cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,h['representative_frame']);ok,im=cap.read();cap.release()
            if not ok:raise ValueError('Calibration overlay decode failed')
            c=h['representative_camera']
            for line in project_table_grid(c,(m['width'],m['height'])):
                cv2.polylines(im,[np.rint(line).astype(np.int32)],False,(0,220,255),2)
            for p in np.rint(h['representative_corners']).astype(int):cv2.circle(im,tuple(p),6,(0,255,0),-1)
            text=f"{group}/{vid} f={c['focal_px']:.0f}px RMSE={c['reprojection_rmse_px']:.2f}px axis={h['edge01_axis']} ({result['physical_axes']})"
            cv2.rectangle(im,(0,0),(im.shape[1],40),(0,0,0),-1)
            cv2.putText(im,text,(10,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            cv2.imwrite(str(out/'overlays'/f'{group}_{vid}_camera{epoch}.png'),im)
            result['overlay_hypothesis']=h['edge01_axis']
        results.append(result)
        if (index+1)%30==0:print('CAMERA_SEGMENTS',index+1,'/',len(all_keys),flush=True)
    write(out/'cameras.json',results)
    summary=dict(assumptions=ASSUMPTIONS,geometry_source=str(args.run),geometry_sha256=sha(args.run/'geometry_samples.jsonl'),
        camera_segments=len(results),usable_segments=sum(r['usable'] for r in results),
        groups={g:dict(videos=len({r['video_id'] for r in results if r['group']==g}),
            estimated_videos=len({r['video_id'] for r in results if r['group']==g and r['usable']}),
            ambiguous_segments=sum(r['usable'] and r['physical_axes']=='ambiguous' for r in results if r['group']==g)) for g in ('public','train')},
        rejection_reasons=dict(Counter(h['reason'] for r in results for h in r['hypotheses'] if not h['usable'])),
        independent_intrinsic_or_3d_validation=False,adopted_in_desktop=False,
        scope='offline estimates of representative frames, not whole-video certified calibration',
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),PROJECT/'development/detection/tt3d_camera_estimation.py']},
        upstream_reference_sha256=sha(PROJECT/'tmp/tt3d-upstream/tt3d/calibration/reproj_optimizer.py'))
    write(out/'summary.json',summary);print('DONE',summary,flush=True)


if __name__=='__main__':main()

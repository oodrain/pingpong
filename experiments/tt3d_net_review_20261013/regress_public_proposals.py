"""Frozen public-set regression of proposals, not independent final classifier scoring."""
import argparse
from pathlib import Path
import json
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.trajectory_events import propose_events
from development.bounce.ge_rule_v2 import replay
from development.evaluation.evaluate_landings import evaluate
from fit_contact_pilot import sha

def jsonl(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def run(root,out):
    project=Path(__file__).resolve().parents[2]
    cache=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1'
    public=root/'participant/participant/pingpang/public_data'
    geom=project/'outputs/experiments/tt3d_geometry_refinement_20261013/runtime_check_02/geometry_samples.jsonl'
    manifest=json.loads((cache/'selected_manifest.json').read_text(encoding='utf-8'))['videos']
    samples=jsonl(geom);out.mkdir(parents=True,exist_ok=False)
    sources={str(geom):sha(geom),str(cache/'selected_manifest.json'):sha(cache/'selected_manifest.json'),
             'rule':sha(project/'development/bounce/trajectory_events.py'),'runner':sha(__file__)}
    predictions=[];baseline=[];details=[]
    for meta in manifest:
        vid=meta['video_id'];video=public/meta['path'];track_path=cache/'trajectories'/f'{vid}.jsonl';raw_path=cache/'raw'/f'{vid}.jsonl'
        tracks=jsonl(track_path);sources[str(track_path)]=sha(track_path);sources[str(raw_path)]=sha(raw_path);sources[str(video)]=sha(video)
        raw=jsonl(raw_path);raw_by_frame={r['input_frame_id']-1:r for r in raw if r['input_frame_id']>0}
        for r in tracks:
            raw_row=raw_by_frame.get(r['frame_id'])
            if raw_row is not None:
                for key in ('detected','x','y','confidence'):
                    if r.get(key)!=raw_row.get(key):raise ValueError('Raw/aligned mismatch')
        cap=cv2.VideoCapture(str(video));timestamps=[]
        while True:
            ok,_=cap.read()
            if not ok:break
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        cap.release()
        if len(timestamps)!=len(tracks) or any(b<=a for a,b in zip(timestamps,timestamps[1:])):raise ValueError('Invalid timestamps/count')
        for row,t in zip(tracks,timestamps):
            row['timestamp_s']=t-timestamps[0];row['in_image']=bool(row['detected'] and 0<=row['x']<meta['width'] and 0<=row['y']<meta['height'])
        old,_=replay(raw);baseline.extend(dict(p,video_id=vid) for p in old)
        candidates=[s for s in samples if s['group']=='public' and s['video_id']==vid and s.get('usable')]
        reference=min(candidates,key=lambda s:s['frame_id']) if candidates else None
        # Offline proposal test: first usable AUTOMATIC geometry sets image scale/axes.
        # This does not assert that this polygon is valid for the entire video.
        proposal=propose_events(tracks,reference['corners']) if reference else None
        if proposal:
            for p in proposal['proposals']:
                row=tracks[p['frame_id']];predictions.append({'video_id':vid,'frame_id':p['frame_id'],'x':row['x'],'y':row['y']})
        details.append({'video_id':vid,'angle':meta['angle'],'reference_frame':reference['frame_id'] if reference else None,
                        'geometry_available':reference is not None,'proposal_result':proposal,'frames':len(tracks)})
        print(vid,meta['angle'],'old',len(old),'new',len(proposal['proposals']) if proposal else 0,flush=True)
    # Commit predictions and hashes BEFORE opening any event labels.
    for name,rows in [('vector_candidates.jsonl',predictions),('baseline_candidates.jsonl',baseline)]:
        (out/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    frozen={'sources':sources,'thresholds_retuned':False,'known_public_regression_not_independent_holdout':True,
            'candidate_generator_only':True,'not_full_classifier_or_subframe_evaluation':True,'uses_offline_first_automatic_geometry':True,'videos':details}
    (out/'inference.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    gt_path=public/'gt_public.jsonl';gt=jsonl(gt_path)
    # Manifest shape is the established evaluator's API: list of video records.
    scores={key:{'without_ignore_exemption':evaluate(rows,gt,manifest,exempt_invalid=False),
                 'with_ignore_exemption':evaluate(rows,gt,manifest,exempt_invalid=True)} for key,rows in [('baseline',baseline),('vector',predictions)]}
    summary={'inference_sha256':sha(out/'inference.json'),'gt_sha256':sha(gt_path),
             'interpretation':'precision/recall here score raw event proposals as table-contact predictions; not final classification accuracy',
             'known_public_regression_not_independent_holdout':True,'geometry_coverage_videos':sum(v['geometry_available'] for v in details),
             'scores':scores}
    (out/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for key,value in scores.items():
        print(key,json.dumps({k:v for k,v in value['without_ignore_exemption'].items() if k not in ('by_video',)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)

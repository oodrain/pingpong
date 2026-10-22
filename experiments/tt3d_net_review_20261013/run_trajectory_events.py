"""Replay all detector tracks; human review joins only after inference."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.trajectory_events import propose_events,select_window,classify
from fit_contact_pilot import search,sha

def run(inputs,source,motion_path,out):
    old=json.loads(source.read_text(encoding='utf-8'));motion=json.loads(motion_path.read_text(encoding='utf-8'))
    if motion['source_sha256']!=sha(source):raise ValueError('Motion evidence mismatch')
    out.mkdir(parents=True,exist_ok=False);videos=[]
    for folder in sorted((inputs/'videos').iterdir()):
        meta=json.loads((folder/'input_summary.json').read_text());camera=json.loads((folder/'camera.json').read_text())
        for name,digest in meta['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Changed input: '+name)
        rows=[json.loads(s) for s in (folder/'timed_tracks.jsonl').read_text().splitlines()];ts=np.array([r['timestamp_s'] for r in rows])
        proposals=propose_events(rows,camera['representative_pose']['projected_corners']);by_id={r['frame_id']:r for r in proposals['proposals']}
        legacy=[c['seed_frame_id'] for c in old['cases'] if c['video_id']==folder.name];results=[]
        for seed in sorted(set(by_id)|set(legacy)):
            pose=min(camera['poses'],key=lambda p:abs(p['frame_id']-seed));q=np.asarray(pose['projected_corners'],np.float32)
            point=rows[seed];distance=cv2.pointPolygonTest(q,(point['x'],point['y']),True) if point['detected'] else -float('inf')
            proposal=by_id.get(seed,{'frame_id':seed,'region':'table_projection' if distance>=-max(3,.01*np.linalg.norm(np.ptp(q,axis=0))) else 'outside_table_projection'})
            window=select_window(rows,seed,list(by_id),proposals['possible_tracking_spikes'])
            supported={r['frame_id']:r['status']=='background_stable' for c in motion['cases'] if c['case_id'].startswith(folder.name+'_') and c['anchor_frame']==pose['frame_id'] for r in c['checks']}
            camera_supported=bool(window['rows']) and all(supported.get(r['frame_id'],False) for r in window['rows'])
            fit=None
            if not window['reasons']:
                lo=float(np.interp(seed-1.5,np.arange(len(ts)),ts));hi=float(np.interp(seed+1.5,np.arange(len(ts)),ts))
                fit=search(window['rows'],pose,lo,hi,constrained=True)
                if fit:
                    fit['fractional_frame']=float(np.interp(fit['tau'],ts,np.arange(len(ts))))
                    fit.pop('time_profile',None)
            decision=classify(proposal,window,fit,camera_supported)
            result={'seed_frame':seed,'new_vector_proposal':seed in by_id,'legacy_candidate':seed in legacy,'proposal':proposal,
                    'selected_frame_ids':[r['frame_id'] for r in window['rows']],'window_reasons':window['reasons'],'camera_background_supported':camera_supported,
                    'fit':fit,'decision':decision}
            results.append(result)
        videos.append({'video_id':folder.name,'proposals':proposals,'events':results})
        print(folder.name,'new proposals',list(by_id),'results',[(r['seed_frame'],r['decision']['event']) for r in results],flush=True)
    # Labels never enter proposal generation, segmentation, fitting, or thresholds.
    journal=source.parent/'human_contact_reviews.jsonl';snapshot=journal.read_bytes();reviews={}
    for line in snapshot.decode('utf-8').splitlines():
        r=json.loads(line)
        if r['results_sha256']==sha(source):reviews[r['case_id']]=r
    evaluation=[]
    for c in old['cases']:
        video=next(v for v in videos if v['video_id']==c['video_id']);review=reviews[c['case_id']]
        same=next(r for r in video['events'] if r['seed_frame']==c['seed_frame_id'])
        interval=review['manual_interval_frames'];proposed=[p['frame_id'] for p in video['proposals']['proposals']]
        matches=[f for f in proposed if interval and interval[0]-1<=f<=interval[1]+1]
        evaluation.append({'case_id':c['case_id'],'human_decision':review['decision'],'human_interval':interval,
                           'new_proposals_within_interval_plus_one_frame':matches,'legacy_seed_decision':same['decision'],
                           'legacy_seed_selected_frames':same['selected_frame_ids']})
    (out/'human_reviews_snapshot.jsonl').write_bytes(snapshot)
    report={'schema_version':1,'mode':'offline_experimental','production_events_modified':False,'human_labels_used_for_inference':False,
            'not_full_three_view_validation':True,'racket_classification_not_implemented_without_racket_evidence':True,
            'proposal_match_tolerance_frames':1,'nonreviewed_new_proposals_are_unlabelled':True,
            'source_hashes':{str(source):sha(source),str(motion_path):sha(motion_path),str(journal):sha(journal),
                            'rule':sha(Path(__file__).resolve().parents[2]/'development/bounce/trajectory_events.py'),'runner':sha(__file__),'optimizer':sha(Path(__file__).with_name('fit_contact_pilot.py'))},
            'videos':videos,'reviewed_evaluation':evaluation}
    (out/'events.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.source,a.motion,a.output)

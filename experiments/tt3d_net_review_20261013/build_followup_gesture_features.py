"""Fill only missing automatic gesture evidence for frozen train candidates.

Reuses ball and geometry caches; reads no human event labels or coordinates.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.detection.video_contact_pipeline import PoseDetector,classify_candidates,merge_pose_alternatives
from development.detection.racket_detector import RacketDetector
from development.bounce.contact_classifier import features,FEATURES
from development.bounce.trajectory_events import observed_runs


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def forced_linear_proposal(tracks,seed,q):
    run=next((r for r in observed_runs(tracks) if r[0]['frame_id']<=seed<=r[-1]['frame_id']),None)
    if run is None:raise ValueError('Frozen candidate has no continuous observations')
    i=next(i for i,r in enumerate(run) if r['frame_id']==seed)
    if i<3 or i+3>=len(run):raise ValueError('Frozen candidate has insufficient linear context')
    subset=run[i-3:i+4];t=np.array([r['timestamp_s']-tracks[seed]['timestamp_s'] for r in subset]);unit=float(np.median(np.diff(t)));t/=unit
    xy=np.array([[r['x'],r['y']] for r in subset]);smooth=np.column_stack((np.ones(7),t,t*t));kink=np.column_stack((np.ones(7),t,np.maximum(t,0)))
    params=np.linalg.lstsq(kink,xy,rcond=None)[0];sse=float(np.sum((smooth@np.linalg.lstsq(smooth,xy,rcond=None)[0]-xy)**2));alternative=float(np.sum((kink@params-xy)**2))
    d=float(cv2.pointPolygonTest(np.asarray(q,np.float32),(tracks[seed]['x'],tracks[seed]['y']),True))
    return {'frame_id':seed,'velocity_before_px_s':(params[1]/unit).tolist(),'velocity_after_px_s':((params[1]+params[2])/unit).tolist(),
        'velocity_change_px_per_nominal_interval':float(np.linalg.norm(params[2])),'smooth_to_kink_gain':(sse-alternative)/max(sse,1e-9),
        'rms_improvement_px':float(np.sqrt(sse/7)-np.sqrt(alternative/7)),'tracking_spike_nearby':False,
        'signed_table_polygon_distance_px':d,'region':'table_projection' if d>=0 else 'outside_table_projection',
        'fit_source':'Same seven-point linear direction fit for a frozen curved candidate; not a human label'}


def run(a):
    protocol=load_protocol();pack=read(a.pack)
    if pack['purpose']!='train_only_event_labels' or pack['split_id']!=protocol['split_id']:raise ValueError('Wrong followup pack')
    records=select(protocol,list(dict.fromkeys(c['video_id'] for c in pack['cases'])),'training')
    for name,digest in pack['sources'].items():
        if sha(name)!=digest:raise ValueError('Frozen selection source changed')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    pose=PoseDetector(PROJECT/'data/pose_weights/keypointrcnn_resnet50_fpn_coco-fc266e95.pth')
    racket=RacketDetector(PROJECT/'data/racket_weights/rtmdet_m_racket_state.pt');groups=defaultdict(list)
    for c in pack['cases']:groups[c['video_id']].append(c)
    all_cases=[];started=time.monotonic();count=0;hashes={str(a.pack):sha(a.pack)}
    for number,(vid,selected) in enumerate(groups.items(),1):
        c=selected[0];track_path=Path(c['tracks_path']);tracks=lines(track_path);sample_path=track_path.with_name(f'{vid}_geometry.json');samples=read(sample_path)
        prop_path=track_path.with_name(f'{vid}_proposals.json');proposals=read(prop_path);q=next(s['corners'] for s in samples if s.get('usable'))
        existing={p['frame_id'] for p in proposals['proposals']}
        for item in selected:
            if item['seed_frame'] not in existing:proposals['proposals'].append(forced_linear_proposal(tracks,item['seed_frame'],q))
        targets={f for c in selected for f in range(c['seed_frame']-2,c['seed_frame']+3)}
        if sha(c['video_path'])!=c['video_sha256'] or sha(track_path)!=c['tracks_sha256']:raise ValueError('Wrong frozen input')
        hashes.update({str(p):sha(p) for p in [track_path,sample_path,prop_path]})
        cap=cv2.VideoCapture(c['video_path']);f=0;people={};full={};rackets={}
        while True:
            ok,image=cap.read()
            if not ok:break
            if f in targets:
                people[f]=pose.detect(image,tracks[f]);full[f]=pose.last_full_people;rackets[f]=racket.detect_near_people(image,tracks[f],people[f]);count+=1
                save(a.output/'progress.json',{'videos_done':number-1,'current_video':vid,'inference_frames':count,'elapsed_seconds':time.monotonic()-started,'human_labels_read':False})
            f+=1
        cap.release()
        if f!=c['n_frames'] or set(people)!=targets:raise ValueError('Exact selected frame decoding failed')
        cases=classify_candidates(tracks,samples,[],proposals,people,rackets)
        cases=merge_pose_alternatives(cases,classify_candidates(tracks,samples,[],proposals,full,rackets))
        seeds={c['seed_frame'] for c in selected};cases=[dict(c,case_id=f"{vid}_{c['seed_frame']}",video_id=vid,source_group=vid.split('_')[0]) for c in cases if c['seed_frame'] in seeds]
        all_cases+=cases;save(a.output/f'{vid}_automatic_evidence.json',{'video_sha256':c['video_sha256'],'poses':people,'full_poses':full,'rackets':rackets,'cases':cases})
        print('Additional train gesture evidence',number,'/',len(groups),'frames',count,'seconds',round(time.monotonic()-started,1),flush=True)
    lookup={c['case_id']:c for c in all_cases};cases=[lookup[c['case_id']] for c in pack['cases']]
    provenance={'training_video_ids':[r['video_id'] for r in records],'split_id':protocol['split_id'],'input_hashes':hashes,
        'models':{'pose':pose.provenance,'racket':racket.provenance},'human_labels_read':False,'human_coordinates_read':False,
        'ball_table_reinferred':False,'new_gesture_inference_frames':count,'elapsed_seconds':time.monotonic()-started}
    save(a.output/'automatic_cases.json',{'cases':cases,'provenance':provenance});save(a.output/'automatic_features.json',{'features':FEATURES,'vectors':[features(c).tolist() for c in cases],'case_ids':[c['case_id'] for c in cases],'provenance':provenance})
    save(a.output/'progress.json',{'status':'completed','inference_frames':count,'elapsed_seconds':time.monotonic()-started})
    (a.output/'source_build_followup_gesture_features.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

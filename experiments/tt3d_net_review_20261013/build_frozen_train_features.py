"""Offline train-only automatic features; never opens human event labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import torch

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import load_protocol,select,sha,verify_inputs
from development.detection.video_contact_pipeline import PoseDetector,classify_candidates,merge_pose_alternatives
from development.detection.racket_detector import RacketDetector
from development.bounce.contact_classifier import FEATURES,features


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(pack_path,out):
    pack=read(pack_path);manifest=load_protocol()
    if pack['split_id']!=manifest['split_id'] or pack.get('purpose')!='train_only_event_labels':
        raise ValueError('Review pack is not bound to the frozen training split')
    ids=list(dict.fromkeys(c['video_id'] for c in pack['cases']))
    records=select(manifest,ids,'training');verify_inputs(records,include_raw_cache=True)
    for name,expected in pack['sources'].items():
        if sha(name)!=expected:raise ValueError('Changed frozen automatic selection input: '+name)
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    pose_path=PROJECT/'data/pose_weights/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
    racket_path=PROJECT/'data/racket_weights/rtmdet_m_racket_state.pt'
    pose=PoseDetector(pose_path);racket=RacketDetector(racket_path)
    groups=defaultdict(list)
    for c in pack['cases']:groups[c['video_id']].append(c)
    started=time.monotonic();all_cases=[];input_hashes={str(pack_path):sha(pack_path)};pose_count=0
    for number,(vid,selected) in enumerate(sorted(groups.items()),1):
        first=selected[0];base=pack_path.parent/'automatic_inputs'
        track_path=base/f'{vid}_tracks.jsonl';geo_path=base/f'{vid}_geometry.json';proposal_path=base/f'{vid}_proposals.json'
        tracks=lines(track_path);samples=read(geo_path);proposals=read(proposal_path)
        for p in (track_path,geo_path,proposal_path):input_hashes[str(p)]=sha(p)
        targets={f for c in selected for f in range(max(0,c['seed_frame']-2),min(first['n_frames'],c['seed_frame']+3))}
        cap=cv2.VideoCapture(first['video_path']);frame_id=0;people={};full_people={};rackets={}
        while True:
            ok,image=cap.read()
            if not ok:break
            if frame_id in targets:
                people[frame_id]=pose.detect(image,tracks[frame_id]);full_people[frame_id]=pose.last_full_people
                rackets[frame_id]=racket.detect_near_people(image,tracks[frame_id],people[frame_id]);pose_count+=1
            frame_id+=1
        cap.release()
        if frame_id!=first['n_frames'] or set(people)!=targets:raise ValueError('Video/pose frame mismatch: '+vid)
        cases=classify_candidates(tracks,samples,[],proposals,people,rackets)
        cases=merge_pose_alternatives(cases,classify_candidates(tracks,samples,[],proposals,full_people,rackets))
        selected_frames={c['seed_frame'] for c in selected}
        selected_cases=[dict(c,video_id=vid,case_id=f"{vid}_{c['seed_frame']}",source_group=vid.split('_')[0]) for c in cases if c['seed_frame'] in selected_frames]
        if {c['seed_frame'] for c in selected_cases}!=selected_frames:raise ValueError('Selected automatic proposal disappeared: '+vid)
        all_cases.extend(selected_cases)
        save(out/f'{vid}_automatic_evidence.json',{'video_sha256':first['video_sha256'],'poses':people,
            'full_poses':full_people,'rackets':rackets,'cases':selected_cases})
        print('Training automatic evidence',number,'/',len(groups),'video',vid,'pose_frames',pose_count,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    lookup={c['case_id']:c for c in all_cases};ordered=[lookup[c['case_id']] for c in pack['cases']]
    provenance={'split_id':manifest['split_id'],'training_video_ids':ids,'video_hashes':{r['video_id']:r['video_sha256'] for r in records},
                'automatic_input_hashes':input_hashes,'ball_table_inputs':'Frozen automatic caches from training sources; no reviewed geometry.',
                'pose_racket_inference':'Fresh inference on selected train windows; no human coordinates.',
                'models':{'pose':pose.provenance,'racket':racket.provenance},'source_sha256':{str(p):sha(p) for p in [Path(__file__),PROJECT/'development/detection/video_contact_pipeline.py',PROJECT/'development/bounce/contact_classifier.py']},
                'human_labels_read':False,'validation_samples_used':False,'public_samples_used':False,
                'pose_frames':pose_count,'elapsed_seconds':time.monotonic()-started}
    save(out/'automatic_cases.json',{'cases':ordered,'provenance':provenance})
    save(out/'automatic_features.json',{'features':FEATURES,'case_ids':[c['case_id'] for c in ordered],
         'vectors':[features(c).tolist() for c in ordered],'provenance':provenance})
    (out/'source_build_frozen_train_features.py').write_bytes(Path(__file__).read_bytes())
    print('Automatic features frozen:',len(ordered),'cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.pack.resolve(),a.output.resolve())

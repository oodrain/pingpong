"""Frozen public auxiliary candidates: wrist evidence first, labels only afterwards."""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import quality_checked_observations,summarize
from development.evaluation.evaluate_landings import evaluate
from fit_contact_pilot import sha

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(root,pilot,weights,out):
    if not sha(weights).startswith('fc266e95'):raise ValueError('Weight checksum mismatch')
    out.mkdir(parents=True,exist_ok=False)
    screening_path=pilot/'auxiliary_screening_01/screening.json';screen=read(screening_path)
    frozen=read(pilot/'public_regression_01/inference.json')
    candidates=screen['selected_review_candidates']
    cache=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1'
    public=root/'participant/participant/pingpang/public_data'
    manifest=read(cache/'selected_manifest.json')['videos'];metadata={v['video_id']:v for v in manifest}
    torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=1120)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    results=[];sources={str(screening_path):sha(screening_path)};start=time.monotonic();done=0
    for vid in sorted({c['video_id'] for c in candidates}):
        meta=metadata[vid];path=public/meta['path'];track_path=cache/'trajectories'/f'{vid}.jsonl'
        for source in (path,track_path):
            digest=sha(source)
            if frozen['sources'][str(source)]!=digest:raise ValueError('Input changed: '+str(source))
            sources[str(source)]=digest
        tracks=lines(track_path);seeds=[c['frame_id'] for c in candidates if c['video_id']==vid]
        targets={f for s in seeds for f in range(s-1,s+2)};poses={};images={}
        cap=cv2.VideoCapture(str(path));fid=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if fid in targets:
                tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
                with torch.inference_mode():pred=model([tensor])[0]
                people=[{'score':float(score),'box':pred['boxes'][i].tolist(),'keypoints':pred['keypoints'][i].tolist(),
                         'keypoint_scores':pred['keypoints_scores'][i].tolist()} for i,score in enumerate(pred['scores']) if score>=.7]
                poses[fid]=people;images[fid]=frame;done+=1
                print(vid,fid,'processed',done,'elapsed_s',round(time.monotonic()-start,1),flush=True)
            fid+=1
        cap.release()
        if fid!=len(tracks) or set(poses)!=targets:raise ValueError('Decode/frame alignment mismatch')
        save(out/f'{vid}_poses.json',{'video_sha256':sources[str(path)],'frames':poses})
        for seed in seeds:
            evidence=[];sheet=np.zeros((480,1280,3),np.uint8)
            for i,f in enumerate(range(seed-1,seed+2)):
                ball=dict(tracks[f]);ball['eligible_high_confidence']=bool(ball.get('detected') and ball.get('confidence',0)>=.75 and 0<=ball['x']<meta['width'] and 0<=ball['y']<meta['height'])
                obs=quality_checked_observations(ball,poses[f]);evidence.append({'frame_id':f,'observations':obs})
                frame=images[f].copy()
                for o in obs:
                    color=(0,255,255) if o['near'] else (80,80,220) if o['quality_reasons'] else (120,120,120)
                    cv2.line(frame,tuple(np.rint(o['elbow_xy']).astype(int)),tuple(np.rint(o['wrist_xy']).astype(int)),color,2)
                    cv2.circle(frame,tuple(np.rint(o['wrist_xy']).astype(int)),6,color,2)
                if ball['detected']:cv2.circle(frame,(round(ball['x']),round(ball['y'])),8,(0,255,0),2)
                sheet[:240,i*426:(i+1)*426]=cv2.resize(frame,(426,240))
                cv2.putText(sheet,f'frame {f}',(i*426+10,270),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                # Crop around ball, so ambiguous/absent wrists cannot hide the event.
                x,y=round(ball['x']),round(ball['y']);left=max(0,min(frame.shape[1]-240,x-120));top=max(0,min(frame.shape[0]-180,y-90))
                sheet[285:465,i*426+90:i*426+330]=frame[top:top+180,left:left+240]
            summary=summarize(evidence)
            case={'video_id':vid,'seed_frame':seed,'angle':meta['angle'],'frames':evidence,**summary,
                'decision':'possible_racket_contact' if summary['status']=='multi_frame_wrist_proximity' else 'contact_type_unresolved',
                'adopt':False,'table_contact_confirmed':False,'camera_calibration_available':False}
            results.append(case)
            cv2.imencode('.jpg',sheet)[1].tofile(str(out/f'{vid}_{seed}_wrists.jpg'))
    report={'sources':sources,'weights_sha256':sha(weights),'script_sha256':sha(__file__),
        'evidence_code_sha256':sha(Path(__file__).resolve().parents[2]/'development/bounce/wrist_evidence.py'),
        'human_event_labels_used_for_inference':False,'production_events_modified':False,
        'known_public_regression_not_independent_holdout':True,'cases':results,
        'status_counts':dict(Counter(c['status'] for c in results)),'processed_frames':done}
    save(out/'wrist_evidence.json',report)
    # Freeze all classifications BEFORE reading GT. Exclusion below is a diagnostic only.
    flagged={(c['video_id'],c['seed_frame']) for c in results if c['decision']=='possible_racket_contact'}
    remaining=[c for c in candidates if (c['video_id'],c['frame_id']) not in flagged]
    gt=lines(public/'gt_public.jsonl');baseline=screen['primary_predictions']
    scores={name:evaluate(rows,gt,manifest,exempt_invalid=False) for name,rows in
            [('baseline',baseline),('all_auxiliary',baseline+candidates),('hypothetical_wrist_exclusion',baseline+remaining)]}
    save(out/'evaluation.json',{'wrist_evidence_sha256':sha(out/'wrist_evidence.json'),'gt_sha256':sha(public/'gt_public.jsonl'),
        'scores':scores,'flagged_count':len(flagged),'exclusion_enabled':False,
        'interpretation':'Wrist overlap is not proof of a hit; hypothetical exclusion only; no threshold tuning.'})
    print(json.dumps({'statuses':report['status_counts'],'scores':{k:v['overall'] for k,v in scores.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','pilot','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.pilot.resolve(),a.weights.resolve(),a.output.resolve())

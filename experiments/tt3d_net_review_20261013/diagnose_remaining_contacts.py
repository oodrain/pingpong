"""Local person crops probe pose extraction failures; fixed contact thresholds."""
import argparse,json,sys,time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import quality_checked_observations,iou
from development.bounce.temporal_contact import transient_wrist_visit,supplement_frozen_direction
from fit_contact_pilot import sha

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def nearest_actor(ball,people):
    if not ball.get('detected') or ball.get('confidence',0)<.75:return None
    xy=np.array([ball['x'],ball['y']]);options=[]
    for index,p in enumerate(people):
        if p['score']<.9:continue
        b=np.array(p['box']);height=b[3]-b[1]
        if height<=0:continue
        distance=float(np.linalg.norm(np.maximum(np.maximum(b[:2]-xy,xy-b[2:]),0)))/height
        # Existing physical quality ceiling for a forearm relative to person height.
        if distance<=.45:options.append((distance,index))
    return min(options)[1] if options else None

def run(root,pilot,weights,out):
    previous_path=pilot/'temporal_contact_01/temporal_evidence.json';previous=read(previous_path)
    direction_path=pilot/'contact_direction_01/direction_evidence.json';direction=read(direction_path)
    if sha(weights)!=previous['weights_sha256']:raise ValueError('Pose weights changed')
    manifest_path=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json'
    meta={r['video_id']:r for r in read(manifest_path)['videos']};public=root/'participant/participant/pingpang/public_data'
    direction_lookup={(c['video_id'],c['seed_frame']):c for c in direction['cases']}
    sources={str(p):sha(p) for p in (previous_path,direction_path,manifest_path)}
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=1120)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    cases=[];processed=0;replaced=0;start=time.monotonic()
    for vid in sorted({c['video_id'] for c in previous['cases']}):
        m=meta[vid];video=public/m['path'];track_path=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories'/f'{vid}.jsonl'
        for path in (video,track_path):
            digest=sha(path)
            if digest!=previous['sources'][str(path)]:raise ValueError('Input changed')
            sources[str(path)]=digest
        pose_path=pilot/'temporal_contact_01'/f'{vid}_poses.json';sources[str(pose_path)]=sha(pose_path)
        original=read(pose_path)['frames'];tracks=lines(track_path)
        seeds=[c['seed_frame'] for c in previous['cases'] if c['video_id']==vid];targets={f for seed in seeds for f in range(seed-2,seed+3)}
        cap=cv2.VideoCapture(str(video));fid=0;poses={};images={};probes={}
        while True:
            ok,image=cap.read()
            if not ok:break
            if fid in targets:
                people=deepcopy(original[str(fid)]);ball=tracks[fid];index=nearest_actor(ball,people)
                probe={'actor_index':index,'replaced':False,'reason':'no_supported_actor'}
                if index is not None:
                    box=people[index]['box'];margin=.2*(box[3]-box[1])
                    x0=max(0,int(np.floor(box[0]-margin)));y0=max(0,int(np.floor(box[1]-margin)))
                    x1=min(m['width'],int(np.ceil(box[2]+margin)));y1=min(m['height'],int(np.ceil(box[3]+margin)))
                    crop=image[y0:y1,x0:x1]
                    tensor=torch.from_numpy(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
                    with torch.inference_mode():pred=model([tensor])[0]
                    processed+=1;candidates=[]
                    for j,score in enumerate(pred['scores']):
                        if score<.9:continue
                        new_box=(pred['boxes'][j]+torch.tensor([x0,y0,x0,y0])).tolist()
                        overlap=iou(box,new_box)
                        if overlap>=.3:
                            keypoints=pred['keypoints'][j].clone();keypoints[:,0]+=x0;keypoints[:,1]+=y0
                            candidates.append((overlap,{'score':float(score),'box':new_box,'keypoints':keypoints.tolist(),'keypoint_scores':pred['keypoints_scores'][j].tolist()}))
                    probe.update(roi=[x0,y0,x1,y1],original_actor=people[index],reason='roi_actor_unmatched')
                    if candidates:
                        overlap,new=max(candidates,key=lambda item:item[0]);people[index]=new;replaced+=1
                        probe.update(replaced=True,matched_iou=overlap,reason='roi_actor_matched',roi_actor=new)
                    print(vid,fid,'crops',processed,'matched',replaced,'elapsed_s',round(time.monotonic()-start,1),flush=True)
                poses[fid]=people;images[fid]=image;probes[fid]=probe
            fid+=1
        cap.release()
        if fid!=len(tracks):raise ValueError('Frame count changed')
        save(out/f'{vid}_pose_probes.json',{'video_sha256':sources[str(video)],'frames':poses,'probes':probes})
        for seed in seeds:
            evidence=[];sheet=np.zeros((430,1280,3),np.uint8)
            for i,f in enumerate(range(seed-2,seed+3)):
                ball=dict(tracks[f]);ball['eligible_high_confidence']=bool(ball.get('detected') and ball.get('confidence',0)>=.75 and 0<=ball['x']<m['width'] and 0<=ball['y']<m['height'])
                obs=quality_checked_observations(ball,poses[f]);evidence.append({'frame_id':f,'observations':obs})
                image=images[f].copy()
                for o in obs:
                    color=(0,255,255) if o['near'] else (80,80,220) if o['quality_reasons'] else (120,120,120)
                    cv2.line(image,tuple(np.rint(o['elbow_xy']).astype(int)),tuple(np.rint(o['wrist_xy']).astype(int)),color,2)
                if ball.get('detected'):cv2.circle(image,(round(ball['x']),round(ball['y'])),8,(0,255,0),2)
                sheet[:144,i*256:(i+1)*256]=cv2.resize(image,(256,144));cv2.putText(sheet,f'frame {f}',(i*256+6,173),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                x=round(ball['x']) if ball.get('detected') else m['width']//2;y=round(ball['y']) if ball.get('detected') else m['height']//2
                left=max(0,min(m['width']-240,x-120));top=max(0,min(m['height']-180,y-90))
                sheet[200:380,i*256+8:i*256+248]=image[top:top+180,left:left+240]
            visit=transient_wrist_visit(evidence,seed);d=direction_lookup[(vid,seed)]
            old=next(c for c in previous['cases'] if c['video_id']==vid and c['seed_frame']==seed)
            decision=supplement_frozen_direction(d['direction'],visit,old['decision']['hit_flag'])
            decision['baseline_temporal_flag']=decision.pop('original_joint_flag')
            cases.append({'video_id':vid,'seed_frame':seed,'angle':m['angle'],'frames':evidence,'roi_visit':visit,
                'baseline_visit':old['visit'],'direction':d['direction'],'decision':decision})
            cv2.imencode('.jpg',sheet)[1].tofile(str(out/f'{vid}_{seed}_roi.jpg'))
    project=Path(__file__).resolve().parents[2]
    report={'sources':sources,'weights_sha256':sha(weights),'runner_sha256':sha(__file__),
        'temporal_rule_sha256':sha(project/'development/bounce/temporal_contact.py'),
        'wrist_quality_rule_sha256':sha(project/'development/bounce/wrist_evidence.py'),
        'cases':cases,'crop_inferences':processed,'matched_crop_actors':replaced,'roi_margin_person_height':.2,
        'contact_thresholds_relaxed':False,'human_labels_used_for_inference':False,
        'method_designed_after_prior_human_review':True,'production_events_modified':False,'independent_holdout':False}
    save(out/'roi_evidence.json',report);(out/'source_diagnose_remaining_contacts.py').write_bytes(Path(__file__).read_bytes())
    human_path=pilot/'human_event_evaluation_01/human_reviews_snapshot.json';human=read(human_path)
    lookup={(c['video_id'],c['seed_frame']):c for c in cases};rows=[]
    for h in human['records']:
        c=lookup[(h['video_id'],h['seed_frame'])]
        rows.append({'case_id':h['case_id'],'human_label':h['candidate_label'],'baseline_flag':c['decision']['baseline_temporal_flag'],'roi_flag':c['decision']['hit_flag']})
    sys.path.insert(0,str(root/'development/annotation'))
    from evaluate_auxiliary_reviews import flag_metrics,table_pool_metrics
    score={'roi_evidence_sha256':sha(out/'roi_evidence.json'),'human_snapshot_sha256':sha(human_path),
        'baseline_flags':flag_metrics(rows,'baseline_flag'),'roi_flags':flag_metrics(rows,'roi_flag'),
        'roi_table_pool':table_pool_metrics(rows,'roi_flag'),'cases':rows,
        'scope':'Already reviewed development candidates; diagnostic ROI probe, not independent validation.'}
    save(out/'evaluation.json',score);print(json.dumps({k:score[k] for k in ('baseline_flags','roi_flags','roi_table_pool')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','pilot','weights','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.pilot.resolve(),a.weights.resolve(),a.output.resolve())

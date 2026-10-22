"""Five-frame pose reuse and flight-piece diagnostics on all frozen candidates."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import quality_checked_observations,summarize
from development.bounce.temporal_contact import transient_wrist_visit,quadratic_phase_diagnostic,supplement_frozen_direction
from development.bounce.trajectory_events import select_window
from fit_contact_pilot import sha

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(root,pilot,weights,out):
    project=Path(__file__).resolve().parents[2];screen_path=pilot/'auxiliary_screening_01/screening.json'
    direction_path=pilot/'contact_direction_01/direction_evidence.json';inference_path=pilot/'public_regression_01/inference.json'
    wrist_path=pilot/'public_auxiliary_wrists_01/wrist_evidence.json';old_wrist=read(wrist_path)
    if sha(weights)!=old_wrist['weights_sha256']:raise ValueError('Changed pose weights')
    screen=read(screen_path);direction=read(direction_path);inference=read(inference_path)
    candidates=screen['selected_review_candidates'];directions={(c['video_id'],c['seed_frame']):c for c in direction['cases']}
    cache=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1';public=root/'participant/participant/pingpang/public_data'
    manifest_path=cache/'selected_manifest.json';meta={v['video_id']:v for v in read(manifest_path)['videos']}
    sources={str(p):sha(p) for p in (screen_path,direction_path,inference_path,wrist_path,manifest_path)}
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=1120)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    start=time.monotonic();processed=0;reused=0;cases=[]
    for vid in sorted({c['video_id'] for c in candidates}):
        m=meta[vid];video=public/m['path'];track_path=cache/'trajectories'/f'{vid}.jsonl'
        for path in (video,track_path):
            digest=sha(path)
            if digest!=inference['sources'][str(path)]:raise ValueError('Changed input')
            sources[str(path)]=digest
        old_cache=pilot/'public_auxiliary_wrists_01'/f'{vid}_poses.json';sources[str(old_cache)]=sha(old_cache)
        cached=read(old_cache)
        if cached['video_sha256']!=sources[str(video)]:raise ValueError('Cached poses belong to different video')
        tracks=lines(track_path);seeds=[c['frame_id'] for c in candidates if c['video_id']==vid]
        targets={f for s in seeds for f in range(max(0,s-2),min(len(tracks),s+3))}
        cap=cv2.VideoCapture(str(video));fid=0;images={};poses={};times=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
            if fid in targets:
                if str(fid) in cached['frames']:people=cached['frames'][str(fid)];reused+=1
                else:
                    tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
                    with torch.inference_mode():pred=model([tensor])[0]
                    people=[{'score':float(score),'box':pred['boxes'][i].tolist(),'keypoints':pred['keypoints'][i].tolist(),
                        'keypoint_scores':pred['keypoints_scores'][i].tolist()} for i,score in enumerate(pred['scores']) if score>=.7]
                    processed+=1;print(vid,fid,'new',processed,'reused',reused,'elapsed_s',round(time.monotonic()-start,1),flush=True)
                images[fid]=frame;poses[fid]=people
            fid+=1
        cap.release()
        if fid!=len(tracks) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Timestamp/frame mismatch')
        for r,t in zip(tracks,times):
            r['timestamp_s']=t-times[0];r['in_image']=bool(r.get('detected') and 0<=r['x']<m['width'] and 0<=r['y']<m['height'])
        save(out/f'{vid}_poses.json',{'video_sha256':sources[str(video)],'frames':poses,'original_pose_cache_sha256':sources[str(old_cache)]})
        proposals=next(v for v in inference['videos'] if v['video_id']==vid)['proposal_result']
        for seed in seeds:
            frames=[];sheet=np.zeros((430,1280,3),np.uint8)
            for i,f in enumerate(range(seed-2,seed+3)):
                ball=dict(tracks[f]);ball['eligible_high_confidence']=bool(ball.get('detected') and ball.get('confidence',0)>=.75 and ball['in_image'])
                observations=quality_checked_observations(ball,poses[f]);frames.append({'frame_id':f,'observations':observations})
                image=images[f].copy()
                for o in observations:
                    col=(0,255,255) if o['near'] else (80,80,220) if o['quality_reasons'] else (120,120,120)
                    cv2.line(image,tuple(np.rint(o['elbow_xy']).astype(int)),tuple(np.rint(o['wrist_xy']).astype(int)),col,2)
                if ball.get('detected'):cv2.circle(image,(round(ball['x']),round(ball['y'])),8,(0,255,0),2)
                sheet[:144,i*256:(i+1)*256]=cv2.resize(image,(256,144))
                cv2.putText(sheet,f'frame {f}',(i*256+6,173),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                x=round(ball['x']) if ball.get('detected') else m['width']//2;y=round(ball['y']) if ball.get('detected') else m['height']//2
                left=max(0,min(m['width']-240,x-120));top=max(0,min(m['height']-180,y-90))
                sheet[200:380,i*256+8:i*256+248]=image[top:top+180,left:left+240]
            visit=transient_wrist_visit(frames,seed);d=directions[(vid,seed)]
            window=select_window(tracks,seed,[p['frame_id'] for p in proposals['proposals']],proposals['possible_tracking_spikes'])
            phase=quadratic_phase_diagnostic(window,seed)
            decision=supplement_frozen_direction(d['direction'],visit,d['decision']['event']=='racket_contact_candidate')
            cases.append({'video_id':vid,'seed_frame':seed,'angle':m['angle'],'frames':frames,'visit':visit,
                'five_frame_wrist_summary':summarize(frames),'selected_window_frames':[r['frame_id'] for r in window['rows']],
                'phase_diagnostic':phase,'decision':decision})
            cv2.imencode('.jpg',sheet)[1].tofile(str(out/f'{vid}_{seed}_temporal.jpg'))
    report={'cases':cases,'sources':sources,'new_pose_frames':processed,'reused_pose_frames':reused,
        'weights_sha256':sha(weights),'runner_sha256':sha(__file__),
        'rule_sha256':sha(project/'development/bounce/temporal_contact.py'),
        'human_labels_used_for_inference':False,'method_designed_after_prior_human_review':True,
        'independent_holdout':False,'production_events_modified':False,'thresholds_retuned_on_results':False}
    save(out/'temporal_evidence.json',report)
    (out/'source_extend_temporal_contact.py').write_bytes(Path(__file__).read_bytes())
    (out/'source_temporal_contact.py').write_bytes((project/'development/bounce/temporal_contact.py').read_bytes())
    # Read manual labels only after all evidence and decisions have been persisted.
    human_path=pilot/'human_event_evaluation_01/human_reviews_snapshot.json';human=read(human_path)
    lookup={(c['video_id'],c['seed_frame']):c for c in cases};rows=[]
    for r in human['records']:
        c=lookup[(r['video_id'],r['seed_frame'])]
        rows.append({'case_id':r['case_id'],'human_label':r['candidate_label'],'old_flag':c['decision']['original_joint_flag'],
            'new_flag':c['decision']['hit_flag'],'phase_fit_available':c['phase_diagnostic']['available']})
    sys.path.insert(0,str(root/'development/annotation'))
    from evaluate_auxiliary_reviews import flag_metrics,table_pool_metrics
    evaluation={'temporal_evidence_sha256':sha(out/'temporal_evidence.json'),'human_snapshot_sha256':sha(human_path),
        'old_flags':flag_metrics(rows,'old_flag'),'new_flags':flag_metrics(rows,'new_flag'),
        'old_table_pool':table_pool_metrics(rows,'old_flag'),'new_table_pool':table_pool_metrics(rows,'new_flag'),
        'cases':rows,'scope':'Development regression on the already reviewed 31 candidates; not independent accuracy.'}
    save(out/'evaluation.json',evaluation);print(json.dumps({k:evaluation[k] for k in ('old_flags','new_flags','new_table_pool')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','pilot','weights','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.pilot.resolve(),a.weights.resolve(),a.output.resolve())

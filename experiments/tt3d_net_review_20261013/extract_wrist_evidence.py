"""CPU keypoint inference on automatically proposed event neighbourhoods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import wrist_observations,summarize
from fit_contact_pilot import sha

def run(events,inputs,weights,out):
    expected=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.url.rsplit('-',1)[1].split('.')[0]
    if not sha(weights).startswith(expected):raise ValueError('Official weight checksum mismatch')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=1120)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    report=json.loads(events.read_text(encoding='utf-8'));results=[];cache_files=[]
    for video in report['videos']:
        name=video['video_id'];folder=inputs/'videos'/name;meta=json.loads((folder/'input_summary.json').read_text())
        if sha(meta['video_path'])!=meta['video_sha256']:raise ValueError('Changed video')
        if sha(folder/'timed_tracks.jsonl')!=meta['files']['timed_tracks.jsonl']:raise ValueError('Changed tracks')
        tracks=[json.loads(s) for s in (folder/'timed_tracks.jsonl').read_text().splitlines()]
        seeds=[p['frame_id'] for p in video['proposals']['proposals']];targets={f for s in seeds for f in range(max(0,s-1),min(len(tracks),s+2))}
        cap=cv2.VideoCapture(meta['video_path']);fid=0;poses={};images={};start=time.monotonic()
        while True:
            ok,frame=cap.read()
            if not ok:break
            if fid in targets:
                tensor=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
                with torch.inference_mode():prediction=model([tensor])[0]
                people=[]
                for i,score in enumerate(prediction['scores']):
                    if score<.7:continue
                    people.append({'score':float(score),'box':prediction['boxes'][i].tolist(),'keypoints':prediction['keypoints'][i].tolist(),'keypoint_scores':prediction['keypoints_scores'][i].tolist()})
                poses[fid]=people;images[fid]=frame
                print(name,fid,'people',len(people),'elapsed',round(time.monotonic()-start,1),flush=True)
            fid+=1
        cap.release()
        if fid!=len(tracks):raise ValueError('Decode mismatch')
        cache=out/(name+'_poses.json');cache.write_text(json.dumps({'video_sha256':meta['video_sha256'],'frames':poses},indent=2),encoding='utf-8');cache_files.append(str(cache))
        for seed in seeds:
            frames=[{'frame_id':f,'observations':wrist_observations(tracks[f],poses[f])} for f in sorted(targets) if abs(f-seed)<=1]
            result={'video_id':name,'seed_frame':seed,'frames':frames,**summarize(frames)}
            event=next(e for e in video['events'] if e['seed_frame']==seed)
            result['trajectory_class']=event['decision']['event']
            result['combined_status']='conflicting_table_and_wrist_evidence' if result['status']=='multi_frame_wrist_proximity' and result['trajectory_class']=='table_contact_candidate' else 'racket_contact_candidate' if result['status']=='multi_frame_wrist_proximity' and result['trajectory_class'] in ('uncertain','off_table_motion_change') else 'insufficient_racket_evidence'
            result['adopt']=False;results.append(result)
            sheet=np.zeros((480,1280,3),np.uint8)
            for i,r in enumerate(frames):
                frame=images[r['frame_id']].copy()
                for o in r['observations']:
                    color=(0,255,255) if o['near'] else (120,120,120)
                    cv2.line(frame,tuple(np.rint(o['elbow_xy']).astype(int)),tuple(np.rint(o['wrist_xy']).astype(int)),color,2)
                    cv2.circle(frame,tuple(np.rint(o['wrist_xy']).astype(int)),6,color,2)
                ball=tracks[r['frame_id']]
                if ball['detected']:cv2.circle(frame,(round(ball['x']),round(ball['y'])),8,(0,255,0),2)
                sheet[:240,i*426:(i+1)*426]=cv2.resize(frame,(426,240))
                cv2.putText(sheet,f"frame {r['frame_id']}",(i*426+10,265),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                observations=r['observations']
                if observations:
                    nearest=min(observations,key=lambda x:x['distance_in_forearms']);x,y=np.rint(nearest['wrist_xy']).astype(int)
                    left=max(0,min(frame.shape[1]-240,x-120));top=max(0,min(frame.shape[0]-180,y-90))
                    sheet[285:465,i*426+90:i*426+330]=frame[top:top+180,left:left+240]
            cv2.imencode('.jpg',sheet)[1].tofile(str(out/(name+'_'+str(seed)+'_wrists.jpg')))
    output={'model':'torchvision KeypointRCNN ResNet50 FPN COCO','model_url':KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.url,'weights_sha256':sha(weights),
            'source_events_sha256':sha(events),'script_sha256':sha(__file__),'evidence_code_sha256':sha(Path(__file__).resolve().parents[2]/'development/bounce/wrist_evidence.py'),
            'torch_version':torch.__version__,'person_score_threshold':.9,'raw_keypoint_score_threshold':2,'max_distance_forearms':1.25,
            'human_event_labels_used':False,'production_events_modified':False,'pose_caches':cache_files,'cases':results}
    (out/'wrist_evidence.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Finished',len(results),'candidate neighbourhoods',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.events,a.inputs,a.weights,a.output)

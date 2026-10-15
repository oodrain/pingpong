"""Experimental video-only contact analysis; no annotation/cache input options.

Models are fixed application assets. Table axes and camera heights remain unknown
unless observed automatically. This runner is an offline CPU prototype, not an
accepted replacement for the competition baseline.
"""
import argparse, hashlib, json, time, sys
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
ROOT=PROJECT.parents[1]
from development.detection.video_ball_detector import VideoBallDetector
from development.detection.racket_detector import RacketDetector
from development.detection.tt3d_segmenter import TT3DSegmenter
from development.detection.auto_table_geometry import GeometryConfig, CausalGeometryTracker, extract_quad
from development.bounce.ge_rule_v2 import replay
from development.bounce.trajectory_events import propose_events, select_window
from development.bounce.wrist_evidence import quality_checked_observations, summarize, iou
from development.bounce.contact_direction import direction_evidence, fuse_direction_wrist
from development.bounce.temporal_contact import transient_wrist_visit, supplement_frozen_direction
from development.bounce.person_motion_contact import person_box_visit, local_table_position, review_suggestion
from development.bounce.racket_contact_evidence import observations, transient_racket_visit
from development.bounce.contact_classifier import load as load_classifier,apply_model
from development.bounce.contact_output_policy import experimental_landings,POLICY
from development.detection.startup_pts_capture import StartupPTSCapture

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save_lines(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')

class PoseDetector:
    def __init__(self,checkpoint):
        self.model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=1120)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));self.model.eval()
        self.provenance={'checkpoint_sha256':sha(checkpoint),'score_threshold':.7,'device':'cpu','roi_person_margin':.2}
    def predict(self,image):
        t=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
        with torch.inference_mode():result=self.model([t])[0]
        return [{'score':float(s),'box':result['boxes'][i].tolist(),'keypoints':result['keypoints'][i].tolist(),
                 'keypoint_scores':result['keypoints_scores'][i].tolist()} for i,s in enumerate(result['scores']) if s>=.7]
    def detect(self,image,ball):
        people=self.predict(image)
        # Keep an independent full-frame branch; a crop must not erase valid
        # observations merely because it has a higher person-box confidence.
        from copy import deepcopy
        self.last_full_people=deepcopy(people)
        if not ball.get('detected') or ball.get('confidence',0)<.75:return people
        xy=np.array([ball['x'],ball['y']]);choices=[]
        for i,p in enumerate(people):
            if p['score']<.9:continue
            b=np.asarray(p['box']);height=b[3]-b[1]
            if height<=0:continue
            distance=float(np.linalg.norm(np.maximum(np.maximum(b[:2]-xy,xy-b[2:]),0)))/height
            if distance<=.45:choices.append((distance,i))
        if choices:
            index=min(choices)[1];b=people[index]['box'];m=.2*(b[3]-b[1]);h,w=image.shape[:2]
            x0=max(0,int(b[0]-m));y0=max(0,int(b[1]-m));x1=min(w,int(b[2]+m));y1=min(h,int(b[3]+m))
            options=[]
            for p in self.predict(image[y0:y1,x0:x1]):
                if p['score']<.9:continue
                box=p['box'];p['box']=[box[0]+x0,box[1]+y0,box[2]+x0,box[3]+y0]
                for k in p['keypoints']:k[0]+=x0;k[1]+=y0
                overlap=iou(b,p['box'])
                if overlap>=.3:options.append((overlap,p))
            if options:people[index]=max(options,key=lambda p:p[0])[1]
        return people

def classify_candidates(tracks,samples,baseline,proposals,pose_frames,racket_frames):
    lookup={p['frame_id']:p for p in proposals['proposals']};primary={p['frame_id'] for p in baseline}
    seeds=sorted(primary|set(lookup));cases=[];q=next((s['corners'] for s in samples if s.get('usable')),None)
    for seed in seeds:
        rows=[{'frame_id':f,'ball':tracks[f],'people':pose_frames.get(f,[])} for f in range(max(0,seed-2),min(len(tracks),seed+3))]
        wf=[{'frame_id':r['frame_id'],'observations':quality_checked_observations(dict(r['ball'],eligible_high_confidence=bool(r['ball']['detected'] and r['ball']['confidence']>=.75)),r['people'])} for r in rows]
        wrist=transient_wrist_visit(wf,seed);body=person_box_visit(rows,seed)
        positions=[dict(local_table_position(samples,r['ball'],r['frame_id']),frame_id=r['frame_id']) for r in rows]
        rf=[{'frame_id':r['frame_id'],'observations':observations(r['ball'],racket_frames.get(r['frame_id'],[]))} for r in rows]
        racket=transient_racket_visit(rf,seed)
        d=None;decision={'review_hit_suggestion':False,'adopt':False};window=None
        if seed in lookup and q is not None:
            p=lookup[seed];d=direction_evidence(p,q,None)  # No reviewed edge-axis assignment.
            window=select_window(tracks,seed,list(lookup),proposals['possible_tracking_spikes'])
            joint=fuse_direction_wrist(d,summarize(wf))['event']=='racket_contact_candidate'
            original=supplement_frozen_direction(d,wrist,joint)['hit_flag']
            decision=review_suggestion(original,d,wrist,body,positions,p)
            # Require the same segmented trajectory context for every supplement.
            if window['reasons']:decision['review_hit_suggestion']=False
            racket_flag=bool(not window['reasons'] and racket['supported'] and not p['tracking_spike_nearby'])
            decision.update(automatic_racket_supplement=racket_flag,review_hit_suggestion=decision['review_hit_suggestion'] or racket_flag)
        kind='racket_contact_candidate' if decision['review_hit_suggestion'] else 'legacy_table_candidate' if seed in primary else 'contact_type_unresolved'
        cases.append({'seed_frame':seed,'timestamp_s':tracks[seed]['timestamp_s'],'source':['baseline'] if seed in primary else ['vector'],
            'event':kind,'confirmed':False,'decision':decision,'direction':d,'wrist_visit':wrist,'person_visit':body,
            'racket_visit':racket,'local_positions':positions,'window_reasons':window['reasons'] if window else ['no_vector_context'],
            'camera_height_verified':False,'physical_table_axis_source':'unknown'})
    return cases

def merge_pose_alternatives(cropped,full):
    lookup={c['seed_frame']:c for c in full}
    if {c['seed_frame'] for c in cropped}!=set(lookup):raise ValueError('Pose branches have different candidate identities')
    for c in cropped:
        other=lookup[c['seed_frame']]
        if c['timestamp_s']!=other['timestamp_s'] or c['window_reasons']!=other['window_reasons']:
            raise ValueError('Pose branches use different temporal context')
        supported=bool(other['decision']['review_hit_suggestion'])
        c['full_frame_pose_decision']=other['decision'];c['full_frame_wrist_visit']=other['wrist_visit']
        c['decision']['full_frame_pose_supplement']=bool(supported and not c['decision']['review_hit_suggestion'])
        c['decision']['review_hit_suggestion']=bool(c['decision']['review_hit_suggestion'] or supported)
        if c['decision']['review_hit_suggestion']:c['event']='racket_contact_candidate'
    return cropped

def run(video,out,models=None,classifier_path=None,threads=4):
    video=Path(video).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(threads)
    assets={'ball':ROOT/'participant/participant/pingpang/participant/weights/ball.onnx',
            'table':PROJECT/'tmp/tt3d-upstream/weights/table_segmentation.ckpt',
            'pose':PROJECT/'data/pose_weights/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
            'racket':PROJECT/'data/racket_weights/rtmdet_m_racket_state.pt'}
    classifier_path=Path(classifier_path) if classifier_path else PROJECT/'data/contact_classifier/source25_train_20261014_v1/model.json'
    classifier=load_classifier(classifier_path)
    if models is None:
        models={'ball':VideoBallDetector(assets['ball']),'table':TT3DSegmenter(assets['table']),
                'pose':PoseDetector(assets['pose']),'racket':RacketDetector(assets['racket'])}
    if hasattr(models['ball'],'reset'):models['ball'].reset()
    StartupPTSCapture.repairs.clear()
    initialised=time.monotonic();print('Models ready; reading video only:',video,flush=True)
    cap=StartupPTSCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open video')
    fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid video FPS')
    cfg=GeometryConfig();tracker=CausalGeometryTracker(cfg);samples=[];raw=[];times=[];sizes=[];fid=0;next_sample=0
    while True:
        ok,image=cap.read()
        if not ok:break
        pts=float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if not np.isfinite(pts) or (times and pts<=times[-1]):raise ValueError('Nonmonotonic decoded video timestamps')
        times.append(pts);sizes.append(image.shape[:2]);det=models['ball'].detect(image)
        raw.append({'input_frame_id':fid,'target_frame_id':fid-1,'detected':det is not None,
                    'x':det['x'] if det else None,'y':det['y'] if det else None,'confidence':det['confidence'] if det else None})
        if fid>=next_sample:
            state=tracker.update(fid,image,extract_quad(image,models['table'].probability(image),cfg));samples.append(state)
            next_sample=fid+max(1,round(fps*(.4 if state['state']=='stable' else .12)))
        fid+=1
        if fid%40==0:print('Ball/table frames',fid,'/',declared,flush=True)
    cap.release()
    if not fid or len(set(sizes))!=1:raise ValueError('Empty video or changing image size')
    if declared!=fid:raise ValueError('Video declared/decoded frame count mismatch')
    h,w=sizes[0];aligned={r['target_frame_id']:r for r in raw if r['target_frame_id']>=0};tracks=[]
    for f,t in enumerate(times):
        r=aligned.get(f);det=bool(r and r['detected'])
        tracks.append({'frame_id':f,'timestamp_s':t-times[0],'detected':det,'x':r['x'] if det else None,
            'y':r['y'] if det else None,'confidence':r['confidence'] if det else None,
            'in_image':bool(det and 0<=r['x']<w and 0<=r['y']<h),'missing_reason':None if det else 'no_detection' if r else 'tail_not_inferred'})
    baseline,_=replay(raw,mode='v0');q=next((s['corners'] for s in samples if s.get('usable')),None)
    proposals=propose_events(tracks,q) if q is not None else {'proposals':[],'possible_tracking_spikes':[]}
    save_lines(out/'ball_tracks.jsonl',tracks);save_lines(out/'geometry_samples.jsonl',samples);save(out/'proposals.json',proposals)
    proposal_ids=[p['frame_id'] for p in proposals['proposals']]
    # A legacy-only seed has no fitted direction and cannot pass the classifier
    # context gate. Avoid expensive pose inference that cannot affect its type.
    seeds={p['frame_id'] for p in proposals['proposals'] if not p['tracking_spike_nearby']
        and not select_window(tracks,p['frame_id'],proposal_ids,proposals['possible_tracking_spikes'])['reasons']}
    targets={f for s in seeds for f in range(max(0,s-2),min(fid,s+3))};people={};full_people={};rackets={}
    cap=StartupPTSCapture(str(video));f=0;processed=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if f in targets:
            people[f]=models['pose'].detect(image,tracks[f]);full_people[f]=models['pose'].last_full_people
            rackets[f]=models['racket'].detect_near_people(image,tracks[f],people[f]);processed+=1
            if processed%5==0:print('Pose/racket frames',processed,'/',len(targets),'elapsed_s',round(time.monotonic()-initialised,1),flush=True)
        f+=1
    cap.release()
    if f!=fid:raise ValueError('Second-pass decode mismatch')
    cases=classify_candidates(tracks,samples,baseline,proposals,people,rackets)
    cases=merge_pose_alternatives(cases,classify_candidates(tracks,samples,baseline,proposals,full_people,rackets))
    cases=apply_model(classifier,cases)
    save(out/'pose_frames.json',people);save(out/'full_pose_frames.json',full_people)
    save(out/'racket_frames.json',rackets);save(out/'event_candidates.json',{'cases':cases})
    # Preserve the legacy output separately; experimental exclusions are disabled.
    save_lines(out/'legacy_landings.jsonl',baseline)
    primary_exclusion=bool(classifier.get('output_policy',{}).get('baseline_exclusion_enabled',False))
    experimental,auxiliary_decisions=experimental_landings(tracks,baseline,proposals,cases,True,primary_exclusion)
    all_ready,_=experimental_landings(tracks,baseline,proposals,cases,False,False)
    policy=dict(POLICY,baseline_exclusion_enabled=primary_exclusion)
    save_lines(out/'experimental_landings.jsonl',experimental)
    save_lines(out/'all_ready_auxiliary_landings.jsonl',all_ready)
    save(out/'experimental_contact_policy.json',{'policy':policy,'auxiliary_decisions':auxiliary_decisions})
    sources=[PROJECT/'development'/sub/p for sub,files in [('detection',['video_contact_pipeline.py','video_ball_detector.py','racket_detector.py','tt3d_segmenter.py','auto_table_geometry.py','table_border_lines.py','startup_pts_capture.py']),
        ('bounce',['trajectory_events.py','wrist_evidence.py','temporal_contact.py','person_motion_contact.py','racket_contact_evidence.py','contact_direction.py','contact_classifier.py','contact_output_policy.py','ge_rule_v2.py','table_geometry.py'])] for p in files]
    report={'video_path':str(video),'video_sha256':sha(video),'n_frames':fid,'fps':fps,'image_size':[w,h],
        'input_contract':'One video plus fixed application model assets; no annotations, reviewed calibration, or prior inference cache.',
        'human_annotation_inputs':[],'manual_axis_inputs':[],'prior_pose_or_track_cache_inputs':[],
        'models':{k:v.provenance for k,v in models.items()},'source_hashes':{str(p):sha(p) for p in sources},
        'pose_evidence_policy':'Full-frame and ROI branches are independently classified, then combined; no historical event flags.',
        'expensive_inference_scope':'Only complete fitted motion-change windows; legacy-only/incomplete-context candidates retain baseline type.',
        'contact_classifier':{'model_sha256':sha(classifier_path),'training_provenance':classifier['training_provenance'],
            'score_threshold':classifier['threshold'],'independent_holdout':False},
        'model_paths':{**{k:str(p) for k,p in assets.items()},'contact_classifier':str(classifier_path)},'candidate_count':len(cases),'automatic_pose_frames':len(people),
        'hit_suggestions':sum(c['event']=='racket_contact_candidate' for c in cases),
        'model_init_seconds':initialised-started,'inference_seconds':time.monotonic()-initialised,
        'elapsed_seconds':time.monotonic()-started,'production_exclusion_enabled':False,
        'experimental_output_policy':policy,'experimental_landings_count':len(experimental),'torch_threads':threads,
        'video_timestamp_repairs':list(StartupPTSCapture.repairs),
        'camera_calibration_verified':False,'independent_holdout':False,'pipeline_status':'experimental_video_only_prototype'}
    save(out/'run.json',report);print('Saved',out,flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();destination=a.output or ROOT/'outputs/video_only_contacts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run(a.video,destination)

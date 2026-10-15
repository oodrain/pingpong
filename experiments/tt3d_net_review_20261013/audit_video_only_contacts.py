"""Post-inference label comparison, plus clearly separated automatic-cache regression."""
import argparse, hashlib, json, sys, math
from pathlib import Path
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));ROOT=PROJECT.parents[1]
from development.detection.video_contact_pipeline import classify_candidates,merge_pose_alternatives
from development.bounce.wrist_evidence import iou
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(pilot,fresh,out):
    out.mkdir(parents=True,exist_ok=False)
    roi_path=pilot/'remaining_contact_diagnosis_01/roi_evidence.json';inference_path=pilot/'public_regression_01/inference.json'
    roi=read(roi_path);inference=read(inference_path)
    geometry_path=PROJECT/'outputs/experiments/tt3d_geometry_refinement_20261013/runtime_check_02/geometry_samples.jsonl';geometry=lines(geometry_path)
    sources={str(p):sha(p) for p in (roi_path,inference_path,geometry_path)};cases=[]
    # Project-local reviewed axes and all prior flags are deliberately ignored.
    for vid in sorted({r['video_id'] for r in roi['cases']}):
        metadata=next(v for v in inference['videos'] if v['video_id']==vid)
        manifest=read(ROOT/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json')
        entry=next(v for v in manifest['videos'] if v['video_id']==vid)
        track=ROOT/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories'/f'{vid}.jsonl'
        video=ROOT/'participant/participant/pingpang/public_data'/entry['path']
        pose_path=pilot/'remaining_contact_diagnosis_01'/f'{vid}_pose_probes.json'
        full_pose_path=pilot/'temporal_contact_01'/f'{vid}_poses.json'
        for p in (track,video,pose_path,full_pose_path):sources[str(p)]=sha(p)
        for p in (track,video):
            if sources[str(p)]!=roi['sources'][str(p)]:raise ValueError('Changed cached evidence input')
        tracks=lines(track);cap=cv2.VideoCapture(str(video));times=[]
        while True:
            ok,_=cap.read()
            if not ok:break
            times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
        cap.release()
        if len(times)!=len(tracks) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Cache timestamp mismatch')
        for r,t in zip(tracks,times):
            r['timestamp_s']=t-times[0];r['in_image']=bool(r.get('detected') and 0<=r['x']<entry['width'] and 0<=r['y']<entry['height'])
        poses={int(k):v for k,v in read(pose_path)['frames'].items()}
        full_pose=read(full_pose_path)
        if full_pose['video_sha256']!=sources[str(video)]:raise ValueError('Full-frame poses belong to another video')
        full_poses={int(k):v for k,v in full_pose['frames'].items()}
        samples=[r for r in geometry if r['group']=='public' and r['video_id']==vid]
        results=classify_candidates(tracks,samples,[],metadata['proposal_result'],poses,{})
        results=merge_pose_alternatives(results,classify_candidates(tracks,samples,[],metadata['proposal_result'],full_poses,{}))
        selected={c['seed_frame'] for c in roi['cases'] if c['video_id']==vid}
        cases.extend(dict(c,video_id=vid,case_id=f"{vid}_{c['seed_frame']}") for c in results if c['seed_frame'] in selected)
    frozen={'cases':cases,'sources':sources,'manual_axes_used':False,'previous_hit_flags_used':False,
            'automatic_racket_detector_in_this_cache_regression':False,'fresh_end_to_end_inference':False,
            'scope':'31 already reviewed development candidates; automatic ball/table/pose caches only.',
            'method_designed_after_prior_human_review':True,'independent_holdout':False}
    save(out/'automatic_cache_predictions.json',frozen)
    # Labels opened only after predictions are persisted. Five spatial labels do
    # not alter any candidate decision or model parameter.
    sys.path.insert(0,str(ROOT/'development/annotation'))
    from evaluate_auxiliary_reviews import flag_metrics,table_pool_metrics
    from racket_evidence_core import RacketJournal
    human_path=pilot/'human_event_evaluation_01/human_reviews_snapshot.json';human=read(human_path)
    lookup={c['case_id']:c for c in cases};rows=[{'case_id':h['case_id'],'human_label':h['candidate_label'],
        'automatic_flag':lookup[h['case_id']]['decision']['review_hit_suggestion']} for h in human['records']]
    evaluation={'prediction_sha256':sha(out/'automatic_cache_predictions.json'),'human_snapshot_sha256':sha(human_path),
        'flags':flag_metrics(rows,'automatic_flag'),'table_pool':table_pool_metrics(rows,'automatic_flag'),'cases':rows,
        'scope':frozen['scope'],'fresh_end_to_end_inference':False,'independent_holdout':False}
    save(out/'cache_evaluation.json',evaluation)
    journal=RacketJournal();reviews=journal.latest()
    if any(reviews.get(f,{}).get('state',{}).get('status')!='completed' for f in journal.pack['frame_ids']):raise ValueError('Spatial review incomplete')
    # Bind the fresh run separately from the cache regression above.
    actual=read(fresh/'run.json');tracks=lines(fresh/'ball_tracks.jsonl');rackets=read(fresh/'racket_frames.json');spatial=[]
    for f in journal.pack['frame_ids']:
        state=reviews[f]['state'];ball=tracks[f];pred=rackets.get(str(f),[]);manual=state['racket_bbox_xyxy']
        error=math.hypot(ball['x']-state['ball_xy'][0],ball['y']-state['ball_xy'][1]) if ball['detected'] and state['ball_xy'] else None
        overlap=max((iou(manual,p['box']) for p in pred),default=0) if manual else None
        spatial.append({'frame_id':f,'ball_error_px':error,'automatic_racket_boxes':pred,'human_spatial_reference':state,
            'maximum_racket_face_box_iou':overlap,'warning':'Human reference marks face only; detector boxes may include handle. This is a localisation diagnostic, not detector mAP.'})
    fresh_cases=read(fresh/'event_candidates.json')['cases'];target=next(c for c in fresh_cases if c['seed_frame']==53)
    report={'fresh_run_sha256':sha(fresh/'run.json'),'fresh_candidates_sha256':sha(fresh/'event_candidates.json'),
        'spatial_journal_sha256':sha(journal.journal),'spatial_pack_sha256':sha(journal.path),
        'labels_read_after_inference':True,'human_coordinates_used_for_inference':False,'model_trained_on_these_five_frames':False,
        'fresh_video_frames':actual['n_frames'],'fresh_automatic_pose_frames':actual['automatic_pose_frames'],
        'fresh_inference_seconds':actual['inference_seconds'],'seed_53_automatic_event':target['event'],
        'spatial_cases':spatial,'remaining_issue':'Actual racket face is not reliably localised across the contact window; hand/blur detections cannot confirm a hit.',
        'independent_holdout':False,'production_exclusion_enabled':False}
    save(out/'spatial_diagnostic.json',report)
    (out/'human_spatial_journal_snapshot.jsonl').write_bytes(journal.journal.read_bytes())
    (out/'source_audit_video_only_contacts.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'automatic_cache_flags':evaluation['flags'],'automatic_cache_table_pool':evaluation['table_pool'],
        'fresh_case_53':target['event'],'ball_errors_px':[r['ball_error_px'] for r in spatial]},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('pilot','fresh','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.pilot.resolve(),a.fresh.resolve(),a.output.resolve())

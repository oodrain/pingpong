"""Fuse cached pose evidence conservatively; never promote 2D proximity to contact."""
import argparse
import json
from pathlib import Path
from fit_contact_pilot import sha

def fuse(event,wrist):
    status=wrist['status'];trajectory=event['decision']['event'];fit=event.get('fit')
    if trajectory=='tracking_anomaly_possible':return 'tracking_anomaly_possible'
    if status!='multi_frame_wrist_proximity':return trajectory
    # Even a boundary-constrained low-residual table hypothesis can explain a bounce
    # beside a player's hand. Preserve this conflict instead of declaring a hit.
    if trajectory=='table_contact_candidate' or (fit is not None and fit['rmse_px']<=3):return 'table_wrist_evidence_conflict'
    if trajectory=='off_table_motion_change':return 'racket_contact_candidate'
    return 'contact_type_uncertain'

def run(events,wrists,source,out):
    e=json.loads(events.read_text(encoding='utf-8'));w=json.loads(wrists.read_text(encoding='utf-8'))
    if w['source_events_sha256']!=sha(events):raise ValueError('Pose evidence mismatch')
    out.mkdir(parents=True,exist_ok=False);cases=[]
    for c in w['cases']:
        video=next(v for v in e['videos'] if v['video_id']==c['video_id']);event=next(v for v in video['events'] if v['seed_frame']==c['seed_frame'])
        cases.append({'video_id':c['video_id'],'seed_frame':c['seed_frame'],'trajectory_class':event['decision']['event'],
                      'wrist_status':c['status'],'near_frame_ids':c['near_frame_ids'],'combined_class':fuse(event,c),'adopt':False})
    # Human review is evaluation only. A "not contact" label does not specify racket.
    evaluation=[]
    for review in e['reviewed_evaluation']:
        video,fid=review['case_id'].rsplit('_',1);matches=[c for c in cases if c['video_id']==video and abs(c['seed_frame']-int(fid))<=2]
        evaluation.append({'case_id':review['case_id'],'human_decision':review['human_decision'],'nearby_results':matches})
    report={'sources':{str(events):sha(events),str(wrists):sha(wrists),str(source):sha(source)},'fusion_script_sha256':sha(__file__),
            'person_role_not_verified':True,'racket_tips_not_detected':True,'no_contact_confirmations':True,
            'human_labels_used_for_inference':False,'production_events_modified':False,'cases':cases,'reviewed_evaluation':evaluation}
    (out/'fusion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(c['video_id'],c['seed_frame'],c['wrist_status'],c['combined_class']) for c in cases],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True);p.add_argument('--wrists',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.events,a.wrists,a.source,a.output)

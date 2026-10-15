"""Audit cached training evidence against reviewed candidate-window events.

No detector inference or reserved/public inputs. Secondary reviewed events are
kept; absence from stage candidate labels is never treated as a negative label.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]; ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.detection.video_contact_pipeline import classify_candidates,merge_pose_alternatives
from development.bounce.contact_classifier import load,apply_model
from development.bounce.contact_output_policy import experimental_landings
from development.bounce.person_motion_contact import local_table_position
from development.bounce.trajectory_events import select_window


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def distance(seed,interval):return max(interval[0]-seed,seed-interval[1],0)


def run(out):
    protocol=load_protocol();cache=DEFAULT/'train_event_review_01/automatic_inputs'
    ids=sorted(p.name.removesuffix('_tracks.jsonl') for p in cache.glob('*_tracks.jsonl'))
    records=select(protocol,ids,'training');metadata={r['video_id']:r for r in records}
    snapshot=DEFAULT/'train_classifier_01/human_reviews_snapshot.json'
    model=load(DEFAULT/'train_classifier_01/model.json')
    if sha(snapshot)!=model['training_provenance']['human_snapshot_sha256']:
        raise ValueError('Human snapshot differs from frozen training run')
    reviews=read(snapshot)['records'];events={};scopes=defaultdict(list)
    for r in reviews:
        if r['status']!='completed':raise ValueError('Incomplete reviewed window')
        scopes[r['video_id']].append(r['state']['review_scope'])
        for event in r['state']['events']:
            key=(r['video_id'],event['event_type'],*event['interval_frames'])
            events.setdefault(key,{'video_id':r['video_id'],'event_type':event['event_type'],
                                  'interval_frames':event['interval_frames'],'review_case_ids':[]})['review_case_ids'].append(r['case_id'])
    select(protocol,list(scopes),'training')
    all_stage=lines(DEFAULT/'baseline_replay_03/label_snapshots/candidate_events.jsonl')
    stage=[r for r in all_stage if r['video_id'] in metadata]
    predictions=defaultdict(list)
    for r in lines(DEFAULT/'baseline_replay_03/predictions.jsonl'):
        if r['video_id'] in metadata:predictions[r['video_id']].append(r)
    out.mkdir(parents=True,exist_ok=False);all_cases=[];outputs={};misses=[];input_hashes={str(snapshot):sha(snapshot)}
    for vid in ids:
        paths=[cache/f'{vid}_tracks.jsonl',cache/f'{vid}_geometry.json',cache/f'{vid}_proposals.json']
        tracks=lines(paths[0]);samples=read(paths[1]);proposals=read(paths[2])
        input_hashes.update({str(p):sha(p) for p in paths})
        evidence=DEFAULT/'train_features_01'/f'{vid}_automatic_evidence.json'
        data=read(evidence) if evidence.exists() else {}
        if data and data['video_sha256']!=metadata[vid]['video_sha256']:raise ValueError('Wrong evidence video')
        if data:input_hashes[str(evidence)]=sha(evidence)
        people={int(f):p for f,p in data.get('poses',{}).items()}
        full={int(f):p for f,p in data.get('full_poses',{}).items()}
        rackets={int(f):p for f,p in data.get('rackets',{}).items()}
        base=predictions[vid]
        cases=classify_candidates(tracks,samples,base,proposals,people,rackets)
        cases=merge_pose_alternatives(cases,classify_candidates(tracks,samples,base,proposals,full,rackets))
        cases=apply_model(model,cases)
        relevant=[e for e in events.values() if e['video_id']==vid]
        for c in cases:
            seed=c['seed_frame'];needed=set(range(max(0,seed-2),min(len(tracks),seed+3)))
            c.update(video_id=vid,case_id=f'{vid}_{seed}',source_group=vid.split('_')[0],
                     automatic_evidence_complete=needed<=set(people)&set(full)&set(rackets),
                     automatic_evidence_frames=sorted(needed&set(people)&set(full)&set(rackets)))
            matches=[e for e in relevant if distance(seed,e['interval_frames'])<=1]
            types={e['event_type'] for e in matches}
            c['reviewed_event_type']=next(iter(types)) if len(types)==1 else 'ambiguous' if types else None
            c['reviewed_event_matches']=matches
            c['stage_table_reference_matches']=[g for g in stage if g['video_id']==vid and abs(g['frame_id']-seed)<=1
                and tracks[seed].get('detected') and ((g['x']-tracks[seed]['x'])**2+(g['y']-tracks[seed]['y'])**2)**.5<=50]
        result,decisions=experimental_landings(tracks,base,proposals,cases,True,True)
        outputs[vid]={'baseline':base,'experimental':result,'auxiliary_decisions':decisions,
                      'geometry_samples':len(samples),'usable_geometry_samples':sum(bool(s.get('usable')) for s in samples),
                      'cached_evidence_frames':len(people)}
        all_cases.extend(cases)
        for e in relevant:
            if e['event_type']!='table_bounce':continue
            interval=e['interval_frames'];old=[p for p in base if distance(p['frame_id'],interval)<=1]
            new=[p for p in result if distance(p['frame_id'],interval)<=1]
            nearby=[c for c in cases if distance(c['seed_frame'],interval)<=2]
            if not new or any(c['event']=='racket_contact_candidate' for c in nearby):
                anchor=interval[0];p=next((p for p in proposals['proposals'] if distance(p['frame_id'],interval)<=1),None)
                reasons=[]
                if old and not new:reasons.append('old_temporal_match_removed')
                if any(c['event']=='racket_contact_candidate' for c in nearby):reasons.append('nearby_hit_suggestion')
                if p is None:reasons.append('no_proposal_within_temporal_tolerance')
                else:
                    screen=next(d for d in decisions if d['seed_frame']==p['frame_id']);reasons+=screen['reasons']
                position=local_table_position(samples,tracks[anchor],anchor)
                if not position.get('available'):reasons.append('geometry_or_ball_unavailable_at_reviewed_interval')
                misses.append({**e,'baseline_temporal_match':bool(old),'experimental_temporal_match':bool(new),
                    'reasons':sorted(set(reasons)),'nearby_case_ids':[c['case_id'] for c in nearby],
                    'position':position,'nearest_proposal_frame':min((p['frame_id'] for p in proposals['proposals']),
                        key=lambda f:distance(f,interval),default=None),
                    'nearby_tracks':tracks[max(0,anchor-4):min(len(tracks),interval[1]+5)]})
    primary_ids={r['case_id'] for r in reviews}
    report={'split_id':protocol['split_id'],'cache_video_ids':ids,'cache_videos':len(ids),'reviewed_window_count':len(reviews),
            'reviewed_unique_event_counts':dict(Counter(e['event_type'] for e in events.values())),
            'automatic_case_count':len(all_cases),'complete_pose_racket_case_count':sum(c['automatic_evidence_complete'] for c in all_cases),
            'reviewed_misfiltered_bounces':[c['case_id'] for c in all_cases if c['reviewed_event_type']=='table_bounce' and c['event']=='racket_contact_candidate'],
            'stage_reference_hit_conflicts':[c['case_id'] for c in all_cases if c['stage_table_reference_matches'] and c['event']=='racket_contact_candidate'],
            'primary_case_count':sum(c['case_id'] in primary_ids for c in all_cases),
            'bounce_miss_or_hit_conflict_count':len(misses),'miss_reason_counts':dict(Counter(r for m in misses for r in m['reasons'])),
            'input_hashes':input_hashes,'classifier_sha256':sha(DEFAULT/'train_classifier_01/model.json'),
            'detector_inference_calls':0,'reserved_or_public_inputs_used':False,
            'metrics_scope':'Reviewed candidate windows only; temporal coverage is not full-video F1. Stage table references are provisional.'}
    save(out/'audit.json',report);save(out/'cases.json',{'cases':all_cases});save(out/'window_events.json',{'events':list(events.values()),'scopes':scopes})
    save(out/'outputs.json',outputs);save(out/'bounce_misses.json',{'cases':misses})
    (out/'source_audit_cached_train_contacts.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)

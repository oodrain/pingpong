"""Merge completed train windows and reuse automatic curved-flight evidence."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'));sys.path.insert(0,str(ROOT/'development/annotation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from auxiliary_event_core import ReviewJournal
from development.bounce.curved_contact_candidates import propose_curved_contacts
from development.bounce.curved_contact_classifier import features,FEATURES,EXTENDED_FEATURES
from development.bounce.person_motion_contact import local_table_position
from development.bounce.curved_contact_context import contextual_features,ballistic_features,net_context_features,CONTEXT_FEATURES,PHYSICS_FEATURES,NET_FEATURES

BASE=ROOT/'outputs/contact_rule_revision_20261014'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(out,extended=False,additional_packs=(),context=False,physics=False,net=False):
    physics=physics or net
    protocol=load_protocol();store=ReviewJournal(BASE/'train_followup_review_02/review_pack.json');snapshot=store.export();new=read(snapshot)
    if new['completed_cases']!=10 or new['known_public_development_set']:raise ValueError('Wrong/incomplete train followup')
    old_path=DEFAULT/'train_classifier_01/human_reviews_snapshot.json'
    old=read(old_path);reviews=old['records']+new['records'];names=NET_FEATURES if net else PHYSICS_FEATURES if physics else CONTEXT_FEATURES if context else EXTENDED_FEATURES if extended else FEATURES
    additional=[]
    for pack in additional_packs:
        extra_store=ReviewJournal(pack);extra_snapshot=extra_store.export();extra=read(extra_snapshot)
        if extra['completed_cases']!=extra['total_cases'] or extra['known_public_development_set']:
            raise ValueError('Incomplete or non-training additional review pack')
        if any(r['candidate_label']=='uncertain' for r in extra['records']):
            raise ValueError('Uncertain additional windows must be isolated before training')
        reviews+=extra['records'];additional.append((extra_store,extra_snapshot))
    if len({r['case_id'] for r in reviews})!=len(reviews):raise ValueError('Duplicate reviewed candidate IDs')
    select(protocol,list(dict.fromkeys(r['video_id'] for r in reviews)),'training')
    events={};scopes=defaultdict(list);timing_offsets=[]
    for r in reviews:
        scopes[r['video_id']].append(r['state']['review_scope'])
        primary=next((e for e in r['state']['events'] if e['event_id']==r['state']['primary_event_id']),None)
        if primary:
            lo,hi=primary['interval_frames'];dist=max(lo-r['seed_frame'],r['seed_frame']-hi,0)
            if dist>2:timing_offsets.append({'case_id':r['case_id'],'candidate_frame':r['seed_frame'],'primary_interval':primary['interval_frames'],
                'distance':dist,'action':'Keep reviewed event interval, not a candidate-frame table-positive label'})
        for e in r['state']['events']:
            key=(r['video_id'],e['event_type'],*e['interval_frames'])
            events.setdefault(key,{'video_id':r['video_id'],'event_type':e['event_type'],'interval_frames':e['interval_frames'],'review_case_ids':[]})['review_case_ids'].append(r['case_id'])
    event_list=list(events.values());cache=DEFAULT/'train_event_review_01/automatic_inputs';automatic={};examples=[];hashes={str(snapshot):sha(snapshot),str(store.journal):sha(store.journal),str(store.path):sha(store.path),str(old_path):sha(old_path)}
    for extra_store,extra_snapshot in additional:
        hashes.update({str(p):sha(p) for p in (extra_store.path,extra_store.journal,extra_snapshot)})
    image_sizes={}
    for path in cache.glob('*_tracks.jsonl'):
        vid=path.name.removesuffix('_tracks.jsonl');metadata=select(protocol,[vid],'training')[0];tracks=lines(path);samples=read(cache/f'{vid}_geometry.json')
        image_sizes[vid]=[metadata['width'],metadata['height']]
        q=next(s['corners'] for s in samples if s.get('usable'));result=propose_curved_contacts(tracks,q)
        hashes.update({str(p):sha(p) for p in [path,cache/f'{vid}_geometry.json']})
        decorated=[]
        for p in result['proposals']:
            seed=p['frame_id'];position=local_table_position(samples,tracks[seed],seed)
            vector=(net_context_features(p,position,q,tracks,samples,image_sizes[vid]) if net else ballistic_features(p,position,q,tracks,samples,image_sizes[vid]) if physics else contextual_features(p,position,q,tracks) if context else features(p,position,q,names)).tolist()
            item={'video_id':vid,'case_id':f'{vid}_{seed}','source_group':vid.split('_')[0],'seed_frame':seed,'proposal':p,'position':position,'vector':vector}
            decorated.append(item)
            if not any(lo<=seed<=hi for lo,hi in scopes.get(vid,[])):continue
            matches=[e for e in event_list if e['video_id']==vid and max(e['interval_frames'][0]-seed,seed-e['interval_frames'][1],0)<=1]
            types={e['event_type'] for e in matches}
            item=dict(item,label_table_contact='table_bounce' in types,event_types=sorted(types),event_matches=matches,
                      negative_basis=None if 'table_bounce' in types else 'No matching table event in a completed, fully annotated candidate window')
            examples.append(item)
        automatic[vid]=decorated
    out.mkdir(parents=True,exist_ok=False)
    if physics:
        for relative in ('bounce/automatic_ballistic_contact.py','bounce/automatic_net_zone.py','bounce/curved_contact_context.py','bounce/curved_contact_candidates.py','detection/tt3d_camera_estimation.py','detection/auto_table_geometry.py'):
            target=out/('feature_'+Path(relative).name);target.write_bytes((PROJECT/'development'/relative).read_bytes());hashes[str(target.resolve())]=sha(target)
    save(out/'dataset.json',{'features':names,'examples':examples,'events':event_list,'scopes':scopes,'timing_offsets':timing_offsets,
        'human_primary_cases':len(reviews),'image_sizes':image_sizes,'human_label_counts':dict(Counter(r['candidate_label'] for r in reviews)),
        'reviewed_event_counts':dict(Counter(e['event_type'] for e in event_list)),'sources':hashes,'split_id':protocol['split_id'],
        'detector_inference_calls':0,'negative_scope_assumption':'User recorded every event in completed review windows; these are still candidate-window labels, not whole-video GT.'})
    save(out/'automatic_curved_cases.json',{'videos':automatic})
    (out/'followup_snapshot.json').write_bytes(snapshot.read_bytes());(out/'followup_journal_snapshot.jsonl').write_bytes(store.journal.read_bytes())
    for index,(extra_store,extra_snapshot) in enumerate(additional):
        (out/f'additional_{index}_snapshot.json').write_bytes(extra_snapshot.read_bytes())
        (out/f'additional_{index}_journal_snapshot.jsonl').write_bytes(extra_store.journal.read_bytes())
    if additional:
        save(out/'additional_review_provenance.json',{'packs':[str(s.path) for s,_ in additional],
            'user_clarifications':[{'case_id':'16_001_45','meaning':'Normal flight; not a table bounce or racket contact at this candidate. Journal other label retained.',
                                   'source':'Explicit user message after completing six windows'},
                                  {'case_id':'06_007_13','meaning':'Net collision; other event label retained.','source':'Explicit user clarification'},
                                  {'case_id':'11_005_68','meaning':'Net graze; other event label retained.','source':'Explicit user clarification'}]})
    (out/'source_build_reviewed_curved_dataset.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'label_counts':dict(Counter(r['candidate_label'] for r in reviews)),
        'events':dict(Counter(e['event_type'] for e in event_list)),'curved_examples':len(examples),
        'curved_positive':sum(e['label_table_contact'] for e in examples),'curved_negative':sum(not e['label_table_contact'] for e in examples),
        'timing_offsets':timing_offsets},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--extended',action='store_true');p.add_argument('--context',action='store_true');p.add_argument('--physics',action='store_true');p.add_argument('--net',action='store_true');p.add_argument('--additional-review-pack',type=Path,action='append',default=[]);a=p.parse_args();run(a.output,a.extended,a.additional_review_pack,a.context,a.physics,a.net)

"""Prepare small train-only followup from cached automatic proposal evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.bounce.curved_contact_candidates import propose_curved_contacts
from development.bounce.contact_consensus import cosine
from development.bounce.person_motion_contact import local_table_position


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(a):
    audit=read(a.audit/'audit.json');protocol=load_protocol();records=select(protocol,audit['cache_video_ids'],'training')
    metadata={r['video_id']:r for r in records};scopes=read(a.audit/'window_events.json')['scopes']
    selected=[]
    for c in read(a.selection/'cv_added.json')['cases']:
        vid=c['video_id'];seed=c['seed_frame']
        if any(lo<=seed<=hi for lo,hi in scopes.get(vid,[])):continue
        selected.append({'video_id':vid,'seed_frame':seed,'reason':'new_curved_auxiliary_outside_previous_review_scope',
                         'fit_bic_gain':c['fit']['bic_gain']})
    # Include weak-direction changes in usable current table projection,
    # without looking at stage/public/reserved event labels or using video IDs
    # as prediction features. Spread the six added windows across recordings.
    cache=DEFAULT/'train_event_review_01/automatic_inputs';options=[]
    for vid in audit['cache_video_ids']:
        tracks=lines(cache/f'{vid}_tracks.jsonl');samples=read(cache/f'{vid}_geometry.json')
        q=next(s['corners'] for s in samples if s.get('usable'))
        for p in propose_curved_contacts(tracks,q)['proposals']:
            seed=p['frame_id'];pos=local_table_position(samples,tracks[seed],seed)
            c=cosine(p['velocity_before_px_s'],p['velocity_after_px_s'])
            if not pos.get('available') or pos.get('outside_projection') or c is None or not -.5<c<.5:continue
            if any(lo-2<=seed<=hi+2 for lo,hi in scopes.get(vid,[])):continue
            options.append({'video_id':vid,'seed_frame':seed,'reason':'weak_image_direction_change_inside_table_projection',
                            'fit_bic_gain':p['bic_gain'],'image_direction_cosine':c})
    weak=[];counts=Counter()
    for unique_groups in (True,False):
        for p in sorted(options,key=lambda p:-p['fit_bic_gain']):
            group=p['video_id'].split('_')[0]
            if len(weak)>=6:break
            if unique_groups and counts[group]:continue
            if any(p['video_id']==s['video_id'] and abs(p['seed_frame']-s['seed_frame'])<=10 for s in selected+weak):continue
            weak.append(p);counts[group]+=1
    selected+=weak
    if len(selected)!=10:raise ValueError('Unexpected followup count; inspect selection before preparing user work')
    a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for index,p in enumerate(selected):
        vid=p['video_id'];r=metadata[vid];seed=p['seed_frame'];track=cache/f'{vid}_tracks.jsonl'
        if sha(r['video_path'])!=r['video_sha256']:raise ValueError('Frozen train video changed')
        cases.append({'case_id':f'{vid}_{seed}','video_id':vid,'seed_frame':seed,'source_group':r['source_group'],
            'split':'train','angle':'训练补充 / 来源 '+r['source_group'],'video_path':r['video_path'],'video_sha256':r['video_sha256'],
            'tracks_path':str(track),'tracks_sha256':sha(track),'width':r['width'],'height':r['height'],'n_frames':r['n_frames'],
            'default_scope':[max(0,seed-12),min(r['n_frames']-1,seed+12)],'priority':index,'selection_reason':p['reason']})
    sources={str(a.selection/'policy.json'):sha(a.selection/'policy.json'),str(a.selection/'selection.json'):sha(a.selection/'selection.json'),
             str(a.audit/'audit.json'):sha(a.audit/'audit.json'),str(DEFAULT/'split_manifest.json'):sha(DEFAULT/'split_manifest.json')}
    snapshot=a.output/'source_prepare_contact_consensus_review.py';snapshot.write_bytes(Path(__file__).read_bytes());sources[str(snapshot)]=sha(snapshot)
    pack={'schema_version':1,'purpose':'train_only_event_labels','split_id':protocol['split_id'],'sources':sources,'cases':cases,
          'known_public_development_set':False,'human_labels_prefilled':False,'selection_uses_train_review_scope_metadata':True,
          'selection_uses_public_or_reserved_event_gt':False,'validation_videos_included':0,'frame_indexing':'zero_based_decoded_frames',
          'selection_policy':'Four unreviewed curved additions plus six weak-direction changes, train sources only; exclude previously reviewed centres.',
          'review_instructions':'Record all events in each window, then identify the event for this candidate; no event or uncertain are valid. Adjacent bounce and hit remain separate events.'}
    save(a.output/'review_pack.json',pack);save(a.output/'selection.json',{'selected':selected,'pack_sha256':sha(a.output/'review_pack.json')})
    print(json.dumps({'cases':len(cases),'ids':[c['case_id'] for c in cases],'sources':sorted({c['source_group'] for c in cases})},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

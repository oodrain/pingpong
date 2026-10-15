"""Freeze dense TCN training sequences from automatic caches and completed windows."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'));sys.path.insert(0,str(ROOT/'development/annotation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from auxiliary_event_core import ReviewJournal
from development.bounce.frame_event_features import automatic_frame_features,interval_targets,FEATURES,EVENT_TYPES

REV=ROOT/'outputs/contact_rule_revision_20261014'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def effective_reviews(reviews):
    """Full-video human adjudication replaces stale window events, never unions."""
    full={}
    for review in reviews:
        if review.get('review_mode')!='complete_video' or review['state']['status']!='completed':continue
        if review['video_id'] in full:raise ValueError('Multiple complete-video adjudications require explicit reconciliation')
        full[review['video_id']]=review
    superseded=[r for r in reviews if r['video_id'] in full and r is not full[r['video_id']]]
    effective=[r for r in reviews if r not in superseded and r['state']['status']=='completed']
    return effective,superseded


def run(output,additional_packs=()):
    protocol=load_protocol();output.mkdir(parents=True,exist_ok=False);reviews=[];sources={};pack_records=[]
    packs=[DEFAULT/'train_event_review_01/review_pack.json']+[REV/name/'review_pack.json' for name in ('train_followup_review_02','train_false_positive_review_04','train_net_contact_review_12')]
    packs+=list(additional_packs)
    for index,path in enumerate(packs):
        store=ReviewJournal(path);snapshot=store.export();data=read(snapshot)
        if data['known_public_development_set'] or store.pack.get('purpose')!='train_only_event_labels':raise ValueError('Non-train review source')
        completed=[r for r in data['records'] if r['status']=='completed']
        if completed:select(protocol,sorted({r['video_id'] for r in completed}),'training')
        reviews+=completed;target=output/f'reviews_{index}_snapshot.json';target.write_bytes(snapshot.read_bytes())
        sources[str(target)]=sha(target);sources[str(path)]=sha(path)
        if store.journal.exists():
            journal=output/f'reviews_{index}_journal_snapshot.jsonl';journal.write_bytes(store.journal.read_bytes());sources[str(journal)]=sha(journal)
        pack_records.append({'pack_path':str(path),'completed_cases':len(completed),'total_cases':len(data['records']),
                             'journal_sha256_at_freeze':data['journal_sha256']})
    if len({r['case_id'] for r in reviews})!=len(reviews):raise ValueError('Duplicate training review cases')
    # A completed full-video reinspection supersedes older window labels for
    # that video; deleted/retimed events must not silently reappear by union.
    reviews,superseded=effective_reviews(reviews)
    events={};by_video=defaultdict(list)
    for review in reviews:
        by_video[review['video_id']].append(review)
        for event in review['state']['events']:
            key=(review['video_id'],event['event_type'],*event['interval_frames'])
            events.setdefault(key,{'video_id':review['video_id'],'event_type':event['event_type'],
                                  'interval_frames':event['interval_frames'],'review_case_ids':[]})['review_case_ids'].append(review['case_id'])
    event_list=list(events.values());records=select(protocol,sorted(by_video),'training');cache=DEFAULT/'train_event_review_01/automatic_inputs'
    sequences=[];fully_visited=0;masked_frame_total=0
    for record in records:
        vid=record['video_id'];tracks_path=cache/f'{vid}_tracks.jsonl';geometry_path=cache/f'{vid}_geometry.json'
        tracks=lines(tracks_path);samples=read(geometry_path)
        if len(tracks)!=record['n_frames']:raise ValueError('Training cache length mismatch')
        sources.update({str(p):sha(p) for p in (tracks_path,geometry_path)})
        if sha(record['video_path'])!=record['video_sha256']:raise ValueError('Frozen training media changed')
        ev=[e for e in event_list if e['video_id']==vid]
        x=automatic_frame_features(tracks,samples,(record['width'],record['height']))
        y,mask=interval_targets(len(tracks),by_video[vid],ev)
        evaluation_mask=mask[:,0].copy()
        for e in ev:
            if e['event_type']=='table_bounce':
                lo,hi=e['interval_frames'];evaluation_mask[max(0,lo-1):min(len(tracks),hi+2)]=True
        for review in by_video[vid]:
            state=review['state'];lo,hi=state['review_scope']
            fully_visited+=int(set(range(lo,hi+1))<=set(state.get('visited_frames',[])))
        path=output/f'{vid}.npz';np.savez_compressed(path,features=x,targets=y,loss_mask=mask,evaluation_mask=evaluation_mask)
        sources[str(path)]=sha(path);masked_frame_total+=int(mask.any(1).sum())
        sequences.append({'video_id':vid,'source_group':record['source_group'],'path':str(path),
            'tracks_path':str(tracks_path),'geometry_path':str(geometry_path),'n_frames':len(tracks),
            'width':record['width'],'height':record['height'],'supervised_frames':int(mask.any(1).sum())})
    feature_source=output/'source_frame_event_features.py';feature_source.write_bytes((PROJECT/'development/bounce/frame_event_features.py').read_bytes());sources[str(feature_source)]=sha(feature_source)
    sources[str(DEFAULT/'split_manifest.json')]=sha(DEFAULT/'split_manifest.json')
    dataset={'schema_version':1,'split_id':protocol['split_id'],'features':FEATURES,'event_types':list(EVENT_TYPES),
        'sequences':sequences,'events':event_list,'reviews':reviews,'review_packs':pack_records,'sources':sources,
        'completed_reviews':len(reviews),'fully_visited_reviews':fully_visited,'supervised_frame_total':masked_frame_total,
        'superseded_window_review_case_ids':[r['case_id'] for r in superseded],
        'event_counts':dict(Counter(e['event_type'] for e in event_list)),
        'background_label_policy':'Visited interiors of completed, non-uncertain review windows only; trim two-frame boundaries. Explicit positive intervals retained; every unreviewed frame masked.',
        'cv_evaluation_policy':'Same trusted touch frames for TCN and rules; allow one-frame neighbours around explicit touch intervals. Temporal one-to-one interval matching, no manual contact coordinates.',
        'actual_timestamp_features':True,'detector_inference_calls':0,'human_geometry_used_for_features':False,
        'known_public_or_reserved_labels_used':False,'candidate_gates_used_for_tcn_training':False}
    save(output/'dataset.json',dataset);(output/'source_build_frame_tcn_dataset.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:dataset[k] for k in ('completed_reviews','fully_visited_reviews','supervised_frame_total','event_counts')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--additional-pack',type=Path,action='append',default=[])
    args=parser.parse_args();run(args.output.resolve(),args.additional_pack)

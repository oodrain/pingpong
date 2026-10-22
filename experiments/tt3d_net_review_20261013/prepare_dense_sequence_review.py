"""Audit training OOF coverage and prepare six complete-video human reviews.

No reserved/public labels or predictions are opened. Prior human events are
draft references, not new confirmations; no human journal is written here.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT),str(ROOT/'development/evaluation'),str(ROOT/'development/annotation')]
from dataset_protocol import load_protocol,select,sha
from development.bounce.frame_event_features import decode_frame_events,interval_targets
from select_cached_contact_consensus import temporal_counts

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def run(args):
    dataset=read(args.dataset/'dataset.json');protocol=load_protocol()
    select(protocol,[s['video_id'] for s in dataset['sequences']],'training')
    for path,digest in dataset['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen dataset input changed: '+path)
    frozen=read(args.model/'model_freeze.json')
    if sha(args.model/'model.pt')!=frozen['model_sha256']:raise ValueError('Baseline TCN model changed')
    chosen=read(args.model/'selection.json')['selected'];sources={str(args.dataset/'dataset.json'):sha(args.dataset/'dataset.json'),
        str(args.model/'selection.json'):sha(args.model/'selection.json'),str(args.model/'model_freeze.json'):sha(args.model/'model_freeze.json')}
    counters=Counter();rows=[];sequence_rows=[];points=[];unchanged_masks=True
    for record in dataset['sequences']:
        vid=record['video_id'];path=args.model/(vid+'_cv_scores.npz');sources[str(path)]=sha(path)
        scores=np.load(path,allow_pickle=False)['scores'];arrays=np.load(record['path'],allow_pickle=False);tracks=lines(record['tracks_path'])
        reviews=[r for r in dataset['reviews'] if r['video_id']==vid];events=[e for e in dataset['events'] if e['video_id']==vid]
        y,mask=interval_targets(record['n_frames'],reviews,events)
        unchanged_masks &= bool(np.array_equal(y,arrays['targets']) and np.array_equal(mask,arrays['loss_mask']))
        output,decisions=decode_frame_events(scores[:,0],tracks,chosen['threshold'])
        trusted=[p for p in output if arrays['evaluation_mask'][p['frame_id']]]
        points += [dict(p,video_id=vid) for p in trusted]
        counters.update(scored_frames=len(tracks),reviewed_evaluation_frames=int(arrays['evaluation_mask'].sum()),
            all_oof_outputs=len(output),trusted_outputs=len(trusted),unreviewed_outputs=len(output)-len(trusted))
        local=[]
        for point in trusted:
            fid=point['frame_id']
            near=lambda kind:any(e['event_type']==kind and max(e['interval_frames'][0]-fid,fid-e['interval_frames'][1],0)<=1 for e in events)
            if near('table_bounce'):continue
            peak=next(d['peak_frame'] for d in decisions if d.get('included') and d['output_frame']==fid)
            row={'video_id':vid,'frame_id':fid,'peak_frame':peak,'touch_score_at_peak':float(scores[peak,0]),
                'hit_score_at_peak':float(scores[peak,1]),'near_reviewed_hit':near('racket_contact'),'near_reviewed_other':near('other')}
            local.append(row);rows.append(row)
        sequence_rows.append(dict(record,reviewed_evaluation_frames=int(arrays['evaluation_mask'].sum()),
            supervised_frames=int(arrays['loss_mask'].any(1).sum()),trusted_non_touch_peaks=len(local),oof_outputs=len(output)))
    if not unchanged_masks:raise ValueError('New unknown masking changed old frozen targets')
    groups=sorted({s['source_group'] for s in sequence_rows});selected=[]
    # One short clip per existing source, each with a reviewed OOF false touch.
    # Selection uses train-only coverage/errors, never held evaluation outcomes.
    for group in groups:
        options=[s for s in sequence_rows if s['source_group']==group and s['trusted_non_touch_peaks']]
        if not options:raise ValueError('No reviewed false-touch clip for source '+group)
        selected.append(min(options,key=lambda s:(s['n_frames'],-s['trusted_non_touch_peaks'],s['video_id'])))
    records=select(protocol,[s['video_id'] for s in selected],'training');cases=[]
    for meta,record in zip(records,selected):
        vid=record['video_id']
        for path,digest in [(meta['video_path'],meta['video_sha256']),
            (record['tracks_path'],dataset['sources'][record['tracks_path']])]:
            if sha(path)!=digest:raise ValueError('Selected review input changed: '+path)
        references=[]
        previous_frames=set();previous_scopes=set();reference_cases=[]
        for review in dataset['reviews']:
            if review['video_id']!=vid:continue
            state=review['state']
            if state['status']!='completed' or state.get('candidate_uncertain'):continue
            lo,hi=state['review_scope'];scope=set(range(lo,hi+1))
            previous_scopes |= scope;previous_frames |= scope & set(state.get('visited_frames',[]))
            reference_cases.append(review['case_id'])
        for event in dataset['events']:
            if event['video_id']!=vid:continue
            lo,hi=event['interval_frames'];kind=event['event_type']
            references.append({'event_id':f'prior_human_{vid}_{kind}_{lo}_{hi}','event_type':kind,'interval_frames':[lo,hi],
                'reference_source_case_ids':event['review_case_ids'],'reference_dataset_sha256':sources[str(args.dataset/'dataset.json')]})
        cases.append({'case_id':vid+'_complete_video','video_id':vid,'seed_frame':0,'angle':meta['official_angle'],
            'review_mode':'complete_video','video_path':meta['video_path'],'video_sha256':meta['video_sha256'],
            'tracks_path':record['tracks_path'],'tracks_sha256':sha(record['tracks_path']),
            'width':meta['width'],'height':meta['height'],'n_frames':meta['n_frames'],
            'default_scope':[0,meta['n_frames']-1],'priority':len(cases),'reference_events':references,
            'reference_visited_frames':sorted(previous_frames) if args.reuse_verified_coverage else [],
            'reference_completed_case_ids':reference_cases,'previous_scope_frames':sorted(previous_scopes)})
    args.output.mkdir(parents=True,exist_ok=False)
    audit={'split_id':protocol['split_id'],'sources':sources,'counts':dict(counters),'trusted_false_touch_count':len(rows),
        'trusted_false_touch_rows':rows,'same_scopes_temporal_metrics':temporal_counts(points,[e for e in dataset['events'] if e['event_type']=='table_bounce']),
        'selected_videos':[s['video_id'] for s in selected],'selection':'Shortest clip with a trusted OOF false touch per existing training recording source',
        'selected_frames':sum(s['n_frames'] for s in selected),'selected_previously_reviewed_evaluation_frames':sum(s['reviewed_evaluation_frames'] for s in selected),
        'reuses_verified_human_browsing':args.reuse_verified_coverage,
        'previous_human_scope_frames':sum(len(c['previous_scope_frames']) for c in cases),
        'reused_human_visited_frames':sum(len(c['reference_visited_frames']) for c in cases),
        'frames_requiring_new_browsing':sum(c['n_frames']-len(c['reference_visited_frames']) for c in cases),
        'old_targets_and_loss_masks_identical':unchanged_masks,'reserved_or_public_inputs_opened':False,
        'detector_inference_calls':0,'new_model_trained':False,'unreviewed_outputs_are_false_positives':False,
        'source_diversity_expanded':False,'limitation':'Six existing training sources only. Partial reviewed scopes and active selection are not unbiased full-video metrics.'}
    save(args.output/'coverage_audit.json',audit)
    pack={'schema_version':1,'purpose':'train_only_event_labels','split_id':protocol['split_id'],
        'sources':{str(args.output/'coverage_audit.json'):sha(args.output/'coverage_audit.json'),**sources},
        'cases':cases,'known_public_development_set':False,'validation_videos_included':0,'collect_other_subtypes':True,
        'human_labels_prefilled':True,'prefill_policy':'Only existing human labels, as draft references requiring complete-video reinspection',
        'prefilled_model_predictions':False,'frame_indexing':'zero_based_decoded_frames',
        'full_video_completion_requires_all_frames_visited':True,'unknown_intervals_are_unsupervised':True,
        'prior_human_coverage_reused':args.reuse_verified_coverage,
        'reference_review_dataset':str(args.dataset/'dataset.json')}
    save(args.output/'review_pack.json',pack)
    (args.output/'source_prepare_dense_sequence_review.py').write_bytes(Path(__file__).read_bytes())
    save(args.output/'freeze.json',{'review_pack_sha256':sha(args.output/'review_pack.json'),
        'coverage_audit_sha256':sha(args.output/'coverage_audit.json'),'human_journal_created':False})
    print(json.dumps({k:audit[k] for k in ('counts','same_scopes_temporal_metrics','selected_videos','selected_frames','selected_previously_reviewed_evaluation_frames')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();base=ROOT/'outputs/frame_tcn_baseline_20261014'
    parser.add_argument('--dataset',type=Path,default=base/'dataset_01');parser.add_argument('--model',type=Path,default=base/'train_01')
    parser.add_argument('--reuse-verified-coverage',action='store_true')
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

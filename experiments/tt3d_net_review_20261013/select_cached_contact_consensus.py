"""Select experimental rules using train-source CV and existing automatic caches."""
import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.bounce.contact_classifier import fit,features,apply_model
from development.bounce.contact_consensus import apply_contact_consensus,curved_auxiliary_landings
from development.bounce.curved_contact_candidates import propose_curved_contacts
from development.bounce.contact_output_policy import experimental_landings


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def restore_rules(cases):
    cases=deepcopy(cases)
    for c in cases:
        d=c['decision'];original=bool(d.get('baseline_flag') or d.get('person_entry_reversal_outside')
            or d.get('transient_wrist_impulse_outside') or d.get('automatic_racket_supplement')
            or c.get('full_frame_pose_decision',{}).get('review_hit_suggestion'))
        d['review_hit_suggestion']=original;d.pop('learned_hit_supplement',None);c.pop('learned_contact',None)
        c['event']='racket_contact_candidate' if original else 'legacy_table_candidate' if 'baseline' in c['source'] else 'contact_type_unresolved'
    return cases


def temporal_counts(points,events):
    owner={}
    def augment(i,seen):
        for j,e in enumerate(events):
            if j in seen or points[i]['video_id']!=e['video_id']:continue
            lo,hi=e['interval_frames']
            if max(lo-points[i]['frame_id'],points[i]['frame_id']-hi,0)>1:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(points)):augment(i,set())
    tp=len(owner);fp=len(points)-tp;fn=len(events)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if points or events else 0}


def run(a):
    audit=read(a.audit/'audit.json');protocol=load_protocol();select(protocol,audit['cache_video_ids'],'training')
    for p,digest in audit['input_hashes'].items():
        if sha(p)!=digest:raise ValueError('Audit input changed: '+p)
    source_cases=restore_rules(read(a.audit/'cases.json')['cases']);primary=read(DEFAULT/'train_features_01/automatic_cases.json')['cases']
    reviews=read(DEFAULT/'train_classifier_01/human_reviews_snapshot.json')['records'];labels={r['case_id']:r['candidate_label'] for r in reviews}
    X=np.array([features(c) for c in primary]);y=np.array([labels[c['case_id']]=='racket_contact' for c in primary],float)
    groups=np.array([c['source_group'] for c in primary]);window=read(a.audit/'window_events.json');events=window['events'];scopes=window['scopes']
    cached=read(a.audit/'outputs.json');cache=DEFAULT/'train_event_review_01/automatic_inputs';tracks={};samples={};proposals={};curves={}
    for vid in audit['cache_video_ids']:
        tracks[vid]=lines(cache/f'{vid}_tracks.jsonl');samples[vid]=read(cache/f'{vid}_geometry.json');proposals[vid]=read(cache/f'{vid}_proposals.json')
        q=next(s['corners'] for s in samples[vid] if s.get('usable'));curves[vid]=propose_curved_contacts(tracks[vid],q)
    folded={};folds=[]
    for group in sorted(set(groups)):
        train=groups!=group;model=fit(X[train],y[train]);ids=[vid for vid in tracks if vid.split('_')[0]==group]
        folded[group]=apply_model(model,[deepcopy(c) for c in source_cases if c['video_id'] in ids])
        folds.append({'held_source':str(group),'training_primary_cases':[c['case_id'] for i,c in enumerate(primary) if train[i]],'held_videos':ids})
    configurations=[]
    for guard,kinematic,rmse,bic in product((0.,-.5,-.8),(False,True),(.5,1.,2.),(6.,20.)):
        configurations.append({'inside_hit_maximum_cosine':guard,'kinematic_supplement_enabled':kinematic,
            'kinematic_maximum_cosine':-.8,'curved_maximum_rmse_px':rmse,'curved_minimum_bic_gain':bic,
            'curved_minimum_direction_cosine':0.})
    def evaluate_config(config,old=False):
        points=[];classified=[];added=[]
        for group,fold in folded.items():
            for vid in [vid for vid in tracks if vid.split('_')[0]==group]:
                cases=[c for c in deepcopy(fold) if c['video_id']==vid]
                if not old:cases=apply_contact_consensus(cases,tracks[vid],config)
                result,_=experimental_landings(tracks[vid],cached[vid]['baseline'],proposals[vid],cases,True,True)
                if not old:
                    result,decisions=curved_auxiliary_landings(tracks[vid],samples[vid],cases,result,config,curves[vid])
                    added.extend(dict(d,video_id=vid) for d in decisions if d['included'])
                classified+=cases
                points.extend(dict(p,video_id=vid) for p in result if any(lo<=p['frame_id']<=hi for lo,hi in scopes.get(vid,[])))
        known=[c for c in classified if c['case_id'] in labels]
        hit_correct=sum(c['event']=='racket_contact_candidate' and labels[c['case_id']]=='racket_contact' for c in known)
        removed=sum(c['event']=='racket_contact_candidate' and labels[c['case_id']]=='table_bounce' for c in known)
        false_hit=sum(c['event']=='racket_contact_candidate' and labels[c['case_id']]!='racket_contact' for c in known)
        metric=temporal_counts(points,[e for e in events if e['event_type']=='table_bounce'])
        return {'config':config,'window_temporal_metrics':metric,'primary_correct_hits':hit_correct,
                'primary_removed_bounces':removed,'primary_false_hit_flags':false_hit,'added_count':len(added)},points,classified,added
    baseline,_,_,_=evaluate_config({},True);reports=[]
    for c in configurations:reports.append(evaluate_config(c)[0])
    admissible=[r for r in reports if r['primary_removed_bounces']==0 and r['primary_false_hit_flags']==0
                and r['primary_correct_hits']>=baseline['primary_correct_hits']
                and r['window_temporal_metrics']['f1']>baseline['window_temporal_metrics']['f1']]
    if not admissible:raise ValueError('No train-CV improvement without reviewed primary regressions')
    chosen=max(admissible,key=lambda r:(r['primary_correct_hits'],r['window_temporal_metrics']['f1'],
        -r['window_temporal_metrics']['fp'],-r['config']['inside_hit_maximum_cosine'],
        -r['config']['curved_maximum_rmse_px'],r['config']['curved_minimum_bic_gain']))
    metrics,points,classified,added=evaluate_config(chosen['config'])
    a.output.mkdir(parents=True,exist_ok=False)
    policy={'schema_version':1,'name':'train_source_curved_consensus_v2','config':chosen['config'],
        'training_provenance':{'split_id':protocol['split_id'],'training_video_ids':audit['cache_video_ids'],
            'training_source_groups':sorted(set(groups)),'primary_label_count':36,'reviewed_window_bounces':43,
            'classification_model_unchanged':True,'classifier_threshold_unchanged':True,'cross_validation_folds':folds,
            'reserved_samples_used_for_selection':False,'public_samples_used_for_selection':False,
            'prior_public_failure_exposure_disclosed':True,'human_labels_at_runtime':False,
            'audit_sha256':sha(a.audit/'audit.json'),'review_snapshot_sha256':sha(DEFAULT/'train_classifier_01/human_reviews_snapshot.json'),
            'selection_path':str(a.output/'selection.json')},
        'status':'experimental_train_selected_not_formally_accepted','model_sha256':audit['classifier_sha256']}
    save(a.output/'policy.json',policy)
    report={'baseline':baseline,'selected':metrics,'search':reports,'policy_sha256':sha(a.output/'policy.json'),
            'folds':folds,'selection_scope':'Six recording-source CV on 36 reviewed primary labels and complete labelled windows',
            'limitation':'Window metrics use intervals only, no 50px spatial scoring; not whole-video F1. Available pose evidence covers original 36 candidates only.',
            'dual_event_user_confirmation':{'case_id':'16_002_33','meaning':'Table bounce at32-33 followed immediately by hit33; labels preserved and not forced exclusive.',
                'runtime_resolution':'Close ambiguous proposals stay deferred; no GT-derived exception in runtime'},
            'detector_inference_calls':0,'selected_policy_frozen_before_reserved_or_public_replay':True}
    save(a.output/'selection.json',report);save(a.output/'cv_cases.json',{'cases':classified});save(a.output/'cv_added.json',{'cases':added});save(a.output/'cv_window_predictions.json',{'predictions':points})
    for name in ('contact_consensus.py','curved_contact_candidates.py','contact_output_policy.py'):
        (a.output/name).write_bytes((PROJECT/'development/bounce'/name).read_bytes())
    (a.output/'source_select_cached_contact_consensus.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'baseline':baseline,'selected':metrics,'policy_sha256':report['policy_sha256']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

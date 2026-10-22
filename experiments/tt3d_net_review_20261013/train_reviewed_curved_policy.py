"""Train grouped table-event model and select policy only on reviewed train windows."""
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
from development.bounce import curved_contact_classifier as table_model
from development.bounce.contact_classifier import fit as hit_fit,features as hit_features,apply_model
from development.bounce.learned_curved_policy import revised_landings,prepare_existing_contacts
from development.bounce.curved_tree_classifier import fit as tree_fit
from select_cached_contact_consensus import restore_rules,temporal_counts


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(a):
    protocol=load_protocol();dataset=read(a.dataset/'dataset.json');automatic=read(a.dataset/'automatic_curved_cases.json')['videos']
    select(protocol,list(automatic),'training')
    for path,digest in dataset['sources'].items():
        if sha(path)!=digest:raise ValueError('Curved dataset source changed')
    examples=dataset['examples'];X=np.array([e['vector'] for e in examples]);y=np.array([e['label_table_contact'] for e in examples],float)
    fit_table=tree_fit if a.tree else table_model.fit
    if a.hard_negatives and not a.tree:raise ValueError('Hard-negative weights require the shallow-tree learner')
    sample_weights=np.array([3. if not e['label_table_contact'] and e['event_types'] else 1. for e in examples])
    def fit_selected(indices,regularization):
        extra={'sample_weights':sample_weights[indices]} if a.hard_negatives else {}
        return fit_table(X[indices],y[indices],regularization,dataset['features'],**extra)
    groups=np.array([e['source_group'] for e in examples]);hit_examples=read(DEFAULT/'train_features_01/automatic_cases.json')['cases']
    hit_labels={r['case_id']:r['candidate_label'] for r in read(DEFAULT/'train_classifier_01/human_reviews_snapshot.json')['records']}
    HX=np.array([hit_features(c) for c in hit_examples]);Hy=np.array([hit_labels[c['case_id']]=='racket_contact' for c in hit_examples],float)
    HG=np.array([c['source_group'] for c in hit_examples]);source_cases=restore_rules(read(a.audit/'cases.json')['cases']);outputs=read(a.audit/'outputs.json')
    cache=DEFAULT/'train_event_review_01/automatic_inputs';tracks={v:lines(cache/f'{v}_tracks.jsonl') for v in automatic}
    samples={v:read(cache/f'{v}_geometry.json') for v in automatic};proposals={v:read(cache/f'{v}_proposals.json') for v in automatic}
    folds=[];held_cases={};held_models={};cv_scores={}
    for group in sorted(set(groups)):
        test=groups==group;train=~test;hit_train=HG!=group
        ids=[v for v in automatic if v.split('_')[0]==group];held_cases[group]=apply_model(hit_fit(HX[hit_train],Hy[hit_train]),[deepcopy(c) for c in source_cases if c['video_id'] in ids])
        folds.append({'held_source':str(group),'train_curve_case_ids':[e['case_id'] for i,e in enumerate(examples) if train[i]],
                      'held_curve_case_ids':[e['case_id'] for i,e in enumerate(examples) if test[i]],
                      'train_hit_case_ids':[c['case_id'] for i,c in enumerate(hit_examples) if hit_train[i]]})
        for regularization in (1.,5.):
            model=fit_selected(train,regularization);held_models[(group,regularization)]=model
            values=table_model.score_matrix(model,X[test])
            for e,value in zip([e for i,e in enumerate(examples) if test[i]],values):cv_scores[(e['case_id'],regularization)]=float(value)
    scopes=dataset['scopes'];events=[e for e in dataset['events'] if e['event_type']=='table_bounce']
    old_points=[dict(p,video_id=vid) for vid in outputs for p in outputs[vid]['experimental'] if any(lo<=p['frame_id']<=hi for lo,hi in scopes.get(vid,[]))]
    baseline=temporal_counts(old_points,events);search=[]
    prepared={}
    for group,fold in held_cases.items():
        for vid in [v for v in automatic if v.split('_')[0]==group]:
            for kinematic in (False,True):
                prepared[(vid,kinematic)]=prepare_existing_contacts(tracks[vid],outputs[vid]['baseline'],proposals[vid],
                    [c for c in fold if c['video_id']==vid],{'kinematic_supplement_enabled':kinematic})
    def evaluate(config,regularization):
        points=[];all_cases=[];decisions=[]
        for group,fold in held_cases.items():
            for vid in [v for v in automatic if v.split('_')[0]==group]:
                result,cases,ds=revised_landings(tracks[vid],samples[vid],outputs[vid]['baseline'],proposals[vid],
                    [c for c in fold if c['video_id']==vid],held_models[(group,regularization)],config,automatic[vid],prepared[(vid,config['kinematic_supplement_enabled'])])
                points.extend(dict(p,video_id=vid) for p in result if any(lo<=p['frame_id']<=hi for lo,hi in scopes.get(vid,[])))
                all_cases+=cases;decisions.extend(dict(d,video_id=vid) for d in ds)
        known=[c for c in all_cases if c['case_id'] in hit_labels]
        report={'config':config,'regularization':regularization,'window_temporal_metrics':temporal_counts(points,events),
            'primary_correct_hits':sum(c['event']=='racket_contact_candidate' and hit_labels[c['case_id']]=='racket_contact' for c in known),
            'primary_false_hit_flags':sum(c['event']=='racket_contact_candidate' and hit_labels[c['case_id']]!='racket_contact' for c in known)}
        return report,points,all_cases,decisions
    cubic_gates=(None,0.,6.) if 'bic_gain_over_smooth_cubic_scaled' in dataset['features'] else (None,)
    rejection_gates=(0.,.2,.35,.5) if a.reject_low_score else (0.,)
    physics_gates=(None,-.05,-.15) if a.physics_veto else (None,)
    for regularization,threshold,near_hit,kinematic,retime,cubic,rejection,physics in product((1.,5.),(.5,.65,.8),(.8,.95,1.01),(False,True),(False,True),cubic_gates,rejection_gates,physics_gates):
        if near_hit<threshold:continue
        config={'table_threshold':threshold,'near_hit_table_threshold':near_hit,'kinematic_supplement_enabled':kinematic,'retime_nearby_enabled':retime}
        if cubic is not None:config['minimum_cubic_bic_gain']=cubic
        if rejection:config['table_rejection_threshold']=rejection
        if physics is not None:config['ballistic_veto_margin']=physics
        search.append(evaluate(config,regularization)[0])
    eligible=[r for r in search if r['primary_correct_hits']>=15 and r['primary_false_hit_flags']==0
        and r['window_temporal_metrics']['f1']>baseline['f1']]
    if not eligible:raise ValueError('No admissible train-source improvement')
    selected=max(eligible,key=lambda r:(r['window_temporal_metrics']['f1'],-r['window_temporal_metrics']['fp'],r['primary_correct_hits'],
        r['config']['table_threshold'],r['config']['near_hit_table_threshold'],not r['config']['retime_nearby_enabled'],r['regularization']))
    result,points,cases,decisions=evaluate(selected['config'],selected['regularization'])
    a.output.mkdir(parents=True,exist_ok=False);model=fit_selected(np.ones(len(y),dtype=bool),selected['regularization'])
    policy={'schema_version':1,'name':'source25_curved_table_'+('shallow_trees' if a.tree else 'logistic'),'config':selected['config'],'table_model':model,
        'model_sha256':sha(DEFAULT/'train_classifier_01/model.json'),
        'training_provenance':{'split_id':protocol['split_id'],'training_video_ids':sorted(automatic),'training_source_groups':sorted(set(groups)),
            'curve_examples':len(examples),'table_positive':int(y.sum()),'non_table':int(len(y)-y.sum()),'reviewed_table_events':len(events),
            'human_primary_cases':dataset.get('human_primary_cases',sum(dataset['human_label_counts'].values())),'classifier_uses_automatic_features_only':True,'pose_hit_classifier_unchanged':True,
            'explicit_reviewed_non_table_events_weight_multiplier':3. if a.hard_negatives else 1.,
            'validation_samples_used_for_selection':False,'public_samples_used_for_selection':False,
            'prior_public_failure_exposure_disclosed':True,'dataset_sha256':sha(a.dataset/'dataset.json'),
            'followup_snapshot_sha256':sha(a.dataset/'followup_snapshot.json'),'selection_path':str(a.output/'selection.json')},
        'status':'experimental_train_selected_not_formally_accepted'}
    save(a.output/'policy.json',policy)
    report={'baseline_expanded_window_metrics':baseline,'selected':result,'search':search,'folds':folds,'policy_sha256':sha(a.output/'policy.json'),
        'cv_score_examples':[dict(e,cv_score=cv_scores[(e['case_id'],selected['regularization'])]) for e in examples],
        'detector_inference_calls':0,'reserved_or_public_selection_inputs':False,
        'metrics_scope':'Training source CV used for rule selection, not unbiased heldout performance. Candidate windows and intervals only, no spatial scoring.',
        'timing_offsets':dataset['timing_offsets']}
    save(a.output/'selection.json',report);save(a.output/'cv_window_predictions.json',{'predictions':points});save(a.output/'cv_cases.json',{'cases':cases});save(a.output/'cv_decisions.json',{'cases':decisions})
    for name in ('learned_curved_policy.py','curved_contact_classifier.py','curved_contact_candidates.py','curved_contact_context.py','curved_tree_classifier.py','automatic_ballistic_contact.py','automatic_net_zone.py','contact_consensus.py','contact_output_policy.py'):
        (a.output/name).write_bytes((PROJECT/'development/bounce'/name).read_bytes())
    (a.output/'source_train_reviewed_curved_policy.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'baseline':baseline,'selected':result,'policy_sha256':report['policy_sha256']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tree',action='store_true');p.add_argument('--reject-low-score',action='store_true');p.add_argument('--hard-negatives',action='store_true');p.add_argument('--physics-veto',action='store_true');run(p.parse_args())

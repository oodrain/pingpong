"""Augment the small strike model with newly reviewed train gesture evidence."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.bounce.contact_classifier import features,fit,apply_model


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(a):
    old=read(DEFAULT/'train_features_01/automatic_cases.json')['cases'];new=read(a.features/'automatic_cases.json')['cases']
    labels=read(DEFAULT/'train_classifier_01/human_reviews_snapshot.json')['records']+read(a.snapshot)['records'];label_map={c['case_id']:c for c in labels}
    retained=[];excluded=[]
    for c in old+new:
        label=label_map[c['case_id']];primary=next((e for e in label['state']['events'] if e['event_id']==label['state']['primary_event_id']),None)
        if primary and max(primary['interval_frames'][0]-c['seed_frame'],c['seed_frame']-primary['interval_frames'][1],0)>2:
            excluded.append({'case_id':c['case_id'],'reason':'primary_event_timing_more_than_two_frames_from_automatic_candidate'});continue
        if label['candidate_label']=='uncertain':continue
        retained.append(c)
    protocol=load_protocol();ids=list(dict.fromkeys(c['video_id'] for c in retained));select(protocol,ids,'training')
    X=np.array([features(c) for c in retained]);y=np.array([label_map[c['case_id']]['candidate_label']=='racket_contact' for c in retained],float)
    groups=np.array([c['source_group'] for c in retained]);cv=[];folds=[]
    for group in sorted(set(groups)):
        test=groups==group;train=~test;model=fit(X[train],y[train]);cases=apply_model(model,deepcopy([c for i,c in enumerate(retained) if test[i]]))
        for c in cases:cv.append({'case_id':c['case_id'],'label':label_map[c['case_id']]['candidate_label'],'rule_flag':next(o for o in retained if o['case_id']==c['case_id'])['decision']['review_hit_suggestion'],
            'model_flag':c['learned_contact']['hit_suggestion'],'combined_flag':c['decision']['review_hit_suggestion'],'score':c['learned_contact']['score']})
        folds.append({'held_source':str(group),'train_case_ids':[c['case_id'] for i,c in enumerate(retained) if train[i]],'held_case_ids':[c['case_id'] for i,c in enumerate(retained) if test[i]]})
    a.output.mkdir(parents=True,exist_ok=False);model=fit(X,y);counts=dict(Counter(label_map[c['case_id']]['candidate_label'] for c in retained))
    model['training_provenance']={'split_id':protocol['split_id'],'training_video_ids':ids,'training_source_groups':sorted(set(groups)),
        'candidate_count':len(retained),'label_counts':counts,'positive_hits':int(y.sum()),'validation_samples_used':False,'public_samples_used':False,
        'automatic_features_sha256':sha(a.features/'automatic_features.json'),'followup_snapshot_sha256':sha(a.snapshot),'excluded_cases':excluded,
        'training_evaluation_path':str(a.output/'evaluation.json'),'past_global_development_exposure':True,'independent_holdout':False}
    model['output_policy']={'baseline_exclusion_enabled':True,'selection_scope':'Experimental augmentation; same original grouped gate, not adopted'}
    save(a.output/'model.json',model)
    def metric(key):
        return {'correct_hits':sum(c[key] and c['label']=='racket_contact' for c in cv),'false_hit_flags':sum(c[key] and c['label']!='racket_contact' for c in cv),
                'removed_table_contacts':sum(c[key] and c['label']=='table_bounce' for c in cv),'known_table_contacts':counts['table_bounce']}
    result={'candidate_count':len(retained),'label_counts':counts,'excluded_cases':excluded,'folds':folds,'cases':cv,
        'metrics':{k:metric(k) for k in ('rule_flag','model_flag','combined_flag')},'classifier_threshold_unchanged':True,'model_sha256':sha(a.output/'model.json'),
        'deployed':False,'limitation':'New reviewed contacts have no supported wrist visits; missing the wrist-near-table conflict subtype.'}
    save(a.output/'evaluation.json',result);(a.output/'source_train_augmented_gesture_classifier.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('candidate_count','label_counts','excluded_cases','metrics')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

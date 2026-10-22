"""Grouped development validation; never called by video inference."""
import argparse,json,sys,hashlib,re
from pathlib import Path
from collections import Counter
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));ROOT=PROJECT.parents[1]
from development.bounce.contact_classifier import features,fit,score_matrix,THRESHOLD,FEATURES
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(pilot,prediction,out):
    out.mkdir(parents=True,exist_ok=False);cases=read(prediction)['cases'];X=np.array([features(c) for c in cases]);ids=[c['case_id'] for c in cases]
    save(out/'automatic_features.json',{'features':FEATURES,'case_ids':ids,'vectors':X.tolist(),
        'source_predictions_sha256':sha(prediction),'manual_spatial_labels_used':False,'manual_table_axis_used':False})
    human=pilot/'human_event_evaluation_01/human_reviews_snapshot.json';records={r['case_id']:r for r in read(human)['records']}
    y=np.array([records[i]['candidate_label']=='racket_contact' for i in ids],float)
    manifest_path=ROOT/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json'
    manifest={v['video_id']:v for v in read(manifest_path)['videos']}
    groups=[]
    for c in cases:
        name=Path(manifest[c['video_id']]['path']).name
        match=re.search(r'_(\d{8,})_\[',name)
        groups.append(match.group(1) if match else 'unverified:'+c['video_id'])
    groups=np.array(groups);cv=np.zeros(len(y));folds=[]
    for group in sorted(set(groups)):
        test=groups==group;train=~test
        if len(set(y[train]))<2:raise ValueError('Group split leaves one class; cannot validate')
        m=fit(X[train],y[train]);cv[test]=score_matrix(m,X[test])
        folds.append({'source_group_proposal':group,'test_case_ids':[ids[i] for i in np.where(test)[0]],
                      'train_count':int(sum(train)),'test_count':int(sum(test))})
    model=fit(X,y);fitted=score_matrix(model,X)
    model['training_provenance']={'candidate_count':len(y),'positive_hits':int(sum(y)),
        'automatic_features_sha256':sha(out/'automatic_features.json'),'human_event_snapshot_sha256':sha(human),
        'already_reviewed_development_data':True,'independent_holdout':False}
    save(out/'model.json',model)
    sys.path.insert(0,str(ROOT/'development/annotation'))
    from evaluate_auxiliary_reviews import flag_metrics,table_pool_metrics
    rows=[]
    for i,c in enumerate(cases):
        rule=bool(c['decision']['review_hit_suggestion'])
        rows.append({'case_id':ids[i],'human_label':records[ids[i]]['candidate_label'],'automatic_rule_flag':rule,
            'source_group_proposal':str(groups[i]),'group_cv_score':float(cv[i]),'fitted_score':float(fitted[i]),
            'group_cv_classifier_flag':bool(cv[i]>=THRESHOLD),'group_cv_combined_flag':bool(rule or cv[i]>=THRESHOLD),
            'fitted_combined_flag':bool(rule or fitted[i]>=THRESHOLD)})
    report={'threshold':THRESHOLD,'fixed_regularization':1.,'thresholds_retuned_after_evaluation':False,
        'group_count':len(folds),'folds':folds,'group_source':'Original recording IDs parsed from filenames; source independence not externally verified.',
        'model_sha256':sha(out/'model.json'),'in_sample_model_scores_are_not_validation':True,'independent_holdout':False,
        'method_designed_after_prior_human_review':True,'cases':rows,
        'metrics':{key:flag_metrics(rows,key) for key in ('automatic_rule_flag','group_cv_classifier_flag','group_cv_combined_flag','fitted_combined_flag')},
        'group_cv_table_pool':table_pool_metrics(rows,'group_cv_combined_flag')}
    save(out/'evaluation.json',report);(out/'source_train_contact_classifier.py').write_bytes(Path(__file__).read_bytes())
    (out/'source_contact_classifier.py').write_bytes((PROJECT/'development/bounce/contact_classifier.py').read_bytes())
    print(json.dumps({'groups':len(folds),'metrics':report['metrics'],'cv_table_pool':report['group_cv_table_pool'],
        'case53':[r for r in rows if r['case_id']=='v000038_53']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('pilot','predictions','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.pilot.resolve(),a.predictions.resolve(),a.output.resolve())

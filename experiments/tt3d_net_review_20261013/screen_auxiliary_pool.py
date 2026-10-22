"""Freeze label-free review screening, then report known-public regression only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.trajectory_events import auxiliary_candidate_pool,screen_auxiliary_candidate
from development.evaluation.evaluate_landings import evaluate
from fit_contact_pilot import sha

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def run(root,folder,out):
    out.mkdir(parents=True,exist_ok=False)
    paths=[folder/n for n in ('inference.json','baseline_candidates.jsonl','vector_candidates.jsonl')]
    inference=read(paths[0]);baseline=lines(paths[1]);vector=lines(paths[2])
    pool=auxiliary_candidate_pool(baseline,vector)
    details={v['video_id']:v['proposal_result'] for v in inference['videos']}
    cases=[{'candidate':p,**screen_auxiliary_candidate(p,details.get(p['video_id']),baseline)} for p in pool['auxiliary_candidates']]
    selected=[c['candidate'] for c in cases if c['status']=='ready_for_event_review']
    report={'policy':'review_readiness_v1','known_public_regression_not_independent_holdout':True,
        'thresholds_retuned':False,'baseline_preserved_exactly':pool['primary_predictions']==baseline,
        'auxiliary_adoption_enabled':False,'candidate_pool_not_final_predictions':True,
        'sources':{str(p):sha(p) for p in paths},'script_sha256':sha(__file__),
        'rule_sha256':sha(Path(__file__).resolve().parents[2]/'development/bounce/trajectory_events.py'),
        'status_counts':dict(Counter(c['status'] for c in cases)),
        'reason_counts':dict(Counter(r for c in cases for r in c['reasons'])),
        'primary_predictions':baseline,'selected_review_candidates':selected,'cases':cases}
    # Persist the complete screening decision before accessing evaluation labels.
    save(out/'screening.json',report)
    public=root/'participant/participant/pingpang/public_data'
    manifest=root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json'
    gt=lines(public/'gt_public.jsonl');videos=read(manifest)['videos']
    scores={name:evaluate(rows,gt,videos,exempt_invalid=False) for name,rows in
        [('baseline',baseline),('baseline_plus_review_ready',baseline+selected)]}
    save(out/'evaluation.json',{'screening_sha256':sha(out/'screening.json'),
        'gt_sha256':sha(public/'gt_public.jsonl'),'scores':scores,
        'interpretation':'Hypothetical candidate pool only. No adoption, no final classifier precision claim; known public regression.'})
    train_path=root/'C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json'
    train=read(train_path)['videos']
    provenance_fields=('recording_source_id','match_id','source_group_id')
    inventory=[]
    for name,rows,path in [('public15',videos,manifest),('train154',train,train_path)]:
        inventory.append({'dataset':name,'video_count':len(rows),'manifest_sha256':sha(path),
            'manifest':str(path),'explicit_source_group_records':sum(any(r.get(k) for k in provenance_fields) for r in rows),
            'known_angle_records':sum(r.get('angle') not in (None,'unknown','') for r in rows),
            'usable_as_new_independent_holdout':False,
            'reason':'Previously used in project development; filename or owner does not verify recording-source independence.'})
    save(out/'holdout_readiness.json',{'scope':'Two existing experiment input manifests only; not an exhaustive disk inventory.',
        'datasets':inventory,'verified_new_holdout_videos':0,
        'required_intake_fields':['video_path','video_sha256','recording_source_id','source_group_verified_by','angle','prior_use_declaration'],
        'requirements':['Use previously untouched recording sources; group adjacent clips and augmentations together.',
            'Freeze source split and rule hashes before looking at labels or metrics.',
            'Keep human contact intervals and racket labels independent from model predictions.']})
    print(json.dumps({'counts':report['status_counts'],'reasons':report['reason_counts'],
        'scores':{k:v['overall'] for k,v in scores.items()}},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.folder,a.output)

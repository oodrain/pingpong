"""Replay a frozen train-selected policy on frozen automatic full-video outputs.

This is cached rule evaluation, not fresh detector inference or speed testing.
All predictions and input hashes are frozen before evaluation labels are read.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha
from development.bounce.contact_consensus import apply_contact_consensus,curved_auxiliary_landings
from development.bounce.contact_output_policy import experimental_landings
from development.bounce.learned_curved_policy import revised_landings


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save_lines(p,r):Path(p).write_text(''.join(json.dumps(s,ensure_ascii=False,allow_nan=False)+'\n' for s in r),encoding='utf-8')


def run(a):
    protocol=load_protocol();policy=read(a.policy);old=a.previous;freeze=read(old/'prediction_freeze.json')
    if policy['training_provenance']['split_id']!=protocol['split_id']:raise ValueError('Wrong policy source split')
    select(protocol,policy['training_provenance']['training_video_ids'],'training')
    if policy['model_sha256']!=freeze['model_sha256']:raise ValueError('Wrong classifier for frozen features')
    for path,digest in freeze['prediction_sha256'].items():
        if sha(path)!=digest:raise ValueError('Old frozen predictions changed')
    records=[]
    for split,purpose in [('validation_reserved','validation'),('public_regression','public_regression')]:
        records+=select(protocol,[r['video_id'] for r in protocol['videos'] if r['split']==split],purpose)
    a.output.mkdir(parents=True,exist_ok=False);input_hashes={};prediction_hashes={};diagnostics=[]
    names=['contact_consensus.py','curved_contact_candidates.py','contact_output_policy.py','trajectory_events.py','person_motion_contact.py']
    if 'table_model' in policy:names+=['curved_contact_classifier.py','curved_contact_context.py','curved_tree_classifier.py','learned_curved_policy.py']
    sources=[PROJECT/'development/bounce'/name for name in names]
    if 'automatic_ballistic_missing' in policy.get('table_model',{}).get('features',[]):
        sources += [PROJECT/'development/bounce/automatic_ballistic_contact.py',PROJECT/'development/bounce/automatic_net_zone.py',PROJECT/'development/detection/tt3d_camera_estimation.py',PROJECT/'development/detection/auto_table_geometry.py']
    plan={'policy_sha256':sha(a.policy),'source_sha256':{str(p):sha(p) for p in sources},
          'classifier_sha256':freeze['model_sha256'],'split_id':protocol['split_id'],
          'previous_prediction_freeze_sha256':sha(old/'prediction_freeze.json'),
          'detector_inference_calls':0,'fresh_video_inference':False,'speed_benchmark':False,
          'policy_selected_using_reserved_or_public_labels':False}
    save(a.output/'plan.json',plan)
    for r in records:
        vid=r['video_id'];folder=old/'videos'/vid;run=read(folder/'run.json')
        if sha(folder/'run.json')!=freeze['run_sha256'][vid] or run['video_sha256']!=r['video_sha256']:
            raise ValueError('Frozen run/video provenance mismatch')
        paths=[folder/name for name in ('ball_tracks.jsonl','geometry_samples.jsonl','proposals.json','event_candidates.json','legacy_landings.jsonl','experimental_landings.jsonl')]
        input_hashes.update({str(p):sha(p) for p in paths})
        tracks=lines(paths[0]);samples=lines(paths[1]);proposals=read(paths[2]);cases=read(paths[3])['cases'];base=lines(paths[4])
        reproduced,_=experimental_landings(tracks,base,proposals,cases,True,True)
        if reproduced!=lines(paths[5]):raise ValueError('Old policy not reproduced: '+vid)
        if 'table_model' in policy:
            result,changed,decisions=revised_landings(tracks,samples,base,proposals,cases,policy['table_model'],policy['config'],image_size=(r['width'],r['height']))
        else:
            changed=apply_contact_consensus(cases,tracks,policy['config'])
            result,_=experimental_landings(tracks,base,proposals,changed,True,True)
            result,decisions=curved_auxiliary_landings(tracks,samples,changed,result,policy['config'])
        destination=a.output/'videos'/vid;destination.mkdir(parents=True)
        save_lines(destination/'landings.jsonl',result);save(destination/'cases.json',{'cases':changed});save(destination/'curved_decisions.json',{'cases':decisions})
        prediction_hashes[str(destination/'landings.jsonl')]=sha(destination/'landings.jsonl')
        diagnostics.append({'video_id':vid,'weak_inside_hits_blocked':[c['seed_frame'] for c in changed if c['contact_consensus']['weak_inside_reversal_blocked']],
            'kinematic_hits_added':[c['seed_frame'] for c in changed if c['contact_consensus']['kinematic_hit_supported']],
            'curved_contacts_added':[d['seed_frame'] for d in decisions if d['included']]})
    save(a.output/'prediction_freeze.json',{'input_hashes':input_hashes,'predictions':prediction_hashes,'plan_sha256':sha(a.output/'plan.json'),
        'policy_sha256':sha(a.policy),'evaluation_labels_opened':False})
    from evaluate_landings import evaluate
    stage=[dict(r,valid=True) for r in lines(old/'label_snapshots/candidate_events.jsonl')]
    public=lines(old/'label_snapshots/gt_public.jsonl');previous=read(old/'evaluation.json');results={}
    for split in ('validation_reserved','public_regression'):
        subset=[r for r in records if r['split']==split];ids={r['video_id'] for r in subset}
        predictions=[dict(p,video_id=r['video_id']) for r in subset for p in lines(a.output/'videos'/r['video_id']/'landings.jsonl')]
        gt=[g for g in (stage if split=='validation_reserved' else public) if g['video_id'] in ids]
        metric=evaluate(predictions,gt,[dict(r,angle=r['official_angle']) for r in subset],exempt_invalid=False)
        if split=='validation_reserved':metric['angle_macro_f1']=None;metric['accuracy_score_estimate']=None
        results[split]={'before':previous['results'][split]['policies']['experimental_landings'],'after':metric,
            'label_status':previous['results'][split]['label_status'],'independent_blind_test':False}
    for path,digest in {**input_hashes,**prediction_hashes}.items():
        if sha(path)!=digest:raise ValueError('Replay inputs/outputs changed during scoring')
    report={**plan,'results':results,'diagnostics':diagnostics,'all_46_previous_policy_outputs_reproduced':True,
            'evaluation_outcomes_used_to_retune_policy':False,'runtime_cache_input_not_required':True,
            'status':'cached_rule_replay_completed_not_fresh_video_test',
            'limitation':'Prior development exposure disclosed; reserved GT is a stage-candidate proxy. No speed or fresh detector equivalence claim.'}
    save(a.output/'evaluation.json',report)
    for p in sources:(a.output/p.name).write_bytes(p.read_bytes())
    (a.output/'source_replay_frozen_contact_consensus.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({s:{'before':r['before']['overall'],'after':r['after']['overall']} for s,r in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--previous',type=Path,default=DEFAULT/'full_video_eval_01');p.add_argument('--output',type=Path,required=True);run(p.parse_args())

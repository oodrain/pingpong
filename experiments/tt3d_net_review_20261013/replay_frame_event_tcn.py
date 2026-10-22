"""Freeze dense TCN predictions on all cached complete videos before scoring."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha,check_model_for_formal_validation
from development.bounce.frame_event_features import automatic_frame_features,decode_frame_events,geometry_supported_touch_scores
from development.bounce.frame_event_tcn import load,predict

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save_lines(path,rows):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def run(args):
    protocol=load_protocol();model,checkpoint=load(args.model);check_model_for_formal_validation(checkpoint,protocol)
    model_freeze=read(args.model.parent/'model_freeze.json')
    if sha(args.model)!=model_freeze['model_sha256']:raise ValueError('TCN model changed after train-only selection')
    previous=args.previous;old_freeze=read(previous/'prediction_freeze.json');current=args.current;current_freeze=read(current/'prediction_freeze.json')
    for path,digest in {**old_freeze['prediction_sha256'],**current_freeze['predictions']}.items():
        if sha(path)!=digest:raise ValueError('Frozen rule predictions changed')
    records=[]
    for split,purpose in [('validation_reserved','validation'),('public_regression','public_regression')]:
        records += select(protocol,[r['video_id'] for r in protocol['videos'] if r['split']==split],purpose)
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    plan={'model_sha256':sha(args.model),'model_metadata_sha256':sha(args.model.parent/'model_metadata.json'),
        'split_id':protocol['split_id'],'training_provenance':checkpoint['training_provenance'],
        'threshold':checkpoint['threshold'],'minimum_separation_s':checkpoint['minimum_separation_s'],
        'geometry_support_policy':checkpoint.get('geometry_support_policy','none'),
        'source_sha256':{str(PROJECT/'development/bounce'/name):sha(PROJECT/'development/bounce'/name) for name in ('frame_event_features.py','frame_event_tcn.py')},
        'candidate_gate_used':False,'detector_inference_calls':0,'fresh_video_inference':False,
        'speed_benchmark':False,'parameter_selection_uses_reserved_or_public_labels':False,
        'old_prediction_freeze_sha256':sha(previous/'prediction_freeze.json'),'current_rule_prediction_freeze_sha256':sha(current/'prediction_freeze.json')}
    save(args.output/'plan.json',plan);inputs={};predictions={};score_hashes={};diagnostics=[];started=time.monotonic()
    for record in records:
        vid=record['video_id'];folder=previous/'videos'/vid;run=read(folder/'run.json')
        if sha(folder/'run.json')!=old_freeze['run_sha256'][vid] or run['video_sha256']!=record['video_sha256']:
            raise ValueError('Frozen automatic run provenance mismatch')
        paths=[folder/'ball_tracks.jsonl',folder/'geometry_samples.jsonl',folder/'run.json']
        inputs.update({str(p):sha(p) for p in paths});tracks=lines(paths[0]);samples=lines(paths[1])
        features=automatic_frame_features(tracks,samples,(record['width'],record['height']))
        scores=predict(model,features,checkpoint['normalization'])
        touch_scores=geometry_supported_touch_scores(scores[:,0],features,samples,checkpoint.get('geometry_support_policy','none'))
        touch,decisions=decode_frame_events(touch_scores,tracks,checkpoint['threshold'],checkpoint['minimum_separation_s'])
        hits,hit_decisions=decode_frame_events(scores[:,1],tracks,checkpoint['threshold'],checkpoint['minimum_separation_s'])
        destination=args.output/'videos'/vid;destination.mkdir(parents=True)
        save_lines(destination/'landings.jsonl',touch);save_lines(destination/'hit_events.jsonl',hits)
        save(destination/'decisions.json',{'table':decisions,'hit':hit_decisions,'world_contact_confirmed':False})
        np.savez_compressed(destination/'frame_scores.npz',scores=scores,touch_scores=touch_scores)
        predictions[str(destination/'landings.jsonl')]=sha(destination/'landings.jsonl');score_hashes[str(destination/'frame_scores.npz')]=sha(destination/'frame_scores.npz')
        diagnostics.append({'video_id':vid,'frames_scored':len(tracks),'table_predictions':len(touch),'hit_predictions':len(hits),
            'table_peaks_without_actual_ball_point':sum(not d['included'] for d in decisions)})
    save(args.output/'prediction_freeze.json',{'input_hashes':inputs,'predictions':predictions,'score_hashes':score_hashes,
        'model_sha256':sha(args.model),'plan_sha256':sha(args.output/'plan.json'),'evaluation_labels_opened':False})
    from evaluate_landings import evaluate
    stage=[dict(g,valid=True) for g in lines(previous/'label_snapshots/candidate_events.jsonl')];public=lines(previous/'label_snapshots/gt_public.jsonl')
    results={};paired={}
    for split in ('validation_reserved','public_regression'):
        subset=[r for r in records if r['split']==split];ids={r['video_id'] for r in subset}
        gt=[g for g in (stage if split=='validation_reserved' else public) if g['video_id'] in ids];manifest=[dict(r,angle=r['official_angle']) for r in subset]
        policies={}
        for name,root,filename in [('original_rule',previous,'experimental_landings.jsonl'),('current_optimized_rule',current,'landings.jsonl'),('dense_tcn',args.output,'landings.jsonl')]:
            points=[dict(p,video_id=r['video_id']) for r in subset for p in lines(root/'videos'/r['video_id']/filename)]
            metric=evaluate(points,gt,manifest,exempt_invalid=False)
            if split=='validation_reserved':metric['angle_macro_f1']=None;metric['accuracy_score_estimate']=None
            policies[name]=metric
        results[split]={'policies':policies,'label_status':'stage_candidate_proxy' if split=='validation_reserved' else 'official_public_valid_invalid_gt',
                        'independent_blind_test':False}
        old={v['video_id']:v for v in policies['current_optimized_rule']['by_video']};changed=[]
        for row in policies['dense_tcn']['by_video']:
            a={m['gt'].get('event_id',m['gt']['frame_id']):m for m in old[row['video_id']]['matches']}
            b={m['gt'].get('event_id',m['gt']['frame_id']):m for m in row['matches']}
            changed.append({'video_id':row['video_id'],'gained_gt_frames':[b[k]['gt']['frame_id'] for k in b.keys()-a.keys()],
                            'lost_gt_frames':[a[k]['gt']['frame_id'] for k in a.keys()-b.keys()]})
        paired[split]=changed
    for path,digest in {**inputs,**predictions,**score_hashes}.items():
        if sha(path)!=digest:raise ValueError('TCN replay inputs or outputs changed during scoring')
    report={**plan,'results':results,'paired_changes_vs_current_rule':paired,'diagnostics':diagnostics,
        'cached_feature_and_tcn_seconds':time.monotonic()-started,'all_frames_scored':sum(d['frames_scored'] for d in diagnostics),
        'evaluation_outcomes_used_to_retune_model':False,'runtime_requires_human_labels_or_previous_external_cache':False,
        'status':'dense_tcn_cached_full_video_comparison_completed','limitation':'Development exposure and small train set disclosed; internal labels remain incomplete stage candidates. This is not fresh detector inference, independent blind validation, or speed acceptance.'}
    save(args.output/'evaluation.json',report)
    for name in ('frame_event_features.py','frame_event_tcn.py'):(args.output/name).write_bytes((PROJECT/'development/bounce'/name).read_bytes())
    (args.output/'source_replay_frame_event_tcn.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({s:{name:m['overall'] for name,m in r['policies'].items()} for s,r in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--previous',type=Path,default=DEFAULT/'full_video_eval_01')
    parser.add_argument('--current',type=Path,default=ROOT/'outputs/contact_rule_revision_20261014/cached_eval_09');parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())

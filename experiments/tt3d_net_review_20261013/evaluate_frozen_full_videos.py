"""Frozen-model complete-video evaluation; labels opened after all predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,verify_inputs,check_model_for_formal_validation,sha

MODELS=None


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):
    p=Path(p);temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,p)


def initialise(threads):
    global MODELS
    import torch
    torch.set_num_threads(threads)
    from development.detection.video_contact_pipeline import PoseDetector
    from development.detection.video_ball_detector import VideoBallDetector
    from development.detection.tt3d_segmenter import TT3DSegmenter
    from development.detection.racket_detector import RacketDetector
    MODELS={'ball':VideoBallDetector(ROOT/'participant/participant/pingpang/participant/weights/ball.onnx',threads),
            'table':TT3DSegmenter(PROJECT/'tmp/tt3d-upstream/weights/table_segmentation.ckpt',threads),
            'pose':PoseDetector(PROJECT/'data/pose_weights/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'),
            'racket':RacketDetector(PROJECT/'data/racket_weights/rtmdet_m_racket_state.pt')}


def process(job):
    from development.detection.video_contact_pipeline import run
    out=Path(job['output']);log=Path(job['log'])
    with log.open('w',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:
            result=run(Path(job['video_path']),out,MODELS,Path(job['model_path']),threads=job['threads'])
            if result['video_sha256']!=job['video_sha256'] or result['n_frames']!=job['n_frames']:
                raise ValueError('Fresh video inference disagrees with frozen input manifest')
            return {'video_id':job['video_id'],'status':'completed','run_sha256':sha(out/'run.json'),
                    'frames':result['n_frames'],'inference_seconds':result['inference_seconds'],
                    'experimental_contacts':result['experimental_landings_count']}
        except Exception as error:
            traceback.print_exc()
            return {'video_id':job['video_id'],'status':'failed','error':str(error),'log':str(log)}


def evaluate(out,manifest,records,plan,completed,seconds):
    frozen_predictions={};bound_runs={}
    for record in records:
        vid=record['video_id'];folder=out/'videos'/vid;run=read(folder/'run.json')
        if run['video_sha256']!=record['video_sha256'] or run['contact_classifier']['model_sha256']!=plan['model_sha256']:
            raise ValueError('Frozen model/video hash changed')
        if run['source_hashes']!=plan['pipeline_source_hashes']:raise ValueError('Inference source changed during batch')
        if run['human_annotation_inputs'] or run['manual_axis_inputs'] or run['prior_pose_or_track_cache_inputs']:
            raise ValueError('Inference used non-video inputs')
        bound_runs[vid]=sha(folder/'run.json')
        for name in ('legacy_landings','all_ready_auxiliary_landings','experimental_landings'):
            path=folder/(name+'.jsonl');frozen_predictions[str(path)]=sha(path)
    save(out/'prediction_freeze.json',{'split_id':manifest['split_id'],'model_sha256':plan['model_sha256'],
         'prediction_sha256':frozen_predictions,'run_sha256':bound_runs,'all_videos_completed':len(records),
         'evaluation_labels_opened':False})
    # No evaluator or event labels are loaded until every complete video result is frozen.
    evaluator_path=ROOT/'development/evaluation/evaluate_landings.py'
    spec=importlib.util.spec_from_file_location('full_video_frozen_evaluator',evaluator_path)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    candidate_path=ROOT/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'
    public_path=ROOT/'participant/participant/pingpang/public_data/gt_public.jsonl'
    for path in (candidate_path,public_path):
        if sha(path)!=plan['evaluation_label_sha256'][str(path)]:raise ValueError('Evaluation labels changed since plan freeze')
    candidates=[dict(r,valid=True) for r in lines(candidate_path)];public=lines(public_path)
    snapshots=out/'label_snapshots';snapshots.mkdir()
    for path in (candidate_path,public_path):(snapshots/path.name).write_bytes(path.read_bytes())
    results={}
    for split in ('validation_reserved','public_regression'):
        subset=[r for r in records if r['split']==split];ids={r['video_id'] for r in subset}
        gt=[r for r in (candidates if split=='validation_reserved' else public) if r['video_id'] in ids]
        metadata=[dict(r,angle=r['official_angle']) for r in subset];policies={}
        for name in ('legacy_landings','all_ready_auxiliary_landings','experimental_landings'):
            predictions=[dict(p,video_id=r['video_id']) for r in subset for p in lines(out/'videos'/r['video_id']/(name+'.jsonl'))]
            score=evaluator.evaluate(predictions,gt,metadata,exempt_invalid=False)
            if split=='validation_reserved':score['angle_macro_f1']=None;score['accuracy_score_estimate']=None
            policies[name]=score
        results[split]={'videos':len(subset),'frames':sum(r['n_frames'] for r in subset),'policies':policies,
            'label_status':'Published stage candidates, matching proxy, not frozen final complete-event GT' if split=='validation_reserved' else 'Official public event GT; development regression only',
            'independent_blind_test':False}
    for path,expected in frozen_predictions.items():
        if sha(path)!=expected:raise ValueError('Predictions changed while scoring')
    timing={'videos':len(records),'frames':sum(r['n_frames'] for r in records),'batch_wall_seconds':seconds,
        'batch_throughput_fps':sum(r['n_frames'] for r in records)/seconds,'workers':plan['workers'],
        'threads_per_worker':plan['threads_per_worker'],'sum_video_inference_seconds':sum(r['inference_seconds'] for r in completed),
        'model_initialisation_and_disk_io_in_batch_wall_time':True,'50fps_acceptance_passed':False}
    report={'split_id':manifest['split_id'],'fresh_complete_video_inference':True,'all_videos_completed':True,
        'model_sha256':plan['model_sha256'],'source_plan_sha256':sha(out/'evaluation_plan.json'),
        'prediction_freeze_sha256':sha(out/'prediction_freeze.json'),'results':results,'timing':timing,
        'reserved_validation_used_for_training_or_thresholds':False,'evaluation_outcomes_used_to_modify_model_or_policy':False,
        'prior_global_development_exposure':True,'production_adoption_enabled':False,
        'full_project_acceptance_complete':False,'evaluation_label_sha256':plan['evaluation_label_sha256'],
        'evaluator_sha256':sha(evaluator_path)}
    save(out/'evaluation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=2);a=parser.parse_args()
    manifest=load_protocol();model=read(a.model);check_model_for_formal_validation(model,manifest)
    records=[]
    for split,purpose in [('validation_reserved','validation'),('public_regression','public_regression')]:
        records.extend(select(manifest,[r['video_id'] for r in manifest['videos'] if r['split']==split],purpose))
    verify_inputs(records)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'videos').mkdir();(out/'logs').mkdir()
    from development.detection.video_contact_pipeline import PROJECT as pipeline_project
    source_files=[PROJECT/'development'/sub/p for sub,names in [('detection',['video_contact_pipeline.py','video_ball_detector.py','racket_detector.py','tt3d_segmenter.py','auto_table_geometry.py','table_border_lines.py']),
        ('bounce',['trajectory_events.py','wrist_evidence.py','temporal_contact.py','person_motion_contact.py','racket_contact_evidence.py','contact_direction.py','contact_classifier.py','contact_output_policy.py','ge_rule_v2.py','table_geometry.py'])] for p in names]
    gt_paths=[ROOT/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl',
              ROOT/'participant/participant/pingpang/public_data/gt_public.jsonl']
    plan={'split_id':manifest['split_id'],'split_manifest_sha256':sha(DEFAULT/'split_manifest.json'),
          'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'model_training_provenance':model['training_provenance'],
          'model_output_policy':model['output_policy'],'pipeline_source_hashes':{str(p):sha(p) for p in source_files},
          'evaluation_label_sha256':{str(p):sha(p) for p in gt_paths},'video_ids':[r['video_id'] for r in records],
          'video_sha256':{r['video_id']:r['video_sha256'] for r in records},'workers':a.workers,'threads_per_worker':a.threads,
          'video_inference_reads_human_labels':False,'inference_inputs':'Each video and fixed model assets only',
          'policy_and_thresholds_frozen_before_video_evaluation':True,'runner_sha256':sha(Path(__file__))}
    save(out/'evaluation_plan.json',plan);(out/'model_snapshot.json').write_bytes(a.model.read_bytes())
    source=out/'source';source.mkdir()
    for path in source_files+[Path(__file__)]:
        relative=path.relative_to(PROJECT);target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    completed=[];failed=[];started=time.monotonic();jobs=[]
    for r in records:
        jobs.append({**r,'output':str(out/'videos'/r['video_id']),'log':str(out/'logs'/(r['video_id']+'.log')),
                     'model_path':str(a.model.resolve()),'threads':a.threads})
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialise,initargs=(a.threads,)) as pool:
        pending={pool.submit(process,job):job for job in jobs}
        while pending:
            done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:result=future.result()
                except Exception as error:result={'video_id':job['video_id'],'status':'failed','error':str(error)}
                (completed if result['status']=='completed' else failed).append(result)
                print('Full video',len(completed)+len(failed),'/',len(records),result['video_id'],result['status'],flush=True)
            active=[job['video_id'] for job in pending.values() if Path(job['log']).exists()]
            save(out/'progress.json',{'status':'running','completed':completed,'failed':failed,'total':len(records),
                 'active_videos':active,'elapsed_seconds':time.monotonic()-started,'evaluation_labels_opened':False})
    if failed:
        save(out/'progress.json',{'status':'failed','completed':completed,'failed':failed,'total':len(records)})
        raise RuntimeError('Some complete-video runs failed; no full evaluation reported')
    report=evaluate(out,manifest,records,plan,completed,time.monotonic()-started)
    save(out/'progress.json',{'status':'completed','completed':completed,'failed':[],'total':len(records),
         'evaluation_sha256':sha(out/'evaluation.json'),'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({split:{k:v['overall'] for k,v in result['policies'].items()} for split,result in report['results'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

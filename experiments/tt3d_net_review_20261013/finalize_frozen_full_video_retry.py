"""Finalize a frozen full-video batch after a decoder-only successful retry."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import load_protocol,select,sha
from evaluate_frozen_full_videos import evaluate


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--retry',type=Path,required=True);parser.add_argument('--video-id',required=True);a=parser.parse_args()
    out=a.evaluation.resolve();progress=read(out/'progress.json');plan=read(out/'evaluation_plan.json')
    if progress['status']!='failed':raise ValueError('Original batch must finish before finalization')
    if [r['video_id'] for r in progress['failed']]!=[a.video_id]:raise ValueError('Additional failed videos require separate repair')
    protocol=load_protocol();record=next(r for r in protocol['videos'] if r['video_id']==a.video_id)
    report=read(a.retry/'run.json')
    if report['source_hashes']!=plan['pipeline_source_hashes'] or report['contact_classifier']['model_sha256']!=plan['model_sha256']:
        raise ValueError('Retry modified frozen model/rules')
    if report['video_sha256']!=record['video_sha256'] or report['n_frames']!=record['n_frames']:
        raise ValueError('Retry changed video input/frame count')
    adapter=report['decoder_compatibility_adapter']
    if not adapter['model_and_policy_unchanged'] or adapter['evaluation_labels_used']:
        raise ValueError('Retry is not isolated decoder compatibility')
    if any(r['other_pts_changed'] or r['frames_reordered'] for r in adapter['repairs']):
        raise ValueError('Retry changed frame order or non-startup timestamps')
    target=out/'videos'/a.video_id
    if list(target.iterdir()):raise ValueError('Failed output is not empty; preserve and audit it before transfer')
    for path in a.retry.iterdir():
        if path.is_file():shutil.copy2(path,target/path.name)
    completed=progress['completed']+[{'video_id':a.video_id,'status':'completed','run_sha256':sha(target/'run.json'),
        'frames':report['n_frames'],'inference_seconds':report['inference_seconds'],
        'experimental_contacts':report['experimental_landings_count'],'decoder_retry':True}]
    expected=set(plan['video_ids'])
    if len(completed)!=len(expected) or {r['video_id'] for r in completed}!=expected:raise ValueError('Incomplete final video batch')
    save(out/'original_batch_failure.json',progress)
    save(out/'retry_completion.json',{'video_id':a.video_id,'source':str(a.retry),'run_sha256':sha(target/'run.json'),
         'decoder_adapter':adapter,'all_other_video_outputs_unchanged':True,
         'model_thresholds_rules_unchanged':True,'new_evaluation_labels_read_before_retry':False})
    records=[]
    for split,purpose in [('validation_reserved','validation'),('public_regression','public_regression')]:
        records.extend(select(protocol,[r['video_id'] for r in protocol['videos'] if r['split']==split],purpose))
    # Plan timestamp precedes model startup/decoding. This conservative wall
    # duration includes decoder-repair attempts and overlap with a third process.
    seconds=time.time()-(out/'evaluation_plan.json').stat().st_mtime
    result=evaluate(out,protocol,records,plan,completed,seconds)
    result['decoder_compatibility_retry']=read(out/'retry_completion.json')
    result['timing']['wall_time_basis']='Elapsed since frozen plan file, including failed and decoder-repair attempts'
    result['timing']['temporary_retry_worker_count']=1
    result['timing']['official_speed_benchmark']=False
    save(out/'evaluation.json',result)
    save(out/'progress.json',{'status':'completed','completed':completed,'failed':[],'total':len(records),
         'original_failed_video_repaired':a.video_id,'evaluation_sha256':sha(out/'evaluation.json'),'elapsed_seconds':seconds})
    (out/'source_finalize_frozen_full_video_retry.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({split:{k:v['overall'] for k,v in value['policies'].items()} for split,value in result['results'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

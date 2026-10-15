"""Optional optimized contact profile: one video and fixed application assets."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
ROOT=PROJECT.parents[1]
from development.detection.video_contact_pipeline import run as base_run,sha,save,save_lines
from development.bounce.learned_curved_policy import revised_landings

DEFAULT_POLICY=PROJECT/'data/contact_classifier/source25_curved_contact_v1/policy.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def run_optimized(video,out,models=None,policy_path=DEFAULT_POLICY):
    policy=read(policy_path)
    policy_sha=sha(policy_path)
    classifier=PROJECT/'data/contact_classifier/source25_train_20261014_v1/model.json'
    if sha(classifier)!=policy['model_sha256']:raise ValueError('Fixed hit classifier differs from optimized contact profile')
    report=base_run(video,out,models=models)
    started=time.monotonic();out=Path(out)
    tracks=lines(out/'ball_tracks.jsonl');samples=lines(out/'geometry_samples.jsonl')
    predictions,cases,decisions=revised_landings(tracks,samples,lines(out/'legacy_landings.jsonl'),
        read(out/'proposals.json'),read(out/'event_candidates.json')['cases'],policy['table_model'],policy['config'],image_size=report['image_size'])
    if sha(policy_path)!=policy_sha or sha(classifier)!=policy['model_sha256']:
        raise ValueError('Fixed model assets changed during analysis')
    save_lines(out/'optimized_landings.jsonl',predictions)
    save(out/'optimized_event_candidates.json',{'cases':cases})
    save(out/'optimized_contact_decisions.json',{'cases':decisions})
    extra=time.monotonic()-started
    sources=[Path(__file__)]+[PROJECT/'development/bounce'/name for name in ('learned_curved_policy.py','curved_contact_context.py','curved_contact_classifier.py','curved_tree_classifier.py','curved_contact_candidates.py','automatic_ballistic_contact.py','automatic_net_zone.py','contact_consensus.py')]
    sources += [PROJECT/'development/detection/tt3d_camera_estimation.py',PROJECT/'development/detection/auto_table_geometry.py']
    report['curved_contact_optimization']={'policy_sha256':policy_sha,'policy_path':str(Path(policy_path).resolve()),
        'config':policy['config'],'training_provenance':policy['training_provenance'],
        'optimized_landings_count':len(predictions),'additional_seconds':extra,
        'runtime_manual_inputs':[],'prior_external_cache_inputs':[],
        'evaluation_status':'experimental_cached_regression_only_not_blind_or_speed_accepted',
        'camera_assumptions_verified':False,'output_file':'optimized_landings.jsonl'}
    report['curved_contact_optimization']['source_hashes']={str(p):sha(p) for p in sources}
    report['inference_seconds']+=extra;report['elapsed_seconds']+=extra
    save(out/'run.json',report)
    print('Optimized contact output:',out/'optimized_landings.jsonl',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or ROOT/'outputs/optimized_video_contacts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_optimized(args.video,out)

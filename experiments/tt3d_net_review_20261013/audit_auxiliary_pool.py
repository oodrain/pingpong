"""Candidate-pool coverage only; preserves baseline outputs exactly."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.trajectory_events import auxiliary_candidate_pool
from development.evaluation.evaluate_landings import evaluate
from fit_contact_pilot import sha

def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

def run(folder,public,manifest):
    baseline=read(folder/'baseline_candidates.jsonl');vector=read(folder/'vector_candidates.jsonl');pool=auxiliary_candidate_pool(baseline,vector)
    assert pool['primary_predictions']==baseline
    gt=read(public/'gt_public.jsonl');videos=json.loads(manifest.read_text(encoding='utf-8'))['videos']
    score=evaluate(pool['primary_predictions']+pool['auxiliary_candidates'],gt,videos,exempt_invalid=False)
    output={'baseline_preserved_exactly':True,'auxiliary_adoption_enabled':False,'candidate_pool_not_final_predictions':True,
            'primary_count':len(baseline),'auxiliary_count':len(pool['auxiliary_candidates']),
            'candidate_pool_matched_gt':score['overall']['tp'],'valid_gt':score['overall']['valid_gt'],'by_angle':score['by_angle'],
            'sources':{str(folder/'baseline_candidates.jsonl'):sha(folder/'baseline_candidates.jsonl'),str(folder/'vector_candidates.jsonl'):sha(folder/'vector_candidates.jsonl')},
            'rule_sha256':sha(Path(__file__).resolve().parents[2]/'development/bounce/trajectory_events.py'),'script_sha256':sha(__file__),**pool}
    target=folder/'auxiliary_pool.json'
    if target.exists():raise ValueError('Output exists')
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Baseline',len(baseline),'auxiliary',output['auxiliary_count'],'pool matched GT',output['candidate_pool_matched_gt'],'of',output['valid_gt'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--public',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();run(a.folder,a.public,a.manifest)

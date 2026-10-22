"""Replay revised geometry at the original sample times, without inference."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.detection.auto_table_geometry import AutoGeometryTimeline
from development.bounce.ge_rule_v2 import RuleConfig,replay
from development.evaluation.evaluate_landings import evaluate,validate_input
from development.evaluation.run_auto_table_v2 import read,rows,write,write_rows,sha
from development.evaluation.run_geometry_d_v2 import changed_matches


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    summary=read(args.run/'summary.json')
    data=args.data_root
    public=data/'participant/participant/pingpang/public_data'
    public_tracks=data/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    manifests={'public':read(public/'manifest.json')['videos'],
               'train':read(train/'selected_manifest.json')['videos']}
    samples=defaultdict(list)
    path=args.run/'fixed_sampling_geometry_samples.jsonl'
    for sample in rows(path):samples[sample['group'],sample['video_id']].append(sample)
    predictions=defaultdict(list);decisions=[];inputs=[path]
    cfg=RuleConfig()
    if summary['rule_config']!=asdict(cfg):raise ValueError('Rule configuration differs')
    for group,meta in manifests.items():
        tracks=public_tracks if group=='public' else train
        for m in meta:
            raw_path=tracks/'raw'/(m['video_id']+'.jsonl');inputs.append(raw_path)
            if summary['input_sha256'].get(str(raw_path.resolve()))!=sha(raw_path):
                raise ValueError('Raw trajectory changed')
            raw=rows(raw_path)
            if len(raw)!=m['n_frames'] or any(r['input_frame_id']!=i for i,r in enumerate(raw)):
                raise ValueError('Raw cache misaligned')
            timeline=AutoGeometryTimeline(samples[group,m['video_id']])
            ps,ds=replay(raw,timeline,'v1a',cfg)
            predictions[group]+=[dict(p,video_id=m['video_id']) for p in ps]
            decisions += [dict(d,group=group,video_id=m['video_id']) for d in ds]
    gt_paths={'public':public/'gt_public.jsonl',
        'train':data/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'}
    results={}
    for group,gt_path in gt_paths.items():
        inputs.append(gt_path)
        if summary['input_sha256'].get(str(gt_path.resolve()))!=sha(gt_path):
            raise ValueError('Scoring labels changed')
        gt=rows(gt_path)
        if group=='train':gt=[dict(g,valid=True) for g in gt]
        validate_input(gt,{m['video_id']:m for m in manifests[group]},ground_truth=True)
        results[group]=evaluate(predictions[group],gt,manifests[group],exempt_invalid=False)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'metrics.json',results)
    for group,ps in predictions.items():write_rows(args.output/(group+'_predictions.jsonl'),ps)
    write_rows(args.output/'candidate_decisions.jsonl',decisions)
    baseline=read(PROJECT/'outputs/experiments/auto_table_v2/runs/20261012T155647_594418Z/metrics.json')
    changes=changed_matches(baseline['public']['auto_region']['without_ignore'],results['public'],'revised_fixed')
    write_rows(args.output/'public_changed_matches.jsonl',changes)
    write(args.output/'summary.json',dict(model_inference_repeated=False,
        sample_seconds=summary['sample_seconds'],additional_warmup_samples_used=False,
        rule_config=asdict(cfg),public=results['public']['overall'],train_proxy=results['train']['overall'],
        public_lost_tp=sum(c['change']=='lost_tp' for c in changes),
        input_sha256={str(p.resolve()):sha(p) for p in inputs},source_sha256=sha(Path(__file__))))
    print({group:r['overall'] for group,r in results.items()})


if __name__=='__main__':main()

"""Retrospective prefix-group audit, without claiming fresh holdout evidence."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.evaluation.evaluate_landings import metrics
from development.evaluation.run_auto_table_v2 import read,write,write_rows,sha


def group_id(video_id):
    parts=video_id.split('_')
    if len(parts)!=2 or not all(p.isdigit() for p in parts):raise ValueError('Unexpected video id')
    return parts[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--geometry-run',type=Path,required=True)
    ap.add_argument('--previous-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest_path=args.data_root/'C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json'
    ms=read(manifest_path)['videos'];groups=defaultdict(list)
    for m in ms:groups[group_id(m['video_id'])].append(m['video_id'])
    current=read(args.data_root/'outputs/experiments/interframe_bounce_v0/CURRENT.json')
    split_path=args.data_root/'outputs/experiments/interframe_bounce_v0'/current['repository_reuse_run']/'split.json'
    split=read(split_path)['assignments']
    old_path=args.previous_run/'window_rule_comparison/metrics.json';new_path=args.geometry_run/'window_rule_comparison/metrics.json'
    reports={'window':read(new_path)['paired']['train']['window_v1']['without_ignore'],
             'previous_geometry':read(old_path)['paired']['train']['window_v1_auto_region']['without_ignore'],
             'current_geometry':read(new_path)['paired']['train']['window_v1_auto_region']['without_ignore']}
    indexed={name:{r['video_id']:r for r in report['by_video']} for name,report in reports.items()}
    for x in indexed.values():
        if set(x)!={m['video_id'] for m in ms}:raise ValueError('Video populations differ')
    records=[]
    for gid,vids in sorted(groups.items()):
        record={'group_id':gid,'video_ids':vids,'old_split_subsets':sorted({split[v] for v in vids})}
        for name,idx in indexed.items():
            record[name]=metrics(*(sum(idx[v][k] for v in vids) for k in ('tp','fp','fn')))
        record['delta_f1_vs_previous']=record['current_geometry']['f1']-record['previous_geometry']['f1']
        records.append(record)
    # A prospective grouping manifest: no metric-informed selection of groups.
    assignments={v:'fold_'+str(int(hashlib.sha256(('table-geometry-group-v1:'+g).encode()).hexdigest()[:8],16)%5) for g,vs in groups.items() for v in vs}
    grouped={'kind':'prefix_group_five_fold_for_future_experiments','assignments':assignments,
             'group_by_video':{v:g for g,vs in groups.items() for v in vs},
             'source_identity_verified':False,'groups_do_not_use_annotation_owner':True,
             'note':'Directory-prefix isolation only; original source identities may span prefixes. Existing data were already examined; this does not create a fresh holdout.'}
    write(args.output/'prospective_group_folds.json',grouped)
    write_rows(args.output/'by_prefix_group.jsonl',records)
    summary={'groups':len(groups),'videos':len(ms),'old_split_crossing_groups':sum(len(r['old_split_subsets'])>1 for r in records),
             'group_macro_f1':{name:sum(r[name]['f1'] for r in records)/len(records) for name in reports},
             'improved_groups_vs_previous':sum(r['delta_f1_vs_previous']>1e-10 for r in records),
             'worsened_groups_vs_previous':sum(r['delta_f1_vs_previous']<-1e-10 for r in records),
             'status':'retrospective_group_diagnostic_not_independent_generalization',
             'source_identity_verified':False,
             'input_sha256':{str(p.resolve()):sha(p) for p in (manifest_path,split_path,old_path,new_path)},'source_sha256':sha(Path(__file__))}
    write(args.output/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

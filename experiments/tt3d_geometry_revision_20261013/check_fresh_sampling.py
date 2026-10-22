"""Fresh model/scheduler verification for the 21 review videos.

Other 148 videos retain the verified frozen-observation replay. The combined
artifact explicitly records mixed sampling provenance; it is not 169 fresh runs.
"""
import argparse
from collections import defaultdict,Counter
import os
from pathlib import Path
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.detection.auto_table_geometry import GeometryConfig,AutoGeometryTimeline
from development.detection.tt3d_segmenter import TT3DSegmenter
from development.evaluation.run_auto_table_v2 import collect,read,rows,write,write_rows,sha
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.validate_table_geometry import corner_error
from development.bounce.ge_rule_v2 import RuleConfig,replay
from development.evaluation.evaluate_landings import evaluate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();cv2.setNumThreads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ('diagnostics','fixed_sampling','fresh_samples'): (out/name).mkdir()
    pack=PROJECT/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    manifest=read(pack/'manifest.json');refs=latest_reviews(pack,manifest)
    keys={(c['group'],c['video_id']) for c in manifest['cases']}
    data=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])));public=data/'participant/participant/pingpang/public_data'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    ms={'public':read(public/'manifest.json')['videos'],'train':read(train/'selected_manifest.json')['videos']}
    metas={(g,m['video_id']):m for g,videos in ms.items() for m in videos}
    summary=read(args.base/'summary.json');cfg=GeometryConfig()
    model=TT3DSegmenter(PROJECT/'outputs/models/table_segmentation.ckpt')
    if model.provenance!=summary['model']:raise ValueError('Model provenance changed')
    fresh=[]
    for index,key in enumerate(sorted(keys)):
        group,vid=key;m=metas[key]
        path=public/m['path'] if group=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(path)!=summary['input_sha256'][str(path.resolve())]:raise ValueError('Video changed')
        ss,elapsed=collect(model,m,path,group,out,cfg,.4,.12)
        fresh+=ss;write_rows(out/'fresh_samples'/f'{group}_{vid}.jsonl',ss)
        print('FRESH',index+1,'/',len(keys),group,vid,'samples',len(ss),'seconds',round(elapsed,1),flush=True)
    combined=[s for s in rows(args.base/'geometry_samples.jsonl') if (s['group'],s['video_id']) not in keys]+fresh
    combined.sort(key=lambda s:(s['group'],s['video_id'],s['frame_id']))
    write_rows(out/'geometry_samples.jsonl',combined)
    fixed=[s for s in rows(args.base/'fixed_sampling_geometry_samples.jsonl') if (s['group'],s['video_id']) not in keys]
    fixed += [s for p in sorted((out/'fixed_sampling').glob('*.jsonl')) for s in rows(p)]
    fixed.sort(key=lambda s:(s['group'],s['video_id'],s['frame_id']))
    write_rows(out/'fixed_sampling_geometry_samples.jsonl',fixed)
    groups=defaultdict(list)
    for s in combined:groups[s['group'],s['video_id']].append(s)
    timelines={k:AutoGeometryTimeline(v) for k,v in groups.items()}
    quality=[dict(group=g,video_id=v,samples=len(ss),any_usable=any(s['usable'] for s in ss),
                  states=dict(Counter(s['state'] for s in ss))) for (g,v),ss in groups.items()]
    write_rows(out/'geometry_quality.jsonl',quality)
    records=read(args.base/'reference_records.json')
    by_case={c['case_id']:c for c in manifest['cases']}
    for r in records:
        c=by_case[r['case_id']];tl=timelines[c['group'],c['video_id']]
        r['frozen_replay_temporal_sample']=r['temporal_sample'];r['frozen_replay_temporal_usable']=r['temporal_usable']
        r['temporal_sample']=tl.sample(c['frame_id']);r['temporal_usable']=tl.lookup(c['frame_id']) is not None
        r.pop('temporal_error',None)
        if r['temporal_usable'] and refs[r['case_id']]['status']=='corners_visible':
            r['temporal_error']=corner_error(r['temporal_sample']['corners'],refs[r['case_id']]['corners'])
    write(out/'reference_records.json',records)
    es=[r['temporal_error'] for r in records if 'temporal_error' in r]
    predictions=defaultdict(list);scores={};decisions=[]
    for g,videos in ms.items():
        tracks=data/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1' if g=='public' else train
        for m in videos:
            p=tracks/'raw'/(m['video_id']+'.jsonl')
            if sha(p)!=summary['input_sha256'][str(p.resolve())]:raise ValueError('Raw input changed')
            ps,ds=replay(rows(p),timelines[g,m['video_id']],'v1a',RuleConfig())
            predictions[g]+=[dict(p,video_id=m['video_id']) for p in ps]
            decisions+=[dict(d,group=g,video_id=m['video_id']) for d in ds]
        p=public/'gt_public.jsonl' if g=='public' else data/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'
        if sha(p)!=summary['input_sha256'][str(p.resolve())]:raise ValueError('Labels changed')
        gt=rows(p)
        if g=='train':gt=[dict(r,valid=True) for r in gt]
        scores[g]=evaluate(predictions[g],gt,videos,exempt_invalid=False)
        write_rows(out/(g+'_predictions.jsonl'),predictions[g])
    write(out/'metrics.json',scores);write_rows(out/'candidate_decisions.jsonl',decisions)
    summary.update(frozen_observation_source=str(args.base),model_inference_repeated=True,
        sampling_provenance=dict(fresh_model_and_scheduler_videos=len(keys),frozen_observation_replay_videos=len(groups)-len(keys),
                                fresh_video_ids=[dict(group=g,video_id=v) for g,v in sorted(keys)]),
        adaptive_scheduler_rerun='21 review videos only',
        coverage={g:dict(videos=len(videos),any_usable=sum(q['any_usable'] for q in quality if q['group']==g)) for g,videos in ms.items()},
        temporal_usable_review_frames=sum(r['temporal_usable'] for r in records),
        temporal_errors=dict(comparisons=len(es),mean_corner_error_px=float(np.mean([e['mean_corner_error_px'] for e in es])) if es else None,
            maximum_corner_error_px=max((e['maximum_corner_error_px'] for e in es),default=None),cases_over_30px=sum(e['maximum_corner_error_px']>30 for e in es)),
        event_metrics={g:r['overall'] for g,r in scores.items()})
    summary['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    write(out/'summary.json',summary)
    print('DONE',summary['temporal_errors'],summary['coverage'],flush=True)


if __name__=='__main__':main()

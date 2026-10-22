"""RGB refinement and causal replay on frozen, independently verified NN seeds.

Observation times are frozen from revision 1, not a rerun of its adaptive
scheduler. No event labels or human corners enter inference.
"""
import argparse
import ast
from collections import defaultdict,Counter
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.detection.auto_table_geometry import (GeometryConfig,CausalGeometryTracker,
    AutoGeometryTimeline,validate_quad_observation,extract_quad)
from development.evaluation.run_auto_table_v2 import read,rows,write,write_rows,sha
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.validate_table_geometry import corner_error
from development.bounce.ge_rule_v2 import RuleConfig,replay
from development.evaluation.evaluate_landings import evaluate


def verify_seed(source):
    old=ast.parse((source/'source/auto_table_geometry.py').read_text(encoding='utf-8-sig'))
    new=ast.parse((PROJECT/'development/detection/auto_table_geometry.py').read_text(encoding='utf-8-sig'))
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='extract_quad')
    b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_seed')
    a.name=b.name
    if ast.dump(a)!=ast.dump(b):raise ValueError('NN seed extraction changed; cache invalid')
    cfg=asdict(GeometryConfig());original=read(source/'summary.json')['geometry_config']
    if any(cfg[k]!=v for k,v in original.items()):raise ValueError('Seed configuration changed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    cv2.setNumThreads(4)
    source=PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/regression_all_01'
    verify_seed(source)
    frozen=read(source/'summary.json');cfg=GeometryConfig()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir();(out/'overlays').mkdir()
    sources=[Path(__file__),PROJECT/'development/detection/auto_table_geometry.py',
             PROJECT/'development/detection/table_border_lines.py']
    for p in sources:shutil.copy2(p,out/'source'/p.name)
    seeds=defaultdict(list)
    for s in rows(source/'geometry_samples.jsonl'):seeds[s['group'],s['video_id']].append(s)
    fixed_ids={(s['group'],s['video_id'],s['frame_id']) for s in rows(source/'fixed_sampling_geometry_samples.jsonl')}
    data=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])))
    public=data/'participant/participant/pingpang/public_data'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    manifests={'public':read(public/'manifest.json')['videos'],'train':read(train/'selected_manifest.json')['videos']}
    metas={(g,m['video_id']):m for g,ms in manifests.items() for m in ms}
    all_samples=[];fixed=[];quality=[];paths=[]
    for index,(key,observations) in enumerate(seeds.items()):
        group,vid=key;m=metas[key]
        path=public/m['path'] if group=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(path)!=frozen['input_sha256'][str(path.resolve())]:raise ValueError('Video changed')
        paths.append(path)
        cap=cv2.VideoCapture(str(path));tracker=CausalGeometryTracker(cfg);fixed_tracker=CausalGeometryTracker(cfg)
        by_frame={s['frame_id']:s for s in observations};local=[]
        try:
            for f in range(m['n_frames']):
                if not cap.grab():raise ValueError('Video truncated')
                if f not in by_frame:continue
                ok,im=cap.retrieve()
                if not ok or im.shape[:2]!=(m['height'],m['width']):raise ValueError('Invalid image')
                old=by_frame[f]
                thumb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,36))
                if not np.array_equal(thumb,np.array(old['scene_thumbnail'])):raise ValueError('Frame identity differs')
                proposal=validate_quad_observation(im,old['raw_proposal'],cfg)
                s=tracker.update(f,im,proposal)
                s.update(group=group,video_id=vid,raw_proposal=proposal,scene_thumbnail=thumb.tolist())
                local.append(s)
                if (group,vid,f) in fixed_ids:
                    fs=fixed_tracker.update(f,im,proposal)
                    fs.update(group=group,video_id=vid,raw_proposal=proposal)
                    fixed.append(fs)
            if cap.grab():raise ValueError('Video extra frames')
        finally:cap.release()
        all_samples+=local
        quality.append(dict(group=group,video_id=vid,samples=len(local),any_usable=any(s['usable'] for s in local),
            usable_samples=sum(s['usable'] for s in local),states=dict(Counter(s['state'] for s in local)),
            reasons=dict(Counter(s['reason'] for s in local))))
        if (index+1)%10==0 or index==len(seeds)-1:print('VIDEOS',index+1,'/',len(seeds),flush=True)
    write_rows(out/'geometry_samples.jsonl',all_samples)
    write_rows(out/'fixed_sampling_geometry_samples.jsonl',fixed)
    write_rows(out/'geometry_quality.jsonl',quality)
    mapping=defaultdict(list)
    for s in all_samples:mapping[s['group'],s['video_id']].append(s)
    timelines={k:AutoGeometryTimeline(v) for k,v in mapping.items()}
    # Scoring is strictly after inference, using immutable human references.
    pack=PROJECT/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    manifest=read(pack/'manifest.json');refs=latest_reviews(pack,manifest)
    records=[];cache=PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
    previous={r['case_id']:r for r in read(PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/regression_evaluation_01/records.json')}
    for case in manifest['cases']:
        im=cv2.imread(str(pack/case['image_url']))
        if sha(pack/case['image_url'])!=case['image_sha256']:raise ValueError('Review frame changed')
        p=extract_quad(im,np.load(cache/(case['case_id']+'.npz'))['probability'])
        ref=refs[case['case_id']];tl=timelines[case['group'],case['video_id']]
        s=tl.sample(case['frame_id']);usable=tl.lookup(case['frame_id']) is not None
        r=dict(case_id=case['case_id'],human_status=ref['status'],proposal=p,temporal_sample=s,
               temporal_usable=usable,previous=previous[case['case_id']])
        for name,q,good in [('single',p,p['usable_proposal']),('temporal',s,usable)]:
            if good and ref['status']=='corners_visible':r[name+'_error']=corner_error(q['corners'],ref['corners'])
        display=im.copy()
        for q,color in [(ref.get('corners'),(0,220,0)),(previous[case['case_id']]['revised_proposal'].get('corners'),(220,0,220)),(p.get('corners') if p['usable_proposal'] else None,(0,220,255))]:
            if q:cv2.polylines(display,[np.rint(q).astype(np.int32)],True,color,3)
        cv2.imwrite(str(out/'overlays'/(case['case_id']+'.png')),display)
        records.append(r)
    write(out/'reference_records.json',records)
    shutil.copy2(pack/'references.jsonl',out/'references_snapshot.jsonl')
    def errors(name):
        es=[r[name+'_error'] for r in records if name+'_error' in r]
        return dict(comparisons=len(es),mean_corner_error_px=float(np.mean([e['mean_corner_error_px'] for e in es])) if es else None,
            maximum_corner_error_px=max((e['maximum_corner_error_px'] for e in es),default=None),
            cases_over_30px=sum(e['maximum_corner_error_px']>30 for e in es))
    predictions=defaultdict(list);decisions=[]
    for group,ms in manifests.items():
        tracks=data/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1' if group=='public' else train
        for m in ms:
            rawpath=tracks/'raw'/(m['video_id']+'.jsonl')
            if sha(rawpath)!=frozen['input_sha256'][str(rawpath.resolve())]:raise ValueError('Trajectory changed')
            ps,ds=replay(rows(rawpath),timelines[group,m['video_id']],'v1a',RuleConfig())
            predictions[group]+=[dict(p,video_id=m['video_id']) for p in ps]
            decisions+=[dict(d,group=group,video_id=m['video_id']) for d in ds]
    scores={}
    for group in manifests:
        gtpath=public/'gt_public.jsonl' if group=='public' else data/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'
        if sha(gtpath)!=frozen['input_sha256'][str(gtpath.resolve())]:raise ValueError('Labels changed')
        gt=rows(gtpath)
        if group=='train':gt=[dict(g,valid=True) for g in gt]
        scores[group]=evaluate(predictions[group],gt,manifests[group],exempt_invalid=False)
        write_rows(out/(group+'_predictions.jsonl'),predictions[group])
    write(out/'metrics.json',scores);write_rows(out/'candidate_decisions.jsonl',decisions)
    summary=dict(completed_videos=len(seeds),geometry_config=asdict(cfg),rule_config=asdict(RuleConfig()),
        model=frozen['model'],sample_seconds=frozen['sample_seconds'],input_sha256=frozen['input_sha256'],
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        frozen_observation_source=str(source),frozen_observations_sha256=sha(source/'geometry_samples.jsonl'),
        seed_extraction_ast_equivalent=True,model_inference_repeated=False,adaptive_scheduler_rerun=False,
        reference_sha256=sha(pack/'references.jsonl'),reference_status='single reviewer, development diagnostics',
        single_errors=errors('single'),temporal_errors=errors('temporal'),
        temporal_usable_review_frames=sum(r['temporal_usable'] for r in records),
        coverage={g:dict(videos=len(ms),any_usable=sum(q['any_usable'] for q in quality if q['group']==g)) for g,ms in manifests.items()},
        event_metrics={g:r['overall'] for g,r in scores.items()},adopted=False)
    write(out/'summary.json',summary)
    print('DONE',summary['single_errors'],summary['temporal_errors'],summary['coverage'],flush=True)


if __name__=='__main__':main()

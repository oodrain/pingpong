"""Paired audit against frozen single-review references and video inputs.

The review set informed development. This is a diagnostic comparison, not an
independent test of generalization. No review submissions or labels are edited.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.detection.auto_table_geometry import GeometryConfig,extract_quad,AutoGeometryTimeline
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.validate_table_geometry import corner_error
from development.evaluation.run_auto_table_v2 import read,rows,write,sha


def timelines(samples):
    groups=defaultdict(list)
    for sample in samples:groups[sample['group'],sample['video_id']].append(sample)
    return {key:AutoGeometryTimeline(value) for key,value in groups.items()}


def error_summary(records,field):
    selected=[r[field] for r in records if field in r]
    return dict(comparisons=len(selected),
        mean_corner_error_px=float(np.mean([r['mean_corner_error_px'] for r in selected])) if selected else None,
        maximum_corner_error_px=max((r['maximum_corner_error_px'] for r in selected),default=None),
        cases_over_30px=sum(r['maximum_corner_error_px']>30 for r in selected))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    pack=PROJECT/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    cache=PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
    manifest=read(pack/'manifest.json');refs=latest_reviews(pack,manifest)
    reference_hash=sha(pack/'references.jsonl');manifest_hash=sha(pack/'manifest.json')
    baseline_run=Path(manifest['geometry_run'])
    baseline_summary=read(baseline_run/'summary.json')
    if sha(baseline_run/'geometry_samples.jsonl')!=manifest['geometry_samples_sha256']:
        raise ValueError('Frozen baseline changed')
    provenance=read(cache/'provenance.json')
    if provenance['manifest_sha256']!=manifest_hash or provenance['model']!=baseline_summary['model']:
        raise ValueError('Cached segmentation identity differs')
    for name,digest in provenance['cache_sha256'].items():
        if sha(cache/name)!=digest:raise ValueError('Probability cache changed')
    baseline_frames={r['case_id']:r['same_frame_proposal'] for r in read(
        PROJECT/'outputs/experiments/tt3d_stage1_20261013/final_diagnostic_20261013_v2/records.json')}
    paths={'baseline':baseline_run/'geometry_samples.jsonl'}
    regression=None
    if args.run:
        regression=read(args.run/'summary.json')
        if regression['model']!=baseline_summary['model'] or regression['geometry_config']!=asdict(GeometryConfig()):
            raise ValueError('Regression provenance differs')
        for source in (PROJECT/'development/detection/auto_table_geometry.py',
                       PROJECT/'development/detection/table_border_lines.py'):
            if regression['source_sha256'].get(str(source.resolve()))!=sha(source):
                raise ValueError('Extraction source differs from regression')
        for path,digest in baseline_summary['input_sha256'].items():
            if regression['input_sha256'].get(path)!=digest:raise ValueError('Regression input changed: '+path)
        paths.update(revised_fixed=args.run/'fixed_sampling_geometry_samples.jsonl',
                     revised_adaptive=args.run/'geometry_samples.jsonl')
    temporal={name:timelines(rows(path)) for name,path in paths.items()}
    if args.run:
        identity=lambda path:{(s['group'],s['video_id'],s['frame_id']) for s in rows(path)}
        if identity(paths['baseline'])!=identity(paths['revised_fixed']):
            raise ValueError('Fixed sampling comparison frames differ')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'overlays').mkdir()
    (args.output/'references_snapshot.jsonl').write_bytes((pack/'references.jsonl').read_bytes())
    write(args.output/'manifest_snapshot.json',manifest)
    records=[]
    for case in manifest['cases']:
        path=pack/case['image_url']
        if sha(path)!=case['image_sha256']:raise ValueError('Original frame changed')
        image=cv2.imread(str(path));human=refs[case['case_id']]
        proposal=extract_quad(image,np.load(cache/(case['case_id']+'.npz'))['probability'])
        old=baseline_frames[case['case_id']]
        record=dict(case_id=case['case_id'],group=case['group'],video_id=case['video_id'],
                    frame_id=case['frame_id'],human_status=human['status'],revised_proposal=proposal,
                    baseline_proposal_usable=old['usable_proposal'])
        for name,p in (('baseline_same_frame',old),('revised_same_frame',proposal)):
            record[name+'_usable']=p['usable_proposal']
            if p['usable_proposal'] and human['status']=='corners_visible':
                record[name+'_error']=corner_error(p['corners'],human['corners'])
        for name,mapping in temporal.items():
            timeline=mapping[case['group'],case['video_id']]
            available=timeline.lookup(case['frame_id']) is not None
            sample=timeline.sample(case['frame_id'])
            record[name+'_usable']=available
            record[name+'_state']=sample['state'] if sample else None
            record[name+'_reason']=sample['reason'] if sample else None
            if available and human['status']=='corners_visible':
                record[name+'_error']=corner_error(sample['corners'],human['corners'])
        draw=image.copy()
        for points,color in ((old.get('corners'),(230,60,230)),
                             (proposal.get('corners') if proposal['usable_proposal'] else None,(0,220,255)),
                             (human['corners'],(60,230,60))):
            if points:cv2.polylines(draw,[np.rint(points).astype(np.int32)],True,color,2)
        cv2.rectangle(draw,(0,0),(draw.shape[1],58),(0,0,0),-1)
        cv2.putText(draw,case['case_id']+' | '+proposal['reason'],(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(draw,'green=human; purple=old candidate; yellow=accepted revised candidate',(8,45),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(args.output/'overlays'/(case['case_id']+'.jpg')),draw)
        records.append(record)
    versions=['baseline_same_frame','revised_same_frame']+list(temporal)
    summary=dict(cases=len(records),human_status_counts=dict(Counter(r['human_status'] for r in records)),
        comparison={name:dict(error_summary(records,name+'_error'),
                             usable_cases=sum(r[name+'_usable'] for r in records)) for name in versions},
        same_frame_reason_counts=dict(Counter(r['revised_proposal']['reason'] for r in records)),
        review_set_used_for_development=True,independent_generalization_validation=False,
        independent_double_review_verified=False,reference_sha256=reference_hash,manifest_sha256=manifest_hash,
        model=provenance['model'],geometry_config=asdict(GeometryConfig()),
        geometry_sha256={name:sha(path) for name,path in paths.items()},
        source_sha256={str(path):sha(path) for path in (Path(__file__),
            PROJECT/'development/detection/auto_table_geometry.py',PROJECT/'development/detection/table_border_lines.py')})
    paired={}
    for name in (name for name in versions if name not in ('baseline_same_frame','baseline')):
        base='baseline_same_frame' if name=='revised_same_frame' else 'baseline'
        pairs=[r for r in records if base+'_error' in r and name+'_error' in r]
        paired[name]=dict(cases=len(pairs),baseline=error_summary(pairs,base+'_error'),
                         revised=error_summary(pairs,name+'_error'))
    summary['paired_errors']=paired
    if regression:
        summary['regression_coverage']=regression['coverage']
        metrics=read(args.run/'metrics.json')
        summary['public_event_diagnostic']=metrics['public']['auto_region']['without_ignore']['overall']
        summary['train_event_proxy']=metrics['train_proxy']['auto_region']['overall']
        fixed=rows(paths['revised_fixed'])
        summary['fixed_sampling_coverage']={g:dict(samples=sum(s['group']==g for s in fixed),
            usable_samples=sum(s['group']==g and s['usable'] for s in fixed),
            any_usable_videos=len({s['video_id'] for s in fixed if s['group']==g and s['usable']})) for g in ('public','train')}
    if sha(pack/'references.jsonl')!=reference_hash:raise ValueError('Human submissions changed during evaluation')
    write(args.output/'records.json',records);write(args.output/'summary.json',summary)
    print(summary['comparison'],flush=True)


if __name__=='__main__':main()

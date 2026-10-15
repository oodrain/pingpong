"""Paired audit of the existing window rule with automatic geometry, without inference."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.bounce.rule_variants import CONFIGS,predict
from development.detection.auto_table_geometry import AutoGeometryTimeline
from development.evaluation.evaluate_landings import evaluate,metrics
from development.evaluation.run_geometry_d_v2 import changed_matches


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(path,data):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in data),encoding='utf-8')


def filter_candidates(candidates,timeline,config,margin=900):
    selected=[];decisions=[];last=-10000
    for p in candidates:
        frame=p['frame_id'];plane=timeline.lookup(frame)
        outside=plane.outside_distance(p['x'],p['y']) if plane else None
        context={timeline.interval_id(f) for f in (p['prev_frame'],frame,p['next_frame'])}
        context.discard(None)
        if len(context)>1:reason='geometry_segment_transition'
        elif outside is not None and outside>margin:reason='outside_table_margin'
        elif frame-last<config.min_gap:reason='temporal_dedup'
        else:
            selected.append({k:p[k] for k in ('frame_id','x','y')});last=frame
            reason='accepted_geometry' if plane else 'accepted_fallback'
        decisions.append(dict(p,geometry_available=plane is not None,outside_distance_mm=outside,
                              reason=reason,accepted=reason.startswith('accepted_')))
    return selected,decisions


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--geometry-run',type=Path,required=True)
    args=ap.parse_args();data=args.data_root
    out=args.geometry_run/'window_rule_comparison';out.mkdir(exist_ok=False)
    samples=rows(args.geometry_run/'geometry_samples.jsonl')
    inputs=[args.geometry_run/'geometry_samples.jsonl']
    cfg=next(c for c in CONFIGS if c.name=='V1_prom3_gapaware2')
    public=data/'participant/participant/pingpang/public_data'
    pub_tracks=data/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    manifests={'public':read(public/'manifest.json')['videos'],'train':read(train/'selected_manifest.json')['videos']}
    sources={'public':pub_tracks,'train':train}
    frozen_path=data/'outputs/bounce_rules154_20261012/V1_prom3_gapaware2_predictions.jsonl'
    frozen=rows(frozen_path);inputs.append(frozen_path)
    predictions=defaultdict(list);decisions=[]
    for group,manifest in manifests.items():
        for meta in manifest:
            vid=meta['video_id'];raw_path=sources[group]/'raw'/(vid+'.jsonl');inputs.append(raw_path)
            baseline,candidates=predict(rows(raw_path),meta,cfg)
            if group=='train' and baseline!=[{k:p[k] for k in ('frame_id','x','y')} for p in frozen if p['video_id']==vid]:
                raise ValueError('Existing V1 window baseline changed: '+vid)
            timeline=AutoGeometryTimeline([s for s in samples if s['group']==group and s['video_id']==vid])
            automatic,ds=filter_candidates(candidates,timeline,cfg)
            for name,ps in [('window_v1',baseline),('window_v1_auto_region',automatic)]:
                predictions[(group,name)]+=[dict(p,video_id=vid) for p in ps]
            decisions += [dict(d,group=group,video_id=vid) for d in ds]
    train_gt_path=data/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'
    gt={'public':rows(public/'gt_public.jsonl'),'train':[dict(g,valid=True) for g in rows(train_gt_path)]}
    inputs += [public/'gt_public.jsonl',train_gt_path,public/'manifest.json',train/'selected_manifest.json']
    reports={group:{name:{'without_ignore':evaluate(predictions[(group,name)],gt[group],manifest,exempt_invalid=False),
                         'with_ignore':evaluate(predictions[(group,name)],gt[group],manifest,exempt_invalid=True)}
                    for name in ('window_v1','window_v1_auto_region')} for group,manifest in manifests.items()}
    changes=[dict(c,group=group) for group in reports for c in changed_matches(
        reports[group]['window_v1']['without_ignore'],reports[group]['window_v1_auto_region']['without_ignore'],'window_v1_auto_region')]
    current=read(data/'outputs/experiments/interframe_bounce_v0/CURRENT.json')
    split_path=data/'outputs/experiments/interframe_bounce_v0'/current['repository_reuse_run']/'split.json'
    split=read(split_path)['assignments'];inputs.append(split_path)
    split_scores={}
    for name,r in reports['train'].items():
        split_scores[name]={}
        for subset in ('train','validation'):
            vs=[v for v in r['without_ignore']['by_video'] if split[v['video_id']]==subset]
            split_scores[name][subset]=dict(metrics(*(sum(v[k] for v in vs) for k in ('tp','fp','fn'))),videos=len(vs))
    write(out/'metrics.json',{'paired':reports,'train_split_proxy':split_scores})
    write_rows(out/'changed_matches.jsonl',changes);write_rows(out/'candidate_decisions.jsonl',decisions)
    for (group,name),ps in predictions.items():write_rows(out/(group+'_'+name+'_predictions.jsonl'),ps)
    code=[Path(__file__),PROJECT/'development/bounce/rule_variants.py',PROJECT/'development/detection/auto_table_geometry.py',
          PROJECT/'development/bounce/table_geometry.py',PROJECT/'development/evaluation/evaluate_landings.py']
    (out/'source').mkdir()
    for path in code:shutil.copy2(path,out/'source'/path.name)
    write(out/'summary.json',{'config':asdict(cfg),'margin_mm':900,'train_window_replay_exact':True,'adopted':False,
          'source_sha256':{str(p.resolve()):sha(p) for p in code},'input_sha256':{str(p.resolve()):sha(p) for p in inputs},
          'train_labels_are_stage_candidates':True,'source_split_verified':False})
    lines=['窗口轨迹规则与自动桌面几何的配对对照','训练154的既有V1输出已逐事件复现，使用同一份自动几何缓存。',
           '900mm固定参数；未重新选参；训练指标为阶段候选代理值。','']
    for group,names in reports.items():
        lines.append(group+'（不豁免）:')
        for name,r in names.items():
            s=r['without_ignore'];o=s['overall'];lines.append(f"{name}: {o['tp']}/{o['fp']}/{o['fn']} F1={o['f1']:.6f} angle_macro={s['angle_macro_f1']:.6f}")
    lines.append('validation31: '+str(split_scores))
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'public':{n:r['without_ignore']['overall'] for n,r in reports['public'].items()},
                      'train_proxy':{n:r['without_ignore']['overall'] for n,r in reports['train'].items()},'validation31':split_scores},indent=2))


if __name__=='__main__':main()

"""Real-video automatic geometry audit; inference never consumes event labels."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.bounce.ge_rule_v2 import RuleConfig,replay
from development.detection.auto_table_geometry import GeometryConfig,extract_quad,CausalGeometryTracker,AutoGeometryTimeline
from development.detection.tt3d_segmenter import TT3DSegmenter
from development.evaluation.evaluate_landings import evaluate,validate_input,metrics
from development.evaluation.run_geometry_d_v2 import changed_matches


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(path,data):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in data),encoding='utf-8')


def collect(model,meta,path,group,out,config,seconds,warmup_seconds=None):
    if meta.get('video_sha256') and sha(path)!=meta['video_sha256']:
        raise ValueError('Video hash mismatch: '+meta['video_id'])
    capture=cv2.VideoCapture(str(path))
    if not capture.isOpened():raise ValueError('Cannot decode '+str(path))
    tracker=CausalGeometryTracker(config)
    fixed_tracker=CausalGeometryTracker(config)
    interval=max(1,round(meta.get('measured_fps',meta['fps'])*seconds))
    warmup_interval=max(1,round(meta.get('measured_fps',meta['fps'])*(warmup_seconds or seconds)))
    next_frame=0
    fixed_samples=[]
    samples=[];best=None;elapsed=0.
    try:
        for frame in range(meta['n_frames']):
            if not capture.grab():raise ValueError('Short video '+meta['video_id'])
            if frame%interval and frame<next_frame:continue
            ok,image=capture.retrieve()
            if not ok or image.shape[:2]!=(meta['height'],meta['width']):raise ValueError('Image size mismatch')
            start=time.perf_counter()
            probability=model.probability(image)
            proposal=extract_quad(image,probability,config)
            state=tracker.update(frame,image,proposal)
            next_frame=frame+(interval if state['state']=='stable' else warmup_interval)
            if frame%interval==0:
                fixed_state=fixed_tracker.update(frame,image,proposal)
                fixed_state.update(group=group,video_id=meta['video_id'],raw_proposal=proposal,
                                   scene_thumbnail=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(64,36)).tolist())
                fixed_samples.append(fixed_state)
            elapsed+=time.perf_counter()-start
            state.update(group=group,video_id=meta['video_id'])
            state['raw_proposal']=proposal
            state['scene_thumbnail']=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(64,36)).tolist()
            samples.append(state)
            if best is None or (state['usable'] and (not best[2]['usable'] or state.get('quality',0)>best[2].get('quality',0))):
                best=(frame,image.copy(),state,probability.copy())
        if capture.grab():raise ValueError('Extra video frames '+meta['video_id'])
    finally:capture.release()
    frame,image,state,probability=best
    mask=cv2.resize((probability>.8).astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    image[mask]=(image[mask]*.8+np.array([0,180,0])*.2).astype(np.uint8)
    if state.get('corners'):
        q=np.array(state['corners']).round().astype(np.int32)
        cv2.polylines(image,[q],True,(0,230,255),3)
        for i,p in enumerate(q):cv2.putText(image,str(i),tuple(p),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,230,255),2)
    im=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,im.width,28),fill='black')
    draw.text((8,8),f"{group} {meta['video_id']} frame={frame} {state['state']} {state['reason']}",fill='white')
    im.save(out/'diagnostics'/(group+'_'+meta['video_id']+'.png'))
    write_rows(out/'fixed_sampling'/f'{group}_{meta["video_id"]}.jsonl',fixed_samples)
    return samples,elapsed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,default=PROJECT/'tmp/tt3d-upstream/weights/table_segmentation.ckpt')
    ap.add_argument('--scope',choices=['public','all'],default='all')
    ap.add_argument('--sample-seconds',type=float,default=.4)
    ap.add_argument('--warmup-sample-seconds',type=float,help='Additional causal samples while unknown or held')
    ap.add_argument('--reuse-geometry',type=Path,help='Reuse only a verified run from identical model/config/input hashes')
    ap.add_argument('--manual-geometry-dir',type=Path,help='Optional actual D public15 annotation directory')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if not np.isfinite(args.sample_seconds) or args.sample_seconds<=0:ap.error('Positive sample interval required')
    if args.warmup_sample_seconds is not None and (not np.isfinite(args.warmup_sample_seconds)
            or not 0<args.warmup_sample_seconds<=args.sample_seconds):
        ap.error('Warmup interval must be positive and no longer than normal interval')
    if args.reuse_geometry and args.warmup_sample_seconds is not None:
        ap.error('Adaptive sampling must decode observations afresh')
    if args.reuse_geometry:
        cache_profile=read(args.reuse_geometry/'summary.json')
        if cache_profile.get('warmup_sample_seconds') is not None:
            ap.error('Adaptive geometry cache cannot be relabelled as fixed sampling')
    data=args.data_root
    public=data/'participant/participant/pingpang/public_data'
    public_tracks=data/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1'
    train=data/'C_train154_20261011_v3/C_train154_20261011_v3'
    pm=read(public/'manifest.json')['videos']
    tm=read(train/'selected_manifest.json')['videos'] if args.scope=='all' else []
    out=args.output or PROJECT/'outputs/experiments/auto_table_v2/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False);(out/'diagnostics').mkdir();(out/'fixed_sampling').mkdir()
    print('RUN',out,flush=True)
    geometry_config=GeometryConfig();rule_config=RuleConfig()
    model=TT3DSegmenter(args.checkpoint)
    jobs=[('public',m,public/m['path'],public_tracks) for m in pm]
    jobs += [('train',m,data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/m['video_id'][:2]/(m['video_id'][3:]+'.mp4'),train) for m in tm]
    inputs=[public/'manifest.json',public/'gt_public.jsonl',public_tracks/'predictions.jsonl',args.checkpoint]
    if tm:inputs += [train/'selected_manifest.json',train/'predictions.jsonl']
    reused={}
    if args.reuse_geometry:
        old=read(args.reuse_geometry/'summary.json')
        if old['geometry_config']!=asdict(geometry_config) or old['sample_seconds']!=args.sample_seconds or old['model']!=model.provenance:
            raise ValueError('Geometry cache provenance mismatch')
        for path in [Path(__file__),PROJECT/'development/detection/auto_table_geometry.py',PROJECT/'development/detection/tt3d_segmenter.py']:
            if old['source_sha256'].get(str(path.resolve()))!=sha(path):raise ValueError('Geometry extraction source changed')
        for sample in rows(args.reuse_geometry/'geometry_samples.jsonl'):
            reused.setdefault((sample['group'],sample['video_id']),[]).append(sample)
    samples=[];quality=[];elapsed=0.;predictions=defaultdict(list);decisions=[]
    # Extraction is finished before any event labels are loaded.
    for index,(group,m,path,tracks) in enumerate(jobs):
        inputs.append(path)
        if args.reuse_geometry:
            if old['input_sha256'].get(str(path.resolve()))!=sha(path):raise ValueError('Video changed')
            ss=reused[(group,m['video_id'])]
            shutil.copy2(args.reuse_geometry/'diagnostics'/(group+'_'+m['video_id']+'.png'),out/'diagnostics')
        else:
            ss,dt=collect(model,m,path,group,out,geometry_config,args.sample_seconds,args.warmup_sample_seconds);elapsed+=dt
        samples+=ss
        quality.append({'group':group,'video_id':m['video_id'],'samples':len(ss),
                        'usable_samples':sum(s['usable'] for s in ss),'any_usable':any(s['usable'] for s in ss),
                        'state_counts':dict(Counter(s['state'] for s in ss)),
                        'reason_counts':dict(Counter(s['reason'] for s in ss))})
        timeline=AutoGeometryTimeline(ss)
        raw_path=tracks/'raw'/(m['video_id']+'.jsonl');inputs.append(raw_path)
        raw=rows(raw_path)
        if len(raw)!=m['n_frames'] or any(r['input_frame_id']!=i for i,r in enumerate(raw)):
            raise ValueError('Malformed raw cache')
        baseline,_=replay(raw,mode='v0')
        gated,ds=replay(raw,timeline,'v1a',rule_config)
        for name,ps in [('v0',baseline),('auto_region',gated)]:
            predictions[(group,name)]+=[dict(p,video_id=m['video_id']) for p in ps]
        decisions += [dict(d,group=group,video_id=m['video_id']) for d in ds]
        if (index+1)%5==0:print('VIDEOS',index+1,'/',len(jobs),'USABLE',sum(q['any_usable'] for q in quality),flush=True)
    write_rows(out/'geometry_samples.jsonl',samples);write_rows(out/'geometry_quality.jsonl',quality)
    if not args.reuse_geometry:
        write_rows(out/'fixed_sampling_geometry_samples.jsonl',
                   [s for path in sorted((out/'fixed_sampling').glob('*.jsonl')) for s in rows(path)])
    write_rows(out/'candidate_decisions.jsonl',decisions)
    for (group,name),ps in predictions.items():write_rows(out/(group+'_'+name+'_predictions.jsonl'),ps)
    for group,tracks in [('public',public_tracks)]+([('train',train)] if tm else []):
        canonical=lambda ps:sorted([{k:p[k] for k in ('video_id','frame_id','x','y')} for p in ps],key=lambda p:(p['video_id'],p['frame_id'],p['x'],p['y']))
        if canonical(predictions[(group,'v0')])!=canonical(rows(tracks/'predictions.jsonl')):
            raise ValueError(group+' V0 replay changed')
    pg=rows(public/'gt_public.jsonl')
    public_scores={name:{'without_ignore':evaluate(predictions[('public',name)],pg,pm,exempt_invalid=False),
                         'with_ignore':evaluate(predictions[('public',name)],pg,pm,exempt_invalid=True)} for name in ('v0','auto_region')}
    train_scores={};split_scores={};changes=[]
    changes += [dict(c,group='public') for c in changed_matches(public_scores['v0']['without_ignore'],public_scores['auto_region']['without_ignore'],'auto_region')]
    if tm:
        gt_path=data/'tmp/repository_labels_audit_20261012/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl'
        gt=[dict(g,valid=True) for g in rows(gt_path)];inputs.append(gt_path)
        validate_input(gt,{m['video_id']:m for m in tm},ground_truth=True)
        train_scores={name:evaluate(predictions[('train',name)],gt,tm,exempt_invalid=False) for name in ('v0','auto_region')}
        changes += [dict(c,group='train') for c in changed_matches(train_scores['v0'],train_scores['auto_region'],'auto_region')]
        current=read(data/'outputs/experiments/interframe_bounce_v0/CURRENT.json')
        split_path=data/'outputs/experiments/interframe_bounce_v0'/current['repository_reuse_run']/'split.json'
        split=read(split_path);inputs.append(split_path)
        for name,report in train_scores.items():
            split_scores[name]={}
            for subset in ('train','validation'):
                vs=[r for r in report['by_video'] if split['assignments'][r['video_id']]==subset]
                counts=[sum(r[k] for r in vs) for k in ('tp','fp','fn')]
                split_scores[name][subset]=dict(metrics(*counts),videos=len(vs))
    write(out/'metrics.json',{'public':public_scores,'train_proxy':train_scores,'train_split_proxy':split_scores})
    write_rows(out/'changed_matches.jsonl',changes)
    coverage={g:{'videos':sum(q['group']==g for q in quality),'any_usable':sum(q['group']==g and q['any_usable'] for q in quality),
                 'samples':sum(q['samples'] for q in quality if q['group']==g),'usable_samples':sum(q['usable_samples'] for q in quality if q['group']==g)} for g in ('public','train') if any(q['group']==g for q in quality)}
    candidates={g:dict(Counter('available' if d['geometry_available'] else 'fallback' for d in decisions if d['group']==g)) for g in coverage}
    missing=[m['video_id'] for m in pm if args.manual_geometry_dir is None or not (args.manual_geometry_dir/(m['video_id']+'.json')).is_file()]
    manual={'available_files':len(pm)-len(missing),'missing_video_ids':missing,
            'report_claim':'D stage1 report mentions 15 manual geometry records; existence/completion not independently confirmed',
            'existence_independently_confirmed':False,
            'true_corner_error_measured':False,'note':'Automatic masks and GT event containment do not substitute for independent corner annotations'}
    # If supplied, report only the acquisition status here; the strict manual replay uses the existing CLI.
    sources=[Path(__file__),PROJECT/'development/detection/tt3d_segmenter.py',PROJECT/'development/detection/auto_table_geometry.py',
             PROJECT/'development/detection/table_border_lines.py',
             PROJECT/'development/bounce/ge_rule_v2.py',PROJECT/'development/bounce/table_geometry.py',PROJECT/'development/evaluation/evaluate_landings.py']
    (out/'source').mkdir()
    for source in sources:shutil.copy2(source,out/'source'/source.name)
    summary={'model':model.provenance,'geometry_config':asdict(geometry_config),'rule_config':asdict(rule_config),
             'sample_seconds':args.sample_seconds,'coverage':coverage,'candidate_geometry':candidates,
             'warmup_sample_seconds':args.warmup_sample_seconds,
             'manual_geometry_audit':manual,'adopted':False,'train_label_status':'723 stage-level candidates; not frozen official GT',
             'validation_split_status':'fixed31 video holdout; source grouping not verified; diagnostic only',
             'sample_geometry_seconds':elapsed,'end_to_end_50fps_verified':False,
             'input_sha256':{str(p.resolve()):sha(p) for p in inputs},'source_sha256':{str(p.resolve()):sha(p) for p in sources},
             'geometry_cache_reused':str(args.reuse_geometry) if args.reuse_geometry else None}
    write(out/'summary.json',summary)
    # All public diagnostics plus a selection of train images, no event-based geometry selection.
    files=sorted((out/'diagnostics').glob('public_*.png'))+sorted((out/'diagnostics').glob('train_*.png'))[:20]
    montage=Image.new('RGB',(640*3,380*((len(files)+2)//3)),(25,25,25))
    for i,file in enumerate(files):
        im=Image.open(file).convert('RGB');im.thumbnail((640,360));montage.paste(im,((i%3)*640,(i//3)*380))
        ImageDraw.Draw(montage).text(((i%3)*640+8,(i//3)*380+362),file.stem,fill='white')
    montage.save(out/'geometry_montage.jpg')
    lines=['自动桌面几何真实视频实验','模型：TT3D公开台面分割 + 图像边线拟合 + 四角求交 + 因果稳定检查',
           '长短轴未知：两种映射都出桌才拒绝；不启用空间去重放行。','全部几何只依赖图像，触台标签仅用于事后评分。','',
           '公开15（不豁免）：版本 TP/FP/FN 汇总F1 三视角平均F1']
    for name,s in public_scores.items():
        r=s['without_ignore'];o=r['overall'];lines.append(f"{name}: {o['tp']}/{o['fp']}/{o['fn']} {o['f1']:.6f} {r['angle_macro_f1']:.6f}")
    if tm:
        lines+=['','训练154（阶段候选标签，代理指标）：']
        for name,s in train_scores.items():
            r=s['overall'];lines.append(f"{name}: {r['tp']}/{r['fp']}/{r['fn']} F1proxy={r['f1']:.6f}; validation={split_scores[name]['validation']['f1']:.6f}")
    lines+=['',f'几何覆盖：{coverage}',f'候选几何：{candidates}',f'独立人工几何文件缺失：{len(missing)}/15；未测量真实四角误差。',
            '结果为独立实验，未替换正式规则；未验证端到端50FPS。',
            '逐候选原因、丢失/新增命中、每视频叠加图、输入/权重/代码哈希均已保存。']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'coverage':coverage,'public':{n:s['without_ignore']['overall'] for n,s in public_scores.items()},
                      'train_proxy':{n:s['overall'] for n,s in train_scores.items()},'lost_matches':dict(Counter(c['group'] for c in changes if c['change']=='lost_tp'))},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()

"""Prepare blind frame reviews, serve locally, and score submitted corner refs.

Frames contain no automatic corners or event markers. References apply ONLY to
their exact decoded frame. Submitted references remain a single-review source,
not independently double-checked GT. No input labels enter the bounce pipeline.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,rows,write,sha
from development.evaluation.validate_table_geometry import corner_error
from development.detection.auto_table_geometry import AutoGeometryTimeline


def prepare(data,run,out):
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    train=read(data/'C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json')['videos']
    public_dir=data/'participant/participant/pingpang/public_data'
    public=read(public_dir/'manifest.json')['videos']
    q=rows(run/'geometry_quality.jsonl');failed={r['video_id'] for r in q if r['group']=='train' and not r['any_usable']}
    groups=defaultdict(list)
    for m in train:
        if m['video_id'] in failed:groups[m['video_id'][:2]].append(m)
    selected=[]
    for vs in groups.values():
        m=min(vs,key=lambda m:m['video_id'])
        selected += [('train',m,round((m['n_frames']-1)*f),'remaining_failure') for f in (.25,.75)]
    for angle in sorted({m['angle'] for m in public}):
        vs=sorted((m for m in public if m['angle']==angle),key=lambda m:m['video_id'])
        for m in (vs[0],vs[-1]):selected.append(('public',m,(m['n_frames']-1)//2,'public_view_control'))
    for vid in ('05_001','21_001'):
        m=next(m for m in train if m['video_id']==vid)
        selected += [('train',m,round((m['n_frames']-1)*f),'new_coverage_control') for f in (.25,.75)]
    for vid,frame in [('05_001',40),('17_002',72)]:
        selected.append(('train',next(m for m in train if m['video_id']==vid),frame,'envelope_disagreement'))
    cases=[];seen=set();run_summary=read(run/'summary.json')
    for group,m,frame,reason in selected:
        vid=m['video_id'];case_id=f'{group}_{vid}_{frame:06d}'
        if case_id in seen:continue
        seen.add(case_id)
        path=public_dir/m['path'] if group=='public' else data/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        video_hash=sha(path)
        if run_summary['input_sha256'].get(str(path.resolve()))!=video_hash:raise ValueError('Video provenance mismatch')
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();cap.release()
        if not ok or image.shape[:2]!=(m['height'],m['width']):raise ValueError('Cannot decode review frame')
        image_path=out/'frames'/(case_id+'.png')
        if not cv2.imwrite(str(image_path),image):raise ValueError('Cannot write review frame')
        cases.append({'case_id':case_id,'group':group,'video_id':vid,'frame_id':frame,
                      'width':m['width'],'height':m['height'],'image_url':'frames/'+image_path.name,
                      'image_sha256':sha(image_path),'video_sha256':video_hash,'selection_reason':reason})
    manifest={'schema_version':1,'frame_convention':'decoded video frame, zero-based; no rule-frame compensation',
              'selection':'fixed fractions, failed-prefix representatives, view controls and two envelope-disagreement frames; no event labels',
              'blinded_to_automatic_corners':True,'independent_human_references_available':False,'cases':cases,
              'geometry_run':str(run.resolve()),'geometry_samples_sha256':sha(run/'geometry_samples.jsonl')}
    write(out/'manifest.json',manifest)
    (out/'index.html').write_text((Path(__file__).with_name('geometry_reference_review.html')).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'pack':str(out.resolve()),'cases':len(cases),'human_references':0},indent=2))


def validate_submission(value,cases):
    if not isinstance(value,dict):raise ValueError('参考格式不正确，请刷新重试')
    case=cases.get(value.get('case_id'))
    if case is None:raise ValueError('未找到核查样本，请刷新重试')
    status=value.get('status')
    if status not in ('corners_visible','occluded','thin_table','no_table','ambiguous_target'):raise ValueError('请选择台面可辨识情况')
    reviewer=value.get('reviewer','')
    if not isinstance(reviewer,str):raise ValueError('请填写复核者姓名或简称')
    reviewer=reviewer.strip()
    if not reviewer or len(reviewer)>80:raise ValueError('请填写复核者姓名或简称，不超过80字')
    points=value.get('corners',[])
    if status=='corners_visible':
        q=np.asarray(points,float)
        if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError('请点选完整的四个角点')
        if np.any(q<0) or np.any(q[:,0]>=case['width']) or np.any(q[:,1]>=case['height']):raise ValueError('角点超出图像，请重新调整')
        if not cv2.isContourConvex(q.astype(np.float32)) or cv2.contourArea(q.astype(np.float32))<1:raise ValueError('请沿台面边界顺序点选四角，避免交叉或重合')
        points=q.tolist()
    elif points:raise ValueError('无法确认四角时，请清空已选点，避免保存猜测角点')
    return {'case_id':case['case_id'],'group':case['group'],'video_id':case['video_id'],'frame_id':case['frame_id'],
            'image_sha256':case['image_sha256'],'video_sha256':case['video_sha256'],'status':status,'corners':points,
            'reviewer':reviewer,'reference_source':'user_submitted_single_review_not_double_checked',
            'submitted_utc':datetime.now(timezone.utc).isoformat()}


def latest_reviews(pack,manifest):
    cases={c['case_id']:c for c in manifest['cases']};latest={}
    path=pack/'references.jsonl'
    if path.exists():
        for r in rows(path):
            case=cases.get(r['case_id'])
            if case is None or r['image_sha256']!=case['image_sha256'] or r['video_sha256']!=case['video_sha256']:
                raise ValueError('Review provenance mismatch')
            validate_submission(r,cases);latest[r['case_id']]=r
    return latest


def serve(pack,port):
    manifest=read(pack/'manifest.json');cases={c['case_id']:c for c in manifest['cases']}
    for c in cases.values():
        if sha(pack/c['image_url'])!=c['image_sha256']:raise ValueError('Review image changed')
    allowed={'/':(pack/'index.html','text/html; charset=utf-8'),'/index.html':(pack/'index.html','text/html; charset=utf-8')}
    allowed.update({'/'+c['image_url']:(pack/c['image_url'],'image/png') for c in cases.values()})
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False).encode('utf-8');self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            route=self.path.split('?',1)[0]
            if route=='/api/pack':return self.reply(200,manifest)
            if route=='/api/references':return self.reply(200,latest_reviews(pack,manifest))
            if route not in allowed:return self.reply(404,{'error':'Not found'})
            path,mime=allowed[route];self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(path.read_bytes())
        def do_POST(self):
            if self.path!='/api/reference':return self.reply(404,{'error':'Not found'})
            if self.headers.get('Origin')!=f'http://127.0.0.1:{port}':return self.reply(403,{'error':'Open the local review page before saving'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Reference request too large/empty')
                value=json.loads(self.rfile.read(length));reference=validate_submission(value,cases)
                with (pack/'references.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(reference,ensure_ascii=False)+'\n')
                self.reply(200,reference)
            except (ValueError,TypeError) as exc:self.reply(400,{'error':str(exc)})
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'REVIEW http://127.0.0.1:{port} cases={len(cases)}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()


def evaluate(pack,run,out):
    manifest=read(pack/'manifest.json');reviews=latest_reviews(pack,manifest)
    for case in manifest['cases']:
        if sha(pack/case['image_url'])!=case['image_sha256']:raise ValueError('Review image changed')
    if sha(run/'geometry_samples.jsonl')!=manifest['geometry_samples_sha256']:raise ValueError('Geometry differs from frozen review pack')
    sample_groups=defaultdict(list)
    for s in rows(run/'geometry_samples.jsonl'):sample_groups[(s['group'],s['video_id'])].append(s)
    timelines={key:AutoGeometryTimeline(ss) for key,ss in sample_groups.items()}
    records=[]
    for case in manifest['cases']:
        r=reviews.get(case['case_id']);timeline=timelines[(case['group'],case['video_id'])];plane=timeline.lookup(case['frame_id'])
        record=dict(case_id=case['case_id'],group=case['group'],video_id=case['video_id'],frame_id=case['frame_id'],
                    geometry_available=plane is not None,reference_status=r['status'] if r else 'pending')
        if r and r['status']=='corners_visible' and plane:
            record.update(corner_error(timeline.sample(case['frame_id'])['corners'],r['corners']))
        records.append(record)
    evaluated=[r for r in records if 'mean_corner_error_px' in r]
    out.mkdir(parents=True,exist_ok=False)
    summary={'cases':len(records),'submitted_references':len(reviews),'corner_comparisons':len(evaluated),
             'status':'single_review_corner_diagnostic' if evaluated else 'awaiting_corner_references',
             'mean_corner_error_px':float(np.mean([r['mean_corner_error_px'] for r in evaluated])) if evaluated else None,
             'independent_double_review_verified':False,'diagnostic_selection_not_representative':True,
             'manifest_sha256':sha(pack/'manifest.json'),'geometry_sha256':sha(run/'geometry_samples.jsonl'),
             'reviews_sha256':sha(pack/'references.jsonl') if (pack/'references.jsonl').exists() else None,
             'source_sha256':sha(Path(__file__))}
    write(out/'summary.json',summary);write(out/'records.json',records);print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);commands=ap.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--geometry-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('serve');p.add_argument('--pack',type=Path,required=True);p.add_argument('--port',type=int,default=8923)
    p=commands.add_parser('evaluate');p.add_argument('--pack',type=Path,required=True);p.add_argument('--geometry-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.data_root,args.geometry_run,args.output)
    elif args.command=='serve':serve(args.pack,args.port)
    else:evaluate(args.pack,args.geometry_run,args.output)


if __name__=='__main__':main()

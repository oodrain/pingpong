"""Audit saved observations and prepare additional frames without human labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])))
CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT/'development/annotation'))
from net_geometry_review import NetReviewStore, backend, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def font(size=18):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def audit(store, out):
    module = backend()
    records = sorted((r for r in store.latest.values() if (r['group'],r['video_id']) == ('public','v000038')), key=lambda r:r['frame_id'])
    joint = module.estimate_net_sequence(records, (1920,1080))
    frames = []
    sheet = Image.new('RGB', (1560,1290), (23,29,37))
    draw = ImageDraw.Draw(sheet)
    draw.text((20,10), 'v000038：粉色人工点 / 黄色共用焦距投影；放大图保留原始像素位置', font=font(22), fill='white')
    for index,(record,pose) in enumerate(zip(records,joint['poses'])):
        im = Image.fromarray(store.image(store.by_id[record['case_id']])[:,:,::-1])
        single = store.estimate(record)['selected_camera']
        observed = np.vstack([module.order_quad(record['corners']), record['net_endpoints']])
        net_projected = np.asarray(pose['projected_net_endpoints'])
        if single['net_endpoint_order_reversed']:
            net_projected = net_projected[::-1]
        projected = np.vstack([pose['projected_corners'],net_projected])
        errors = projected-observed
        labels = ['C1','C2','C3','C4','N1','N2']
        points = [dict(point=label, observed_px=o.tolist(), projected_px=p.tolist(), dx_px=float(e[0]), dy_px=float(e[1]), distance_px=float(np.linalg.norm(e))) for label,o,p,e in zip(labels,observed,projected,errors)]
        frames.append(dict(frame_id=record['frame_id'],corners_adjusted=record['corners_adjusted'], single_focal_px=single['focal_px'],single_rmse_px=single['reprojection_rmse_px'],joint_rmse_px=pose['reprojection_rmse_px'],points=points))
        y=55+index*405
        draw.text((20,y),f"第 {record['frame_id']} 帧：单帧 RMSE {single['reprojection_rmse_px']:.2f}px / 共用焦距 RMSE {pose['reprojection_rmse_px']:.2f}px",font=font(19),fill='white')
        # All six patches, including the physical image evidence around each point.
        for j,(label,o,p,e) in enumerate(zip(labels,observed,projected,errors)):
            x=20+j*255
            left,top=int(round(o[0]))-30,int(round(o[1]))-30
            patch=im.crop((left,top,left+60,top+60)).resize((240,240),Image.Resampling.NEAREST)
            pd=ImageDraw.Draw(patch)
            a=(float(o[0]-left)*4,float(o[1]-top)*4);b=(float(p[0]-left)*4,float(p[1]-top)*4)
            pd.line((a,b),fill='white',width=2)
            for pt,color in [(a,(255,105,205)),(b,(255,225,70))]:
                pd.ellipse((pt[0]-5,pt[1]-5,pt[0]+5,pt[1]+5),outline=color,width=3)
            draw.text((x,y+34),f'{label}  偏差 {np.linalg.norm(e):.2f}px',font=font(18),fill='white')
            sheet.paste(patch,(x,y+65))
            draw.text((x,y+313),f'dx {e[0]:+.2f} / dy {e[1]:+.2f}',font=font(16),fill='white')
    sheet.save(out/'v000038_point_deviations.png')
    # This measures projection changes, not proof of zoom or bad human annotation.
    sizes=[]
    for r in records:
        q=np.asarray(r['corners']);sizes.append(dict(frame_id=r['frame_id'],area_px2=float(abs(cv2.contourArea(q.astype(np.float32)))),edge_lengths_px=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).tolist()))
    report=dict(shared_focal_px=joint['shared_focal_px'],joint_rmse_px=joint['reprojection_rmse_px'],joint_maximum_px=joint['maximum_point_error_px'],frames=frames,table_projection_sizes=sizes,point_labels='C1..C4 canonical corner order; N1/N2 original human click order',interpretation='Shared focal rejected. Point residuals do not distinguish annotation error, endpoint model error, or changing camera intrinsics.',human_coordinates_changed=False)
    write(out/'v000038_diagnostic.json',report)


def candidates(store, out):
    manifest=store.source.manifest;geometry=Path(manifest['geometry_source'])
    expected=manifest['source_sha256'][str((geometry/'geometry_samples.jsonl').resolve())]
    if sha(geometry/'geometry_samples.jsonl')!=expected:raise ValueError('Geometry source changed')
    geo=read(geometry/'summary.json')
    public=ROOT/'participant/participant/pingpang/public_data'
    train=ROOT/'C_train154_20261011_v3/C_train154_20261011_v3'
    metas={('public',m['video_id']):m for m in read(public/'manifest.json')['videos']}
    metas.update({('train',m['video_id']):m for m in read(train/'selected_manifest.json')['videos']})
    accepted={(r['group'],r['video_id'],store.by_id[r['case_id']].get('camera_epoch',0)) for r in store.latest.values() if r['calibration']['usable']}
    already={(c['group'],c['video_id'],c['frame_id']) for c in store.cases}
    seeds={(c['group'],c['video_id'],c.get('camera_epoch',0)):c for c in store.cases}
    grouped={key:[] for key in accepted}
    for line in (geometry/'geometry_samples.jsonl').read_text().splitlines():
        s=json.loads(line);key=(s['group'],s['video_id'],s['camera_epoch'])
        if key in grouped and s['usable'] and s['state']=='stable' and (s['group'],s['video_id'],s['frame_id']) not in already:grouped[key].append(s)
    pool=out/'candidate_pool';pool.mkdir();(pool/'frames').mkdir();cases=[]
    for key in sorted(accepted):
        group,vid,epoch=key;m=metas[group,vid];seed=seeds[key]
        path=public/m['path'] if group=='public' else ROOT/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(path)!=geo['input_sha256'][str(path.resolve())] or sha(path)!=seed['video_sha256']:raise ValueError('Video changed')
        wanted={s['frame_id']:s for s in grouped[key]}
        if len(wanted)<2:raise ValueError(f'Insufficient stable candidates for {key}')
        cap=cv2.VideoCapture(str(path))
        try:
            for f in range(max(wanted)+1):
                if not cap.grab():raise ValueError('Truncated video')
                if f not in wanted:continue
                ok,im=cap.retrieve();s=wanted[f]
                if not ok or im.shape[:2]!=(m['height'],m['width']):raise ValueError('Frame decode mismatch')
                if not np.array_equal(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,36)),np.asarray(s['scene_thumbnail'])):raise ValueError('Frame identity mismatch')
                case_id=f'{group}_{vid}_camera{epoch}_{f:06d}';file=pool/'frames'/(case_id+'.png')
                if not cv2.imwrite(str(file),im):raise ValueError('Frame write failed')
                cases.append(dict(seed,case_id=case_id,frame_id=f,image_url='frames/'+file.name,image_sha256=sha(file),proposed_corners=backend().order_quad(s['corners']).tolist(),corner_source='automatic_proposal_requires_confirmation',priority='supplement',selection_scope='stable geometry; net visibility must be confirmed by human'))
        finally:cap.release()
    write(pool/'manifest.json',dict(manifest,cases=cases,human_submissions_generated=False))
    for page,start in enumerate(range(0,len(cases),12),1):
        canvas=Image.new('RGB',(1500,1100),(23,29,37));d=ImageDraw.Draw(canvas)
        for index,c in enumerate(cases[start:start+12]):
            im=Image.open(pool/c['image_url']);q=np.array(c['proposed_corners']);lo=q.min(axis=0)-[130,180];hi=q.max(axis=0)+[130,70]
            crop=im.crop((max(0,int(lo[0])),max(0,int(lo[1])),min(im.width,int(hi[0])),min(im.height,int(hi[1]))));crop.thumbnail((480,215))
            x=10+(index%3)*500;y=10+(index//3)*270;canvas.paste(crop,(x,y));d.text((x,y+220),f"{c['group']} {c['video_id']}  f{c['frame_id']}",font=font(),fill='white')
        canvas.save(out/f'candidate_sheet_{page:02d}.jpg')
    print(json.dumps(dict(candidates=len(cases),videos=len(accepted),directory=str(out)),ensure_ascii=False),flush=True)


def finish(store,pool,selection,out):
    manifest=read(pool/'manifest.json');available={(c['group'],c['video_id'],c['frame_id']):c for c in manifest['cases']}
    choices=read(selection);selected=[]
    accepted={(r['group'],r['video_id']) for r in store.latest.values() if r['calibration']['usable']}
    if {(v['group'],v['video_id']) for v in choices}!=accepted:raise ValueError('Selection must cover all single-frame accepted videos')
    for item in choices:
        if len(set(item['frames']))!=2:raise ValueError('Two additional distinct frames per video required')
        selected.extend(available[item['group'],item['video_id'],f] for f in item['frames'])
    raw=store.path.read_bytes();out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();(out/'prior_sources').mkdir()
    snapshot=out/'prior_sources/net_reference_at_preparation.jsonl';snapshot.write_bytes(raw)
    (out/'prior_sources/manifest_at_preparation.json').write_bytes(store.source.manifest_path.read_bytes())
    (out/'selection.json').write_bytes(selection.read_bytes())
    rechecks=[dict(c,priority='recheck',review_reason='v000038：复核真实台面外边缘四角和球网顶带对应点；不按黄色投影挪点。') for c in store.cases if (c['group'],c['video_id'])==('public','v000038')]
    rechecks.sort(key=lambda c:c['frame_id']);selected.sort(key=lambda c:(c['group'],c['video_id'],c['frame_id']))
    cases=[]
    for case in rechecks+selected:
        source=(store.pack if case['priority']=='recheck' else pool)/case['image_url']
        if sha(source)!=case['image_sha256']:raise ValueError('Source frame changed')
        shutil.copy2(source,out/case['image_url']);cases.append(case)
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate review case')
    parent=dict(pack=str(store.pack),manifest_sha256=store.source.manifest_hash,
        reference_snapshot='prior_sources/'+snapshot.name,reference_snapshot_sha256=sha(snapshot))
    result=dict(manifest,cases=cases,purpose='v000038 recheck and nine-video supplementary calibration review',
        parent_review_source=parent,human_submissions_generated=False,
        frame_selection_scope='Agent visual screening for clearer corners/net; no human geometry labels generated',
        prior_pack=str(store.pack),prior_manifest_sha256=store.source.manifest_hash)
    result.pop('pilot_videos',None);result.pop('pilot_frames_per_video',None)
    result['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    result['source_sha256'][str(selection.resolve())]=sha(selection)
    write(out/'manifest.json',result)
    (out/'核查说明.txt').write_text('本轮共21帧：先复核v000038的36、110、170帧，再核查18个补充帧。\n'
        '四角仍是待确认草稿。按真实台面外边缘核对，白线有宽度时保持一致；遮挡时如实登记，不按投影补猜。\n'
        '球网仍采用原来的两个固定对应点，不支持顶线上任意取两点。看不清对应位置时登记遮挡。\n'
        '每帧确认四角、标球网、确认长宽方向，按S保存。全部完成后点“多帧标定”。\n'
        '导出会读取原核查包的最新追加记录，同一帧以本轮保存记录为准；原54帧记录不会被改写。\n'
        '补充帧已核对原视频哈希、连续解码帧号和缓存缩略图；视觉筛选不代表标定已通过。\n',encoding='utf-8')
    if store.path.read_bytes()!=raw:raise ValueError('Parent journal changed during preparation')
    report=dict(cases=len(cases),rechecks=len(rechecks),additional_frames=len(selected),videos=len(accepted),
        parent_reference_sha256=sha(store.path),source_frames_verified=True,human_submissions=0,
        selected_videos=choices,originals_unchanged=True)
    write(out/'preparation_summary.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pool',type=Path);ap.add_argument('--selection',type=Path);a=ap.parse_args()
    cv2.setNumThreads(4);store=NetReviewStore(a.pack)
    if a.pool or a.selection:
        if not a.pool or not a.selection:ap.error('--pool and --selection are required together')
        finish(store,a.pool,a.selection,a.output);return
    raw=store.path.read_bytes();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'parent_net_references_snapshot.jsonl').write_bytes(raw)
    audit(store,a.output);candidates(store,a.output)
    if store.path.read_bytes()!=raw:raise ValueError('Human journal changed during preparation')
    write(a.output/'provenance.json',dict(parent_pack=str(store.pack),parent_manifest_sha256=sha(store.pack/'manifest.json'),parent_net_reference_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=sha(__file__),originals_unchanged=True))


if __name__=='__main__':main()

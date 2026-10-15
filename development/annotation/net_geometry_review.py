"""Native six-point review; append-only net records never rewrite corner history."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
from annotation_core import OutputLock,write_image,write_json
from annotate import close_window,font,window_visible
from geometry_review import ReferenceStore,GeometryReviewWindow,default_pack,sha
from gui_dialogs import ask,pick_file

ROOT=Path(__file__).resolve().parents[2]
TITLE='Pingpong - Net and Camera Review'
STATUSES={'net_visible':'球网端点可见','net_occluded':'球网端点被遮挡','table_unusable':'四角无法可靠确认'}
_BACKEND_IDENTITY=None


def backend():
    global _BACKEND_IDENTITY
    # Executable code comes from the known local checkout, never a pack field.
    checkout=ROOT/'worktrees/table-geometry-v2'
    if not (checkout/'development/detection/tt3d_net_calibration.py').is_file():
        raise ValueError('未找到本地球网标定模块；仍可保存核查坐标。')
    if str(checkout) not in sys.path:sys.path.insert(0,str(checkout))
    from development.detection import tt3d_net_calibration as module
    if Path(module.__file__).resolve()!=checkout/'development/detection/tt3d_net_calibration.py':
        raise ValueError('标定模块来自其他目录，请重新启动核查程序。')
    sources=[checkout/'development/detection'/name for name in ('tt3d_net_calibration.py','tt3d_camera_estimation.py','auto_table_geometry.py')]
    identity={str(p.resolve()):sha(p) for p in sources}
    if _BACKEND_IDENTITY is not None and _BACKEND_IDENTITY!=identity:
        raise ValueError('标定代码已更新，请重新启动核查程序；人工坐标仍可保存。')
    _BACKEND_IDENTITY=identity
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def ordered(points):
    q=np.asarray(points,float).reshape(4,2);centre=q.mean(axis=0)
    q=q[np.argsort(np.arctan2(q[:,1]-centre[1],q[:,0]-centre[0]))]
    return np.roll(q,-np.argmin(q.sum(axis=1)),axis=0).tolist()


def default_net_pack():
    candidates=sorted((ROOT/'outputs/experiments/tt3d_net_review_20261013').glob('review_pack*'),reverse=True)
    candidates.extend(sorted((ROOT/'worktrees').glob('*/outputs/experiments/tt3d_net_review_20261013/review_pack*'),reverse=True))
    # Resume actual human work before choosing a newer empty pack.
    saved=[p for p in candidates if (p/'net_references.jsonl').is_file() and (p/'net_references.jsonl').stat().st_size]
    candidates=saved+candidates
    return next((p for p in candidates if (p/'manifest.json').is_file()),default_pack())


class NetReviewStore:
    def __init__(self,pack):
        self.source=ReferenceStore(pack);self.pack=self.source.pack
        self.cases=[c for c in self.source.cases if c.get('proposed_corners') or
                    self.source.latest.get(c['case_id'],{}).get('status')=='corners_visible']
        if not self.cases:raise ValueError('没有可供球网核查的四角，请先完成人工四角核查。')
        self.by_id={c['case_id']:c for c in self.cases}
        self.path=self.pack/'net_references.jsonl';self.latest=self.load()

    def image(self,case):return self.source.image(case)

    def base_corners(self,case):
        self.source.latest=self.source.load()
        manual=self.source.latest.get(case['case_id'])
        if manual and manual['status']=='corners_visible':
            seed=dict(source='existing_single_review',corners=ordered(manual['corners']),reference=manual)
        else:
            if not case.get('proposed_corners'):raise ValueError('原四角参考已不可用，请先重新核查四角。')
            seed=dict(source='automatic_proposal_requires_confirmation',corners=ordered(case['proposed_corners']))
        return seed,digest(seed)

    def validate(self,value):
        if not isinstance(value,dict) or value.get('case_id') not in self.by_id:raise ValueError('未知核查样本。')
        case=self.by_id[value['case_id']];status=value.get('status')
        if status not in STATUSES:raise ValueError('请选择球网或台面可辨识状态。')
        reviewer=value.get('reviewer','')
        if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer.strip())>80:raise ValueError('请填写实际核查者，不超过80字。')
        axis=value.get('edge01_axis','unknown')
        if axis not in ('length','width','unknown'):raise ValueError('未知长宽方向。')
        if type(value.get('corner_confirmed')) is not bool:raise ValueError('缺少四角确认状态。')
        q=value.get('corners',[]);net=value.get('net_endpoints',[])
        if status=='table_unusable':
            if q!=[] or net!=[] or axis!='unknown' or value['corner_confirmed']:raise ValueError('台面不可辨识时不保存猜测坐标或方向。')
        else:
            if not value['corner_confirmed']:raise ValueError('请先核对并点击“确认当前四角”，再保存。')
            try:points=np.asarray(q,float)
            except (ValueError,TypeError) as error:raise ValueError('四角格式不正确。') from error
            if points.shape!=(4,2) or not np.isfinite(points).all():raise ValueError('需要四个有限角点。')
            if np.any(points<0) or np.any(points[:,0]>=case['width']) or np.any(points[:,1]>=case['height']):raise ValueError('四角超出原图。')
            if not cv2.isContourConvex(points.astype(np.float32)) or cv2.contourArea(points.astype(np.float32))<1:raise ValueError('四角需沿边界顺序点选，不能交叉或共线。')
            q=ordered(points)
            if status=='net_visible':
                try:n=np.asarray(net,float)
                except (ValueError,TypeError) as error:raise ValueError('球网端点格式不正确。') from error
                if n.shape!=(2,2) or not np.isfinite(n).all():raise ValueError('请点选两个球网顶线端点。')
                if np.any(n<0) or np.any(n[:,0]>=case['width']) or np.any(n[:,1]>=case['height']):raise ValueError('球网端点超出原图。')
                if np.linalg.norm(n[1]-n[0])<5:raise ValueError('两个球网端点过近或重合。')
                net=n.tolist()
            elif net!=[]:raise ValueError('球网端点遮挡时请清空球网点。')
        return dict(schema_version=1,case_id=case['case_id'],group=case['group'],video_id=case['video_id'],
            frame_id=case['frame_id'],image_sha256=case['image_sha256'],video_sha256=case['video_sha256'],
            manifest_sha256=self.source.manifest_hash,status=status,corners=q,net_endpoints=net,
            corner_confirmed=value['corner_confirmed'],edge01_axis=axis,
            axis_confirmed=axis!='unknown',reviewer=reviewer.strip(),
            base_corner_sha256=value.get('base_corner_sha256'),
            reference_source='user_reviewed_six_point_calibration_not_independent_ground_truth')

    def load(self):
        latest={}
        if self.path.is_file():
            for line in self.path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():continue
                record=json.loads(line);normal=self.validate(record)
                for field in ('schema_version','group','video_id','frame_id','image_sha256','video_sha256','manifest_sha256','axis_confirmed','reference_source'):
                    if record.get(field)!=normal[field]:raise ValueError('球网核查记录与原帧来源不符。')
                latest[record['case_id']]=record
        return latest

    def estimate(self,value):
        if value['status']!='net_visible':return dict(usable=False,status='net_or_table_unavailable')
        case=self.by_id[value['case_id']]
        try:
            module=backend()
            result=module.estimate_net_camera(value['corners'],(case['width'],case['height']),value['net_endpoints'],value['edge01_axis'])
            checkout=Path(module.__file__).resolve().parents[2]
            sources=[Path(module.__file__),checkout/'development/detection/tt3d_camera_estimation.py',
                     checkout/'development/detection/auto_table_geometry.py']
            result['source_sha256']={str(p.resolve()):sha(p) for p in sources}
            return result
        except (ImportError,ValueError,cv2.error) as error:
            return dict(usable=False,status='estimation_unavailable',detail=str(error))

    def save(self,value):
        record=self.validate(value)
        self.image(self.by_id[record['case_id']])
        record['calibration']=self.estimate(record)
        record['submitted_utc']=datetime.now(timezone.utc).isoformat()
        with OutputLock(self.pack):
            self.image(self.by_id[record['case_id']])
            seed,current_digest=self.base_corners(self.by_id[record['case_id']])
            if record['base_corner_sha256']!=current_digest:raise ValueError('四角来源已更新，请重新打开本帧核查。')
            record['original_corner_source']=seed['source']
            record['corners_adjusted']=bool(record['corners'] and not np.allclose(record['corners'],seed['corners'],atol=1e-6,rtol=0))
            latest=self.load()
            with self.path.open('ab') as stream:
                stream.write((json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
                stream.flush();os.fsync(stream.fileno())
            latest[record['case_id']]=record;self.latest=latest
        return record

    def prior_reviews(self):
        """Read a declared parent journal; accept append-only corrections, never rewrite it."""
        source=self.source.manifest.get('parent_review_source')
        if not source:return {},{},None
        parent_path=Path(source['pack']).resolve()
        if parent_path==self.pack or not parent_path.is_relative_to(ROOT.resolve()):
            raise ValueError('补充核查的原核查包路径无效。')
        snapshot_path=(self.pack/source['reference_snapshot']).resolve()
        if not snapshot_path.is_relative_to(self.pack) or sha(snapshot_path)!=source['reference_snapshot_sha256']:
            raise ValueError('补充核查的原始记录快照已变化。')
        prepared=snapshot_path.read_bytes()
        with OutputLock(parent_path):
            parent=NetReviewStore(parent_path)
            if parent.source.manifest.get('parent_review_source'):
                raise ValueError('补充核查只支持一个原核查包，不能递归继承。')
            if parent.source.manifest_hash!=source['manifest_sha256']:
                raise ValueError('原核查清单已变化，请重新准备补充帧。')
            raw=parent.path.read_bytes()
            if not raw.startswith(prepared):
                raise ValueError('原核查记录被改写；仅接受保留历史的追加修正。')
            selected={(c['group'],c['video_id'],c.get('camera_epoch',0)) for c in self.cases}
            records={};cases={};stale=[]
            for case_id,record in parent.latest.items():
                case=parent.by_id[case_id]
                if (case['group'],case['video_id'],case.get('camera_epoch',0)) not in selected:continue
                parent.image(case)
                if parent.base_corners(case)[1]!=record['base_corner_sha256']:
                    stale.append(case_id);continue
                if case_id in self.by_id:
                    current=self.by_id[case_id]
                    fields=('group','video_id','frame_id','image_sha256','video_sha256','width','height')
                    if any(case.get(k)!=current.get(k) for k in fields) or case.get('camera_epoch',0)!=current.get('camera_epoch',0):
                        raise ValueError('补充核查与原核查样本编号发生冲突。')
                records[case_id]=record;cases[case_id]=case
            meta=dict(pack=str(parent_path),reference_bytes=raw,manifest_bytes=parent.source.manifest_path.read_bytes(),
                corner_bytes=parent.source.path.read_bytes() if parent.source.path.is_file() else b'',
                reference_sha256=hashlib.sha256(raw).hexdigest(),stale_cases=stale)
        return records,cases,meta

    def export(self,on_progress=None):
        groups=defaultdict(list);stale=[]
        if on_progress:on_progress('正在核对已保存标注与原帧来源…')
        with OutputLock(self.pack):
            self.latest=self.load()
            snapshot=self.path.read_bytes() if self.path.is_file() else b''
            manifest_snapshot=self.source.manifest_path.read_bytes()
            corner_snapshot=self.source.path.read_bytes() if self.source.path.is_file() else b''
            prior,prior_cases,parent=self.prior_reviews()
            combined=dict(prior);combined.update(self.latest)
            all_cases=dict(prior_cases);all_cases.update(self.by_id)
            for case_id,record in combined.items():
                case=all_cases[case_id]
                if case_id not in self.latest:
                    if record['status']=='net_visible' and record['corner_confirmed'] and record['axis_confirmed']:
                        groups[(case['group'],case['video_id'],case.get('camera_epoch',0))].append(record)
                    continue
                self.image(case)
                if self.base_corners(case)[1]!=record['base_corner_sha256']:stale.append(case_id);continue
                if record['status']=='net_visible' and record['corner_confirmed'] and record['axis_confirmed']:
                    groups[(case['group'],case['video_id'],case.get('camera_epoch',0))].append(record)
        results=[]
        for index,((group,vid,epoch),records) in enumerate(groups.items(),1):
            if on_progress:on_progress(f'正在计算 {index}/{len(groups)}：{vid}（{len(records)}帧）…')
            records.sort(key=lambda r:r['frame_id']);case=all_cases[records[0]['case_id']]
            if len(records)<3:result=dict(usable=False,status='at_least_three_reviewed_frames_required')
            elif any((all_cases[r['case_id']]['width'],all_cases[r['case_id']]['height'])!=(case['width'],case['height']) for r in records):
                result=dict(usable=False,status='image_size_changed')
            elif any(all_cases[r['case_id']]['video_sha256']!=case['video_sha256'] for r in records):
                result=dict(usable=False,status='video_source_changed')
            else:
                try:result=backend().estimate_net_sequence(records,(case['width'],case['height']))
                except (ValueError,ImportError,cv2.error) as error:result=dict(usable=False,status='joint_estimation_unavailable',detail=str(error))
            results.append(dict(group=group,video_id=vid,camera_epoch=epoch,reviewed_frames=len(records),result=result))
        directory=self.pack/'calibration_exports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'net_references_snapshot.jsonl').write_bytes(snapshot)
        (directory/'manifest_snapshot.json').write_bytes(manifest_snapshot)
        (directory/'corner_references_snapshot.jsonl').write_bytes(corner_snapshot)
        if parent:
            (directory/'parent_net_references_snapshot.jsonl').write_bytes(parent['reference_bytes'])
            (directory/'parent_manifest_snapshot.json').write_bytes(parent['manifest_bytes'])
            (directory/'parent_corner_references_snapshot.jsonl').write_bytes(parent['corner_bytes'])
        write_json(directory/'cameras.json',results)
        report=dict(reviewed_cases=len(self.latest),total_cases=len(self.cases),pending_cases=len(self.cases)-len(self.latest),
            pending_axis_cases=[key for key,r in self.latest.items() if r['status']=='net_visible' and not r['axis_confirmed']],
            unavailable_cases=[key for key,r in self.latest.items() if r['status']!='net_visible'],
            joint_minimum_frames=3,
            joint_usable_segments=sum(r['result']['usable'] for r in results),stale_cases=stale,
            prior_reviewed_cases=len(prior),combined_reviewed_cases=len(combined),
            parent_review_source=dict(pack=parent['pack'],net_reference_sha256=parent['reference_sha256'],
                stale_cases=parent['stale_cases']) if parent else None,
            review_record_origins={key:'current_pack' if key in self.latest else 'parent_pack' for key in combined},
            net_reference_sha256=hashlib.sha256(snapshot).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_snapshot).hexdigest(),
            original_corner_reference_sha256=hashlib.sha256(corner_snapshot).hexdigest(),
            source_sha256=dict(_BACKEND_IDENTITY or {},**{str(Path(__file__).resolve()):sha(__file__)}),
            independent_3d_validation=False,desktop_event_flow_replaced=False)
        write_json(directory/'summary.json',report)
        return directory,report


CALIBRATION_REASONS={
    'estimated_under_human_axes_and_tt3d_assumptions':'通过',
    'at_least_three_reviewed_frames_required':'可用帧不足3帧',
    'joint_reprojection_error':'投影误差超过阈值',
    'weak_focal_constraint':'焦距约束不足',
    'invalid_camera_pose':'相机位置不合理',
    'focal_bound_hit':'焦距达到搜索边界',
    'image_size_changed':'图像尺寸不一致',
    'video_source_changed':'视频来源不一致',
    'joint_estimation_unavailable':'计算失败，详见导出记录',
}


class NetReviewWindow(GeometryReviewWindow):
    def __init__(self,store,*,video_id=None,frame_id=None,group=None):
        self.export_busy=False;self.export_result=None;self.result_open=False;self.result_scroll=0;self.error_notice=None
        super().__init__(store,video_id=video_id,frame_id=frame_id,group=group)
        self.message=('当前视频无可靠标定样本，已显示核查清单。' if video_id and not any(c['video_id']==video_id for c in store.cases) else
                      '已选最近稳定样本，帧号以标题为准。' if video_id else '')+'先确认四角，再点球网并确认长宽。'

    def show_last_result(self):
        directories=sorted((self.store.pack/'calibration_exports').glob('*'),reverse=True)
        for directory in directories:
            if not (directory/'summary.json').is_file() or not (directory/'cameras.json').is_file():continue
            report=json.loads((directory/'summary.json').read_text(encoding='utf-8-sig'))
            cameras=json.loads((directory/'cameras.json').read_text(encoding='utf-8-sig'))
            self.export_result=(directory,report,cameras);self.result_open=True;self.result_scroll=0
            return True
        self.message='尚无多帧标定结果，请点击“多帧标定”。';return False

    def export_progress(self,message):
        self.message=message
        if getattr(self,'ui_running',False):
            cv2.imshow(TITLE,self.draw());cv2.waitKeyEx(1)

    def draw_result(self):
        directory,report,cameras=self.export_result
        image=Image.new('RGB',(1400,900),(24,28,35));paint=ImageDraw.Draw(image);self.buttons=[];self.rows=[]
        def label(x,y,text,size=18,color=(230,234,241)):paint.text((x,y),str(text),font=font(size),fill=color)
        def button(x,y,w,text,action):
            paint.rectangle((x,y,x+w,y+42),fill=(52,90,125));label(x+10,y+10,text,17)
            self.buttons.append((x,y,x+w,y+42,action))
        label(28,20,'多帧标定已完成 · 结果已保存',28)
        label(28,68,f"通过 {report['joint_usable_segments']}/{len(cameras)} 个视频段",25,(150,225,170))
        label(28,109,f"本轮已登记 {report['reviewed_cases']}/{report['total_cases']} 帧 · 合并记录 {report.get('combined_reviewed_cases',report['reviewed_cases'])} 帧",19)
        raw=self.store.path.read_bytes() if self.store.path.is_file() else b''
        stale=hashlib.sha256(raw).hexdigest()!=report.get('net_reference_sha256') or self.store.source.manifest_hash!=report.get('manifest_sha256')
        parent=report.get('parent_review_source')
        if parent:
            parent_file=Path(parent['pack'])/'net_references.jsonl'
            stale=stale or not parent_file.is_file() or sha(parent_file)!=parent['net_reference_sha256']
        label(28,144,'标注随后有更新；此页显示上次结果，请重新计算。' if stale else '结果对应当前已保存标注；单帧通过不等于多帧通过。',17,(255,210,128))
        columns=[(28,'视频'),(260,'有效帧'),(390,'结果'),(570,'原因'),(1090,'RMSE（原图像素）')]
        for x,title in columns:label(x,190,title,19)
        self.result_scroll=min(max(0,self.result_scroll),max(0,len(cameras)-9))
        for offset,camera in enumerate(cameras[self.result_scroll:self.result_scroll+9]):
            r=camera['result'];y=230+offset*48;good=r['usable'];color=(150,225,170) if good else (255,210,128)
            paint.rectangle((20,y-4,1380,y+39),fill=(35,43,52))
            label(28,y,camera['video_id']);label(260,y,camera['reviewed_frames'])
            label(390,y,'通过' if good else '未通过',18,color)
            label(570,y,CALIBRATION_REASONS.get(r['status'],r['status']),18,color)
            value=r.get('reprojection_rmse_px');label(1090,y,f'{value:.2f}' if value is not None else '—')
        label(28,685,'各视频段需至少3帧可靠标注；通过结果仍基于主点居中、等焦距、零畸变假设。',17)
        path=str(directory)
        label(28,722,'保存位置：'+path[:105],14)
        if len(path)>105:label(28,745,path[105:],14)
        button(28,800,220,'返回核查','result_close');button(270,800,220,'打开结果文件夹','result_folder')
        button(512,800,220,'重新计算','export')
        label(28,861,'原始人工标注已保留；结果页面可再次打开。',17,(150,215,170))
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    def load_case(self,index):
        case=self.store.cases[index];image=self.store.image(case)
        self.store.latest=self.store.load();saved=self.store.latest.get(case['case_id'])
        seed,base_hash=self.store.base_corners(case)
        fresh=saved is not None and saved['base_corner_sha256']==base_hash
        self.index=index;self.original=image;self.saved=saved
        self.corners=[list(p) for p in (saved['corners'] if fresh else seed['corners'])]
        self.points=[list(p) for p in saved['net_endpoints']] if fresh else []
        self.status=saved['status'] if fresh else 'net_visible'
        self.axis=saved['edge01_axis'] if fresh else 'unknown'
        self.corner_confirmed=saved['corner_confirmed'] if fresh else seed['source']=='existing_single_review'
        self.base_hash=base_hash;self.mode='net';self.preview=None;self.dirty=False
        self.zoom=1;self.center=(case['width']/2,case['height']/2);self.drag=None;self.transform=None
        if saved and not fresh:self.message='四角来源有更新，请重新核查本帧。'

    def value(self):
        return dict(case_id=self.case['case_id'],reviewer=self.reviewer,status=self.status,
                    corners=self.corners,net_endpoints=self.points,corner_confirmed=self.corner_confirmed,
                    edge01_axis=self.axis,base_corner_sha256=self.base_hash)

    def changed(self):
        self.dirty=True;self.preview=None
        if self.mode=='corners':self.corner_confirmed=False;self.axis='unknown'

    def draw(self):
        if self.result_open and self.export_result and not self.export_busy:return self.draw_result()
        image=Image.new('RGB',(1400,900),(24,28,35));paint=ImageDraw.Draw(image)
        self.buttons=[];self.rows=[]
        def label(x,y,text,size=16,color=(230,234,241)):paint.text((x,y),str(text),font=font(size),fill=color)
        def button(x,y,width,text,action,active=False):
            box=(x,y,x+width,y+36);paint.rectangle(box,fill=(52,90,125) if active else (45,51,62))
            label(x+8,y+8,text);self.buttons.append((*box,action))
        label(18,12,'球网与相机标定核查',25)
        label(18,49,f"{self.case['group']} / {self.case['video_id']} · 原始帧 {self.case['frame_id']}（0基）· 样本 {self.index+1}/{len(self.store.cases)}")
        label(18,78,'球网点：顶线在两侧台面边线上方的端点，勿点伸出台面的网柱。',16)
        vh,vw=max(1,int(self.case['height']/self.zoom)),max(1,int(self.case['width']/self.zoom))
        ox=int(np.clip(self.center[0]-vw/2,0,self.case['width']-vw));oy=int(np.clip(self.center[1]-vh/2,0,self.case['height']-vh))
        scale=min(968/vw,558/vh);dw,dh=max(1,round(vw*scale)),max(1,round(vh*scale))
        left,top=16+(968-dw)//2,108+(558-dh)//2
        frame=Image.fromarray(cv2.cvtColor(self.original[oy:oy+vh,ox:ox+vw],cv2.COLOR_BGR2RGB))
        image.paste(frame.resize((dw,dh),Image.Resampling.LANCZOS),(left,top))
        self.transform=(left,top,dw,dh,ox,oy,vw,vh)
        layer=Image.new('RGBA',(dw,dh),(0,0,0,0));draw=ImageDraw.Draw(layer)
        def mapped(points):return [((p[0]-ox)*dw/vw,(p[1]-oy)*dh/vh) for p in points]
        for points,color,prefix in [(self.corners,(70,235,190) if self.corner_confirmed else (140,165,190),''),(self.points,(245,115,220),'N')]:
            local=mapped(points)
            if len(local)>1:draw.line(local+([local[0]] if len(local)==4 else []),fill=color,width=2)
            for i,(x,y) in enumerate(local):
                draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='black')
                draw.text((x+8,y-23),prefix+str(i+1),font=font(18),fill=color)
        if self.preview and self.preview.get('selected_camera'):
            projected=mapped(self.preview['selected_camera'].get('projected_net_endpoints',[]))
            if len(projected)==2:draw.line(projected,fill=(255,220,80),width=2)
        image.paste(layer,(left,top),layer)
        for x,width,text,action in [(18,135,'保存并计算 S','save'),(163,112,'上一帧 P','previous'),
            (285,112,'下一帧 N','next'),(407,112,'撤销一点','undo'),(529,130,'恢复已存草稿','clear'),
            (669,140,f'放大 {self.zoom}× Z','zoom'),(819,160,'原图全貌 0','reset')]:button(x,680,width,text,action)
        button(18,727,155,'编辑四角 C','mode:corners',self.mode=='corners')
        button(183,727,155,'点球网端点 E','mode:net',self.mode=='net')
        button(348,727,250,'确认当前四角','confirm_corners',self.corner_confirmed)
        button(608,727,210,'检查投影 I','inspect')
        button(828,727,150,'多帧标定','export')
        button(828,776,150,'查看结果','result_last')
        label(18,777,'绿：已确认四角 · 灰：待确认草稿 · 粉：人工球网点 · 黄：估计投影',15)
        label(18,803,self.message[:82],16,(255,210,128))
        label(18,834,'独立保存：net_references.jsonl；既有四角核查记录不会改写。',15)
        label(18,865,'已修改，尚未保存' if self.dirty else '当前无未保存修改',16,(255,210,128) if self.dirty else (150,215,170))
        label(1015,12,'核查者：'+(self.reviewer or '尚未填写'),18)
        button(1015,49,365,'填写 / 更换核查者','reviewer');button(1015,95,365,'打开其他核查包','pack')
        label(1015,143,f'已登记 {len(self.store.latest)}/{len(self.store.cases)} · 球网点 {len(self.points)}/2',17)
        for i,(key,title) in enumerate(STATUSES.items()):button(1015,179+i*42,365,title,'status:'+key,self.status==key)
        label(1015,310,'四角已确认' if self.corner_confirmed else '四角待确认：请核对灰色草稿',16,(150,215,170) if self.corner_confirmed else (255,210,128))
        for y,title,axis in [(340,'1—2边是长边（2.74米）','length'),(383,'1—2边是短边（1.525米）','width'),(426,'长宽方向尚不确定','unknown')]:button(1015,y,365,title,'axis:'+axis,self.axis==axis)
        for i,p in enumerate(self.points):label(1015,475+i*25,f'N{i+1}: 原图 ({p[0]:.2f}, {p[1]:.2f})',15)
        label(1015,529,'左右端点点选顺序不限；可拖动调整。',15)
        label(1015,555,'遮挡时登记遮挡，不补猜测球网点。',15)
        if self.preview:
            c=self.preview.get('selected_camera') or {}
            label(1015,583,f"焦距 {c.get('focal_px',0):.0f}px · RMSE {c.get('reprojection_rmse_px',0):.2f}px",15)
            label(1015,606,'投影通过检查' if self.preview.get('usable') else '待确认或未通过投影检查',15,(150,215,170) if self.preview.get('usable') else (255,210,128))
        else:label(1015,587,'主点居中、等焦距、零畸变；网高15.25cm。',14)
        label(1015,630,'核查样本（滚轮翻页）',16)
        self.scroll=min(max(0,self.scroll),max(0,len(self.store.cases)-7))
        for i in range(self.scroll,min(len(self.store.cases),self.scroll+7)):
            c=self.store.cases[i];y=662+(i-self.scroll)*31
            if i==self.index:paint.rectangle((1008,y-2,1386,y+28),fill=(43,65,85))
            marker='已存' if c['case_id'] in self.store.latest else '待核查'
            label(1015,y,f"{i+1:02d} {c['video_id']} f{c['frame_id']} {marker}",15)
            self.rows.append((1008,y-2,1386,y+28,i))
        if self.export_busy:
            paint.rectangle((100,360,1300,510),fill=(35,55,75),outline=(110,170,220),width=3)
            label(140,382,'正在进行多帧标定，请稍候…',27)
            label(140,443,self.message,22,(255,225,130))
        if self.error_notice:
            paint.rectangle((90,310,1310,590),fill=(75,35,40),outline=(245,130,130),width=3)
            label(125,333,'操作未完成',27,(255,185,185))
            for i in range(3):label(125,390+i*35,self.error_notice[i*49:(i+1)*49],20)
            button(125,527,210,'返回核查','error_close')
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    def action(self,action):
        if self.export_busy:return
        if action=='export':
            if self.dirty:raise ValueError('请先保存当前帧，再导出多帧标定。')
            self.export_busy=True
            try:path,result=self.store.export(on_progress=self.export_progress)
            except Exception as error:
                self.error_notice=str(error);self.result_open=False;raise
            finally:self.export_busy=False;self.pending=None
            cameras=json.loads((path/'cameras.json').read_text(encoding='utf-8-sig'))
            self.export_result=(path,result,cameras);self.result_open=True;self.result_scroll=0
            self.message=f"标定完成：通过 {result['joint_usable_segments']}/{len(cameras)} 个视频段，结果已保存。"
            print('多帧标定输出：'+str(path),flush=True)
        elif action=='result_last':self.show_last_result()
        elif action=='result_close':self.result_open=False
        elif action=='error_close':self.error_notice=None
        elif action=='result_folder':
            if self.export_result:os.startfile(str(self.export_result[0]))
        elif action=='reviewer':
            value=ask('核查者','填写实际核查者姓名或简称',initial=self.reviewer)
            if value is not None:
                if not value.strip() or len(value.strip())>80:raise ValueError('核查者不能为空，且不超过80字。')
                self.reviewer=value.strip()
        elif action=='save':
            if not self.reviewer:self.action('reviewer')
            self.saved=self.store.save(self.value());self.preview=self.saved['calibration'];self.dirty=False
            self.message='已追加保存；标定通过检查。' if self.preview.get('usable') else '坐标已保存；方向待确认、端点不可见或标定未通过检查。'
        elif action=='inspect':
            value=self.store.validate(self.value()) if self.reviewer else self.store.validate(dict(self.value(),reviewer='preview only; not submitted'))
            self.preview=self.store.estimate(value)
            hint=self.preview.get('axis_diagnostic',{}).get('suggested_edge01_axis','unknown')
            self.message={'length':'球网方向提示：1—2边为长边，请人工确认。','width':'球网方向提示：1—2边为短边，请人工确认。','unknown':'投影已计算；未自动替你确认长宽方向。'}[hint]
        elif action=='confirm_corners':
            if self.status=='table_unusable':raise ValueError('台面不可辨识时不能确认四角。')
            # Reuse geometric validation without pretending net points exist.
            self.store.validate(dict(self.value(),reviewer=self.reviewer or 'preview only',status='net_occluded',
                                     net_endpoints=[],corner_confirmed=True))
            self.corners=ordered(self.corners);self.corner_confirmed=True;self.dirty=True;self.preview=None
            self.message='当前四角已人工确认；请继续点球网端点并确认1—2边长宽。'
        elif action.startswith('mode:'):
            self.mode=action.split(':')[1];self.drag=None
            self.message='正在编辑四角；调整后需重新确认。' if self.mode=='corners' else '正在点球网顶线端点，左右顺序不限。'
        elif action.startswith('axis:'):
            if self.status=='table_unusable':raise ValueError('台面无法确认时不登记长宽方向。')
            if not self.corner_confirmed:raise ValueError('请先确认四角，再指定编号1—2边的长宽。')
            self.axis=action.split(':')[1];self.dirty=True;self.preview=None
        elif action.startswith('status:'):
            status=action.split(':')[1]
            if status==self.status:return
            self.status=status;self.points=[];self.preview=None;self.dirty=True
            if self.status=='table_unusable':self.corners=[];self.corner_confirmed=False;self.axis='unknown'
        elif action=='undo':
            target=self.corners if self.mode=='corners' else self.points
            if target:target.pop();self.changed()
        elif action in ('quit','pack','next','previous') or action.startswith('case:'):
            if self.dirty:self.message='有未保存修改，请先保存或恢复已存草稿。';return
            if action=='quit':self.running=False
            elif action=='pack':
                path=pick_file('选择球网核查包的manifest.json',file_filter='JSON|*.json')
                if path:self.store=NetReviewStore(Path(path).parent);self.load_case(0);self.scroll=0
            else:
                index=int(action.split(':')[1]) if action.startswith('case:') else self.index+(1 if action=='next' else -1)
                self.load_case(max(0,min(len(self.store.cases)-1,index)));self.scroll=max(0,self.index-3)
        elif action=='clear':self.load_case(self.index);self.message='已恢复上次记录或初始草稿。'
        elif action=='zoom':self.zoom={1:2,2:4,4:1}[self.zoom];self.transform=None
        elif action=='reset':self.zoom=1;self.transform=None

    def mouse(self,event,x,y,flags,_):
        if self.export_busy:return
        if self.error_notice:
            if event==cv2.EVENT_LBUTTONDOWN:
                for x0,y0,x1,y1,action in self.buttons:
                    if action=='error_close' and x0<=x<=x1 and y0<=y<=y1:self.pending=action;return
            return
        if self.result_open:
            if event==cv2.EVENT_MOUSEWHEEL:
                delta=(flags>>16)&0xffff;delta=delta-0x10000 if delta>=0x8000 else delta
                self.result_scroll+=-3 if delta>0 else 3
            elif event==cv2.EVENT_LBUTTONDOWN:
                for x0,y0,x1,y1,action in self.buttons:
                    if x0<=x<=x1 and y0<=y<=y1:self.pending=action;return
            return
        if event==cv2.EVENT_MOUSEWHEEL and x>=1000 and y>=640:
            delta=(flags>>16)&0xffff;delta=delta-0x10000 if delta>=0x8000 else delta
            self.scroll+=-3 if delta>0 else 3;return
        point=self.original_point(x,y);target=self.corners if self.mode=='corners' else self.points
        if event==cv2.EVENT_MOUSEMOVE and self.drag is not None:
            if point:target[self.drag]=point;self.changed()
            return
        if event==cv2.EVENT_LBUTTONUP:self.drag=None;return
        if event==cv2.EVENT_RBUTTONDOWN and point:self.center=tuple(point);self.transform=None;return
        if event!=cv2.EVENT_LBUTTONDOWN:return
        for x0,y0,x1,y1,action in self.buttons:
            if x0<=x<=x1 and y0<=y<=y1:self.pending=action;return
        for x0,y0,x1,y1,index in self.rows:
            if x0<=x<=x1 and y0<=y<=y1:self.pending='case:'+str(index);return
        if not point or self.status=='table_unusable' or (self.mode=='net' and self.status!='net_visible'):return
        _,_,dw,dh,_,_,vw,vh=self.transform
        for i,p in enumerate(target):
            if np.hypot((p[0]-point[0])*dw/vw,(p[1]-point[1])*dh/vh)<=10:self.drag=i;return
        limit=4 if self.mode=='corners' else 2
        if len(target)<limit:target.append(point);self.changed()

    def key(self,key):
        if self.export_busy:return
        if self.error_notice:
            if key in (27,13):self.pending='error_close'
            return
        if self.result_open:
            if key in (27,ord('q'),ord('Q')):self.pending='result_close'
            return
        if 65<=key<=90:key+=32
        actions={ord('s'):'save',13:'save',ord('n'):'next',ord('p'):'previous',ord('z'):'zoom',
                 ord('0'):'reset',8:'undo',ord('k'):'clear',ord('q'):'quit',27:'quit',
                 ord('c'):'mode:corners',ord('e'):'mode:net',ord('i'):'inspect'}
        if key in actions:self.pending=actions[key]

    def run(self,smoke_seconds=0):
        self.ui_running=True
        cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL);cv2.resizeWindow(TITLE,1260,810);cv2.setMouseCallback(TITLE,self.mouse)
        deadline=time.perf_counter()+smoke_seconds if smoke_seconds else None
        try:
            while self.running:
                cv2.imshow(TITLE,self.draw());self.key(cv2.waitKeyEx(30))
                if not window_visible(TITLE):
                    if self.dirty:
                        cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL);cv2.resizeWindow(TITLE,1260,810);cv2.setMouseCallback(TITLE,self.mouse)
                        self.message='有未保存修改，请先保存或恢复草稿再关闭。'
                    else:break
                action,self.pending=self.pending,None
                if action:
                    try:self.action(action)
                    except (ValueError,OSError,KeyError,TypeError) as error:
                        self.message=str(error);self.error_notice=str(error);self.result_open=False
                        print('核查操作未完成：'+str(error),flush=True)
                if deadline and time.perf_counter()>=deadline:break
        finally:self.ui_running=False;close_window(TITLE)


def open_net_review(*,pack=None,video_id=None,frame_id=None,group=None,smoke_seconds=0):
    path=Path(pack) if pack else default_net_pack()
    if path is None:
        chosen=pick_file('选择球网核查包manifest.json',file_filter='JSON|*.json')
        if not chosen:return
        path=Path(chosen).parent
    NetReviewWindow(NetReviewStore(path),video_id=video_id,frame_id=frame_id,group=group).run(smoke_seconds)


def main():
    parser=argparse.ArgumentParser(description='原生桌面球网、长宽方向与相机标定核查')
    parser.add_argument('--pack',type=Path,default=default_net_pack());parser.add_argument('--preview',type=Path)
    parser.add_argument('--smoke-seconds',type=float,default=0)
    parser.add_argument('--show-last-calibration',action='store_true',help='直接显示上次已保存的多帧标定结果')
    args=parser.parse_args()
    if args.pack is None:open_net_review();return
    window=NetReviewWindow(NetReviewStore(args.pack))
    if args.show_last_calibration:window.show_last_result()
    if args.preview:
        args.preview.parent.mkdir(parents=True,exist_ok=True);write_image(args.preview,window.draw())
    else:window.run(args.smoke_seconds)


if __name__=='__main__':main()

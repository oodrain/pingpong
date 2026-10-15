"""Append-only spatial annotations, separate from existing event labels."""
from copy import deepcopy
from datetime import datetime, timezone
import json, math, os
from pathlib import Path
from uuid import uuid4
from annotation_core import OutputLock, sha256, write_json

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'worktrees/table-geometry-v2/outputs/experiments/tt3d_trajectory_pilot_20261013'
DEFAULT_PACK=PILOT/'racket_evidence_review_01/review_pack.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare_pack(target=DEFAULT_PACK):
    target=Path(target)
    if target.exists():return target
    source=PILOT/'auxiliary_event_review_01/review_pack.json'
    case=next(c for c in read(source)['cases'] if c['case_id']=='v000038_53')
    evidence=PILOT/'person_motion_fusion_02/evidence.json'
    for path,key in ((case['video_path'],'video_sha256'),(case['tracks_path'],'tracks_sha256')):
        if sha256(Path(path))!=case[key]:raise ValueError('原核查输入已变化')
    pack={'schema_version':1,'case':case,'frame_ids':list(range(51,56)),
          'target_actor_hint':'近侧蓝衣球员的球拍拍面，不含手柄',
          'sources':{str(p):sha256(p) for p in (source,evidence)},
          'purpose':'Spatial evidence for unresolved development case; not independent validation.',
          'labels_prefilled':False,'event_labels_modified':False,'frame_indexing':'zero_based_decoded_frames'}
    with OutputLock(target.parent/'pack_lock'):
        if not target.exists():write_json(target,pack)
    return target

def blank(fid):return {'frame_id':fid,'status':'draft','ball_status':None,'ball_xy':None,
                       'ball_source':None,'racket_status':None,'racket_bbox_xyxy':None}

def validate(pack,state):
    c=pack['case'];fid=state.get('frame_id')
    if type(fid) is not int or fid not in pack['frame_ids']:raise ValueError('帧号不在补标范围')
    if state.get('status') not in ('draft','completed'):raise ValueError('未知完成状态')
    ball=state.get('ball_status');racket=state.get('racket_status')
    if ball not in (None,'visible','not_visible','uncertain') or racket not in (None,'visible','not_visible','uncertain'):
        raise ValueError('未知可见性状态')
    def coords(values,n):
        if not isinstance(values,list) or len(values)!=n or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('标记坐标无效')
        if any(not 0<=v<(c['width'] if i%2==0 else c['height']) for i,v in enumerate(values)):
            raise ValueError('标记超出原图范围')
    if ball=='visible':
        coords(state.get('ball_xy'),2)
        if state.get('ball_source') not in ('manual_click','detector_confirmation'):raise ValueError('缺少球点来源')
    elif state.get('ball_xy') is not None or state.get('ball_source') is not None:raise ValueError('不可见球点不能携带坐标')
    if racket=='visible':
        b=state.get('racket_bbox_xyxy');coords(b,4)
        if b[2]-b[0]<2 or b[3]-b[1]<2:raise ValueError('拍面框太小或起止颠倒')
    elif state.get('racket_bbox_xyxy') is not None:raise ValueError('不可见拍面不能携带框')
    if state['status']=='completed' and (ball is None or racket is None):raise ValueError('请分别核对球点与拍面，或选择看不见 / 不确定')

class RacketJournal:
    def __init__(self,pack=DEFAULT_PACK):
        self.path=Path(pack).resolve();self.pack=read(self.path);self.pack_sha=sha256(self.path)
        self.journal=self.path.parent/'human_racket_evidence.jsonl';self.verify()
    def verify(self):
        if sha256(self.path)!=self.pack_sha:raise ValueError('补标包已变化，请重新打开')
        for p,h in self.pack['sources'].items():
            if sha256(Path(p))!=h:raise ValueError('补标来源已变化')
        c=self.pack['case']
        for p,k in ((c['video_path'],'video_sha256'),(c['tracks_path'],'tracks_sha256')):
            if sha256(Path(p))!=c[k]:raise ValueError('视频或球点已变化，停止保存')
    def latest(self):
        rows={}
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                r=json.loads(line)
                if r['review_pack_sha256']!=self.pack_sha:raise ValueError('记录与补标包不一致')
                validate(self.pack,r['state']);rows[r['state']['frame_id']]=r
        return rows
    def load(self,fid):
        r=self.latest().get(fid)
        return (deepcopy(r['state']),r['revision']) if r else (blank(fid),None)
    def save(self,state,revision):
        validate(self.pack,state);self.verify()
        with OutputLock(self.path.parent/'journal_lock'):
            old=self.latest().get(state['frame_id'])
            if (old['revision'] if old else None)!=revision:raise ValueError('另一窗口已修改本帧，请重新载入')
            row={'revision':str(uuid4()),'previous_revision':revision,'review_pack_sha256':self.pack_sha,
                 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'state':deepcopy(state),
                 'event_labels_modified':False}
            with self.journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        return row['revision']
    def export(self):
        self.verify()
        with OutputLock(self.path.parent/'journal_lock'):
            rows=self.latest()
            if any(rows.get(f,{}).get('state',{}).get('status')!='completed' for f in self.pack['frame_ids']):
                raise ValueError('请先完成全部 5 帧')
            destination=self.path.parent/'exports'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
            write_json(destination,{'case_id':self.pack['case']['case_id'],'review_pack_sha256':self.pack_sha,
                'journal_sha256':sha256(self.journal),'records':[rows[f] for f in self.pack['frame_ids']],
                'independent_holdout':False,'event_labels_modified':False,'automatic_rule_modified':False})
        return destination

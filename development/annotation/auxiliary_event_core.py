"""Independent human event reviews with immutable inputs and append-only revisions."""
from copy import deepcopy
from datetime import datetime, timezone
import json, os
from pathlib import Path
from uuid import uuid4
from annotation_core import OutputLock, sha256 as file_sha256, write_json

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'worktrees/table-geometry-v2/outputs/experiments/tt3d_trajectory_pilot_20261013'
DEFAULT_PACK=PILOT/'auxiliary_event_review_01/review_pack.json'
TYPES={'table_bounce','racket_contact','other'}
OTHER_KINDS={'net_collision','net_graze','normal_flight','tracking_error','unspecified'}

def sha256(path):return file_sha256(Path(path))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def prepare_pack(target=DEFAULT_PACK):
    target=Path(target)
    if target.exists():return target
    screen_path=PILOT/'auxiliary_screening_01/screening.json'
    inference_path=PILOT/'public_regression_01/inference.json'
    direction_path=PILOT/'contact_direction_01/direction_evidence.json'
    manifest_path=ROOT/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json'
    screen=read(screen_path);inference=read(inference_path);direction=read(direction_path)
    metadata={r['video_id']:r for r in read(manifest_path)['videos']}
    priorities={(c['video_id'],c['seed_frame']):0 if c['decision']['event']=='racket_contact_candidate' else 1 if c['decision']['event']=='wrist_motion_conflict' else 2 for c in direction['cases']}
    cases=[]
    for c in screen['selected_review_candidates']:
        vid=c['video_id'];fid=c['frame_id'];meta=metadata[vid]
        video=ROOT/'participant/participant/pingpang/public_data'/meta['path']
        tracks=ROOT/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories'/f'{vid}.jsonl'
        for path in (video,tracks):
            if sha256(path)!=inference['sources'][str(path)]:raise ValueError('核查输入已变化：'+str(path))
        cases.append({'case_id':f'{vid}_{fid}','video_id':vid,'seed_frame':fid,'angle':meta['angle'],
            'video_path':str(video),'video_sha256':inference['sources'][str(video)],
            'tracks_path':str(tracks),'tracks_sha256':inference['sources'][str(tracks)],
            'width':meta['width'],'height':meta['height'],'n_frames':meta['n_frames'],
            'priority':priorities[(vid,fid)],'default_scope':[max(0,fid-12),min(meta['n_frames']-1,fid+12)]})
    cases.sort(key=lambda c:(c['priority'],c['video_id'],c['seed_frame']))
    sources={str(p):sha256(p) for p in (screen_path,inference_path,direction_path,manifest_path)}
    pack={'schema_version':1,'sources':sources,'cases':cases,'known_public_development_set':True,
        'priority_uses_experimental_evidence':True,'human_labels_prefilled':False,'frame_indexing':'zero_based_decoded_frames'}
    with OutputLock(target.parent/'pack_lock'):
        if not target.exists():write_json(target,pack)
    return target

def blank(case):
    state={'status':'draft','primary_event_id':None,'candidate_uncertain':False,'no_events_in_scope':False,'events':[],
        'review_scope':list(case['default_scope']),'displayed_frame':case['seed_frame'],'visited_frames':[]}
    if case.get('review_mode')=='complete_video':
        state.update(events=deepcopy(case.get('reference_events',[])),unknown_intervals=[],
            visited_frames=list(case.get('reference_visited_frames',[])))
    return state

def validate(case,state):
    if state.get('status') not in ('draft','completed'):raise ValueError('未知核查状态')
    scope=state.get('review_scope')
    if not isinstance(scope,list) or len(scope)!=2 or any(type(f) is not int for f in scope) or not 0<=scope[0]<=scope[1]<case['n_frames']:
        raise ValueError('核查范围超出视频')
    events=state.get('events',[]);ids=set();identities=set()
    for e in events:
        if not e.get('event_id') or e['event_id'] in ids:raise ValueError('事件编号重复')
        ids.add(e['event_id']);interval=e.get('interval_frames')
        if e.get('event_type') not in TYPES:raise ValueError('未知事件类型')
        if 'other_kind' in e and (e['event_type']!='other' or e['other_kind'] not in OTHER_KINDS):
            raise ValueError('其他变化细分类型无效')
        if not isinstance(interval,list) or len(interval)!=2 or any(type(f) is not int for f in interval) or not scope[0]<=interval[0]<=interval[1]<=scope[1]:
            raise ValueError('事件区间超出核查范围')
        identity=(e['event_type'],*interval)
        if identity in identities:raise ValueError('同类型、同区间已记录，选择列表中的事件进行修改')
        identities.add(identity)
    primary=state.get('primary_event_id')
    if primary is not None and primary not in ids:raise ValueError('候选对应事件已不存在')
    if state.get('candidate_uncertain') and primary is not None:raise ValueError('无法判断与已指定候选事件冲突')
    if state.get('no_events_in_scope'):
        if events or primary is not None or state.get('candidate_uncertain'):raise ValueError('窗口无事件与其他判断冲突')
        if not set(range(scope[0],scope[1]+1))<=set(state.get('visited_frames',[])):raise ValueError('请先浏览完整窗口再标记无事件')
    complete_video=case.get('review_mode')=='complete_video'
    unknown=state.get('unknown_intervals',[])
    if not isinstance(unknown,list):raise ValueError('看不清区间格式无效')
    for interval in unknown:
        if not isinstance(interval,list) or len(interval)!=2 or any(type(f) is not int for f in interval) or not scope[0]<=interval[0]<=interval[1]<=scope[1]:
            raise ValueError('看不清区间超出核查范围')
        if any(max(interval[0],e['interval_frames'][0])<=min(interval[1],e['interval_frames'][1]) for e in events):
            raise ValueError('看不清区间与已记录事件重叠，请核对事件或调整区间')
    if complete_video:
        if scope!=[0,case['n_frames']-1]:raise ValueError('整段核查必须覆盖完整视频')
        if state.get('candidate_uncertain'):raise ValueError('请用时间区间标记看不清的部分，其他帧继续核查')
        if state.get('no_events_in_scope') and unknown:raise ValueError('有看不清区间时不能声明整段无事件')
        if state['status']=='completed':
            missing=set(range(case['n_frames']))-set(state.get('visited_frames',[]))
            if missing:raise ValueError(f'整段还有 {len(missing)} 帧未浏览，请慢放或逐帧检查后完成')
            if not state.get('all_events_reviewed'):raise ValueError('请确认整段所有可见事件已记录')
    if not complete_video and state['status']=='completed' and primary is None and not state.get('candidate_uncertain') and not state.get('no_events_in_scope'):
        raise ValueError('请指定本候选事件，或选择窗口无事件 / 无法判断')
    if type(state.get('displayed_frame')) is not int or not 0<=state['displayed_frame']<case['n_frames']:raise ValueError('当前帧超出视频')
    if any(type(f) is not int or not 0<=f<case['n_frames'] for f in state.get('visited_frames',[])):raise ValueError('浏览帧号无效')

class ReviewJournal:
    def __init__(self,pack_path=DEFAULT_PACK):
        self.path=Path(pack_path).resolve();self.pack=read(self.path);self.source_sha=sha256(self.path)
        self.cases={c['case_id']:c for c in self.pack['cases']}
        if len(self.cases)!=len(self.pack['cases']):raise ValueError('核查包候选编号重复')
        for path,digest in self.pack['sources'].items():
            if sha256(path)!=digest:raise ValueError('实验来源已变化，请生成新的核查包')
        self.journal=self.path.parent/'human_event_reviews.jsonl'
    def latest(self):
        result={}
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if row['review_pack_sha256']!=self.source_sha:raise ValueError('核查记录与当前核查包不一致')
                if row['case_id'] not in self.cases:raise ValueError('记录包含未知候选')
                validate(self.cases[row['case_id']],row['state']);result[row['case_id']]=row
        return result
    def load(self,case_id):
        row=self.latest().get(case_id)
        return (deepcopy(row['state']),row['revision']) if row else (blank(self.cases[case_id]),None)
    def save(self,case_id,state,expected_revision=None):
        case=self.cases[case_id];validate(case,state)
        if sha256(self.path)!=self.source_sha:raise ValueError('核查包已变化，请重新打开')
        # Validate media identities on every save, so a changed video cannot silently
        # receive annotations from a stale viewer. Existing labels are not rewritten.
        for path,key in ((case['video_path'],'video_sha256'),(case['tracks_path'],'tracks_sha256')):
            if sha256(path)!=case[key]:raise ValueError('视频或球点已变化，停止保存')
        with OutputLock(self.path.parent/'journal_lock'):
            current=self.latest().get(case_id)
            if (current['revision'] if current else None)!=expected_revision:raise ValueError('另一窗口已修改本例，请重新载入后继续')
            row={'schema_version':1,'case_id':case_id,'review_pack_sha256':self.source_sha,'revision':str(uuid4()),
                'previous_revision':expected_revision,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                'video_sha256':case['video_sha256'],'state':deepcopy(state),'production_events_modified':False}
            with self.journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
        return row['revision']
    def export(self):
        if sha256(self.path)!=self.source_sha:raise ValueError('核查包已变化')
        with OutputLock(self.path.parent/'journal_lock'):
            latest=self.latest();records=[];events={}
            for case in self.pack['cases']:
                row=latest.get(case['case_id']);state=row['state'] if row else blank(case)
                if row:
                    for path,key in ((case['video_path'],'video_sha256'),(case['tracks_path'],'tracks_sha256')):
                        if sha256(path)!=case[key]:raise ValueError('核查输入已变化，停止导出')
                primary=next((e for e in state['events'] if e['event_id']==state['primary_event_id']),None)
                records.append({'case_id':case['case_id'],'video_id':case['video_id'],'seed_frame':case['seed_frame'],'review_mode':case.get('review_mode','candidate_window'),
                    'status':state['status'] if row else 'unreviewed','candidate_label':primary['event_type'] if primary else 'uncertain' if state['candidate_uncertain'] else 'no_event' if state.get('no_events_in_scope') else None,
                    'revision':row['revision'] if row else None,'state':state})
                if row and state['status']=='completed':
                    for e in state['events']:
                        key=(case['video_id'],e['event_type'],*e['interval_frames'])
                        item=events.setdefault(key,{'video_id':case['video_id'],'event_type':e['event_type'],'interval_frames':e['interval_frames'],'source_case_ids':[]})
                        item['source_case_ids'].append(case['case_id'])
            target=self.path.parent/'exports'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
            write_json(target,{'review_pack_sha256':self.source_sha,'journal_sha256':sha256(self.journal) if self.journal.exists() else None,
                'completed_cases':sum(r['status']=='completed' for r in records),'total_cases':len(records),
                'frame_indexing':'zero_based_decoded_frames','known_public_development_set':self.pack.get('known_public_development_set',False),
                'review_scope_is_candidate_windows_not_full_video_gt':not all(c.get('review_mode')=='complete_video' for c in self.pack['cases']),
                'complete_video_review_count':sum(r['status']=='completed' and r['review_mode']=='complete_video' for r in records),
                'unknown_intervals_are_unsupervised':True,'records':records,'completed_events_exact_deduplicated':list(events.values())})
        return target

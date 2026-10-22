"""Native multi-event annotation of the frozen 31 auxiliary candidates."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json,time
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from PIL import Image,ImageDraw
from annotate import font,window_visible,close_window
from annotation_core import write_image
from auxiliary_event_core import DEFAULT_PACK,ReviewJournal,prepare_pack,sha256

TITLE='Pingpong - Auxiliary Event Review'
LABELS={'table_bounce':'触台','racket_contact':'击球','other':'其他变化'}
OTHER_LABELS={'unspecified':'未细分','net_collision':'撞网','net_graze':'擦网','normal_flight':'正常飞行','tracking_error':'检测错点'}

class SequentialFrames:
    """Exact decoded indices; backwards access resets and decodes sequentially."""
    def __init__(self,path,count):
        self.path=str(path);self.count=count;self.cap=None;self.next_id=0;self.cache=OrderedDict();self.reset()
    def reset(self):
        if self.cap is not None:self.cap.release()
        self.cap=cv2.VideoCapture(self.path);self.next_id=0
        if not self.cap.isOpened():raise ValueError('无法打开视频')
    def get(self,fid):
        if not 0<=fid<self.count:raise ValueError('帧号超出视频')
        if fid in self.cache:
            self.cache.move_to_end(fid);return self.cache[fid]
        if fid<self.next_id:self.reset()
        while self.next_id<=fid:
            ok,frame=self.cap.read()
            if not ok:raise ValueError('视频提前结束，停止核查')
            at=self.next_id;self.next_id+=1
            self.cache[at]=(frame,float(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
            if len(self.cache)>32:self.cache.popitem(last=False)
        return self.cache[fid]
    def close(self):
        if self.cap is not None:self.cap.release();self.cap=None

class AuxiliaryWindow:
    def __init__(self,pack=DEFAULT_PACK,prior_coverage_pack=None):
        self.store=ReviewJournal(pack);self.cases=self.store.pack['cases'];self.reader=None;self.loaded_video=None
        self.prior_coverage={};self.prior_coverage_sha=None
        if prior_coverage_pack:
            reference=ReviewJournal(prior_coverage_pack);self.prior_coverage_sha=reference.source_sha
            for case in self.cases:
                prior=reference.cases.get(case['case_id'])
                if not prior or any(prior.get(k)!=case.get(k) for k in ('video_sha256','tracks_sha256','n_frames','review_mode')):
                    raise ValueError('旧浏览记录与当前核查视频不一致')
                self.prior_coverage[case['case_id']]=set(prior.get('reference_visited_frames',[]))
        self.reuse_browsing=bool(self.prior_coverage) or self.store.pack.get('prior_human_coverage_reused',False)
        self.buttons=[];self.pending=None;self.play=False;self.last_tick=0;self.overlay=True;self.zoom_center=None
        self.message=('已有人工事件与核查记录已沿用；点“下一未浏览帧”补查缺口，看不清请标区间。' if self.reuse_browsing else '整段慢放并记录所有触台、击球和撞网/擦网；已有人工事件供复核，看不清请标区间。'
            if self.cases[0].get('review_mode')=='complete_video' else '先选事件类型和时刻，再添加；同一窗口可分别记录触台与击球。')
        latest=self.store.latest();index=next((i for i,c in enumerate(self.cases) if latest.get(c['case_id'],{}).get('state',{}).get('status')!='completed'),0)
        self.select(index)
    def select(self,index):
        self.index=index%len(self.cases);self.case=self.cases[self.index];self.play=False;self.zoom_center=None
        c=self.case
        self.complete_video=c.get('review_mode')=='complete_video'
        if self.loaded_video!=c['video_path']:
            if sha256(c['video_path'])!=c['video_sha256'] or sha256(c['tracks_path'])!=c['tracks_sha256']:raise ValueError('视频或球点已变化')
            if self.reader:self.reader.close()
            self.reader=SequentialFrames(c['video_path'],c['n_frames']);self.loaded_video=c['video_path']
            self.tracks={r['frame_id']:r for r in (json.loads(s) for s in Path(c['tracks_path']).read_text(encoding='utf-8').splitlines() if s.strip())}
        self.state,self.revision=self.store.load(c['case_id']);self.frame=self.state['displayed_frame'];self.scope=list(self.state['review_scope'])
        editor=self.state.get('editor',{});self.event_type=editor.get('event_type');self.mode=editor.get('mode','frame');self.range=list(editor.get('range',[None,None]))
        self.selected=editor.get('selected');self.selected_unknown=editor.get('selected_unknown');self.event_scroll=0;self.visited=set(self.state.get('visited_frames',[]));self.dirty=False
        inherited=self.prior_coverage.get(c['case_id'],set())-self.visited
        if inherited:
            self.visited |= inherited;self.dirty=True
            self.state['inherited_human_coverage_pack_sha256']=self.prior_coverage_sha
        self.other_kind=editor.get('other_kind','unspecified')
        self.current_image()
    def current_image(self):
        image,timestamp=self.reader.get(self.frame);self.visited.add(self.frame);return image,timestamp
    def interval(self):
        if self.mode=='frame':return [self.frame,self.frame]
        if self.mode=='between':
            if self.frame>=self.scope[1]:raise ValueError('两帧之间需要后一帧；请向前移动一帧或扩大范围')
            return [self.frame,self.frame+1]
        if None in self.range:raise ValueError('请先设置区间起点与终点')
        if self.range[0]>self.range[1]:raise ValueError('区间起点不能晚于终点')
        return list(self.range)
    def save(self,completed=False):
        updated=deepcopy(self.state);updated.update(review_scope=list(self.scope),displayed_frame=self.frame,visited_frames=sorted(self.visited),
            editor={'event_type':self.event_type,'mode':self.mode,'range':list(self.range),'selected':self.selected,'other_kind':self.other_kind,'selected_unknown':self.selected_unknown})
        if completed:
            updated['status']='completed'
            if self.complete_video:updated['all_events_reviewed']=True
        self.revision=self.store.save(self.case['case_id'],updated,self.revision);self.state=updated;self.dirty=False
        self.message='已保存并标记本例完成。' if completed else '已保存；下次打开可继续。'
    def mutate(self,callback):
        previous=deepcopy(self.state)
        try:
            callback();self.state['status']='draft'
            if self.complete_video:self.state['all_events_reviewed']=False
            self.save()
        except Exception:
            self.state=previous;raise
    def add(self,update=False):
        if self.event_type not in LABELS:raise ValueError('请先选择触台、击球或其他变化')
        interval=self.interval()
        if update and not self.selected:raise ValueError('先点击右侧事件列表选择要修改的一项')
        event_id=self.selected if update else str(uuid4())
        old_selected=self.selected;old_unknown=self.selected_unknown;self.selected_unknown=None
        def change():
            event={'event_id':event_id,'event_type':self.event_type,'interval_frames':interval}
            if self.event_type=='other' and self.store.pack.get('collect_other_subtypes'):
                event['other_kind']=self.other_kind
            self.state['no_events_in_scope']=False
            if update:
                if not any(e['event_id']==event_id for e in self.state['events']):raise ValueError('选中事件已不存在')
                self.state['events']=[event if e['event_id']==event_id else e for e in self.state['events']]
            else:self.state['events'].append(event)
            if not self.complete_video and self.state['primary_event_id'] is None and not self.state['candidate_uncertain']:self.state['primary_event_id']=event_id
            self.selected=event_id
        try:self.mutate(change)
        except Exception:self.selected=old_selected;self.selected_unknown=old_unknown;raise
        self.event_scroll=max(0,len(self.state['events'])-7)
        self.message='事件已保存。需要记录另一事件时，点击“新建事件”。'
    def move(self,fid):
        self.play=False;self.frame=max(self.scope[0],min(self.scope[1],fid));self.zoom_center=None;self.dirty=True;self.current_image()
    def action(self,action):
        if action in ('previous','next','next_unreviewed'):
            if self.dirty:self.save()
            if action=='next_unreviewed':
                latest=self.store.latest();index=self.index
                for offset in range(1,len(self.cases)+1):
                    candidate=(self.index+offset)%len(self.cases)
                    if latest.get(self.cases[candidate]['case_id'],{}).get('state',{}).get('status')!='completed':index=candidate;break
            else:index=self.index+(-1 if action=='previous' else 1)
            self.select(index)
        elif action=='back':self.move(self.frame-1)
        elif action=='forward':self.move(self.frame+1)
        elif action=='back5':self.move(self.frame-5)
        elif action=='forward5':self.move(self.frame+5)
        elif action=='seed':self.move(self.case['seed_frame'])
        elif action=='next_unseen':
            unseen=[f for f in range(self.scope[0],self.scope[1]+1) if f not in self.visited]
            if unseen:self.move(next((f for f in unseen if f>self.frame),unseen[0]))
            else:self.message='整段已浏览；请核对事件列表和看不清区间，再确认完成。'
        elif action=='play':self.play=not self.play;self.last_tick=time.monotonic()
        elif action=='overlay':self.overlay=not self.overlay
        elif action=='scope':
            if self.complete_video:
                self.message='本任务固定为完整视频范围；请标出整段的所有事件。';return
            target=[0,self.case['n_frames']-1] if self.scope==self.case['default_scope'] else list(self.case['default_scope'])
            if any(not target[0]<=e['interval_frames'][0]<=e['interval_frames'][1]<=target[1] for e in self.state['events']):raise ValueError('已有事件超出候选窗口，请保留全片范围')
            if target!=self.scope and self.state.get('no_events_in_scope'):
                self.state['no_events_in_scope']=False;self.state['status']='draft'
            self.scope=target;self.move(self.frame)
        elif action.startswith('type:'):self.event_type=action.split(':')[1];self.dirty=True
        elif action.startswith('mode:'):self.mode=action.split(':')[1];self.dirty=True
        elif action=='start':self.mode='range';self.range[0]=self.frame;self.dirty=True
        elif action=='end':self.mode='range';self.range[1]=self.frame;self.dirty=True
        elif action=='new':self.selected=None;self.selected_unknown=None;self.event_type=None;self.mode='frame';self.range=[None,None];self.other_kind='unspecified';self.dirty=True
        elif action=='other_kind':
            if self.event_type!='other':raise ValueError('先选择“其他变化”，再细分撞网、擦网或正常飞行')
            choices=list(OTHER_LABELS);self.other_kind=choices[(choices.index(self.other_kind)+1)%len(choices)];self.dirty=True
            self.message='其他变化细分：'+OTHER_LABELS[self.other_kind]+'；可继续点击切换，然后添加或更新事件。'
        elif action=='add':self.add()
        elif action=='update':self.add(update=True)
        elif action.startswith('event:'):
            self.selected_unknown=None
            self.selected=action.split(':',1)[1];e=next(e for e in self.state['events'] if e['event_id']==self.selected)
            self.event_type=e['event_type'];self.other_kind=e.get('other_kind','unspecified');self.mode='range';self.range=list(e['interval_frames']);self.move(self.range[0])
        elif action=='primary':
            if self.complete_video:raise ValueError('整段核查不需要指定本候选事件')
            if not self.selected:raise ValueError('请先在列表中选择本候选对应的事件')
            def change():self.state.update(primary_event_id=self.selected,candidate_uncertain=False,no_events_in_scope=False)
            self.mutate(change)
        elif action=='uncertain':
            if self.complete_video:
                interval=self.interval()
                def change_unknown():
                    self.state.setdefault('unknown_intervals',[]).append(interval)
                    self.state['unknown_intervals']=sorted({tuple(v) for v in self.state['unknown_intervals']})
                    self.state['unknown_intervals']=[list(v) for v in self.state['unknown_intervals']]
                    self.state['no_events_in_scope']=False
                self.mutate(change_unknown);self.selected_unknown=interval;self.selected=None
                self.event_scroll=max(0,len(self.state['events'])+len(self.state['unknown_intervals'])-7)
                self.message='该区间已记录为看不清；训练时排除，不当作无事件。';return
            def change():self.state.update(primary_event_id=None,candidate_uncertain=True,no_events_in_scope=False)
            self.mutate(change);self.message='本候选已记录为无法判断；其他已记录事件仍保留。'
        elif action=='no_events':
            if self.state['events']:raise ValueError('列表中已有事件，请先核对；确认误标后逐项删除，再标记窗口无事件')
            missing=set(range(self.scope[0],self.scope[1]+1))-self.visited
            if missing:raise ValueError(f'窗口还有 {len(missing)} 帧未浏览，请用逐帧或慢放检查完整窗口')
            def change():self.state.update(primary_event_id=None,candidate_uncertain=False,no_events_in_scope=True)
            self.mutate(change);self.message='已记录当前范围内无事件，按 C 完成本例；遮挡看不清时应选无法判断。'
        elif action=='delete':
            if self.complete_video and self.selected_unknown is not None:
                interval=list(self.selected_unknown)
                def remove_unknown():self.state['unknown_intervals']=[v for v in self.state.get('unknown_intervals',[]) if v!=interval]
                self.mutate(remove_unknown);self.selected_unknown=None;return
            if not self.selected:raise ValueError('请先在列表中选择一项')
            selected=self.selected
            def change():
                self.state['events']=[e for e in self.state['events'] if e['event_id']!=selected]
                if self.state['primary_event_id']==selected:self.state['primary_event_id']=None
            self.mutate(change);self.selected=None;self.event_scroll=max(0,min(self.event_scroll,len(self.state['events'])-7))
        elif action=='list_up':self.event_scroll=max(0,self.event_scroll-1)
        elif action.startswith('unknown:'):
            self.selected=None;self.selected_unknown=list(map(int,action.split(':')[1:]));self.mode='range';self.range=list(self.selected_unknown);self.move(self.range[0])
        elif action=='list_down':self.event_scroll=min(max(0,len(self.state['events'])+len(self.state.get('unknown_intervals',[]))-7),self.event_scroll+1)
        elif action=='save':self.save()
        elif action=='complete':self.save(completed=True)
        elif action=='export':
            if self.dirty:self.save()
            self.message='已导出：'+str(self.store.export())
        elif action=='reload':self.select(self.index);self.message='已重新读取已保存记录。'
        elif action=='analysis':
            if self.dirty:self.save()
            if self.store.pack.get('purpose')=='train_only_event_labels':
                self.message='这是训练集事件标记包；尚未训练新模型，暂无核查分析结果。'
                return
            from tt3d_review_analysis import open_analysis
            self.play=False;close_window(TITLE)
            try:open_analysis(direction=True,human_review=(self.store.path.parent.parent/'human_event_evaluation_01/evaluation.json').exists())
            finally:
                self.select(self.index)
                cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(TITLE,self.mouse)
    def mouse(self,event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        for box,action in self.buttons:
            if box[0]<=x<=box[2] and box[1]<=y<=box[3]:self.pending=action;return
        if 18<=x<=978 and 701<=y<=720:
            self.move(round(self.scope[0]+(x-18)/960*(self.scope[1]-self.scope[0])));return
        left,top,width,height,scale=self.video_rect
        if left<=x<=left+width and top<=y<=top+height:self.zoom_center=((x-left)/scale,(y-top)/scale)
    def draw(self):
        frame,timestamp=self.current_image();c=self.case
        image=Image.new('RGB',(1440,900),(22,27,35));d=ImageDraw.Draw(image);self.buttons=[]
        def txt(x,y,value,size=17,color=(225,232,243)):d.text((x,y),str(value),font=font(size),fill=color)
        def button(box,title,action,active=False):
            d.rectangle(box,fill=(45,95,122) if active else (47,66,87));txt(box[0]+6,box[1]+7,title,16);self.buttons.append((box,action))
        completed=sum(r['state']['status']=='completed' for r in self.store.latest().values())
        heading='训练集整段 · 逐帧事件核查' if self.complete_video else '训练集候选 · 触台 / 击球核查' if self.store.pack.get('purpose')=='train_only_event_labels' else '辅助候选 · 触台 / 击球人工核查'
        txt(18,10,heading,25)
        position=f"全片 {c['n_frames']} 帧" if self.complete_video else f"候选 {c['seed_frame']} 帧"
        txt(610,17,f"{self.index+1}/{len(self.cases)}  {c['video_id']} / {position}  {c['angle']}  完成 {completed}/{len(self.cases)}")
        guide='沿用已有标注，只补未浏览片段；看不清请设区间后按 U。A/D 逐帧，空格慢放，S 保存。' if self.reuse_browsing else '复核已有人工事件，补齐整段所有事件；看不清请设区间后按 U。A/D 逐帧，空格慢放，S 保存。'
        txt(18,46,guide if self.complete_video else '可添加多个事件；首次添加标为本候选。A/D 逐帧，I/K 区间起止，S 保存，Q 保存后退出。',16)
        scale=min(960/frame.shape[1],540/frame.shape[0]);w,h=round(frame.shape[1]*scale),round(frame.shape[0]*scale)
        left=18+(960-w)//2;top=78+(540-h)//2;self.video_rect=(left,top,w,h,scale)
        annotated=frame.copy();ball=self.tracks.get(self.frame)
        if self.overlay and ball and ball.get('detected') and ball.get('x') is not None:
            cv2.circle(annotated,(round(ball['x']),round(ball['y'])),8,(65,255,170),2)
        image.paste(Image.fromarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)).resize((w,h)),(left,top))
        seen=len(self.visited & set(range(self.scope[0],self.scope[1]+1)))
        txt(18,625,f"当前帧 {self.frame}（0 基）  视频时间 {timestamp:.3f} s  范围 {self.scope[0]}～{self.scope[1]}  窗口已浏览 {seen}/{self.scope[1]-self.scope[0]+1} 帧",17)
        for box,title,action in [((18,653,126,691),'前一帧 A','back'),((136,653,244,691),'后一帧 D','forward'),((254,653,350,691),'前 5 帧','back5'),((360,653,456,691),'后 5 帧','forward5'),((466,653,580,691),'慢放 / 空格','play'),((590,653,704,691),'回到开头' if self.complete_video else '回候选帧','seed'),((714,653,836,691),'整段范围' if self.complete_video else '全片/窗口','scope'),((846,653,978,691),'球点显示','overlay')]:button(box,title,action)
        d.rectangle((18,701,978,720),fill=(50,60,74));x=18+(self.frame-self.scope[0])/max(1,self.scope[1]-self.scope[0])*960;d.rectangle((x-3,701,x+3,720),fill=(80,220,170))
        for box,title,action in [((18,737,225,777),'新建事件','new'),((238,737,445,777),'添加事件 Enter','add'),((458,737,665,777),'更新选中事件','update'),((678,737,978,777),'下一未浏览帧' if self.complete_video else '本候选对应选中事件','next_unseen' if self.complete_video else 'primary'),((18,789,168,831),'上一例 P','previous'),((178,789,328,831),'下一例 N','next'),((338,789,518,831),'下一未完成例','next_unreviewed'),((528,789,718,831),'查看核查分析','analysis'),((728,789,978,831),'导出核查结果','export')]:button(box,title,action)
        txt(1000,78,'选择类型，再选时间定位方式',19)
        for i,(kind,title) in enumerate(LABELS.items()):button((1000+i*142,111,1134+i*142,147),title,'type:'+kind,self.event_type==kind)
        button((1000,157,1204,193),'当前帧接触','mode:frame',self.mode=='frame');button((1215,157,1425,193),'当前与后一帧之间','mode:between',self.mode=='between')
        button((1000,203,1204,239),'设区间起点 I','start');button((1215,203,1425,239),'设区间终点 K','end')
        try:interval=self.interval();time_text=f"{interval[0]} 帧" if interval[0]==interval[1] else f"{interval[0]}～{interval[1]} 帧"
        except ValueError:time_text='区间起止未设置完整'
        txt(1000,248,'待添加：'+(LABELS.get(self.event_type) or '尚未选类型')+' / '+time_text,17,(255,210,128))
        txt(1000,277,'人工事件 / 看不清区间（均可修改）' if self.complete_video else '已记录事件（★ 为本候选对应事件）',17)
        entries=[('event',e) for e in self.state['events']]+[('unknown',v) for v in self.state.get('unknown_intervals',[])]
        if self.complete_video:entries.sort(key=lambda row:row[1]['interval_frames'] if row[0]=='event' else row[1])
        for i,(kind,e) in enumerate(entries[self.event_scroll:self.event_scroll+7]):
            if kind=='unknown':
                a,b=e;title=f'看不清  {a}～{b} 帧';action=f'unknown:{a}:{b}';active=self.selected_unknown==e
            else:
                a,b=e['interval_frames'];star='★ ' if e['event_id']==self.state['primary_event_id'] else ''
                name=OTHER_LABELS.get(e.get('other_kind'),LABELS[e['event_type']]);title=star+name+f'  {a}～{b} 帧'
                action='event:'+e['event_id'];active=self.selected==e['event_id']
            button((1000,306+i*28,1425,331+i*28),title,action,active)
        if self.store.pack.get('collect_other_subtypes'):
            button((1000,512,1074,543),'列表 ↑','list_up');button((1080,512,1154,543),'列表 ↓','list_down')
            button((1160,512,1289,543),'删除选中事件','delete');button((1295,512,1425,543),'细分：'+OTHER_LABELS[self.other_kind],'other_kind',self.event_type=='other')
        else:
            button((1000,512,1100,543),'列表 ↑','list_up');button((1110,512,1210,543),'列表 ↓','list_down');button((1220,512,1425,543),'删除选中事件','delete')
        center=self.zoom_center
        if center is None and ball and ball.get('detected'):center=(ball['x'],ball['y'])
        if center is None:center=(frame.shape[1]/2,frame.shape[0]/2)
        rw=min(240,frame.shape[1]);rh=min(135,frame.shape[0]);zx=max(0,min(frame.shape[1]-rw,round(center[0]-rw/2)));zy=max(0,min(frame.shape[0]-rh,round(center[1]-rh/2)))
        image.paste(Image.fromarray(cv2.cvtColor(annotated[zy:zy+rh,zx:zx+rw],cv2.COLOR_BGR2RGB)).resize((425,239)),(1000,555))
        txt(1000,796,'局部放大：可点击左侧画面改变位置',15)
        button((1000,820,1190,855),'保存 S','save');button((1200,820,1425,855),'确认整段完成 C' if self.complete_video else '本例核查完成 C','complete')
        button((1000,860,1205,893),'整段无事件 V' if self.complete_video else '本窗口无事件 V','no_events');button((1215,860,1425,893),'此区间看不清 U' if self.complete_video else '无法判断 U','uncertain')
        button((820,838,978,862),'重新载入','reload')
        primary=next((e for e in self.state['events'] if e['event_id']==self.state['primary_event_id']),None)
        summary=f"整段：{len(self.state['events'])} 个事件，{len(self.state.get('unknown_intervals',[]))} 个看不清区间" if self.complete_video else '本候选：'+(LABELS[primary['event_type']] if primary else '无法判断' if self.state['candidate_uncertain'] else '已查窗口内无事件' if self.state.get('no_events_in_scope') else '未指定')
        txt(18,838,summary+'  /  '+('已完成' if self.state['status']=='completed' else '草稿'),17)
        # Wrap status so a long export path remains readable.
        for i in range(2):txt(18,863+i*17,self.message[i*68:(i+1)*68],14,(255,210,128))
        return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    def run(self,smoke_seconds=0):
        cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(TITLE,self.mouse);start=time.monotonic()
        try:
            while True:
                if self.play and time.monotonic()-self.last_tick>1/6:
                    self.frame=self.scope[0] if self.frame>=self.scope[1] else self.frame+1;self.zoom_center=None;self.last_tick=time.monotonic();self.dirty=True
                cv2.imshow(TITLE,self.draw());key=cv2.waitKeyEx(30)
                if smoke_seconds and time.monotonic()-start>=smoke_seconds:break
                if key in (27,ord('q')) or not window_visible(TITLE):
                    try:
                        if self.dirty:self.save()
                    except Exception as error:
                        self.message='未能保存：'+str(error)+'；请重新载入后继续。'
                        cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(TITLE,self.mouse)
                        continue
                    break
                action=self.pending;self.pending=None
                action={ord('a'):'back',ord('d'):'forward',ord('p'):'previous',ord('n'):'next',32:'play',13:'add',ord('i'):'start',ord('k'):'end',ord('s'):'save',ord('c'):'complete',ord('u'):'uncertain',ord('v'):'no_events',ord('t'):'type:table_bounce',ord('h'):'type:racket_contact',ord('o'):'type:other'}.get(key,action)
                try:
                    if action:self.play=False if action not in ('play','overlay') else self.play;self.action(action)
                except Exception as error:self.message='未完成：'+str(error)
        finally:self.reader.close();close_window(TITLE)

def ensure_review_pack(pack,allow_prepare=False):
    pack=Path(pack)
    if pack.exists():return pack
    if allow_prepare or pack.resolve()==DEFAULT_PACK.resolve():return prepare_pack(pack)
    raise FileNotFoundError('指定核查包不存在，请先生成该批核查任务：'+str(pack))


def open_auxiliary_review(pack=DEFAULT_PACK,smoke_seconds=0):
    ensure_review_pack(pack);AuxiliaryWindow(pack).run(smoke_seconds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,default=DEFAULT_PACK);p.add_argument('--prior-coverage-pack',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--preview',type=Path);p.add_argument('--smoke-test',action='store_true');a=p.parse_args()
    ensure_review_pack(a.pack,allow_prepare=a.prepare)
    if a.prepare:print(a.pack)
    else:
        window=AuxiliaryWindow(a.pack,a.prior_coverage_pack)
        if a.preview:
            try:write_image(a.preview,window.draw())
            finally:window.reader.close()
        else:window.run(1.5 if a.smoke_test else 0)

"""Native ball-point and racket-face box review on five pinned source frames."""
import argparse, json, time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from annotate import font, window_visible, close_window
from annotation_core import write_image
from auxiliary_event_review import SequentialFrames
from racket_evidence_core import DEFAULT_PACK, RacketJournal, prepare_pack, read

TITLE='Pingpong - Racket Spatial Evidence'
LABELS={None:'未核对','visible':'已标可见','not_visible':'看不见','uncertain':'不确定'}

class RacketWindow:
    def __init__(self,pack=DEFAULT_PACK):
        self.store=RacketJournal(pack);self.case=self.store.pack['case'];self.ids=self.store.pack['frame_ids']
        self.reader=SequentialFrames(self.case['video_path'],self.case['n_frames'])
        self.tracks={r['frame_id']:r for r in (json.loads(s) for s in Path(self.case['tracks_path']).read_text(encoding='utf-8-sig').splitlines() if s.strip())}
        self.mode='racket';self.overlay=False;self.drag=None;self.drag_end=None;self.views=[];self.buttons=[];self.pending=None
        self.message='拖框标拍面（不含手柄），切换球点模式后单击球心；看不清可直接选择看不见或不确定。'
        latest=self.store.latest();self.select(next((i for i,f in enumerate(self.ids) if latest.get(f,{}).get('state',{}).get('status')!='completed'),0))
    def select(self,index):
        self.index=index%len(self.ids);self.frame=self.ids[self.index];self.state,self.revision=self.store.load(self.frame)
        self.dirty=False;self.drag=None;self.drag_end=None;self.zoom_center=None
    def save(self,completed=False):
        updated=deepcopy(self.state)
        if completed:updated['status']='completed'
        self.revision=self.store.save(updated,self.revision);self.state=updated;self.dirty=False
    def changed(self):self.state['status']='draft';self.dirty=True
    def source_point(self,x,y):
        for left,top,w,h,sx,sy,ox,oy in self.views:
            if left<=x<left+w and top<=y<top+h:
                return [min(self.case['width']-1,max(0,(x-left)/sx+ox)),min(self.case['height']-1,max(0,(y-top)/sy+oy))]
        return None
    def action(self,action):
        if action in ('back','forward'):
            if self.dirty:self.save()
            self.select(self.index+(-1 if action=='back' else 1))
        elif action in ('ball','racket'):self.mode=action
        elif action=='overlay':self.overlay=not self.overlay
        elif action=='confirm_detector':
            b=self.tracks[self.frame]
            if not self.overlay:raise ValueError('先显示绿色检测点并目视核对，或切换球点模式单击真实球心')
            if not b.get('detected'):raise ValueError('当前帧没有检测点，请手动点选或标记看不见')
            self.state.update(ball_status='visible',ball_xy=[float(b['x']),float(b['y'])],ball_source='detector_confirmation');self.changed()
        elif ':' in action:
            kind,status=action.split(':');self.state[kind+'_status']=status
            self.state['ball_xy' if kind=='ball' else 'racket_bbox_xyxy']=None
            if kind=='ball':self.state['ball_source']=None
            self.changed()
        elif action=='clear':
            self.state.update(ball_status=None,ball_xy=None,ball_source=None,racket_status=None,racket_bbox_xyxy=None);self.changed()
        elif action=='save':self.save();self.message='当前帧草稿已保存。'
        elif action=='complete':
            self.save(True);latest=self.store.latest()
            pending=next((i for i,f in enumerate(self.ids) if latest.get(f,{}).get('state',{}).get('status')!='completed'),None)
            if pending is None:self.message='全部 5 帧已完成并导出：'+str(self.store.export())
            else:self.select(pending);self.message='上一帧已完成，已跳到下一未完成帧。'
        elif action=='export':self.message='已导出：'+str(self.store.export())
        elif action=='reload':self.select(self.index);self.message='已重新载入当前帧。'
    def mouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            for box,action in self.buttons:
                if box[0]<=x<=box[2] and box[1]<=y<=box[3]:self.pending=action;return
        point=self.source_point(x,y)
        if event==cv2.EVENT_RBUTTONDOWN and point:self.zoom_center=point;return
        if event==cv2.EVENT_LBUTTONDOWN and point:
            if self.mode=='ball':self.state.update(ball_status='visible',ball_xy=point,ball_source='manual_click');self.changed()
            else:self.drag=point;self.drag_end=point
        elif event==cv2.EVENT_MOUSEMOVE and self.drag and point:self.drag_end=point
        elif event==cv2.EVENT_LBUTTONUP and self.drag:
            start=self.drag;self.drag=None;self.drag_end=None
            if point and abs(point[0]-start[0])>=2 and abs(point[1]-start[1])>=2:
                self.state.update(racket_status='visible',racket_bbox_xyxy=[min(start[0],point[0]),min(start[1],point[1]),max(start[0],point[0]),max(start[1],point[1])]);self.changed()
            else:self.message='拍面框未保存：请在图像内拖出有效区域。'
    def draw(self):
        frame,pts=self.reader.get(self.frame);image=Image.new('RGB',(1440,900),(22,27,35));draw=ImageDraw.Draw(image);self.buttons=[]
        def text(x,y,value,size=17,color=(228,234,242)):draw.text((x,y),str(value),font=font(size),fill=color)
        def button(box,title,action,active=False):
            draw.rectangle(box,fill=(45,95,122) if active else (47,66,87));text(box[0]+7,box[1]+7,title,16);self.buttons.append((box,action))
        count=sum(r['state']['status']=='completed' for r in self.store.latest().values())
        text(18,12,'球拍与球点 · 补充空间证据',25)
        text(680,18,f"{self.case['case_id']}   {self.index+1}/5 帧   已完成 {count}/5")
        text(18,51,'原图与放大图均可标记；右键改变放大位置。A/D 切帧，C 完成本帧，S 保存，Q 保存退出。',16)
        annotated=frame.copy();b=self.tracks[self.frame]
        if self.overlay and b.get('detected'):cv2.circle(annotated,(round(b['x']),round(b['y'])),8,(0,255,0),2)
        if self.state['ball_xy'] is not None:cv2.drawMarker(annotated,tuple(round(v) for v in self.state['ball_xy']),(255,200,0),cv2.MARKER_CROSS,18,2)
        box=self.state['racket_bbox_xyxy']
        if self.drag and self.drag_end:box=[*self.drag,*self.drag_end]
        if box is not None:cv2.rectangle(annotated,tuple(round(v) for v in box[:2]),tuple(round(v) for v in box[2:]),(60,90,255),2)
        scale=min(960/frame.shape[1],540/frame.shape[0]);w,h=round(frame.shape[1]*scale),round(frame.shape[0]*scale);left=18+(960-w)//2;top=85+(540-h)//2
        image.paste(Image.fromarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)).resize((w,h)),(left,top))
        center=self.zoom_center or ([b['x'],b['y']] if b.get('detected') else [frame.shape[1]/2,frame.shape[0]/2])
        rw=min(320,frame.shape[1]);rh=min(240,frame.shape[0]);ox=max(0,min(frame.shape[1]-rw,round(center[0]-rw/2)));oy=max(0,min(frame.shape[0]-rh,round(center[1]-rh/2)))
        image.paste(Image.fromarray(cv2.cvtColor(annotated[oy:oy+rh,ox:ox+rw],cv2.COLOR_BGR2RGB)).resize((425,319)),(1000,85))
        self.views=[(left,top,w,h,w/frame.shape[1],h/frame.shape[0],0,0),(1000,85,425,319,425/rw,319/rh,ox,oy)]
        text(1000,414,'放大位置默认来自检测点；标签为空。',15)
        text(1000,442,'当前模式：'+('拖框标拍面' if self.mode=='racket' else '单击标球心'),19,(255,208,120))
        button((1000,477,1205,517),'拍面模式 R','racket',self.mode=='racket');button((1215,477,1425,517),'球点模式 B','ball',self.mode=='ball')
        text(1000,529,'拍面：'+LABELS[self.state['racket_status']],18)
        button((1000,560,1205,600),'拍面看不见','racket:not_visible');button((1215,560,1425,600),'拍面不确定','racket:uncertain')
        text(1000,612,'球点：'+LABELS[self.state['ball_status']],18)
        button((1000,643,1205,683),'球看不见','ball:not_visible');button((1215,643,1425,683),'球点不确定','ball:uncertain')
        button((1000,697,1205,737),'显示检测球点 O','overlay',self.overlay);button((1215,697,1425,737),'确认检测球点','confirm_detector')
        text(18,644,f"当前帧 {self.frame}（0 基）  原视频时间 {pts:.3f} s  {'已完成' if self.state['status']=='completed' else '草稿'}",19)
        text(18,684,'1. 标近侧蓝衣球员的拍面（不含手柄）；看不清时选择对应状态。',19)
        text(18,724,'2. 切换球点模式，在原图或放大图中点球心；只标实际可见小球。',19)
        text(18,764,'3. 点“完成本帧”进入下一帧；完成 5 帧后自动导出。',19)
        for box,title,action in [((18,805,180,845),'前一帧 A','back'),((190,805,352,845),'后一帧 D','forward'),((362,805,525,845),'清空当前标记','clear'),((535,805,698,845),'重新载入','reload'),((1000,753,1205,793),'保存草稿 S','save'),((1215,753,1425,793),'完成本帧 C','complete'),((1000,807,1425,847),'导出全部 5 帧','export')]:button(box,title,action)
        for i in range(2):text(18,853+i*20,self.message[i*94:(i+1)*94],14,(255,208,120))
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    def run(self,smoke_seconds=0):
        cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(TITLE,self.mouse);start=time.monotonic()
        try:
            while True:
                cv2.imshow(TITLE,self.draw());key=cv2.waitKeyEx(40)
                if smoke_seconds and time.monotonic()-start>=smoke_seconds:break
                if key in (27,ord('q')) or not window_visible(TITLE):
                    try:
                        if self.dirty:self.save()
                    except Exception as error:
                        self.message='保存失败：'+str(error);cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(TITLE,self.mouse);continue
                    break
                action=self.pending;self.pending=None
                action={ord('a'):'back',ord('d'):'forward',ord('b'):'ball',ord('r'):'racket',ord('o'):'overlay',ord('s'):'save',ord('c'):'complete'}.get(key,action)
                try:
                    if action:self.action(action)
                except Exception as error:self.message='未完成：'+str(error)
        finally:self.reader.close();close_window(TITLE)

def open_racket_review(pack=DEFAULT_PACK,smoke_seconds=0):
    prepare_pack(pack);RacketWindow(pack).run(smoke_seconds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,default=DEFAULT_PACK);p.add_argument('--preview',type=Path);p.add_argument('--smoke-test',action='store_true');a=p.parse_args();prepare_pack(a.pack)
    window=RacketWindow(a.pack)
    if a.preview:
        try:write_image(a.preview,window.draw())
        finally:window.reader.close()
    else:window.run(1.5 if a.smoke_test else 0)

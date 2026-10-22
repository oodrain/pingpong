"""Standalone CPU-only desktop review; raw videos and model weights are supplied separately."""
import argparse
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk
import core


class FrameReader:
    """Sequential decoding preserves frame identity; short backward jumps use a bounded cache."""
    def __init__(self, path):
        self.path = str(path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise ValueError('无法打开视频')
        self.next_frame = 0
        self.cache = OrderedDict()

    def get(self, frame):
        if frame in self.cache:
            return self.cache[frame]
        if frame < self.next_frame:
            self.capture.release()
            self.capture = cv2.VideoCapture(self.path)
            self.next_frame = 0
        while self.next_frame <= frame:
            okay, bgr = self.capture.read()
            if not okay:
                raise ValueError('视频在请求帧前结束，请核对数据版本与零起始帧号')
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            self.cache[self.next_frame] = (Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), timestamp)
            self.next_frame += 1
            while len(self.cache) > 12:
                self.cache.popitem(last=False)
        return self.cache[frame]

    def close(self):
        self.capture.release()


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.meta, self.cases = core.videos(), core.cases()
        self.dataset = args.dataset_root
        self.baseline, self.proposal = args.baseline, args.proposal
        self.output = args.output
        self.reviews = core.read(self.output) if self.output.is_file() else {'schema': 1, 'reviews': {}}
        self.reader = None
        self.loaded = None
        self.index, self.frame = 0, 0
        self.playing = False
        self.timer = None
        self.center = None
        self.setting_scale = False
        self.visibility = tk.StringVar(value=core.VISIBILITY[''])
        self.judgment = tk.StringVar(value=core.JUDGMENTS[''])
        self.note, self.jump = tk.StringVar(), tk.StringVar()
        self.overlay, self.review_model = tk.BooleanVar(value=True), tk.StringVar(value='baseline')
        root.title('比赛球检测核查 · 原图 / 球框 / 基线 / 新模型')
        root.geometry('1350x900')
        root.minsize(1150, 800)
        top = ttk.Frame(root); top.pack(fill='x', padx=8, pady=6)
        for label, action in [('选择数据根目录', self.choose_dataset), ('载入基线预测目录', lambda: self.choose_predictions(False)),
                              ('载入新模型预测目录', lambda: self.choose_predictions(True))]:
            ttk.Button(top, text=label, command=action).pack(side='left', padx=3)
        ttk.Checkbutton(top, text='叠加球框/球点（O）', variable=self.overlay, command=self.redraw).pack(side='left', padx=8)
        self.header = ttk.Label(root, text='先选择包含 rallies_videos 和 annotations 的数据根目录')
        self.header.pack(fill='x', padx=10)
        body = ttk.Frame(root); body.pack(fill='both', expand=True, padx=8, pady=5)
        left = ttk.Frame(body); left.pack(side='left', fill='y')
        self.list = tk.Listbox(left, width=27, exportselection=False)
        scroll = ttk.Scrollbar(left, orient='vertical', command=self.list.yview)
        self.list.configure(yscrollcommand=scroll.set)
        self.list.pack(side='left', fill='y'); scroll.pack(side='left', fill='y')
        for c in self.cases:
            self.list.insert('end', f"{c['video_id']}  f{c['frame_id']}  {c['angle']}°")
        self.list.bind('<<ListboxSelect>>', self.selected)
        middle = ttk.Frame(body); middle.pack(side='left', fill='both', expand=True, padx=8)
        self.canvas = tk.Canvas(middle, background='#18202c', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Button-1>', self.clicked)
        self.canvas.bind('<Configure>', lambda e: self.redraw())
        self.scale = ttk.Scale(middle, from_=0, to=1, command=self.slider)
        self.scale.pack(fill='x')
        right = ttk.Frame(body, width=265); right.pack(side='right', fill='y')
        self.zoom = ttk.Label(right); self.zoom.pack(pady=5)
        ttk.Label(right, text='点击原图移动放大区域\n绿色框=已有标注（待核查）\n青色圆=基线；橙色叉=新模型\n空心点=低置信；实心点=可信').pack(anchor='w')
        self.detail = ttk.Label(right, text='', wraplength=265, justify='left'); self.detail.pack(fill='x', pady=12)
        ttk.Label(right, text='当前帧可见性').pack(anchor='w')
        ttk.Combobox(right, state='readonly', values=list(core.VISIBILITY.values()), textvariable=self.visibility).pack(fill='x', pady=3)
        ttk.Label(right, text='核查哪条预测').pack(anchor='w', pady=(8,0))
        ttk.Combobox(right, state='readonly', values=['baseline', 'proposal'], textvariable=self.review_model).pack(fill='x')
        ttk.Label(right, text='检测现象').pack(anchor='w', pady=(8,0))
        ttk.Combobox(right, state='readonly', values=list(core.JUDGMENTS.values()), textvariable=self.judgment).pack(fill='x')
        ttk.Label(right, text='说明（可留空）').pack(anchor='w', pady=(8,0))
        ttk.Entry(right, textvariable=self.note).pack(fill='x')
        ttk.Button(right, text='保存当前帧核查（S）', command=self.save_review).pack(fill='x', pady=10)
        controls = ttk.Frame(root); controls.pack(fill='x', padx=8, pady=5)
        for label, action in [('上一例 P', lambda: self.change_case(-1)), ('下一例 N', lambda: self.change_case(1)),
                              ('−5帧', lambda: self.move(-5)), ('前一帧 ←', lambda: self.move(-1)),
                              ('后一帧 →', lambda: self.move(1)), ('+5帧', lambda: self.move(5)),
                              ('播放/暂停 空格', self.toggle_play)]:
            ttk.Button(controls, text=label, command=action).pack(side='left', padx=2)
        ttk.Entry(controls, width=7, textvariable=self.jump).pack(side='left', padx=5)
        ttk.Button(controls, text='跳帧', command=self.go_to).pack(side='left')
        self.footer = ttk.Label(root, text='帧号从0开始；核查只写独立文件，原视频和原标注保持原样。')
        self.footer.pack(fill='x', padx=10, pady=7)
        for key, action in [('<Left>', lambda: self.move(-1)), ('<Right>', lambda: self.move(1)),
                            ('<space>', self.toggle_play), ('n', lambda: self.change_case(1)),
                            ('p', lambda: self.change_case(-1)), ('s', self.save_review),
                            ('o', self.toggle_overlay)]:
            root.bind(key, lambda event, fn=action: self.hotkey(event, fn))
        root.protocol('WM_DELETE_WINDOW', self.close)
        if self.dataset:
            self.select_case(0)

    def hotkey(self, event, fn):
        if event.widget.winfo_class() not in ('TEntry', 'Entry', 'TCombobox'):
            fn(); return 'break'

    def alert(self, fn):
        try:
            return fn()
        except Exception as error:
            if self.args.smoke or self.args.preview:
                raise
            messagebox.showerror('请检查输入', str(error))

    def choose_dataset(self):
        value = filedialog.askdirectory(title='选择同时包含 rallies_videos 和 annotations 的目录')
        if value:
            self.dataset = Path(value); self.loaded = None
            self.select_case(self.index)

    def choose_predictions(self, proposal):
        value = filedialog.askdirectory(title='选择含 视频编号.json / jsonl 的预测目录')
        if value:
            if proposal: self.proposal = Path(value)
            else: self.baseline = Path(value)
            self.loaded = None; self.select_case(self.index)

    def selected(self, event):
        if self.list.curselection(): self.select_case(self.list.curselection()[0])

    def select_case(self, index):
        if not self.dataset:
            self.choose_dataset(); return
        def load():
            self.pause(); self.index = max(0, min(len(self.cases)-1, index))
            case = self.cases[self.index]; vid = case['video_id']; meta = self.meta[vid]
            if self.loaded != vid:
                if self.reader: self.reader.close()
                self.ann = core.annotation(self.dataset, meta)
                self.reader = FrameReader(core.data_path(self.dataset, meta['video_path']))
                self.base, self.base_hash = core.tracks(self.baseline, vid, self.ann['num_frames'])
                self.new, self.new_hash = core.tracks(self.proposal, vid, self.ann['num_frames'])
                self.loaded = vid
            self.frame = case['frame_id']; self.center = None
            self.scale.configure(to=self.ann['num_frames']-1)
            self.list.selection_clear(0,'end'); self.list.selection_set(self.index); self.list.see(self.index)
            self.load_review(); self.redraw()
        self.alert(load)

    def change_case(self, delta): self.select_case(self.index + delta)

    def move(self, delta):
        if not self.reader: return
        self.frame = max(0, min(self.ann['num_frames']-1, self.frame+delta))
        self.load_review(); self.alert(self.redraw)

    def go_to(self):
        def go(): self.move(int(self.jump.get())-self.frame)
        self.alert(go)

    def slider(self, value):
        if self.reader and not self.setting_scale: self.move(round(float(value))-self.frame)

    def clicked(self, event):
        if not self.reader: return
        x,y,scale = self.transform
        px,py = (event.x-x)/scale,(event.y-y)/scale
        if 0 <= px < self.ann['frame_width'] and 0 <= py < self.ann['frame_height']:
            self.center = (px,py); self.redraw()

    def redraw(self):
        if not self.reader: return
        raw, timestamp = self.reader.get(self.frame)
        if raw.size != (self.ann['frame_width'], self.ann['frame_height']):
            raise ValueError('视频尺寸与标注不一致')
        im = raw.copy(); draw = ImageDraw.Draw(im)
        bb = core.boxes(self.ann, self.frame)
        if self.overlay.get():
            for box in bb: draw.rectangle(tuple(box), outline='#55ee77', width=2)
            for tracks,color in [(self.base,'#00d4ff'),(self.new,'#ff9900')]:
                if tracks is None: continue
                row = tracks[self.frame]
                if row['detected']:
                    x,y = row['x'],row['y']; r=6
                    draw.ellipse((x-r,y-r,x+r,y+r), outline=color, width=2,
                                 fill=color if core.trusted(row) else None)
                    if tracks is self.new:
                        draw.line((x-r,y-r,x+r,y+r), fill=color, width=2)
                        draw.line((x-r,y+r,x+r,y-r), fill=color, width=2)
        self.last_render = im
        width,height=max(100,self.canvas.winfo_width()),max(100,self.canvas.winfo_height())
        scale=min(width/im.width,height/im.height); w,h=max(1,round(im.width*scale)),max(1,round(im.height*scale))
        self.transform=((width-w)/2,(height-h)/2,scale)
        self.photo=ImageTk.PhotoImage(im.resize((w,h),Image.Resampling.BILINEAR))
        self.canvas.delete('all'); self.canvas.create_image(width/2,height/2,image=self.photo)
        center = self.center or ((bb[0][0]+bb[0][2])/2,(bb[0][1]+bb[0][3])/2) if bb else self.center
        if center is None:
            row = self.base[self.frame] if self.base else None
            center=(row['x'],row['y']) if row and row['detected'] else (im.width/2,im.height/2)
        x,y=map(round,center)
        zoom=im.crop((x-48,y-48,x+48,y+48)).resize((240,240),Image.Resampling.NEAREST)
        self.zoom_photo=ImageTk.PhotoImage(zoom); self.zoom.configure(image=self.zoom_photo)
        baseline = self.base[self.frame] if self.base else None
        proposal = self.new[self.frame] if self.new else None
        case=self.cases[self.index]
        self.header.configure(text=f"{self.loaded}  {case['angle']}°  f{self.frame}/{self.ann['num_frames']-1}  解码时间 {timestamp:.6f}s")
        self.detail.configure(text='基线：'+core.status(baseline)+'\n\n新模型：'+core.status(proposal)+
                              '\n\n已有球框：'+str(len(bb))+'\n有球框不保证这一帧清晰可见；无球框也不代表无球。'+
                              '\n\n清单原状态：'+case['category']+'（仅对应关注帧）')
        self.setting_scale=True; self.scale.set(self.frame); self.setting_scale=False
        self.jump.set(str(self.frame))

    def load_review(self):
        key=f'{self.loaded}:{self.frame}'
        row=self.reviews['reviews'].get(key,{})
        self.visibility.set(core.VISIBILITY.get(row.get('visibility',''),'未核查'))
        self.judgment.set(core.JUDGMENTS.get(row.get('judgment',''),'未判断'))
        self.review_model.set(row.get('model','baseline')); self.note.set(row.get('note',''))

    def save_review(self):
        if not self.reader: return
        def save():
            visibility=next(k for k,v in core.VISIBILITY.items() if v==self.visibility.get())
            judgment=next(k for k,v in core.JUDGMENTS.items() if v==self.judgment.get())
            model=self.review_model.get(); digest=self.base_hash if model=='baseline' else self.new_hash
            if judgment not in ('', 'unknown') and digest is None:
                raise ValueError('请先载入所选模型预测；没有载入预测不能标成模型漏检。只核查可见性时，请把检测现象留为未判断。')
            self.reviews['reviews'][f'{self.loaded}:{self.frame}']=dict(video_id=self.loaded,frame_id=self.frame,
                visibility=visibility,judgment=judgment,model=model,prediction_sha256=digest,note=self.note.get())
            core.save(self.output,self.reviews)
            self.footer.configure(text=f'已保存 {self.loaded} f{self.frame}；原标注未修改。')
        self.alert(save)

    def toggle_overlay(self): self.overlay.set(not self.overlay.get()); self.redraw()

    def toggle_play(self):
        if not self.reader: return
        if self.playing: self.pause()
        else: self.playing=True; self.tick()

    def pause(self):
        self.playing=False
        if self.timer is not None:
            self.root.after_cancel(self.timer); self.timer=None

    def tick(self):
        if not self.playing: return
        if self.frame >= self.ann['num_frames']-1:
            self.playing=False; return
        self.move(1)
        self.timer=self.root.after(100,self.tick)  # Deliberate 10 decoded frames/s review speed, not an FPS benchmark.

    def close(self):
        self.pause()
        if self.reader: self.reader.close()
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--proposal',type=Path)
    parser.add_argument('--output',type=Path,default=core.HERE/'local_output/reviews.json')
    parser.add_argument('--video-id',choices=list(core.videos()))
    parser.add_argument('--frame',type=int)
    parser.add_argument('--smoke',action='store_true',help='Construct and exercise the desktop UI without manual interaction')
    parser.add_argument('--preview',type=Path,help='Save the current video overlay without opening an interactive window')
    args=parser.parse_args()
    if (args.smoke or args.preview) and not args.dataset_root:
        parser.error('--smoke/--preview requires --dataset-root')
    root=tk.Tk()
    if args.smoke or args.preview: root.withdraw()
    app=App(root,args)
    if args.video_id:
        app.select_case(next(i for i,c in enumerate(app.cases) if c['video_id']==args.video_id))
    if args.frame is not None and app.reader: app.move(args.frame-app.frame)
    root.update_idletasks()
    if args.smoke:
        original=app.frame; app.move(1); app.move(-1); assert app.frame==original
        app.toggle_overlay(); app.toggle_overlay()
        app.visibility.set(core.VISIBILITY['uncertain']); app.save_review()
        assert f'{app.loaded}:{app.frame}' in core.read(args.output)['reviews']
        print('DESKTOP SMOKE PASSED')
    if args.preview:
        args.preview.parent.mkdir(parents=True,exist_ok=True); app.last_render.save(args.preview)
    if args.smoke or args.preview: app.close()
    else: root.mainloop()


if __name__=='__main__': main()

"""A local, frame-accurate video and annotation workspace for AI Pingpong."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from math import atan2, ceil, degrees, hypot
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from uuid import uuid4

_LOCAL_TCL = Path(sys.prefix) / "tcl"
if (_LOCAL_TCL / "tcl8.6/init.tcl").is_file():
    os.environ["TCL_LIBRARY"] = str(_LOCAL_TCL / "tcl8.6")
    os.environ["TK_LIBRARY"] = str(_LOCAL_TCL / "tk8.6")

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageOps, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "annotation"))

from core import (ROOT, VideoEntry, SequentialVideo, SessionStore,
                  discover_catalog, entry_from_path, load_reference)
from overlays import interval_of, load_history, read_own_session, timed_tail, trajectory_segments
import review_theme
from review_playback import ReviewPlayback

BG = "#eff3f7"
WHITE = "#ffffff"
INK = "#18283b"
MUTED = "#6b7d90"
LINE = "#dce4ed"
ACCENT = "#087f86"
VIDEO = "#111c2a"
ROLES = {"development": "开发集", "reserved": "内部保留", "public_regression": "公开验证", "imported": "自行导入"}
TYPES = {"触台": "table_bounce", "触拍": "racket_contact", "触网": "net_contact", "落地": "floor_contact", "其他": "other"}
TYPE_NAMES = {v: k for k, v in TYPES.items()}
COLORS = {"table_bounce": "#32e2c4", "racket_contact": "#ffbd69", "net_contact": "#b49cff", "floor_contact": "#f588a1", "other": "#9bd0ff"}
VIDEO_TYPES = [("视频", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm"), ("所有文件", "*.*")]
REVIEW_ZOOMS = ("适应", "1.25×", "1.5×", "2×", "3×", "4×", "6×", "8×")
TRAIL_SECONDS = 0.48
# Oldest to newest. The leading end gets both a lighter color and a wider stroke.
TRAIL_STROKES = (("#2b505b", 1.0), ("#376472", 1.2), ("#447d89", 1.5), ("#5296a1", 1.9),
                 ("#61afbc", 2.3), ("#73c8d4", 2.8), ("#8be2ec", 3.3))


class Studio(tk.Tk):
    def __init__(self, output: Path, initial_video=None, portable=False, table_demo=None):
        super().__init__()
        self.portable = portable
        self.table_demo = table_demo
        self.source_frame_offset = (table_demo or {}).get("source_frame_offset", 0)
        self.source_time_offset = (table_demo or {}).get("source_time_offset", 0)
        self.table_view = None
        self.review_view = None
        self.review_layout = None
        self.timeline_markers = []
        self.timeline_hover_x = None
        self.preview_end = None
        self.cpu_busy = False
        self.cpu_result = None
        self.cpu_track_frames = []
        self.title("Pingpong · 触台复核工作台" if table_demo else "Pingpong · 视频工作台")
        if portable or table_demo:
            width = max(900, min(1260, self.winfo_screenwidth() - 72))
            height = max(610, min(820, self.winfo_screenheight() - 120))
            self.geometry(f"{width}x{height}")
            self.minsize(900, 610)
        else:
            self.geometry("1460x900")
            self.minsize(1160, 760)
        self.configure(bg=review_theme.BACKGROUND if table_demo else BG)
        self.output = output
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-decode")
        self.analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")
        self.results = queue.Queue()
        self.catalog = [] if portable or table_demo else discover_catalog()
        self.reader = None
        self.store = None
        self.entry = None
        self.frame = self.target_frame = 0
        self.image_bgr = None
        self.photo = None
        self.picture_item = None
        self.picture_key = None
        self.render_job = None
        self.preview_window_size = None
        self.pts = None
        self.generation = 0
        self.loading = self.reading = self.playing = self.closed = False
        self.play_at = 0.0
        self.review_playback = None
        self.stream_preview = None
        self.displayed_event = None
        self.draft = None
        self.interval_start = None
        self.roi = None
        self.pan_start = None
        self.transform = None
        self.references = {"official": [], "trajectory": [], "history": [], "imported": []}
        self.saved_own = None
        self.boxes = {}
        self.reference_names = {}
        self.mode = tk.StringVar(value="观察")
        self.search = tk.StringVar()
        self.role_filter = tk.StringVar(value="全部视频")
        self.event_type = tk.StringVar(value="触台")
        self.reason = tk.StringVar()
        self.interval_touch = tk.BooleanVar(value=False)
        self.speed = tk.StringVar(value="1×")
        self.zoom = tk.StringVar(value="适应")
        self.jump = tk.StringVar(value="0")
        self.status = tk.StringVar(value="就绪 · 选择视频开始观察")
        self.point_status = tk.StringVar(value="尚未选点")
        self.interval_status = tk.StringVar(value="未设置区间起点")
        self.save_status = tk.StringVar(value="观察模式")
        self.layers = {name: tk.BooleanVar(value=(name in {"own", "history", "official", "trajectory"})) for name in ("own", "history", "official", "trajectory", "imported", "boxes")}
        self.point_radius = tk.StringVar(value="2")
        self.event_scope = tk.StringVar(value="整段位置")
        if table_demo:
            self.show_time_tail = tk.BooleanVar(value=True)
            self.show_table_outline = tk.BooleanVar(value=False)
            self.timeline_gradient = None
            self.timeline_gradient_width = 0
            self.timeline_axis = None
            self.timeline_playhead = 10
        self.startup_jobs = []
        self._style()
        self._build()
        self._bind_keys()
        self._load_preferences()
        self._filter_library()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.report_callback_exception = self._tk_error
        self.poll_job = self.after(25, self._poll)
        self.autosave_job = self.after(15000, self._autosave)
        self.startup_jobs.append(self.after(200, self._show_window))
        if initial_video:
            path = Path(initial_video).resolve()
            entry = next((e for e in self.catalog if e.path.resolve() == path), None) or self._entry_from_path(path)
            if entry not in self.catalog:
                self.catalog.append(entry)
                self._filter_library()
            self.startup_jobs.append(self.after(150, lambda: self.open_entry(entry)))

    def _show_window(self):
        # pythonw may inherit SW_HIDE from the console-free launcher.
        if self.table_demo and sys.platform == "win32":
            from ctypes import c_void_p, windll

            if self.state() == "withdrawn":
                self.deiconify()
            handle = c_void_p(int(self.wm_frame(), 0))
            if not windll.user32.IsWindowVisible(handle):
                windll.user32.ShowWindow(handle, 5)
        else:
            self.withdraw()
            self.deiconify()
            self.lift()
        if self.review_layout is not None:
            self.review_layout.restore_sizes()

    def _style(self):
        self.option_add("*Menu.font" if self.table_demo else "*Font", "{Microsoft YaHei UI} 10")
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", font=("Microsoft YaHei UI", 10), foreground=INK, background=WHITE)
        s.configure("TFrame", background=WHITE)
        s.configure("TLabel", background=WHITE)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        s.configure("Section.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        s.configure("TButton", padding=(12, 8), borderwidth=0, background="#eaf0f5")
        s.map("TButton", background=[("active", "#dce8ee"), ("disabled", "#f1f4f7")], foreground=[("disabled", "#a5afba")])
        s.configure("Accent.TButton", background=ACCENT, foreground=WHITE)
        s.map("Accent.TButton", background=[("active", "#096c72"), ("disabled", "#b2cbcd")], foreground=[("disabled", "#f4f8f8")])
        s.configure("Small.TButton", padding=(8, 5))
        s.configure("ReviewAction.TButton", padding=(5, 5))
        s.configure("Panel.TMenubutton", padding=(6, 2), font=("Microsoft YaHei UI", 9))
        s.configure("TRadiobutton", padding=(10, 7), background=WHITE)
        s.map("TRadiobutton", background=[("selected", "#e1f2f1")])
        s.configure("TCheckbutton", padding=(1, 6), background=WHITE)
        s.configure("TEntry", padding=6, fieldbackground="#f6f8fb", bordercolor=LINE)
        s.configure("TCombobox", padding=5, fieldbackground="#f6f8fb", bordercolor=LINE)
        s.map("TCombobox", fieldbackground=[("readonly", "#f6f8fb")])
        s.configure("Treeview", rowheight=34, borderwidth=0, fieldbackground=WHITE, background=WHITE)
        s.map("Treeview", background=[("selected", "#dcf0ef")], foreground=[("selected", "#076970")])
        s.configure("Treeview.Heading", background="#f3f6f9", relief="flat", padding=(7, 8), font=("Microsoft YaHei UI", 9))
        s.configure("Review.Treeview", rowheight=30)
        s.configure("Review.Treeview.Heading", padding=(4, 4))
        s.configure("TNotebook", background=WHITE, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(13, 10), background="#edf2f7")
        s.map("TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", ACCENT)])
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG, borderwidth=0)
        s.configure("TPanedwindow", background=LINE)
        if self.table_demo:
            self.review_button_images = review_theme.apply(s)
            self.option_add("*TCombobox*Listbox.background", review_theme.CONTROL)
            self.option_add("*TCombobox*Listbox.foreground", review_theme.TEXT)
            self.option_add("*TCombobox*Listbox.selectBackground", review_theme.SELECTED)
            self.option_add("*TCombobox*Listbox.selectForeground", review_theme.TEXT)

    def _build(self):
        top = ttk.Frame(self, padding=(18, 10) if self.table_demo else (22, 14))
        top.pack(fill="x")
        brand = ttk.Frame(top)
        brand.pack(side="left")
        if not self.table_demo:
            ttk.Label(brand, text="PINGPONG", foreground=ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(brand, text="触台复核工作台" if self.table_demo else "视频工作台", style="Title.TLabel").pack(anchor="w")
        if self.table_demo:
            ttk.Button(top, text="使用说明", style="Quiet.TButton", command=self.help).pack(side="right")
        modes = ttk.Frame(top, padding=(0 if self.table_demo else 34, 0))
        modes.pack(side="right" if self.table_demo else "left")
        if not self.portable and not self.table_demo:
            for mode in ("观察", "标注"):
                ttk.Radiobutton(modes, text=mode + "模式", value=mode, variable=self.mode, command=self._mode_changed).pack(side="left", padx=3)
        if self.table_demo:
            ttk.Checkbutton(modes, text="球轨迹", variable=self.show_time_tail,
                            command=self.render).pack(side="left", padx=(0, 12))
            ttk.Checkbutton(modes, text="台面线", variable=self.show_table_outline,
                            command=self.render).pack(side="left", padx=(0, 12))
        else:
            ttk.Button(top, text="使用说明", command=self.help).pack(side="right", padx=(8, 0))
        self.export_button = ttk.Button(top, text="导出落点" if self.portable else "导出", command=self.export, state="disabled")
        if not self.table_demo:
            self.export_button.pack(side="right", padx=(8, 0))
        self.save_button = ttk.Button(top, text="保存  Ctrl+S", command=self.save, style="Accent.TButton", state="disabled")
        if not self.portable and not self.table_demo:
            self.save_button.pack(side="right", padx=(8, 0))
        if not self.table_demo:
            ttk.Button(top, text="＋ 导入视频", command=self.import_videos).pack(side="right")
        self.analyze_button = ttk.Button(top, text="自动分析 · CPU" if self.portable else "自动分析 · 第三轮",
                                         command=self.analyze_current, state="disabled" if self.portable else "normal")
        if not self.table_demo:
            self.analyze_button.pack(side="right", padx=6)
        if not self.table_demo:
            ttk.Separator(self).pack(fill="x")
        self.panes = ttk.Panedwindow(self, orient="horizontal")
        self.panes.pack(fill="both", expand=True, padx=14, pady=(10, 0) if self.table_demo else 14)
        self.library = ttk.Frame(self.panes, padding=14, width=220)
        self.center = ttk.Frame(self.panes, padding=14, style="ReviewCard.TFrame" if self.table_demo else "TFrame")
        self.inspector = ttk.Frame(self.panes, padding=0 if self.table_demo else 12, width=302,
                                   style="Workspace.TFrame" if self.table_demo else "TFrame")
        if not self.table_demo:
            self.panes.add(self.library, weight=0)
        self.panes.add(self.center, weight=1)
        self.panes.add(self.inspector, weight=0)
        if not self.table_demo:
            self.startup_jobs.append(self.after(100, self._initial_sashes))
        self._build_library()
        self._build_player()
        if self.table_demo:
            self._build_review_panels(modes)
        else:
            self._build_inspector()
        bar = ttk.Frame(self, padding=(14, 5) if self.table_demo else (20, 7),
                        style="Workspace.TFrame" if self.table_demo else "TFrame")
        if self.table_demo:
            bar.pack(side="bottom", fill="x", before=self.panes)
        else:
            bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status, style="Workspace.TLabel" if self.table_demo else "Muted.TLabel").pack(side="left")
        if self.table_demo:
            ttk.Label(bar, textvariable=self.save_status, style="Workspace.TLabel").pack(side="right")
        else:
            ttk.Label(bar, textvariable=self.save_status, foreground=ACCENT).pack(side="right")

    def _initial_sashes(self):
        w = self.panes.winfo_width()
        self.panes.sashpos(0, 220)
        self.panes.sashpos(1, w - 310)

    def _build_review_panels(self, toolbar):
        from review_layout import ReviewLayout, TrackView
        from review_view import ReviewView
        from table_view import TableView
        self.review_layout = ReviewLayout(self, toolbar)
        self.table_view = TableView(self.review_layout.hosts["table"], self, self.table_demo)
        self.review_view = ReviewView(self.review_layout.hosts["review"], self)
        self.track_view = TrackView(self.review_layout.hosts["track"], self)
        self.review_layout.cards.update(table=self.table_view, review=self.review_view, track=self.track_view)
        for pane in self.review_layout.cards.values():
            pane.pack(fill="both", expand=True)
        self.review_layout.load()
        self.save_status.set("复核自动保存")

    def _build_library(self):
        row = ttk.Frame(self.library)
        row.pack(fill="x", pady=(0, 14))
        ttk.Label(row, text="视频库", style="Section.TLabel").pack(side="left")
        self.count_label = ttk.Label(row, style="Muted.TLabel")
        self.count_label.pack(side="right")
        entry = ttk.Entry(self.library, textvariable=self.search)
        entry.pack(fill="x", pady=(0, 9))
        self.search.trace_add("write", lambda *_: self._filter_library())
        combo = ttk.Combobox(self.library, textvariable=self.role_filter, values=["全部视频", *ROLES.values()], state="readonly", width=16)
        if not self.portable:
            combo.pack(fill="x", pady=(0, 12))
        combo.bind("<<ComboboxSelected>>", lambda _: self._filter_library())
        table = ttk.Frame(self.library)
        table.pack(fill="both", expand=True)
        self.video_list = ttk.Treeview(table, columns=("role",), show="tree headings", selectmode="browse")
        self.video_list.heading("#0", text="视频编号")
        self.video_list.heading("role", text="用途")
        self.video_list.column("#0", width=97, minwidth=70)
        self.video_list.column("role", width=74, minwidth=58, stretch=False)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.video_list.yview)
        self.video_list.configure(yscrollcommand=scroll.set)
        self.video_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.video_list.bind("<<TreeviewSelect>>", self._library_selected)
        ttk.Button(self.library, text="导入文件夹", command=self.import_folder).pack(fill="x", pady=(12, 6))
        ttk.Label(self.library, text="搜索编号或文件名\n单击打开 · 视频支持独立导入", style="Muted.TLabel", font=("Microsoft YaHei UI", 9)).pack(anchor="w")

    def _build_player(self):
        heading = ttk.Frame(self.center)
        heading.pack(fill="x", pady=(0, 9 if self.table_demo else 12))
        self.video_title = ttk.Label(heading, text="原始视频" if self.table_demo else "选择一段视频", style="Section.TLabel" if self.table_demo else "Title.TLabel")
        self.video_meta = ttk.Label(heading, text="" if self.table_demo else "逐帧观察 · 事件标注 · 参考结果叠加",
                                    style="VideoMeta.TLabel" if self.table_demo else "Muted.TLabel")
        if self.table_demo:
            self.video_title.pack(side="left", padx=(0, 10))
            self.video_meta.pack(side="left")
        else:
            self.video_title.pack(anchor="w")
            self.video_meta.pack(anchor="w", pady=(5, 0))
        self.canvas = tk.Canvas(self.center, bg=review_theme.CANVAS if self.table_demo else VIDEO,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._resize_player)
        self.canvas.bind("<Button-1>", self._canvas_click)
        if self.table_demo:
            self.canvas.bind("<ButtonPress-3>", self._begin_pan)
            self.canvas.bind("<B3-Motion>", self._move_pan)
            self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        else:
            self.canvas.bind("<Button-3>", self._zoom_center)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.progress = ttk.Progressbar(self.center, mode="indeterminate")
        if not self.table_demo:
            self.progress.pack(fill="x")
        if self.table_demo:
            frame_bar = ttk.Frame(self.center, style="PlayerStatus.TFrame", padding=(10, 4))
            frame_bar.pack(fill="x", pady=(8, 0))
            self.frame_number = ttk.Label(frame_bar, text="f—", width=3, style="FrameNumber.TLabel")
            self.frame_number.pack(side="left", padx=(0, 7))
            self.frame_info = ttk.Label(frame_bar, text="/ f—  ·  PTS —", style="FrameInfo.TLabel")
            self.frame_info.pack(side="left")
            self.frame_event_info = ttk.Label(frame_bar, style="FrameEvent.TLabel", width=1, anchor="e")
            self.frame_event_info.pack(side="right", fill="x", expand=True, padx=(8, 0))
        else:
            self.frame_info = ttk.Label(self.center, text="0 基帧号 · 坐标以原始视频像素保存", style="Muted.TLabel")
            self.frame_info.pack(fill="x", pady=(8, 0))
        self.timeline = tk.Canvas(self.center, bg=review_theme.CANVAS if self.table_demo else WHITE,
                                   height=52, highlightthickness=0, cursor="hand2")
        self.timeline.pack(fill="x", pady=(0, 5) if self.table_demo else (4, 2))
        self.timeline.bind("<Configure>", lambda _: self.draw_timeline())
        self.timeline.bind("<Button-1>", self._timeline_click)
        self.timeline.bind("<B1-Motion>", lambda event: self._timeline_click(event, scrub=True))
        if self.table_demo:
            self.timeline.bind("<Motion>", lambda event: self._timeline_hover(event.x if event.y < 25 else None))
            self.timeline.bind("<Leave>", lambda _: self._timeline_hover(None))
        controls = ttk.Frame(self.center)
        controls.pack(fill="x", pady=(2, 6 if self.table_demo else 9))
        before = (("‹  上一帧  A", -1),) if self.table_demo else (("−10", -10), ("上一帧", -1))
        after = (("下一帧  D  ›", 1),) if self.table_demo else (("下一帧", 1), ("＋10", 10))
        for label, delta in before:
            ttk.Button(controls, text=label, style="Small.TButton", command=lambda n=delta: self.step(n)).pack(side="left", padx=(0, 5))
        play_options = {"image": self.review_button_images["播放"], "compound": "left"} if self.table_demo else {}
        self.play_button = ttk.Button(controls, text="播放", width=6, style="Accent.TButton", command=self.toggle_play, **play_options)
        self.play_button.pack(side="left", padx=(0, 5))
        for label, delta in after:
            ttk.Button(controls, text=label, style="Small.TButton", command=lambda n=delta: self.step(n)).pack(side="left", padx=(0, 5))
        speed = ttk.Combobox(controls, textvariable=self.speed, values=("0.1×", "0.25×", "0.5×", "1×", "2×"), width=5, state="readonly")
        speed.pack(side="right")
        if self.table_demo:
            speed.bind("<<ComboboxSelected>>", lambda _: self.canvas.focus_set())
            ttk.Label(controls, text="倍速", style="Muted.TLabel").pack(side="right", padx=(0, 5))
        secondary = ttk.Frame(self.center)
        secondary.pack(fill="x")
        ttk.Label(secondary, text="跳至帧", style="Muted.TLabel").pack(side="left", padx=(0, 7))
        jump = ttk.Entry(secondary, textvariable=self.jump, width=7)
        jump.pack(side="left")
        jump.bind("<Return>", lambda _: self.jump_to())
        utility_style = "Quiet.TButton" if self.table_demo else "Small.TButton"
        ttk.Button(secondary, text="前往", style=utility_style, command=self.jump_to).pack(side="left", padx=6)
        ttk.Button(secondary, text="截图", style=utility_style, command=self.snapshot).pack(side="right")
        zoom = ttk.Combobox(secondary, textvariable=self.zoom, values=REVIEW_ZOOMS if self.table_demo else ("适应", "2×", "4×", "8×"),
                            width=6, state="readonly")
        zoom.pack(side="right", padx=6)
        zoom.bind("<<ComboboxSelected>>", lambda _: self._set_zoom())
        if self.table_demo:
            ttk.Button(secondary, text="适应画面", style=utility_style, command=self._fit_review_video).pack(side="right")
        else:
            ttk.Label(secondary, text="放大", style="Muted.TLabel").pack(side="right")
        help_text = ("空格 播放    A/D 或 ←/→ 逐帧    滚轮缩放    右键拖动"
                     if self.table_demo else "Space 播放    ← → 逐帧    滚轮调帧    右键设置放大中心")
        ttk.Label(self.center, text=help_text, style="Muted.TLabel", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(6 if self.table_demo else 12, 0))

    def _build_inspector(self):
        ttk.Label(self.inspector, text="工作面板", style="Section.TLabel").pack(anchor="w", pady=(0, 13))
        self.notebook = ttk.Notebook(self.inspector)
        self.notebook.pack(fill="both", expand=True)
        note = ttk.Frame(self.notebook, padding=(8, 14))
        records = ttk.Frame(self.notebook, padding=(8, 14))
        layers = ttk.Frame(self.notebook, padding=(8, 14))
        self.notebook.add(note, text="标注")
        self.notebook.add(records, text="记录")
        self.notebook.add(layers, text="图层")
        if self.portable:
            self.notebook.hide(note)
            self.notebook.hide(records)
            self.notebook.tab(layers, text="分析结果")
        self.note_hint = ttk.Label(note, text="切换到标注模式后开始记录。\n观察模式下点击画面不会写入标注。", style="Muted.TLabel", wraplength=260)
        self.note_hint.pack(anchor="w", pady=(0, 16))
        ttk.Label(note, text="事件类型", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        grid = ttk.Frame(note)
        grid.pack(fill="x")
        self.edit_widgets = []
        for i, name in enumerate(TYPES):
            b = ttk.Radiobutton(grid, text=name, variable=self.event_type, value=name, command=self._type_changed)
            b.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 4), pady=2)
            self.edit_widgets.append(b)
        ttk.Label(note, text="在画面上单击，选择事件位置", style="Muted.TLabel", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(12, 5))
        ttk.Label(note, textvariable=self.point_status, foreground=ACCENT).pack(anchor="w", pady=(0, 12))
        ttk.Label(note, text="备注 / 判断依据").pack(anchor="w", pady=(0, 6))
        reason = ttk.Entry(note, textvariable=self.reason)
        reason.pack(fill="x")
        self.edit_widgets.append(reason)
        ttk.Label(note, text="备注可选；疑难区间仍需填写说明", style="Muted.TLabel", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(5, 12))
        add = ttk.Button(note, text="＋ 添加当前事件  Enter", style="Accent.TButton", command=self.add_event)
        add.pack(fill="x")
        self.edit_widgets.append(add)
        ttk.Button(note, text="清除选点  Esc", command=self.clear_draft, style="Small.TButton").pack(fill="x", pady=(6, 16))
        ttk.Separator(note).pack(fill="x", pady=(0, 14))
        ttk.Label(note, text="疑难区间", style="Section.TLabel").pack(anchor="w")
        ttk.Label(note, textvariable=self.interval_status, style="Muted.TLabel").pack(anchor="w", pady=(8, 8))
        row = ttk.Frame(note)
        row.pack(fill="x")
        for text, command in (("设起点  I", self.start_interval), ("设终点并记录  O", self.end_interval)):
            b = ttk.Button(row, text=text, command=command, style="Small.TButton")
            b.pack(side="left", padx=(0, 5))
            self.edit_widgets.append(b)
        ttk.Label(note, text="遮挡、无法确定触台时，用区间记录。", style="Muted.TLabel", wraplength=260, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(8, 2))
        ttk.Checkbutton(note, text="区间内确实发生了触台（帧间接触）", variable=self.interval_touch).pack(anchor="w", pady=(0, 12))
        ttk.Separator(note).pack(fill="x", pady=(0, 14))
        complete = ttk.Button(note, text="标记整段人工核查完成", command=self.complete)
        complete.pack(fill="x")
        self.edit_widgets.append(complete)
        ttk.Label(note, text="播放结束不会自动标记核查完成。", style="Muted.TLabel", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(8, 0))
        self.record_count = ttk.Label(records, text="本次标注 · 0 条", style="Section.TLabel")
        self.record_count.pack(anchor="w", pady=(0, 10))
        self.record_list = ttk.Treeview(records, columns=("kind", "frame"), show="headings", selectmode="browse", height=10)
        self.record_list.heading("kind", text="事件")
        self.record_list.heading("frame", text="帧 / 区间")
        self.record_list.column("kind", width=90, minwidth=70)
        self.record_list.column("frame", width=100, minwidth=80)
        self.record_list.pack(fill="both", expand=True)
        self.record_list.bind("<<TreeviewSelect>>", self._record_selected)
        self.record_detail = ttk.Label(records, text="选中记录定位到对应帧。", style="Muted.TLabel", wraplength=240)
        self.record_detail.pack(fill="x", pady=12)
        for text, command in (("修改备注", self.edit_reason), ("删除选中记录", self.delete_record)):
            b = ttk.Button(records, text=text, command=command)
            b.pack(fill="x", pady=(0, 6))
            self.edit_widgets.append(b)
        undo_row = ttk.Frame(records)
        undo_row.pack(fill="x", pady=(5, 0))
        for text, command in (("撤销 Ctrl+Z", self.undo), ("重做 Ctrl+Y", self.redo)):
            b = ttk.Button(undo_row, text=text, command=command, style="Small.TButton")
            b.pack(side="left", padx=(0, 5))
            self.edit_widgets.append(b)
        ttk.Label(layers, text="显示内容", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        names = {"own": "工作台标注 · 青绿", "history": "此前人工标注 · 紫色", "official": "官方触台 GT · 绿色", "trajectory": "球轨迹 · 蓝色", "imported": "导入参考 · 橙色", "boxes": "原始目标框 · 黄色"}
        self.layer_buttons = {}
        for key, text in names.items():
            if self.portable and key not in ("trajectory", "imported"):
                continue
            button = ttk.Checkbutton(layers, text=text, variable=self.layers[key], command=self._layers_changed)
            button.pack(anchor="w", fill="x")
            self.layer_buttons[key] = button
        row = ttk.Frame(layers)
        row.pack(fill="x", pady=(8, 5))
        ttk.Label(row, text="球心半径").pack(side="left")
        radius = ttk.Combobox(row, textvariable=self.point_radius, values=("1", "2", "3", "4"), width=3, state="readonly")
        radius.pack(side="right")
        radius.bind("<<ComboboxSelected>>", lambda _: self.render())
        row = ttk.Frame(layers)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="事件点位").pack(side="left")
        scope = ttk.Combobox(row, textvariable=self.event_scope, values=("当前帧", "整段位置"), width=9, state="readonly")
        scope.pack(side="right")
        scope.bind("<<ComboboxSelected>>", lambda _: self.render())
        ttk.Label(layers, text="轨迹显示此前24帧，缺失处断线。\n整段位置：非当前帧为空心点并标帧号。", style="Muted.TLabel", wraplength=245, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(4, 8))
        if not self.portable:
            ttk.Button(layers, text="导入标签 / 预测 / 轨迹", command=self.import_reference).pack(fill="x")
        self.layer_info = ttk.Label(layers, text="打开视频后显示可用参考。", style="Muted.TLabel", wraplength=245)
        self.layer_info.pack(anchor="w", pady=8)
        ttk.Label(layers, text="事件定位 · 单击跳帧", style="Section.TLabel").pack(anchor="w", pady=(5, 6))
        self.reference_list = ttk.Treeview(layers, columns=("kind", "frame"), show="headings", selectmode="browse", height=4)
        self.reference_list.heading("kind", text="来源 / 类型")
        self.reference_list.heading("frame", text="帧 / 区间")
        self.reference_list.column("kind", width=145, minwidth=95)
        self.reference_list.column("frame", width=75, minwidth=60)
        scroll = ttk.Scrollbar(layers, orient="vertical", command=self.reference_list.yview)
        self.reference_list.configure(yscrollcommand=scroll.set)
        self.reference_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.reference_list.bind("<<TreeviewSelect>>", self._reference_selected)
        self._set_editing(False)

    def _bind_keys(self):
        bind = self.bind_all if self.table_demo else self.bind
        for key, fn in (("<space>", self.toggle_play), ("<Left>", lambda: self.step(-1)), ("<Right>", lambda: self.step(1)),
                        ("<Shift-Left>", lambda: self.step(-10)), ("<Shift-Right>", lambda: self.step(10))):
            bind(key, lambda event, f=fn: self._shortcut(event, f))
        if self.table_demo:
            for key, delta in (("a", -1), ("A", -1), ("d", 1), ("D", 1)):
                bind(f"<KeyPress-{key}>", lambda event, n=delta: self._shortcut(event, lambda: self.step(n)))
            for key, delta in (("j", -1), ("k", 1)):
                bind(f"<KeyPress-{key}>", lambda event, n=delta: self._shortcut(event, lambda: self.review_view.navigate(n)))
            for key, state in (("f", "confirmed"), ("x", "rejected"), ("q", "uncertain")):
                bind(f"<KeyPress-{key}>", lambda event, value=state: self._shortcut(event, lambda: self.review_view.decide(value)))
            bind("<Escape>", lambda event: self._shortcut(event, self._cancel_review_tool))
            return
        for key, fn in (("<Return>", self.add_event), ("<Escape>", self.clear_draft),
                        ("i", self.start_interval), ("o", self.end_interval)):
            self.bind(key, lambda event, f=fn: self._shortcut(event, f))
        for key, fn in (("<Control-s>", self.save), ("<Control-z>", self.undo), ("<Control-y>", self.redo)):
            self.bind(key, lambda event, f=fn: self._shortcut(event, f, global_key=True))

    def _cancel_review_tool(self):
        if self.review_view.adding:
            self.review_view.cancel_add()
        if self.table_view.pending is not None:
            self.table_view.toggle_marking()

    def _shortcut(self, event, fn, global_key=False):
        if not global_key and isinstance(self.focus_get(), (ttk.Entry, ttk.Combobox, tk.Entry, tk.Text)):
            return
        fn()
        return "break"

    def _load_preferences(self):
        path = self.output / "workspace.json"
        if not path.exists():
            return
        try:
            paths = json.loads(path.read_text(encoding="utf-8"))["imported_videos"]
            known = {e.path.resolve() for e in self.catalog}
            for item in paths:
                p = Path(item)
                if p.is_file() and p.resolve() not in known:
                    self.catalog.append(self._entry_from_path(p))
                    known.add(p.resolve())
        except (ValueError, OSError, KeyError) as e:
            logging.warning("Workspace preferences: %s", e)

    def _save_preferences(self):
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "workspace.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"imported_videos": [str(e.path) for e in self.catalog if e.role == "imported"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _filter_library(self):
        if not hasattr(self, "video_list"):
            return
        self.video_list.delete(*self.video_list.get_children())
        query = self.search.get().lower().strip()
        role = self.role_filter.get()
        for i, e in enumerate(self.catalog):
            if query and query not in (e.video_id + " " + e.path.name).lower():
                continue
            if role != "全部视频" and ROLES.get(e.role, "自行导入") != role:
                continue
            self.video_list.insert("", "end", iid=str(i), text=e.video_id, values=(ROLES.get(e.role, "自行导入"),))
        self.count_label.configure(text=f"{len(self.video_list.get_children())} 段")

    def _library_selected(self, _=None):
        selected = self.video_list.selection()
        if selected:
            entry = self.catalog[int(selected[0])]
            if self.entry is None or entry.path != self.entry.path:
                self.open_entry(entry)

    def import_videos(self):
        paths = filedialog.askopenfilenames(parent=self, title="导入视频", filetypes=VIDEO_TYPES)
        self._import_paths(paths)

    def import_folder(self):
        folder = filedialog.askdirectory(parent=self, title="选择视频文件夹（包含子文件夹）")
        if folder:
            paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"})
            self._import_paths(paths)

    def _import_paths(self, paths):
        known = {e.path.resolve(): e for e in self.catalog}
        first = None
        for p in paths:
            p = Path(p).resolve()
            entry = known.get(p)
            if entry is None:
                entry = self._entry_from_path(p)
                self.catalog.append(entry)
                known[p] = entry
            first = first or entry
        if first:
            self.search.set("")
            self.role_filter.set("全部视频")
            self._filter_library()
            self._save_preferences()
            self.open_entry(first)

    def _entry_from_path(self, path):
        return VideoEntry(path.stem, path) if self.portable or self.table_demo else entry_from_path(path)

    def open_entry(self, entry):
        if not self._release_session():
            return
        self.pause()
        self.generation += 1
        generation = self.generation
        if self.reader:
            self.executor.submit(self.reader.close)
        self.reader = None
        self.entry = entry
        self.cpu_result = None
        self.cpu_track_frames = []
        if self.portable:
            self.analyze_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
        self.image_bgr = None
        self.stream_preview = None
        self.loading = True
        self.reading = False
        self.frame = self.target_frame = 0
        self.zoom.set("适应")
        self.roi = None
        self.pan_start = None
        self.clear_draft()
        self.interval_start = None
        self.interval_status.set("未设置区间起点")
        self.references = {"official": [], "trajectory": [], "history": [], "imported": []}
        self.saved_own = None
        self.reference_names = {}
        self.boxes = {}
        self._set_editing(False)
        self.refresh_records()
        title = self.table_demo.get("title", entry.video_id) if self.table_demo else entry.video_id
        self.video_title.configure(text=f"原片 · {title}" if self.table_demo else title)
        role = "" if self.table_demo else f"{ROLES.get(entry.role, '自行导入')}  ·  "
        self.video_meta.configure(text=f"{role}正在核对视频…")
        self.status.set("正在打开原始视频…" if self.portable else "正在顺序解码并核对帧数，首次打开请稍候…")
        if self.table_demo:
            self.progress.pack(fill="x", after=self.canvas)
        self.progress.start(45 if self.table_demo else 12)
        self.render()
        def load():
            if self.portable:
                from source_video import PreviewVideo
                reader = PreviewVideo(entry.path)
            else:
                reader = SequentialVideo(entry.path, seek_preroll=2 if self.table_demo else None)
            try:
                image, pts = reader.get(0)
                return reader, image, pts
            except Exception:
                reader.close()
                raise
        self.executor.submit(load).add_done_callback(lambda f: self.results.put(("open", generation, f)))

    def analyze_current(self):
        if not self.entry or not self.reader or self.loading:
            return
        if self.portable:
            self.analyze_cpu()
            return
        from datetime import datetime, timezone
        from video_only_entry import run_process
        generation = self.generation
        video = self.entry.path
        output = ROOT/'outputs/ball_default_video_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.analyze_button.configure(state='disabled')
        self.status.set('第三轮球点与触台分析中，CPU处理需要数分钟；可以继续观察视频。')
        def analyze():
            run_process(video, output, progress=lambda _:None)
            return output
        self.analysis_executor.submit(analyze).add_done_callback(lambda f:self.results.put(('analysis', generation, f)))

    def analyze_cpu(self):
        if self.cpu_busy:
            return
        generation = self.generation
        worker = Path(__file__).with_name("local_analysis.py")
        identifier = str(uuid4())
        log = self.output / "analysis" / "jobs" / (identifier + ".log")
        executable = Path(sys.executable)
        if sys.platform == "win32":
            executable = executable.with_name("python.exe")
        args = [str(executable), "-E", "-s", "-B", "-X", "utf8", "-u", str(worker),
                "--video", str(self.entry.path), "--workspace", str(self.output),
                "--video-id", self.entry.video_id, "--run-id", identifier, "--sha256", self.reader.sha256]
        self.cpu_busy = True
        self.analyze_button.configure(state="disabled")
        self.status.set("正在启动本地 CPU 分析…")

        def analyze():
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as output, subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ) as process:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    if line.startswith("正在"):
                        self.results.put(("cpu_status", generation, line.strip()))
                if process.wait():
                    raise RuntimeError(f"本地分析失败。诊断日志：\n{log}")

        self.analysis_executor.submit(analyze).add_done_callback(
            lambda future: self.results.put(("cpu_analysis", generation, future)))

    def _load_cpu_results(self):
        from core import _read
        from portable_results import latest, rows_for_video
        self.cpu_result = None
        self.cpu_track_frames = []
        self.references["imported"] = []
        self.references["trajectory"] = []
        self.export_button.configure(state="disabled")
        result = latest(self.output, self.reader.sha256)
        if result is not None:
            folder, _ = result
            rows = rows_for_video(result, self.entry.video_id, self.reader.width,
                                  self.reader.height, self.reader.n_frames)
            self.references["imported"] = [dict(row, event_type="table_bounce", prediction_only=True) for row in rows]
            self.references["trajectory"] = _read(folder / "tracks.json")
            self.cpu_track_frames = [row["frame_id"] for row in self.references["trajectory"]]
            self.reference_names["trajectory"] = "CPU 分析球点"
            self.cpu_result = result
            self.layers["imported"].set(True)
            self.export_button.configure(state="normal")
        self._refresh_layer_summary()

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, generation, future = self.results.get_nowait()
                if kind == "cpu_status":
                    if generation == self.generation:
                        self.status.set(future)
                    continue
                if kind == "cpu_analysis":
                    self.cpu_busy = False
                    if self.reader is not None:
                        self.analyze_button.configure(state="normal")
                    try:
                        future.result()
                        if generation == self.generation:
                            self._load_cpu_results()
                            self.render()
                            self.status.set(f"分析完成 · {self.cpu_result[1]['event_count']} 个预测落点")
                    except Exception as e:
                        self.error(str(e))
                    continue
                if kind == 'analysis':
                    self.analyze_button.configure(state='normal')
                    try:
                        output = future.result()
                        if generation == self.generation:
                            self._auto_references()
                            self.render()
                            self.status.set('第三轮分析完成：球轨迹与触台预测已载入，预测不等于人工标注。')
                    except Exception as e:
                        self.error(str(e))
                    continue
                if generation != self.generation:
                    if kind == "open" and not future.exception():
                        self.executor.submit(future.result()[0].close)
                    continue
                try:
                    result = future.result()
                    if kind == "open":
                        self.reader, self.image_bgr, self.pts = result
                        self.loading = False
                        self.progress.stop()
                        if self.table_demo:
                            self.progress.pack_forget()
                        if self.table_view is not None:
                            from core import _read
                            result_path = ROOT / self.table_demo["result"]
                            self.review_tracks = _read(result_path / "tracks.json")
                            self.review_observations = {row["frame_id"]: row for row in self.review_tracks}
                            self.review_times = {row["frame_id"]: row["timestamp_s"] for row in self.review_tracks}
                            self.table_view.load(self.reader)
                            self.review_view.load(self.reader, result_path, self.table_view.predictions)
                            self.layers["imported"].set(True)
                            self.event_scope.set("当前帧")
                            self.refresh_reviews()
                        elif self.portable:
                            self._load_cpu_results()
                            self.analyze_button.configure(state="disabled" if self.cpu_busy else "normal")
                        else:
                            self._auto_references()
                        r = self.reader
                        if self.table_demo:
                            self.frame_number.configure(width=len(str(self.source_frame_id(r.n_frames - 1))) + 1)
                        role = "" if self.table_demo else f"{ROLES.get(self.entry.role, '自行导入')}  ·  "
                        self.video_meta.configure(text=f"{role}{r.width} × {r.height}  ·  {r.n_frames} 帧  ·  {r.fps:.2f} fps")
                        if self.source_frame_offset:
                            self.status.set(f"视频就绪 · 原片 f{self.source_frame_offset}–f{self.source_frame_id(r.n_frames - 1)}")
                        else:
                            self.status.set("视频就绪 · 帧号从 0 开始")
                        if self.table_view is not None:
                            self.seek(self.table_view.frame_id)
                        else:
                            self._mode_changed()
                    else:
                        self.reading = False
                        if self.table_demo and result[0] != self.target_frame:
                            continue
                        self.frame, self.image_bgr, self.pts = result
                        self.stream_preview = None
                    self.jump.set(str(self.source_frame_id(self.frame)))
                    if self.store:
                        self.store.state["last_frame"] = self.frame
                    self.render()
                except Exception as e:
                    self.loading = self.reading = False
                    self.target_frame = self.frame
                    self.progress.stop()
                    if self.table_demo:
                        self.progress.pack_forget()
                    self.pause()
                    self.error(str(e))
        except queue.Empty:
            pass
        if self.reader and not self.loading and not self.reading:
            now = time.monotonic()
            if self.target_frame != self.frame:
                self._request_frame()
            elif self.playing:
                if self.table_demo:
                    try:
                        self._advance_review_playback(now)
                    except Exception as error:
                        self.pause()
                        self.error(str(error))
                elif now >= self.play_at:
                    if self.frame >= self.reader.n_frames - 1 or self.preview_end is not None and self.frame >= self.preview_end:
                        self.pause()
                    else:
                        self.target_frame = self.frame + 1
                        self.play_at = now + self._frame_delay()
                        self._request_frame()
        polling_ms = 10 if self.portable else 15
        if self.table_demo and (self.playing or self.reading):
            polling_ms = 5
            if self.review_playback is not None and self.play_at is not None:
                remaining = ceil((self.play_at - time.monotonic()) * 1000)
                if remaining > 0 or not self.review_playback.frames.empty():
                    polling_ms = max(1, min(20, remaining))
        self.poll_job = self.after(polling_ms, self._poll)

    def _frame_delay(self):
        times = self.reader.timestamps
        if self.portable:
            before, after = times.get(self.frame), times.get(self.frame + 1)
        else:
            before, after = times[self.frame], times[self.frame + 1]
        interval = (after - before) if before is not None and after is not None and after > before else 1 / self.reader.fps
        return interval / float(self.speed.get().rstrip("×"))

    def _advance_review_playback(self, now):
        last = self.preview_end if self.preview_end is not None else self.reader.n_frames - 1
        if self.frame >= last:
            self.pause()
            return
        if self.review_playback is None:
            bounds = self._video_geometry(self.canvas.winfo_width(), self.canvas.winfo_height())[2:]
            self.review_playback = ReviewPlayback(self.executor, self.reader, self.frame + 1, last, bounds, self._display_pixels)
            self.play_at = None
        stream = self.review_playback
        if stream.task.done():
            stream.task.result()
        if self.play_at is None:
            if stream.frames.qsize() < stream.startup_frames:
                return
            stream.start_clock()
            self.play_at = now
        if now < self.play_at:
            return
        try:
            frame = stream.frames.get_nowait()
        except queue.Empty:
            return
        self.frame, self.image_bgr, self.pts, self.stream_preview = frame
        self.target_frame = self.frame
        if self.frame < last:
            interval = self._frame_delay()
            self.play_at = max(self.play_at + interval, now + interval / 4)
        self.render()
        if self.frame == last:
            self.pause()

    def _request_frame(self):
        self.reading = True
        fid, reader, generation = self.target_frame, self.reader, self.generation

        def read():
            image, pts = reader.get(fid)
            return fid, image, pts

        self.executor.submit(read).add_done_callback(lambda f: self.results.put(("frame", generation, f)))

    def seek(self, frame):
        if self.reader and not self.loading:
            if self.review_playback is not None:
                self.pause()
            self.target_frame = max(0, min(int(frame), self.reader.n_frames - 1))
            self.clear_draft()
            if self.table_demo and not self.reading and self.target_frame != self.frame:
                if self.render_job is not None:
                    self.after_cancel(self.render_job)
                    self.render_job = None
                self._request_frame()

    def step(self, delta):
        self.pause()
        self.seek(self.target_frame + delta)

    def source_frame_id(self, frame):
        return self.source_frame_offset + frame

    def jump_to(self):
        try:
            n = int(self.jump.get())
        except ValueError:
            self.status.set("请输入整数帧号")
            return
        self.pause()
        self.seek(n - self.source_frame_offset)
        self.canvas.focus_set()

    def toggle_play(self):
        if not self.reader or self.loading:
            return
        if self.playing:
            self.pause()
        else:
            resized = self.table_demo and self.preview_window_size != (self.winfo_width(), self.winfo_height())
            if self.frame == self.reader.n_frames - 1:
                self.seek(0)
            self.clear_draft()
            self.playing = True
            self.play_at = time.monotonic()
            self._set_play_button("暂停")
            if self.table_demo and not self.reading and self.target_frame == self.frame:
                self._advance_review_playback(self.play_at)
            if resized:
                # Finish window layout while the decode worker fills the playback buffer.
                self.update_idletasks()

    def pause(self):
        was_playing = self.playing
        self.playing = False
        if self.review_playback is not None:
            self.review_playback.cancel()
            self.review_playback = None
            self.stream_preview = None
        self.preview_end = None
        if self.table_demo and was_playing:
            self.target_frame = self.frame
            self.jump.set(str(self.source_frame_id(self.frame)))
        if hasattr(self, "play_button"):
            self._set_play_button("播放")
        if self.table_demo and was_playing:
            self._queue_render()

    def _set_play_button(self, caption):
        if self.play_button.cget("text") != caption:
            options = {"image": self.review_button_images[caption]} if self.table_demo else {}
            self.play_button.configure(text=caption, **options)

    def preview_event(self, frame):
        if not self.reader or self.loading:
            return
        self.pause()
        timestamps = self.reader.timestamps
        stamp = timestamps[frame]
        if stamp is None:
            radius = round(self.reader.fps * .25)
            start, end = max(0, frame - radius), min(self.reader.n_frames - 1, frame + radius)
        else:
            start = bisect_left(timestamps, stamp - .25)
            end = bisect_right(timestamps, stamp + .25) - 1
        self.seek(start)
        self.preview_end = end
        self.playing = True
        self.play_at = time.monotonic()
        self._set_play_button("暂停")

    def _wheel(self, event):
        if not event.delta:
            return
        if not self.table_demo or event.state & 1:
            self.step(-1 if event.delta > 0 else 1)
            return "break"
        point = self._to_original(event.x, event.y)
        if point is None:
            return
        zoom = REVIEW_ZOOMS.index(self.zoom.get())
        offset = max(1, abs(event.delta) // 120) * (1 if event.delta > 0 else -1)
        next_zoom = max(0, min(len(REVIEW_ZOOMS) - 1, zoom + offset))
        if next_zoom != zoom:
            left, top, dw, dh = self.transform[:4]
            self.zoom.set(REVIEW_ZOOMS[next_zoom])
            self._set_zoom(point, anchor=((event.x - left) / dw, (event.y - top) / dh))
        return "break"

    def _timeline_click(self, event, *, scrub=False):
        if not self.reader or self.loading:
            return
        if self.table_demo:
            self.canvas.focus_set()
            event_id = self._timeline_marker_at(event.x) if not scrub and event.y < 25 else None
            if event_id is not None:
                self.review_view.select_and_go(event_id)
                return
        ratio = max(0, min(1, (event.x - 10) / max(1, self.timeline.winfo_width() - 20)))
        stamps = self.reader.timestamps if self.table_demo else ()
        if len(stamps) > 1 and stamps[0] is not None:
            stamp = stamps[0] + ratio * (stamps[-1] - stamps[0])
            after = min(bisect_left(stamps, stamp), len(stamps) - 1)
            before = max(0, after - 1)
            frame = before if stamp - stamps[before] < stamps[after] - stamp else after
        else:
            frame = round(ratio * (self.reader.n_frames - 1))
        self.pause()
        self.seek(frame)

    def _timeline_marker_at(self, cursor_x):
        if cursor_x is None or not self.timeline_markers:
            return None
        selected = self.review_view.selected_id
        x, key = min(self.timeline_markers, key=lambda marker: (abs(marker[0] - cursor_x), marker[1] != selected))
        return key if abs(x - cursor_x) <= 8 else None

    def _timeline_hover(self, x):
        from review_view import STATE_COLORS, STATE_NAMES

        self.timeline_hover_x = x
        self.timeline.delete("hover")
        key = self._timeline_marker_at(x)
        if key is not None:
            row = self.review_view.rows[key]
            frame_id = self.source_frame_id(row["frame_id"])
            self.timeline.create_text(self.timeline.winfo_width() / 2, 44, tags="hover",
                                      text=f"f{frame_id}  ·  {STATE_NAMES[row['status']]}",
                                      fill=STATE_COLORS[row["status"]], font=("Microsoft YaHei UI", 9))

    def _fit_review_video(self):
        self.zoom.set("适应")
        self._set_zoom()

    def _set_zoom(self, center=None, anchor=(.5, .5)):
        if not self.reader:
            return
        if self.pan_start is not None:
            self._end_pan(None)
        factor = 1 if self.zoom.get() == "适应" else float(self.zoom.get().rstrip("×"))
        w, h = self.reader.width, self.reader.height
        if factor == 1:
            self.roi = None
        else:
            cx, cy = center or (self.draft[1:] if self.draft else ((self.roi[0] + self.roi[2] / 2, self.roi[1] + self.roi[3] / 2) if self.roi else (w / 2, h / 2)))
            rw, rh = max(1, int(w / factor)), max(1, int(h / factor))
            self.roi = self._bounded_roi(cx - rw * anchor[0], cy - rh * anchor[1], rw, rh)
        if self.table_demo:
            self.canvas.focus_set()
        self.render()

    def _bounded_roi(self, x, y, w, h):
        return (int(max(0, min(self.reader.width - w, x))),
                int(max(0, min(self.reader.height - h, y))), w, h)

    def _begin_pan(self, event):
        if self.roi is None or self._to_original(event.x, event.y) is None:
            return
        self.pan_start = (event.x, event.y, *self.roi, self.transform[2], self.transform[3])
        self.canvas.focus_set()
        self.canvas.configure(cursor="fleur")

    def _move_pan(self, event):
        if self.pan_start is None:
            return
        x, y, ox, oy, w, h, dw, dh = self.pan_start
        roi = self._bounded_roi(ox - (event.x - x) * w / dw, oy - (event.y - y) * h / dh, w, h)
        if roi != self.roi:
            self.roi = roi
            self._queue_render()

    def _end_pan(self, _event):
        self.pan_start = None
        self.canvas.configure(cursor="crosshair")
        if _event is not None:
            self.render()

    def _zoom_center(self, event):
        point = self._to_original(event.x, event.y)
        if point:
            if self.zoom.get() == "适应":
                self.zoom.set("2×")
            self._set_zoom(point)

    def _to_original(self, x, y):
        if not self.transform:
            return None
        left, top, dw, dh, ox, oy, rw, rh = self.transform
        if not (left <= x < left + dw and top <= y < top + dh):
            return None
        return ox + (x - left) * rw / dw, oy + (y - top) * rh / dh

    def _display_point(self, x, y):
        if not self.transform:
            return None
        left, top, dw, dh, ox, oy, rw, rh = self.transform
        if not (ox <= x < ox + rw and oy <= y < oy + rh):
            return None
        return left + (x - ox) * dw / rw, top + (y - oy) * dh / rh

    def _resize_player(self, event):
        if not self.table_demo:
            self._queue_render()
        elif self.review_playback is not None:
            self.review_playback.bounds = self._video_geometry(event.width, event.height)[2:]
        elif not self.closed and self.review_playback is None:
            if self.render_job is not None:
                self.after_cancel(self.render_job)
            self.render_job = self.after(45, self._render_queued)

    def _queue_render(self):
        if not self.closed and self.render_job is None:
            self.render_job = self.after(16, self._render_queued)

    def _render_queued(self):
        self.render_job = None
        self.render()

    def _video_geometry(self, width, height):
        source_height, source_width = self.image_bgr.shape[:2]
        ox, oy, rw, rh = self.roi or (0, 0, source_width, source_height)
        scale = min(width / rw, height / rh)
        dw, dh = max(1, round(rw * scale)), max(1, round(rh * scale))
        return (width - dw) // 2, (height - dh) // 2, dw, dh, ox, oy, rw, rh

    def render(self):
        if not hasattr(self, "canvas"):
            return
        if self.render_job is not None:
            self.after_cancel(self.render_job)
            self.render_job = None
        canvas = self.canvas
        if self.table_view is not None:
            self.table_view.show_frame(self.frame)
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if self.image_bgr is None:
            canvas.delete("all")
            self.picture_item = self.picture_key = self.photo = None
            self.transform = None
            if self.table_demo:
                self.frame_number.configure(text="f—")
                self.frame_info.configure(text="/ f—  ·  PTS —")
                self.frame_event_info.configure(text="")
                self.displayed_event = None
                label = "正在读取原片…" if self.loading else "等待载入样片"
                canvas.create_text(cw / 2, ch / 2, text=label, fill=review_theme.MUTED, font=("Microsoft YaHei UI", 12))
            else:
                canvas.create_text(cw / 2, ch / 2 - 22, text="PINGPONG", fill="#8db1be", font=("Segoe UI", 22, "bold"))
                canvas.create_text(cw / 2, ch / 2 + 17, text="正在准备视频…" if self.loading else "从左侧选择视频，或导入本地文件", fill="#8ea3b7", font=("Microsoft YaHei UI", 11))
            self.draw_timeline()
            return
        self.transform = self._video_geometry(cw, ch)
        if self.table_demo and not self.playing:
            self.preview_window_size = (self.winfo_width(), self.winfo_height())
        _, _, _, _, ox, oy, rw, rh = self.transform
        if self.review_playback is not None:
            self.review_playback.bounds = self.transform[2:]
        canvas.delete("!video_frame")
        fast = bool(self.table_demo and (self.playing or self.pan_start is not None))
        picture_key = self.generation, self.frame, self.transform, fast
        if picture_key != self.picture_key:
            self._update_picture(fast)
            self.picture_key = picture_key
        if self.table_view is not None and self.review_layout.visible("table") and (
                self.show_table_outline.get() or self.table_view.pending is not None):
            self.table_view.draw_source(canvas, self._display_point, self.frame)
        if self.review_layout is not None and self.review_layout.visible("track"):
            self.track_view.show_frame(self.frame)
        if self.review_view is not None and self.show_time_tail.get() and hasattr(self, "review_tracks"):
            self._draw_time_tail()
        if self.layers["boxes"].get():
            for item in self.boxes.get(self.frame, []):
                for rect in item.get("bbox_xyxy", []):
                    if len(rect) != 4:
                        continue
                    x1, y1, x2, y2 = rect
                    x1, y1, x2, y2 = max(ox, x1), max(oy, y1), min(ox + rw - 1, x2), min(oy + rh - 1, y2)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    a, b = self._display_point(x1, y1), self._display_point(x2, y2)
                    if a and b:
                        canvas.create_rectangle(*a, *b, outline="#ffe394", width=1)
                        canvas.create_text(a[0] + 3, a[1] + 3, anchor="nw", text=item.get("label", ""), fill="#ffe394", font=("Segoe UI", 9))
        if self.layers["trajectory"].get():
            tracks = self.references["trajectory"]
            if self.portable:
                start = bisect_left(self.cpu_track_frames, max(0, self.frame - 24))
                stop = bisect_right(self.cpu_track_frames, self.frame)
                tracks = tracks[start:stop]
            for segment in trajectory_segments(tracks, self.frame):
                visible = []
                for _, x, y in segment:
                    point = self._display_point(x, y)
                    if point is None:
                        if len(visible) >= 4:
                            canvas.create_line(*visible, fill="#7fc9ff", width=1.4, tags=("trajectory",))
                        visible = []
                    else:
                        visible.extend(point)
                if len(visible) >= 4:
                    canvas.create_line(*visible, fill="#7fc9ff", width=1.4, tags=("trajectory",))
            for item in self.references["trajectory"]:
                if item.get("frame_id") == self.frame and item.get("detected", True) and item.get("x") is not None and item.get("y") is not None:
                    low = item.get('confidence') is not None and item['confidence'] < .75
                    reviewed = item.get('user_review_state') == 'user_generally_acceptable'
                    color = '#72e09b' if reviewed else '#a8a8a8' if low else '#7fc9ff'
                    label = f"低分 {item['confidence']:.3f}" if low else ''
                    disposition = item.get('case_disposition') or {}
                    if disposition.get('status') == 'deferred_source_near_duplicate_frames':
                        label += ' · 源画面卡顿，暂缓'
                    self._draw_point(item['x'], item['y'], color, label, hollow=low)
        frame_events = []
        for name, color in (("official", "#83f69b"), ("history", "#d3adff"), ("imported", "#ffbe6b"), ("own", "#32e2c4")):
            if not self.layers[name].get():
                continue
            rows = self._own_rows() if name == "own" else self.references[name]
            for item in rows:
                interval = interval_of(item)
                if interval is None:
                    continue
                current = interval[0] <= self.frame <= interval[1]
                label = ("GT" if name == "official" else TYPE_NAMES.get(item.get("event_type"), "疑难"))
                event_color = item.get("review_color", color)
                if item.get("review_label"):
                    label += " · " + item["review_label"]
                if current:
                    frame_events.append((label, event_color))
                if (current or self.event_scope.get() == "整段位置") and item.get("x") is not None and item.get("y") is not None:
                    valid = item.get("valid", True)
                    suffix = f" f{interval[0]}" if not current else ""
                    point_label = "" if self.table_demo and item.get("review_label") else label + suffix
                    self._draw_point(item["x"], item["y"], event_color if valid else "#a8a8a8", point_label, hollow=not valid or not current)
        if self.table_demo:
            text = " / ".join(dict.fromkeys(label for label, _ in frame_events))
            color = frame_events[0][1] if frame_events else review_theme.MUTED
            if self.displayed_event != (text, color):
                self.frame_event_info.configure(text=text, foreground=color)
                self.displayed_event = text, color
        elif frame_events:
            text = "当前帧事件：" + " / ".join(dict.fromkeys(label for label, _ in frame_events))
            canvas.create_rectangle(8, 8, min(cw - 8, 24 + len(text) * 13), 34, fill=VIDEO, outline="")
            canvas.create_text(16, 21, text=text, anchor="w", fill=frame_events[0][1], font=("Microsoft YaHei UI", 9))
        if self.draft and self.draft[0] == self.frame:
            self._draw_point(*self.draft[1:], "#ffde75", "待确认", cross=True)
        timestamp = (f"PTS {self.pts + self.source_time_offset:.3f} s" if self.pts is not None else
                     f"估算 {self.frame / max(1, self.reader.fps) + self.source_time_offset:.3f} s")
        if self.table_demo:
            self.frame_number.configure(text=f"f{self.source_frame_id(self.frame)}")
            end_frame = self.source_frame_id(self.reader.n_frames - 1)
            self.frame_info.configure(text=f"/ f{end_frame}  ·  {timestamp}  ·  {self.zoom.get()}")
        else:
            self.frame_info.configure(text=f"帧 {self.frame} / {self.reader.n_frames - 1}    ·    {timestamp}    ·    {self.zoom.get()}")
        self.draw_timeline()

    @staticmethod
    def _display_pixels(image, bounds, *, fast):
        dw, dh, ox, oy, rw, rh = bounds
        display = image[oy:oy + rh, ox:ox + rw]
        if (dw, dh) != (rw, rh):
            interpolation = cv2.INTER_AREA if not fast and dw < rw else cv2.INTER_LINEAR
            display = cv2.resize(display, (dw, dh), interpolation=interpolation)
        return Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    def _update_picture(self, fast):
        left, top, dw, dh = self.transform[:4]
        bounds = self.transform[2:]
        if fast and self.stream_preview is not None and self.stream_preview[0] == bounds:
            image = self.stream_preview[1]
        else:
            image = self._display_pixels(self.image_bgr, bounds, fast=fast)
        if self.photo is not None and (dw, dh) == (self.photo.width(), self.photo.height()):
            self.photo.paste(image)
        else:
            self.photo = ImageTk.PhotoImage(image, master=self)
            if self.picture_item is None:
                self.picture_item = self.canvas.create_image(left, top, image=self.photo, anchor="nw", tags="video_frame")
            else:
                self.canvas.itemconfigure(self.picture_item, image=self.photo)
        self.canvas.coords(self.picture_item, left, top)

    def _draw_time_tail(self):
        origin = self.reader.timestamps[0]
        now = self.pts - origin if self.pts is not None and origin is not None else self.review_times.get(self.frame)
        if now is None:
            return
        canvas = self.canvas
        paths = []
        path = style = head = None
        for point, previous in timed_tail(self.review_tracks, now, duration=TRAIL_SECONDS):
            stamp, fid, x, y = point
            target = self._display_point(x, y)
            if target is None:
                path = None
                continue
            freshness = min(1, max(0, 1 - (now - stamp) / TRAIL_SECONDS))
            stroke = TRAIL_STROKES[round(freshness * (len(TRAIL_STROKES) - 1))]
            start = self._display_point(previous[2], previous[3]) if previous is not None else None
            if start is None:
                path = None
            elif path is None or stroke != style:
                path = [*start, *target]
                paths.append((stroke, path))
                style = stroke
            else:
                path.extend(target)
            if fid == self.frame:
                head = target, start
            elif start is None:
                px, py = target
                canvas.create_oval(px - 1.3, py - 1.3, px + 1.3, py + 1.3, fill=stroke[0], outline="", tags="time_tail")

        # Keep every observed vertex but draw neighboring segments of the same style together.
        for (color, width), pixels in paths:
            canvas.create_line(*pixels, fill=color, width=width, joinstyle="round", capstyle="round", tags="time_tail")
        if head is not None:
            (px, py), start = head
            ring = (px - 7, py - 7, px + 7, py + 7)
            canvas.create_oval(*ring, outline="#64bac8", tags=("time_tail", "trail_head"))
            if start is not None:
                dx, dy = px - start[0], py - start[1]
                if hypot(dx, dy) >= 2:
                    heading = degrees(atan2(-dy, dx))
                    canvas.create_arc(*ring, start=heading - 55, extent=110, style="arc",
                                      outline="#d4faff", width=2, tags=("time_tail", "trail_direction"))
            canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#effdff", outline="#17343d",
                               tags=("time_tail", "trail_head"))

    def refresh_reviews(self):
        from review_view import STATE_COLORS, STATE_NAMES
        if self.review_view.path is None or self.table_view.plane is None:
            return
        self.table_view.set_events(self.review_view.events())
        self.review_view.show_rows(self.table_view.rows)
        self.table_view.select(self.review_view.selected_id)
        self.references["imported"] = [dict(
            row, event_type="table_bounce", valid=row["status"] != "rejected",
            review_color=STATE_COLORS[row["status"]], review_label=STATE_NAMES[row["status"]],
        ) for row in self.table_view.rows]
        self.render()

    def _draw_point(self, x, y, color, text="", hollow=False, cross=False):
        point = self._display_point(x, y)
        if point is None:
            return
        px, py = point
        radius = int(self.point_radius.get())
        self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, outline=color, width=1, fill="" if hollow or cross else color)
        if cross:
            self.canvas.create_line(px - 7, py, px - 3, py, fill=color)
            self.canvas.create_line(px + 3, py, px + 7, py, fill=color)
            self.canvas.create_line(px, py - 7, px, py - 3, fill=color)
            self.canvas.create_line(px, py + 3, px, py + 7, fill=color)
        if text:
            self.canvas.create_text(px + 8, py - 7, anchor="w", text=text, fill="#152330", font=("Microsoft YaHei UI", 8))
            self.canvas.create_text(px + 7, py - 8, anchor="w", text=text, fill=color, font=("Microsoft YaHei UI", 8))

    def draw_timeline(self, *, refresh=False):
        if not hasattr(self, "timeline"):
            return
        c = self.timeline
        w = max(20, c.winfo_width())
        if self.table_demo:
            reader = self.reader if not self.loading else None
            axis = reader, w
            if refresh or axis != self.timeline_axis:
                self._draw_review_timeline(c, w)
                self.timeline_axis = axis
            if reader is not None and self.review_view is not None:
                x = self._review_timeline_x(self.frame, w)
                if x != self.timeline_playhead:
                    c.coords("track_glow", 10, 28, x, 28)
                    c.coords("track_mask", x, 27, w - 10, 29)
                    c.move("playhead", x - self.timeline_playhead, 0)
                    self.timeline_playhead = x
            return
        c.delete("all")
        c.create_line(10, 27, w - 10, 27, fill=LINE, width=5)
        if not self.reader or self.loading:
            return
        total = max(1, self.reader.n_frames - 1)
        pos = lambda f: 10 + f / total * (w - 20)
        for name, color in (("own", ACCENT), ("official", "#73bd82"), ("history", "#a585ca"), ("imported", "#d99646")):
            if self.layers[name].get():
                for e in (self._own_rows() if name == "own" else self.references[name]):
                    interval = interval_of(e)
                    if interval is None:
                        continue
                    row = 15 if name == "own" else (7 if name == "official" else 33)
                    shade = color if e.get("valid", True) else "#a8a8a8"
                    if interval[0] == interval[1]:
                        c.create_line(pos(interval[0]), row, pos(interval[0]), row + 7, fill=shade, width=2)
                    else:
                        c.create_rectangle(pos(interval[0]), row, max(pos(interval[0]) + 3, pos(interval[1])), row + 5, fill=shade, outline="")
        x = pos(self.frame)
        c.create_line(10, 27, x, 27, fill=ACCENT, width=5)
        c.create_oval(x - 6, 21, x + 6, 33, fill=ACCENT, outline=WHITE, width=2)
        c.create_text(10, 46, text="0", fill=MUTED, anchor="w", font=("Segoe UI", 8))
        c.create_text(w - 10, 46, text=str(self.reader.n_frames - 1), fill=MUTED, anchor="e", font=("Segoe UI", 8))

    def _review_timeline_x(self, frame, width):
        stamps = self.reader.timestamps
        duration = stamps[-1] - stamps[0] if stamps[0] is not None else 0
        ratio = (stamps[frame] - stamps[0]) / duration if duration > 0 else frame / max(1, len(stamps) - 1)
        return 10 + ratio * (width - 20)

    def _draw_review_timeline(self, c, width):
        from review_view import STATE_COLORS

        c.delete("all")
        self.timeline_markers = []
        self.timeline_playhead = 10
        track_color = "#293646"
        c.create_rectangle(10, 27, width - 10, 29, fill=track_color, outline="")
        if not self.reader or self.loading or self.review_view is None:
            return
        stamps = self.reader.timestamps
        duration = stamps[-1] - stamps[0] if stamps[0] is not None else 0

        track_width = width - 20
        if track_width > 0:
            if self.timeline_gradient_width != track_width:
                ramp = Image.linear_gradient("L").transpose(Image.Transpose.ROTATE_90).resize((track_width, 2))
                colors = ImageOps.colorize(ramp, "#3d6977", review_theme.ACCENT)
                self.timeline_gradient = ImageTk.PhotoImage(colors, master=self)
                self.timeline_gradient_width = track_width
            c.create_line(10, 28, 10, 28, fill="#1b3942", width=4, capstyle="round", tags="track_glow")
            c.create_image(10, 27, image=self.timeline_gradient, anchor="nw")
            c.create_rectangle(10, 27, width - 10, 29, fill=track_color, outline="", tags="track_mask")
        selected = None
        for key in self.review_view.tree.get_children():
            row = self.review_view.rows[key]
            x = self._review_timeline_x(row["frame_id"], width)
            self.timeline_markers.append((x, key))
            shade = STATE_COLORS[row["status"]]
            if key == self.review_view.selected_id:
                selected = x, shade
            else:
                c.create_line(x, 12, x, 21, fill=shade, width=1, capstyle="round")
        if selected is not None:
            x, shade = selected
            c.create_line(x, 8, x, 22, fill="#334b52", width=5, capstyle="round")
            c.create_line(x, 8, x, 22, fill=shade, width=2, capstyle="round")
        c.create_oval(2, 20, 18, 36, outline="#294752", width=2, tags="playhead")
        c.create_oval(5, 23, 15, 33, fill=review_theme.CANVAS, outline=review_theme.ACCENT, tags="playhead")
        c.create_line(10, 25, 10, 31, fill=review_theme.ACCENT, tags="playhead")
        c.create_text(10, 45, text="0.00 s" if duration > 0 else "f0", fill=review_theme.MUTED, anchor="w",
                      font=(review_theme.NUMERIC, 9))
        c.create_text(width - 10, 45, text=f"{duration:.2f} s" if duration > 0 else f"f{self.source_frame_id(len(stamps) - 1)}",
                      fill=review_theme.MUTED, anchor="e", font=(review_theme.NUMERIC, 9))
        if self.timeline_hover_x is not None:
            self._timeline_hover(self.timeline_hover_x)

    def _auto_references(self):
        from default_ball import references as default_ball_references

        e = self.entry
        notes = []
        public = ROOT / "participant/participant/pingpang/public_data/gt_public.jsonl"
        if e.role == "public_regression" and public.exists():
            self.references["official"] = load_reference(public, e.video_id)
        tracks = [ROOT / "C_train154_20261011_v3/C_train154_20261011_v3/trajectories" / f"{e.video_id}.jsonl",
                  ROOT / "outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories" / f"{e.video_id}.jsonl"]
        for path in tracks:
            if path.exists():
                self.references["trajectory"] = load_reference(path, e.video_id)
                break
        default = default_ball_references(self.reader.sha256, e.video_id)
        if default is not None:
            self.references['trajectory'], predictions = default
            self.references['imported'] = [p for p in self.references['imported'] if p.get('ball_profile') != 'round3_reviewed'] + predictions
            self.reference_names['trajectory'] = '第三轮默认球点'
        if e.annotation_path and e.annotation_path.exists():
            try:
                data = json.loads(e.annotation_path.read_text(encoding="utf-8-sig"))
                self.boxes = {r["frame_id"]: r.get("annotations", []) for r in data.get("frames", [])}
            except (ValueError, KeyError) as ex:
                notes.append(f"目标框无法读取：{ex}")
        self.saved_own = read_own_session(self.output, e, self.reader)
        self.references["history"], warnings = load_history(ROOT, e, self.reader)
        notes.extend(warnings)
        if notes:
            logging.warning("Reference loading: %s", "; ".join(notes))
        self._refresh_layer_summary()
        self.refresh_records()

    def _own_rows(self):
        state = self.store.state if self.store else self.saved_own
        return state.get("events", []) + state.get("uncertain", []) if state else []

    def _refresh_layer_summary(self):
        counts = {name: len(self.references[name]) for name in self.references}
        counts["own"] = len(self._own_rows())
        counts["boxes"] = sum(bool(v) for v in self.boxes.values())
        names = {"own": "工作台标注", "history": "此前人工标注", "official": "官方触台 GT", "trajectory": "球轨迹", "imported": "导入参考", "boxes": "原始目标框"}
        if self.portable:
            names["imported"] = "自动触台预测"
        if self.reference_names.get('trajectory'):
            names['trajectory'] = self.reference_names['trajectory']
        for name, button in self.layer_buttons.items():
            button.configure(text=f"{names[name]} · {counts[name]} {'帧' if name in {'boxes', 'trajectory'} else '条'}", state="normal" if counts[name] or name == "own" else "disabled")
        if self.portable:
            self.layer_info.configure(text="单击预测跳到原视频帧。\n仅显示实际取样得到的球点。")
            self._refresh_reference_list()
            return
        if self.entry.role == "public_regression":
            valid = sum(r.get("valid", True) for r in self.references["official"])
            gt = f"官方 GT：{valid} 个有效，{counts['official'] - valid} 个无效。"
        else:
            gt = "本视频不提供官方触台 GT。"
        self.layer_info.configure(text=gt + "\n此前人工标注独立显示，保持只读。")
        self._refresh_reference_list()

    def _layers_changed(self):
        self._refresh_reference_list()
        self.render()

    def _refresh_reference_list(self):
        self.reference_list.delete(*self.reference_list.get_children())
        self.reference_records = {}
        names = {"official": "GT", "history": "历史", "imported": "参考", "own": "我的"}
        if self.portable:
            names["imported"] = "预测"
        for name in ("official", "history", "imported", "own"):
            if not self.layers[name].get():
                continue
            for row in (self._own_rows() if name == "own" else self.references[name]):
                interval = interval_of(row)
                if interval is None:
                    continue
                key = f"ref_{len(self.reference_records)}"
                self.reference_records[key] = row
                kind = TYPE_NAMES.get(row.get("event_type"), "疑难")
                if not row.get("valid", True):
                    kind += "(无效)"
                span = str(interval[0]) if interval[0] == interval[1] else f"{interval[0]}–{interval[1]}"
                self.reference_list.insert("", "end", iid=key, values=(f"{names[name]} / {kind}", span))

    def _reference_selected(self, _=None):
        selected = self.reference_list.selection()
        if selected:
            row = self.reference_records.get(selected[0])
            if row:
                self.pause()
                self.seek(interval_of(row)[0])
                if self.portable:
                    self.status.set(f"定位预测触台 · 源帧 {row['frame_id']}")
                else:
                    self.status.set(f"已定位参考事件 · {row.get('source_path', '工作台标注')}")

    def import_reference(self):
        if not self.reader:
            self.status.set("请先打开视频")
            return
        path = filedialog.askopenfilename(parent=self, title="选择参考标签或模型结果", filetypes=[("JSON / JSONL", "*.json *.jsonl"), ("全部", "*.*")])
        if path:
            try:
                records = load_reference(Path(path), self.entry.video_id)
                invalid = [e for e in records if isinstance(e.get("frame_id"), int) and not 0 <= e["frame_id"] < self.reader.n_frames]
                if invalid:
                    raise ValueError("参考文件包含超出当前视频的帧号，请核对视频与结果是否匹配")
                self.references["imported"] = records
                self.layers["imported"].set(True)
                self._refresh_layer_summary()
                self.render()
                self.status.set(f"已导入 {len(records)} 条参考记录")
            except Exception as ex:
                self.error(str(ex))

    def _mode_changed(self):
        self.pause()
        editing = self.mode.get() == "标注" and self.reader is not None and not self.loading
        if editing and self.store is None:
            try:
                self.store = SessionStore(self.output / "sessions", self.entry, self.reader)
                self.seek(self.store.state.get("last_frame", self.frame))
            except Exception as ex:
                self.mode.set("观察")
                editing = False
                self.error(str(ex))
        self._set_editing(editing)
        self.note_hint.configure(text="先选类型，再单击画面选点，按 Enter 确认。\n所有坐标按原始视频像素保存。" if editing else "切换到标注模式后开始记录。\n观察模式下点击画面不会写入标注。")
        self.save_status.set("标注模式 · 每 15 秒自动保存" if editing else "观察模式")
        if not editing:
            self.clear_draft()
        self.refresh_records()
        if self.reader:
            self._refresh_layer_summary()
        self.render()

    def _set_editing(self, enabled):
        for w in getattr(self, "edit_widgets", []):
            w.configure(state="normal" if enabled else "disabled")
        buttons = ["save_button"] if self.portable else ["save_button", "export_button"]
        for w in (getattr(self, name, None) for name in buttons):
            if w:
                w.configure(state="normal" if enabled else "disabled")

    def _can_edit(self):
        if self.mode.get() != "标注" or not self.store:
            self.status.set("请先打开视频并切换到标注模式")
            return False
        if self.reading or self.target_frame != self.frame:
            self.status.set("正在定位帧，请稍候再标注")
            return False
        self.pause()
        return True

    def _canvas_click(self, event):
        if self.review_view is not None and self.review_view.adding:
            if not self.loading and not self.reading and self.target_frame == self.frame:
                self.pause()
                self.review_view.add(self.frame, self._to_original(event.x, event.y))
            return
        if self.table_view is not None and self.table_view.pending is not None:
            if not self.loading and not self.reading:
                self.table_view.add_corner(self._to_original(event.x, event.y), self.frame)
            return
        if self.table_demo:
            self.canvas.focus_set()
            return
        self.canvas.focus_set()
        if self.mode.get() != "标注" or not self._can_edit():
            return
        point = self._to_original(event.x, event.y)
        if point:
            self.draft = (self.frame, *point)
            self.point_status.set(f"帧 {self.frame}  ·  x {point[0]:.1f}  /  y {point[1]:.1f}")
            self.render()

    def _type_changed(self):
        self.status.set(f"当前类型：{self.event_type.get()} · 触台必须选点，其余事件可只记录帧号")

    def clear_draft(self):
        if self.table_demo and self.draft is None:
            return
        self.draft = None
        self.point_status.set("尚未选点")
        self.render()

    def add_event(self):
        if not self._can_edit():
            return
        x, y = self.draft[1:] if self.draft and self.draft[0] == self.frame else (None, None)
        try:
            self.store.add_event(TYPES[self.event_type.get()], self.frame, x=x, y=y, reason=self.reason.get().strip())
            self.reason.set("")
            self.clear_draft()
            self._changed("事件已添加")
        except Exception as ex:
            self.error(str(ex))

    def start_interval(self):
        if self._can_edit():
            self.interval_start = self.frame
            self.interval_status.set(f"起点：帧 {self.frame}，请定位到终点")

    def end_interval(self):
        if not self._can_edit():
            return
        if self.interval_start is None:
            self.status.set("请先设置区间起点（I）")
            return
        try:
            touch = bool(self.interval_touch.get())
            self.store.add_uncertain(min(self.interval_start, self.frame), max(self.interval_start, self.frame),
                                     self.reason.get().strip(), touch=touch)
            self.interval_start = None
            self.interval_status.set("未设置区间起点")
            self.reason.set("")
            self._changed("疑难区间已记录（含触台）" if touch else "疑难区间已记录")
        except Exception as ex:
            self.error(str(ex))

    def refresh_records(self):
        if not hasattr(self, "record_list"):
            return
        self.record_list.delete(*self.record_list.get_children())
        self.records = {}
        if self.store or self.saved_own:
            for e in sorted(self._own_rows(), key=lambda e: e.get("frame_id", e.get("frame_start", 0))):
                kind = TYPE_NAMES.get(e.get("event_type"), "疑难区间")
                if "frame_start" in e and e.get("touch"):
                    kind = "疑难·含触台"
                frame = str(e["frame_id"]) if "frame_id" in e else f"{e['frame_start']} – {e['frame_end']}"
                self.records[e["id"]] = e
                self.record_list.insert("", "end", iid=e["id"], values=(kind, frame))
        self.record_count.configure(text=f"本次标注 · {len(self.records)} 条")

    def _record_selected(self, _=None):
        selected = self.record_list.selection()
        if selected and selected[0] in self.records:
            e = self.records[selected[0]]
            self.pause()
            self.seek(e.get("frame_id", e.get("frame_start", 0)))
            detail = e.get("reason", "") or "无备注"
            if e.get("x") is not None:
                detail = f"x {e['x']:.2f} / y {e['y']:.2f}\n{detail}"
            self.record_detail.configure(text=detail)

    def delete_record(self):
        if self._can_edit() and self.record_list.selection():
            self.store.delete(self.record_list.selection()[0])
            self._changed("记录已删除 · 可撤销")

    def edit_reason(self):
        if not self._can_edit() or not self.record_list.selection():
            return
        e = self.records[self.record_list.selection()[0]]
        reason = simpledialog.askstring("修改备注", "备注 / 判断依据", parent=self, initialvalue=e.get("reason", ""))
        if reason is None:
            return
        if "frame_start" in e and not reason.strip():
            self.error("疑难区间需要填写说明")
            return
        try:
            self.store.update_reason(e["id"], reason.strip())
            self._changed("备注已修改")
        except Exception as ex:
            self.error(str(ex))

    def undo(self):
        if self._can_edit() and self.store.undo():
            self._changed("已撤销")

    def redo(self):
        if self._can_edit() and self.store.redo():
            self._changed("已重做")

    def _changed(self, text):
        self.refresh_records()
        self._refresh_layer_summary()
        self.render()
        self.save_status.set("有修改 · 自动保存中")
        self.status.set(text)
        self.save(quiet=True)

    def save(self, quiet=False):
        if not self.store:
            return True
        try:
            self.store.state["last_frame"] = self.frame
            self.store.save()
            self.save_status.set(("人工核查完成 · " if self.store.state.get("completed") else "草稿 · ") + "已保存 " + time.strftime("%H:%M:%S"))
            if not quiet:
                self.status.set("标注已保存，可关闭工具后继续")
            return True
        except Exception as ex:
            self.error("保存失败，当前标注仍在内存中：" + str(ex))
            return False

    def _autosave(self):
        if self.closed:
            return
        if self.store:
            self.save(quiet=True)
        self.autosave_job = self.after(15000, self._autosave)

    def complete(self):
        if not self._can_edit():
            return
        if messagebox.askyesno("人工核查完成", "是否已经人工检查整段视频，并记录所有事件和无法判断的区间？\n\n此状态代表你的人工确认，不由播放进度自动生成。", parent=self):
            try:
                if not any(e["event_type"] == "table_bounce" for e in self.store.state["events"]):
                    note = simpledialog.askstring("无触台说明", "本段未标注触台，请说明原因（例如整段无触台 / 均位于疑难区间）：", parent=self, initialvalue=self.store.state.get("notes", ""))
                    if not note or not note.strip():
                        return
                    self.store.state["notes"] = note.strip()
                self.store.mark_complete(True)
                self._changed("已标记整段人工核查完成")
            except Exception as ex:
                self.error(str(ex))

    def export(self):
        if self.portable:
            self.export_predictions()
            return
        if not self.store or not self.save(quiet=True):
            return
        dialog = tk.Toplevel(self)
        dialog.title("导出标注")
        dialog.geometry("430x285")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        panel = ttk.Frame(dialog, padding=24)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="导出当前视频的标注", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(panel, text="通用包适用于全部视频；团队标准包适用于\n有原始对象 JSON 的训练视频初标。", style="Muted.TLabel").pack(anchor="w", pady=(0, 16))
        member = tk.StringVar(value="A")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(0, 14))
        ttk.Label(row, text="团队标准包成员").pack(side="left", padx=(0, 15))
        ttk.Combobox(row, textvariable=member, values=("A", "B", "C", "D"), state="readonly", width=5).pack(side="left")
        def run(team):
            folder = filedialog.askdirectory(parent=dialog, title="选择导出位置（自动创建独立导出包）")
            if not folder:
                return
            try:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.entry.video_id)
                target = Path(folder) / f"{safe}_{'team' if team else 'annotations'}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}"
                path = self.store.export_team(target, annotator=member.get()) if team else self.store.export(target)
                dialog.destroy()
                self.status.set(f"已导出：{path}")
                messagebox.showinfo("导出成功", f"标注包已保存到：\n{path}", parent=self)
            except Exception as ex:
                messagebox.showerror("无法导出", str(ex), parent=dialog)
        ttk.Button(panel, text="导出通用标注包", style="Accent.TButton", command=lambda: run(False)).pack(fill="x", pady=(0, 8))
        ttk.Button(panel, text="导出团队标准包（初标）", command=lambda: run(True)).pack(fill="x")

    def export_predictions(self):
        if self.cpu_result is None:
            return
        from portable_results import export_result
        folder = filedialog.askdirectory(parent=self, title="选择落点导出位置")
        if folder:
            try:
                path = export_result(self.cpu_result, Path(folder), self.entry.video_id)
                self.status.set(f"已导出落点：{path}")
            except (OSError, ValueError, KeyError) as error:
                self.error(str(error))

    def snapshot(self):
        if self.image_bgr is None or not self.reader:
            return
        folder = self.output / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.entry.video_id)
        path = folder / f"{safe}_f{self.source_frame_id(self.frame):06d}_{time.time_ns()}.png"
        ok, encoded = cv2.imencode(".png", self.image_bgr)
        if not ok:
            self.error("截图编码失败")
            return
        encoded.tofile(str(path))
        self.status.set(f"已保存原始帧截图：{path.name}")

    def show_source(self):
        if self.entry:
            os.startfile(str(self.entry.path.parent))

    def show_output(self):
        self.output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output))

    def help(self):
        if self.table_view is not None:
            messagebox.showinfo("触台复核原型",
                                "点击时间轴圆点、球桌落点或清单中的事件跳回原片。时间轴随清单状态筛选。使用确认、驳回、标疑难记录判断。"
                                "要补记漏报，先点补记按钮，定位原片帧后在左侧画面点击球心。复核自动保存。\n\n"
                                "俯视图默认随播放只画最新出现的一个落点，也可切换仅触台帧或全部落点。"
                                "打开「台面线」可在原画面检查四角连线，重标时自动显示。"
                                "原视频上滚轮以光标为中心缩放，右键拖动，A/D 逐帧。"
                                "点「适应画面」恢复全图，Shift+滚轮也可逐帧。\n\n"
                                "「回看片段」播放触台前后各约 0.25 秒，J/K 跳到上一处或下一处。F 确认，X 驳回，Q 标疑难。"
                                "校正点位允许重新选择原始帧和落点。\n\n"
                                "布局菜单支持独立弹出、停靠、隐藏和保存。拖动分隔条可以调整大小。\n\n"
                                "重标四角时，先选 A 端的一个角，沿同一条长边到 B 端，"
                                "横跨 B 端到第三角，最后沿另一条长边回到 A 端。"
                                "四个角在同一帧上选取。A 端显示在示意图左侧。",
                                parent=self)
            return
        if self.portable:
            messagebox.showinfo("本地 CPU 工作台", "导入视频 → 自动分析 → 点击落点回看 → 导出落点。\n"
                                "支持按原始帧号逐帧前后查看。高帧率视频按时间采样，预测回写到实际源帧。",
                                parent=self)
            return
        messagebox.showinfo("视频工作台 · 使用说明", "1. 左侧筛选或导入视频，单击打开。\n2. 观察模式：播放、逐帧、拖时间轴、放大、截图。\n3. 标注模式：选类型，点击画面，再按 Enter。\n4. I 设置疑难区间起点；O 记录终点，需要备注。\n5. 记录页可定位、修改备注、删除、撤销 / 重做。\n6. 图层页可显示官方 GT、球轨迹、对象框，或导入参考。\n7. 每次标注自动保存，支持关闭后继续。\n\nSpace：播放 / 暂停；← →：前后帧；Shift+方向键：10帧\nCtrl+S：保存；Ctrl+Z / Y：撤销 / 重做\n右键：设置放大中心；截图保存不含叠加的原始帧\n\n帧号从0开始；事件坐标始终是原图像素。\n参考层不自动转成人工标签。视频音轨暂不播放。", parent=self)

    def error(self, text):
        logging.error(text)
        self.status.set(text[:130])
        messagebox.showerror("视频工作台", text, parent=self)

    def _tk_error(self, exc, value, traceback):
        logging.error("UI callback error", exc_info=(exc, value, traceback))
        self.error(str(value))

    def _release_session(self):
        if self.store:
            if not self.save(quiet=True):
                return False
            self.store.close()
            self.store = None
        return True

    def close(self):
        if not self._release_session():
            return
        self.closed = True
        for job in (*self.startup_jobs, self.poll_job, self.autosave_job):
            self.after_cancel(job)
        if self.render_job is not None:
            self.after_cancel(self.render_job)
        if self.review_view is not None and self.review_view.reveal_job is not None:
            self.review_view.tree.after_cancel(self.review_view.reveal_job)
        self.pause()
        if self.reader:
            self.executor.submit(self.reader.close)
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.analysis_executor.shutdown(wait=False, cancel_futures=False)
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="Pingpong unified video workspace")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--table-demo", type=Path)
    args = parser.parse_args()
    table_demo = None
    if args.table_demo:
        from core import _read
        table_demo = _read(args.table_demo)
        args.video = ROOT / table_demo["video"]
    if args.output is None:
        args.output = (Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PingpongStudio"
                       if args.portable else ROOT / "outputs/video_studio")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output / "video_studio.log", level=logging.INFO, encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        Studio(args.output, args.video, portable=args.portable, table_demo=table_demo).mainloop()
    except Exception:
        logging.exception("Application failed")
        raise


if __name__ == "__main__":
    main()

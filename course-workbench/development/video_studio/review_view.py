"""Review decisions kept separately from the source predictions."""
from tkinter import StringVar, ttk
from uuid import uuid4

from core import _Lock, _read, _sha, _write
from flight_review import FLIGHT_HINTS, flight_hints


STATE_NAMES = {"pending": "待复核", "confirmed": "已确认", "rejected": "已驳回", "uncertain": "疑难"}
STATE_COLORS = {"pending": "#ffc270", "confirmed": "#81ecc3", "rejected": "#a6b6c7", "uncertain": "#d6b7fc"}
AIRBORNE = "疑似空中"


class ReviewView(ttk.Frame):
    def __init__(self, parent, studio):
        super().__init__(parent, padding=(13, 10), style="ReviewCard.TFrame")
        self.studio = studio
        self.path = None
        self.manual_path = None
        self.predictions = []
        self.flight = {}
        self.state = {"decisions": {}, "manual": []}
        self.rows = {}
        self.selected_id = None
        self.reveal_job = None
        self.adding = False
        self.correction_id = None
        self.filter = StringVar(value="全部状态")
        self.note = StringVar(value="正在载入已有预测…")
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="事件复核", style="Section.TLabel").pack(side="left")
        studio.review_layout.add_panel_menu(header, "review")
        self.counter = ttk.Label(header, style="Muted.TLabel")
        self.counter.pack(side="left", padx=(8, 0))
        controls = ttk.Frame(self)
        controls.pack(side="bottom", fill="x", pady=(5, 0))
        decisions = ttk.Frame(controls)
        decisions.pack(fill="x", pady=(0, 4))
        self.actions = []
        for index, (state, text, style) in enumerate((("confirmed", "确认  F", "Confirm"),
                                                     ("rejected", "驳回  X", "Reject"),
                                                     ("uncertain", "标疑难  Q", "Uncertain"))):
            button = ttk.Button(decisions, text=text, state="disabled", style=f"{style}.ReviewAction.TButton",
                                command=lambda value=state: self.decide(value), width=1)
            decisions.columnconfigure(index, weight=1, uniform="verdicts")
            button.grid(row=0, column=index, sticky="ew", padx=(0, 4) if index < 2 else 0)
            self.actions.append(button)
        self.add_button = ttk.Button(controls, text="补记漏报", state="disabled", style="ReviewAction.TButton",
                                     command=self.toggle_add)
        self.add_button.pack(side="left")
        self.correct_button = ttk.Button(controls, text="校正点位", state="disabled", style="ReviewAction.TButton",
                                         command=self.toggle_correction)
        self.correct_button.pack(side="left", padx=(4, 0))
        ttk.Label(self, textvariable=self.note, style="Muted.TLabel", wraplength=340).pack(side="bottom", fill="x", pady=(4, 0))
        nav = ttk.Frame(self)
        nav.pack(fill="x", pady=(0, 5))
        for label, command in (("‹  上一处", lambda: self.navigate(-1)), ("下一处  ›", lambda: self.navigate(1))):
            ttk.Button(nav, text=label, command=command, style="Quiet.TButton").pack(side="left", padx=(0, 3))
        ttk.Button(nav, text="回看片段", command=self.preview_selected, style="Small.TButton").pack(side="left")
        picker = ttk.Combobox(nav, textvariable=self.filter, values=("全部状态", *STATE_NAMES.values(), AIRBORNE),
                              width=8, state="readonly")
        picker.pack(side="right")
        picker.bind("<<ComboboxSelected>>", self.refilter)
        area = ttk.Frame(self)
        area.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(area, columns=("frame", "source", "status", "position"),
                                 height=4, show="headings", selectmode="browse", style="Review.Treeview")
        for state, color in STATE_COLORS.items():
            if state != "pending":
                self.tree.tag_configure(state, foreground=color)
        for name, label, width in (("frame", "原片帧", 58), ("source", "来源", 82),
                                   ("status", "复核状态", 72), ("position", "投影状态", 70)):
            self.tree.heading(name, text=label)
            self.tree.column(name, width=width, minwidth=38, anchor="center")
        scrollbar = ttk.Scrollbar(area, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<ButtonRelease-1>", self.pick_row)
        self.tree.bind("<Return>", self.pick_selected)
        self.tree.bind("<<TreeviewSelect>>", self.selection_changed)
        self.tree.bind("<Configure>", self.reveal_after_resize)
        self.empty = ttk.Label(area, text="当前筛选无事件", style="Empty.TLabel")

    def reveal_after_resize(self, _event):
        if self.reveal_job is None:
            self.reveal_job = self.tree.after(40, self.keep_selection_visible)

    def keep_selection_visible(self):
        self.reveal_job = None
        if self.selected_id is not None and self.tree.exists(self.selected_id):
            bounds = self.tree.bbox(self.selected_id)
            if not bounds or bounds[1] + bounds[3] > self.tree.winfo_height():
                self.tree.see(self.selected_id)

    def clear(self):
        self.path = self.manual_path = self.selected_id = None
        self.predictions = []
        self.flight = {}
        self.state = {"decisions": {}, "manual": []}
        self.cancel_add()
        self.show_rows([])
        self.note.set("导入视频后开始复核。")

    def load(self, reader, prediction_file, predictions, *, shared_manual=False):
        directory = self.studio.output / "landing_reviews"
        fingerprint = _sha(prediction_file) if prediction_file is not None else "manual"
        self.path = directory / f"{reader.sha256}_{fingerprint}.json"
        self.manual_path = directory / f"{reader.sha256}_manual.json" if shared_manual else self.path
        self.predictions = [dict(row, review_id=f"prediction-{i}", source="prediction")
                            for i, row in enumerate(predictions)]
        self.flight = flight_hints(predictions, self.studio.review_tracks, self.studio.review_times, reader.width)
        self.state = self.read_state()
        self.selected_id = next((row["review_id"] for row in self.predictions
                                 if row["frame_id"] == self.studio.table_view.frame_id), None)
        self.add_button.configure(state="normal")
        self.note.set("点击清单或时间轴圆点跳帧。")

    def read_state(self):
        state = _read(self.path) if self.path.is_file() else {"decisions": {}, "manual": []}
        if self.manual_path != self.path and self.manual_path.is_file():
            manual = _read(self.manual_path)
            state["manual"] = manual["manual"]
            state["decisions"].update(manual["decisions"])
            state.setdefault("corrections", {}).update(manual.get("corrections", {}))
        return state

    def save_edit(self, edit, *, manual=False):
        path = self.manual_path if manual else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _Lock(path.with_suffix(".lock"))
        try:
            state = _read(path) if path.is_file() else {"decisions": {}, "manual": []}
            edit(state)
            _write(path, state)
        finally:
            lock.close()
        self.state = self.read_state()

    def events(self):
        decisions = self.state["decisions"]
        corrections = self.state.get("corrections", {})
        table = self.studio.table_view
        adjustments = table.adjustments if table.refine.get() else {}
        rows = []
        for original in self.predictions + self.state["manual"]:
            key = original["review_id"]
            manual = original["source"] == "manual"
            row = dict(original, status=decisions.get(key, "confirmed" if manual else "pending"),
                       position_source="manual" if manual else "prediction")
            hint = self.flight.get(original["frame_id"]) if not manual else None
            if hint is not None:
                row["flight_hint"] = hint
            if key in corrections:
                row.update(corrections[key], position_source="corrected", original_frame_id=original["frame_id"])
            elif key in adjustments and hint is None:
                row.update(adjustments[key], position_source="track", original_frame_id=original["frame_id"])
            rows.append(row)
        return sorted(rows, key=lambda row: row["frame_id"])

    def show_rows(self, rows):
        self.rows = {row["review_id"]: row for row in rows}
        self.refilter()
        pending = sum(row["status"] == "pending" for row in rows)
        self.counter.configure(text=f"待复核 {pending} / {len(rows)}")

    def refilter(self, event=None):
        choices = {"prediction": "预测", "manual": "人工", "track": "轨迹校正", "corrected": "人工校正"}
        active = self.filter.get()
        self.tree.delete(*self.tree.get_children())
        for key, row in self.rows.items():
            if active == AIRBORNE and row["map_note"] != AIRBORNE:
                continue
            if active not in ("全部状态", AIRBORNE) and STATE_NAMES[row["status"]] != active:
                continue
            source = choices[row["position_source"]]
            position = row["side"] or row["map_note"]
            frame_id = self.studio.source_frame_id(row["frame_id"])
            self.tree.insert("", "end", iid=key, values=(f"f{frame_id}", source, STATE_NAMES[row["status"]], position),
                             tags=(row["status"],))
        if self.tree.get_children():
            self.empty.place_forget()
        else:
            self.empty.place(relx=.5, rely=.57, anchor="center")
        if self.selected_id is not None and self.tree.exists(self.selected_id):
            self.tree.selection_set(self.selected_id)
            self.tree.see(self.selected_id)
        self.update_actions()
        self.studio.draw_timeline(refresh=True)
        if event is not None:
            self.tree.focus_set()

    def update_actions(self):
        available = self.selected_id is not None and self.tree.exists(self.selected_id) and not self.adding
        for button in self.actions:
            button.configure(state="normal" if available else "disabled")
        self.correct_button.configure(state="normal" if available or self.correction_id is not None else "disabled")
        self.add_button.configure(state="disabled" if self.path is None or self.correction_id is not None else "normal")

    def selection_changed(self, _event=None):
        selection = self.tree.selection()
        if selection:
            self.selected_id = selection[0]
            self.studio.table_view.select(self.selected_id)
            self.update_actions()
            self.studio.draw_timeline(refresh=True)

    def pick_row(self, event):
        key = self.tree.identify_row(event.y)
        if key in self.rows:
            self.select_and_go(key)

    def pick_selected(self, _event):
        if self.selected_id in self.rows:
            self.select_and_go(self.selected_id)
        return "break"

    def select_and_go(self, key):
        self.selected_id = key
        if not self.tree.exists(key):
            self.filter.set("全部状态")
            self.refilter()
        self.tree.selection_set(key)
        self.tree.see(key)
        self.studio.table_view.select(key)
        self.studio.table_view.go(self.rows[key]["frame_id"])
        self.update_actions()
        row = self.rows[key]
        if not self.adding:
            frame = self.studio.source_frame_id(row["frame_id"])
            original = row.get("original_frame_id")
            detail = f" · 原预测 f{self.studio.source_frame_id(original)}" if original is not None else ""
            if row.get("flight_hint") and row["status"] in ("pending", "uncertain"):
                detail += f" · {FLIGHT_HINTS[row['flight_hint']]}"
            self.note.set(f"f{frame} · {STATE_NAMES[row['status']]}{detail}")
        self.studio.draw_timeline(refresh=True)

    def navigate(self, delta):
        visible = self.tree.get_children()
        if visible:
            at = visible.index(self.selected_id) if self.selected_id in visible else (-1 if delta > 0 else len(visible))
            self.select_and_go(visible[max(0, min(len(visible) - 1, at + delta))])

    def preview_selected(self):
        if self.selected_id is not None and self.tree.exists(self.selected_id):
            self.studio.preview_event(self.rows[self.selected_id]["frame_id"])

    def decide(self, status):
        key = self.selected_id
        if key is None or not self.tree.exists(key) or self.adding:
            return
        if self.studio.loading or self.studio.reading:
            self.note.set("正在定位原片，请看清当前画面后再记录。")
            return
        try:
            self.save_edit(lambda state: state["decisions"].__setitem__(key, status), manual=key.startswith("manual-"))
        except (ValueError, OSError) as error:
            self.studio.error(str(error))
            return
        self.studio.refresh_reviews()
        frame_id = self.studio.source_frame_id(self.rows[key]["frame_id"])
        self.note.set(f"f{frame_id} · {STATE_NAMES[status]} · 已保存")

    def toggle_add(self):
        if self.adding:
            self.cancel_add()
            return
        self.begin_edit()

    def toggle_correction(self):
        if self.correction_id is not None:
            self.cancel_add()
        elif self.selected_id is not None and self.tree.exists(self.selected_id):
            self.begin_edit(self.selected_id)

    def begin_edit(self, key=None):
        s = self.studio
        if s.image_bgr is None or s.loading or s.reading:
            return
        if s.table_view.pending is not None:
            s.table_view.toggle_marking()
        s.pause()
        self.adding = True
        self.correction_id = key
        button = self.correct_button if key else self.add_button
        button.configure(text="取消校正" if key else "取消补记")
        self.note.set("先定位正确帧，再在原片点击落点。")
        s.status.set("校正点位 · 定位原片后点击保存" if key else "补记漏报 · 定位原片后点击保存")
        self.update_actions()

    def cancel_add(self):
        self.adding = False
        self.correction_id = None
        self.add_button.configure(text="补记漏报")
        self.correct_button.configure(text="校正点位")
        self.note.set("点选事件跳回原片。复核操作自动保存。")
        self.studio.status.set("复核就绪")
        self.update_actions()

    def add(self, frame_id, point):
        if point is None:
            return
        original_key = self.correction_id
        key = original_key or "manual-" + uuid4().hex
        location = dict(frame_id=frame_id, x=point[0], y=point[1])
        def edit(state):
            if original_key:
                state.setdefault("corrections", {})[original_key] = location
            else:
                state["manual"].append(dict(location, review_id=key, source="manual"))
        try:
            self.save_edit(edit, manual=key.startswith("manual-"))
        except (ValueError, OSError) as error:
            self.studio.error(str(error))
            return
        self.selected_id = key
        self.cancel_add()
        self.filter.set("全部状态")
        self.studio.refresh_reviews()
        self.note.set(f"f{self.studio.source_frame_id(frame_id)} · 已保存人工{'校正' if original_key else '补记'}")

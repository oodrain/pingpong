"""Manual fixed-camera calibration and synchronized top-down event preview."""
from bisect import bisect_right
from math import isfinite
import tkinter as tk
from tkinter import ttk

import numpy as np

from core import _read, _write
from landing_refinement import adjacent_observations
from review_view import STATE_COLORS
import review_theme as ui
from review_geometry.table_geometry import (
    TABLE_L, TABLE_W, GeometryError, build_plane,
)


LABELS = ("A端 · 起点", "B端 · 同一长边", "B端 · 另一长边", "A端 · 另一长边")
FOLLOW = "随播放（单点）"
AT_EVENT = "仅触台帧"
OVERVIEW = "全部落点"


def project_events(plane, events):
    rows = []
    for event in events:
        row = dict(event)
        u = v = None
        if plane is not None:
            try:
                u, v = plane.project(row["x"], row["y"])
            except GeometryError:
                pass
        row.update(u=u, v=v, inside=u is not None and 0 <= u <= TABLE_L and 0 <= v <= TABLE_W)
        row["side"] = ("A 端" if u < TABLE_L / 2 else "B 端") if row["inside"] else None
        rows.append(row)
    return rows


def load_events(result, reader):
    metadata = _read(result / "run.json")
    if (metadata["video_sha256"] != reader.sha256
            or metadata["image_size"] != [reader.width, reader.height]
            or metadata["frames"] != reader.n_frames):
        raise ValueError("已有预测与当前样片的指纹、尺寸或帧数不一致。")
    events = _read(result / "landings.json")
    for row in events:
        fid, x, y = row["frame_id"], row["x"], row["y"]
        if (type(fid) is not int or not 0 <= fid < reader.n_frames or not isfinite(x)
                or not isfinite(y) or not (0 <= x < reader.width and 0 <= y < reader.height)):
            raise ValueError("已有预测含非法的原图坐标或源帧号。")
    return sorted(events, key=lambda row: row["frame_id"])


class TableView(ttk.Frame):
    def __init__(self, parent, studio):
        super().__init__(parent, padding=(13, 10), style="ReviewCard.TFrame")
        self.studio = studio
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 4))
        ttk.Label(header, text="俯视球桌", style="Section.TLabel").pack(side="left")
        studio.review_layout.add_panel_menu(header, "table")
        self.mode = tk.StringVar(value=FOLLOW)
        self.mode_picker = ttk.Combobox(header, textvariable=self.mode, values=(FOLLOW, AT_EVENT, OVERVIEW),
                                        width=15, state="readonly")
        self.mode_picker.pack(side="right")
        self.mode_picker.bind("<<ComboboxSelected>>", self.change_mode)
        self.refine = tk.BooleanVar(value=False)
        self.canvas = tk.Canvas(self, bg=ui.CANVAS, height=150, highlightthickness=0, cursor="hand2")
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<Button-1>", self.pick_event)
        self.rows = []
        self.source_frames = []
        self.source_frame = -1
        self.events = []
        self.predictions = []
        self.adjustments = {}
        self.hit_points = []
        self.corners = []
        self.net = []
        self.plane = None
        self.selected_id = None
        self.pending = None
        self.frame_id = 0
        self.saved_path = None
        self.note = tk.StringVar(value="正在载入已有预测…")
        buttons = ttk.Frame(self)
        buttons.pack(side="bottom", fill="x")
        ttk.Label(self, textvariable=self.note, style="Muted.TLabel", wraplength=340).pack(side="bottom", fill="x", pady=(4, 5))
        self.mark_button = ttk.Button(buttons, text="重标四角", style="Small.TButton", command=self.toggle_marking, state="disabled")
        self.mark_button.pack(side="left")
        self.flip_button = ttk.Button(buttons, text="交换 A/B", style="Quiet.TButton", command=self.flip, state="disabled")
        self.flip_button.pack(side="left", padx=5)
        self.calibration_button = ttk.Button(buttons, text="回到标定帧", style="Quiet.TButton",
                                             command=lambda: self.go(self.frame_id), state="disabled")
        self.calibration_button.pack(side="left")
        ttk.Checkbutton(buttons, text="轨迹校正", variable=self.refine, command=studio.refresh_reviews).pack(side="right")
        self.canvas.pack(fill="both", expand=True)

    def clear(self):
        self.plane = self.pending = self.selected_id = self.saved_path = None
        self.frame_id, self.source_frame = 0, -1
        self.corners, self.net, self.predictions = [], [], []
        self.adjustments = {}
        self.mark_button.configure(text="标定四角", state="disabled")
        self.flip_button.configure(state="disabled")
        self.calibration_button.configure(state="disabled")
        self.set_events([])

    def load(self, reader, predictions, case=None):
        self.refine.set(bool(case and case.get("refine", True)))
        self.predictions = predictions
        self.events = self.predictions
        self.saved_path = self.studio.output / "table_calibration" / (reader.sha256 + ".json")
        saved = self.saved_path.is_file()
        source = _read(self.saved_path) if saved else case
        if source is not None:
            self.set_plane(source["corners"], source["frame_id"], save=False)
        else:
            self.set_events(predictions)
        self.mark_button.configure(state="normal")
        if saved:
            self.note.set("已载入你的四角标定。A 端在示意图左侧。")

    def set_predictions(self, predictions):
        self.predictions = predictions
        if self.plane is not None:
            self.update_adjustments()

    def set_plane(self, corners, frame_id, *, save):
        reader = self.studio.reader
        if type(frame_id) is not int or not 0 <= frame_id < reader.n_frames:
            raise ValueError("标定帧超出源视频范围。")
        points = np.asarray(corners, dtype=float)
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or (points < 0).any() or (points >= [reader.width, reader.height]).any()):
            raise ValueError("四个台角须落在原图范围内。")
        plane = build_plane(dict(corners=points, edge01_axis="length"))
        inverse = np.linalg.inv(plane.matrix)
        net = [inverse @ [TABLE_L / 2, across, 1] for across in (0, TABLE_W)]
        net = [(float(p[0] / p[2]), float(p[1] / p[2])) for p in net]
        if save:
            _write(self.saved_path, dict(frame_id=frame_id, corners=points.tolist()))
        self.corners, self.frame_id, self.net = points.tolist(), frame_id, net
        self.plane = plane
        self.mark_button.configure(text="重标四角")
        self.flip_button.configure(state="normal")
        self.calibration_button.configure(state="normal")
        self.update_adjustments()
        if self.studio.review_view.path is not None:
            self.studio.refresh_reviews()
        else:
            self.set_events(self.events)

    def update_adjustments(self):
        reader = self.studio.reader
        self.adjustments = adjacent_observations(self.predictions, self.studio.review_tracks, self.plane,
                                                 self.corners, (reader.width, reader.height))

    def set_events(self, events):
        self.events = events
        self.rows = project_events(self.plane, events)
        self.source_frames = [row["frame_id"] for row in self.rows]
        shown = sum(row["inside"] for row in self.rows)
        if self.plane is None and self.pending is None:
            self.note.set("导入视频后，标定球桌四角即可显示俯视落点。")
        elif self.pending is None or len(self.pending) == 4:
            adjusted = sum(row.get("position_source") == "track" for row in self.rows)
            frame_id = self.studio.source_frame_id(self.frame_id)
            self.note.set(f"标定 f{frame_id} · {shown}/{len(self.rows)} 点在台内 · {adjusted} 点轨迹校正")
        self.draw()

    def toggle_marking(self):
        if self.pending is not None:
            self.pending = None
            has_plane = self.plane is not None
            self.mark_button.configure(text="重标四角" if has_plane else "标定四角")
            self.flip_button.configure(state="normal" if has_plane else "disabled")
            self.set_events(self.events)
            self.studio.render()
            return
        s = self.studio
        if s.loading or s.reading or s.image_bgr is None:
            return
        if s.review_view.adding:
            s.review_view.cancel_add()
        s.pause()
        self.pending = []
        self.mark_frame = s.frame
        self.mark_button.configure(text="取消标定")
        self.flip_button.configure(state="disabled")
        self.note.set("同一帧按四角顺序点击：先选 A 端起点，再沿长边到 B 端。")
        s.render()

    def add_corner(self, point, frame_id):
        if point is None:
            return
        if frame_id != self.mark_frame:
            frame_id = self.studio.source_frame_id(self.mark_frame)
            self.note.set(f"请回到 f{frame_id} 继续，或取消后在当前帧重标。")
            return
        self.pending.append(point)
        if len(self.pending) == 4:
            try:
                self.set_plane(self.pending, frame_id, save=True)
            except (ValueError, OSError, np.linalg.LinAlgError) as error:
                self.pending.clear()
                self.note.set(f"标定失败，请从 A 端重新点击四角。{error}")
            else:
                self.pending = None
                self.mark_button.configure(text="重标四角")
                self.flip_button.configure(state="normal")
        else:
            self.note.set(f"已标 {len(self.pending)}/4。下一点 {LABELS[len(self.pending)]}。")
        self.studio.render()

    def flip(self):
        try:
            self.set_plane(self.corners[2:] + self.corners[:2], self.frame_id, save=True)
        except (ValueError, OSError) as error:
            self.note.set(str(error))
        self.studio.render()

    def go(self, frame_id):
        self.studio.pause()
        self.studio.seek(frame_id)

    def pick_event(self, event):
        if not self.hit_points:
            return
        x, y, key = min(self.hit_points, key=lambda p: (p[0] - event.x)**2 + (p[1] - event.y)**2)
        if (x - event.x)**2 + (y - event.y)**2 <= 15**2:
            self.studio.review_view.select_and_go(key)

    def select(self, key):
        if key != self.selected_id:
            self.selected_id = key
            self.draw()

    def change_mode(self, _event=None):
        self.studio.canvas.focus_set()
        self.draw()

    def marker_index(self):
        index = bisect_right(self.source_frames, self.source_frame) - 1
        if index < 0 or (self.mode.get() == AT_EVENT and self.source_frames[index] != self.source_frame):
            return -1
        # Predictions and manual records can share a source frame. A selected
        # record at that same instant remains the one displayed.
        for candidate in range(index, -1, -1):
            if self.source_frames[candidate] != self.source_frames[index]:
                break
            if self.rows[candidate].get("review_id") == self.selected_id:
                return candidate
        return index

    def show_frame(self, frame):
        if frame == self.source_frame:
            return
        before = self.marker_index() if self.mode.get() != OVERVIEW else -1
        self.source_frame = frame
        if self.mode.get() != OVERVIEW and before != self.marker_index():
            self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        self.hit_points = []
        width = min(c.winfo_width() - 36, (c.winfo_height() - 34) * TABLE_L / TABLE_W)
        if width < 30:
            return
        height = width * TABLE_W / TABLE_L
        left, top = (c.winfo_width() - width) / 2, (c.winfo_height() - height) / 2 + 4
        right, bottom = left + width, top + height
        c.create_rectangle(left - 3, top - 3, right + 3, bottom + 3, outline="#19333e")
        c.create_rectangle(left, top, right, bottom, fill="#193944", outline="#5a8994")
        c.create_rectangle(left + width / 2, top + 1, right - 1, bottom - 1, fill="#1e414b", outline="")
        for fraction in (.125, .25, .375, .625, .75, .875):
            x = left + width * fraction
            c.create_line(x, top + 1, x, bottom - 1, fill="#294e57")
        for fraction in (.25, .75):
            y = top + height * fraction
            c.create_line(left + 1, y, right - 1, y, fill="#294e57")
        c.create_line(left, top + height / 2, right, top + height / 2, fill="#668890")
        midpoint = left + width / 2
        c.create_line(midpoint, top - 4, midpoint, bottom + 4, fill="#9ab8c0", width=1, dash=(3, 3))
        c.create_text(left + width / 4, top - 13, text="A 端", fill=ui.MUTED, font=("Microsoft YaHei UI", 9))
        c.create_text(right - width / 4, top - 13, text="B 端", fill=ui.MUTED, font=("Microsoft YaHei UI", 9))
        if self.mode.get() == OVERVIEW:
            displayed = sorted(self.rows, key=lambda item: item.get("review_id", "") == self.selected_id)
        else:
            index = self.marker_index()
            displayed = () if index < 0 else (self.rows[index],)
        for row in displayed:
            if not row["inside"]:
                continue
            x, y = left + row["u"] / TABLE_L * width, top + row["v"] / TABLE_W * height
            key = row.get("review_id")
            active = key is not None and key == self.selected_id
            radius = 5 if active else 4
            status = row.get("status", "pending")
            color = STATE_COLORS[status]
            if active:
                c.create_oval(x - 12, y - 12, x + 12, y + 12, outline="#315d66")
                c.create_oval(x - 8, y - 8, x + 8, y + 8, outline=ui.ACCENT)
                c.create_line(x - 12, y, x + 12, y, fill=ui.ACCENT)
                c.create_line(x, y - 12, x, y + 12, fill=ui.ACCENT)
            if status == "rejected":
                c.create_line(x - radius, y - radius, x + radius, y + radius, fill=color, width=2)
                c.create_line(x + radius, y - radius, x - radius, y + radius, fill=color, width=2)
            else:
                dot = 2 if active else radius
                c.create_oval(x - dot, y - dot, x + dot, y + dot, fill=color,
                              outline="#102f36", width=1)
            prefix = "+" if row.get("source") == "manual" else ""
            label_y = y - 19 if y > top + 24 else y + 19
            frame_id = self.studio.source_frame_id(row["frame_id"])
            label = c.create_text(x, label_y, text=f"{prefix}f{frame_id}", fill=color, font=(ui.NUMERIC, 10, "bold"))
            x0, y0, x1, y1 = c.bbox(label)
            chip = c.create_rectangle(x0 - 3, y0, x1 + 3, y1, fill=ui.CANVAS, outline="")
            c.tag_lower(chip, label)
            if key is not None:
                self.hit_points.insert(0 if active else len(self.hit_points), (x, y, key))

    def draw_source(self, canvas, locate, frame_id):
        if self.pending is not None and frame_id != self.mark_frame:
            return
        corners = self.pending if self.pending is not None else self.corners
        for index, corner in enumerate(corners):
            point = locate(*corner)
            if not point:
                continue
            x, y = point
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#50ffdf", outline="#123b48")
            canvas.create_text(x, y - 12, text=f"{index + 1}{LABELS[index][0]}", fill="white",
                               font=("Segoe UI", 10, "bold"))
            if index or len(corners) == 4:
                previous = locate(*corners[index - 1])
                if previous:
                    canvas.create_line(*previous, *point, fill="#50ffdf", width=2)
        if self.pending is None and self.net:
            a, b = [locate(*point) for point in self.net]
            if a and b:
                canvas.create_line(*a, *b, fill="#50ffdf", width=2, dash=(4, 4))

"""Dock, float and restore the review workbench's individual panels."""
import logging
import re
import tkinter as tk
from tkinter import ttk

from core import _read, _write
import review_theme as ui


TITLES = {"table": "俯视球桌", "review": "复核清单", "track": "轨迹详情"}
DEFAULTS = {"table": "dock", "review": "dock", "track": "hide"}


class ReviewLayout:
    def __init__(self, studio, toolbar):
        self.studio = studio
        self.path = studio.output / "review_layout.json"
        self.split = ttk.Panedwindow(studio.inspector, orient="vertical")
        self.split.pack(fill="both", expand=True)
        self.hosts = {key: tk.Frame(self.split, background=ui.BACKGROUND) for key in TITLES}
        self.cards = {}
        self.gap_pad = 2
        studio.bind("<Configure>", self.resize_spacing, add="+")
        self.slots = {key: tk.StringVar(value=value) for key, value in DEFAULTS.items()}
        self.floating = set()
        self.docked = ()
        self.order = list(TITLES)
        self.width = 405
        self.sashes = {}
        self.geometry = {}
        self.close_commands = {key: studio.register(lambda name=key: self.set_slot(name, "dock")) for key in TITLES}
        control = ttk.Menubutton(toolbar, text="布局")
        control.pack(side="left", padx=(0, 8))
        menu = self.make_menu(control)
        control.configure(menu=menu)
        menu.add_command(label="视频和复核", command=lambda: self.preset(("table", "review")))
        menu.add_command(label="视频、复核和轨迹", command=lambda: self.preset(tuple(TITLES)))
        menu.add_command(label="专注视频", command=lambda: self.preset(()))
        menu.add_separator()
        for key, title in TITLES.items():
            submenu = self.make_menu(menu)
            self.panel_commands(submenu, key)
            menu.add_cascade(label=title, menu=submenu)
        menu.add_separator()
        menu.add_command(label="保存布局", command=self.save)
        menu.add_command(label="恢复默认布局", command=self.reset)

    def resize_spacing(self, event):
        if event.widget is not self.studio:
            return
        padding = 5 if event.width >= 1080 else 2
        if self.gap_pad != padding:
            self.gap_pad = padding
            self.apply_spacing()

    def apply_spacing(self):
        for key, card in self.cards.items():
            if key in self.docked:
                position = self.docked.index(key)
                top = self.gap_pad if position > 0 else 0
                bottom = self.gap_pad if position < len(self.docked) - 1 else 0
                card.pack_configure(padx=(2 * self.gap_pad, 0), pady=(top, bottom))
            else:
                card.pack_configure(padx=0, pady=0)

    def make_menu(self, parent):
        return tk.Menu(parent, tearoff=False, background=ui.CONTROL, foreground=ui.TEXT,
                       activebackground=ui.HOVER, activeforeground=ui.TEXT, selectcolor=ui.ACCENT)

    def panel_commands(self, menu, key):
        for value, label in (("dock", "停靠"), ("float", "浮动窗口"), ("hide", "隐藏")):
            menu.add_radiobutton(label=label, value=value, variable=self.slots[key], command=self.change)
        menu.add_separator()
        menu.add_command(label="在侧栏上移", command=lambda: self.move_up(key))

    def add_panel_menu(self, heading, key):
        button = ttk.Menubutton(heading, text="窗口", style="Panel.TMenubutton")
        button.pack(side="right", padx=(4, 0))
        menu = self.make_menu(button)
        self.panel_commands(menu, key)
        button.configure(menu=menu)

    def load(self):
        try:
            if self.path.is_file():
                saved = _read(self.path)
                order = saved.get("order", [])
                if sorted(order) == sorted(TITLES):
                    self.order = order
                for key, setting in saved.get("slots", {}).items():
                    if key in self.slots and setting in ("dock", "float", "hide"):
                        self.slots[key].set(setting)
                self.geometry = {key: geometry for key, geometry in saved.get("geometry", {}).items()
                                 if key in TITLES and isinstance(geometry, str)
                                 and re.fullmatch(r"\d+x\d+[+-]\d+[+-]\d+", geometry)}
                width = saved.get("sidebar_width")
                if type(width) is int and width > 0:
                    self.width = width
                self.sashes = {key: positions for key, positions in saved.get("sashes", {}).items()
                               if isinstance(positions, list) and all(type(p) is int and p > 0 for p in positions)}
                self.studio.show_time_tail.set(bool(saved.get("time_tail", True)))
                self.studio.show_table_outline.set(bool(saved.get("table_outline", False)))
        except (ValueError, OSError, TypeError, AttributeError):
            logging.warning("Could not restore review layout %s", self.path, exc_info=True)
        self.arrange()

    def visible(self, key):
        return self.slots[key].get() != "hide"

    def capture(self):
        if self.docked and self.studio.panes.winfo_width() > 100:
            self.width = self.studio.panes.winfo_width() - self.studio.panes.sashpos(0)
            if len(self.docked) > 1 and self.split.winfo_height() > 100:
                points = [self.split.sashpos(index) for index in range(len(self.docked) - 1)]
                if all(p > 0 for p in points):
                    self.sashes[",".join(self.docked)] = points
        for key in self.floating:
            self.geometry[key] = self.wm("geometry", key)

    def restore_sizes(self):
        if not self.docked:
            return
        app = self.studio
        width = app.panes.winfo_width()
        if width < 680:
            return
        app.panes.sashpos(0, max(365, width - max(315, self.width)))
        height = self.split.winfo_height()
        if height < 100:
            return
        count = len(self.docked)
        points = self.sashes.get(",".join(self.docked), [])
        if len(points) != count - 1 or points and points[-1] >= height - 55:
            suggested = {"table": 6, "review": 10, "track": 5} if count == 3 else {"table": 4, "review": 6, "track": 3}
            weights = [suggested[key] for key in self.docked]
            points = [round(height * sum(weights[:i]) / sum(weights)) for i in range(1, count)]
        if count == 2 and self.gap_pad == 2 and height >= 430:
            if self.docked[0] == "review":
                points = [max(280, points[0])]
            elif self.docked[1] == "review":
                points = [min(points[0], height - 286)]
        for index, position in enumerate(points):
            self.split.sashpos(index, position)

    def wm(self, command, key, *args):
        return self.studio.tk.call("wm", command, self.hosts[key], *args)

    def arrange(self):
        app = self.studio
        new_floating = {key for key in self.order if self.slots[key].get() == "float"}
        new_docked = tuple(key for key in self.order if self.slots[key].get() == "dock")
        if not self.visible("table") and app.table_view.pending is not None:
            app.table_view.toggle_marking()
        if not self.visible("review") and app.review_view.adding:
            app.review_view.cancel_add()
        if new_docked != self.docked:
            for name in self.split.panes():
                self.split.forget(name)
        for key in self.floating - new_floating:
            self.wm("forget", key)
        for key in new_floating - self.floating:
            self.wm("manage", key)
            self.wm("title", key, "Pingpong · " + TITLES[key])
            self.wm("minsize", key, 315, 210)
            self.wm("protocol", key, "WM_DELETE_WINDOW", self.close_commands[key])
            offset = 55 * (self.order.index(key) + 1)
            default_x = max(0, min(app.winfo_rootx() + offset, app.winfo_screenwidth() - 490))
            default_y = max(0, min(app.winfo_rooty() + offset, app.winfo_screenheight() - 370))
            self.wm("geometry", key, self.geometry.get(key, f"440x340+{default_x}+{default_y}"))
        if new_docked != self.docked:
            for key in new_docked:
                self.split.add(self.hosts[key], weight=1)
        sidebar = str(app.inspector) in app.panes.panes()
        if new_docked and not sidebar:
            app.panes.add(app.inspector, weight=0)
        elif not new_docked and sidebar:
            app.panes.forget(app.inspector)
        self.floating, self.docked = new_floating, new_docked
        self.apply_spacing()
        app.update_idletasks()
        self.restore_sizes()
        app.render()

    def change(self):
        self.capture()
        self.arrange()

    def set_slot(self, key, value):
        self.slots[key].set(value)
        self.change()

    def move_up(self, key):
        at = self.order.index(key)
        if at > 0:
            self.order[at - 1], self.order[at] = self.order[at], self.order[at - 1]
            self.change()

    def preset(self, keys):
        for key in TITLES:
            self.slots[key].set("dock" if key in keys else "hide")
        self.change()

    def reset(self):
        self.order = list(TITLES)
        self.width = 405
        self.sashes = {}
        self.geometry = {}
        for key, value in DEFAULTS.items():
            self.slots[key].set(value)
        self.studio.show_time_tail.set(True)
        self.studio.show_table_outline.set(False)
        self.arrange()
        self.studio.status.set("已恢复默认布局")

    def save(self):
        self.capture()
        try:
            _write(self.path, dict(order=self.order, slots={key: slot.get() for key, slot in self.slots.items()},
                                   geometry=self.geometry, sidebar_width=self.width, sashes=self.sashes,
                                   time_tail=self.studio.show_time_tail.get(),
                                   table_outline=self.studio.show_table_outline.get()))
        except OSError as error:
            self.studio.error(str(error))
            return
        self.studio.status.set("已保存布局")


class TrackView(ttk.Frame):
    def __init__(self, parent, studio):
        super().__init__(parent, padding=(13, 10), style="ReviewCard.TFrame")
        self.studio = studio
        self.info = ttk.Label(self, text="等待球轨迹", style="Muted.TLabel")
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="轨迹详情", style="Section.TLabel").pack(side="left")
        studio.review_layout.add_panel_menu(header, "track")
        self.info.pack(fill="x", pady=(3, 2))
        self.canvas = tk.Canvas(self, height=130, background=ui.CANVAS, highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<Button-1>", self.jump)
        self.frame = -1
        self.hits = []

    def show_frame(self, frame):
        if frame != self.frame:
            self.frame = frame
            self.draw()

    def draw(self):
        s = self.studio
        if not hasattr(s, "review_tracks") or self.frame < 0:
            return
        row = s.review_observations.get(self.frame)
        stamp = s.review_times.get(self.frame)
        if stamp is None:
            return
        positions = f"  x {row['x']:.1f}  y {row['y']:.1f} px" if row and row["detected"] else "  缺球点"
        self.info.configure(text=f"f{s.source_frame_id(self.frame)}  ·  {stamp + s.source_time_offset:.3f}s{positions}")
        c = self.canvas
        c.delete("all")
        self.hits = []
        x0, x1, y0, y1 = 16, c.winfo_width() - 16, 21, c.winfo_height() - 12
        if x1 <= x0 or y1 <= y0:
            return
        c.create_rectangle(x0, y0, x1, y1, outline=ui.BORDER)
        c.create_line((x0 + x1)/2, y0, (x0 + x1)/2, y1, fill=ui.MUTED, dash=(3, 3))
        c.create_text(x0 + 45, 10, text="蓝 x  ·  绿 y", fill=ui.MUTED)
        previous = None
        for point in s.review_tracks:
            timestamp = point["timestamp_s"]
            if timestamp < stamp - .32:
                continue
            if timestamp > stamp + .32:
                break
            x = x0 + ((timestamp - stamp + .32) / .64) * (x1 - x0)
            self.hits.append((x, point["frame_id"]))
            if not point["detected"]:
                previous = None
                continue
            values = (y1 - point["x"]/s.reader.width*(y1-y0), y1 - point["y"]/s.reader.height*(y1-y0))
            for axis, color in enumerate(("#84c9ff", ui.ACCENT)):
                if previous and timestamp - previous[2] <= .085:
                    c.create_line(previous[0], previous[1][axis], x, values[axis], fill=color, width=2)
                c.create_oval(x-2, values[axis]-2, x+2, values[axis]+2, fill=color, outline="")
            previous = x, values, timestamp

    def jump(self, event):
        if self.hits:
            _, frame = min(self.hits, key=lambda pair: abs(pair[0] - event.x))
            self.studio.pause()
            self.studio.seek(frame)

"""CPU-only native annotation window. Run --prepare or --smoke-test without labelling."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from annotation_core import (DEFAULT_ANNOTATION, DEFAULT_OUTPUT, DEFAULT_VIDEO, OutputLock,
                             Transform, prepare, read_image, write_image)

TITLE = "Pingpong Annotation"
EVENT_LABELS = {"table_bounce": "触台", "net_contact": "触网", "floor_contact": "触地",
                "racket_contact": "击球", "other": "其他"}
ZOOM_TITLE = "Original crop - click to refine (Z closes)"
MAIN_W, MAIN_H, SIDE_W = 1000, 564, 360
HEADER, FOOTER = 95, 155
CANVAS_W, CANVAS_H = MAIN_W + SIDE_W, HEADER + MAIN_H + FOOTER


def window_visible(name):
    """Native X can remove a Win32 window before HighGUI state is polled."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error as error:
        if error.code == -27 or 'NULL window' in str(error):
            return False
        raise


def close_window(name):
    """Closing an already removed window is harmless; other GUI errors still propagate."""
    try:
        cv2.destroyWindow(name)
    except cv2.error as error:
        if error.code != -27 and 'NULL window' not in str(error):
            raise


@lru_cache(maxsize=8)
def font(size):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def text(draw, pos, content, size=17, color=(220, 225, 235)):
    draw.text(pos, content, fill=color, font=font(size))


class Viewer:
    def __init__(self, session, annotation, *, tracking=True):
        self.session = session
        self.annotation = {item["frame_id"]: item for item in annotation["frames"]}
        self.tracking = tracking
        self.frame = session.progress["current_frame"]
        self.overlay = True
        self.playing = False
        self.pending = None
        self.range_start = self.range_end = None
        self.selected = None
        self.scroll = 0
        self.message = "完整查看视频；点球心后Enter保存；1—5选类型；C确认完成。"
        self.event_type = "table_bounce"
        self.show_reference = False
        self.exit_action = None
        self.zoom_open = False
        self.zoom_center = None
        self.zoom_transform = None
        self.transform = None
        self.rows = []
        self.dirty = True
        self.next_play = time.perf_counter()
        self.last_save = time.perf_counter()
        self.last_account = time.perf_counter()
        self.image = None
        self.load_frame(self.frame)

    def boxes(self, frame, label="ball"):
        item = self.annotation.get(frame, {})
        return [b for a in item.get("annotations", []) if a["label"] == label
                for b in a["bbox_xyxy"]]

    def load_frame(self, frame):
        frame = min(max(frame, 0), self.session.meta["n_frames"] - 1)
        image = read_image(self.session.frames_dir / f"{frame:06d}.png")
        if image.shape[:2] != (self.session.meta["height"], self.session.meta["width"]):
            raise ValueError("缓存图片尺寸异常，暂停查看。")
        self.image, self.frame = image, frame
        self.pending = None
        self.selected = None
        if self.tracking:
            self.session.set_frame(frame)
        boxes = self.boxes(frame)
        self.zoom_center = (((boxes[0][0] + boxes[0][2]) / 2,
                             (boxes[0][1] + boxes[0][3]) / 2) if boxes else
                            (self.session.meta["width"] / 2, self.session.meta["height"] / 2))
        self.dirty = True

    def account(self):
        now = time.perf_counter()
        if self.tracking and self.session.progress["completed_at"] is None:
            self.session.progress["window_seconds"] += now - self.last_account
        self.last_account = now

    def save(self):
        if self.tracking:
            self.account()
            self.session.save()
        self.last_save = time.perf_counter()

    def dialog(self, title, prompt, *, integer=False, initial=""):
        from gui_dialogs import ask
        self.playing = False
        try:
            return ask(title, prompt, integer=integer, initial=initial,
                       minimum=0, maximum=self.session.meta["n_frames"] - 1)
        finally:
            self.dirty = True

    def visible_records(self):
        records = [("event", e) for e in self.session.events]
        records += [("non_table", e) for e in self.session.state["non_table"] if e["round"] == self.session.round]
        records += [("uncertain", e) for e in self.session.state["uncertain"]
                    if e["round"] == self.session.round]
        return sorted(records, key=lambda item: item[1].get("frame_id", item[1].get("frame_start")))

    def render(self):
        meta, p = self.session.meta, self.session.progress
        scale = min(MAIN_W / meta["width"], MAIN_H / meta["height"])
        dw, dh = round(meta["width"] * scale), round(meta["height"] * scale)
        left, top = (MAIN_W - dw) // 2, HEADER + (MAIN_H - dh) // 2
        self.transform = Transform(left, top, dw, dh, 0, 0, meta["width"], meta["height"])
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (26, 23, 20)
        canvas[top:top + dh, left:left + dw] = cv2.resize(self.image, (dw, dh))

        def point(x, y):
            px, py = self.transform.to_display(x, y)
            return round(px), round(py)

        if self.overlay:
            for label, color in (("table", (170, 130, 40)), ("ball", (40, 230, 70))):
                for x1, y1, x2, y2 in self.boxes(self.frame, label):
                    cv2.rectangle(canvas, point(x1, y1), point(x2, y2), color, 1)
                    if label == "ball":
                        cv2.circle(canvas, point((x1+x2)/2, (y1+y2)/2), 2, color, -1)
            # Connect only consecutive labelled frames: gaps must remain visible.
            last = None
            for f in range(max(0, self.frame - 3), min(meta["n_frames"], self.frame + 4)):
                boxes = self.boxes(f)
                if len(boxes) != 1:
                    last = None
                    continue
                b = boxes[0]
                center = point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if last is not None:
                    cv2.line(canvas, last, center, (0, 200, 255), 1)
                last = center
        if self.show_reference:
            for e in self.session.state["events"]["initial"]:
                if e["frame_id"] == self.frame:
                    cv2.drawMarker(canvas, point(e["x"], e["y"]), (0, 80, 255), cv2.MARKER_DIAMOND, 20, 2)
        # Only current-round annotations appear until review is completed.
        for e in self.session.events:
            if e["frame_id"] == self.frame:
                cv2.circle(canvas, point(e["x"], e["y"]), 8, (255, 150, 0), 2)
        if self.pending:
            cv2.drawMarker(canvas, point(*self.pending), (255, 80, 255), cv2.MARKER_CROSS, 16, 2)

        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        labels = {"initial": "初标", "selfcheck": "个人自查（初标隐藏）", "review": "交叉复核（先独立判断）"}
        label = labels[self.session.round]
        member = self.session.state["participants"].get(self.session.round, "未登记")
        frozen = "【已结束，记录冻结】" if p["completed_at"] is not None else ""
        text(draw, (15, 8), f"{meta['video_id']}  |  {member} · {label} {frozen}", 22)
        text(draw, (15, 40), f"帧 {self.frame} / {meta['n_frames'] - 1}  ·  {meta['fps']:g} FPS  ·  原图 {meta['width']}×{meta['height']}")
        text(draw, (15, 66), f"已查看 {len(p['seen_frames'])}/{meta['n_frames']} · 完整检查需按C确认 · 当前类型 {EVENT_LABELS[self.event_type]}")
        draw.rectangle((MAIN_W, HEADER, CANVAS_W, HEADER+MAIN_H), fill=(32, 36, 44))
        text(draw, (MAIN_W + 12, HEADER + 6), "本轮事件与疑难（滚轮翻页）", 19)
        records = self.visible_records()
        capacity = (MAIN_H - 85) // 39
        self.scroll = max(0, min(self.scroll, max(0, len(records) - capacity)))
        self.rows = []
        for index, (kind, record) in enumerate(records[self.scroll:self.scroll + capacity]):
            y = HEADER + 40 + index * 39
            if self.selected == record["event_id"]:
                draw.rectangle((MAIN_W+4, y-2, CANVAS_W-4, y+33), fill=(55, 79, 103))
            if kind == "event":
                line = f"触台 f{record['frame_id']}  ({record['x']:.1f}, {record['y']:.1f})"
                target = record["frame_id"]
            elif kind == "non_table":
                line = f"{EVENT_LABELS[record['event_type']]} f{record['frame_id']}"
                target = record["frame_id"]
            else:
                line = f"疑难 f{record['frame_start']}—{record['frame_end']}"
                target = record["frame_start"]
            text(draw, (MAIN_W + 12, y), line, 16)
            self.rows.append((y - 2, y + 34, record["event_id"], target))
        text(draw, (MAIN_W + 12, HEADER + MAIN_H - 39), f"触台 {len(self.session.events)} · 干扰 {sum(k=='non_table' for k, _ in records)} · 疑难 {sum(k=='uncertain' for k, _ in records)}", 17)
        footer = HEADER + MAIN_H + 7
        for idx, line in enumerate((
            "A/D 前后帧 · 空格播放 · B 框/轨迹 · G 跳帧 · Z 放大 · 1触台 2触网 3触地 4击球 5其他",
            "点球心→Enter保存；干扰可不点坐标 · I/O/U疑难 · 点击列表→Delete/X删除 · K清草稿 · V可选",
            ("C 人工确认完成 · S保存 · Q/L视频列表 · N下一段 · E重新编辑 · H复核后对照"
             if self.session.state.get('completion_policy') == 'manual_attestation' else
             "C 确认完成 · S保存 · Q/L任务表 · N下一段 · E重新编辑 · H复核后对照 · R个人自查"))):
            text(draw, (12, footer + idx * 25), line, 17)
        pending = f"待保存点：({self.pending[0]:.2f}, {self.pending[1]:.2f})" if self.pending else "尚未点选球心"
        text(draw, (12, footer + 79), f"{pending}   疑难起止：{self.range_start} / {self.range_end}", 17)
        text(draw, (12, footer + 107), self.message[:88], 17, (255, 208, 120))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def zoom_image(self):
        w, h = self.session.meta["width"], self.session.meta["height"]
        sw, sh = min(96, w), min(96, h)
        cx, cy = self.zoom_center
        x0 = min(max(round(cx-sw/2), 0), w-sw)
        y0 = min(max(round(cy-sh/2), 0), h-sh)
        self.zoom_transform = Transform(0, 0, 480, 480, x0, y0, sw, sh)
        crop = self.image[y0:y0+sh, x0:x0+sw].copy()
        zoom = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST)
        for e in self.session.events:
            if e["frame_id"] == self.frame and x0 <= e["x"] < x0+sw and y0 <= e["y"] < y0+sh:
                x, y = self.zoom_transform.to_display(e["x"], e["y"])
                cv2.circle(zoom, (round(x), round(y)), 7, (255, 150, 0), 2)
        if self.pending and x0 <= self.pending[0] < x0+sw and y0 <= self.pending[1] < y0+sh:
            x, y = self.zoom_transform.to_display(*self.pending)
            cv2.drawMarker(zoom, (round(x), round(y)), (255, 80, 255), cv2.MARKER_CROSS, 18, 1)
        return zoom

    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL and x >= MAIN_W:
            self.scroll += -1 if flags > 0 else 1
            self.dirty = True
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.playing = False
        if x >= MAIN_W:
            for top, bottom, event_id, target in self.rows:
                if top <= y <= bottom:
                    self.load_frame(target)
                    self.selected = event_id
                    self.message = "已选中记录；可Delete/X删除。"
                    break
        else:
            point = self.transform.to_original(x, y) if self.transform else None
            if point:
                self.pending = point
                self.zoom_center = point
                self.message = "点选仅是草稿，确认触台后按Enter保存；Z放大查看。"
            else:
                self.message = "请在视频画面内点选。"
        self.dirty = True

    def zoom_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and self.zoom_transform:
            point = self.zoom_transform.to_original(x, y)
            if point:
                self.playing = False
                self.pending = point
                # Keep crop stable during fine clicks.
                self.message = "已在原图局部放大画面细调点；Enter保存。"
                self.dirty = True

    def confirm(self, title, prompt):
        from gui_dialogs import ask
        self.playing = False
        try:
            return bool(ask(title, prompt, confirm=True))
        finally:
            self.dirty = True

    def sync_zoom_window(self):
        if self.zoom_open and not window_visible(ZOOM_TITLE):
            self.zoom_open = False
            self.zoom_transform = None
            self.message = '放大窗口已关闭；按Z重新打开。'
            self.dirty = True

    def toggle_zoom(self):
        # Recheck native state: Z may arrive immediately after the user clicks X.
        if self.zoom_open and window_visible(ZOOM_TITLE):
            close_window(ZOOM_TITLE)
            self.zoom_open = False
            self.zoom_transform = None
        else:
            self.zoom_open = False
            cv2.namedWindow(ZOOM_TITLE, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(ZOOM_TITLE, self.zoom_mouse)
            self.zoom_open = True

    def handle_key(self, key):
        if key < 0:
            return True
        character = chr(key).lower() if 0 <= key < 256 else ""
        try:
            if character in ("q", "l", "n") or key == 27:
                if (self.pending or self.range_start is not None or self.range_end is not None) and not self.confirm("未保存草稿", "当前点选或疑难区间尚未保存，是否放弃草稿并离开？"):
                    return True
                self.exit_action = "next" if character == "n" else "tasks"
                return False
            if character in "12345" and character:
                self.event_type = {"1": "table_bounce", "2": "net_contact", "3": "floor_contact",
                                   "4": "racket_contact", "5": "other"}[character]
                self.message = f"当前类型：{EVENT_LABELS[self.event_type]}；Enter保存。"
            elif character in ("a", "d") or key in (2424832, 2555904, 65361, 65363):
                self.playing = False
                direction = -1 if character == "a" or key in (2424832, 65361) else 1
                self.load_frame(self.frame + direction)
            elif key == 32:
                self.playing = not self.playing
                self.next_play = time.perf_counter() + 1 / self.session.meta["fps"]
            elif character == "b":
                self.overlay = not self.overlay
            elif character == "z":
                self.toggle_zoom()
            elif character == "g":
                target = self.dialog("跳转", "输入0基帧号：", integer=True)
                if target is not None:
                    self.load_frame(target)
            elif character == "v":
                self.playing = False
                self.session.mark_reviewed()
                self.load_frame(self.frame + 1)
                self.message = "该帧已确认检查；到尾帧后按C结束本轮。"
                self.save()
            elif key in (13, 10):
                self.playing = False
                if self.event_type == "table_bounce":
                    if not self.pending:
                        raise ValueError("先在视频或放大图中点选可见球心；看不清用疑难区间。")
                    self.session.put_event(self.frame, *self.pending)
                else:
                    reason = self.dialog("干扰事件依据", "简述判断依据；没有可见球心时可以不填坐标：")
                    if reason is None:
                        return True
                    x, y = self.pending if self.pending else (None, None)
                    self.session.put_non_table(self.frame, self.event_type, x, y, reason)
                self.session.mark_reviewed()
                self.pending = None
                self.message = "记录已保存；同帧同类型再次保存会更新。"
                self.save()
            elif character == "i":
                self.playing = False
                self.range_start = self.frame
            elif character == "o":
                self.playing = False
                self.range_end = self.frame
            elif character == "u":
                self.session.ensure_editable()
                if self.range_start is None or self.range_end is None:
                    raise ValueError("先用I和O分别标记疑难起止帧。")
                if self.range_start > self.range_end:
                    raise ValueError("疑难起始帧不能大于结束帧。")
                reason = self.dialog("疑难原因", "填写原因，例如遮挡、两帧之间接触、球心无法判断：")
                if reason is not None:
                    self.session.put_uncertain(self.range_start, self.range_end, reason)
                    self.range_start = self.range_end = None
                    self.message = "疑难区间已保存，未强填触台帧和坐标。"
                    self.save()
            elif character == "x" or key in (8, 127, 3014656, 65535):
                if not self.selected:
                    raise ValueError("先点击右侧事件或疑难列表选择记录。")
                self.session.delete(self.selected)
                self.selected = None
                self.message = "已删除所选记录。"
                self.save()
            elif character == "s":
                self.save()
                self.message = "标签和进度已保存。"
            elif character == "k":
                self.pending = None
                self.range_start = self.range_end = None
                self.message = "已清除未保存草稿，已有记录保持不变。"
            elif character == "c":
                self.playing = False
                if self.pending or self.range_start is not None or self.range_end is not None:
                    raise ValueError("尚有未保存点/疑难草稿，先Enter/U保存，或K清除。")
                if (self.session.state.get("completion_policy") != "manual_attestation"
                        and len(self.session.progress["seen_frames"]) != self.session.meta["n_frames"]):
                    raise ValueError("尚有未查看帧，请完整查看；无需每帧按V。")
                if self.session.progress["completed_at"] is not None:
                    self.message = "本阶段已完成；H查看复核对照，E可保留快照后重开。"
                    return True
                if not self.confirm("确认完整检查", "确认已完整检查此视频，并记录确定事件及疑难？播放过不等于已判断。"):
                    return True
                reason = ""
                if not self.session.events:
                    reason = self.dialog("无触台说明", "本阶段没有触台记录，请说明原因：")
                    if reason is None:
                        return True
                self.account()
                self.session.complete(reason, inspected_all=True)
                self.save()
                self.message = "本阶段已完成并冻结；初标不等于交叉复核。N下一段，Q/L任务表。"
            elif character == "r":
                if self.session.state.get("completion_policy") == "manual_attestation":
                    raise ValueError("本地模式请返回视频列表选择初标或独立复核；不切换个人自查。")
                self.playing = False
                self.account()
                self.session.switch_round("selfcheck")
                self.show_reference = False
                self.range_start = self.range_end = None
                self.scroll = 0
                self.load_frame(self.session.progress["current_frame"])
                self.save()
                self.message = "已进入可选个人自查；无需等待，初标隐藏。组内复核请从任务表启动。"
            elif character == "e":
                reason = self.dialog("重新编辑", "请填写重新编辑原因；会先保存旧版本快照：")
                if reason is not None:
                    self.session.reopen(reason)
                    self.show_reference = False
                    self.save()
                    self.message = "已保留历史快照并重新打开本阶段。"
            elif character == "h":
                if self.session.round != "review" or self.session.progress["completed_at"] is None:
                    raise ValueError("先完成独立交叉复核，再显示初标对照。")
                self.show_reference = not self.show_reference
                result = self.session.state
                from annotation_core import compare_events
                diff = compare_events(result["events"]["initial"], self.session.events)
                self.message = f"初标用红色菱形：配对{len(diff['matches'])}，仅初标{len(diff['initial_only'])}，仅复核{len(diff['selfcheck_only'])}；详情见review_comparison.json。"
        except ValueError as error:
            self.message = str(error)
            self.playing = False
        self.dirty = True
        return True

    def run(self, smoke_test=False):
        cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(TITLE, self.mouse)
        start = time.perf_counter()
        try:
            while True:
                if not window_visible(TITLE):
                    break
                self.sync_zoom_window()
                if self.dirty:
                    frame = self.render()
                    cv2.imshow(TITLE, frame)
                    if self.zoom_open:
                        cv2.imshow(ZOOM_TITLE, self.zoom_image())
                    # Mark seen only after rendering. Smoke tests never count as review.
                    if self.tracking:
                        self.session.mark_seen()
                    self.dirty = False
                key = cv2.waitKeyEx(5)
                if smoke_test and time.perf_counter() - start >= 1.5:
                    break
                if not smoke_test and not self.handle_key(key):
                    break
                now = time.perf_counter()
                if self.playing and now >= self.next_play:
                    if self.frame >= self.session.meta["n_frames"] - 1:
                        self.playing = False
                    else:
                        self.load_frame(self.frame + 1)
                    # Never skip frames if rendering is slower than the video.
                    self.next_play = max(self.next_play + 1 / self.session.meta["fps"], now)
                if self.tracking and now - self.last_save >= 15:
                    self.save()
        finally:
            if self.tracking:
                self.save()
            cv2.destroyAllWindows()
        return self.exit_action


def run_command(args):
    session, annotation = prepare(args.video, args.annotation, args.output,
                                  rebuild_cache=args.rebuild_cache)
    if args.annotator:
        existing_member = session.state["participants"].get(args.round or session.round)
        if existing_member and existing_member != args.annotator:
            raise ValueError("此工作目录已有其他成员记录，请使用独立输出目录。")
        session.state["participants"][args.round or session.round] = args.annotator
    if args.round:
        session.switch_round(args.round)
    if args.prepare or args.report:
        session.save()
        print(json.dumps({"ok": True, "video": session.meta, "output": str(session.output)}, ensure_ascii=False))
        return 0
    if args.preview:
        # Previews use a deep copy, so validation cannot create personal review evidence.
        viewer = Viewer(copy.deepcopy(session), annotation, tracking=False)
        for frame in (0, min(8, session.meta["n_frames"]-1), session.meta["n_frames"]-1):
            viewer.load_frame(frame)
            write_image(session.output / f"preview_{frame:06d}.png", viewer.render())
        write_image(session.output / "preview_zoom.png", viewer.zoom_image())
        print("UI preview saved; no annotation/review progress changed.")
        return 0
    if args.round and not args.smoke_test:
        session.save()
    viewer = Viewer(session, annotation, tracking=not args.smoke_test)
    viewer.run(smoke_test=args.smoke_test)
    print("GUI smoke test passed (no personal annotation/review recorded)." if args.smoke_test else "标签与查看进度已保存。")
    return 0


def main():
    # Keep Chinese diagnostics readable through Windows terminals and captured pipes.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="单视频触台/干扰/疑难标注；批量任务用team_annotation.py")
    parser.add_argument("--annotator", help="当前操作者姓名或编号")
    parser.add_argument("--rebuild-cache", action="store_true", help="输入哈希一致时重建PNG，不删除标签")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round", choices=("initial", "selfcheck"))
    parser.add_argument("--prepare", action="store_true", help="仅准备PNG并核对数据")
    parser.add_argument("--report", action="store_true", help="从session重建导出与报告")
    parser.add_argument("--smoke-test", action="store_true", help="短暂打开窗口并退出，不修改查看进度")
    parser.add_argument("--preview", action="store_true", help="保存首帧/缺球框帧/尾帧的UI预览，不计查看进度")
    args = parser.parse_args()
    try:
        with OutputLock(args.output):
            return run_command(args)
    except (ValueError, OSError, cv2.error) as error:
        print(f"暂停：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Native table-corner review, compatible with the existing reference evaluator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from annotation_core import OutputLock, write_image
from annotate import close_window, font, window_visible
from gui_dialogs import ask, pick_file

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Pingpong - Table Corner Review'
W, H = 1400, 900
STATUSES = {
    'corners_visible': '四角可见',
    'occluded': '角点被遮挡',
    'thin_table': '台面太薄，看不清',
    'no_table': '没有球台',
    'ambiguous_target': '无法确定目标球台',
}
SOURCE = 'user_submitted_single_review_not_double_checked'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def default_pack():
    candidates = [ROOT / 'outputs/experiments/geometry_reference_audit_20261013/review_pack']
    candidates.extend(sorted((ROOT / 'worktrees').glob(
        '*/outputs/experiments/geometry_reference_audit_20261013/review_pack')))
    return next((p for p in candidates if (p / 'manifest.json').is_file()), None)


def validate_reference(value, cases):
    """Keep the web evaluator's schema and original-frame coordinate convention."""
    if not isinstance(value, dict) or value.get('case_id') not in cases:
        raise ValueError('未找到核查样本。')
    case = cases[value['case_id']]
    status = value.get('status')
    if status not in STATUSES:
        raise ValueError('请选择台面可辨识情况。')
    reviewer = value.get('reviewer', '')
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer.strip()) > 80:
        raise ValueError('请填写核查者姓名或简称，不超过80字。')
    points = value.get('corners', [])
    if status == 'corners_visible':
        try:
            q = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('角点格式不正确。') from error
        if q.shape != (4, 2) or not np.isfinite(q).all():
            raise ValueError('请沿台面边界顺序点选完整四角。')
        if np.any(q < 0) or np.any(q[:, 0] >= case['width']) or np.any(q[:, 1] >= case['height']):
            raise ValueError('角点超出原图范围。')
        if not cv2.isContourConvex(q.astype(np.float32)) or cv2.contourArea(q.astype(np.float32)) < 1:
            raise ValueError('四角不能交叉、重合或共线，请沿边界顺序点选。')
        points = q.tolist()
    elif points != []:
        raise ValueError('角点无法确认时请清空点选，不保存猜测角点。')
    return {
        'case_id': case['case_id'], 'group': case['group'], 'video_id': case['video_id'],
        'frame_id': case['frame_id'], 'image_sha256': case['image_sha256'],
        'video_sha256': case['video_sha256'], 'status': status, 'corners': points,
        'reviewer': reviewer.strip(), 'reference_source': SOURCE,
        'submitted_utc': datetime.now(timezone.utc).isoformat(),
    }


class ReferenceStore:
    def __init__(self, pack):
        self.pack = Path(pack).resolve()
        self.manifest_path = self.pack / 'manifest.json'
        self.manifest = read(self.manifest_path)
        self.manifest_hash = sha(self.manifest_path)
        self.cases = self.manifest['cases']
        self.by_id = {c['case_id']: c for c in self.cases}
        if not self.cases or len(self.cases) != len(self.by_id):
            raise ValueError('核查清单为空或样本编号重复。')
        for case in self.cases:
            if (type(case['width']) is not int or type(case['height']) is not int
                    or min(case['width'], case['height']) <= 0
                    or type(case['frame_id']) is not int or case['frame_id'] < 0):
                raise ValueError('核查清单尺寸或帧号不正确。')
            self.image_path(case)
        self.path = self.pack / 'references.jsonl'
        self.latest = self.load()

    def image_path(self, case):
        path = (self.pack / case['image_url']).resolve()
        if not path.is_relative_to(self.pack):
            raise ValueError('原帧路径超出核查包。')
        return path

    def image(self, case):
        if sha(self.manifest_path) != self.manifest_hash:
            raise ValueError('核查清单已变化，请重新打开核查窗口。')
        path = self.image_path(case)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != case['image_sha256']:
            raise ValueError('原帧哈希已变化，停止核查以免错配。')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (case['height'], case['width']):
            raise ValueError('原帧尺寸或解码结果不匹配。')
        return image

    def load(self):
        latest = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                valid = validate_reference(record, self.by_id)
                if any(record.get(k) != valid[k] for k in
                       ('group', 'video_id', 'frame_id', 'image_sha256', 'video_sha256', 'reference_source')):
                    raise ValueError('已有核查记录与原帧来源不符。')
                latest[record['case_id']] = record
        return latest

    def save(self, value):
        record = validate_reference(value, self.by_id)
        # Verify the source again at submission, even if it changed after display.
        self.image(self.by_id[record['case_id']])
        with OutputLock(self.pack):
            latest = self.load()
            data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            with self.path.open('ab') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            latest[record['case_id']] = record
            self.latest = latest
        return record


class GeometryReviewWindow:
    def __init__(self, store, *, video_id=None, frame_id=None, group=None):
        self.store = store
        matches = [(i, c) for i, c in enumerate(store.cases)
                   if c['video_id'] == video_id and (group is None or c['group'] == group)]
        self.index = min(matches, key=lambda item: abs(item[1]['frame_id'] - (frame_id or 0)))[0] if matches else 0
        self.reviewer = ''
        self.message = ('已打开核查包中的最近样本帧，帧号以标题为准。' if matches else
                        '本核查包没有当前视频，显示全部核查样本。' if video_id else
                        '沿台面上缘顺序点选四角；遮挡角点请登记遮挡。')
        self.buttons, self.rows = [], []
        self.transform = None
        self.pending = None
        self.running = True
        self.scroll = max(0, self.index - 4)
        self.drag = None
        self.load_case(self.index)

    @property
    def case(self):
        return self.store.cases[self.index]

    def load_case(self, index):
        image = self.store.image(self.store.cases[index])
        # Refresh submissions made since this window was opened.
        self.store.latest = self.store.load()
        saved = self.store.latest.get(self.store.cases[index]['case_id'])
        self.index, self.original = index, image
        self.points = [list(p) for p in saved['corners']] if saved else []
        self.status = saved['status'] if saved else 'corners_visible'
        self.saved = saved
        self.dirty = False
        self.zoom = 1
        self.center = (self.case['width'] / 2, self.case['height'] / 2)
        self.drag = None
        self.transform = None

    def original_point(self, x, y):
        if self.transform is None:
            return None
        left, top, dw, dh, ox, oy, vw, vh = self.transform
        if left <= x < left + dw and top <= y < top + dh:
            return [ox + (x - left) * vw / dw, oy + (y - top) * vh / dh]
        return None

    def draw(self):
        image = Image.new('RGB', (W, H), (24, 28, 35))
        paint = ImageDraw.Draw(image)
        self.buttons, self.rows = [], []
        def label(x, y, text, size=17, color=(230, 234, 241)):
            paint.text((x, y), str(text), font=font(size), fill=color)
        def button(x, y, width, text, action, active=False):
            box = (x, y, x + width, y + 36)
            paint.rectangle(box, fill=(52, 90, 125) if active else (45, 51, 62))
            label(x + 9, y + 7, text)
            self.buttons.append((*box, action))
        label(18, 12, '人工四角核查', 25)
        label(18, 50, f"{self.case['group']} / {self.case['video_id']} · 原始帧 {self.case['frame_id']}（0基）"
              f" · {self.case['width']}×{self.case['height']} · 样本 {self.index+1}/{len(self.store.cases)}")
        label(18, 78, '仅显示原帧及人工点选。点台面上缘四角，顺时针或逆时针均可。', 16)
        vh, vw = max(1, int(self.case['height'] / self.zoom)), max(1, int(self.case['width'] / self.zoom))
        ox = int(np.clip(self.center[0] - vw / 2, 0, self.case['width'] - vw))
        oy = int(np.clip(self.center[1] - vh / 2, 0, self.case['height'] - vh))
        scale = min(968 / vw, 558 / vh)
        dw, dh = max(1, round(vw * scale)), max(1, round(vh * scale))
        left, top = 16 + (968 - dw) // 2, 108 + (558 - dh) // 2
        frame = Image.fromarray(cv2.cvtColor(self.original[oy:oy+vh, ox:ox+vw], cv2.COLOR_BGR2RGB))
        image.paste(frame.resize((dw, dh), Image.Resampling.LANCZOS), (left, top))
        self.transform = (left, top, dw, dh, ox, oy, vw, vh)
        mapped = [(left + (p[0]-ox) * dw/vw, top + (p[1]-oy) * dh/vh) for p in self.points]
        # Draw into a clipped overlay so offscreen points cannot cover the controls.
        layer = Image.new('RGBA', (dw, dh), (0, 0, 0, 0))
        overlay = ImageDraw.Draw(layer)
        local = [(x-left, y-top) for x, y in mapped]
        if len(local) > 1:
            overlay.line(local + ([local[0]] if len(local) == 4 else []), fill=(70, 235, 190), width=2)
        for i, (x, y) in enumerate(local):
            overlay.ellipse((x-5, y-5, x+5, y+5), fill=(255, 205, 90), outline='black', width=1)
            overlay.text((x+8, y-22), str(i+1), font=font(20), fill=(255, 230, 130))
        image.paste(layer, (left, top), layer)
        for x, width, text, action in (
            (18, 132, '保存 S', 'save'), (160, 126, '上一个 P', 'previous'),
            (296, 126, '下一个 N', 'next'), (432, 126, '撤销一点', 'undo'),
            (568, 126, '清空草稿', 'clear'), (704, 126, f'放大 {self.zoom}× Z', 'zoom'),
            (840, 140, '原图全貌 0', 'reset')):
            button(x, 695, width, text, action)
        label(18, 745, '左键点选 / 拖动已选角点 · 右键定位放大中心 · S保存 · Q返回', 16)
        label(18, 775, self.message[:84], 17, (255, 210, 128))
        button(18, 809, 215, '球网与方向核查 G', 'net_review')
        label(245, 818, '保存文件：核查包内 references.jsonl', 15)
        pack_label = str(self.store.pack)
        label(18, 835, '核查包：' + (pack_label if len(pack_label) <= 111 else '…' + pack_label[-110:]), 13)
        label(18, 864, '已修改，尚未保存' if self.dirty else '当前无未保存修改', 16,
              (255, 210, 128) if self.dirty else (150, 215, 170))
        label(1015, 12, '核查者：' + (self.reviewer or '尚未填写'), 18)
        button(1015, 49, 365, '填写 / 更换核查者', 'reviewer')
        button(1015, 95, 365, '打开其他核查包', 'pack')
        label(1015, 144, f'已登记 {len(self.store.latest)}/{len(self.store.cases)} · 点选 {len(self.points)}/4', 18)
        for i, (status, title) in enumerate(STATUSES.items()):
            button(1015, 182+i*42, 365, title, 'status:' + status, self.status == status)
        for i, point in enumerate(self.points):
            label(1015, 401+i*27, f'{i+1}: 原图 ({point[0]:.2f}, {point[1]:.2f})', 16)
        if self.saved:
            label(1015, 514, '上次保存：' + self.saved['reviewer'][:18], 15, (150, 215, 170))
        label(1015, 541, '核查样本（滚轮翻页）', 17)
        self.scroll = min(max(0, self.scroll), max(0, len(self.store.cases)-9))
        for i in range(self.scroll, min(len(self.store.cases), self.scroll+9)):
            case = self.store.cases[i]
            y = 571 + (i-self.scroll)*32
            if i == self.index:
                paint.rectangle((1008, y-2, 1386, y+29), fill=(43, 65, 85))
            marker = '已存' if case['case_id'] in self.store.latest else '待核查'
            label(1015, y, f"{i+1:02d} {case['video_id']}  f{case['frame_id']}  {marker}", 16)
            self.rows.append((1008, y-2, 1386, y+29, i))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def action(self, action):
        if action == 'net_review':
            if self.dirty:
                self.message = '当前有未保存修改，请先保存或清空草稿，再进入球网核查。'
                return
            from net_geometry_review import open_net_review
            close_window(TITLE)
            try:
                open_net_review(video_id=self.case['video_id'],frame_id=self.case['frame_id'],group=self.case['group'])
            finally:
                cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(TITLE,1260,810)
                cv2.setMouseCallback(TITLE,self.mouse)
        elif action == 'save':
            if not self.reviewer:
                self.action('reviewer')
            self.saved = self.store.save(dict(case_id=self.case['case_id'], reviewer=self.reviewer,
                                               status=self.status, corners=self.points))
            self.dirty = False
            self.message = '已保存当前原始帧参考；重复保存追加历史，下次打开可继续。'
        elif action == 'reviewer':
            value = ask('核查者', '填写实际核查者姓名或简称', initial=self.reviewer)
            if value is not None:
                if not value.strip() or len(value.strip()) > 80:
                    raise ValueError('核查者不能为空，且不超过80字。')
                self.reviewer = value.strip()
        elif action in ('next', 'previous') or action.startswith('case:'):
            if self.dirty:
                self.message = '当前有未保存修改，请先保存，或点击清空草稿恢复上次记录。'
                return
            index = (int(action.split(':')[1]) if action.startswith('case:')
                     else self.index + (1 if action == 'next' else -1))
            self.load_case(max(0, min(len(self.store.cases)-1, index)))
            self.scroll = max(0, self.index-4)
            self.message = '已切换原始帧。'
        elif action.startswith('status:'):
            status = action.split(':')[1]
            if status not in STATUSES:
                raise ValueError('未知核查状态。')
            if status != self.status:
                self.status = status
                if status != 'corners_visible':
                    self.points = []
                self.dirty = True
                self.message = '状态已修改，按S保存。不可辨识状态不保存猜测角点。'
        elif action == 'undo':
            if self.points:
                self.points.pop()
                self.dirty = True
        elif action == 'clear':
            # Explicitly discard the draft and restore the saved record.
            self.load_case(self.index)
            self.message = '未保存草稿已清除，已恢复上次保存记录。'
        elif action == 'zoom':
            self.zoom = {1: 2, 2: 4, 4: 1}[self.zoom]
            self.transform = None
        elif action == 'reset':
            self.zoom = 1
            self.transform = None
        elif action in ('quit', 'pack'):
            if self.dirty:
                self.message = '有未保存修改，请先保存或清空草稿，再返回 / 更换核查包。'
                return
            if action == 'quit':
                self.running = False
            else:
                path = pick_file('选择四角核查包的manifest.json', file_filter='JSON|*.json')
                if path:
                    replacement = ReferenceStore(Path(path).parent)
                    replacement.image(replacement.cases[0])
                    self.store = replacement
                    self.scroll = 0
                    self.load_case(0)
                    self.message = '已打开新核查包。'

    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL and x >= 1000 and y >= 560:
            delta = (flags >> 16) & 0xffff
            if delta >= 0x8000:
                delta -= 0x10000
            self.scroll += -3 if delta > 0 else 3
            return
        point = self.original_point(x, y)
        if event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            if point:
                self.points[self.drag] = point
                self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None
            return
        if event == cv2.EVENT_RBUTTONDOWN and point:
            self.center = tuple(point)
            self.transform = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x0, y0, x1, y1, action in self.buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.pending = action
                return
        for x0, y0, x1, y1, index in self.rows:
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.pending = 'case:' + str(index)
                return
        if point and self.status == 'corners_visible':
            if self.zoom == 1:
                self.center = tuple(point)
            # Hit tolerance is measured in displayed pixels, independent of zoom.
            _, _, dw, dh, _, _, vw, vh = self.transform
            for i, p in enumerate(self.points):
                if np.hypot((p[0]-point[0])*dw/vw, (p[1]-point[1])*dh/vh) <= 10:
                    self.drag = i
                    return
            if len(self.points) < 4:
                self.points.append(point)
                self.dirty = True
                self.message = f'已点选{len(self.points)}/4角；四角齐全后按S保存。'
            else:
                self.message = '已有四角，可拖动黄色角点调整，或撤销一点重新点选。'

    def key(self, key):
        if 65 <= key <= 90:
            key += 32
        actions = {ord('s'): 'save', 13: 'save', ord('n'): 'next', ord('p'): 'previous',
                   ord('z'): 'zoom', ord('0'): 'reset', 8: 'undo', ord('k'): 'clear',
                   ord('q'): 'quit', 27: 'quit'}
        actions[ord('g')] = 'net_review'
        if key in actions:
            self.pending = actions[key]

    def run(self, *, smoke_seconds=0):
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, 1260, 810)
        cv2.setMouseCallback(TITLE, self.mouse)
        deadline = time.perf_counter() + smoke_seconds if smoke_seconds else None
        try:
            while self.running:
                cv2.imshow(TITLE, self.draw())
                self.key(cv2.waitKeyEx(30))
                if not window_visible(TITLE):
                    if self.dirty:
                        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(TITLE, 1260, 810)
                        cv2.setMouseCallback(TITLE, self.mouse)
                        self.message = '有未保存修改，请先保存或清空草稿，再关闭窗口。'
                    else:
                        break
                action, self.pending = self.pending, None
                if action:
                    try:
                        self.action(action)
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        self.message = str(error)
                if deadline and time.perf_counter() >= deadline:
                    break
        finally:
            close_window(TITLE)


def open_review(*, pack=None, video_id=None, frame_id=None, group=None, smoke_seconds=0):
    path = Path(pack) if pack else default_pack()
    if path is None:
        chosen = pick_file('选择四角核查包的manifest.json', file_filter='JSON|*.json')
        if not chosen:
            return
        path = Path(chosen).parent
    GeometryReviewWindow(ReferenceStore(path), video_id=video_id, frame_id=frame_id, group=group).run(
        smoke_seconds=smoke_seconds)


def main():
    parser = argparse.ArgumentParser(description='原生桌面人工四角核查')
    parser.add_argument('--pack', type=Path, default=default_pack())
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--smoke-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.pack is None:
        open_review()
        return
    window = GeometryReviewWindow(ReferenceStore(args.pack))
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        write_image(args.preview, window.draw())
    else:
        window.run(smoke_seconds=args.smoke_seconds)


if __name__ == '__main__':
    main()

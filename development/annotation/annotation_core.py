"""State, sequential decoding, coordinates and comparison for the annotation pilot."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT / "examples/training/03_007/video.mp4"
DEFAULT_ANNOTATION = PROJECT / "examples/training/03_007/annotation.json"
DEFAULT_OUTPUT = PROJECT / "outputs/annotation_pilot/03_007"
ROUNDS = ("initial", "selfcheck", "review")
SELF_CHECK_DELAY = 0  # Kept for old imports; team review has no waiting period.
EVENT_TYPES = {"table_bounce", "net_contact", "floor_contact", "racket_contact", "other"}


class OutputLock:
    """Prevent two viewers/report exporters from overwriting the same session."""
    def __init__(self, output):
        self.output = Path(output)
        self.stream = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = (self.output / ".session.lock").open("a+b")
        if self.stream.seek(0, 2) == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError("该输出目录已有工具在使用；请先关闭另一个标注窗口。") from error
        return self

    def __exit__(self, *_):
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    # Windows scanners may briefly hold a newly written file without delete sharing.
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.03 * (attempt + 1))


def write_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path: Path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片无法读取：{path}")
    return image


def write_image(path: Path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"PNG编码失败：{path}")
    encoded.tofile(str(path))


@dataclass(frozen=True)
class Transform:
    """A displayed rectangle maps to a full-image rectangle or zoom ROI."""
    left: int
    top: int
    display_width: int
    display_height: int
    origin_x: int
    origin_y: int
    source_width: int
    source_height: int

    def to_original(self, x, y):
        if not (self.left <= x < self.left + self.display_width
                and self.top <= y < self.top + self.display_height):
            return None
        return (self.origin_x + (x - self.left) * self.source_width / self.display_width,
                self.origin_y + (y - self.top) * self.source_height / self.display_height)

    def to_display(self, x, y):
        return (self.left + (x - self.origin_x) * self.display_width / self.source_width,
                self.top + (y - self.origin_y) * self.display_height / self.source_height)


def prepare(video=DEFAULT_VIDEO, annotation=DEFAULT_ANNOTATION, output=DEFAULT_OUTPUT,
            *, frames_dir=None, rebuild_cache=False, completion_policy=None):
    """Verify immutable inputs and sequentially rebuild a portable frame cache."""
    video, annotation, output = map(Path, (video, annotation, output))
    started = time.perf_counter()
    declared = read_json(annotation)
    inputs = {"video": str(video.resolve()), "annotation": str(annotation.resolve()),
              "video_sha256": sha256(video), "annotation_sha256": sha256(annotation)}
    output.mkdir(parents=True, exist_ok=True)
    session_path = output / "session.json"
    existing = read_json(session_path) if session_path.exists() else None
    if existing:
        if any(existing["inputs"].get(k) != inputs[k]
               for k in ("video_sha256", "annotation_sha256")):
            raise ValueError("输入文件哈希已改变，请使用新的输出目录；不覆盖既有标签。")
        meta = existing["video"]
        if (meta["video_id"] != declared["video_id"]
                or (meta["n_frames"], meta["width"], meta["height"]) !=
                (declared["num_frames"], declared["frame_width"], declared["frame_height"])):
            raise ValueError("旧标签与当前视频元数据不一致，暂停恢复。")
    if frames_dir is not None:
        frames = Path(frames_dir)
    else:
        saved_cache = Path(existing.get("cache", {}).get("directory", output / "frames")) if existing else output / "frames"
        frames = saved_cache if saved_cache.is_dir() else output / "frames"
    expected_size = (declared["frame_width"], declared["frame_height"])
    expected_count = declared["num_frames"]
    if [item["frame_id"] for item in declared["frames"]] != list(range(expected_count)):
        raise ValueError("JSON帧号不是0基连续帧号，暂停标注。")
    frames.parent.mkdir(parents=True, exist_ok=True)
    identity = {"video_sha256": inputs["video_sha256"],
                "annotation_sha256": inputs["annotation_sha256"],
                "n_frames": expected_count, "size": list(expected_size)}
    # A second annotator may reuse frames, but never a different video with the same ID.
    with OutputLock(frames.parent / ("." + frames.name + "-cache-lock")):
        marker_path = frames / "cache.json"
        marker = read_json(marker_path) if marker_path.exists() else None
        if marker and marker.get("identity") != identity:
            raise ValueError("缓存属于不同输入，请换缓存目录；不覆盖其他视频。")
        ready = all((frames / f"{i:06d}.png").is_file() for i in range(expected_count))
        trusted = marker is not None or (existing is not None and frames_dir is None)
        if not ready or not trusted or rebuild_cache:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError("无法打开MP4，请检查视频路径和OpenCV后端。")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            backend = cap.getBackendName()
            sizes, count = set(), 0
            # Build separately: decoding failure must not damage labels or a valid cache.
            with tempfile.TemporaryDirectory(prefix="frames-build-", dir=frames.parent) as temp:
                staged = Path(temp)
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok:
                            break
                        sizes.add((image.shape[1], image.shape[0]))
                        write_image(staged / f"{count:06d}.png", image)
                        count += 1
                finally:
                    cap.release()
                issues = []
                if count != expected_count:
                    issues.append(f"实际解码{count}帧，JSON声明{expected_count}帧")
                if sizes != {expected_size}:
                    issues.append(f"实际尺寸{sorted(sizes)}，JSON声明{expected_size}")
                if reported_count != count:
                    issues.append(f"容器报告{reported_count}帧，但解码{count}帧")
                if not math.isfinite(fps) or fps <= 0:
                    issues.append("未读取到有效视频帧率")
                if existing and (count, *expected_size, fps) != (
                        existing["video"]["n_frames"], existing["video"]["width"],
                        existing["video"]["height"], existing["video"]["fps"]):
                    issues.append("重新解码信息与旧会话不一致")
                validation = {"ok": not issues, "issues": issues, "decoded_frames": count,
                              "reported_frames": reported_count, "decoded_sizes": sorted(sizes),
                              "fps": fps, "backend": backend, "inputs": inputs,
                              "prepare_seconds": time.perf_counter() - started}
                write_json(output / "decode_validation.json", validation)
                if issues:
                    raise ValueError("数据核对失败，保留原标签：" + "；".join(issues))
                frames.mkdir(parents=True, exist_ok=True)
                for i in range(count):
                    (staged / f"{i:06d}.png").replace(frames / f"{i:06d}.png")
                marker = {"identity": identity, "fps": fps, "backend": backend}
                write_json(marker_path, marker)
        elif marker is None:
            marker = {"identity": identity, "fps": existing["video"]["fps"], "backend": "legacy"}
            write_json(marker_path, marker)
    if existing:
        state = existing
        if completion_policy is not None:
            state['completion_policy'] = completion_policy
        state["inputs"] = inputs  # Same hashes; local paths may change across computers.
        session = Session(output, state)
        session.frames_dir = frames
        # Import standalone legacy interference records without changing their provenance.
        non_table = output / "non_table_events.jsonl"
        if non_table.exists() and not session.state["non_table"]:
            for line in non_table.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    record = json.loads(line)
                    record.setdefault("round", "initial")
                    session.state["non_table"].append(record)
            session.validate()
        return session, declared
    meta = {"video_id": declared["video_id"], "width": expected_size[0],
            "height": expected_size[1], "n_frames": expected_count, "fps": marker["fps"]}
    state = {"schema_version": 2, "inputs": inputs, "video": meta,
             "created_at": time.time(), "current_round": "initial",
             "rounds": {}, "events": {}, "uncertain": [], "non_table": [],
             "participants": {}, "timing": {"decode_seconds": time.perf_counter() - started},
             "label_purpose": "team_exploration_pending_review"}
    if completion_policy is not None:
        state['completion_policy'] = completion_policy
    session = Session(output, state)
    session.frames_dir = frames
    session.save()
    return session, declared


def compare_events(first, second):
    """Maximum-cardinality one-to-one matching; nearest-first alone is insufficient."""
    edges = {}
    for i, event in enumerate(first):
        candidates = []
        for j, other in enumerate(second):
            dt = abs(event["frame_id"] - other["frame_id"])
            distance = math.hypot(event["x"] - other["x"], event["y"] - other["y"])
            if dt <= 1 and distance <= 50:
                candidates.append((dt, distance, j))
        edges[i] = [j for _, _, j in sorted(candidates)]
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in range(len(first)):
        augment(i, set())
    pairs = sorted((i, j) for j, i in owners.items())
    matched_first, matched_second = {i for i, _ in pairs}, {j for _, j in pairs}
    return {"matches": [{"initial_event_id": first[i]["event_id"],
                         "selfcheck_event_id": second[j]["event_id"],
                         "frame_difference": second[j]["frame_id"] - first[i]["frame_id"],
                         "distance_px": math.hypot(first[i]["x"] - second[j]["x"],
                                                   first[i]["y"] - second[j]["y"])}
                        for i, j in pairs],
            "initial_only": [e for i, e in enumerate(first) if i not in matched_first],
            "selfcheck_only": [e for j, e in enumerate(second) if j not in matched_second]}


class Session:
    def __init__(self, output: Path, state: dict):
        self.output = Path(output)
        self.state = state
        for r in ROUNDS:
            self.state.setdefault("rounds", {}).setdefault(r, {
                "current_frame": 0, "seen_frames": [], "reviewed_frames": [],
                "completed_at": None, "window_seconds": 0.0, "no_events_reason": ""})
            self.state.setdefault("events", {}).setdefault(r, [])
        self.state.setdefault("non_table", [])
        self.state.setdefault("participants", {})
        self.state.setdefault("timing", {"decode_seconds": 0})
        cached = Path(self.state.get("cache", {}).get("directory", self.output / "frames"))
        self.frames_dir = cached if cached.is_dir() else self.output / "frames"
        self.validate()

    @property
    def meta(self):
        return self.state["video"]

    @property
    def round(self):
        return self.state["current_round"]

    @property
    def progress(self):
        return self.state["rounds"][self.round]

    @property
    def events(self):
        return self.state["events"][self.round]

    def validate(self):
        n, w, h = self.meta["n_frames"], self.meta["width"], self.meta["height"]
        if self.round not in ROUNDS:
            raise ValueError("无效标注轮次")
        for r in ROUNDS:
            frames = []
            for e in self.state["events"][r]:
                if (e["round"] != r or e["video_id"] != self.meta["video_id"]
                    or type(e["frame_id"]) is not int or not 0 <= e["frame_id"] < n
                    or not math.isfinite(e["x"]) or not math.isfinite(e["y"])
                    or not 0 <= e["x"] < w or not 0 <= e["y"] < h
                    or e["event_type"] != "table_bounce" or e["review_state"] != "provisional"):
                    raise ValueError("事件包含无效帧号/坐标/状态")
                frames.append(e["frame_id"])
            if len(frames) != len(set(frames)):
                raise ValueError("同轮次有重复事件帧")
            p = self.state["rounds"][r]
            for f in [p["current_frame"], *p["seen_frames"], *p["reviewed_frames"]]:
                if type(f) is not int or not 0 <= f < n:
                    raise ValueError("无效查看进度")
        for e in self.state["non_table"]:
            if (e.get("round") not in ROUNDS or e["video_id"] != self.meta["video_id"]
                    or type(e["frame_id"]) is not int or not 0 <= e["frame_id"] < n
                    or e["event_type"] not in EVENT_TYPES - {"table_bounce"}):
                raise ValueError("无效干扰事件")
            if (e.get("x") is None) != (e.get("y") is None):
                raise ValueError("干扰坐标应同时填写或同时留空")
            if e.get("x") is not None and not (
                    math.isfinite(e["x"]) and math.isfinite(e["y"])
                    and 0 <= e["x"] < w and 0 <= e["y"] < h):
                raise ValueError("干扰事件坐标越界")
        for e in self.state["uncertain"]:
            if (e["round"] not in ROUNDS or e["video_id"] != self.meta["video_id"]
                or type(e["frame_start"]) is not int or type(e["frame_end"]) is not int
                or not 0 <= e["frame_start"] <= e["frame_end"] < n or not e["reason"].strip()):
                raise ValueError("无效疑难记录")

    def set_frame(self, frame):
        self.progress["current_frame"] = min(max(int(frame), 0), self.meta["n_frames"] - 1)

    def mark_seen(self):
        self.progress["seen_frames"] = sorted(set(self.progress["seen_frames"])
                                               | {self.progress["current_frame"]})
        if len(self.progress['seen_frames']) == self.meta['n_frames']:
            self.progress['displayed_unavailable'] = False

    def mark_reviewed(self):
        self.mark_seen()
        self.progress["reviewed_frames"] = sorted(set(self.progress["reviewed_frames"])
                                                   | {self.progress["current_frame"]})

    def ensure_editable(self):
        if self.progress["completed_at"] is not None:
            raise ValueError("本轮已完成并冻结；可查看，不能修改。初标与自查证据分别保留。")

    def put_event(self, frame, x, y):
        self.ensure_editable()
        if (type(frame) is not int or not 0 <= frame < self.meta["n_frames"]
            or not math.isfinite(x) or not math.isfinite(y)
            or not 0 <= x < self.meta["width"] or not 0 <= y < self.meta["height"]):
            raise ValueError("点选位置或帧号越界")
        event = {"event_id": f"{self.meta['video_id']}_{self.round}_{frame:06d}",
                 "video_id": self.meta["video_id"], "frame_id": frame,
                 "x": round(float(x), 4), "y": round(float(y), 4),
                 "event_type": "table_bounce", "review_state": "provisional", "round": self.round}
        event["annotator"] = self.state["participants"].get(self.round, "未登记")
        existing = next((e for e in self.events if e["frame_id"] == frame), None)
        if existing:
            event["event_id"] = existing["event_id"]
            existing.update(event)
        else:
            self.events.append(event)
        self.events.sort(key=lambda e: e["frame_id"])
        self.progress["no_events_reason"] = ""
        return event

    def put_uncertain(self, start, end, reason):
        self.ensure_editable()
        if (type(start) is not int or type(end) is not int
            or not 0 <= start <= end < self.meta["n_frames"] or not reason.strip()):
            raise ValueError("疑难区间必须合法且填写原因")
        event_id = f"{self.meta['video_id']}_{self.round}_uncertain_{start:06d}_{end:06d}"
        record = {"event_id": event_id, "video_id": self.meta["video_id"],
                  "frame_start": start, "frame_end": end, "reason": reason.strip(),
                  "review_state": "uncertain", "round": self.round,
                  "annotator": self.state['participants'].get(self.round, '未登记')}
        existing = next((e for e in self.state["uncertain"] if e['round'] == self.round
                         and e['frame_start'] == start and e['frame_end'] == end), None)
        if existing:
            record['event_id'] = existing['event_id']
            existing.update(record)
        else:
            self.state["uncertain"].append(record)
        return record

    def delete(self, event_id):
        self.ensure_editable()
        self.state["events"][self.round] = [e for e in self.events if e["event_id"] != event_id]
        self.state["non_table"] = [e for e in self.state["non_table"]
                                   if not (e["round"] == self.round and e["event_id"] == event_id)]
        self.state["uncertain"] = [e for e in self.state["uncertain"]
                                  if not (e["round"] == self.round and e["event_id"] == event_id)]

    def put_non_table(self, frame, event_type, x=None, y=None, reason=""):
        self.ensure_editable()
        if (type(frame) is not int or not 0 <= frame < self.meta["n_frames"]
                or event_type not in EVENT_TYPES - {"table_bounce"}):
            raise ValueError("干扰事件类型或帧号无效")
        if (x is None) != (y is None) or (x is not None and not (
                math.isfinite(x) and math.isfinite(y) and 0 <= x < self.meta["width"]
                and 0 <= y < self.meta["height"])):
            raise ValueError("干扰事件坐标越界")
        if not reason.strip():
            raise ValueError("请填写干扰判断依据")
        record = {"event_id": f"{self.meta['video_id']}_{self.round}_{event_type}_{frame:06d}",
                  "video_id": self.meta["video_id"], "frame_id": frame, "x": x, "y": y,
                  "event_type": event_type, "round": self.round, "reason": reason.strip(),
                  "review_state": "provisional",
                  "annotator": self.state["participants"].get(self.round, "未登记")}
        prior = next((e for e in self.state["non_table"] if e["round"] == self.round
                      and e["frame_id"] == frame and e["event_type"] == event_type), None)
        if prior:
            record["event_id"] = prior["event_id"]
            prior.update(record)
        else:
            self.state["non_table"].append(record)
        return record

    def complete(self, no_events_reason="", now=None, *, inspected_all=False):
        if self.progress["completed_at"] is not None:
            return
        if (self.state.get("completion_policy") != "manual_attestation"
                and len(set(self.progress["seen_frames"])) != self.meta["n_frames"]):
            raise ValueError("尚有未查看帧，请完整查看视频后再结束；无需每帧按V。")
        if not inspected_all and len(set(self.progress["reviewed_frames"])) != self.meta["n_frames"]:
            raise ValueError("请确认已经完整检查；播放覆盖不自动代表人工完成。")
        if not self.events and not no_events_reason.strip():
            raise ValueError("没有触台记录时请填写原因。")
        self.progress["no_events_reason"] = no_events_reason.strip()
        self.progress["inspected_all_attested"] = True
        self.progress["completion_basis"] = "manual_attestation"
        self.progress["completed_at"] = time.time() if now is None else now

    def switch_round(self, round_name, now=None):
        if round_name not in ROUNDS:
            raise ValueError("无效轮次")
        if round_name in ("selfcheck", "review"):
            if self.state["rounds"]["initial"]["completed_at"] is None:
                raise ValueError("初标完成后才能开始第二轮。")
            if round_name == "review":
                initial = self.state["participants"].get("initial")
                reviewer = self.state["participants"].get("review")
                if not reviewer or not initial or reviewer == initial:
                    raise ValueError("交叉复核需要登记与初标不同的成员，请从任务窗口启动。")
            elif "selfcheck" not in self.state["participants"]:
                self.state["participants"]["selfcheck"] = self.state["participants"].get("initial", "未登记")
        self.state["current_round"] = round_name

    def reopen(self, reason):
        if not reason.strip():
            raise ValueError("请说明重新编辑原因。")
        # Retain the entire pre-edit evidence before allowing changes.
        history = self.output / "history"
        history.mkdir(exist_ok=True)
        write_json(history / f"session-{time.time_ns()}.json", self.state)
        self.progress["completed_at"] = None
        self.progress["inspected_all_attested"] = False
        self.progress["completion_basis"] = None
        self.state.setdefault("edit_history", []).append({"round": self.round,
            "reason": reason.strip(), "time": time.time()})

    def save(self):
        self.validate()
        original = self.output / "session.json"
        if original.exists() and not (self.output / "session.legacy-backup.json").exists():
            prior = read_json(original)
            if prior.get("schema_version", 1) < 2:
                write_json(self.output / "session.legacy-backup.json", prior)
        self.state["schema_version"] = 2
        self.state["cache"] = {"directory": str(self.frames_dir.resolve())}
        write_json(original, self.state)
        for r in ROUNDS:
            atomic_text(self.output / f"events_{r}.jsonl", "".join(
                json.dumps(e, ensure_ascii=False) + "\n" for e in self.state["events"][r]))
        for name, key in (("uncertain", "uncertain"), ("non_table_events", "non_table")):
            atomic_text(self.output / f"{name}.jsonl", "".join(
                json.dumps(e, ensure_ascii=False) + "\n" for e in self.state[key]))
        result = compare_events(self.state["events"]["initial"], self.state["events"]["selfcheck"])
        if self.state.get("completion_policy") != "manual_attestation":
            write_json(self.output / "comparison.json", result)
        # Cross-review output is withheld until independent viewing is completed.
        review_result = None
        if self.state["rounds"]["review"]["completed_at"] is not None:
            review_result = compare_events(self.state["events"]["initial"], self.state["events"]["review"])
            write_json(self.output / "review_comparison.json", review_result)
        elif (self.output / "review_comparison.json").exists():
            (self.output / "review_comparison.json").unlink()
        self.write_report(result, review_result)

    def write_report(self, comparison, review_result=None):
        labels = {"initial": "初标", "selfcheck": "个人自查（可选）", "review": "交叉复核"}
        initial_done = self.state["rounds"]["initial"]["completed_at"] is not None
        review_done = self.state["rounds"]["review"]["completed_at"] is not None
        status = ("交叉复核记录完成，分歧待讨论；不自动升级标签" if review_done else
                  "初标完成，待交叉复核" if initial_done else "初标未完成（试点未完成）")
        lines = [f"# {self.meta['video_id']} 标注记录", "", f"状态：{status}。", "",
                 "个人第二遍是可选功能；无需20分钟间隔。一致不等于正确，疑难和未看范围不能作负例。", "",
                 f"实际视频：{self.meta['n_frames']}帧，{self.meta['width']}×{self.meta['height']}，{self.meta['fps']:g} FPS。", "",
                 "|阶段|成员|已查看帧|触台|干扰|疑难|完整检查声明|完成|",
                 "|---|---|---:|---:|---:|---:|---|---|"]
        for r in ROUNDS:
            p = self.state["rounds"][r]
            lines.append(f"|{labels[r]}|{self.state['participants'].get(r, '未登记')}|"
                         f"{len(p['seen_frames'])}/{self.meta['n_frames']}|{len(self.state['events'][r])}|"
                         f"{sum(e['round']==r for e in self.state['non_table'])}|"
                         f"{sum(e['round']==r for e in self.state['uncertain'])}|"
                         f"{'是' if p.get('inspected_all_attested') else '旧逐帧记录' if p['completed_at'] else '否'}|"
                         f"{'是' if p['completed_at'] else '否'}|")
        if review_result is not None:
            lines += ["", "## 初标与复核对照", "",
                      "帧差≤1、距离≤50像素，一对一匹配；只说明一致性。",
                      f"配对 {len(review_result['matches'])}；仅初标 {len(review_result['initial_only'])}；"
                      f"仅复核 {len(review_result['selfcheck_only'])}。",
                      "分歧和干扰类型需回看讨论，初标/复核原始记录分别保留。"]
        if all(self.state["rounds"][r]["completed_at"] is not None for r in ("initial", "selfcheck")):
            lines += ["", f"两遍查看完成（个人自查），配对 {len(comparison['matches'])}，不等于正确。"]
        lines += ["", "## 疑难", ""]
        lines += [f"- {e['round']} 帧{e['frame_start']}—{e['frame_end']}：{e['reason']}"
                  for e in self.state["uncertain"]]
        lines += ["", "## 耗时", "",
                  f"本次准备/解码 {self.state['timing'].get('decode_seconds', 0):.1f}秒。"]
        lines += [f"- {labels[r]}窗口 {self.state['rounds'][r]['window_seconds']:.1f}秒（含停留与播放）。"
                  for r in ROUNDS]
        atomic_text(self.output / "pilot_report.md", "\n".join(lines) + "\n")

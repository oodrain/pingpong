"""Exact-frame video access and portable, independently stored review annotations.

This module never writes source videos, object annotations, official GT, or model
results. All frame identifiers are zero-based sequential decoder indices.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

import cv2

ROOT = Path(__file__).resolve().parents[2]
EVENT_TYPES = {"table_bounce", "racket_contact", "net_contact", "floor_contact", "other"}
FRAME_INDEXING = "zero_based_decoded_frames"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".write-", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _team_validator():
    """The existing validator is standalone; resolve it even from an external CWD."""
    path = ROOT / "development/annotation/submission.py"
    spec = importlib.util.spec_from_file_location("video_studio_team_submission", path)
    if spec is None or spec.loader is None:
        raise ValueError("无法载入项目团队标注校验器。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    path: Path
    role: str = "imported"
    annotation_path: Path | None = None


def discover_catalog(root=ROOT):
    """Read explicit manifests; never recursively scan the project or media tree."""
    root = Path(root).resolve()
    candidates = [root / "data/splits/team_split_v1.json",
                  root / "outputs/team_integration_20261022/stage/data/splits/team_split_v1.json"]
    for manifest in candidates:
        if manifest.is_file():
            result = []
            for row in _read(manifest).get("videos", []):
                rel = row.get("video_relative_path") or row.get("path")
                if not rel:
                    continue
                ann = row.get("object_annotation_relative_path")
                result.append(VideoEntry(row["video_id"], (root / rel).resolve(),
                                         row.get("role", "imported"),
                                         (root / ann).resolve() if ann else None))
            return result
    result, known = [], set()
    public = root / "participant/participant/pingpang/public_data"
    if (public / "manifest.json").is_file():
        for row in _read(public / "manifest.json").get("videos", []):
            result.append(VideoEntry(row["video_id"], (public / row["path"]).resolve(), "public_regression"))
            known.add(row["video_id"])
    # Only the documented annotation layout is visited, never arbitrary folders.
    train = root / "pp_train_data/pp_train_data"
    anns = train / "annotations"
    if anns.is_dir():
        for ann in sorted(anns.glob("[0-9][0-9]/*.json")):
            data = _read(ann)
            vid = data.get("video_id", f"{ann.parent.name}_{ann.stem}")
            if vid in known:
                continue
            paths = [train / str(data.get("video_path", ""))]
            paths += [train / "rallies_videos" / member / ann.parent.name / (ann.stem + ".mp4")
                      for member in "ABCD"]
            matches = [p for p in paths if p.is_file()]
            if matches:
                role = "reserved" if ann.parent.name in {"09", "12", "15", "17", "24"} else "development"
                result.append(VideoEntry(vid, matches[0].resolve(), role, ann.resolve()))
                known.add(vid)
    return result


def entry_from_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("所选视频不存在。")
    for entry in discover_catalog():
        if entry.path == path:
            return entry
    annotation = path.with_suffix(".json")
    video_id = path.stem
    if annotation.is_file():
        try:
            data = _read(annotation)
            if "frames" in data and isinstance(data.get("video_id"), str):
                return VideoEntry(data["video_id"], path, "imported", annotation)
        except (ValueError, OSError, TypeError):
            pass
    return VideoEntry(video_id, path)


class SequentialVideo:
    """Preflight every decoded frame, then read exact indices with bounded memory.

    Construct this object on a worker thread. Access from one thread at a time.
    No approximate keyframe seeking or input_frame_id conversion is performed.
    """
    def __init__(self, path, *, seek_preroll=None):
        self.path = Path(path).resolve()
        self.cache = OrderedDict()
        self.cap = None
        self.next_frame = 0
        self.seek_preroll = seek_preroll
        before = self.path.stat()
        self.sha256 = _sha(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("无法打开视频，请检查路径和编码格式。")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = 0
        self.width = self.height = 0
        times = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if self.n_frames and (width, height) != (self.width, self.height):
                    raise ValueError("视频中途分辨率改变，不能在同一坐标系内标注。")
                self.width, self.height = width, height
                times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
                self.n_frames += 1
        finally:
            cap.release()
        after = self.path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("读取期间视频发生变化，请重新打开。")
        if not self.n_frames:
            raise ValueError("视频没有可解码的帧。")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("视频没有有效帧率，无法可靠播放。")
        if self.reported_frames > 0 and self.reported_frames != self.n_frames:
            raise ValueError(f"视频声明 {self.reported_frames} 帧但实际解码 {self.n_frames} 帧，请检查文件完整性。")
        self.timestamp_warning = ""
        if (any(not math.isfinite(t) or t < 0 for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            self.timestamp_warning = "视频时间戳不连续；帧号保持精确，显示时间按帧率估算。"
            self.timestamps = [None] * self.n_frames
        else:
            self.timestamps = times
        self._reset()

    def _reset(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError("视频重新打开失败。")
        self.next_frame = 0

    def get(self, fid):
        if type(fid) is not int or not 0 <= fid < self.n_frames:
            raise ValueError("帧号超出视频范围。")
        if self.cap is None:
            raise ValueError("视频已关闭。")
        if fid in self.cache:
            self.cache.move_to_end(fid)
            return self.cache[fid], self.timestamps[fid]
        if fid < self.next_frame:
            self._reset()
        if self.seek_preroll is not None and fid - self.next_frame >= 24:
            # Decode in source order and convert only the frames near the target to BGR.
            while self.next_frame < fid - self.seek_preroll:
                if not self.cap.grab():
                    raise ValueError(f"视频在第 {self.next_frame} 帧提前结束。")
                self.next_frame += 1
        while self.next_frame <= fid:
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f"视频在第 {self.next_frame} 帧提前结束。")
            if frame.shape[:2] != (self.height, self.width):
                raise ValueError("视频尺寸已改变。")
            self.cache[self.next_frame] = frame
            self.next_frame += 1
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
        return self.cache[fid], self.timestamps[fid]

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()


class _Lock:
    def __init__(self, path):
        self.stream = Path(path).open("a+b")
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
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError("这个视频的标注会话正在另一窗口中使用。") from error

    def close(self):
        if not self.stream.closed:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()


class SessionStore:
    def __init__(self, output_root, entry, video):
        self.entry, self.video = entry, video
        slug = re.sub(r"[^\w.-]+", "_", entry.video_id)[:100] or "video"
        self.directory = Path(output_root).resolve() / (slug + "-" + video.sha256[:12])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "session.json"
        self._lock = _Lock(self.directory / ".session.lock")
        self._undo, self._redo = [], []
        self._source_stat = self._stat()
        self._annotation_sha = _sha(entry.annotation_path) if entry.annotation_path and entry.annotation_path.is_file() else None
        try:
            if _sha(entry.path) != video.sha256:
                raise ValueError("打开会话前源视频已变化，请重新载入视频。")
            self.state = _read(self.path) if self.path.is_file() else {
                "schema_version": 1, "tool": "pingpong_video_studio/v1",
                "video_id": entry.video_id, "video_path": str(entry.path), "role": entry.role,
                "video_sha256": video.sha256, "width": video.width, "height": video.height,
                "n_frames": video.n_frames, "fps": video.fps, "frame_indexing": FRAME_INDEXING,
                "annotation_path": str(entry.annotation_path) if entry.annotation_path else None,
                "annotation_sha256": self._annotation_sha,
                "events": [], "uncertain": [], "completed": False, "last_frame": 0,
                "created_at": _now(), "updated_at": _now(), "checked_ranges": [], "notes": ""}
            for key, expected in (("video_id", entry.video_id), ("video_sha256", video.sha256),
                                  ("width", video.width), ("height", video.height), ("n_frames", video.n_frames)):
                if self.state.get(key) != expected:
                    raise ValueError("已有标注与当前视频身份或尺寸不一致，请使用新的工作目录。")
            if self.state.get("annotation_sha256") != self._annotation_sha:
                raise ValueError("原始对象标注身份已变化，请使用新的工作目录后核对。")
            self._validate()
        except BaseException:
            self._lock.close()
            raise

    def _stat(self):
        value = Path(self.entry.path).stat()
        return value.st_size, value.st_mtime_ns

    def _verify_source(self):
        current = self._stat()
        if current != self._source_stat:
            if _sha(self.entry.path) != self.video.sha256:
                raise ValueError("源视频已被修改，停止保存；请恢复原文件或重新打开新视频。")
            self._source_stat = current

    def _frame(self, value):
        if type(value) is not int or not 0 <= value < self.video.n_frames:
            raise ValueError("帧号必须是视频范围内的0基整数。")

    def _coords(self, x, y, required=False):
        if x is None and y is None and not required:
            return
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("请选择原图中的有效位置。")
        if not 0 <= x < self.video.width or not 0 <= y < self.video.height:
            raise ValueError("坐标超出原图范围。")

    def _validate(self):
        ids, identities = set(), set()
        for event in self.state["events"]:
            if event["event_type"] not in EVENT_TYPES:
                raise ValueError("未知事件类型。")
            self._frame(event["frame_id"])
            self._coords(event.get("x"), event.get("y"), event["event_type"] == "table_bounce")
            if not isinstance(event.get("reason", ""), str):
                raise ValueError("事件说明必须为文字。")
            identity = (event["event_type"], event["frame_id"])
            if identity in identities:
                raise ValueError("同一帧出现了重复的同类型事件。")
            identities.add(identity)
            if event["id"] in ids:
                raise ValueError("标注编号重复。")
            ids.add(event["id"])
        for interval in self.state["uncertain"]:
            self._frame(interval["frame_start"])
            self._frame(interval["frame_end"])
            if (interval["frame_start"] > interval["frame_end"]
                    or not isinstance(interval.get("reason"), str) or not interval["reason"].strip()):
                raise ValueError("疑难区间或说明无效。")
            if interval["id"] in ids:
                raise ValueError("标注编号重复。")
            ids.add(interval["id"])
        self._frame(self.state.get("last_frame", 0))
        if type(self.state.get("completed")) is not bool:
            raise ValueError("完成状态无效。")
        if self.state["completed"] and self.state.get("checked_ranges") != [[0, self.video.n_frames - 1]]:
            raise ValueError("完成状态缺少完整人工核查范围。")

    def _before_change(self):
        self._undo.append(deepcopy(self.state))
        self._undo = self._undo[-100:]
        self._redo.clear()
        self.state["completed"] = False
        self.state["checked_ranges"] = []

    def add_event(self, event_type, frame_id, x=None, y=None, reason=""):
        if event_type not in EVENT_TYPES:
            raise ValueError("未知事件类型。")
        self._frame(frame_id)
        self._coords(x, y, event_type == "table_bounce")
        if not isinstance(reason, str):
            raise ValueError("备注必须是文字。")
        if any(e["event_type"] == event_type and e["frame_id"] == frame_id for e in self.state["events"]):
            raise ValueError("当前帧已有同类型事件，请删除旧记录后重新标注。")
        self._before_change()
        record = {"id": uuid4().hex, "event_type": event_type, "frame_id": frame_id,
                  "x": float(x) if x is not None else None, "y": float(y) if y is not None else None,
                  "reason": str(reason).strip()}
        self.state["events"].append(record)
        self.state["events"].sort(key=lambda e: (e["frame_id"], e["event_type"]))
        return record

    def add_uncertain(self, start, end, reason, touch=False):
        self._frame(start)
        self._frame(end)
        if start > end or not isinstance(reason, str) or not reason.strip():
            raise ValueError("疑难区间起点不能晚于终点，并且需要填写说明。")
        if any(e["frame_start"] == start and e["frame_end"] == end for e in self.state["uncertain"]):
            raise ValueError("已存在相同疑难区间。")
        self._before_change()
        record = {"id": uuid4().hex, "frame_start": start, "frame_end": end,
                  "reason": str(reason).strip(), "touch": bool(touch)}
        self.state["uncertain"].append(record)
        self.state["uncertain"].sort(key=lambda e: (e["frame_start"], e["frame_end"]))
        return record

    def update_reason(self, id, reason):
        record = next((e for name in ("events", "uncertain") for e in self.state[name] if e["id"] == id), None)
        if record is None:
            raise ValueError("这条标注已不存在，请重新选择记录。")
        required = "frame_start" in record
        if not isinstance(reason, str) or (required and not reason.strip()):
            raise ValueError("疑难区间必须填写文字说明。")
        reason = reason.strip()
        if record.get("reason", "") != reason:
            self._before_change()
            record["reason"] = reason
        return record

    def delete(self, id):
        if not any(e["id"] == id for name in ("events", "uncertain") for e in self.state[name]):
            return
        self._before_change()
        for name in ("events", "uncertain"):
            self.state[name] = [e for e in self.state[name] if e["id"] != id]

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(deepcopy(self.state))
        self.state = self._undo.pop()
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(deepcopy(self.state))
        self.state = self._redo.pop()
        return True

    def mark_complete(self, value=True):
        if type(value) is not bool:
            raise ValueError("完成状态必须是布尔值。")
        if self.state["completed"] != value:
            self._before_change()
            self.state["completed"] = value
            self.state["checked_ranges"] = [[0, self.video.n_frames - 1]] if value else []

    def save(self):
        if self._lock is None:
            raise ValueError("标注会话已关闭。")
        self._verify_source()
        self._validate()
        self.state["updated_at"] = _now()
        if self.path.exists():
            history = self.directory / "history"
            history.mkdir(exist_ok=True)
            target = history / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
            shutil.copy2(self.path, target)
            for old in sorted(history.glob("*.json"))[:-20]:
                old.unlink()
        _write(self.path, self.state)
        return self.path

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def export(self, folder):
        """Create a new generic exchange directory, never replace an existing one."""
        self.save()
        folder = Path(folder).resolve()
        if folder.exists():
            raise ValueError("导出目录已存在，请选一个新目录，避免覆盖历史结果。")
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".video-studio-export-", dir=folder.parent) as staging:
            stage = Path(staging) / "bundle"
            stage.mkdir()
            _write(stage / "session.json", self.state)
            metadata = {key: deepcopy(value) for key, value in self.state.items() if key not in {"events", "uncertain"}}
            metadata["coordinate_system"] = "original_image_pixels"
            metadata["interval_endpoints"] = "inclusive"
            metadata["completion_basis"] = "manual_attestation" if self.state["completed"] else None
            _write(stage / "metadata.json", metadata)
            _jsonl(stage / "events.jsonl", [dict(e, event_id=e["id"], video_id=self.entry.video_id) for e in self.state["events"]])
            _jsonl(stage / "uncertain.jsonl", [dict(e, event_id=e["id"], video_id=self.entry.video_id) for e in self.state["uncertain"]])
            os.rename(stage, folder)
        return folder

    def export_team(self, folder, annotator="A"):
        """Export compatible initial v2 labels, gated by the existing team validator."""
        if not re.fullmatch(r"\d{2}_\d{3}", self.entry.video_id):
            raise ValueError("团队标准包仅适用于 NN_NNN 编号的训练视频；该视频请使用通用导出。")
        if annotator not in "ABCD" or len(annotator) != 1:
            raise ValueError("团队成员必须为 A、B、C 或 D。")
        if self.entry.annotation_path is None or not self.entry.annotation_path.is_file():
            raise ValueError("团队标准包需要原始对象 annotation.json。")
        annotation = _read(self.entry.annotation_path)
        if (annotation.get("video_id") != self.entry.video_id
                or annotation.get("num_frames") != self.video.n_frames
                or annotation.get("frame_width") != self.video.width
                or annotation.get("frame_height") != self.video.height
                or [e.get("frame_id") for e in annotation.get("frames", [])] != list(range(self.video.n_frames))):
            raise ValueError("原始对象标注与当前视频不一致。")
        if _sha(self.entry.annotation_path) != self._annotation_sha:
            raise ValueError("原始对象标注已变化，请重新打开后核对。")
        self.save()
        validator = _team_validator()
        target = Path(folder).resolve() / annotator / self.entry.video_id / "initial"
        if target.exists():
            raise ValueError("这个团队导出目录已存在，请选择新的导出根目录。")
        metadata = {"schema_version": 2, "video_id": self.entry.video_id, "annotator": annotator,
                    "round": "initial", "width": self.video.width, "height": self.video.height,
                    "n_frames": self.video.n_frames, "fps": self.video.fps,
                    "video_sha256": self.video.sha256, "annotation_sha256": self._annotation_sha,
                    "completed": self.state["completed"], "checked_ranges": self.state.get("checked_ranges", []),
                    "no_events_reason": self.state.get("notes", "").strip() if not any(
                        e["event_type"] == "table_bounce" for e in self.state["events"]) else "",
                    "label_version": 1, "updated_at": _now(), "tool": "pingpong_video_studio/v1",
                    "source_type": "manual", "review_of": None,
                    "completion_basis": "manual_attestation" if self.state["completed"] else None,
                    "displayed_ranges": None}
        events, uncertain, non_table = [], [], []
        for item in self.state["events"]:
            row = {"event_id": item["id"], "video_id": self.entry.video_id, "annotator": annotator,
                   "round": "initial", "review_state": "provisional",
                   **{key: item[key] for key in ("event_type", "frame_id", "x", "y", "reason")}}
            (events if item["event_type"] == "table_bounce" else non_table).append(row)
        for item in self.state["uncertain"]:
            uncertain.append({"event_id": item["id"], "video_id": self.entry.video_id, "annotator": annotator,
                              "round": "initial", "review_state": "uncertain",
                              **{key: item[key] for key in ("frame_start", "frame_end", "reason")}})
        if metadata["completed"] and not events and not metadata["no_events_reason"]:
            raise ValueError("完整核查后没有触台时，请在会话 notes 中说明原因后导出团队标准包。")
        validator.validate_data(metadata, events, uncertain, non_table)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".team-export-", dir=target.parent) as staging:
            stage = Path(staging) / annotator / self.entry.video_id / "initial"
            stage.mkdir(parents=True)
            _write(stage / "metadata.json", metadata)
            _jsonl(stage / "events.jsonl", events)
            _jsonl(stage / "uncertain.jsonl", uncertain)
            _jsonl(stage / "non_table_events.jsonl", non_table)
            validator.validate_folder(stage)
            os.rename(stage, target)
        return target


def load_reference(path, video_id=None):
    """Import references as detached records while preserving valid:false and provenance."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / ("metadata.json" if (path / "metadata.json").exists() else "session.json")
    records, context = [], None
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        meta = path.parent / "metadata.json"
        if meta.is_file():
            context = _read(meta).get("video_id")
    else:
        raw = _read(path)
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            context = raw.get("video_id") or raw.get("video", {}).get("video_id")
            grouped = raw.get("videos")
            if isinstance(grouped, dict):
                for vid, rows in grouped.items():
                    if isinstance(rows, list):
                        records.extend(dict(row, video_id=vid) for row in rows)
            elif isinstance(grouped, list):
                for video in grouped:
                    records.extend(dict(row, video_id=video["video_id"]) for row in video.get("events", []))
            if path.name == "metadata.json":
                for name in ("events.jsonl", "uncertain.jsonl", "non_table_events.jsonl"):
                    adjacent = path.parent / name
                    if adjacent.exists():
                        records.extend(json.loads(line) for line in adjacent.read_text(encoding="utf-8-sig").splitlines() if line.strip())
            else:
                event_data = raw.get("events", [])
                if isinstance(event_data, dict):
                    for phase, rows in event_data.items():
                        records.extend(dict(row, round=row.get("round", phase)) for row in rows)
                elif isinstance(event_data, list):
                    records.extend(event_data)
                for key in ("uncertain", "non_table", "non_table_events", "completed_events_exact_deduplicated", "tracks", "landings"):
                    if isinstance(raw.get(key), list):
                        records.extend(raw[key])
                    elif isinstance(raw.get(key), dict):
                        for phase, rows in raw[key].items():
                            if isinstance(rows, list):
                                records.extend(dict(row, round=row.get("round", phase)) for row in rows)
                if "frame_id" in raw or "interval_frames" in raw:
                    records.append(raw)
        else:
            raise ValueError("不支持的参考文件格式。")
    result = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError("参考记录必须是JSON对象。")
        row = deepcopy(item)
        vid = row.get("video_id", context)
        if video_id is not None and vid != video_id:
            continue
        if not any(k in row for k in ("frame_id", "frame_start", "interval_frames")):
            continue
        row.setdefault("video_id", vid)
        row.setdefault("id", row.get("event_id") or row.get("gt_id") or f"ref_{index}")
        row.setdefault("source_path", str(path))
        if "event_type" not in row:
            row["event_type"] = "ball_track" if "detected" in row else ("uncertain" if "frame_start" in row else "table_bounce")
        result.append(row)
    return result

"""Read-only compatibility layers; no inferred event coordinates or frame IDs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

try:
    from .core import load_reference
except ImportError:
    from core import load_reference


def interval_of(record):
    if type(record.get("frame_id")) is int:
        return record["frame_id"], record["frame_id"]
    value = record.get("interval_frames")
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(type(f) is int for f in value):
        return tuple(value)
    if all(type(record.get(k)) is int for k in ("frame_start", "frame_end")):
        return record["frame_start"], record["frame_end"]
    return None


def trajectory_segments(records, frame, tail=24):
    """Only join adjacent, detected samples; missing detections break the line."""
    segments, segment, previous = [], [], None
    for row in sorted(records, key=lambda r: r.get("frame_id", -1)):
        fid = row.get("frame_id")
        if type(fid) is not int or not max(0, frame - tail) <= fid <= frame:
            continue
        valid = row.get("detected", True) and all(
            type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("x", "y"))
        if not valid or (previous is not None and fid != previous + 1):
            if segment:
                segments.append(segment)
            segment = []
        if valid:
            segment.append((fid, row["x"], row["y"]))
        previous = fid
    if segment:
        segments.append(segment)
    return segments


def timed_tail(records, time_s, duration=0.6, max_gap=0.085):
    """Yield actual observations and a short display-only link to the previous point."""
    previous = None
    for row in records:
        stamp = row.get("timestamp_s")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            previous = None
            continue
        if stamp < time_s - duration or stamp > time_s + 1e-8:
            continue
        x, y = row.get("x"), row.get("y")
        if (not row.get("detected", True) or not row.get("in_image", True)
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y))):
            previous = None
            continue
        point = (stamp, row["frame_id"], x, y)
        link = (previous if previous is not None and point[1] == previous[1] + 1
                and 0 < stamp - previous[0] <= max_gap else None)
        yield point, link
        previous = point


def read_own_session(output, entry, video):
    slug = re.sub(r"[^\w.-]+", "_", entry.video_id)[:100] or "video"
    path = Path(output) / "sessions" / (slug + "-" + video.sha256[:12]) / "session.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    if state.get("video_sha256") != video.sha256 or (state.get("width"), state.get("height"), state.get("n_frames")) != (video.width, video.height, video.n_frames):
        raise ValueError("已保存的工作台标注与当前视频身份不一致")
    return state


def load_history(root, entry, video):
    root = Path(root)
    sources, warnings, records = [], [], []
    for manifest in (root / "data/splits/team_split_v1.json", root / "outputs/team_integration_20261022/stage/data/splits/team_split_v1.json"):
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            declared = next((r for r in data.get("videos", []) if r.get("video_id") == entry.video_id), {})
            if declared.get("video_sha256") and declared["video_sha256"] != video.sha256:
                return [], ["视频编号相同但内容与项目清单不同，未加载历史标注"]
            break
    for base in (root / "labels", root / "tmp/repository_labels_audit_20261012/labels", root / "outputs/local_annotation"):
        for member in "ABCD":
            for phase in ("initial", "review"):
                folder = base / member / entry.video_id / phase
                path = folder / "metadata.json"
                if not path.is_file():
                    path = folder / "session.json"
                if path.is_file():
                    sources.append((path, member, phase))
    snapshot = root / "outputs/team_integration_20261022/stage/labels/local_development_20261022"
    for name in ("confirmed_positive_union.json", "precise_scopes.json"):
        path = snapshot / name
        if path.is_file():
            sources.append((path, "", "已核查"))
    seen = set()
    for path, member, phase in sources:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            identity = data.get("video_sha256") or data.get("inputs", {}).get("video_sha256")
            if isinstance(data.get("videos"), list):
                meta = next((v for v in data["videos"] if v.get("video_id") == entry.video_id), {})
                identity = meta.get("video_sha256", identity)
            if identity and identity != video.sha256:
                warnings.append(f"跳过视频身份不匹配的 {path.name}")
                continue
            for row in load_reference(path, entry.video_id):
                interval = interval_of(row)
                if interval is None or not 0 <= interval[0] <= interval[1] < video.n_frames:
                    continue
                row.setdefault("annotator", member)
                row.setdefault("round", phase)
                key = (row.get("event_type"), interval, row.get("x"), row.get("y"), row.get("annotator"), row.get("round"), row.get("reason"))
                if key in seen:
                    continue
                seen.add(key)
                row["id"] = "history_" + str(len(records))
                records.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f"{path.name}：{error}")
    return records, warnings

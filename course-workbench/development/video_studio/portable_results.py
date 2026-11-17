"""Commit, reopen and export portable automatic results."""
import math
from pathlib import Path
import tempfile
from uuid import UUID, uuid4

from core import _jsonl, _read, _write


SCHEMA = "pingpong-studio-cpu-v1"


def latest(workspace, source_hash):
    pointer = workspace / "analysis" / "by_video" / (source_hash + ".json")
    if not pointer.is_file():
        return None
    identifier = str(UUID(_read(pointer)["run_id"]))
    folder = workspace / "analysis" / "runs" / identifier
    manifest = _read(folder / "run.json")
    if (manifest["schema"], manifest["status"], manifest["source_sha256"], manifest["run_id"]) != (
        SCHEMA, "succeeded", source_hash, identifier
    ):
        raise ValueError("自动分析结果与当前视频不符。")
    return folder, manifest


def commit(workspace, manifest, tracks, landings):
    source_hash, run_id = manifest["source_sha256"], manifest["run_id"]
    runs = workspace / "analysis" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    folder = runs / run_id
    with tempfile.TemporaryDirectory(dir=runs, prefix=".pending-") as staging:
        staging = Path(staging)
        _write(staging / "tracks.json", tracks)
        _jsonl(staging / "predictions.jsonl", landings)
        _write(staging / "run.json", manifest)
        staging.rename(folder)
    _write(workspace / "analysis" / "by_video" / (source_hash + ".json"), {"run_id": run_id})


def rows_for_video(result, video_id, width, height, frames):
    import json
    folder, manifest = result
    if (width, height, frames) != (manifest["width"], manifest["height"], manifest["source_frames"]):
        raise ValueError("分析结果的原图尺寸或源帧数与播放器不符。")
    predictions = []
    with (folder / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            frame, x, y = row["frame_id"], row["x"], row["y"]
            if (type(frame) is not int or not 0 <= frame < frames or not math.isfinite(x)
                    or not math.isfinite(y) or not 0 <= x < width or not 0 <= y < height):
                raise ValueError("分析结果包含非法落点。")
            predictions.append(dict(video_id=video_id, frame_id=frame, x=x, y=y))
    if len(predictions) != manifest["event_count"]:
        raise ValueError("自动分析事件数量与运行记录不符。")
    return predictions


def export_result(result, destination, video_id):
    _, manifest = result
    rows = rows_for_video(result, video_id, manifest["width"], manifest["height"], manifest["source_frames"])
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_id)[:48]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{safe_id}_落点_{uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(dir=destination, prefix=".pending-") as staging:
        staging = Path(staging)
        _jsonl(staging / "predictions.jsonl", rows)
        _write(staging / "run.json", dict(manifest, video_id=video_id))
        staging.rename(target)
    return target

"""离线回放官方 V0 触台规则：读 C 的轨迹缓存 raw/*.jsonl，输出官方格式 predictions.jsonl。

逐行复刻官方参考实现 participant/solution.py 的 _emit 逻辑：
- 置信度门槛 CONF_TH=0.75（轨迹只收 conf>=0.75 的点）
- 局部极大：y > prev_y 且 y >= next_y
- 幅度门控：(y-prev_y) >= FALL_AMP(2.0) 且 (y-next_y) >= RISE_AMP(1.0)
- MIN_GAP=10：与上次发出间隔小于 10 帧则跳过
- 帧补偿：上报 frame_id = input_frame_id - 1（不小于 0）
- process 阶段 need_future=2；finish 阶段 need_future=1

用法：
    python replay_v0.py --raw <raw 目录> --out <predictions.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_GAP = 10
CONF_TH = 0.75
FALL_AMP = 2.0
RISE_AMP = 1.0


def emit(traj: list[tuple[int, float, float, float]], last_emit: int, need_future: int):
    out = []
    n = len(traj)
    for j in range(1, n - need_future):
        fid, x, y, conf = traj[j]
        if fid - last_emit < MIN_GAP:
            continue
        prev_y = traj[j - 1][2]
        next_y = traj[j + 1][2]
        if (y > prev_y and y >= next_y
                and (y - prev_y) >= FALL_AMP
                and (y - next_y) >= RISE_AMP):
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y})
            last_emit = fid
    return out, last_emit


def replay_video(rows: list[dict]) -> list[dict]:
    traj: list[tuple[int, float, float, float]] = []
    last_emit = -10_000
    landings: list[dict] = []
    for r in rows:
        if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH:
            traj.append((r["input_frame_id"], r["x"], r["y"], r["confidence"]))
        res, last_emit = emit(traj, last_emit, 2)
        landings.extend(res)
    res, last_emit = emit(traj, last_emit, 1)
    landings.extend(res)
    return landings


def main() -> None:
    ap = argparse.ArgumentParser(description="离线回放官方 V0 触台规则")
    ap.add_argument("--raw", required=True, help="C 导出的 raw 目录（每视频一个 jsonl）")
    ap.add_argument("--out", required=True, help="输出 predictions.jsonl 路径")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"raw 目录为空: {raw_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for path in files:
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows.sort(key=lambda r: r["input_frame_id"])
            video_id = rows[0]["video_id"]
            landings = replay_video(rows)
            for item in landings:
                f.write(json.dumps({"video_id": video_id, "frame_id": item["frame_id"],
                                    "x": item["x"], "y": item["y"]}) + "\n")
            total += len(landings)
            print(f"{video_id}: {len(rows)} 输入帧 -> {len(landings)} 个落点")
    print(f"合计 {total} 个落点 -> {out_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M1.5 首尾边界取证：
A) 公开15 + 154(v2)：找出所有"标签在轨迹首/尾 2 帧之内"的漏报，打印首/尾窗口；
B) 风险人口：统计有多少视频的首点满足 y[0] > y[1]（起步即上升）/ 尾点满足 y[-1] > y[-2]（结尾仍下降），
   并核对这些位置附近是否有标签——决定补偿发射的 FP 风险面。
"""
import importlib.util
import json
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")
RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12
CFG = {"F1": True, "F2": True, "F3": True}
FULL = {**CFG, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0,
        "D_dmin": 2, "D3": True, "D3_ratio": 0.45}


def eligible(rows):
    return [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= 0.75]


def scan_edges(tag, by_v, raw_dir):
    print(f"===== {tag} =====")
    n_head = n_tail = 0
    for vid in sorted(by_v):
        rows = G.load(raw_dir / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        traj = eligible(rows)
        if not traj:
            continue
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, None, FULL)]
        mm_, fp, fn = G.match(preds, by_v[vid])
        gts = by_v[vid]
        first_real, last_real = traj[0][0] - 1, traj[-1][0] - 1
        for jj in fn:
            g = gts[jj]
            if not g.get("valid", True):
                continue
            f = int(g["frame_id"])
            if f <= first_real + 1 or f >= last_real - 1:
                kind = "首" if f <= first_real + 1 else "尾"
                win = [f"{t[0]-1}:{t[2]:.0f}" for t in (traj[:4] if kind == "首" else traj[-4:])]
                print(f"  {vid} 标{f}{kind} | 轨迹首帧f{first_real} 尾帧f{last_real} | 窗口 {win} | 标签({g['x']:.0f},{g['y']:.0f})")
                n_head += kind == "首"
                n_tail += kind == "尾"
    print(f"  首部漏报 {n_head}，尾部漏报 {n_tail}")
    # 风险人口
    risk_h = risk_t = 0
    head_no_label = tail_no_label = 0
    for vid in sorted(by_v):
        rows = G.load(raw_dir / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        traj = eligible(rows)
        if len(traj) < 3:
            continue
        gts = by_v[vid]
        if traj[0][2] > traj[1][2]:
            risk_h += 1
            if not any(abs(g["frame_id"] - (traj[0][0] - 1)) <= 1 for g in gts):
                head_no_label += 1
        if traj[-1][2] > traj[-2][2]:
            risk_t += 1
            if not any(abs(g["frame_id"] - (traj[-1][0] - 1)) <= 1 for g in gts):
                tail_no_label += 1
    print(f"  风险人口：首点上升(y0>y1) {risk_h} 条（其中近处无标签 {head_no_label}）；"
          f"尾点下降(yl>yl-1) {risk_t} 条（其中近处无标签 {tail_no_label}）")


def main():
    # 公开15
    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    by15 = defaultdict(list)
    for g in gt:
        by15[g["video_id"]].append(g)
    scan_edges("公开15(v0 直接漏报口径按 F1F2F3+D2/D3 复合配置)", by15, G.RAW)
    # 154
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by154 = defaultdict(list)
    for e in ev2:
        by154[e["video_id"]].append(e)
    scan_edges("154(v2)", by154, RAW154)


if __name__ == "__main__":
    main()

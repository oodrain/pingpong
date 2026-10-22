# -*- coding: utf-8 -*-
"""FP 侧取证（③）：公开 15 当前候选栈的 38 个误报全量档案 + 廉价判别式测量。

对每个 FP 输出：位置、投影出桌距离、与最近标签的帧差/距离、网线距离、复核码（v2 工作表交叉）；
并测量若干判别式在 FP/TP 上的分离度（网线邻近性、出桌距离分层、近邻标签距离）。
"""
import csv
import importlib.util
import json
import math
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EA = Path("E:/pingpong-data/error_analysis_v1_20261011")

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

FULL = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
        "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2, "gate_quad": 4.0,
        "D3": True, "D3_ratio": 0.45}


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def main():
    # 复核码索引（v2 工作表：按 (视频, 帧±2, 类别组) 找最近）
    review = []
    for r in csv.DictReader((EA / "review" / "复核工作表_已填写.csv").read_text(encoding="gbk").splitlines()):
        if r["数据集"] == "public15":
            review.append((r["视频"], int(r["帧"]), (r["人工结论"] or "").strip(), r["AI预归因"]))

    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    by_v = defaultdict(list)
    for g in gt:
        by_v[g["video_id"]].append(g)

    rows_out = []
    stat_fp = Counter()
    net_d_fp, net_d_tp = [], []
    out_fp, out_tp = [], []
    for vid in man:
        doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
        seg = doc["segments"][0]
        H = G.build_H(seg)
        net = seg.get("net")
        width = 1920.0
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=seg["corners"]))]
        gts = by_v.get(vid, [])
        mm_, fp, fn = G.match(preds, gts)
        for i in fp:
            p = preds[i]
            u, v = G.project(H, p["x"], p["y"])
            outmm = max(u - 2740, -u, v - 1525, -v, 0)
            # 最近标签
            best = min(((abs(g["frame_id"] - p["frame_id"]),
                         math.dist((g["x"], g["y"]), (p["x"], p["y"])), g["frame_id"]) for g in gts),
                       default=(999, 999, -1))
            nd = seg_dist((p["x"], p["y"]), net[0], net[1]) if net else None
            code = ""
            for rv, rf, rc, ra in review:
                if rv == vid and abs(rf - p["frame_id"]) <= 2 and rc:
                    code = rc
                    break
            out_fp.append(outmm)
            if nd is not None:
                net_d_fp.append(nd / width)
            rows_out.append((vid, p["frame_id"], round(p["x"]), round(p["y"]), round(outmm),
                             best[0], round(best[1]), None if nd is None else round(nd / width, 3), code,
                             p.get("why", "")))
        for i, j in mm_:
            p = preds[i]
            u, v = G.project(H, p["x"], p["y"])
            out_tp.append(max(u - 2740, -u, v - 1525, -v, 0))
            if net:
                net_d_tp.append(seg_dist((p["x"], p["y"]), net[0], net[1]) / width)

    print(f"公开15 当前 FP = {len(rows_out)} 个；TP = {len(out_tp)} 个")
    print(f"\n{'视频':8s} {'帧':>4s} {'位置':>12s} {'出桌mm':>7s} {'Δ标签':>5s} {'距标签':>6s} {'网距(w)':>8s} {'复核':>3s} {'why':>6s}")
    for r in rows_out:
        print(f"{r[0]:8s} {r[1]:>4d} ({r[2]:>4d},{r[3]:>4d}) {r[4]:>7.0f} {r[5]:>4d} {r[6]:>6d} "
              f"{str(r[7]):>8s} {r[8]:>3s} {r[9]:>6s}")
    cc = Counter(r[8] for r in rows_out)
    print(f"\n复核码分布: {dict(cc)}")
    print(f"出桌距离: FP 中位 {np.median(out_fp):.0f}mm / TP 中位 {np.median(out_tp):.0f}mm")
    if net_d_fp:
        print(f"网线距离(画面宽占比): FP 中位 {np.median(net_d_fp):.3f} / TP 中位 {np.median(net_d_tp):.3f}")
        for th in (0.01, 0.02, 0.04):
            fhc = sum(1 for d in net_d_fp if d < th)
            thc = sum(1 for d in net_d_tp if d < th)
            print(f"  阈值 {th}: 覆盖 FP {fhc}/{len(net_d_fp)}，误伤 TP {thc}/{len(net_d_tp)}")


if __name__ == "__main__":
    main()

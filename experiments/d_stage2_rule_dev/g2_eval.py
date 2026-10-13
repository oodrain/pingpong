# -*- coding: utf-8 -*-
"""几何门双判评估：图像四边形通道（含膨胀 δ）+ 投影门。

背景（M1.2 取证）：V1 的 900mm 投影门在桌面图像近边缘处会"透视爆炸"——
球在桌缘附近触台时，发射点比接触点低十几像素就会投影出桌 1000+mm，被误杀
（v000001 f58：标注点投影 0mm 在桌内、发射点低 13px 却 1226mm 出桌）。

双判：发射点若落在桌面图像四边形内（或外扩 δ px 内）→ 直接放行；
否则照旧走 900mm 投影门。评估增量：分别叠加在 F1F2F3 与 M1.2 候选（+D2c）之上。
"""
import importlib.util
import json
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

BASE = {"gate": 900, "F1": True, "F2": True, "F3": True}
D2C = {**BASE, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
       "D_smax": 50.0, "D_dmin": 2}

CFGS = [
    ("F1F2F3", dict(BASE)),
    ("F1F2F3+quad2", {**BASE, "gate_quad": 2.0}),
    ("F1F2F3+quad6", {**BASE, "gate_quad": 6.0}),
    ("+D2c", dict(D2C)),
    ("+D2c+quad2", {**D2C, "gate_quad": 2.0}),
    ("+D2c+quad4", {**D2C, "gate_quad": 4.0}),
    ("+D2c+quad8", {**D2C, "gate_quad": 8.0}),
]


def main():
    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    gts_by_v = defaultdict(list)
    for g in gt:
        gts_by_v[g["video_id"]].append(g)

    videos = {}
    for vid in man:
        doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
        seg = doc["segments"][0]
        H = G.build_H(seg)
        quad = seg["corners"]
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"), quad)

    angles = ["0度", "45度", "90度", "全部"]
    results, detail = {}, {}
    for name, cfg in CFGS:
        per_video, sets = {}, {}
        for vid, (H, rows, gts, _, quad) in videos.items():
            cfg_v = dict(cfg, quad=quad)
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, cfg_v)]
            mm_, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
            per_video[vid] = (mv, len(fp) + (len(mm_) - mv),
                              sum(1 for j in fn if gts[j].get("valid", True)))
            sets[vid] = {
                "matched": {gts[j]["frame_id"] for i, j in mm_ if gts[j].get("valid", True)},
                "fp": {preds[i]["frame_id"] for i in fp},
                "fn": {gts[j]["frame_id"] for j in fn if gts[j].get("valid", True)},
            }
        results[name] = per_video
        detail[name] = sets

    print(f"{'配置':16s} | {'0度':>14s} | {'45度':>14s} | {'90度':>14s} | {'全部':>16s}")
    for name, _ in CFGS:
        cells = []
        for a in angles:
            vs = [v for v in videos if a == "全部" or videos[v][3] == a]
            tp = sum(results[name][v][0] for v in vs)
            fp = sum(results[name][v][1] for v in vs)
            fn = sum(results[name][v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
        print(f"{name:16s} | {cells[0]:>14s} | {cells[1]:>14s} | {cells[2]:>14s} | {cells[3]:>16s}")

    # 明细：quad 相对各自基线的变化
    for name in ("F1F2F3+quad2", "F1F2F3+quad6", "+D2c+quad2", "+D2c+quad4", "+D2c+quad8"):
        base_name = "F1F2F3" if name.startswith("F1") else "+D2c"
        base, cur = detail[base_name], detail[name]
        nm = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["matched"] - base[v]["matched"])]
        lm = [f"{v} f{f}" for v in videos for f in sorted(base[v]["matched"] - cur[v]["matched"])]
        nf = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["fp"] - base[v]["fp"])]
        rf = [f"{v} f{f}" for v in videos for f in sorted(base[v]["fp"] - cur[v]["fp"])]
        nfn = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["fn"] - base[v]["fn"])]
        ffn = [f"{v} f{f}" for v in videos for f in sorted(base[v]["fn"] - cur[v]["fn"])]
        print(f"\n[{name} vs {base_name}] 新命中 {nm} | 丢命中 {lm} | 新增FP {nf} | 消除FP {rf} | 修复FN {ffn} | 新增FN {nfn}")

    (HERE / "g2_stats.json").write_text(
        json.dumps({n: {v: results[n][v] for v in videos} for n, _ in CFGS}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("\n已保存 g2_stats.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""公共15 的 31 个漏报逐例取证：为 V2 规则（上下文特征）提供设计依据。

对每个 FN（有效 GT 未匹配）输出：
- 标注帧该帧的原始检测状态（未检测/低置信/坐标）
- 局部 y 窗口（f-6 .. f+7，真实帧号；M=不合格点、X=无记录）
- 窗口内的局部极大及幅度（FALL/RISE）与相邻合格点间距
- 最近 V0 / V1(区域门) 发射的位置与帧差、像素距离
- 最近极大点的几何投影出桌距离（用于判断特征规则是否会被区域门拦截）
"""
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("gv1", "E:/pingpong-data/geometry_v1_20261012/ge_rule_v1.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
gt = G.load(G.GT_P)
gts_by_v = defaultdict(list)
for g in gt:
    gts_by_v[g["video_id"]].append(g)


def ywin_str(byin, lo, hi):
    parts = []
    for fr in range(lo, hi + 1):
        r = byin.get(fr + 1)  # input = 真实帧 + 1
        if r is None:
            parts.append(f"{fr}:X")
        elif not r.get("detected"):
            parts.append(f"{fr}:M")
        elif r.get("confidence") is None or r["confidence"] < 0.75:
            parts.append(f"{fr}:m{r['confidence']:.2f}")
        else:
            parts.append(f"{fr}:{r['y']:.0f}")
    return " ".join(parts)


total = 0
for vid, meta in man.items():
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    byin = {r["input_frame_id"]: r for r in rows}
    gts = gts_by_v.get(vid, [])
    preds0 = [{"video_id": vid, **e} for e in G.replay(rows, H, "v0")]
    G.MARGIN = 900.0
    preds1 = [{"video_id": vid, **e} for e in G.replay(rows, H, "v1a")]

    _, _, fn0 = G.match(preds0, gts)
    for j in fn0:
        g = gts[j]
        if not g.get("valid", True):
            continue
        total += 1
        f = int(g["frame_id"])
        rec = byin.get(f + 1)
        if rec is None:
            rawst = "X(无记录)"
        elif not rec.get("detected"):
            rawst = "未检测"
        else:
            c = rec.get("confidence")
            rawst = (f"det=({rec['x']:.0f},{rec['y']:.0f}) conf={c:.2f}" if c is not None
                     else "det conf=?")

        elig = [(r["input_frame_id"] - 1, r["y"]) for r in rows
                if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= 0.75]
        maxs = []
        for k in range(1, len(elig) - 1):
            fr, y = elig[k]
            pv, nx = elig[k - 1][1], elig[k + 1][1]
            if y > pv and y >= nx:
                maxs.append((fr, round(y - pv, 1), round(y - nx, 1),
                             elig[k][0] - elig[k - 1][0], elig[k + 1][0] - elig[k][0]))
        maxs_w = [m for m in maxs if abs(m[0] - f) <= 4]

        def nearest(preds):
            best = None
            for p in preds:
                d = math.dist((p["x"], p["y"]), (g["x"], g["y"]))
                if best is None or d < best[2]:
                    best = (p["frame_id"], abs(p["frame_id"] - f), round(d))
            return best

        n0, n1 = nearest(preds0), nearest(preds1)

        geo = "-"
        if maxs_w and rec is not None and rec.get("detected"):
            pass
        if maxs_w:
            mfr = maxs_w[0][0]
            p = byin.get(mfr + 1)
            if p is not None and p.get("detected"):
                u, v = G.project(H, p["x"], p["y"])
                geo = f"{max(u - G.TABLE_L, -u, v - G.TABLE_W, -v, 0):.0f}mm"

        ld = "-"
        if rec is not None and rec.get("detected") and rec.get("x") is not None:
            ld = f"{math.dist((rec['x'], rec['y']), (g['x'], g['y'])):.0f}px"

        print(f"\n{vid} f{f} | raw@f: {rawst} | 标注点距当帧球位: {ld}")
        print(f"   y窗(f{f-6}..f{f+7}): {ywin_str(byin, f - 6, f + 7)}")
        print(f"   窗内极大(fr,fall,rise,前间距,后间距): {maxs_w}")
        print(f"   最近V0发射: {n0} | 最近V1发射: {n1} | 最近极大几何出桌: {geo}")
        emis0 = [(p["frame_id"], round(p["x"]), round(p["y"])) for p in preds0 if abs(p["frame_id"] - f) <= 18]
        emis1 = [(p["frame_id"], round(p["x"]), round(p["y"])) for p in preds1 if abs(p["frame_id"] - f) <= 18]
        print(f"   V0发射(±18帧): {emis0}")
        print(f"   V1发射(±18帧): {emis1}")

print(f"\n共 {total} 例")

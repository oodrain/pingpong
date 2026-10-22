# -*- coding: utf-8 -*-
"""r2_eval 配套诊断：
A) v000095 f107 / v000106 f134 为何在组合配置下仍未恢复（打印预测/GT/匹配）
B) 补充配置：V1+F2、V1+F1 的完整指标
C) F3 素材：所有"通过幅度门但被 MIN_GAP 拦下"的候选 —— 距离上次发射的像素距离分布
   （真实相邻触台案例：f10-057≈120px / f111-117≈180px / f122-117≈436px / f134-106≈647px）
"""
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G = mod("gv1", "E:/pingpong-data/geometry_v1_20261012/ge_rule_v1.py")
R = mod("r2", str(HERE / "r2_eval.py"))

man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
gt = G.load(G.GT_P)
gts_by_v = defaultdict(list)
for g in gt:
    gts_by_v[g["video_id"]].append(g)

MIN_GAP, CONF_TH = 10, 0.75
CFG_FULL = {"gate": 900, "F1": True, "F2": True, "F3": True}

print("========== A) 两例调试（组合配置下） ==========")
for vid, lo, hi in (("v000095", 95, 122), ("v000106", 124, 140)):
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    gts = gts_by_v[vid]
    preds = [{"video_id": vid, **e} for e in R.replay_v2(rows, H, CFG_FULL)]
    m, fp, fn = G.match(preds, gts)
    print(f"\n{vid} 预测 f{lo}..{hi}: {[(p['frame_id'], round(p['x']), round(p['y'])) for p in preds if lo <= p['frame_id'] <= hi]}")
    print(f"{vid} GT   f{lo}..{hi}: {[(g['frame_id'], round(g['x']), round(g['y']), g.get('valid', True)) for g in gts if lo <= g['frame_id'] <= hi]}")
    pairs = [(preds[i]["frame_id"], gts[j]["frame_id"]) for i, j in m
             if lo <= gts[j]["frame_id"] <= hi or lo <= preds[i]["frame_id"] <= hi]
    print("匹配对(pred帧→gt帧):", pairs)
    print("该范围FP:", [(preds[i]["frame_id"], round(preds[i]["x"]), round(preds[i]["y"])) for i in fp if lo <= preds[i]["frame_id"] <= hi])

print("\n========== B) 补充配置 ==========")
videos = {}
for vid in man:
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"))

for name, cfg in (("V1+F2", {"gate": 900, "F2": True}), ("V1+F1", {"gate": 900, "F1": True})):
    cells = []
    for a in ("0度", "45度", "90度", "全部"):
        tp = fp_n = fn_n = 0
        for vid, (H, rows, gts, ang) in videos.items():
            if a != "全部" and ang != a:
                continue
            preds = [{"video_id": vid, **e} for e in R.replay_v2(rows, H, cfg)]
            m, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in m if gts[j].get("valid", True))
            tp += mv
            fp_n += len(fp) + (len(m) - mv)
            fn_n += sum(1 for j in fn if gts[j].get("valid", True))
        f1 = 2 * tp / (2 * tp + fp_n + fn_n) if tp else 0.0
        cells.append(f"{tp}/{fp_n}/{fn_n}/{f1:.3f}")
    print(f"{name:8s} | 0度 {cells[0]} | 45度 {cells[1]} | 90度 {cells[2]} | 全部 {cells[3]}")

print("\n========== C) 被 MIN_GAP 拦下的候选（通过幅度门，F1/F2 宽口径） ==========")
blocked = []
for vid, (H, rows, gts, ang) in videos.items():
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    last_fid, last_xy = -10_000, None
    for j in range(1, n - 1):
        fid, x, y = traj[j]
        if not (y > traj[j - 1][2] and y >= traj[j + 1][2]):
            continue
        fall1 = y - traj[j - 1][2]
        rise1 = y - traj[j + 1][2]
        fall2 = y - traj[j - 2][2] if j >= 2 else fall1
        rise2 = y - traj[j + 2][2] if j + 2 < n else rise1
        ampl = (fall1 >= 2.0 and rise1 >= 1.0) or (fall2 >= 2.0 and rise2 >= 2.0) or (rise1 >= 8.0 and (j + 2 >= n or traj[j + 2][2] <= traj[j + 1][2]))
        if not ampl:
            continue
        if fid - last_fid < MIN_GAP:
            d = math.dist((x, y), last_xy) if last_xy else 0
            blocked.append((round(d), vid, fid - 1, round(fall1, 1), round(rise1, 1), last_fid - 1))
            continue
        last_fid, last_xy = fid, (x, y)

print(f"共 {len(blocked)} 个被拦截候选；按距离降序（截取前 40）：")
for b in sorted(blocked, reverse=True)[:40]:
    print("  dist=%4dpx  %s f%d  fall=%.1f rise=%.1f  拦者=f%d" % b)
print("\n距离分布（含全部）:")
import collections
buckets = collections.Counter()
for d, *_ in blocked:
    b = "<40" if d < 40 else ("40-79" if d < 80 else ("80-119" if d < 120 else ("120-199" if d < 200 else ">=200")))
    buckets[b] += 1
print(dict(buckets))

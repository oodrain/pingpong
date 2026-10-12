# -*- coding: utf-8 -*-
"""区域门边距选型的质量分析：
1) V0 发射点中「真触台（TP）」与「误报（FP）」的投影出桌距离分布 → 看可分界；
2) 全量 F1-边距曲线；
3) 留一交叉验证（LOOCV）：在 14 条上选边距、在留出 1 条上验证增益，判断是否过拟合。
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gv1", HERE / "ge_rule_v1.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
gt = G.load(G.GT_P)
gt_by_v = {}
for g in gt:
    gt_by_v.setdefault(g["video_id"], []).append(g)

margins = [400, 600, 800, 900, 1000, 1200]

tp_out, fp_out = [], []
per_video = {}   # vid -> {"v0": (tp,fp,fn), margin -> (tp,fp,fn)}
for vid in man:
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    gts = gt_by_v.get(vid, [])
    res = {}

    def run(mode):
        preds = [{"video_id": vid, **e} for e in G.replay(rows, H, mode)]
        m, fp, fn = G.match(preds, gts)
        fn_valid = sum(1 for j in fn if gts[j].get("valid", True))
        m_valid = sum(1 for i, j in m if gts[j].get("valid", True))
        fp_a = len(fp) + (len(m) - m_valid)
        return preds, m, fn_valid, m_valid, fp_a

    preds, m, fnv, mv, fpa = run("v0")
    res["v0"] = (mv, fpa, fnv)
    for i, j in m:
        u, v = G.project(H, preds[i]["x"], preds[i]["y"])
        od = max(u - G.TABLE_L, -u, v - G.TABLE_W, -v, 0.0)
        (tp_out if gts[j].get("valid", True) else fp_out).append(od)
    for i in range(len(preds)):
        if all(i != a for a, _ in m):
            u, v = G.project(H, preds[i]["x"], preds[i]["y"])
            fp_out.append(max(u - G.TABLE_L, -u, v - G.TABLE_W, -v, 0.0))
    for mrg in margins:
        G.MARGIN = float(mrg)
        _, m2, fnv2, mv2, fpa2 = run("v1a")
        res[mrg] = (mv2, fpa2, fnv2)
    per_video[vid] = res

G.MARGIN = 900.0


def stats(a):
    a = np.array(a)
    return (len(a), np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 95), np.percentile(a, 99), a.max())


n, p50, p90, p95, p99, mx = stats(tp_out)
print(f"真触台(TP) {n} 个，投影出桌距离mm: 50%={p50:.0f} 90%={p90:.0f} 95%={p95:.0f} 99%={p99:.0f} max={mx:.0f}")
n, p50, p90, p95, p99, mx = stats(fp_out)
print(f"误报(FP) {n} 个，投影出桌距离mm: 50%={p50:.0f} 90%={p90:.0f} 95%={p95:.0f} 99%={p99:.0f} max={mx:.0f}")
print()
for mrg in margins:
    lose = sum(1 for d in tp_out if d > mrg)
    kill = sum(1 for d in fp_out if d > mrg)
    print(f"边距 {mrg:>5}mm: 误杀TP {lose:>2} / 杀掉FP {kill:>3}  (净收益 若按 FP×1 vs TP×2 权重 = {kill - 2 * lose})")

print("\n全量 F1 曲线:")
for mrg in margins:
    tp = sum(per_video[v][mrg][0] for v in man)
    fp = sum(per_video[v][mrg][1] for v in man)
    fn = sum(per_video[v][mrg][2] for v in man)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"  {mrg:>5}mm: TP{tp} FP{fp} FN{fn} F1={f1:.3f}")

print("\nLOOCV（在14条上选F1最优边距，验证留出条）:")
chosen = []
gains = []
for v in man:
    others = [u for u in man if u != v]

    def f1_of(u, mrg):
        tp, fp, fn = per_video[u][mrg]
        return 2 * tp / (2 * tp + fp + fn) if tp * 2 + fp + fn else 0.0

    def f1_v(u, mrg):
        tp, fp, fn = per_video[u][mrg]
        return 2 * tp / (2 * tp + fp + fn) if tp * 2 + fp + fn else 0.0

    best_m = max(margins, key=lambda m: sum(f1_of(u, m) for u in others))
    others_f1 = sum(f1_of(u, best_m) for u in others)
    v0_f1 = sum(f1_v(u, "v0") for u in others)
    chosen.append(best_m)
    gains.append(f1_v(v, best_m) - f1_v(v, "v0"))
from collections import Counter
print("  各条被选中的边距分布:", dict(sorted(Counter(chosen).items())))
print(f"  留出条平均增益: {np.mean(gains):+.3f}（min {min(gains):+.3f}, max {max(gains):+.3f}）")
print(f"  全量增益（900mm 相对 V0）: ", end="")
tp0 = sum(per_video[v]["v0"][0] for v in man); fp0 = sum(per_video[v]["v0"][1] for v in man); fn0 = sum(per_video[v]["v0"][2] for v in man)
f1_0 = 2 * tp0 / (2 * tp0 + fp0 + fn0)
tp9 = sum(per_video[v][900][0] for v in man); fp9 = sum(per_video[v][900][1] for v in man); fn9 = sum(per_video[v][900][2] for v in man)
f1_9 = 2 * tp9 / (2 * tp9 + fp9 + fn9)
print(f"{f1_9 - f1_0:+.3f}")

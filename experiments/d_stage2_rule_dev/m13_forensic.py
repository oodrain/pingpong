# -*- coding: utf-8 -*-
"""M1.3 取证：v2 标签下 154 剩余漏报的机制分解。

对 F1F2F3 配置（无门）回放后仍未命中的每个有效标签，按回放内部判定链逐环归因：
  no_detect / low_conf       —— 标签帧无达标检测
  edge / not_max             —— 不被扫描的边界点 / y 非局部极大
  ampl_fail                  —— 是极大但幅度链全失败（记录 fall1/rise1/fall2/rise2 与全片典型位移）
  blocked_rej                —— 通过幅度但被 MIN_GAP 拦截且 F3 守卫条件不全过（记录守卫值）
  emitted_offset             —— 实际有发射但帧差>1 或距离>50（位移/位置问题）
另：按视角（0/45/90）汇总；输出样本供 M1.3（幅度相对化）设计。
"""
import importlib.util
import json
import math
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
MIN_GAP, CONF_TH = 10, 0.75


def attr(vid, ev_list, g, preds):
    f = int(g["frame_id"])
    fid = f + 1
    rows = attr.rows_cache.get(vid)
    if rows is None:
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        attr.rows_cache[vid] = rows
    by = {r["input_frame_id"]: r for r in rows}
    rec = by.get(fid)
    if rec is None:
        return "out_of_range", {}, None
    if not rec.get("detected"):
        return "no_detect", {}, None
    if rec.get("confidence") is None or rec["confidence"] < CONF_TH:
        return "low_conf", {"conf": round(rec.get("confidence") or 0, 2)}, None
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    j = next((k for k, t in enumerate(traj) if t[0] == fid), None)
    if j is None:
        return "not_in_traj", {}, None
    ds = [math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) for k in range(n - 1)]
    v_med = float(np.median(ds)) if ds else 0.0
    if j == 0 or j >= n - 1 or (j >= n - 2):
        # 实际扫描范围 range(1,n-2) + {n-2}
        if not (1 <= j <= n - 2):
            return "edge", {}, v_med
    if not (traj[j][2] > traj[j - 1][2] and traj[j][2] >= traj[j + 1][2]):
        info = {"prev": round(traj[j - 1][2]), "y": round(traj[j][2]), "next": round(traj[j + 1][2]),
                "v_med": round(v_med, 1)}
        # 附近 ±4 帧内是否有极大（位移类）
        near_max = None
        for k in range(max(1, j - 4), min(n - 1, j + 5)):
            if traj[k][2] > traj[k - 1][2] and traj[k][2] >= traj[k + 1][2]:
                if near_max is None or abs(traj[k][0] - fid) < abs(traj[near_max][0] - fid):
                    near_max = k
        if near_max is not None:
            info["near_max"] = f"f{traj[near_max][0] - 1}（Δ{traj[near_max][0] - fid:+d}）"
            return "nm_邻近极大(位移类)", info, v_med
        if traj[j][2] < traj[j - 1][2]:
            return "nm_标注帧上行", info, v_med
        return "nm_继续下行无极大", info, v_med
    fall1 = traj[j][2] - traj[j - 1][2]
    rise1 = traj[j][2] - traj[j + 1][2]
    fall2 = traj[j][2] - traj[j - 2][2] if j >= 2 else fall1
    rise2 = traj[j][2] - traj[j + 2][2] if j + 2 < n else rise1
    ampl_base = fall1 >= 2.0 and rise1 >= 1.0
    f2 = fall2 >= 2.0 and rise2 >= 2.0
    f1 = rise1 >= 8.0 and (j + 2 >= n or traj[j + 2][2] <= traj[j + 1][2])
    if not (ampl_base or f2 or f1):
        info = {"fall1": round(fall1, 1), "rise1": round(rise1, 1),
                "fall2": round(fall2, 1), "rise2": round(rise2, 1), "v_med": round(v_med, 1)}
        return "ampl_fail", info, v_med
    # 通过幅度：检查附近发射（±4 帧内）反推
    near_emis = [(p["frame_id"] - f, round(math.dist((p["x"], p["y"]), (g["x"], g["y"])))) for p in preds
                 if abs(p["frame_id"] - f) <= 4]
    if near_emis:
        near_emis.sort(key=lambda t: (abs(t[0]), t[1]))
        dlt, dd = near_emis[0]
        return "有发射但未配(位移/占用)", {"Δ": dlt, "dist": dd}, v_med
    return "gap拦截(F3拒绝)", {"fall1": round(fall1, 1), "rise1": round(rise1, 1)}, v_med


attr.rows_cache = {}


def main():
    import csv
    ang = {}
    for r in csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines()):
        ang[r["video_id"]] = int(r["angle_degrees"])
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)

    cats = Counter()
    cats_angle = defaultdict(Counter)
    samples = defaultdict(list)
    for vid in sorted(by_v):
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, None, CFG)]
        mm_, fp, fn = G.match(preds, by_v[vid])
        for jj in fn:
            g = by_v[vid][jj]
            if not g.get("valid", True):
                continue
            cat, info, v_med = attr(vid, None, g, preds)
            cats[cat] += 1
            cats_angle[ang.get(vid, 0)][cat] += 1
            if len(samples[cat]) < 12:
                samples[cat].append((vid, g["frame_id"], info))

    tot = sum(cats.values())
    print(f"154 剩余漏报（F1F2F3，v2 标签）: {tot} 例")
    print(f"{'类别':16s} {'全部':>5s} | {'0°':>4s} {'45°':>4s} {'90°':>4s}")
    for c, n in cats.most_common():
        a = cats_angle
        print(f"{c:16s} {n:>5d} | {a[0][c]:>4d} {a[45][c]:>4d} {a[90][c]:>4d}")
    print()
    for c, ss in samples.items():
        print(f"--- {c} 样例 ---")
        for vid, f, info in ss[:8]:
            print(f"  {vid} f{f}: {info}")
        print()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""上行族取证：v2 标签下"标注帧上行"漏报的逐例详情与聚合。

筛选条件（与 m13_forensic 的 nm_标注帧上行 一致）：
F1F2F3 回放后未命中的标签，其帧在达标轨迹中不是局部极大、且 y[j] < y[j-1]（球在上升）。

对每例输出：
- y 窗口 f-8..f+8（达标点；X=无）；
- 标注帧最近达标点的位置与距离；
- **标签前** 最近局部极大（15 帧内）的帧差 Δ 与位置距标签距离；**标签后** 最近极大同理；
- ±15 帧内最近发射（Δ、距离）；标注帧置信度；v_med。
聚合：前峰/后峰/无峰的分布、Δ 分布、距离分布。
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
CONF_TH = 0.75


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

    cases = []
    for vid in sorted(by_v):
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
                if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
        by_in = {r["input_frame_id"]: r for r in rows}
        n = len(traj)
        if n < 3:
            continue
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, None, CFG)]
        mm_, fp, fn = G.match(preds, by_v[vid])
        ds = [math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) for k in range(n - 1)]
        v_med = float(np.median(ds)) if ds else 0.0
        for jj in fn:
            g = by_v[vid][jj]
            f = int(g["frame_id"])
            fid = f + 1
            j = next((k for k, t in enumerate(traj) if t[0] == fid), None)
            if j is None or j == 0 or j > n - 2:
                continue
            is_max = traj[j][2] > traj[j - 1][2] and traj[j][2] >= traj[j + 1][2]
            if is_max:
                continue
            near_max = any(traj[k][2] > traj[k - 1][2] and traj[k][2] >= traj[k + 1][2]
                           for k in range(max(1, j - 4), min(n - 1, j + 5)))
            if near_max:
                continue  # 属"邻近极大"类，不归本族（与 m13 分类一致）
            if not (traj[j][2] < traj[j - 1][2]):
                continue
            # —— 上行族 ——
            parts = []
            traj_by_f = {t[0] - 1: t for t in traj}
            for fr in range(f - 8, f + 9):
                t = traj_by_f.get(fr)
                r = by_in.get(fr + 1)
                if t is not None:
                    parts.append(f"{fr}:{t[2]:.0f}")
                elif r is None:
                    parts.append(f"{fr}:X")
                else:
                    parts.append(f"{fr}:x")  # 有记录但未达标
            win = " ".join(parts)
            # 最近达标点
            near = min(traj, key=lambda t: abs(t[0] - 1 - f))
            npt_dist = math.dist((near[1], near[2]), (g["x"], g["y"]))
            # 前/后最近极大
            pre = post = None
            for k in range(1, n - 1):
                if traj[k][2] > traj[k - 1][2] and traj[k][2] >= traj[k + 1][2]:
                    dfr = (traj[k][0] - 1) - f
                    if -15 <= dfr <= -1:
                        if pre is None or abs(dfr) < abs((traj[pre][0] - 1) - f):
                            pre = k
                    if 1 <= dfr <= 15:
                        if post is None or abs(dfr) < abs((traj[post][0] - 1) - f):
                            post = k
            pre_info = post_info = None
            if pre is not None:
                pre_info = {"Δ": (traj[pre][0] - 1) - f,
                            "dist": round(math.dist((traj[pre][1], traj[pre][2]), (g["x"], g["y"])))}
            if post is not None:
                post_info = {"Δ": (traj[post][0] - 1) - f,
                             "dist": round(math.dist((traj[post][1], traj[post][2]), (g["x"], g["y"])))}
            emis = [(p["frame_id"] - f, round(math.dist((p["x"], p["y"]), (g["x"], g["y"]))))
                    for p in preds if abs(p["frame_id"] - f) <= 15]
            emis.sort(key=lambda t: (abs(t[0]), t[1]))
            rec = by_in.get(fid)
            cases.append({
                "vid": vid, "angle": ang.get(vid, -1), "f": f,
                "label": [round(g["x"]), round(g["y"])],
                "conf": round(rec["confidence"], 2) if rec and rec.get("confidence") else None,
                "v_med": round(v_med, 1), "near_dist": round(npt_dist),
                "pre": pre_info, "post": post_info,
                "emis": emis[:3], "win": win,
            })

    print(f"上行族共 {len(cases)} 例\n")
    agg = Counter()
    for c in cases:
        if c["pre"] and c["post"]:
            agg["前后都有峰"] += 1
        elif c["pre"]:
            agg["仅前峰"] += 1
        elif c["post"]:
            agg["仅后峰"] += 1
        else:
            agg["无峰(±15)"] += 1
    print("峰分布:", dict(agg))
    pre_d = Counter(c["pre"]["Δ"] for c in cases if c["pre"])
    print("前峰 Δ 分布:", dict(sorted(pre_d.items())))
    pre_dist_ok = sum(1 for c in cases if c["pre"] and c["pre"]["dist"] <= 50)
    print(f"前峰位置距标签 ≤50px 的例数: {pre_dist_ok}/{sum(pre_d.values())}")
    emis_near = sum(1 for c in cases if c["emis"] and c["emis"][0][1] <= 50 and abs(c["emis"][0][0]) <= 1)
    print(f"±15 内有 ≤1 帧且 ≤50px 的发射(占用类): {emis_near}")
    confs = [c["conf"] for c in cases if c["conf"]]
    if confs:
        print(f"标注帧置信度: 中位 {np.median(confs):.2f}, 最低 {min(confs):.2f}")

    print("\n逐例：")
    for c in cases:
        print(f"[{c['vid']} {c['angle']}° f{c['f']}] label=({c['label'][0]},{c['label'][1]}) conf={c['conf']} "
              f"v_med={c['v_med']} 最近点距={c['near_dist']}px 前峰={c['pre']} 后峰={c['post']} emis={c['emis']}")
        print(f"   y窗口: {c['win']}")

    (HERE / "m13b_up_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n已保存 m13b_up_cases.json（{len(cases)} 例）")


if __name__ == "__main__":
    main()

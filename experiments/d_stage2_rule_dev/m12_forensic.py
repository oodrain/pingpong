# -*- coding: utf-8 -*-
"""M1.2 取证：接触时刻系统偏差（标注帧 vs 发射帧 vs 轨迹极大）。

对公开 15 的"仍漏 FN"（V1+F1F2F3 配置下）逐例：
- 列出 ±8 帧内所有未匹配发射（FP），给出 有符号帧差 Δ=发射帧−标注帧 与位置距离
- 列出 ±5/±6 帧轨迹窗口 y 序列、窗口内局部极大（帧号、幅度）
- 标签帧最近轨迹点的位置距离；窗口内相邻位移统计（快慢段度量）
另附：154 的 paired_time_offset 12 例（同类现象的既有 FP 证据）的 Δ/距离表。
"""
import importlib.util
import json
import math
import statistics
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

CFG_FULL = {"gate": 900, "F1": True, "F2": True, "F3": True}

man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
gt = G.load(G.GT_P)
gts_by_v = defaultdict(list)
for g in gt:
    gts_by_v[g["video_id"]].append(g)

cases = []
for vid, meta in man.items():
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    gts = gts_by_v.get(vid, [])
    preds = [{"video_id": vid, **e} for e in R.replay_v2(rows, H, CFG_FULL)]
    m, fp_idx, fn_idx = G.match(preds, gts)
    if not fn_idx:
        continue
    fps = [preds[i] for i in fp_idx]
    elig = [(r["input_frame_id"] - 1, r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= 0.75]
    for j in fn_idx:
        g = gts[j]
        if not g.get("valid", True):
            continue
        f = int(g["frame_id"])

        near_fps = sorted(
            [(p["frame_id"] - f, math.dist((p["x"], p["y"]), (g["x"], g["y"])), p["frame_id"],
              round(p["x"]), round(p["y"])) for p in fps if abs(p["frame_id"] - f) <= 8],
            key=lambda t: (abs(t[0]), t[1]))

        win = [(fr, round(x), round(y)) for fr, x, y in elig if f - 5 <= fr <= f + 6]
        win_y = {fr: y for fr, x, y in elig}
        maxs = []
        for k in range(1, len(elig) - 1):
            fr, x, y = elig[k]
            if f - 6 <= fr <= f + 6 and y > elig[k - 1][2] and y >= elig[k + 1][2]:
                maxs.append((fr, fr - f, round(y - elig[k - 1][2], 1), round(y - elig[k + 1][2], 1)))

        near_pt = min(win, key=lambda t: abs(t[0] - f)) if win else None
        pt_dist = round(math.dist((near_pt[1], near_pt[2]), (g["x"], g["y"]))) if near_pt else None

        # 窗口内相邻位移（快慢段：用 ±3 帧内相邻达标点位移）
        seg = [(fr, x, y) for fr, x, y in elig if f - 4 <= fr <= f + 2]
        ds = [round(math.dist((seg[k][1], seg[k][2]), (seg[k + 1][1], seg[k + 1][2])))
              for k in range(len(seg) - 1)]
        dv = [round(y2 - y1, 1) for (_, _, y1), (_, _, y2) in zip(seg, seg[1:])]

        cases.append({
            "video_id": vid, "angle": doc.get("angle"), "frame_id": f,
            "label": [round(g["x"]), round(g["y"])],
            "near_fps": near_fps, "maxs": maxs,
            "near_pt": near_pt, "pt_dist": pt_dist,
            "win": win, "win_y": win_y, "dxy": ds, "dy": dv,
        })

print(f"公开15 仍漏有效 FN: {len(cases)} 例（V1+F1F2F3 配置）\n")
for c in cases:
    print(f"[{c['video_id']} {c['angle']} f{c['frame_id']}] label=({c['label'][0]},{c['label'][1]})")
    fpstr = ["Δ%+d %dpx@f%d" % (d, dist, fr) for (d, dist, fr, _, _) in c["near_fps"]] or ["无"]
    print(f"  邻近FP: {fpstr}")
    print(f"  局部极大(帧,Δ,fall,rise): {c['maxs']}")
    print(f"  标签帧最近轨迹点 f{c['near_pt'][0]}({c['near_pt'][1]},{c['near_pt'][2]}) 距标签 {c['pt_dist']}px")
    print(f"  y窗口 f{f-5}..f{f+6}: " +
          " ".join(f"{fr}:{c['win_y'].get(fr, 'X')}" for fr in range(c['frame_id'] - 5, c['frame_id'] + 7)))
    print(f"  相邻位移(±3): {c['dxy']} | Δy: {c['dy']}")
    print()

print("\n========== 154 的 paired_time_offset（同类现象，FP 侧） ==========")
ev154 = [json.loads(l) for l in (HERE.parent / "error_analysis_v1_20261011" / "train154_events.jsonl")
         .read_text(encoding="utf-8").splitlines() if l.strip()]
rows154 = [e for e in ev154 if e["kind"] == "FP" and e["category"] == "paired_time_offset"]
print(f"共 {len(rows154)} 例：")
deltas = []
for e in rows154:
    g = e["pair"]
    d = round(math.dist((e["x"], e["y"]), (g["x"], g["y"])))
    df = int(e["frame_id"]) - int(g["frame_id"])
    deltas.append(df)
    print(f"  {e['video_id']} 发射f{e['frame_id']} ↔ 标签f{g['frame_id']}  Δ={df:+d}  位置差{d}px")
print(f"Δ 分布: {sorted(deltas)}")

(HERE / "m12_pairs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"\n已保存 m12_pairs.json（{len(cases)} 例）")

# -*- coding: utf-8 -*-
"""V2 规则第一迭代：上下文特征恢复 + 消融对照（公开15，官方GT）。

三个候选特征（基于 31 例漏报取证）：
- F1 强回升恢复：rise1 >= 8 且后续持续上升（放宽"下落幅度"要求；接触时下落被压平）
- F2 宽窗幅度：±2 窗口 fall2/rise2 均 >= 2（慢启动下落/延迟回升）
- F3 强候选破间隔：被 MIN_GAP 抑制时，若"强形状 + 距上次发射>=100px + 无跳点/缺口"则放行

实现要点（v2 修正）：
- 两遍扫描严格等价官方逐帧：pass1 扫 j<=n-3，pass2 只扫 j=n-2（修复重复发射 bug）
- 接触位置插值：非"基础幅度且未被拦截"的恢复型发射，用 frac=fall1/(fall1+rise1)
  在 j/j+1 之间插值出接触帧与球心（修复"位置差 13-17px 超容差"的近失案例）
"""
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("gv1", "E:/pingpong-data/geometry_v1_20261012/ge_rule_v1.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

MIN_GAP, CONF_TH = 10, 0.75

CFGS = [
    ("V0", {}),
    ("V1门900", {"gate": 900}),
    ("V0+F1", {"F1": True}),
    ("V0+F2", {"F2": True}),
    ("V0+F3", {"F3": True}),
    ("V1+F1", {"gate": 900, "F1": True}),
    ("V1+F2", {"gate": 900, "F2": True}),
    ("V1+F1F2", {"gate": 900, "F1": True, "F2": True}),
    ("V1+F1F2F3", {"gate": 900, "F1": True, "F2": True, "F3": True}),
]


def replay_v2(rows, H, cfg):
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    out = []
    last_fid = -10_000
    last_xy = None

    def scan(j):
        nonlocal last_fid, last_xy
        fid, x, y = traj[j]
        if not (y > traj[j - 1][2] and y >= traj[j + 1][2]):
            return
        fall1 = y - traj[j - 1][2]
        rise1 = y - traj[j + 1][2]
        fall2 = y - traj[j - 2][2] if j >= 2 else fall1
        rise2 = y - traj[j + 2][2] if j + 2 < n else rise1

        ampl_base = (fall1 >= 2.0 and rise1 >= 1.0)
        ampl = ampl_base
        if not ampl and cfg.get("F2") and fall2 >= 2.0 and rise2 >= 2.0:
            ampl = True
        if not ampl and cfg.get("F1") and rise1 >= 8.0 and (j + 2 >= n or traj[j + 2][2] <= traj[j + 1][2]):
            ampl = True
        if not ampl:
            return

        if cfg.get("gate"):
            u, v = G.project(H, x, y)
            g = cfg["gate"]
            if not (-g <= u <= G.TABLE_L + g and -g <= v <= G.TABLE_W + g):
                return

        blocked = fid - last_fid < MIN_GAP
        if blocked:
            if not cfg.get("F3"):
                return
            strong = ((fall1 >= 3.0 and rise1 >= 3.0)
                      or (fall2 >= 3.0 and rise2 >= 5.0)
                      or (rise1 >= 8.0 and fall1 >= 0.5))
            far_ok = last_xy is not None and math.dist((x, y), last_xy) >= 100.0
            jump_ok = True
            for k in range(max(1, j - 2), min(n - 1, j + 2)):
                if math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) > 120.0:
                    jump_ok = False
                    break
            gap_ok = ((j < 2 or traj[j][0] - traj[j - 1][0] <= 2)
                      and (traj[j + 1][0] - traj[j][0] <= 2)
                      and (j + 2 >= n or traj[j + 2][0] - traj[j + 1][0] <= 2))
            if not (strong and far_ok and jump_ok and gap_ok):
                return

        if ampl_base and not blocked:
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y})
        else:
            denom = fall1 + rise1
            frac = fall1 / denom if denom > 0.5 else 0.5
            frac = min(1.0, max(0.0, frac))
            x2, y2 = traj[j + 1][1], traj[j + 1][2]
            out.append({"frame_id": max(0, int(round((fid - 1) + frac))),
                        "x": x + frac * (x2 - x), "y": y + frac * (y2 - y)})
        last_fid = fid
        last_xy = (x, y)

    for j in range(1, n - 2):
        scan(j)
    if n >= 3:
        scan(n - 2)
    return out


def main():
    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    gts_by_v = defaultdict(list)
    for g in gt:
        gts_by_v[g["video_id"]].append(g)

    videos = {}
    for vid in man:
        doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
        H = G.build_H(doc["segments"][0])
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"))

    angles = ["0度", "45度", "90度", "全部"]
    results, fn_detail = {}, {}
    for name, cfg in CFGS:
        per_video, fn_sets = {}, {}
        for vid, (H, rows, gts, _) in videos.items():
            preds = [{"video_id": vid, **e} for e in replay_v2(rows, H, cfg)]
            m, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in m if gts[j].get("valid", True))
            fp_a = len(fp) + (len(m) - mv)
            fnv = sum(1 for j in fn if gts[j].get("valid", True))
            per_video[vid] = (mv, fp_a, fnv, len(preds))
            fn_sets[vid] = {gts[j]["frame_id"] for j in fn if gts[j].get("valid", True)}
        results[name] = per_video
        fn_detail[name] = fn_sets

    print(f"{'配置':12s} | {'0度':>14s} | {'45度':>14s} | {'90度':>14s} | {'全部':>16s}")
    for name, _ in CFGS:
        cells = []
        for a in angles:
            vs = [v for v in videos if a == "全部" or videos[v][3] == a]
            tp = sum(results[name][v][0] for v in vs)
            fp = sum(results[name][v][1] for v in vs)
            fn = sum(results[name][v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
        print(f"{name:12s} | {cells[0]:>14s} | {cells[1]:>14s} | {cells[2]:>14s} | {cells[3]:>16s}")

    base = fn_detail["V0"]
    for key in ("V1+F1F2", "V1+F1F2F3"):
        fin = fn_detail[key]
        recovered = [f"{vid} f{fr}" for vid in videos for fr in sorted(base.get(vid, set())) if fr not in fin.get(vid, set())]
        still = [f"{vid} f{fr}" for vid in videos for fr in sorted(base.get(vid, set())) if fr in fin.get(vid, set())]
        print(f"\n[{key}] 恢复 {len(recovered)}: {recovered}")
        print(f"[{key}] 仍漏 {len(still)}: {still}")

    (Path(__file__).resolve().parent / "r2_stats.json").write_text(
        json.dumps({n: {v: results[n][v] for v in videos} for n, _ in CFGS}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("\n已保存 r2_stats.json")


if __name__ == "__main__":
    main()

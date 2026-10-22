# -*- coding: utf-8 -*-
"""M1.2 追踪：对公开15 的每个仍漏 FN，用带日志的回放逐点打印拒绝原因。

与 r2_eval.replay_v2 完全同逻辑（每个点都走全程，只对标签 ±7 帧内的点记录日志），
保证 last_fid / last_xy 状态一致。
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

CFG = {"gate": 900, "F1": True, "F2": True, "F3": True}
MIN_GAP, CONF_TH = 10, 0.75


def replay_trace(rows, H, cfg, f0, f1, log):
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    out = []
    last_fid, last_xy = -10_000, None

    def scan(j):
        nonlocal last_fid, last_xy
        fid, x, y = traj[j]
        w = f0 <= fid - 1 <= f1
        if not (y > traj[j - 1][2] and y >= traj[j + 1][2]):
            if w:
                log.append(f"    f{fid - 1} y={y:.0f}: 非极大 (p={traj[j - 1][2]:.0f}, n={traj[j + 1][2]:.0f})")
            return
        fall1 = y - traj[j - 1][2]
        rise1 = y - traj[j + 1][2]
        fall2 = y - traj[j - 2][2] if j >= 2 else fall1
        rise2 = y - traj[j + 2][2] if j + 2 < n else rise1
        ampl_base = (fall1 >= 2.0 and rise1 >= 1.0)
        ampl = ampl_base
        tag = "base" if ampl_base else ""
        if not ampl and cfg.get("F2") and fall2 >= 2.0 and rise2 >= 2.0:
            ampl, tag = True, "F2"
        if not ampl and cfg.get("F1") and rise1 >= 8.0 and (j + 2 >= n or traj[j + 2][2] <= traj[j + 1][2]):
            ampl, tag = True, "F1"
        if not ampl:
            if w:
                log.append(f"    f{fid - 1} y={y:.0f}: 极大但幅度不足 "
                           f"(fall1={fall1:.1f} rise1={rise1:.1f} fall2={fall2:.1f} rise2={rise2:.1f})")
            return
        gate_note = "无门"
        if cfg.get("gate"):
            u, v = G.project(H, x, y)
            g = cfg["gate"]
            out_mm = max(u - G.TABLE_L, -u, v - G.TABLE_W, -v, 0)
            if not (-g <= u <= G.TABLE_L + g and -g <= v <= G.TABLE_W + g):
                if w:
                    log.append(f"    f{fid - 1} y={y:.0f} [{tag}]: 【几何门拦截】"
                               f"投影({u:.0f},{v:.0f}) 出桌 {out_mm:.0f}mm > {g}")
                return
            gate_note = f"门过(出桌{out_mm:.0f}mm)"
        blocked = fid - last_fid < MIN_GAP
        if blocked:
            if not cfg.get("F3"):
                if w:
                    log.append(f"    f{fid - 1} y={y:.0f} [{tag}] {gate_note}: 【MIN_GAP拦截】(上次发射fid={last_fid})")
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
                if w:
                    log.append(f"    f{fid - 1} y={y:.0f} [{tag}] {gate_note}: 【MIN_GAP拦截】"
                               f"(上次fid={last_fid}) F3: strong={strong} far={far_ok} "
                               f"jump={jump_ok} gap={gap_ok}")
                return
            if w:
                log.append(f"    f{fid - 1} y={y:.0f} [{tag}] {gate_note}: MIN_GAP拦截但F3放行 → 发射")
        else:
            if w:
                log.append(f"    f{fid - 1} y={y:.0f} [{tag}] {gate_note}: 发射")
        if ampl_base and not blocked:
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y, "why": "base"})
        else:
            denom = fall1 + rise1
            frac = fall1 / denom if denom > 0.5 else 0.5
            frac = min(1.0, max(0.0, frac))
            x2, y2 = traj[j + 1][1], traj[j + 1][2]
            out.append({"frame_id": max(0, int(round((fid - 1) + frac))),
                        "x": x + frac * (x2 - x), "y": y + frac * (y2 - y), "why": tag})
        last_fid, last_xy = fid, (x, y)

    for j in range(1, n - 2):
        scan(j)
    if n >= 3:
        scan(n - 2)
    return out


man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
gt = G.load(G.GT_P)
gts_by_v = defaultdict(list)
for g in gt:
    gts_by_v[g["video_id"]].append(g)

pairs = json.loads((HERE / "m12_pairs.json").read_text(encoding="utf-8"))
by_vid = defaultdict(list)
for c in pairs:
    by_vid[c["video_id"]].append(c)

for vid, cases in sorted(by_vid.items()):
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    for c in cases:
        f = c["frame_id"]
        log = []
        emis = replay_trace(rows, H, CFG, f - 7, f + 7, log)
        near_emis = [(e["frame_id"], round(e["x"]), round(e["y"]), e.get("why")) for e in emis
                     if abs(e["frame_id"] - f) <= 12]
        print(f"=== {vid} FN f{f} label=({c['label'][0]},{c['label'][1]}) ===")
        print(f"  标签±12 发射: {near_emis}")
        for line in log:
            print(line)
        print()

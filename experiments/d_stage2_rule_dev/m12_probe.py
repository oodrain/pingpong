# -*- coding: utf-8 -*-
"""探针：列出 D2 每次回溯的（j→k, s_prev, s_at, ratio, 距离）及最终命中去留。
用于找出"真回溯"（命中/修复）与"假回溯"（破坏原命中/变 FP）的判别特征。
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
MIN_GAP, CONF_TH = 10, 0.75
CFG = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
       "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2}
BASE = {"gate": 900, "F1": True, "F2": True, "F3": True}


def replay(rows, H, cfg, shifts=None):
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    out = []
    last_fid, last_xy = -10_000, None

    def d_trigger(k):
        if k < 1 or k + 1 >= n:
            return False
        if traj[k][2] <= traj[k - 1][2]:
            return False
        s_prev = traj[k][2] - traj[k - 1][2]
        s_at = traj[k + 1][2] - traj[k][2]
        if (cfg.get("D_smin", 8.0) <= s_prev <= cfg.get("D_smax", 1e9) and s_at > 0
                and s_at <= cfg.get("D_ratio", 0.45) * s_prev
                and traj[k + 1][0] - traj[k][0] <= 2):
            return s_prev, s_at
        return False

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
        j_emit = j
        sh = None
        if cfg.get("D2"):
            for k in range(j - 1, max(0, j - 1 - cfg.get("D2_back", 5)), -1):
                if j - k < cfg.get("D_dmin", 1):
                    continue
                t = d_trigger(k)
                if t:
                    j_emit = k
                    sh = (traj[j][0] - 1, traj[k][0] - 1, t[0], t[1], j - k)
                    break
        efid, ex, ey = traj[j_emit]
        if cfg.get("gate"):
            u, v = G.project(H, ex, ey)
            g = cfg["gate"]
            if not (-g <= u <= G.TABLE_L + g and -g <= v <= G.TABLE_W + g):
                return
        blocked = efid - last_fid < MIN_GAP
        if blocked:
            if not cfg.get("F3"):
                return
            sf1 = ey - traj[j_emit - 1][2] if j_emit >= 1 else fall1
            sr1 = ey - traj[j_emit + 1][2] if j_emit + 1 < n else rise1
            sf2 = ey - traj[j_emit - 2][2] if j_emit >= 2 else sf1
            sr2 = ey - traj[j_emit + 2][2] if j_emit + 2 < n else sr1
            strong = ((sf1 >= 3.0 and sr1 >= 3.0) or (sf2 >= 3.0 and sr2 >= 5.0)
                      or (sr1 >= 8.0 and sf1 >= 0.5))
            far_ok = last_xy is not None and math.dist((ex, ey), last_xy) >= 100.0
            jump_ok = True
            for kk in range(max(1, j_emit - 2), min(n - 1, j_emit + 2)):
                if math.dist((traj[kk][1], traj[kk][2]), (traj[kk + 1][1], traj[kk + 1][2])) > 120.0:
                    jump_ok = False
                    break
            gap_ok = ((j_emit < 2 or traj[j_emit][0] - traj[j_emit - 1][0] <= 2)
                      and (traj[j_emit + 1][0] - traj[j_emit][0] <= 2)
                      and (j_emit + 2 >= n or traj[j_emit + 2][0] - traj[j_emit + 1][0] <= 2))
            if not (strong and far_ok and jump_ok and gap_ok):
                return
        if j_emit != j:
            out.append({"frame_id": max(0, efid - 1), "x": ex, "y": ey, "why": "shift"})
            if shifts is not None:
                shifts.append(sh)
        elif ampl_base and not blocked:
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y, "why": "base"})
        else:
            denom = fall1 + rise1
            frac = fall1 / denom if denom > 0.5 else 0.5
            frac = min(1.0, max(0.0, frac))
            x2, y2 = traj[j + 1][1], traj[j + 1][2]
            out.append({"frame_id": max(0, int(round((fid - 1) + frac))),
                        "x": x + frac * (x2 - x), "y": y + frac * (y2 - y), "why": "interp"})
        last_fid, last_xy = efid, (ex, ey)

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

print("回溯明细（j=原极大帧, k=回溯帧, s_prev/s_at/ratio, d=回溯距离）| 结果: 命中/新FP/丢失/无关")
for vid in man:
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    H = G.build_H(doc["segments"][0])
    rows = G.load(G.RAW / f"{vid}.jsonl")
    rows.sort(key=lambda r: r["input_frame_id"])
    gts = gts_by_v.get(vid, [])
    shifts = []
    preds_d2 = [{"video_id": vid, **e} for e in replay(rows, H, CFG, shifts)]
    preds_b = [{"video_id": vid, **e} for e in replay(rows, H, BASE)]
    m_d2, fp_d2, fn_d2 = G.match(preds_d2, gts)
    m_b, fp_b, fn_b = G.match(preds_b, gts)
    matched_labels_b = {gts[j]["frame_id"] for i, j in m_b if gts[j].get("valid", True)}
    matched_labels_d2 = {gts[j]["frame_id"] for i, j in m_d2 if gts[j].get("valid", True)}
    fp_frames_d2 = {preds_d2[i]["frame_id"] for i in fp_d2}
    if not shifts:
        continue
    print(f"\n{vid}:")
    for (jf, kf, sp, sa, dd) in shifts:
        tag = []
        # 回溯点附近是否有标签（±1）
        near_lab = [g["frame_id"] for g in gts if abs(g["frame_id"] - kf) <= 1
                    and math.dist((g["x"], g["y"]), (0, 0)) >= 0]
        k_lab = [g["frame_id"] for g in gts if abs(g["frame_id"] - kf) <= 1]
        j_lab = [g["frame_id"] for g in gts if abs(g["frame_id"] - jf) <= 1]
        res = []
        for lf in k_lab:
            if lf in matched_labels_d2 and lf not in matched_labels_b:
                res.append(f"修复FN(label f{lf})")
            elif lf in matched_labels_d2:
                res.append(f"命中(label f{lf},基线也命中)")
        for lf in j_lab:
            if lf in matched_labels_b and lf not in matched_labels_d2:
                res.append(f"!!丢失原命中(label f{lf})")
        if kf in fp_frames_d2:
            res.append("新FP")
        if not res:
            res.append("无标签相关")
        print(f"  j f{jf} → k f{kf}  s_prev={sp:.1f} s_at={sa:.1f} r={sa/max(sp,0.1):.2f} d={dd} | {', '.join(res)}")

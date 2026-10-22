# -*- coding: utf-8 -*-
"""M1.2 评估：D 特征（急减速触台恢复）+ F3 守卫微调 的消融矩阵与留一验证。

背景（m12_trace/m12_forensic 取证）：
- 仅靠 y 局部极大的发射点，在快速段会偏离标注帧 2–5 帧（FP+FN 双罚）；
- 真实触台的图像特征 = 下落速度骤降（球在触台瞬间减速/反向，被拖影与趋近相机效应掩盖）；
- D 特征：对"仍在下降但速度骤降"的点（非局部极大，y[k]>y[k-1]，y[k+1]>y[k]，
  s_prev=y[k]-y[k-1]>=8，s_at=y[k+1]-y[k] <= ratio*s_prev，相邻输入帧距<=2）→ 在该点发射。

对比配置见 CFGS。评估口径与 r2_eval 一致（公开15 官方 GT，口径 A）。
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


def replay_m12(rows, H, cfg):
    traj = [(r["input_frame_id"], r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    out = []
    last_fid, last_xy = -10_000, None

    def d_trigger(k):
        """减速点：仍在下降段（y[k]>y[k-1]）但下降速度骤降（s_at <= ratio*s_prev）。"""
        if k < 1 or k + 1 >= n:
            return False
        if traj[k][2] <= traj[k - 1][2]:
            return False
        s_prev = traj[k][2] - traj[k - 1][2]
        s_at = traj[k + 1][2] - traj[k][2]
        return (cfg.get("D_smin", 8.0) <= s_prev <= cfg.get("D_smax", 1e9) and s_at > 0
                and s_at <= cfg.get("D_ratio", 0.45) * s_prev
                and traj[k + 1][0] - traj[k][0] <= 2)

    last_d3_fid = -10_000

    def d3_trigger(k):
        """远侧掩蔽触台的"上升速度浅坑"：y 单调下降（图像上升）但速度骤减后骤增。"""
        if k < 2 or k + 1 >= n:
            return False
        if traj[k + 1][0] - traj[k][0] > 2 or traj[k][0] - traj[k - 1][0] > 2 or traj[k - 1][0] - traj[k - 2][0] > 2:
            return False
        if not (traj[k][2] < traj[k - 1][2] and traj[k + 1][2] < traj[k][2]):
            return False
        rho_in = traj[k - 1][2] - traj[k][2]
        rho_prev = traj[k - 2][2] - traj[k - 1][2]
        rho_out = traj[k][2] - traj[k + 1][2]
        r = cfg.get("D3_ratio", 0.45)
        return (rho_prev >= cfg.get("D3_min", 3.0) and rho_out >= cfg.get("D3_min", 3.0)
                and rho_in <= r * rho_prev and rho_in <= r * rho_out)

    def scan(j):
        nonlocal last_fid, last_xy, last_d3_fid
        fid, x, y = traj[j]
        # D3 附加：不改变主扫描 MIN_GAP 状态（与主发射保持间隔，与自身保持间隔）
        if cfg.get("D3") and d3_trigger(j) and fid - last_fid >= MIN_GAP and fid - last_d3_fid >= MIN_GAP:
            passed = True
            if cfg.get("gate"):
                quad = cfg.get("quad")
                passed = False
                if quad is not None and cfg.get("gate_quad", 0.0) > 0:
                    import cv2 as _cv2
                    import numpy as _np
                    dist = _cv2.pointPolygonTest(_np.array(quad, _np.float32).reshape(-1, 1, 2),
                                                 (float(x), float(y)), True)
                    passed = dist >= -cfg["gate_quad"]
                if not passed:
                    u, v = G.project(H, x, y)
                    g = cfg["gate"]
                    passed = (-g <= u <= G.TABLE_L + g and -g <= v <= G.TABLE_W + g)
            if passed:
                out.append({"frame_id": max(0, fid - 1), "x": x, "y": y, "why": "D3"})
                last_d3_fid = fid
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

        # D2 回溯修正：极大点前 ≤D2_back 帧内若存在减速点，则把发射修正到该点
        j_emit = j
        if cfg.get("D2"):
            for k in range(j - 1, max(0, j - 1 - cfg.get("D2_back", 5)), -1):
                if j - k >= cfg.get("D_dmin", 1) and d_trigger(k):
                    j_emit = k
                    break

        efid, ex, ey = traj[j_emit]
        # 跳点守卫（全路径可选）：邻域存在 >120px 相邻位移 → 该极大多半源于假检测点
        if cfg.get("jump_all"):
            for k in range(max(1, j_emit - 2), min(n - 1, j_emit + 2)):
                if math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) > 120.0:
                    return
        if cfg.get("gate"):
            passed = False
            quad = cfg.get("quad")
            if quad is not None and cfg.get("gate_quad", 0.0) > 0:
                import cv2 as _cv2
                import numpy as _np
                dist = _cv2.pointPolygonTest(_np.array(quad, _np.float32).reshape(-1, 1, 2),
                                             (float(ex), float(ey)), True)
                passed = dist >= -cfg["gate_quad"]
            if not passed:
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
            strong = ((sf1 >= 3.0 and sr1 >= 3.0)
                      or (sf2 >= 3.0 and sr2 >= 5.0)
                      or (sr1 >= 8.0 and sf1 >= 0.5))
            if cfg.get("F3_strong2") and not strong:
                strong = (sr1 >= 5.0 and sf1 >= 2.0)
            far_ok = last_xy is not None and math.dist((ex, ey), last_xy) >= cfg.get("F3_far", 100.0)
            jump_ok = True
            for k in range(max(1, j_emit - 2), min(n - 1, j_emit + 2)):
                if math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) > 120.0:
                    jump_ok = False
                    break
            gap_ok = ((j_emit < 2 or traj[j_emit][0] - traj[j_emit - 1][0] <= 2)
                      and (traj[j_emit + 1][0] - traj[j_emit][0] <= 2)
                      and (j_emit + 2 >= n or traj[j_emit + 2][0] - traj[j_emit + 1][0] <= 2))
            if not (strong and far_ok and jump_ok and gap_ok):
                return

        if j_emit != j:
            out.append({"frame_id": max(0, efid - 1), "x": ex, "y": ey, "why": "shift"})
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

    # 固定范围门（后处理，2026-10-17）：事件点离桌面四边形 ≤ max(frac×对角线, min_px)
    # 才保留；需 cfg 的 quad 与 range_frac>0。语义与 event_rules.detect_events 一致。
    rf = cfg.get("range_frac", 0)
    quad = cfg.get("quad")
    if rf and quad is not None:
        import cv2 as _cv2
        import numpy as _np
        qa = _np.array(quad, _np.float32).reshape(-1, 1, 2)
        p4 = _np.array(quad, float)
        diag = max(math.dist(p4[a], p4[b]) for a in range(4) for b in range(a + 1, 4))
        tol = max(rf * diag, cfg.get("range_min_px", 4.0))
        out = [e for e in out
               if _cv2.pointPolygonTest(qa, (float(e["x"]), float(e["y"])), True) >= -tol]
    return out


BASE = {"gate": 900, "F1": True, "F2": True, "F3": True}
CFGS = [
    ("基准 V1+F1F2F3", dict(BASE)),
    ("+D2b(smax50,d2)", {**BASE, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.45,
                          "D_smax": 50.0, "D_dmin": 2}),
    ("+D2c(smax50,d2,r40)", {**BASE, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
                              "D_smax": 50.0, "D_dmin": 2}),
    ("+D2d(smax60,d2)", {**BASE, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.45,
                          "D_smax": 60.0, "D_dmin": 2}),
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
        H = G.build_H(doc["segments"][0])
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"))

    angles = ["0度", "45度", "90度", "全部"]
    results, detail = {}, {}
    for name, cfg in CFGS:
        per_video, per_video_sets = {}, {}
        for vid, (H, rows, gts, _) in videos.items():
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, cfg)]
            m, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in m if gts[j].get("valid", True))
            fp_a = len(fp) + (len(m) - mv)
            fnv = sum(1 for j in fn if gts[j].get("valid", True))
            per_video[vid] = (mv, fp_a, fnv)
            per_video_sets[vid] = {
                "matched": {gts[j]["frame_id"] for i, j in m if gts[j].get("valid", True)},
                "fp": {preds[i]["frame_id"] for i in fp},
                "fn": {gts[j]["frame_id"] for j in fn if gts[j].get("valid", True)},
            }
        results[name] = per_video
        detail[name] = per_video_sets

    print(f"{'配置':28s} | {'0度':>14s} | {'45度':>14s} | {'90度':>14s} | {'全部':>16s}")
    for name, _ in CFGS:
        cells = []
        for a in angles:
            vs = [v for v in videos if a == "全部" or videos[v][3] == a]
            tp = sum(results[name][v][0] for v in vs)
            fp = sum(results[name][v][1] for v in vs)
            fn = sum(results[name][v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
        print(f"{name:28s} | {cells[0]:>14s} | {cells[1]:>14s} | {cells[2]:>14s} | {cells[3]:>16s}")

    # 相对基准的变化明细（标签级）
    base = detail["基准 V1+F1F2F3"]
    for name, _ in CFGS[1:]:
        cur = detail[name]
        new_m, lost_m, new_fp, rem_fp, new_fn, fix_fn = [], [], [], [], [], []
        for vid in videos:
            new_m += [f"{vid} f{f}" for f in sorted(cur[vid]["matched"] - base[vid]["matched"])]
            lost_m += [f"{vid} f{f}" for f in sorted(base[vid]["matched"] - cur[vid]["matched"])]
            new_fp += [f"{vid} f{f}" for f in sorted(cur[vid]["fp"] - base[vid]["fp"])]
            rem_fp += [f"{vid} f{f}" for f in sorted(base[vid]["fp"] - cur[vid]["fp"])]
            new_fn += [f"{vid} f{f}" for f in sorted(cur[vid]["fn"] - base[vid]["fn"])]
            fix_fn += [f"{vid} f{f}" for f in sorted(base[vid]["fn"] - cur[vid]["fn"])]
        print(f"\n[{name}]")
        print(f"  新命中标签 {len(new_m)}: {new_m}")
        print(f"  丢命中标签 {len(lost_m)}: {lost_m}")
        print(f"  新增FP {len(new_fp)}: {new_fp}")
        print(f"  消除FP {len(rem_fp)}: {rem_fp}")
        print(f"  修复FN {len(fix_fn)}: {fix_fn}")
        print(f"  新增FN {len(new_fn)}: {new_fn}")

    (HERE / "m12_stats.json").write_text(
        json.dumps({n: {v: results[n][v] for v in videos} for n, _ in CFGS}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("\n已保存 m12_stats.json")


if __name__ == "__main__":
    main()

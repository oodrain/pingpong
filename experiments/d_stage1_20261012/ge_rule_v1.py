# -*- coding: utf-8 -*-
"""几何约束第一版（V1-geo）对照实验：公开 15 上 V0 vs V1。

V1（已定稿）= V0 + 桌面区域门（唯一改动）：
  发射点经桌面单应投影到桌面坐标系（2740×1525mm），距桌面矩形超过 MARGIN → 拒绝发射。
  MARGIN=900mm 的选型依据见 margin_analysis.py（TP/FP 出桌距离分布 + 留一交叉验证）。

本轮尝试并否决的两个变体（v1b 模式代码保留备查）：
  - 幅度放宽（1.0/0.5）：捕捉检测噪声并级联挤出真发射，F1 反而大降；
  - 智能去重（>500mm 放行相邻发射）：放行的大多是噪声重复。
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path("E:/pingpong-data")
ANN = BASE / "table_annotator/annotations"
RAW = BASE / "C_public15_20261009_v1/raw"
GT_P = BASE / "public_data_20261010/gt_public.jsonl"
MAN = BASE / "public_data_20261010/manifest.json"
OUT = Path("E:/pingpong-data/geometry_v1_20261012")
OUT.mkdir(exist_ok=True)

TABLE_L, TABLE_W = 2740.0, 1525.0
MIN_GAP, CONF_TH = 10, 0.75
MARGIN = 900.0      # 区域门边距 mm（依据：TP 中位出桌 0mm / FP 中位 620mm；900 为 F1 拐点，LOOCV 14/15 选中）
DUP_MM = 500.0      # 智能去重：桌面距离阈值


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_H(seg):
    c = np.array(seg["corners"], np.float32)
    net = seg.get("net")
    dst_a = np.array([[0, 0], [TABLE_L, 0], [TABLE_L, TABLE_W], [0, TABLE_W]], np.float32)
    dst_b = np.array([[0, 0], [TABLE_W, 0], [TABLE_W, TABLE_L], [0, TABLE_L]], np.float32)
    best = None
    for dst in (dst_a, dst_b):
        H = cv2.getPerspectiveTransform(c, dst)
        score = 0.0
        if net and len(net) == 2:
            pts = cv2.perspectiveTransform(np.array(net, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            score = float(abs(pts[0][0] - pts[1][0]))  # 网段映射后 x 跨度；方向选小的
        if best is None or score < best[0]:
            best = (score, H)
    return best[1]


def project(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
    return float(p[0]), float(p[1])


def in_region(u, v, margin):
    return -margin <= u <= TABLE_L + margin and -margin <= v <= TABLE_W + margin


def replay(rows, H, mode):
    traj = [(r["input_frame_id"], r["x"], r["y"], r["confidence"]) for r in rows
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]
    n = len(traj)
    out = []
    last_fid = -10_000
    last_mm = None

    def emit(need_future):
        nonlocal last_fid, last_mm
        for j in range(1, n - need_future):
            fid, x, y, conf = traj[j]
            prev_y, next_y = traj[j - 1][2], traj[j + 1][2]
            if not (y > prev_y and y >= next_y):
                continue
            fall, rise = y - prev_y, y - next_y
            if fall < 2.0 or rise < 1.0:
                continue
            if mode == "v0":
                if fid - last_fid < MIN_GAP:
                    continue
            else:
                u, v = project(H, x, y)
                if not in_region(u, v, MARGIN):
                    continue
                if fid - last_fid < MIN_GAP:
                    if mode == "v1a" or last_mm is None or math.dist((u, v), last_mm) <= DUP_MM:
                        continue
                last_mm = (u, v)
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y})
            last_fid = fid

    # 两阶段扫描等价于官方逐帧回放：条件只看局部点，首次可评估时即评估，顺序与逐帧一致
    emit(2)
    emit(1)
    return out


def match(preds, gts):
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p["video_id"] != g["video_id"]:
                continue
            df = abs(p["frame_id"] - g["frame_id"])
            if df > 1:
                continue
            d = math.dist((p["x"], p["y"]), (g["x"], g["y"]))
            if d <= 50:
                pairs.append((d, df, i, j))
    pairs.sort()
    up, ug = set(), set()
    m = []
    for d, df, i, j in pairs:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); m.append((i, j))
    fp = [i for i in range(len(preds)) if i not in up]
    fn = [j for j in range(len(gts)) if j not in ug]
    return m, fp, fn


def main():
    man = {v["video_id"]: v for v in json.loads(MAN.read_text(encoding="utf-8"))["videos"]}
    gt = load(GT_P)
    gt_by_v = {}
    for g in gt:
        gt_by_v.setdefault(g["video_id"], []).append(g)

    stats = {}
    for vid, meta in man.items():
        doc = json.loads((ANN / f"{vid}.json").read_text(encoding="utf-8"))
        seg = doc["segments"][0]
        H = build_H(seg)
        rows = load(RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        gts = gt_by_v.get(vid, [])
        res = {}
        for mode in ("v0", "v1a", "v1b"):
            preds = [{"video_id": vid, **e} for e in replay(rows, H, mode)]
            m, fp, fn = match(preds, gts)
            fn_valid = sum(1 for j in fn if gts[j].get("valid", True))
            m_valid = sum(1 for i, j in m if gts[j].get("valid", True))
            fp_a = len(fp) + (len(m) - m_valid)  # 口径A：命中干扰点算 FP
            res[mode] = {"preds": len(preds), "tp": m_valid, "fpA": fp_a, "fnA": fn_valid, "fpRaw": len(fp), "fnRaw": len(fn)}
            # 诊断：V0 中匹配到 GT 的发射，其投影出桌距离
            if mode == "v0":
                outd = []
                for i, j in m:
                    if not gts[j].get("valid", True):
                        continue
                    u, v = project(H, preds[i]["x"], preds[i]["y"])
                    od = max(u - TABLE_L, -u, v - TABLE_W, -v, 0.0)
                    outd.append(od)
                if outd:
                    big = sum(1 for d in outd if d > MARGIN)
                    res["v0"]["tp_out_beyond_margin"] = big
        stats[vid] = {"angle": doc.get("angle"), **{f"{k}_{kk}": vv for k, v in res.items() for kk, vv in v.items()}}
        r0, ra, rb = res["v0"], res["v1a"], res["v1b"]
        print(f"{vid} {doc.get('angle')}: V0 TP{r0['tp']} FP{r0['fpA']} FN{r0['fnA']}"
              f" | V1a TP{ra['tp']} FP{ra['fpA']} FN{ra['fnA']}"
              f" | V1b TP{rb['tp']} FP{rb['fpA']} FN{rb['fnA']}"
              f" | V0中TP投影出桌>{MARGIN:.0f}mm: {r0.get('tp_out_beyond_margin', 0)}")

    # 汇总（口径A，按视角）
    print("\n===== 汇总（口径A：命中干扰点计 FP；FN 只计有效点）=====")
    print(f"{'视角':6s} | {'V0: TP/FP/FN/F1':>22s} | {'V1a(仅区域门): TP/FP/FN/F1':>26s} | {'V1b(+智能去重): TP/FP/FN/F1':>28s}")
    angles = ["0度", "45度", "90度", "全部"]
    agg = {}
    for a in angles:
        vs = [v for v, s in stats.items() if a == "全部" or s["angle"] == a]
        row = {}
        for mode in ("v0", "v1a", "v1b"):
            tp = sum(stats[v][f"{mode}_tp"] for v in vs)
            fp = sum(stats[v][f"{mode}_fpA"] for v in vs)
            fn = sum(stats[v][f"{mode}_fnA"] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            row[mode] = (tp, fp, fn, f1)
        agg[a] = row
        cells = []
        for mode in ("v0", "v1a", "v1b"):
            t = row[mode]
            cells.append(f"{t[0]:>3d}/{t[1]:>3d}/{t[2]:>3d}/{t[3]:.3f}")
        print(f"{a:6s} | {cells[0]:>22s} | {cells[1]:>26s} | {cells[2]:>28s}")
    (OUT / "ge_v1_stats.json").write_text(json.dumps({"per_video": stats, "agg": agg}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n已保存:", OUT / "ge_v1_stats.json")


if __name__ == "__main__":
    main()

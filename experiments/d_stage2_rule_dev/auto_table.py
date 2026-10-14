# -*- coding: utf-8 -*-
"""自动桌面四角估计 v1（提交侧几何门前置 / 阶段三方案原型）。

思路：逐帧三通道（白线/蓝面/暗蓝，复用标注工具候选实现）产生候选四边形，
经三重先验（中央位置 / 球点覆盖率 / 面积合理性）筛选后，按时序融合（固定机位
多帧中值+离群剔除）得到稳定四角。

先验依据：
- 中央：转播固定机位下球桌必在画面中部（排除场馆墙面/挡板误锁定）；
- 球点覆盖率：球大部分时间在桌面一带飞行（真桌候选覆盖高、横幅/挡板覆盖低）；
- 面积：桌面图像占比的合理范围（排除巨型合并区域与细条横幅）。
角点精度要求由噪声实验界定：≤20px 对门后 F1 基本无损；覆盖率失败视频自动
回退"纯规则"（无门），系统仍优于无几何基线。

用法：
    python auto_table.py <video.mp4> [n_frames] [raw.jsonl]   → 打印角点 JSON
模块函数：
    estimate_table(video_path, n_frames=24, tol=60.0, ball_pts=None)
    返回 {"corners": [[x,y]x4], "frames_ok", "n_frames", "spread_px"} 或 None
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

MIN_AREA_RATIO, MAX_AREA_RATIO = 0.012, 0.28
FRAC_T1, FRAC_T2 = 0.50, 0.40   # 两级球点覆盖率门槛
BALL_TOL_FRAC = 0.03            # 球点判定的膨胀容差（画面宽占比）


def _quads_from_mask(mask, W, H, min_area_ratio=0.004, topk=3):
    """掩码中最大的 topk 个四合分量 [(score, quad_ds), ...]，降序。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        ar = area / (W * H)
        if ar < min_area_ratio or ar > 0.45:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y <= 3:
            continue
        if x <= 5 and x + bw >= W - 5:
            continue
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        if ha <= 0:
            continue
        fill = area / ha
        score = 0.65 * fill + 0.35 * min(ar / 0.06, 1.0)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.05, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(-1, 2).astype(np.float32)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        scored.append((score, quad))
    scored.sort(key=lambda t: -t[0])
    return scored[:topk]


def _canonical_order(quad: np.ndarray) -> np.ndarray:
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    order = np.argsort(ang)
    quad = quad[order]
    k = int(np.argmin(quad[:, 0] + quad[:, 1]))
    return np.roll(quad, -k, axis=0)


def _frame_candidates(frame, ball_pts=None):
    """单帧选角：三通道 top3 候选 → 中心/面积/球点覆盖率先验 → 首个合规者。

    返回 ((quad 原图坐标, score, name)) 或 None。"""
    h, w = frame.shape[:2]
    scale = 960.0 / w
    img = cv2.resize(frame, (960, max(1, int(h * scale))))
    W, H = img.shape[1], img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hh, S, V = cv2.split(hsv)
    cands = []

    wm = cv2.inRange(hsv, (0, 0, 145), (179, 75, 255))
    lines = cv2.HoughLinesP(wm, 1, np.pi / 180, 55, minLineLength=110, maxLineGap=20)
    if lines is not None:
        segs = np.asarray(lines).reshape(-1, 4).tolist()
        keep = [s for s in segs if ((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2) ** 0.5 >= 110]
        if keep:
            canvas = np.zeros_like(wm)
            for x1, y1, x2, y2 in keep:
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, 13)
            canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            for sc, quad in _quads_from_mask(canvas, W, H, 0.008, topk=3):
                cands.append((quad, float(sc) - 0.10, "line"))

    sat = (S > 60) & (V > 40)
    if sat.sum() >= 500:
        hist = np.bincount(Hh[sat], minlength=180).astype(float)
        zone = hist.copy()
        zone[:20] = 0
        zone[160:] = 0
        h0 = int(zone.argmax())
        m = cv2.inRange(hsv, np.array([max(0, h0 - 15), 60, 40]), np.array([min(179, h0 + 15), 255, 255]))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        for sc, quad in _quads_from_mask(m, W, H, 0.004, topk=3):
            cands.append((quad, float(sc), f"blob{h0}"))

    dark = (S >= 30) & (V >= 28) & (V <= 170)
    if dark.sum() >= 500:
        hist = np.bincount(Hh[dark], minlength=180).astype(float)
        zone = hist.copy()
        zone[:90] = 0
        zone[145:] = 0
        if zone.sum() > 0:
            h0 = int(zone.argmax())
            m = cv2.inRange(hsv, np.array([max(0, h0 - 15), 30, 25]), np.array([min(179, h0 + 15), 255, 175]))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            for sc, quad in _quads_from_mask(m, W, H, 0.004, topk=3):
                cands.append((quad, float(sc), f"dark{h0}"))

    if not cands:
        return None
    frame_area = float(w * h)
    pts_ds = None
    if ball_pts:
        pts_ds = [np.array([x * scale, y * scale], np.float32) for x, y in ball_pts]
    tol_ds = BALL_TOL_FRAC * 960.0

    def centered(q):
        c = q.mean(axis=0)
        return (0.10 * W <= c[0] <= 0.90 * W) and (0.08 * H <= c[1] <= 0.92 * H)

    def ok_frame(q_ds):
        qo = q_ds / scale
        ar = float(cv2.contourArea(qo.astype(np.float32))) / frame_area
        return MIN_AREA_RATIO <= ar <= MAX_AREA_RATIO

    def ball_frac(q_ds):
        if pts_ds is None:
            return 1.0
        qc = np.asarray(q_ds, np.float32).reshape(-1, 1, 2)
        inside = sum(1 for p in pts_ds
                     if cv2.pointPolygonTest(qc, (float(p[0]), float(p[1])), True) >= -tol_ds)
        return inside / max(1, len(pts_ds))

    cands.sort(key=lambda t: -t[1])
    # 一级：首个（中心+面积+覆盖率≥0.5）候选；二级回退：覆盖率≥0.4
    for frac_th in (FRAC_T1, FRAC_T2):
        for q_ds, sc, name in cands:
            if centered(q_ds) and ok_frame(q_ds) and ball_frac(q_ds) >= frac_th:
                return _canonical_order(q_ds) / scale, sc, name
    return None


def _fuse(quads, tol=60.0):
    """时序融合：以先到的候选为种子找最大一致簇（容差 tol），取簇内中值。"""
    if not quads:
        return None
    arrs = [q for q, _ in quads]
    best = None
    for seed in arrs[:6]:
        kept = [q for q in arrs if np.max(np.linalg.norm(q - seed, axis=1)) <= tol]
        if best is None or len(kept) > best[0]:
            best = (len(kept), kept)
    n_kept, kept = best
    med = np.median(np.stack(kept), axis=0)
    spread = float(np.mean(np.linalg.norm(np.stack(kept) - med, axis=2)))
    return med.astype(np.float32), n_kept, spread


def estimate_table(video_path, n_frames=24, tol=60.0, ball_pts=None):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    quads = []
    names = []
    fw = fh = None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        if fw is None:
            fh, fw = fr.shape[:2]
        r = _frame_candidates(fr, ball_pts=ball_pts)
        if r is not None:
            quads.append((r[0], r[2]))
            names.append(r[2])
    cap.release()
    fused = _fuse(quads, tol=tol)
    if fused is None:
        return None
    quad, kept, spread = fused
    # 终极校验：融合结果本身必须通过球点覆盖率（否则判失败 → 上层回退纯规则，
    # 防止错误角点把整条视频的门杀光，v000047 类灾难案例的防护）
    if ball_pts and fw:
        qc = np.asarray(quad, np.float32).reshape(-1, 1, 2)
        tol_px = BALL_TOL_FRAC * fw
        inside = sum(1 for (x, y) in ball_pts
                     if cv2.pointPolygonTest(qc, (float(x), float(y)), True) >= -tol_px)
        if inside / max(1, len(ball_pts)) < 0.45:
            return None
    return {
        "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
        "frames_ok": kept,
        "n_frames": len(quads),
        "spread_px": round(spread, 2),
        "channels": sorted(set(names)),
    }


if __name__ == "__main__":
    path = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    ball = None
    if len(sys.argv) > 3:
        rows = [json.loads(l) for l in open(sys.argv[3], encoding="utf-8").splitlines() if l.strip()]
        ball = [(r["x"], r["y"]) for r in rows
                if r.get("detected") and r.get("x") is not None and (r.get("confidence") or 0) >= 0.75]
    res = estimate_table(path, nf, ball_pts=ball)
    print(json.dumps(res, ensure_ascii=False))

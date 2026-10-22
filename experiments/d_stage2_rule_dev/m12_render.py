# -*- coding: utf-8 -*-
"""M1.2 关键案例单帧放大渲染（3.2x）：帧内画桌面四边形 + 当前帧检测点 + 指定标记点。"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G = mod("gv1", "E:/pingpong-data/geometry_v1_20261012/ge_rule_v1.py")
VID_DIR = Path("E:/pingpong-data/error_analysis_v1_20261011/vids_ascii")
OUT = HERE / "m12_zoom"
OUT.mkdir(exist_ok=True)

# (输出名, 视频, 真实帧, 中心, 窗口, 缩放, [(x,y,color)])
CASES = []
for f in (74, 76, 78):
    CASES.append((f"z1_f{f}", "v000001", f, (975, 465), (320, 240), 3.2,
                  [(958, 460, (0, 255, 255))]))
for f in (44, 45, 46):
    CASES.append((f"z1_f{f}", "v000001", f, (1000, 360), (320, 240), 3.2,
                  [(1006, 355, (0, 255, 255))]))
for f in (57, 58, 60):
    CASES.append((f"z1_f{f}", "v000001", f, (1080, 480), (340, 250), 3.2,
                  [(1082, 497, (0, 255, 255))]))
for f in (20, 21, 24, 25):
    CASES.append((f"z10_f{f}", "v000010", f, (1100, 450), (340, 250), 3.2,
                  [(1066, 444, (0, 255, 255)), (1142, 454, (255, 0, 255))]))

for name, vid, f, center, (w, h), sc, marks in CASES:
    doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
    corners = doc["segments"][0]["corners"]
    net = doc["segments"][0].get("net")
    rows = G.load(G.RAW / f"{vid}.jsonl")
    by_in = {r["input_frame_id"]: r for r in rows}
    cap = cv2.VideoCapture(str(VID_DIR / f"{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        continue
    Hh, Ww = fr.shape[:2]
    x0 = int(min(max(center[0] - w / 2, 0), Ww - w))
    y0 = int(min(max(center[1] - h / 2, 0), Hh - h))
    crop = fr[y0:y0 + h, x0:x0 + w].copy()
    tile = cv2.resize(crop, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)

    def T(p):
        return tuple(np.round((np.array(p) - (x0, y0)) * sc).astype(int))

    cv2.polylines(tile, [np.array([T(p) for p in corners], np.int32)], True, (0, 200, 0), 2)
    if net and len(net) == 2:
        cv2.line(tile, T(net[0]), T(net[1]), (255, 200, 0), 2)
    rec = by_in.get(f + 1)
    if rec and rec.get("detected") and rec.get("confidence") is not None:
        tx, ty = T((rec["x"], rec["y"]))
        cv2.drawMarker(tile, (tx, ty), (255, 255, 0), cv2.MARKER_CROSS, 32, 3)
        cv2.putText(tile, f"det({rec['x']:.0f},{rec['y']:.0f}) conf={rec['confidence']:.2f}",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    else:
        cv2.putText(tile, f"raw @input{f+1}: 无检测/低置信", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for (mx, my, color) in marks:
        tx, ty = T((mx, my))
        cv2.circle(tile, (tx, ty), int(12 * sc // 3), color, 3)
        cv2.circle(tile, (tx, ty), 3, color, -1)
    cv2.putText(tile, f"{vid} f{f}", (10, tile.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f"{name}.png"), tile)
    print(f"写出 {name}.png")

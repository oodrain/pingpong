# -*- coding: utf-8 -*-
"""M3.1 训练包桌框抽查：覆盖率 / 稳定性 / 与真实桌面的贴合度（渲染目检）。

训练包的每个对象 JSON 在每帧含 "table" bbox（轴对齐矩形）。本脚本：
1) 抽取 20 条视频（三视角覆盖、跨成员目录），统计桌框覆盖帧数与 bbox 中心/尺寸漂移；
2) 对 4 条样本渲染首/中/末帧叠加图（bbox + 全幅缩小），供判断"粗筛"可用性。
输出：控制台汇总 + m31_overlays/*.png
"""
import importlib.util
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ADJ = Path("E:/annotation-data/adjudication")
V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")
OUT = HERE / "m31_overlays"
OUT.mkdir(exist_ok=True)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    import csv
    ang = {}
    for r in csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines()):
        ang[r["video_id"]] = int(r["angle_degrees"])

    # 收集全部 json 索引
    index = {}
    for member in "ABCD":
        for p in (ADJ / member / "json").rglob("*.json"):
            vid = f"{p.parent.name}_{p.stem}"
            index[vid] = p

    # 抽样：每个视角多少条（0:8, 45:6, 90:6），组内随机
    random.seed(42)
    pick = []
    for a, n in ((0, 8), (45, 6), (90, 6)):
        cands = sorted(v for v in index if ang.get(v) == a)
        pick += random.sample(cands, min(n, len(cands)))
    print(f"抽样 {len(pick)} 条：", pick)

    print(f"\n{'视频':8s} {'视角':>3s} | {'总帧':>5s} {'有桌框':>5s} | 中心漂移(px) 尺寸漂移(px) | 首帧bbox")
    render_set = set(random.sample(pick, 4))
    rows = []
    for vid in pick:
        d = load(index[vid])
        frames = d.get("frames", [])
        boxes = []
        for fr in frames:
            for an in fr.get("annotations", []):
                if an.get("label") == "table" and an.get("bbox_xyxy"):
                    boxes.append(an["bbox_xyxy"][0])
                    break
        if not boxes:
            print(f"{vid:8s} {ang.get(vid,0):>3d} | {len(frames):>5d} {'0':>5s} | 无桌框!")
            continue
        cx = [ (b[0]+b[2])/2 for b in boxes]; cy = [ (b[1]+b[3])/2 for b in boxes]
        w = [ b[2]-b[0] for b in boxes]; h = [ b[3]-b[1] for b in boxes]
        dcx = max(cx)-min(cx); dcy = max(cy)-min(cy); dw = max(w)-min(w); dh = max(h)-min(h)
        print(f"{vid:8s} {ang.get(vid,0):>3d} | {len(frames):>5d} {len(boxes):>5d} | "
              f"({dcx:5.1f},{dcy:5.1f}) ({dw:5.1f},{dh:5.1f}) | {[round(v,1) for v in boxes[0]]}")
        rows.append((vid, ang.get(vid,0), len(frames), len(boxes), dcx, dcy, dw, dh))

        if vid in render_set:
            # 渲染首/中/末帧
            vpath = None
            for member in "ABCD":
                for p in (ADJ / member / "videos").rglob(f"{index[vid].stem}.mp4"):
                    if p.parent.name == index[vid].parent.name:
                        vpath = p; break
                if vpath: break
            if not vpath:
                continue
            cap = cv2.VideoCapture(str(vpath))
            nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tiles = []
            for fi in (0, nf // 2, nf - 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    continue
                tile = cv2.resize(fr, (960, 540))
                sc = 960 / fr.shape[1]
                # 找该帧的桌框（按帧序近似取 boxes[min(len-1, fi)]）
                b = boxes[min(len(boxes)-1, fi)]
                cv2.rectangle(tile, (int(b[0]*sc), int(b[1]*sc)), (int(b[2]*sc), int(b[3]*sc)),
                              (0, 255, 0), 3)
                cv2.putText(tile, f"{vid} f{fi}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                tiles.append(tile)
            cap.release()
            if tiles:
                canvas = np.hstack([cv2.resize(t, (640, 360)) for t in tiles])
                cv2.imwrite(str(OUT / f"{vid}.png"), canvas)
                print(f"  渲染 {vid}.png")

    if rows:
        dcxs = [r[4] for r in rows]; dcys = [r[5] for r in rows]
        print(f"\n汇总：{len(rows)} 条有桌框；中心漂移 中位 ({np.median(dcxs):.1f},{np.median(dcys):.1f})px，"
              f"最大 ({max(dcxs):.1f},{max(dcys):.1f})px")
        n_ov = sum(1 for r in rows if r[3] < r[2])
        print(f"覆盖不完整的视频数: {n_ov}/{len(rows)}")


if __name__ == "__main__":
    main()

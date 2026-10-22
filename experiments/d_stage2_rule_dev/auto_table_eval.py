# -*- coding: utf-8 -*-
"""自动桌面验收（公开15）：
A) 自动角点 vs 人工 GT 角点的逐角误差分布；
B) 用自动角点跑完整候选栈（带几何门）→ F1，对比 GT 角点（0.744）与无几何（0.697）。
输出：auto_table_public15.json（角点 + 指标）+ 控制台报告。
"""
import importlib.util
import json
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("auto_table", HERE / "auto_table.py")
AT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AT)

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

VID = Path("E:/pingpong-data/error_analysis_v1_20261011/vids_ascii")
FULL = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
        "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2, "gate_quad": 4.0,
        "D3": True, "D3_ratio": 0.45}
NOG = {k: v for k, v in FULL.items() if k not in ("gate", "gate_quad")}


def main():
    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    gts_by_v = defaultdict(list)
    for g in gt:
        gts_by_v[g["video_id"]].append(g)
    report = {}
    tot_gt, tot_auto, tot_sys = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    errs = []
    print(f"{'视频':8s} {'角点误差(均值/最大)px':>22s} | {'帧ok/总':>8s} {'spread':>7s} | {'F1(GT角)':>8s} {'F1(自动角)':>9s}")
    for vid in man:
        doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
        seg = doc["segments"][0]
        corners_gt = np.array(seg["corners"], float)
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        gts = gts_by_v.get(vid, [])

        ball_pts = [(r["x"], r["y"]) for r in rows
                    if r.get("detected") and r.get("x") is not None
                    and (r.get("confidence") or 0) >= 0.75]
        res = AT.estimate_table(VID / f"{vid}.mp4", ball_pts=ball_pts)
        if res is None:
            preds_nog = [{"video_id": vid, **e} for e in replay_m12(rows, None, NOG)]
            mm_, fp, fn = G.match(preds_nog, gts)
            for t3 in ((sum(1 for i, j in mm_ if gts[j].get("valid", True)),
                        len(fp), sum(1 for j in fn if gts[j].get("valid", True))),):
                pass
            mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
            t_nog = (mv, len(fp) + (len(mm_) - mv), sum(1 for j in fn if gts[j].get("valid", True)))
            for i3 in range(3):
                tot_sys[i3] += t_nog[i3]
            f_nog = 2 * t_nog[0] / (2 * t_nog[0] + t_nog[1] + t_nog[2]) if t_nog[0] else 0.0
            print(f"{vid:8s} 自动检测失败 → 回退纯规则 (F1 {f_nog:.3f})")
            continue
        corners_auto = np.array(res["corners"], float)
        d = np.linalg.norm(corners_auto - corners_gt, axis=1)
        errs.append((vid, d.mean(), d.max()))
        report[vid] = {"corners_auto": res["corners"], "err_mean": round(float(d.mean()), 1),
                       "err_max": round(float(d.max()), 1), "spread": res["spread_px"],
                       "frames_ok": res["frames_ok"]}

        def f1_with(corners):
            H = G.build_H({**seg, "corners": corners.tolist()})
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=corners.tolist()))]
            mm_, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
            t = (mv, len(fp) + (len(mm_) - mv), sum(1 for j in fn if gts[j].get("valid", True)))
            return 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t[0] else 0.0, t

        f_gt, t_gt = f1_with(corners_gt)
        f_auto, t_auto = f1_with(corners_auto)
        # 系统口径：自动角点成功→带门；失败→回退纯规则（无几何）
        cfg_nog = {k: v for k, v in FULL.items() if k not in ("gate", "gate_quad")}
        preds_nog = [{"video_id": vid, **e} for e in replay_m12(rows, None, cfg_nog)]
        mm_, fp, fn = G.match(preds_nog, gts)
        mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
        t_nog = (mv, len(fp) + (len(mm_) - mv), sum(1 for j in fn if gts[j].get("valid", True)))
        for i in range(3):
            tot_gt[i] += t_gt[i]
            tot_auto[i] += t_auto[i]
            tot_sys[i] += t_auto[i] if res is not None else t_nog[i]
        print(f"{vid:8s} {d.mean():>10.1f} / {d.max():<8.1f} | {res['frames_ok']:>4d}/{res['n_frames']:<3d} "
              f"{res['spread_px']:>6.1f} | {f_gt:>8.3f} {f_auto:>9.3f}")

    F = lambda t: 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t[0] else 0.0
    print(f"\n汇总：GT 角点 {tot_gt} = {F(tot_gt):.3f} | 自动角点(成功视频) {tot_auto} = {F(tot_auto):.3f}")
    print(f"系统口径（自动角点+失败回退纯规则）: {tot_sys} = {F(tot_sys):.3f}（对照：全 GT 角点 0.744 / 纯规则 0.697）")
    em = np.array([e[1] for e in errs]); ex = np.array([e[2] for e in errs])
    print(f"角点误差：均值 中位 {np.median(em):.1f}px 最大 {em.max():.1f}px；单角最大 {ex.max():.1f}px")
    worst = sorted(errs, key=lambda t: -t[1])[:3]
    print("误差最大三例:", [(w[0], round(w[1], 1), round(w[2], 1)) for w in worst])
    (HERE / "auto_table_public15.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("已保存 auto_table_public15.json")


if __name__ == "__main__":
    main()

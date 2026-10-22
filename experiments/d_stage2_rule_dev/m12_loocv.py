# -*- coding: utf-8 -*-
"""M1.2 留一验证 + 154 尺度检验（D2c：急减速回溯修正）。

A) 公开15 留一：每次留 1 条视频，在其余 14 条上比较 {基准, +D2c} 的 F1，选优后在留出条上评估。
B) 154 尺度：V0+F1F2F3 vs +D2c（均无几何门），候选口径 TP/FP/FN 与逐视频极值检查。
"""
import importlib.util
import json
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 从 m12_eval.py 中取出 replay_m12（不执行其 main）
src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

BASE_CFG = {"gate": 900, "F1": True, "F2": True, "F3": True}
D2C = {**BASE_CFG, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
       "D_smax": 50.0, "D_dmin": 2}
BASE_NG = {"F1": True, "F2": True, "F3": True}
D2C_NG = {**BASE_NG, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
          "D_smax": 50.0, "D_dmin": 2}


def f1_of(per_video, vids):
    tp = sum(per_video[v][0] for v in vids)
    fp = sum(per_video[v][1] for v in vids)
    fn = sum(per_video[v][2] for v in vids)
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), tp, fp, fn


def eval_videos(videos, cfg):
    out = {}
    for vid, (H, rows, gts) in videos.items():
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, cfg)]
        mm, fp, fn = G.match(preds, gts)
        mv = sum(1 for i, j in mm if gts[j].get("valid", True))
        out[vid] = (mv, len(fp) + (len(mm) - mv),
                    sum(1 for j in fn if gts[j].get("valid", True)))
    return out


def main():
    # ---------- A) 公开15 留一 ----------
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
        videos[vid] = (H, rows, gts_by_v.get(vid, []))

    per_base = eval_videos(videos, BASE_CFG)
    per_d2c = eval_videos(videos, D2C)
    print("===== A) 公开15 留一验证 =====")
    print("每条视频 F1（基准 → +D2c）:")
    gains = []
    chose_b = 0
    for vid in man:
        f1b, *_ = f1_of(per_base, [vid])
        f1d, *_ = f1_of(per_d2c, [vid])
        gains.append(f1d - f1b)
        vids_wo = [v for v in man if v != vid]
        f1r_b, *_ = f1_of(per_base, vids_wo)
        f1r_d, *_ = f1_of(per_d2c, vids_wo)
        pick = "D2c" if f1r_d >= f1r_b else "基准"
        if pick == "D2c":
            chose_b += 1
        held = f1d if pick == "D2c" else f1b
        base_held = f1b
        print(f"  {vid}: {f1b:.3f} → {f1d:.3f} | 其余14条选: {pick} | 留出增益 {held - base_held:+.3f}")
    print(f"14/15 条选择 +D2c: {chose_b}/15；留出平均增益 {sum(gains)/len(gains):+.4f}"
          f"（全量增益 {f1_of(per_d2c, man)[0] - f1_of(per_base, man)[0]:+.4f} 为上限口径）")

    # ---------- B) 154 尺度 ----------
    print("\n===== B) 154 尺度（候选口径，无几何门） =====")
    RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")
    CAND = Path("E:/pingpong/experiments/manual_label_audit/adjudicated_label_candidates_v1/candidate_events.jsonl")
    cand = G.load(CAND)
    cand_by_v = defaultdict(list)
    for c in cand:
        cand_by_v[c["video_id"]].append(c)
    res = {}
    for name, cfg in (("基准 F1F2F3", BASE_NG), ("+D2c", D2C_NG)):
        per_video = {}
        for vid in sorted(cand_by_v):
            rows = G.load(RAW154 / f"{vid}.jsonl")
            rows.sort(key=lambda r: r["input_frame_id"])
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H=None, cfg=cfg)]
            mm, fp, fn = G.match(preds, cand_by_v[vid])
            per_video[vid] = (len(mm), len(fp), len(fn))
        res[name] = per_video
        tp = sum(v[0] for v in per_video.values())
        fp = sum(v[1] for v in per_video.values())
        fn = sum(v[2] for v in per_video.values())
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"{name}: TP={tp} FP={fp} FN={fn} | P={tp/(tp+fp):.3f} R={tp/(tp+fn):.3f} F1={f1:.3f}")
    deltas = {v: (res["+D2c"][v][0] - res["基准 F1F2F3"][v][0],
                  res["+D2c"][v][1] - res["基准 F1F2F3"][v][1],
                  res["+D2c"][v][2] - res["基准 F1F2F3"][v][2]) for v in res["+D2c"]}
    worst = sorted(deltas.items(), key=lambda kv: kv[1][0] - kv[1][1] - kv[1][2])[:6]
    print("变化最大（劣）的视频 (ΔTP, ΔFP, ΔFN):", worst)
    n_tp = sum(1 for d in deltas.values() if d[0] > 0)
    n_fp = sum(1 for d in deltas.values() if d[1] > 0)
    print(f"TP 增加的视频: {n_tp}/154；FP 增加的视频: {n_fp}/154")
    print("合计 Δ(TP,FP,FN):", tuple(sum(d[i] for d in deltas.values()) for i in range(3)))

    (HERE / "m12_loocv_stats.json").write_text(json.dumps(
        {"per_base": per_base, "per_d2c": per_d2c}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

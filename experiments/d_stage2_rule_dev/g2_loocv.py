# -*- coding: utf-8 -*-
"""几何门双判留一验证：{F1F2F3+D2c} vs {F1F2F3+D2c+quad4} 逐折选择与留出增益。"""
import importlib.util
import json
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

D2C = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
       "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2}
Q4 = {**D2C, "gate_quad": 4.0}


def f1_of(per, vids):
    tp = sum(per[v][0] for v in vids)
    fp = sum(per[v][1] for v in vids)
    fn = sum(per[v][2] for v in vids)
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def eval_videos(videos, cfg):
    out = {}
    for vid, (H, rows, gts, quad) in videos.items():
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(cfg, quad=quad))]
        mm_, fp, fn = G.match(preds, gts)
        mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
        out[vid] = (mv, len(fp) + (len(mm_) - mv),
                    sum(1 for j in fn if gts[j].get("valid", True)))
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
        seg = doc["segments"][0]
        H = G.build_H(seg)
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        videos[vid] = (H, rows, gts_by_v.get(vid, []), seg["corners"])

    per_d = eval_videos(videos, D2C)
    per_q = eval_videos(videos, Q4)
    print("每条视频 F1（+D2c → +D2c+quad4）:")
    gains, chose = [], 0
    for vid in man:
        f1_d = f1_of(per_d, [vid])
        f1_q = f1_of(per_q, [vid])
        gains.append(f1_q - f1_d)
        wo = [v for v in man if v != vid]
        pick = "quad" if f1_of(per_q, wo) >= f1_of(per_d, wo) else "D2c"
        chose += pick == "quad"
        print(f"  {vid}: {f1_d:.3f} → {f1_q:.3f} | 其余14条选: {pick} | 留出增益 {f1_q - f1_d:+.3f}")
    print(f"\n15/15 折选择 +quad4: {chose}/15；留出平均增益 {sum(gains)/len(gains):+.4f}"
          f"（全量 {f1_of(per_q, man) - f1_of(per_d, man):+.4f} 为上限口径）")


if __name__ == "__main__":
    main()

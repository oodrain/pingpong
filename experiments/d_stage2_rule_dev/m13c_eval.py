# -*- coding: utf-8 -*-
"""D3（远侧掩蔽触台·上升速度浅坑）消融：公开15 候选栈上叠加 + 154/v2 分视角增量。

公开 15：候选栈 = V1+F1F2F3+D2c+quad4（当前最优），在其上叠 D3 的 ratio 变体。
154：F1F2F3 与 F1F2F3+D2c 为基，叠 D3；分视角 + 恢复/新增明细。
"""
import importlib.util
import json
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")
RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

STACK = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
         "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2, "gate_quad": 4.0}
NG = {"F1": True, "F2": True, "F3": True}
NG_D2C = {**NG, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
          "D_smax": 50.0, "D_dmin": 2}

CFGS15 = [
    ("候选栈(基准)", dict(STACK)),
    ("+D3(0.45)", {**STACK, "D3": True, "D3_ratio": 0.45}),
    ("+D3(0.35)", {**STACK, "D3": True, "D3_ratio": 0.35}),
    ("+D3(0.55)", {**STACK, "D3": True, "D3_ratio": 0.55}),
]
CFGS154 = [
    ("F1F2F3", dict(NG), dict(NG)),
    ("F1F2F3+D3(0.45)", {**NG, "D3": True, "D3_ratio": 0.45}, dict(NG)),
    ("+D2c", dict(NG_D2C), dict(NG)),
    ("+D2c+D3(0.45)", {**NG_D2C, "D3": True, "D3_ratio": 0.45}, dict(NG_D2C)),
]


def eval15():
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
        videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"), seg["corners"])

    angles = ["0度", "45度", "90度", "全部"]
    results, detail = {}, {}
    for name, cfg in CFGS15:
        per_v, sets = {}, {}
        for vid, (H, rows, gts, _, quad) in videos.items():
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(cfg, quad=quad))]
            mm_, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
            per_v[vid] = (mv, len(fp) + (len(mm_) - mv),
                          sum(1 for j in fn if gts[j].get("valid", True)))
            sets[vid] = {"matched": {gts[j]["frame_id"] for i, j in mm_ if gts[j].get("valid", True)},
                         "fp": {preds[i]["frame_id"] for i in fp},
                         "fn": {gts[j]["frame_id"] for j in fn if gts[j].get("valid", True)}}
        results[name] = per_v
        detail[name] = sets
    print("== 公开 15 ==")
    print(f"{'配置':16s} | {'0度':>14s} | {'45度':>14s} | {'90度':>14s} | {'全部':>16s}")
    for name, _ in CFGS15:
        cells = []
        for a in angles:
            vs = [v for v in videos if a == "全部" or videos[v][3] == a]
            tp = sum(results[name][v][0] for v in vs)
            fp = sum(results[name][v][1] for v in vs)
            fn = sum(results[name][v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
        print(f"{name:16s} | {cells[0]:>14s} | {cells[1]:>14s} | {cells[2]:>14s} | {cells[3]:>16s}")
    base = detail["候选栈(基准)"]
    for name, _ in CFGS15[1:]:
        cur = detail[name]
        nm = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["matched"] - base[v]["matched"])]
        lm = [f"{v} f{f}" for v in videos for f in sorted(base[v]["matched"] - cur[v]["matched"])]
        nf = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["fp"] - base[v]["fp"])]
        rf = [f"{v} f{f}" for v in videos for f in sorted(base[v]["fp"] - cur[v]["fp"])]
        nfn = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["fn"] - base[v]["fn"])]
        print(f"[{name}] 新命中 {nm} | 丢命中 {lm} | 新增FP {nf} | 消除FP {rf} | 新增FN {nfn}")


def eval154():
    import csv
    ang = {}
    for r in csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines()):
        ang[r["video_id"]] = int(r["angle_degrees"])
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)
    print("\n== 154（v2 候选标签，分视角） ==")
    print(f"{'配置':18s} | {'0度(82)':>15s} | {'45度(11)':>14s} | {'90度(61)':>15s} | {'全部':>15s}")
    detail = {}
    for name, cfg, _ in CFGS154:
        per_v, sets = {}, {}
        for vid in sorted(by_v):
            rows = G.load(RAW154 / f"{vid}.jsonl")
            rows.sort(key=lambda r: r["input_frame_id"])
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, None, cfg)]
            mm_, fp, fn = G.match(preds, by_v[vid])
            per_v[vid] = (len(mm_), len(fp), len(fn))
            sets[vid] = {"matched": {by_v[vid][j]["frame_id"] for i, j in mm_},
                         "fp": {preds[i]["frame_id"] for i in fp},
                         "fn": {by_v[vid][j]["frame_id"] for j in fn}}
        detail[name] = sets
        cells = []
        for a in (0, 45, 90, None):
            vs = [v for v in by_v if a is None or ang.get(v) == a]
            tp = sum(per_v[v][0] for v in vs)
            fp = sum(per_v[v][1] for v in vs)
            fn = sum(per_v[v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
            cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
        print(f"{name:18s} | {cells[0]:>15s} | {cells[1]:>14s} | {cells[2]:>15s} | {cells[3]:>15s}")
    for base_n, cur_n in (("F1F2F3", "F1F2F3+D3(0.45)"), ("+D2c", "+D2c+D3(0.45)")):
        base, cur = detail[base_n], detail[cur_n]
        nm = [f"{v} f{f}" for v in by_v for f in sorted(cur[v]["matched"] - base[v]["matched"])]
        lm = [f"{v} f{f}" for v in by_v for f in sorted(base[v]["matched"] - cur[v]["matched"])]
        nf = [f"{v} f{f}" for v in by_v for f in sorted(cur[v]["fp"] - base[v]["fp"])]
        nfn = [f"{v} f{f}" for v in by_v for f in sorted(cur[v]["fn"] - base[v]["fn"])]
        print(f"[{cur_n} vs {base_n}] 新命中 {len(nm)} | 丢命中 {len(lm)} | 新增FP {len(nf)} | 新增FN {len(nfn)}")
        print(f"  新命中: {nm}")
        if lm or nf or nfn:
            print(f"  丢命中: {lm} | 新增FP: {nf} | 新增FN: {nfn}")


if __name__ == "__main__":
    eval15()
    eval154()

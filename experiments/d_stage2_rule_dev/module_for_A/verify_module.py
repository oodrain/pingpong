# -*- coding: utf-8 -*-
"""验证 event_rules.py 与已验证回放（m12_eval.replay_m12 候选栈全配置）逐条一致。

- 公开 15：带桌面几何（门+双判+固定范围门 range_frac=0.03/window=2 启用）；顺带复算 F1
- 154：无几何；顺带复算（v2 标签，应 = 585/459/150 ≈ 0.658，范围门不生效）
"""
import importlib.util
import json
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("event_rules", HERE / "event_rules.py")
ER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ER)

src_path = (HERE / "m12_eval.py") if (HERE / "m12_eval.py").exists() else (HERE.parent / "m12_eval.py")
src = src_path.read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

CFG = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
       "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2, "gate_quad": 4.0,
       "D3": True, "D3_ratio": 0.45, "D3_min": 3.0,
       "range_frac": 0.03, "range_window": 2, "range_min_px": 4.0}

V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")
RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")


def key(events):
    return [(e["frame_id"], round(e["x"], 6), round(e["y"], 6)) for e in events]


def main():
    # —— 公开 15 ——
    man = {v["video_id"]: v for v in json.loads(G.MAN.read_text(encoding="utf-8"))["videos"]}
    gt = G.load(G.GT_P)
    gts_by_v = defaultdict(list)
    for g in gt:
        gts_by_v[g["video_id"]].append(g)
    ok = True
    tp = fp = fn = 0
    for vid in man:
        doc = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))
        seg = doc["segments"][0]
        H = G.build_H(seg)
        rows = G.load(G.RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        a = ER.detect_events(rows, table={"corners": seg["corners"], "net": seg.get("net")})
        b = [{"frame_id": e["frame_id"], "x": e["x"], "y": e["y"]}
             for e in replay_m12(rows, H, dict(CFG, quad=seg["corners"]))]
        same = key(a) == key(b)
        if not same:
            ok = False
            print(f"[不一致] {vid}: 模块 {len(a)} vs 回放 {len(b)}")
            sa, sb = set(map(tuple, key(a))), set(map(tuple, key(b)))
            print("  仅模块:", sorted(sa - sb)[:5], " 仅回放:", sorted(sb - sa)[:5])
        preds = [{"video_id": vid, **e} for e in a]
        gts = gts_by_v.get(vid, [])
        mm_, f_, n_ = G.match(preds, gts)
        mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
        tp += mv
        fp += len(f_) + (len(mm_) - mv)
        fn += sum(1 for j in n_ if gts[j].get("valid", True))
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"公开15：逐条一致 = {ok}；模块 F1 复算 = {tp}/{fp}/{fn} = {f1:.3f}（GT 角点 + v1.2 范围门；无范围门 0.744）")

    # —— 154 ——
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)
    ok2 = True
    tp = fp = fn = 0
    cfg154 = {k: v for k, v in CFG.items() if k != "gate"}  # 154 无几何
    for vid in sorted(by_v):
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        a = ER.detect_events(rows)
        b = [{"frame_id": e["frame_id"], "x": e["x"], "y": e["y"]}
             for e in replay_m12(rows, None, cfg154)]
        if key(a) != key(b):
            ok2 = False
            print(f"[不一致] {vid}: 模块 {len(a)} vs 回放 {len(b)}")
        preds = [{"video_id": vid, **e} for e in a]
        mm_, f_, n_ = G.match(preds, by_v[vid])
        tp += len(mm_)
        fp += len(f_)
        fn += len(n_)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"154：逐条一致 = {ok2}；模块复算 = {tp}/{fp}/{fn} = {f1:.3f}（应 585/459/150 ≈ 0.658）")


if __name__ == "__main__":
    main()

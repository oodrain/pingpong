# -*- coding: utf-8 -*-
"""154 评估（候选标签 v2 + 视角映射）：F1F2F3 与 +D2c 的总体与分视角指标。

输入：
- B 的候选标签 v2（735 事件）：E:/pingpong-data/adjudicated_candidates_v2_20261013/.../candidate_events.jsonl
- 视角映射：同目录 data/manifests/video_angles.csv
- 轨迹：E:/pingpong-data/C_train154_20261011_v3/raw
输出：控制台表格 + m12_154_v2_stats.json
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
exec(compile(src, "p", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

BASE_NG = {"F1": True, "F2": True, "F3": True}
D2C_NG = {**BASE_NG, "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
          "D_smax": 50.0, "D_dmin": 2}


def load_list(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ev2 = load_list(V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)
    print(f"v2 候选标签: {len(ev2)} 条 / {len(by_v)} 视频")

    import csv
    ang = {}
    for r in csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines()):
        ang[r["video_id"]] = int(r["angle_degrees"])

    res = {}
    added_events = [e for e in ev2 if e.get("source", "").startswith("missing_label_audit")]
    for name, cfg in (("F1F2F3", BASE_NG), ("F1F2F3+D2c", D2C_NG)):
        per_video = {}
        matched_frames = set()
        for vid in sorted(by_v):
            rows = G.load(RAW154 / f"{vid}.jsonl")
            rows.sort(key=lambda r: r["input_frame_id"])
            preds = [{"video_id": vid, **e} for e in replay_m12(rows, H=None, cfg=cfg)]
            mm, fp, fn = G.match(preds, by_v[vid])
            per_video[vid] = (len(mm), len(fp), len(fn))
            for i, j in mm:
                matched_frames.add((vid, by_v[vid][j]["frame_id"], by_v[vid][j].get("source", "")))
        res[name] = {"per_video": per_video, "matched": matched_frames}

    print(f"\n{'配置':14s} | {'0度(82)':>12s} | {'45度(11)':>11s} | {'90度(61)':>12s} | {'全部(154)':>14s}")
    for name in res:
        cells = []
        for a in (0, 45, 90, None):
            vs = [v for v in by_v if a is None or ang.get(v) == a]
            tp = sum(res[name]["per_video"][v][0] for v in vs)
            fp = sum(res[name]["per_video"][v][1] for v in vs)
            fn = sum(res[name]["per_video"][v][2] for v in vs)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
            p = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            cells.append(f"{tp}/{fp}/{fn} P{p:.2f} R{r_:.2f} F1{f1:.3f}")
        print(f"{name:14s} | {cells[0]:>26s} | {cells[1]:>25s} | {cells[2]:>26s} | {cells[3]:>30s}")

    # 对账：审计新增 14 条是否被我们命中
    base_m = res["F1F2F3"]["matched"]
    d2c_m = res["F1F2F3+D2c"]["matched"]
    print(f"\n审计新增事件（14 条）匹配情况:")
    hit_b = hit_d = miss = 0
    for e in added_events:
        key = (e["video_id"], e["frame_id"], e.get("source", ""))
        kb = any(v == e["video_id"] and f == e["frame_id"] for (v, f, s) in base_m)
        kd = any(v == e["video_id"] and f == e["frame_id"] for (v, f, s) in d2c_m)
        tag = "D2c命中" if kd else ("基准命中" if kb else "未命中")
        print(f"  {e['video_id']} f{e['frame_id']} ({e.get('source','')}) -> {tag}")
        hit_d += kd
        hit_b += kb and not kd
        miss += not kd and not kb
    print(f"  小计: D2c 命中 {hit_d}，仅基准命中 {hit_b}，未命中 {miss}")

    (HERE / "m12_154_v2_stats.json").write_text(
        json.dumps({n: res[n]["per_video"] for n in res} | {"angles": ang}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("\n已保存 m12_154_v2_stats.json")


if __name__ == "__main__":
    main()

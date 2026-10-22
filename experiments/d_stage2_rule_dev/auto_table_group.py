# -*- coding: utf-8 -*-
"""自动桌面·按来源组投票版（154 演练 + 提交形态）。

流程：
1) 采集：对 154 条视频各取 24 帧候选池（auto_table.frame_pool，含球点覆盖率）；
2) 组共识：同来源组（2 位前缀，固定机位）内全部候选一起聚类投票，取
   "成员数优先 + 平均覆盖率门槛"的最大簇中值为该组桌面四边形；
3) 评估：每条视频用本组共识角点跑带门回放（候选标签 v2）；
   安全守卫——带门发射数 < 无门发射数×20% 时判几何不可靠，回退无门；
4) 客观校验：共识四边形 vs 训练包逐帧 table bbox（粗对照）。

用法：python auto_table_group.py collect   （采集，缓存 auto154_pools.json）
      python auto_table_group.py eval      （共识 + 评估）
"""
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADJ = Path("E:/annotation-data/adjudication")
RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")
V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")

spec = importlib.util.spec_from_file_location("auto_table", HERE / "auto_table.py")
AT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AT)

src = (HERE / "m12_eval.py").read_text(encoding="utf-8").split("BASE =")[0]
m = types.ModuleType("m12mod")
m.__dict__["__file__"] = str(HERE / "m12_eval.py")
exec(compile(src, "m12_eval_part.py", "exec"), m.__dict__)
G = m.G
replay_m12 = m.replay_m12

FULL = {"gate": 900, "F1": True, "F2": True, "F3": True, "D2": True, "D2_back": 5,
        "D_smin": 8.0, "D_ratio": 0.40, "D_smax": 50.0, "D_dmin": 2, "gate_quad": 4.0,
        "D3": True, "D3_ratio": 0.45}
NOG = {k: v for k, v in FULL.items() if k not in ("gate", "gate_quad")}

N_FRAMES = 24


def video_path(vid):
    grp, seq = vid.split("_")
    for member in "ABCD":
        p = ADJ / member / "videos" / grp / f"{seq}.mp4"
        if p.exists():
            return p
    return None


def ball_pts_of(vid):
    rows = G.load(RAW154 / f"{vid}.jsonl")
    return [(r["x"], r["y"]) for r in rows
            if r.get("detected") and r.get("x") is not None and (r.get("confidence") or 0) >= 0.75]


def collect():
    import cv2
    import csv
    vids = [r["video_id"] for r in csv.DictReader((V2 / "data/manifests/video_angles.csv")
            .read_text(encoding="utf-8-sig").splitlines())]
    pools = {}
    for k, vid in enumerate(vids):
        vp = video_path(vid)
        ball = ball_pts_of(vid)
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, min(N_FRAMES, max(2, total))).astype(int)
        pool = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            for (q, sc, f, name) in AT.frame_pool(fr, ball):
                pool.append([float(q[0][0]), float(q[0][1]), float(q[1][0]), float(q[1][1]),
                             float(q[2][0]), float(q[2][1]), float(q[3][0]), float(q[3][1]),
                             round(float(sc), 3), round(float(f), 3)])
        cap.release()
        pools[vid] = pool
        if (k + 1) % 10 == 0:
            print(f'  采集 {k+1}/{len(vids)}')
    (HERE / "auto154_pools.json").write_text(json.dumps(pools), encoding="utf-8")
    print(f'已保存 auto154_pools.json（{len(pools)} 条视频）')


def consensus_of(pool, tol=60.0, min_size=4, min_frac=0.5):
    """组内候选投票：按覆盖率降序贪心聚簇，取成员数优先的最大簇中值。"""
    if not pool:
        return None
    entries = [(np.array(e[:8], float).reshape(4, 2), e[8], e[9]) for e in pool]
    clusters = []
    for q, sc, f in sorted(entries, key=lambda t: -t[2]):
        placed = False
        for cl in clusters:
            if np.max(np.linalg.norm(q - cl["med"], axis=1)) <= tol:
                cl["members"].append((q, f))
                cl["med"] = np.median(np.stack([mm[0] for mm in cl["members"]]), axis=0)
                placed = True
                break
        if not placed:
            clusters.append({"med": q.copy(), "members": [(q, f)]})
    if not clusters:
        return None
    best = max(clusters, key=lambda cl: (len(cl["members"]),
                                         float(np.mean([mm[1] for mm in cl["members"]]))))
    mf = float(np.mean([mm[1] for mm in best["members"]]))
    if len(best["members"]) < min_size or mf < min_frac:
        return None
    return best["med"], len(best["members"]), mf


def group_bbox(grp):
    """组内训练包 table bbox 中值（固定机位 → 稳定）；无则 None。"""
    boxes = []
    for member in "ABCD":
        d = ADJ / member / "json" / grp
        if not d.exists():
            continue
        for jp in d.glob("*.json"):
            doc = json.loads(jp.read_text(encoding="utf-8"))
            for fr in doc.get("frames", []):
                for an in fr.get("annotations", []):
                    if an.get("label") == "table" and an.get("bbox_xyxy"):
                        boxes.append(an["bbox_xyxy"][0])
    if len(boxes) < 5:
        return None
    import numpy as _np
    arr = _np.array(boxes, float)
    med = _np.median(arr, axis=0)
    ok = _np.all(_np.abs(arr - med) <= 0.08 * 1920 + 30, axis=1)  # 剔除漂移明显的脏帧
    if ok.sum() >= 5:
        med = _np.median(arr[ok], axis=0)
    return med.tolist()


def aabb(qs):
    xs = [p[0] for p in qs]
    ys = [p[1] for p in qs]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def eval_154():
    import csv
    pools = json.loads((HERE / "auto154_pools.json").read_text(encoding="utf-8"))
    ang = {r["video_id"]: int(r["angle_degrees"]) for r in
           csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines())}
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)

    groups = defaultdict(list)
    for vid in by_v:
        groups[vid.split("_")[0]].append(vid)

    cons = {}
    print("组共识（含训练包 bbox 客观校验）：")
    for grp, vids in sorted(groups.items()):
        pool = [e for vid in vids for e in pools.get(vid, [])]
        r = consensus_of(pool)
        bb = group_bbox(grp)
        verdict = ""
        if r and bb is not None:
            iv = iou(aabb(r[0]), bb)
            if iv < 0.3:
                verdict = f" → bbox校验不过(IoU {iv:.2f})，弃用"
                r = None
            else:
                verdict = f" → bbox校验通过(IoU {iv:.2f})"
        cons[grp] = r
        if r:
            q, n, mf = r
            print(f"  组{grp}: {len(vids)}条视频, 共识成员 {n} 候选, 平均覆盖率 {mf:.2f}, "
                  f"中心=({q.mean(axis=0)[0]:.0f},{q.mean(axis=0)[1]:.0f}){verdict}")
        else:
            print(f"  组{grp}: {len(vids)}条视频, 无可用共识{verdict}")

    tot_gate = [0, 0, 0]
    tot_nog = [0, 0, 0]
    guard_hits = []
    for vid in sorted(by_v):
        grp = vid.split("_")[0]
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        gts = by_v[vid]
        preds_nog = [{"video_id": vid, **e} for e in replay_m12(rows, None, NOG)]
        mmn, fpn, fnn = G.match(preds_nog, gts)
        t_nog = (len(mmn), len(fpn), len(fnn))
        use = None
        if cons.get(grp):
            q = cons[grp][0]
            H = G.build_H({"corners": q.tolist(), "net": None})
            preds_g = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=q.tolist()))]
            if len(preds_g) < 0.2 * max(1, len(preds_nog)):
                guard_hits.append(vid)
                t = t_nog
            else:
                mmg, fpg, fng = G.match(preds_g, gts)
                t = (len(mmg), len(fpg), len(fng))
        else:
            t = t_nog
        for i in range(3):
            tot_gate[i] += t[i]
            tot_nog[i] += t_nog[i]
    F = lambda t: 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t[0] else 0.0
    print(f"\n154 对照：无门 {tot_nog} = {F(tot_nog):.3f}｜组共识+门(含守卫) {tot_gate} = {F(tot_gate):.3f}")
    print(f"守卫触发（带门<20%回退）: {len(guard_hits)} 条 {guard_hits[:8]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if mode == "collect":
        collect()
    else:
        eval_154()

# -*- coding: utf-8 -*-
"""固定球桌范围门（C 方案复现）：候选点必须在图像桌面四边形 ± 容差内。

背景（C 的 TT3D-TCN 分支 LEGACY_20261016）：其旧入口在神经事件解码后加"固定球桌范围"
后处理——球心到四边形有符号欧氏距离 ≤ max(2% 四边对角线, 4px) 才保留；公共 15 上
候选 115→97、严格口径 77/38/15 → 77/20/15（81.48%），TP 全保留。

我们的几何门是"毫米投影 900mm 或图像四边形 ±4px"双判（失败放行），对"击球等悬在台面上方、
毫米投影仍贴近台面"的 FP 偏宽松。本实验在现有候选栈（F1F2F3+D2c+D3+双判）之后加一道
图像空间固定范围门，做：
  A) 公开 15 · 人工角点：距离分布诊断（TP vs FP 的可分性）+ 容差扫描；
  B) 公开 15 · 自动角点（auto_table_public15.json）＋ 20% 守卫：系统口径是否受益；
  C) 154 · 组共识角点（auto154_group_quads.json，仅 accepted 组）：纯规则 / 门 / 门+范围门。

口径与既有实验一致：±1 帧/50px 一对一，官方 GT 口径 A（公开 15）；候选标签 v2（154）。
"""
import importlib.util
import json
import math
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = Path("E:/pingpong-data/adjudicated_candidates_v2_20261013")
RAW154 = Path("E:/pingpong-data/C_train154_20261011_v3/raw")

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
FRACS = [None, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]


def quad_arr(quad):
    return np.array(quad, np.float32).reshape(-1, 1, 2)


def signed_dist(quad, x, y):
    return float(cv2.pointPolygonTest(quad_arr(quad), (float(x), float(y)), True))


def quad_diag(quad):
    p = np.array(quad, float)
    return max(math.dist(p[i], p[j]) for i in range(4) for j in range(i + 1, 4))


def tol_of(frac, quad):
    return max(frac * quad_diag(quad), 4.0)


def apply_range(events, quad, frac):
    if frac is None or quad is None:
        return list(events)
    t = tol_of(frac, quad)
    return [e for e in events if signed_dist(quad, e["x"], e["y"]) >= -t]


def metrics(per_v, vids):
    tp = sum(per_v[v][0] for v in vids)
    fp = sum(per_v[v][1] for v in vids)
    fn = sum(per_v[v][2] for v in vids)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return tp, fp, fn, f1


def fmt_cells(per_v, videos, angles):
    cells = []
    for a in angles:
        vs = [v for v in videos if a == "全部" or videos[v][3] == a]
        tp, fp, fn, f1 = metrics(per_v, vs)
        cells.append(f"{tp}/{fp}/{fn}/{f1:.3f}")
    return cells


# ---------------- A) 公开 15 · 人工角点 ----------------
def part_a(videos):
    print("=" * 100)
    print("A) 公开 15 · 人工角点：候选栈（F1F2F3+D2c+D3+双判）之后加固定范围门")
    angles = ["0度", "45度", "90度", "全部"]
    results, detail, diag = {}, {}, []
    for vid, (H, rows, gts, _a, quad) in videos.items():
        preds = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=quad))]
        mm_, fp, fn = G.match(preds, gts)
        mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
        matched_idx = {i for i, j in mm_ if gts[j].get("valid", True)}
        for i, p in enumerate(preds):
            diag.append({"video": vid, "f": p["frame_id"], "x": p["x"], "y": p["y"],
                         "tp": i in matched_idx,
                         "d": round(signed_dist(quad, p["x"], p["y"]), 1),
                         "diag": round(quad_diag(quad), 1)})
        for frac in FRACS:
            keep = apply_range(preds, quad, frac)
            mm2, fp2, fn2 = G.match(keep, gts)
            mv2 = sum(1 for i, j in mm2 if gts[j].get("valid", True))
            results.setdefault(frac, {})[vid] = (mv2, len(fp2) + (len(mm2) - mv2),
                                                 sum(1 for j in fn2 if gts[j].get("valid", True)))
            detail.setdefault(frac, {})[vid] = {
                "matched": {gts[j]["frame_id"] for i, j in mm2 if gts[j].get("valid", True)},
                "fp": {keep[i]["frame_id"] for i in fp2}}
    print(f"{'容差(frac)':14s} | " + " | ".join(f"{a:>16s}" for a in angles))
    for frac in FRACS:
        label = "无范围门" if frac is None else f"{frac:.3f}(≥4px)"
        cells = fmt_cells(results[frac], videos, angles)
        print(f"{label:14s} | " + " | ".join(f"{c:>16s}" for c in cells))
    base = detail[None]
    for frac in FRACS[1:]:
        cur = detail[frac]
        lost = [f"{v} f{f}" for v in videos for f in sorted(base[v]["matched"] - cur[v]["matched"])]
        rem = [f"{v} f{f}" for v in videos for f in sorted(base[v]["fp"] - cur[v]["fp"])]
        newfp = [f"{v} f{f}" for v in videos for f in sorted(cur[v]["fp"] - base[v]["fp"])]
        print(f"  [frac={frac}] 消除FP {len(rem)} | 丢命中 {len(lost)} {lost[:6]}"
              f"{' …' if len(lost) > 6 else ''} | 新增FP {len(newfp)}")
    # 距离分布：TP/FP 在容差线上的可分性
    tps = [e for e in diag if e["tp"]]
    fps = [e for e in diag if not e["tp"]]
    def summarize(name, arr):
        ds = sorted(round(e["d"] / e["diag"] * 100, 2) for e in arr)
        if not ds:
            print(f"  {name}: 无"); return
        q = lambda p: ds[min(len(ds) - 1, int(p * (len(ds) - 1)))]
        print(f"  {name}: n={len(ds)} 距离/对角线%  p0={ds[0]} p10={q(.1)} p25={q(.25)} "
              f"中位={q(.5)} p75={q(.75)} p90={q(.9)} max={ds[-1]}")
        print(f"      超出 2%（即需删除）的条数: {sum(1 for d in ds if d < -2)}"
              f" | 超出 4px 的条数: {sum(1 for e in arr if e['d'] < -4)}")
    print("  距离分布（有符号，负=四边形外，按对角线归一化%）：")
    summarize("TP（有效命中）", tps)
    summarize("FP（误报）", fps)
    worst_fp = sorted(fps, key=lambda e: e["d"])
    print("  最靠外的 12 个 FP:", [(e["video"], e["f"], e["d"]) for e in worst_fp[:12]])
    print("  最靠外的 6 个 TP:", [(e["video"], e["f"], e["d"])
                                  for e in sorted(tps, key=lambda e: e["d"])[:6]])
    # 留一验证：容差在其余 14 条上选，评估留出条
    from collections import Counter
    fr = [f for f in FRACS if f is not None]
    tot, picks = [0, 0, 0], []
    for h in videos:
        best = None
        for f in fr:
            tp = fp = fn = 0
            for v in videos:
                if v == h:
                    continue
                c = results[f][v]
                tp += c[0]
                fp += c[1]
                fn += c[2]
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if best is None or f1 >= best[0] - 1e-12:
                best = (f1, f)
        picks.append(best[1])
        c = results[best[1]][h]
        for k in range(3):
            tot[k] += c[k]
    base_tot = [sum(results[None][v][k] for v in videos) for k in range(3)]
    F = lambda x: 2 * x[0] / (2 * x[0] + x[1] + x[2])
    print(f'  留一验证（人工角点，容差在其余14条上选）：选中分布 {dict(sorted(Counter(picks).items()))}；'
          f'留出合计 {tot} = {F(tot):.3f}（对照 无范围门 {base_tot} = {F(base_tot):.3f}）')
    (HERE / "range_gate_diag15.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return results


# ---------------- B) 公开 15 · 自动角点 + 20% 守卫 ----------------
def part_b(videos):
    print("\n" + "=" * 100)
    print("B) 公开 15 · 自动角点（auto_table_public15.json）+ 20% 守卫：系统口径（容差扫描）")
    auto = json.loads((HERE / "auto_table_public15.json").read_text(encoding="utf-8"))
    br_fracs = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    tot = {"无门(纯规则)": [0, 0, 0], "自动角点门(系统口径基准)": [0, 0, 0]}
    tot.update({f"+范围门{int(f*100)}%": [0, 0, 0] for f in br_fracs})
    guard = []
    lost_detail = {f: [] for f in br_fracs}
    bper = {f: {} for f in br_fracs}
    bper_base = {}
    for vid, (H0, rows, gts, _a, quad_gt) in videos.items():
        nog = [{"video_id": vid, **e} for e in replay_m12(rows, None, NOG)]

        def t_of(preds):
            mm_, fp, fn = G.match(preds, gts)
            mv = sum(1 for i, j in mm_ if gts[j].get("valid", True))
            return (mv, len(fp) + (len(mm_) - mv),
                    sum(1 for j in fn if gts[j].get("valid", True))), len(preds)

        def matched_set(preds):
            mm_, _, _ = G.match(preds, gts)
            return {preds[i]["frame_id"] for i, j in mm_ if gts[j].get("valid", True)}

        t_nog, n_nog = t_of(nog)
        for k in range(3):
            tot["无门(纯规则)"][k] += t_nog[k]
        if vid in auto:
            cq = auto[vid]["corners_auto"]
            net = json.loads((G.ANN / f"{vid}.json").read_text(encoding="utf-8"))["segments"][0].get("net")
            H = G.build_H({"corners": cq, "net": net})
            gated = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=cq))]
            t_g, n_g = t_of(gated)
            use_gate = n_g >= 0.2 * max(1, n_nog)
            if not use_gate:
                guard.append(vid)
            base_preds = gated if use_gate else nog
            t_base, _ = t_of(base_preds)
            for k in range(3):
                tot["自动角点门(系统口径基准)"][k] += t_base[k]
            bper_base[vid] = t_base
            m0 = matched_set(base_preds)
            for f in br_fracs:
                if use_gate:
                    rf = [{"video_id": vid, **e} for e in apply_range(base_preds, cq, f)]
                else:
                    rf = list(base_preds)
                t_rf, _ = t_of(rf)
                for k in range(3):
                    tot[f"+范围门{int(f*100)}%"][k] += t_rf[k]
                bper[f][vid] = t_rf
                lost = m0 - matched_set(rf)
                if lost:
                    lost_detail[f].append((vid, sorted(lost),
                                           auto[vid].get("err_mean"), auto[vid].get("frames_ok")))
        else:
            guard.append(vid)
            bper_base[vid] = t_nog
            for f in br_fracs:
                bper[f][vid] = t_nog
            for name in tot:
                if name == "无门(纯规则)":
                    continue
                for k in range(3):
                    tot[name][k] += t_nog[k]
    F = lambda t: 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t[0] else 0.0
    for name, t in tot.items():
        print(f"  {name:26s}: {t[0]}/{t[1]}/{t[2]} = {F(t):.3f}")
    print(f"  自动检测失败/守卫回退: {sorted(set(guard))}")
    for f in br_fracs:
        if lost_detail[f]:
            print(f"  丢命中 frac={f}: " + "; ".join(
                f"{v} f{fs} (自动角点误差 {e}px, {ok}帧)" for v, fs, e, ok in lost_detail[f]))
    print(f"  （对照：参考件 auto_table_eval 报告 系统口径 0.748）")
    # 自动角点路径 · 范围门7% 分视角
    line = "  自动角点路径 · 范围门7% 分视角: "
    for a in ("0度", "45度", "90度", "全部"):
        vs = [v for v in videos if a == "全部" or videos[v][3] == a]
        t = [sum(bper[0.07][v][k] for v in vs) for k in range(3)]
        line += f"{a} {t[0]}/{t[1]}/{t[2]}/{F(t):.3f}  " if t[0] else f"{a} 0/-/-  "
    print(line)


# ---------------- C) 154 · 组共识角点 ----------------
def part_c():
    print("\n" + "=" * 100)
    print("C) 154 · 组共识角点（accepted 组）：纯规则 / 门 / 门+范围门")
    import csv
    ang = {r["video_id"]: int(r["angle_degrees"]) for r in
           csv.DictReader((V2 / "data/manifests/video_angles.csv").read_text(encoding="utf-8-sig").splitlines())}
    ev2 = [json.loads(l) for l in
           (V2 / "experiments/manual_label_audit/adjudicated_label_candidates_v2/candidate_events.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_v = defaultdict(list)
    for e in ev2:
        by_v[e["video_id"]].append(e)
    groups = json.loads((HERE / "auto154_group_quads.json").read_text(encoding="utf-8"))
    acc = {g: d["corners"] for g, d in groups.items() if d.get("accepted") and d.get("corners")}

    names = ["纯规则", "组门", "组门+范围门1%", "组门+范围门2%", "组门+范围门3%", "纯规则+范围门2%"]
    stat = {n: defaultdict(lambda: [0, 0, 0]) for n in names}
    guard = []
    for vid in sorted(by_v):
        grp = vid.split("_")[0]
        rows = G.load(RAW154 / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        gts = by_v[vid]
        nog = [{"video_id": vid, **e} for e in replay_m12(rows, None, NOG)]
        def t_of(preds):
            mm_, fp, fn = G.match(preds, gts)
            return (len(mm_), len(fp), len(fn)), len(preds)
        t_nog, n_nog = t_of(nog)
        t = {n: (t_nog if not n.startswith("组门") and n != "纯规则+范围门2%" else None) for n in names}
        if grp in acc:
            cq = acc[grp]
            H = G.build_H({"corners": cq, "net": None})
            gated = [{"video_id": vid, **e} for e in replay_m12(rows, H, dict(FULL, quad=cq))]
            t_g, n_g = t_of(gated)
            use = n_g >= 0.2 * max(1, n_nog)
            if not use:
                guard.append(vid)
            for n, frac in [("组门", None), ("组门+范围门1%", 0.01),
                            ("组门+范围门2%", 0.02), ("组门+范围门3%", 0.03)]:
                if use:
                    preds = [{"video_id": vid, **e} for e in apply_range(gated, cq, frac)]
                    t[n] = t_of(preds)[0]
                else:
                    t[n] = t_nog
            if use:
                t["纯规则+范围门2%"] = t_of([{"video_id": vid, **e} for e in
                                            apply_range(nog, cq, 0.02)])[0]
            else:
                t["纯规则+范围门2%"] = t_nog
        else:
            for n in names:
                t[n] = t_nog
        for n in names:
            for k in range(3):
                stat[n][vid][k] = t[n][k]
    F = lambda t: 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t[0] else 0.0
    print(f"{'配置':18s} | " + " | ".join(f"{a:>16s}" for a in ["0度(82)", "45度(11)", "90度(61)", "全部(154)"]))
    for n in names:
        cells = []
        for a in (0, 45, 90, None):
            vs = [v for v in by_v if a is None or ang.get(v) == a]
            tp = sum(stat[n][v][0] for v in vs)
            fp = sum(stat[n][v][1] for v in vs)
            fn = sum(stat[n][v][2] for v in vs)
            cells.append(f"{tp}/{fp}/{fn}/{2*tp/(2*tp+fp+fn):.3f}" if tp else "0/-/-")
        print(f"{n:18s} | " + " | ".join(f"{c:>16s}" for c in cells))
    print(f"  accepted 来源组: {sorted(acc)}（共 {len(acc)} 组）；守卫回退视频数: {len(guard)}")


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
        videos[vid] = (H, rows, gts_by_v.get(vid, []), doc.get("angle"), seg["corners"])
    part_a(videos)
    part_b(videos)
    part_c()


if __name__ == "__main__":
    main()

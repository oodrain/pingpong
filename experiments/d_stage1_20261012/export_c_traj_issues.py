# -*- coding: utf-8 -*-
"""导出给 C 的轨迹问题清单（回答"61 跳点 + 56 缺口具体是哪些"）。

来源：
- 错误分类 v1 事件表：error_analysis_v1_20261011/{public15,train154}_events.jsonl
- 人工复核（185 张采样）：review/复核工作表_已填写.csv

输出：
- C_轨迹问题清单_v1.csv（UTF-8-BOM，Excel 可直接打开）
- 控制台摘要（校验重算与原始分类的一致性）

帧号约定（与轨迹缓存/预测一致）：
- 真实帧 = predictions.jsonl 的 frame_id = 标签 frame_id = 输入帧 - 1
- 输入帧 = raw/*.jsonl 的 input_frame_id

定位重算规则（与分类 v1 完全一致）：
- 跳点：误报点邻域 ±3 点内相邻位移 > max(80px, 5×全片相邻位移中位)
- 缺口附近：误报点前后相邻达标点输入帧间隔 >= 3（即缺 >= 2 帧）
- 邻居缺口：漏报点与前后邻居输入帧间隔 >= 3
"""
import csv
import json
import math
import statistics
from pathlib import Path

EA = Path("E:/pingpong-data/error_analysis_v1_20261011")
OUT = Path(__file__).resolve().parent

RAW = {
    "train154": Path("E:/pingpong-data/C_train154_20261011_v3/raw"),
    "public15": Path("E:/pingpong-data/C_public15_20261009_v1/raw"),
}

CODE_MEAN = {
    "T": "T=确认真实触台（根因是标签缺失/待 B 核对，非轨迹误发）",
    "K": "K=确认 AI 技术归因（轨迹问题属实）",
    "D": "D=触地", "J": "J=击球", "W": "W=触网", "F": "F=飞行高点",
    "B": "B=帧间触台", "N": "N=不是触台", "?": "?=判断不了",
}

TYPE_ORDER = ["跳点", "缺口附近", "未检测", "低置信", "邻居缺口", "边框（尾帧）", "边框（超范围）"]
SORT_DATASET = {"train154": 0, "public15": 1}
CAT2TYPE = {
    "jump": "跳点", "near_gap": "缺口附近", "no_detection": "未检测",
    "low_conf": "低置信", "neighbor_gap": "邻居缺口",
    "tail_edge": "边框（尾帧）", "out_of_range": "边框（超范围）",
}
WANT = set(CAT2TYPE)


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


class Traj:
    def __init__(self, raw_path):
        self.rows = load_jsonl(raw_path)
        self.rows.sort(key=lambda r: r["input_frame_id"])
        self.by_input = {r["input_frame_id"]: r for r in self.rows}
        self.pts = [r for r in self.rows
                    if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= 0.75]
        self.n_frames = (self.rows[-1]["input_frame_id"] + 1) if self.rows else 0
        ds = [math.dist((self.pts[k]["x"], self.pts[k]["y"]), (self.pts[k + 1]["x"], self.pts[k + 1]["y"]))
              for k in range(len(self.pts) - 1)]
        self.med_vel = float(statistics.median(ds)) if ds else 0.0

    def idx_of_fid(self, fid):
        for k, r in enumerate(self.pts):
            if r["input_frame_id"] == fid:
                return k
        return None


def rf(x):  # input_frame_id -> 真实帧
    return x - 1


def loc_jump(t, fid):
    j = t.idx_of_fid(fid)
    lo, hi = max(0, j - 3), min(len(t.pts) - 1, j + 3)
    hits = []
    for k in range(lo, hi):
        dk = math.dist((t.pts[k]["x"], t.pts[k]["y"]), (t.pts[k + 1]["x"], t.pts[k + 1]["y"]))
        if dk > max(80.0, 5 * t.med_vel):
            hits.append((t.pts[k]["input_frame_id"], t.pts[k + 1]["input_frame_id"], dk))
    if not hits:
        return None, None
    a, b, dk = hits[0]
    txt = (f"邻域跳点：达标点输入帧 {a}→{b}（真实帧 {rf(a)}→{rf(b)}）相邻位移 {dk:.0f}px"
           f"（全片相邻位移中位 {t.med_vel:.0f}px）")
    if b - a > 1:
        txt += f"；该两帧间另有 {b - a - 1} 帧未达标（跳点/缺口并发）"
    if len(hits) > 1:
        txt += f"；窗内共 {len(hits)} 处超阈位移"
    return txt, (a, b, dk)


def gap_segs(t, fid):
    j = t.idx_of_fid(fid)
    segs = []
    if j >= 1 and t.pts[j]["input_frame_id"] - t.pts[j - 1]["input_frame_id"] >= 3:
        segs.append(("前侧", t.pts[j - 1]["input_frame_id"], t.pts[j]["input_frame_id"]))
    if j + 1 < len(t.pts) and t.pts[j + 1]["input_frame_id"] - t.pts[j]["input_frame_id"] >= 3:
        segs.append(("后侧", t.pts[j]["input_frame_id"], t.pts[j + 1]["input_frame_id"]))
    return segs


def loc_gap(t, fid):
    segs = gap_segs(t, fid)
    if not segs:
        return None, None
    parts = []
    for side, a, b in segs:
        parts.append(f"{side}缺口：输入帧 {a + 1}–{b - 1} 缺 {b - a - 1} 帧"
                     f"（对应真实帧 {rf(a + 1)}–{rf(b - 1)}）")
    return "；".join(parts), segs


def loc_neighbor_gap(t, fid):
    j = t.idx_of_fid(fid)
    if j is None:
        return "（漏报点输入帧不在达标轨迹点序列中）", None
    segs = []
    if j >= 1 and fid - t.pts[j - 1]["input_frame_id"] >= 3:
        segs.append(("前侧", t.pts[j - 1]["input_frame_id"], fid))
    if j + 1 < len(t.pts) and t.pts[j + 1]["input_frame_id"] - fid >= 3:
        segs.append(("后侧", fid, t.pts[j + 1]["input_frame_id"]))
    parts = [f"{side}缺口：输入帧 {a + 1}–{b - 1} 缺 {b - a - 1} 帧（对应真实帧 {rf(a + 1)}–{rf(b - 1)}）"
             for side, a, b in segs]
    return "；".join(parts) if parts else "（无 ≥2 帧缺口？）", segs


def neighbors_ctx(t, fid):
    prev = [p["input_frame_id"] for p in t.pts if p["input_frame_id"] < fid]
    nxt = [p["input_frame_id"] for p in t.pts if p["input_frame_id"] > fid]
    p = max(prev) if prev else None
    q = min(nxt) if nxt else None
    s = []
    s.append(f"前最近达标输入帧 {p}（距 {fid - p}）" if p is not None else "前无达标帧")
    s.append(f"后最近达标输入帧 {q}（距 {q - fid}）" if q is not None else "后无达标帧")
    return "；".join(s)


def review_index():
    keyed = {}
    p = EA / "review" / "复核工作表_已填写.csv"
    for r in csv.DictReader(p.read_text(encoding="gbk").splitlines()):
        key = (r["数据集"], r["视频"], int(r["帧"]), r["AI预归因"])
        code = (r["人工结论"] or "").strip()
        keyed[key] = (code, (r["备注"] or "").strip())
    return keyed


def main():
    review = review_index()
    events = {}
    for ds in ("train154", "public15"):
        events[ds] = [e for e in load_jsonl(EA / f"{ds}_events.jsonl")
                      if e["category"] in WANT and (e["kind"] == "FP" or e["kind"].startswith("FN_valid"))]

    trajs = {}
    rows_out = []
    mismatches, review_tab = [], {}
    for ds, evs in events.items():
        for e in evs:
            vid, cat = e["video_id"], e["category"]
            f = int(e["frame_id"])
            fid = f + 1
            key = (ds, vid)
            if key not in trajs:
                trajs[key] = Traj(RAW[ds] / f"{vid}.jsonl")
            t = trajs[key]
            kind = "FP" if e["kind"] == "FP" else "FN"
            ev_txt = "FP（误报：规则多射）" if kind == "FP" else "FN（漏报：规则漏抓）"
            loc = ""
            detail = e.get("detail", "")
            if cat == "jump":
                loc, chk = loc_jump(t, fid)
                if loc is None:
                    loc = "（重算未发现超阈跳点？）"
                elif chk:
                    # 校验：原始 detail 形如"邻近位移 80px（全片中位 14px）"
                    try:
                        dk0 = float(detail.split("邻近位移 ")[1].split("px")[0])
                        if abs(dk0 - chk[2]) > 1.0:
                            mismatches.append((ds, vid, f, cat, detail, loc))
                    except Exception:
                        mismatches.append((ds, vid, f, cat, detail, loc))
            elif cat == "near_gap":
                loc, segs = loc_gap(t, fid)
                if loc is None:
                    loc = "（重算未发现 ≥2 帧缺口？）"
                else:
                    try:
                        n0 = int(detail.split("缺口 ")[1].split(" 帧")[0])
                        n1 = sum(b - a - 1 for _, a, b in segs)
                        if n0 != n1:
                            mismatches.append((ds, vid, f, cat, detail, loc))
                    except Exception:
                        mismatches.append((ds, vid, f, cat, detail, loc))
            elif cat == "neighbor_gap":
                loc, _ = loc_neighbor_gap(t, fid)
            elif cat == "no_detection":
                rec = t.by_input.get(fid)
                state = "无记录" if rec is None else ("未检测" if not rec.get("detected") else "有检测")
                loc = f"raw 输入帧 {fid}（真实帧 {f}）：{state}；{neighbors_ctx(t, fid)}"
            elif cat == "low_conf":
                rec = t.by_input.get(fid)
                conf = rec.get("confidence") if rec else None
                loc = (f"raw 输入帧 {fid}（真实帧 {f}）：conf={conf if conf is None else round(conf, 2)}"
                       f" <0.75 被阈值过滤；{neighbors_ctx(t, fid)}")
            elif cat == "tail_edge":
                loc = f"漏报点位于达标轨迹末点（真实帧 {f}），其后无帧（尾帧未推）"
            elif cat == "out_of_range":
                loc = f"标签真实帧 {f}（输入帧 {fid}）超出 raw 范围（0..{t.n_frames - 1}）"

            rcode, rnote = review.get((ds, vid, f, cat), ("", ""))
            tabs = review_tab.setdefault(cat, {})
            if rcode:
                tabs[rcode] = tabs.get(rcode, 0) + 1

            rows_out.append({
                "数据集": ds, "问题类型": CAT2TYPE[cat], "事件": ev_txt,
                "video_id": vid, "真实帧": f, "输入帧": fid,
                "x": round(e["x"], 1), "y": round(e["y"], 1),
                "问题定位": loc, "证据图(本地)": Path(e["evidence"]).name if e.get("evidence") else "",
                "复核结论(采样)": CODE_MEAN.get(rcode, rcode), "复核备注(采样)": rnote,
            })

    rows_out.sort(key=lambda r: (TYPE_ORDER.index(r["问题类型"]), SORT_DATASET[r["数据集"]],
                                 r["video_id"], r["真实帧"]))
    out_csv = OUT / "C_轨迹问题清单_v1.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"写出 {out_csv} 共 {len(rows_out)} 行")
    import collections
    cnt = collections.Counter((r["数据集"], r["问题类型"]) for r in rows_out)
    for (ds, tp), n in sorted(cnt.items(), key=lambda x: (SORT_DATASET[x[0][0]], TYPE_ORDER.index(x[0][1]))):
        print(f"  {ds:9s} {tp}: {n}")
    print("\n复核采样结论分布（按问题类型）:")
    for cat, tab in review_tab.items():
        print(f"  {cat}: {dict(tab)}")
    print(f"\n重算与原分类不一致: {len(mismatches)} 条")
    for m in mismatches[:10]:
        print("  ", m)


if __name__ == "__main__":
    main()

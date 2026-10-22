# -*- coding: utf-8 -*-
"""按团队整合协议（integration/team-sync-20261022，2026-10-22）冻结验收 D 事件模块 v1.2。

协议：123 段开发池；42 段精确核查（可信/完整范围，166 事件）；625 条去重确认正例防误删；
31 段保留集与 15 段公开样例不参与。本脚本只做冻结评估，不调参；几何来自自动组共识角点。

变体（同一模块）：pure（无几何）/ geom_v12（模块默认：双判门 + 邻帧范围门 3%/±2）/
geom_v10（同角点、range_frac=0，隔离范围门增量）。

选型说明：v1.2 的 (window ±2, tol 3%) 在 9 组固定预算内、按拟合来源
（01/02/04/07/08/13/18/19/21/22）完整 F1 选出，并在内层来源（00/11/16）验证；
本脚本复现该评估（fit/inner 分列）。
"""
import importlib.util
import json
import sys
from pathlib import Path

REF = Path("E:/pingpong-data/team_sync_ref_20261022")
OUT = Path("E:/pingpong-data/rule_v2_dev/team_protocol_20261022")
RAW = Path("E:/pingpong-data/C_train154_20261011_v3/raw")
M = Path("E:/pingpong-data/rule_v2_dev")

sys.path.insert(0, str(REF / "development" / "evaluation"))
import team_data as TD  # noqa: E402

spec = importlib.util.spec_from_file_location("event_rules", M / "event_rules.py")
ER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ER)

FIT = {"01", "02", "04", "07", "08", "13", "18", "19", "21", "22"}
INNER = {"00", "11", "16"}


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = TD.read(REF / "data/splits/team_split_v1.json")
    base = REF / "labels/local_development_20261022"
    precise = TD.read(base / "precise_scopes.json")["videos"]
    positives = TD.read(base / "confirmed_positive_union.json")["videos"]
    dev = [v["video_id"] for v in split["videos"] if v["role"] == "development"]
    groups = json.loads((M / "auto154_group_quads.json").read_text(encoding="utf-8"))
    acc = {g: d["corners"] for g, d in groups.items() if d.get("accepted") and d.get("corners")}
    fit = [s for s in precise if s["source_group"] in FIT]
    inn = [s for s in precise if s["source_group"] in INNER]

    preds = {"pure": {}, "geom_v12": {}, "geom_v10": {}}
    geom_used = 0
    for vid in dev:
        rows = load(RAW / f"{vid}.jsonl")
        rows.sort(key=lambda r: r["input_frame_id"])
        pure = ER.detect_events(rows)
        preds["pure"][vid] = pure
        q = acc.get(vid.split("_")[0])
        if q:
            geom_used += 1
            table = {"corners": q, "net": None}
            preds["geom_v12"][vid] = ER.detect_events(rows, table=table)
            preds["geom_v10"][vid] = ER.detect_events(rows, config={"range_frac": 0}, table=table)
        else:
            preds["geom_v12"][vid] = pure
            preds["geom_v10"][vid] = pure
    for name, d in preds.items():
        (OUT / f"predictions_{name}.json").write_text(
            json.dumps(d, ensure_ascii=False), encoding="utf-8")

    print(f"开发池 {len(dev)} 段；有自动组角点 {geom_used} 段（accepted 组 {sorted(acc)}）")
    out = {}
    for name in ("pure", "geom_v10", "geom_v12"):
        r = {}
        for tag, ss in (("all42", precise), ("fit", fit), ("inner", inn)):
            r[tag] = {
                "trusted": TD.scoped_score(preds[name], ss, "trusted"),
                "complete": TD.scoped_score(preds[name], ss, "complete"),
            }
        out[name] = r
        def fmt(x):
            return f"{x['tp']}/{x['fp']}/{x['fn']}={x['f1']:.3f}"
        print(f"{name:9s} | 42可信 {fmt(r['all42']['trusted'])} | 42完整 {fmt(r['all42']['complete'])} "
              f"| fit完整 {fmt(r['fit']['complete'])} | inner可信 {fmt(r['inner']['trusted'])} "
              f"| inner完整 {fmt(r['inner']['complete'])}")
    print("\n防误删（625 正例，123 开发段）：")
    for old, new in (("pure", "geom_v10"), ("pure", "geom_v12"), ("geom_v10", "geom_v12")):
        p = TD.preserve(preds[old], preds[new], positives)
        print(f"  {old} -> {new}: 基线命中 {p['baseline_matched']}, 保留 {p['retained']}, "
              f"丢失 {len(p['lost'])} {'PASS' if p['passed'] else 'FAIL: ' + str(p['lost'][:3])}")
        out.setdefault("preservation", {})[f"{old}->{new}"] = {
            "baseline_matched": p["baseline_matched"], "retained": p["retained"],
            "lost": p["lost"], "passed": p["passed"]}
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    print(f"\n预测与摘要已写入 {OUT}")


if __name__ == "__main__":
    main()

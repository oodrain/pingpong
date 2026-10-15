"""Replay repaired D rules on cached detections and publish reproducible evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from development.bounce.ge_rule_v2 import RuleConfig, replay
from development.bounce.table_geometry import GeometryTimeline
from development.evaluation.evaluate_landings import evaluate, validate_input

PUBLIC = PROJECT / "participant/participant/pingpang/public_data"
TRACKS = PROJECT / "outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1"
MODES = ("v0", "v1a", "v1b")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, data):
    path.write_text("".join(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n" for d in data), encoding="utf-8")


def score(predictions, gt, manifest):
    return {"without_ignore_exemption": evaluate(predictions, gt, manifest, exempt_invalid=False),
            "with_ignore_exemption": evaluate(predictions, gt, manifest, exempt_invalid=True)}


def changed_matches(baseline, other, mode):
    def key(g):
        return g["video_id"], g["frame_id"], g["x"], g["y"]
    old = {key(m["gt"]): m for v in baseline["by_video"] for m in v["matches"]}
    new = {key(m["gt"]): m for v in other["by_video"] for m in v["matches"]}
    return [dict(mode=mode, change=change, **values[k])
            for keys, values, change in ((old.keys() - new.keys(), old, "lost_tp"),
                                        (new.keys() - old.keys(), new, "gained_tp"))
            for k in sorted(keys)]


def margin_analysis(cached, gt, manifest, config, margins):
    scans = {}
    for margin in margins:
        predictions = []
        for vid, (raw, timeline) in cached.items():
            pred, _ = replay(raw, timeline, "v1a", replace(config, margin_mm=margin))
            predictions.extend(dict(p, video_id=vid) for p in pred)
        scans[str(margin)] = score(predictions, gt, manifest)
    # Select by the same angle-macro objective that is reported, within each fold.
    folds = []
    for held_out in manifest:
        training = [m for m in manifest if m["video_id"] != held_out["video_id"]]
        if not training:
            continue
        def train_macro(margin):
            data = scans[str(margin)]["without_ignore_exemption"]["by_video"]
            by_angle = {}
            for m in training:
                row = next(r for r in data if r["video_id"] == m["video_id"])
                counts = by_angle.setdefault(m["angle"], [0, 0, 0])
                for i, k in enumerate(("tp", "fp", "fn")):
                    counts[i] += row[k]
            return sum(2*t/(2*t+p+n) if 2*t+p+n else 0.0
                       for t,p,n in by_angle.values()) / len(by_angle)
        # On equal scores select the more conservative (larger) region.
        chosen = max(margins, key=lambda m: (train_macro(m), m))
        held_row = next(r for r in scans[str(chosen)]["without_ignore_exemption"]["by_video"]
                        if r["video_id"] == held_out["video_id"])
        folds.append({"held_out": held_out["video_id"], "angle": held_out["angle"],
                      "chosen_margin_mm": chosen, **{k: held_row[k] for k in ("tp", "fp", "fn", "f1")}})
    held_angles = {}
    for row in folds:
        counts = held_angles.setdefault(row["angle"], [0, 0, 0])
        for i, k in enumerate(("tp", "fp", "fn")):
            counts[i] += row[k]
    held_macro = (sum(2*t/(2*t+p+n) if 2*t+p+n else 0.0 for t,p,n in held_angles.values())
                  / len(held_angles)) if held_angles else None
    return {"objective": "angle_macro_f1_without_ignore", "scans": scans, "leave_one_out": folds,
            "leave_one_out_angle_macro_f1": held_macro,
            "selected_margin_mm": max(margins, key=lambda m: (
                scans[str(m)]["without_ignore_exemption"]["angle_macro_f1"], m))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=PROJECT, help="Shared input workspace, separate from this code checkout")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--geometry-dir", type=Path, help="One VIDEO_ID.json geometry document per video")
    ap.add_argument("--baseline-predictions", type=Path, help="Optional exact V0 replay comparison")
    ap.add_argument("--margin-mm", type=float, default=900.0, help="Experiment parameter, not a validated default")
    ap.add_argument("--margins", type=float, nargs="+", default=[400, 600, 800, 900, 1000, 1200])
    ap.add_argument("--require-geometry", action="store_true", help="Fail unless all V1 candidates have usable geometry")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    public = args.data_root / "participant/participant/pingpang/public_data"
    tracks = args.data_root / "outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1"
    args.raw_dir = args.raw_dir or tracks / "raw"
    args.manifest = args.manifest or public / "manifest.json"
    args.gt = args.gt or public / "gt_public.jsonl"
    config = RuleConfig(margin_mm=args.margin_mm)
    margins = sorted(set(args.margins))
    if not margins or any(m < 0 or not math.isfinite(m) for m in margins):
        ap.error("Margins must be finite and nonnegative")
    manifest, gt = read(args.manifest)["videos"], rows(args.gt)
    metadata = {m["video_id"]: m for m in manifest}
    if not manifest or len(metadata) != len(manifest):
        raise ValueError("Empty manifest or duplicate video IDs")
    validate_input(gt, metadata, ground_truth=True)
    predictions = {mode: [] for mode in MODES}
    decisions = {mode: [] for mode in MODES}
    cached, issues, missing_geometry = {}, {}, []
    inputs = [args.manifest, args.gt]
    for m in manifest:
        vid = m["video_id"]
        path = args.raw_dir / (vid + ".jsonl")
        raw = rows(path)
        if len(raw) != m["n_frames"] or any(r["input_frame_id"] != i or
                r.get("video_id", vid) != vid for i, r in enumerate(raw)):
            raise ValueError(f"{vid}: raw cache does not cover the manifest frames exactly")
        inputs.append(path)
        document = None
        if args.geometry_dir is not None:
            geometry_path = args.geometry_dir / (vid + ".json")
            if geometry_path.is_file():
                document = read(geometry_path)
                if document.get("video_id", vid) != vid:
                    raise ValueError(f"{vid}: geometry video_id mismatch")
                inputs.append(geometry_path)
        if document is None:
            missing_geometry.append(vid)
        timeline = GeometryTimeline(document)
        if timeline.issues:
            issues[vid] = timeline.issues
        cached[vid] = raw, timeline
        for mode in MODES:
            pred, diagnostics = replay(raw, timeline, mode, config)
            predictions[mode].extend(dict(p, video_id=vid) for p in pred)
            decisions[mode].extend(dict(d, video_id=vid) for d in diagnostics)
    for pred in predictions.values():
        validate_input(pred, metadata)
    comparison = None
    if args.baseline_predictions:
        expected = rows(args.baseline_predictions)
        inputs.append(args.baseline_predictions)
        canonical = lambda data: sorted(data, key=lambda p: (p["video_id"], p["frame_id"], p["x"], p["y"]))
        fields = lambda data: [{k: p[k] for k in ("video_id", "frame_id", "x", "y")} for p in data]
        comparison = canonical(fields(expected)) == canonical(predictions["v0"])
        if not comparison:
            raise ValueError("Repaired V0 replay differs from supplied baseline")
    known = sum(d["geometry_available"] for d in decisions["v1a"])
    total = len(decisions["v1a"])
    if args.require_geometry and (missing_geometry or issues or known != total):
        raise ValueError("Geometry incomplete; see missing files, invalid segments or uncovered candidate frames")
    scores = {mode: score(predictions[mode], gt, manifest) for mode in MODES}
    scans = margin_analysis(cached, gt, manifest, config, margins) if known else {
        "status": "not_run_no_usable_geometry", "selected_margin_mm": None}
    changed = [d for mode in ("v1a", "v1b") for d in changed_matches(
        scores["v0"]["without_ignore_exemption"], scores[mode]["without_ignore_exemption"], mode)]
    status = "fallback_verified_geometry_pending" if not known else "partial_geometry_experiment" if known != total or issues or missing_geometry else "geometry_experiment_complete"
    output = args.output or PROJECT / "outputs/experiments/d_geometry_v2/runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), PROJECT / "development/bounce/ge_rule_v2.py",
               PROJECT / "development/bounce/table_geometry.py", PROJECT / "development/evaluation/evaluate_landings.py"]
    (output / "source").mkdir()
    for path in sources:
        shutil.copy2(path, output / "source" / path.name)
    summary = {"status": status, "config": asdict(config), "videos": len(manifest),
               "candidate_geometry": {"known": known, "total": total}, "missing_geometry_videos": missing_geometry,
               "geometry_issues": issues, "v0_matches_baseline": comparison,
               "reason_counts": {mode: dict(Counter(d["reason"] for d in ds)) for mode, ds in decisions.items()},
               "scorer": "project maximum matching; valid GT first; +/-1 frame, 50px; both ignore policies; unofficial estimate",
               "adopted": False, "margin_selection": scans.get("selected_margin_mm"),
               "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
               "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    write(output / "summary.json", summary)
    write(output / "metrics.json", scores)
    write(output / "margin_analysis.json", scans)
    write_rows(output / "changed_matches.jsonl", changed)
    for mode in MODES:
        write_rows(output / (mode + "_predictions.jsonl"), predictions[mode])
        write_rows(output / (mode + "_decisions.jsonl"), decisions[mode])
    lines = ["D几何规则修正版实验", "状态：" + status, f"视频数：{len(manifest)}；有几何的候选：{known}/{total}",
             "评分：项目最大匹配、有效GT优先；非赛事正式成绩；尚未接入生产规则。",
             "900mm仅为待重验证实验参数；平面投影无法恢复球的真实高度。", "",
             "版本 | TP/FP/FN | 汇总F1 | 三视角平均F1（不豁免干扰点预测）"]
    for mode in MODES:
        s = scores[mode]["without_ignore_exemption"]
        o = s["overall"]
        lines.append(f"{mode} | {o['tp']}/{o['fp']}/{o['fn']} | {o['f1']:.6f} | {s['angle_macro_f1']:.6f}")
    if not known:
        lines += ["", "缺少可用桌面几何，本轮验证回退到V0；不能作为几何收益或边距选型证据。"]
    lines += ["", "补齐几何后，以 --geometry-dir 重跑；严格验收加 --require-geometry。",
              "输入及代码哈希在summary.json，逐事件决策与匹配变化均已输出。"]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": status, "geometry_candidates": known,
                      "v0_matches_baseline": comparison,
                      "metrics_without_ignore": {mode: scores[mode]["without_ignore_exemption"]["overall"] for mode in MODES}},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

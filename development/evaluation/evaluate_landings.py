"""CPU-only public landing evaluation; independent of the protected framework.

Uses maximum-cardinality bipartite matching (not greedy matching). Valid GT has
priority. Unmatched predictions near invalid GT can be exempted from FP using
the original +/-1 frame, 50 px window. The released package does not specify the
exact ignore policy, so both with-ignore and without-exemption scores are saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "participant/participant/pingpang/public_data"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def within(a, b, frames=1, pixels=50):
    return abs(a["frame_id"] - b["frame_id"]) <= frames and distance(a, b) <= pixels


def maximum_matching(predictions, gt, frames=1, pixels=50):
    adjacency = [sorted(
        (j for j, g in enumerate(gt) if within(p, g, frames, pixels)),
        key=lambda j: (abs(p["frame_id"] - gt[j]["frame_id"]), distance(p, gt[j]), j)
    ) for p in predictions]
    owner = {}

    def augment(i, visited):
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or augment(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return sorted((i, j) for j, i in owner.items())


def metrics(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def score_video(predictions, valid_gt, invalid_gt, frames=1, pixels=50,
                exempt_invalid=True):
    pairs = maximum_matching(predictions, valid_gt, frames, pixels)
    used_p, used_g = {i for i, _ in pairs}, {j for _, j in pairs}
    # Fixed ignore window prevents sensitivity runs from enlarging ignore regions.
    ignored = [i for i, p in enumerate(predictions)
               if exempt_invalid and i not in used_p
               and any(within(p, g, 1, 50) for g in invalid_gt)]
    ignored_set = set(ignored)
    fp_indices = [i for i in range(len(predictions))
                  if i not in used_p and i not in ignored_set]
    fn_indices = [j for j in range(len(valid_gt)) if j not in used_g]
    return {**metrics(len(pairs), len(fp_indices), len(fn_indices)),
            "predictions": len(predictions), "valid_gt": len(valid_gt),
            "invalid_gt": len(invalid_gt), "ignored_predictions": len(ignored),
            "matches": [{"prediction": predictions[i], "gt": valid_gt[j],
                         "frame_error": predictions[i]["frame_id"] - valid_gt[j]["frame_id"],
                         "distance_px": distance(predictions[i], valid_gt[j])}
                        for i, j in pairs],
            "ignored": [predictions[i] for i in ignored],
            "false_positives": [predictions[i] for i in fp_indices],
            "false_negatives": [valid_gt[j] for j in fn_indices]}


def summarize(rows):
    counts = {k: sum(row[k] for row in rows)
              for k in ("tp", "fp", "fn", "predictions", "valid_gt", "invalid_gt",
                        "ignored_predictions")}
    return {**counts, **metrics(counts["tp"], counts["fp"], counts["fn"])}


def validate_input(records, metadata, ground_truth=False):
    for index, r in enumerate(records, 1):
        if r.get("video_id") not in metadata:
            raise ValueError(f"Record {index}: unknown video_id")
        m = metadata[r["video_id"]]
        f = r.get("frame_id")
        if type(f) is not int or not 0 <= f < m["n_frames"]:
            raise ValueError(f"Record {index}: invalid frame_id")
        for key, limit in (("x", m["width"]), ("y", m["height"])):
            v = r.get(key)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < limit:
                raise ValueError(f"Record {index}: invalid {key}")
        if ground_truth and type(r.get("valid")) is not bool:
            raise ValueError(f"Record {index}: missing/invalid GT valid flag")


def evaluate(predictions, gt, manifest, frames=1, pixels=50, exempt_invalid=True):
    pred_by, gt_by = defaultdict(list), defaultdict(list)
    for p in predictions:
        pred_by[p["video_id"]].append(p)
    for g in gt:
        gt_by[g["video_id"]].append(g)
    rows = []
    for m in manifest:
        vid = m["video_id"]
        valid = [g for g in gt_by[vid] if g["valid"]]
        invalid = [g for g in gt_by[vid] if not g["valid"]]
        if "n_valid_landings" in m and len(valid) != m["n_valid_landings"]:
            raise ValueError(f"{vid}: manifest and valid GT count disagree")
        rows.append({"video_id": vid, "angle": m["angle"],
                     **score_video(pred_by[vid], valid, invalid, frames, pixels, exempt_invalid)})
    angles = {angle: summarize([r for r in rows if r["angle"] == angle])
              for angle in dict.fromkeys(m["angle"] for m in manifest)}
    return {"by_video": rows, "by_angle": angles, "overall": summarize(rows),
            "angle_macro_f1": sum(a["f1"] for a in angles.values()) / len(angles),
            "accuracy_score_estimate": sum(a["f1"] * 25 for a in angles.values())}


def nearest(point, candidates, point_is_gt=False):
    if not candidates:
        return {}
    other = min(candidates, key=lambda g: (
        abs(point["frame_id"] - g["frame_id"]) / 1 + distance(point, g) / 50))
    return {"nearest_frame_id": other["frame_id"], "nearest_x": other["x"],
            "nearest_y": other["y"],
            "signed_frame_error": (other["frame_id"] - point["frame_id"] if point_is_gt
                                   else point["frame_id"] - other["frame_id"]),
            "distance_px": distance(point, other)}


def write_artifacts(output, report, predictions, gt):
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    result = report["with_ignore_exemption"]
    fields = ["video_id", "angle", "predictions", "valid_gt", "invalid_gt", "tp", "fp", "fn",
              "ignored_predictions", "precision", "recall", "f1"]
    with (output / "per_video.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["by_video"]:
            writer.writerow({k: row[k] for k in fields})
    fields = ["video_id", "angle", "kind", "frame_id", "x", "y", "nearest_frame_id",
              "nearest_x", "nearest_y", "signed_frame_error", "distance_px"]
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["by_video"]:
            vid = row["video_id"]
            for kind, points, candidates in (
                ("FP", row["false_positives"], [g for g in gt if g["video_id"] == vid and g["valid"]]),
                ("FN", row["false_negatives"], [p for p in predictions if p["video_id"] == vid])):
                for p in points:
                    writer.writerow({"video_id": vid, "angle": row["angle"], "kind": kind,
                                     "frame_id": p["frame_id"], "x": p["x"], "y": p["y"],
                                     **nearest(p, candidates, point_is_gt=(kind == "FN"))})

    lines = ["# 官方乒乓球基线：公开验证集本地精度评估", "",
             "输入为线上推理结果；所有评估在 CPU 上完成，未修改官方工程。", "",
             "## 评估口径", "",
             "正常 GT 优先，在帧差 ≤1、欧氏距离 ≤50 px 的候选边中做最大数量一对一匹配。",
             "GT valid=false 不计正常落点；未匹配预测落入忽略点相同容差窗口时免计 FP。",
             "包内没有正式评分器，忽略窗口/冲突处理未获官方确认，因此同时报告不豁免预测的结果。",
             "所有分数仅是公开集自测估算；容差放宽实验用于诊断，不能作为参赛成绩。", "",
             "## 按视角汇总（忽略点命中预测豁免）", "",
             "| 视角 | 有效GT | 预测 | TP | FP | FN | 忽略预测 | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for angle, r in [*result["by_angle"].items(), ("整体（micro）", result["overall"])]:
        lines.append(f"| {angle} | {r['valid_gt']} | {r['predictions']} | {r['tp']} | {r['fp']} | "
                     f"{r['fn']} | {r['ignored_predictions']} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", "精度分按三个视角 F1 分别乘 25 后求和，不能直接使用整体 micro F1 ×75。", "",
              "## 口径敏感性与性能", "",
              "| 口径 | 整体 F1 | 视角平均 F1 | 精度分估算 /75 | 总分估算 /100 |",
              "|---|---:|---:|---:|---:|"]
    for key, label in (("with_ignore_exemption", "忽略点命中预测豁免"),
                       ("without_ignore_exemption", "不豁免未匹配预测")):
        r = report[key]
        total = r["accuracy_score_estimate"] + report["speed_score_estimate"]
        lines.append(f"| {label} | {r['overall']['f1']:.4f} | {r['angle_macro_f1']:.4f} | "
                     f"{r['accuracy_score_estimate']:.2f} | {total:.2f} |")
    s = report["run_status"]
    lines += ["", f"- 运行状态：{s['status']}；总帧数：{s['totals']['frames']}；预测数：{s['totals']['predictions']}。",
              f"- 推理 FPS：{s['performance']['infer_fps']}；逐视频 FPS 算术均值：{s['performance']['avg_video_fps']}。",
              f"- prepare：{s['timing']['prepare_sec']} s；infer：{s['timing']['infer_sec']} s；total：{s['timing']['total_sec']} s。",
              f"- 解码合计：{sum(v['decode_sec'] for v in s['videos']):.2f} s；算法合计：{sum(v['solution_sec'] for v in s['videos']):.2f} s。",
              f"- 本次 engine 构建耗时：{s['timing']['engine_build_sec']} s；后端：{s['solution_info']['backend']}。缓存命中时未计入此前首次构建的耗时。",
              "- infer_sec 包含解码、算法、写出及调度开销，不能当作纯神经网络耗时。", "",
              "## 各视频结果", "",
              "| 视频 | 视角 | TP | FP | FN | 忽略预测 | F1 | FP帧号 | FN帧号 |",
              "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in result["by_video"]:
        fps = ", ".join(str(p["frame_id"]) for p in r["false_positives"]) or "—"
        fns = ", ".join(str(g["frame_id"]) for g in r["false_negatives"]) or "—"
        lines.append(f"| {r['video_id']} | {r['angle']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{r['ignored_predictions']} | {r['f1']:.4f} | {fps} | {fns} |")
    lines += ["", "## 容差诊断（忽略窗口固定，非正式成绩）", "",
              "| 时间容差/帧 | 空间容差/px | TP | FP | FN | 整体 F1 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in report["tolerance_diagnostics"]:
        m = r["overall"]
        lines.append(f"| {r['frames']} | {r['pixels']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['f1']:.4f} |")
    base_tp = result["overall"]["tp"]
    time_gain = next(r["overall"]["tp"] for r in report["tolerance_diagnostics"]
                     if r["frames"] == 5 and r["pixels"] == 50) - base_tp
    space_gain = next(r["overall"]["tp"] for r in report["tolerance_diagnostics"]
                      if r["frames"] == 1 and r["pixels"] == 100) - base_tp
    lines += ["", "诊断仅判断扩大哪种容差能增加匹配，不证明误差来自检测器还是落点判定器。",
              "errors.csv 中最近点按 |帧差|/1 + 距离/50 排序，只用于排查，不用于评分匹配；signed_frame_error 始终为预测帧减 GT 帧。", "",
              f"本批结果时间容差放宽到 ±5 帧多匹配 {time_gain} 个点；空间容差放宽到 100 px 多匹配 {space_gain} 个点。",
              "上述增量帮助判断容差敏感性；仅有预测落点，尚不能确定是球检测、",
              "轨迹断裂还是触台判定造成，需要逐帧轨迹或视频叠加。", "",
              "## 下一步", "",
              "优先分析 0° 的误检和漏检。先在视频中叠加球轨迹、GT 和预测，确认是否为时间偏移、",
              "球位检测错误或非触台转折，再决定改轨迹与事件规则还是重训检测器。",
              "公开验证集只有 15 条，后续需从训练数据另设开发划分，避免只针对这 15 条调参。", "",
              "复现命令（在项目根目录）：", "", "```bash",
              "python development/evaluation/evaluate_landings.py", "```", ""]
    (output / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=ROOT / "outputs/baseline_run/predictions.jsonl")
    ap.add_argument("--gt", type=Path, default=PUBLIC / "gt_public.jsonl")
    ap.add_argument("--manifest", type=Path, default=PUBLIC / "manifest.json")
    ap.add_argument("--run-status", type=Path, default=ROOT / "outputs/baseline_run/run_status.json")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/baseline_evaluation")
    args = ap.parse_args()
    pred, gt = read_jsonl(args.predictions), read_jsonl(args.gt)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))["videos"]
    meta = {m["video_id"]: m for m in manifest}
    if len(meta) != len(manifest):
        raise ValueError("Duplicate manifest video_id")
    validate_input(pred, meta)
    validate_input(gt, meta, ground_truth=True)
    status = json.loads(args.run_status.read_text(encoding="utf-8-sig"))
    if status["status"] != "ok" or status["totals"]["predictions"] != len(pred):
        raise ValueError("Run status not ok or prediction count disagrees")
    if status["totals"]["frames"] != sum(m["n_frames"] for m in manifest):
        raise ValueError("Run frame count disagrees with manifest")
    run_videos = {v["video_id"]: v for v in status["videos"]}
    if set(run_videos) != set(meta):
        raise ValueError("Run videos disagree with manifest")
    for vid, m in meta.items():
        v = run_videos[vid]
        if v["status"] != "ok" or v["frames"] != m["n_frames"] or v["predictions"] != sum(p["video_id"] == vid for p in pred):
            raise ValueError(f"{vid}: run status disagrees with predictions/manifest")
    fps = status["performance"]["avg_video_fps"]
    speed = 25.0 if fps >= 50 else (0.0 if fps < 1 else 1 + (fps - 1) * 24 / 49)
    report = {
        "method": {"frames": 1, "pixels": 50, "matching": "maximum cardinality bipartite",
                   "ignore_policy": "valid GT first; unmatched predictions within +/-1 frame and 50 px of any invalid GT exempt",
                   "official_scorer_available": False},
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                         for key, path in (("predictions", args.predictions), ("gt", args.gt),
                                           ("manifest", args.manifest), ("run_status", args.run_status))},
        "run_status": status,
        "speed_score_estimate": speed,
        "with_ignore_exemption": evaluate(pred, gt, manifest),
        "without_ignore_exemption": evaluate(pred, gt, manifest, exempt_invalid=False),
        "tolerance_diagnostics": [],
    }
    for frames, pixels in ((1, 50), (2, 50), (3, 50), (5, 50), (1, 75), (1, 100), (5, 100)):
        result = evaluate(pred, gt, manifest, frames, pixels)
        report["tolerance_diagnostics"].append({"frames": frames, "pixels": pixels,
                                                "overall": result["overall"]})
    write_artifacts(args.output, report, pred, gt)
    print(json.dumps({"with_ignore": report["with_ignore_exemption"]["overall"],
                      "without_ignore": report["without_ignore_exemption"]["overall"],
                      "by_angle": report["with_ignore_exemption"]["by_angle"],
                      "accuracy_score_estimate": report["with_ignore_exemption"]["accuracy_score_estimate"],
                      "speed_score_estimate": speed,
                      "tolerance_diagnostics": report["tolerance_diagnostics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

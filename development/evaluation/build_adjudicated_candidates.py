#!/usr/bin/env python3
"""Build stage-level adjudicated label candidates from annotation rounds.

The source labels are read-only. Consensus uses one-to-one matching with the
official tolerances (frame +/-1, Euclidean position <=50px). Third-review
points replace nearby consensus points, otherwise they are added. Uncertain
and no-bounce decisions exclude only disputed source events, never unrelated
consensus events that happen to lie in the context window.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MEMBERS = "ABCD"
FRAME_TOLERANCE = 1
POSITION_TOLERANCE = 50.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def distance(left: dict, right: dict) -> float:
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))


def compatible(left: dict, right: dict) -> bool:
    return (
        abs(int(left["frame_id"]) - int(right["frame_id"])) <= FRAME_TOLERANCE
        and distance(left, right) <= POSITION_TOLERANCE
    )


def match_events(initial: list[dict], review: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Maximum-cardinality one-to-one matching with deterministic nearest-first edges."""
    edges: dict[int, list[int]] = {}
    for i, left in enumerate(initial):
        choices = [j for j, right in enumerate(review) if compatible(left, right)]
        edges[i] = sorted(
            choices,
            key=lambda j: (
                abs(int(left["frame_id"]) - int(review[j]["frame_id"])),
                distance(left, review[j]),
                int(review[j]["frame_id"]),
                j,
            ),
        )

    right_to_left: dict[int, int] = {}

    def place(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in right_to_left or place(right_to_left[j], seen):
                right_to_left[j] = i
                return True
        return False

    for i in sorted(range(len(initial)), key=lambda idx: (len(edges[idx]), int(initial[idx]["frame_id"]), idx)):
        place(i, set())

    pairs = sorted(((i, j) for j, i in right_to_left.items()), key=lambda pair: (pair[0], pair[1]))
    used_i = {i for i, _ in pairs}
    used_j = {j for _, j in pairs}
    return pairs, [i for i in range(len(initial)) if i not in used_i], [j for j in range(len(review)) if j not in used_j]


def normalized_third_events(record: dict) -> list[dict]:
    if record.get("decision") != "confirmed":
        return []
    raw = record.get("final_events")
    if isinstance(raw, list):
        events = raw
    elif all(record.get(key) is not None for key in ("final_frame_id", "final_x", "final_y")):
        events = [{"frame_id": record["final_frame_id"], "x": record["final_x"], "y": record["final_y"]}]
    else:
        events = []
    return sorted(
        [
            {"frame_id": int(event["frame_id"]), "x": round(float(event["x"]), 3), "y": round(float(event["y"]), 3)}
            for event in events
        ],
        key=lambda event: (event["frame_id"], event["x"], event["y"]),
    )


def locate_videos(labels: Path) -> dict[str, dict]:
    videos: dict[str, dict] = {}
    for owner_dir in sorted(path for path in labels.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in owner_dir.iterdir() if path.is_dir()):
            initial_dir, review_dir = video_dir / "initial", video_dir / "review"
            if not (initial_dir / "events.jsonl").is_file() or not (review_dir / "events.jsonl").is_file():
                continue
            video_id = video_dir.name
            if video_id in videos:
                raise ValueError(f"duplicate canonical label package for {video_id}")
            initial_meta = read_json(initial_dir / "metadata.json")
            review_meta = read_json(review_dir / "metadata.json")
            videos[video_id] = {
                "owner": owner_dir.name,
                "initial_meta": initial_meta,
                "review_meta": review_meta,
                "initial": read_jsonl(initial_dir / "events.jsonl"),
                "review": read_jsonl(review_dir / "events.jsonl"),
            }
    return videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="repository containing labels and adjudication_package")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    labels = repo / "labels"
    package = repo / "experiments" / "manual_label_audit" / "adjudication_package"
    template_rows = read_jsonl(package / "decision_template.jsonl")
    cases = {row["case_id"]: row for row in template_rows}
    if len(cases) != len(template_rows):
        raise ValueError("duplicate case_id in decision_template.jsonl")

    decisions: dict[str, dict] = {}
    member_counts: dict[str, int] = {}
    for member in MEMBERS:
        rows = read_jsonl(package / "third_review" / f"{member}.jsonl")
        member_counts[member] = len(rows)
        for row in rows:
            case_id = row.get("case_id")
            if case_id in decisions:
                raise ValueError(f"duplicate third-review decision: {case_id}")
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"unknown third-review case: {case_id}")
            if row.get("third_reviewer") != member:
                raise ValueError(f"{case_id}: third_reviewer does not match {member}.jsonl")
            if member in {case["initial_annotator"], case["reviewer"]}:
                raise ValueError(f"{case_id}: third reviewer participated in an earlier round")
            if row.get("completed") is not True:
                raise ValueError(f"{case_id}: decision is not completed")
            if row.get("decision") not in {"confirmed", "no_bounce", "uncertain"}:
                raise ValueError(f"{case_id}: invalid decision")
            if row.get("decision") == "uncertain" and not str(row.get("notes", "")).strip():
                raise ValueError(f"{case_id}: uncertain decision has no reason")
            decisions[case_id] = row
    if set(decisions) != set(cases):
        missing = sorted(set(cases) - set(decisions))
        extra = sorted(set(decisions) - set(cases))
        raise ValueError(f"third-review coverage mismatch; missing={missing}, extra={extra}")

    videos = locate_videos(labels)
    if len(videos) != 154:
        raise ValueError(f"expected 154 canonical videos, found {len(videos)}")

    consensus: dict[str, list[dict]] = defaultdict(list)
    unmatched_initial_ids: set[str] = set()
    unmatched_review_ids: set[str] = set()
    match_distances: list[float] = []
    for video_id, source in sorted(videos.items()):
        initial = source["initial"]
        review = source["review"]
        pairs, initial_only, review_only = match_events(initial, review)
        for i, j in pairs:
            left, right = initial[i], review[j]
            match_distances.append(distance(left, right))
            consensus[video_id].append(
                {
                    "video_id": video_id,
                    "frame_id": int(right["frame_id"]),
                    "x": round(float(right["x"]), 3),
                    "y": round(float(right["y"]), 3),
                    "event_type": "table_bounce",
                    "review_state": "confirmed",
                    "round": "adjudicated_candidate_v1",
                    "source": "initial_review_agreement",
                    "initial_event_id": left.get("event_id"),
                    "review_event_id": right.get("event_id"),
                    "initial_frame_id": int(left["frame_id"]),
                    "review_frame_id": int(right["frame_id"]),
                    "pixel_distance": round(distance(left, right), 4),
                }
            )
        unmatched_initial_ids.update(str(initial[i].get("event_id")) for i in initial_only)
        unmatched_review_ids.update(str(review[j].get("event_id")) for j in review_only)

    template_initial_ids = {
        str(event.get("event_id")) for case in template_rows for event in case.get("initial_only_events", [])
    }
    template_review_ids = {
        str(event.get("event_id")) for case in template_rows for event in case.get("review_only_events", [])
    }
    if unmatched_initial_ids != template_initial_ids or unmatched_review_ids != template_review_ids:
        raise ValueError(
            "decision template no longer matches source labels: "
            f"initial unmatched={len(unmatched_initial_ids)}/{len(template_initial_ids)}, "
            f"review unmatched={len(unmatched_review_ids)}/{len(template_review_ids)}"
        )

    final_by_video = {video_id: list(consensus.get(video_id, [])) for video_id in videos}
    replacements = 0
    third_point_count = 0
    uncertain_rows: list[dict] = []
    no_bounce_rows: list[dict] = []
    decision_counts = Counter()

    for case in template_rows:
        case_id = case["case_id"]
        record = decisions[case_id]
        decision = record["decision"]
        decision_counts[decision] += 1
        audit_row = {
            "case_id": case_id,
            "video_id": case["video_id"],
            "frame_start": int(case["frame_start"]),
            "frame_end": int(case["frame_end"]),
            "initial_annotator": case["initial_annotator"],
            "reviewer": case["reviewer"],
            "third_reviewer": record["third_reviewer"],
            "decision": decision,
            "notes": record.get("notes", ""),
            "initial_only_events": case.get("initial_only_events", []),
            "review_only_events": case.get("review_only_events", []),
        }
        if decision == "uncertain":
            uncertain_rows.append(audit_row)
            continue
        if decision == "no_bounce":
            no_bounce_rows.append(audit_row)
            continue

        events = normalized_third_events(record)
        if not events:
            raise ValueError(f"{case_id}: confirmed decision has no points")
        meta = videos[case["video_id"]]["initial_meta"]
        for event in events:
            if not int(case["frame_start"]) <= event["frame_id"] <= int(case["frame_end"]):
                raise ValueError(f"{case_id}: frame {event['frame_id']} is outside case range")
            if not (0 <= event["x"] < float(meta["width"]) and 0 <= event["y"] < float(meta["height"])):
                raise ValueError(f"{case_id}: point is outside image bounds: {event}")

            current = final_by_video[case["video_id"]]
            candidates = [
                (idx, row)
                for idx, row in enumerate(current)
                if row.get("source") == "initial_review_agreement"
                and compatible(event, row)
            ]
            replaced = False
            if candidates:
                idx, _ = min(
                    candidates,
                    key=lambda item: (
                        abs(event["frame_id"] - int(item[1]["frame_id"])),
                        distance(event, item[1]),
                        item[0],
                    ),
                )
                del current[idx]
                replacements += 1
                replaced = True
            current.append(
                {
                    "video_id": case["video_id"],
                    **event,
                    "event_type": "table_bounce",
                    "review_state": "confirmed",
                    "round": "adjudicated_candidate_v1",
                    "source": "third_review",
                    "case_id": case_id,
                    "third_reviewer": record["third_reviewer"],
                    "replaces_consensus": replaced,
                }
            )
            third_point_count += 1

    duplicate_pairs: list[dict] = []
    final_rows: list[dict] = []
    grouped_rows: list[dict] = []
    for video_id in sorted(videos):
        events = sorted(
            final_by_video.get(video_id, []),
            key=lambda event: (int(event["frame_id"]), float(event["x"]), float(event["y"]), event["source"]),
        )
        for i, left in enumerate(events):
            for right in events[i + 1 :]:
                if int(right["frame_id"]) - int(left["frame_id"]) > FRAME_TOLERANCE:
                    break
                if compatible(left, right):
                    duplicate_pairs.append(
                        {
                            "video_id": video_id,
                            "left": {key: left.get(key) for key in ("frame_id", "x", "y", "source", "case_id")},
                            "right": {key: right.get(key) for key in ("frame_id", "x", "y", "source", "case_id")},
                            "pixel_distance": round(distance(left, right), 4),
                        }
                    )
        clean_events = []
        for index, event in enumerate(events, 1):
            row = {"event_id": f"{video_id}_candidate_{index:04d}", **event}
            clean_events.append(row)
            final_rows.append(row)
        source = videos[video_id]
        grouped_rows.append(
            {
                "video_id": video_id,
                "owner": source["owner"],
                "width": int(source["initial_meta"]["width"]),
                "height": int(source["initial_meta"]["height"]),
                "n_frames": int(source["initial_meta"]["n_frames"]),
                "events": [
                    {key: event[key] for key in ("event_id", "frame_id", "x", "y", "event_type", "source")}
                    for event in clean_events
                ],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_events.jsonl", final_rows)
    write_jsonl(output / "candidate_by_video.jsonl", grouped_rows)
    write_jsonl(output / "excluded_uncertain.jsonl", uncertain_rows)
    write_jsonl(output / "rejected_no_bounce.jsonl", no_bounce_rows)
    write_jsonl(output / "duplicate_candidates.jsonl", duplicate_pairs)

    source_counts = Counter(row["source"] for row in final_rows)
    report = {
        "status": "pass" if not duplicate_pairs else "needs_duplicate_review",
        "policy": {
            "agreement": "one-to-one match with frame +/-1 and Euclidean distance <=50px; review point represents agreement",
            "third_review": "third-review point replaces a compatible consensus point in the same video, otherwise it is added",
            "uncertain": "disputed source events excluded; unrelated consensus context retained",
            "no_bounce": "disputed source events excluded",
        },
        "videos": len(videos),
        "member_cases": member_counts,
        "decision_cases": dict(decision_counts),
        "initial_review_consensus_before_replacement": sum(len(rows) for rows in consensus.values()),
        "third_review_points": third_point_count,
        "third_points_replacing_consensus": replacements,
        "candidate_events": len(final_rows),
        "candidate_source_counts": dict(source_counts),
        "excluded_uncertain_cases": len(uncertain_rows),
        "rejected_no_bounce_cases": len(no_bounce_rows),
        "duplicate_candidates": len(duplicate_pairs),
        "max_consensus_pixel_distance": round(max(match_distances), 4) if match_distances else None,
        "source_unmatched_initial_events": len(unmatched_initial_ids),
        "source_unmatched_review_events": len(unmatched_review_ids),
    }
    (output / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 154条视频裁决汇总候选标签 v1\n\n"
        "本目录是 B 本阶段的数据整理交付物，由初标、独立复核和第三复核非破坏性汇总生成；"
        "原始 `labels/` 不修改。它是后续数据划分、抽查、训练和统一评估的候选输入，"
        "不等同于官方 GT，也不表示项目最终标签已经冻结。\n\n"
        "- `candidate_events.jsonl`：逐触台候选点总表。\n"
        "- `candidate_by_video.jsonl`：154条视频分组结果；固定训练/开发划分后按对应子集读取。\n"
        "- `excluded_uncertain.jsonl`：第三复核仍为纯疑难的条目，本版候选标签不纳入。\n"
        "- `rejected_no_bounce.jsonl`：第三复核确认非触台的条目。\n"
        "- `duplicate_candidates.jsonl`：同一容差窗口内的重复候选检查结果；为空仅表示未发现此类重复，"
        "不代表标签已完成最终验收。\n"
        "- `merge_report.json`：数量、汇总口径和完整性报告。\n\n"
        "汇总匹配阈值沿用评测容差：帧号前后1帧、欧氏位置距离50像素，并执行一对一匹配。"
        "该阈值用于形成阶段候选结果，不是对真实触台正确性的额外证明。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not duplicate_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())

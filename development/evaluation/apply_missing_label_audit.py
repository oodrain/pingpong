#!/usr/bin/env python3
"""Apply B's suspected-missing-label audit to candidate labels v1.

The audit UI records independent actions.  This merger therefore treats
candidate_removed, remove_event_ids, and final_events as authoritative rather
than trying to infer the final label set from the summary decision string.
Public-15 cases are retained for audit only and never modify training labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FRAME_TOLERANCE = 1
POSITION_TOLERANCE = 50.0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def compatible(left: dict, right: dict) -> bool:
    return (
        abs(int(left["frame_id"]) - int(right["frame_id"])) <= FRAME_TOLERANCE
        and math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
        <= POSITION_TOLERANCE
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--audit-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v1 = args.v1.resolve()
    audit = args.audit_package.resolve()
    output = args.output.resolve()

    base_events = read_jsonl(v1 / "candidate_events.jsonl")
    base_groups = read_jsonl(v1 / "candidate_by_video.jsonl")
    templates = read_jsonl(audit / "decision_template.jsonl")
    decisions = read_jsonl(audit / "third_review" / "B.jsonl")
    case_by_id = {str(row["case_id"]): row for row in templates}
    decision_by_id = {str(row["case_id"]): row for row in decisions}
    if len(case_by_id) != len(templates) or len(decision_by_id) != len(decisions):
        raise ValueError("duplicate case_id in audit input")
    if set(case_by_id) != set(decision_by_id):
        raise ValueError("audit decision coverage does not match decision template")
    if any(row.get("completed") is not True for row in decisions):
        raise ValueError("all audit decisions must be completed")

    meta_by_video = {str(row["video_id"]): row for row in base_groups}
    events_by_video: dict[str, list[dict]] = defaultdict(list)
    base_ids: set[str] = set()
    for event in base_events:
        event_id = str(event["event_id"])
        if event_id in base_ids:
            raise ValueError(f"duplicate v1 event_id: {event_id}")
        base_ids.add(event_id)
        events_by_video[str(event["video_id"])].append(dict(event))

    training_cases: list[dict] = []
    public_cases: list[dict] = []
    change_rows: list[dict] = []
    removed_ids: set[str] = set()
    added_rows: list[dict] = []
    action_counts = Counter()

    for case_id in sorted(case_by_id):
        case = case_by_id[case_id]
        record = decision_by_id[case_id]
        audit_row = {
            "case_id": case_id,
            "video_id": case["video_id"],
            "dataset": case.get("dataset"),
            "decision": record.get("decision"),
            "candidate_removed": bool(record.get("candidate_removed", False)),
            "remove_event_ids": list(record.get("remove_event_ids") or []),
            "final_events": list(record.get("final_events") or []),
            "completed_at": record.get("completed_at"),
        }
        if str(case["video_id"]).startswith("v") or case.get("dataset") == "公共15":
            public_cases.append(audit_row)
            continue

        training_cases.append(audit_row)
        video_id = str(case["video_id"])
        if video_id not in meta_by_video:
            raise ValueError(f"unknown training video in audit: {video_id}")
        meta = meta_by_video[video_id]

        case_removed: list[str] = []
        allowed_existing = {str(event.get("event_id")) for event in case.get("initial_only_events", [])}
        for event_id in record.get("remove_event_ids") or []:
            event_id = str(event_id)
            if event_id not in allowed_existing:
                raise ValueError(f"{case_id}: removal is not one of the displayed existing labels: {event_id}")
            if event_id not in base_ids:
                raise ValueError(f"{case_id}: v1 removal target does not exist: {event_id}")
            if event_id in removed_ids:
                raise ValueError(f"v1 event removed by more than one audit case: {event_id}")
            removed_ids.add(event_id)
            case_removed.append(event_id)
        if case_removed:
            action_counts["removed_existing"] += len(case_removed)

        case_added: list[dict] = []
        if not bool(record.get("candidate_removed", False)) and record.get("decision") != "already_labeled":
            candidate_events = case.get("review_only_events") or []
            if len(candidate_events) != 1:
                raise ValueError(f"{case_id}: expected exactly one D candidate")
            source = candidate_events[0]
            case_added.append({
                "video_id": video_id,
                "frame_id": int(source["frame_id"]),
                "x": round(float(source["x"]), 3),
                "y": round(float(source["y"]), 3),
                "event_type": "table_bounce",
                "review_state": "confirmed",
                "round": "adjudicated_candidate_v2",
                "source": "missing_label_audit_d_candidate",
                "case_id": case_id,
                "auditor": record.get("third_reviewer"),
                "source_event_id": source.get("event_id"),
            })
            action_counts["retained_d_candidate"] += 1
        elif bool(record.get("candidate_removed", False)):
            action_counts["removed_d_candidate"] += 1

        for index, event in enumerate(record.get("final_events") or [], 1):
            case_added.append({
                "video_id": video_id,
                "frame_id": int(event["frame_id"]),
                "x": round(float(event["x"]), 3),
                "y": round(float(event["y"]), 3),
                "event_type": "table_bounce",
                "review_state": "confirmed",
                "round": "adjudicated_candidate_v2",
                "source": "missing_label_audit_manual",
                "case_id": case_id,
                "auditor": record.get("third_reviewer"),
                "manual_index": index,
            })
            action_counts["manual_event"] += 1

        for event in case_added:
            if not int(case["frame_start"]) <= event["frame_id"] <= int(case["frame_end"]):
                raise ValueError(f"{case_id}: added frame outside audit segment: {event}")
            if not (0 <= event["frame_id"] < int(meta["n_frames"])):
                raise ValueError(f"{case_id}: added frame outside video: {event}")
            if not (0 <= event["x"] < float(meta["width"]) and 0 <= event["y"] < float(meta["height"])):
                raise ValueError(f"{case_id}: added point outside image bounds: {event}")
            added_rows.append(event)

        change_rows.append({**audit_row, "removed_v1_event_ids": case_removed, "added_events": case_added})

    final_by_video: dict[str, list[dict]] = defaultdict(list)
    for video_id, events in events_by_video.items():
        for event in events:
            if str(event["event_id"]) in removed_ids:
                continue
            updated = dict(event)
            updated["source_round"] = updated.get("round")
            updated["round"] = "adjudicated_candidate_v2"
            final_by_video[video_id].append(updated)

    new_id_counter = Counter()
    for event in added_rows:
        video_id = str(event["video_id"])
        new_id_counter[video_id] += 1
        added = dict(event)
        added["event_id"] = f"{video_id}_candidate_v2_add_{new_id_counter[video_id]:04d}"
        final_by_video[video_id].append(added)

    duplicates: list[dict] = []
    final_rows: list[dict] = []
    grouped_rows: list[dict] = []
    final_ids: set[str] = set()
    for video_id in sorted(meta_by_video):
        events = sorted(final_by_video.get(video_id, []), key=lambda row: (int(row["frame_id"]), float(row["x"]), float(row["y"]), str(row["event_id"])))
        for event in events:
            event_id = str(event["event_id"])
            if event_id in final_ids:
                raise ValueError(f"duplicate v2 event_id: {event_id}")
            final_ids.add(event_id)
        for index, left in enumerate(events):
            for right in events[index + 1:]:
                if int(right["frame_id"]) - int(left["frame_id"]) > FRAME_TOLERANCE:
                    break
                if compatible(left, right):
                    duplicates.append({
                        "video_id": video_id,
                        "left_event_id": left["event_id"],
                        "right_event_id": right["event_id"],
                        "frame_delta": abs(int(left["frame_id"]) - int(right["frame_id"])),
                        "pixel_distance": round(math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"])), 4),
                    })
        final_rows.extend(events)
        meta = meta_by_video[video_id]
        grouped_rows.append({
            "video_id": video_id,
            "owner": meta["owner"],
            "width": meta["width"],
            "height": meta["height"],
            "n_frames": meta["n_frames"],
            "events": [
                {key: event.get(key) for key in ("event_id", "frame_id", "x", "y", "event_type", "source")}
                for event in events
            ],
        })

    report = {
        "status": "pass" if not duplicates else "warning",
        "version": "adjudicated_candidate_v2",
        "policy": {
            "scope": "apply only the 33 training-154 audit cases; public-15 cases remain audit-only",
            "candidate_removed": "exclude the displayed D candidate",
            "remove_event_ids": "remove only the explicitly selected v1 events",
            "remaining_d_candidate": "add as confirmed audit candidate unless decision is already_labeled",
            "final_events": "add all manually saved centers independently",
        },
        "videos": len(grouped_rows),
        "v1_events": len(base_events),
        "training_audit_cases": len(training_cases),
        "public_audit_only_cases": len(public_cases),
        "removed_v1_events": len(removed_ids),
        "added_events": len(added_rows),
        "v2_events": len(final_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_pairs_within_official_tolerance": len(duplicates),
    }

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_events.jsonl", final_rows)
    write_jsonl(output / "candidate_by_video.jsonl", grouped_rows)
    write_jsonl(output / "audit_changes.jsonl", change_rows)
    write_jsonl(output / "public15_audit_only.jsonl", public_cases)
    write_jsonl(output / "duplicate_candidates.jsonl", duplicates)
    (output / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 154条视频裁决汇总候选标签 v2\n\n"
        "本目录在v1基础上应用B对D阶段一35条疑似漏标的人工核对结果。仅33条训练154记录会修改候选标签；"
        "公共15的2条仅写入`public15_audit_only.jsonl`，不会修改官方GT。v1保持不变。\n\n"
        "- `candidate_events.jsonl`：v2逐事件候选标签。\n"
        "- `candidate_by_video.jsonl`：按154条视频分组。\n"
        "- `audit_changes.jsonl`：33条训练核对的逐条增删记录。\n"
        "- `public15_audit_only.jsonl`：2条公共数据核对意见，仅审计。\n"
        "- `duplicate_candidates.jsonl`：按帧±1、位置50px检查的重复候选。\n"
        "- `merge_report.json`：合并口径和完整性统计。\n\n"
        "本目录仍是阶段候选标签，不冒充官方GT或冻结版最终标签。\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

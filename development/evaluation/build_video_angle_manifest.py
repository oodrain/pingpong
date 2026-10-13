#!/usr/bin/env python3
"""Expand visually verified source-group camera angles to all 154 videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


GROUP_ANGLES = {
    "00": "0", "01": "0", "02": "90", "03": "90", "04": "90",
    "05": "90", "06": "0", "07": "0", "08": "0", "09": "90",
    "10": "90", "11": "45", "12": "0", "13": "90", "14": "0",
    "15": "0", "16": "0", "17": "45", "18": "0", "19": "0",
    "20": "45", "21": "90", "22": "90", "23": "0", "24": "90",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument("--group-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))["tasks"]
    ids = [str(row["video_id"]) for row in tasks]
    if len(tasks) != 154 or len(set(ids)) != 154:
        raise ValueError(f"expected 154 unique tasks, got rows={len(tasks)} unique={len(set(ids))}")

    grouped = defaultdict(list)
    video_rows = []
    for task in sorted(tasks, key=lambda row: str(row["video_id"])):
        video_id = str(task["video_id"])
        group = video_id.split("_", 1)[0]
        if group not in GROUP_ANGLES:
            raise ValueError(f"unclassified source group: {group}")
        grouped[group].append(video_id)
        video_rows.append({
            "video_id": video_id,
            "angle_degrees": GROUP_ANGLES[group],
            "source_group": group,
            "confidence": "high",
            "basis": "source-group first/last frames matched to official 0/45/90 references",
        })

    group_rows = []
    for group in sorted(grouped):
        videos = grouped[group]
        group_rows.append({
            "source_group": group,
            "angle_degrees": GROUP_ANGLES[group],
            "video_count": len(videos),
            "first_video_id": videos[0],
            "last_video_id": videos[-1],
            "confidence": "high",
            "basis": "visual check of first and last representative frames against official references",
        })

    for path, rows in ((args.video_output, video_rows), (args.group_output, group_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "videos": len(video_rows),
        "source_groups": len(group_rows),
        "video_angle_counts": dict(sorted(Counter(row["angle_degrees"] for row in video_rows).items())),
        "group_angle_counts": dict(sorted(Counter(row["angle_degrees"] for row in group_rows).items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

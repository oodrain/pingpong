"""比对两份 predictions.jsonl（官方格式）：逐条精确一致检查 + 差异清单 + 容差互配统计。

用法：
    python compare_predictions.py a.jsonl b.jsonl [--tol-frames 1] [--tol-px 50]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="比对两份官方格式预测")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol-frames", type=int, default=1)
    ap.add_argument("--tol-px", type=float, default=50.0)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    print(f"数量: A={len(A)} | B={len(B)}")

    same_len = len(A) == len(B)
    exact = same_len and all(a == b for a, b in zip(A, B))
    print(f"逐条精确一致: {'是' if exact else '否'}")
    if not exact:
        shown = 0
        for i, (a, b) in enumerate(zip(A, B)):
            if a != b:
                print(f"  第 {i+1} 条: A={a} | B={b}")
                shown += 1
                if shown >= 10:
                    print("  ...（最多显示 10 条）")
                    break
        if not same_len:
            ka = {(r["video_id"], r["frame_id"]) for r in A}
            kb = {(r["video_id"], r["frame_id"]) for r in B}
            only_a = sorted(ka - kb)
            only_b = sorted(kb - ka)
            print(f"  仅存在于 A 的: {len(only_a)} 条，例如 {only_a[:5]}")
            print(f"  仅存在于 B 的: {len(only_b)} 条，例如 {only_b[:5]}")

    # 官方容差口径（±1 帧 / 50px）一对一贪心互配
    used: set[int] = set()
    matched = 0
    for a in A:
        for j, b in enumerate(B):
            if j in used or b["video_id"] != a["video_id"]:
                continue
            if abs(b["frame_id"] - a["frame_id"]) <= args.tol_frames and \
               math.dist((a["x"], a["y"]), (b["x"], b["y"])) <= args.tol_px:
                used.add(j)
                matched += 1
                break
    print(f"容差内(±{args.tol_frames}帧/{args.tol_px:g}px)互配: {matched} 对（A {len(A)} 条，B {len(B)} 条）")


if __name__ == "__main__":
    main()

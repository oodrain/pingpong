"""Freeze AI-only notes, then compare frozen and same-frame automatic geometry.

Distances to coarse AI estimates are diagnostic, never human corner accuracy.
No reference file in the human review pack is written or changed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from experiments.tt3d_stage1_20261013.audit_review_inputs import read, sha
from development.detection.auto_table_geometry import GeometryConfig, extract_quad, AutoGeometryTimeline
from development.detection.tt3d_segmenter import TT3DSegmenter


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distances(automatic, estimate):
    a, b = np.asarray(automatic, float), np.asarray(estimate, float)
    alternatives = [np.linalg.norm(a - np.roll(order, n, axis=0), axis=1)
                    for order in (b, b[::-1]) for n in range(4)]
    d = min(alternatives, key=lambda x: x.mean())
    return {"mean_distance_to_ai_estimate_px": float(d.mean()),
            "maximum_distance_to_ai_estimate_px": float(d.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--notes", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest, notes = read(args.pack / "manifest.json"), read(args.notes)
    if notes["human_ground_truth"] is not False or notes["automatic_corner_coordinates_used_for_annotation"] is not False:
        raise ValueError("AI provenance must be explicit")
    if sorted(n["index"] for n in notes["cases"]) != list(range(1, len(manifest["cases"]) + 1)):
        raise ValueError("Visual notes must cover each case exactly once")
    indexed = {n["index"]: n for n in notes["cases"]}
    reviews = []
    for number, case in enumerate(manifest["cases"], 1):
        path = args.pack / case["image_url"]
        if sha(path) != case["image_sha256"]:
            raise ValueError("Review image changed")
        note = indexed[number]
        if note["visibility"] not in ("occluded", "four_corners_reviewable"):
            raise ValueError("Unknown visual visibility")
        q = note.get("estimated_corners_px", [])
        if note.get("estimated_corners_crop_pixels"):
            mapped = np.asarray(note["estimated_corners_crop_pixels"]) + np.asarray(note["crop_origin_original_px"]) - np.asarray(note["crop_image_offset_px"])
            if not np.array_equal(mapped, np.asarray(q)):
                raise ValueError("Crop-to-original coordinate conversion mismatch")
        if q:
            a = np.asarray(q, float)
            if note["visibility"] != "four_corners_reviewable" or a.shape != (4,2) or not np.isfinite(a).all():
                raise ValueError("No estimates allowed for hidden corners")
            if np.any(a < 0) or np.any(a[:,0] >= case["width"]) or np.any(a[:,1] >= case["height"]) or not cv2.isContourConvex(a.astype(np.float32)):
                raise ValueError("Invalid cyclic visual estimate")
        reviews.append(dict(case, **note, reference_source=notes["source"],
                            human_ground_truth=False, pixel_accurate_reference=False,
                            double_review_verified=False))
    run = Path(manifest["geometry_run"])
    frozen_hash = sha(run / "geometry_samples.jsonl")
    if frozen_hash != manifest["geometry_samples_sha256"]:
        raise ValueError("Frozen geometry changed")
    run_summary = read(run / "summary.json")
    for source in (PROJECT / "development/detection/auto_table_geometry.py",
                   PROJECT / "development/detection/tt3d_segmenter.py"):
        if run_summary["source_sha256"].get(str(source.resolve())) != sha(source):
            raise ValueError("Same-frame diagnostics require the frozen extraction code")
    args.output.mkdir(parents=True, exist_ok=False)
    overlays = args.output / "comparison_overlays"
    overlays.mkdir()
    # Write annotation copy before inspecting any automatic coordinates.
    ref_path = args.output / "ai_visual_reviews.jsonl"
    ref_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews), encoding="utf-8")
    review_digest = sha(ref_path)
    groups = defaultdict(list)
    for line in (run / "geometry_samples.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        groups[sample["group"], sample["video_id"]].append(sample)
    timelines = {key: AutoGeometryTimeline(value) for key, value in groups.items()}
    model = TT3DSegmenter(args.checkpoint)
    if model.provenance != run_summary["model"]:
        raise ValueError("Model/environment differs from frozen run")
    records = []
    for index, reference in enumerate(reviews, 1):
        image = cv2.imread(str(args.pack / reference["image_url"]))
        if image is None:
            raise ValueError("Cannot decode exact review frame")
        t0 = time.perf_counter()
        proposal = extract_quad(image, model.probability(image), GeometryConfig())
        elapsed = time.perf_counter() - t0
        timeline = timelines[reference["group"], reference["video_id"]]
        sample = timeline.sample(reference["frame_id"])
        available = timeline.lookup(reference["frame_id"]) is not None
        record = {"case_id": reference["case_id"], "visibility": reference["visibility"],
                  "selection_reason": reference["selection_reason"],
                  "group": reference["group"], "video_id": reference["video_id"], "frame_id": reference["frame_id"],
                  "coarse_ai_estimate_available": bool(reference.get("estimated_corners_px")),
                  "frozen_causal_geometry_available": available,
                  "frozen_sample_frame": sample["frame_id"] if sample else None,
                  "frozen_sample_age_frames": reference["frame_id"] - sample["frame_id"] if sample else None,
                  "frozen_state": sample["state"] if sample else None,
                  "same_frame_proposal": proposal,
                  "same_frame_inference_seconds_cpu": elapsed,
                  "same_frame_proposal_is_not_temporally_confirmed": True}
        estimate = reference.get("estimated_corners_px", [])
        if estimate and available:
            record["frozen_geometry_vs_ai"] = distances(sample["corners"], estimate)
        if estimate and proposal["usable_proposal"]:
            record["same_frame_vs_ai"] = distances(proposal["corners"], estimate)
        draw = image.copy()
        if proposal.get("corners"):
            cv2.polylines(draw, [np.asarray(proposal["corners"]).round().astype(np.int32)], True, (0,220,255), 2)
        if estimate:
            cv2.polylines(draw, [np.asarray(estimate).round().astype(np.int32)], True, (255,100,30), 2)
        cv2.rectangle(draw, (0,0), (draw.shape[1],58), (0,0,0), -1)
        cv2.putText(draw, f"AI diagnostic only | {reference['case_id']} | {reference['visibility']}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.putText(draw, "yellow=automatic candidate; blue=coarse AI estimate; NOT HUMAN GT", (8,44), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.imwrite(str(overlays / (reference["case_id"] + ".jpg")), draw)
        records.append(record)
        print(f"{index:02d}/{len(reviews)} {reference['case_id']} proposal={proposal['usable_proposal']} causal={available}", flush=True)
    if sha(ref_path) != review_digest:
        raise ValueError("AI annotations changed during automatic comparison")
    compared = [r for r in records if "same_frame_vs_ai" in r]
    frozen_compared = [r for r in records if "frozen_geometry_vs_ai" in r]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "cases": len(records), "ai_visibility_counts": dict(Counter(r["visibility"] for r in records)),
               "coarse_ai_estimates": sum(r["coarse_ai_estimate_available"] for r in records),
               "same_frame_usable_proposals": sum(r["same_frame_proposal"]["usable_proposal"] for r in records),
               "frozen_causal_geometry_available": sum(r["frozen_causal_geometry_available"] for r in records),
               "same_frame_comparisons_to_ai": len(compared), "frozen_comparisons_to_ai": len(frozen_compared),
               "same_frame_mean_distance_to_ai_estimates_px": float(np.mean([r["same_frame_vs_ai"]["mean_distance_to_ai_estimate_px"] for r in compared])) if compared else None,
               "frozen_mean_distance_to_ai_estimates_px": float(np.mean([r["frozen_geometry_vs_ai"]["mean_distance_to_ai_estimate_px"] for r in frozen_compared])) if frozen_compared else None,
               "human_corner_accuracy_measured": False, "independent_human_reference_count": 0,
               "diagnostic_selection_not_representative": True,
               "same_frame_candidates_do_not_replace_causal_geometry": True,
               "changes_to_detector_event_labels_or_geometry_algorithm": False,
               "arithmetic_corrections_after_comparison": sum(bool(r.get("arithmetic_correction_after_comparison")) for r in reviews),
               "corrected_annotation_context": notes.get("correction_context"),
               "ai_reviews_sha256": review_digest, "notes_sha256": sha(args.notes),
               "geometry_samples_sha256": frozen_hash, "model": model.provenance,
               "source_sha256": sha(Path(__file__)),
               "limits": ["All comparisons are to coarse AI visual estimates, not human ground truth.",
                          "Two arithmetic corrections followed viewing comparison overlays; this is not a new blinded evaluation.",
                          "Single-frame proposals have not passed temporal confirmation.",
                          "Sampling and occlusion make aggregate distances unsuitable as overall accuracy."]}
    write(args.output / "summary.json", summary)
    write(args.output / "records.json", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

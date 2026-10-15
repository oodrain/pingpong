"""Correct documented crop arithmetic and summarize existing inference only.

This is a post-comparison AI diagnostic revision, not a fresh blinded evaluation.
It never writes the human reference pack or runs a new model inference.
"""
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from experiments.tt3d_stage1_20261013.audit_review_inputs import read, sha
from experiments.tt3d_stage1_20261013.compare_visual_diagnostics import distances, write


def main():
    source = PROJECT / "experiments/tt3d_stage1_20261013"
    outputs = PROJECT / "outputs/experiments/tt3d_stage1_20261013"
    frozen = outputs / "visual_diagnostic_corrected_20261013T1210"
    pack = PROJECT / "outputs/experiments/geometry_reference_audit_20261013/review_pack"
    notes_path = source / "visual_notes.json"
    notes = read(notes_path)
    corrections = {
        17: [[336,202],[662,201],[658,308],[279,307]],
        18: [[336,201],[664,201],[659,308],[279,307]],
    }
    destination = outputs / "final_diagnostic_20261013_v2"
    destination.mkdir(exist_ok=False)
    # Preserve the second notes revision before correcting two more arithmetic errors.
    (destination / "notes_before_final_arithmetic_correction.json").write_bytes(notes_path.read_bytes())
    for note in notes["cases"]:
        if note["index"] in corrections:
            crop = np.asarray(corrections[note["index"]])
            note["estimated_corners_crop_pixels"] = crop.tolist()
            note["crop_origin_original_px"] = [192,158]
            note["crop_image_offset_px"] = [60,50]
            note["estimated_corners_px"] = (crop + [192,158] - [60,50]).tolist()
            note["arithmetic_correction_after_comparison"] = True
            correction_note = " 裁剪坐标换算的40px纵向误差已修正；非新一轮盲评。"
            if correction_note not in note["note"]:
                note["note"] += correction_note
    write(notes_path, notes)
    records = read(frozen / "records.json")
    frozen_notes = [json.loads(line) for line in
                    (frozen / "ai_visual_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = read(pack / "manifest.json")
    if len(records) != len(notes["cases"]) or len(records) != len(manifest["cases"]):
        raise ValueError("Mismatched case counts")
    overlays = destination / "comparison_overlays"
    overlays.mkdir()
    rejected_reviewable = []
    for index, (record, note, case) in enumerate(zip(records, notes["cases"], manifest["cases"]), 1):
        if note["index"] != index or record["case_id"] != case["case_id"]:
            raise ValueError("Case alignment changed")
        if sha(pack / case["image_url"]) != case["image_sha256"]:
            raise ValueError("Frame changed")
        estimate = note.get("estimated_corners_px")
        if note.get("estimated_corners_crop_pixels"):
            mapped = (np.asarray(note["estimated_corners_crop_pixels"])
                      + note["crop_origin_original_px"] - note["crop_image_offset_px"])
            if not np.array_equal(mapped, np.asarray(estimate)):
                raise ValueError("Invalid crop conversion")
        if estimate:
            q = np.asarray(estimate, np.float32)
            if q.shape != (4,2) or not cv2.isContourConvex(q):
                raise ValueError("Invalid polygon")
            if np.any(q < 0) or np.any(q[:,0] >= case["width"]) or np.any(q[:,1] >= case["height"]):
                raise ValueError("Out-of-bounds estimate")
        proposal = record["same_frame_proposal"]
        # A retained causal comparison must have exactly the same annotation as its inference source.
        annotation_changed = frozen_notes[index - 1].get("estimated_corners_px") != estimate
        if annotation_changed and "frozen_geometry_vs_ai" in record:
            raise ValueError("Corrected causal comparison requires original causal coordinates")
        if estimate and proposal["usable_proposal"]:
            record["same_frame_vs_ai"] = distances(proposal["corners"], estimate)
        record["ai_annotation_is_human_reference"] = False
        record["annotation_revision_after_comparison"] = bool(note.get("arithmetic_correction_after_comparison"))
        if record["visibility"] == "four_corners_reviewable" and not proposal["usable_proposal"]:
            gates = []
            if proposal.get("mask_iou", 0) < .70:
                gates.append("mask_iou_below_0.70")
            if proposal.get("edge_support", 0) < .40:
                gates.append("edge_support_below_0.40")
            rejected_reviewable.append({
                "case_id": record["case_id"], "mask_iou": proposal.get("mask_iou"),
                "edge_support": proposal.get("edge_support"), "failed_gates": gates,
                "returned_generic_reason": proposal["reason"],
            })
        image = cv2.imread(str(pack / case["image_url"]))
        if image is None:
            raise ValueError("Cannot decode original")
        if proposal.get("corners"):
            cv2.polylines(image, [np.asarray(proposal["corners"]).round().astype(np.int32)], True, (0,220,255), 2)
        if estimate:
            cv2.polylines(image, [np.asarray(estimate).round().astype(np.int32)], True, (255,100,30), 2)
        cv2.rectangle(image, (0,0), (image.shape[1],58), (0,0,0), -1)
        cv2.putText(image, f"AI DIAGNOSTIC | {case['case_id']} | post-comparison arithmetic revision",
                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        cv2.putText(image, "yellow=automatic; blue=coarse AI estimate; NOT HUMAN GROUND TRUTH",
                    (8,44), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        if not cv2.imwrite(str(overlays / (case["case_id"] + ".jpg")), image):
            raise ValueError("Cannot save overlay")
    previous = read(frozen / "summary.json")
    summary = {
        "cases": len(records),
        "ai_visibility_counts": dict(Counter(r["visibility"] for r in records)),
        "same_frame_usable_proposals": sum(r["same_frame_proposal"]["usable_proposal"] for r in records),
        "frozen_causal_geometry_available": sum(r["frozen_causal_geometry_available"] for r in records),
        "reviewable_rejected_cases": rejected_reviewable,
        "arithmetic_corrected_cases": [n["index"] for n in notes["cases"] if n.get("arithmetic_correction_after_comparison")],
        "new_blinded_evaluation": False,
        "human_ground_truth": False,
        "human_corner_accuracy_measured": False,
        "human_reference_records": read(outputs / "human_reference_readiness_20261013T1215/summary.json")["submitted_references"],
        "diagnostic_selection_not_representative": True,
        "source_inference_records_sha256": sha(frozen / "records.json"),
        "notes_sha256": sha(notes_path),
        "manifest_sha256": sha(pack / "manifest.json"),
        "frozen_geometry_sha256": previous["geometry_samples_sha256"],
        "model": previous["model"],
        "model_inference_repeated": False,
        "production_geometry_or_events_changed": False,
    }
    write(destination / "summary.json", summary)
    write(destination / "records.json", records)
    write(destination / "ai_notes_final.json", notes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

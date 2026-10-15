"""Read-only provenance checks and unmarked contact sheets for TT3D stage 1.

Existing references, geometry, event labels and frozen runs are never modified.
AI visual notes belong in a separate file; they must not be submitted as human
references to geometry_reference_review.py.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.pack / "manifest.json")
    frozen_run = Path(manifest["geometry_run"])
    geometry_digest = sha(frozen_run / "geometry_samples.jsonl")
    if geometry_digest != manifest["geometry_samples_sha256"]:
        raise ValueError("Frozen geometry hash differs from the review manifest")
    public_root = args.data_root / "participant/participant/pingpang/public_data"
    public = {v["video_id"]: v for v in read(public_root / "manifest.json")["videos"]}
    train = {v["video_id"]: v for v in read(args.data_root / "C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json")["videos"]}
    source_code = Path(__file__).resolve()
    video_digests = {}
    records = []
    for number, case in enumerate(manifest["cases"], 1):
        path = args.pack / case["image_url"]
        digest = sha(path)
        with Image.open(path) as frame:
            size = frame.size
        if digest != case["image_sha256"] or size != (case["width"], case["height"]):
            raise ValueError("Review frame changed: " + case["case_id"])
        if case["group"] == "public":
            meta = public[case["video_id"]]
            video = public_root / meta["path"]
        else:
            meta = train[case["video_id"]]
            video = args.data_root / "pp_train_data/pp_train_data/rallies_videos" / meta["owner"] / case["video_id"][:2] / (case["video_id"][3:] + ".mp4")
        if video not in video_digests:
            video_digests[video] = sha(video)
        if video_digests[video] != case["video_sha256"]:
            raise ValueError("Video changed: " + case["case_id"])
        records.append(dict(case, review_index=number, image_verified=True,
                            video_verified=True, angle=meta.get("angle", "unknown")))
    # Check all inputs before creating a new output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    sheets = args.output / "blind_contact_sheets"
    sheets.mkdir()
    for start in range(0, len(records), 6):
        canvas = Image.new("RGB", (1280, 1170), "#202830")
        draw = ImageDraw.Draw(canvas)
        for local, case in enumerate(records[start:start + 6]):
            x, y = (local % 2) * 640, (local // 2) * 390
            with Image.open(args.pack / case["image_url"]) as original:
                thumb = original.convert("RGB")
                thumb.thumbnail((636, 358))
                canvas.paste(thumb, (x + (640 - thumb.width) // 2, y + 29))
            draw.text((x + 6, y + 7), f"{case['review_index']:02d} {case['case_id']} [{case['width']}x{case['height']}]", fill="white")
        canvas.save(sheets / f"sheet_{start // 6 + 1:02d}.jpg", quality=95)
    refs = args.pack / "references.jsonl"
    ref_rows = [json.loads(line) for line in refs.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if refs.exists() else []
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_pack": str(args.pack.resolve()), "frozen_geometry_run": str(frozen_run),
        "cases": len(records), "unique_videos": len(video_digests),
        "selection_counts": dict(Counter(r["selection_reason"] for r in records)),
        "all_image_hashes_and_dimensions_verified": True,
        "all_video_hashes_verified": True, "frozen_geometry_hash_verified": True,
        "existing_reference_records": len(ref_rows),
        "existing_reference_cases": len({r["case_id"] for r in ref_rows}),
        "human_corner_accuracy_status": "requires_reference_provenance_and_review",
        "corner_accuracy_claimed": False,
        "manifest_sha256": sha(args.pack / "manifest.json"),
        "geometry_samples_sha256": geometry_digest,
        "existing_references_sha256": sha(refs) if refs.exists() else None,
        "source_sha256": sha(source_code),
        "notes": ["Diagnostic selection is not representative of all frames.",
                  "Contact sheets contain no automatic corners or event markers.",
                  "AI visual estimates are not independent human ground truth."]}
    for name, content in (("input_integrity.json", summary), ("case_index.json", records)):
        (args.output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

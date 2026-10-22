"""Check exact detector-cache compatibility on public15 and train154; no GT needed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from development.bounce.ge_rule_v2 import replay


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=PROJECT)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    public = args.data_root / "outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1"
    train = args.data_root / "C_train154_20261011_v3/C_train154_20261011_v3"
    results, inputs = {}, []
    for name, root in (("public15", public), ("train154", train)):
        manifest = root / "selected_manifest.json"
        baseline_path = root / "predictions.jsonl"
        metadata = json.loads(manifest.read_text(encoding="utf-8-sig"))["videos"]
        baseline = rows(baseline_path)
        inputs.extend([manifest, baseline_path])
        total, frames = 0, 0
        for meta in metadata:
            vid = meta["video_id"]
            raw_path = root / "raw" / (vid + ".jsonl")
            raw = rows(raw_path)
            inputs.append(raw_path)
            if len(raw) != meta["n_frames"]:
                raise ValueError(vid + ": cache frame count mismatch")
            expected = [{k: p[k] for k in ("frame_id", "x", "y")} for p in baseline if p["video_id"] == vid]
            for mode in ("v0", "v1a", "v1b"):
                predictions, _ = replay(raw, mode=mode)
                if predictions != expected:
                    raise ValueError(vid + ": repaired " + mode + " replay mismatch")
            total += len(expected)
            frames += len(raw)
        results[name] = {"videos": len(metadata), "raw_frames": frames, "predictions": total,
                         "v0_exact_replay": True, "v1a_missing_geometry_exact_fallback": True,
                         "v1b_missing_geometry_exact_fallback": True}
    sources = [Path(__file__), PROJECT / "development/bounce/ge_rule_v2.py",
               PROJECT / "development/bounce/table_geometry.py"]
    result = {"status": "passed", "datasets": results,
              "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "note": "Cache compatibility and unknown-geometry fallback only; no geometry accuracy claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "datasets": results}, indent=2))


if __name__ == "__main__":
    main()

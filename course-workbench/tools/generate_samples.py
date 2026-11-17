"""Build short, fully synthetic fixtures for frame-accurate review."""
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "development" / "video_studio"
SIZE = (960, 540)
DURATION = 5
# (tenths of a second, position across the length, position across the width).
KNOTS = [
    (0, .12, .32), (8, .25, .26), (18, .68, .54),
    (28, .43, .68), (38, .81, .63), (46, .59, .38), (50, .74, .34),
]


def embed_canvas(corners):
    unit = np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(unit, corners)
    return matrix


def map_point(matrix, u, v):
    homogeneous = matrix @ np.array([u, v, 1.0])
    return tuple(int(round(x)) for x in homogeneous[:2] / homogeneous[2])


def draw_table(corners, matrix, fps):
    width, height = SIZE
    frame = np.full((height, width, 3), (33, 29, 23), dtype=np.uint8)
    cv2.rectangle(frame, (0, 268), (width, height), (47, 43, 36), -1)
    cv2.fillConvexPoly(frame, corners.astype(np.int32) + [0, 10], (24, 35, 40))
    cv2.fillConvexPoly(frame, corners.astype(np.int32), (103, 102, 35))
    cv2.polylines(frame, [corners.astype(np.int32)], True, (193, 224, 217), 2, cv2.LINE_AA)
    cv2.line(frame, map_point(matrix, 0, .5), map_point(matrix, 1, .5), (155, 185, 173), 1, cv2.LINE_AA)
    a, b = (map_point(matrix, .5, v) for v in (0, 1))
    cv2.line(frame, a, b, (183, 214, 208), 2, cv2.LINE_AA)
    cv2.line(frame, (a[0], a[1]-19), (b[0], b[1]-19), (230, 239, 230), 2, cv2.LINE_AA)
    cv2.line(frame, a, (a[0], a[1]-19), (230, 239, 230), 2, cv2.LINE_AA)
    cv2.line(frame, b, (b[0], b[1]-19), (230, 239, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, "PINGPONG COURSE / SYNTHETIC FIXTURE", (30, 44),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (222, 232, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps} fps | fixed camera | 5 seconds", (30, 74),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (171, 215, 211), 1, cv2.LINE_AA)
    return frame


def ball_position(matrix, frame_id, knots: Sequence):
    for (start, u0, v0), (end, u1, v1) in zip(knots, knots[1:]):
        if frame_id <= end:
            progress = (frame_id - start) / (end - start)
            x, floor_y = map_point(matrix, u0 + (u1-u0)*progress, v0 + (v1-v0)*progress)
            return x, int(round(floor_y - 4 * progress * (1 - progress) * 77))
    raise ValueError("frame falls beyond the synthetic flight path")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate(fps):
    config = json.loads((STUDIO / f"table_demo_course{fps}.json").read_text(encoding="utf-8"))
    directory = ROOT / config["result"]
    directory.mkdir(parents=True, exist_ok=True)
    video = ROOT / config["video"]
    corners = np.asarray(config["corners"], dtype=np.float32)
    matrix = embed_canvas(corners)
    background = draw_table(corners, matrix, fps)
    knots = [(round(t * fps / 10), u, v) for t, u, v in KNOTS]
    contact_frames = {point[0] for point in knots[1:-1]}
    metadata = directory / "run.json"
    metadata.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create a video at {video}")
    tracks, landings = [], []
    try:
        for frame_id in range(fps * DURATION):
            pixels = background.copy()
            x, y = ball_position(matrix, frame_id, knots)
            cv2.circle(pixels, (x, y), 7, (220, 232, 255), -1, cv2.LINE_AA)
            cv2.putText(pixels, f"f{frame_id:03d}   {frame_id/fps:.3f} s", (30, 505),
                        cv2.FONT_HERSHEY_SIMPLEX, .60, (221, 233, 230), 1, cv2.LINE_AA)
            writer.write(pixels)
            row = dict(frame_id=frame_id, x=x, y=y, timestamp_s=frame_id/fps)
            tracks.append(dict(row, confidence=1.0, detected=True, in_image=True))
            if frame_id in contact_frames:
                landings.append(row)
    finally:
        writer.release()
    fingerprint = hashlib.sha256(video.read_bytes()).hexdigest()
    save_json(directory / "tracks.json", tracks)
    save_json(directory / "landings.json", landings)
    save_json(metadata, dict(video_sha256=fingerprint, image_size=SIZE, frames=len(tracks),
                             native_fps=fps, synthetic=True))
    print(f"{fps} fps: {len(tracks)} frames, {len(landings)} events -> {video.relative_to(ROOT)}")


if __name__ == "__main__":
    for rate in (25, 50):
        generate(rate)

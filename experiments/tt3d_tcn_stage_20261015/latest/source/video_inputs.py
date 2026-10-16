"""Automatic inputs for the retained model, with explicit fixed asset paths."""
from pathlib import Path
import cv2
import numpy as np
import torch
from .auto_table_geometry import GeometryConfig, CausalGeometryTracker, extract_quad
from .geometry import SecondsGeometryTracker
from .startup_pts_capture import StartupPTSCapture
from .frame_features import automatic_frame_features
from .revision import motion_features
from .ball_repair import repair_video


def automatic_inputs(video, ball_model, table_model):
    from .video_ball_detector import VideoBallDetector
    from .tt3d_segmenter import TT3DSegmenter
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    video = Path(video)
    ball = VideoBallDetector(ball_model, threads=4)
    table = TT3DSegmenter(table_model)
    cap = StartupPTSCapture(str(video))
    times = np.asarray(cap.times, float) / 1000
    times -= times[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError('Nonmonotonic actual timestamps')
    fps = 1 / np.median(np.diff(times)) if len(times) > 1 else 25.
    cfg = GeometryConfig()
    cadence_driver = CausalGeometryTracker(cfg)
    geometry = SecondsGeometryTracker(times, cfg)
    samples, raw, detections = [], [], []
    fid = next_sample = 0
    size = None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            current_size = image.shape[1], image.shape[0]
            size = current_size if size is None else size
            if current_size != size:
                raise ValueError('Video resolution changed')
            detection = ball.detect(image)
            detections.append({'target_frame_id': fid - 1, 'detection': detection})
            if fid >= next_sample:
                import copy
                proposal = extract_quad(image, table.probability(image), cfg)
                raw.append({'frame_id': fid, 'proposal': copy.deepcopy(proposal)})
                driver_state = cadence_driver.update(fid, image, copy.deepcopy(proposal))
                samples.append(geometry.update(fid, image, copy.deepcopy(proposal)))
                next_sample = fid + max(1, round(fps * (.4 if driver_state['state'] == 'stable' else .12)))
            fid += 1
    finally:
        cap.release()
    if fid != len(times):
        raise ValueError('Decoded frame count changed')
    aligned = {r['target_frame_id']: r['detection'] for r in detections if r['target_frame_id'] >= 0}
    tracks = []
    for i, timestamp in enumerate(times):
        detection = aligned.get(i)
        good = bool(detection)
        tracks.append({'frame_id': i, 'timestamp_s': float(timestamp), 'detected': good,
                       'x': detection['x'] if good else None, 'y': detection['y'] if good else None,
                       'confidence': detection['confidence'] if good else None,
                       'in_image': bool(good and 0 <= detection['x'] < size[0] and 0 <= detection['y'] < size[1]),
                       'missing_reason': None if good else 'no_detection' if i in aligned else 'tail_not_inferred'})
    tracks, _ = repair_video(video, tracks, size, ball)
    features = np.c_[automatic_frame_features(tracks, samples, size), motion_features(tracks, size)].astype(np.float32)
    return features, tracks, samples, raw

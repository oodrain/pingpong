"""Official lifecycle adapter for the unchanged retained video-only TCN."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from core.types import Landing, Solution as OfficialSolution

from .auto_table_geometry import GeometryConfig, CausalGeometryTracker, extract_quad
from .ball_repair import repair_video
from .contract import CONTRACT
from .frame_features import automatic_frame_features
from .geometry import SecondsGeometryTracker
from .revision import motion_features
from .runtime import infer, load_asset
from .timestamp_reader import read_timestamps
from .image_evidence import FrameEvidence, TimingProfile
from .tt3d_segmenter import TT3DSegmenter
from .video_ball_detector import VideoBallDetector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class BaseSolution(OfficialSolution):
    def prepare(self, context):
        self.context = context
        self.root = Path(__file__).resolve().parents[2]
        self.device = os.environ.get('TCN_DEVICE', 'cuda')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('TCN_DEVICE must be cpu or cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU. Use TCN_DEVICE=cpu for CPU checks only, or start a GPU instance.')
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        config = json.loads((self.root / 'configs/tcn_frozen.json').read_text(encoding='utf-8'))
        if config != CONTRACT:
            raise ValueError('Frozen TCN configuration changed; restore configs/tcn_frozen.json')
        for name, expected in CONTRACT['asset_sha256'].items():
            if sha256(self.root / 'weights' / name) != expected:
                raise ValueError('Model checksum mismatch: ' + name)
        self.ball = VideoBallDetector(self.root / 'weights/ball.onnx', device=self.device,
                                      profile_prefix=os.environ.get('TCN_ORT_PROFILE_PREFIX'))
        self.table = TT3DSegmenter(self.root / 'weights/table_segmentation.ckpt', device=self.device)
        self.models = load_asset(self.root / 'weights/model.pt', device=self.device)
        # Model warmup belongs to prepare; all video processing stays in timed reset/frame/finish calls.
        blank = np.zeros((320, 640, 3), np.uint8)
        self.ball.detect(blank)
        self.table.probability(blank)
        for states in self.models.values():
            for model, _ in states:
                width = model.config['input_features']
                with torch.inference_mode():
                    model(torch.zeros((1, width, 31), device=self.device))
        if self.device == 'cuda':
            torch.cuda.synchronize()
        context.info.update({
            'adapter': CONTRACT['schema'], 'tcn_device': self.device,
            'table_device': str(next(self.table.model.parameters()).device),
            'ball_providers': self.ball.session.get_providers(),
            'model_sha256': CONTRACT['asset_sha256']['model.pt'],
            'precision': 'float32', 'human_annotation_inputs': False,
            'historical_detection_inputs': False, 'future_context': True,
            'timestamp_policy': 'original_checked_video_pts', 'tcn_video_timings': [], 'tcn_video_profiles': [], 'engineering_variant': 'B_shared_evidence_pts',
        })

    def reset(self, video_meta):
        self.meta = video_meta
        self.path = Path(self.context.input_dir) / video_meta.path
        self.profile = TimingProfile()
        self.timing = dict.fromkeys(('timestamp_probe_sec', 'ball_sec', 'table_model_sec',
                                    'table_geometry_sec', 'roi_repair_sec', 'features_sec', 'tcn_sec'), 0.)
        start = time.perf_counter()
        # Keep the original actual-PTS feature contract; never substitute frame_id/manifest_fps.
        self.times, self.timestamp_info = read_timestamps(self.path, video_meta.n_frames)
        self.times = np.asarray(self.times, float) / 1000
        self.times -= self.times[0]
        if len(self.times) != video_meta.n_frames or np.any(np.diff(self.times) <= 0):
            raise ValueError('Actual video timestamps do not match official metadata')
        self.timing['timestamp_probe_sec'] += time.perf_counter() - start
        self.fps = 1 / np.median(np.diff(self.times)) if len(self.times) > 1 else 25.
        self.cfg = GeometryConfig()
        self.cadence = CausalGeometryTracker(self.cfg)
        self.geometry = SecondsGeometryTracker(self.times, self.cfg)
        self.ball.reset()
        self.ball.profile = TimingProfile()
        self.samples, self.detections = [], []
        self.fid = self.next_sample = 0
        self.size = (video_meta.width, video_meta.height)
        self.finished = False
        self.last_debug = None

    def process_frame(self, frame_rgb, frame_meta):
        if self.finished or frame_meta['frame_id'] != self.fid or frame_meta['video_id'] != self.meta.video_id:
            raise ValueError('Official frame sequence is not contiguous or belongs to another video')
        if frame_rgb.shape != (self.size[1], self.size[0], 3) or frame_rgb.dtype != np.uint8:
            raise ValueError('Expected official RGB uint8 frames with unchanged dimensions')
        image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        start = time.perf_counter()
        detection = self.ball.detect(image)
        self.timing['ball_sec'] += time.perf_counter() - start
        self.detections.append({'target_frame_id': self.fid - 1, 'detection': detection})
        if self.fid >= self.next_sample:
            start = time.perf_counter()
            probability = self.table.probability(image)
            self.timing['table_model_sec'] += time.perf_counter() - start
            start = time.perf_counter()
            evidence = FrameEvidence(image, self.cfg.fitting_width, self.profile)
            proposal = extract_quad(image, probability, self.cfg,frame_evidence=evidence)
            state = self.cadence.update(self.fid, image, copy.deepcopy(proposal),frame_evidence=evidence)
            self.samples.append(self.geometry.update(self.fid, image, copy.deepcopy(proposal),frame_evidence=evidence))
            self.next_sample = self.fid + max(1, round(self.fps * (.4 if state['state'] == 'stable' else .12)))
            self.timing['table_geometry_sec'] += time.perf_counter() - start
        self.fid += 1
        # The frozen model uses future observations and full-video sparse ROI evidence.
        return []

    def finish_video(self, video_meta):
        if self.finished:
            return []
        if video_meta.video_id != self.meta.video_id or self.fid != len(self.times):
            raise ValueError('Video finish does not match the complete decoded frame sequence')
        aligned = {r['target_frame_id']: r['detection'] for r in self.detections if r['target_frame_id'] >= 0}
        tracks = []
        for fid, timestamp in enumerate(self.times):
            detection = aligned.get(fid)
            good = bool(detection)
            tracks.append({'frame_id': fid, 'timestamp_s': float(timestamp), 'detected': good,
                           'x': detection['x'] if good else None, 'y': detection['y'] if good else None,
                           'confidence': detection['confidence'] if good else None,
                           'in_image': bool(good and 0 <= detection['x'] < self.size[0] and 0 <= detection['y'] < self.size[1]),
                           'missing_reason': None if good else 'no_detection' if fid in aligned else 'tail_not_inferred'})
        start = time.perf_counter()
        tracks, repair = repair_video(self.path, tracks, self.size, self.ball)
        self.timing['roi_repair_sec'] += time.perf_counter() - start
        start = time.perf_counter()
        features = np.c_[automatic_frame_features(tracks, self.samples, self.size),
                         motion_features(tracks, self.size)].astype(np.float32)
        self.timing['features_sec'] += time.perf_counter() - start
        start = time.perf_counter()
        points, hits, scores, _ = infer(self.models, features, tracks)
        self.automatic_tracks = tracks
        points = self.review_points(points, tracks)
        self.timing['tcn_sec'] += time.perf_counter() - start
        self.context.info['tcn_video_profiles'].append(dict(video_id=video_meta.video_id, timestamps=self.timestamp_info,
            geometry=self.profile.snapshot(), ball=self.ball.profile.snapshot(), geometry_samples=len(self.samples)))
        self.context.info['tcn_video_timings'].append(dict(
            video_id=video_meta.video_id, frames=self.fid, roi_repair_triggered=repair['triggered'],
            roi_extra_model_calls=repair['model_calls'],
            **{key: round(value, 6) for key, value in self.timing.items()}))
        if os.environ.get('TCN_SAVE_DEBUG') == '1':
            # Debug is opt-in, writes no frames and is disabled for normal speed measurements.
            safe_id = ''.join(c for c in video_meta.video_id if c.isalnum() or c in '_-')
            if not safe_id:
                raise ValueError('Invalid debug video ID')
            out = Path(self.context.output_dir) / 'tcn_debug' / safe_id
            out.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(out / 'features.npz', control=features, **scores)
            (out / 'tracks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tracks), encoding='utf-8')
            (out / 'landings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in points), encoding='utf-8')
            self.last_debug = dict(features=features, tracks=tracks, samples=self.samples,
                                   points=points, hits=hits, scores=scores)
        self.finished = True
        return [Landing(frame_id=int(p['frame_id']), x=float(p['x']), y=float(p['y'])) for p in points]

"""Frozen two-branch inference; device selection changes no parameters."""
from pathlib import Path

import hashlib

import json

import numpy as np

import torch

from .model import load, predict

from .observation_features import augment

from .decoder import original_control

from .frame_features import decode_frame_events

SCHEMA = 'retained_tcn_progress_v1'

def load_asset(path, device="cpu"):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if set(checkpoint['checkpoints']) != {'calibrated_event', 'fps_context'} or any(len(v) != 3 for v in checkpoint['checkpoints'].values()):
        raise ValueError('Expected the unchanged frozen two-branch three-model ensemble')
    if checkpoint['schema'] != SCHEMA:
        raise ValueError('Unsupported model schema')
    return {branch: [(load(s).to(device), s['normalization']) for s in states]
            for branch, states in checkpoint['checkpoints'].items()}

def infer(models, features, tracks):
    features = np.asarray(features, np.float32)
    if features.shape != (len(tracks), 51) or not np.isfinite(features).all():
        raise ValueError('Expected the unchanged automatic 51-feature contract')
    if [r['frame_id'] for r in tracks] != list(range(len(tracks))):
        raise ValueError('Track IDs must be contiguous decoded frame IDs')
    scores = {branch: np.mean([predict(model, augment(features, tracks, branch), stats)
              for model, stats in states], axis=0) for branch, states in models.items()}
    points, admission = original_control(scores['calibrated_event'], scores['fps_context'], tracks)
    hits = decode_frame_events(scores['calibrated_event'][:, 1], tracks, .5, .12)[0]
    return points, hits, scores, admission

"""Causal seconds-based geometry confirmation; spatial acceptance stays fixed."""
import cv2
import numpy as np
from .auto_table_geometry import (
    GeometryConfig, order_quad, align_quad, table_diagonal, perimeter_evidence,
)

POLICY = {
    'schema': 2, 'hold_seconds': .8,
    'confirmation_window_seconds': 'max(.8, (required_observations-1)*rounded_stable_cadence) plus one nominal PTS interval',
    'required_consistent_observations': 3, 'spatial_thresholds': 'unchanged GeometryConfig',
    'rejected_observation': 'Does not certify geometry; retain recent good history only without contradictory perimeter.',
    'emission': 'Three consistent real proposals, plus current RGB verification; held geometry also reverified.',
    'cut': 'Same grayscale cut rule; reset history and held geometry.',
    'sampling': 'Unchanged legacy tracker controls sample times for exact cache/runtime comparability.',
    'event_labels_used': False, 'human_geometry_at_runtime': False,
}


class SecondsGeometryTracker:
    def __init__(self, timestamps, config=None):
        self.times = np.asarray(timestamps, float)
        assert self.times.ndim == 1 and len(self.times) and np.isfinite(self.times).all()
        assert np.all(np.diff(self.times) > 0)
        self.config = config or GeometryConfig()
        nominal=float(np.median(np.diff(self.times))) if len(self.times)>1 else .04
        rounded_stable_cadence=max(1,round(.4/nominal))*nominal
        self.confirmation_window=max(.8,(self.config.stability_samples-1)*rounded_stable_cadence)+nominal
        self.history = []
        self.last_good = None
        self.previous_image = None
        self.previous_frame = -1
        self.epoch = self.camera_epoch = 0

    def reset(self):
        self.history = []
        self.last_good = None
        self.epoch += 1

    def perimeter(self, image, q, frame_evidence=None):
        cfg = self.config
        scale = min(1., cfg.fitting_width / image.shape[1])
        im = frame_evidence.fitting_image if frame_evidence is not None else cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
        ev = perimeter_evidence(im, np.asarray(q, np.float32)*scale,frame_evidence=frame_evidence)
        okay = all(s >= cfg.minimum_straight_span or n >= .75 for s, n in zip(
            ev['straight_span_by_side'], ev['neutral_border_support_by_side']))
        return okay, ev

    def expiry(self, timestamp):
        return int(np.searchsorted(self.times, timestamp+POLICY['hold_seconds']+1e-9, side='right')-1)

    def update(self, frame_id, image, proposal, frame_evidence=None):
        assert self.previous_frame < frame_id < len(self.times)
        self.previous_frame = frame_id
        now = self.times[frame_id]
        thumb = frame_evidence.scene_thumb if frame_evidence is not None else cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64,36)).astype(float)
        cut = self.previous_image is not None and float(np.mean(abs(thumb-self.previous_image)>45)) > .65
        self.previous_image = thumb
        if cut:
            self.reset(); self.camera_epoch += 1
        self.history = [(t,q) for t,q in self.history if now-t <= self.confirmation_window+1e-9]
        if self.last_good and now-self.last_good['confirmed_timestamp_s'] > POLICY['hold_seconds']+1e-9:
            self.last_good = None
        reference = self.history[-1][1] if self.history else np.asarray(self.last_good['corners']) if self.last_good else None
        cfg = self.config
        if not proposal['usable_proposal']:
            ev = proposal.get('image_perimeter', {})
            if reference is not None and proposal.get('corners') and min(ev.get('straight_span_by_side', [0])) >= cfg.minimum_straight_span:
                q = align_quad(proposal['corners'], reference)
                if np.max(np.linalg.norm(q-reference, axis=1))/table_diagonal(reference) > cfg.maximum_corner_jitter:
                    self.reset()
        else:
            q = order_quad(proposal['corners']) if reference is None else align_quad(proposal['corners'], reference)
            if reference is not None and np.max(np.linalg.norm(q-reference, axis=1))/table_diagonal(reference) > cfg.maximum_corner_jitter:
                self.reset()
            self.history.append((now,q)); self.history = self.history[-cfg.stability_samples:]
            median = np.median([p for _,p in self.history], axis=0)
            ready = len(self.history) == cfg.stability_samples and max(
                np.max(np.linalg.norm(p-median, axis=1))/table_diagonal(median) for _,p in self.history) <= cfg.maximum_corner_jitter
            if ready:
                okay, ev = self.perimeter(image, median,frame_evidence=frame_evidence)
                if okay:
                    self.last_good = dict(proposal, corners=median.tolist(), confirmed_timestamp_s=float(now), last_good_frame=frame_id)
                    return dict(self.last_good, frame_id=frame_id, epoch=self.epoch, usable=True, state='stable',
                                reason='seconds_consensus_current_perimeter', current_perimeter=ev,
                                valid_until=self.expiry(now), physical_axes='ambiguous', camera_epoch=self.camera_epoch)
        if self.last_good:
            okay, ev = self.perimeter(image, self.last_good['corners'],frame_evidence=frame_evidence)
            if okay:
                return dict(self.last_good, frame_id=frame_id, epoch=self.epoch, usable=True, state='held',
                            reason='seconds_hold_current_perimeter', current_perimeter=ev,
                            valid_until=self.expiry(self.last_good['confirmed_timestamp_s']), physical_axes='ambiguous', camera_epoch=self.camera_epoch)
            self.reset()
        return dict(proposal, frame_id=frame_id, epoch=self.epoch, usable=False, state='unknown',
                    reason='camera_cut' if cut else proposal.get('reason','rejected') if not proposal['usable_proposal'] else 'awaiting_seconds_consensus',
                    valid_until=frame_id, physical_axes='unknown', camera_epoch=self.camera_epoch)

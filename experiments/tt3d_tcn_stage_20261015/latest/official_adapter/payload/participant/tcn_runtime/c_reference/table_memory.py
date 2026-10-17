"""Bounded table memory. Inputs are automatic proposals, prior states and RGB only."""
from copy import deepcopy
import cv2
import numpy as np

POLICY = dict(hold_seconds=.8, maximum_check_age_seconds=.45,
              minimum_straight_span=.25, minimum_neutral_support=.75,
              minimum_visible_adjacent_edges=2, maximum_corner_shift_fraction=.025,
              minimum_corner_uncertainty_fraction=.003,
              uncertainty_growth_px_at_640_per_second=2.,
              cut_pixel_delta=45., cut_fraction=.65,
              contradictory_observations_to_clear=2)


def quad(value):
    try:
        q = np.asarray(value, np.float32)
    except (ValueError, TypeError):
        return None
    if q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) <= 1:
        return None
    return q


def diagonal(q):
    return max(float(np.linalg.norm(np.ptp(q, axis=0))), 1.)


def align(q, reference):
    return min((np.roll(a, k, axis=0) for a in (q, q[::-1]) for k in range(4)),
               key=lambda a: float(np.square(a-reference).sum()))


def unknown(frame, reason, epoch=0):
    return dict(frame_id=int(frame), usable=False, state='unknown', reason=reason,
                valid_until=int(frame), memory_camera_epoch=int(epoch))


class BorderVerifier:
    """The frozen RGB measurements; a hidden edge is not a negative table label."""
    def __init__(self, image):
        from .auto_table_geometry import perimeter_evidence
        self.measure = perimeter_evidence
        self.scale = min(1., 640/image.shape[1])
        self.image = cv2.resize(image, (round(image.shape[1]*self.scale), round(image.shape[0]*self.scale)))

    def __call__(self, q):
        return self.measure(self.image, np.asarray(q, np.float32)*self.scale)


def visible_edges(evidence):
    a, b = evidence.get('straight_span_by_side', []), evidence.get('neutral_border_support_by_side', [])
    if len(a) != 4 or len(b) != 4:
        return [False]*4
    return [bool(s >= POLICY['minimum_straight_span'] or n >= POLICY['minimum_neutral_support']) for s, n in zip(a, b)]


class TableMemory:
    def __init__(self, timestamps, size):
        self.times = np.asarray(timestamps, float)
        if not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('Need actual increasing video timestamps')
        self.size = tuple(size)
        self.good = None
        self.thumb = None
        self.last_frame = self.last_sample = -1
        self.epoch = 0
        self.confirmed_after = -float('inf')
        self.conflicts = 0
        self.cuts = []

    def observe_frame(self, frame, image):
        if frame != self.last_frame+1:
            raise ValueError('Observe every frame once in order')
        self.last_frame = frame
        thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
        cut = self.thumb is not None and float(np.mean(abs(thumb-self.thumb) > POLICY['cut_pixel_delta'])) > POLICY['cut_fraction']
        self.thumb = thumb
        if cut:
            self.good = None
            self.conflicts = 0
            self.epoch += 1
            self.confirmed_after = float(self.times[frame])
            self.cuts.append(frame)
        return bool(cut)

    def update(self, frame, legacy, proposal, verifier):
        if frame <= self.last_sample or frame != self.last_frame:
            raise ValueError('Sample must follow its current RGB frame exactly once')
        self.last_sample = frame
        now = float(self.times[frame])
        confirmed = float(legacy.get('confirmed_timestamp_s', self.times[legacy['frame_id']]))
        q = quad(legacy.get('corners'))
        if legacy.get('usable') and frame <= legacy.get('valid_until', frame) and q is not None and confirmed >= self.confirmed_after:
            self.good = deepcopy(legacy)
            self.conflicts = 0
            result = deepcopy(legacy)
            result.update(memory_origin='unchanged_verified_geometry', memory_camera_epoch=self.epoch,
                          memory_checked_timestamp_s=now, corner_uncertainty_px=POLICY['minimum_corner_uncertainty_fraction']*diagonal(q))
            return result
        if not self.good:
            return unknown(frame, 'no_prior_verified_table', self.epoch)
        age = now-float(self.good['confirmed_timestamp_s'])
        if age > POLICY['hold_seconds']+1e-9:
            self.good = None
            return unknown(frame, 'bounded_table_memory_expired', self.epoch)
        reference = quad(self.good.get('corners'))
        if reference is None:
            self.good = None
            return unknown(frame, 'invalid_prior_table', self.epoch)
        evidence = verifier(reference)
        visible = visible_edges(evidence)
        current = quad(proposal.get('corners')) if proposal.get('usable_proposal') else None
        shift = 0.
        if current is not None:
            current = align(current, reference)
            shift = float(np.linalg.norm(current-reference, axis=1).max())
            if shift > POLICY['maximum_corner_shift_fraction']*diagonal(reference):
                self.conflicts += 1
                if self.conflicts >= POLICY['contradictory_observations_to_clear']:
                    self.good = None
                    self.confirmed_after = now
                return unknown(frame, 'contradictory_table_position', self.epoch)
            self.conflicts = 0
        adjacent = any(visible[i] and visible[(i+1)%4] for i in range(4))
        if sum(visible) < POLICY['minimum_visible_adjacent_edges'] or not adjacent:
            # Keep the stored table, but do not certify it on this frame.
            return unknown(frame, 'insufficient_visible_border_evidence', self.epoch)
        expiry = min(float(self.good['confirmed_timestamp_s'])+POLICY['hold_seconds'],
                     now+POLICY['maximum_check_age_seconds'])
        result = deepcopy(self.good)
        uncertainty = max(shift, POLICY['minimum_corner_uncertainty_fraction']*diagonal(reference))
        uncertainty += POLICY['uncertainty_growth_px_at_640_per_second']*self.size[0]/640*age
        result.update(frame_id=int(frame), usable=True, state='partial_hold',
                      reason='recent_verified_table_with_visible_adjacent_edges',
                      memory_origin='bounded_history', memory_camera_epoch=self.epoch,
                      memory_source_frame=int(self.good['frame_id']), memory_age_seconds=age,
                      memory_checked_timestamp_s=now, current_perimeter=evidence,
                      visible_edges=visible, corner_uncertainty_px=uncertainty,
                      valid_until=int(np.searchsorted(self.times, expiry+1e-9, side='right')-1))
        return result


def timeline(samples, timestamps, cuts=()):
    result, at, active = [], 0, None
    samples = sorted(samples, key=lambda r:r['frame_id'])
    if len({r['frame_id'] for r in samples}) != len(samples):
        raise ValueError('Duplicate table sample')
    cutset = set(cuts)
    for frame, now in enumerate(timestamps):
        if frame in cutset:
            active = None
        while at < len(samples) and samples[at]['frame_id'] == frame:
            active = samples[at]
            at += 1
        okay = active and active.get('usable') and frame <= active.get('valid_until', active['frame_id'])
        if okay and 'memory_checked_timestamp_s' in active:
            okay = now-active['memory_checked_timestamp_s'] <= POLICY['maximum_check_age_seconds']+1e-9
        result.append(active if okay else unknown(frame, 'missing_failed_or_expired_geometry'))
    if at != len(samples):
        raise ValueError('Table sample outside video')
    return result


def replay(video, tracks, proposals, legacy_samples, size):
    raw = {s['frame_id']:s['proposal'] for s in proposals}
    legacy = {s['frame_id']:s for s in legacy_samples}
    if set(raw) != set(legacy) or len(raw) != len(proposals):
        raise ValueError('Use unchanged automatic sample cadence')
    memory = TableMemory([t['timestamp_s'] for t in tracks], size)
    samples = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot decode video')
    frame = 0
    try:
        while True:
            okay, image = cap.read()
            if not okay:
                break
            if frame >= len(tracks) or (image.shape[1], image.shape[0]) != tuple(size):
                raise ValueError('RGB and cache differ')
            cut = memory.observe_frame(frame, image)
            if frame in raw:
                samples.append(memory.update(frame, legacy[frame], raw[frame], BorderVerifier(image)))
            elif cut:
                samples.append(unknown(frame, 'camera_cut', memory.epoch))
            frame += 1
    finally:
        cap.release()
    if frame != len(tracks):
        raise ValueError('Actual decoded frame count differs')
    return samples, memory.cuts

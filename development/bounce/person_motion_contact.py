"""Additional review suggestions, not confirmed contact or production exclusion."""
import math
import cv2
import numpy as np
from development.bounce.wrist_evidence import iou

def person_box_visit(frames, seed):
    by_frame = {r['frame_id']: r for r in frames}
    visits = []
    def eligible(row):
        b = row['ball']
        return (b.get('detected') and b.get('confidence', 0) >= .75
                and math.isfinite(b['x']) and math.isfinite(b['y']))
    def distance(row, person):
        b = np.asarray(person['box'], float); xy = np.array([row['ball']['x'], row['ball']['y']])
        return float(np.linalg.norm(np.maximum(np.maximum(b[:2]-xy, xy-b[2:]), 0)))
    for fid in range(seed-1, seed+2):
        rows = [by_frame.get(f) for f in (fid-1, fid, fid+1)]
        if any(r is None or not eligible(r) for r in rows): continue
        for p in rows[1]['people']:
            if p['score'] < .9 or p['box'][2] <= p['box'][0] or p['box'][3] <= p['box'][1]: continue
            if distance(rows[1], p) != 0: continue
            matched = []
            for r in (rows[0], rows[2]):
                choices = [q for q in r['people'] if q['score'] >= .9 and iou(p['box'], q['box']) >= .3]
                matched.append(max(choices, key=lambda q: iou(p['box'], q['box'])) if choices else None)
            if any(q is None for q in matched): continue
            distances = [distance(rows[0], matched[0]), 0., distance(rows[2], matched[1])]
            if min(distances[0], distances[2]) >= 2:
                visits.append({'closest_frame': fid, 'person_box': p['box'],
                               'distance_px_before_at_after': distances})
    return {'supported': bool(visits), 'visits': visits, 'person_role_verified': False}

def local_table_position(samples, ball, frame):
    """Latest state wins, including invalidation; no backfill through future states."""
    eligible = [r for r in samples if r['frame_id'] <= frame]
    if not eligible: return {'available': False, 'reason': 'no_prior_geometry'}
    row = max(eligible, key=lambda r: r['frame_id'])
    result = {'available': False, 'geometry_frame': row['frame_id'],
              'epoch': row.get('epoch'), 'camera_epoch': row.get('camera_epoch')}
    if not row.get('usable') or frame > row.get('valid_until', row['frame_id']):
        return dict(result, reason='invalid_or_expired_geometry')
    if not ball.get('detected') or ball.get('confidence', 0) < .75:
        return dict(result, reason='weak_or_missing_ball')
    q = np.asarray(row['corners'], np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all() or not np.isfinite([ball['x'], ball['y']]).all():
        return dict(result, reason='invalid_coordinates')
    d = float(cv2.pointPolygonTest(q, (float(ball['x']), float(ball['y'])), True))
    return dict(result, available=True, signed_distance_px=d, outside_projection=d <= -2,
                physical_contact_height_known=False)

def review_suggestion(baseline_flag, direction, wrist_visit, body_visit, positions, proposal):
    outside = {r['frame_id'] for r in positions if r.get('available') and r.get('outside_projection')}
    kink = (proposal['smooth_to_kink_gain'] >= .35 and proposal['rms_improvement_px'] >= .3
            and proposal['velocity_change_px_per_nominal_interval'] >= 2
            and not proposal.get('tracking_spike_nearby', False))
    body = bool(kink and direction['strong_image_reversal']
                and direction['longitudinal_direction'] not in ('same_direction', 'uncertain')
                and any(v['closest_frame'] in outside for v in body_visit['visits']))
    wrist = bool(kink and any(v['closest_frame'] in outside for v in wrist_visit['visits']))
    return {'review_hit_suggestion': bool(baseline_flag or body or wrist),
            'baseline_flag': bool(baseline_flag), 'person_entry_reversal_outside': body,
            'transient_wrist_impulse_outside': wrist,
            'frozen_experimental_exclusion_flag': bool(baseline_flag),
            'production_exclusion_flag': False, 'adopt': False,
            'racket_contact_confirmed': False,
            'limitations': ['Person box is not a racket; unverified actor role.',
                           'Image table projection does not measure airborne height.',
                           'Suggestions require independent validation before exclusion.']}

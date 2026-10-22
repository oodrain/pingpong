"""Projected trajectory contact hypotheses, with explicit uncertainty and fallback.

This estimates an image-space change point. It does not recover ball height or
prove a table bounce. No event labels, manual points or camera files are inputs.
"""
import cv2
import numpy as np
try:
    from .table_memory import quad, diagonal, timeline
except ImportError:
    from table_memory import quad, diagonal, timeline

POLICY = dict(radius_frames=5, knot_radius_frames=1.5, knot_steps=13,
              minimum_points_per_side=3, minimum_confidence=.75,
              maximum_gap_intervals=2.5, maximum_support_seconds=.25,
              residual_limit_px_at_640=6., residual_limit_table_fraction=.015,
              relative_credible_rss_slack=.25, absolute_credible_rmse_slack_px_at_640=1.,
              boundary_margin_px_at_640=6., boundary_margin_table_fraction=.03,
              geometry_uncertainty_multiplier=2., fit_uncertainty_multiplier=2.,
              maximum_leave_one_out_shift_px_at_640=12.)


def actual(t):
    return bool(t.get('detected') and t.get('in_image', True) and
                (t.get('confidence') or 0) >= POLICY['minimum_confidence'] and
                all(t.get(k) is not None and np.isfinite(t[k]) for k in ('x','y')))


def design(t, quadratic):
    positive = np.maximum(t, 0)
    return np.c_[np.ones(len(t)), t, positive, t*t, positive*positive] if quadratic else np.c_[np.ones(len(t)), t, positive]


def fit_contact(tracks, seed, size, table_diagonal):
    times = np.asarray([r['timestamp_s'] for r in tracks], float)
    if len(times) != len(tracks) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need actual strictly increasing timestamps')
    unit = float(np.median(np.diff(times))) if len(times)>1 else .04
    radius = POLICY['radius_frames']
    ids = [i for i in range(max(0, seed-radius), min(len(tracks), seed+radius+1))
           if actual(tracks[i]) and abs(times[i]-times[seed]) <= POLICY['maximum_support_seconds']+1e-9]
    fail = dict(available=False, support_frames=ids, observation_points_filled=False)
    if len(ids) < 2*POLICY['minimum_points_per_side'] or np.max(np.diff(times[ids]), initial=0) > POLICY['maximum_gap_intervals']*unit:
        return dict(fail, reason='insufficient_continuous_observations')
    t = (times[ids]-times[seed])/unit
    xy = np.asarray([[tracks[i]['x'],tracks[i]['y']] for i in ids], float)
    weights = np.sqrt([tracks[i]['confidence'] for i in ids])[:,None]
    limit = max(POLICY['residual_limit_px_at_640']*size[0]/640,
                POLICY['residual_limit_table_fraction']*table_diagonal)
    choices = []
    for offset in np.linspace(-POLICY['knot_radius_frames'], POLICY['knot_radius_frames'], POLICY['knot_steps']):
        dt = t-offset
        if min(sum(dt<0), sum(dt>=0)) < POLICY['minimum_points_per_side']:
            continue
        for quadratic in (False, True):
            a = design(dt, quadratic)
            params, _, rank, _ = np.linalg.lstsq(a*weights, xy*weights, rcond=None)
            if rank < a.shape[1]:
                continue
            error = np.linalg.norm(a@params-xy, axis=1)
            rmse = float(np.sqrt(np.mean(error**2)))
            if rmse > limit or float(error.max()) > 2*limit:
                continue
            # A single questionable observation cannot pin a contact position.
            shifts = []
            for j in range(len(ids)):
                keep = np.arange(len(ids)) != j
                cp, _, rk, _ = np.linalg.lstsq(a[keep]*weights[keep], xy[keep]*weights[keep], rcond=None)
                if rk == a.shape[1]:
                    shifts.append(float(np.linalg.norm(cp[0]-params[0])))
            loo = max(shifts, default=float('inf'))
            if loo > POLICY['maximum_leave_one_out_shift_px_at_640']*size[0]/640:
                continue
            choices.append(dict(offset_frames=float(offset), time_s=float(times[seed]+offset*unit),
                                xy=params[0].tolist(), rmse_px=rmse, max_residual_px=float(error.max()),
                                leave_one_out_shift_px=loo, family='quadratic' if quadratic else 'linear',
                                rss=float(np.sum(error**2))))
    if not choices:
        return dict(fail, reason='no_stable_single_contact_fit')
    # Include every admissible family/knot near the best residual, rather than
    # cherry-picking a table-compatible point or claiming a calibrated posterior.
    best = min(choices, key=lambda a:(a['rmse_px'],a['family']=='quadratic',abs(a['offset_frames'])))
    slack = POLICY['absolute_credible_rmse_slack_px_at_640']*size[0]/640
    plausible = [c for c in choices if c['rmse_px'] <= max(best['rmse_px']*(1+POLICY['relative_credible_rss_slack']), best['rmse_px']+slack)]
    return dict(available=True, support_frames=ids, best=best, plausible=plausible,
                time_interval_s=[min(c['time_s'] for c in plausible),max(c['time_s'] for c in plausible)],
                observation_points_filled=False, physical_contact_confirmed=False,
                uncertainty_is_engineering_envelope=True)


def decide(point, tracks, states, size, cuts=()):
    frame = int(point['frame_id'])
    g = states[frame]
    result = dict(frame_id=frame, included=True, geometry_usable=bool(g.get('usable')),
                  original_point=dict(point), event_time_unchanged=True)
    if not g.get('usable'):
        return dict(result, reason='unknown_geometry_fallback')
    q = quad(g.get('corners'))
    if q is None:
        return dict(result, reason='invalid_geometry_fallback')
    lo, hi = max(0, frame-POLICY['radius_frames']), min(len(tracks)-1, frame+POLICY['radius_frames'])
    if any(lo < c <= hi for c in cuts):
        return dict(result, reason='camera_boundary_fallback')
    fit = fit_contact(tracks, frame, size, diagonal(q))
    result['contact_fit'] = fit
    if not fit['available']:
        return dict(result, reason=fit['reason'])
    # Keep a conservative local observed envelope as well: estimated airborne
    # coordinates at neighbouring frames cannot safely be treated as a bounce.
    locations = [c['xy'] for c in fit['plausible']]
    locations.extend([[tracks[j]['x'],tracks[j]['y']] for j in range(max(0,frame-1),min(len(tracks),frame+2)) if actual(tracks[j])])
    corner_error = float(g.get('corner_uncertainty_px', .003*diagonal(q)))
    fit_error = max(max(c['rmse_px'],c['leave_one_out_shift_px']) for c in fit['plausible'])
    margin = max(POLICY['boundary_margin_px_at_640']*size[0]/640, POLICY['boundary_margin_table_fraction']*diagonal(q))
    margin += POLICY['geometry_uncertainty_multiplier']*corner_error+POLICY['fit_uncertainty_multiplier']*fit_error
    distances = [-float(cv2.pointPolygonTest(q,(float(p[0]),float(p[1])),True)) for p in locations]
    excluded = min(distances) > margin+1e-9
    return dict(result, included=not excluded,
                reason='all_contact_hypotheses_outside_table' if excluded else 'inside_or_uncertain_contact_region',
                estimated_contact=fit['best'], minimum_outside_distance_px=min(distances),
                uncertainty_margin_px=margin, table_state=g.get('state'), table_sample_frame=g['frame_id'])


def review(points, tracks, samples, size, cuts=()):
    states = timeline(samples,[t['timestamp_s'] for t in tracks],cuts)
    decisions = [decide(p,tracks,states,size,cuts) for p in points]
    return [dict(p) for p,d in zip(points,decisions) if d['included']], decisions

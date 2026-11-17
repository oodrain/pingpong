"""Flag projected tracks that look like continuous flight near a candidate."""
from bisect import bisect_left, bisect_right

import numpy as np

from review_geometry.contact_review import actual


FLIGHT_HINTS = {
    "smooth_apex": "轨迹平滑经过高点",
    "smooth_pass": "轨迹连续通过，未见反弹转折",
}


def flight_hints(events, tracks, timestamps, width):
    support = [row for row in tracks if actual(row)]
    times = [row["timestamp_s"] for row in support]
    if len(times) < 7 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return {}
    # 640-pixel reference coordinates keep the tolerances independent of resolution.
    xy = np.array([(row["x"], row["y"]) for row in support]) * (640 / width)
    hints = {}
    for event in events:
        frame = event["frame_id"]
        seed = timestamps.get(frame)
        if seed is None or not np.isfinite(seed):
            continue
        lo, hi = bisect_left(times, seed - .16 - 1e-9), bisect_right(times, seed + .16 + 1e-9)
        nearby = np.asarray(times[lo:hi]) - seed
        hint = continuous_flight(nearby, xy[lo:hi], np.array([event["x"], event["y"]]) * (640 / width))
        if hint is not None:
            hints[frame] = hint
    return hints


def continuous_flight(times, points, candidate):
    if (len(times) < 7 or np.count_nonzero(times < -1e-6) < 3 or np.count_nonzero(times > 1e-6) < 3
            or times[0] > -.10 or times[-1] < .10 or np.min(abs(times)) > .027
            or np.max(np.diff(times)) > .071):
        return None
    t = times / .1
    design = np.column_stack((np.ones(len(t)), t, t * t))
    model = np.linalg.lstsq(design, points, rcond=None)[0]
    fitted = design @ model
    errors = np.linalg.norm(fitted - points, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    if rmse > 1.2 or np.max(errors) > 2.4 or np.linalg.norm(model[0] - candidate) > 3:
        return None
    # A bounce introduces an abrupt change. Compare with a model that allows
    # a slope break at or shortly around the candidate even if the arc fits.
    for knot in (-.4, -.2, 0., .2, .4):
        broken = np.column_stack((design, np.maximum(t - knot, 0.)))
        params = np.linalg.lstsq(broken, points, rcond=None)[0]
        kink_rmse = np.sqrt(np.mean(np.sum((broken @ params - points)**2, axis=1)))
        if np.linalg.norm(params[3]) > 3 and rmse - kink_rmse > .25:
            return None
    before, after = model[1] - 2 * model[2], model[1] + 2 * model[2]
    if model[2, 1] > .7 and before[1] < -1.6 and after[1] > 1.6:
        if abs(model[1, 1] / (2 * model[2, 1])) <= .5:
            return "smooth_apex"
    if np.linalg.norm(before) > 5 and np.linalg.norm(after - before) <= .20 * np.linalg.norm(before):
        return "smooth_pass"
    return None

"""Conservative frame refinement for the offline landing review view."""
import numpy as np

from review_geometry.contact_review import (
    actual, fit_contact,
)
from review_geometry.table_geometry import (
    TABLE_L, TABLE_W, GeometryError,
)


def adjacent_observations(events, tracks, plane, corners, size):
    """Return adjacent *observed* ball positions for supported changes of frame."""
    if not tracks or any(row["frame_id"] != index for index, row in enumerate(tracks)):
        return {}
    times = np.asarray([row["timestamp_s"] for row in tracks], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return {}
    table_diagonal = float(np.linalg.norm(np.ptp(np.asarray(corners), axis=0)))
    result = {}
    for index, event in enumerate(events):
        seed = event["frame_id"]
        if not 1 <= seed < len(tracks) - 1 or not actual(tracks[seed]):
            continue
        fit = fit_contact(tracks, seed, size, table_diagonal)
        if not fit["available"]:
            continue
        best = fit["best"]["offset_frames"]
        if not .65 <= abs(best) <= 1.25:
            continue
        direction = -1 if best < 0 else 1
        # Count unique time hypotheses so two model families at the same knot
        # cannot count as independent votes for moving the observation.
        votes = np.unique([row["offset_frames"] for row in fit["plausible"]]) * direction
        if np.median(votes) < .55 or np.mean(votes > 0) < .75:
            continue
        alternate = tracks[seed + direction]
        if not actual(alternate) or abs(alternate["timestamp_s"] - tracks[seed]["timestamp_s"]) > .08:
            continue
        try:
            first = plane.project(event["x"], event["y"])
            second = plane.project(alternate["x"], alternate["y"])
        except GeometryError:
            continue
        if not all(0 <= u <= TABLE_L and 0 <= v <= TABLE_W for u, v in (first, second)):
            continue
        if (first[0] < TABLE_L / 2) != (second[0] < TABLE_L / 2):
            continue
        result[f"prediction-{index}"] = {key: alternate[key] for key in ("frame_id", "x", "y")}
    return result

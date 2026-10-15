"""D geometry experiment repaired: chronological replay with explicit fallback.

Image-y extrema are candidates, not physical proof of table contact. This module
consumes detector observations and optional table geometry, never event labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from development.bounce.table_geometry import GeometryError, GeometryTimeline


@dataclass(frozen=True)
class RuleConfig:
    confidence: float = .75
    min_gap: int = 10
    fall: float = 2.0
    rise: float = 1.0
    margin_mm: float = 900.0
    duplicate_mm: float = 500.0

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("Invalid confidence threshold")
        if type(self.min_gap) is not int or self.min_gap < 1:
            raise ValueError("Invalid min_gap")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.fall, self.rise, self.margin_mm, self.duplicate_mm)):
            raise ValueError("Thresholds must be finite and nonnegative")


def replay(raw, geometry=None, mode="v0", config=None):
    if mode not in ("v0", "v1a", "v1b"):
        raise ValueError("Unknown rule mode")
    cfg = config or RuleConfig()
    timeline = geometry or GeometryTimeline()
    points = []
    previous = -1
    for row in raw:
        fid = row["input_frame_id"]
        if type(fid) is not int or fid < 0 or fid <= previous:
            raise ValueError("Raw input_frame_id must be nonnegative and strictly increasing")
        previous = fid
        if not row.get("detected"):
            continue
        confidence = row.get("confidence")
        if confidence is None:
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid detection confidence")
        if confidence < cfg.confidence:
            continue
        x, y = row.get("x"), row.get("y")
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Invalid detected coordinates")
        points.append({"frame_id": max(0, fid - 1), "x": float(x), "y": float(y)})
    predictions, decisions = [], []
    last_frame, last_mm, last_segment = -10000, None, None
    # A candidate is evaluated once when its following accepted point is known.
    for before, point, after in zip(points, points[1:], points[2:]):
        y = point["y"]
        if not (y > before["y"] and y >= after["y"] and
                y - before["y"] >= cfg.fall and y - after["y"] >= cfg.rise):
            continue
        fid = point["frame_id"]
        plane = timeline.lookup(fid) if mode != "v0" else None
        decision = dict(point, mode=mode, evaluated_at_frame=after["frame_id"],
                        geometry_available=plane is not None, outside_distance_mm=None,
                        segment_id=plane.segment_id if plane else None,
                        geometry_axis_source=plane.axis_source if plane else None)
        projected = None
        reason = None
        if mode != "v0":
            context = [timeline.interval_id(p["frame_id"]) for p in (before, point, after)]
            if len({sid for sid in context if sid is not None}) > 1:
                reason = "geometry_segment_transition"
            if plane is not None and reason is None:
                try:
                    projected = plane.project(point["x"], y)
                    outside = plane.outside_distance(point["x"], y)
                    decision["outside_distance_mm"] = outside
                    if outside > cfg.margin_mm:
                        reason = "outside_table_margin"
                except GeometryError:
                    plane = None
                    decision.update(geometry_available=False, segment_id=None)
        if reason is None and fid - last_frame < cfg.min_gap:
            spatial_exception = (mode == "v1b" and fid > last_frame and plane is not None
                                 and plane.axis_source != "ambiguous_conservative"
                                 and last_mm is not None and last_segment == plane.segment_id
                                 and math.dist(projected, last_mm) > cfg.duplicate_mm)
            if not spatial_exception:
                reason = "temporal_dedup"
        if reason is None:
            predictions.append(dict(point))
            last_frame = fid
            last_mm = projected if plane else None
            last_segment = plane.segment_id if plane else None
            reason = "accepted_geometry" if plane else "accepted_v0" if mode == "v0" else "accepted_fallback"
        decision.update(accepted=reason.startswith("accepted_"), reason=reason)
        decisions.append(decision)
    return predictions, decisions

"""Validated table-plane geometry, in millimetres; does not estimate ball height."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

TABLE_L, TABLE_W = 2740.0, 1525.0


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class TablePlane:
    matrix: np.ndarray
    segment_id: str
    axis_source: str
    width_mm: float = TABLE_L
    height_mm: float = TABLE_W

    def project(self, x, y):
        p = self.matrix @ np.array([x, y, 1.0], dtype=float)
        if not np.isfinite(p).all() or abs(p[2]) < 1e-10:
            raise GeometryError("Point projects to infinity")
        u, v = p[:2] / p[2]
        return float(u), float(v)

    def outside_distance(self, x, y):
        u, v = self.project(x, y)
        # Same L-infinity expansion as D, with consistent canonical dimensions.
        return max(-u, u - self.width_mm, -v, v - self.height_mm, 0.0)


def build_plane(segment, segment_id="static"):
    """Corners must be cyclic. Edge 0->1 needs an axis label or a usable net.

    Rotating the corner order changes which edge is the length. A valid net in
    unit-square coordinates runs across the width at half the table length.
    Screen edge lengths cannot establish the physical axes under perspective.
    """
    try:
        corners = np.asarray(segment["corners"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryError("Missing/malformed corners") from exc
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise GeometryError("Expected four finite cyclic corners")
    q = corners.astype(np.float32)
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 1.0:
        raise GeometryError("Corners are crossed or degenerate")
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    unit_h = cv2.getPerspectiveTransform(q, unit)
    declared = segment.get("edge01_axis")
    if declared not in (None, "length", "width"):
        raise GeometryError("edge01_axis must be length or width")
    inferred = None
    if segment.get("net") is not None:
        try:
            net = np.asarray(segment["net"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise GeometryError("Malformed net endpoints") from exc
        if net.shape != (2, 2) or not np.isfinite(net).all():
            raise GeometryError("Expected two finite net endpoints")
        homogeneous = np.column_stack([net, np.ones(2)]) @ unit_h.T
        if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
            raise GeometryError("Net projects to infinity")
        points = homogeneous[:, :2] / homogeneous[:, 2, None]
        dx, dy = np.abs(points[1] - points[0])
        if dy >= .4 and dx <= .2 * dy and np.max(np.abs(points[:, 0] - .5)) <= .2:
            inferred = "length"
        elif dx >= .4 and dy <= .2 * dx and np.max(np.abs(points[:, 1] - .5)) <= .2:
            inferred = "width"
        else:
            raise GeometryError("Net does not establish a transverse centre line")
        if declared is not None and declared != inferred:
            raise GeometryError("Axis label and net disagree")
    axis = declared or inferred
    if axis is None:
        raise GeometryError("Unknown physical axes: provide edge01_axis or net")
    if axis == "width":
        q = np.roll(q, -1, axis=0)
    target = np.array([[0, 0], [TABLE_L, 0], [TABLE_L, TABLE_W], [0, TABLE_W]], np.float32)
    matrix = cv2.getPerspectiveTransform(q, target)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise GeometryError("Singular homography")
    return TablePlane(matrix, str(segment_id), "declared" if declared else "net")


class GeometryTimeline:
    """Explicit inclusive ranges in output frame coordinates, not raw input IDs.

    Invalid segment geometry is unknown and falls back. Malformed/overlapping
    ranges fail immediately so camera segments cannot silently shadow each other.
    """
    def __init__(self, document=None):
        self.segments = []
        self.issues = []
        if document is None:
            self.issues.append({"reason": "missing_geometry"})
            return
        segments = document.get("segments")
        if not isinstance(segments, list) or not segments:
            raise GeometryError("Geometry document requires nonempty segments")
        previous_end = -1
        for index, segment in enumerate(segments):
            if len(segments) > 1 and not {"start_frame", "end_frame"} <= segment.keys():
                raise GeometryError("Multiple segments require explicit start_frame/end_frame")
            start = segment.get("start_frame", 0)
            end = segment.get("end_frame")
            if type(start) is not int or start < 0 or (end is not None and
                    (type(end) is not int or end < start)):
                raise GeometryError("Invalid inclusive segment range")
            if start <= previous_end:
                raise GeometryError("Overlapping or unordered segments")
            if end is None and index != len(segments) - 1:
                raise GeometryError("Only final segment may have an open end")
            sid = str(segment.get("id", index))
            if any(s[2] == sid for s in self.segments):
                raise GeometryError("Duplicate segment id")
            try:
                plane = build_plane(segment, sid)
            except GeometryError as exc:
                plane = None
                self.issues.append({"segment_id": sid, "reason": str(exc)})
            self.segments.append((start, end, sid, plane))
            previous_end = math.inf if end is None else end

    def lookup(self, frame_id):
        for start, end, _, plane in self.segments:
            if start <= frame_id and (end is None or frame_id <= end):
                return plane
        return None

    def interval_id(self, frame_id):
        for start, end, sid, _ in self.segments:
            if start <= frame_id and (end is None or frame_id <= end):
                return sid
        return None

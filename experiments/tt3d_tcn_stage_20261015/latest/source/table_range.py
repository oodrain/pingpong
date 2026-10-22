"""Offline fixed-camera table range, selected only from automatic observations."""
import math

import cv2
import numpy as np


POLICY = {
    'schema': 'fixed_video_table_range_v1',
    'selection': 'highest quality usable automatic proposal in the entire video; earliest frame breaks ties',
    'camera_assumption': 'one fixed camera and one stationary table per input clip',
    'margin_diagonal_fraction': .02,
    'minimum_margin_px': 4.,
    'unknown_geometry': 'keep original predictions',
    'scope': 'table_bounce only; hits and neural features are unchanged',
}


def select_fixed_table(observations, image_size):
    """Reject failed/malformed proposals before ranking their quality heuristic.

    A usable proposal has already passed the segmentation and RGB border checks.
    Per-frame consensus expiry is irrelevant to this whole-clip fixed-camera gate.
    """
    width, height = image_size
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('Expected a positive finite image size')
    candidates = []
    for row in observations:
        proposal = row.get('proposal', {})
        if proposal.get('usable_proposal') is not True:
            continue
        try:
            frame = row['frame_id']
            quality = float(proposal['quality'])
            q = np.asarray(proposal['corners'], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if (type(frame) is not int or frame < 0 or not math.isfinite(quality)
                or not 0 <= quality <= 1 or q.shape != (4, 2) or not np.isfinite(q).all()
                or not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) <= 1
                or np.any(q < 0) or np.any(q[:, 0] > width) or np.any(q[:, 1] > height)):
            continue
        candidates.append((quality, -frame, q, proposal))
    if not candidates:
        return {'available': False, 'reason': 'no_usable_automatic_table',
                'candidate_count': 0, 'corners': None}
    quality, negative_frame, q, proposal = max(candidates, key=lambda item: item[:2])
    diagonal = float(np.linalg.norm(np.ptp(q, axis=0)))
    return {'available': True, 'source_frame': -negative_frame, 'quality': quality,
            'candidate_count': len(candidates), 'corners': q.tolist(),
            'diagonal_px': diagonal, 'mask_iou': proposal.get('mask_iou'),
            'edge_support': proposal.get('edge_support'),
            'fitting_source': proposal.get('fitting_source')}


def constrain_landings(points, observations, image_size, *,
                       margin_fraction=POLICY['margin_diagonal_fraction'],
                       minimum_margin_px=POLICY['minimum_margin_px']):
    """Keep points inside/on the polygon or within a Euclidean edge tolerance.

    Missing geometry preserves all original predictions; it never means outside.
    Coordinates and event times are not moved or refitted.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (margin_fraction, minimum_margin_px)):
        raise ValueError('Expected finite nonnegative edge tolerances')
    selected = select_fixed_table(observations, image_size)
    margin = max(minimum_margin_px, margin_fraction * selected['diagonal_px']) if selected['available'] else None
    polygon = np.asarray(selected['corners'], np.float32) if selected['available'] else None
    kept, audit = [], []
    for point in points:
        xy = float(point['x']), float(point['y'])
        if not all(math.isfinite(v) for v in xy):
            raise ValueError('Nonfinite landing coordinates')
        distance = float(cv2.pointPolygonTest(polygon, xy, True)) if polygon is not None else None
        included = distance is None or distance >= -margin - 1e-6
        if included:
            kept.append(dict(point))
        reason = ('unknown_table_kept' if distance is None else 'outside_fixed_table'
                  if not included else 'inside_table' if distance >= 0 else 'edge_tolerance')
        audit.append(dict(point, included=included, signed_edge_distance_px=distance, reason=reason))
    return kept, {'policy': dict(POLICY, margin_diagonal_fraction=margin_fraction,
                                minimum_margin_px=minimum_margin_px),
                  'selected_table': selected, 'margin_px': margin,
                  'input_count': len(points), 'kept_count': len(kept),
                  'rejected_count': len(points) - len(kept), 'events': audit}

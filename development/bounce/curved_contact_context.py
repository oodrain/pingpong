"""Automatic projection and longer-flight context, without physical-axis claims."""
import cv2
import numpy as np
from .curved_contact_candidates import fit_curved_window
from .curved_contact_classifier import features, EXTENDED_FEATURES, CONTEXT_FEATURES, PHYSICS_FEATURES,LEGACY_PHYSICS_FEATURES,NET_FEATURES
from .trajectory_events import observed_runs

def contextual_features(proposal, position, corners, tracks):
    """Coordinates lie on the image plane mapped to a unit square, not world Z."""
    vector = features(proposal, position, corners, EXTENDED_FEATURES).tolist()
    seed = proposal['frame_id']
    q = np.asarray(corners, np.float32)
    plane = cv2.getPerspectiveTransform(q, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    ball = tracks[seed]
    uv = cv2.perspectiveTransform(np.array([[[ball['x'], ball['y']]]], np.float32), plane)[0, 0]
    if not np.isfinite(uv).all():
        raise ValueError('Nonfinite rectified ball projection')
    distances = sorted(np.clip(np.abs(uv - .5), 0, 3).tolist())
    wide = None
    run = next((r for r in observed_runs(tracks) if r[0]['frame_id'] <= seed <= r[-1]['frame_id']), None)
    if run:
        index = next(i for i, row in enumerate(run) if row['frame_id'] == seed)
        if 4 <= index < len(run) - 4:
            wide = fit_curved_window(run[index-4:index+5], seed, np.arange(-1.5, 1.501, .25))
    vector += distances + [float(wide is None),
                           abs(wide['contact_offset_intervals']) if wide else 0.,
                           float(np.log((1+wide['fit_rmse_px'])/(1+proposal['fit_rmse_px']))) if wide else 0.]
    return np.asarray(vector, float)


def ballistic_features(proposal,position,corners,tracks,samples,image_size,feature_names=PHYSICS_FEATURES):
    from .automatic_ballistic_contact import automatic_ballistic_evidence
    if image_size is None:raise ValueError('Automatic ballistic model requires decoded video dimensions')
    vector=contextual_features(proposal,position,corners,tracks).tolist()
    evidence=automatic_ballistic_evidence(tracks,samples,proposal['frame_id'],image_size)
    if evidence['available']:
        best=evidence['best'];vector += [0.,float(np.clip(np.log1p(best['fit_rmse_px']),0,5)),float(best['opposing_vertical_velocity_margin'])]
    else:vector += [1.,0.,0.]
    if feature_names==PHYSICS_FEATURES:
        vector.append(max(f['opposing_vertical_velocity_margin'] for f in evidence['hypotheses']) if evidence['available'] else 0.)
    elif feature_names!=LEGACY_PHYSICS_FEATURES:raise ValueError('Unsupported ballistic feature schema')
    return np.asarray(vector,float)


def net_context_features(proposal,position,corners,tracks,samples,image_size):
    from .automatic_net_zone import net_zone_evidence
    vector=ballistic_features(proposal,position,corners,tracks,samples,image_size).tolist()
    evidence=net_zone_evidence(tracks,samples,proposal['frame_id'],image_size)
    if evidence['available']:
        vector += [0.,float(np.clip(evidence['maximum_signed_mesh_distance_fraction'],-1,1)),float(np.clip(evidence['minimum_top_distance_fraction'],0,1))]
    else:vector += [1.,0.,0.]
    return np.asarray(vector,float)

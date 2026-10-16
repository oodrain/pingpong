"""Previous fixed camera acceptance, reused by the consensus diagnostic."""
import numpy as np
from functools import lru_cache
from .physics import POLICY
from .camera import estimate_camera
DECODER={"maximum_camera_history_s":.6}

def cameras(corners,size):return estimate_camera(np.array(corners).reshape(4,2),size)['candidates']

def camera_at(geometry,tracks,i,size):
    prior=[g for g in geometry if g['frame_id']<=i];g=prior[-1] if prior else None
    if not g or not g.get('usable') or i>g.get('valid_until',g['frame_id']):return [],dict(reason='current_geometry_unknown')
    tau=tracks[i]['timestamp_s'];past=[q for q in prior if q.get('usable') and q.get('epoch')==g.get('epoch') and 0<=tau-tracks[q['frame_id']]['timestamp_s']<=DECODER['maximum_camera_history_s']]
    if len(past)<POLICY['camera_minimum_past_stable_samples']:return [],dict(reason='insufficient_stable_camera_history')
    q=np.array(g['corners']);scale=max(np.linalg.norm(np.ptp(q,axis=0)),1);drift=max(np.linalg.norm(np.asarray(a['corners'])-q,axis=1).max()/scale for a in past[-3:])
    if drift>POLICY['camera_maximum_corner_drift_fraction']:return [],dict(reason='automatic_corners_temporally_unstable',drift_fraction=float(drift))
    allc=cameras(tuple(q.ravel()),tuple(size));usable=[c for c in allc if c['usable']]
    return usable,dict(reason='conditional_camera_hypotheses' if usable else 'no_usable_metric_camera',geometry_frame=g['frame_id'],epoch=g.get('epoch'),drift_fraction=float(drift),camera_candidates=[dict(axis=c['edge01_axis'],usable=c['usable'],reason=c['reason']) for c in allc])

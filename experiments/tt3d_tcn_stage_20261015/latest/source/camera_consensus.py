"""Causal automatic-corner consensus with an unused latest-frame check."""
from . import physics as gravity
from . import camera_history as old_pipeline
import numpy as np
from functools import lru_cache
from .camera import estimate_camera

POLICY=dict(history_seconds=.8,maximum_hold_seconds=.16,minimum_samples=3,
    minimum_span_seconds=.12,minimum_quality=.90,minimum_mask_iou=.85,
    maximum_corner_drift_fraction=.02,held_out_corner_rmse_diagonal_fraction=.003,
    held_out_corner_maximum_diagonal_fraction=.006)
@lru_cache(maxsize=4096)
def estimate(corners,size):return estimate_camera(np.array(corners).reshape(4,2),size)['candidates']
def camera_at(geometry,tracks,i,size):
    old,oldcheck=old_pipeline.camera_at(geometry,tracks,i,size)
    prior=[g for g in geometry if g['frame_id']<=i]
    if not prior:return [],dict(reason='no_camera_history',old_available=bool(old))
    latest=prior[-1];now=tracks[i]['timestamp_s'];epoch=latest.get('camera_epoch',latest.get('epoch'))
    # A held geometry carries the old proposal's quality/corners. It is not a
    # fresh four-corner observation, even when its perimeter was checked now.
    history=[g for g in prior if g.get('last_good_frame',g['frame_id'])==g['frame_id'] and g.get('state')!='held' and
        g.get('camera_epoch',g.get('epoch'))==epoch and now-tracks[g['frame_id']]['timestamp_s']<=POLICY['history_seconds'] and
        g.get('corners') and g.get('usable_proposal',g.get('usable',False)) and g.get('quality',0)>=POLICY['minimum_quality'] and g.get('mask_iou',0)>=POLICY['minimum_mask_iou']]
    if len(history)<POLICY['minimum_samples']:return old,dict(reason='original_camera' if old else 'insufficient_consensus_history',old_available=bool(old),original=oldcheck)
    history=history[-6:];hold=now-tracks[history[-1]['frame_id']]['timestamp_s'];span=tracks[history[-1]['frame_id']]['timestamp_s']-tracks[history[0]['frame_id']]['timestamp_s']
    if hold>POLICY['maximum_hold_seconds'] or span<POLICY['minimum_span_seconds']:return old,dict(reason='original_camera' if old else 'consensus_expired_or_short',old_available=bool(old),original=oldcheck)
    # Latest accepted proposal is not part of the calibration fit.
    q=np.median(np.asarray([g['corners'] for g in history[:-1]]),axis=0);scale=max(1.,np.linalg.norm(np.ptp(q,axis=0)))
    drift=max(np.max(np.linalg.norm(np.asarray(g['corners'])-q,axis=1))/scale for g in history)
    if drift>POLICY['maximum_corner_drift_fraction']:return old,dict(reason='original_camera' if old else 'consensus_corner_instability',old_available=bool(old),drift_fraction=float(drift),original=oldcheck)
    allc=estimate(tuple(q.ravel()),tuple(size));cs=[];checks=[];held=np.array(history[-1]['corners'])
    for c in allc:
        passed=False;rmse=None;maximum=None
        if c['usable']:
            errors=np.linalg.norm(np.array(c['projected_corners'])-held,axis=1);rmse=float(np.sqrt(np.mean(errors**2)));maximum=float(max(errors))
            passed=rmse<=max(3.,np.linalg.norm(size)*POLICY['held_out_corner_rmse_diagonal_fraction']) and maximum<=max(6.,np.linalg.norm(size)*POLICY['held_out_corner_maximum_diagonal_fraction'])
        checks.append(dict(axis=c['edge01_axis'],calibration_reason=c['reason'],heldout_rmse_px=rmse,heldout_maximum_px=maximum,passed=bool(passed)))
        if passed:cs.append(c)
    if not cs:return old,dict(reason='original_camera' if old else 'consensus_calibration_not_observable',old_available=bool(old),checks=checks,original=oldcheck)
    return cs,dict(reason='past_consensus_heldout_checked',old_available=bool(old),fit_frames=[g['frame_id'] for g in history[:-1]],heldout_frame=history[-1]['frame_id'],hold_seconds=float(hold),drift_fraction=float(drift),checks=checks,
        validation_scope='automatic 2D temporal consistency only, not independent intrinsic or metric certification',axis_ambiguity_preserved=True)

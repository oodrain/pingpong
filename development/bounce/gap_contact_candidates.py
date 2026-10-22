"""Fit contacts near one or two missing observations without filling tracks.

Uses four measured positions before and after a bounded gap, actual PTS,
shared image curvature, and a searched velocity jump. Estimated contact
coordinates are hypotheses; they are never reported as detector observations.
"""
import cv2
import numpy as np
from .curved_contact_candidates import fit_curved_window


def propose_gap_contacts(tracks,corners,maximum_gap=2):
    q=np.asarray(corners,float)
    if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError('Invalid gap-fit table projection')
    scale=float(np.linalg.norm(np.ptp(q,axis=0)))
    axes=np.array([q[1]-q[0],q[3]-q[0]]);lengths=np.linalg.norm(axes,axis=1)
    if q.shape!=(4,2) or scale<=0 or min(lengths)<=0:raise ValueError('Invalid gap-fit table projection')
    axes/=lengths[:,None]
    valid=[bool(r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=.75
                and r.get('x') is not None and r.get('y') is not None
                and np.isfinite([r['x'],r['y']]).all()) for r in tracks]
    candidates=[];i=0
    while i<len(tracks):
        if valid[i]:i+=1;continue
        lo=i
        while i<len(tracks) and not valid[i]:i+=1
        hi=i-1
        if hi-lo+1>maximum_gap or lo<4 or hi+4>=len(tracks):continue
        before=tracks[lo-4:lo];after=tracks[hi+1:hi+5]
        if not all(valid[lo-4:lo]) or not all(valid[hi+1:hi+5]):continue
        rows=before+after;times=np.array([r['timestamp_s'] for r in rows]);unit=float(np.median(np.diff(times)))
        if unit<=0 or np.any(np.diff(times)<=0):continue
        # A large timing edit is not a missing ball observation.
        if tracks[hi+1]['timestamp_s']-tracks[lo-1]['timestamp_s']>(maximum_gap+1.1)*unit:continue
        xy=np.array([[r['x'],r['y']] for r in rows])
        if np.linalg.norm(xy[4]-xy[3])>.5*scale:continue
        anchor=lo-1;span=(tracks[hi+1]['timestamp_s']-tracks[anchor]['timestamp_s'])/unit
        fit=fit_curved_window(rows,anchor,np.arange(-.5,span+.501,.25))
        if (fit['fit_rmse_px']>2 or fit['bic_gain']<6 or fit['smooth_to_kink_gain']<.35
            or fit['rms_improvement_px']<.3 or fit['velocity_change_px_per_nominal_interval']<max(2,.01*scale)):
            continue
        fractional=float(np.interp(fit['contact_time_s'],[r['timestamp_s'] for r in tracks],[r['frame_id'] for r in tracks]))
        seed=int(np.floor(fractional+.5))
        if not max(0,lo-2)<=seed<=min(len(tracks)-1,hi+2):continue
        # Re-express subframe offset around the output frame, not the fit anchor.
        fit['contact_offset_intervals']=float((fit['contact_time_s']-tracks[seed]['timestamp_s'])/unit)
        d=float(cv2.pointPolygonTest(q.astype(np.float32),tuple(map(float,fit['position_px'])),True))
        fit.update(frame_id=seed,proposal_method='bounded_gap_two_measured_flights',
            estimated_position_px=fit['position_px'],missing_frame_ids=list(range(lo,hi+1)),
            measured_frame_ids=[r['frame_id'] for r in rows],tracking_spike_nearby=False,
            signed_table_polygon_distance_px=d,region='table_projection' if d>=0 else 'outside_table_projection',
            table_axis_image_components_before=(axes@np.asarray(fit['velocity_before_px_s'])*unit).tolist(),
            table_axis_image_components_after=(axes@np.asarray(fit['velocity_after_px_s'])*unit).tolist(),
            support_minimum_confidence=min(r['confidence'] for r in rows),observations_interpolated=False,
            evaluated_at_frame=rows[-1]['frame_id'],world_contact_confirmed=False)
        candidates.append(fit)
    return candidates

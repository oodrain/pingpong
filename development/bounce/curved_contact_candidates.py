"""Experimental image-space curved-flight contact proposals.

Fit shared curvature and a velocity jump instead of comparing a smooth curve
with two straight lines. No image-y sign rule, event labels, or 3D claims.
The module is not enabled in the production or video-only pipeline.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from .trajectory_events import observed_runs,EventConfig


@dataclass(frozen=True)
class CurvedContactConfig:
    half_window: int=3
    minimum_gain: float=.35
    minimum_rms_improvement_px: float=.3
    minimum_bic_gain: float=6.
    minimum_velocity_change_px: float=2.
    maximum_contact_rmse_px: float=3.
    merge_radius: int=2


def fit_curved_window(rows,centre_frame,offset_grid=(-.5,-.25,0.,.25,.5)):
    centre=next(r for r in rows if r['frame_id']==centre_frame)
    dt=np.diff([r['timestamp_s'] for r in rows]);unit=float(np.median(dt))
    if unit<=0 or not np.isfinite(unit):raise ValueError('Invalid timestamp cadence')
    t=np.array([(r['timestamp_s']-centre['timestamp_s'])/unit for r in rows]);xy=np.array([[r['x'],r['y']] for r in rows])
    smooth=np.column_stack((np.ones(len(t)),t,t*t))
    smooth_sse=float(np.sum((smooth@np.linalg.lstsq(smooth,xy,rcond=None)[0]-xy)**2))
    cubic=np.column_stack((smooth,t*t*t))
    cubic_sse=float(np.sum((cubic@np.linalg.lstsq(cubic,xy,rcond=None)[0]-xy)**2))
    choices=[]
    for tau in offset_grid:
        if min(np.sum(t<tau),np.sum(t>=tau))<3:continue
        design=np.column_stack((smooth,np.maximum(t-tau,0)))
        params=np.linalg.lstsq(design,xy,rcond=None)[0]
        sse=float(np.sum((design@params-xy)**2))
        choices.append((sse,tau,params))
    if not choices:raise ValueError('Contact fit needs three actual observations on each side')
    sse,tau,params=min(choices,key=lambda c:c[0]);n=2*len(rows)
    # Two jump components and one searched contact time, versus smooth flight.
    bic_gain=n*math.log(max(smooth_sse,1e-9)/max(sse,1e-9))-3*math.log(n)
    cubic_bic_gain=n*math.log(max(cubic_sse,1e-9)/max(sse,1e-9))-math.log(n)
    before=params[1]+2*tau*params[2];after=before+params[3]
    position=params[0]+params[1]*tau+params[2]*tau*tau
    return {'centre_frame':centre_frame,'contact_offset_intervals':tau,
            'contact_time_s':centre['timestamp_s']+tau*unit,'nominal_interval_s':unit,
            'position_px':position.tolist(),'velocity_before_px_s':(before/unit).tolist(),
            'velocity_after_px_s':(after/unit).tolist(),'velocity_change_px_per_nominal_interval':float(np.linalg.norm(params[3])),
            'smooth_to_kink_gain':(smooth_sse-sse)/max(smooth_sse,1e-9),
            'rms_improvement_px':math.sqrt(smooth_sse/len(rows))-math.sqrt(sse/len(rows)),
            'fit_rmse_px':math.sqrt(sse/len(rows)),'bic_gain':bic_gain,'score':smooth_sse-sse,
            'smooth_cubic_rmse_px':math.sqrt(cubic_sse/len(rows)),'bic_gain_over_smooth_cubic':cubic_bic_gain,
            'shared_image_curvature':params[2].tolist(),'window_frame_bounds':[rows[0]['frame_id'],rows[-1]['frame_id']]}


def propose_curved_contacts(tracks,corners,config=CurvedContactConfig()):
    q=np.asarray(corners,float)
    if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError('Invalid table quadrilateral')
    scale=float(np.linalg.norm(np.ptp(q,axis=0)));axes=np.array([q[1]-q[0],q[3]-q[0]])
    if scale<=0 or min(np.linalg.norm(axes,axis=1))<=0:raise ValueError('Degenerate quadrilateral')
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    runs=observed_runs(tracks,EventConfig());candidates=[];spikes=[];h=config.half_window
    for run in runs:
        for a,b,c in zip(run,run[1:],run[2:]):
            p0,p,p1=[np.array([r['x'],r['y']]) for r in (a,b,c)]
            weight=(b['timestamp_s']-a['timestamp_s'])/(c['timestamp_s']-a['timestamp_s'])
            residual=float(np.linalg.norm(p-((1-weight)*p0+weight*p1)))
            if residual>max(20,.08*scale) and np.linalg.norm(p1-p0)<.5*residual:spikes.append(b['frame_id'])
        for i in range(h,len(run)-h):
            fit=fit_curved_window(run[i-h:i+h+1],run[i]['frame_id'])
            if (fit['smooth_to_kink_gain']<config.minimum_gain or fit['rms_improvement_px']<config.minimum_rms_improvement_px
                or fit['bic_gain']<config.minimum_bic_gain or fit['fit_rmse_px']>config.maximum_contact_rmse_px
                or fit['velocity_change_px_per_nominal_interval']<max(config.minimum_velocity_change_px,.01*scale)):
                continue
            centre=run[i];distance=float(cv2.pointPolygonTest(q.astype(np.float32),tuple(map(float,fit['position_px'])),True))
            fit.update(frame_id=centre['frame_id'],region='table_projection' if distance>=-max(3,.01*scale) else 'outside_table_projection',
                       signed_table_polygon_distance_px=distance,
                       table_axis_image_components_before=(axes@np.asarray(fit['velocity_before_px_s'])*fit['nominal_interval_s']).tolist(),
                       table_axis_image_components_after=(axes@np.asarray(fit['velocity_after_px_s'])*fit['nominal_interval_s']).tolist(),
                       tracking_spike_nearby=any(abs(centre['frame_id']-s)<=h for s in spikes),
                       evaluated_at_frame=run[i+h]['frame_id'],proposal_method='shared_image_curvature_velocity_jump',
                       world_contact_confirmed=False)
            candidates.append(fit)
    chosen=[]
    for c in sorted(candidates,key=lambda c:c['bic_gain'],reverse=True):
        if all(abs(c['frame_id']-r['frame_id'])>config.merge_radius for r in chosen):chosen.append(c)
    return {'proposals':sorted(chosen,key=lambda c:c['frame_id']),'possible_tracking_spikes':sorted(set(spikes)),
            'observed_runs':[[r[0]['frame_id'],r[-1]['frame_id']] for r in runs]}

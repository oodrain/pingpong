"""A table-height ballistic hypothesis from automatic corners and actual ball points.

Camera assumptions are unverified. These are diagnostic features, never proof of
contact or a substitute for detections. Both physical table-axis assignments are
estimated; no manually reviewed axes, net points, or camera files are consumed.
"""
from functools import lru_cache
import cv2
import numpy as np
from development.detection.tt3d_camera_estimation import estimate_camera
from .trajectory_events import observed_runs


@lru_cache(maxsize=256)
def camera_hypotheses(corners, image_size):
    return estimate_camera(np.asarray(corners).reshape(4, 2), image_size)['candidates']


def fit_ballistic_contact(rows, centre_frame, camera):
    centre=next(r for r in rows if r['frame_id']==centre_frame)
    times=np.array([r['timestamp_s']-centre['timestamp_s'] for r in rows]);unit=float(np.median(np.diff(times)))
    xy=np.array([[r['x'],r['y']] for r in rows],float)
    if unit<=0 or not np.isfinite(xy).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid ballistic observations')
    k=np.array(camera['K']);rotation=np.array(camera['rotation_world_to_camera']);translation=np.array(camera['tvec_m'])
    rays=(np.linalg.inv(k)@np.column_stack((xy,np.ones(len(xy)))).T).T
    choices=[]
    for offset in (-.5,-.25,0.,.25,.5):
        t=times-offset*unit
        if min(sum(t<0),sum(t>=0))<3:continue
        a=[];b=[];matrices=[];known=[]
        for dt,ray in zip(t,rays):
            basis=np.zeros((3,6));basis[0,[0,2]]=[1,dt];basis[1,[1,3]]=[1,dt];basis[2,[4,5]]=[dt,max(dt,0)]
            constant=rotation@np.array([0,0,-.5*9.81*dt*dt])+translation
            projected=rotation@basis;cross=np.array([[1,0,-ray[0]/ray[2]],[0,1,-ray[1]/ray[2]]])
            a.extend(cross@projected);b.extend(-cross@constant);matrices.append(projected);known.append(constant)
        parameters,_,rank,_=np.linalg.lstsq(np.asarray(a),np.asarray(b),rcond=None)
        if rank<6:continue
        camera_points=np.array([m@parameters+c for m,c in zip(matrices,known)])
        if np.any(camera_points[:,2]<=0):continue
        uv=(k@camera_points.T).T;uv=uv[:,:2]/uv[:,2,None]
        rmse=float(np.sqrt(np.mean(np.sum((uv-xy)**2,axis=1))))
        before=float(parameters[4]);after=float(parameters[4]+parameters[5])
        choices.append({'fit_rmse_px':rmse,'contact_offset_intervals':offset,
            'contact_time_s':centre['timestamp_s']+offset*unit,
            'vertical_velocity_before_m_s':before,'vertical_velocity_after_m_s':after,
            'opposing_vertical_velocity_margin':min(-before,after)/(abs(before)+abs(after)+1),
            'horizontal_velocity_m_s':parameters[2:4].tolist(),'contact_xy_m':parameters[:2].tolist(),
            'world_contact_confirmed':False,'gravity_m_s2':9.81})
    return min(choices,key=lambda r:r['fit_rmse_px']) if choices else None


def automatic_ballistic_evidence(tracks, samples, seed, image_size):
    geometry=max((s for s in samples if s['frame_id']<=seed),key=lambda s:s['frame_id'],default=None)
    if geometry is None or not geometry.get('usable') or seed>geometry.get('valid_until',geometry['frame_id']):
        return {'available':False,'reason':'current_geometry_unusable'}
    run=next((r for r in observed_runs(tracks) if r[0]['frame_id']<=seed<=r[-1]['frame_id']),None)
    if run is None:return {'available':False,'reason':'ball_support_missing'}
    index=next(i for i,r in enumerate(run) if r['frame_id']==seed)
    if index<3 or index+3>=len(run):return {'available':False,'reason':'insufficient_actual_ball_support'}
    try:
        cameras=camera_hypotheses(tuple(np.asarray(geometry['corners']).ravel()),tuple(image_size))
    except (ValueError,cv2.error,np.linalg.LinAlgError):
        return {'available':False,'reason':'camera_estimation_failed'}
    fits=[]
    for camera in cameras:
        if not camera['usable']:continue
        fit=fit_ballistic_contact(run[index-3:index+4],seed,camera)
        if fit is not None:fits.append(dict(fit,table_edge01_assignment=camera['edge01_axis'],camera_focal_px=camera['focal_px']))
    if not fits:return {'available':False,'reason':'camera_or_ballistic_hypothesis_unobservable'}
    return {'available':True,'hypotheses':fits,'best':min(fits,key=lambda r:r['fit_rmse_px']),
            'human_calibration_used':False,'world_contact_confirmed':False,'camera_assumptions_verified':False}

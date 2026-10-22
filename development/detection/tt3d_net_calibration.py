"""Six-point camera fitting: four table corners and two net-top endpoints.

Net points are vertically above the two table sideline intersections, at the
TT3D net height, not at extended support posts. Human axis confirmation is an
explicit input; suggestions never silently override it.
"""
import math
import cv2
import numpy as np
from development.detection.auto_table_geometry import order_quad
from development.detection.tt3d_camera_estimation import object_corners,ASSUMPTIONS

NET_HEIGHT_M=.1525


def infer_net_axis(corners,net_points):
    q=order_quad(corners).astype(float);net=np.asarray(net_points,float).reshape(2,2)
    unit=net[1]-net[0];length=np.linalg.norm(unit)
    if length<1:raise ValueError('Net endpoints coincide')
    unit/=length;mid=net.mean(axis=0)
    lines=[np.cross(np.r_[a,1],np.r_[b,1]) for a,b in zip(q,np.roll(q,-1,axis=0))]
    angles={}
    for axis,i,j in [('length',1,3),('width',0,2)]:
        vp=np.cross(lines[i],lines[j]);direction=vp[:2]-mid*vp[2]
        norm=np.linalg.norm(direction)
        if norm<1e-12:angles[axis]=90.;continue
        cosine=np.clip(abs(float(direction@unit))/norm,0.,1.)
        angles[axis]=math.degrees(math.acos(cosine))
    ranking=sorted(angles,key=angles.get);best,other=ranking
    supported=angles[best]<=7. and angles[other]-angles[best]>=10.
    return dict(suggested_edge01_axis=best if supported else 'unknown',
                angle_error_degrees=angles,confirmation_required=True)


def net_object_points(axis):
    q=object_corners(axis)
    x,y=abs(q[0,0]),abs(q[0,1])
    net=np.array([[0,y,NET_HEIGHT_M],[0,-y,NET_HEIGHT_M]]) if axis=='length' else np.array([[-x,0,NET_HEIGHT_M],[x,0,NET_HEIGHT_M]])
    return np.vstack([q,net])


def _minimize(cost):
    grid=np.linspace(np.log(500),np.log(20000),41)
    values=[cost(f) for f in grid];best=int(np.argmin(values))
    left,right=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
    ratio=(math.sqrt(5)-1)/2
    c=right-ratio*(right-left);d=left+ratio*(right-left);fc,fd=cost(c),cost(d)
    for _ in range(42):
        if fc<fd:
            right,d,fd=d,c,fc;c=right-ratio*(right-left);fc=cost(c)
        else:
            left,c,fc=c,d,fd;d=left+ratio*(right-left);fd=cost(d)
    return float(np.exp(min((grid[best],(left+right)/2),key=cost)))


def _solve(obj,image_points,size,f):
    width,height=size;K=np.array([[f,0,width/2],[0,f,height/2],[0,0,1]],float)
    ok,r,t=cv2.solvePnP(obj[:4],image_points[:4],K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('Initial planar pose failed')
    ok,r,t=cv2.solvePnP(obj,image_points,K,np.zeros(5),r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('Six-point pose failed')
    projected=cv2.projectPoints(obj,r,t,K,np.zeros(5))[0].reshape(-1,2)
    return K,r,t,projected


def _fit(q,net,size,axis,reverse):
    obj=net_object_points(axis)
    image_points=np.vstack([q,net[::-1] if reverse else net]).astype(np.float64)
    def solve(f):return _solve(obj,image_points,size,f)
    def cost(logf):
        return float(np.sum((solve(float(np.exp(logf)))[3]-image_points)**2))
    f=_minimize(cost);K,r,t,projected=solve(f)
    error=np.linalg.norm(projected-image_points,axis=1)
    rmse=float(np.sqrt(np.mean(error**2)))
    R=cv2.Rodrigues(r)[0];centre=(-R.T@t).ravel()
    depths=(R@obj.T+t).T[:,2]
    profile=[dict(focal_px=f*m,rmse_px=float(np.sqrt(np.mean(np.sum((solve(f*m)[3]-image_points)**2,axis=1))))) for m in (.9,1.,1.1)]
    sensitivity=min(profile[0]['rmse_px'],profile[2]['rmse_px'])-rmse
    reason='estimated'
    if depths.min()<=0 or centre[2]<=0:reason='invalid_camera_pose'
    elif f<=505 or f>=19950:reason='focal_bound_hit'
    elif rmse>3 or error.max()>6:reason='reprojection_error'
    elif sensitivity<.25:reason='weak_focal_constraint'
    return dict(edge01_axis=axis,usable=reason=='estimated',reason=reason,focal_px=f,K=K.tolist(),
        rvec=r.ravel().tolist(),tvec_m=t.ravel().tolist(),camera_centre_m=centre.tolist(),
        rotation_world_to_camera=R.tolist(),reprojection_rmse_px=rmse,
        corner_rmse_px=float(np.sqrt(np.mean(error[:4]**2))),net_rmse_px=float(np.sqrt(np.mean(error[4:]**2))),
        maximum_point_error_px=float(error.max()),per_point_error_px=error.tolist(),
        projected_corners=projected[:4].tolist(),projected_net_endpoints=projected[4:].tolist(),
        observed_net_endpoints=image_points[4:].tolist(),net_endpoint_order_reversed=reverse,
        focal_profile=profile,focal_10percent_rmse_increase_px=float(sensitivity))


def estimate_net_camera(corners,image_size,net_points,confirmed_axis='unknown'):
    q=order_quad(corners).astype(float);net=np.asarray(net_points,float)
    width,height=image_size
    if net.shape!=(2,2) or not np.isfinite(net).all() or not np.isfinite(q).all():raise ValueError('Invalid points')
    if min(width,height)<=0 or confirmed_axis not in ('length','width','unknown'):raise ValueError('Invalid camera inputs')
    if not cv2.isContourConvex(q.astype(np.float32)) or abs(cv2.contourArea(q.astype(np.float32)))<100:raise ValueError('Degenerate table quadrilateral')
    if np.linalg.norm(net[1]-net[0])<5:raise ValueError('Net endpoints too close')
    candidates=[]
    for axis in ('length','width'):
        fitted=[]
        for reverse in (False,True):
            try:fitted.append(_fit(q,net,(width,height),axis,reverse))
            except (cv2.error,ValueError):continue
        candidates.append(min(fitted,key=lambda c:c['reprojection_rmse_px']) if fitted else dict(edge01_axis=axis,usable=False,reason='pose_solver_failed'))
    selected=next((c for c in candidates if c['edge01_axis']==confirmed_axis),None)
    usable=selected is not None and selected['usable']
    return dict(image_size=[width,height],corners=q.tolist(),net_endpoints=net.tolist(),
        assumptions=dict(ASSUMPTIONS,net_height_m=NET_HEIGHT_M),candidates=candidates,
        axis_diagnostic=infer_net_axis(q,net),confirmed_edge01_axis=confirmed_axis,
        usable=bool(usable),selected_camera=selected,
        status='estimated_under_human_axis_and_tt3d_assumptions' if usable else
               'axis_confirmation_required' if confirmed_axis=='unknown' else 'selected_axis_fit_rejected',
        validation_scope='six manually reviewed image points; no independent intrinsic or 3D validation')


def estimate_net_sequence(observations,image_size):
    """Shared focal, individually fitted poses for a reviewed camera segment."""
    frames=[v['frame_id'] for v in observations]
    if len(observations)<3 or frames!=sorted(set(frames)):
        raise ValueError('At least three distinct chronological reviewed frames required')
    data=[]
    for value in observations:
        axis=value.get('edge01_axis','unknown')
        if axis not in ('length','width'):raise ValueError('All frame axes require human confirmation')
        initial=estimate_net_camera(value['corners'],image_size,value['net_endpoints'],axis)
        selected=initial['selected_camera']
        if not selected or 'net_endpoint_order_reversed' not in selected:raise ValueError('No six-point pose available')
        q=np.asarray(initial['corners'],float);net=np.asarray(value['net_endpoints'],float)
        if selected['net_endpoint_order_reversed']:net=net[::-1]
        data.append((value,net_object_points(axis),np.vstack([q,net])))
    def residual(f):return np.concatenate([(_solve(obj,points,image_size,f)[3]-points).ravel() for value,obj,points in data])
    f=_minimize(lambda logf:float(np.sum(residual(float(np.exp(logf)))**2)))
    poses=[];all_errors=[];valid_pose=True
    for value,obj,points in data:
        K,r,t,projected=_solve(obj,points,image_size,f);R=cv2.Rodrigues(r)[0]
        error=np.linalg.norm(projected-points,axis=1);all_errors.extend(error.tolist())
        centre=(-R.T@t).ravel();depths=(R@obj.T+t).T[:,2]
        valid_pose=valid_pose and depths.min()>0 and centre[2]>0
        poses.append(dict(frame_id=value['frame_id'],edge01_axis=value['edge01_axis'],K=K.tolist(),
            rvec=r.ravel().tolist(),tvec_m=t.ravel().tolist(),camera_centre_m=centre.tolist(),
            projected_corners=projected[:4].tolist(),projected_net_endpoints=projected[4:].tolist(),
            reprojection_rmse_px=float(np.sqrt(np.mean(error**2))),maximum_point_error_px=float(error.max())))
    rmse=float(np.sqrt(np.mean(np.square(all_errors))))
    profile=[dict(focal_px=f*m,rmse_px=float(np.sqrt(np.mean(residual(f*m)**2)*2))) for m in (.9,1.,1.1)]
    sensitivity=min(profile[0]['rmse_px'],profile[2]['rmse_px'])-rmse
    reason='estimated_under_human_axes_and_tt3d_assumptions'
    if not valid_pose:reason='invalid_camera_pose'
    elif f<=505 or f>=19950:reason='focal_bound_hit'
    elif rmse>3 or max(p['reprojection_rmse_px'] for p in poses)>3 or max(all_errors)>6:reason='joint_reprojection_error'
    elif sensitivity<.25:reason='weak_focal_constraint'
    return dict(usable=reason=='estimated_under_human_axes_and_tt3d_assumptions',status=reason,
        shared_focal_px=f,image_size=list(image_size),frames=frames,poses=poses,reprojection_rmse_px=rmse,
        maximum_point_error_px=max(all_errors),focal_profile=profile,focal_10percent_rmse_increase_px=float(sensitivity),
        assumptions=dict(ASSUMPTIONS,net_height_m=NET_HEIGHT_M),
        unique_player_end_direction_verified=False,
        coordinate_scope='ordered-corner coordinates per frame; player-end correspondence not independently verified',
        validation_scope='offline shared-focal fit of reviewed frames; not whole-video or independent 3D certification')

"""Single-plane camera estimates under TT3D's restricted intrinsic model.

Four corners cannot verify lens distortion or resolve physical table direction.
Report both long/short assignments and focal sensitivity, never a universal
calibration certificate. Units are metres, image coordinates are pixels.
"""
import cv2
import numpy as np
from .auto_table_geometry import order_quad

TABLE_WIDTH=1.525
TABLE_LENGTH=2.740
ASSUMPTIONS=dict(principal_point='image centre',fx_equals_fy=True,
                 skew=0,distortion=[0.,0.,0.,0.,0.],table_dimensions_m=[TABLE_WIDTH,TABLE_LENGTH],
                 world_origin='table centre',world_z='table top normal')


def object_corners(edge01_axis):
    x,y=(TABLE_LENGTH,TABLE_WIDTH) if edge01_axis=='length' else (TABLE_WIDTH,TABLE_LENGTH)
    # Image coordinates increase downwards; reversed planar winding makes +Z
    # the visible table-top normal, matching TT3D's world-coordinate convention.
    return np.array([[-x/2,y/2,0],[x/2,y/2,0],[x/2,-y/2,0],[-x/2,-y/2,0]],np.float64)


def estimate_assignment(corners,image_size,axis):
    width,height=image_size
    q=np.asarray(corners,np.float64).reshape(4,2)
    obj=object_corners(axis)
    centred=q-[width/2,height/2]
    H=cv2.getPerspectiveTransform(obj[:,:2].astype(np.float32),centred.astype(np.float32)).astype(float)
    a,b=H[:,0],H[:,1]
    # Orthogonality and equal metric scale, linear in 1/f^2. This also detects
    # fronto-parallel planes, where focal and camera distance are inseparable.
    coefficient=np.array([a[0]*b[0]+a[1]*b[1],a[0]**2+a[1]**2-b[0]**2-b[1]**2])
    offset=np.array([a[2]*b[2],a[2]**2-b[2]**2])
    inverse_f2=-float(coefficient@offset)/max(float(coefficient@coefficient),1e-30)
    if np.linalg.norm(offset)<1e-10 or inverse_f2<=0:
        return dict(edge01_axis=axis,usable=False,reason='unobservable_or_inconsistent_focal',
                    homography=H.tolist())
    analytic_f=1/np.sqrt(inverse_f2)
    if not 500<analytic_f<20000:
        return dict(edge01_axis=axis,usable=False,reason='focal_outside_tt3d_bounds',analytic_focal_px=float(analytic_f))
    def solve(f):
        K=np.array([[f,0,width/2],[0,f,height/2],[0,0,1]],np.float64)
        ok,r,t=cv2.solvePnP(obj,q,K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise ValueError('PnP failed')
        projected=cv2.projectPoints(obj,r,t,K,np.zeros(5))[0].reshape(4,2)
        return K,r,t,projected
    def cost(logf):return float(np.sum((solve(float(np.exp(logf)))[3]-q)**2))
    # One bounded focal parameter: coarse profile followed by golden-section
    # minimization avoids adding a new scientific runtime dependency.
    grid=np.unique(np.r_[np.linspace(np.log(500),np.log(20000),41),np.log(analytic_f)])
    costs=[cost(v) for v in grid];best=int(np.argmin(costs))
    left,right=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
    ratio=(np.sqrt(5)-1)/2
    c=right-ratio*(right-left);d=left+ratio*(right-left)
    fc,fd=cost(c),cost(d)
    for _ in range(45):
        if fc<fd:
            right,d,fd=d,c,fc;c=right-ratio*(right-left);fc=cost(c)
        else:
            left,c,fc=c,d,fd;d=left+ratio*(right-left);fd=cost(d)
    logf=min((grid[best],(left+right)/2),key=cost)
    f=float(np.exp(logf));K,r,t,projected=solve(f)
    rotation=cv2.Rodrigues(r)[0]
    depths=(rotation@obj.T+t).T[:,2]
    rmse=float(np.sqrt(np.mean(np.sum((projected-q)**2,axis=1))))
    per_corner=np.linalg.norm(projected-q,axis=1)
    # Local focal sensitivity after refitting pose. A tiny 10% focal change
    # must cause a detectable reprojection change; otherwise mark uncertain.
    profile=[]
    for multiplier in (.9,1.,1.1):
        ff=f*multiplier
        qq=solve(ff)[3]
        profile.append(dict(focal_px=ff,rmse_px=float(np.sqrt(np.mean(np.sum((qq-q)**2,axis=1))))))
    sensitivity=min(profile[0]['rmse_px'],profile[2]['rmse_px'])-rmse
    reason='estimated'
    if np.min(depths)<=0:reason='nonpositive_table_depth'
    elif float((-rotation.T@t)[2,0])<=0:reason='camera_below_table_plane'
    elif f<=505 or f>=19950:reason='focal_bound_hit'
    elif rmse>3:reason='reprojection_error'
    elif sensitivity<.25:reason='weak_focal_constraint'
    return dict(edge01_axis=axis,usable=reason=='estimated',reason=reason,focal_px=f,
        K=K.tolist(),rvec=r.ravel().tolist(),tvec_m=t.ravel().tolist(),
        camera_centre_m=(-rotation.T@t).ravel().tolist(),rotation_world_to_camera=rotation.tolist(),
        reprojection_rmse_px=rmse,per_corner_error_px=per_corner.tolist(),
        projected_corners=projected.tolist(),minimum_table_depth_m=float(np.min(depths)),
        focal_profile=profile,focal_10percent_rmse_increase_px=float(sensitivity),
        optimizer_success=True,analytic_focal_px=float(analytic_f))


def estimate_camera(corners,image_size):
    width,height=image_size
    q=order_quad(corners).astype(float)
    if not np.isfinite(q).all() or width<=0 or height<=0:raise ValueError('Invalid camera inputs')
    if not cv2.isContourConvex(q.astype(np.float32)) or abs(cv2.contourArea(q.astype(np.float32)))<100:
        raise ValueError('Degenerate table quadrilateral')
    candidates=[]
    for axis in ('length','width'):
        try:candidates.append(estimate_assignment(q,(width,height),axis))
        except (cv2.error,ValueError) as exc:candidates.append(dict(edge01_axis=axis,usable=False,reason='pose_solver_failed',detail=str(exc)))
    usable=[c for c in candidates if c['usable']]
    return dict(image_size=[width,height],corners=q.tolist(),assumptions=ASSUMPTIONS,
        usable=bool(usable),status='estimated_under_assumptions' if usable else 'insufficient_constraint',
        physical_axes='ambiguous' if len(usable)>1 else 'conditional_metric_hypothesis' if usable else 'unknown',
        unique_physical_direction_verified=False,candidates=candidates,
        validation_scope='four-corner reprojection only; no independent 3D or intrinsic validation')


def project_table_grid(candidate,image_size):
    obj=object_corners(candidate['edge01_axis']);x,y=abs(obj[0,0]),abs(obj[0,1])
    lines=[]
    for xx in np.linspace(-x,x,5):lines.append(np.array([[xx,-y,0],[xx,y,0]],float))
    for yy in np.linspace(-y,y,5):lines.append(np.array([[-x,yy,0],[x,yy,0]],float))
    return [cv2.projectPoints(line,np.array(candidate['rvec']),np.array(candidate['tvec_m']),
            np.array(candidate['K']),np.zeros(5))[0].reshape(-1,2) for line in lines]

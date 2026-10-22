"""Competing world-space hypotheses on identical observations, not plane projection proof."""
import numpy as np
from scipy.optimize import lsq_linear,minimize_scalar
from .camera import object_corners

POLICY=dict(gravity_m_s2=9.81,ball_radius_m=.02,table_height_m=.76,window_s=.24,search_s=.10,
    search_frames=2.5,maximum_observations_each_side=6,minimum_observations_each_side=3,maximum_cross_gap_s=.12,
    minimum_confidence=.75,maximum_condition=1e6,noise_image_diagonal_fraction=.002,minimum_noise_px=2.,
    maximum_rmse_image_diagonal_fraction=.0075,maximum_speed_m_s=40.,airborne_min_height_above_table_m=.10,
    airborne_max_height_above_table_m=2.5,normal_restitution_range=[.2,1.1],maximum_time_profile_frames=1.5,
    maximum_axis_contact_spread_px=5.,maximum_axis_time_spread_frames=1.,time_grid_points=17,
    camera_minimum_past_stable_samples=2,camera_maximum_corner_drift_fraction=.02,
    strong_motion_gain=.5,strong_motion_jump=.3,close_event_radius_frames=1.25)
KINDS=['smooth_flight','table_bounce','airborne_impulse','floor_bounce']
def observed(r):
    return bool(r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=POLICY['minimum_confidence'] and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y','timestamp_s')))
def project(x,camera):
    r=np.asarray(camera['rotation_world_to_camera']);t=np.asarray(camera['tvec_m']);k=np.asarray(camera['K']);p=np.asarray(x)@r.T+t
    q=p@k.T;return q[...,:2]/q[...,2,None],p[...,2]
def design(times,tau,kind):
    dt=np.asarray(times)-tau;n=len(dt);plane=kind in ('table_bounce','floor_bounce');dim=6 if kind=='smooth_flight' else 8 if plane else 9
    b=np.zeros((n,3,dim));c=np.zeros((n,3));c[:,2]=-.5*POLICY['gravity_m_s2']*dt*dt
    if kind=='smooth_flight':b[:,:,:3]=np.eye(3);b[:,:,3:]=dt[:,None,None]*np.eye(3)
    else:
        if plane:
            b[:,0,0]=1;b[:,1,1]=1;c[:,2]+=POLICY['ball_radius_m']-(POLICY['table_height_m'] if kind=='floor_bounce' else 0);base=2
        else:b[:,:,:3]=np.eye(3);base=3
        for i,d in enumerate(dt):b[i,:,base+(0 if d<0 else 3):base+(3 if d<0 else 6)]=np.eye(3)*d
    return b,c
def fit_at(rows,camera,tau,kind,size):
    times=np.array([r['timestamp_s'] for r in rows]);dt=times-tau;n=len(rows);before=dt<0
    if kind!='smooth_flight' and min(before.sum(),(~before).sum())<POLICY['minimum_observations_each_side']:return None
    if kind!='smooth_flight' and times[~before].min()-times[before].max()>POLICY['maximum_cross_gap_s']+1e-9:return None
    xy=np.array([[r['x'],r['y']] for r in rows]);k=np.asarray(camera['K']);rot=np.asarray(camera['rotation_world_to_camera']);trans=np.asarray(camera['tvec_m'])
    rays=(np.linalg.inv(k)@np.c_[xy,np.ones(n)].T).T;rays/=rays[:,2,None]
    cross=np.zeros((n,2,3));cross[:,0,0]=1;cross[:,1,1]=1;cross[:,:,2]=-rays[:,:2]
    basis,constant=design(times,tau,kind);mat=np.einsum('ij,njk->nik',rot,basis);const=constant@rot.T+trans
    aa=np.einsum('nij,njk->nik',cross,mat).reshape(2*n,-1);bb=-np.einsum('nij,nj->ni',cross,const).ravel();pdim=aa.shape[1]
    condition=float(np.linalg.cond(aa))
    if not np.isfinite(condition) or condition>POLICY['maximum_condition']:return None
    lo=np.full(pdim,-40.);hi=np.full(pdim,40.)
    if kind in ('table_bounce','floor_bounce'):
        bounds=np.max(abs(object_corners(camera['edge01_axis'])[:,:2]),axis=0)+.02 if kind=='table_bounce' else np.array([6.,6.])
        lo[:2]=-bounds;hi[:2]=bounds;hi[4]=-.1;lo[7]=.1
    else:
        lo[:2]=-6;hi[:2]=6
        lo[2]=POLICY['airborne_min_height_above_table_m'] if kind=='airborne_impulse' else -.74
        hi[2]=POLICY['airborne_max_height_above_table_m'] if kind=='airborne_impulse' else 3.
    weights=np.ones(2*n)
    for _ in range(3):
        a=aa*weights[:,None];b=bb*weights
        p,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
        if rank<pdim:return None
        if np.any(p<lo) or np.any(p>hi):
            result=lsq_linear(a,b,bounds=(lo,hi),method='bvls',tol=1e-8,max_iter=100)
            if not result.success:return None
            p=result.x
        world=np.einsum('nij,j->ni',basis,p)+constant;uv,depth=project(world,camera)
        if np.any(depth<=.1) or not np.isfinite(uv).all():return None
        weights=np.repeat(np.median(depth)/depth,2)
    mse=float(np.mean(np.sum((uv-xy)**2,axis=1)));rmse=float(np.sqrt(mse));sigma=max(POLICY['minimum_noise_px'],np.linalg.norm(size)*POLICY['noise_image_diagonal_fraction'])
    if kind=='smooth_flight':centre=p[:3];vel_pre=vel_post=p[3:6]
    elif kind=='airborne_impulse':centre=p[:3];vel_pre=p[3:6];vel_post=p[6:9]
    else:
        centre=np.r_[p[:2],POLICY['ball_radius_m']-(POLICY['table_height_m'] if kind=='floor_bounce' else 0.)];vel_pre=p[2:5];vel_post=p[5:8]
        restitution=vel_post[2]/max(-vel_pre[2],1e-9)
        if not POLICY['normal_restitution_range'][0]<=restitution<=POLICY['normal_restitution_range'][1]:return None
        if np.min(world[:,2])<centre[2]-.03:return None
    if max(np.linalg.norm(vel_pre),np.linalg.norm(vel_post))>POLICY['maximum_speed_m_s']:return None
    contact,depth=project(centre,camera)
    if depth<=0:return None
    # Known-noise Gaussian deviance with parameter penalty; time is an extra
    # searched parameter for impulse models, not for continuous flight.
    parameter_count=pdim+(kind!='smooth_flight');score=n*mse/sigma**2+parameter_count*np.log(2*n)
    surface=centre.copy();surface[2]-=POLICY['ball_radius_m'];surface_px,_=project(surface,camera)
    return dict(kind=kind,time_s=float(tau),rmse_px=rmse,penalized_score=float(score),parameter_count=int(parameter_count),condition=condition,
        world_centre_m=centre.tolist(),image_ball_centre_px=contact.tolist(),image_surface_contact_px=surface_px.tolist(),
        velocity_before_m_s=vel_pre.tolist(),velocity_after_m_s=vel_post.tolist(),support_frame_ids=[r['frame_id'] for r in rows],
        maximum_rmse_passed=bool(rmse<=np.linalg.norm(size)*POLICY['maximum_rmse_image_diagonal_fraction']),coordinates_are_estimates=True)
def compare(rows,camera,seed_time,time_bounds,size,unit):
    grid=np.linspace(*time_bounds,POLICY['time_grid_points']);all_fits={};profiles={}
    for kind in KINDS:
        fits=[fit_at(rows,camera,float(t),kind,size) for t in ([seed_time] if kind=='smooth_flight' else grid)];fits=[f for f in fits if f is not None]
        if fits:
            best=min(fits,key=lambda f:f['penalized_score'])
            if kind!='smooth_flight':
                step=(time_bounds[1]-time_bounds[0])/(len(grid)-1);lo=max(time_bounds[0],best['time_s']-step);hi=min(time_bounds[1],best['time_s']+step)
                def cost(t):
                    f=fit_at(rows,camera,float(t),kind,size);return f['penalized_score'] if f else 1e12
                result=minimize_scalar(cost,bounds=(lo,hi),method='bounded',options={'xatol':max(unit*.01,1e-5),'maxiter':12})
                f=fit_at(rows,camera,float(result.x),kind,size)
                if f is not None:fits.append(f);best=min(fits,key=lambda f:f['penalized_score'])
                profile=[f['time_s'] for f in fits if f['penalized_score']<=best['penalized_score']+3.84]
                width=(max(profile)-min(profile))/unit
                best=dict(best,time_profile_s=[min(profile),max(profile)],time_profile_width_frames=float(width),time_observable=bool(width<=POLICY['maximum_time_profile_frames'] and min(best['time_s']-time_bounds[0],time_bounds[1]-best['time_s'])>=unit*.15))
            all_fits[kind]=best;profiles[kind]=len(fits)
    usable={k:f for k,f in all_fits.items() if f['maximum_rmse_passed']}
    if not usable:return dict(available=False,reason='no_physically_valid_low_residual_model',models=all_fits)
    winner=min(usable,key=lambda k:usable[k]['penalized_score']);table=usable.get('table_bounce');others=[f for k,f in usable.items() if k!='table_bounce']
    # An invalid table fit is not enough to prove another event type. Compare
    # against the fitted table cost, when such a hypothesis can be evaluated.
    table_any=all_fits.get('table_bounce');margin=None
    if table_any and others:margin=min(f['penalized_score'] for f in others)-table_any['penalized_score']
    return dict(available=True,winner=winner,table_advantage=margin,models=all_fits,camera_axis=camera['edge01_axis'],
        accepted_table_by_geometry_alone=False,all_models_use_same_observations=True)

def aggregate(comparisons,margin):
    if not comparisons or not all(c.get('available') for c in comparisons):return dict(kind='unknown',reason='camera_or_model_unavailable')
    advantages=[c.get('table_advantage') for c in comparisons]
    if any(a is None for a in advantages):return dict(kind='unknown',reason='missing_competing_hypothesis')
    if all(a>=margin and c['winner']=='table_bounce' and c['models']['table_bounce'].get('time_observable') for a,c in zip(advantages,comparisons)):
        fits=[c['models']['table_bounce'] for c in comparisons];xy=np.array([f['image_ball_centre_px'] for f in fits]);tm=np.array([f['time_s'] for f in fits])
        if np.max(np.linalg.norm(xy-xy.mean(0),axis=1))*2>POLICY['maximum_axis_contact_spread_px']:return dict(kind='unknown',reason='camera_axis_contact_disagreement')
        return dict(kind='table_bounce',reason='all_camera_hypotheses_prefer_table',fit=min(fits,key=lambda f:f['penalized_score']),axis_times_s=tm.tolist())
    if all(a<=-margin and c['winner']!='table_bounce' for a,c in zip(advantages,comparisons)):
        return dict(kind='non_table',reason='all_camera_hypotheses_prefer_non_table',model_winners=[c['winner'] for c in comparisons])
    return dict(kind='unknown',reason='insufficient_margin_or_axis_disagreement')

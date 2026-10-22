"""Reprojection fitting with quadratic drag and piecewise-constant Magnus spin.

This is a flight model, not a verified recovery of rotation. Bounce velocities
remain independently fitted; restitution and nonpenetration are checked.
"""
from . import physics as gravity
import numpy as np
from scipy.optimize import least_squares
from .camera import object_corners

POLICY=dict(drag_over_mass=3.8e-4/2.7e-3,magnus_over_mass=4.86e-6/2.7e-3,
    max_spin_rad_s=400.,spin_prior_rad_s=200.,maximum_step_s=.02,
    max_nfev=50,maximum_joint_nfev=60,minimum_outer_observations=3,
    minimum_middle_observations=2,minimum_joint_time_separation_frames=.5,
    table_feasibility_rmse_diagonal=.0075,negative_rmse_ratio=.35,
    margin=10.,maximum_time_uncertainty_frames=1.5,
    early_exit_on_feasible_single_table=True,automatic_positive_confirmation=False,
    minimum_table_height_confidence_m=.10,maximum_parameter_condition=1e8)

def advance(p,v,dt,spin,mode):
    dt=np.atleast_1d(dt).astype(float);p=np.broadcast_to(np.asarray(p,float),(len(dt),3)).copy();v=np.broadcast_to(np.asarray(v,float),p.shape).copy()
    if mode=='gravity':
        p+=v*dt[:,None];p[:,2]-=4.905*dt**2;v[:,2]-=9.81*dt;return p,v
    spin=np.asarray(spin)
    def acc(x):
        a=-POLICY['drag_over_mass']*np.linalg.norm(x,axis=1)[:,None]*x;a[:,2]-=9.81
        if mode=='magnus':
            a+=POLICY['magnus_over_mass']*np.column_stack((spin[1]*x[:,2]-spin[2]*x[:,1],spin[2]*x[:,0]-spin[0]*x[:,2],spin[0]*x[:,1]-spin[1]*x[:,0]))
        return a
    steps=max(1,int(np.ceil(np.max(abs(dt))/POLICY['maximum_step_s'])));h=dt[:,None]/steps
    for _ in range(steps):
        a=acc(v);v2=v+h*a/2;b=acc(v2);v3=v+h*b/2;c=acc(v3);v4=v+h*c;d=acc(v4)
        p+=h*(v+2*v2+2*v3+v4)/6;v+=h*(a+2*b+2*c+d)/6
    return p,v

def unpack(x,kind,mode,joint=False):
    plane=kind in ('table_bounce','floor_bounce');smooth=kind=='smooth_flight';base=2 if plane else 3
    p=np.r_[x[:2],.02-(.76 if kind=='floor_bounce' else 0.)] if plane else x[:3]
    before=x[base:base+3];after=before if smooth else x[base+3:base+6]
    cursor=base+(3 if smooth else 6);omega1=omega2=np.zeros(3)
    if mode=='magnus':
        omega1=x[cursor:cursor+3];cursor+=3
        omega2=omega1 if smooth else x[cursor:cursor+3];cursor+=0 if smooth else 3
    tau=x[cursor];cursor+=1
    hit=(x[cursor],x[cursor+1:cursor+4]) if joint else None
    return p,before,after,omega1,omega2,tau,hit

def world_positions(x,times,kind,mode,joint=False):
    p,b,a,w1,w2,tau,hit=unpack(x,kind,mode,joint);out=np.zeros((len(times),3))
    for side,vel,spin in ((times<tau,b,w1),(times>=tau,a,w2)):
        ids=np.flatnonzero(side)
        if not len(ids):continue
        tt=times[ids];out[ids]=advance(p,vel,tt-tau,spin,mode)[0]
        if hit is not None:
            th,dv=hit;cross=(tt<th) if th<tau else (tt>th)
            if (th<tau and np.any(tt<tau)) or (th>=tau and np.any(tt>=tau)):
                ph,vh=advance(p,vel,np.array([th-tau]),spin,mode);vh=vh[0]+(-dv if th<tau else dv)
                if cross.any():out[ids[cross]]=advance(ph[0],vh,tt[cross]-th,spin,mode)[0]
    return out

def seed_and_bounds(rows,cam,tau,bounds,kind,mode,old,hit_time=None,unit=.04):
    plane=kind in ('table_bounce','floor_bounce');smooth=kind=='smooth_flight'
    if old:
        p=old['world_centre_m'];b=old['velocity_before_m_s'];a=old['velocity_after_m_s'];tau=old['time_s']
    else:
        r=min(rows,key=lambda r:abs(r['timestamp_s']-tau));rot=np.asarray(cam['rotation_world_to_camera']);centre=-rot.T@np.asarray(cam['tvec_m']);ray=rot.T@np.linalg.solve(np.asarray(cam['K']),[r['x'],r['y'],1.]);z=.02 if plane else .3
        if kind=='floor_bounce':z=-.74
        den=ray[2] if abs(ray[2])>1e-7 else -1e-7;p=centre+(z-centre[2])/den*ray;p=np.clip(p,[-5,-5,-.74],[5,5,2.5]);b=[0,5,-2];a=[0,5,2]
    lim=np.max(abs(object_corners(cam['edge01_axis'])[:,:2]),axis=0)+.02 if kind=='table_bounce' else np.array([6.,6.])
    if plane:
        initial=list(p[:2])+list(b)+list(a);lo=list(-lim)+[-40,-40,-40,-40,-40,.1];hi=list(lim)+[40,40,-.1,40,40,40]
    else:
        initial=list(p)+list(b)+(list(a) if not smooth else []);lo=[-6,-6,.1 if kind=='airborne_impulse' else -.74]+[-40]*(3 if smooth else 6);hi=[6,6,2.5 if kind=='airborne_impulse' else 3]+[40]*(3 if smooth else 6)
    if mode=='magnus':
        count=3 if smooth else 6;initial +=[0.]*count;lo +=[-POLICY['max_spin_rad_s']]*count;hi +=[POLICY['max_spin_rad_s']]*count
    initial +=[tau];lo +=[bounds[0]];hi +=[bounds[1]]
    if hit_time is not None:
        initial +=[hit_time,0.,0.,0.];lo +=[max(rows[0]['timestamp_s']+unit,hit_time-unit),-40,-40,-40];hi +=[min(rows[-1]['timestamp_s']-unit,hit_time+unit),40,40,40]
    return np.array(initial),np.array(lo),np.array(hi)

def fit(rows,cam,tau,bounds,size,unit,kind,mode,old=None,hit_time=None):
    times=np.array([r['timestamp_s'] for r in rows]);xy=np.array([[r['x'],r['y']] for r in rows]);joint=hit_time is not None
    x,lo,hi=seed_and_bounds(rows,cam,tau,bounds,kind,mode,old,hit_time,unit)
    smooth=kind=='smooth_flight';fixed_tau=float(x[-1])
    if smooth:x=x[:-1];lo=lo[:-1];hi=hi[:-1]
    if np.any(hi<=lo):return None
    sigma=max(2.,np.linalg.norm(size)*.002)
    def residual(x):
        if smooth:x=np.r_[x,fixed_tau]
        uv,depth=gravity.project(world_positions(x,times,kind,mode,joint),cam);data=((uv-xy)/sigma).ravel()
        if not np.isfinite(data).all():return np.full(len(data)+(3 if kind=='smooth_flight' else 6) if mode=='magnus' else len(data),1e6)
        if mode=='magnus':
            _,_,_,w1,w2,_,_=unpack(x,kind,mode,joint);data=np.r_[data,w1/POLICY['spin_prior_rad_s']] if kind=='smooth_flight' else np.r_[data,w1/POLICY['spin_prior_rad_s'],w2/POLICY['spin_prior_rad_s']]
        return data
    try:
        opt=least_squares(residual,np.clip(x,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=POLICY['maximum_joint_nfev'] if joint else POLICY['max_nfev'],x_scale='jac',ftol=1e-5,xtol=1e-5,gtol=1e-5)
    except (ValueError,FloatingPointError):return None
    fitted=np.r_[opt.x,fixed_tau] if smooth else opt.x
    p,b,a,w1,w2,tau,hit=unpack(fitted,kind,mode,joint);world=world_positions(fitted,times,kind,mode,joint);uv,depth=gravity.project(world,cam)
    before=times<tau
    if kind!='smooth_flight' and min(before.sum(),(~before).sum())<3:return None
    if np.any(depth<=.1) or not np.isfinite(uv).all() or max(np.linalg.norm(b),np.linalg.norm(a))>40:return None
    if kind in ('table_bounce','floor_bounce'):
        if not .2<=a[2]/max(-b[2],1e-9)<=1.1 or world[:,2].min()<p[2]-.03:return None
    middle=0;order=None
    if joint:
        th,dv=hit;middle=int(np.sum((times>min(tau,th))&(times<max(tau,th))))
        if abs(tau-th)<unit*POLICY['minimum_joint_time_separation_frames']:return None
        if min(np.sum(times<min(tau,th)),np.sum(times>max(tau,th)))<3:return None
        order='table_then_hit' if tau<th else 'hit_then_table'
    rmse=float(np.sqrt(np.mean(np.sum((uv-xy)**2,axis=1))));jac=opt.jac[:2*len(rows)];condition=float(np.linalg.cond(jac));rank=int(np.linalg.matrix_rank(jac));full_rank=rank==len(opt.x)
    covariance=np.linalg.pinv(jac.T@jac);tindex=len(opt.x)-(5 if joint else 1);time_sd=float(np.sqrt(max(0,covariance[tindex,tindex]))) if full_rank and not smooth else None
    px,_=gravity.project(p,cam);surface=p.copy();surface[2]-=.02;sx,_=gravity.project(surface,cam)
    return dict(kind=kind,mode=mode,joint=joint,rmse_px=rmse,penalized_score=float(np.sum(residual(opt.x)**2)+len(opt.x)*np.log(2*len(rows))),parameter_count=len(opt.x),
        time_s=float(tau),world_centre_m=p.tolist(),image_ball_centre_px=px.tolist(),image_surface_contact_px=sx.tolist(),velocity_before_m_s=b.tolist(),velocity_after_m_s=a.tolist(),
        spin_before_rad_s=w1.tolist(),spin_after_rad_s=w2.tolist(),spin_is_estimated_not_measured=True,
        jacobian_full_rank=full_rank,condition=condition if np.isfinite(condition) else None,time_sd_s=time_sd,
        time_observable=bool(time_sd is not None and full_rank and time_sd<unit*.75 and condition<POLICY['maximum_parameter_condition'] and min(tau-bounds[0],bounds[1]-tau)>unit*.15),
        converged=bool(opt.success),function_evaluations=int(opt.nfev),maximum_rmse_passed=bool(rmse<=np.linalg.norm(size)*.0075),
        hit_time_s=float(hit[0]) if joint else None,joint_order=order,middle_observations=middle,
        joint_observable=bool(joint and middle>=POLICY['minimum_middle_observations'] and full_rank),support_frame_ids=[r['frame_id'] for r in rows])

def compare(rows,cam,tau,bounds,size,unit,mode,hit_time=None):
    old=gravity.compare(rows,cam,tau,bounds,size,unit);fits={}
    f=fit(rows,cam,tau,bounds,size,unit,'table_bounce',mode,old.get('models',{}).get('table_bounce'))
    if f:fits['table_bounce']=f
    if hit_time is not None:
        for shift in (-unit,unit):
            f=fit(rows,cam,float(np.clip(tau+shift,*bounds)),bounds,size,unit,'table_bounce',mode,None,hit_time)
            if f and ('table_and_hit' not in fits or f['penalized_score']<fits['table_and_hit']['penalized_score']):fits['table_and_hit']=f
    # A feasible table already prevents a veto. Do not spend extra solves to
    # rank ambiguous heights or claim an unverified positive table contact.
    feasible=any(v['converged'] and v['maximum_rmse_passed'] for v in fits.values())
    if not feasible:
        for kind in ('smooth_flight','airborne_impulse','floor_bounce'):
            f=fit(rows,cam,tau,bounds,size,unit,kind,mode,old.get('models',{}).get(kind))
            if f:fits[kind]=f
    usable={k:v for k,v in fits.items() if v['converged'] and v['maximum_rmse_passed']}
    return dict(models=fits,available=bool(usable),winner=min(usable,key=lambda k:usable[k]['penalized_score']) if usable else None,camera_axis=cam['edge01_axis'],joint_attempted=hit_time is not None)

def decision(comparisons,size,has_nearby_hit=False):
    # Negative evidence requires failure of *every evaluable table hypothesis*,
    # not a relative preference among ambiguous monocular fits.
    if not comparisons or not all(c['available'] for c in comparisons):return dict(kind='unknown',reason='camera_or_model_unavailable')
    negative=[];positive=[]
    for c in comparisons:
        models=c['models'];tables=[v for k,v in models.items() if k.startswith('table') and v['converged']]
        if not tables:return dict(kind='unknown',reason='table_feasibility_untested')
        if has_nearby_hit and (not models.get('table_and_hit') or not models['table_and_hit']['joint_observable']):return dict(kind='unknown',reason='joint_event_not_observable')
        feasible=[t for t in tables if t['maximum_rmse_passed']]
        others=[v for k,v in models.items() if not k.startswith('table') and v['converged'] and v['maximum_rmse_passed']]
        if not feasible and others:
            t=min(tables,key=lambda t:t['rmse_px']);o=min(others,key=lambda t:t['penalized_score'])
            negative.append(o['rmse_px']<=POLICY['negative_rmse_ratio']*t['rmse_px'] and t['penalized_score']-o['penalized_score']>=POLICY['margin'])
        else:negative.append(False)
        if feasible and others:
            t=min(feasible,key=lambda t:t['penalized_score']);margin=min(o['penalized_score'] for o in others)-t['penalized_score']
            if margin>=POLICY['margin'] and t['time_observable'] and (not t['joint'] or t['joint_observable']):positive.append(t)
    if all(negative):return dict(kind='non_table',reason='all_camera_table_hypotheses_fail_absolute_feasibility')
    if POLICY['automatic_positive_confirmation'] and len(positive)==len(comparisons):
        xy=np.array([f['image_ball_centre_px'] for f in positive]);ts=[f['time_s'] for f in positive]
        if np.max(np.linalg.norm(xy-xy.mean(0),axis=1))*2<=5:return dict(kind='table_bounce',reason='identifiable_table_model',fit=min(positive,key=lambda t:t['penalized_score']),axis_times_s=ts)
    return dict(kind='unknown',reason='feasible_table_or_unresolved_depth_ambiguity')

"""Auditable TT3D-inspired local ballistic pilot (not full TT3D spin/drag).

All coordinates are metres; ball centre at contact is 0.02 m above table.
Conditional on contact time, initialize from ray constraints and refine in pixels.
No detector point deletion, label changes, or production event replacement.
"""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np

G=9.81
RADIUS=.02

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project(points, pose):
    rot=cv2.Rodrigues(np.asarray(pose['rvec'],float))[0]
    cam=np.asarray(points)@rot.T+pose['tvec_m']
    k=np.asarray(pose['K'])
    return cam[:,:2]/cam[:,2,None]*[k[0,0],k[1,1]]+k[:2,2],cam[:,2]

def design(times, tau, bounce=True):
    dt=np.asarray(times)-tau
    a=np.zeros((len(dt),3,8 if bounce else 6))
    base=np.zeros((len(dt),3));base[:,2]=-.5*G*dt**2
    if bounce:
        base[:,2]+=RADIUS
        a[:,0,0]=1;a[:,1,1]=1
        for i,d in enumerate(dt):a[i,:,2 if d<0 else 5:5 if d<0 else 8]=np.eye(3)*d
    else:
        a[:,:,:3]=np.eye(3)
        a[:,:,3:]=np.eye(3)[None]*dt[:,None,None]
    return a,base

def conditional(times, xy, pose, tau, bounce=True, initial=None, bounds=None):
    a,base=design(times,tau,bounce)
    rot=cv2.Rodrigues(np.asarray(pose['rvec'],float))[0]
    k=np.asarray(pose['K']);f=np.array([k[0,0],k[1,1]])
    ac=np.einsum('ij,njk->nik',rot,a);bc=base@rot.T+pose['tvec_m']
    ray=(xy-k[:2,2])/f
    lhs=(ac[:,:2,:]-ray[:,:,None]*ac[:,2:3,:]).reshape(-1,a.shape[2])
    rhs=(ray*bc[:,2,None]-bc[:,:2]).ravel()
    p=np.linalg.lstsq(lhs,rhs,rcond=None)[0] if initial is None else np.asarray(initial,float).copy()
    if bounds is not None:p=np.clip(p,*bounds)
    def evaluate(p):
        cam=bc+np.einsum('nij,j->ni',ac,p)
        if np.any(cam[:,2]<=.05):return None
        uv=cam[:,:2]/cam[:,2,None]*f+k[:2,2]
        residual=(uv-xy).ravel()
        jac=(ac[:,:2,:]/cam[:,2,None,None]-cam[:,:2,None]*ac[:,2:3,:]/cam[:,2,None,None]**2)*f[None,:,None]
        return float(residual@residual),residual,jac.reshape(-1,len(p)),uv
    e=evaluate(p)
    if e is None:return None
    damping=1e-4
    for _ in range(80 if bounds is not None else 35):
        h=e[2].T@e[2];gradient=e[2].T@e[1]
        active=np.zeros(len(p),bool) if bounds is None else (((p<=bounds[0]+1e-9)&(gradient>0))|((p>=bounds[1]-1e-9)&(gradient<0)))
        free=~active;step=np.zeros(len(p))
        if not np.any(free):break
        hf=h[np.ix_(free,free)]
        step[free]=np.linalg.solve(hf+damping*np.diag(np.maximum(np.diag(hf),1)), -gradient[free])
        candidate=p+step if bounds is None else np.clip(p+step,*bounds)
        trial=evaluate(candidate)
        if trial is not None and trial[0]<e[0]:
            step=candidate-p;p=candidate;e=trial;damping=max(damping/3,1e-10)
            if np.linalg.norm(step)<1e-7:break
        else:damping*=10
    return {'tau':float(tau),'parameters':p.tolist(),'sse':e[0],'rmse_px':float(np.sqrt(e[0]/len(times))),
            'max_error_px':float(np.linalg.norm(e[1].reshape(-1,2),axis=1).max()),'projected':e[3].tolist()}

def physical_bounds(times, tau, pose):
    limits=np.array([2.740,1.525] if pose['edge01_axis']=='length' else [1.525,2.740])/2
    lower=np.r_[-limits,[-40,-40,-40],[-40,-40,max(.1,.5*G*max(0,max(times)-tau))]]
    upper=np.r_[limits,[40,40,-max(.1,.5*G*max(0,tau-min(times)))],[40,40,40]]
    return lower,upper

def search(rows, pose, lo, hi, constrained=False):
    times=np.array([r['timestamp_s'] for r in rows]);xy=np.array([[r['x'],r['y']] for r in rows])
    profile=[]
    for tau in np.linspace(lo,hi,61):
        if min(np.sum(times<tau),np.sum(times>=tau))<3:continue
        fit=conditional(times,xy,pose,tau,bounds=physical_bounds(times,tau,pose) if constrained else None)
        if fit:profile.append(fit)
    if not profile:return None
    best=min(profile,key=lambda f:f['sse']);step=(hi-lo)/60
    for _ in range(3):
        for tau in np.linspace(max(lo,best['tau']-step),min(hi,best['tau']+step),9):
            if min(np.sum(times<tau),np.sum(times>=tau))<3:continue
            fit=conditional(times,xy,pose,tau,bounds=physical_bounds(times,tau,pose) if constrained else None)
            if fit:profile.append(fit)
        best=min(profile,key=lambda f:f['sse']);step/=4
    best=dict(best)
    if constrained:
        bounds=physical_bounds(times,best['tau'],pose);p=np.array(best['parameters'])
        best['active_constraint_indices']=np.flatnonzero((abs(p-bounds[0])<1e-5)|(abs(p-bounds[1])<1e-5)).tolist()
    # A descriptive 1-pixel RMS cost band, NOT a confidence interval.
    band=[f['tau'] for f in profile if f['sse']<=best['sse']+len(rows)]
    best['one_pixel_cost_band_s']=[min(band),max(band)]
    best['time_profile']=[{'time_s':f['tau'],'rmse_px':f['rmse_px']} for f in sorted(profile,key=lambda f:f['tau'])]
    best['search_boundary']=bool(min(best['tau']-lo,hi-best['tau'])<(hi-lo)/100)
    nulls=[conditional(times,xy,pose,float(times.mean()),False)]
    p=np.asarray(best['parameters']);centre=float(times.mean());delta=centre-best['tau']
    for velocity in (p[2:5],p[5:8],(p[2:5]+p[5:8])/2):
        position=np.r_[p[:2],RADIUS]+velocity*delta+np.array([0,0,-.5*G*delta**2])
        nulls.append(conditional(times,xy,pose,centre,False,np.r_[position,velocity+[0,0,-G*delta]]))
    null=min((n for n in nulls if n is not None),key=lambda n:n['sse'],default=None)
    best['smooth_ballistic_rmse_px']=null['rmse_px'] if null else None
    best['bic_gain_over_smooth']=float(2*len(rows)*np.log(max(null['sse'],1e-12)/max(best['sse'],1e-12))-3*np.log(2*len(rows))) if null else None
    return best

def stability(frames, pose, targets):
    anchor=pose['frame_id'];gray=cv2.cvtColor(frames[anchor],cv2.COLOR_BGR2GRAY)
    q=np.asarray(pose['projected_corners'],np.float32)
    mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(q).astype(np.int32),255)
    # Keep features on table plane; omit net and its immediate shadow.
    net=np.asarray(pose['projected_net_endpoints'],np.float32)
    cv2.line(mask,tuple(np.rint(net[0]).astype(int)),tuple(np.rint(net[1]).astype(int)),0,65)
    mask=cv2.erode(mask,np.ones((5,5),np.uint8))
    points=cv2.goodFeaturesToTrack(gray,400,.005,7,mask=mask)
    results=[]
    for fid in targets:
        item={'frame_id':fid,'status':'insufficient_features'}
        if points is not None and len(points)>=12:
            other=cv2.cvtColor(frames[fid],cv2.COLOR_BGR2GRAY)
            p1,s1,_=cv2.calcOpticalFlowPyrLK(gray,other,points,None,winSize=(25,25),maxLevel=3)
            p0,s0,_=cv2.calcOpticalFlowPyrLK(other,gray,p1,None,winSize=(25,25),maxLevel=3)
            good=(s1.ravel()>0)&(s0.ravel()>0)&(np.linalg.norm(p0-points,axis=2).ravel()<.8)
            src=points[good];dst=p1[good]
            if len(src)>=12:
                h,inliers=cv2.findHomography(src,dst,cv2.RANSAC,1.5)
                if h is not None and inliers is not None and inliers.sum()>=12:
                    keep=inliers.ravel()>0
                    coverage=cv2.contourArea(cv2.convexHull(src[keep]))/cv2.contourArea(q)
                    residual=np.linalg.norm(cv2.perspectiveTransform(src[keep],h)-dst[keep],axis=2).ravel()
                    motion=np.linalg.norm(cv2.perspectiveTransform(q[None],h)[0]-q,axis=1)
                    supported=coverage>=.15 and np.percentile(residual,95)<=1.5
                    item.update(status=('stable' if motion.max()<=3 else 'motion_detected') if supported else 'insufficient_coverage',
                                inliers=int(keep.sum()),coverage=float(coverage),p95_residual_px=float(np.percentile(residual,95)),max_corner_motion_px=float(motion.max()))
        results.append(item)
    return {'anchor_frame':anchor,'status':'stable' if all(r['status']=='stable' for r in results) else ('motion_detected' if any(r['status']=='motion_detected' for r in results) else 'unverified'),
            'method':'table-plane LK forward/backward and RANSAC homography; threshold 3 px; no camera-motion compensation', 'frames':results}

def run(inputs, output):
    output.mkdir(parents=True,exist_ok=False)
    cases=[];sources={}
    for folder in sorted((inputs/'videos').iterdir()):
        meta=json.loads((folder/'input_summary.json').read_text());camera=json.loads((folder/'camera.json').read_text())
        for name,digest in meta['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Input changed: '+name)
        video=Path(meta['video_path'])
        if sha(video)!=meta['video_sha256']:raise ValueError('Video changed')
        sources[str(folder/'input_summary.json')]=sha(folder/'input_summary.json')
        tracks=[json.loads(s) for s in (folder/'timed_tracks.jsonl').read_text().splitlines()]
        seeds=json.loads((folder/'candidate_contexts.json').read_text())
        cap=cv2.VideoCapture(str(video));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release()
        if len(frames)!=len(tracks):raise ValueError('Decode count mismatch')
        ts=np.array([r['timestamp_s'] for r in tracks]);dt=float(np.median(np.diff(ts)))
        for seed in seeds:
            fid=seed['seed_frame_id'];pose=min(camera['poses'],key=lambda p:abs(p['frame_id']-fid))
            indices=list(range(max(0,fid-8),min(len(tracks),fid+9)))
            rows=[tracks[i] for i in indices if tracks[i]['eligible_high_confidence']]
            stable=stability(frames,pose,indices)
            lo=float(np.interp(fid-2.5,np.arange(len(ts)),ts));hi=float(np.interp(fid+2.5,np.arange(len(ts)),ts))
            fit=search(rows,pose,lo,hi)
            case={'case_id':folder.name+'_'+str(fid),'video_id':folder.name,'video_path':str(video),'video_sha256':meta['video_sha256'],
                  'seed_frame_id':fid,'seed_time_s':float(ts[fid]),'tracks':tracks[indices[0]:indices[-1]+1],
                  'camera':pose,'camera_stability':stable,'fit':fit,'reasons':[]}
            reasons=case['reasons']
            if stable['status']!='stable':reasons.append('camera_'+stable['status'])
            if fit:
                fit['fractional_frame']=float(np.interp(fit['tau'],ts,np.arange(len(ts))))
                fit['delta_frames']=fit['fractional_frame']-fid
                p=np.array(fit['parameters']);a,b=design([r['timestamp_s'] for r in rows],fit['tau']);xyz=b+np.einsum('nij,j->ni',a,p)
                fit['contact_xyz_m']=[float(p[0]),float(p[1]),RADIUS]
                fit['velocity_before_m_s']=p[2:5].tolist();fit['velocity_after_m_s']=p[5:8].tolist()
                fit['fit_frame_ids']=[r['frame_id'] for r in rows]
                limits=np.array([2.740,1.525] if pose['edge01_axis']=='length' else [1.525,2.740])/2
                if np.any(np.abs(p[:2])>limits+RADIUS):reasons.append('contact_outside_table')
                if not (p[4]<-.1 and p[7]>.1):reasons.append('not_down_then_up')
                if np.min(xyz[:,2])<RADIUS-.01:reasons.append('trajectory_below_table')
                if max(np.linalg.norm(p[2:5]),np.linalg.norm(p[5:8]))>40:reasons.append('implausible_speed')
                if p[4]<-.1 and not .2<=p[7]/(-p[4])<=1.2:reasons.append('implausible_vertical_restitution')
                if fit['rmse_px']>3 or fit['max_error_px']>6:reasons.append('large_projection_residual')
                if fit['search_boundary']:reasons.append('time_at_search_boundary')
                if fit['bic_gain_over_smooth'] is None or fit['bic_gain_over_smooth']<6:reasons.append('smooth_path_not_excluded')
                if np.ptp(fit['one_pixel_cost_band_s'])/dt>.75:reasons.append('broad_time_cost_band')
                sensitivities=[]
                for alt_pose in camera['poses']:
                    alt=search(rows,alt_pose,lo,hi)
                    if alt:sensitivities.append({'kind':'reviewed_camera','frame':alt_pose['frame_id'],'tau':alt['tau'],'rmse_px':alt['rmse_px']})
                short=[r for r in rows if abs(r['frame_id']-fid)<=6]
                alt=search(short,pose,lo,hi)
                if alt:sensitivities.append({'kind':'six_frame_half_window','tau':alt['tau'],'rmse_px':alt['rmse_px']})
                else:reasons.append('short_window_insufficient')
                fit['sensitivity']=sensitivities
                fit['sensitivity_spread_frames']=float(np.ptp([fit['tau']]+[s['tau'] for s in sensitivities])/dt)
                if fit['sensitivity_spread_frames']>.5:reasons.append('camera_or_window_sensitive')
                times=np.linspace(ts[indices[0]],ts[indices[-1]],161);a,b=design(times,fit['tau']);xyz=b+np.einsum('nij,j->ni',a,p);uv,_=project(xyz,pose)
                fit['curve']=[{'time_s':float(t),'xyz_m':x.tolist(),'xy_px':u.tolist()} for t,x,u in zip(times,xyz,uv)]
            else:reasons.append('fit_unavailable')
            physical={'contact_outside_table','not_down_then_up','trajectory_below_table','implausible_speed','implausible_vertical_restitution','large_projection_residual','fit_unavailable'}
            case['status']='model_inconsistent' if physical.intersection(reasons) else ('needs_review' if reasons else 'provisional_pass')
            cases.append(case)
            print(case['case_id'],case['status'],reasons, 'time',round(fit['fractional_frame'],3) if fit else None,flush=True)
    report={'schema_version':1,'model':'TT3D-inspired gravity-only piecewise ballistic fit; no drag/spin; independent pre/post velocities',
            'not_full_tt3d':True,'human_contact_ground_truth_used':False,'production_events_modified':False,
            'thresholds_are_experimental':True,'sources':sources,'script_sha256':sha(__file__),'cases':cases}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs,args.output)

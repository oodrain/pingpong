"""Automatic sparse-track ROI retry with stateless aligned three-frame inference."""
import cv2,numpy as np
from .roi import automatic_roi
POLICY={'schema':1,'trigger':'fewer than 25 percent actual >=.75 observations and at least five coarse observations',
        'minimum_confidence':.75,'original_pixel_threshold':.7,'minimum_roi_fraction':.6,'roi_margin_fraction':.1,
        'retain_original_high_confidence_exactly':True,'retry_targets':'missing or confidence <.75; no terminal-context fabrication',
        'component_mass_fraction':.85,'support_window_s':.12,'max_speed_image_diagonals_per_s':3.,
        'roi_from_full_video_automatic_coarse_points_only':True,'offline_two_pass':True,'observations_interpolated':False}
def good(r):return bool(r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=.75 and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y')))
def plan(tracks,size):
    n=len(tracks);w,h=size;coarse=[[r['x'],r['y']] for r in tracks if r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=.7 and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y'))]
    trigger=n>0 and sum(good(r) for r in tracks)/n<.25 and len(coarse)>=5
    roi=automatic_roi(coarse,w,h) if trigger else None
    return {'triggered':bool(trigger and roi is not None),'roi':roi,'target_frames':[i for i in range(n-1) if not good(tracks[i])] if roi is not None else [],'original_high_frames':[i for i,r in enumerate(tracks) if good(r)]}
def infer_triplet(detector,images,roi):
    x,y,w,h=roi;norm=[]
    for im in images:
        norm.append(detector.normalize(im[y:y+h,x:x+w]) if im is not None else detector.normalize(np.zeros((h,w,3),np.uint8)))
    heat=detector.session.run(None,{detector.name:np.concatenate(norm,axis=0)[None]})[0][0,1];mask=heat>.7
    if not mask.any():return None
    yy,xx=np.nonzero(mask);weights=heat[mask].astype(np.float64);total=weights.sum();_,labels=cv2.connectedComponents(mask.astype(np.uint8));masses=np.bincount(labels[mask],weights=weights)[1:];fraction=float(masses.max()/total) if len(masses) else 0.
    return {'x':float(xx@weights/total*w/512+x),'y':float(yy@weights/total*h/288+y),'confidence':float(heat[mask].max()),'dominant_component_mass_fraction':fraction}
def merge(tracks,candidates,size):
    w,h=size;diag=float(np.hypot(w,h));times=np.array([r['timestamp_s'] for r in tracks]);pool={i:dict(r) for i,r in enumerate(tracks) if good(r)}
    eligible={i:c for i,c in candidates.items() if c and c['confidence']>=.75 and c['dominant_component_mass_fraction']>=.85 and 0<=c['x']<w and 0<=c['y']<h}
    pool.update(eligible);accepted=[];result=[dict(r) for r in tracks]
    for i,c in eligible.items():
        if good(tracks[i]):continue
        neighbours=[j for j,p in pool.items() if j!=i and 0<abs(times[j]-times[i])<=.12+1e-9 and np.hypot(c['x']-p['x'],c['y']-p['y'])/abs(times[j]-times[i])<=3*diag]
        if not neighbours:continue
        result[i].update(detected=True,in_image=True,x=c['x'],y=c['y'],confidence=c['confidence'],missing_reason=None,
                         coordinate_source='automatic_sparse_roi_retry',observations_interpolated=False)
        accepted.append(i)
    for i,r in enumerate(tracks):
        if good(r):assert result[i]==r
    return result,accepted
def repair_video(video,tracks,size,detector):
    from .startup_pts_capture import StartupPTSCapture
    p=plan(tracks,size)
    if not p['triggered']:return [dict(r) for r in tracks],dict(p,model_calls=0,candidates={},accepted_frames=[])
    targets=set(p['target_frames']);cap=StartupPTSCapture(str(video));fid=0;times=[];history=[None,None];candidates={}
    while True:
        ok,im=cap.read()
        if not ok:break
        times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
        history=(history+[im])[-3:]
        if fid-1 in targets:candidates[fid-1]=infer_triplet(detector,history,p['roi'])
        fid+=1
    cap.release();assert fid==len(tracks);assert np.allclose(np.array(times)-times[0],[r['timestamp_s'] for r in tracks],atol=1e-6,rtol=0)
    assert set(candidates)==targets
    repaired,accepted=merge(tracks,candidates,size)
    return repaired,dict(p,model_calls=len(candidates),candidates={str(k):v for k,v in candidates.items()},accepted_frames=accepted)

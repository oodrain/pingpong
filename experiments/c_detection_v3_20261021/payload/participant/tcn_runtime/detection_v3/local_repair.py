"""Experimental wrapper; original global sparse repair is preserved."""
import cv2,numpy as np
from .. import ball_repair as original
from ..roi import automatic_roi
from .policy import local_targets
from .curve_rescue import propose,shifted_roi,merge
infer_triplet=original.infer_triplet

def plan(tracks,size):
    old=original.plan(tracks,size)
    if old['triggered']:
        return dict(old,branch='original_global_sparse',local_support={})
    targets,support=local_targets(tracks)
    points=[[r['x'],r['y']] for r in tracks if r.get('detected') and r.get('in_image',True)
        and (r.get('confidence') or 0)>=.7 and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y'))]
    roi=automatic_roi(points,*size) if targets else None
    return dict(triggered=bool(targets and roi is not None),roi=roi,
        target_frames=targets if roi is not None else [],original_high_frames=old['original_high_frames'],
        branch='local_original_support',local_support={str(i):v for i,v in support.items()},
        original_global_triggered=False)

def repair_video(video,tracks,size,detector):
    from ..startup_pts_capture import StartupPTSCapture
    p=plan(tracks,size)
    if not p['triggered']:return [dict(r) for r in tracks],dict(p,model_calls=0,candidates={},accepted_frames=[])
    targets=set(p['target_frames']);cap=StartupPTSCapture(str(video));fid=0;times=[];history=[None,None];candidates={};checks={}
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000);history=(history+[im])[-3:]
            if fid-1 in targets:
                target=fid-1;c=infer_triplet(detector,history,p['roi']);candidates[target]=c
                if p['branch']=='local_original_support':
                    decision=propose(tracks,target,c,size,p['roi'])
                    if decision['needs_image_check']:
                        _,ids=original.merge(tracks,{target:c},size)
                        shift=shifted_roi(p['roi'],size)
                        if target in ids:
                            import hashlib
                            checks[str(target)]=dict(roi=shift,candidate=infer_triplet(detector,history,shift),
                                source_RGB_sha256={str(i):hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest() for i,im in zip([target-1,target,target+1],history)})
                        else:checks[str(target)]=dict(roi=shift,candidate=None,reason='no_original_merge_support')
            fid+=1
    finally:cap.release()
    assert fid==len(tracks) and set(candidates)==targets
    assert np.allclose(np.array(times)-times[0],[r['timestamp_s'] for r in tracks],atol=1e-6,rtol=0)
    previous,ids=original.merge(tracks,candidates,size)
    source=dict(coarse=tracks,tracks=previous,repair=dict(p,candidates={str(i):v for i,v in candidates.items()},accepted_frames=ids))
    enriched,compatible,info=merge(source,original,size,checks)
    shift_calls=sum('source_RGB_sha256' in r for r in checks.values())
    info.update(model_calls=len(candidates)+shift_calls,base_roi_calls=len(candidates),shifted_roi_calls=shift_calls,
        compatibility_tracks=compatible,new_base_inference=True,new_shifted_inference=True)
    return enriched,info

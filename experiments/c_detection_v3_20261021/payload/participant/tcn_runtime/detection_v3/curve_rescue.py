"""Add a fixed, image-confirmed curved-path rescue to the frozen v2 gate."""
import math
from .admission_v2 import judge as v2_judge,good
POLICY=dict(schema=3,v2_points_retained=True,original_global_branch_preserved=True,
    rescue='v2 position rejection only; nearest original bracketing anchors plus nearest third original trusted point within .12s',
    trajectory='quadratic Lagrange position check; never emits predicted coordinates',
    radius='absolute Lagrange weights times full-frame heat-cell diagonal plus ROI heat-cell diagonal',
    image_check='same144 model and triplet in automatic ROI translated one heat cell in x/y, clamped in image',
    image_thresholds='original confidence .75 and component fraction .85',
    image_agreement='distance no larger than sum of the two ROI heat-cell diagonals',
    event_compatibility='enriched v3 tracks feed unchanged TCN; v2 tracks retained for diagnostics only',
    manual_frame_or_box_inputs=False,training_steps=0,parameter_sweep=False)

def propose(tracks,fid,candidate,size,roi):
    old=v2_judge(tracks,fid,candidate,size,roi)
    if old['allow']:return dict(path='v2',allow=True,v2=old,needs_image_check=False)
    fail=dict(path='rejected',allow=False,v2=old,needs_image_check=False)
    if old['reason']!='outside_position_corridor':return fail
    t=tracks[fid]['timestamp_s'];ids=[i for i,r in enumerate(tracks) if good(r) and 0<abs(r['timestamp_s']-t)<=.120000001]
    anchors=old['anchor_frames']
    if not min(tracks[i]['timestamp_s'] for i in anchors)<t<max(tracks[i]['timestamp_s'] for i in anchors):
        return dict(fail,rescue_reason='no_original_bracketing')
    others=[i for i in ids if i not in anchors]
    if not others:return dict(fail,rescue_reason='fewer_than3_original_anchors')
    third=min(others,key=lambda i:(abs(tracks[i]['timestamp_s']-t),i));anchors=sorted([*anchors,third],key=lambda i:tracks[i]['timestamp_s'])
    relative=[tracks[i]['timestamp_s']-t for i in anchors];weights=[]
    for a in range(3):
        value=1.
        for b in range(3):
            if b!=a:value*=(-relative[b])/(relative[a]-relative[b])
        weights.append(value)
    xy=[sum(w*tracks[i][key] for w,i in zip(weights,anchors)) for key in ['x','y']]
    radius=sum(abs(w) for w in weights)*math.hypot(size[0]/512,size[1]/288)+math.hypot(roi[2]/512,roi[3]/288)
    residual=math.hypot(candidate['x']-xy[0],candidate['y']-xy[1])
    curve=dict(anchor_frames=anchors,weights=weights,predicted_xy=xy,residual_px=residual,radius_px=radius,
        predicted_coordinates_are_observations=False)
    if residual>radius+1e-9:return dict(fail,curve=curve,rescue_reason='outside_curved_corridor')
    return dict(path='curve_pending_image',allow=False,v2=old,curve=curve,needs_image_check=True)

def shifted_roi(roi,size):
    x,y,w,h=roi;dx=max(1,math.ceil(w/512));dy=max(1,math.ceil(h/288))
    return [min(max(x+dx,0),size[0]-w),min(max(y+dy,0),size[1]-h),w,h]

def image_agrees(candidate,other,roi,shifted):
    limit=math.hypot(roi[2]/512,roi[3]/288)+math.hypot(shifted[2]/512,shifted[3]/288)
    if shifted==roi:return dict(allow=False,reason='clamped_ROI_identical',limit_px=limit)
    if other is None:return dict(allow=False,reason='shifted_no_detection',limit_px=limit)
    distance=math.hypot(other['x']-candidate['x'],other['y']-candidate['y'])
    quality=other['confidence']>=.75 and other['dominant_component_mass_fraction']>=.85
    return dict(allow=bool(quality and distance<=limit+1e-9),reason='image_consistent' if quality and distance<=limit+1e-9 else 'shifted_quality_or_location_disagreement',distance_px=distance,limit_px=limit)

def merge(source,original,size,image_checks):
    coarse=source['coarse'];info=source['repair'];candidates={int(i):c for i,c in info['candidates'].items()}
    if info['branch']=='original_global_sparse':
        tracks,accepted=original.merge(coarse,candidates,size)
        return tracks,tracks,dict(info,curve_audit={},curve_rescued_frames=[],image_checks={})
    import copy
    event_tracks=copy.deepcopy(coarse);tracks=copy.deepcopy(coarse);accepted=[];v2accepted=[];rescued=[];audit={}
    for fid,c in candidates.items():
        decision=propose(coarse,fid,c,size,info['roi']);decision['accepted']=False
        selected=bool(decision['allow'])
        if decision['needs_image_check']:
            check=image_checks[str(fid)];shift=shifted_roi(info['roi'],size)
            assert check['roi']==shift
            agree=image_agrees(c,check['candidate'],info['roi'],shift);decision['image_agreement']=agree;selected=agree['allow']
        if selected:
            result,ids=original.merge(coarse,{fid:c},size)
            if fid in ids:
                tracks[fid]=result[fid];accepted.append(fid);decision['accepted']=True
                if decision['path']=='v2':event_tracks[fid]=result[fid];v2accepted.append(fid)
                else:rescued.append(fid);decision['path']='curve_and_image'
        audit[str(fid)]=decision
    return tracks,event_tracks,dict(info,accepted_frames=accepted,v2_accepted_frames=v2accepted,
        curve_rescued_frames=rescued,curve_audit=audit,image_checks=image_checks,
        events_use_enriched_curve_points=True)

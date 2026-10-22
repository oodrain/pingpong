"""Resolution-scaled positional support from original observations only.

This is a heuristic localization gate, not a calibrated uncertainty bound.
Predicted coordinates are never emitted as detections.
"""
import math
POLICY=dict(schema=2,support_window_s=.12,anchors='original trustworthy only',
    anchor_choice='nearest bracketing; otherwise nearest2 same-side',
    one_sided_extension='at most one anchor interval',
    tolerance='sum absolute interpolation weights times full-frame heat-cell diagonal plus ROI heat-cell diagonal',
    heat_size=[512,288],original_global_sparse_branch_unchanged=True,
    candidate_corroboration=False,synthetic_observations=False,parameter_sweep=False)

def good(r):
    return bool(r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=.75
        and all(r.get(k) is not None and math.isfinite(r[k]) for k in ('x','y')))

def judge(tracks,fid,candidate,size,roi):
    if candidate is None:return dict(allow=False,reason='no_candidate')
    if good(tracks[fid]):return dict(allow=False,reason='retain_original_trusted')
    if candidate['confidence']<.75 or candidate['dominant_component_mass_fraction']<.85:
        return dict(allow=False,reason='original_candidate_quality_gate')
    w,h=size
    if not 0<=candidate['x']<w or not 0<=candidate['y']<h:return dict(allow=False,reason='outside_image')
    t=tracks[fid]['timestamp_s']
    indices=[i for i,r in enumerate(tracks) if good(r) and 0<abs(r['timestamp_s']-t)<=.12+1e-9]
    left=sorted([i for i in indices if tracks[i]['timestamp_s']<t],key=lambda i:tracks[i]['timestamp_s'],reverse=True)
    right=sorted([i for i in indices if tracks[i]['timestamp_s']>t],key=lambda i:tracks[i]['timestamp_s'])
    if left and right:a,b=left[0],right[0];mode='bracket'
    elif len(left)>=2:a,b=left[1],left[0];mode='extrapolate_after'
    elif len(right)>=2:a,b=right[0],right[1];mode='extrapolate_before'
    else:return dict(allow=False,reason='fewer_than_two_original_anchors')
    ta,tb=tracks[a]['timestamp_s'],tracks[b]['timestamp_s'];interval=tb-ta
    assert interval>0
    if mode=='extrapolate_after' and t-tb>interval+1e-9:return dict(allow=False,reason='extrapolation_beyond_one_interval',anchor_frames=[a,b])
    if mode=='extrapolate_before' and ta-t>interval+1e-9:return dict(allow=False,reason='extrapolation_beyond_one_interval',anchor_frames=[a,b])
    alpha=(t-ta)/interval;weights=[1-alpha,alpha]
    prediction=[weights[0]*tracks[a][key]+weights[1]*tracks[b][key] for key in ['x','y']]
    coarse_cell=math.hypot(w/512,h/288);roi_cell=math.hypot(roi[2]/512,roi[3]/288)
    radius=(abs(weights[0])+abs(weights[1]))*coarse_cell+roi_cell
    residual=math.hypot(candidate['x']-prediction[0],candidate['y']-prediction[1])
    return dict(allow=residual<=radius+1e-9,reason='position_supported' if residual<=radius+1e-9 else 'outside_position_corridor',
        anchor_frames=[a,b],mode=mode,weights=weights,predicted_xy=prediction,residual_px=residual,radius_px=radius,
        predicted_coordinates_are_observations=False)

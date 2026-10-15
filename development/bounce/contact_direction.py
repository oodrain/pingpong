"""Image motion evidence for contact review; no airborne 3D velocity claim."""
import numpy as np

def direction_evidence(proposal,corners,edge01_axis=None):
    before=np.asarray(proposal['velocity_before_px_s'],float)
    after=np.asarray(proposal['velocity_after_px_s'],float)
    q=np.asarray(corners,float)
    if q.shape!=(4,2) or not np.isfinite(q).all() or not np.isfinite([before,after]).all():
        raise ValueError('Nonfinite direction evidence')
    if edge01_axis not in (None,'length','width'):raise ValueError('Unknown axis declaration')
    delta=float(np.linalg.norm(after-before))
    # Recover the time unit of the frozen proposal's fitted kink; not video FPS.
    unit=proposal['velocity_change_px_per_nominal_interval']/delta if delta>0 else 0
    pre=before*unit;post=after*unit
    speeds=[float(np.linalg.norm(v)) for v in (pre,post)]
    cosine=float(np.dot(pre,post)/(speeds[0]*speeds[1])) if min(speeds)>0 else None
    # The existing proposal rule uses a 2px minimum change. Apply the same
    # resolution floor to signed components; values near zero cannot decide a hit.
    axes=np.array([q[1]-q[0],q[3]-q[0]])
    lengths=np.linalg.norm(axes,axis=1)
    if min(lengths)<=0:raise ValueError('Degenerate axes')
    axes/=lengths[:,None]
    a=axes@pre;b=axes@post
    states=['uncertain' if min(abs(x),abs(y))<2 else 'reversed' if x*y<0 else 'same_direction' for x,y in zip(a,b)]
    long_index=0 if edge01_axis=='length' else 1 if edge01_axis=='width' else None
    long_state=states[long_index] if long_index is not None else 'physical_axis_unknown'
    return {'image_direction_cosine':cosine,'speeds_px_per_nominal_interval':speeds,
        'signed_edge_components_before':a.tolist(),'signed_edge_components_after':b.tolist(),
        'edge_direction_states':states,'edge01_axis':edge01_axis,'longitudinal_direction':long_state,
        'strong_image_reversal':bool(cosine is not None and cosine<0 and min(speeds)>=2),
        'nominal_interval_s':float(unit),'minimum_signed_component_px':2,
        'limitations':['Image direction is not physical longitudinal velocity.',
            'Automatic polygon may not remain valid through the clip; axis labels are supplementary only.']}

def fuse_direction_wrist(direction,wrist):
    """Keep disagreements for review; never classify absence of a wrist as bounce."""
    multi=wrist['status']=='multi_frame_wrist_proximity'
    # An explicit contradictory length direction blocks automatic confirmation.
    reversal=direction['strong_image_reversal'] and direction['longitudinal_direction'] not in ('same_direction','uncertain')
    if multi and reversal:kind='racket_contact_candidate'
    elif multi:kind='wrist_motion_conflict'
    elif reversal:kind='motion_reversal_unresolved'
    else:kind='contact_type_unresolved'
    return {'event':kind,'adopt':False,'racket_contact_confirmed':False,'table_contact_confirmed':False,
        'hypothetical_auxiliary_exclusion':kind=='racket_contact_candidate'}

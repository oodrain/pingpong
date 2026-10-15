"""Temporal image evidence; no human intervals and no confirmed contact labels."""
import numpy as np
from development.bounce.wrist_evidence import iou

def transient_wrist_visit(frames,seed):
    by_frame={r['frame_id']:r['observations'] for r in frames};visits=[];incomplete=[]
    for fid in range(seed-1,seed+2):
        for anchor in by_frame.get(fid,[]):
            if not anchor['near']:continue
            neighbours=[]
            for other_fid in (fid-1,fid+1):
                matches=[o for o in by_frame.get(other_fid,[]) if o['hand']==anchor['hand']
                    and not o.get('quality_reasons') and iou(o['person_box'],anchor['person_box'])>=.3]
                neighbours.append(max(matches,key=lambda o:iou(o['person_box'],anchor['person_box'])) if matches else None)
            if any(o is None for o in neighbours):incomplete.append(fid);continue
            distances=[neighbours[0]['distance_px'],anchor['distance_px'],neighbours[1]['distance_px']]
            # Existing image resolution floor (2px), without relaxing wrist proximity.
            if min(distances[0]-distances[1],distances[2]-distances[1])>=2:
                visits.append({'closest_frame':fid,'hand':anchor['hand'],'person_box':anchor['person_box'],
                    'distance_px_before_at_after':distances,'distance_in_forearms_at_closest':anchor['distance_in_forearms'],
                    'neighbour_frame_ids':[fid-1,fid+1]})
    return {'supported':bool(visits),'visits':visits,'near_anchor_without_matched_neighbours':sorted(set(incomplete)),
        'minimum_distance_change_px':2,'racket_contact_confirmed':False,
        'limitations':['Closest wrist approach does not prove racket contact.','Hand identity is anatomical, not verified playing hand.']}

def quadratic_phase_diagnostic(window,seed):
    """Independent quadratic flight pieces; derivatives are diagnostic only."""
    reasons=list(window['reasons']);rows=window['rows'];centre=next((r for r in rows if r['frame_id']==seed),None)
    if reasons or centre is None:return {'available':False,'reasons':reasons or ['seed_not_available']}
    unit=float(np.median(np.diff([r['timestamp_s'] for r in rows])))
    if unit<=0:raise ValueError('Invalid timestamps')
    pieces=[]
    for part in ([r for r in rows if r['frame_id']<seed],[r for r in rows if r['frame_id']>seed]):
        if len(part)<3:return {'available':False,'reasons':['insufficient_flight_points']}
        t=np.array([(r['timestamp_s']-centre['timestamp_s'])/unit for r in part]);xy=np.array([[r['x'],r['y']] for r in part])
        design=np.column_stack([np.ones(len(t)),t,t*t]);coef=np.linalg.lstsq(design,xy,rcond=None)[0]
        residual=np.linalg.norm(design@coef-xy,axis=1)
        pieces.append({'frame_ids':[r['frame_id'] for r in part],'coefficients':coef.tolist(),
            'rmse_px':float(np.sqrt(np.mean(residual**2))),'max_error_px':float(max(residual))})
    before=np.asarray(pieces[0]['coefficients'])[1];after=np.asarray(pieces[1]['coefficients'])[1]
    speeds=[float(np.linalg.norm(v)) for v in (before,after)]
    cosine=float(before@after/(speeds[0]*speeds[1])) if min(speeds)>0 else None
    return {'available':True,'pieces':pieces,'image_direction_cosine_at_seed':cosine,
        'nominal_interval_s':unit,'fit_quality_supported':all(p['rmse_px']<=3 and p['max_error_px']<=6 for p in pieces),
        'used_for_hit_classification':False,'note':'Fits may contain another unobserved event; derivative at seed is not a physical 3D velocity.'}

def supplement_frozen_direction(direction,visit,original_joint_flag):
    reversal=direction['strong_image_reversal'] and direction['longitudinal_direction'] not in ('same_direction','uncertain')
    supplemental=bool(reversal and visit['supported'])
    return {'hit_flag':bool(original_joint_flag or supplemental),'original_joint_flag':bool(original_joint_flag),
        'transient_wrist_supplement':supplemental,'adopt':False,'racket_contact_confirmed':False,'table_contact_confirmed':False}

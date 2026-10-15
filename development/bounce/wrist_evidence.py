"""Conservative image-space wrist proximity, never proof of racket contact."""
import math
import numpy as np

def wrist_observations(ball,people):
    if not ball.get('eligible_high_confidence'):return []
    xy=np.array([ball['x'],ball['y']],float);observations=[]
    for person_id,p in enumerate(people):
        if p['score']<.9:continue
        key=np.array(p['keypoints'],float);scores=np.array(p['keypoint_scores'],float)
        for elbow,wrist,side in ((7,9,'left'),(8,10,'right')):
            # These are raw heatmap scores, not calibrated probabilities.
            if min(scores[elbow],scores[wrist])<2:continue
            arm=float(np.linalg.norm(key[wrist,:2]-key[elbow,:2]))
            if not math.isfinite(arm) or arm<10:continue
            distance=float(np.linalg.norm(xy-key[wrist,:2]));ratio=distance/arm
            observations.append({'person_index':person_id,'person_box':p['box'],'hand':side,'distance_px':distance,
                                 'forearm_px':arm,'distance_in_forearms':ratio,'wrist_score_raw':float(scores[wrist]),
                                 'wrist_xy':key[wrist,:2].tolist(),'elbow_xy':key[elbow,:2].tolist(),'near':ratio<=1.25})
    return observations

def iou(a,b):
    left=np.maximum(a[:2],b[:2]);right=np.minimum(a[2:],b[2:]);inter=float(np.prod(np.maximum(0,right-left)))
    total=float(np.prod(np.maximum(0,np.array(a[2:])-a[:2]))+np.prod(np.maximum(0,np.array(b[2:])-b[:2]))-inter)
    return inter/total if total>0 else 0

def quality_checked_observations(ball,people):
    """Retain rejected observations for audit; exclude ambiguous person assignment."""
    result=wrist_observations(ball,people)
    for observation in result:
        p=people[observation['person_index']];box=p['box'];height=box[3]-box[1]
        wrist=np.array(observation['wrist_xy']);reasons=[]
        if height<=0 or observation['forearm_px']>.45*height:reasons.append('forearm_implausible_relative_to_person_box')
        if any(wrist<np.array(box[:2])-5) or any(wrist>np.array(box[2:])+5):reasons.append('wrist_outside_own_box')
        overlapping=[]
        for index,other in enumerate(people):
            if index==observation['person_index'] or other['score']<.9 or iou(box,other['box'])>=.7:continue
            b=other['box']
            if b[0]<=wrist[0]<=b[2] and b[1]<=wrist[1]<=b[3]:overlapping.append(index)
        if overlapping:reasons.append('wrist_inside_another_person_box')
        observation.update(geometrically_near=observation['near'],quality_reasons=reasons,overlapping_person_indices=overlapping,
                           near=bool(observation['near'] and not reasons))
    return result

def summarize(frames):
    near=[(r['frame_id'],o) for r in frames for o in r['observations'] if o['near']]
    consistent=any(0<abs(f-g)<=2 and a['hand']==b['hand'] and iou(a['person_box'],b['person_box'])>=.3 for f,a in near for g,b in near)
    return {'status':'multi_frame_wrist_proximity' if consistent else 'single_frame_wrist_proximity' if near else 'no_supported_wrist_proximity',
            'near_frame_ids':sorted(set(f for f,_ in near)),'racket_contact_confirmed':False,
            'limitations':['2D overlap does not prove contact','wrists are not racket tips','person role and handedness unverified','missing/occluded wrists cannot rule out a hit']}

"""Frozen experimental auxiliary contact policy, independent of evaluation GT."""
from .trajectory_events import screen_auxiliary_candidate

POLICY={'schema_version':1,'name':'preserve_v0_add_ready_projection_aux_filter_hits_v1',
        'baseline_exclusion_enabled':False,'near_baseline_tolerance_frames':2,
        'auxiliary_requires_complete_motion_context':True,
        'auxiliary_requires_current_usable_table_projection':True,
        'auxiliary_requires_inside_projection':True,'human_inputs':False,
        'projection_membership_is_not_verified_world_contact':True}


def experimental_landings(tracks,baseline,proposals,cases,filter_hits=True,exclude_primary_hits=False):
    lookup={c['seed_frame']:c for c in cases}
    def primary_hit(seed):
        c=lookup.get(seed)
        return bool(c and c['event']=='racket_contact_candidate' and c.get('direction') and not c['window_reasons']
                    and (c.get('wrist_visit',{}).get('supported') or c.get('full_frame_wrist_visit',{}).get('supported')
                         or c.get('person_visit',{}).get('supported') or c.get('racket_visit',{}).get('supported')
                         or c.get('decision',{}).get('kinematic_hit_supported')))
    result=[dict(p) for p in baseline if not (filter_hits and exclude_primary_hits and primary_hit(p['frame_id']))]
    decisions=[]
    primary=[dict(p,video_id='input_video') for p in baseline]
    for p in proposals['proposals']:
        seed=p['frame_id'];ball=tracks[seed];case=lookup[seed]
        candidate={'video_id':'input_video','frame_id':seed,'x':ball['x'],'y':ball['y']}
        screen=screen_auxiliary_candidate(candidate,proposals,primary)
        position=next((r for r in case['local_positions'] if r['frame_id']==seed),{})
        reasons=[]
        if screen['status']!='ready_for_event_review':reasons.append(screen['status'])
        if not ball.get('detected'):reasons.append('ball_missing')
        if not position.get('available'):reasons.append('current_geometry_unavailable')
        elif position.get('outside_projection'):reasons.append('outside_current_projection')
        if case['window_reasons']:reasons.append('motion_context_incomplete')
        if filter_hits and case['event']=='racket_contact_candidate':reasons.append('automatic_hit_suggestion')
        if not reasons:result.append({'frame_id':seed,'x':ball['x'],'y':ball['y']})
        decisions.append({'seed_frame':seed,'included':not reasons,'reasons':reasons,
                          'world_contact_confirmed':False})
    return sorted(result,key=lambda r:r['frame_id']),decisions

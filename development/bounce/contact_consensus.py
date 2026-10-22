"""Train-selected experimental contact safeguards and curved auxiliaries.

Operates on automatic evidence only. Image projection and motion cannot verify
world height. The original contact model and formal baseline remain separate.
"""
from copy import deepcopy
import numpy as np
from .curved_contact_candidates import fit_curved_window,propose_curved_contacts,CurvedContactConfig
from .trajectory_events import observed_runs
from .person_motion_contact import local_table_position


def cosine(before,after):
    a=np.asarray(before,float);b=np.asarray(after,float);den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/den) if den>0 else None


def apply_contact_consensus(cases,tracks,config):
    cases=deepcopy(cases);runs=observed_runs(tracks)
    for c in cases:
        seed=c['seed_frame'];direction=c.get('direction') or {};raw_cos=direction.get('image_direction_cosine')
        position=next((p for p in c['local_positions'] if p['frame_id']==seed),{})
        inside=bool(position.get('available') and not position.get('outside_projection'))
        original=c['event']=='racket_contact_candidate';blocked=False;kinematic=False;fit=None
        if (original and inside and not c['racket_visit']['supported']
            and (raw_cos is None or raw_cos>config['inside_hit_maximum_cosine'])):
            blocked=True
        if config['kinematic_supplement_enabled'] and not original and direction and not c['window_reasons']:
            run=next((r for r in runs if r[0]['frame_id']<=seed<=r[-1]['frame_id']),None)
            if run is not None:
                i=next(i for i,r in enumerate(run) if r['frame_id']==seed)
                if i>=3 and i+3<len(run):
                    fit=fit_curved_window(run[i-3:i+4],seed)
                    curved_cos=cosine(fit['velocity_before_px_s'],fit['velocity_after_px_s'])
                    kinematic=bool(c.get('learned_contact',{}).get('score',0)>=c.get('learned_contact',{}).get('threshold',.8)
                        and raw_cos is not None and raw_cos<=config['kinematic_maximum_cosine']
                        and curved_cos is not None and curved_cos<=config['kinematic_maximum_cosine']
                        and fit['bic_gain']>=6 and fit['fit_rmse_px']<=3
                        and fit['velocity_change_px_per_nominal_interval']>=2)
        flag=bool((original and not blocked) or kinematic)
        c['contact_consensus']={'original_hit_suggestion':original,'weak_inside_reversal_blocked':blocked,
            'kinematic_hit_supported':kinematic,'hit_suggestion':flag,'world_contact_confirmed':False}
        c['decision']['kinematic_hit_supported']=kinematic;c['decision']['review_hit_suggestion']=flag
        c['event']='racket_contact_candidate' if flag else 'legacy_table_candidate' if 'baseline' in c['source'] else 'contact_type_unresolved'
    return cases


def curved_auxiliary_landings(tracks,samples,cases,existing,config,proposals=None):
    q=next((s['corners'] for s in samples if s.get('usable')),None)
    if q is None:return list(existing),[]
    if proposals is None:proposals=propose_curved_contacts(tracks,q)
    output=[dict(p) for p in existing];decisions=[]
    for p in proposals['proposals']:
        seed=p['frame_id'];ball=tracks[seed];position=local_table_position(samples,ball,seed);reasons=[]
        d=cosine(p['velocity_before_px_s'],p['velocity_after_px_s'])
        if p['fit_rmse_px']>config['curved_maximum_rmse_px'] or p['bic_gain']<config['curved_minimum_bic_gain']:
            reasons.append('curved_fit_below_selected_quality')
        if p['tracking_spike_nearby']:reasons.append('possible_tracking_spike')
        if not position.get('available'):reasons.append('current_geometry_unavailable')
        elif position.get('outside_projection'):reasons.append('outside_current_projection')
        if d is None or d<config['curved_minimum_direction_cosine']:reasons.append('turnaround_requires_contact_type_evidence')
        if any(abs(c['seed_frame']-seed)<=2 and c['event']=='racket_contact_candidate' for c in cases):
            reasons.append('near_hit_candidate_cannot_separate_events')
        if any(abs(r['frame_id']-seed)<=2 for r in output):reasons.append('near_existing_contact')
        if not reasons:output.append({'frame_id':seed,'x':ball['x'],'y':ball['y']})
        decisions.append({'seed_frame':seed,'included':not reasons,'reasons':reasons,'fit':p,
                          'position':position,'image_direction_cosine':d,'human_label_inputs':False})
    return sorted(output,key=lambda p:p['frame_id']),decisions

"""Experimental table-event scoring for automatic curved proposals.

Human event labels are used offline to build fixed model assets only. The
runtime consumes automatic evidence; mixed events can retain table evidence
without erasing the separately estimated hit type.
"""
from copy import deepcopy
from .curved_contact_candidates import propose_curved_contacts
from .curved_contact_classifier import features,score_matrix
from .curved_contact_context import contextual_features,ballistic_features,net_context_features,CONTEXT_FEATURES,PHYSICS_FEATURES,LEGACY_PHYSICS_FEATURES,NET_FEATURES
from .contact_consensus import apply_contact_consensus
from .contact_output_policy import experimental_landings
from .person_motion_contact import local_table_position


def prepare_existing_contacts(tracks,baseline,proposals,cases,config):
    # Keep the old gesture branch; a weak screen direction alone does not veto
    # an established hit. A separate learned table branch evaluates touch.
    cases=apply_contact_consensus(cases,tracks,{
        'inside_hit_maximum_cosine':1.,'kinematic_supplement_enabled':config['kinematic_supplement_enabled'],
        'kinematic_maximum_cosine':-.8})
    result,_=experimental_landings(tracks,baseline,proposals,cases,True,True)
    return result,cases


def revised_landings(tracks,samples,baseline,proposals,cases,model,config,automatic=None,prepared=None,image_size=None):
    result,cases=prepare_existing_contacts(tracks,baseline,proposals,cases,config) if prepared is None else deepcopy(prepared)
    q=next((s['corners'] for s in samples if s.get('usable')),None)
    if q is None:return result,cases,[]
    if automatic is None:
        automatic=[]
        for p in propose_curved_contacts(tracks,q)['proposals']:
            seed=p['frame_id'];position=local_table_position(samples,tracks[seed],seed)
            if model['features']==NET_FEATURES:vector=net_context_features(p,position,q,tracks,samples,image_size)
            elif model['features'] in (PHYSICS_FEATURES,LEGACY_PHYSICS_FEATURES):vector=ballistic_features(p,position,q,tracks,samples,image_size,model['features'])
            elif model['features']==CONTEXT_FEATURES:vector=contextual_features(p,position,q,tracks)
            else:vector=features(p,position,q,model['features'])
            automatic.append({'seed_frame':seed,'proposal':p,'position':position,'vector':vector.tolist()})
    if not automatic:return result,cases,[]
    scores=score_matrix(model,[p['vector'] for p in automatic]);decisions=[]
    def ballistic_veto(item):
        if 'ballistic_veto_margin' not in config:return False
        if model['features'] not in (PHYSICS_FEATURES,NET_FEATURES):raise ValueError('Ballistic veto requires consensus across camera hypotheses')
        missing,rmse_log,_,maximum_margin=item['vector'][len(CONTEXT_FEATURES):len(PHYSICS_FEATURES)]
        return missing==0 and rmse_log<=1.0986122886681098 and maximum_margin<config['ballistic_veto_margin']
    rejection_threshold=config.get('table_rejection_threshold',0.)
    for existing in list(result):
        neighbours=[(item,float(score)) for item,score in zip(automatic,scores)
                    if abs(item['seed_frame']-existing['frame_id'])<=1 and item['position'].get('available')
                    and not item['proposal']['tracking_spike_nearby']]
        low_score=bool(neighbours and max(score for _,score in neighbours)<rejection_threshold)
        physical_rejection=bool(neighbours and all(ballistic_veto(item) for item,_ in neighbours))
        if low_score or physical_rejection:
            result.remove(existing)
            decisions.append({'seed_frame':existing['frame_id'],'included':False,'existing_removed':True,
                'reasons':['existing_contact_inconsistent_ballistic_hypotheses' if physical_rejection else 'existing_contact_low_automatic_table_score'],'table_score':max(score for _,score in neighbours),
                'threshold':rejection_threshold,'proposal':neighbours[0][0]['proposal'],
                'score_is_calibrated_probability':False,'world_contact_confirmed':False})
    for a,value in sorted(zip(automatic,scores),key=lambda v:float(v[1]),reverse=True):
        p=a['proposal'];seed=a['seed_frame'];position=a['position'];value=float(value);reasons=[]
        hit_neighbours=[c['seed_frame'] for c in cases if abs(c['seed_frame']-seed)<=2 and c['event']=='racket_contact_candidate']
        threshold=config['near_hit_table_threshold'] if hit_neighbours else config['table_threshold']
        if value<threshold:reasons.append('learned_table_score_below_selected_threshold')
        if ballistic_veto(a):reasons.append('inconsistent_ballistic_hypotheses')
        if p['tracking_spike_nearby']:reasons.append('possible_tracking_spike')
        if p.get('bic_gain_over_smooth_cubic',float('inf'))<config.get('minimum_cubic_bic_gain',float('-inf')):
            reasons.append('smooth_cubic_flight_not_excluded')
        if not position.get('available'):reasons.append('current_geometry_unavailable')
        elif position.get('outside_projection'):reasons.append('outside_current_projection')
        neighbours=[r for r in result if abs(r['frame_id']-seed)<=2]
        replace=None
        if neighbours:
            if config['retime_nearby_enabled'] and len(neighbours)==1 and abs(neighbours[0]['frame_id']-seed)==2:
                old_seed=neighbours[0]['frame_id'];old_scores=[float(v) for item,v in zip(automatic,scores) if item['seed_frame']==old_seed]
                if old_scores and value-max(old_scores)>=.2:replace=neighbours[0]
            if replace is None:reasons.append('near_existing_contact')
        if not reasons:
            if replace is not None:result.remove(replace)
            ball=tracks[seed];result.append({'frame_id':seed,'x':ball['x'],'y':ball['y']})
        decisions.append({'seed_frame':seed,'included':not reasons,'reasons':reasons,'table_score':value,'threshold':threshold,
            'nearby_hit_frames':hit_neighbours,'retimed_from_frame':replace['frame_id'] if not reasons and replace else None,
            'position':position,'proposal':p,'score_is_calibrated_probability':False,'world_contact_confirmed':False})
    return sorted(result,key=lambda p:p['frame_id']),cases,decisions

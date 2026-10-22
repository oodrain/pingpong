"""Experimental offline event proposals and windows. No labels or image-axis sign rule.

Uses both image coordinates and actual timestamps. Regions are image projections,
not reconstructed airborne ball positions. Racket identity requires extra evidence.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass(frozen=True)
class EventConfig:
    confidence: float=.75
    half_window: int=3
    merge_radius: int=2
    minimum_gain: float=.35
    minimum_rms_improvement_px: float=.3
    minimum_velocity_change_fraction: float=.01

def auxiliary_candidate_pool(baseline,vector):
    """Preserve baseline byte-level values; extra proposals never become final output.

    Deduplicate exact observations only. Temporal neighbours remain auxiliary
    evidence until an event classifier and one-event deduplication are validated.
    """
    primary=[dict(p) for p in baseline];seen={(p['video_id'],p['frame_id'],p['x'],p['y']) for p in primary};auxiliary=[]
    for proposal in vector:
        identity=(proposal['video_id'],proposal['frame_id'],proposal['x'],proposal['y'])
        if identity in seen:continue
        seen.add(identity);auxiliary.append(dict(proposal))
    return {'primary_predictions':primary,'auxiliary_candidates':auxiliary,'auxiliary_adoption_enabled':False}

def observed_runs(rows,config=EventConfig()):
    last_id=-1;last_time=-math.inf;runs=[];current=[]
    intervals=np.diff([r['timestamp_s'] for r in rows]);dt=float(np.median(intervals)) if len(intervals) else 1/30
    for r in rows:
        fid=r['frame_id'];t=r['timestamp_s']
        if not isinstance(fid,int) or fid<=last_id or not math.isfinite(t) or t<=last_time:raise ValueError('Frames and timestamps must increase')
        last_id=fid;last_time=t
        valid=r.get('detected') and r.get('in_image',True) and r.get('confidence') is not None and r['confidence']>=config.confidence
        if valid and not all(r.get(k) is not None and math.isfinite(r[k]) for k in ('x','y')):raise ValueError('Nonfinite detection')
        if not valid:
            if current:runs.append(current);current=[]
            continue
        if current and (fid!=current[-1]['frame_id']+1 or t-current[-1]['timestamp_s']>2*dt):
            runs.append(current);current=[]
        current.append(r)
    if current:runs.append(current)
    return runs

def screen_auxiliary_candidate(candidate, proposal_result, baseline):
    """Label review readiness from frozen inference evidence, never from event GT.

    Nearby baseline events remain timing alternatives, not additional contacts.
    Image polygon membership routes review only; it cannot establish world height.
    Uses the existing three-point context and two-frame event neighbourhood.
    """
    seed=candidate['frame_id'];reasons=[]
    proposals=proposal_result.get('proposals',[]) if proposal_result else []
    proposal=next((p for p in proposals if p['frame_id']==seed),None)
    runs=proposal_result.get('observed_runs',[]) if proposal_result else []
    run=next((r for r in runs if r[0]<=seed<=r[1]),None)
    spikes=proposal_result.get('possible_tracking_spikes',[]) if proposal_result else []
    if proposal is None:reasons.append('proposal_evidence_missing')
    if run is None:reasons.append('continuous_context_missing')
    bounds=None
    if run is not None:
        lo=max(run[0],seed-6);hi=min(run[1],seed+6)
        boundaries=set(p['frame_id'] for p in proposals)|set(spikes)
        left=[b for b in boundaries if b<seed-2];right=[b for b in boundaries if b>seed+2]
        if left:lo=max(lo,max(left)+1)
        if right:hi=min(hi,min(right)-1)
        bounds=[lo,hi]
        if seed-lo<3 or hi-seed<3:reasons.append('insufficient_context_after_segmentation')
        if any(lo<=s<=hi for s in spikes):reasons.append('possible_tracking_spike')
    if proposal and proposal.get('tracking_spike_nearby') and 'possible_tracking_spike' not in reasons:
        reasons.append('possible_tracking_spike')
    neighbours=sorted({p['frame_id'] for p in baseline if p['video_id']==candidate['video_id'] and abs(p['frame_id']-seed)<=2})
    status='deferred_insufficient_evidence' if reasons else ('timing_alternative' if neighbours else 'ready_for_event_review')
    return {'status':status,'reasons':reasons,'nearby_baseline_frames':neighbours,'window_frame_bounds':bounds,
            'review_route':'table_model' if proposal and proposal.get('region')=='table_projection' else 'contact_type',
            'adopt':False,'confirmed_event_type':None}

def propose_events(rows,corners,config=EventConfig()):
    q=np.asarray(corners,float);scale=float(np.linalg.norm(np.ptp(q,axis=0)))
    if q.shape!=(4,2) or scale<=0 or not np.isfinite(q).all():raise ValueError('Invalid table polygon')
    candidates=[];spikes=[];runs=observed_runs(rows,config)
    for run in runs:
        for a,b,c in zip(run,run[1:],run[2:]):
            p0,p,p1=[np.array([r['x'],r['y']]) for r in (a,b,c)]
            weight=(b['timestamp_s']-a['timestamp_s'])/(c['timestamp_s']-a['timestamp_s'])
            residual=float(np.linalg.norm(p-((1-weight)*p0+weight*p1)))
            if residual>max(20,.08*scale) and np.linalg.norm(p1-p0)<.5*residual:spikes.append(b['frame_id'])
        h=config.half_window
        for i in range(h,len(run)-h):
            subset=run[i-h:i+h+1];centre=run[i];dt=np.array([r['timestamp_s']-centre['timestamp_s'] for r in subset]);unit=float(np.median(np.diff(dt)));t=dt/unit
            xy=np.array([[r['x'],r['y']] for r in subset])
            smooth=np.column_stack((np.ones(len(t)),t,t*t));kink=np.column_stack((np.ones(len(t)),t,np.maximum(t,0)))
            params=np.linalg.lstsq(kink,xy,rcond=None)[0]
            sse=float(np.sum((smooth@np.linalg.lstsq(smooth,xy,rcond=None)[0]-xy)**2));alternative=float(np.sum((kink@params-xy)**2))
            gain=(sse-alternative)/max(sse,1e-9);improvement=math.sqrt(sse/len(t))-math.sqrt(alternative/len(t));change=float(np.linalg.norm(params[2]))
            if gain<config.minimum_gain or improvement<config.minimum_rms_improvement_px or change<max(2,config.minimum_velocity_change_fraction*scale):continue
            fid=centre['frame_id'];distance=cv2.pointPolygonTest(q.astype(np.float32),(float(centre['x']),float(centre['y'])),True)
            # Orientation is expressed along projected table edges, not screen x/y.
            axes=np.array([q[1]-q[0],q[3]-q[0]]);axes/=np.linalg.norm(axes,axis=1)[:,None]
            candidates.append({'frame_id':fid,'score':float(sse-alternative),'smooth_to_kink_gain':gain,'rms_improvement_px':improvement,
                'velocity_change_px_per_nominal_interval':change,'velocity_before_px_s':(params[1]/unit).tolist(),'velocity_after_px_s':((params[1]+params[2])/unit).tolist(),
                'table_axis_image_components_before':(axes@params[1]).tolist(),'table_axis_image_components_after':(axes@(params[1]+params[2])).tolist(),
                'signed_table_polygon_distance_px':float(distance),'region':'table_projection' if distance>=-max(3,.01*scale) else 'outside_table_projection',
                'tracking_spike_nearby':any(abs(fid-s)<=h for s in spikes),'evaluated_at_frame':subset[-1]['frame_id']})
    chosen=[]
    for candidate in sorted(candidates,key=lambda r:r['score'],reverse=True):
        if all(abs(candidate['frame_id']-r['frame_id'])>config.merge_radius for r in chosen):chosen.append(candidate)
    return {'proposals':sorted(chosen,key=lambda r:r['frame_id']),'possible_tracking_spikes':sorted(set(spikes)),
            'observed_runs':[[r[0]['frame_id'],r[-1]['frame_id']] for r in runs]}

def select_window(rows,seed,proposal_frames,spikes,maximum_half_window=6):
    # Adjacent event evidence stops a fit before another event, never deletes interior points.
    runs=observed_runs(rows);run=next((r for r in runs if r[0]['frame_id']<=seed<=r[-1]['frame_id']),None)
    if run is None:return {'rows':[],'reasons':['seed_missing_or_low_confidence']}
    boundaries=sorted(set(proposal_frames)|set(spikes));lo=max(run[0]['frame_id'],seed-maximum_half_window);hi=min(run[-1]['frame_id'],seed+maximum_half_window)
    # Proposals within two frames describe the same event uncertainty, not a cut.
    left=[b for b in boundaries if b<seed-2];right=[b for b in boundaries if b>seed+2]
    if left:lo=max(lo,max(left)+1)
    if right:hi=min(hi,min(right)-1)
    selected=[r for r in run if lo<=r['frame_id']<=hi];reasons=[]
    if sum(r['frame_id']<seed for r in selected)<3 or sum(r['frame_id']>seed for r in selected)<3:reasons.append('insufficient_context_after_segmentation')
    if any(lo<=s<=hi for s in spikes):reasons.append('possible_tracking_spike')
    return {'rows':selected,'reasons':reasons,'window_frame_bounds':[lo,hi]}

def classify(proposal,window,fit,camera_supported=False):
    reasons=list(window['reasons'])
    if proposal.get('tracking_spike_nearby') or 'possible_tracking_spike' in reasons:
        return {'event':'tracking_anomaly_possible','reasons':reasons+['isolated_image_jump'],'adopt':False}
    if reasons or fit is None:return {'event':'uncertain','reasons':reasons+(['fit_unavailable'] if fit is None else []),'adopt':False}
    if proposal.get('region')=='outside_table_projection' and fit['rmse_px']>3:
        return {'event':'off_table_motion_change','reasons':['table_contact_model_not_supported','racket_contact_requires_racket_or_player_evidence'],'adopt':False}
    if fit['active_constraint_indices']:reasons.append('physical_boundary_solution')
    if fit['rmse_px']>3 or fit['max_error_px']>6:reasons.append('large_projection_residual')
    p=fit['parameters'];ratio=p[7]/(-p[4])
    if not .2<=ratio<=1.2:reasons.append('implausible_restitution')
    if max(np.linalg.norm(p[2:5]),np.linalg.norm(p[5:8]))>40:reasons.append('implausible_speed')
    if fit['search_boundary']:reasons.append('time_at_search_boundary')
    if fit['bic_gain_over_smooth'] is None or fit['bic_gain_over_smooth']<6:reasons.append('smooth_flight_not_excluded')
    if not camera_supported:reasons.append('camera_evidence_insufficient')
    return {'event':'table_contact_candidate' if not reasons else 'uncertain','reasons':reasons,
            'adopt':False,'note':'experimental proposal; requires window/camera sensitivity and independent validation'}

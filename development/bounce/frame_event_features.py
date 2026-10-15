"""Dense automatic frame features, interval targets, and event peak decoding."""
import cv2
import numpy as np

EVENT_TYPES=('table_bounce','racket_contact')
FEATURES=['context_present','ball_x_n','ball_y_n','ball_available','ball_confidence','dt_ratio','time_gap',
          'velocity_x_n','velocity_y_n','velocity_available','acceleration_x_n','acceleration_y_n','acceleration_available',
          'table_available']+[f'corner_{i}_{axis}_n' for i in range(4) for axis in ('x','y')]+[
          'table_scale_n','rectified_u','rectified_v','rectified_available','signed_projection_distance_fraction']


def automatic_frame_features(tracks,samples,image_size):
    width,height=image_size
    if width<=0 or height<=0 or not tracks:raise ValueError('Invalid automatic sequence dimensions')
    times=np.array([r['timestamp_s'] for r in tracks],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic automatic timestamps')
    if [r['frame_id'] for r in tracks]!=list(range(len(tracks))):raise ValueError('Noncontiguous decoded frame indices')
    nominal=float(np.median(np.diff(times))) if len(times)>1 else .04
    samples=sorted(samples,key=lambda r:r['frame_id']);at=0;geometry=None;features=[];previous=None;previous_velocity=None
    for fid,row in enumerate(tracks):
        while at<len(samples) and samples[at]['frame_id']<=fid:geometry=samples[at];at+=1
        confidence=row.get('confidence') or 0.;xy=[row.get('x'),row.get('y')]
        available=bool(row.get('detected') and row.get('in_image',True) and confidence>0 and all(v is not None and np.isfinite(v) for v in xy))
        if available and not (0<=xy[0]<width and 0<=xy[1]<height):available=False
        point=np.array([xy[0]/width,xy[1]/height]) if available else np.zeros(2)
        dt=times[fid]-times[fid-1] if fid else nominal;gap=dt>2*nominal
        velocity_available=bool(available and previous is not None and not gap)
        velocity=(point-previous)*.04/dt if velocity_available else np.zeros(2)
        acceleration_available=bool(velocity_available and previous_velocity is not None)
        acceleration=(velocity-previous_velocity)*.04/dt if acceleration_available else np.zeros(2)
        previous=point if available else None;previous_velocity=velocity if velocity_available else None
        table_available=False;corners=np.zeros((4,2));scale=0.;uv=np.zeros(2);rectified=False;distance=0.
        if geometry and geometry.get('usable') and fid<=geometry.get('valid_until',geometry['frame_id']):
            q=np.asarray(geometry.get('corners'),np.float32)
            if q.shape==(4,2) and np.isfinite(q).all() and cv2.isContourConvex(q) and abs(cv2.contourArea(q))>1:
                table_available=True;corners=q/[width,height];diagonal=float(np.linalg.norm(np.ptp(q,axis=0)))
                scale=diagonal/max(width,height)
                if available:
                    matrix=cv2.getPerspectiveTransform(q,np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
                    uv=cv2.perspectiveTransform(np.array([[xy]],np.float32),matrix)[0,0]
                    rectified=bool(np.isfinite(uv).all())
                    if not rectified:uv=np.zeros(2)
                    distance=float(cv2.pointPolygonTest(q,tuple(map(float,xy)),True))/max(diagonal,1)
        vector=[1.,*point,float(available),float(np.clip(confidence,0,1)) if available else 0.,np.clip(dt/nominal,0,5),float(gap),
                *np.clip(velocity,-2,2),float(velocity_available),*np.clip(acceleration,-2,2),float(acceleration_available),
                float(table_available),*corners.ravel(),scale,*np.clip(uv,-2,3),float(rectified),np.clip(distance,-1,1)]
        features.append(vector)
    result=np.asarray(features,np.float32)
    if result.shape!=(len(tracks),len(FEATURES)) or not np.isfinite(result).all():raise ValueError('Invalid dense frame features')
    return result


def interval_targets(n_frames,reviews,events,radius=2,sigma=.8):
    targets=np.zeros((n_frames,len(EVENT_TYPES)),np.float32);mask=np.zeros_like(targets,dtype=bool)
    # A visited frame in a completed window can supervise background only away
    # from its boundary; unseen context and ambiguous windows remain masked.
    for review in reviews:
        state=review['state']
        if state['status']!='completed' or state.get('candidate_uncertain'):continue
        lo,hi=state['review_scope'];visited=set(state.get('visited_frames',[]))
        for fid in range(max(0,lo+radius),min(n_frames,hi-radius+1)):
            if fid in visited:mask[fid,:]=True
    for event in events:
        if event['event_type'] not in EVENT_TYPES:continue
        channel=EVENT_TYPES.index(event['event_type']);lo,hi=event['interval_frames']
        if not 0<=lo<=hi<n_frames:raise ValueError('Event interval outside decoded video')
        for fid in range(max(0,lo-radius),min(n_frames,hi+radius+1)):
            distance=max(lo-fid,fid-hi,0);value=np.exp(-.5*(distance/sigma)**2)
            targets[fid,channel]=max(targets[fid,channel],value)
        # Explicit positive intervals remain usable even at a review boundary.
        mask[lo:hi+1,channel]=True
    # Occluded/ambiguous spans are unknown for both heads. Include the same
    # two-frame event shoulder so nearby labels cannot create negative targets
    # on an unobservable event; explicit labels elsewhere remain available.
    for review in reviews:
        for lo,hi in review['state'].get('unknown_intervals',[]):
            if not 0<=lo<=hi<n_frames:raise ValueError('Unknown interval outside decoded video')
            mask[max(0,lo-radius):min(n_frames,hi+radius+1),:]=False
    unknown=[v for r in reviews for v in r['state'].get('unknown_intervals',[])]
    for event in events:
        if event['event_type'] not in EVENT_TYPES:continue
        lo,hi=event['interval_frames']
        if not any(max(lo,a)<=min(hi,b) for a,b in unknown):
            mask[lo:hi+1,EVENT_TYPES.index(event['event_type'])]=True
    return targets,mask


def geometry_supported_touch_scores(scores,features,samples,policy='none'):
    """Optional image-plane support; unknown/invalid geometry never rejects.

    Reduce scores clearly outside a currently stable polygon. The model still
    evaluates every frame and the polygon does not certify physical height.
    """
    result=np.asarray(scores,np.float32).copy()
    if policy=='none':return result
    if policy!='stable_polygon_soft':raise ValueError('Unknown dense geometry support policy')
    x=np.asarray(features)
    if x.shape!=(len(result),len(FEATURES)):raise ValueError('Dense geometry feature shape mismatch')
    samples=sorted(samples,key=lambda r:r['frame_id']);at=0;geometry=None
    for fid in range(len(result)):
        while at<len(samples) and samples[at]['frame_id']<=fid:geometry=samples[at];at+=1
        if geometry and geometry.get('state')=='stable' and x[fid,FEATURES.index('table_available')]>0 and x[fid,FEATURES.index('ball_available')]>0:
            if x[fid,FEATURES.index('signed_projection_distance_fraction')]<-.02:result[fid]*=.25
    return result


def decode_frame_events(scores,tracks,threshold=.5,minimum_separation_s=.08,observation_radius=1,minimum_confidence=.75):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or len(scores)!=len(tracks) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Invalid dense event scores')
    if not 0<=threshold<=1 or minimum_separation_s<0 or observation_radius<0:raise ValueError('Invalid dense peak policy')
    maxima=[i for i,s in enumerate(scores) if s>=threshold and s>(scores[i-1] if i else -1) and s>=(scores[i+1] if i+1<len(scores) else -1)]
    chosen=[];points=[];diagnostics=[];seen=set()
    for seed in sorted(maxima,key=lambda i:(-scores[i],i)):
        if any(abs(tracks[seed]['timestamp_s']-tracks[j]['timestamp_s'])<minimum_separation_s-1e-9 for j in chosen):continue
        chosen.append(seed)
        candidates=[]
        for fid in range(max(0,seed-observation_radius),min(len(tracks),seed+observation_radius+1)):
            r=tracks[fid]
            if r.get('detected') and r.get('in_image',True) and (r.get('confidence') or 0)>=minimum_confidence and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y')):
                candidates.append(fid)
        if not candidates:
            diagnostics.append({'peak_frame':seed,'included':False,'reason':'no_actual_ball_point_within_one_frame','score':float(scores[seed])});continue
        fid=min(candidates,key=lambda j:(abs(j-seed),-scores[j],-(tracks[j].get('confidence') or 0),j))
        if fid in seen:continue
        seen.add(fid);r=tracks[fid];points.append({'frame_id':fid,'x':r['x'],'y':r['y']})
        diagnostics.append({'peak_frame':seed,'output_frame':fid,'included':True,'score':float(scores[seed]),
                            'coordinate_source':'actual_automatic_detection','observations_interpolated':False})
    return sorted(points,key=lambda p:p['frame_id']),sorted(diagnostics,key=lambda p:p['peak_frame'])

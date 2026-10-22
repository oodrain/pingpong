"""Conservative transient racket-box proximity; no manual overrides."""
import math
import numpy as np
from development.bounce.wrist_evidence import iou

def observations(ball,rackets):
    if not ball.get('detected') or ball.get('confidence',0)<.75:return []
    result=[];xy=np.array([ball['x'],ball['y']])
    for r in rackets:
        b=np.asarray(r['box']);diag=float(np.linalg.norm(b[2:]-b[:2]))
        if r['score']<.3 or diag<10 or not np.isfinite(b).all():continue
        distance=float(np.linalg.norm(np.maximum(np.maximum(b[:2]-xy,xy-b[2:]),0)))
        result.append(dict(r,distance_px=distance,distance_in_box_diagonals=distance/diag,near=distance/diag<=.25))
    return result

def transient_racket_visit(frames,seed):
    rows={r['frame_id']:r['observations'] for r in frames};visits=[]
    for fid in range(seed-1,seed+2):
        for anchor in rows.get(fid,[]):
            if not anchor['near']:continue
            neighbours=[]
            for other in (fid-1,fid+1):
                options=[r for r in rows.get(other,[]) if iou(anchor['box'],r['box'])>=.3]
                neighbours.append(max(options,key=lambda r:iou(anchor['box'],r['box'])) if options else None)
            if any(r is None for r in neighbours):continue
            distances=[neighbours[0]['distance_px'],anchor['distance_px'],neighbours[1]['distance_px']]
            if min(distances[0]-distances[1],distances[2]-distances[1])>=2:
                visits.append({'closest_frame':fid,'box':anchor['box'],'distance_px_before_at_after':distances})
    return {'supported':bool(visits),'visits':visits,'racket_contact_confirmed':False,
            'limitations':['Detector may confuse hands or blur with a racket.','Boxes include handle and background.']}

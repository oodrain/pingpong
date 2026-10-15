"""Offline rule prototypes on cached detections; no detector or reference ball-label input.

Raw input IDs are compensated exactly once. All emitted x/y come from model
observations. The table-box experiment is separately named oracle and is not
a deployment-ready method. These ablations are diagnostic, not tuned final rules.
"""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class Config:
    name: str
    conf: float = .75
    min_gap: int = 10
    fall: float = 2.0
    rise: float = 1.0
    max_neighbor_gap: int | None = None
    prominence_window: int = 0
    normalize_pixels: bool = False
    spatial_nms: bool = False
    remove_isolated_spikes: bool = False
    mode: str = 'y_max'
    oracle_table: bool = False
    oracle_reference_ball: bool = False
    adaptive_gap: bool = False

CONFIGS = [
    Config('V0_exact'),
    Config('A_gap6',min_gap=6),
    Config('A_gap3',min_gap=3),
    Config('A_conf070',conf=.70),
    Config('A_amp1_05',fall=1,rise=.5),
    Config('A_gapaware2',max_neighbor_gap=2),
    Config('A_normalize720',normalize_pixels=True),
    Config('A_prominence3',prominence_window=3),
    Config('V1_prom3_gapaware2',prominence_window=3,max_neighbor_gap=2),
    Config('V1_prom3_gapaware3',prominence_window=3,max_neighbor_gap=3),
    Config('V1_prom3_gap6_gapaware2',prominence_window=3,min_gap=6,max_neighbor_gap=2),
    Config('V1_prom3_adaptive_gap',prominence_window=3,max_neighbor_gap=2,adaptive_gap=True),
    Config('A_spatial_nms3',min_gap=3,spatial_nms=True),
    Config('A_remove_spikes',remove_isolated_spikes=True),
    Config('V1_gap3_prom3',min_gap=3,max_neighbor_gap=2,prominence_window=3,spatial_nms=True),
    Config('V1_conf070_gap3_prom3',conf=.70,min_gap=3,max_neighbor_gap=2,prominence_window=3,spatial_nms=True),
    Config('A_y_both',mode='y_both'),
    Config('A_turn35_gap3',min_gap=3,max_neighbor_gap=2,mode='turn35',spatial_nms=True),
    Config('ORACLE_V0_tablebox',oracle_table=True),
    Config('ORACLE_V1_prom3_gapaware2_tablebox',prominence_window=3,max_neighbor_gap=2,oracle_table=True),
    Config('ORACLE_V1_tablebox',min_gap=3,max_neighbor_gap=2,prominence_window=3,spatial_nms=True,oracle_table=True),
    Config('ORACLE_turn35_tablebox',min_gap=3,max_neighbor_gap=2,mode='turn35',spatial_nms=True,oracle_table=True),
    Config('ORACLE_V0_reference_ball',oracle_reference_ball=True),
]

def angle(a,b,c):
    u=(b['x']-a['x'],b['y']-a['y'])
    v=(c['x']-b['x'],c['y']-b['y'])
    nu,nv=math.hypot(*u),math.hypot(*v)
    if min(nu,nv)<1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/(nu*nv)))))

def in_table(point,table,margin=15):
    if not table:
        return False
    x1,y1,x2,y2=table
    return x1-margin<=point['x']<=x2+margin and y1-margin<=point['y']<=y2+margin

def accepted_points(raw,cfg,meta):
    points=[dict(r,frame_id=r['input_frame_id']-1) for r in raw if r['detected'] and r['confidence']>=cfg.conf]
    if cfg.remove_isolated_spikes:
        diag=math.hypot(meta['width'],meta['height'])
        bad=set()
        for j in range(1,len(points)-1):
            a,b,c=points[j-1:j+2]
            if b['frame_id']-a['frame_id']!=1 or c['frame_id']-b['frame_id']!=1:
                continue
            if math.dist((a['x'],a['y']),(c['x'],c['y']))<.025*diag and min(math.dist((b['x'],b['y']),(a['x'],a['y'])),math.dist((b['x'],b['y']),(c['x'],c['y'])))>.05*diag:
                bad.add(j)
        points=[p for j,p in enumerate(points) if j not in bad]
    return points

def generate_candidates(raw,meta,cfg,tables=None):
    points=accepted_points(raw,cfg,meta)
    out=[]
    scale=720/meta['height'] if cfg.normalize_pixels else 1
    for j in range(1,len(points)-1):
        a,b,c=points[j-1:j+2]
        dg0=b['frame_id']-a['frame_id']
        dg1=c['frame_id']-b['frame_id']
        if cfg.max_neighbor_gap is not None and max(dg0,dg1)>cfg.max_neighbor_gap:
            continue
        signs=(1,-1) if cfg.mode=='y_both' else (1,)
        gate=False
        strength=0
        for sign in signs:
            y=b['y']*scale*sign
            before=[a['y']*scale*sign]
            after=[c['y']*scale*sign]
            peak=y>before[0] and y>=after[0]
            if cfg.prominence_window:
                before=[p['y']*scale*sign for p in points[max(0,j-cfg.prominence_window):j] if 0<b['frame_id']-p['frame_id']<=cfg.prominence_window]
                after=[p['y']*scale*sign for p in points[j+1:j+1+cfg.prominence_window] if 0<p['frame_id']-b['frame_id']<=cfg.prominence_window]
            if before and after:
                fall=y-min(before)
                rise=y-min(after)
                if peak and fall>=cfg.fall and rise>=cfg.rise:
                    gate=True
                    strength=max(strength,min(fall,rise))
        turning=angle(a,b,c)
        if cfg.mode=='turn35':
            v0=math.dist((a['x'],a['y']),(b['x'],b['y']))/dg0
            v1=math.dist((b['x'],b['y']),(c['x'],c['y']))/dg1
            gate=turning>=35 and min(v0,v1)>=1.5
            strength=turning/10
        if not gate:
            continue
        fid=max(0,b['frame_id'])
        if cfg.oracle_table and not in_table(b,tables.get(fid) if tables else None):
            continue
        out.append({'frame_id':fid,'x':b['x'],'y':b['y'],'confidence':b['confidence'],
                    'prev_frame':a['frame_id'],'next_frame':c['frame_id'],
                    'gap_before':dg0,'gap_after':dg1,'turn_angle_deg':turning,
                    'local_fall':(b['y']-a['y'])*scale,'local_rise':(b['y']-c['y'])*scale,
                    'strength':strength})
    return out

def predict(raw,meta,cfg,tables=None):
    candidates=generate_candidates(raw,meta,cfg,tables)
    if cfg.adaptive_gap:
        selected=[]
        diag=math.hypot(meta['width'],meta['height'])
        for p in candidates:
            if selected:
                dt=p['frame_id']-selected[-1]['frame_id']
                d=math.dist((p['x'],p['y']),(selected[-1]['x'],selected[-1]['y']))
                short_exception=dt>=3 and d>=.04*diag and p['confidence']>=.85 and p['strength']>=3
                if dt<cfg.min_gap and not short_exception:
                    continue
            selected.append(p)
    elif cfg.spatial_nms:
        # Greedy chronological duplicate suppression; retain a close-in-time event
        # if it has moved more than 50 original-image pixels from the previous one.
        selected=[]
        for p in candidates:
            if selected and p['frame_id']-selected[-1]['frame_id']<cfg.min_gap and math.dist((p['x'],p['y']),(selected[-1]['x'],selected[-1]['y']))<=50:
                continue
            selected.append(p)
    else:
        selected=[]
        last=-10000
        for p in candidates:
            if p['frame_id']-last>=cfg.min_gap:
                selected.append(p)
                last=p['frame_id']
    return [{k:p[k] for k in ('frame_id','x','y')} for p in selected],candidates

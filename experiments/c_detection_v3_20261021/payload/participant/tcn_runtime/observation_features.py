"""Actual PTS and observed support only. No detector or annotations imported."""
import numpy as np

NAMES=['nominal_period_over_40ms','past_actual_gap_over_200ms','future_actual_gap_over_200ms',
       'past_actual_support_200ms','future_actual_support_200ms']
ACTIVE={'calibrated_event':(), 'fps_context':(0,), 'gap_context':(1,2),
        'support_context':(3,4),'all_context':tuple(range(5))}

def good(row):
    return bool(row.get('detected') and row.get('in_image',True) and (row.get('confidence') or 0)>=.75
        and all(row.get(k) is not None and np.isfinite(row[k]) for k in ('x','y')))

def context(tracks):
    t=np.asarray([r['timestamp_s'] for r in tracks],float)
    if not len(t) or not np.isfinite(t).all() or not np.all(np.diff(t)>0):raise ValueError('Invalid actual PTS')
    nominal=float(np.median(np.diff(t))) if len(t)>1 else .04;present=np.array([good(r) for r in tracks]);denom=max(round(.2/nominal),1);result=[]
    for i in range(len(tracks)):
        prev=np.flatnonzero(present&(t<t[i]));nxt=np.flatnonzero(present&(t>t[i]))
        pg=min((t[i]-t[prev[-1]])/.2,5.) if len(prev) else 5.;ng=min((t[nxt[0]]-t[i])/.2,5.) if len(nxt) else 5.
        left=(t<t[i])&(t>=t[i]-.2-1e-9);right=(t>t[i])&(t<=t[i]+.2+1e-9)
        result.append([np.clip(nominal/.04,.25,4),pg,ng,min(float((present&left).sum())/denom,1),min(float((present&right).sum())/denom,1)])
    z=np.asarray(result,np.float32);assert z.shape==(len(tracks),5) and np.isfinite(z).all();return z

def augment(base,tracks,variant):
    if variant not in ACTIVE:raise ValueError('Unknown ablation')
    x=np.asarray(base,np.float32);assert x.shape==(len(tracks),51) and np.isfinite(x).all()
    if variant=='calibrated_event':return x.copy()
    c=context(tracks);masked=np.zeros_like(c);masked[:,ACTIVE[variant]]=c[:,ACTIVE[variant]]
    return np.c_[x,masked].astype(np.float32)

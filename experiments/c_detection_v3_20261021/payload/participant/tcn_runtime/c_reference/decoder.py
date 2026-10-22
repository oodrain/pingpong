"""Frozen admission uses the original score, before any auxiliary review."""
import numpy as np
from .frame_features import decode_frame_events

def anchor_admitted(original_probability,fps_probability):
    return bool(original_probability>=.5 and (original_probability<.65 or fps_probability>=.65))

def original_control(original,fps,tracks):
    pts,details=decode_frame_events(np.asarray(original)[:,0],tracks,.5,.12);by={p['frame_id']:p for p in pts};kept=[];audit=[]
    for d in details:
        if not d['included']:continue
        peak=d['peak_frame'];support=list(range(max(0,peak-1),min(len(tracks),peak+2)));accept=anchor_admitted(float(original[peak,0]),float(fps[support,0].max()))
        audit.append(dict(peak_frame=peak,output_frame=d['output_frame'],included=accept,original_probability=float(original[peak,0]),fps_probability=float(fps[support,0].max()),admission_frozen_before_review=True))
        if accept:kept.append(by[d['output_frame']])
    return sorted(kept,key=lambda p:p['frame_id']),audit

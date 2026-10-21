"""Automatic time-supported missing-point selection, no label inputs."""
import math
POLICY=dict(schema=1,support_window_s=.12,require_real_triplet=True,retain_original_trusted=True,
    original_global_sparse_branch_preserved=True,automatic_ROI='unchanged min.6 margin.1',
    selection='nontrusted interior frame within original .12s window of an original trusted point',
    original_merge_unchanged=True,manual_frame_ids=False,manual_boxes=False,parameter_search=False)

def good(row):
    return bool(row.get('detected') and row.get('in_image',True) and (row.get('confidence') or 0)>=.75
        and all(row.get(k) is not None and math.isfinite(row[k]) for k in ('x','y')))

def local_targets(tracks):
    n=len(tracks);times=[r['timestamp_s'] for r in tracks]
    assert all(math.isfinite(t) for t in times) and all(a<b for a,b in zip(times,times[1:]))
    trusted=[i for i,r in enumerate(tracks) if good(r)]
    support={i:[j for j in trusted if 0<abs(times[j]-times[i])<=.12+1e-9]
        for i in range(1,n-1) if not good(tracks[i])}
    support={i:ids for i,ids in support.items() if ids}
    return sorted(support),support

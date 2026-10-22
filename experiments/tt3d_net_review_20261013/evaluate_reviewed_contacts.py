"""Evaluate constrained fits on completed development reviews, without fitting labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_contact_pilot import search,sha

def interval_error(value,interval):
    return max(interval[0]-value,0,value-interval[1])

def run(source,inputs,out):
    data=json.loads(source.read_text(encoding='utf-8'));journal=source.parent/'human_contact_reviews.jsonl'
    snapshot=journal.read_bytes();reviews={}
    for line in snapshot.decode('utf-8').splitlines():
        r=json.loads(line)
        if r['results_sha256']==sha(source):reviews[r['case_id']]=r
    if set(reviews)!=set(c['case_id'] for c in data['cases']):raise ValueError('Incomplete or mismatched reviews')
    out.mkdir(parents=True,exist_ok=False);(out/'human_reviews_snapshot.jsonl').write_bytes(snapshot)
    results=[]
    for c in data['cases']:
        folder=inputs/'videos'/c['video_id'];meta=json.loads((folder/'input_summary.json').read_text())
        for name,expected in meta['files'].items():
            if sha(folder/name)!=expected:raise ValueError('Input changed: '+name)
        tracks=[json.loads(x) for x in (folder/'timed_tracks.jsonl').read_text().splitlines()]
        camera=json.loads((folder/'camera.json').read_text());ts=np.array([r['timestamp_s'] for r in tracks]);fid=c['seed_frame_id']
        lo=float(np.interp(fid-2.5,np.arange(len(ts)),ts));hi=float(np.interp(fid+2.5,np.arange(len(ts)),ts))
        variants=[]
        # Same fixed windows and search interval for every case, including noncontacts.
        # Human labels are read only by the evaluation code below, not the optimizer.
        for width in (4,6,8):
            rows=[r for r in tracks if abs(r['frame_id']-fid)<=width and r['eligible_high_confidence']]
            fit=search(rows,c['camera'],lo,hi,constrained=True)
            if fit:
                fit['fractional_frame']=float(np.interp(fit['tau'],ts,np.arange(len(ts))))
                fit['fit_frame_ids']=[r['frame_id'] for r in rows]
            variants.append({'half_window':width,'fit':fit})
        primary=variants[1]['fit'];reasons=[];sensitivity=[]
        same_rows=[r for r in tracks if abs(r['frame_id']-fid)<=6 and r['eligible_high_confidence']]
        unconstrained_same_window=search(same_rows,c['camera'],lo,hi)
        if unconstrained_same_window:unconstrained_same_window['fractional_frame']=float(np.interp(unconstrained_same_window['tau'],ts,np.arange(len(ts))))
        if c['camera_stability']['status']!='stable':reasons.append('camera_'+c['camera_stability']['status'])
        if primary:
            rows=[r for r in tracks if abs(r['frame_id']-fid)<=6 and r['eligible_high_confidence']]
            for pose in camera['poses']:
                alt=search(rows,pose,lo,hi,constrained=True)
                if alt:sensitivity.append({'camera_frame':pose['frame_id'],'fractional_frame':float(np.interp(alt['tau'],ts,np.arange(len(ts)))),'rmse_px':alt['rmse_px']})
            p=primary['parameters'];ratio=p[7]/(-p[4]);primary['vertical_restitution']=ratio
            if primary['active_constraint_indices']:reasons.append('optimum_on_physical_boundary')
            if primary['rmse_px']>3 or primary['max_error_px']>6:reasons.append('large_projection_residual')
            if not .2<=ratio<=1.2:reasons.append('implausible_vertical_restitution')
            if max(np.linalg.norm(p[2:5]),np.linalg.norm(p[5:8]))>40:reasons.append('implausible_speed')
            if primary['search_boundary']:reasons.append('time_at_search_boundary')
            if primary['bic_gain_over_smooth'] is None or primary['bic_gain_over_smooth']<6:reasons.append('smooth_path_not_excluded')
            if np.ptp(primary['one_pixel_cost_band_s'])/np.median(np.diff(ts))>.75:reasons.append('broad_time_cost_band')
            values=[v['fit']['fractional_frame'] for v in variants if v['fit']]+[v['fractional_frame'] for v in sensitivity]
            primary['sensitivity_spread_frames']=float(np.ptp(values))
            if np.ptp(values)>.5:reasons.append('camera_or_window_sensitive')
            if any(v['fit'] is None for v in variants):reasons.append('short_window_insufficient')
        else:reasons.append('fit_unavailable')
        review=reviews[c['case_id']];interval=review['manual_interval_frames'];evaluation=None
        if interval is not None:
            evaluation={'human_interval':interval,'old_seed_interval_error_frames':interval_error(fid,interval),
                        'unconstrained_interval_error_frames':interval_error(c['fit']['fractional_frame'],interval),
                        'unconstrained_same_window_interval_error_frames':interval_error(unconstrained_same_window['fractional_frame'],interval) if unconstrained_same_window else None,
                        'constrained_interval_error_frames':interval_error(primary['fractional_frame'],interval) if primary else None}
        result={'case_id':c['case_id'],'human_review':review,'old_fit_frame':c['fit']['fractional_frame'],
                'old_fit_rmse_px':c['fit']['rmse_px'],'primary_half_window':6,'unconstrained_same_window':unconstrained_same_window,'variants':variants,'camera_sensitivity':sensitivity,
                'reasons':reasons,'automatic_adoption_allowed':not reasons,'evaluation':evaluation}
        results.append(result)
        print(c['case_id'],review['decision'],'new',round(primary['fractional_frame'],3) if primary else None,
              'rmse',round(primary['rmse_px'],2) if primary else None,'reasons',reasons,flush=True)
    evaluated=[r['evaluation'] for r in results if r['evaluation'] is not None]
    metrics={}
    for key in ('old_seed','unconstrained','unconstrained_same_window','constrained'):
        values=[r[key+'_interval_error_frames'] for r in evaluated if r[key+'_interval_error_frames'] is not None]
        metrics[key]={'available':len(values),'inside_closed_human_interval':sum(v<1e-9 for v in values),'mean_distance_to_interval_frames':float(np.mean(values)) if values else None}
    report={'schema_version':1,'development_set_only':True,'not_independent_test':True,'human_labels_used_by_optimizer':False,
            'selection':'primary half-window 6 for all cases; 4 and 8 are sensitivity diagnostics, never chosen using labels',
            'constraints':'contact at ball-centre height .02 m; contact inside table; downward then upward world Z; no table penetration within fitted window',
            'limits':'gravity only, no spin/drag; inherited unresolved camera checks; labels identify frame intervals, not precise subframe ground truth',
            'sources':{str(source):sha(source),str(journal):sha(journal),'optimizer_sha256':sha(Path(__file__).with_name('fit_contact_pilot.py')),'runner_sha256':sha(__file__)},
            'metrics':metrics,'cases':results}
    if journal.read_bytes()!=snapshot:raise ValueError('Reviews changed during evaluation')
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.inputs,a.output)

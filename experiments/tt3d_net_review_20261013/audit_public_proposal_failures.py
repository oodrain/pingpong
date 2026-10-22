"""Post-score failure attribution. Never changes frozen proposals or thresholds."""
import argparse
import collections
import json
from pathlib import Path
from fit_contact_pilot import sha

def key(gt):return (gt['video_id'],gt['frame_id'],gt['x'],gt['y'])

def run(folder):
    inference=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
    scores=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))['scores']
    old={key(m['gt']):m for v in scores['baseline']['without_ignore_exemption']['by_video'] for m in v['matches']}
    new={key(m['gt']):m for v in scores['vector']['without_ignore_exemption']['by_video'] for m in v['matches']}
    diagnostic={v['video_id']:v for v in inference['videos']};lost=[]
    for identity in sorted(old.keys()-new.keys()):
        gt=old[identity]['gt'];v=diagnostic[gt['video_id']];p=v['proposal_result'];fid=gt['frame_id']
        if p is None:reason='no_usable_geometry'
        elif not any(start<=candidate-3 and candidate+3<=end for start,end in p['observed_runs'] for candidate in range(max(0,fid-1),fid+2)):
            reason='no_seven_consecutive_eligible_points_near_event'
        elif any(abs(r['frame_id']-fid)<=1 for r in p['proposals']):reason='candidate_position_or_one_to_one_match_failure'
        else:reason='change_score_or_temporal_selection_failed'
        lost.append({'gt':gt,'angle':v['angle'],'reason':reason,'old_prediction':old[identity]['prediction']})
    report={'source_inference_sha256':sha(folder/'inference.json'),'source_evaluation_sha256':sha(folder/'evaluation.json'),
            'script_sha256':sha(__file__),'thresholds_retuned':False,'lost_matches':lost,
            'gained_matches':[new[k] for k in sorted(new.keys()-old.keys())],
            'lost_reason_counts':dict(collections.Counter(r['reason'] for r in lost)),
            'same_matched_gt':len(old.keys()&new.keys()),'lost_count':len(old.keys()-new.keys()),'gained_count':len(new.keys()-old.keys())}
    target=folder/'failure_audit.json'
    if target.exists():raise ValueError('Audit already exists')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report['lost_reason_counts'],'lost',report['lost_count'],'gained',report['gained_count'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)

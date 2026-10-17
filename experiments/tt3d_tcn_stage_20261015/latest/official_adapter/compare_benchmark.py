"""Compare event identity and subpixel coordinates; never read event GT."""
import argparse
import json
import math
from pathlib import Path


def prediction_file(path):
    p=Path(path)
    if p.is_file():return p
    if (p/'predictions.jsonl').is_file():return p/'predictions.jsonl'
    if (p/'run_1/predictions.jsonl').is_file():return p/'run_1/predictions.jsonl'
    raise ValueError('No predictions.jsonl found: '+str(p))


def compare(reference,candidate,tolerance=.001):
    def rows(path):
        values=[json.loads(s) for s in prediction_file(path).read_text(encoding='utf-8').splitlines() if s]
        return sorted(values,key=lambda r:(r['video_id'],r['frame_id'],r['x'],r['y']))
    a,b=rows(reference),rows(candidate)
    identities=[(r['video_id'],r['frame_id']) for r in a]==[(r['video_id'],r['frame_id']) for r in b]
    maximum=max((math.hypot(x['x']-y['x'],x['y']-y['y']) for x,y in zip(a,b)),default=0.) if identities else None
    passed=bool(identities and maximum<=tolerance)
    return dict(passed=passed,reference_events=len(a),candidate_events=len(b),
                event_video_and_frame_ids_exact=identities,maximum_coordinate_error_px=maximum,
                allowed_coordinate_error_px=tolerance,event_GT_used=False,
                comparison_is_not_F1_or_accuracy_evaluation=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=compare(a.reference,a.candidate)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(2)

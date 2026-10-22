"""Replay frozen automatic features; explicitly not an end-to-end FPS test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import HERE,read,sha,sequences,write
from development.bounce.frame_event_features import decode_frame_events
from development.bounce.frame_event_tcn import load,predict

def replay(output=None):
    stage=read(HERE/'stage.json');asset=HERE/'assets/model.pt'
    if sha(asset)!=stage['model_sha256']:raise ValueError('Frozen model hash mismatch')
    torch.set_num_threads(4);model,checkpoint=load(asset);manifest,rows=sequences('eval')
    maximum=0.;touch_count=hit_count=frames=0;records=[]
    for row in rows:
        scores=predict(model,row['features'],checkpoint['normalization'])
        error=float(np.max(np.abs(scores-row['expected_scores'])));maximum=max(maximum,error)
        if error>1e-5:raise ValueError('Score reproduction failed: '+row['video_id'])
        output_heads=[]
        for channel in (0,1):
            points,_=decode_frame_events(scores[:,channel],row['tracks'],checkpoint['threshold'],checkpoint['minimum_separation_s'])
            expected=row['expected_table_events'] if channel==0 else row['expected_racket_events']
            if points!=expected:raise ValueError('Event reproduction failed: '+row['video_id'])
            output_heads.append(points)
        touch_count+=len(output_heads[0]);hit_count+=len(output_heads[1]);frames+=len(scores)
        if output:
            path=output/row['video_id'];path.mkdir(parents=True,exist_ok=True)
            for name,points in zip(('landings.jsonl','hit_events.jsonl'),output_heads):
                (path/name).write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in points),encoding='utf-8')
        records.append({'video_id':row['video_id'],'frames':len(scores),'maximum_score_difference':error})
    report={'videos':len(rows),'frames':frames,'maximum_score_difference':maximum,'touch_outputs':touch_count,
        'hit_outputs':hit_count,'all_frozen_events_identical':True,'detector_inference_calls':0,
        'end_to_end_fps_benchmark':False,'ground_truth_scoring_performed':False,'records':records}
    if output:write(output/'replay_verification.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=replay(args.output);print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

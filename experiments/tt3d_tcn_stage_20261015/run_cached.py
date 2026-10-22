"""Predict from your own automatic track/geometry JSON; no human labels read."""
import argparse
import json
from pathlib import Path
import torch
from common import HERE,read,sha,write
from development.bounce.frame_event_features import automatic_frame_features,decode_frame_events
from development.bounce.frame_event_tcn import load,predict

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks',type=Path,required=True);parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--width',type=int,required=True);parser.add_argument('--height',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=[json.loads(s) for s in args.tracks.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    text=args.geometry.read_text(encoding='utf-8-sig').strip()
    geometry=json.loads(text) if text.startswith('[') else [json.loads(s) for s in text.splitlines() if s.strip()]
    asset=HERE/'assets/model.pt'
    if sha(asset)!=read(HERE/'stage.json')['model_sha256']:raise ValueError('Frozen model hash mismatch')
    torch.set_num_threads(4);model,checkpoint=load(asset)
    scores=predict(model,automatic_frame_features(rows,geometry,(args.width,args.height)),checkpoint['normalization'])
    args.output.mkdir(parents=True,exist_ok=False)
    for channel,name in [(0,'landings.jsonl'),(1,'hit_events.jsonl')]:
        points,_=decode_frame_events(scores[:,channel],rows,checkpoint['threshold'],checkpoint['minimum_separation_s'])
        (args.output/name).write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in points),encoding='utf-8')
    write(args.output/'run.json',{'frames':len(rows),'model_sha256':sha(asset),'human_labels_used_at_runtime':False,
        'raw_video_inference':False,'detector_inference_calls':0,'fps_benchmark':False})
    print(args.output)

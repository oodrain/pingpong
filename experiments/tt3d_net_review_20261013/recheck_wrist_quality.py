"""Recheck cached poses; preserves every old inference and human annotation."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import quality_checked_observations,summarize
from fit_contact_pilot import sha

def run(original,inputs,out):
    prior=json.loads(original.read_text(encoding='utf-8'));out.mkdir(parents=True,exist_ok=False);cases=[];sources={str(original):sha(original)}
    for c in prior['cases']:
        posefile=original.parent/(c['video_id']+'_poses.json');poses=json.loads(posefile.read_text())['frames'];sources[str(posefile)]=sha(posefile)
        folder=inputs/'videos'/c['video_id'];path=folder/'timed_tracks.jsonl';meta=json.loads((folder/'input_summary.json').read_text())
        if sha(path)!=meta['files']['timed_tracks.jsonl']:raise ValueError('Changed tracks')
        rows={r['frame_id']:r for r in [json.loads(s) for s in path.read_text().splitlines()]}
        frames=[{'frame_id':f['frame_id'],'observations':quality_checked_observations(rows[f['frame_id']],poses[str(f['frame_id'])])} for f in c['frames']]
        cases.append({'video_id':c['video_id'],'seed_frame':c['seed_frame'],'previous_status':c['status'],'frames':frames,**summarize(frames)})
    result={'source_events_sha256':prior['source_events_sha256'],'model':prior['model'],'weights_sha256':prior['weights_sha256'],
            'cached_pose_sources':sources,'rule_sha256':sha(Path(__file__).resolve().parents[2]/'development/bounce/wrist_evidence.py'),
            'script_sha256':sha(__file__),'new_model_inference':False,'human_event_labels_used':False,'production_events_modified':False,'cases':cases}
    (out/'wrist_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(c['video_id'],c['seed_frame'],c['previous_status'],c['status']) for c in cases if c['previous_status']!=c['status']],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.original,a.inputs,a.output)

"""Required actual-video C/C2 ball parity before cloud benchmark; no event labels."""
import argparse,importlib,json,os,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-dir',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--decoder',choices=['cpu','gpu'],default='gpu');a=p.parse_args()
 task=a.task_dir.resolve();data=(a.input or task/'public_data').resolve();sys.path.insert(0,str(task));os.environ['TCN_DEVICE']='cuda'
 from core.types import RunContext,VideoMeta
 from core.decoder import decode_video
 from participant.solution import Solution
 from participant.tcn_runtime.batch_ball import BatchBall
 from participant.tcn_runtime.c_reference.batch_ball import BatchBall as OriginalBatch
 context=RunContext(input_dir=str(data),output_dir=str(task.parent/'outputs'),decoder=a.decoder)
 solution=Solution();solution.prepare(context)
 videos=json.loads((data/'manifest.json').read_text(encoding='utf-8'))['videos'];chosen={}
 for v in videos:chosen.setdefault(str(v['angle']),v)
 rows=[]
 for angle,v in sorted(chosen.items()):
  meta=VideoMeta.from_entry(v);size=(v['width'],v['height']);old=OriginalBatch(solution.ball,size);new=BatchBall(solution.ball,size);aa=[];bb=[]
  for f,rgb in decode_video(meta,data,a.decoder):aa.extend(old.push(rgb));bb.extend(new.push(rgb))
  aa.extend(old.flush());bb.extend(new.flush());assert len(aa)==len(bb)==v['n_frames']
  error=0.;confidence=0.;bad=[]
  for x,y in zip(aa,bb):
   assert x['target_frame_id']==y['target_frame_id']
   q,r=x['detection'],y['detection']
   if (q is None)!=(r is None):bad.append(x['target_frame_id']);continue
   if q is not None:
    error=max(error,float(np.hypot(q['x']-r['x'],q['y']-r['y'])));confidence=max(confidence,abs(q['confidence']-r['confidence']))
    if (q['confidence']>=.75)!=(r['confidence']>=.75):bad.append(x['target_frame_id'])
  row=dict(video_id=v['video_id'],angle=angle,frames=len(aa),detection_or_admission_mismatches=bad,max_coordinate_error_px=error,max_confidence_error=confidence)
  rows.append(row);print(row,flush=True)
  if bad or error>.001:raise RuntimeError('C2 GPU ball parity failed: stop benchmark and retain results')
 receipt=dict(passed=True,rows=rows,event_GT_used=False,middle_graph_CUDA_provider=solution.ball.middle.session.get_providers()[0],actual_decoder=a.decoder)
 dest=task.parent/'outputs/c2_gpu_ball_check.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 print('C2 GPU BALL PARITY PASSED: '+str(dest))
if __name__=='__main__':main()

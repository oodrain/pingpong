"""C2: output-equivalent geometry/ball engineering, optional accepted quick geometry."""
import json,time
import numpy as np
from .base_solution import BaseSolution
from .batch_ball import BatchBall
from .middle_ball import MiddleBall
from .quick_geometry import QuickGeometry
from .pipeline import GeometryWorker
from .geometry_engine import GeometryEngine
from .auto_table_geometry import extract_quad
from .contact_review import review

class Solution(BaseSolution):
 middle_model_name = "ball_middle.onnx"
 def prepare(self,context):
  super().prepare(context)
  self.engineering=json.loads((self.root/'configs/engineering_C2.json').read_text(encoding='utf-8'))
  if self.engineering['schema']!='C2_engineering_v1' or self.engineering['batch_size']!=8:raise ValueError('Unknown C2 recipe')
  self.ball.middle=MiddleBall(self.ball,self.root/'weights'/self.middle_model_name,self.device)
  check=self.ball.middle.warmup_check()
  context.info.update(engineering_variant='C2_quick' if self.engineering['geometry_quick'] else 'C2_equivalent',
   c_pipeline_timings=[],middle_graph_check=check,ball_batch_size=8)
 def reset(self,meta):
  previous=getattr(self,'worker',None)
  if previous is not None and not previous.closed:previous.abort()
  super().reset(meta);self.quick=QuickGeometry()
  def provider(frame,image,evidence):
   start=time.perf_counter();probability=self.table.probability(image);duration=time.perf_counter()-start
   proposal=None
   if self.engineering['geometry_quick']:
    proposal=self.quick.propose(image,probability,evidence,self.engine.memory.good,float(self.times[frame]))
   if proposal is None:proposal=extract_quad(image,probability,self.cfg,frame_evidence=evidence)
   return proposal,duration
  self.engine=GeometryEngine(self.times,self.size,provider,'safe');self.worker=GeometryWorker(self.engine)
  self.batched=BatchBall(self.ball,self.size);self.pipeline_wait_seconds=0.
 def process_frame(self,rgb,meta):
  if self.finished or meta['frame_id']!=self.fid or meta['video_id']!=self.meta.video_id:raise ValueError('Frame sequence mismatch')
  if rgb.shape!=(self.size[1],self.size[0],3) or rgb.dtype!=np.uint8:raise ValueError('Expected official RGB')
  start=time.perf_counter();self.worker.submit(self.fid,rgb);self.pipeline_wait_seconds+=time.perf_counter()-start
  start=time.perf_counter();self.detections.extend(self.batched.push(rgb));self.timing['ball_sec']+=time.perf_counter()-start
  self.fid+=1;return []
 def review_points(self,points,tracks):
  points,self.geometry_review_audit=review(points,tracks,self.review_samples,self.size,self.geometry_cuts);return points
 def finish_video(self,meta):
  if self.finished:return []
  start=time.perf_counter();self.detections.extend(self.batched.flush());self.timing['ball_sec']+=time.perf_counter()-start
  start=time.perf_counter();g=self.worker.close();join=time.perf_counter()-start
  self.samples=g['samples'];self.review_samples=g['review_samples'];self.geometry_cuts=g['cuts'];self.profile=self.engine.profile
  self.timing['table_model_sec']=g['table_model_sec'];self.timing['table_geometry_sec']=g['table_geometry_sec']
  result=super().finish_video(meta)
  self.context.info['c_pipeline_timings'].append(dict(video_id=meta.video_id,full_geometry_calls=g['full_calls'],
   maximum_queue_frames=g['maximum_queue_frames'],enqueue_wait_sec=self.pipeline_wait_seconds,finish_join_sec=join,
   ball_batches=self.batched.batches,ball_model_items=self.batched.model_items,worker_and_main_timings_overlap=True,
   quick_geometry_accepted=self.quick.accepted,middle_graph_calls=self.ball.middle.calls,numeric_guard_calls=self.ball.middle.guards))
  if self.last_debug is not None:self.last_debug.update(review_samples=self.review_samples,geometry_review_audit=self.geometry_review_audit)
  return result

"""Video-only C lifecycle: bounded CPU geometry / batched GPU ball pipeline."""
import json
import time
import numpy as np
from .base_solution import BaseSolution
from .batch_ball import BatchBall
from .pipeline import GeometryWorker
from .geometry_engine import GeometryEngine
from .auto_table_geometry import extract_quad
from .contact_review import review


class Solution(BaseSolution):
    def prepare(self,context):
        super().prepare(context)
        self.engineering=json.loads((self.root/'configs/engineering_C.json').read_text(encoding='utf-8'))
        if self.engineering['schema']!='tcn_engineering_C_v1' or self.engineering['batch_size']!=8 or self.engineering['queue_frames']!=8:
            raise ValueError('Unsupported C engineering recipe')
        if self.engineering['geometry_mode'] not in ('safe','tracked'):raise ValueError('Unknown frozen C mode')
        # No real video is touched in prepare; warm only the registered shape.
        self.ball.session.run(None,{self.ball.name:np.zeros((8,9,288,512),np.float32)})
        context.info.update(engineering_variant='C_'+self.engineering['geometry_mode'],c_pipeline_timings=[],
                            geometry_review='unchanged_previous_selected_review',ball_batch_size=8)

    def reset(self,video_meta):
        previous=getattr(self,'worker',None)
        if previous is not None and not previous.closed:previous.abort()
        super().reset(video_meta)
        def provider(frame,image,evidence):
            start=time.perf_counter();probability=self.table.probability(image);model_seconds=time.perf_counter()-start
            return extract_quad(image,probability,self.cfg,frame_evidence=evidence),model_seconds
        self.engine=GeometryEngine(self.times,self.size,provider,self.engineering['geometry_mode'])
        self.worker=GeometryWorker(self.engine)
        self.batched=BatchBall(self.ball,self.size)
        self.pipeline_wait_seconds=0.

    def process_frame(self,frame_rgb,frame_meta):
        if self.finished or frame_meta['frame_id']!=self.fid or frame_meta['video_id']!=self.meta.video_id:
            raise ValueError('Official frame sequence mismatch')
        if frame_rgb.shape!=(self.size[1],self.size[0],3) or frame_rgb.dtype!=np.uint8:
            raise ValueError('Expected unchanged official RGB frame')
        start=time.perf_counter();self.worker.submit(self.fid,frame_rgb);self.pipeline_wait_seconds+=time.perf_counter()-start
        start=time.perf_counter();self.detections.extend(self.batched.push(frame_rgb));self.timing['ball_sec']+=time.perf_counter()-start
        self.fid+=1
        return []

    def review_points(self,points,tracks):
        points,self.geometry_review_audit=review(points,tracks,self.review_samples,self.size,self.geometry_cuts)
        return points

    def finish_video(self,video_meta):
        if self.finished:return []
        start=time.perf_counter();self.detections.extend(self.batched.flush());self.timing['ball_sec']+=time.perf_counter()-start
        start=time.perf_counter();geometry=self.worker.close();join_seconds=time.perf_counter()-start
        self.samples=geometry['samples'];self.review_samples=geometry['review_samples'];self.geometry_cuts=geometry['cuts']
        self.profile=self.engine.profile
        self.timing['table_model_sec']=geometry['table_model_sec'];self.timing['table_geometry_sec']=geometry['table_geometry_sec']
        output=super().finish_video(video_meta)
        self.context.info['c_pipeline_timings'].append(dict(video_id=video_meta.video_id,full_geometry_calls=geometry['full_calls'],
            maximum_queue_frames=geometry['maximum_queue_frames'],enqueue_wait_sec=self.pipeline_wait_seconds,
            finish_join_sec=join_seconds,ball_batches=self.batched.batches,ball_model_items=self.batched.model_items,
            worker_and_main_timings_overlap=True))
        if self.last_debug is not None:
            self.last_debug.update(review_samples=self.review_samples,geometry_review_audit=self.geometry_review_audit)
        return output

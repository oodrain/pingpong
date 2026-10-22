"""Same batch8 input, frame delay, normalization, and tail; reusable window storage."""
import time
import cv2
import numpy as np
from .video_ball_detector import VideoBallDetector
from .middle_ball import decode_cpu as decode_heat
BATCH_SIZE=8;NUMERIC_GUARD=2e-6
def normalize_rgb(rgb):
 x=cv2.resize(rgb,(512,288)).astype(np.float32)/255
 x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
 return x.transpose(2,0,1).copy()
class BatchBall:
 def __init__(self,detector,size,batch_size=BATCH_SIZE):
  if batch_size!=8:raise ValueError('Only frozen batch8')
  self.detector=detector;self.size=tuple(size);self.frame=0;self.ids=[]
  blank=detector.normalize(np.zeros((288,512,3),np.uint8));self.history=[blank,blank]
  self.tensor=np.empty((8,9,288,512),np.float32);self.batches=0;self.model_items=0
 def push(self,rgb):
  start=time.perf_counter();self.history=(self.history+[normalize_rgb(rgb)])[-3:]
  j=len(self.ids)
  for k,v in enumerate(self.history):self.tensor[j,k*3:(k+1)*3]=v
  self.ids.append(self.frame-1);self.frame+=1
  self.detector.profile.add('normalization_and_window',time.perf_counter()-start)
  return self.flush() if len(self.ids)==8 else []
 def flush(self):
  if not self.ids:return []
  n=len(self.ids);rows=self.detector.middle.infer(self.tensor[:n],self.size)
  result=[dict(target_frame_id=i,detection=d) for i,d in zip(self.ids,rows)]
  self.batches+=1;self.model_items+=n;self.ids=[];return result

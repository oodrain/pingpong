"""FP32 batched three-frame inference with unchanged middle-frame semantics."""
import time
import cv2
import numpy as np
from .video_ball_detector import VideoBallDetector

BATCH_SIZE=8
NUMERIC_GUARD=2e-6


def normalize_rgb(rgb):
    x=cv2.resize(rgb,(512,288)).astype(np.float32)/255
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return x.transpose(2,0,1).copy()


def decode_heat(heat,size):
    mask=heat>.7
    if not mask.any():return None
    yy,xx=np.nonzero(mask);weights=heat[mask].astype(np.float64);total=weights.sum()
    width,height=size
    return dict(x=float(xx@weights/total*width/512),y=float(yy@weights/total*height/288),confidence=float(heat[mask].max()))


class BatchBall:
    def __init__(self,detector,size,batch_size=BATCH_SIZE):
        if batch_size!=BATCH_SIZE:raise ValueError('Only the frozen batch8 recipe is supported')
        self.detector=detector;self.size=tuple(size);self.pending=[];self.frame=0
        blank=detector.normalize(np.zeros((288,512,3),np.uint8))
        self.history=[blank,blank];self.batches=0;self.model_items=0;self.serial_guard_calls=0

    def push(self,rgb):
        start=time.perf_counter()
        self.history=(self.history+[normalize_rgb(rgb)])[-3:]
        self.pending.append((self.frame-1,np.concatenate(self.history,axis=0)))
        self.frame+=1
        self.detector.profile.add('normalization_and_window',time.perf_counter()-start)
        return self.flush() if len(self.pending)>=BATCH_SIZE else []

    def flush(self):
        if not self.pending:return []
        ids=[x[0] for x in self.pending]
        tensor=np.stack([x[1] for x in self.pending])
        start=time.perf_counter()
        heat=self.detector.session.run(None,{self.detector.name:tensor})[0]
        if len(ids)>1:
            for j,h in enumerate(heat[:,1]):
                if np.any(abs(h-.7)<=NUMERIC_GUARD) or abs(float(h.max())-.75)<=NUMERIC_GUARD:
                    heat[j]=self.detector.session.run(None,{self.detector.name:tensor[j:j+1]})[0][0]
                    self.serial_guard_calls+=1
        self.detector.profile.add('batch_inference_and_transfers',time.perf_counter()-start)
        if heat.shape!=(len(ids),3,288,512):raise ValueError('Unexpected batch heatmap shape')
        start=time.perf_counter()
        result=[dict(target_frame_id=i,detection=decode_heat(h[1],self.size)) for i,h in zip(ids,heat)]
        self.detector.profile.add('heatmap_decode',time.perf_counter()-start)
        self.batches+=1;self.model_items+=len(ids);self.pending=[]
        return result

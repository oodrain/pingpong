"""Official middle-channel detector via ONNX Runtime; zero manual inputs."""
import hashlib
from pathlib import Path
import cv2
import numpy as np

EXPECTED_SHA256='a5a467f134ae7e2fc5679e395dca9af12b6dfcc65cee51e633c465beb0c61543'

class VideoBallDetector:
    def __init__(self,checkpoint,threads=4):
        import onnxruntime as ort
        digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if digest!=EXPECTED_SHA256:raise ValueError('Unexpected official ball model')
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(checkpoint),sess_options=options,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].type!='tensor(float)':raise ValueError('Unexpected ball model dtype')
        self.name=self.session.get_inputs()[0].name;self.reset()
        self.provenance={'checkpoint_sha256':digest,'backend':'onnxruntime_cpu','version':ort.__version__,
                         'threshold':.7,'middle_channel_delay_frames':1,'manual_labels_used':False}
    @staticmethod
    def normalize(bgr):
        x=cv2.cvtColor(cv2.resize(bgr,(512,288)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return x.transpose(2,0,1).copy()
    def reset(self):
        blank=self.normalize(np.zeros((288,512,3),np.uint8));self.buffer=[blank,blank]
    def detect(self,image):
        self.buffer=(self.buffer+[self.normalize(image)])[-3:]
        heat=self.session.run(None,{self.name:np.concatenate(self.buffer,axis=0)[None]})[0][0,1]
        mask=heat>.7
        if not mask.any():return None
        yy,xx=np.nonzero(mask);w=heat[mask].astype(np.float64);total=w.sum();h,width=image.shape[:2]
        return {'x':float(xx@w/total*width/512),'y':float(yy@w/total*h/288),'confidence':float(heat[mask].max())}

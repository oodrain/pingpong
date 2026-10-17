"""Unchanged FP32 middle-frame model; explicit device I/O and centroid reduction."""
import time,os
from pathlib import Path
import numpy as np

GUARD=2e-6

def decode_cpu(heat,size):
 mask=heat>.7
 if not mask.any():return None
 y,x=np.nonzero(mask);v=heat[mask].astype(np.float64);total=v.sum()
 return dict(x=float(x@v/total*size[0]/512),y=float(y@v/total*size[1]/288),confidence=float(heat[mask].max()))

class MiddleBall:
 def __init__(self,detector,path,device):
  import onnxruntime as ort
  self.detector=detector;self.device=device;self.calls=0;self.guards=0
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  prefix=os.environ.get('TCN_ORT_PROFILE_PREFIX')
  if prefix:opts.enable_profiling=True;opts.profile_file_prefix=prefix+'_middle'
  if device=='cuda':
   import torch
   self.torch=torch;self.stream=torch.cuda.Stream(device=0)
   providers=[('CUDAExecutionProvider',dict(device_id=0,use_tf32='0',user_compute_stream=str(self.stream.cuda_stream))),'CPUExecutionProvider']
  else:providers=['CPUExecutionProvider']
  self.session=ort.InferenceSession(str(path),sess_options=opts,providers=providers)
  self.name=self.session.get_inputs()[0].name;self.output=self.session.get_outputs()[0].name
  if device=='cuda':
   if self.session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('Middle graph is not using CUDA')
   with torch.cuda.stream(self.stream):
    self.input=torch.empty((8,9,288,512),device='cuda',dtype=torch.float32)
    self.heat=torch.empty((8,1,288,512),device='cuda',dtype=torch.float32)
    self.host=torch.empty((8,9,288,512),pin_memory=True,dtype=torch.float32)
    self.xx=torch.arange(512,device='cuda',dtype=torch.float64).reshape(1,1,512)
    self.yy=torch.arange(288,device='cuda',dtype=torch.float64).reshape(1,288,1)
   self.stream.synchronize()

 def run_heat(self,tensor):
  # The fixed CPU screen found the one-channel graph slower. Preserve C's
  # faster original CPU graph; CUDA retains the derived graph and device I/O.
  if self.device!='cuda':return self.detector.session.run(None,{self.detector.name:tensor})[0][:,1]
  torch=self.torch;n=len(tensor)
  with torch.cuda.stream(self.stream):
   np.copyto(self.host.numpy()[:n],tensor)
   self.input[:n].copy_(self.host[:n],non_blocking=True)
   binding=self.session.io_binding()
   binding.bind_input(self.name,'cuda',0,np.float32,(n,9,288,512),self.input.data_ptr())
   binding.bind_output(self.output,'cuda',0,np.float32,(n,1,288,512),self.heat.data_ptr())
   self.session.run_with_iobinding(binding)
  return self.heat[:n,0]

 def infer(self,tensor,size):
  start=time.perf_counter();heat=self.run_heat(tensor);self.calls+=1
  if self.device=='cuda':
   torch=self.torch
   with torch.cuda.stream(self.stream):
    mask=heat>.7;values=torch.where(mask,heat,0.).to(torch.float64)
    total=values.sum((1,2));maximum=heat.amax((1,2))
    near=((heat-.7).abs()<=GUARD).any(dim=2).any(dim=1)|((maximum-.75).abs()<=GUARD)
    stats=torch.stack(((values*self.xx).sum((1,2)),(values*self.yy).sum((1,2)),total,maximum.to(torch.float64),near.to(torch.float64)),dim=1).cpu().numpy()
   rows=[]
   for sx,sy,den,maximum,near in stats:
    rows.append(None if den==0 else dict(x=float(sx/den*size[0]/512),y=float(sy/den*size[1]/288),confidence=float(maximum)))
   guards=stats[:,-1].astype(bool)
  else:
   rows=[decode_cpu(h,size) for h in heat]
   guards=np.array([np.any(abs(h-.7)<=GUARD) or abs(float(h.max())-.75)<=GUARD for h in heat])
  for j in np.flatnonzero(guards):
   old=self.detector.session.run(None,{self.detector.name:tensor[j:j+1]})[0][0,1]
   rows[j]=decode_cpu(old,size);self.guards+=1
  self.detector.profile.add('middle_inference_device_IO_decode',time.perf_counter()-start)
  return rows

 def warmup_check(self):
  # No video is read; actual-video cloud comparison is a separate required check.
  x=np.zeros((8,9,288,512),np.float32)
  old=self.detector.session.run(None,{self.detector.name:x})[0][:,1]
  new=self.run_heat(x) if self.device=='cuda' else self.session.run(None,{self.name:x})[0][:,0]
  if self.device=='cuda':
   with self.torch.cuda.stream(self.stream):new=new.cpu().numpy()
  if not np.array_equal(old>.7,new>.7) or float(abs(old-new).max())>5e-6:
   raise RuntimeError('Derived middle graph warmup differs from original; do not benchmark')
  return dict(max_heat_error=float(abs(old-new).max()),threshold_mask_exact=True,device_IO=self.device=='cuda')

"""RacketVision RTMDet-M inference on CPU, no annotation or dataset inputs.

Uses the pinned Apache-2.0 OpenMMLab architecture classes and forward methods.
Only inference components are compiled; no MMCV custom CUDA operators needed.
Configuration: official RacketVision rtmdet_m_racket_infer.py (3 sport classes).
"""
import ast, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version

EXPECTED_SHA256='e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd'
INFERENCE_STATE_SHA256='7da20afc3386fea344d0a4f1c3b00005789695794c26b74d1446bfbd6b0e9d10'
# COCO category order after filtering to the three training classes; verified
# against the published COCO file and official SPORT_TO_DET_CAT_ID mapping.
CLASS_NAMES=('badminton_racket','tabletennis_racket','tennis_racket')
DECLARED_METADATA_CLASSES=('tabletennis_racket','badminton_racket','tennis_racket')
SOURCES=Path(__file__).with_name('racket_model_sources')

def architecture_classes():
    manifest=json.loads((SOURCES/'sources.json').read_text(encoding='utf-8'))
    ns={'torch':torch,'nn':nn,'math':math,'BaseModule':BaseModule,'ConvModule':ConvModule,
        'DepthwiseSeparableConvModule':DepthwiseSeparableConvModule,'digit_version':digit_version,
        '_BatchNorm':nn.modules.batchnorm._BatchNorm}
    selections=[('se_layer.py',['ChannelAttention']),('csp_layer.py',['DarknetBottleneck','CSPNeXtBlock','CSPLayer']),
                ('csp_darknet.py',['SPPBottleneck']),('cspnext.py',['CSPNeXt']),('cspnext_pafpn.py',['CSPNeXtPAFPN'])]
    def tree(name):
        path=SOURCES/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[name]['sha256']:raise ValueError('Changed model architecture source')
        return ast.parse(path.read_text(encoding='utf-8'))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    for filename,names in selections:
        body=[n for n in tree(filename).body if isinstance(n,ast.ClassDef) and n.name in names]
        if len(body)!=len(names):raise ValueError('Missing pinned class')
        for cls in body:cls.decorator_list=[]
        module=ast.fix_missing_locations(ast.Module(body=[future,*body],type_ignores=[]))
        exec(compile(module,str(SOURCES/filename),'exec'),ns)
    # Keep the upstream layer sharing and exact exp(regression) forward method.
    head=next(n for n in tree('rtmdet_head.py').body if isinstance(n,ast.ClassDef) and n.name=='RTMDetSepBNHead')
    methods=[n for n in head.body if isinstance(n,ast.FunctionDef) and n.name in ('_init_layers','forward')]
    for method in methods:
        module=ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[]))
        exec(compile(module,str(SOURCES/'rtmdet_head.py'),'exec'),ns)
    return ns

def build_model():
    ns=architecture_classes()
    class Head(nn.Module):
        _init_layers=ns['_init_layers'];forward=ns['forward']
        def __init__(self):
            super().__init__();self.prior_generator=SimpleNamespace(strides=[(8,8),(16,16),(32,32)])
            self.in_channels=self.feat_channels=192;self.stacked_convs=2;self.num_base_priors=1
            self.cls_out_channels=3;self.pred_kernel_size=1;self.conv_cfg=None;self.norm_cfg={'type':'BN'}
            self.act_cfg={'type':'SiLU','inplace':True};self.use_depthwise=False;self.with_objectness=False
            self.share_conv=True;self.exp_on_reg=True;self._init_layers()
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=ns['CSPNeXt'](arch='P5',expand_ratio=.5,deepen_factor=.67,widen_factor=.75,
                channel_attention=True,norm_cfg={'type':'BN'},act_cfg={'type':'SiLU','inplace':True})
            self.neck=ns['CSPNeXtPAFPN'](in_channels=[192,384,768],out_channels=192,num_csp_blocks=2,
                use_depthwise=False,expand_ratio=.5,norm_cfg={'type':'BN'},act_cfg={'type':'SiLU','inplace':True})
            self.bbox_head=Head()
        def forward(self,x):return self.bbox_head(self.neck(self.backbone(x)))
    return Model()

def preprocess(image):
    h,w=image.shape[:2];ratio=min(640/w,640/h);rw,rh=round(w*ratio),round(h*ratio)
    resized=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
    padded=np.full((640,640,3),114,np.float32);padded[:rh,:rw]=resized
    padded=(padded-np.array([103.53,116.28,123.675],np.float32))/np.array([57.375,57.12,58.395],np.float32)
    return torch.from_numpy(padded.transpose(2,0,1).copy()).unsqueeze(0),[rw/w,rh/h]

def decode(outputs,scale,image_size,threshold=.3):
    cls_outputs,dist_outputs=outputs;boxes=[];scores=[];labels=[]
    for cls,dist,stride in zip(cls_outputs,dist_outputs,(8,16,32)):
        h,w=cls.shape[-2:];s=cls[0].permute(1,2,0).reshape(-1,3).sigmoid()
        # Upstream nms_pre is applied over all sport-specific class scores.
        ids,classes=torch.where(s>.05)
        if not len(ids):continue
        conf=s[ids,classes];order=conf.argsort(descending=True)[:1000];ids=ids[order];classes=classes[order];conf=conf[order]
        d=dist[0].permute(1,2,0).reshape(-1,4)[ids]
        xy=torch.stack([ids%w,ids//w],dim=1).to(d.dtype)*stride
        b=torch.cat([xy-d[:,:2],xy+d[:,2:]],dim=1)
        boxes.append(b);scores.append(conf);labels.append(classes)
    if not boxes:return []
    boxes=torch.cat(boxes);scores=torch.cat(scores);labels=torch.cat(labels)
    height,width=image_size;boxes/=torch.tensor([*scale,*scale],dtype=boxes.dtype)
    boxes[:,0::2].clamp_(0,width-1);boxes[:,1::2].clamp_(0,height-1)
    keep=[]
    for kind in range(3):
        ids=torch.where(labels==kind)[0]
        keep.extend(ids[nms(boxes[ids],scores[ids],.6)].tolist())
    keep=sorted(keep,key=lambda i:float(scores[i]),reverse=True)[:100]
    return [{'box':boxes[i].tolist(),'score':float(scores[i]),'class_id':int(labels[i]),
             'class_name':CLASS_NAMES[int(labels[i])],
             'face_only':False} for i in keep if int(labels[i])==1 and float(scores[i])>=threshold
            and 0<float((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]))/(width*height)<.5]

class RacketDetector:
    def __init__(self,checkpoint,threads=4):
        torch.set_num_threads(threads);path=Path(checkpoint);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=INFERENCE_STATE_SHA256:raise ValueError('Unexpected inference state SHA256')
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        if loaded.get('original_sha256')!=EXPECTED_SHA256 or tuple(loaded.get('classes',[]))!=DECLARED_METADATA_CLASSES:
            raise ValueError('Unexpected training checkpoint or class mapping')
        self.model=build_model()
        self.model.load_state_dict(loaded['state_dict'],strict=True);self.model.eval()
        self.provenance={'checkpoint_sha256':digest,'original_checkpoint_sha256':EXPECTED_SHA256,
            'model':'RacketVision RTMDet-M','sport_class_id':1,
            'class_mapping_source':'published COCO category order filtered by training classes; official inference mapping',
            'score_threshold':.3,'nms_iou':.6,'source_manifest_sha256':hashlib.sha256((SOURCES/'sources.json').read_bytes()).hexdigest(),
            'device':'cpu','torch':torch.__version__,'manual_labels_used':False,'face_pose_estimator_used':False}
    def detect(self,image):
        tensor,scale=preprocess(image)
        with torch.inference_mode():outputs=self.model(tensor)
        return decode(outputs,scale,image.shape[:2])
    def detect_near_people(self,image,ball,people):
        result=[dict(p,source='full_frame') for p in self.detect(image)]
        if not ball.get('detected') or ball.get('confidence',0)<.75:return result
        xy=np.array([ball['x'],ball['y']]);choices=[]
        for i,p in enumerate(people):
            if p['score']<.9:continue
            b=np.asarray(p['box']);height=b[3]-b[1]
            if height<=0:continue
            distance=float(np.linalg.norm(np.maximum(np.maximum(b[:2]-xy,xy-b[2:]),0)))/height
            if distance<=.45:choices.append((distance,i))
        if choices:
            p=people[min(choices)[1]];b=p['box'];margin=.2*(b[3]-b[1]);h,w=image.shape[:2]
            x0=max(0,int(b[0]-margin));y0=max(0,int(b[1]-margin));x1=min(w,int(b[2]+margin));y1=min(h,int(b[3]+margin))
            for r in self.detect(image[y0:y1,x0:x1]):
                box=r['box'];r=dict(r,box=[box[0]+x0,box[1]+y0,box[2]+x0,box[3]+y0],
                    source='automatic_person_crop',person_box=p['box'],roi=[x0,y0,x1,y1]);result.append(r)
        if not result:return []
        keep=nms(torch.tensor([p['box'] for p in result]),torch.tensor([p['score'] for p in result]),.6).tolist()
        return [result[i] for i in keep]

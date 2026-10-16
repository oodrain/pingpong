"""Automatic motion context and a small TCN; no labels in inference features."""
import numpy as np
import torch
from torch import nn

from .frame_features import FEATURES as BASE_FEATURES
from .temporal import TemporalBlock

MOTION_FEATURES=[]
for radius in (3,5):
    MOTION_FEATURES += [f'{name}_{radius}' for name in (
        'fit_available','pre_vx','pre_vy','post_vx','post_vy',
        'pre_speed','post_speed','turn_cosine','relative_speed_change',
        'fit_discontinuity','flight_residual','observation_fraction')]


def motion_features(tracks,image_size):
    """Fit observed points before/after each frame using real timestamps.

    At most five future frames are used. Coordinates are never filled for
    missing detections; availability and support fraction remain explicit.
    """
    width,height=image_size
    times=np.asarray([r['timestamp_s'] for r in tracks],float)
    if width<=0 or height<=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid automatic motion sequence')
    result=np.zeros((len(tracks),len(MOTION_FEATURES)),np.float32)
    good=np.asarray([bool(r.get('detected') and r.get('in_image',True)
        and (r.get('confidence') or 0)>=.5
        and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x','y'))
        and 0<=r['x']<width and 0<=r['y']<height) for r in tracks])
    xy=np.asarray([[r['x']/width,r['y']/height] if g else [0.,0.] for r,g in zip(tracks,good)])
    confidence=np.asarray([(r.get('confidence') or 0) if g else 0. for r,g in zip(tracks,good)])
    nominal=float(np.median(np.diff(times))) if len(times)>1 else .04
    for fid in range(len(tracks)):
        for level,radius in enumerate((3,5)):
            before=np.asarray([j for j in range(max(0,fid-radius),fid) if good[j]],int)
            after=np.asarray([j for j in range(fid+1,min(len(tracks),fid+radius+1)) if good[j]],int)
            offset=level*12
            result[fid,offset+11]=(len(before)+len(after))/(2*radius)
            if len(before)<2 or len(after)<2:continue
            support=np.r_[before,after]
            if np.max(np.diff(times[support]))>2.5*nominal:continue
            def fit(indices):
                t=(times[indices]-times[fid])/.04
                design=np.c_[np.ones(len(t)),t]
                weights=np.sqrt(confidence[indices])[:,None]
                return np.linalg.lstsq(design*weights,xy[indices]*weights,rcond=None)[0]
            pre,post=fit(before),fit(after)
            v1,v2=pre[1],post[1];s1,s2=np.linalg.norm(v1),np.linalg.norm(v2)
            cosine=float(v1@v2/max(s1*s2,1e-8))
            # Smooth quadratic flight explains perspective drift without an
            # image-y sign requirement; residual is evidence, never a veto.
            t=(times[support]-times[fid])/.04
            design=np.c_[np.ones(len(t)),t,t*t]
            weights=np.sqrt(confidence[support])[:,None]
            curve=np.linalg.lstsq(design*weights,xy[support]*weights,rcond=None)[0]
            residual=float(np.sqrt(np.mean((design@curve-xy[support])**2)))
            result[fid,offset:offset+11]=[1.,*np.clip(v1,-2,2),*np.clip(v2,-2,2),
                min(s1,2),min(s2,2),np.clip(cosine,-1,1),np.clip((s2-s1)/max(s1+s2,1e-6),-1,1),
                min(float(np.linalg.norm(pre[0]-post[0])),1),min(residual,1)]
    return result


def feature_names(mode):
    return BASE_FEATURES+(MOTION_FEATURES if mode=='motion' else [])


def augment_horizontal(features,mode):
    """Mirror a feature sequence while preserving absent-value masks."""
    x=np.asarray(features,np.float32).copy();names=feature_names(mode)
    for name,mask in [('ball_x_n','ball_available')]+[(f'corner_{i}_x_n','table_available') for i in range(4)]:
        i=names.index(name);valid=x[:,names.index(mask)]>0
        x[valid,i]=1-x[valid,i]
    for name in ('velocity_x_n','acceleration_x_n'):
        x[:,names.index(name)]*=-1
    if mode=='motion':
        for radius in (3,5):
            for name in ('pre_vx','post_vx'):x[:,names.index(f'{name}_{radius}')]*=-1
    return x


def training_weights(targets,mask,events,event_weight=1.):
    """Extra known non-touch events, only where human background is trusted.

    Explicit touches and their soft shoulders are never made negative, and
    concurrent touch/hit labels stay independent.
    """
    weights=np.asarray(mask,np.float32).copy()
    if event_weight==1:return weights
    for event in events:
        if event['event_type']=='table_bounce':continue
        lo,hi=event['interval_frames']
        for fid in range(max(0,lo-1),min(len(targets),hi+2)):
            if mask[fid,0] and targets[fid,0]==0:weights[fid,0]=event_weight
    return weights


class RevisionTCN(nn.Module):
    def __init__(self,input_features,channels=32,dilations=(1,2,4,8),dropout=.1):
        super().__init__()
        self.config={'input_features':input_features,'channels':channels,'dilations':list(dilations),'dropout':dropout}
        self.input=nn.Conv1d(input_features,channels,1)
        self.blocks=nn.ModuleList(TemporalBlock(channels,d,dropout) for d in dilations)
        self.output=nn.Conv1d(channels,2,1)
    def forward(self,x):
        x=torch.relu(self.input(x))
        for block in self.blocks:x=block(x)
        return self.output(x)


def statistics(sequences):
    x=np.concatenate([s['features'][s['loss_mask'].any(1)] for s in sequences])
    mean=x.mean(0);scale=x.std(0);scale[scale<.001]=.001
    mean[0]=0.;scale[0]=1.
    return {'mean':mean.tolist(),'scale':scale.tolist(),'fit_uses_training_sequences_only':True}


def normalize(x,stats):
    return np.clip((x-np.asarray(stats['mean'],np.float32))/np.asarray(stats['scale'],np.float32),-10,10).astype(np.float32)


def predict(model,features,stats):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(normalize(features,stats).T.copy())[None]))[0].T.numpy()

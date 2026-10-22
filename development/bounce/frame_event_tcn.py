"""Small dense temporal model. Runtime inputs are automatic sequences only."""
import numpy as np
import torch
from torch import nn
from .frame_event_features import FEATURES,EVENT_TYPES


class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__();self.conv=nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation)
        self.norm=nn.LayerNorm(channels);self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        y=self.conv(x);y=self.norm(y.transpose(1,2)).transpose(1,2)
        return torch.relu(x+self.dropout(torch.relu(y)))


class FrameEventTCN(nn.Module):
    def __init__(self,channels=32,dilations=(1,2,4,8),dropout=.1):
        super().__init__();self.config={'channels':channels,'dilations':list(dilations),'dropout':dropout}
        self.receptive_field=1+2*sum(dilations)
        self.input=nn.Conv1d(len(FEATURES),channels,1)
        self.blocks=nn.ModuleList(TemporalBlock(channels,d,dropout) for d in dilations)
        self.output=nn.Conv1d(channels,len(EVENT_TYPES),1)
    def forward(self,x):
        x=torch.relu(self.input(x))
        for block in self.blocks:x=block(x)
        return self.output(x)


def normalization(sequences):
    rows=np.concatenate([s['features'][s['loss_mask'].any(1)] for s in sequences])
    mean=rows.mean(0);scale=rows.std(0);scale[scale<.001]=.001
    # Preserve explicit real-frame presence to distinguish convolution padding.
    mean[0]=0.;scale[0]=1.
    return {'mean':mean.tolist(),'scale':scale.tolist(),'fit_uses_training_sequences_only':True}


def normalize(features,stats):
    x=np.asarray(features,np.float32);mean=np.asarray(stats['mean'],np.float32);scale=np.asarray(stats['scale'],np.float32)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or mean.shape!=(len(FEATURES),) or scale.shape!=(len(FEATURES),) or not np.isfinite(x).all() or not np.isfinite([*mean,*scale]).all() or np.any(scale<=0):
        raise ValueError('Invalid TCN feature normalization')
    return np.clip((x-mean)/scale,-10,10).astype(np.float32)


def predict(model,features,stats):
    model.eval()
    with torch.inference_mode():
        tensor=torch.from_numpy(normalize(features,stats).T.copy())[None]
        return torch.sigmoid(model(tensor))[0].T.cpu().numpy()


def load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['features']!=FEATURES or checkpoint['event_types']!=list(EVENT_TYPES):raise ValueError('TCN checkpoint feature schema mismatch')
    model=FrameEventTCN(**checkpoint['architecture']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model,checkpoint

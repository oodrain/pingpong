"""Inference-only frozen model; identical tensor names and float32 math."""
import numpy as np
import torch
from .revision import RevisionTCN, normalize
class PaddingTCN(RevisionTCN):
    def __init__(self,mask_padding=True,**kwargs):
        super().__init__(**kwargs);assert mask_padding;self.config['mask_padding']=True
    def forward(self,x):
        present=(x[:,0:1,:]>.5).to(x.dtype);h=torch.relu(self.input(x))*present
        for block in self.blocks:h=block(h)*present
        return self.output(h)*present

def predict(model,x,stats):
    model.eval()
    with torch.inference_mode():return torch.sigmoid(model(torch.from_numpy(normalize(x,stats).T.copy())[None].to(next(model.parameters()).device)))[0].T.cpu().numpy()

def load(cp):
    cfg=dict(cp['architecture']);cls=ContextTCN if cfg.pop('context_projection',False) else PaddingTCN
    model=cls(**cfg);model.load_state_dict(cp['state_dict'],strict=True);model.eval();return model

class ContextTCN(PaddingTCN):
    def __init__(self,input_features=56,**kwargs):
        assert input_features==56
        super().__init__(input_features=51,**kwargs)
        self.context_weight=torch.nn.Parameter(torch.zeros(self.config['channels'],5,1))
        self.config.update(input_features=56,context_projection=True)
    def forward(self,x):
        present=(x[:,0:1,:]>.5).to(x.dtype)
        base=self.input(x[:,:51,:].contiguous())
        extra=torch.nn.functional.conv1d(x[:,51:,:].contiguous(),self.context_weight)
        h=torch.relu(base+extra)*present
        for block in self.blocks:h=block(h)*present
        return self.output(h)*present

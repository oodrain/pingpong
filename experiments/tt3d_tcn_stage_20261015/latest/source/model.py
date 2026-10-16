"""Same padding-correct TCN; objective coefficients computed from fitting labels only."""
import numpy as np,torch
from .revision import RevisionTCN,statistics,normalize
from .protocol import FIXED,VARIANTS,SEEDS

class PaddingTCN(RevisionTCN):
    def __init__(self,mask_padding=True,**kwargs):
        super().__init__(**kwargs);assert mask_padding;self.config['mask_padding']=True
    def forward(self,x):
        present=(x[:,0:1,:]>.5).to(x.dtype);h=torch.relu(self.input(x))*present
        for block in self.blocks:h=block(h)*present
        return self.output(h)*present

def event_weights(seqs,targets,mask):
    """Positive and known negative event groups never overlap or use unknowns."""
    weights=torch.zeros_like(mask);audit={}
    for channel,kind in enumerate(('table_bounce','racket_contact')):
        pos=torch.zeros_like(mask[:,channel]);neg=pos.clone();np_events=nn_events=0
        for i,s in enumerate(seqs):
            n=s['n_frames'];index=torch.arange(mask.shape[2]);y=targets[i,channel];valid=mask[i,channel]>0
            for e in s['events']:
                lo,hi=e['interval_frames'];scope=(index>=max(0,lo-2))&(index<=min(n-1,hi+2))&valid
                if e['event_type']==kind:
                    support=scope*(y>0)*y
                    if support.sum()>0:pos[i]+=support/support.sum();np_events+=1
                else:
                    support=(scope&(y==0)).to(mask.dtype)
                    if support.sum()>0:neg[i]+=support/support.sum();nn_events+=1
        bg=((mask[:,channel]>0)&(targets[:,channel]==0)&(neg==0)).to(mask.dtype)
        parts=[]
        for name,v,mass in [('positive',pos,2.),('known_negative',neg,1.),('background',bg,1.)]:
            if v.sum()>0:parts.append((name,v/v.sum(),mass))
        assert parts
        total=sum(mass for _,_,mass in parts)
        for _,v,mass in parts:weights[:,channel]+=v*(mass/total)/2
        audit[kind]={'positive_events':np_events,'known_negative_events':nn_events,'background_entries':int(bg.sum()),
            'group_mass':{name:mass/total/2 for name,_,mass in parts}}
        assert not torch.any(weights[:,channel][mask[:,channel]==0])
        assert not torch.any(neg[targets[:,channel]>0])
    assert abs(float(weights.sum())-1)<1e-5
    return weights,audit

def predict(model,x,stats):
    model.eval()
    with torch.inference_mode():return torch.sigmoid(model(torch.from_numpy(normalize(x,stats).T.copy())[None]))[0].T.numpy()

def load(cp):
    cfg=dict(cp['architecture']);cls=ContextTCN if cfg.pop('context_projection',False) else PaddingTCN
    model=cls(**cfg);model.load_state_dict(cp['state_dict'],strict=True);model.eval();return model

def calibrated_objective(seqs,y,mask):
    """Match positive AND negative loss coefficient mass separately per head."""
    wb=mask/mask.sum().clamp_min(1)
    pos=(y*mask).sum((0,2));neg=((1-y)*mask).sum((0,2))
    bb=(neg/pos.clamp_min(1)).clamp(1,12)[None,:,None]
    we,groups=event_weights(seqs,y,mask)
    bp=(wb*y*bb).sum((0,2));bn=(wb*(1-y)).sum((0,2))
    ep=(we*y).sum((0,2));en=(we*(1-y)).sum((0,2))
    factor=torch.ones_like(bp);scale=torch.zeros_like(bp)
    for c in range(2):
        if bn[c]>0 and en[c]>0:scale[c]=bn[c]/en[c]
        if bp[c]>0:
            assert ep[c]>0 and scale[c]>0
            factor[c]=bp[c]/(ep[c]*scale[c])
    we=we*scale[None,:,None];be=factor[None,:,None]
    ap=(we*y*be).sum((0,2));an=(we*(1-y)).sum((0,2))
    torch.testing.assert_close(ap,bp,rtol=2e-6,atol=2e-7);torch.testing.assert_close(an,bn,rtol=2e-6,atol=2e-7)
    audit={'base_positive':bp.tolist(),'base_negative':bn.tolist(),'calibrated_positive':ap.tolist(),'calibrated_negative':an.tolist(),
           'event_pos_weight':factor.tolist(),'event_head_scale':scale.tolist(),'event_groups':groups,
           'unknown_weight':float(we[mask==0].sum()),'computed_from_fitting_labels_only':True}
    return wb,bb,we,be,audit

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

def fitting_statistics(seqs):
    if seqs[0]['features'].shape[1]==51:return statistics(seqs)
    prefix=[dict(s,features=s['features'][:,:51].copy()) for s in seqs]
    base=statistics(prefix)
    extra=np.concatenate([s['features'][s['loss_mask'].any(1),51:] for s in seqs])
    mean=extra.mean(0);scale=extra.std(0);scale[scale<.001]=.001
    return dict(base,mean=base['mean']+mean.tolist(),scale=base['scale']+scale.tolist())

def train(seqs,variant,seed):
    assert variant in VARIANTS and seed in SEEDS
    torch.manual_seed(seed);np.random.seed(seed);stats=fitting_statistics(seqs)
    width=FIXED['feature_widths'][variant];assert all(s['features'].shape[1]==width for s in seqs)
    cls=PaddingTCN if variant=='calibrated_event' else ContextTCN
    model=cls(input_features=width,channels=FIXED['channels'],dilations=FIXED['dilations'],dropout=FIXED['dropout'])
    maximum=max(s['n_frames'] for s in seqs);x=np.zeros((len(seqs),width,maximum),np.float32);y=np.zeros((len(seqs),2,maximum),np.float32);mask=np.zeros_like(y)
    for i,s in enumerate(seqs):
        n=s['n_frames'];x[i,:,:n]=normalize(s['features'],stats).T;y[i,:,:n]=s['targets'].T;mask[i,:,:n]=s['loss_mask'].T
    x,y,mask=map(torch.from_numpy,(x,y,mask));assert mask.sum()>0
    wb,bb,we,be,audit=calibrated_objective(seqs,y,mask)
    opt=torch.optim.AdamW(model.parameters(),lr=FIXED['learning_rate'],weight_decay=FIXED['weight_decay'])
    for epoch in range(FIXED['epochs']):
        model.train();opt.zero_grad(set_to_none=True);logits=model(x)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=be,reduction='none')*we).sum()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),FIXED['gradient_clip']);opt.step()
    return model,stats,dict(audit,padding_length=maximum,unknown_weight=float(we[mask==0].sum()),feature_width=width,extra_projection_zero_initialized=width==56)

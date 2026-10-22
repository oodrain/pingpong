"""Reproduce train-only CV or verify the selected final model weights on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import HERE,read,sequences,write
from development.bounce.frame_event_features import FEATURES,EVENT_TYPES,decode_frame_events
from development.bounce.frame_event_tcn import FrameEventTCN,normalization,normalize,predict,load

EPOCHS=(30,60,120);THRESHOLDS=(.35,.5,.65,.8);SEED=1729

def fit(rows,epochs,callback=None):
    torch.manual_seed(SEED);np.random.seed(SEED);model=FrameEventTCN();stats=normalization(rows)
    maximum=max(r['n_frames'] for r in rows);batch=len(rows)
    x=np.zeros((batch,len(FEATURES),maximum),np.float32)
    y=np.zeros((batch,len(EVENT_TYPES),maximum),np.float32);mask=np.zeros_like(y)
    for i,row in enumerate(rows):
        n=row['n_frames'];x[i,:,:n]=normalize(row['features'],stats).T;y[i,:,:n]=row['targets'].T;mask[i,:,:n]=row['loss_mask'].T
    x=torch.from_numpy(x);y=torch.from_numpy(y);mask=torch.from_numpy(mask)
    positive=(y*mask).sum((0,2));negative=((1-y)*mask).sum((0,2))
    weight=(negative/positive.clamp_min(1)).clamp(1,12)[None,:,None]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(x)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=weight,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if callback and epoch in EPOCHS:callback(epoch,model,stats)
    return model,stats

def temporal_counts(points,events):
    owner={}
    def augment(i,seen):
        for j,event in enumerate(events):
            if j in seen or points[i]['video_id']!=event['video_id']:continue
            lo,hi=event['interval_frames']
            if max(lo-points[i]['frame_id'],points[i]['frame_id']-hi,0)>1:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(points)):augment(i,set())
    tp=len(owner);fp=len(points)-tp;fn=len(events)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if points or events else 0}

def group_selection(dataset,rows):
    scores={epoch:{} for epoch in EPOCHS};groups=sorted({r['source_group'] for r in rows})
    for group in groups:
        train=[r for r in rows if r['source_group']!=group];held=[r for r in rows if r['source_group']==group]
        def checkpoint(epoch,model,stats):
            for row in held:scores[epoch][row['video_id']]=predict(model,row['features'],stats)
        fit(train,max(EPOCHS),checkpoint);print('Completed held training source',group,flush=True)
    events=[e for e in dataset['events'] if e['event_type']=='table_bounce'];search=[]
    for epoch in EPOCHS:
        for threshold in THRESHOLDS:
            points=[]
            for row in rows:
                output,_=decode_frame_events(scores[epoch][row['video_id']][:,0],row['tracks'],threshold)
                points.extend(dict(p,video_id=row['video_id']) for p in output if row['evaluation_mask'][p['frame_id']])
            search.append({'epochs':epoch,'threshold':threshold,'window_temporal_metrics':temporal_counts(points,events)})
    selected=max(search,key=lambda r:(r['window_temporal_metrics']['f1'],-r['window_temporal_metrics']['fp'],r['window_temporal_metrics']['tp'],-r['epochs'],r['threshold']))
    return selected,search

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cv',action='store_true')
    parser.add_argument('--verify-frozen',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    dataset,rows=sequences('train');selection=read(HERE/'assets/training_selection.json');selected=selection['selected'];report={}
    if args.cv:
        selected,search=group_selection(dataset,rows)
        if selected!=selection['selected'] or search!=selection['search']:raise ValueError('Training CV selection did not reproduce')
        report.update(cv_selection_reproduced=True,selected=selected)
    model,stats=fit(rows,selected['epochs'])
    if args.verify_frozen:
        frozen,checkpoint=load(HERE/'assets/model.pt')
        if stats!=checkpoint['normalization']:raise ValueError('Training normalization did not reproduce')
        if any(not torch.equal(v,frozen.state_dict()[k]) for k,v in model.state_dict().items()):raise ValueError('Final weights did not reproduce')
        report.update(frozen_weights_identical=True,training_normalization_identical=True)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        torch.save({'schema_version':1,'features':FEATURES,'event_types':list(EVENT_TYPES),'architecture':model.config,
            'state_dict':model.state_dict(),'normalization':stats,'threshold':selected['threshold'],'minimum_separation_s':.08,
            'training_provenance':{'split_id':dataset['split_id'],'training_video_ids':[r['video_id'] for r in rows],
                'training_source_groups':sorted({r['source_group'] for r in rows}),'parameter_selection_uses_train_sources_only':True}},args.output/'model.pt')
        write(args.output/'verification.json',report)
    print(json.dumps(report,indent=2))

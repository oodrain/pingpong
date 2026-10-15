"""Train-only grouped selection of a small dense TCN baseline."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import load_protocol,select,sha
from development.bounce.frame_event_features import FEATURES,EVENT_TYPES,decode_frame_events
from development.bounce.frame_event_tcn import FrameEventTCN,normalization,normalize,predict
from select_cached_contact_consensus import temporal_counts

EPOCHS=(30,60,120);THRESHOLDS=(.35,.5,.65,.8);SEED=1729
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load_sequences(dataset):
    sequences=[]
    for record in dataset['sequences']:
        arrays=np.load(record['path'],allow_pickle=False)
        sequences.append(dict(record,**{k:arrays[k] for k in arrays.files},tracks=lines(record['tracks_path'])))
    return sequences


def train_model(sequences,epochs,callback=None,seed=SEED):
    torch.manual_seed(seed);np.random.seed(seed);model=FrameEventTCN();stats=normalization(sequences)
    maximum=max(s['n_frames'] for s in sequences);batch=len(sequences)
    x=np.zeros((batch,len(FEATURES),maximum),np.float32);y=np.zeros((batch,len(EVENT_TYPES),maximum),np.float32);mask=np.zeros_like(y)
    for i,s in enumerate(sequences):
        n=s['n_frames'];x[i,:,:n]=normalize(s['features'],stats).T;y[i,:,:n]=s['targets'].T;mask[i,:,:n]=s['loss_mask'].T
    x=torch.from_numpy(x);y=torch.from_numpy(y);mask=torch.from_numpy(mask)
    positive=(y*mask).sum((0,2));negative=((1-y)*mask).sum((0,2));weight=(negative/positive.clamp_min(1)).clamp(1,12)[None,:,None]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    log=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(x)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=weight,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epoch%15==0:
            log.append({'epoch':epoch,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started})
            print('Epoch',epoch,'loss',round(float(loss.detach()),4),'seconds',round(time.monotonic()-started,1),flush=True)
        if callback and epoch in EPOCHS:callback(epoch,model,stats)
    return model,stats,{'epochs':epochs,'training_sequences':batch,'seed':seed,'positive_weights':weight.flatten().tolist(),
                       'history':log,'elapsed_seconds':time.monotonic()-started}


def run(args):
    protocol=load_protocol();dataset=read(args.dataset/'dataset.json')
    select(protocol,[s['video_id'] for s in dataset['sequences']],'training')
    for path,digest in dataset['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen TCN dataset changed: '+path)
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    sequences=load_sequences(dataset);groups=sorted({s['source_group'] for s in sequences});folds=[];scores={e:{} for e in EPOCHS}
    plan={'architecture':{'channels':32,'dilations':[1,2,4,8],'dropout':.1},'receptive_field_frames':31,
        'epoch_checkpoints':list(EPOCHS),'threshold_candidates':list(THRESHOLDS),'minimum_separation_s':.08,'seed':SEED,
        'dataset_sha256':sha(args.dataset/'dataset.json'),'training_source_groups':groups,'selection_inputs':'Completed training windows, source-group out-of-fold predictions only',
        'reserved_or_public_selection_inputs':False,'detector_inference_calls':0}
    save(args.output/'plan.json',plan)
    for index,group in enumerate(groups):
        train=[s for s in sequences if s['source_group']!=group];held=[s for s in sequences if s['source_group']==group]
        print('Held source',group,'train videos',len(train),'held videos',len(held),flush=True)
        def checkpoint(epoch,model,stats):
            for sequence in held:scores[epoch][sequence['video_id']]=predict(model,sequence['features'],stats)
        _,_,log=train_model(train,max(EPOCHS),checkpoint)
        folds.append({'held_source':group,'training_video_ids':[s['video_id'] for s in train],
            'held_video_ids':[s['video_id'] for s in held],'training_log':log})
        save(args.output/'progress.json',{'phase':'group_cv','folds_completed':index+1,'folds_total':len(groups),'last_source':group})
    events=[e for e in dataset['events'] if e['event_type']=='table_bounce'];search=[]
    for epoch in EPOCHS:
        for threshold in THRESHOLDS:
            points=[]
            for sequence in sequences:
                output,_=decode_frame_events(scores[epoch][sequence['video_id']][:,0],sequence['tracks'],threshold)
                points += [dict(p,video_id=sequence['video_id']) for p in output if sequence['evaluation_mask'][p['frame_id']]]
            search.append({'epochs':epoch,'threshold':threshold,'window_temporal_metrics':temporal_counts(points,events)})
    selected=max(search,key=lambda row:(row['window_temporal_metrics']['f1'],-row['window_temporal_metrics']['fp'],row['window_temporal_metrics']['tp'],-row['epochs'],row['threshold']))
    baseline=read(ROOT/'outputs/contact_rule_revision_20261014/train_cache_audit_01/outputs.json');old=[]
    for sequence in sequences:
        old += [dict(p,video_id=sequence['video_id']) for p in baseline[sequence['video_id']]['experimental'] if sequence['evaluation_mask'][p['frame_id']]]
    save(args.output/'selection.json',{'plan':plan,'selected':selected,'search':search,'folds':folds,
        'current_rule_same_scopes':temporal_counts(old,events),'metrics_are_training_selection_not_unbiased_validation':True})
    for sequence in sequences:
        path=args.output/(sequence['video_id']+'_cv_scores.npz');np.savez_compressed(path,scores=scores[selected['epochs']][sequence['video_id']])
    print('Selected',json.dumps(selected),flush=True);print('Training final fixed asset on all training sources',flush=True)
    model,stats,final_log=train_model(sequences,selected['epochs'])
    checkpoint={'schema_version':1,'features':FEATURES,'event_types':list(EVENT_TYPES),'architecture':model.config,
        'normalization':stats,'state_dict':model.state_dict(),'threshold':selected['threshold'],'minimum_separation_s':.08,
        'training_provenance':{'split_id':protocol['split_id'],'training_video_ids':[s['video_id'] for s in sequences],
            'training_source_groups':groups,'completed_review_windows':dataset['completed_reviews'],
            'supervised_frames':dataset['supervised_frame_total'],'event_counts':dataset['event_counts'],
            'dataset_sha256':sha(args.dataset/'dataset.json'),'parameter_selection_uses_train_sources_only':True,
            'candidate_gate_used':False,'manual_geometry_used_for_features':False,'independent_blind_test':False,
            'prior_public_failure_exposure_disclosed':True}}
    torch.save(checkpoint,args.output/'model.pt')
    save(args.output/'model_metadata.json',{k:v for k,v in checkpoint.items() if k!='state_dict'})
    save(args.output/'final_training_log.json',final_log)
    for name in ('frame_event_features.py','frame_event_tcn.py'):(args.output/name).write_bytes((PROJECT/'development/bounce'/name).read_bytes())
    (args.output/'source_train_frame_event_tcn.py').write_bytes(Path(__file__).read_bytes())
    save(args.output/'model_freeze.json',{'model_sha256':sha(args.output/'model.pt'),'metadata_sha256':sha(args.output/'model_metadata.json'),
        'plan_sha256':sha(args.output/'plan.json'),'selection_sha256':sha(args.output/'selection.json'),'validation_labels_opened':False})
    save(args.output/'progress.json',{'phase':'completed','model_sha256':sha(args.output/'model.pt')})
    print('Model frozen',sha(args.output/'model.pt'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

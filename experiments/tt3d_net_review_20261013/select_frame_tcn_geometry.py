"""Train-CV-only selection of one optional stable-table image support policy."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import load_protocol,select,sha
from development.bounce.frame_event_features import geometry_supported_touch_scores,decode_frame_events
from development.bounce.frame_event_tcn import load
from select_cached_contact_consensus import temporal_counts
from train_frame_event_tcn import load_sequences,read,save


def run(args):
    protocol=load_protocol();dataset=read(args.dataset/'dataset.json');sequences=load_sequences(dataset)
    select(protocol,[s['video_id'] for s in sequences],'threshold_selection')
    for path,digest in dataset['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen TCN dataset changed')
    _,checkpoint=load(args.previous/'model.pt');events=[e for e in dataset['events'] if e['event_type']=='table_bounce']
    search=[]
    for policy in ('none','stable_polygon_soft'):
        for threshold in (.35,.5,.65,.8):
            points=[]
            for sequence in sequences:
                scores=np.load(args.previous/(sequence['video_id']+'_cv_scores.npz'),allow_pickle=False)['scores'][:,0]
                samples=read(sequence['geometry_path']);adjusted=geometry_supported_touch_scores(scores,sequence['features'],samples,policy)
                output,_=decode_frame_events(adjusted,sequence['tracks'],threshold)
                points += [dict(p,video_id=sequence['video_id']) for p in output if sequence['evaluation_mask'][p['frame_id']]]
            search.append({'geometry_support_policy':policy,'threshold':threshold,'window_temporal_metrics':temporal_counts(points,events)})
    selected=max(search,key=lambda row:(row['window_temporal_metrics']['f1'],-row['window_temporal_metrics']['fp'],row['geometry_support_policy']=='none',row['threshold']))
    args.output.mkdir(parents=True,exist_ok=False)
    report={'search':search,'selected':selected,'epoch_checkpoint_fixed':read(args.previous/'selection.json')['selected']['epochs'],
        'selection_video_ids':[s['video_id'] for s in sequences],'selection_uses_reserved_or_public_labels':False,
        'previous_model_sha256':sha(args.previous/'model.pt'),'dataset_sha256':sha(args.dataset/'dataset.json'),
        'selection_scope':'Same source-held-out TCN predictions and trusted manual training frames as baseline; geometry settings fixed to 2% polygon margin and 0.25 multiplier.'}
    save(args.output/'selection.json',report)
    checkpoint['threshold']=selected['threshold'];checkpoint['geometry_support_policy']=selected['geometry_support_policy']
    checkpoint['training_provenance']['postprocessing_selection_sha256']=sha(args.output/'selection.json')
    checkpoint['training_provenance']['postprocessing_uses_train_source_cv_only']=True
    torch.save(checkpoint,args.output/'model.pt');save(args.output/'model_metadata.json',{k:v for k,v in checkpoint.items() if k!='state_dict'})
    for name in ('frame_event_features.py','frame_event_tcn.py'):(args.output/name).write_bytes((PROJECT/'development/bounce'/name).read_bytes())
    (args.output/'source_select_frame_tcn_geometry.py').write_bytes(Path(__file__).read_bytes())
    save(args.output/'model_freeze.json',{'model_sha256':sha(args.output/'model.pt'),'metadata_sha256':sha(args.output/'model_metadata.json'),
        'selection_sha256':sha(args.output/'selection.json'),'validation_labels_opened':False})
    print(json.dumps(selected,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

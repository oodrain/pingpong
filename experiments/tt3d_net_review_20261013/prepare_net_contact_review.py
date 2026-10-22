"""Four unreviewed training net-zone windows, without assumed event labels."""
import argparse,json,sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(ROOT/'development/evaluation'))
from dataset_protocol import DEFAULT,load_protocol,select,sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(args):
    protocol=load_protocol();dataset=read(args.dataset/'dataset.json');automatic=read(args.dataset/'automatic_curved_cases.json')['videos']
    records={r['video_id']:r for r in select(protocol,list(automatic),'training')};options=[]
    for vid,items in automatic.items():
        scopes=dataset['scopes'].get(vid,[])
        for item in items:
            seed=item['seed_frame'];vector=item['vector'];proposal=item['proposal'];position=item['position']
            if any(lo-2<=seed<=hi+2 for lo,hi in scopes):continue
            if not position.get('available') or position.get('outside_projection') or proposal['tracking_spike_nearby']:continue
            if vector[-3] or vector[-2]<-.02 or vector[-1]>.06:continue
            lo=max(0,seed-12);hi=min(records[vid]['n_frames']-1,seed+12)
            prior=[b for a,b in scopes if b<seed];later=[a for a,b in scopes if a>seed]
            if prior:lo=max(lo,max(prior)+1)
            if later:hi=min(hi,min(later)-1)
            if seed-lo<3 or hi-seed<3:continue
            options.append(dict(item,review_scope=[lo,hi],net_top_distance_fraction=vector[-1]))
    options.sort(key=lambda item:item['net_top_distance_fraction']);chosen=[]
    for unique_group in (True,False):
        for item in options:
            if len(chosen)>=4:break
            if any(item['video_id']==c['video_id'] for c in chosen):continue
            if unique_group and any(item['source_group']==c['source_group'] for c in chosen):continue
            chosen.append(item)
    if len(chosen)!=4:raise ValueError('Insufficient diverse, unreviewed net-zone windows')
    args.output.mkdir(parents=True,exist_ok=False);cache=DEFAULT/'train_event_review_01/automatic_inputs';cases=[]
    for index,item in enumerate(chosen):
        record=records[item['video_id']];track=cache/(item['video_id']+'_tracks.jsonl')
        if sha(record['video_path'])!=record['video_sha256']:raise ValueError('Training video changed')
        cases.append({'case_id':item['case_id'],'video_id':item['video_id'],'seed_frame':item['seed_frame'],
            'source_group':record['source_group'],'split':'train','angle':'网投影假设 / 来源 '+record['source_group'],
            'video_path':record['video_path'],'video_sha256':record['video_sha256'],
            'tracks_path':str(track),'tracks_sha256':sha(track),'width':record['width'],'height':record['height'],
            'n_frames':record['n_frames'],'default_scope':item['review_scope'],'priority':index,
            'selection_reason':'unreviewed_automatic_net_zone_hypothesis_event_type_unknown'})
    snapshot=args.output/Path(__file__).name;snapshot.write_bytes(Path(__file__).read_bytes())
    inputs=[args.dataset/'dataset.json',args.dataset/'automatic_curved_cases.json',DEFAULT/'split_manifest.json',snapshot]
    pack={'schema_version':1,'purpose':'train_only_event_labels','split_id':protocol['split_id'],
        'sources':{str(p.resolve()):sha(p) for p in inputs},'cases':cases,'known_public_development_set':False,
        'human_labels_prefilled':False,'validation_videos_included':0,'collect_other_subtypes':True,
        'selection_uses_public_or_reserved_event_gt':False,'selection_uses_train_stage_event_gt':False,
        'frame_indexing':'zero_based_decoded_frames','selection_policy':'Four distinct train videos near automatically projected net hypotheses; trim scopes before completed windows.',
        'review_instructions':'Record all events. For other, cycle subtype net collision, net graze, normal flight or tracking error before adding the event. Net zone is a hypothesis, not a label.'}
    save(args.output/'review_pack.json',pack);save(args.output/'selection.json',{'selected':chosen,'pack_sha256':sha(args.output/'review_pack.json')})
    print(json.dumps({'cases':len(cases),'ids':[c['case_id'] for c in cases],'scopes':[c['default_scope'] for c in cases]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

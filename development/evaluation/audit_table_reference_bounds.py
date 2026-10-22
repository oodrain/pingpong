"""Audit table envelope consistency against existing TRAIN object boxes.

Boxes are evaluation-only inputs: no extraction/tuning uses them. A table box
does not establish four true corners, plane axes, or height.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,rows,write,write_rows,sha


def table_boxes(frame):
    boxes=[]
    for annotation in frame.get('annotations',[]):
        if annotation.get('label')!='table':continue
        for value in annotation.get('bbox_xyxy',[]):
            b=np.asarray(value,float)
            if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:
                raise ValueError('Malformed table reference box')
            boxes.append(b)
    return boxes


def envelope_metrics(corners,box):
    q=np.asarray(corners,float)
    if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError('Malformed corners')
    a=np.array([q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()])
    b=np.asarray(box,float)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    area_a=(a[2:]-a[:2]).prod();area_b=(b[2:]-b[:2]).prod()
    width,height=b[2:]-b[:2]
    return {'automatic_envelope':a.tolist(),'reference_box':b.tolist(),
            'bbox_iou':float(intersection/(area_a+area_b-intersection)),
            'reference_box_coverage':float(intersection/area_b),
            'max_side_error_normalized':float(np.max(np.abs(a-b)/[width,height,width,height])),
            'side_errors_px':(a-b).tolist()}


def audit(run,data):
    manifest=read(data/'C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json')['videos']
    metas={m['video_id']:m for m in manifest}
    samples=[s for s in rows(run/'geometry_samples.jsonl') if s['group']=='train']
    cache={};hashes={};records=[]
    for s in samples:
        vid=s['video_id'];m=metas[vid]
        if vid not in cache:
            path=data/'pp_train_data/pp_train_data/annotations'/vid[:2]/(vid[3:]+'.json')
            document=read(path);hashes[str(path.resolve())]=sha(path)
            if (document['video_id'],document['frame_width'],document['frame_height'])!=(vid,m['width'],m['height']):
                raise ValueError('Reference metadata mismatch')
            fs=document['frames'];cache[vid]={f['frame_id']:f for f in fs}
            if len(cache[vid])!=len(fs):raise ValueError('Duplicate reference frame')
        frame=cache[vid].get(s['frame_id'])
        boxes=table_boxes(frame) if frame else []
        status='reference_frame_missing' if frame is None else 'reference_box_missing' if not boxes else 'multiple_reference_boxes' if len(boxes)>1 else 'single_reference_box'
        r={'video_id':vid,'frame_id':s['frame_id'],'geometry_usable':s['usable'],
           'geometry_state':s['state'],'reference_status':status,'reference_box_count':len(boxes)}
        if status=='single_reference_box' and s['usable']:r.update(envelope_metrics(s['corners'],boxes[0]))
        records.append(r)
    return records,hashes


def aggregate(records):
    valid=[r for r in records if 'bbox_iou' in r]
    return {'samples':len(records),'usable_samples':sum(r['geometry_usable'] for r in records),
            'reference_status_counts':dict(Counter(r['reference_status'] for r in records)),
            'geometry_and_single_box_samples':len(valid),
            'mean_envelope_iou':float(np.mean([r['bbox_iou'] for r in valid])) if valid else None,
            'median_envelope_iou':float(np.median([r['bbox_iou'] for r in valid])) if valid else None,
            'large_side_disagreement_samples':sum(r['max_side_error_normalized']>.2 for r in valid),
            'low_envelope_iou_samples':sum(r['bbox_iou']<.7 for r in valid)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--geometry-run',type=Path,required=True)
    ap.add_argument('--previous-run',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    current,hashes=audit(args.geometry_run,args.data_root)
    summary={'current':aggregate(current),'corner_accuracy_verified':False,
             'reference_type':'provided training table bounding boxes; not four-corner truth',
             'disagreement_thresholds':'IoU<0.7 or envelope side error>20% of reference width/height; diagnostic, not adoption gates',
             'reference_used_in_inference':False,'annotation_sha256':hashes,
             'geometry_samples_sha256':sha(args.geometry_run/'geometry_samples.jsonl'),'source_sha256':sha(Path(__file__))}
    write_rows(out/'current_envelope_records.jsonl',current)
    if args.previous_run:
        previous,old_hashes=audit(args.previous_run,args.data_root)
        if hashes!=old_hashes:raise ValueError('Reference population changed')
        write_rows(out/'previous_envelope_records.jsonl',previous)
        old={(r['video_id'],r['frame_id']):r for r in previous}
        common=[];newly=[]
        for r in current:
            p=old[(r['video_id'],r['frame_id'])]
            if 'bbox_iou' in r and 'bbox_iou' in p:
                common.append(dict(video_id=r['video_id'],frame_id=r['frame_id'],previous_iou=p['bbox_iou'],current_iou=r['bbox_iou'],delta_iou=r['bbox_iou']-p['bbox_iou']))
            elif 'bbox_iou' in r:newly.append(r)
        summary['previous']=aggregate(previous)
        summary['common_samples']={'samples':len(common),'mean_iou_delta':float(np.mean([r['delta_iou'] for r in common])) if common else None}
        summary['newly_available_samples']=aggregate(newly)
        write_rows(out/'common_sample_changes.jsonl',common)
        write_rows(out/'newly_available_envelopes.jsonl',newly)
    # Worst cases selected only by box disagreement, not event labels.
    manifest=read(args.data_root/'C_train154_20261011_v3/C_train154_20261011_v3/selected_manifest.json')['videos']
    metas={m['video_id']:m for m in manifest}
    samples={(s['video_id'],s['frame_id']):s for s in rows(args.geometry_run/'geometry_samples.jsonl') if s['group']=='train'}
    (out/'worst_overlays').mkdir()
    worst=sorted((r for r in current if 'bbox_iou' in r),key=lambda r:r['bbox_iou'])[:24]
    for r in worst:
        v=r['video_id'];m=metas[v];path=args.data_root/'pp_train_data/pp_train_data/rallies_videos'/m['owner']/v[:2]/(v[3:]+'.mp4')
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame_id']);ok,im=cap.read();cap.release()
        if not ok:raise ValueError('Cannot decode audit frame')
        s=samples[(v,r['frame_id'])];cv2.polylines(im,[np.array(s['corners']).round().astype(np.int32)],True,(0,230,255),2)
        x,y,x2,y2=map(round,r['reference_box']);cv2.rectangle(im,(x,y),(x2,y2),(255,200,0),2)
        cv2.putText(im,f"{v} frame={r['frame_id']} bboxIoU={r['bbox_iou']:.3f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.imwrite(str(out/'worst_overlays'/(v+'_'+str(r['frame_id'])+'.png')),im)
    write(out/'summary.json',summary);write_rows(out/'worst_cases.jsonl',worst)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('annotation_sha256',)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

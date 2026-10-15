"""Separate contact consistency from independently annotated corner accuracy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from development.detection.auto_table_geometry import AutoGeometryTimeline
from development.bounce.table_geometry import GeometryTimeline


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def corner_error(automatic,reference):
    a,b=np.asarray(automatic,float),np.asarray(reference,float)
    # Only boundary-cycle/reflection ambiguity is allowed, not arbitrary permutations.
    assignments=[np.roll(order,k,axis=0) for order in (b,b[::-1]) for k in range(4)]
    distances=min((np.linalg.norm(a-order,axis=1) for order in assignments),key=lambda d:d.mean())
    return {'mean_corner_error_px':float(distances.mean()),'maximum_corner_error_px':float(distances.max())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry-run',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--reference-dir',type=Path)
    args=ap.parse_args()
    out=args.geometry_run/'geometry_validation'
    out.mkdir(parents=True,exist_ok=False);(out/'contact_overlays').mkdir()
    data=args.data_root/'participant/participant/pingpang/public_data'
    manifest=read(data/'manifest.json')['videos'];gt=rows(data/'gt_public.jsonl')
    samples=[s for s in rows(args.geometry_run/'geometry_samples.jsonl') if s['group']=='public']
    contacts=[];comparisons=[];missing=[];reference_hashes={}
    for meta in manifest:
        vid=meta['video_id'];ss=[s for s in samples if s['video_id']==vid]
        timeline=AutoGeometryTimeline(ss)
        reference_path=args.reference_dir/(vid+'.json') if args.reference_dir else None
        if reference_path is None or not reference_path.is_file():
            missing.append(vid)
        else:
            document=read(reference_path);manual=GeometryTimeline(document)
            reference_hashes[str(reference_path.resolve())]=hashlib.sha256(reference_path.read_bytes()).hexdigest()
            for s in ss:
                plane=manual.lookup(s['frame_id'])
                if not s['usable'] or plane is None:continue
                sid=manual.interval_id(s['frame_id'])
                segment=next(seg for index,seg in enumerate(document['segments']) if str(seg.get('id',index))==sid)
                comparisons.append(dict(video_id=vid,frame_id=s['frame_id'],
                                        **corner_error(s['corners'],segment['corners'])))
        capture=cv2.VideoCapture(str(data/meta['path']))
        try:
            for point in [g for g in gt if g['video_id']==vid and g['valid']]:
                frame=point['frame_id'];plane=timeline.lookup(frame);sample=timeline.sample(frame)
                signed=None;outside=None
                if plane:
                    signed=float(cv2.pointPolygonTest(np.array(sample['corners'],np.float32),(point['x'],point['y']),True))
                    outside=plane.outside_distance(point['x'],point['y'])
                row=dict(point,geometry_available=plane is not None,geometry_frame=sample['frame_id'] if sample else None,
                         signed_polygon_distance_px=signed,conservative_outside_mm=outside,
                         centre_inside_image_polygon=signed>=0 if signed is not None else None)
                contacts.append(row)
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
                if not ok:raise ValueError('Cannot decode contact frame')
                if plane:cv2.polylines(image,[np.array(sample['corners']).round().astype(np.int32)],True,(0,230,255),3)
                cv2.circle(image,(round(point['x']),round(point['y'])),9,(0,255,0),3)
                cv2.putText(image,f"{vid} GT frame={frame} geometry={'available' if plane else 'unknown'}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
                cv2.imwrite(str(out/'contact_overlays'/(vid+'_'+str(frame)+'.png')),image)
        finally:capture.release()
    (out/'contact_consistency.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in contacts),encoding='utf-8')
    write(out/'corner_comparisons.json',comparisons)
    known=[r for r in contacts if r['geometry_available']]
    counts={'valid_contacts':len(contacts),'contacts_with_geometry':len(known),
            'centres_inside_polygon':sum(r['centre_inside_image_polygon'] for r in known),
            'centres_outside_polygon':sum(not r['centre_inside_image_polygon'] for r in known),
            'contacts_beyond_900mm_conservative':sum(r['conservative_outside_mm']>900 for r in known)}
    write(out/'summary.json',{'status':'corner_reference_compared' if comparisons else 'contact_consistency_only',
          'contact_consistency':counts,'reference_files':len(reference_hashes),'reference_hashes':reference_hashes,
          'missing_reference_video_ids':missing,'independent_corner_comparisons':len(comparisons),
          'mean_corner_error_px':float(np.mean([r['mean_corner_error_px'] for r in comparisons])) if comparisons else None,
          'note':'Valid GT ball-centre containment is a consistency diagnostic, not corner GT; ball radius/projection can move edge contacts outside.'})
    print(json.dumps({'contact_consistency':counts,'corner_comparisons':len(comparisons),'missing_references':len(missing)},indent=2))


if __name__=='__main__':main()

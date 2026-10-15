"""Independent background motion evidence and temporal-window diagnostics.

Background transforms diagnose image motion only: they are never applied to ball
pixels or promoted to camera calibration. No human event labels choose windows.
"""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
from fit_contact_pilot import sha,search,design,project

def background_features(frame,corners):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);mask=np.full(gray.shape,255,np.uint8)
    table=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(table,np.rint(corners).astype(np.int32),255)
    table=cv2.dilate(table,np.ones((81,81),np.uint8));mask[table>0]=0
    # Ignore broadcast score overlays at the outermost image border.
    mask[:25]=0;mask[-25:]=0;mask[:,:25]=0;mask[:,-25:]=0
    key,desc=cv2.SIFT_create(nfeatures=1800).detectAndCompute(gray,mask)
    return np.array([k.pt for k in key],np.float32),desc

def compare_background(reference,target,size,corners):
    p,d=reference;q,e=target
    if d is None or e is None or min(len(d),len(e))<20:return {'status':'insufficient_features'}
    pairs=cv2.BFMatcher().knnMatch(d,e,k=2)
    pairs=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    if len(pairs)<20:return {'status':'insufficient_matches','matches':len(pairs)}
    src=np.array([p[m.queryIdx] for m in pairs]);dst=np.array([q[m.trainIdx] for m in pairs])
    transform,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=3000,confidence=.999)
    if transform is None:return {'status':'no_consensus'}
    keep=mask.ravel()>0;src=src[keep];dst=dst[keep]
    coverage=cv2.contourArea(cv2.convexHull(src))/(size[0]*size[1])
    error=np.linalg.norm(cv2.transform(src[None],transform)[0]-dst,axis=1)
    motion=np.linalg.norm(cv2.transform(np.asarray(corners,np.float32)[None],transform)[0]-corners,axis=1)
    strong=len(src)>=30 and coverage>=.3 and np.percentile(error,95)<=1.5
    return {'status':('background_stable' if motion.max()<=3 else 'background_motion') if strong else 'weak_background_support',
            'inliers':len(src),'coverage':float(coverage),'p95_error_px':float(np.percentile(error,95)),
            'max_table_location_shift_px':float(motion.max()),'image_similarity_transform':transform.tolist()}

def contact_sheet(frames,tracks,seed,path):
    ids=[max(0,min(len(frames)-1,seed+d)) for d in (-8,-5,-2,-1,0,1,4,8)]
    sheet=np.zeros((2*330,4*360,3),np.uint8)
    for n,fid in enumerate(ids):
        frame=frames[fid].copy();row=tracks[fid];x0=(n%4)*360;y0=(n//4)*330
        if row['detected']:
            x,y=round(row['x']),round(row['y']);cv2.circle(frame,(x,y),9,(0,255,0),2)
            raw=frames[fid];left=max(0,min(raw.shape[1]-160,x-80));top=max(0,min(raw.shape[0]-100,y-50))
            sheet[y0+215:y0+315,x0+100:x0+260]=raw[top:top+100,left:left+160]
        sheet[y0+10:y0+212,x0:x0+360]=cv2.resize(frame,(360,202))
        cv2.putText(sheet,f'frame {fid}',(x0+5,y0+325),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.imencode('.jpg',sheet)[1].tofile(str(path))

def run(source,inputs,out):
    data=json.loads(source.read_text(encoding='utf-8'));out.mkdir(parents=True,exist_ok=False);cases=[]
    for video in sorted(set(c['video_id'] for c in data['cases'])):
        folder=inputs/'videos'/video;meta=json.loads((folder/'input_summary.json').read_text())
        for name,digest in meta['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Changed input '+name)
        if sha(meta['video_path'])!=meta['video_sha256']:raise ValueError('Changed video')
        tracks=[json.loads(s) for s in (folder/'timed_tracks.jsonl').read_text().splitlines()]
        cap=cv2.VideoCapture(meta['video_path']);frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release();cache={};ts=np.array([r['timestamp_s'] for r in tracks])
        for c in (c for c in data['cases'] if c['video_id']==video):
            seed=c['seed_frame_id'];pose=c['camera'];corners=np.array(pose['projected_corners'])
            # Each case uses its calibrated reference polygon for all matching masks.
            anchor=pose['frame_id'];reference=background_features(frames[anchor],corners)
            checks=[]
            for fid in range(max(0,seed-8),min(len(frames),seed+9)):
                key=(anchor,fid)
                if key not in cache:cache[key]=background_features(frames[fid],corners)
                checks.append({'frame_id':fid,**compare_background(reference,cache[key],meta['image_size'],corners)})
            lo=float(np.interp(seed-2.5,np.arange(len(ts)),ts));hi=float(np.interp(seed+2.5,np.arange(len(ts)),ts))
            windows=[]
            for before,after in ((4,4),(4,6),(6,4),(6,6),(8,8)):
                rows=[r for r in tracks if seed-before<=r['frame_id']<=seed+after and r['eligible_high_confidence']]
                fit=search(rows,pose,lo,hi,constrained=True)
                windows.append({'before':before,'after':after,'fit_frame_ids':[r['frame_id'] for r in rows],
                                'fractional_frame':float(np.interp(fit['tau'],ts,np.arange(len(ts)))) if fit else None,
                                'rmse_px':fit['rmse_px'] if fit else None,'active_constraints':fit['active_constraint_indices'] if fit else None,
                                'velocity_before_m_s':fit['parameters'][2:5] if fit else None,'velocity_after_m_s':fit['parameters'][5:8] if fit else None})
            image=out/(c['case_id']+'_frames.jpg');contact_sheet(frames,tracks,seed,image)
            status='background_stable' if all(x['status']=='background_stable' for x in checks) else ('background_motion' if any(x['status']=='background_motion' for x in checks) else 'unverified')
            result={'case_id':c['case_id'],'background_status':status,'anchor_frame':anchor,'checks':checks,'windows':windows,'contact_sheet':str(image)}
            cases.append(result);print(c['case_id'],status,[(w['before'],w['after'],round(w['rmse_px'],2) if w['rmse_px'] else None) for w in windows],flush=True)
    report={'source_sha256':sha(source),'script_sha256':sha(__file__),'optimizer_sha256':sha(Path(__file__).with_name('fit_contact_pilot.py')),
            'human_labels_used':False,'camera_parameters_modified':False,'background_stability_is_not_table_pose_validation':True,
            'windows_are_diagnostics_not_selected_for_deployment':True,'cases':cases}
    (out/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def short_probes(source,inputs,out):
    data=json.loads(source.read_text(encoding='utf-8'));results=[]
    out.mkdir(parents=True,exist_ok=True)
    target=out/'short_window_probes.json'
    if target.exists():raise ValueError('Output already exists')
    for c in data['cases']:
        folder=inputs/'videos'/c['video_id'];meta=json.loads((folder/'input_summary.json').read_text())
        path=folder/'timed_tracks.jsonl'
        if sha(path)!=meta['files']['timed_tracks.jsonl']:raise ValueError('Changed tracks')
        rows=[json.loads(s) for s in path.read_text().splitlines()];ts=np.array([r['timestamp_s'] for r in rows]);fid=c['seed_frame_id']
        lo=float(np.interp(fid-2.5,np.arange(len(ts)),ts));hi=float(np.interp(fid+2.5,np.arange(len(ts)),ts));variants=[]
        for before,after in ((3,3),(4,3),(3,4)):
            admitted=[r for r in rows if fid-before<=r['frame_id']<=fid+after and r['eligible_high_confidence']]
            fit=search(admitted,c['camera'],lo,hi,constrained=True)
            if fit:fit['fractional_frame']=float(np.interp(fit['tau'],ts,np.arange(len(ts))))
            variants.append({'before':before,'after':after,'frame_ids':[r['frame_id'] for r in admitted],'fit':fit})
        extensions=[];central=variants[0]['fit']
        if central is not None:
            for side,frame_id in (('before',fid-4),('after',fid+4)):
                if 0<=frame_id<len(rows) and rows[frame_id]['eligible_high_confidence']:
                    row=rows[frame_id];a,b=design([row['timestamp_s']],central['tau'])
                    xyz=b+np.einsum('nij,j->ni',a,central['parameters']);xy,depth=project(xyz,c['camera'])
                    error=float(np.linalg.norm(xy[0]-[row['x'],row['y']]))
                    extensions.append({'side':side,'frame_id':frame_id,'prediction_error_px':error,
                                       'extension_inconsistent':bool(central['rmse_px']<=3 and (depth[0]<=0 or error>6)),
                                       'interpretation':'window inconsistency cue, not proof of racket contact'})
        results.append({'case_id':c['case_id'],'variants':variants,'one_frame_extension_checks':extensions})
    target.write_text(json.dumps({'source_sha256':sha(source),'script_sha256':sha(__file__),'optimizer_sha256':sha(Path(__file__).with_name('fit_contact_pilot.py')),
                                 'human_labels_used':False,'not_automatically_selected':True,'cases':results},indent=2,allow_nan=False),encoding='utf-8')
    print('Saved short-window probes for',len(results),'cases',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--short-only',action='store_true')
    a=p.parse_args();(short_probes if a.short_only else run)(a.source,a.inputs,a.output)

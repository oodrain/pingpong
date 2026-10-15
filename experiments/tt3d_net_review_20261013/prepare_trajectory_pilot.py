"""Freeze reviewed cameras and verified, already aligned detector tracks for trials."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np

ROOT=Path(os.environ.get('AI_PINGPONG_DATA_ROOT', str(Path(__file__).resolve().parents[2])))
CHECKOUT=ROOT/'worktrees/table-geometry-v2'
sys.path.insert(0,str(ROOT/'development/annotation'))
from net_geometry_review import NetReviewStore,backend,sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    store=NetReviewStore(args.pack);review_bytes=store.path.read_bytes()
    if len(store.latest)!=len(store.cases):raise ValueError('Human review is incomplete')
    export=next(p for p in sorted((store.pack/'calibration_exports').glob('*'),reverse=True) if (p/'summary.json').is_file() and (p/'cameras.json').is_file())
    summary=read(export/'summary.json')
    if summary['net_reference_sha256']!=hashlib.sha256(review_bytes).hexdigest() or summary['manifest_sha256']!=store.source.manifest_hash:raise ValueError('Calibration export is stale')
    prior,_,parent=store.prior_reviews()
    if parent and summary['parent_review_source']['net_reference_sha256']!=parent['reference_sha256']:raise ValueError('Parent review changed')
    accepted=[c for c in read(export/'cameras.json') if c['result']['usable']]
    dataset=ROOT/'C_train154_20261011_v3/C_train154_20261011_v3'
    metas={m['video_id']:m for m in read(dataset/'selected_manifest.json')['videos']}
    metadata=read(dataset/'export_metadata.json')
    # Export metadata hashes the exporter's *input* manifest, not selected_manifest.json.
    # Bind the output delivery to the previously frozen geometry run instead.
    geometry_summary=read(Path(store.source.manifest['geometry_source'])/'summary.json')
    if sha(dataset/'selected_manifest.json')!=geometry_summary['input_sha256'].get(str((dataset/'selected_manifest.json').resolve())):raise ValueError('Detector delivery manifest changed')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'sources').mkdir();(args.output/'videos').mkdir()
    for file in ['summary.json','cameras.json','net_references_snapshot.jsonl','parent_net_references_snapshot.jsonl','manifest_snapshot.json','parent_manifest_snapshot.json']:
        if (export/file).is_file():shutil.copy2(export/file,args.output/'sources'/file)
    for file in ['selected_manifest.json','export_metadata.json']:
        shutil.copy2(dataset/file,args.output/'sources'/file)
    module=backend()
    from development.bounce.ge_rule_v2 import replay
    reports=[]
    for camera in accepted:
        if camera['group']!='train':raise ValueError('This pilot preparer expects the verified train delivery')
        vid=camera['video_id'];meta=metas[vid];result=camera['result']
        video=ROOT/'pp_train_data/pp_train_data/rallies_videos'/meta['owner']/vid[:2]/(vid[3:]+'.mp4')
        if sha(video)!=meta['video_sha256']:raise ValueError('Video changed')
        observations=rows(dataset/'trajectories'/(vid+'.jsonl'));raw=rows(dataset/'raw'/(vid+'.jsonl'))
        if sha(dataset/'raw'/(vid+'.jsonl'))!=geometry_summary['input_sha256'].get(str((dataset/'raw'/(vid+'.jsonl')).resolve())):raise ValueError('Frozen detector raw cache changed')
        n=meta['n_frames']
        if len(observations)!=n or [r['frame_id'] for r in observations]!=list(range(n)):raise ValueError('Aligned trajectory coverage mismatch')
        if len(raw)!=n or [r['input_frame_id'] for r in raw]!=list(range(n)):raise ValueError('Raw coverage mismatch')
        for r in observations[:-1]:
            if r['input_frame_id']!=r['frame_id']+1:raise ValueError('Unexpected detector delay')
            original=raw[r['input_frame_id']]
            if original['target_frame_id']!=r['frame_id'] or any(original.get(k)!=r.get(k) for k in ('detected','x','y','confidence')):raise ValueError('Aligned/raw detector mismatch')
        if observations[-1]['missing_reason']!='tail_not_inferred' or observations[-1]['input_frame_id'] is not None:raise ValueError('Unexpected tail semantics')
        cap=cv2.VideoCapture(str(video));timestamps=[]
        try:
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            while True:
                ok,image=cap.read()
                if not ok:break
                if image.shape[:2]!=(meta['height'],meta['width']):raise ValueError('Decoded image size mismatch')
                timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.)
        finally:cap.release()
        if len(timestamps)!=n or abs(fps-meta['measured_fps'])>1e-6:raise ValueError('Decoded timebase/frame count mismatch')
        times=np.asarray(timestamps)
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Non-monotonic decoded timestamps')
        times-=times[0]
        prepared=[]
        for row,t in zip(observations,times):
            detected=row['detected'];valid=detected and all(row.get(k) is not None and np.isfinite(row[k]) for k in ('x','y','confidence'))
            if detected and not valid:raise ValueError('Invalid detector observation')
            in_image=valid and 0<=row['x']<meta['width'] and 0<=row['y']<meta['height']
            prepared.append(dict(row,timestamp_s=float(t),in_image=bool(in_image),eligible_high_confidence=bool(in_image and row['confidence']>=.75)))
        seeds,decisions=replay(raw,mode='v0');contexts=[]
        for seed in seeds:
            f=seed['frame_id'];before=[r for r in prepared[max(0,f-8):f] if r['eligible_high_confidence']];after=[r for r in prepared[f+1:min(n,f+9)] if r['eligible_high_confidence']]
            contexts.append(dict(seed_frame_id=f,seed_source='original_v0_detector_rule_not_human_ground_truth',timestamp_s=float(times[f]),before_frame_ids=[r['frame_id'] for r in before],after_frame_ids=[r['frame_id'] for r in after],has_three_points_each_side=len(before)>=3 and len(after)>=3))
        best=min(result['poses'],key=lambda p:p['reprojection_rmse_px'])
        object_points=module.net_object_points(best['edge01_axis'])[:4]
        baseline=cv2.projectPoints(object_points,np.asarray(best['rvec']),np.asarray(best['tvec_m']),np.asarray(best['K']),np.zeros(5))[0].reshape(4,2)
        drift=[]
        for pose in result['poses']:
            p=cv2.projectPoints(object_points,np.asarray(pose['rvec']),np.asarray(pose['tvec_m']),np.asarray(pose['K']),np.zeros(5))[0].reshape(4,2)
            drift.append(dict(frame_id=pose['frame_id'],maximum_corner_projection_change_px=float(np.linalg.norm(p-baseline,axis=1).max())))
        directory=args.output/'videos'/vid;directory.mkdir()
        shutil.copy2(dataset/'trajectories'/(vid+'.jsonl'),directory/'aligned_tracks_source.jsonl');shutil.copy2(dataset/'raw'/(vid+'.jsonl'),directory/'raw_source.jsonl')
        (directory/'timed_tracks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in prepared),encoding='utf-8')
        write(directory/'camera.json',dict(result,representative_pose=best,reviewed_pose_projection_change=drift,full_video_static_camera_verified=False))
        write(directory/'candidate_contexts.json',contexts)
        report=dict(video_id=vid,video_path=str(video),video_sha256=sha(video),image_size=[meta['width'],meta['height']],frames=n,measured_fps=fps,timestamps_source='sequential OpenCV decoder presentation timestamps, seconds normalized to first frame',median_frame_interval_s=float(np.median(np.diff(times))),
            decoded_span_s=float(times[-1]),nominal_fps=meta['fps'],detected_points=sum(r['detected'] for r in prepared),high_confidence_points=sum(r['eligible_high_confidence'] for r in prepared),missing_reasons=dict(Counter(r['missing_reason'] for r in prepared if not r['detected'])),
            original_rule_candidates=len(contexts),candidates_with_three_points_each_side=sum(c['has_three_points_each_side'] for c in contexts),camera_rmse_px=result['reprojection_rmse_px'],representative_camera_frame=best['frame_id'],reviewed_camera_frames=result['frames'],
            maximum_reviewed_projection_change_px=max(d['maximum_corner_projection_change_px'] for d in drift),camera_static_verification_pending=True,player_end_direction_verified=False,detector_tracks_are_human_ground_truth=False,
            files={f.name:sha(f) for f in directory.iterdir() if f.is_file()})
        write(directory/'input_summary.json',report);reports.append(report)
    if store.path.read_bytes()!=review_bytes:raise ValueError('Review changed during preparation')
    report=dict(status='verified_inputs_prepared_for_offline_trajectory_trials',created_utc=datetime.now(timezone.utc).isoformat(),videos=len(reports),inputs=reports,source_calibration_export=str(export),annotation_sha256=sha(store.path),script_sha256=sha(__file__),
        frame_alignment='trajectories.frame_id already zero-based; no additional -1 correction',low_confidence_and_missing_frames_preserved=True,event_labels_consumed=False,trajectory_fit_completed=False,subframe_contact_time_estimated=False,desktop_event_flow_replaced=False,
        next_step='Check camera stability across candidate windows, then fit pre/post trajectories and trial subframe contact times. No additional whole-pack geometry labeling requested.')
    write(args.output/'summary.json',report)
    print(json.dumps(dict(output=str(args.output),videos=len(reports),inputs=[dict(video_id=r['video_id'],points=r['high_confidence_points'],candidates=r['original_rule_candidates'],contexts=r['candidates_with_three_points_each_side'],camera_projection_change_px=r['maximum_reviewed_projection_change_px']) for r in reports]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

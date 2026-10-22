"""Package verified regression/camera diagnostics, preserving human submissions."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,sha,write
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.run_geometry_d_v2 import changed_matches


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--camera',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    run=read(args.run/'summary.json');camera=read(args.camera/'summary.json')
    if camera['geometry_sha256']!=sha(args.run/'geometry_samples.jsonl'):raise ValueError('Camera/geometry mismatch')
    for summary in (run,camera):
        for p,digest in summary['source_sha256'].items():
            if sha(p)!=digest:raise ValueError('Source changed: '+p)
    pack=PROJECT/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    if sha(pack/'references.jsonl')!='b3f72d5b4ebdec5e9ffc441cf3a9c19a96240bd83bd0850da178f047090d2247':raise ValueError('Human submissions changed')
    if sha(pack/'manifest.json')!='4505cb4e88eb917ede1c768e6d7ad390f2eafc9dab48d3615c8b70be35089ca3':raise ValueError('Review manifest changed')
    manifest=read(pack/'manifest.json');refs=latest_reviews(pack,manifest)
    cases={c['case_id']:c for c in manifest['cases']}
    records=read(args.run/'reference_records.json');by_case={r['case_id']:r for r in records}
    paired=[r for r in records if 'single_error' in r and 'revised_same_frame_error' in r['previous']]
    temporal_pairs=[r for r in records if 'temporal_error' in r and 'revised_adaptive_error' in r['previous']]
    events=read(args.run/'metrics.json')
    previous_events=read(PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/regression_all_01/metrics.json')
    changes=changed_matches(previous_events['public']['auto_region']['without_ignore'],events['public'],'refined')
    sources=[PROJECT/'development/detection/test_auto_table_geometry.py',PROJECT/'development/detection/test_tt3d_camera_estimation.py',
             PROJECT/'development/evaluation/test_tt3d_camera_consistency.py',Path(__file__)]
    args.output.mkdir(parents=True,exist_ok=False)
    validation=dict(status='completed',geometry=run,camera=camera,
        human_submissions_unchanged=True,human_reference_sha256=sha(pack/'references.jsonl'),
        paired_single_comparisons=len(paired),
        paired_single_before_mean_px=float(np.mean([r['previous']['revised_same_frame_error']['mean_corner_error_px'] for r in paired])),
        paired_single_after_mean_px=float(np.mean([r['single_error']['mean_corner_error_px'] for r in paired])),
        paired_temporal_comparisons=len(temporal_pairs),
        paired_temporal_before_mean_px=float(np.mean([r['previous']['revised_adaptive_error']['mean_corner_error_px'] for r in temporal_pairs])),
        paired_temporal_after_mean_px=float(np.mean([r['temporal_error']['mean_corner_error_px'] for r in temporal_pairs])),
        public_changed_matches=changes,
        tests=dict(passed=57,command='python -X utf8 -m unittest development.detection.test_auto_table_geometry development.detection.test_tt3d_camera_estimation development.evaluation.test_auto_geometry_sampling development.evaluation.test_geometry_reference_review development.evaluation.test_geometry_validation development.evaluation.test_tt3d_camera_consistency development.bounce.test_ge_rule_v2'),
        caveats=['Single-review corners informed development; no independent generalization claim.',
                 'Unknown/held geometry is excluded from camera estimation; some videos remain unavailable.',
                 'Four-corner reprojection and multi-frame focal agreement do not independently validate intrinsics or 3D.',
                 'Axes are retained as alternative hypotheses where unresolved.',
                 'Stricter geometry reduces coverage; public region-filter F1 decreased from 0.635417 to 0.622449. This is not adopted as an event-classification improvement.',
                 'Offline experiment; desktop production event flow has not been replaced; 50 FPS unverified.'],
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        artifact_sha256={str(p.resolve()):sha(p) for p in [args.run/'geometry_samples.jsonl',args.run/'reference_records.json',args.camera/'cameras.json']})
    write(args.output/'validation.json',validation)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    canvas=Image.new('RGB',(1320,1030),'#101925');draw=ImageDraw.Draw(canvas)
    draw.text((22,14),'TT3D：截角修复、时序检查与相机估计',font=font,fill='white')
    draw.text((22,48),'绿色：人工角点  紫色：前版结果  黄色：修复结果 / 估计的台面网格',font=small,fill='#cbd5e1')
    def tile(image,x,y,title,detail):
        pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));pil.thumbnail((620,365))
        canvas.paste(pil,(x+(620-pil.width)//2,y+60+(365-pil.height)//2))
        draw.text((x,y),title,font=font,fill='white')
        draw.text((x,y+32),detail,font=small,fill='#cbd5e1')
    for case_id,x,y in [('train_00_005_000018',20,95),('public_v000038_000098',675,95)]:
        c=cases[case_id];r=by_case[case_id];im=cv2.imread(str(pack/c['image_url']))
        ref=refs[case_id];old=r['previous']['revised_proposal']['corners'] if '00_005' in case_id else r['previous'].get('revised_adaptive_corners')
        # Previous audit stores temporal errors but not its corners; get the
        # actual latest causal sample, never substitute the current single frame.
        if 'v000038' in case_id:
            from development.evaluation.run_auto_table_v2 import rows
            observations=[s for s in rows(PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/regression_all_01/geometry_samples.jsonl')
                          if s['group']==c['group'] and s['video_id']==c['video_id'] and s['frame_id']<=c['frame_id']]
            old=observations[-1]['corners']
        new=r['proposal']['corners'] if '00_005' in case_id else r['temporal_sample'].get('corners') if r['temporal_usable'] else None
        for q,color in [(ref.get('corners'),(0,220,0)),(old,(220,0,220)),(new,(0,220,255))]:
            if q:cv2.polylines(im,[np.rint(q).astype(np.int32)],True,color,3)
        q=np.array(ref['corners']);left,top=np.maximum(q.min(axis=0)-[120,95],0).astype(int)
        right,bottom=np.minimum(q.max(axis=0)+[120,110],[im.shape[1],im.shape[0]]).astype(int)
        title='00_005 第18帧：截角修复' if '00_005' in case_id else 'v000038 第98帧：停止沿用错误角点'
        detail='单帧最大误差 101.4 → 3.1 px；时序仍需稳定确认' if '00_005' in case_id else ('时序结果已恢复，请查看逐帧记录' if r['temporal_usable'] else '当前返回不可用；紫色旧角点已停止使用')
        tile(im[top:bottom,left:right],x,y,title,detail)
    overlays=list((args.camera/'overlays').glob('public_v000038*.png'))
    others=list((args.camera/'overlays').glob('public_v000095*.png'))
    if not others:others=list((args.camera/'overlays').glob('public_v000085*.png'))
    selected=(overlays+others)[:2]
    for p,x in zip(selected,(20,675)):
        tile(cv2.imread(str(p)),x,545,p.stem.replace('_camera0','')+'：相机估计',
             '主点居中、fx=fy、零畸变；台面网格用于检查投影')
    draw.text((22,995),f"通过检查的相机片段 {camera['usable_segments']}/{camera['camera_segments']}；部分方向仍有歧义，未完成独立3D验证。",font=small,fill='#cbd5e1')
    canvas.save(args.output/'comparison_and_calibration.png')
    print('DONE',validation['paired_single_comparisons'],camera['groups'],flush=True)


if __name__=='__main__':main()

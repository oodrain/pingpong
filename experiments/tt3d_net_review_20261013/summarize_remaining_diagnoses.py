"""Error-focused diagnosis only; preserves human labels and classifier thresholds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.wrist_evidence import iou
from fit_contact_pilot import sha

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def run(pilot,out):
    temporal_path=pilot/'temporal_contact_01/temporal_evidence.json';evaluation_path=pilot/'temporal_contact_01/evaluation.json'
    roi_path=out/'roi_evidence.json';roi_eval_path=out/'evaluation.json'
    temporal=read(temporal_path);baseline=read(evaluation_path);roi=read(roi_path);score=read(roi_eval_path)
    old={c['video_id']+'_'+str(c['seed_frame']):c for c in temporal['cases']}
    new={c['video_id']+'_'+str(c['seed_frame']):c for c in roi['cases']};labels={c['case_id']:c for c in score['cases']}
    diagnoses=[];sources={str(p):sha(p) for p in (temporal_path,evaluation_path,roi_path,roi_eval_path)}
    for row in baseline['cases']:
        if row['human_label']!='racket_contact' or row['new_flag']:continue
        name=row['case_id'];c=old[name];r=new[name]
        observations=[(f['frame_id'],o) for f in c['frames'] for o in f['observations']]
        anchor_fid,anchor=min(observations,key=lambda item:item[1]['distance_in_forearms'])
        full_path=pilot/'temporal_contact_01'/f"{c['video_id']}_poses.json";sources[str(full_path)]=sha(full_path)
        full=read(full_path)['frames'];per_frame=[]
        elbow,wrist=(8,10) if anchor['hand']=='right' else (7,9)
        for frame in c['frames']:
            fid=frame['frame_id'];people=full[str(fid)]
            actor=max(people,key=lambda p:iou(p['box'],anchor['person_box']))
            overlap=iou(actor['box'],anchor['person_box'])
            raw=[actor['keypoint_scores'][elbow],actor['keypoint_scores'][wrist]]
            arm=float(np.linalg.norm(np.array(actor['keypoints'][elbow][:2])-actor['keypoints'][wrist][:2]))
            per_frame.append({'frame_id':fid,'person_iou_to_anchor':overlap,'anatomical_hand':anchor['hand'],
                'elbow_score_raw':raw[0],'wrist_score_raw':raw[1],'forearm_px':arm,
                'raw_score_gate_passed':bool(min(raw)>=2),'actor_match_supported':overlap>=.3})
        phase=c['phase_diagnostic'];d=r['direction']
        if c['visit']['supported'] and not d['strong_image_reversal']:
            kind='direction_assumption';summary='手腕证据有效，二维方向条件未通过'
            detail='不同拟合给出不同方向，且后段残差未通过' if not phase.get('fit_quality_supported') else '两种方向估计均未反向；真实击球不满足该条件'
            next_step='需视角与时序联合判断，不能要求所有击球都二维反向'
        elif any(not f['raw_score_gate_passed'] for f in per_frame):
            kind='weak_arm_keypoints';summary='关键邻帧右臂关键点质量不足'
            failed=[f['frame_id'] for f in per_frame if not f['raw_score_gate_passed']]
            detail='未通过热图分值门槛的帧：'+', '.join(map(str,failed))
            next_step='检查局部人物姿态、遮挡与跨帧关联'
        else:
            kind='wrist_localization_and_scale';summary='手腕定位与前臂归一化尺度不稳定'
            lengths=[f['forearm_px'] for f in per_frame]
            detail=f"前臂估计范围 {min(lengths):.1f}～{max(lengths):.1f} px；原最近距离比仍超门槛"
            next_step='检查局部手腕定位；不通过放宽距离阈值补漏'
        diagnoses.append({'case_id':name,'video_id':c['video_id'],'seed_frame':c['seed_frame'],'root_cause':kind,
            'summary':summary,'detail':detail,'next_step':next_step,'original_closest_anchor_frame':anchor_fid,
            'original_actor_keypoints':per_frame,'old_direction_cosine':d['image_direction_cosine'],
            'quadratic_direction_cosine':phase.get('image_direction_cosine_at_seed'),'quadratic_fit_quality_supported':phase.get('fit_quality_supported'),
            'roi_visit_supported':r['roi_visit']['supported'],'roi_hit_flag':labels[name]['roi_flag'],
            'human_label_used_for_error_selection':True,'not_independent_wrist_ground_truth':True})
    report={'sources':sources,'runner_sha256':sha(__file__),'diagnoses':diagnoses,'baseline_flags':score['baseline_flags'],
        'roi_flags':score['roi_flags'],'roi_table_pool':score['roi_table_pool'],'crop_inferences':roi['crop_inferences'],
        'matched_crop_actors':roi['matched_crop_actors'],'all_31_candidates_probed':True,'thresholds_relaxed':False,
        'production_events_modified':False,'scope':'Error diagnosis and retrospective development comparison; not independent evaluation.'}
    (out/'root_cause_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'source_summarize_remaining_diagnoses.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'diagnoses':[(r['case_id'],r['root_cause'],r['roi_hit_flag']) for r in diagnoses],
        'score':score['roi_flags'],'table_pool':score['roi_table_pool']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.pilot.resolve(),a.output.resolve())

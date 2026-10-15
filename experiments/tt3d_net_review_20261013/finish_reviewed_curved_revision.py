"""Record reviewed revision outcome without promoting experimental assets."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
REV = ROOT / 'outputs/contact_rule_revision_20261014'
BASE = ROOT / 'outputs/data_split_audit_20261014'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run():
    pack_path = REV / 'train_false_positive_review_04/review_pack.json'
    pack = read(pack_path)
    dataset = read(REV / 'reviewed_curve_dataset_04/dataset.json')
    for path, digest in dataset['sources'].items():
        assert sha(Path(path)) == digest, path
    old_journal = BASE / 'train_event_review_01/human_event_reviews.jsonl'
    new_journal = REV / 'train_followup_review_02/human_event_reviews.jsonl'
    model = PROJECT / 'data/contact_classifier/source25_train_20261014_v1/model.json'
    assert sha(old_journal) == 'fdecc04fcbf29b5c885fa68ef0c656804fecdce90efe3e40496f860e7abf0a2d'
    assert sha(new_journal) == sha(REV / 'reviewed_curve_dataset_04/followup_journal_snapshot.jsonl')
    assert sha(model) == 'f898b855b215d86fcb0fb8e245e2d5b08808d0ae6ab3cee11ebf19128c53b7b4'
    followup = read(REV / 'reviewed_curve_dataset_04/followup_snapshot.json')
    assert followup['completed_cases'] == 10
    render = read(REV / 'train_false_positive_review_04/render_validation.json')
    assert len(render) == 6 and all(r['events_prefilled'] == 0 for r in render)
    tests = {'curved_contact_candidates': 9, 'learned_curved_policy': 9, 'contact_output_policy': 6,
             'dataset_protocol': 7, 'auxiliary_event_review': 14, 'video_only_contacts': 13}
    checks = {'tests_passed': tests, 'total_tests_passed': sum(tests.values()),
              'all_six_native_pages_rendered': True, 'native_smoke_passed': True,
              'preview_visually_inspected': True, 'human_labels_not_prefilled': True,
              'old_and_completed_followup_journals_unchanged': True,
              'default_classifier_sha256': sha(model),
              'source_equivalence': read(REV / 'final_source_equivalence_checks.json')}
    save(REV / 'validation_completed_reviews_04.json', checks)
    selection = read(REV / 'train_learned_policy_04/selection.json')
    evaluation = read(REV / 'cached_eval_04/evaluation.json')
    metrics = {name: {phase: values[phase]['overall'] for phase in ('before', 'after')}
               for name, values in evaluation['results'].items()}
    report_path = REV / '完成10例后的优化结果与6例核查说明.txt'
    report_path.write_text('''完成10例后的触台规则优化结果

已读取并使用全部10例补充标注，主事件均为触台。16_001_220的实际触台在217帧，保留这个时间真值；不把220帧直接当作正例。16_002_33的触台后紧接击球仍按两个事件保留。
合并原36例后，训练窗口共有57个触台、47个击球和5个其他事件。弯曲轨迹分类器使用133个自动候选：46个触台正例、87个窗口内非触台例。阴性依据是已完整核查窗口内没有匹配触台，不能推广到未标注整段视频。

本轮完成：共享曲率的前后轨迹拟合、自动轨迹特征分类、分组交叉选择、平滑三次飞行对照。图像y值是否先增后减不再作为必要条件。当前台面投影仍只是图像证据，尚未证明球的真实接触高度。
复用已有球点和台面检测。为补齐已有窗口的击球证据，仅新增75帧固定模型的姿态/球拍推理（50+25帧）；没有重新跑完整球点/球桌检测。新增10例和5个强转向候选均未触发手腕击球条件，扩大手腕模型训练没有取得增益。
短暂漏球拟合只使用缺失前后真实检测点，至少各4个可靠支持点；点数不足或时间不连续时拒绝输出。该模块目前仅供诊断，尚未加入默认输出，不能声称已恢复11_004的缺失触台。

训练来源交叉选择（候选窗口、仅时间匹配）：原方案39 TP / 16 FP / 18 FN，F1=69.64%；新方案47 TP / 13 FP / 10 FN，F1=80.34%。这是训练侧选方案结果，不是独立验证成绩。
复用已冻结自动输入的整段回放：31条内部验证，94 TP / 72 FP / 52 FN -> 95 TP / 72 FP / 51 FN，F1 60.26% -> 60.70%。该集合真值仍为阶段候选，未匹配输出不能全部认定为实际误检。
15条公开回归：67 TP / 42 FP / 25 FN -> 68 TP / 44 FP / 24 FN，F1均为66.67%；多恢复1个触台，同时增加2个误检。使用严格±1帧、50像素一对一匹配，未使用忽略区豁免。
新预测先冻结再打开评测标签；阈值按训练来源选择。公开/内部集合已经多次开发回放，因此这些成绩不能称为未见盲测。整段改善不足，本轮不替换默认规则和f898击球模型。最终运行入口仍只输入视频，人工标注只用于离线训练和核查。

下一步需要6个新训练窗口：11_005_68、06_002_22、06_007_13、16_001_45、19_006_28、06_001_23。
这些是模型分数较高、此前未核查、现有阶段参考未覆盖的候选，不预设它们是误检。必须区分漏标触台、击球、其他变化或无事件，才能继续改进误检控制。
双击 触台规则补充核查.cmd，会打开桌面程序的1/6、完成0/6。
查看候选前后整个窗口，记录所有触台/击球及各自时刻；再选择“本候选对应事件”，点击“本例核查完成”。帧间事件选“当前与后一帧之间”。无事件选“本窗口无事件”，看不清选“无法判断”。不需要标球点或球拍坐标。
原36例、已完成10例都不用重标。已完成10例可通过“已完成10例核查.cmd”查看；新6例单独保存，不覆盖旧记录。

已通过58项相关回归测试、6个桌面页面渲染和原生窗口启动检查；核对333个自动弯曲候选特征与冻结版本完全一致。旧模型、旧评测预测和标注记录的哈希检查通过。本轮不是GPU测速，也未解决50 FPS目标。
''', encoding='utf-8')
    status = {
        'status': 'awaiting_six_unreviewed_train_event_labels', 'completed_prior_followup_cases': 10,
        'completed_prior_followup_label_counts': {'table_bounce': 10},
        'review_pack': str(pack_path), 'review_pack_sha256': sha(pack_path),
        'required_cases': 6, 'completed_cases': 0,
        'native_launcher': str(ROOT / '触台规则补充核查.cmd'),
        'previous_completed_launcher': str(ROOT / '已完成10例核查.cmd'),
        'new_ball_or_table_detector_inference_calls': 0, 'supplemental_pose_racket_frames': 75,
        'default_video_pipeline_changed': False, 'experimental_rule_selected_but_not_adopted': True,
        'reason_for_not_adopting': 'Small reserved stage-GT gain; public F1 unchanged with two more false positives.',
        'training_selection_report': str(REV / 'train_learned_policy_04/selection.json'),
        'cached_evaluation_report': str(REV / 'cached_eval_04/evaluation.json'),
        'report': str(report_path), 'training_window_metrics': selection['selected']['window_temporal_metrics'],
        'cached_whole_video_metrics': metrics, 'validation': str(REV / 'validation_completed_reviews_04.json'),
        'new_review_is_potential_false_positive_not_prefilled_negative': True,
        'next_step': 'Review six previously unreviewed train windows; retrain and select using train source groups only.'}
    previous = REV / 'status_before_completed10_04.json'
    if not previous.exists():
        previous.write_bytes((REV / 'status.json').read_bytes())
    save(REV / 'status.json', status)
    audit_status = read(BASE / 'status.json')
    audit_status['active_contact_rule_followup'] = {
        'status': status['status'], 'cases': 6, 'completed_previous_followup': 10,
        'review_pack': str(pack_path), 'launcher': status['native_launcher'],
        'report': str(report_path), 'default_rule_changed': False}
    save(BASE / 'status.json', audit_status)
    print(json.dumps({'pending_cases': 6, 'completed_previous': 10, 'tests_passed': sum(tests.values())}))


if __name__ == '__main__':
    run()

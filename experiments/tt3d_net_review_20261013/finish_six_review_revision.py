"""Freeze this revision's optional profile and required new net-zone review."""
import hashlib,json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2];ROOT=PROJECT.parents[1]
REV=ROOT/'outputs/contact_rule_revision_20261014';BASE=ROOT/'outputs/data_split_audit_20261014'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run():
    dataset=read(REV/'reviewed_curve_dataset_11/dataset.json')
    six_journal=REV/'train_false_positive_review_04/human_event_reviews.jsonl'
    assert all(sha(p)==digest for p,digest in dataset['sources'].items() if Path(p)!=six_journal)
    assert sha(BASE/'train_event_review_01/human_event_reviews.jsonl')=='fdecc04fcbf29b5c885fa68ef0c656804fecdce90efe3e40496f860e7abf0a2d'
    assert sha(REV/'train_followup_review_02/human_event_reviews.jsonl')==sha(REV/'reviewed_curve_dataset_11/followup_journal_snapshot.jsonl')
    six_frozen=REV/'reviewed_curve_dataset_11/additional_0_journal_snapshot.jsonl'
    assert six_journal.read_bytes().startswith(six_frozen.read_bytes())
    latest={}
    for line in six_journal.read_text(encoding='utf-8').splitlines():
        row=json.loads(line);latest[row['case_id']]=row
    snapshot=read(REV/'reviewed_curve_dataset_11/additional_0_snapshot.json')
    fields=('events','primary_event_id','review_scope','status','candidate_uncertain','no_events_in_scope')
    for row in snapshot['records']:
        assert all(row['state'].get(k)==latest[row['case_id']]['state'].get(k) for k in fields)
    view_update={'journal_sha256_at_training':sha(six_frozen),'latest_journal_sha256':sha(six_journal),
                 'training_event_labels_identical':True,'append_only_history_preserved':True,
                 'later_changes_are_view_state_only':sha(six_frozen)!=sha(six_journal)}
    classifier=PROJECT/'data/contact_classifier/source25_train_20261014_v1/model.json'
    assert sha(classifier)=='f898b855b215d86fcb0fb8e245e2d5b08808d0ae6ab3cee11ebf19128c53b7b4'
    notes={'source':'Explicit user messages; original event journals retained.',
        'cases':{'16_001_45':'normal_flight','06_007_13':'net_collision','11_005_68':'net_graze'}}
    save(REV/'six_review_user_clarifications_12.json',notes)
    old=BASE/'full_video_eval_01';freeze=read(old/'prediction_freeze.json')
    assert all(sha(p)==digest for p,digest in freeze['prediction_sha256'].items())
    assert all(sha(old/'videos'/vid/'run.json')==digest for vid,digest in freeze['run_sha256'].items())
    cached=REV/'cached_eval_09';new_freeze=read(cached/'prediction_freeze.json')
    assert all(sha(p)==digest for p,digest in {**new_freeze['input_hashes'],**new_freeze['predictions']}.items())
    asset=PROJECT/'data/contact_classifier/source25_curved_contact_v1/policy.json'
    assert sha(asset)==sha(REV/'train_learned_policy_09/policy.json')==new_freeze['policy_sha256']
    evaluation=read(cached/'evaluation.json');paired={}
    for name,result in evaluation['results'].items():
        before={v['video_id']:v for v in result['before']['by_video']};lost=[];added=[]
        for video in result['after']['by_video']:
            am={m['gt'].get('event_id',m['gt']['frame_id']):m for m in before[video['video_id']]['matches']}
            bm={m['gt'].get('event_id',m['gt']['frame_id']):m for m in video['matches']}
            lost += [{'video_id':video['video_id'],'frame_id':am[k]['gt']['frame_id']} for k in am.keys()-bm.keys()]
            added += [{'video_id':video['video_id'],'frame_id':bm[k]['gt']['frame_id']} for k in bm.keys()-am.keys()]
        assert not lost
        paired[name]={'old_matched_events_lost':lost,'new_matched_events':added}
    pack_path=REV/'train_net_contact_review_12/review_pack.json';pack=read(pack_path)
    assert len(pack['cases'])==4 and not (pack_path.parent/'human_event_reviews.jsonl').exists()
    tests={'curved_contact_candidates':9,'learned_curved_policy':13,'automatic_ballistic_contact':6,
           'contact_output_policy':6,'dataset_protocol':7,'auxiliary_event_review':15,
           'video_only_contacts':13,'optimized_video_only_contacts':3}
    checks={'tests_passed':tests,'total_tests_passed':sum(tests.values()),'original_old_journals_models_and_predictions_unchanged':True,
        'completed_six_journal_audit':view_update,
        'four_native_review_pages_rendered':True,'native_review_smoke_passed':True,'optional_results_viewer_smoke_passed':True,
        'review_and_results_previews_visually_inspected':True,'optional_profile_reproduction':read(REV/'optional_profile_reproduction_12.json'),
        'paired_event_matches':paired,'profile_sha256':sha(asset),'review_pack_sha256':sha(pack_path)}
    save(REV/'validation_completed_six_12.json',checks)
    report=REV/'完成6例后的优化结果与网区域核查说明.txt'
    report.write_text('''6例补标后的触台规则优化结果

6例全部已读取：2例触台、1例击球、3例其他。用户明确16_001_45为正常飞行、06_007_13为撞网、11_005_68为擦网；解释另存，原始标注日志未修改。
合并后共有52个主候选标签（26触台、19击球、6其他、1无事件），完整核查窗口包含66个触台、57个击球、8个其他事件。153个自动弯曲候选用于训练，其中51个触台正例、102个窗口内非触台例。训练仅用冻结训练来源，未将公开/内部验证标签加入训练。

这轮没有重新运行球、球桌、姿态或球拍检测。复用已有自动结果，补充较长前后轨迹、自动台面区域和自动四角下的重力轨迹假设。空间高度和相机内参仍未独立验证；四角不可靠时三维假设保持缺失。未用图像y值先增后减作为必要条件，未使用人工相机、球网或球点作为推理输入。
增加了小型浅树分类器处理特征组合，并在有可靠自动候选证据时允许筛除原输出中的低分候选。推理可使用NumPy/OpenCV与原固定模型。三维硬否决和网投影区域方案未通过本轮接受检查，没有加入选用版本。网区域方案只是可疑区域，长短边不确定时可能投到错误位置，不能等同于真实网接触。

保留的可选实验版本09：训练来源交叉选择（66个已标触台，窗口内仅时间匹配），旧45 TP / 18 FP / 21 FN -> 新47 TP / 12 FP / 19 FN；F1 69.77% -> 75.20%。它用于选择参数，不是独立验证成绩。
31条内部验证缓存回放：旧94 TP / 72 FP / 52 FN -> 新95 TP / 70 FP / 51 FN；F1 60.26% -> 61.09%。内部参考仍是阶段候选，未匹配输出不全等同于真实误检。
15条公开回归：旧67 TP / 42 FP / 25 FN -> 新68 TP / 42 FP / 24 FN；F1 66.67% -> 67.33%。严格±1帧、50像素一对一匹配，无忽略区豁免。旧方案已匹配的触台均保留。所有新预测先冻结再评分；多轮开发回放不属于新素材盲测。
对比方案07减少误检更多，但训练窗口漏检较多；补充强负样本权重、三维硬否决、网投影区域的方案未取得稳定综合收益。未根据公开某一视频/帧编写例外规则或回调阈值。

16_001_45在版本09的已标训练视频回放中不再输出为触台；2个已标真触台保留。06_002_22击球、06_007_13撞网、11_005_68擦网仍有误报风险，不能声称6例已经全部解决。当前主要缺口是区分网附近的不同事件，以及相邻触台/击球和短暂漏球造成的候选缺失。支持点不足时不会补造球点。

已增加独立可选入口：视频自动分析（优化实验版）.cmd。
它只选择一个视频，运行自动检测与固定版本09，再写optimized_landings.jsonl；原experimental_landings.jsonl仍保留。分析完成后打开原生结果窗口；E页显示该版本缓存回放成绩，明确不是当前所选视频准确率。原“视频自动分析.cmd”和正式比赛输出未替换。

下一步需要4个未标窗口：16_001_27、11_005_123、00_002_69、16_003_14。选择只使用训练来源的自动网区域假设，不预填真假或类型，窗口已截在旧核查范围之前/之后，避免重复浏览已标片段。它们可能是触台、击球、正常过网、撞网/擦网或检测问题。
双击 触台规则补充核查.cmd，应显示1/4、完成0/4。
记录窗口内所有事件及其时刻，然后指定本候选对应事件。若选“其他变化”，点击右侧“细分”循环选择撞网、擦网、正常飞行或检测错点，再添加/更新事件。看不清可保留未细分或无法判断，不必猜测。最后点“本例核查完成”。无需标球点、球拍、球台或球网坐标。
旧36例、10例和刚完成的6例均不用重标。“已完成6例核查.cmd”可查看刚完成记录。

验证：72项相关测试通过；4个核查页面与原生窗口检查通过；优化结果界面已渲染核对。当前代码重新计算46条缓存视频，输出与冻结版本09完全一致。视频入口使用合成视频和固定资产检查，确认不读取人工标注和历史外部缓存。没有重跑全部真实视频检测，没有进行新一轮GPU/50 FPS测速。优化仍是可用实验版本，尚未完成最终规则验收。
''',encoding='utf-8')
    status={'status':'awaiting_four_net_zone_training_reviews','completed_previous_followups':{'ten_windows':10,'six_windows':6},
        'review_pack':str(pack_path),'review_pack_sha256':sha(pack_path),'required_cases':4,'completed_cases':0,
        'native_launcher':str(ROOT/'触台规则补充核查.cmd'),'report':str(report),
        'default_video_pipeline_changed':False,'default_hit_classifier_sha256':sha(classifier),
        'optional_profile_available':True,'optional_profile_path':str(asset),'optional_profile_sha256':sha(asset),
        'optional_launcher':str(ROOT/'视频自动分析（优化实验版）.cmd'),'formal_production_adopted':False,
        'cached_evaluation_report':str(cached/'evaluation.json'),'automatic_detector_inference_calls_this_round':0,
        'known_unresolved_training_cases':['06_002_22','06_007_13','11_005_68'],
        'user_semantics':notes['cases'],'validation_report':str(REV/'validation_completed_six_12.json'),
        'next_step':'Review four distinct unreviewed train windows and other-event subtypes; preserve train-source-only selection.'}
    previous=REV/'status_before_completed6_12.json'
    if not previous.exists():previous.write_bytes((REV/'status.json').read_bytes())
    save(REV/'status.json',status)
    parent=read(BASE/'status.json');parent['active_contact_rule_followup']={
        'status':status['status'],'cases':4,'completed_ten':10,'completed_six':6,
        'review_pack':str(pack_path),'launcher':status['native_launcher'],'report':str(report),
        'default_rule_changed':False,'optional_optimized_profile':str(asset)}
    save(BASE/'status.json',parent)
    print(json.dumps({'completed_previous':6,'pending_new':4,'tests_passed':sum(tests.values()),'optional_profile_sha256':sha(asset)}))

if __name__=='__main__':run()

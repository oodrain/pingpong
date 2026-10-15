"""Read-only native comparison of the completed contact reviews."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
from annotate import font,window_visible,close_window
from annotation_core import write_image

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'worktrees/table-geometry-v2/outputs/experiments/tt3d_trajectory_pilot_20261013/constrained_02/comparison.json'
TITLE='Pingpong - Contact Review Analysis'
PILOT=REPORT.parent.parent

def render_regression():
    folder=PILOT/'public_regression_01';evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'failure_audit.json').read_text(encoding='utf-8'));pool=json.loads((folder/'auxiliary_pool.json').read_text(encoding='utf-8'))
    scores=evaluation['scores'];image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'三视角公开回归：新候选规则不能替换基线',26)
    text(24,65,'公开 15 段非独立留出集；对比原始 V0 候选基线。固定阈值、无干扰点豁免，只评候选生成。',18)
    columns=[24,230,520,800,1060]
    for x,h in zip(columns,['视角','基线 TP / FP / FN','新规则 TP / FP / FN','基线 F1','新规则 F1']):text(x,122,h,18)
    for i,angle in enumerate(scores['baseline']['without_ignore_exemption']['by_angle']):
        old=scores['baseline']['without_ignore_exemption']['by_angle'][angle];new=scores['vector']['without_ignore_exemption']['by_angle'][angle];y=175+i*68
        for x,v in zip(columns,[angle,f"{old['tp']} / {old['fp']} / {old['fn']}",f"{new['tp']} / {new['fp']} / {new['fn']}",f"{old['f1']:.3f}",f"{new['f1']:.3f}"]):text(x,y,v,23)
    old=scores['baseline']['without_ignore_exemption']['overall'];new=scores['vector']['without_ignore_exemption']['overall']
    text(24,402,f"总 F1：{old['f1']:.3f} → {new['f1']:.3f}；因此不采用直接替换。",24,(255,190,100))
    text(24,455,f"原命中丢失 {audit['lost_count']} 个，新获得 {audit['gained_count']} 个。",20)
    text(24,497,'丢失归因：13 个变化评分/时间选择；11 个连续有效球点不足；4 个缺少可用几何。',18)
    text(24,543,f"保留基线 {pool['primary_count']} 个候选，另存 {pool['auxiliary_count']} 个辅助候选；没有自动采用新增项。",18)
    text(24,585,f"合并候选池覆盖 {pool['candidate_pool_matched_gt']}/{pool['valid_gt']} 个有效标注；这是候选覆盖，不是最终规则精度。",18)
    screening=PILOT/'auxiliary_screening_01/screening.json'
    if screening.exists():
        counts=json.loads(screening.read_text(encoding='utf-8'))['status_counts']
        text(24,626,f"筛选：{counts.get('ready_for_event_review',0)} 个待判接触类型；{counts.get('timing_alternative',0)} 个时间备选；{counts.get('deferred_insufficient_evidence',0)} 个证据不足暂缓。",18)
        filtered=json.loads((screening.parent/'evaluation.json').read_text(encoding='utf-8'))['scores']
        before=filtered['baseline']['overall'];after=filtered['baseline_plus_review_ready']['overall']
        text(24,661,f"全部加入会命中 {after['tp']}、误报 {after['fp']}（基线 {before['tp']} / {before['fp']}），仍未自动采用；尚无已确认的独立测试集。",17,(255,194,110))
    else:
        text(24,637,'下一步：验证辅助候选筛选、球拍接触证据与新来源测试集；不继续用这 15 段调参。',18)
    text(24,710,'U 新增候选核查；R 本页；W 原手腕证据；S/D/E 其他分析；Esc / Q 关闭。',18)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_wrists(detail=None):
    report=json.loads((PILOT/'wrist_fusion_02/fusion.json').read_text(encoding='utf-8'))
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    names={'table_contact_candidate':'触台候选','uncertain':'不确定','off_table_motion_change':'台面投影外变化','racket_contact_candidate':'击球候选（未确认）','contact_type_uncertain':'接触类型待确认','table_wrist_evidence_conflict':'触台/手腕证据冲突','tracking_anomaly_possible':'疑似跟踪异常'}
    text(24,16,'手腕辅助证据：人体姿态与球轨迹',26)
    if detail is not None:
        index=detail%len(report['cases']);c=report['cases'][index]
        path=PILOT/'wrist_evidence_01'/f"{c['video_id']}_{c['seed_frame']}_wrists.jpg"
        image.paste(Image.open(path).convert('RGB'),(40,145))
        text(24,65,f"{index+1}/{len(report['cases'])}  {c['video_id']} / {c['seed_frame']} 帧    {names[c['combined_class']]}",22)
        text(24,108,'原始姿态叠加：绿圈为球点；黄色为几何邻近，尚未经过人物归属过滤。',18)
        quality=json.loads((PILOT/'wrist_quality_02/wrist_evidence.json').read_text(encoding='utf-8'))
        item=next(r for r in quality['cases'] if r['video_id']==c['video_id'] and r['seed_frame']==c['seed_frame'])
        rejected=sorted({r['frame_id'] for r in item['frames'] for o in r['observations'] if o.get('geometrically_near') and not o['near']})
        text(24,650,'质量过滤排除的邻近帧：'+(', '.join(map(str,rejected)) or '无')+'；画面邻近不等于实际接触。',18,(255,194,110))
    else:
        text(24,62,'13 个自动候选附近，共 39 帧；CPU 离线提取，未接入实时桌面推理。')
        columns=[24,210,530,780]
        for x,h in zip(columns,['候选','手腕邻近证据','邻近帧','保守融合结果']):text(x,105,h,17)
        for i,c in enumerate(report['cases']):
            y=141+i*31;draw.rectangle((18,y-2,1340,y+28),fill=(33,40,50) if i%2==0 else (27,33,43))
            status={'multi_frame_wrist_proximity':'多帧、同侧手腕匹配','single_frame_wrist_proximity':'仅单帧邻近','no_supported_wrist_proximity':'未取得有效邻近证据'}[c['wrist_status']]
            for x,v in zip(columns,[f"{c['video_id']}_{c['seed_frame']}",status,', '.join(map(str,c['near_frame_ids'])) or '—',names[c['combined_class']]]):text(x,y,v,16)
        text(24,565,'19_003 / 29：击球候选；11_003 / 79：多帧手腕邻近，但接触类型仍需确认。',18)
        text(24,601,'19_007 / 14：原单帧邻近存在人物归属歧义，已排除；触台候选判断保留。',18)
        text(24,637,'未检测球拍尖端；未验证人物角色；遮挡或姿态错位时，不能用“没靠近”排除击球。',18,(255,194,110))
    text(24,708,'J 证据图，N/P 切换，W 手腕汇总；R 公开回归，S/D/E 其他分析页；Esc 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_events():
    report=json.loads((PILOT/'event_rules_01/events.json').read_text(encoding='utf-8'))
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    labels={'table_contact_candidate':'触台候选','uncertain':'不确定','off_table_motion_change':'台外运动变化','tracking_anomaly_possible':'疑似跟踪异常'}
    count=sum(len(v['proposals']['proposals']) for v in report['videos'])
    text(24,16,'自动切分与事件规则：实验回放',26)
    text(24,62,f'完整回放 3 段视频，生成 {count} 个新候选；下表对照已有人工核查的 9 处事件。')
    columns=[24,220,460,680,900]
    for x,h in zip(columns,['已核查候选','新提议帧（邻近匹配）','实验分类','自动选取窗口','主要限制']):text(x,110,h,17)
    for i,c in enumerate(report['reviewed_evaluation']):
        y=146+i*43;draw.rectangle((18,y-3,1340,y+35),fill=(33,40,50) if i%2==0 else (27,33,43))
        frames=c['legacy_seed_selected_frames'];decision=c['legacy_seed_decision'];reasons=decision['reasons']
        issue='仍需独立验证' if not reasons else ('分段后有效球点不足' if 'insufficient_context_after_segmentation' in reasons else '空间反弹估计不可靠' if 'physical_boundary_solution' in reasons else '缺少球拍/人体接触证据')
        values=[c['case_id'],', '.join(map(str,c['new_proposals_within_interval_plus_one_frame'])) or '非触台，未计匹配',labels[decision['event']],f'{frames[0]}～{frames[-1]}' if frames else '无',issue]
        for x,v in zip(columns,values):text(x,y,v,17)
    text(24,550,'7 个已标触台区间均有邻近候选（区间外允许 1 帧）；这不是完整视频的召回率。',18)
    text(24,585,'候选来自二维向量变化，不要求图像 y 转向；遇到相邻变化或缺失检测自动截断窗口。',18)
    text(24,620,'台外变化不能直接叫击球：缺少球拍/人体证据时保留类别，不强行判定。',18,(255,194,110))
    text(24,655,'所有结果仍为实验输出；新提议中的未核查事件不能算作正确或错误。正式事件未替换。',18)
    text(24,710,'按 W 查看手腕证据；S 拟合对照，D 相机/窗口诊断，E 本页；Esc / Q 关闭。',18)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_diagnostics():
    motion=json.loads((PILOT/'diagnostics_01/diagnostics.json').read_text(encoding='utf-8'))
    short=json.loads((PILOT/'window_consistency_01/short_window_probes.json').read_text(encoding='utf-8'))
    short={c['case_id']:c for c in short['cases']}
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'新增排查：背景运动与拟合窗口',26)
    text(24,62,'背景匹配独立于球点；测试窗口未使用人工触台区间进行选择。')
    columns=[24,205,420,595,770,985]
    for x,h in zip(columns,['候选','背景证据','最大位移 px','短窗误差 px','扩窗异常帧','短窗仍触及约束']):text(x,110,h,17)
    for i,c in enumerate(motion['cases']):
        s=short[c['case_id']];f=s['variants'][0]['fit'];y=146+i*43
        draw.rectangle((18,y-3,1340,y+35),fill=(33,40,50) if i%2==0 else (27,33,43))
        shift=max(r.get('max_table_location_shift_px',0) for r in c['checks'])
        flagged=[str(r['frame_id']) for r in s['one_frame_extension_checks'] if r['extension_inconsistent']]
        values=[c['case_id'],'检测到位移' if c['background_status']=='background_motion' else ('支持背景稳定' if c['background_status']=='background_stable' else '证据不足'),f'{shift:.2f}',f"{f['rmse_px']:.2f}" if f else '有效点不足',', '.join(flagged) or '无提示','是' if f and f['active_constraint_indices'] else ('否' if f else '无法拟合')]
        for x,v in zip(columns,values):text(x,y,v,17)
    text(24,550,'11_003 / 76：加入第 80 帧后明显失配，逐帧画面也显示球拍接球附近的运动。',18)
    text(24,585,'19_003 / 67：第 63、71 帧扩窗预测不一致；不能将整段当作一次触台的自由飞行。',18)
    text(24,620,'19_007 / 35：背景稳定，缩窗后误差降低，但反弹速度仍贴物理边界，三维解释未解决。',18,(255,194,110))
    text(24,655,'背景稳定不等于标定准确；短窗口误差低也不等于时间更准确。尚未替换正式结果。',18)
    text(24,710,'按 S 返回拟合对照；按 E 查看新事件规则；Esc / Q 关闭。',18)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render(report=REPORT):
    data=json.loads(Path(report).read_text(encoding='utf-8'))
    stale=[]
    for path,digest in data['sources'].items():
        if path.endswith('.jsonl'):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:stale.append(path)
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'触台核查分析：人工记录与两种拟合对照',26)
    text(24,62,'9 个候选 / 3 段视频；已标记 7 个触台区间、2 个非触台。结果仅用于开发实验。')
    columns=[24,210,395,575,755,935,1080]
    headings=['候选','人工判断','旧拟合 ±8 帧','无约束 ±6 帧','有约束 ±6 帧','新误差 px','仍未通过']
    for x,h in zip(columns,headings):text(x,110,h,17)
    for i,c in enumerate(data['cases']):
        y=146+i*43;draw.rectangle((18,y-3,1340,y+35),fill=(33,40,50) if i%2==0 else (27,33,43))
        e=c['evaluation'];primary=c['variants'][1]['fit'];old=c['unconstrained_same_window']
        label=f"{e['human_interval'][0]}～{e['human_interval'][1]} 帧触台" if e else '不是触台'
        more=[r for r in c['reasons'] if not r.startswith('camera_')]
        issue='相机待确认' if not more else ('投影误差大等' if 'large_projection_residual' in more else '物理边界/反弹异常')
        values=[c['case_id'],label,f"{c['old_fit_frame']:.3f}",f"{old['fractional_frame']:.3f}" if old else '无解',f"{primary['fractional_frame']:.3f}" if primary else '无解',f"{primary['rmse_px']:.2f}" if primary else '—',issue]
        for x,v in zip(columns,values):text(x,y,v,16)
    m=data['metrics']
    text(24,552,f"落入人工区间：旧拟合 {m['unconstrained']['inside_closed_human_interval']}/7；同窗口无约束 {m['unconstrained_same_window']['inside_closed_human_interval']}/7；有约束 {m['constrained']['inside_closed_human_interval']}/7。",19)
    text(24,588,'同窗口对照尚未证明时间精度提高；空间速度受约束，不等于已恢复真实三维运动。',18,(255,194,110))
    text(24,623,'旧整数候选也都在人工区间端点内；区间标记不能证明小数帧比整数帧更准确。',18)
    text(24,658,'19_007 的 35～36 帧仍贴物理边界，暂不采用。全部案例仍有相机检查或模型问题。',18)
    text(24,698,'U 新增候选；R 公开回归；W 手腕证据；D 窗口排查；E 事件规则；Esc / Q 关闭。',18)
    if stale:text(24,735,'注意：核查记录在分析后发生变化，请重新计算报告。',16,(255,140,100))
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_auxiliary(detail=None):
    folder=PILOT/'public_auxiliary_wrists_01'
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'新增候选：手腕证据与接触类型核查',26)
    if not (folder/'wrist_evidence.json').exists():
        text(24,90,'证据尚未生成完毕；请稍后重新按 U。')
    else:
        report=json.loads((folder/'wrist_evidence.json').read_text(encoding='utf-8'));cases=report['cases']
        names={'multi_frame_wrist_proximity':'多帧邻近：可能击球','single_frame_wrist_proximity':'单帧邻近：类别待定','no_supported_wrist_proximity':'无有效邻近：类别待定'}
        if detail is not None:
            c=cases[detail%len(cases)]
            text(24,66,f"{detail%len(cases)+1}/{len(cases)}  {c['video_id']} / {c['seed_frame']} 帧  {c['angle']}  {names[c['status']]}",22)
            image.paste(Image.open(folder/f"{c['video_id']}_{c['seed_frame']}_wrists.jpg").convert('RGB'),(40,140))
            text(24,641,'绿圈：球点；黄色：有效邻近手腕；红色：质量过滤排除；下排：球点附近放大图。',18)
        else:
            text(24,65,f"固定的 {len(cases)} 个候选，{report['processed_frames']} 帧姿态证据；未使用触台答案决定分类。",18)
            for i,c in enumerate(cases):
                x=24+(i//16)*665;y=120+(i%16)*29
                text(x,y,f"{c['video_id']} / {c['seed_frame']:3d}  {names[c['status']]}",17)
            evaluation=folder/'evaluation.json'
            if evaluation.exists():
                scores=json.loads(evaluation.read_text(encoding='utf-8'))['scores']
                before=scores['all_auxiliary']['overall'];after=scores['hypothetical_wrist_exclusion']['overall']
                text(24,619,f"诊断性排除多帧邻近项：命中 {before['tp']}→{after['tp']}，误报 {before['fp']}→{after['fp']}；未启用自动排除。",19)
        text(24,678,'手腕邻近不等于球拍接触；没有邻近也不能确认为触台。公开集仅用于回归。',18,(255,194,110))
    text(24,720,'B 标记与保存；F 方向对照；U 汇总；J 证据图；N/P 切换；R 公开回归；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_direction():
    folder=PILOT/'contact_direction_01';image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'击球候选：前后运动方向与手腕证据',26)
    if not (folder/'evaluation.json').exists():
        text(24,90,'方向分析尚未完成；稍后重新按 F。')
    else:
        report=json.loads((folder/'direction_evidence.json').read_text(encoding='utf-8'))
        scores=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))['scores']
        text(24,65,'31 个固定候选；表中列出原 9 个多帧手腕邻近项。仅为公开回归诊断，未启用排除。',18)
        columns=[24,250,525,750,1030]
        for x,h in zip(columns,['候选','球台长轴方向','二维速度夹角 cos','方向+手腕结果','处理']):text(x,111,h,17)
        names={'reversed':'投影分量反向','same_direction':'投影分量同向','uncertain':'投影分量过小','physical_axis_unknown':'长轴未确认'}
        cases=[c for c in report['cases'] if c['wrist_status']=='multi_frame_wrist_proximity']
        for i,c in enumerate(cases):
            d=c['direction'];y=148+i*34;candidate=c['decision']['event']=='racket_contact_candidate'
            values=[f"{c['video_id']} / {c['seed_frame']}",names[d['longitudinal_direction']],f"{d['image_direction_cosine']:.3f}",'击球候选' if candidate else '手腕与方向不支持','仅诊断性排除' if candidate else '保留核查']
            for x,v in zip(columns,values):text(x,y,v,17)
        for i,(label,key) in enumerate([('全部辅助候选','all_auxiliary'),('只按手腕排除','hypothetical_wrist_exclusion'),('方向+手腕排除','hypothetical_direction_and_wrist_exclusion')]):
            s=scores[key]['overall'];text(24,483+i*36,f"{label}：命中 {s['tp']}，误报 {s['fp']}，漏检 {s['fn']}；候选池 F1 {s['f1']:.3f}",19)
        text(24,600,'v000037 / 44、v000057 / 10：二维运动没有反向，保留了先前被手腕规则误删的触台命中。',17)
        text(24,635,'仅 10/31 个候选可对应已有人工长轴标记；图像方向不是三维速度，几何未覆盖整段。',17)
        text(24,672,'仍缺击球真值与独立验证；候选池误报减少不能当作击球分类准确率。',18,(255,194,110))
    text(24,720,'M 人工核查评估；F 本页；U 候选汇总；J/N/P 三帧图；R 公开回归；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_human_review():
    path=PILOT/'human_event_evaluation_01/evaluation.json'
    image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'人工核查完成：击球筛选的实际表现',26)
    if not path.exists():text(24,90,'人工核查评估尚未生成。')
    else:
        r=json.loads(path.read_text(encoding='utf-8'));labels=r['human_label_counts']
        text(24,65,f"已完成 {r['reviewed_cases']} 个候选：触台 {labels.get('table_bounce',0)}、击球 {labels.get('racket_contact',0)}、其他 {labels.get('other',0)}、无事件 {labels.get('no_event',0)}。",19)
        text(24,102,'公开开发集中的 31 个辅助候选，单人核查；这些指标不代表完整视频或新视频的准确率。',17)
        for x,h in zip([24,320,625,940],['方法','筛出项中确认为击球','人工击球中筛出','误排除触台']):text(x,150,h,18)
        for i,(name,key,pool_key) in enumerate([('只按多帧手腕邻近','wrist_flags','wrist_exclusion'),('方向＋手腕联合','joint_flags','joint_exclusion')]):
            m=r[key];pool=r['table_candidate_pool'][pool_key];y=194+i*45
            for x,v in zip([24,320,625,940],[name,f"{m['tp']}/{m['flagged_count']}",f"{m['tp']}/{m['tp']+m['unflagged_true_hits']}（{m['recall_within_reviewed_candidates']:.1%}）",pool['removed_table_contacts']]):text(x,y,v,20)
        text(24,299,'联合规则保住全部 15 个已标触台，但还有 8 个击球未被筛出；其余候选不能直接判作触台。',18,(255,194,110))
        reasons={'motion_does_not_support_reversal':'二维方向未支持反向','single_frame_wrist_only':'仅单帧手腕邻近','no_supported_wrist_evidence':'没有有效手腕证据'}
        for x,h in zip([24,280,615],['漏筛的击球候选','人工区间','主要证据缺口']):text(x,346,h,17)
        for i,c in enumerate(r['missed_hit_cases']):
            y=382+i*29;interval=c['human_primary_interval']
            for x,v in zip([24,280,615],[c['case_id'],f'{interval[0]}～{interval[1]} 帧',reasons[c['unflagged_hit_reason']]]):text(x,y,v,17)
        text(24,647,'下一步检查短窗中的触台/击球相邻变化与遮挡；保存了原始记录快照，未按新标签调整阈值。',17)
        stale=[p for p,h in r['sources'].items() if p.endswith('human_event_reviews.jsonl') and hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        if stale:text(24,681,'核查记录已更新，请重新导出并评估后再使用本页。',17,(255,145,100))
    text(24,718,'G 剩余原因诊断；L 五帧时序；M 本页；F 原规则；B 人工核查；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_temporal_contact(detail=None):
    folder=PILOT/'temporal_contact_01';image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'短窗时序：接近手腕、最近、离开',26)
    if not (folder/'evaluation.json').exists():text(24,90,'五帧证据正在生成，请稍后重新按 L。')
    else:
        report=json.loads((folder/'temporal_evidence.json').read_text(encoding='utf-8'))
        evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));cases=report['cases']
        if detail is not None:
            index=detail%len(cases);c=cases[index];visit=c['visit'];phase=c['phase_diagnostic']
            text(24,65,f"{index+1}/{len(cases)}  {c['video_id']} / {c['seed_frame']} 帧   短暂邻近证据：{'支持' if visit['supported'] else '不足'}",22)
            image.paste(Image.open(folder/f"{c['video_id']}_{c['seed_frame']}_temporal.jpg").convert('RGB'),(40,137))
            closest=', '.join(str(v['closest_frame']) for v in visit['visits']) or '无'
            text(24,592,'符合接近/离开的最近帧：'+closest+'；黄色为有效邻近，红色为质量不通过。',18)
            quality='残差支持' if phase.get('fit_quality_supported') else '残差不支持 / 有效点不足'
            text(24,629,'前后独立二次曲线：'+quality+'；此项只用于诊断，没有用来放宽击球判断。',17)
        else:
            old=evaluation['old_flags'];new=evaluation['new_flags'];pool=evaluation['new_table_pool']
            text(24,65,f"全部 31 个候选统一扩至 5 帧：复用 {report['reused_pose_frames']} 帧，补算 {report['new_pose_frames']} 帧。",18)
            text(24,108,'这是根据已有失败案例设计的方法，在同一批人工标记上回顾性评估；不是独立测试。',17)
            text(24,157,f"筛出的击球：{old['tp']}/{old['flagged_count']} → {new['tp']}/{new['flagged_count']}；13 个人工击球中筛出 {new['tp']} 个。",23)
            text(24,201,f"击球覆盖：{old['recall_within_reviewed_candidates']:.1%} → {new['recall_within_reviewed_candidates']:.1%}；误排除触台 {pool['removed_table_contacts']} 个。",21)
            reasons={'motion_does_not_support_reversal':'二维方向未支持反向','single_frame_wrist_only':'原先仅单帧邻近','no_supported_wrist_evidence':'原先无有效手腕'}
            original=json.loads((PILOT/'human_event_evaluation_01/evaluation.json').read_text(encoding='utf-8'))
            results={c['case_id']:c for c in evaluation['cases']};lookup={c['video_id']+'_'+str(c['seed_frame']):c for c in cases}
            for x,h in zip([24,270,610,935],['原漏筛候选','原证据缺口','接近/离开证据','新增筛选结果']):text(x,260,h,17)
            for i,c in enumerate(original['missed_hit_cases']):
                result=results[c['case_id']];v=lookup[c['case_id']];y=303+i*34
                for x,value in zip([24,270,610,935],[c['case_id'],reasons[c['unflagged_hit_reason']],'支持' if v['visit']['supported'] else '不足','击球候选' if result['new_flag'] else '仍待解决']):text(x,y,value,18)
            text(24,619,'保持原方向门槛与手腕距离门槛；短暂接近证据可补充“必须多帧邻近”的限制。',17)
        text(24,676,'新增结果仍未自动采用；没有手腕或没有反向证据的案例，不能直接判为触台。',18,(255,194,110))
    text(24,718,'L 本页；J 五帧图，N/P 切换；M 人工评估；F 原规则；B 人工核查；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_remaining_diagnosis(detail=None):
    folder=PILOT/'remaining_contact_diagnosis_01';image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'剩余漏筛击球：原因核对与局部人物重提取',26)
    if not (folder/'root_cause_report.json').exists():text(24,90,'诊断报告尚未生成。')
    else:
        r=json.loads((folder/'root_cause_report.json').read_text(encoding='utf-8'));cases=r['diagnoses']
        if detail is not None:
            index=detail%len(cases);c=cases[index]
            text(24,65,f"{index+1}/{len(cases)}  {c['case_id']}  {'局部证据补出击球候选' if c['roi_hit_flag'] else '仍待解决'}",22)
            image.paste(Image.open(folder/f"{c['video_id']}_{c['seed_frame']}_roi.jpg").convert('RGB'),(40,137))
            text(24,592,c['summary'],20);text(24,628,c['detail'],18)
        else:
            before=r['baseline_flags'];after=r['roi_flags'];pool=r['roi_table_pool']
            text(24,65,f"全部 31 个候选使用同一人物裁剪流程；完成 {r['crop_inferences']} 次局部姿态推理，均通过人物框对应检查。",17)
            text(24,108,'当前样本击球覆盖 '+f"{before['tp']}/13 → {after['tp']}/13（{after['recall_within_reviewed_candidates']:.1%}）；误排除触台 {pool['removed_table_contacts']} 个。",23)
            for x,h in zip([24,255,850,1120],['原漏筛候选','主要原因','具体证据','局部结果']):text(x,161,h,17)
            evidence={'direction_assumption':'手腕有效，方向门槛未过','weak_arm_keypoints':'邻帧右臂关键点低分','wrist_localization_and_scale':'前臂尺度跨帧抖动'}
            for i,c in enumerate(cases):
                y=205+i*72
                for x,value in zip([24,255,850,1120],[c['case_id'],c['summary'],evidence[c['root_cause']],'补出候选' if c['roi_hit_flag'] else '仍待解决']):text(x,y,value,16)
                text(255,y+27,c['detail'],15,(180,197,215))
            text(24,602,'v000057 / 58：局部姿态补齐邻帧右腕证据；其他四例仍受方向假设、定位或遮挡限制。',17)
            text(24,639,'保持原质量、距离与方向门槛；没有修改人工标记。局部结果尚未接入正式事件输出。',17)
        text(24,676,'这是已标记开发样本上的诊断；还没有独立视频验证，不能作为完整视频准确率。',18,(255,194,110))
    text(24,718,'H 补充击球提示；G 本页；J 局部五帧图，N/P 切换；L 原五帧；B 核查；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def render_person_motion(detail=None):
    folder=PILOT/'person_motion_fusion_02';image=Image.new('RGB',(1360,770),(24,28,36));draw=ImageDraw.Draw(image)
    def text(x,y,value,size=18,color=(225,232,240)):draw.text((x,y),str(value),font=font(size),fill=color)
    text(24,16,'人物、轨迹与局部台面位置：补充击球提示',26)
    if not (folder/'evaluation.json').exists():text(24,90,'补充证据尚未生成。')
    else:
        evidence=json.loads((folder/'evidence.json').read_text(encoding='utf-8'));evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));cases=evidence['cases']
        if detail is not None:
            c=cases[detail%len(cases)];d=c['decision'];pos=c['local_positions']
            text(24,65,f"{detail%len(cases)+1}/{len(cases)}  {c['case_id']}   {'击球提示' if d['review_hit_suggestion'] else '类型仍待判断'}",22)
            image.paste(Image.open(folder/(c['case_id']+'_spatial.jpg')).convert('RGB'),(40,137))
            text(24,585,'人物进入/离开框：'+('支持' if c['person_visit']['supported'] else '不足')+'；短暂接近手腕：'+('支持' if c['wrist_visit']['supported'] else '不足'),18)
            distances='；'.join(f"{p['frame_id']}帧 {p['signed_distance_px']:+.1f}px" if p['available'] else f"{p['frame_id']}帧 无有效台面" for p in pos)
            text(24,618,'距局部台面投影边界（负值在外）：'+distances,16)
            reasons=[]
            if d['baseline_flag']:reasons.append('原有提示保留')
            if d['person_entry_reversal_outside']:reasons.append('人物框短暂进入 + 反向 + 台面投影外')
            if d['transient_wrist_impulse_outside']:reasons.append('手腕接近/离开 + 轨迹突变 + 台面投影外')
            text(24,651,'依据：'+('；'.join(reasons) or '补充分支证据不足，保留待判'),17)
        else:
            old=evaluation['baseline_flags'];new=evaluation['suggestion_flags'];pool=evaluation['suggestion_table_pool']
            text(24,65,f"全部 31 个候选统一规则：原提示 {old['tp']}/13 → 补充提示 {new['tp']}/13（{new['recall_within_reviewed_candidates']:.1%}）。",23)
            text(24,110,f"这批样本补充提示误报 {new['fp']} 个，误筛触台 {pool['removed_table_contacts']} 个；15 个触台全部保留。",20)
            rows=[('v000095 / 159','人物框进入/离开 + 轨迹反向 + 台面投影外','补出提示'),
                  ('v000117 / 83','人物框进入/离开 + 轨迹反向 + 台面投影外','补出提示'),
                  ('v000117 / 25','手腕接近/离开 + 轨迹突变 + 台面投影外','补出提示'),
                  ('v000038 / 53','手腕有效，但位于台面投影内，二维轨迹未反向','仍待判断')]
            for i,(cid,reason,status) in enumerate(rows):
                y=185+i*70;text(24,y,cid,20);text(265,y,reason,18);text(1130,y,status,18)
            text(24,505,'新增分支仅提供核查提示，没有加入正式事件排除；人物框不能证明接触球拍。',18)
            text(24,548,'台面只用于当前帧的投影区域，失效/过期时保留待判；不能据此测量小球高度。',18)
            text(24,591,'尚未使用击球—触台—击球顺序：缺少独立可靠的自动触台锚点，不能用人工答案补齐。',17)
            text(24,634,'12/13 来自已标记开发样本的回顾评估；还需要独立视频检验，不能当作整体准确率。',17,(255,194,110))
    text(24,718,'H 本页；J 五帧证据，N/P 切换；K 球拍与球点补标；G 局部诊断；B 事件核查；Esc / Q 关闭。',17)
    return cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

def open_analysis(smoke_seconds=0,diagnostics=False,events=False,wrists=False,regression=False,auxiliary=False,direction=False,human_review=False,temporal=False,remaining=False,person_motion=False):
    frame=render_person_motion() if person_motion else render_remaining_diagnosis() if remaining else render_temporal_contact() if temporal else render_human_review() if human_review else render_direction() if direction else render_auxiliary() if auxiliary else render_regression() if regression else render_wrists() if wrists else render_events() if events else render_diagnostics() if diagnostics else render();auxiliary=auxiliary or direction or human_review or temporal or remaining or person_motion;cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE);start=time.monotonic();detail=-1
    try:
        while True:
            cv2.imshow(TITLE,frame);key=cv2.waitKeyEx(40)
            if key in (ord('d'),ord('s'),ord('e'),ord('w'),ord('r'),ord('u'),ord('f'),ord('m'),ord('g'),ord('h')):temporal=False
            if key in (ord('d'),ord('s'),ord('e'),ord('w'),ord('r'),ord('u'),ord('f'),ord('m'),ord('l'),ord('h')):remaining=False
            if key in (ord('d'),ord('s'),ord('e'),ord('w'),ord('r'),ord('u'),ord('f'),ord('m'),ord('l'),ord('g')):person_motion=False
            if key in (ord('d'),ord('s'),ord('e'),ord('w'),ord('r')):auxiliary=False
            if key==ord('d'):frame=render_diagnostics();detail=-1
            elif key==ord('s'):frame=render();detail=-1
            elif key==ord('e'):frame=render_events();detail=-1
            elif key==ord('w'):frame=render_wrists();detail=-1
            elif key==ord('r'):frame=render_regression();detail=-1
            elif key==ord('u'):auxiliary=True;detail=-1;frame=render_auxiliary()
            elif key==ord('f'):auxiliary=True;detail=-1;frame=render_direction()
            elif key==ord('m'):auxiliary=True;detail=-1;frame=render_human_review()
            elif key==ord('l'):temporal=True;auxiliary=True;detail=-1;frame=render_temporal_contact()
            elif key==ord('g'):remaining=True;auxiliary=True;detail=-1;frame=render_remaining_diagnosis()
            elif key==ord('h'):person_motion=True;auxiliary=True;detail=-1;frame=render_person_motion()
            elif key==ord('k'):
                from racket_evidence_review import open_racket_review
                close_window(TITLE)
                try:open_racket_review()
                finally:cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE)
            elif key==ord('b'):
                from auxiliary_event_review import open_auxiliary_review
                close_window(TITLE)
                try:open_auxiliary_review()
                finally:cv2.namedWindow(TITLE,cv2.WINDOW_AUTOSIZE)
            elif key==ord('j'):detail=0;frame=render_person_motion(detail) if person_motion else render_remaining_diagnosis(detail) if remaining else render_temporal_contact(detail) if temporal else render_auxiliary(detail) if auxiliary else render_wrists(detail)
            elif key in (ord('n'),ord('p')) and detail>=0:
                detail+=1 if key==ord('n') else -1
                frame=render_person_motion(detail) if person_motion else render_remaining_diagnosis(detail) if remaining else render_temporal_contact(detail) if temporal else render_auxiliary(detail) if auxiliary else render_wrists(detail)
            if key in (27,ord('q')) or not window_visible(TITLE) or (smoke_seconds and time.monotonic()-start>smoke_seconds):break
    finally:close_window(TITLE)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',type=Path);parser.add_argument('--smoke-test',action='store_true');parser.add_argument('--diagnostics',action='store_true');parser.add_argument('--events',action='store_true');parser.add_argument('--wrists',action='store_true');parser.add_argument('--regression',action='store_true');parser.add_argument('--auxiliary',action='store_true');parser.add_argument('--direction',action='store_true');parser.add_argument('--human-review',action='store_true');parser.add_argument('--temporal',action='store_true');parser.add_argument('--remaining',action='store_true');parser.add_argument('--person-motion',action='store_true');args=parser.parse_args()
    if args.preview:write_image(args.preview,render_person_motion() if args.person_motion else render_remaining_diagnosis() if args.remaining else render_temporal_contact() if args.temporal else render_human_review() if args.human_review else render_direction() if args.direction else render_auxiliary() if args.auxiliary else render_regression() if args.regression else render_wrists() if args.wrists else render_events() if args.events else render_diagnostics() if args.diagnostics else render())
    else:open_analysis(1.5 if args.smoke_test else 0,args.diagnostics,args.events,args.wrists,args.regression,args.auxiliary,args.direction,args.human_review,args.temporal,args.remaining,args.person_motion)

"""Render the completed paired diagnostic as repository documentation."""
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,write,write_rows,sha
from development.evaluation.run_geometry_d_v2 import changed_matches


def link(path,label):return f'[{label}]({path.resolve().as_posix()})'
def number(value):return f'{value:.3f}' if value is not None else '不可比'


def main():
    output=PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013'
    audit=output/'regression_evaluation_01'
    final=read(audit/'summary.json');records=read(audit/'records.json')
    fixed=read(output/'fixed_event_replay_01/metrics.json')
    run=output/'regression_all_01';new=read(run/'metrics.json')
    baseline=PROJECT/'outputs/experiments/auto_table_v2/runs/20261012T155647_594418Z'
    old_summary=read(baseline/'summary.json');old=read(baseline/'metrics.json')
    changes=changed_matches(old['public']['auto_region']['without_ignore'],new['public']['auto_region']['without_ignore'],'revised_adaptive')
    write_rows(output/'public_old_to_revised_changed_matches.jsonl',changes)
    validation=read(output/'validation.json')
    validation.update(full_video_regression_status='completed',videos=169,
        geometry_source_sha256=final['source_sha256'],
        model_provenance_matched=True,baseline_video_and_label_hashes_matched=True,
        baseline_v0_predictions_exactly_reproduced=True,
        regression_summary_sha256=sha(run/'summary.json'),paired_audit_sha256=sha(audit/'summary.json'))
    write(output/'validation.json',validation)
    report_dir=PROJECT/'experiments/tt3d_geometry_revision_20261013';report_dir.mkdir(exist_ok=True)
    lines=['# TT3D 球台几何首次修订与回归','',
        f"2026-10-13。完成四角提取修订、因果初始加密采样，以及15个公开视频、154个训练视频的回归。{validation['tests_passed']}项相关测试通过。当前仍为实验分支，未接入相机标定或3D轨迹，也未替换桌面正式事件流程。本轮为离线CPU验证，未验证实时50FPS。",'',
        '## 修改内容','',
        '- 从原图寻找包住台面分割区域的四条外边线，检查完整边线支持、实际观测长度和角点之外的边线延伸。只在原候选有截断迹象、替代边线证据足够时重建；证据冲突时返回不可用。',
        '- 每条边都需要图像支持，避免平均分掩盖某一条边落在球网、球员或台面内部的问题。遮挡时以可见掩膜覆盖和强边线共同判断，保留有可靠四边外轮廓的候选。',
        '- 正常采样仍为0.4秒；初始或观测丢失时增加约0.12秒的采样。保留三次稳定确认、20帧保持期限和切镜重置，不使用未来帧。另保留原0.4秒采样时刻的对照。',
        '- 细分掩膜不足、图像边线不足、外轮廓歧义和边线延伸未被解释的拒绝原因。','',
        '## 人工参考对照','',
        '沿用原36帧、39次提交中的最新36条参考：20个四角可见，16个遮挡。原始提交未改写。这些单人参考参与了本轮方法开发，结果属于开发诊断，不能称为独立泛化验证。','',
        '|样本|原单帧最大误差|修订单帧最大误差|处理|','|---|---:|---:|---|']
    for cid in ('public_v000094_000048','train_17_002_000072','train_21_001_000106','train_00_005_000018'):
        r=next(r for r in records if r['case_id']==cid)
        a=r.get('baseline_same_frame_error',{}).get('maximum_corner_error_px')
        b=r.get('revised_same_frame_error',{}).get('maximum_corner_error_px')
        lines.append(f"|{cid}|{number(a)}px|{number(b)}{'px' if b is not None else ''}|{r['revised_proposal']['reason']}|")
    paired=final['paired_errors']['revised_same_frame']
    lines += ['',f"单帧前后都可用的{paired['cases']}个相同样本，平均角点误差从{number(paired['baseline']['mean_corner_error_px'])}px变为{number(paired['revised']['mean_corner_error_px'])}px。前后各有13个可见样本可比，但样本集合不同，不能把两组独立均值当作同一组样本的提升。",
        '', 'train/00_005 frame18仍有明显误差；其近端左角附近有球员遮挡，单人参考和自动结果均需后续复核。保留用户原记录，不自动改变可见性或四角。','',
        '|时序版本|36帧中可用|与可见参考可比|平均角点误差|最大角点误差|','|---|---:|---:|---:|---:|']
    for name,label in (('baseline','冻结原版'),('revised_fixed','修订＋原采样'),('revised_adaptive','修订＋加密初始采样')):
        c=final['comparison'][name]
        lines.append(f"|{label}|{c['usable_cases']}|{c['comparisons']}|{number(c['mean_corner_error_px'])}px|{number(c['maximum_corner_error_px'])}px|")
    lines += ['', '时序比较只在核查帧进行，不能把单帧候选的误差当作实际可用时序几何的误差。各版本可比集合不同，相同样本对照见摘要中的paired_errors。完整逐样本记录包含恢复、失去几何及拒绝原因。','',
        '加密采样与原版都可用的5个相同可见样本，平均角点误差从9.863px变为4.352px。train/17_002 frame72的时序最大角点误差从110.798px降到10.532px；public/v000094 frame48当前仍等待稳定确认，没有将改善后的单帧候选强行作为可用标定。','',
        '新增可用时序样本中仍有两例明显误差：train/00_005 frame18最大78.550px，public/v000038 frame98最大50.748px。后者处于旧几何保持状态。下一轮需优先检查保持旧几何时的形状漂移及遮挡角点；覆盖增加和触台F1小幅改善均不足以证明标定合格。','',
        '## 固定视频回归','',
        '|组别|冻结原版有几何的视频|修订＋原采样|修订＋加密初始采样|','|---|---:|---:|---:|']
    for group,total in (('public',15),('train',154)):
        old_count=old_summary['coverage'][group]['any_usable']
        fix_count=final['fixed_sampling_coverage'][group]['any_usable_videos']
        new_count=final['regression_coverage'][group]['any_usable']
        lines.append(f'|{group}|{old_count}/{total}|{fix_count}/{total}|{new_count}/{total}|')
    lines += ['', '“视频中曾有可用几何”不等于整段覆盖，也不证明四角正确。采样频率不同，采样点总数不能直接当作覆盖提升比例。','',
        '|触台区域过滤版本|公开集TP/FP/FN|公开集汇总F1|训练阶段候选标签TP/FP/FN|训练代理F1|','|---|---|---:|---|---:|']
    for label,p,t in (
        ('原V0',old['public']['v0']['without_ignore']['overall'],old['train_proxy']['v0']['overall']),
        ('冻结原版几何',old['public']['auto_region']['without_ignore']['overall'],old['train_proxy']['auto_region']['overall']),
        ('修订＋原采样',fixed['public']['overall'],fixed['train']['overall']),
        ('修订＋加密初始采样',new['public']['auto_region']['without_ignore']['overall'],new['train_proxy']['auto_region']['overall'])):
        lines.append(f"|{label}|{p['tp']}/{p['fp']}/{p['fn']}|{p['f1']:.6f}|{t['tp']}/{t['fp']}/{t['fn']}|{t['f1']:.6f}|")
    lost=[c for c in changes if c['change']=='lost_tp']
    lines += ['',f'公开集相对冻结原版几何的匹配变化：丢失命中{len(lost)}个。逐项记录见下方产物。公开集使用原92个有效事件，不豁免无效标签附近预测；训练指标使用冻结723个阶段候选事件，不能替代最新735候选版本或正式真值。',
        '', '触台规则、轨迹、标签、模型权重保持相同；V0的逐条预测已与原缓存严格核对一致。这里测试的是区域过滤，未实现TT3D相机位姿或帧间触台拟合。','',
        '## 当前边界与下一步','',
        '1. 已明确的截角和遮挡掩膜拒绝得到一轮修订；仍有误差或不可用样本，不能对所有视频直接开展自动标定。',
        '2. 对剩余截角、边线冲突和时序覆盖不足逐项处理，并增加没有参与本轮开发的四角参考。当前参数为开发启发式，质量分不是正确率。',
        '3. 在可靠几何片段上确认球台长宽方向、球网位置和相机内参/畸变来源，再接相机位姿与分段轨迹。四角参考仅适用于所标帧，不能自动当作整段相机标定。','',
        '本地TT3D标定源码使用fx=fy、主点位于画面中心、零畸变的假设，并在500—20000像素范围内优化焦距。接入时需要保留这些假设与多解诊断，独立计算真实像素重投影均方根；上游optimize_f返回的error是优化器cost，不能直接称为像素RMSE。源码：'+link(PROJECT/'tmp/tt3d-upstream/tt3d/calibration/reproj_optimizer.py','reproj_optimizer.py')+'。','',
        '## 可检查的产物','',
        '- '+link(audit/'summary.json','前后比较摘要'),
        '- '+link(audit/'records.json','36帧逐项结果'),
        '- '+link(run/'summary.json','169视频运行与输入来源'),
        '- '+link(output/'public_old_to_revised_changed_matches.jsonl','公开集匹配变化'),
        '- '+link(output/'validation.json','验证记录'),
        '- '+link(PROJECT/'development/detection/auto_table_geometry.py','四角与时序实现'),
        '- '+link(PROJECT/'development/detection/table_border_lines.py','外边线候选与延伸检查'),
        '- '+link(PROJECT/'development/evaluation/run_auto_table_v2.py','回归与初始加密采样'),
        '', '人工参考SHA256：'+final['reference_sha256'],
        '冻结原版几何SHA256：'+final['geometry_sha256']['baseline'],'']
    (report_dir/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Compact image artifact for inspection in Codex; this is a computed overlay.
    selected=['public_v000094_000048','train_17_002_000072','train_19_001_000024','train_00_005_000018']
    canvas=Image.new('RGB',(1000,600),(22,22,22));draw=ImageDraw.Draw(canvas)
    submitted={r['case_id']:r for r in (json.loads(line) for line in
        (audit/'references_snapshot.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    for i,cid in enumerate(selected):
        image=Image.open(audit/'overlays'/(cid+'.jpg')).convert('RGB')
        points=np.asarray(submitted[cid]['corners']);left,top=np.floor(points.min(axis=0)-55).astype(int)
        right,bottom=np.ceil(points.max(axis=0)+55).astype(int)
        image=image.crop((max(0,left),max(60,top),min(image.width,right),min(image.height,bottom)))
        image.thumbnail((490,260));x=(i%2)*500;y=(i//2)*300
        canvas.paste(image,(x+(500-image.width)//2,y+28+(260-image.height)//2))
        draw.text((x+8,y+7),cid,fill='white')
        draw.text((x+8,y+281),'green: human   purple: old   yellow: accepted revision',fill='white')
    canvas.save(output/'revision_comparison.png')
    print(report_dir/'REPORT.md')


if __name__=='__main__':main()

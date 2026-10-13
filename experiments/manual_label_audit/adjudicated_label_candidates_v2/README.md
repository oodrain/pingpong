# 154条视频裁决汇总候选标签 v2

本目录在v1基础上应用B对D阶段一35条疑似漏标的人工核对结果。仅33条训练154记录会修改候选标签；公共15的2条仅写入`public15_audit_only.jsonl`，不会修改官方GT。v1保持不变。

- `candidate_events.jsonl`：v2逐事件候选标签。
- `candidate_by_video.jsonl`：按154条视频分组。
- `audit_changes.jsonl`：33条训练核对的逐条增删记录。
- `public15_audit_only.jsonl`：2条公共数据核对意见，仅审计。
- `duplicate_candidates.jsonl`：按帧±1、位置50px检查的重复候选。
- `merge_report.json`：合并口径和完整性统计。

本目录仍是阶段候选标签，不冒充官方GT或冻结版最终标签。

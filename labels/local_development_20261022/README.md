# 本地补充标注快照 v1

版本：`local_development_20261022_v1`。复用已有人工判断，未新增标注。输入视频只在本地按哈希定位，快照无视频、模型、预测缓存或绝对本机路径。

| 文件 | 用途 |
|---|---|
| precise_scopes.json | 42段精确事件、可信评分范围、完整评分范围和训练已知范围 |
| confirmed_positive_union.json | 123段中625条去重正例；包括精确范围外既有确认正例，用于独立防误删检查 |
| supplemental_completed.jsonl | 120条最新completed人工补充记录，包含事件、状态和少量空间标记；彼此可能重叠 |
| supplemental_drafts.jsonl | 13条球网细分类草稿，不能作为已完成GT |
| journals/ | 对应人工日志的修订历史，草稿、复用旧事件和最终判断均保留 |
| geometry_references.jsonl | 人工台面/球网参考标记；保留原确认状态，去掉派生相机拟合结果 |
| source_hashes.json | 本地源文件相对路径及SHA-256，用于溯源；不是要求仓库拥有全部历史缓存 |
| summary.json | 各部分数量和来源摘要 |

可信窗口内有166次触台；其中20段完整核查视频包含112次，是166的子集。625条正例清单包含与这些事件的复用关系，不能相加。120条补充记录也不是120次新增触台。

先使用precise_scopes确定精确评分范围，再用confirmed_positive_union独立检查原方案命中过的正例是否丢失。625正例清单的缺项仍为未知，不能据此计算全视频误报。其他初标、复标和v2裁决继续保留在原labels及experiments/manual_label_audit目录，不覆盖。

补充日志是来源与人工判断记录；同一视频的新整段核查会替代旧窗口监督，不能将所有日志盲目拼接。precise_scopes是本快照已整理好的精确范围入口。补充事件不因为被导出就自动获得训练资格，仍受data/splits/team_split_v1.json约束；公开样例辅助记录仅供开发回归。

帧号均从0开始，事件区间包含两端，空间坐标为原视频像素。公开v000038的51～55帧是人工球点与拍面补标，与数据集原有逐帧对象球框是不同来源。

核查与评分：

```bash
python development/evaluation/team_data.py
python development/evaluation/team_data.py --baseline old.json --candidate new.json --output audit.json
```

输入JSON为`{video_id: [{frame_id, x, y}, ...]}`；也支持JSONL每行`{video_id, points: [...]}`。评分程序用于离线评估，不进入只输入视频的推理流程。丢失任意原方案已匹配正例时退出码为2。

需要从完整本地开发档案重新导出时，安装NumPy并运行：

```bash
python tools/team_sync/export_local_annotations.py --data-root /path/to/development_archive --output /path/to/checkout
```

普通团队使用者直接读取已发布快照即可，不需要复原历史outputs。标注人的身份未被日志记录时不补造姓名；实际初标/复核身份仍以原交换文件为准。

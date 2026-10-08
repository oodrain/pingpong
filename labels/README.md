# 小组人工标注结果

仓库按初标来源成员归档复核，让同一视频的initial和review相邻。例如A复核D时，文件放labels/D/视频/review，metadata和每条记录的annotator仍是A，review_of.annotator仍是D。按复核者存放的交换包也兼容。初标目录必须匹配初标者；复核目录必须匹配实际复核者或绑定的初标者，不能改写真实身份来适配目录。使用仓库自带的新版校验器：python tools/validate_labels.py validate labels。

每段按成员/视频/轮次提交，例如A/03_007/initial或B/03_007/review。

标准要求metadata.json、events.jsonl、uncertain.jsonl，可选non_table_events.jsonl；新工具始终导出四个文件。空JSONL保留。校验器接受v1/v2，本地窗口默认v2。

执行标准、JSON示例和GitHub步骤见[标注提交标准](../docs/guides/标注文件与GitHub提交标准.md)。本地窗口可一键或批量导出，也可用submission.py export；默认校验不限制固定任务分配，自制工具遵守相同格式即可。

标注工具可从 [tools/](../tools/README.md) 下载。校验仓库结果使用下方自带脚本，Python 3.10及以上即可，无需安装图形工具依赖。在仓库根目录运行：

```powershell
python tools/validate_labels.py validate labels
```

这里保存各成员的正式初标与独立复核；分歧仍需人工确认，不自动作为最终标签。

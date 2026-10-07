# 小组人工标注结果

每段按成员/视频/轮次提交，例如A/03_007/initial或B/03_007/review。

标准要求metadata.json、events.jsonl、uncertain.jsonl，可选non_table_events.jsonl；新工具始终导出四个文件。空JSONL保留。校验器接受v1/v2，本地窗口默认v2。

执行标准、JSON示例和GitHub步骤见[标注提交标准](../docs/guides/标注文件与GitHub提交标准.md)。本地窗口可一键或批量导出，也可用submission.py export；默认校验不限制固定任务分配，自制工具遵守相同格式即可。

先从 [tools/](../tools/README.md) 下载并解压工具包。在仓库根目录运行以下命令，将 `<工具包目录>` 替换为解压得到的 `pingpong-annotation-tool` 绝对路径：

```powershell
python "<工具包目录>/development/annotation/submission.py" validate labels
```

这里目前只建立了提交目录和规则，没有把示例或检查副本作为人工成果放入。

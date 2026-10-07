# pingpong

乒乓球多视角落点检测。

当前建立协作目录骨架。已有分工、标注格式说明和标注工具已归入对应位置；新增团队算法文件为空，未新增数据划分、人工标签、实验结果或自动检查逻辑。

## 从这里开始

- [已有分工总结](docs/team/乒乓球项目分工总结.md)
- [已有标注工具格式要求](docs/guides/标注工具格式要求.md)
- [标注与 GitHub 提交标准](docs/guides/标注文件与GitHub提交标准.md)
- [标注工具操作说明](docs/guides/标注工具_组员上手说明.md)
- [下载标注工具 ZIP](tools/pingpong-annotation-local-20261007.zip)
- [官方工程连接说明](official/README.md)

## 目录职责

|目录|用途|当前状态|
|---|---|---|
|solution/|团队最终推理源码，后续对应官方乒乓球参赛者代码区|Python 文件与配置、权重目录留空|
|tools/|供组员下载的标注工具 ZIP|工具包随仓库提交，解压后使用|
|development/detection/|轨迹导出与检测诊断工具|空目录|
|development/bounce/|离线规则回放与诊断工具；复用 solution 中的规则|空目录|
|development/evaluation/|评估与错误可视化工具|空目录|
|development/packaging/|后续生成官方运行工程的打包工具|空目录|
|development/training/|后续训练及模型导出工具|空目录|
|labels/|A/B/C/D 的人工初标与复核结果|已有说明，成员目录为空|
|data/manifests/|团队数据来源、路径与哈希索引|已有任务清单 tasks.json，其余待补|
|data/splits/|开发集与内部检查集划分|空目录|
|data/label_selection/|评估采用的标签版本与裁决记录|空目录|
|data/artifacts/|外部轨迹与大结果的位置、版本、哈希|空目录|
|docs/|现有使用说明、分工与后续接口文档|已有文档已整理；interfaces 留空|
|examples/、experiments/|后续小型样例与实验记录|空目录|
|official/participant/|官方参考工程|框架、参考算法和输入清单；数据、GT 和权重另取|
|.github/|后续工作流与任务模板|仅空目录|

## 与官方工程的对应

团队源码目录 `solution/` 的内容，后续由打包工具放入 `official/participant/pingpang/participant/` 对应的运行工程代码区。当前尚未实现打包工具，空白团队文件不能直接覆盖官方参考算法。

官方工程根是 `official/participant/`：其中有 `run_all.sh`、`pingpang/`、`basketball/`。提交时将该层工程放到容器 `/participant`。受保护框架文件保持原字节，后续算法通过 `participant.solution:Solution` 接入。

推理配置以 `solution/configs/` 为后续维护位置；实验记录保存使用的配置版本。模型、视频、官方 GT 和真实原始数据按约定渠道获取，不存放在这个上传目录中。运行时需在自己的开发工程中补齐。

## 组员下载与使用工具

1. 从 [tools/](tools/README.md) 下载 `pingpong-annotation-local-20261007.zip`。
2. 将整个 ZIP 解压到自己的固定目录，进入 `pingpong-annotation-tool/`。
3. 安装 Python 3.12 x64，双击解压目录里的 `安装标注环境.cmd`。
4. 双击同一目录里的 `启动标注.cmd`，导入分配的视频和对象 JSON。
5. 将标准结果导出到本地仓库的 `labels/`，只提交对应成员、视频和轮次的标准文件。

仓库中的 [任务清单](data/manifests/tasks.json) 从原有工具包复制，分配内容未改动。工具不要求使用固定数据目录；显式按清单校验输入时，需核对路径或指定的数据根目录。ZIP 内保留原有兼容任务表。

上传目录只保留工具 ZIP。组员将 ZIP 解压到仓库之外的固定目录，安装启动脚本、环境、缓存和工作记录都留在自己的工具目录。

提交标签前，在仓库根目录执行以下命令，将 `<工具包目录>` 替换为自己解压得到的 `pingpong-annotation-tool` 绝对路径：

```powershell
python "<工具包目录>/development/annotation/submission.py" validate labels
```

校验器只需要 Python 3.10 及以上和标准库；图形标注工具使用 Python 3.12 x64。详见 [组员上手说明](docs/guides/标注工具_组员上手说明.md)。

`labels/A`、`B`、`C`、`D` 为空目录，未放置 `.gitkeep`，以免被已有校验器误判为提交。Git 不保留空成员目录，首次提交真实标签后自然出现。

## 仓库用途

本仓库用于小组交流代码、标签、文档和标注工具 ZIP。本地解压工具、环境、缓存、原始数据和备份存放在仓库之外。

组员领取具体任务后，在任务分支提交改动并提出 Pull Request，由相关成员检查后合并。

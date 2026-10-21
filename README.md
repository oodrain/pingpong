# pingpong

乒乓球多视角落点检测。

本分支保存截至 2026-10-17 的 **TCN 触台识别、自动台面几何及球检测流程阶段成果**。当前内部整段时间事件 F1 为 **87.22%**，可信窗口为 **89.70%**；这些数据已有开发曝光，不能当作官方空间与时间联合评分或独立盲测成绩。

本次继续更新原阶段目录，包含当前 C2 官方工程适配源码与汇总。球检测仍保留官方原权重，主要更新局部重检、批处理和传输流程。模型、视频、标注、缓存和本机运行记录不随 Git 分发。新增 GPU 路径与端到端 50 FPS 尚待平台验收。

请先阅读 [最新进展与分视角结果](experiments/tt3d_tcn_stage_20261015/latest/README.md)，使用 [当前官方工程适配入口](experiments/tt3d_tcn_stage_20261015/latest/official_adapter/README.md)。[2026-10-15 阶段总结](experiments/tt3d_tcn_stage_20261015/README.md)及早期公开成绩保留为历史记录。

组员开展球检测优化，请先看 [轻量桌面核查工具与修改摘要](tools/ball_detection_review/README.md)，下载 [核查工具 ZIP](tools/ball_detection_review.zip)。内置六段视频的 57 个关注帧索引，可复用已有视频和球框；基线预测包通过组内共享存储另行提供。工具不需要 GPU、TCN 或模型权重。

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
|development/detection/|自动球点、台面几何及相关检测工具|已包含阶段源码|
|development/bounce/|轨迹事件、候选规则与逐帧 TCN|已包含阶段源码|
|development/evaluation/|来源划分、评测与错误分析工具|已包含阶段源码|
|development/packaging/|后续生成官方运行工程的打包工具|空目录|
|development/training/|后续训练及模型导出工具|空目录|
|labels/|A/B/C/D 的人工初标与复核结果|已有成员提交；以实际文件为准|
|data/manifests/|团队数据来源、路径与哈希索引|已有任务清单 tasks.json，其余待补|
|data/splits/|开发集与内部检查集划分|空目录|
|data/label_selection/|评估采用的标签版本与裁决记录|空目录|
|data/artifacts/|外部轨迹与大结果的位置、版本、哈希|空目录|
|docs/|现有使用说明、分工与后续接口文档|已有文档已整理；interfaces 留空|
|examples/、experiments/|小型样例与实验记录|experiments 已包含阶段实现与报告|
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

成员标签按实际提交保存在 `labels/A`、`B`、`C`、`D` 下。不要放置 `.gitkeep`，以免被已有校验器误判为提交。

## 仓库用途

本仓库用于小组交流代码、标签、文档和标注工具 ZIP。本地解压工具、环境、缓存、原始数据和备份存放在仓库之外。

组员领取具体任务后，在任务分支提交改动并提出 Pull Request，由相关成员检查后合并。

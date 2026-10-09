# 标注工具下载

第三复核请下载 [pingpong-annotation-local-20261009.zip](pingpong-annotation-local-20261009.zip)。在 GitHub 打开 ZIP 后，点击下载原文件；也可以克隆或下载整个仓库后从本目录取得 ZIP。10月7日版本只保留作旧版初标兼容，不支持本轮第三复核。

1. 解压整个 ZIP 到自己的固定目录，进入 `pingpong-annotation-tool/`。
2. 安装 Python 3.12 x64，双击解压目录中的 `安装标注环境.cmd`，首次安装需联网。
3. 双击同一目录中的 `启动标注.cmd`，选择成员。初标/复核时导入分配的视频和对象 JSON。
4. 第三复核时点击 `第三复核/裁决`，裁决包选择仓库中的 `experiments/manual_label_audit/adjudication_package`，数据根目录选择同时含 A/B/C/D 原始视频和对象 JSON 的本地目录。
5. 第三复核结果自动保存到裁决包的 `third_review/<成员>.jsonl`，不需要再导出。

请把工具解压到仓库之外的固定目录。环境、缓存和工作记录保留在自己的工具目录；上传目录只保存工具 ZIP 和交流成果。ZIP 不包含视频、对象 JSON、模型、真实标签或 Python 环境。

使用说明见 [组员上手说明](../docs/guides/标注工具_组员上手说明.md)，结果要求见 [标注提交标准](../docs/guides/标注文件与GitHub提交标准.md)。团队任务分配见 [tasks.json](../data/manifests/tasks.json)；ZIP 内的任务表是兼容副本，后续分配以仓库清单和组长安排为准。

本目录用于分发标注工具 ZIP。初标/复核结果统一交到仓库 `labels/`；第三复核结果交到 `experiments/manual_label_audit/adjudication_package/third_review/`。为保持盲审，各成员先提交到自己的分支，四人全部完成前不要把第三复核结果合并进公共主分支。

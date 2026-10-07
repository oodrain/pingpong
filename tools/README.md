# 标注工具下载

下载 [pingpong-annotation-local-20261007.zip](pingpong-annotation-local-20261007.zip)。在 GitHub 打开 ZIP 后，点击下载原文件；也可以克隆或下载整个仓库后从本目录取得 ZIP。

1. 解压整个 ZIP 到自己的固定目录，进入 `pingpong-annotation-tool/`。
2. 安装 Python 3.12 x64，双击解压目录中的 `安装标注环境.cmd`，首次安装需联网。
3. 双击同一目录中的 `启动标注.cmd`，选择成员并导入分配的视频和对象 JSON。
4. 导出根目录选择本地仓库的 `labels/`，提交标准结果文件。

请把工具解压到仓库之外的固定目录。环境、缓存和工作记录保留在自己的工具目录；上传目录只保存工具 ZIP 和交流成果。ZIP 不包含视频、对象 JSON、模型、真实标签或 Python 环境。

使用说明见 [组员上手说明](../docs/guides/标注工具_组员上手说明.md)，结果要求见 [标注提交标准](../docs/guides/标注文件与GitHub提交标准.md)。团队任务分配见 [tasks.json](../data/manifests/tasks.json)；ZIP 内的任务表是兼容副本，后续分配以仓库清单和组长安排为准。

本目录用于分发标注工具 ZIP；组员的人工标注结果统一交到仓库 `labels/`。

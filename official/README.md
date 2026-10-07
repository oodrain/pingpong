# 官方参考工程

来源：本机 `D:/AI-pingpong/participant/participant/` 中的官方参考工程副本。

`participant/` 下保留官方 README、联合入口，以及乒乓球和篮球的框架、完整性清单、自检脚本、参考算法和公开输入清单。复制保持原文件字节；两个任务的受保护文件与原 `.integrity.json` 均核对一致。

此上传目录不包含视频、公开 GT、ONNX 权重、Python 环境、运行结果或 TensorRT 缓存。运行时在自己的开发工程中按 `public_data/manifest.json` 的相对路径准备视频，并取得相应权重和评估数据。本目录是框架参考，不是包含完整数据与模型的运行包。

## 对应位置

|团队仓库|官方运行工程|
|---|---|
|solution/solution.py|participant/pingpang/participant/solution.py|
|solution/ball_detector.py|participant/pingpang/participant/ball_detector.py|
|solution/trajectory.py|participant/pingpang/participant/trajectory.py|
|solution/landing_detector.py|participant/pingpang/participant/landing_detector.py|
|solution/configs/|participant/pingpang/participant/configs/|
|选定版本的权重|participant/pingpang/participant/weights/|

团队源码当前为空，映射是后续接入约定，尚未执行覆盖。打包工具留在 `development/packaging/`，暂未实现。正式算法应自包含于任务的 `participant/` 代码区，不依赖仓库外层的标注、评估或实验目录。

官方 `Solution` 接口包括 prepare、reset、process_frame、finish_video。算法返回 `Landing(frame_id, x, y)`，官方负责补 video_id 和写出结果。接口定义见 [types.py](participant/pingpang/core/types.py)，参考实现见 [solution.py](participant/pingpang/participant/solution.py)。

官方 `core/`、`scripts/`、入口、校验器与完整性清单保持原样，不重新生成框架哈希。篮球参考实现原样保留。根目录 `.gitattributes` 对本目录关闭文本换行转换，避免 Git 改变受保护文件字节。

团队人工标签用于开发评估，不直接充当正式预测输出。官方 GT、模型和视频另行取得，存放在自己的开发工程中；根目录 `.gitignore` 也保留相应排除规则。

官方使用和提交规则见 [官方 README](participant/README.md)。

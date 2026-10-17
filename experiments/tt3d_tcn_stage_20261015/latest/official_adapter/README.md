# 当前官方工程适配源码（C2）

更新日期：2026-10-17。该目录是当前部署入口，默认配置为 `C2_equivalent`，`geometry_quick=false`。
输入为视频；推理时不读取人工标注、人工四角或历史检测缓存。原 C 对照保留在 `payload/participant/tcn_runtime/c_reference/`，与 C2 使用同一组基础模型资产。

## 范围与状态

- 触台：保留当前双分支 TCN、原阈值与真实时间戳，加入短时可信台面保留及轨迹接触点的保守几何复核。
- 球检测：仍使用官方原权重；保留自动稀疏轨迹局部重检，增加 batch8、缓冲复用及 CUDA 中间热图/I/O Binding 路径。后者需要实际 GPU 输出等价检查，不代表检测精度已经提升。
- 几何：共享采样画面的图像证据，按需搜索；复用重复拟合、优化像素计数与分位数计算。独立快速复核试验损失过一次真实触台，因此默认关闭。
- 帧号、真实 PTS、原可信球点和事件坐标契约保留。当前没有用三维物理拟合替换最终落点。
- C2 默认 CPU 路径保留原球推理图。已完成本地输出等价检查；新增 CUDA 路径尚未云端验收，尚未证明达到 50 FPS。

## 模型资产单独准备

Git 只分发源码、配置与汇总；视频、模型权重、标注、特征缓存及本机运行记录不随本目录发布。模型许可与获取渠道沿用已有授权。本目录不能在缺少模型时直接推理。

`PACKAGE.json` 分别记录源码与四项模型资产的 SHA256；`MODEL_ASSETS.json` 说明导出关系。`ball_middle.onnx` 只保留原最后一层的中间输出通道，未经训练；不能称为新训练模型。

在本目录运行：

```bash
python verify_bundle.py --source-only
python prepare_assets.py --source /path/to/authorized_local_C2_bundle
python verify_bundle.py
```

来源可以是已有完整 C2 包根目录、其 `payload/participant/` 或 `weights/`。脚本先验证全部资产，再复制到被 Git 忽略的权重目录；不下载模型、不改变参数、不覆盖不同哈希的文件。早期桌面入口使用的原始台面训练检查点与这里清除训练元数据后的部署检查点不能混用。

## 接入现有官方工程

保留官方基础环境，额外依赖按固定约束安装：

```bash
python -m pip install -r requirements-extra.txt -c constraints-official.txt
python deploy.py --task-dir /path/to/official/pingpang
python run_stage.py --task-dir /path/to/official/pingpang --mode cpu-smoke
```

安装器先备份受影响的算法文件，校验官方受保护框架。GPU 平台完成对应 CUDA 环境后执行：

```bash
python run_stage.py --task-dir /path/to/official/pingpang --mode gpu-smoke
python check_ball_gpu.py --task-dir /path/to/official/pingpang
python benchmark_pair.py --task-dir /path/to/official/pingpang --runs 1
```

任一步失败即停止。最后一条比较原 C 和 C2，保存事件等价检查、官方 `infer_fps` 与分段计时。并发阶段耗时不能直接相加；微基准加速不能代替完整视频速度。平台日志留在本地，分享前移除设备及环境信息。

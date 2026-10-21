# C 球检测 v3 · 2026-10-21

当前 C 选用 **144组微调权重＋v3局部补查**。本目录包含可安装源码、两份球检测 ONNX 权重、固定配置、哈希及开发回归摘要。保留原 [C2 适配包](../tt3d_tcn_stage_20261015/latest/official_adapter/README.md)，本次不覆盖该历史入口。

## 结果与范围

在固定的12段开发材料、相同显式601输入下，三轮GPU完整流程得到：

| 项目 | 原144流程 | 144＋v2 | 本版144＋v3 |
| --- | ---: | ---: | ---: |
| 符合既有定位口径的正确点 | 984 | 990 | **992** |
| 原8段主回归事件F1 | 0.915254 | 0.915254 | 0.915254 |
| 合计12段事件F1 | 0.857143 | 0.857143 | 0.857143 |
| 合并后TP / FP / FN | 36 / 9 / 3 | 36 / 9 / 3 | 36 / 9 / 3 |
| 12段整段耗时中位数（秒） | 58.085 | 65.731 | 64.904 |

v3比原144增加8个正确点，同颜色比较没有损失既有正确点；耗时增加约11%。其中4帧原来无坐标，另外4帧原来置信度不足。新增点来自真实图像推理，曲线只负责接纳检查，不生成球坐标。v3相对v2恢复20_004/29与06_001/36。计时波动不能证明v3比v2更快。

这些是已曝光开发回归，不是独立来源或官方比赛成绩。球点增加没有提升本轮触台分数。与native颜色输入相比仍有2个旧正确点损失；此处只发布已验证的显式601兼容路线，不宣称颜色问题解决。旧GPU间歇异常在本轮未复现，根因尚未定位。TOTNet两种官方预处理均明显损失现有正确点，未替换本版。

详见 [冻结验证摘要](evidence/acceptance.json) 与 [新增点清单](evidence/gains.json)。原始视频、标注、私有运行路径和服务器信息不在本包内。

## 运行

需要已有的官方 `pingpang` 工程、视频及其 `manifest.json`。Python依赖沿用官方环境，并安装本目录的 `requirements-extra.txt`；GPU验收环境为Torch 2.1.2+cu118、ORT GPU 1.18.1、NumPy 1.26.4。CPU只用于功能检查。

Git已包含 `ball_v3.onnx` 与 `ball_v3_middle.onnx`（各约5.3MB）。TCN、台面及旧参考球权重仍从已有C2资产包取得，按 `MODEL_ASSETS.json` 和 `PACKAGE.json` 校验；不需要重新训练。

在本目录执行，将示例路径换成自己的绝对路径：

```powershell
python verify_bundle.py --source-only
python prepare_assets.py --source "D:/assets/C2/payload/participant"
python deploy.py --task-dir "D:/work/official/participant/pingpang"
python check_adapter.py --task-dir "D:/work/official/participant/pingpang" --device cuda
python run_v3.py --task-dir "D:/work/official/participant/pingpang" --input "D:/data/pingpang" --output "D:/outputs/v3" --device cuda
```

Linux使用相同命令并替换路径。无GPU的本地检查使用 `--device cpu`；环境版本不同的准备检查可加 `--allow-version-difference`，不计GPU验收。

**必须通过 `run_v3.py` 运行完整流程。** 它在当前进程中显式绑定601解码，保留原始帧数和PTS；直接执行官方 `run.py` 会使用其native颜色路径，不能复现本表。GPU要求NVDEC，无CPU解码回退。本入口不是已获官方平台认可的解码配置，正式平台采用前需确认其输入契约。

安装器只替换 `participant/` 下的算法文件，先备份受影响文件，并在前后运行官方完整性校验。框架文件不修改。输出为官方 `predictions.jsonl`、`run_status.json` 等；可加 `--debug-tracks` 导出轨迹与特征，但调试输出不应用于速度比较。

## 冻结实现

- 原整段稀疏触发分支保留；其他局部缺口要求原始可信球点在0.12秒内提供支持。
- v2位置约束通过时保留原判断。仅对位置约束拒绝的候选，使用三枚原始可信锚点检查弯曲球路，再换平移一个热图单元的自动ROI做真实图像复查。
- 不替换原始高置信球点，不以插值输出球位，不使用人工球框、事件标签或逐视频特判。
- **完整v3轨迹送入原TCN**，v2兼容轨迹仅作诊断。打包时修正了旧实验文件中与实际运行不符的说明字段，接纳数值和阈值保持冻结。

`PACKAGE.json` 固定发布文件与外部资产的字节数/SHA-256。`evidence/source_mapping.json` 记录从冻结实验源到包内相对导入的对应；包装验证另见 `evidence/packaging_validation.json`。

打包验证通过：干净工程安装前后框架完整性检查、CPU模型预热、五个核心模块数值AST核对、40份冻结GPU记录的规则重放，以及03_003／20_004／06_001三段原视频共288帧CPU完整流程。三段轨迹逐项等于冻结CPU结果，15个最终事件帧号一致、坐标差小于1e-6像素。本次打包未重新进行GPU计时，性能表沿用前述冻结GPU验收。

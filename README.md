# 乒乓球触台检测：团队整合工作区

当前整合日期：2026-10-22。最终流程只输入视频，输出触台时刻和原图像素位置。原方案保留为对照；C球检测是可选实验，本次整合没有因球点增加而直接替换正式方案。

## 团队从这里开始

| 任务 | 入口 |
|---|---|
| 了解本次整合与分支关系 | [整合交接](docs/team/整合分支交接_20261022.md) |
| 使用统一数据划分与实验要求 | [数据协作规范](docs/team/数据与实验协作规范.md) |
| 读取本地补标与核查范围 | [补标快照](labels/local_development_20261022/README.md) |
| 查看最新开发检查与未采用原因 | [当前状态](experiments/team_integration_20261022/README.md) |
| 准备并运行原方案C2 | [C2适配入口](experiments/tt3d_tcn_stage_20261015/latest/official_adapter/README.md) |
| 对比C球检测 | [C固定实验包](experiments/c_detection_v3_20261021/README.md) |
| 核查球点 | [轻量核查工具](tools/ball_detection_review/README.md) |

## 数据约定

开发池123段、20个录像来源；内部保留31段，来源09/12/15/17/24；公开样例15段。保留数据近期不参与训练或选型，公开样例用于工程回归。开发池按来源做内层拟合、校准与留出，不是123段全部固定训练。

42段精确核查数据的可信范围有166次触台，20段完整核查视频有112次，是166的子集。其余81段以确认正例和未知位置为主。625条去重正例清单用于额外防误删，不能与112/166相加。完整定义见快照。

源码与JSON标注可随仓库共享；原视频、官方对象JSON、外部资产按允许渠道准备。模型和输入按各模块清单核对哈希，避免比较不同版本。正式推理不读取人工标签，标签仅用于训练和离线评测。

## 快速校验

Python 3.10及以上，仅校验共享数据与安全契约无需模型：

```bash
python development/evaluation/team_data.py
python -m unittest discover -s development/evaluation -p test_team_data.py
python tools/team_sync/verify_integration.py
```

比较已生成预测：

```bash
python development/evaluation/team_data.py --baseline old.json --candidate new.json --output audit.json
```

预测格式和评分范围见补标快照README。原正确触台有任何丢失时需要逐例复核，不能仅凭总体F1提高就替换模型。

## 开发与历史

其他规则候选和旧回放保存在各自experiments及development目录，代码归档不等于默认采用。历史阶段报告使用过不同数据和评分范围，不直接横向相减。当前结果属于已使用开发数据上的回归；独立泛化、官方输入兼容和GPU整链路性能分别验收。

每次团队提交保留来源清单、标签/模型/代码版本、固定预算和原方案对照。补标前先查已有记录。工作区可保留在个人目录，待共享成果复制到Git整合工作区后审阅提交，不整体覆盖仓库。

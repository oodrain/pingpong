# development/bounce —— 离线触台规则回放与诊断（负责人：D）

## 文件

- `replay_v0.py`：读 C 导出的轨迹缓存 `raw/*.jsonl`，逐行复刻官方 V0 触台规则
  （`official/participant/pingpang/participant/solution.py` 的 `_emit`：CONF_TH=0.75、
  y 局部极大 + 幅度门控 FALL_AMP=2.0 / RISE_AMP=1.0、MIN_GAP=10、发出时 -1 帧补偿、
  process 阶段 need_future=2 / finish 阶段 need_future=1），输出官方格式 `predictions.jsonl`。
- `compare_predictions.py`：比对两份 `predictions.jsonl`（逐条精确一致检查 + 差异清单 +
  ±1 帧 / 50px 容差互配统计）。

## 用法

```bash
python replay_v0.py --raw <C 的 raw 目录> --out <predictions.jsonl>
python compare_predictions.py a.jsonl b.jsonl
```

## 已知验证（2026-10-09）

- 对 C 的 `C_public15_20261009_v1` 全量回放：15 条视频共 2109 输入帧 → 109 个落点，
  与 C 的 `predictions.jsonl` **逐条精确一致**（帧号与坐标浮点级相同）。
- 与 A 的 `pingpang_baseline/predictions.jsonl` 的直接比对，待 A 提供文件后补记
  （C 的交付文档记载两者逐条相同）。

## 接口约定（源：C 的交付文档 `C_public15_20261009_v1/README.md`）

- `raw/`：每个输入帧一行；`target_frame_id = input_frame_id - 1`（-1 为启动预填）；
  精确回放用 `input_frame_id`、官方 `CONF_TH`，再沿用官方发出时的 -1 帧补偿。
- `trajectories/`：已按真实帧对齐并完成补偿，**不要再减 1**。
- 低置信度记录保留在缓存中；是否采用由规则决定，不能导出时删除。
- 官方不补推最后一帧（`missing_reason=tail_not_inferred`）。

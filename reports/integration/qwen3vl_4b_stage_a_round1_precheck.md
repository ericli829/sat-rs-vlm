# Qwen3-VL-4B Stage-A Round 1 Precheck

日期：2026-08-18

## 当前仓库

- 分支：`master`
- HEAD：`aecb31c9b207beb8ce1468286efccadf636bbb3c`
- 未执行 reset、checkout、commit 或 push。
- Round 1 相关改动保留为本地工作区修改。

## Round 0 产物审计

用户提供的结果目录：

`D:\Desktop\tzb-2026\results\qwen3vl_4b_stage_a_20260817_205641_reports`

该目录不是只包含 Round 0 的结果，实际包含：

`round_000`、`round_001`、`round_002`、`round_003`、`round_004`、`round_005`、`final_adapter`。

`stage_a_result.json` 的结构指纹为 Qwen3-VL-4B：

- `hidden_size=2560`
- `num_hidden_layers=36`
- `num_attention_heads=32`
- `vision_hidden_size=1024`
- `vision_depth=24`
- LoRA trainable parameters：`33,030,144`
- trainable ratio：约 `0.7388%`

本任务明确使用：

`round_000/adapter`

不能把该目录的 `final_adapter` 自动当作 Round 0，因为它对应历史 Round 5。

## 历史 provenance 风险

当前目录中的 `cycle_manifest.json` SHA256 与 `stage_a_result.json` 记录一致：

`b0e5df303708e5a07b81041f373e7de32519693400288e8eb3781342a2ac509c`

manifest 本身报告：

- 全周期 population：`176465`
- 全周期 unique samples：`176465`
- coverage：`100%`
- duplicate：`0`
- protected E3 overlap：`0`

但历史 `stage_a_result.json` 与当前 manifest 的 round 样本数不一致：

| Round | stage_a_result | cycle_manifest |
|---|---:|---:|
| Round 0 | 54550 | 54560 |
| Round 1 | 52211 | 51442 |

因此历史输出目录应视为“可审计但存在 round provenance 不一致”的结果。正式 Round 1
启动前必须在云端重新检查 `round_001_train.jsonl` 的行数与 SHA，并以云端当前
`cycle_manifest.json` 为唯一依据；不能直接复用历史目录中的 `round_001` 输出。

## 当前实现审计

- `scripts/training/run_qwen3vl_4b_stage_a.py`：负责 cycle 准备、round 串联、dry-run、真实 forward-only 和正式训练。
- `scripts/train_qwen3vl_lora.py`：负责 Qwen3-VL、processor、LoRA、assistant-only labels、task-weighted loss 和 Trainer。
- `src/sat_rs_vlm/training/config.py`：负责 Pydantic 配置和路径覆盖。
- `src/sat_rs_vlm/data/cyclic_training.py`：负责固定 seed 的 full-coverage cycle、source/task 分桶和 E3 泄漏保护。
- Round 0 adapter 通过 `lora.initial_adapter_dir` / `--initial-adapter` 注入；Round 1 普通 transition 不使用 `resume_from_checkpoint`。
- Round 1 保持 `task_weighted`，六类任务权重均为 `1.0`。
- vision tuning 保持关闭；LoRA 参数保持 `r=16`、`alpha=32`、`dropout=0.05`，target modules 不变。

## 本次实现

- 新增 `configs/train/qwen3vl_4b_stage_a_round1_4090.yaml`。
- Stage-A runner 支持显式 `--initial-adapter`，适配 parent run 与新输出 run 分离的场景。
- runner 在加载模型前校验 adapter 权重、LoRA 类型、base fingerprint，并拒绝 2B、H1/H2 adapter。
- runner 根据真实 `N`、batch、gradient accumulation 和 `WORLD_SIZE` 计算：
  `effective_batch`、`steps_per_epoch`、50% `mid_round_save_step` 和 `final_step`。
- `train_qwen3vl_lora.py` 新增 `--save-steps`，由 runner 动态覆盖配置。
- 每个 round 新增 `round_plan.json`，并在 `round_result.json` 记录 parent/output fingerprint、SHA、source/task 分布、保存点和 resume 状态。
- runner 默认拒绝覆盖已有 round 输出；已有历史结果必须使用新的 `--run-root`。

## Round 1 预期配置

- train file：`round_001_train.jsonl`
- parent：`round_000/adapter`
- learning rate：`1e-5`
- epochs：`1`
- per-device batch：`4`
- gradient accumulation：`4`
- 单卡 effective batch：`16`
- `resume_from_checkpoint`：`null`
- training-time generation evaluation：关闭

以当前历史 stage result 的 Round 1 样本数 `52211` 作为暂时参考时：

- `steps_per_epoch=3264`
- `mid_round_save_step=1632`
- `final_step=3264`

正式启动前应以云端实际 JSONL 行数重新计算，不使用这个暂时参考值替代云端事实。

## 验证结果

已完成：

- `python -m compileall -q src scripts`：通过。
- Stage-A runner 与训练配置回归测试：`19 passed`。

未在本机执行真实 Qwen3-VL-4B forward-only，因为当前本机没有该模型和云端数据资产。

## AutoDL 正式命令

以下命令使用新的输出根目录，保护历史 Round 0 到 Round 5：

```bash
cd /root/autodl-tmp/sat-rs-vlm
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export DATA_ROOT=/root/autodl-tmp/datasets
export OUTPUT_ROOT=/root/autodl-tmp/outputs
export ROUND_0_FINAL_ADAPTER=/root/autodl-tmp/outputs/<stage_a_run>/round_000/adapter
export ROUND1_RUN_ROOT=/root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1_$(date +%Y%m%d_%H%M%S)

bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh \
  --train-config configs/train/qwen3vl_4b_stage_a_round1_4090.yaml \
  --start-round 1 \
  --end-round 1 \
  --initial-adapter "$ROUND_0_FINAL_ADAPTER" \
  --run-root "$ROUND1_RUN_ROOT" \
  --dry-run

bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh \
  --train-config configs/train/qwen3vl_4b_stage_a_round1_4090.yaml \
  --start-round 1 \
  --end-round 1 \
  --initial-adapter "$ROUND_0_FINAL_ADAPTER" \
  --run-root "$ROUND1_RUN_ROOT" \
  --forward-only

bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh \
  --train-config configs/train/qwen3vl_4b_stage_a_round1_4090.yaml \
  --start-round 1 \
  --end-round 1 \
  --initial-adapter "$ROUND_0_FINAL_ADAPTER" \
  --run-root "$ROUND1_RUN_ROOT" \
  --skip-e2-eval
```

最后一条命令不在训练期间运行昂贵的 E2 generation。训练完成后，使用 Round 1 的中间
checkpoint 做 E1，使用 final adapter 做 E2；评测报告应同时保留 checkpoint provenance
和对应 tier SHA。

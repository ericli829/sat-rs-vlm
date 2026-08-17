# Qwen3-VL-4B Stage-A 多源全覆盖训练

## 目标与边界

Stage-A 从 `Qwen3-VL-4B-Instruct` base 直接开始，用 VRSBench 与 LEVIR-CC 做首次
多任务 task alignment。它不是 H1/H2、2B Replay 或困难样本专项训练。视觉编码器保持
冻结，训练参数仅为 LoRA；assistant-only mask 继续由 `Qwen3VLDataCollator` 构造。

```text
4B Base
  -> full-cycle bucket preparation
  -> round 0 adapter
  -> round 1 adapter
  -> ...
  -> final_adapter
  -> Unified E2 v2
```

## 两种选择语义

历史 `legacy_round_sampling` 会使用 `seed + round_index` 重新抽取 task quota，并对
LEVIR variant 做 modulo rotation。它可复现，但不同 round 可能重叠，也不能证明最终
覆盖完整 population。旧配置和旧脚本继续使用该行为。

新的 `cyclic_full_coverage` 将 VRSBench 每个 task 以固定 seed 独立打乱，再按配置中的
bucket size 切成不相交切片；LEVIR-CC 按图像对分组，对每组 caption variant 切片，
最后不足一组的 variant 原样保留，不回卷、不补样。`cycle_manifest.json` 对全局、source、
task 和 LEVIR variant 分别记录覆盖率、重复与遗漏。

训练 sampler 仍采用 batch-level `VRSBench,VRSBench,VRSBench,LEVIR-CC` 偏好。历史
`truncate` 会由较小 source 决定 epoch 长度；Stage-A 使用 `coverage_first`，完整 pattern
结束后排空各 source 尾部，因此尾部可以偏离 3:1，但每个 bucket 样本恰好暴露一次。

## 训练协议

- LoRA：`r=16`、`alpha=32`、`dropout=0.05`。
- targets：`q/k/v/o_proj` 与 `gate/up/down_proj`；每项必须实际命中。
- Loss：`task_weighted`，六类任务权重均为 `1.0`，不是 token mean。
- Detection：仅输出 normalized `[0,1]` 的 `label+bbox` JSON。
- Counting：仅输出整数；Scene：仅类别；VQA：仅短答案。
- LEVIR-CC 保持 free-form change caption，不改成二分类 target。
- 每个 bucket 训练 1 epoch；optimizer/scheduler 每轮重建，模型权重通过上一轮 adapter 串联。
- round 0 LR 为 `2e-5`，后续为 `1e-5`；数组耗尽后沿用最后一个值。
- RTX 4090 默认 batch 4、gradient accumulation 4，effective batch 为 16。只有真实 OOM
  才调整到 batch 2、accumulation 8。

完整 E3 v2 ID 是受保护 population。cycle 构建前必须读取
`data/evaluation/tiers_v2/evaluation_tiers_manifest.json`，任一 train ID 与 E3 重叠都会
立即失败。

## 产物

数据目录包含 `cycle_manifest.json`、`round_NNN_train.jsonl`、对应 report 与统一
`validation.jsonl`。训练 run 包含每轮 `adapter/`、`round_result.json`、最终
`final_adapter/` 与 `stage_a_result.json`。正式 cycle 完成后 runner 默认调用固定
Unified E2 v2；训练期间不运行完整 generation evaluation。

每个 adapter 的 `strategy_manifest.json` 保存 base model fingerprint、逐 target 命中数、
LoRA 可训练参数量和比例。后续 round 缺少 fingerprint 或 hidden size 等结构字段不一致时，
会在 PEFT 挂载前失败，从而阻止 2B adapter 误接到 4B。

## AutoDL 命令

```bash
cd /root/autodl-tmp/sat-rs-vlm
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export DATA_ROOT=/root/autodl-tmp/datasets
export OUTPUT_ROOT=/root/autodl-tmp/outputs
```

准备并验证完整 cycle：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh --prepare-only
```

配置/路径 dry-run 与真实双 source forward probe：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh --dry-run
bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh --forward-only
```

正式训练较久，建议后台运行：

```bash
RUN_ROOT="$OUTPUT_ROOT/qwen3vl_4b_stage_a_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh \
  --run-root "$RUN_ROOT" \
  > "$RUN_ROOT.launch.log" 2>&1 &
echo $! > "$RUN_ROOT.pid"
```

查看和停止：

```bash
tail -f "$RUN_ROOT.launch.log"
kill "$(cat "$RUN_ROOT.pid")"
```

从已有 run 的 round 2 继续：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh \
  --resume \
  --start-round 2 \
  --run-root /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_<timestamp>
```

只做 smoke/debug 时才可传 `--max-train-samples`，正式命令会拒绝该参数。使用
`--skip-e2-eval` 可把 E2 延后，但默认工作流会在 cycle 完成后评测。

## 为什么使用循环分桶

循环分桶避免一次性组织超大 epoch，同时提供独立 checkpoint、中断恢复、LEVIR caption
variant 轮换以及逐轮漂移分析。Stage-A 追求完整代表性 task alignment；H2 则根据已有模型
错误做 hardness-aware refinement，两者的数据选择假设和实验问题不同。

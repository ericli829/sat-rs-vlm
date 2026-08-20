# Qwen3-VL-4B Stage-A v2

## 设计

旧 fixed task bucket 会让低频任务在前两轮耗尽，而基于单 round 的 probe 会进一步造成静默
quota 失真。v2 不再把 round 文件视为完整训练 population：

```text
Canonical Legal Population
  |-- Balanced Probe
  |-- R0: full legal VRSBench, fresh strong LoRA
  `-- R1: full VRSBench + full/replayed LEVIR-CC, visual reinforcement
```

Canonical builder 复用正式 multisource normalization、图片路径改写和 prompt strengthening。
它从 tiers manifest 实际收集 E1/E2/E3 全部 ID，要求最终 train 与 protected eval 交集为零，
并写出 source input、prompt profile、样本数、task distribution 与 SHA256。canonical 文件不含
replay exposure。

## R0

- 起点：Qwen3-VL-4B base，fresh LoRA，不接受 initial adapter。
- 数据：`legal_vrs_train.jsonl` 全部 unique 样本，一轮。
- LoRA：r16、alpha32、dropout 0.05、七个 text projection targets。
- 优化：LR `1e-4`，batch 4，accumulation 4，effective batch 16，BF16/SDPA/cosine。
- 冻结：ViT、main/deepstack merger、patch embed 与全部非 LoRA base 参数。
- 保存：根据样本量动态计算约 0.5 epoch checkpoint 和 final adapter。

R0 的 `trainable_parameters.json` 若发现非 LoRA trainable 参数会立即失败。final adapter 完成
后先运行 Unified E1 v2，之后才允许进入 R1。

## R1

R1 必须从正式 R0 final adapter 继续，校验 PEFT 类型、r/alpha/targets、4B architecture
fingerprint、adapter 权重 SHA 和 `training_stage=qwen3vl_4b_stage_a_v2_r0`。

Stage2 固定数据保留全部 legal VRS 和全部 unique legal LEVIR。目标 LEVIR exposure 为
`ceil(VRS/3)`；仅当 unique LEVIR 不足时确定性 replay，使用 deep copy、synthetic exposure
ID、`replay_original_id` 与 replay provenance。若 unique LEVIR 已超过目标，不丢数据强求
3:1。`coverage_first` alternating sampler 运行前验证无遗漏、无重复、无 source starvation。

R1 只允许三个不重叠 optimizer groups：

| Group | Surface | LR |
|---|---|---:|
| LoRA | R0 adapter 参数 | `2e-5` |
| main merger | `visual.merger` | `1e-5` |
| vision blocks | 动态解析的最后 2 blocks | `2e-6` |

DeepStack、patch embed、早期 ViT blocks 和 LLM base 保持冻结。完整参数名、tensor 数、参数量
与 LR 写入 audit/manifest；视觉权重沿用现有 sidecar checkpoint 机制。

## 评测与恢复

R0 与 R1 使用相同 E1 配置、tier SHA、processor、prompt 和 generation protocol。请求 batch
为 4；首次 CUDA OOM 时清 cache 并回退到 2。如果 R1 首次暴露 batch-4 OOM，runner 会将
R0/R1 的 batch-4 产物归档，并以 batch 2 重跑两者，保证 paired comparison 一致。
E2 仅在显式 `--run-e2` 时执行。

`--resume` 区分两类状态：已完成 adapter 直接复用；中断训练使用最新 Trainer
`checkpoint-*` 恢复 optimizer/scheduler。R1 的 `--initial-adapter` 始终表示 R0 权重链，不
等同于 Trainer resume。状态写入 `run_manifest.json`。

## AutoDL 命令

```bash
cd /root/autodl-tmp/sat-rs-vlm
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export DATA_ROOT=/root/autodl-tmp/datasets
export OUTPUT_ROOT=/root/autodl-tmp/outputs
```

Canonical population 与 Stage2 prepare：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh --prepare-only
```

R0 dry-run；R1 dry-run 需提供一个已完成且契约合法的 R0 adapter：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh --dry-run
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh \
  --dry-run --r0-adapter /path/to/formal-r0-adapter
```

Forward-only 同理：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh --forward-only
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh \
  --forward-only --r0-adapter /path/to/formal-r0-adapter
```

一键正式运行与 resume：

```bash
RUN_ROOT="$OUTPUT_ROOT/qwen3vl_4b_stage_a_v2_$(date +%Y%m%d_%H%M%S)"
screen -dmS qwen4b-stage-a-v2 bash -lc "
  cd /root/autodl-tmp/sat-rs-vlm
  bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh \
    --run-root '$RUN_ROOT'
"

bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh \
  --run-root "$RUN_ROOT" --resume
```

只有正式流程允许显式关机；诊断模式永远忽略关机请求：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh \
  --run-root "$RUN_ROOT" --resume --shutdown
```

查看日志：`tail -f "$RUN_ROOT"/logs/stage_a_v2_*.log`。停止 screen 中前台进程可使用
`screen -S qwen4b-stage-a-v2 -X stuff $'\003'`；不要删除已有 Trainer checkpoint。

旧 `run_autodl_qwen3vl_4b_stage_a.sh`、旧 configs 和旧数据 SHA 均未改变，它们只用于历史
复现，不再是新 4B 正式训练推荐入口。

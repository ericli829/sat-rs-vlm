# Qwen3-VL-4B LR + Visual Merger 快速诊断实验

本实验矩阵用于回答两个工程问题：4B Stage-A 的 LoRA 学习率是否过小，以及冻结的 visual merger 是否限制 Scene、Counting、VQA 等视觉语义任务。它不是新的正式训练阶段，也不是从 base model 重新训练。

## 固定初始化与数据

所有实验都从同一个 Round1 final LoRA adapter 启动：

```text
Qwen3-VL-4B base + Round1 final LoRA
```

不会从已经 ViT-tuned 的 checkpoint 继续训练。已有 ViT checkpoint 只作为可选静态比较项，通过 `--existing-vit-checkpoint` 评测。

训练数据优先复用 `data/processed/experiments/qwen3vl_4b_vit_probe/train.jsonl`。启动前会检查 JSONL SHA、唯一 ID，以及与 E1/E2/E3 protected IDs 的交集。数据缺失时，runner 调用已有 deterministic probe builder；不会运行时随机重新采样。

所有模型使用 Unified Evaluation Tiers v2 的 E1，保持相同 evaluation manifest、prompt、processor、生成参数和 Evaluation v1.5 evaluator。E1 只作为快速方向诊断；正式结论仍使用 E2。

本轮 sweep 使用独立配置 `configs/eval/qwen3vl_4b_e1_sweep_4090.yaml`，默认
`eval_batch_size=4`、`group_by_task=true`。这只是吞吐调整，不改变 generation protocol。
runner 的第一次 E1 若出现 CUDA OOM，只执行一次 `4 -> 2` fallback；随后 Round1 baseline、
A1-A3、B1-B3 以及可选 ViT checkpoint 全部固定使用同一个 effective batch size。结果目录中
会写入 `evaluation_metadata.json`，记录 batch size、样本数、运行时间、samples/s、峰值显存
（若 evaluator 能提供）。历史 baseline/E2/E3 配置保持不变。

## 实验矩阵

| ID | 训练面 | LoRA LR | merger LR | ViT |
|---|---|---:|---:|---:|
| A1 | LoRA only | 2e-5 | - | frozen |
| A2 | LoRA only | 5e-5 | - | frozen |
| A3 | LoRA only | 1e-4 | - | frozen |
| B1 | LoRA + main merger | Phase A selected | 1e-5 | frozen |
| B2 | LoRA + main merger | Phase A selected | 3e-5 | frozen |
| B3 | LoRA + main merger + last2 ViT | Phase A selected | 1e-5 | last 2 blocks |

默认训练预算为 100 optimizer steps、batch 4、gradient accumulation 4、effective batch 16、BF16、gradient checkpointing、cosine scheduler 和 seed 42。OOM 时只对当前实验重试一次 batch 2 / accumulation 8。

`vision_tuning.enabled=true` 现在允许 `unfreeze_last_n_blocks=0`，但必须打开 main merger、DeepStack merger 或 patch embedding 中至少一项。B1/B2 因此是严格的 merger-only visual tuning；它们不会偷偷解冻 ViT。

## Phase A 选择逻辑

Phase A 每个实验单独从 Round1 启动并进行 E1。选择 Phase B LoRA LR 时使用：

```text
0.30 * counting exact
+ 0.15 * counting within-1
+ 0.25 * scene normalized
+ 0.20 * VQA normalized
+ 0.10 * detection mIoU
```

候选还必须满足 parse success 不低于 0.99，并且 Detection mIoU、LEVIR F1、Caption ROUGE-L 相对 Round1 不下降超过 0.04。全部候选违反 guard 时使用 5e-5，并在 `phase_a_selection.json` 中记录 `fallback_due_to_guard_failure`。3.45% 是历史 2B LoRA/Base 参考量级，不是本轮优化目标。

## 输出

每个实验单独写入：

```text
<output-root>/A1/
├── checkpoint/
│   ├── final adapter
│   ├── checkpoint-100/
│   ├── processor/
│   ├── strategy_manifest.json
│   └── trainable_parameters.json
├── resolved_training_config.yaml
├── experiment_status.json
├── train.log
└── evaluation_e1/
```

汇总目录包含：

```text
<report-dir>/
├── summary.json
├── summary.csv
├── summary.md
├── phase_a_selection.json
├── lora_analysis/<experiment>/
├── merger_analysis/<experiment>/
└── figures/
```

LoRA 分析包含 delta W、LoRA/Base ratio、projection/layer 统计、相对 Round1 的 cosine 和 relative change，以及 participation ratio、top-8 energy、r95。B1/B2/B3 额外分析 main merger sidecar 的 absolute/relative Frobenius update。缺少 matplotlib 时 CSV/JSON 仍是权威结果，不会使训练失败。

## AutoDL 命令

准备和校验资产：

```bash
python scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --prepare-only
```

不关机的一键正式矩阵：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.sh \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --output-root /ABS/PATH/TO/OUTPUT_ROOT \
  --report-dir /ABS/PATH/TO/REPORT_DIR
```

正式执行时不需要额外传 `--batch-size`；sweep 会从上述独立配置默认使用 4，并在
`reports/summary.json`、`sweep_status.json` 和每个 `evaluation_e1/evaluation_metadata.json`
中记录最终值。若发生 fallback，报告会包含：

```json
{
  "requested": 4,
  "effective": 2,
  "reason": "cuda_oom"
}
```

如果已有同一 Round1 adapter 的合法 E1，可以显式复用，runner 会校验 E1 tier/SHA；旧目录没有 provenance 时只给出 warning，不会修改旧评测目录：

```bash
python scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --baseline-evaluation-dir /ABS/PATH/TO/EXISTING_BASELINE_E1
```

带可选旧 ViT checkpoint 的版本：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.sh \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --existing-vit-checkpoint /ABS/PATH/TO/VIT_CHECKPOINT_100 \
  --shutdown-after-run
```

AutoDL 正式运行建议始终通过 `run_autodl_qwen3vl_4b_lr_merger_sweep.sh` 启动。它会复用
Stage-A 已验证的 `activate_autodl_python.sh` 和 `--shutdown-after-run` 处理，避免 screen
内退回 base Python。`--shutdown` 仍作为 Python 入口的兼容别名保留，但不建议绕过 shell
wrapper 直接使用；`--prepare-only`、`--dry-run`、`--forward-only` 永远不会关机。

编排 smoke：

```bash
python scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --max-steps 2 \
  --dry-run
```

服务器中断后恢复：

```bash
bash scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.sh \
  --config configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml \
  --initial-adapter /ABS/PATH/TO/ROUND1_FINAL_ADAPTER \
  --output-root /ABS/PATH/TO/OUTPUT_ROOT \
  --report-dir /ABS/PATH/TO/REPORT_DIR \
  --resume
```

`ANALYZED` 实验不会重训，只有 checkpoint 但未评测的实验从 evaluation 继续，失败实验默认不自动重试；需要显式加 `--retry-failed`。

## 如何阅读结果

先看 `summary.md` 的 Phase-A selector 和 guard，再看：

1. LoRA/Base ratio 是否随 LR 增加而增长；
2. Scene、Counting、VQA 是否同步改善；
3. B1/B2 是否损害 Detection、LEVIR、Caption；
4. B3 是否在弱任务和 Detection/LEVIR 之间形成互补；
5. `trainable_parameters.json` 与 optimizer group 是否只有期望的参数面。

通过快速 E1 后，候选模型应再使用同一个固定 E2 做标准比较。不要把 E1 结果直接当作最终模型结论。

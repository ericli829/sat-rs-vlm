# Qwen3-VL-4B ViT Last-2 Probe

这是一个受控诊断实验，不是新的正式训练阶段。起点必须是已经完成的
Qwen3-VL-4B Stage-A Round1 `final_adapter`，不是 base model，也不是 Round0。

## 实验变量

实验只改变一个视觉变量：在原 Round1 LoRA adapter 上动态解冻 ViT 最后 2 个
transformer blocks。LoRA `r=16`、`alpha=32`、dropout、target modules、assistant-only
loss、bbox `[0,1]` 协议、E1 JSONL、generation config 和图像预处理均保持不变。

第一版关闭 `main merger`、DeepStack merger 和 `patch_embed`，因此训练参数只有：

- Round1 LoRA adapter：`5e-6`
- ViT last-2 blocks：`1e-6`

当前 optimizer schema 仍要求未使用的 merger 学习率位于两者之间；配置中的
`visual_merger_lr=2e-6` 不会产生 merger 参数组。

## 数据与泄漏保护

`build_qwen3vl_4b_vit_probe_dataset.py` 从训练 population 中固定 seed=42、无放回抽样，
目标为 VRSBench:LEVIR-CC 约 4500:1500，并尽量按 VRSBench 五类任务各 900 条组织。
它会读取 `data/evaluation/tiers_v2/evaluation_tiers_manifest.json` 的 E1/E2/E3 IDs，
发现任何 overlap 立即失败。输出目录为：

```text
data/processed/experiments/qwen3vl_4b_vit_probe/
├── train.jsonl
└── manifest.json
```

`manifest.json` 记录 source/task 分布、输入文件 SHA256、输出 SHA256、唯一 ID 数和
protected evaluation overlap。E1/E2/E3 样本仍然只用于评测。

## AutoDL 流程

先设置模型、数据和 Round1 adapter：

```bash
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export DATA_ROOT=/root/autodl-tmp/data
export OUTPUT_ROOT=/root/autodl-tmp/outputs
export PYTHONUNBUFFERED=1
```

准备数据并执行泄漏审计：

```bash
python scripts/training/run_autodl_qwen3vl_4b_vit_probe.py \
  --config configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml \
  --initial-adapter /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --prepare-only
```

训练前 dry-run 和 real forward-only：

```bash
python scripts/training/run_autodl_qwen3vl_4b_vit_probe.py \
  --config configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml \
  --initial-adapter /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --dry-run

python scripts/training/run_autodl_qwen3vl_4b_vit_probe.py \
  --config configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml \
  --initial-adapter /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --forward-only
```

5-step smoke：

```bash
python scripts/training/run_autodl_qwen3vl_4b_vit_probe.py \
  --config configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml \
  --initial-adapter /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --max-steps 5 --skip-eval
```

200-step probe 和自动 E1 baseline/midpoint/final 评测：

```bash
python scripts/training/run_autodl_qwen3vl_4b_vit_probe.py \
  --config configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml \
  --initial-adapter /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --max-steps 200
```

训练输出包括 `checkpoint-100`、`checkpoint-200`、`trainable_parameters.json`、
`strategy_manifest.json` 和 `visual_trainable_weights.safetensors`。评测报告位于：

```text
reports/experiments/qwen3vl_4b_vit_probe_last2/
├── baseline_e1/
├── checkpoint100_e1/
├── checkpoint200_e1/
├── comparison.json
└── comparison.md
```

中间 checkpoint 的 sidecar、processor 和 manifest 由 runner 补齐，因此可以直接被
现有 `evaluate_rs_vlm.py` 加载。也可以单独运行：

```bash
python scripts/evaluation/compare_vit_probe.py \
  --baseline-dir reports/experiments/qwen3vl_4b_vit_probe_last2/baseline_e1 \
  --checkpoint100-dir reports/experiments/qwen3vl_4b_vit_probe_last2/checkpoint100_e1 \
  --checkpoint200-dir reports/experiments/qwen3vl_4b_vit_probe_last2/checkpoint200_e1 \
  --output-dir reports/experiments/qwen3vl_4b_vit_probe_last2
```

## 如何判断

重点看 Detection 的 small/medium/large、Counting 的 `6-10` 和 `11+`、Scene，
同时检查 Caption、VQA、LEVIR 是否退化。结果分类为：

- `STRONG POSITIVE`：Detection/Counting/Scene 至少两个明确改善，且 broad tasks 无系统性退化；
- `PROMISING`：一个视觉敏感任务明显改善，其余基本持平；
- `NEGATIVE`：大部分任务无改善或 broad capability 明显退化。

E1 只用于快速方向判断，不用于最终统计结论。只有 positive/promising 才进入
last-4、打开 main merger 或 E2 复核；negative 时不要自动扩大 ViT 解冻范围。

## 已知限制

当前 probe 不测试 last-4、bbox_2d、LoRA+、AdaLoRA、视觉分辨率调整或新的 loss。
4090 上应记录 `trainable_parameters.json`、optimizer group 和 peak VRAM；如果 batch=4
OOM，保持 effective batch=16，改为 batch=2、gradient accumulation=8。


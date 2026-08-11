# 量化敏感度分析

## 目标

敏感度分析回答“哪些 Linear 层或组件量化后导致 Evaluation v1.5 主指标下降”。它不使用
Keyword Hit Rate 作为精度代理，也不把延迟改善混入准确率分数。

## 算法

1. 扫描实际模型中的 `torch.nn.Linear` 完整模块名。
2. 根据多个命名候选分为视觉编码器、multimodal projector、语言模型和其他组件，或按
   Transformer block 形成可解释的层组。
3. 在 GPU 上加载 FP16 baseline；对每个组重新加载模型，只将该组转换为 bitsandbytes
   `Linear8bitLt`，其余层保持 FP16。
4. 使用与 baseline 相同的样本和生成参数得到 predictions。
5. predictions 进入 Evaluation v1.5，提取 IoU、计数准确率、VQA 归一化准确率、token F1、
   ROUGE-L/chrF 和变化事件 F1 等可用主指标。
6. 对“越大越好”和“越小越好”的指标分别计算有害方向变化，先在任务内加权，再在任务间
   加权，避免某个任务因为指标数量多而支配总分。
7. 强制检查 baseline 与量化组的成功样本 ID 和失败率，再按分数排序。
8. 生成 JSON、Markdown、进度状态、速度统计和可选 PNG 图表。

配置中的 `include_modules` 和 `skip_modules` 可以使用 `vision_encoder` 等组件名，也可以对完整
模块名做包含匹配。
因此默认跳过视觉编码器时仍能单独测试 `visual.merger` 这类 multimodal projector。没有匹配
层、指定层不存在或目标层实际未转换为 `Linear8bitLt` 时立即失败，不生成虚假成功报告。

## Dry-run

只检查配置、模型目录占位、JSONL 和图片，不加载模型。该检查保留轻量 CPU smoke 配置，不代表
正式敏感度脚本仍支持 CPU 量化：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/quantization_sensitivity_smoke.yaml \
  --dry-run
```

## 真实 GPU 实验

绘图前安装可选依赖：`python -m pip install -e ".[plot]"`。

先设置 merge 后模型目录：

```bash
export MERGED_MODEL_DIR=/root/autodl-tmp/models/qwen3vl_vrsbench_levircc_merged

# 组件级初筛
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_component_autodl.yaml \
  --plot

# Transformer 层组复测
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_layer_autodl.yaml \
  --plot
```

## 当前阶段推荐流程

以下流程使用准备好的 VRSBench + LEVIR-CC 联合验证集，并通过固定 `samples_per_task` 保证每次
实验选择相同样本。

1. 运行双数据集 base smoke，确认模型、图片和显存配置可用。
2. 完整评测 base 模型，建立两个数据集的未微调基线。
3. 对 merge 后 LoRA 模型执行组件级 GPU bitsandbytes INT8 敏感度初筛。
4. 根据组件报告调整 `include_modules`，按 Transformer block 复测候选组件。
5. 对完整模型执行 BF16 与 bitsandbytes INT8 的固定样本精度、显存和延迟对比。

## GPU 层敏感度

GPU 模式使用 bitsandbytes LLM.int8。每次加载模型时，脚本通过
`llm_int8_skip_modules` 保持非目标层为 FP16，只把当前层组转换为 `Linear8bitLt`。加载后会
验证目标层和跳过层的实际类型；若 Transformers 没有按配置转换，实验立即失败，避免把 FP16
误报成 INT8。

脚本还会自动排除与其他参数共享权重的 Linear 层，例如 Qwen3-VL 通常与词嵌入绑定的
`lm_head`。bitsandbytes 不能安全量化该类层；它们会在报告的
`automatically_skipped_tied_linear_modules` 字段中列出。当前 GPU 配置统一使用 FP16，避免
LLM.int8 内核在每次 MatMul 时将 BF16 激活值临时转换为 FP16。

该模式同时记录精度、CUDA 峰值显存和单样本延迟：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_layer_autodl.yaml \
  --plot
```

GPU 模式需要 `bitsandbytes`，并且仍会为每个层组重新加载一次模型。报告中的局部量化速度主要
用于判断某个层组是否引入明显开销；最终部署速度仍应使用完整 bnb INT8 配置确认。

GPU 配置将 `benchmark.inference_batch_size` 设为 16。评测按任务类型分桶后再批量生成，避免
不同 `max_new_tokens` 的任务混在同一批；报告中的延迟改为 `batch_amortized_per_sample`，即总
批处理时间除以批内样本数，并不表示单请求时延。

组件初筛：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_component_autodl.yaml \
  --plot
```

层组复测：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_layer_autodl.yaml \
  --plot
```

完整 GPU bitsandbytes INT8 确认：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/full_bnb_int8_multidataset_autodl.yaml
```

## 输出

```text
reports/evaluation/quantization/sensitivity/
├── baseline/
├── groups/<group>/
├── sensitivity_report.json
├── sensitivity_report.md
├── sensitivity_progress.json
└── figures/
```

报告中的高敏感组是混合精度候选提示，不是对显存、延迟或部署收益的精确预测。正式结论应
扩大样本量、固定 seed，并在目标卫星计算平台或等价硬件上复测。

# 量化敏感度分析

## 目标

敏感度分析回答“哪些 Linear 层或组件量化后导致 Evaluation v1.5 主指标下降”。它不使用
Keyword Hit Rate 作为精度代理，也不把延迟改善混入准确率分数。

## 算法

1. 扫描实际模型中的 `torch.nn.Linear` 完整模块名。
2. 根据多个命名候选分为视觉编码器、multimodal projector、语言模型和其他组件，或按
   Transformer block 形成可解释的层组。
3. 对每个组重新加载 FP32 模型，只量化该组，避免同时深拷贝两个 2B 模型。
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
层、指定层不存在或实际未发生 dynamic quantization 时立即失败，不生成虚假成功报告。

## Dry-run

只检查配置、模型目录占位、JSONL、图片和后端能力，不加载模型：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/quantization_sensitivity_smoke.yaml \
  --dry-run
```

## 真实组件实验

绘图前安装可选依赖：`python -m pip install -e ".[plot]"`。

先在 `quantization_eval.yaml` 中设置 `model.merged_model`、数据路径和样本数：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/quantization_eval.yaml \
  --method component_wise \
  --plot
```

## 当前阶段推荐流程

以下流程使用准备好的 VRSBench + LEVIR-CC 联合验证集，并通过固定 `samples_per_task` 保证每次
实验选择相同样本。

1. 运行双数据集 base smoke，确认模型、图片和显存配置可用。
2. 完整评测 base 模型，建立两个数据集的未微调基线。
3. 对 merge 后 LoRA 模型执行组件级 CPU dynamic INT8 敏感度初筛。
4. 根据组件报告调整 `include_modules`，按 Transformer block 复测候选组件。
5. 对完整模型执行 FP32 与 CPU dynamic INT8 的固定样本精度和延迟对比。
6. 如果目标平台是 4090，再执行 BF16 与 bitsandbytes INT8 的 GPU 对比。

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

完整 CPU INT8 确认：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/full_int8_multidataset_autodl.yaml
```

可选 GPU bitsandbytes INT8 确认：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/full_bnb_int8_multidataset_autodl.yaml
```

逐层组实验：

```bash
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/quantization_eval.yaml \
  --method layer_wise
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

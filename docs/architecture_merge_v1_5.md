# Master + Evaluation v1.5 + dev-dqt 架构合并说明

## 来源

| 职责 | 来源提交 | 合并方式 |
|---|---|---|
| 主工程、LoRA、推理、AutoDL、数据和可靠性 | `master@da9f6d9` | 保持现有实现 |
| 多任务评估、语义诊断、配对比较和绘图 | `feature/evaluation-v1.5@4f66174` | 选择性迁入并接入现有生成入口 |
| AutoDL 4090 参数、INT8 和敏感度思路 | `dev-dqt@af6ddba` | 重构到统一配置与顶层量化模块 |

本次没有执行普通 `git merge`。分支中的已生成 predictions、实验 reports、processed data、
硬编码 Windows/AutoDL 路径和以 Keyword Hit Rate 为主指标的旧脚本没有迁入。

## 最终模块边界

```text
真实模型/Checkpoint
        |
        v
predictions.jsonl
        |
        v
sat_rs_vlm.evaluation.runner (contract v1.5)
        |----------------------|
        v                      v
metrics.json             evaluated_predictions.jsonl
        |
        v
evaluation.comparison -> comparison.json -> evaluation.plotting
```

量化只修改模型加载和 Linear 权重表示，不实现另一套指标：

```text
sat_rs_vlm.quantization.quantizer
        |
        v
quantization.benchmark -- baseline/INT8 predictions
        |
        v
同一个 Evaluation v1.5 runner 和 comparison
```

## 兼容性

- `scripts/train_qwen3vl_lora.py`、本地推理、数据格式和 checkpoint loader 保持不变。
- `scripts/evaluate_rs_vlm.py` 保留原命令和旧 summary 路径，但 summary 内容来自 v1.5。
- `sat_rs_vlm.compression.quantization.*` 仅重导出顶层实现，旧 Python import 不会立即失效。
- `quantize_int8*.py`、`quantize_lora_int8_cpu.py` 和 `quantize_merged_model.py` 是薄入口。
- 原 `configs/compression/` 已迁到 `configs/quantization/`；Makefile 和文档使用新路径。

## 故意不保留的实现

- `scripts/eval_quantized_model.py`：不保留，量化输出进入统一 Evaluation v1.5。
- dev-dqt 内置 VRSBench 原始 JSON 读取：不保留，继续使用项目 messages JSONL Dataset。
- Keyword Hit Rate 敏感度公式：不保留，改用 v1.5 主任务指标退化。
- 动态 INT8 整模型可部署声明：不保留；未完成 reload smoke 时明确标记 benchmark-only。
- 分支中已经生成的模型结果：不作为新主分支事实，真实结果需要按当前配置重新运行。

## 验证层级

1. 默认 pytest 使用 fixture、fake module tree 和 toy Linear，不下载模型、不要求 GPU。
2. dry-run 校验 YAML、环境变量、模型/数据路径和图片存在性。
3. 离线 fixture 验证 baseline、quantized、comparison 与 figures 全链路产物。
4. 真实 FP32/INT8 与敏感度结论必须在本地模型或 AutoDL 环境显式运行。

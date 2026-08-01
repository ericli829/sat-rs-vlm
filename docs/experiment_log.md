# 第一版实验记录

## 实验编号

`V1-SMOKE-001`

## 目标

验证本地 Qwen3-VL-2B、VRSBench 转换数据、LoRA 训练脚本和 adapter 工件流转可以在同一条端到端链路中运行。该实验是工程 smoke test，不用于报告模型能力或 benchmark 排名。

## 基座与数据

| 项目 | 记录 |
| --- | --- |
| 基座模型 | `${LOCAL_MODEL_DIR}` |
| 原始数据 | `${DATA_ROOT}`（VRSBench 根目录） |
| 训练 JSONL | `data/processed/qwen3vl_train.jsonl`，142390 条 |
| 验证 JSONL | `data/processed/qwen3vl_val.jsonl`，62918 条 |
| 图片根目录 | `${DATA_ROOT}`（VRSBench 根目录） |

## 训练方法

- LoRA，冻结视觉编码器。
- 最大序列长度：1024。
- smoke 配置仅采样少量数据，`max_steps=2`。
- 目标模块、秩和 alpha 以 `configs/train/qwen3vl_local_smoke.yaml` 为准。

## 已验证结果

| 检查项 | 结果 |
| --- | --- |
| VRSBench 转换 | 通过，生成训练/验证 Qwen3-VL JSONL |
| 资产校验 | 通过 |
| 单 batch 前向传播 | 通过 |
| 两步 LoRA smoke 训练 | 通过 |
| 训练时长 | 约 27.42 秒 |
| 最终训练 loss | 约 21.7669 |
| 峰值显存 | 约 4834 MiB（RTX 4060） |
| adapter 输出 | `checkpoints/smoke/qwen3vl-local-smoke/` |

## 评估状态

早期评估输出曾出现全空预测，该结果不能作为模型质量结论。原因是评估输入没有按 generation 场景构造，且 batch 存在 CPU/CUDA 设备混用。当前代码已改用 generation collator，移除参考答案、补充 generation prompt，并在调用 `generate()` 前迁移全部模型输入。

需要使用修复后的 `scripts/evaluate_rs_vlm.py` 重新执行评估。smoke adapter 只有两步训练，重跑得到的分数仍只用于验证链路，不代表正式模型性能。

## 下一步

1. 以独立配置运行完整 VRSBench LoRA 实验，并记录数据采样、随机种子、超参数和 adapter 版本。
2. 对正式 adapter 重跑验证集，按任务补充 mAP、CIDEr、BLEU/ROUGE、IoU 与变化检测指标。
3. 对比 LoRA、QLoRA、蒸馏和量化配置，记录显存、延迟、精度和容错影响。

# 第一版实验记录

## 实验编号

`V1-SMOKE-001`

## 目标

验证本地 Qwen3-VL-2B、VRSBench 转换数据、LoRA 训练脚本和 adapter 工件流转可以在同一条端到端链路中运行。该实验是工程 smoke test，不用于报告模型能力或 benchmark 排名。

## 基座与数据

| 项目 | 记录 |
| --- | --- |
| 基座模型 | `D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct` |
| 原始数据 | `F:\VIT-data\VRSBench` |
| 训练 JSONL | `data/processed/qwen3vl_train.jsonl`，142390 条 |
| 验证 JSONL | `data/processed/qwen3vl_val.jsonl`，62918 条 |
| 图片根目录 | `F:\VIT-data\VRSBench` |

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

---

## 实验编号

`V2-AUTODL-4090-FULL`

## 目标

在 AutoDL 云平台上使用 RTX 4090 GPU 进行完整的 LoRA 训练，验证大规模数据训练效果，并与本地 CPU smoke 训练结果进行对比。

## 基座与数据

| 项目 | 记录 |
| --- | --- |
| 基座模型 | `/root/autodl-tmp/models/Qwen3-VL-2B-Instruct` |
| 原始数据 | `/root/autodl-tmp/datasets/VRSBench` |
| 训练 JSONL | `data/processed/qwen3vl_train.jsonl`，142390 条 |
| 验证 JSONL | `data/processed/qwen3vl_val.jsonl`，62918 条（评估采样 1024 条） |
| 图片根目录 | `/root/autodl-tmp/datasets/VRSBench` |

## 训练方法

- LoRA (r=16, alpha=32, dropout=0.05)，冻结视觉编码器。
- 最大序列长度：1024。
- 完整训练：2 epochs，batch_size=16，gradient_accumulation=1。
- 学习率：0.0001，warmup_ratio=0.03，cosine 调度。
- 梯度检查点：启用。
- 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj。

## 硬件环境

| 项目 | 记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 (24564 MB) |
| 平台 | AutoDL 云服务器 |
| 训练精度 | BF16 |

## 训练结果

| 指标 | 数值 |
| --- | --- |
| 总训练步数 | 17800 |
| 训练 Epoch | 2.0 |
| 最终训练 Loss | 0.7175 |
| 平均训练 Loss | 0.7338 |
| 最终验证 Loss | 0.5522 |
| 训练时长 | 24110.9 秒（约 6.7 小时） |
| 训练样本/秒 | 11.811 |
| 峰值显存 | 18214 MB |
| 峰值保留显存 | 20782 MB |
| 可训练参数 | 17.4M / 2.14B (0.81%) |

## 验证 Loss 变化趋势

| Step | Eval Loss |
| --- | --- |
| 1000 | 0.6612 |
| 2000 | 0.6309 |
| 3000 | 0.6082 |
| 4000 | 0.6016 |
| 5000 | 0.5955 |
| 6000 | 0.5878 |
| 7000 | 0.5806 |
| 8000 | 0.5708 |
| 9000 | 0.5676 |
| 10000 | 0.5670 |
| 11000 | 0.5630 |
| 12000 | 0.5606 |
| 13000 | 0.5576 |
| 14000 | 0.5570 |
| 15000 | 0.5546 |
| 16000 | 0.5529 |
| 17000 | 0.5522 |
| 17800 | 0.5522 |

## 与本地 CPU Smoke 训练对比

| 指标 | 本地 CPU Smoke (V1) | AutoDL 4090 Full (V2) | 变化 |
| --- | --- | --- | --- |
| 训练设备 | Intel Core Ultra 7 155H (CPU) | RTX 4090 (GPU) | GPU 加速 |
| 训练步数 | 50 | 17800 | 356× |
| 训练样本 | 318 | 142390 | 448× |
| 最终训练 Loss | 6.629 | 0.717 | -89.2% |
| 最终验证 Loss | N/A | 0.552 | — |
| 训练时长 | 768.3 秒 (12.8 min) | 24110.9 秒 (6.7 h) | 31.4× |
| 峰值显存 | 0 (CPU) | 18214 MB | — |
| 可训练参数 | 17.4M (0.81%) | 17.4M (0.81%) | 相同 |

## 评估状态

- 完整训练 adapter 已保存至 `checkpoints/lora/autodl_4090_full/`
- 评估配置已创建：`configs/eval/qwen3vl_autodl_4090_eval.yaml`
- 待执行本地评估以验证模型质量

## 工件清单

| 文件 | 路径 |
| --- | --- |
| Adapter 权重 | `checkpoints/lora/autodl_4090_full/adapter_model.safetensors` |
| Adapter 配置 | `checkpoints/lora/autodl_4090_full/adapter_config.json` |
| 训练配置 | `checkpoints/lora/autodl_4090_full/training_config.yaml` |
| 训练报告 | `checkpoints/lora/autodl_4090_full/train_report.json` |
| 训练状态 | `checkpoints/lora/autodl_4090_full/trainer_state.json` |
| Processor | `checkpoints/lora/autodl_4090_full/processor/` |

## 下一步

1. 使用本地环境运行评估，验证模型在各遥感任务上的表现。
2. 对比 LoRA 与 QLoRA 在相同数据上的训练效率和模型质量。
3. 尝试不同的 LoRA 秩和 alpha 组合，优化模型性能。
4. 评估模型在变化检测等新任务上的泛化能力。

# SHC 结构化微调实验

## 共享训练协议

`Qwen3VLDataCollator` 分别编码 generation prompt 与完整对话，根据有效 token 长度只保留
assistant 答案 labels。user prompt、图像 token 和 padding 均为 `-100`；left/right padding、
单图和多图使用同一逻辑。截断后没有 assistant token 时抛出包含 sample id 的异常。

Detection 监督答案为 `{"label":"...","bbox":[x1,y1,x2,y2]}`，坐标目标格式固定为
`normalized_0_1`。Counting 为 `{"count":n}`。模板定义在
`src/sat_rs_vlm/data/prompt_templates.py`，parser 定义在 `data/task_protocol.py`。

## 实验矩阵

| ID | 数据组成 | 采样方式 | LoRA | 目的 |
| --- | --- | --- | --- | --- |
| E0 | 均衡验证集 | 无训练 | 无 adapter | zero-shot 下限 |
| E1 | balanced quota | uniform | r=16 | 结构化 LoRA 基线 |
| E1b | 与 E1 相同 | uniform | r=32 | 只比较容量 |
| E1d-data | detection/counting quota | uniform | r=16 | 只比较数据组成 |
| E1d-sampler | balanced quota | weighted | r=16 | 只比较采样概率 |
| E1d-combined | detection/counting quota | weighted | r=16 | 联合干预 |

先生成派生数据：

```bash
python scripts/prepare_e1_datasets.py
```

这些 JSONL 位于 `data/processed/e1/` 并被 Git 忽略。仓库只保存生成脚本、配额、配置和统计，
不保存 8k/1k 派生数据。

训练与评测示例：

```bash
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1_balanced.yaml
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e0_zeroshot.yaml
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e1.yaml
```

所有实验应设置 `MODEL_ROOT`、`DATA_ROOT`、`OUTPUT_ROOT`，并使用相同 seed、验证集和生成参数。
当前本地只验证了协议、采样和配置代码；真实 E0/E1/E1b/E1d 结果需重新运行后才能报告。

# 实验记录

## 实验编号

EXP-0001

## Base Model

Qwen/Qwen3-VL-8B-Instruct

## Dataset Version

sample-v0，后续替换为 VRSBench / MME Real RS / XLRS-bench / LEVIR-CC 转换结果。

## Training Method

QLoRA，默认冻结视觉编码器。

## LoRA 参数

- r: 16
- alpha: 32
- dropout: 0.05
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 数据量

- train: 待填写
- val: 待填写
- test: 待填写

## 训练命令

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml
```

## 评测结果

- exact_match: 待填写
- keyword_hit_rate: 待填写
- counting_mae: 待填写
- valid_json_rate: 待填写

## 问题记录

- 是否出现显存不足：待填写
- 是否需要降低 max_seq_length：待填写
- 是否需要更换更小模型：待填写

## 下一步计划

- 接入真实遥感 benchmark。
- 增加 mAP、CIDEr、BLEU/ROUGE 等任务指标。
- 尝试更小模型、蒸馏模型和量化部署。

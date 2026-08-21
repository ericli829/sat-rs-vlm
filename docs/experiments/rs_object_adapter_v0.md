# RS Object Adapter v0

## 目的与假设

这是一个严格受控的 object-centric 视觉适配实验，不是新的 Qwen3-VL 全模型微调
阶段。实验假设是：在当前 Qwen3-VL-4B Round1 checkpoint 的视觉表征上，增加一个
小型、类别条件化的 object proposal adapter，可能改善遥感目标计数和定位。Qwen
视觉塔、原始 merger、Qwen LLM 和原有 LoRA 都保持冻结；只有新 adapter 参数更新。
这样可以把结论限定为 object adapter 的增益，而不是把语言生成或 ViT 重新训练混在
一起。

当前 4B Round1 E2 counting reference（历史结果，只作比较基线，不写入代码）为：
`n=377`、exact `0.5385`、within-1 `0.8090`、MAE `1.021`、RMSE `2.109`、
bias `-0.596`；6-10 count 子集 MAE `2.52`、bias `-2.35`。建议将实验视为有
价值的条件是总体 MAE <= `0.94`、6-10 子集 MAE <= `2.22`，且 within-1 不下降
超过 1 个百分点。proposal IoU 仅是上界诊断，不能直接等价为 Qwen 生成检测指标。

## 数据与防泄漏

唯一训练来源是：

`data/processed/multisource/qwen3vl_4b_stage_a_v2/legal_vrs_train.jsonl`

唯一保护来源是：

`data/processed/multisource/vrsbench_levircc_eval_full.jsonl`

builder 先用完整 portable image identity 做集合差集，再按 `(image, class)` 聚合。
同图同类 IoU >= 0.95 的检测框去重，结果按 sample id 排序。数据划分使用
`sha256(seed + image)` 的 image-level 95/5 稳定划分，训练和内部验证不会共享图片。
没有合法解析的数据不会被猜测修复。

counting 类别先读取 `target_class/object_class/category/label` metadata；没有 metadata
时，只从 `How many <target> ...`、`What is the (total) number of <target> ...`
这两类 single-class exact-cardinality 问题的 target 开头做确定性 alias 匹配，不再扫描
整句。属性限定（如 `large planes`、`small vehicles`）、比较和二值问题不属于 v0
监督；不能把 `cars`、`boats`、子集飞机等粗暴映射到父类，否则会污染 full-set /
partial-set 语义。少量 VRSBench taxonomy 命名差异使用固定等价 alias 归一化，不使用
模糊匹配或语义猜测。resolution rate 仍以 eligible exact-cardinality 问题为分母，hard
blocker 仍为 < 90%；无匹配或多匹配样本会保留在审计诊断中。

输出固定为：

```text
data/processed/rs_object_adapter_v0/
├── train.jsonl
├── val.jsonl
├── class_vocab.json
├── audit.json
└── manifest.json
```

训练中不得使用 E2 或最终评测数据；E2 只由
`data/evaluation/tiers_v2/evaluation_tiers_manifest.json` 解析，不能在 evaluator
中随机抽样。

## 监督类型与损失

每个 `(image, class)` pair 只使用以下监督类型：

| 类型 | 语义 |
| --- | --- |
| `full_set` | D == N，包括 N == 0；允许 full-set negatives，负 query 权重为 0.1 |
| `partial_set` | 0 < D < N；只监督已匹配的正框，不把未匹配框当负例 |
| `count_only` | D == 0, N > 0；只做 count 和二值化约束 |
| `detection_only` | 没有 count 但 D > 0；不制造负监督 |
| `conflict` | D > N 或 count 冲突；排除并记录 |

Hungarian 匹配使用 detached CPU cost：`5 * L1 + 2 * (1-GIoU) + positive BCE`。
loss 先按 sample 对每个启用的 component 归一化，再做 batch mean，避免答案/框数量
直接改变样本权重。总损失固定为：

```text
1.0 objectness + 5.0 bbox L1 + 2.0 GIoU + 1.0 count + 0.01 binarization
```

`partial_set` 和 `count_only` 才启用 binarization；不新增 task-specific loss 或
其他 hidden loss。

## Adapter 结构

从动态解析的 Qwen3-VL visual module 中确认恰好 24 个 blocks，并 hook `[5, 11, 17,
23]` 的 patch hidden states。每层执行独立 `LayerNorm + Linear(1024 -> 256)`，用
可学习 softmax 融合；加入 normalized 2D patch center 和 class embedding，之后用
64 个 query、2 层 Transformer decoder、8 heads、FFN 1024 输出 objectness 与
sigmoid `cxcywh`。视觉 token 数必须和 `image_grid_thw` 相符，否则立即失败。

## 命令

先在 AutoDL 或已准备好正式数据/模型的机器上设置路径并构建数据：

```bash
export DATA_ROOT=/root/autodl-tmp/data
python scripts/data/build_object_adapter_v0_data.py \
  --config configs/experiments/rs_object_adapter_v0_4090.yaml
```

如果审计被阻断，应先处理 `audit.json` 中的具体原因；`--allow-blocked` 只用于查看
中间资产，不得用于训练。

训练起点必须是现有 4B Round1 adapter：

```bash
python scripts/training/train_object_adapter_v0.py \
  --config configs/experiments/rs_object_adapter_v0_4090.yaml \
  --checkpoint-dir /path/to/qwen3vl_4b_round1/final_adapter
```

两步真机 smoke（只在数据审计通过后运行）：

```bash
python scripts/training/train_object_adapter_v0.py \
  --config configs/experiments/rs_object_adapter_v0_4090.yaml \
  --checkpoint-dir /path/to/qwen3vl_4b_round1/final_adapter \
  --max-train-groups 8 --max-steps 2
```

`--dry-run` 会加载并审计 R1 视觉结构，但不会 optimizer step；checkpoint 只保存
新 adapter 的 `adapter_model.safetensors`、manifest 和 class vocabulary，不保存
Qwen 权重、optimizer 或 resume state。

使用 epoch 2 adapter 在固定 E2 上评测：

```bash
python scripts/evaluation/evaluate_object_adapter_v0.py \
  --config configs/experiments/rs_object_adapter_v0_4090.yaml \
  --checkpoint outputs/experiments/rs_object_adapter_v0_4090/checkpoint_epoch_2 \
  --r1-checkpoint-dir /path/to/qwen3vl_4b_round1/final_adapter \
  --batch-size 4
```

Object Adapter checkpoint 只包含自身的 adapter 权重；评测会从
`adapter_manifest.json` 的 `source_r1_checkpoint` 加载 R1 Qwen/PEFT 模型和
visual sidecar，也可以用 `--r1-checkpoint-dir` 显式覆盖路径。输出包括
`predictions.jsonl`、`e2_metrics.json` 和 `evaluation_metadata.json`，其中记录 E2
SHA、样本数、运行时间和源 R1 manifest。内部 val 每 epoch 保存
`val_epoch_1.json` / `val_epoch_2.json`。

## AutoDL 训练后 E1 流程

需要在 AutoDL 上执行正式训练并在训练完成后自动做一次 E1 时，使用独立
launcher。它会先校验 E1 manifest 和 SHA，再执行数据审计、训练、最新
`checkpoint_epoch_*` 选择和 E1 评测。只有显式传入 `--shutdown-after-run` 才会
在成功或失败后尝试关机；关机实现优先执行 Python 的
`os.system("/usr/bin/shutdown")`，再使用兼容性 fallback。

```bash
bash scripts/training/run_autodl_rs_object_adapter_v0_e1.sh \
  --r1-checkpoint-dir /root/autodl-tmp/outputs/qwen3vl_4b_stage_a_round1/final_adapter \
  --data-root /root/autodl-tmp/datasets \
  --output-root /root/autodl-tmp/outputs \
  --run-root /root/autodl-tmp/outputs/rs_object_adapter_v0_e1_$(date +%Y%m%d_%H%M%S) \
  --shutdown-after-run
```

结果位于 run root 下的 `training/`、`evaluation_e1/`、`reports/` 和 `logs/`；
E1 评测文件为 `e1_metrics.json`、`predictions.jsonl` 和
`evaluation_metadata.json`。短 smoke 可加 `--max-train-groups 8 --max-steps 2`，
正式训练不要使用这两个限制。

## 判定与后续实验

只有总体/6-10 count 目标和 within-1 守住，并且 proposal 诊断没有显示明显退化时，
才考虑下一阶段。后续路线只记录不在 v0 实现：last-4 ViT blocks、提高 visual
resolution/token budget、Qwen-native `bbox_2d + scaled_0_1000` 独立协议、LoRA+。
AdaLoRA 不属于本实验。当前也没有测试 bbox_2d、LoRA+、AdaLoRA、last-4 或分辨率
调整。

本地工作区若缺少上述 `multisource` 正式源、`tiers_v2` manifest 或真实 Qwen 依赖，
builder/evaluator 会明确失败；不能用旧的 `qwen3vl_train.jsonl` 或 legacy tier
替代，因为那会改变数据契约和泄漏审计结论。

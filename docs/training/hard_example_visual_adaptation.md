# H1 Hard Example Visual Adaptation

## 目标与边界

H1 是对现有最终 LoRA checkpoint 的短程继续训练，不是从 Qwen3-VL base model
重新训练。起点必须是已经完成 Stage A（VRSBench）和 Stage B
（VRSBench + LEVIR-CC）的 adapter。历史 checkpoint、593 条固定评测集、历史 predictions、
Evaluation v1.5 报告和量化敏感度实验均保持不变。

本阶段不修改 LoRA rank、assistant-only label mask、bbox 训练协议、visual resolution 或
`min_pixels/max_pixels`，也不加入 LoRA+、AdaLoRA 或 DoRA。多任务 loss 聚合使用正式的
可替换接口；默认 `task_weighted`，历史对照可显式切换为 `token_mean`。

## 为什么不从 Base Model 重训

Stage A/B 已经建立遥感任务能力，重新从 base model 开始会增加成本并混入无关变量。
H1 只检验两个假设：当前模型的真实失败样本是否值得重点继续训练，以及最后少量视觉层
是否能改善遥感小目标表征。它保留已有 adapter 并使用较短 `max_steps`，因此结果可以和
当前 baseline 做清晰的 paired comparison。

## Assistant-only Supervision

`Qwen3VLDataCollator` 已正确实现 assistant-only loss：prompt、user、图像占位 token 和
padding 的 label 都是 `-100`，只有 assistant answer 满足 `labels != -100`。截断后没有
assistant token 会直接报错并包含 sample ID。H1 没有修改这段 mask。

`task_sampling_weights` 只改变样本被抽到的概率。独立的 `loss.task_weights` 才参与
loss 聚合，两者不能混用。H1 的 hard/replay 比例是数据组成，同样不是 loss weighting。

默认 `task_weighted` 先对每条样本的 assistant token CE 求均值，再按
`loss.task_weights` 聚合样本；因此 captioning 不会只因答案更长获得更高梯度权重。
当前所有任务权重均为 `1.0`，尚未人为提高 detection 或 counting 优先级。需要复现
Stage A/B 的历史 loss 数值时，使用 `loss.mode: token_mean`。两种聚合方式的
`train_loss/eval_loss` 数值不可直接比较，模型质量继续通过 Evaluation v1.5 的任务指标比较。

## 训练前统计

```bash
python scripts/training/analyze_training_data.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml \
  --run-name h1_precheck
```

输出：

```text
reports/training_statistics/h1_precheck/
├── summary.json
└── summary.md
```

监督 token 数使用真实 Collator 的 `labels != -100`。截断率通过同一 processor、同一 chat
template 的 capped 与 `truncation=False` 编码比较，不用字符串长度猜测。报告同时给出
数据源/任务占比、prompt/assistant/total token 分布、bbox 类别与面积、count 分布、图像尺寸、
`grid_thw` 和近似视觉 token 数。处理器未返回或低成本不可取得的数据标为 `unavailable`。

small/medium/large 阈值来自 YAML：

```yaml
bbox_area_thresholds:
  small_max: 0.01
  medium_max: 0.10
```

面积是 `normalized_0_1` xyxy 框的 `(x2-x1)*(y2-y1)`。

## Hard Example Mining 与泄漏保护

先在 train split 或专用 training mining split 上生成 predictions，再由 Evaluation v1.5
得到 `evaluated_predictions.jsonl`。hard mining 复用其中的 parser 和逐样本指标，不维护
第二套评测口径。

```bash
python scripts/training/build_hard_example_dataset.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml
```

必须配置：

- `H1_MINING_EVALUATION_DIR`：training mining split 的 Evaluation v1.5 输出目录。
- `H1_MINING_TRAIN_JSONL`：与预测 ID 对应的训练源 JSONL。
- `data/evaluation/tiers/evaluation_tiers_manifest.json`：E1/E2/E3 全部冻结评测 ID；H1 默认读取 E3 ID 集并全部排除。
- `FINAL_LORA_CHECKPOINT`：Stage-B 最终 adapter。

生产配置默认 fail closed：未提供排除 ID 或 ID 数少于 593 时不写 H1 数据。预测必须能按
ID 内连接到训练源；不在训练源中的 prediction 会进入 manifest 的
`unmatched_prediction_ids`，不会训练。最终 593 条样本仅用于 H1 前后 paired evaluation，
严禁进入 `hard_train.jsonl`、`replay_train.jsonl` 或 `h1_train.jsonl`。

Detection hard score 综合 `(1-IoU)`、label error、parse/coordinate failure、center distance
和 small-object bonus；同时保留 GIoU、bbox area bucket 和具体原因。Counting 使用 parse、
absolute error、exact 和 within-1，并把 `abs_error >= 2` 标为高优先级原因。VQA 保存
`qa_type`；Caption 综合 ROUGE-L、chrF 和 CIDEr-D approximation。Change 样本只使用可信的
caption/semantic 诊断，不根据异常的 binary predicted-change-rate 自动加权。

输出：

```text
data/processed/hard_examples/
├── hard_train.jsonl
├── replay_train.jsonl
├── h1_train.jsonl
└── hard_manifest.json
```

manifest 记录起点 checkpoint、预测来源、Evaluation contract、权重/阈值、hard/replay ID、
任务/来源分布、seed、SHA256、创建时间和全部排除 ID。

## Hard + Replay

默认目标是 70% hard + 30% regular replay。replay 使用固定 seed，并优先覆盖可获得的每个
`training_source × task_type` 单元，包括 VRSBench detection/counting/VQA/caption/scene 和
LEVIR-CC change detection。小型数据集为保证覆盖可能略偏离目标比例，实际比例会写入
manifest，避免用名义比例掩盖实际组成。正式配置启用 `enforce_replay_coverage`；缺少要求的
source 或 task 时直接失败，不能静默生成偏科 replay。

## Partial ViT 范围

H1 在 PEFT adapter 加载完成后执行：

1. 冻结全部参数；
2. 恢复全部 LoRA A/B 参数；
3. 通过对象结构解析 `visual.blocks`，使用 `len(blocks)` 选择最后 N 层；
4. 解冻主 `visual.merger`；
5. 保持 patch embed、position 模块、前面 blocks 和 deep-stack mergers 冻结；
6. 审计全部 trainable 参数后才创建 optimizer。

当前本地 Qwen3-VL-2B 实际有 24 个 visual blocks，权重键为
`model.visual.blocks.0..23`，主 merger 为 `model.visual.merger`，deep-stack mergers 为
`model.visual.deepstack_merger_list.0..2`。代码不硬编码 24 或 block 22/23，PEFT 包装前缀
变化也不影响对象解析。第一版只解冻 last-2，是为了控制显存和 catastrophic forgetting；
last-4 留给 H2。

## 参数审计与分组学习率

训练前写入：

```text
reports/training/h1_hard_visual_adaptation/trainable_parameters.json
<checkpoint>/trainable_parameters.json
```

审计分类为 LoRA、vision blocks、visual merger、optional visual 和 unexpected trainable，
记录参数名、tensor 数、参数量、总参数量及比例。视觉 block 或 merger 未真正匹配时立即
失败；默认 `fail_on_unexpected_trainable=true`。

优化器按参数 identity 保证互斥：

```text
LoRA          1e-5
Visual merger 5e-6
Last ViT      1e-6
```

配置强制 `LR_ViT < LR_Merger < LR_LoRA`。Trainer 正式接收预构建 AdamW optimizer，未对
Trainer 做运行时 monkey patch。weight decay、warmup、cosine scheduler、max grad norm 和
bf16 延续已有训练参数。

## Step 预算与 Checkpoint

H1 使用 `max_steps`，`num_train_epochs` 必须为 `null`。先读取 Stage-B 的最终 global step：

```bash
python scripts/training/estimate_h1_steps.py \
  --reference-stage-steps <STAGE_B_GLOBAL_STEPS>
```

脚本给出 10%/15%/20%/25% 候选，不自动选值。结合 hard 数据量、统计报告和显存结果后，
手动更新 H1 YAML。默认示例 1000 steps、每 250 steps 保存一次，近似保留 25% 间隔；应随
最终 max_steps 同步调整 `save_steps/eval_steps`。

标准 PEFT 保存只包含 adapter，无法包含解冻后的 base visual 参数。因此 H1 的每个 Trainer
checkpoint 额外保存 `h1_visual_weights.safetensors`；最终根目录的
`strategy_manifest.json` 标记 `adapter_with_visual_sidecar`。Evaluation v1.5 会先加载 base、
再挂 adapter、最后按 manifest 加载 visual sidecar，缺文件时明确失败。

## H1 训练

Dry-run 不加载模型：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml \
  --dry-run
```

AutoDL 正式训练：

```bash
source environments/autodl.env
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml
```

`FINAL_LORA_CHECKPOINT` 与新 output directory 必须不同。配置缺少起点 adapter、使用 qlora、
缺少 max_steps 或设置 epoch 时都会在加载模型前失败。

## Evaluation v1.5 与 Small Object 分析

593 条固定 evaluation set 上分别评估 Stage-B baseline 和 H1：

```bash
python scripts/evaluation/run_evaluation.py \
  --config configs/evaluation/evaluation_v1_5.yaml \
  --checkpoint "$FINAL_LORA_CHECKPOINT"

python scripts/evaluation/run_evaluation.py \
  --config configs/evaluation/evaluation_v1_5.yaml \
  --checkpoint "$H1_CHECKPOINT"
```

配对比较：

```bash
python scripts/evaluation/compare_evaluations.py \
  --baseline-dir reports/evaluation/stage_b \
  --candidate-dir reports/evaluation/h1 \
  --output-dir reports/evaluation/h1_comparison
```

small/medium/large 与视觉预算相关性：

```bash
python scripts/evaluation/analyze_visual_adaptation.py \
  --before reports/evaluation/stage_b \
  --after reports/evaluation/h1 \
  --output reports/evaluation/h1_comparison/visual_analysis.json
```

该报告比较各 bbox size 的 mIoU、GIoU、center distance、label match 和 parse success，并在
元数据存在时统计 IoU 与 bbox area、visual token count、image pixel count 的 Pearson
相关性。相关性只用于诊断，不表示因果，缺失字段不插值。如果 small-object IoU 提升而
counting/VQA/caption/change 基本不退化，才支持 partial ViT adaptation 的假设。

绘图继续使用 Evaluation v1.5：

```bash
python scripts/evaluation/plot_evaluation_results.py \
  --evaluation "stage_b=reports/evaluation/stage_b" \
  --evaluation "h1=reports/evaluation/h1" \
  --comparison "h1_vs_stage_b=reports/evaluation/h1_comparison" \
  --output-dir reports/evaluation/h1_comparison/figures \
  --overwrite
```

## BBox 协议保持不变

本阶段仍使用：

```json
{"label":"<class>","bbox":[0.0,0.0,1.0,1.0]}
```

不会修改 prompt templates、VRSBench converter、task protocol 或 Evaluation parser 的主
schema。未来 P1 可从 Current/H1 checkpoint 做短程 detection protocol adaptation，独立测试
Qwen-native `bbox_2d + scaled_0_1000`，无需从 base model 重训。

## 后续路线

- H2：仅当 H1 证明视觉适配有效时测试 last-4 ViT blocks。
- R1：仅当错误与 visual token shortage 明显相关时提高视觉分辨率/token budget。
- P1：独立测试 `bbox_2d + scaled_0_1000`。
- L+：H1 完成且有额外预算时再测试 LoRA+。
- AdaLoRA：当前低优先级，不在 H1 实现。

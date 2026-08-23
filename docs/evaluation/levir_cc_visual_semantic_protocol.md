# LEVIR-CC 影像级细粒度变化语义评测协议

## 目标

本协议响应组委会对 LEVIR-CC 的补充要求。在 CIDEr、BLEU-4 等 Caption 文本指标之外，分别评价：

1. 双时相影像中是否发生变化；
2. 变化对象是否正确；
3. 变化方向或类型是否正确；
4. 对象与方向的联合事件是否正确。

金标必须由标注者查看同一对“时相一 / 时相二”图像后给出；不得把参考 Caption 当作图像事实金标。

## 标签体系

对象字段 gold_changed_objects 可多选并使用竖线分隔：

- building、road、parking_area、bridge、sports_field；
- water_body、vegetation_landcover、other_permanent_structure。

方向字段 gold_change_directions 可多选并使用竖线分隔：

- appearance_construction：新增、出现、新建；
- disappearance_demolition：消失、拆除、移除；
- expansion、reduction；
- replacement_modification；
- state_change_unspecified：确认有变化但无法可靠归入前述方向。

事件字段 gold_change_events 是正式评分依据，使用`对象:方向`并以`|`分隔，例如：

```text
building:appearance_construction|road:disappearance_demolition
```

它避免把“建筑新增、道路拆除”错误扩展为四种对象—方向组合。`gold_changed_objects`和
`gold_change_directions`必须分别与事件字段中的对象集合、方向集合完全一致。

无变化填 gold_change_label=0、对象与方向均填 none。图像不可判定时填 U、对象与方向均填 unknown，保留审计但不进入二分类分母。

appearance_construction 与 disappearance_demolition、expansion 与 reduction 是相反时序语义。若预测同一对象为相反方向，单独计为 opposite_temporal_error_count，不能只笼统写作“方向错误”。

## 标注与汇总

两位标注者独立查看图像对。标注表不显示模型 Caption、参考 Caption 或数据集 changeflag，避免锚定偏差；分歧项由第三人裁决。

模型生成 Caption 经固定版本的规则抽取器转为对象、方向和“对象-方向”事件，再与影像金标比较。二分类不从 Caption 关键词二次推断：必须优先读取预测文件中已写回的服务器规则或离线本地小模型结果；缺失时标为 unresolved。汇总包括：

- 二分类判定覆盖率、未决数，以及已判定样本上的 Accuracy、Balanced Accuracy、变化类 Precision / Recall / F1、无变化 Recall；
- 保守全样本 Accuracy（将未决视为未完成判定）仅作覆盖诊断，不与已判定样本 Accuracy 混淆；
- 对象的 Micro Precision / Recall / F1 与集合 Exact Match（仅真实有变化样本）；
- 方向的 Micro Precision / Recall / F1 与集合 Exact Match（仅真实有变化样本）；
- 对象-方向联合事件的 Micro Precision / Recall / F1 与 Exact Match（仅真实有变化样本）；
- 相反时序错误数。

全部新增语义指标标记为“内部、规则抽取”，并在报告中说明它们衡量的是 Caption 中表达出的变化语义与视觉金标的一致性。

同一份 `predictions.jsonl` 的二分类结果可重复用于指标重算和绘图；只有模型、Prompt、生成参数或数据集发生变化、导致 Caption 重新生成时，才需要对规则未决样本重新运行离线小模型判定。

## Prompt 可复现性

每次正式推理必须记录 Prompt 原文、图像顺序（时相一为 before，时相二为 after）、do_sample、温度、top_p、max_new_tokens、num_beams 和输出后处理版本。

当前历史 predictions.jsonl 未完整保存原始 Prompt 文本，因此不能事后假设其与数据集作者 Prompt 一致。正式语义基线应从已声明 Prompt profile 的推理结果开始；不同 Prompt profile 的结果分开展示，不直接横向比较。评测脚本会核验生成清单的`predictions_sha256`与输入预测文件一致。

## 运行

python scripts/evaluation/evaluate_levir_visual_semantics.py --gold-csv <visual_gold.csv> --predictions <predictions.jsonl> --generation-manifest <generation_manifest.json> --output-dir <new_output_dir> --prompt-profile <prompt_profile_id>

评测脚本拒绝复用非空输出目录，并保存输入 SHA256、逐样本事件、汇总指标和审计行。
## 标注与冻结流程

影像级金标必须在不展示模型输出、参考 Caption 或 `changeflag` 的条件下形成。仓库提供以下本地流程，所有中间 CSV 与图像路径都不应上传至 GitHub：

```powershell
# 1. 由图像映射生成两份独立、盲化的标注表；可用 --sample-size 200 冻结首批子集
python scripts/evaluation/prepare_levir_visual_semantic_audit.py `
  --image-mapping <image_mapping.csv> `
  --sample-size 200 `
  --output-dir <visual_semantic_audit>

# 2. 检查两位标注者填写的表；输出只含校验摘要
python scripts/evaluation/validate_levir_visual_semantic_annotations.py `
  --annotations <annotator_a_visual_semantic.csv> `
  --output <annotator_a_validation.json>

# 3. 自动导出仅包含分歧样本的裁决表，第三位标注者完成其中的 gold_* 字段
python scripts/evaluation/prepare_levir_visual_semantic_adjudication.py `
  --annotator-a <annotator_a_visual_semantic.csv> `
  --annotator-b <annotator_b_visual_semantic.csv> `
  --output <visual_semantic_adjudication.csv>

# 4. 冻结唯一的图像金标；有分歧但没有裁决时程序会失败
python scripts/evaluation/finalize_levir_visual_semantic_gold.py `
  --annotator-a <annotator_a_visual_semantic.csv> `
  --annotator-b <annotator_b_visual_semantic.csv> `
  --adjudicated <visual_semantic_adjudication.csv> `
  --output-dir <frozen_visual_semantic_gold>
```

最终生成的 `visual_semantic_gold_standard.csv` 才是 `evaluate_levir_visual_semantics.py --gold-csv` 的输入。金标冻结后不得为了某个模型的结果而修改；如需修订，必须形成新版本金标和新的 SHA256 清单。

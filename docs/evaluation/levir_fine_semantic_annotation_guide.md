# LEVIR-CC 细粒度变化语义标注规范

## 目的与边界

本规范将已经完成的 LEVIR-CC Caption 二分类人工金标进一步拆为“变化对象”和“变化方向”。它服务于两件事：

- 为本地小模型提供高质量监督数据，减少把成像差异、车辆等误判为永久结构变化的情况；
- 为变化描述建立可解释的细粒度内部评测指标。

标注对象仍然只是模型 Caption 表达的语义：**不查看图像、参考 Caption、`changeflag` 或样本 ID**。因此结果是“Caption 语义金标”，不是图像级事实金标。

## 已有数据划分

| 划分 | 数量 | 用途 | 使用限制 |
|---|---:|---|---|
| `development` | 300 | 优化提示词、选择规则、训练/微调本地小模型 | 可用于开发 |
| `holdout` | 200 | 定型后的独立验证 | 在模型、提示词和规则冻结前不得读取其语义标注结果 |

二分类标签已由双人标注和裁决确定。本次只补标 `human_changed_objects`、`human_change_directions` 和置信度；不要改动已填写的 `human_change_label`。

## 字段填写

多项标签用英文竖线 `|` 分隔，例如：`building|road`。不要输入自由文本、逗号或中文分隔符。

| 字段 | 可选值 | 含义 |
|---|---|---|
| `human_changed_objects` | `building`、`road`、`parking_area`、`bridge`、`sports_field`、`water_body`、`vegetation_landcover`、`other_permanent_structure` | Caption 明确提及发生实质变化的对象，可多选 |
| `human_change_directions` | `appearance_construction`、`disappearance_demolition`、`expansion`、`reduction`、`replacement_modification`、`count_change`、`state_change_unspecified` | Caption 明确陈述的变化方向，可多选 |
| `human_annotation_confidence` | `high`、`medium`、`low` | 对这两个细粒度字段的把握程度 |
| `human_semantic_note` | 可选 | 仅简短记录歧义或裁决原因 |

补充值只在特定情况下使用：

- 若锁定的 `human_change_label=0`，对象和方向必须为 `none`；
- 若为 `U`，对象和方向必须为 `unknown`；
- 若为 `1`，必须完成对象和方向两个字段，且方向不能为 `none`。Caption 只说“发生了变化”但未说明对象或方向时，分别填 `unknown` 或 `state_change_unspecified`，并把置信度设为 `low`。

## 判定口径

### 变化对象

- 新房屋、厂房、建筑群：`building`
- 新道路、道路消失或改线：`road`
- 停车场本体的新增/消失/改造：`parking_area`；仅车辆出现或消失不属于该项
- 桥梁：`bridge`
- 球场、跑道等明确人工场地：`sports_field`
- 水塘、河道等水体变化：`water_body`
- 树林、植被、裸地等土地覆盖改变：`vegetation_landcover`
- 其他明确永久人工设施：`other_permanent_structure`

### 变化方向

- 出现、新建、建成、增加一个对象：`appearance_construction`
- 消失、拆除、移除：`disappearance_demolition`
- 面积扩大、延伸：`expansion`
- 面积缩小、减少：`reduction`
- 被另一种结构替代、结构形态明显改造：`replacement_modification`
- 只明确数量增减、但无法对应出现/消失对象：`count_change`
- 确认有实质改变但方向没有说明：`state_change_unspecified`

## 示例

| Caption | 对象 | 方向 |
|---|---|---|
| `A new building and a road appeared.` | `building|road` | `appearance_construction` |
| `Two houses were demolished.` | `building` | `disappearance_demolition` |
| `The parking area was expanded.` | `parking_area` | `expansion` |
| `Only a vehicle disappeared.` | `none` | `none` |
| `The scene is brighter, but structures remain unchanged.` | `none` | `none` |
| `The area changed, but the caption gives no details.` | `unknown` | `state_change_unspecified` |

## 质量控制与定型流程

1. 两位标注者独立填写对象、方向和置信度。
2. 程序先检查格式和与二分类金标的一致性。
3. 一致项直接进入金标；不一致项由第三人裁决。
4. 只用 `development` 金标调整本地小模型或提示词。
5. 配置冻结后，才打开 `holdout` 的细粒度结果，报告二分类、对象、方向和联合正确率。

即使某个模型在参考 Caption 语义上得分较高，也不能据此称其“图像级事实正确”。该结论必须另有看图人工审计或像素/实例级真值支持。

裁决表由以下命令生成；它会忽略多标签的填写顺序，只列出对象、方向不一致或格式不合法的 `human_change_label=1` 行：

```powershell
python scripts/evaluation/prepare_levir_fine_semantic_adjudication.py `
  --annotator-a <annotator_a_semantic_completed.csv> `
  --annotator-b <annotator_b_semantic_completed.csv> `
  --output-csv <semantic_adjudication.csv> `
  --summary-json <semantic_adjudication_summary.json>
```

# LEVIR-CC 影像级细粒度语义评测材料需求

## 1. 目的

本需求用于建立 LEVIR-CC 的影像级细粒度变化语义评测。除 CIDEr、BLEU-4、ROUGE-L 等 Caption 文本指标外，新增评价：

1. 是否正确判断双时相影像发生变化；
2. 是否正确识别变化对象；
3. 是否正确识别变化方向或类型；
4. 是否将新增与拆除、出现与消失等相反时序语义混淆。

金标由标注者直接查看同一对双时相影像后给出。参考 Caption 只用于传统文本指标，不能作为对象和方向的图像事实金标。

## 2. 材料清单

| 优先级 | 材料 | 是否必须 | 用途 |
|---|---|---|---|
| P0 | 双时相图像与样本 ID 的对应关系 | 必须 | 人工查看图像并建立对象、方向视觉金标 |
| P0 | 与图像对应的模型预测文件 | 必须 | 对模型生成 Caption 做语义评测 |
| P0 | Prompt 与生成配置及预测文件 SHA256 | 正式报告必须 | 保证模型比较可复现、不同 Prompt 不混比 |
| P1 | 数据划分与 changeflag | 建议 | 分层抽样、二分类核验、样本统计 |
| P1 | 模型/LoRA/量化版本信息 | 建议 | 结果追溯与性能对比 |
| P2 | 图像时间戳、分辨率、来源版本 | 可选 | 数据追溯与异常样本审计 |

已有 predictions.jsonl 无需重复导出；只需确认其具体对应的模型、Prompt profile 和生成配置。

## 3. P0：双时相图像定位信息

必须能够从 predictions.jsonl 的 id 唯一定位变化前和变化后图像。以下三种提供方式任选其一。

### 方式 A：图像映射 CSV（推荐）

示例字段：

    id,image_t1_path,image_t2_path,split
    levircc_val_7938_39693,D:\datasets\LEVIR-CC\A\7938.png,D:\datasets\LEVIR-CC\B\39693.png,val

字段要求：

| 字段 | 要求 |
|---|---|
| id | 与 predictions.jsonl 的 id 完全一致、非空、唯一 |
| image_t1_path | 变化前图像的可读取本地路径 |
| image_t2_path | 变化后图像的可读取本地路径 |
| split | 建议记录 val、test 等划分 |

### 方式 B：数据集根目录与命名规则

若不另行生成映射文件，可提供：

- LEVIR-CC 数据集根目录；
- 划分文件位置；
- 从预测 id 定位 t1、t2 两张图的完整规则；
- 明确说明 t1 为变化前、t2 为变化后。

规则必须能唯一解释任一预测 ID 对应的两张图。

### 方式 C：最小评测包

若暂不共享完整数据集，可先提供 200–300 个样本组成的评测包，包内包含：

    visual_semantic_subset/
    ├── image_mapping.csv
    ├── before/
    ├── after/
    └── predictions.jsonl

影像要求：

- 支持 PNG、JPG、JPEG、TIFF；
- 保留原始影像内容和分辨率，不另行裁剪或压缩；
- 两张图应为空间对应的同一区域；
- 文件可在本地离线读取，不需要上传至 GitHub；
- 图像路径可使用绝对路径或相对数据包路径，但必须统一。

## 4. P0：模型预测文件

预测文件使用 JSONL，每行一个样本。最少字段如下：

    {
      "id": "levircc_val_7938_39693",
      "task_type": "change_detection",
      "prediction": "A new building was constructed.",
      "reference": "a building appears .",
      "metadata": {"dataset": "LEVIR-CC", "split": "val", "changeflag": 1}
    }

要求：

- id 与图像定位信息一一对应；
- prediction 为模型原始输出，不能人工修改；
- 每份文件只对应一种模型、LoRA/量化配置、Prompt profile 和生成设置；
- 不同配置必须导出为不同结果文件；
- reference 和 changeflag 可保留用于传统指标与分层抽样，但不展示给视觉标注者；
- 评测时记录输入文件 SHA256，避免结果文件被替换后无法追溯。

## 5. P0：Prompt 与生成配置

每份正式预测结果必须附带 generation_manifest.json 或等价配置文件，至少包括：

    {
      "prompt_profile": "levir_caption_v1",
      "prompt_text_verbatim": "Compare the two remote-sensing images and describe the changes.",
      "image_t1_role": "before",
      "image_t2_role": "after",
      "input_image_order": ["image_t1", "image_t2"],
      "do_sample": false,
      "temperature": 0.0,
      "top_p": 1.0,
      "num_beams": 1,
      "max_new_tokens": 128,
      "model_id": "Qwen3-VL-2B-Instruct",
      "adapter_id": "optional_lora_identifier",
      "quantization": "none",
      "code_version": "git_commit_or_release_identifier",
      "output_postprocessing": "strip_only",
      "predictions_sha256": "与本次 predictions.jsonl 完全一致的 SHA256"
    }

必须明确：

- Prompt 原文，包括 system prompt 或模板文本；
- 双时相图像输入顺序；
- do_sample、temperature、top_p、num_beams、max_new_tokens；
- 基础模型、LoRA、量化与代码版本；
- 输出后处理方法。

不同 Prompt profile 的结果必须单独报告，不能直接横向比较。

## 6. 影像级语义金标范围

首批建议冻结 200–300 个样本。应分层覆盖：

- 真实有变化和无变化；
- 建筑新增与建筑拆除；
- 道路、停车区域、植被/土地覆盖等实际出现的变化类型；
- 清晰样本和困难样本。

视觉标注表将由评测侧生成。标注者只查看 sample_id、image_t1_path、image_t2_path、gold_change_label、gold_changed_objects、gold_change_directions、gold_change_events、annotation_confidence、annotation_note。

标注表不显示模型 Caption、参考 Caption 或 changeflag。两人独立标注，分歧项再裁决。

标签规则：

| 字段 | 取值 |
|---|---|
| gold_change_label | 0、1、U |
| gold_changed_objects | building、road、parking_area、bridge、sports_field、water_body、vegetation_landcover、other_permanent_structure、none、unknown |
| gold_change_directions | appearance_construction、disappearance_demolition、expansion、reduction、replacement_modification、state_change_unspecified、none、unknown |
| gold_change_events | 一个或多个`对象:方向`对，以`|`分隔；例如`building:appearance_construction|road:disappearance_demolition` |

无变化：label=0，objects=none，directions=none，events留空。
不可可靠判定：label=U，objects=unknown，directions=unknown，events留空；保留审计，不进入主指标分母。

## 7. 评测输出

得到影像金标后，评测侧将输出：

- 二分类 Accuracy、Balanced Accuracy、变化类 Precision / Recall / F1、无变化 Recall；
- 变化对象 Micro Precision / Recall / F1、对象集合 Exact Match；
- 变化方向 Micro Precision / Recall / F1、方向集合 Exact Match；
- 对象-方向联合事件 Micro Precision / Recall / F1、Exact Match；
- 相反时序错误数：新增预测为拆除、拆除预测为新增、扩大预测为缩小等；
- 逐样本语义结果、错误清单、输入 SHA256 与运行清单。

新增语义指标统一标注为“内部、规则抽取”，并与传统 Caption 指标分开展示。

## 8. 提供前检查

- [ ] 图像映射中的 id 与 predictions.jsonl 完全匹配；
- [ ] t1、t2 顺序已明确为 before、after；
- [ ] 图像文件可本地打开；
- [ ] predictions.jsonl 保留原始 prediction；
- [ ] 每份结果均有 Prompt 和生成配置；
- [ ] 不同模型或不同 Prompt 未混在同一预测文件；
- [ ] 用于正式汇报的语义评测子集 ID 已冻结。

# LEVIR-CC 图像级语义评测材料清单

## 一、数据与图像材料（优先提供）

- LEVIR-CC 双时相图像本地位置，明确：
  - `t1`：变化前图像（before）；
  - `t2`：变化后图像（after）。

- 图像映射 CSV，至少包含：

```csv
id,image_t1_path,image_t2_path,split
sample_001,D:\LEVIR-CC\A\001.png,D:\LEVIR-CC\B\001.png,val
```

要求：

- `id` 与预测文件中的 `id` 完全一致；
- 每个 ID 唯一；
- 两张图像可在本地打开；
- 路径可为绝对路径或相对路径，但全表保持一致；
- 提供 `split`，例如 `val`、`test`。

如暂时无法提供完整数据集，可先提供 200–300 条样本的最小评测包：

```text
visual_semantic_subset/
├── image_mapping.csv
├── before/
├── after/
└── predictions.jsonl
```

## 二、冻结评测样本范围

首批建议提供或确认 200–300 条冻结样本，覆盖：

- 有变化、无变化；
- 建筑新增、建筑拆除；
- 道路、停车区、植被/土地覆盖等变化；
- 清晰样本与难例。

冻结后，同一批样本用于所有模型横向比较。

## 三、模型预测结果

每个模型、LoRA、量化配置或 Prompt 单独提供一个原始预测文件：

```json
{
  "id": "sample_001",
  "task_type": "change_detection",
  "prediction": "A new building was constructed.",
  "reference": "a building appears .",
  "metadata": {
    "dataset": "LEVIR-CC",
    "split": "val",
    "changeflag": 1
  }
}
```

要求：

- 文件格式为 `predictions.jsonl`；
- `prediction` 为模型原始输出，不人工修改；
- 不同模型或不同配置不能混在同一个文件；
- 保留 `reference` 与 `changeflag`，但不向人工图像标注者展示。

## 四、每份预测结果对应的生成配置

每份 `predictions.jsonl` 同时提供 `generation_manifest.json`，至少包含：

```json
{
  "prompt_profile": "levir_caption_v1",
  "prompt_text_verbatim": "完整 Prompt 原文",
  "image_t1_role": "before",
  "image_t2_role": "after",
  "input_image_order": ["image_t1", "image_t2"],
  "do_sample": false,
  "temperature": 0.0,
  "top_p": 1.0,
  "num_beams": 1,
  "max_new_tokens": 128,
  "model_id": "基础模型名称",
  "adapter_id": "LoRA 名称或 none",
  "quantization": "none / int8 / int4",
  "code_version": "Git commit 或版本号",
  "output_postprocessing": "输出后处理方式",
  "predictions_sha256": "预测文件 SHA256"
}
```

## 提供前检查

- [ ] 图像映射中的 `id` 与 `predictions.jsonl` 完全匹配；
- [ ] `t1`、`t2` 顺序已明确为 before、after；
- [ ] 图像文件可在本地打开；
- [ ] 每份预测文件只对应一种模型和一套生成配置；
- [ ] Prompt、图像输入顺序、生成参数、模型/LoRA/量化和代码版本均已记录；
- [ ] 不同 Prompt profile 的结果未混在同一预测文件中。

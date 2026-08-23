# LEVIR-CC 历史结果与正式推理记录

## 历史结果

缺少 Prompt 原文、双时相图像顺序或完整生成参数的既有预测，只能标记为“历史探索性结果”。它们可用于：定位模型短板、筛选人工金标样本、形成后续优化假设；不得作为严格可复现的正式模型对比结论。

不要根据预测文本、训练配置或记忆反推并填写历史 Prompt；这会把未知信息误写为事实。

## 后续正式推理

每个正式 LEVIR-CC 推理实验均从
`configs/eval/levir_cc_formal_generation_template.yaml` 复制一份专用配置，并在运行前填写：

- `prompt_text`：实际送入模型的完整原文；
- `image_t1_role`、`image_t2_role`、`image_input_order`：固定为本次双时相图像的真实语义和排列；
- `generation_profile_name`：本次 Prompt 版本名；
- `model_id`、`adapter_id`、`quantization`、`code_version`；
- 数据集/测试集路径和模型路径。

设置 `require_complete: true` 后，任一字段为空时推理会在加载模型前终止。成功推理会在输出目录同时保存：

- `predictions.jsonl`：逐样本模型输出；
- `performance_report.json`：显存、时延、吞吐等性能监测；
- `generation_manifest.json`：Prompt、图像顺序、生成参数、模型/Adapter 指纹和输入输出 SHA256；
- `model_run_manifest.json`：同内容的兼容文件名。

图像级语义评测必须使用与 `predictions.jsonl` SHA256 完全一致的 `generation_manifest.json`。更换 Prompt、模型、Adapter、量化方式、数据集或生成参数后，均应视为新的实验，重新生成预测和清单。

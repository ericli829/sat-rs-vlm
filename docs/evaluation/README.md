# 多任务离线评测 v1.5

本模块读取仓库推理脚本生成的 `predictions.jsonl`，在不加载模型的情况下完成质量评测、逐样本扩展、结果追溯和同集配对比较。

## 输入字段

每行至少包含：

```json
{
  "id": "sample-1",
  "task_type": "detection",
  "prediction": "{\"label\":\"ship\",\"bbox\":[0.1,0.2,0.4,0.5]}",
  "reference": "{\"label\":\"ship\",\"bbox\":[0.1,0.2,0.4,0.5]}",
  "metadata": {"dataset": "VRSBench", "source_task": "referring"},
  "inference_latency_ms": 12.5
}
```

量化、故障和恢复实验的扩展字段会原样保留。严格模式会拒绝重复ID、缺失必填字段、非法参考答案和非法LEVIR-CC `changeflag`。

## 协议路由

- `VRSBench + detection + source_task=referring`：Visual Grounding。
- `VRSBench + counting`：Counting。
- `VRSBench + captioning`：Detailed Caption。
- `VRSBench + vqa/scene_classification`：Open VQA内部文本指标。
- `LEVIR-CC + change_detection`：变化二分类与变化描述。
- 其他数据集按 `task_type` 进入通用内部协议。

坐标格式按 `metadata.bbox_target_format`、Dataset Manifest `coordinate_range` 的顺序确定，不根据框数值猜测。

### LEVIR-CC变化判定

变化二分类使用可追溯的 `levir_relaxed_no_change_v2` 解析器，不再要求模型输出与少量固定句子完全一致：

- 直接接受 `0`（无变化）和 `1`（有变化）；
- 接受 `changeflag`、`change_flag`、`changed`、`has_change` JSON字段中的二分类值；
- 接受完整、全局性的常见无变化表达及其时态、单复数和轻微措辞变化，例如
  `No changes were observed between the two images`、`There were no significant changes`、
  `Both images appear unchanged`；
- 不使用简单关键词包含规则。`No building changed, but a road appeared`、
  `No change in buildings; however vegetation was removed`等复合变化描述仍判为有变化；
- 空回答解析失败；其他非空且不满足完整无变化模式的变化描述判为有变化。

逐样本结果记录 `change_parser_version` 和 `change_parse_mode`，汇总结果同时给出各解析模式的使用率，
便于定位“全部被判为有变化”是模型输出问题还是解析规则问题。调整模型提示词使其直接输出0/1可以减少歧义，
但使用本解析器不要求重新训练模型。

## 运行评测

```powershell
python scripts/evaluation/evaluate_predictions.py `
  --predictions outputs/evaluation/model/predictions.jsonl `
  --output-dir outputs/evaluation/model/v1.5 `
  --latency-semantics batch_amortized_per_sample `
  --eval-batch-size 16 `
  --group-by-task
```

输出目录默认拒绝覆盖已有非空目录：

```text
evaluated_predictions.jsonl
summary.json
evaluation_manifest.json
```

## 配对比较

```powershell
python scripts/evaluation/compare_evaluations.py `
  --baseline-dir outputs/evaluation/baseline/v1.5 `
  --candidate-dir outputs/evaluation/candidate/v1.5 `
  --output-dir outputs/evaluation/comparisons/baseline-vs-candidate `
  --bootstrap-resamples 1000 `
  --seed 20260806
```

比较前会强制核验ID集合、任务类型和参考答案，随后输出逐样本改善/退化结果和配对Bootstrap 95%置信区间。

## 统一绘图

绘图工具读取一个或多个完整评测目录；每个目录必须包含 `summary.json`，存在
`evaluated_predictions.jsonl` 时会额外绘制Grounding IoU CDF和Counting误差分布。配对比较目录必须包含
`comparison_summary.json`。

安装仓库已有的可选绘图依赖：

```powershell
python -m pip install -e ".[reliability-plot]"
```

运行示例：

```powershell
python scripts/evaluation/plot_evaluation_results.py `
  --evaluation baseline=outputs/evaluation/baseline/v1.5 `
  --evaluation candidate=outputs/evaluation/candidate/v1.5 `
  --evaluation levir=outputs/evaluation/levir/v1.5 `
  --comparison vrsbench=outputs/evaluation/comparisons/baseline-vs-candidate `
  --output-dir outputs/evaluation/figures/v1.5 `
  --formats png svg
```

`--evaluation`与`--comparison`均使用 `LABEL=DIR` 格式。默认拒绝写入非空输出目录；只有显式指定
`--overwrite`时才允许替换同名生成文件，不会删除目录中的其他文件。

输出包括：

- 任务样本分布、VRSBench核心指标和任务短板图；
- Grounding IoU CDF、Counting误差方向和VQA QA-Type准确率；
- Caption质量/长度与参考文本语义诊断；
- 配对改善置信区间和Win/Tie/Loss；
- LEVIR-CC混淆矩阵、二分类指标和变化描述指标；
- 测量口径一致时的Mean/P50/P95延迟图；
- 记录输入哈希、契约版本、生成文件和跳过原因的 `plot_manifest.json`。

图中文字使用英文以避免跨平台字体差异。所有高级语义图均明确标为参考文本内部诊断，不表示图像级事实正确率。

## 指标边界

- Grounding、Counting、VQA、Caption和LEVIR-CC当前自动指标均为内部指标。
- 名称包含 `Approx` 的文本指标不能作为官方榜单分数。
- 语义诊断只比较预测文本和参考文本，不读取图像，不能称为图像级幻觉率。
- 参数量、模型文件大小、显存和吞吐率不能从Prediction JSONL推测。
- `repository_native_v2`用于保持仓库原有 `v2_task_metrics` 结果口径。

## 测试

```powershell
pytest tests/unit/evaluation
pytest tests/integration/evaluation/test_plot_evaluation_cli.py
ruff check src/sat_rs_vlm/evaluation scripts/evaluation tests/unit/evaluation
```

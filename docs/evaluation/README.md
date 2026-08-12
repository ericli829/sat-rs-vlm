# 多任务离线评测 v1.6（P0变化二分类）

本模块读取仓库推理脚本生成的 `predictions.jsonl`，在不加载模型的情况下完成质量评测、逐样本扩展、结果追溯和同集配对比较。v1.6在冻结的v1.5基础上新增P0独立变化二分类；v1.5契约文件保持原有行为，仅用于历史结果复现。

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

P0采用“独立二分类 + 变化描述”双输出。推理脚本对同一对图像先生成原变化描述，再用固定二分类问题要求模型输出
`0=无语义变化`或`1=有语义变化`。这不是从描述文本中再次猜测，而是模型对图像对执行的第二次独立推理。

新生成的Prediction JSONL会增加：

```json
{
  "prediction": "A new building appeared.",
  "prediction_changeflag": 1,
  "binary_prediction": "1",
  "binary_prediction_parse_ok": true,
  "binary_prompt_version": "levir_semantic_change_binary_v1",
  "binary_inference_latency_ms": 18.4,
  "total_inference_latency_ms": 137.2
}
```

评测优先级固定为：

1. 合法的顶层 `prediction_changeflag` 整数0/1；
2. 独立问题的原始 `binary_prediction`，只按显式二分类格式解析；
3. 两个字段均不存在时，旧Prediction JSONL才使用描述文本兼容解析。

独立二分类解析不会套用描述性白名单。无法解析的独立回答记为解析失败，不会用Caption结果掩盖。逐样本结果记录
`binary_prediction_source`，汇总给出 `explicit_binary_decision_rate` 和 `caption_fallback_decision_rate`，可检查一次评测究竟用了P0还是旧兼容口径。

旧结果的兼容回退使用可追溯的 `levir_contextual_no_change_v3` 解析器，不要求模型输出与少量固定句子完全一致：

- 直接接受 `0`（无变化）和 `1`（有变化）；
- 接受 `changeflag`、`change_flag`、`changed`、`has_change` JSON字段中的二分类值；
- 接受完整、全局性的常见无变化表达及其时态、单复数和轻微措辞变化，例如
  `No changes were observed between the two images`、`There were no significant changes`、
  `Both images appear unchanged`；
- 对多个句子的回答逐句判定；只有每个分句都表达全局无变化时才判为无变化，例如
  `The second image is identical to the first image. There are no visible changes between the two images.`；
- 允许在场景说明之后给出全局无变化结论，例如先描述两幅图的视角，再说明
  `There are no discernible changes in the environment`；只有未发现新增、移除、替换、建造等明确变化证据时才接受该结论；
- 不使用简单关键词包含规则。`No building changed, but a road appeared`、
  `No change in buildings; however vegetation was removed`等复合变化描述仍判为有变化；
- 空回答解析失败；其他非空且不满足完整无变化模式的变化描述判为有变化。

逐样本结果同时记录 `change_decision_version`、`change_parser_version` 和 `change_parse_mode`。旧文件无需重导出即可继续评测，
但要得到P0独立二分类结果，必须用更新后的推理脚本重新生成预测；无需重新训练模型。

普通评测与量化评测配置默认启用P0；可显式控制：

```yaml
generation:
  change_binary_enabled: true
  change_binary_max_new_tokens: 8
```

`inference_latency_ms`仍只表示原变化描述的生成延迟，便于与历史结果比较；新增二分类延迟单独记录在
`binary_inference_latency_ms`，两次推理总延迟记录在 `total_inference_latency_ms`。

## 运行评测

```powershell
python scripts/evaluation/evaluate_predictions.py `
  --predictions outputs/evaluation/model/predictions.jsonl `
  --output-dir outputs/evaluation/model/v1.6 `
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
  --baseline-dir outputs/evaluation/baseline/v1.6 `
  --candidate-dir outputs/evaluation/candidate/v1.6 `
  --output-dir outputs/evaluation/comparisons/baseline-vs-candidate `
  --bootstrap-resamples 1000 `
  --seed 20260806
```

比较前会强制核验ID集合、任务类型和参考答案，随后输出逐样本改善/退化结果和配对Bootstrap 95%置信区间。

## 统一绘图

绘图工具同时支持v1.5和v1.6结果。一次运行中的所有评测目录必须使用同一个契约版本，
配对比较也必须与该版本一致，避免把不同判定口径直接画在同一张图上。每个目录必须包含 `summary.json`，存在
`evaluated_predictions.jsonl` 时会额外绘制Grounding IoU CDF和Counting误差分布。配对比较目录必须包含
`comparison_summary.json`。

安装仓库已有的可选绘图依赖：

```powershell
python -m pip install -e ".[reliability-plot]"
```

运行示例：

```powershell
python scripts/evaluation/plot_evaluation_results.py `
  --evaluation baseline=outputs/evaluation/baseline/v1.6 `
  --evaluation candidate=outputs/evaluation/candidate/v1.6 `
  --evaluation levir=outputs/evaluation/levir/v1.6 `
  --comparison vrsbench=outputs/evaluation/comparisons/baseline-vs-candidate `
  --output-dir outputs/evaluation/figures/v1.6 `
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

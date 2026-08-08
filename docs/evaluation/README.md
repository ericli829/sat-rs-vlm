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

## 指标边界

- Grounding、Counting、VQA、Caption和LEVIR-CC当前自动指标均为内部指标。
- 名称包含 `Approx` 的文本指标不能作为官方榜单分数。
- 语义诊断只比较预测文本和参考文本，不读取图像，不能称为图像级幻觉率。
- 参数量、模型文件大小、显存和吞吐率不能从Prediction JSONL推测。
- `repository_native_v2`用于保持仓库原有 `v2_task_metrics` 结果口径。

## 测试

```powershell
pytest tests/unit/evaluation
ruff check src/sat_rs_vlm/evaluation scripts/evaluation tests/unit/evaluation
```

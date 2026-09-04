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
- `MME-RealWorld + official_subtask=Remote Sensing`：官方 A-E 单选解析与准确率。
- `XLRS-Bench + VQA`：官方 A-D 单选；Overall Land Use Classification
  自动进入多选集合精确匹配。
- 其他数据集按 `task_type` 进入通用内部协议。

坐标格式按 `metadata.bbox_target_format`、Dataset Manifest `coordinate_range` 的顺序确定，不根据框数值猜测。

MME/XLRS 的 `protocol_provenance` 固定记录官方来源 commit。官方 parser 可以在任意子集上
运行，但只有数据版本、split、语言、官方任务层级、原始 prompt profile 全部齐全，并显式
声明 `evaluation_scope=official_full_split` 时，`official_comparability` 才会标为
`eligible_for_official_comparison`；否则为 `protocol_only`。

官方 JSON/JSONL 可先转换为主推理链路直接读取的 messages JSONL：

```powershell
python scripts/data/prepare_official_benchmark.py `
  --dataset mme-realworld-rs `
  --input D:\datasets\MME_RealWorld.json `
  --output data\processed\mme_realworld_rs_eval.jsonl `
  --dataset-version official-2024 `
  --split train `
  --language en `
  --source-repository https://github.com/MME-Benchmarks/MME-RealWorld `
  --source-commit bee58edb82c883843ae93be61c2ae3b452c781d1 `
  --expected-records <remote-sensing-count> `
  --official-full-split
```

`--dataset xlrs-vqa` 使用 XLRS 官方单选/多选 prompt。转换器默认不声称输入是完整
官方 split；只有确认原始文件完整、来源 commit 和转换后样本数时才应传
`--official-full-split`。转换器会拒绝重复样本 ID 和样本数不一致的声明。

完整系统评测使用 [scripts/evaluate_taskgraph.py](../../scripts/evaluate_taskgraph.py)。
它逐样本调用 `TaskGraphRuntime`，并输出 `predictions.jsonl`、`evaluation/`、
`system_manifest.json` 和 `evaluation_metadata.json`。输入 JSONL 支持 `id`、`dataset`、
`task_type`、`question`、`images`、`options`、`reference`、`metadata` 和可选的
`graph` 字段：

```powershell
python scripts/evaluate_taskgraph.py `
  --config configs/taskgraph/runtime.real.example.yaml `
  --input data/processed/official_taskgraph_eval.jsonl `
  --output-dir reports/official/taskgraph/run-001 `
  --contract configs/eval/evaluation_contract_v1.8_local_complete.yaml `
  --warmup-runs 1 `
  --repeat-runs 3
```

评测产物提交前可在不加载模型的情况下执行性能审计：

```powershell
python scripts/evaluation/audit_performance_report.py `
  --run-dir reports/official/taskgraph/run-001 `
  --submission `
  --require-official
```

开发 smoke 不加 `--submission` 时，缺少 CUDA、完整权重清点或重复采样只会报告为 warning；
`--submission` 会将这些条件升级为 blocker，并写出 `performance_audit.json`。该审计器不
推测缺失的性能数据，也不修改原始评测产物。

该入口的 `system_manifest.json` 汇总完整 provider inventory、典型路径和最重路径；
每条预测的 telemetry 保存完整系统 E2E、planner/executor/postprocess 分段、资源、
激活模型、TTFT、decode-only Token/s 和视觉输入统计。懒加载或非模型 provider 的
未知参数不会被估算，而是保留 `null` 并标记 `partial`。

## 当前覆盖状态

下表按当前代码状态更新。`已支持`表示代码和协议适配器已存在，但不等于已经在
完整官方 split 上跑出正式结果；只有满足上文的 provenance 条件才具有官方可比性。

| 成果要求 | 当前项目 | 判断 |
| --- | --- | --- |
| VRSBench | Grounding 官方确定性指标、Counting 内部指标、VQA/Caption 内部指标 | 部分完成 |
| MME-RealWorld-RS | 官方 A-E 单选 parser/scoring 已接入 | 代码已支持；完整官方运行未完成 |
| XLRS-Bench | 官方 VQA 单选和 Overall Land Use Classification 多选已接入 | 部分完成；Grounding/Caption 未实现 |
| LEVIR-CC | change caption、server-rule/local-judge 流程已存在 | 内部/临时协议，非官方可比 |
| VQA Accuracy / EM | Exact Match、normalized accuracy、macro QA-type accuracy | 已支持 |
| Grounding IoU | mean IoU 及相关诊断 | 已支持 |
| Acc@0.5 / Acc@0.7 | VRSBench Grounding，含 Unique / Non-Unique / All 分组 | 已支持 |
| Counting Exact Match | `exact_count_accuracy` | 已支持 |
| Counting RMSE | `rmse_on_parsed` | 已支持 |
| 单样本延迟 | standalone 和 TaskGraph 完整系统均有 telemetry；完整系统入口使用单样本 E2E | 已支持 |
| 峰值 GPU 显存 | allocated/reserved peak 已写入评测元数据 | standalone 已支持 |
| 峰值 CPU 内存 | CPU RSS peak 已写入评测元数据 | standalone 已支持 |
| 输出 Token 数 | 重新 tokenize 解码文本并记录 `output_token_count` | 已支持；不是 decode-only Token/s |
| TTFT | Transformers 首个 logits step hook；不支持 hook 的后端显式 unavailable | 已支持 |
| Decode-only Token/s | 首 token 到生成结束的纯生成阶段计时 | 已支持 |
| 视觉 Token 数 | `image_grid_thw` 按视觉 merge area 统计 | 已支持；后端无 grid 时为 null |
| 图像 resize / tile / crop 统计 | 原图尺寸、处理后 grid/尺寸、tile/crop 数量逐样本记录 | 已支持 |
| 完整系统参数量 | TaskGraph trace 和 run manifest 均记录 provider inventory | 已支持；未知懒加载模型显式标 partial |
| 各路径激活参数量 | 典型路径、最重路径和路径分布均写入 system manifest | 已支持；未知模型不估算 |
| 全部权重实际存储 | standalone 和已加载本地 provider 均统计权重文件大小 | 已支持；远程/懒加载权重可为 null |
| 模型加载/冷启动时间 | standalone 模型加载和完整系统 runtime 初始化/首请求路径均记录 | 已支持；provider 懒加载时间包含在首请求 E2E |
| 失败样本数 | 批失败自动退回单样本；失败记录写入 JSONL，并同步到 summary/manifest | 已支持；失败样本仍需按提交规则决定是否纳入分母 |
| warmup / repeat | standalone 和完整系统入口均支持配置、执行和元数据记录 | 已支持 |
| OS/CPU/GPU/driver/CUDA/framework | 运行环境 manifest 已采集 | standalone 和完整系统均已支持 |
| Prompt/version/hash | MME/XLRS provenance、输入/配置/契约 hash、完整系统 prompt hash | 已支持；真实运行前仍需锁定最终配置 |
| 官方协议与自定义协议隔离 | official/internal 标签和 `official_comparability` 已有 | 部分完成；仍有未完成官方任务 |

本表不把旧量化 benchmark 作为本次成果要求。量化目录可以保留作历史或内部实验，
但不应进入正式评测、资源统计或成果提交链路。

## 运行评测

```powershell
python scripts/evaluation/evaluate_predictions.py `
  --config configs/eval/evaluation_v1_5.yaml `
  --predictions outputs/evaluation/model/predictions.jsonl `
  --output-dir outputs/evaluation/model/v1.5 `
  --latency-semantics batch_amortized_per_sample `
  --eval-batch-size 16 `
  --group-by-task
```

输出目录默认拒绝覆盖已有非空目录：

```text
evaluated_predictions.jsonl
metrics.json
summary.json
evaluation_manifest.json
```

`metrics.json` 是标准指标产物，`summary.json` 为兼容 Evaluation v1.5 原始命名的同内容
别名。模型生成、量化 baseline 和量化 candidate 均调用同一 runner。

## 配对比较

```powershell
python scripts/evaluation/compare_evaluations.py `
  --config configs/eval/evaluation_v1_5.yaml `
  --baseline-dir outputs/evaluation/baseline/v1.5 `
  --candidate-dir outputs/evaluation/candidate/v1.5 `
  --output-dir outputs/evaluation/comparisons/baseline-vs-candidate `
  --bootstrap-resamples 1000 `
  --seed 20260806
```

比较前会强制核验ID集合、任务类型和参考答案，随后输出逐样本改善/退化结果和配对Bootstrap 95%置信区间。

## 统一绘图

绘图工具读取一个或多个完整评测目录；每个目录必须包含 `summary.json`（与
`metrics.json` 内容相同），存在
`evaluated_predictions.jsonl` 时会额外绘制Grounding IoU CDF和Counting误差分布。配对比较目录必须包含
`comparison_summary.json`。

安装仓库已有的可选绘图依赖：

```powershell
python -m pip install -e ".[plot]"
```

运行示例：

```powershell
python scripts/evaluation/plot_evaluation_results.py `
  --config configs/eval/evaluation_v1_5.yaml `
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

- VRSBench Counting/VQA/Caption 和 LEVIR-CC 当前自动指标均为内部指标。
- VRSBench Grounding 的 official_acc_at_*_{unique,non_unique,all} 与官方确定性 evaluator
  对齐；其它 Grounding 诊断仍保留为 internal。
- MME-RealWorld-RS MCQ 与 XLRS VQA 使用固定官方 parser/scoring；XLRS-lite macro
  只有在完整官方任务覆盖下可直接与 lmms-eval 比较。
- XLRS Grounding/Caption 仍为已注册未实现，因为公开仓库尚未提供可核验 evaluator。
- VRSBench VQA/Caption 的官方最终版本使用 GPT-based 评审，本地离线链路不自动调用外部模型，
  因此继续单独标记为 internal。
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

# 初赛评测与成果要求对照核验

核验日期：2026-09-03
核验对象：`sat-rs-vlm` 当前本地工作树
依据材料：

- `第一次统一答疑0801.pdf`
- 项目内关于“将资源统计并入主评测链路和完整系统 Executor”的说明

## 一、结论

当前项目已经基本落实了答疑材料中关于“可审计评测工程”和“完整系统资源统计”的技术要求，主要包括：

- 有真正的完整系统评测入口，而不是只测单个 VLM 的 `generate()`。
- 资源统计已经进入主评测链路，逐样本保存端到端时间、资源、路由和视觉输入信息。
- 已支持 TTFT、输出 Token 数、decode-only Token/s、视觉 Token、resize、tile/crop 等字段的记录。
- 已支持完整系统模型清单、典型路径和最重路径的参数统计与权重存储统计。
- 已支持冷启动、warmup、repeat、失败样本、环境信息和 prompt provenance。
- MME-RealWorld-RS 的官方单选协议、XLRS-Bench 的单选和多选协议已在当前 `v1.8` contract 中注册为 `implemented`。
- 量化 benchmark 没有继续作为本次成果检查主链路，符合说明中的处理原则。

但目前仍不能写成“所有官方数据集结果已经完成并可直接提交”。原因是：

1. VRSBench VQA/caption 当前仍是 `implemented_internal_only`。
2. LEVIR-CC 当前仍是 `implemented_internal_only`，现有实现是内部的 server-rule/local-judge 流程，不等同于官方完整协议。
3. XLRS 当前 contract 明确覆盖的是官方 MCQ 和 multi-select；XLRS grounding/caption 尚未形成对应的官方 adapter。
4. 本地机器没有 CUDA，当前只能验证代码、Mock、schema、审计和 smoke 流程，不能提供真实模型的 GPU 显存、真实 TTFT、Token/s 和正式模型参数结果。
5. fake smoke 只能证明产物链路正确，不能作为模型效果、真实性能或官方榜单结果。

因此，准确判断是：**工程基础和资源审计链路基本达到要求；官方协议覆盖和真实云端实验结果尚未全部达到最终提交状态。**

## 二、答疑要求与当前实现

| 答疑要求 | 当前实现 | 判断 |
| --- | --- | --- |
| 基于 VRSBench、MME-RealWorld-RS、XLRS-Bench、LEVIR-CC 开展验证 | 四个数据集均有数据/协议识别入口，MME 和 XLRS 部分官方协议已接入 | 部分达到，需补齐各数据集正式运行 |
| 官方协议结果与调整后的结果分开报告 | contract 区分 `official`、`internal`、`implemented_internal_only`，评测结果带 protocol provenance | 工程上已具备，需正式数据运行确认 |
| 不强制统一 prompt、缩放、切片和视觉 Token 预算，但必须披露 | 保存 prompt profile/version/hash、原始/处理尺寸、视觉网格和 tile/crop 记录 | 已达到 |
| VQA Accuracy / Exact Match | 已有 Exact Match、normalized accuracy、QA type macro accuracy | 已达到底层指标要求 |
| Grounding IoU、Acc@0.5、Acc@0.7 | VRS Grounding 已有 mean IoU、GIoU、中心距离和 Acc@0.5/0.7 | 已达到底层指标要求 |
| Counting Exact Match、RMSE | 已有 exact count accuracy、绝对误差、平方误差和 RMSE | 已达到 |
| 检测/变化区域 Precision、Recall、F1、IoU/mIoU | 项目已有相关指标组件，但应按对应公开数据集实际任务格式运行 | 已具备，待正式数据验证 |
| 完整系统参数总量不超过 32B | 已有完整 system inventory 和多模型遍历统计 | 统计机制已达到，真实总量待云端模型加载后确认 |
| 全部独立模型参数和权重实际存储占用 | `system_manifest.json` 保存模型清单、参数量、存储量和 accounting status | 已达到机制要求，真实数值待云端确认 |
| 动态路由的典型路径和最重路径激活参数 | manifest 保存 `paths.typical`、`paths.heaviest` 和 path distribution | 已达到 |
| 单样本端到端时间 | 主入口覆盖 preprocess、planner、executor、postprocess 和 e2e | 已达到机制要求 |
| 峰值 GPU 显存和峰值 CPU 内存 | telemetry 保存 allocated/reserved GPU memory 和 peak CPU RSS | 已达到机制要求，真实 GPU 数值待云端 |
| TTFT、输出 Token 数、Token/s | generation telemetry 支持 TTFT、output token count、decode-only Token/s | 已达到机制要求，fake 运行无真实生成值 |
| 视觉 Token、resize、tile/crop | `vision_input` 保存 visual token、原始/处理尺寸、image grid、tile/crop 记录 | 已达到机制要求 |
| 冷启动和一次性模型加载时间 | `cold_start` 独立记录，和正常样本 e2e 分开 | 已达到 |
| 测试数、失败数、warmup、repeat、batch/cache | manifest 和 metadata 保存这些字段，失败样本逐条保留错误信息 | 已达到 |
| OS、CPU、GPU、driver、CUDA、框架、精度和配置 | runtime environment、provider inventory、配置 hash 已接入 | 已达到机制要求，真实云端环境待采集 |
| 离线、自动、连续运行，不调用在线模型 API | real config 使用本地模型路径和 `local_files_only`，主评测为连续自动入口 | 工程设计符合，需云端运行日志最终确认 |

## 三、公开数据集协议状态

当前正式 contract：`configs/eval/evaluation_contract_v1.8_local_complete.yaml`。

| 数据集/任务 | contract 状态 | 当前含义 | 提交风险 |
| --- | --- | --- | --- |
| VRSBench Grounding | `implemented`，含 `vrsbench_grounding_v1` | 已按定位协议计算 IoU、Acc@0.5、Acc@0.7，并支持 unique/non-unique 分层 | 低，仍需用官方完整 split 实跑 |
| VRSBench Counting | `implemented` | 已有正式计数误差指标 | 中，需核对官方输入输出格式和完整数据覆盖 |
| VRSBench VQA | `implemented_internal_only` | 有内部文本评分，不应直接当官方榜单结果 | 中高 |
| VRSBench Caption | `implemented_internal_only` | 有内部 caption 评分，不应直接当官方协议结果 | 中高 |
| MME-RealWorld-RS MCQ | `implemented`，`official` | 已配置官方选择题解析、exact match 和 weighted micro accuracy | 中，需正式数据和原始 prompt 验证 |
| XLRS-Bench single-choice | `implemented`，`official` | 已配置官方单选解析和 macro/micro 聚合 | 中，需正式数据和语言版本分别运行 |
| XLRS-Bench multi-select | `implemented`，`official` | 已配置集合 exact match 和 macro/micro 聚合 | 中，需正式数据和语言版本分别运行 |
| XLRS-Bench grounding/caption | 当前没有对应完整官方 adapter | 不能声称 XLRS 全任务官方覆盖 | 高 |
| LEVIR-CC change caption | `implemented_internal_only` | 当前是内部 server-rule/local-judge 流程，需单独标记 | 中高 |

PDF 对 XLRS 中英文评测的表述是“鼓励但不是必须同时提交”。因此只做一种语言可以接受，但必须在报告中写清楚数据版本、语言、任务和结果范围，不能把中英文混成一个数字。

## 四、已经落地的工程证据

### 1. 完整系统评测入口

入口：`scripts/evaluate_taskgraph.py`

该入口从 JSONL 读取样本，执行完整 TaskGraph runtime，并生成：

- `predictions.jsonl`
- `evaluation_metadata.json`
- `system_manifest.json`
- `evaluation/metrics.json`
- `evaluation/summary.json`
- `evaluation/evaluated_predictions.jsonl`
- `evaluation/evaluation_manifest.json`

每个样本同时保存执行 trace 和 telemetry，不再把单个模型生成耗时误称为完整系统 e2e。

### 2. 系统级资源与性能统计

主要实现位于：

- `src/sat_rs_vlm/infrastructure/telemetry.py`
- `src/sat_rs_vlm/evaluation/performance_audit.py`
- `src/sat_rs_vlm/taskgraph/runtime.py`
- `src/sat_rs_vlm/models/hf_vlm_engine.py`

已覆盖：

- preprocess、planner、executor、postprocess、e2e、TTFT；
- generated/output tokens 和 decode-only Token/s；
- visual token count、原始尺寸、处理尺寸、image grid；
- tile 数、crop 数及 provider 级记录；
- peak CPU RSS、peak GPU allocated/reserved；
- provider/model inventory、参数量、dtype、模型文件实际存储；
- cold start、warmup、repeat、failure count；
- OS、Python、PyTorch、Transformers、CUDA/GPU 信息；
- prompt profile、version、hash、输入文件 hash、contract hash、Git provenance。

### 3. 动态路径统计

`system_manifest.json` 记录：

- 完整系统模型与权重清单；
- 实际执行路径分布；
- 典型路径 `paths.typical`；
- 参数量最大的最重路径 `paths.heaviest`；
- 每条路径的 activated providers、已知参数量和存储量；
- 无法真实计量时的 `partial` 状态，而不是填入猜测值。

## 五、本地验证结果

当前 `.venv` 环境检查结果：

- 基础依赖：齐全；
- 模型依赖：齐全，包括 `transformers`、`safetensors`、`fastapi`；
- CUDA：不可用；
- 输出目录：可写。

代码验证结果：

```text
767 passed, 9 skipped
```

fake 完整系统 smoke 已成功生成全部 7 类评测产物，并验证了：

- `failed_samples = 0`；
- warmup/repeat 字段存在；
- cold start 字段存在；
- prompt aggregate hash 存在；
- typical/heaviest path 字段存在；
- fake provider 没有真实权重时，参数和存储 accounting 正确显示为 `partial`。

这证明的是评测链路和审计结构可运行，不代表真实模型性能结果已经完成。

## 六、提交前剩余工作

| 工作项 | 是否需要 GPU | 必要性 | 说明 |
| --- | --- | --- | --- |
| 用真实 Qwen、LAE-DINO、GeoRSCLIP 跑完整系统 | 是 | 必须 | 采集真实路径、参数、显存、TTFT、Token/s 和视觉 Token |
| 用正式四数据集和固定 split 运行 | 通常是 | 必须 | 保存数据版本、语言、split、输入 hash 和原始日志 |
| 补 VRSBench VQA/caption 官方 profile | 编码不需要，实跑通常需要 | 必须 | 不能继续只报 `implemented_internal_only` 作为官方结果 |
| 补 LEVIR-CC 官方可比协议 | 编码不需要，实跑通常需要 | 必须 | 内部 judge 结果与官方结果分开报告 |
| 补 XLRS grounding/caption 官方 adapter | 编码不需要，实跑通常需要 | 按任务需要 | 如果成果范围包含这些 XLRS 子任务，则必须补齐 |
| 核对总参数量不超过 32B | 不需要 | 必须 | 依据所有独立模型 manifest 和实际加载模型确认 |
| 整理正式报告表格和图 | 不需要 | 必须 | 只使用真实正式 run，不使用 fake smoke 数字 |
| 提交前清理工作树、固定 commit 和 provenance | 不需要 | 必须 | 只纳入代码、配置、脚本和必要报告，不纳入实验缓存 |

## 七、最终建议表述

目前对外汇报可以写：

> 项目已经完成面向完整 TaskGraph 系统的可审计评测链路建设，覆盖官方/内部协议区分、端到端性能、Token 与视觉输入预算、模型参数和权重存储、典型/最重路径、冷启动、失败样本、运行环境和 prompt provenance。当前本地已完成代码级和 Mock 级验证；MME-RealWorld-RS 及 XLRS 的部分官方协议已接入。VRSBench VQA/caption、LEVIR-CC 官方可比协议以及真实模型在云端的正式资源数据仍需在提交前完成并单独报告。

不建议写成：

> 四个数据集官方评测和所有真实性能数据已经全部完成。

因为这一表述目前会把内部协议、Mock 结果和真实官方 run 混在一起，和答疑材料要求的“真实、完整、可复现、口径分开”不一致。

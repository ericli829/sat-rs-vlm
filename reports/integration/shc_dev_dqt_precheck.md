# SHC / dev-dqt 整合前检查

## 仓库基线

- 目标项目：`sat-rs-vlm`，当前分支 `server-adaption`。
- 只读来源：同级 `sat-rs-vlm-temp`，来源工作区状态为空，当前检出 `shc`。
- SHC：`shc` / `d309e57d2c16d88b49e1f37c020811903e12b9f7`。
- dev-dqt：本地分支名不存在；`origin/dev-dqt` 与给定 SHA
  `40d2806ad7135aeaa213462adac7f5182ab3f2d1` 一致。
- 读取方式：只使用 `git show`、`git diff`、`git log` 和 `git rev-parse`；没有 checkout、
  merge、cherry-pick、rebase、stash、commit 或 push。

## 目标工作区状态

整合前工作区不是干净工作区，原有修改必须完整保留：

- 已暂存：78 个服务器适配文件，主要位于 `configs/base|cloud|local`、`configuration/`、
  `data/manifest.py`、`scripts/environment|training|storage`、环境约束和双环境测试。
- 已暂存后又修改：`docs/training_commands.md`、三个 `scripts/data/*.py`、
  `scripts/training/run_train.py`、`configuration/layered.py`、`training/experiment.py`。
- 未暂存修改：README、Makefile、pyproject、数据/评测配置、原 LoRA 训练与评测脚本、
  数据转换脚本、插件脚本、可靠性兼容模块和测试入口。
- 未跟踪：5 个可靠性配置、4 份可靠性文档、5 个可靠性脚本/入口、
  10 个可靠性源码文件和 11 个可靠性测试文件。
- `git diff --stat`（未暂存部分）：31 files changed, 549 insertions, 350 deletions。
- `git diff --cached --stat`：78 files changed, 3772 insertions。

关键保护范围：

```text
configuration/               分层配置、环境变量和本地/AutoDL 路径
training/experiment.py       实验目录、manifest 和运行记录
data/manifest.py             数据集 manifest
plugins/                     外部微调插件边界
models/reliability/          bit flip、checksum、输出验证、保护和恢复
evaluation/reliability/      可靠性指标与报告
scripts/reliability/         可靠性实验入口
```

来源可能覆盖的现有文件包括 `README.md`、`Makefile`、`pyproject.toml`、
`configs/data/remote_sensing_data.yaml`、`configs/eval/qwen3vl_eval.yaml`、
`scripts/train_qwen3vl_lora.py`、`scripts/evaluate_rs_vlm.py`、
`data/qwen3vl_collator.py`、`data/vrsbench.py` 和现有测试。上述文件只能按功能编辑。

## 来源提交范围

SHC 相对目标基线新增/修改 44 个文件，其中包含 18000 条 E1 派生 JSONL。有效代码为：

- prompt 模板、assistant-only mask、task sampler、分任务指标；
- VRSBench detection/counting 结构化输出和场景问题映射思路；
- E0/E1/E1b/E1d 配置以及分层/配额数据生成脚本；
- 训练和评测链路接入测试。

dev-dqt 相对目标基线修改 12 个文件。有效代码仅来自
`quantize_int8.py` 和 `quantize_int8_cpu.py` 的后端与 benchmark 思路；四个缩减后的 processed
JSONL、三个本地报告、绝对路径和 README 实验结论不作为迁移来源。

## 功能对照

| 能力 | 目标项目 | shc | dev-dqt | 重复情况 | 最终方案 |
| --- | --- | --- | --- | --- | --- |
| assistant-only mask | 仅 pad mask/TODO | 有 | 无 | SHC 补缺 | 在现有 collator 中重写并校验空监督 |
| prompt 模板 | 分散在转换器 | 有 | 无 | 部分重复 | 建立唯一 `data/prompt_templates.py` |
| detection JSON | VRSBench 已输出 | 有 | 仅评测使用 | 部分重复 | 共用结构化协议和 parser |
| counting JSON | 未统一 | 有，数字支持有限 | 无 | SHC 需修复 | 统一 `{"count":n}`，支持英文数字与 unresolved |
| bbox 解析 | reliability validator 内有 | 有 | 两脚本各一套 | 三套 | 抽取 task protocol，指标和 validator 分层复用 |
| 任务指标 | reliability 指标与旧 eval 摘要 | 有 | 错误的包含关系准确率 | 三套 | 统一 `evaluation/metrics.py` |
| Qwen3-VL 加载 | 训练、eval/checkpoint loader | 无新增 | 各脚本重复 | dev-dqt 重复 | 量化复用兼容模型类与现有设备工具 |
| LoRA adapter 加载 | checkpoint loader/eval 已有 | 使用现有逻辑 | 未统一 | 重复 | 提供共享推理 loader，不另写 PEFT 流程 |
| 推理生成 | `evaluate_rs_vlm.py` | 增加任务指标 | 两套重复 | 三套 | 抽取统一生成 helper，解码新增 token |
| 数据样本解析 | Qwen3VLDataset/messages | 同协议 | CPU 脚本部分支持 | 重复 | benchmark 只读 messages Dataset |
| CPU INT8 | 无 | 无 | 有 | 新能力 | `torch_dynamic_int8` 后端，默认 benchmark-only |
| bitsandbytes INT8 | 无 | 无 | 有 | 新能力 | `bnb_int8` 后端，可选依赖且能力探测 |
| 性能 benchmark | profiler 基础统计 | 无 | 有两套 | 部分重复 | 统一 latency/memory/artifact 报告 |
| 输出验证 | reliability validator | 指标 parser | 简单 JSON 检查 | 职责交叉 | 共享 parser，保留验证/指标/benchmark 上层边界 |
| 可靠性报告 | 已有稳定 schema | 无 | 无 | 无 | 扩展公共 prediction 字段，不覆盖可靠性报告 |
| 配置解析 | 分层 YAML + 环境展开 | 独立 YAML | argparse/硬编码 | 三套 | 复用 `configuration`，Pydantic 量化配置 |
| 报告 schema | eval 与 reliability 各有 | 扩展 eval | 两套 dataclass | 重复 | 建立 JSON-safe 量化 manifest 和统一 prediction 基础字段 |

## 预定处理

1. 先统一 task protocol、坐标格式、prediction schema 和 task metrics。
2. 将 SHC 功能嵌入当前 Dataset/Collator/训练/评测边界，不覆盖脚本。
3. 建立 `compression/quantization`，用一个入口承载 baseline、CPU dynamic INT8 和 bnb INT8。
4. E1d 拆为 data、sampler、combined 三种配置语义。
5. 不复制任何完整 processed JSONL、checkpoint、本地报告、绝对路径配置或临时应用脚本。

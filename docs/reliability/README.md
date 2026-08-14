# 可靠性实验模块

该模块用于研究单粒子翻转（single-event upset）对遥感 VLM LoRA Adapter 和输出的影响，
并提供可重复的本地流程验证与云端真实推理入口。它是工程实验工具，不是经过航天认证的
容错实现。

## 能力范围

| 层级 | 能力 | 支持类型 | 当前状态 |
|---|---|---|---|
| 基础值 | 定点/随机、单 bit/多 bit、固定 seed | bytes、bytearray、非负整数 | 本地已测试 |
| Tensor | 定点/随机、结构化记录、默认复制 | float16、bfloat16、float32、int8、uint8 | 本地 CPU 已测试 |
| State dict | 名称、正则、模块、层号和随机目标 | 上述 tensor dtype | 本地 CPU 已测试 |
| LoRA Adapter | 全部 LoRA、A、B、模块、层、正则 | safetensors | 小型 Adapter 本地已测试 |
| 文件完整性 | SHA-256 manifest 构建/验证 | 任意文件 | 本地已测试 |
| 输出层 | 通用、counting、detection、VQA 验证 | 文本、JSON、统一结果字典 | 本地已测试 |
| 保护/恢复 | no protection、checksum recovery、output guard vote、weight clamp | 文件、输出、state dict | 本地 smoke 已测试 |
| 真实实验 | Qwen3-VL clean/fault/recovered 推理 | 标准 LoRA checkpoint | 代码已实现，等待 AutoDL 验证 |
| v1.5 sensitivity | target/layer/bit-plane/repeat + paired Bootstrap | E1/E2/E3 | 本地控制流已测试 |
| Activation Guard | research 记录、deployment fail closed | NaN/Inf/max_abs | 本地已测试 |
| 冗余恢复 | checksum、warm/golden replica、scrub | Adapter bundle | 本地已测试 |

每条 `BitFlipRecord` 包含目标名、元素/字节/bit 地址、dtype、shape、翻转前后值或
字节、随机 seed。随机注入在候选 bit 地址空间中无放回抽样，固定 seed 可复现。

## 保护层级

- `no_protection`：不修改故障输出，只作为影响基线。
- `checksum_recovery`：文件级检测，从干净备份复制到临时文件，校验后原子替换，再校验。
- `output_guard_vote`：输出层合法性过滤、字符串/数值规范化和多数投票；无合法值时显式 fallback。
- `weight_clamp`：根据干净权重范围生成新的修复 state dict。该方法依赖干净参考范围，属于实验性
  启发式方法，不是完整纠错方案。

checksum recovery、output guard 和 weight clamp 分别作用在文件层、输出层和参数层，报告不能
直接横向解释为同一种保护强度。

## 两种执行模式

`smoke_mock` 在 CPU 上使用固定预测和小型权重，只验证注入、记录、保护、恢复、指标与目录流程。
所有报告均包含 `execution_mode: smoke_mock`，结果不代表真实 Qwen3-VL 鲁棒性。

`real_inference` 复用现有 `scripts/evaluate_rs_vlm.py`、checkpoint loader、Processor、
Qwen3-VL/PEFT 模型加载和 Qwen3VL Dataset/Collator。缺少 CUDA、模型依赖、数据或标准 Adapter
checkpoint 时直接失败，绝不降级为 Mock。评测通过 `--output-dir` 写入可靠性运行目录，干净
Adapter 保持只读。

## 输出结构

```text
${OUTPUT_ROOT}/reliability/<experiment>/<run_id>/
├── config_resolved.yaml
├── command.txt
├── environment.json
├── git_commit.txt
├── run_report.json
├── logs/
├── clean/predictions.jsonl
├── faults/
│   ├── injection_records.jsonl
│   └── adapters/
├── predictions/clean_fault_pairs.jsonl
├── metrics/
│   ├── summary.json
│   └── by_task.json
├── protection/strategy_results.json
├── figures/
└── artifacts/
```

同名运行默认拒绝覆盖；只有显式 `--overwrite` 才会替换该 run。`--resume` 必须配合现有
`--run-id`，并复用已经生成的 clean/fault 预测。

## 当前限制

- 默认指标包含 changed/invalid/empty/exact match/recovery，以及 counting MAE/accuracy 和首框 IoU；
  正式论文实验仍应补充完整 mAP、CIDEr、BLEU/ROUGE 和遥感任务指标。
- weight clamp 真实模式当前生成独立修复 Adapter 和统计；默认正式配置未启用，仍需在 AutoDL
  单独验证其修复后推理和收益。
- bit flip 是软件级故障模型，不模拟缓存层次、总线、电磁环境或真实器件故障率。
- bfloat16/float16/float32 Adapter 已由本地小文件覆盖；大型分片或非标准 PEFT 文件布局需另测。

Evaluation v1.5 sensitivity、恢复运行和 task-metric risk policy 的操作方法见
[SEU Sensitivity 与保护策略](seu_v15_operation.md)。Reliability sweep 默认 E1，训练后模型
评测默认 E2；两者用途不同。

后续可扩展 ECC、故障时间模型、量化权重注入和卫星平台硬件在环实验。Consensus Guard 当前只
保留 detector/recovery 接口，尚未实现未经验证的 generalist/specialist ensemble。


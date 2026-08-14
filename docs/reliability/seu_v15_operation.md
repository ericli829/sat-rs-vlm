# SEU Sensitivity 与保护策略

本流程对已加载的 Qwen3-VL/LoRA 参数执行内存 bit flip，checkpoint 始终只读。每个故障条件
使用与 clean baseline 完全相同的 Evaluation v1.5 固定 tier，并通过 paired Bootstrap 比较
任务指标。

## Tier 语义

- E1：默认 sensitivity sweep，593 条固定诊断样本。
- E2：确认 E1 发现的高风险 target/layer/bit-plane。
- E3：仅用于最终少量关键故障和正式报告。

baseline 与 fault condition 的 `evaluation_tier` 和 `evaluation_tier_sha256` 必须相同，否则
paired comparison 会拒绝执行。训练后的默认模型评测仍是 E2，不受可靠性 sweep 默认 E1 的
影响。

## 预检与计划

```bash
python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --preflight

python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --run-id seu-e1-pilot --dry-run
```

`condition_plan.json` 绑定故障条件、seed、evaluation tier、数据 SHA256 和评测 contract。恢复时
如果计划不一致会立即失败，不能悄悄继续另一个实验。

## 正式运行与恢复

```bash
python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --run-id seu-e1-pilot \
  --activation-guard --activation-guard-mode research

python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --run-id seu-e1-pilot --resume
```

`research` guard 记录 NaN、Inf 和超阈值 activation，但不会把检测成功误标为程序失败；
`deployment` guard 遇到相同异常时 fail closed。

## 结果和策略

```text
<OUTPUT_ROOT>/reliability/v15_sensitivity/<run-id>/
├── condition_plan.json
├── sensitivity_progress.json
├── sensitivity_report.json
├── sensitivity_groups.json
├── baseline/evaluation_v1_5/
└── conditions/<condition>/
```

`sensitivity_report.json` 保存原始 repeats；`sensitivity_groups.json` 按 target、layer、
bit-plane、fault intensity 聚合 mean/std/95% CI，并保留 Detection、Counting、Scene、VQA、
Caption、LEVIR 的任务退化指标。

```bash
python scripts/reliability/build_protection_policy.py \
  --sensitivity-summary <run-dir>/sensitivity_report.json \
  --output <run-dir>/protection_policy.json
```

脚本同时生成 `sensitivity_groups.json`。checksum、warm/golden replica、scrub 和 output guard
是当前软件实现；ECC、dual execution、TMR 和硬件 rollback 只作为平台集成建议，不宣称已经
由本仓库实现。

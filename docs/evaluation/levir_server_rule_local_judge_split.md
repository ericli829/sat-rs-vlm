# LEVIR-CC 服务器规则评测与本地语言模型补全

## 目的

服务器端评测不得依赖本地语言模型权重或 `transformers` 环境。变化 Caption 的完整语义判定分为两个互不阻塞的阶段：

```text
原始 predictions.jsonl
  ├─ 服务器：确定性规则路由 + 基础评测
  └─ 本地：仅对服务器未决 Caption 调用文本小模型 + 全量变化检测评测
```

Caption 的 BLEU、ROUGE-L、METEOR、chrF、CIDEr-D 和所有非 LEVIR 任务指标始终由服务器基础评测计算，不等待本地语言模型。
当前推荐流程不重新启用 v1.6 的第二次图像二分类推理；主 VLM 只生成一次自然语言 Caption，后续判定只消费 Caption 文本。

## 一、服务器：无语言模型规则路由

该步骤只使用 Python 规则，不导入或加载 Qwen、Torch、Transformers 或 LoRA：

```powershell
python scripts/evaluation/route_change_captions_rules.py `
  --predictions <raw_predictions.jsonl> `
  --output-dir <server_rule_routing_dir>

python scripts/evaluation/evaluate_predictions.py `
  --predictions <server_rule_routing_dir/rule_routed_predictions.jsonl> `
  --output-dir <server_evaluation_dir> `
  --contract configs/eval/evaluation_contract_v1.8_server_rule_only.yaml
```

规则只处理三类高置信 Caption：

- 完整无变化表达；
- 明确建筑、道路等永久结构发生新增、拆除、扩建等变化；
- 明确只描述车辆、光照、阴影、植被或成像差异等非目标变化。

其余 Caption 写入：

```text
prediction_changeflag = null
binary_prediction_source = server_rule_unresolved
```

并进入 `local_judge_queue.jsonl`。服务器结果中的二分类指标标记为 `partial_coverage`；必须同时展示 `binary_decision_coverage` 与 `binary_unresolved_rate`，不能作为完整 LEVIR-CC 二分类总分。

带有提示注入/指令样式的输入不会进入本地模型队列，而是写入
`manual_audit_queue.jsonl`（来源为 `server_input_guard`）。因此 `local_judge_queue.jsonl`
严格只包含 `server_rule_unresolved`，两类队列不能合并。

## 二、本地：仅补判未决 Caption

本地小模型只处理 `server_rule_unresolved`，不会再次推理已被服务器规则判定的样本：

```powershell
python scripts/evaluation/judge_change_captions.py `
  --predictions <server_rule_routing_dir/rule_routed_predictions.jsonl> `
  --model <local_qwen3_1.7b_path> `
  --only-unresolved `
  --output-dir <local_judge_dir>

python scripts/evaluation/evaluate_predictions.py `
  --predictions <local_judge_dir/judged_predictions.jsonl> `
  --output-dir <local_complete_evaluation_dir> `
  --contract configs/eval/evaluation_contract_v1.8_local_complete.yaml
```

本地阶段只在拥有本地 Qwen3-1.7B（可选 LoRA）权重的设备执行。`U` 或非法小模型输出保留在审计队列，不应静默改写为 0 或 1。

`evaluation_contract_v1.8_local_postjudge.yaml` 是明确的 post-router 契约；旧的
`evaluation_contract_v1.8_local_complete.yaml` 保留为兼容别名。若仍有 `U`，结果会明确标记
覆盖率和审计状态，不能把它包装成完整二分类分数。

## 三、报告规则

| 结果类型 | 可报告内容 | 限制 |
|---|---|---|
| 服务器规则结果 | 全部传统文本指标、性能指标、规则覆盖率、已覆盖样本的部分二分类诊断 | 不得称为完整变化检测总分 |
| 本地补全结果 | 全量变化检测 Accuracy、F1、FPR、FNR、MCC、Kappa | 标明本地评审器版本和覆盖情况 |

二者统计分母不同，不能直接横向比较。所有输出目录均不可覆盖已有结果。

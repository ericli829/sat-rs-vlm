# ViT 主线与 Reliability v1.5 功能级整合报告

日期：2026-08-14  
工作分支：`vit`  
Reliability 来源：`origin/lyz-reliability-v15-migration`  
共同 merge base：`2d8504e960e5814016adec1aebdb9434889fd45f`

## 1. 分支职责

`vit` 是最终功能主线，保留以下正式实现：

- task-weighted/token-mean 可替换 loss；
- assistant-only labels；
- hard-example mining 与训练数据统计；
- LoRA + optional last-N ViT blocks + visual merger；
- LoRA/merger/ViT differential LR；
- trainable parameter audit；
- E1/E2/E3 固定分层评测与 tier SHA256 契约；
- Evaluation v1.5 paired comparison。

`lyz-reliability-v15-migration` 提供：

- target/layer/bit-plane in-memory bit flip；
- resumable v1.5 sensitivity；
- output degeneration guard；
- activation guard；
- checksum、warm/golden replicas、scrub/recovery；
- tiered risk policy 和 deployment guard。

本次没有执行粗粒度 `git merge`，而是先比较 `master..vit` 与
`master..origin/lyz-reliability-v15-migration`，再按模块职责迁移。

## 2. 冲突处理

### Fault injector

以 `vit` 已有 state-dict/Adapter 注入 API 为基础，加入：

- `all_parameters`、`lora_adapter`、`lora_a`、`lora_b`；
- `vision_encoder`、`visual_blocks`、`visual_merger`；
- `language_model`、`attention`、`mlp`、`embeddings`；
- sign/exponent/mantissa/all bit-plane；
- layer selector、模型 fault inventory、无放回地址抽样。

visual merger 的内部 MLP 会优先归类为 `visual_merger`，不会误归类成通用 LLM MLP。
checkpoint 文件保持只读，故障只注入当前进程中的已加载参数。

### Output validator

保留 `vit` 的 detection/counting/VQA 结构化校验，在其上加入 repeated character、repeated
token、symbol-only 和 long low-diversity guard。`yes`、`no`、`3`、`north` 等合法短答案
不会被误杀。

### Evaluation

没有引入第二套 evaluation。Reliability inference 继续调用现有
`scripts/evaluate_rs_vlm.py`，比较继续调用 Evaluation v1.5 comparison。

## 3. Reliability 最终流水线

```text
Clean model + fixed tier
        ↓
Evaluation v1.5 baseline
        ↓
in-memory fault condition
        ↓
same tier + same SHA256
        ↓
paired comparison + Bootstrap CI
        ↓
sensitivity_report.json (raw repeats)
        ↓
sensitivity_groups.json (target/layer/plane/intensity)
        ↓
task-metric degradation
        ↓
protection_policy.json
```

Risk policy 支持 Detection IoU/IoU@0.5、Counting exact/MAE、Scene/VQA normalized
accuracy、Caption ROUGE-L/CIDEr diagnostic 和 LEVIR balanced/binary accuracy。历史
changed/invalid/exact-match 字段只作为兼容 diagnostics。

Activation Guard 有两种模式：

- `research`：记录 `guard_triggered` 与完整 anomaly，condition 正常结束；
- `deployment`：检测到 NaN/Inf/max_abs 后 fail closed。

## 4. Evaluation tiers

- E1：Reliability 大规模 sensitivity sweep 默认层级；
- E2：确认 E1 高风险条件，也是正式训练后的默认模型评测；
- E3：少量最终关键 fault condition 和正式结论。

`condition_plan.json` 记录 tier、eval JSONL SHA256、contract 和全部条件。`--resume` 时计划
必须完全一致。baseline/candidate tier 或 SHA256 不一致时 paired comparison 拒绝执行。

## 5. Visual checkpoint

新 partial-ViT checkpoint 使用：

```text
adapter_model.safetensors
visual_trainable_weights.safetensors
visual_trainable_manifest.json
strategy_manifest.json
processor/
```

visual manifest 记录 selected ViT blocks、merger/optional visual、参数名、base checkpoint、
初始 adapter 和 SHA256。Evaluation loader 会校验 sidecar manifest/hash。历史
`h1_visual_weights.safetensors` 仍由 manifest 和目录加载兼容逻辑支持。

Reliability selector 可分别注入 LoRA、selected visual blocks 和 visual merger。

## 6. Dynamic step protection

新增 `training/training_plan.py`：

```text
effective_batch = per_device_batch × gradient_accumulation × world_size
steps_per_epoch = ceil(unique_samples / effective_batch)
resolved_steps = ceil(steps_per_epoch × target_effective_epochs)
```

H1 配置从固定 `max_steps: 1000` 改为：

```yaml
max_steps: null
target_effective_epochs: 1.5
max_effective_epochs: 2.0
allow_overtrain: false
```

显式 max_steps 超过上限默认失败。训练前打印并保存 unique samples、effective batch、
steps/epoch、resolved max steps、expected effective epochs 和 sample exposures。原始 LoRA 的
epoch/max_steps 配置继续兼容。

## 7. API 清理与兼容

Canonical API：

- `build_training_parameter_groups()`；
- `create_grouped_optimizer()`；
- `visual_trainable_weights.safetensors`；
- `sat_rs_vlm.quantization`；
- `aggregate_sensitivity_conditions()`。

保留兼容 wrapper：

- `build_h1_parameter_groups()`；
- `create_h1_optimizer()`；
- legacy H1 sidecar 文件名加载；
- `sat_rs_vlm.compression.quantization` 薄导入层。

量化审计确认真正实现只位于 `sat_rs_vlm.quantization`。旧 compression 路径仍有历史测试
引用，因此未删除，只保留无业务逻辑的 compatibility wrapper。

## 8. RTX 4090 profile

正式 H1 4090 profile 保持 single-GPU BF16、batch 4、gradient accumulation 4、eval batch
4、12 workers、pin memory 和 persistent workers。基础 LoRA 4090 profile仍为 batch 16、
gradient accumulation 1。

仓库已有 `scripts/training/benchmark_autodl_training.py`，可记录 samples/sec、peak VRAM 和
GPU utilization，并只在成功且显存安全的候选中给出建议。本机检测结果为 RTX 4060 Laptop
8GB，不是 RTX 4090，因此没有执行或伪造 4090 benchmark，前后性能数据与 peak VRAM 均记为
`not measured locally`。

AutoDL 短 benchmark：

```bash
python scripts/training/benchmark_autodl_training.py \
  --max-steps 20 --max-train-samples 512
```

## 9. 验证结果

- `python -m compileall -q src scripts`：通过。
- `python -m pytest tests/unit -q`：`251 passed, 26 skipped`。
- Reliability focused（首轮）：`62 passed, 15 skipped`；遗漏审计后新增 model-level
  injector/density/output-vote 覆盖并纳入完整 unit 结果。
- Integration（`PYTHONPATH=src`，排除缺少接口依赖的 CLI/HTTP/bootstrap）：
  `25 passed, 3 skipped`。
- Reliability plan dry-run：4 个 visual_blocks/visual_merger/LoRA exponent 条件，成功。
- `git diff --check`：通过。

本机默认 Anaconda 环境缺少 `torch`、`typer`、`fastapi`、`ruff` 和部分模型依赖，因此：

- torch/safetensors 单测按 marker 跳过；
- CLI/HTTP integration 未收集；
- `ruff check` 未执行；使用 Black/isort、compileall 和 `git diff --check` 做本地静态兜底；
- 未运行真实 Qwen3-VL inference、SEU E1 或 4090 training benchmark。

云端完整验证前应先执行 `pip install -e ".[dev,model,qlora]"`，再运行：

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --preflight
```

## 10. 未实现能力

- ECC、TMR、dual execution 和硬件 rollback 仅出现在策略建议中，需要目标硬件/运行时；
- Consensus Guard 只有 output/consensus/recovery Protocol 扩展点，没有实现未经验证的 ensemble；
- 没有重新训练正式模型，也没有改变 bbox protocol 或 Evaluation v1.5 数学定义；
- 没有删除历史 checkpoint、量化报告或评测资产。

## 11. 主要修改文件

新增：Reliability guards、redundancy、risk/sensitivity、v1.5 scripts/config/tests、
`training_plan.py`、SEU 文档和本报告。

修改：fault injector、output validator、LoRA training entry、training config/optimizer/
vision tuning、checkpoint loader、H1 YAML/文档、README，以及两个本地 smoke 兼容修复
（dataset script `src` bootstrap、quantization dry-run 不强制 torch）。

本次未 commit、未 push。

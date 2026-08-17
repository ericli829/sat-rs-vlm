# Qwen3-VL-4B Stage-A 集成报告

## Modified files

```text
README.md
configs/data/autodl_vrsbench_levircc.yaml
configs/data/autodl_qwen3vl_4b_stage_a.yaml
configs/eval/qwen3vl_4b_stage_a_e2_v2.yaml
configs/train/qwen3vl_4b_stage_a_multisource_4090.yaml
docs/training/qwen3vl_4b_stage_a_multisource.md
reports/integration/qwen3vl_4b_stage_a_multisource_report.md
scripts/data/prepare_multisource_training_data.py
scripts/train_qwen3vl_lora.py
scripts/training/run_autodl_qwen3vl_4b_stage_a.sh
scripts/training/run_qwen3vl_4b_stage_a.py
src/sat_rs_vlm/data/cyclic_training.py
src/sat_rs_vlm/data/prompt_templates.py
src/sat_rs_vlm/data/task_sampler.py
src/sat_rs_vlm/training/config.py
src/sat_rs_vlm/training/model_audit.py
tests/unit/test_coverage_first_sampler.py
tests/unit/test_cyclic_training.py
tests/unit/test_prepare_multisource_training_data.py
tests/unit/test_training_config.py
tests/unit/training/test_model_audit.py
tests/unit/training/test_stage_a_runner.py
```

## 基线与审计

- 分支：`master`；开始时 HEAD：`1ca8ddcc71d7e4980208c32796111264b752d1e0`。
- 历史 VRS quota：`seed + round_index * 1000` 后重新抽样，不保证 round 间互斥。
- 历史 LEVIR：`(round_index * variants_per_group) % group_size`，会在 cycle 尾部回卷。
- 历史 alternating sampler 是 batch-level 同源 pattern；其 epoch 长度取各 source 可完成
  pattern 数的最小值，不 oversample，但会截断较大 source 尾部。
- 每次旧 runner 调用都会重新创建 Trainer，因此 optimizer/scheduler 每轮重置；模型参数由
  `--initial-adapter` 串联。每轮 `num_train_epochs=1` 只表示对当轮 sampler 输出的一次 pass。
- 结论：历史实现是 deterministic re-sampling，不是可证明的 full-cycle coverage。

## 新架构

新增 `cyclic_full_coverage`，保留 `legacy_round_sampling`。VRSBench 按 task 做稳定 shuffle 与
不相交切片；LEVIR 按 image pair 对 variant 做不回卷切片；最后 partial bucket 保留。
`coverage_first` sampler 在 3:1 pattern 后排空尾部，不重复、不丢弃。cycle manifest 保存
每轮 JSONL SHA256、source/task 分布，以及全局/source/task/LEVIR 覆盖证明。

训练前读取 Unified E3 v2 IDs 并 fail-fast 检查泄漏。4B loader 继续使用共享
`compatible_model_class()`；model 与 processor 均来自 `QWEN3VL_4B_MODEL_DIR`。LoRA 注入前
逐 target 审计，续训前比较 adapter/base fingerprint；输出 manifest 保存 trainable audit。

## 本地证据

本地 synthetic fixture 覆盖 10 条样本、3 个 round，结果为：unique=10、coverage=100%、
duplicate=0、missing=0；单 LEVIR pair 的 5 个 variants 按 `2,2,1` 完整覆盖。coverage-first
fixture 对 13 个不均衡 source 样本全部暴露一次，历史 truncate fixture 仍保持 12 次暴露。

本机当前无法访问 VRSBench 原始图像，也没有被忽略的 `tiers_v2` 冻结 JSONL/manifest，且
没有本地 4B 权重。因此正式 population 的 round 数、每轮分布、source 实际比例、4B target
命中数和 trainable 参数量必须由 AutoDL `prepare-only` / `forward-only` 生成；此处不伪造。
正式 prepare 后，可直接从 `cycle_manifest.json` 读取真实 round 数、逐 round VRS/LEVIR
数量与 task distribution、unique samples、coverage/duplicate rate、LEVIR variant coverage
和 source exposure ratio。真实 forward 后，可从每轮 `strategy_manifest.json` 读取七类
LoRA target 的命中数、trainable parameter count 与 ratio。

## Stage-A 参数

- batch `4`，accumulation `4`，effective batch `16`，eval batch `2`。
- 每 bucket `1 epoch`，vision frozen，LoRA `r16/alpha32/dropout0.05`。
- round 0 `2e-5`；后续 `1e-5`；cosine、warmup `0.03`、weight decay `0.01`。
- task-weighted loss，六类任务权重均为 `1.0`；assistant-only mask 未修改。
- 训练期间 `do_eval=false`；完成后默认 Unified E2 v2。

## 验证状态

- `compileall`：通过。
- 完整 unit tests：`291 passed, 26 skipped`；新增定向测试全部通过。
- 全量 pytest：默认 Anaconda 环境缺少既有接口测试所需的 `typer`、`fastapi`，在
  integration collection 阶段出现 2 个 dependency errors；没有测试执行失败。
- Ruff：默认环境未安装，未执行；Mypy 对新增核心模块与 runner 检查通过。
- 真实 4B forward、LoRA target 数量、trainable 参数量、显存和正式训练：未在本机执行。
- 正式 AutoDL cycle 的真实统计将在 `cycle_manifest.json`、每轮 report、
  `strategy_manifest.json` 与 `stage_a_result.json` 中落盘。

## AutoDL

完整命令与 resume、日志查看方式见
`docs/training/qwen3vl_4b_stage_a_multisource.md`。执行顺序为 prepare-only、dry-run、
forward-only，三者通过后再后台启动正式 full cycle。runner 默认在最终 adapter 上运行
`configs/eval/qwen3vl_4b_baseline_e2_v2.yaml`。

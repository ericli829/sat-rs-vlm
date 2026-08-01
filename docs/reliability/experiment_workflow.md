# 可靠性实验工作流

## 标准流程

```text
分层配置与环境变量
        ↓
固定、均衡的评测 JSONL
        ↓
干净 Adapter 推理
        ↓
只读复制并注入 bit flip
        ↓
故障 Adapter 重载和 hash 校验
        ↓
故障推理与输出验证
        ↓
文件恢复 / 输出投票 / 参数裁剪
        ↓
恢复后推理或输出层决策
        ↓
统一指标、报告与绘图数据
```

## 配置和路径

可靠性入口使用现有 `LayeredConfigRequest`，优先级仍为：

```text
CLI > 环境变量 > 实验配置 > local/cloud 配置 > reliability/base > base/default
```

核心模块不读取 YAML，也不知道 AutoDL 或 Windows 路径。数据、模型、输出和缓存目录由
`PathConfig` 与 `PROJECT_ROOT`、`DATA_ROOT`、`MODEL_ROOT`、`OUTPUT_ROOT` 提供。

## 固定评测样本

`build_reliability_eval_manifest.py` 读取当前 `DatasetManifest` 声明的 split，检查
train/validation/test 样本 ID 泄漏，再按 captioning、VQA、counting、detection 和
scene classification 均衡抽样。输出只保存相对图片路径，并写出同名 `.stats.json`。

## Adapter 注入

1. 检查 `adapter_model.safetensors` 和 `adapter_config.json`。
2. 计算源权重 SHA-256。
3. 把整个 checkpoint 复制到独立临时目录，保留 Processor、manifest 和 metadata。
4. 根据 LoRA A/B、模块、层和正则选择候选参数。
5. 固定 seed 抽取 bit 地址，生成新 state dict。
6. 写出 safetensors，重载并比较实际参数变化。
7. 再次校验源 hash，并要求故障 hash 与源不同。
8. 写出 `fault_records.jsonl` 和 `fault_record.json` 后发布故障目录。

源 Adapter 不会被原地写入。真实评测也使用独立 `--output-dir`，不会把预测写回源 checkpoint。

## 模式语义

- `baseline`：只运行干净 checkpoint 推理。
- `inject`：只生成和验证故障 Adapter，不加载 Qwen3-VL。
- `compare`：比较配置给出的 clean/fault 预测文件。
- `protect`：对给定预测执行输出 guard 和统一指标。
- `recover`：生成故障 Adapter 并执行文件级恢复，不运行模型。
- `full`：干净推理、注入、故障推理、已配置保护/恢复、恢复后评测和统一报告。

`real_inference` 的 baseline/full 需要 CUDA；inject/recover 是文件和 tensor 操作，可在 CPU
执行，但报告仍标记为真实资产流程，不会生成伪造模型预测。

## 指标解释

总体和按任务均输出 `changed_rate`、`invalid_rate`、`empty_rate`、clean/fault exact match、
exact match drop、recovery success rate 和 post-recovery exact match。缺少恢复结果时字段为
`null`，不会用 0 冒充结果。

报告始终包含 `schema_version`、`execution_mode`、`experiment_name` 和 `run_id`。真实运行另有
`real_inference_manifest.json`，记录模型、Adapter、数据、split、设备、dtype、seed、故障与
保护配置。


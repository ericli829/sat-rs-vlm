# Qwen3-VL 量化与统一评估

## 架构边界

量化实现位于 `src/sat_rs_vlm/quantization/`：

| 模块 | 职责 |
|---|---|
| `quantizer.py` | 后端注册、dynamic INT8、bnb INT8、未来 GPTQ/AWQ 接口 |
| `benchmark.py` | 固定样本生成、延迟/内存/大小统计、Evaluation v1.5 编排 |
| `sensitivity.py` | Linear 扫描、层/组件分组、局部量化、主指标退化和绘图 |
| `config.py` | YAML、环境变量和 Pydantic 校验 |
| `artifacts.py` | JSON 安全序列化、目录大小和量化 manifest |

`sat_rs_vlm.compression.quantization` 只保留旧导入兼容层，所有业务实现都在顶层包中。

## 后端

| 后端 | 设备 | 方法 | LoRA 约束 | 产物状态 |
|---|---|---|---|---|
| `baseline` | CPU/CUDA | 无量化 | 支持 | 原模型或 Adapter |
| `torch_dynamic_int8` | CPU | PyTorch Linear dynamic qint8 | 必须先 merge | 默认 benchmark-only |
| `bnb_int8` | CUDA | bitsandbytes 8-bit 加载 | 可配置，需真实环境验证 | 保存后仍需 reload smoke |

`method: dynamic_int8` 会映射到 `torch_dynamic_int8`。未知的 INT4、GPTQ、AWQ、QAT 或
mixed precision 方法会清晰失败，不会回退到 baseline。

## 配置

主配置为 `configs/quantization/quantization_eval.yaml`：

```yaml
model:
  base_model: "${LOCAL_MODEL_DIR}"
  merged_model: null

quantization:
  method: "dynamic_int8"
  device: "cpu"

evaluation:
  contract: "configs/eval/evaluation_contract_v1.5.yaml"
  dataset: "VRSBench"
  tasks: [detection, counting, captioning, vqa, scene_classification]
  sample_num: 20

output:
  output_dir: "reports/evaluation/quantization"
```

将 `model.merged_model` 设置为 `scripts/merge_lora.py` 的输出目录。路径使用 YAML 或环境变量，
脚本中没有本机绝对路径。

## 公平对比流程

1. 加载 merge 后 FP32 模型并在固定 messages 样本上生成 predictions。
2. 使用相同 Processor、样本顺序、prompt、reference、seed、warmup 和生成参数加载 INT8。
3. 两组 predictions 分别进入同一个 Evaluation v1.5 contract。
4. `evaluation.comparison` 依据样本 ID 做配对比较和 bootstrap 置信区间。
5. 延迟、Python/CUDA 峰值内存、参数量和序列化大小单独记录，不冒充任务准确率。

Keyword Hit 只存在于旧仓库指标兼容诊断中，量化比较与敏感度评分均不把它作为主指标。

## 命令

配置与资产 dry-run，不加载真实 Qwen3-VL：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/qwen3vl_torch_dynamic_int8_smoke.yaml \
  --dry-run
```

FP32 与 CPU dynamic INT8 完整比较：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/quantization_eval.yaml
```

CUDA bitsandbytes INT8：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/qwen3vl_bnb_int8.yaml
```

旧命令 `quantize_int8.py`、`quantize_int8_cpu.py`、`quantize_lora_int8_cpu.py` 和
`quantize_merged_model.py` 是薄兼容入口，均要求 `--config`。未 merge LoRA 与 CPU dynamic
INT8 的组合会明确失败。

## 输出

```text
reports/evaluation/quantization/
├── raw_predictions/
│   ├── baseline.jsonl
│   └── quantized.jsonl
├── baseline/
│   ├── metrics.json
│   ├── summary.json
│   ├── evaluated_predictions.jsonl
│   └── evaluation_manifest.json
├── quantized/
├── comparison/
│   ├── comparison.json
│   ├── paired_comparison.jsonl
│   └── comparison_manifest.json
└── benchmark_report.json
```

CPU dynamic INT8 state dict 仍标记 `benchmark_only=true` 和 `reload_supported=false`，不能称为
可部署模型。真实部署前必须实现模型结构重建与 reload smoke。

## 当前验证状态

- 默认 pytest：配置、toy Linear、统一评估和配对比较，不需要 GPU 或本地模型。
- 本地/AutoDL：需要显式运行真实 Qwen3-VL benchmark。
- bnb INT8：需要 CUDA 与可用 bitsandbytes；缺失时不会影响 LoRA 或 CPU 测试。
- INT4、GPTQ、AWQ、QAT、mixed precision：仅保留注册扩展边界，尚未实现。

# Qwen3-VL INT8 量化实验

## 后端边界

| 后端 | 设备 | 方法 | Adapter | 产物状态 |
| --- | --- | --- | --- | --- |
| `baseline` | CPU/CUDA | 无量化 | 支持 | 原模型/adapter |
| `torch_dynamic_int8` | CPU | PyTorch Linear 动态 qint8 | 未合并 LoRA 不支持 | 默认 benchmark-only |
| `bnb_int8` | CUDA | bitsandbytes `load_in_8bit` | 可配置，需真实环境验证 | 保存后仍须 reload smoke |

`bnb_int8` 不是动态量化。缺少 bitsandbytes 或 CUDA 时会明确失败，不回退到 baseline。
CPU dynamic INT8 若只保存 state dict，会同时写量化层与 manifest，并明确标记
`benchmark_only`、`reload_supported=false`，不能称为部署产物。

AutoDL 上先用 `bash scripts/environment/setup_autodl.sh --install-model --install-qlora`
安装可选依赖；基础安装不会携带 bitsandbytes。

## 公平比较

统一入口从 messages JSONL 固定样本 ID，直接解析单图/多图路径。baseline 与 quantized 使用
相同 Processor、adapter、prompt、reference、seed、warmup 和 generation 配置；生成只解码
输入长度之后的新 token。报告延迟为单样本端到端延迟，包含 mean、median、p50、p95、
min、max 和样本数。

任务表现复用普通评测指标：detection 分开报告 JSON、坐标范围、label 和 IoU；counting
报告 parsable、MAE、exact 和 ±1；caption 报告 BLEU/ROUGE-L；VQA/场景报告 exact match。
关键词命中只用于诊断。参数量、序列化字节、CPU/CUDA 内存分别记录，不用参数量冒充文件压缩率。

## 命令

只验证配置、依赖、路径、messages 和图片，不加载模型：

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/compression/qwen3vl_torch_dynamic_int8.yaml --dry-run
```

CPU 与 CUDA 实验：

```bash
python scripts/quantize_rs_vlm.py --config configs/compression/qwen3vl_torch_dynamic_int8.yaml
python scripts/quantize_rs_vlm.py --config configs/compression/qwen3vl_bnb_int8.yaml
```

服务器沿用 `MODEL_ROOT`、`DATA_ROOT`、`OUTPUT_ROOT`。可用 `--skip-baseline` 单独运行量化变体，
此时 speedup 和 accuracy retention 保持 `null`。旧 `quantize_int8*.py` 仅为弃用 wrapper。

## 可靠性关系

量化和 bit flip 共享 prediction schema、样本 manifest、task parser、task metrics、checksum 与
环境路径，但评测指标、输出合法性验证和压缩 benchmark 保持独立职责。量化 base + fault adapter、
量化恢复及 bnb 保存重载尚未在真实 Qwen3-VL 环境验证，不做成功声明。

# Qwen3-VL-2B 本地 LoRA 训练

## 第一版范围

第一版的真实模型基线是本地 `Qwen3-VL-2B-Instruct`，训练数据来自 VRSBench，并经过项目内转换器生成统一的遥感指令 JSONL 和 Qwen3-VL messages JSONL。已完成资产校验、数据转换、LoRA 两步 smoke 训练以及训练前向传播验证。

第一版并不宣称已经完成全量训练或取得 benchmark 分数。LoRA 基线保持原入口；
实验性微调方法已经移到 Git 忽略的本地插件包，使用方式见
`docs/external_plugins.md`。稳定 LoRA 不依赖该插件包。

## 当前资产

```text
本地模型: D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct
VRSBench: F:\VIT-data\VRSBench
项目根目录: D:\Desktop\tzb-2026\sat-rs-vlm
```

转换后的数据位于 `data/processed/`：

| 文件 | 当前记录数 |
| --- | ---: |
| `rs_train.jsonl` | 142390 |
| `rs_val.jsonl` | 62918 |
| `qwen3vl_train.jsonl` | 142390 |
| `qwen3vl_val.jsonl` | 62918 |

转换器写入的图片路径相对于 VRSBench 根目录，例如 `Images/Images_train/...`；训练和评估必须把 `image_root` 指向 `F:\VIT-data\VRSBench`。

## 训练策略

- 使用 LoRA 训练低秩 adapter，避免全参微调的显存和存储成本。
- 默认冻结视觉编码器，优先适配语言侧与跨模态连接层。
- Windows 环境默认使用 LoRA；仅在 bitsandbytes、CUDA 和版本兼容时启用 QLoRA。
- 训练脚本会将 batch 中的文本和视觉 tensor 移动到模型输入嵌入层所在设备，避免 CPU/CUDA 混用。

## 环境检查

在同一个已安装项目依赖的 Python 环境中执行。PowerShell 示例：

```powershell
$env:LOCAL_MODEL_DIR="D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="F:\VIT-data\VRSBench"
$env:TRAIN_JSONL="data\processed\qwen3vl_train.jsonl"
$env:VAL_JSONL="data\processed\qwen3vl_val.jsonl"

python scripts/check_env.py
python scripts/validate_training_assets.py --config configs/train/qwen3vl_local_smoke.yaml
```

`validate_training_assets.py` 只检查路径、模型文件、JSONL 和图片可访问性，不会加载完整权重。

## 数据转换

首次使用或 VRSBench 原始标注更新后，按顺序运行：

```powershell
python scripts/prepare_rs_instruction_data.py --config configs/data/remote_sensing_data.yaml
python scripts/convert_to_qwen3vl_format.py --config configs/data/remote_sensing_data.yaml
```

第一条命令会重建 `rs_train.jsonl`、`rs_val.jsonl` 和 `rs_test.jsonl`；第二条命令据此重建 Qwen3-VL 格式文件。不要将个人手工样本放在这些输出文件中。测试样本已隔离到 `data/processed/sample/`。

## 推荐执行顺序

### 1. 仅验证配置

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --dry-run
```

### 2. 验证模型前向传播

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --model-dir "$env:LOCAL_MODEL_DIR" `
  --train-file "$env:TRAIN_JSONL" `
  --val-file "$env:VAL_JSONL" `
  --image-root "$env:DATA_ROOT" `
  --max-seq-length 1024 `
  --forward-only
```

成功时会输出 batch tensor shape 和 loss。此步骤不保存 adapter。

### 3. 两步 smoke 训练

```powershell
python scripts/run_local_smoke_train.py `
  --model-dir "$env:LOCAL_MODEL_DIR" `
  --train-file "$env:TRAIN_JSONL" `
  --val-file "$env:VAL_JSONL" `
  --image-root "$env:DATA_ROOT"
```

默认使用少量样本和 `max_steps=2`，产物写入 `checkpoints/smoke/qwen3vl-local-smoke/`，并生成 `reports/training/smoke_train_report.json`。

### 4. 全量本地训练

先根据显存、目标模块和保存策略检查 `configs/train/qwen3vl_local.yaml`，再运行：

```powershell
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local.yaml
```

全量训练前建议先复制并命名一份实验配置，避免覆盖既有 smoke 或正式实验工件。

## 评估

当前评估配置使用本地 2B 基座、VRSBench 验证集和已存在的 smoke adapter：

```powershell
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml
```

输出位于 `reports/eval/`：

- `qwen3vl_predictions.jsonl`：逐样本预测与参考答案。
- `qwen3vl_eval_summary.json`：基础文本指标、计数指标与 `empty_prediction_rate`。

正式训练后，将 `configs/eval/qwen3vl_eval.yaml` 的 `model.adapter_path` 改为该训练产生的 adapter 目录。adapter 目录必须包含 `adapter_config.json` 和 adapter 权重；普通 checkpoint 目录不能直接作为 adapter 加载。

## 常见问题

- `ModuleNotFoundError: sat_rs_vlm`：使用 `python -m pip install -e ".[model]"`，并确保命令使用同一个 Python 解释器。
- `c10.dll` 或 `torch` 导入失败：当前虚拟环境中的 PyTorch 二进制与解释器或 CUDA 不匹配。重建环境并安装与本机 CUDA/CPU 匹配的 PyTorch 后，再安装项目依赖。
- `Expected all tensors to be on the same device`：更新到第一版当前代码；训练和评估均已在模型调用前迁移 batch。仍出现时请确认没有用旧脚本或旧安装副本执行。
- 生成为空：确认评估使用 generation collator、有效 adapter 和修复后的 `evaluate_rs_vlm.py`；摘要的 `empty_prediction_rate` 可快速定位此问题。
- CUDA out of memory：降低 `max_seq_length`，维持 batch size 为 1，增加梯度累积，冻结视觉编码器，或缩小图像 token 数。
- 图片找不到：确认 JSONL 内路径以 `Images/Images_train/` 等 VRSBench 根目录相对路径保存，并把 `image_root` 设置为 VRSBench 根目录。

## 后续扩展

该训练链路的接口已为 QLoRA、真实遥感 benchmark、知识蒸馏、剪枝、量化、模型合并和星载容错实验保留位置。接入新数据集时，优先实现 `BaseRemoteSensingDataset`/注册表适配，再复用统一的指令与 messages 转换流程。

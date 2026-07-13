# sat-rs-vlm 使用说明

本文说明当前框架能够完成的工作、推荐操作顺序，以及如何使用本地
`Qwen3-VL-2B-Instruct` 和遥感数据开展推理与 LoRA 训练。

以下命令默认在项目根目录执行：

```text
D:\Desktop\tzb-2026\sat-rs-vlm
```

## 1. 当前能力与边界

当前框架包含三条可独立使用的链路：

| 链路 | 用途 | 是否需要 GPU |
| --- | --- | --- |
| Mock 推理 | 验证任务路由、统一结果、CLI 和 HTTP 接口 | 否 |
| HuggingFace 推理 | 使用本地或 HuggingFace 兼容 VLM 做基础生成式推理 | 建议使用 |
| Qwen3-VL 多策略微调 | LoRA baseline 与七种独立策略、统一评估和实验对比 | 训练时需要 |

需要特别注意：

- `MockVLMEngine` 返回确定性的模拟结果，不读取图像内容，适合接口联调和测试。
- `HuggingFaceVLMEngine` 是通用模型适配器，目前把真实模型输出统一收敛到
  `answer`，还没有把文本自动解析成检测框或分割掩膜。
- Qwen3-VL 训练脚本支持项目内部 JSONL 和 Qwen3-VL `messages` JSONL。
- `prepare_rs_instruction_data.py` 已支持 VRSBench 逐图标注，能够展开 caption、
  referring/detection 和 QA，并把越界框裁剪到 `[0,1]`。
- `__MACOSX` 及其中的 `._*` 文件是 macOS 压缩包元数据，不是训练图片或标注，
  数据扫描时应忽略。

## 2. 总体调用流程

```text
CLI / HTTP 请求
      |
      v
InferenceService
      |
      +-- TaskRouter：根据关键词或第二张图判断任务
      |
      +-- BaseVLMEngine
             |-- MockVLMEngine
             `-- HuggingFaceVLMEngine
      |
      v
InferenceResult：answer / boxes / masks / count / confidence / raw_output
```

接口层只负责输入输出，任务路由与模型调用集中在应用层。后续替换真实模型时，
应实现 `BaseVLMEngine`，无需修改 CLI 和 HTTP 协议。

## 3. 创建和检查环境

要求 Python 3.10 或更高版本。推荐为本项目使用独立 `.venv`，避免与 Streamlit
等已有环境发生 `rich` 版本冲突。

只运行 Mock、API 和测试：

```powershell
python scripts/bootstrap_env.py
.\.venv\Scripts\Activate.ps1
python scripts/check_env.py
```

需要真实模型推理或训练：

```powershell
python scripts/bootstrap_env.py --with-model `
  --torch-index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\Activate.ps1
python scripts/check_env.py --require-model
```

上面的 `cu130` 适用于支持 CUDA 13.x 的 NVIDIA 驱动。其他机器应在 PyTorch 官方
安装页面选择与驱动和操作系统匹配的 wheel index。不要仅凭 `pip show torch` 判断环境
正常；`check_env.py --require-model` 会实际导入 PyTorch 并验证 CUDA runtime。

也可以手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,model]"
```

确认当前命令使用的是项目虚拟环境：

```powershell
python -c "import sys, sat_rs_vlm; print(sys.executable); print(sat_rs_vlm.__file__)"
```

## 4. 使用 Mock 快速体验

默认配置 `configs/default.yaml` 使用 `mock` 后端，因此不需要模型、CUDA，也不要求
图像包含真实内容。项目内已有可用示例图 `data/samples/demo_image.png`。

查看当前配置：

```powershell
python -m sat_rs_vlm.interfaces.cli config
```

图像描述：

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --image data/samples/demo_image.png `
  --prompt "请描述这张遥感图像中的主要地物。"
```

目标检测：

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --image data/samples/demo_image.png `
  --prompt "请检测并框出图像中的建筑物位置。"
```

双时相变化检测：

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --image data/samples/before.png `
  --second-image data/samples/after.png `
  --prompt "请说明前后两张遥感图像中的变化。"
```

只要提供 `--second-image`，任务路由器就会优先选择 `change_detection`。单图任务按
关键词判断，支持 detection、scene_classification、segmentation、counting、
captioning 和 VQA；不能识别时返回 unknown。

CLI 输出为统一 JSON。例如检测任务会使用以下字段：

```json
{
  "task_type": "detection",
  "answer": "检测到疑似建筑物、道路和开阔地目标。",
  "boxes": [
    {
      "label": "building",
      "x_min": 0.12,
      "y_min": 0.18,
      "x_max": 0.34,
      "y_max": 0.42,
      "confidence": 0.86
    }
  ],
  "masks": [],
  "count": null,
  "confidence": 0.82,
  "raw_output": {
    "engine": "mock",
    "profile": {}
  }
}
```

Mock 检测响应中的 `boxes` 使用归一化坐标并包含目标置信度。

## 5. 启动 HTTP API

启动服务：

```powershell
uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000
```

可访问：

- 健康检查：`http://127.0.0.1:8000/health`
- Swagger 文档：`http://127.0.0.1:8000/docs`
- OpenAPI 定义：`http://127.0.0.1:8000/openapi.json`

PowerShell 健康检查：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

单图推理：

```powershell
$body = @{
  image_path = "data/samples/demo_image.png"
  prompt = "请统计图像中的建筑物数量。"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/infer" `
  -ContentType "application/json" `
  -Body $body
```

双图请求只需增加：

```powershell
second_image_path = "data/samples/after.png"
```

HTTP 后端在应用启动时读取 `SAT_RS_VLM_CONFIG`。如需指定另一份配置，应先设置变量
再启动 Uvicorn：

```powershell
$env:SAT_RS_VLM_CONFIG="configs/default.yaml"
uvicorn sat_rs_vlm.interfaces.http.app:app --host 127.0.0.1 --port 8000
```

## 6. 使用本地 Qwen3-VL 做基础推理

本地模型目录为：

```text
D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct
```

使用通用 HuggingFace 后端：

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --backend huggingface `
  --model-id "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --image "data\samples\demo_image.png" `
  --prompt "请描述这张遥感图像。"
```

这条链路适合验证模型能否加载和生成。Qwen3-VL 的正式训练链路使用专用 processor、
chat template 和视觉输入整理逻辑，功能比通用推理适配器更完整。若通用后端与当前
Transformers 版本的 Qwen3-VL 自动类不兼容，应优先完成第 9 节的 `forward-only`
检查，再为 `HuggingFaceVLMEngine` 增加 Qwen3-VL 专用适配器。

## 7. 准备训练数据

### 7.1 转换 VRSBench

生成内部格式数据：

```powershell
python scripts/prepare_rs_instruction_data.py `
  --config configs/data/remote_sensing_data.yaml
```

生成 Qwen3-VL `messages` 格式：

```powershell
python scripts/convert_to_qwen3vl_format.py `
  --config configs/data/remote_sensing_data.yaml
```

输出文件包括：

```text
data/processed/rs_train.jsonl
data/processed/rs_val.jsonl
data/processed/rs_test.jsonl
data/processed/qwen3vl_train.jsonl
data/processed/qwen3vl_val.jsonl
data/processed/qwen3vl_test.jsonl
```

默认配置使用 `F:/VIT-data/VRSBench`，保留官方 train/val 划分。VRSBench 没有独立
test 标注，因此 `rs_test.jsonl` 和 `qwen3vl_test.jsonl` 为空文件。只验证少量图片时：

```powershell
python scripts/prepare_rs_instruction_data.py `
  --config configs/data/remote_sensing_data.yaml `
  --max-images-per-split 2
```

该参数会覆盖同名 processed 文件，仅用于转换冒烟。需要生成原来的占位样本时使用：

```powershell
python scripts/prepare_rs_instruction_data.py `
  --config configs/data/sample_data.yaml
```

sample 配置固定写入 `data/processed/sample/`，不会覆盖真实 VRSBench 的 processed 文件。

### 7.2 内部 JSONL 格式

每行一个样本：

```json
{
  "id": "sample_001",
  "task_type": "captioning",
  "images": ["Images/Images_train/000001.png"],
  "instruction": "请描述这张遥感图像。",
  "answer": "图像中包含建筑、道路和植被。",
  "metadata": {"dataset": "VRSBench", "split": "train"}
}
```

变化检测在 `images` 中按“变化前、变化后”顺序放两张图。`images` 可以使用绝对路径，
也可以使用相对于训练参数 `image_root` 的路径。

### 7.3 Qwen3-VL messages 格式

```json
{
  "id": "sample_001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "Images/Images_train/000001.png"},
        {"type": "text", "text": "请描述这张遥感图像。"}
      ]
    },
    {"role": "assistant", "content": "图像中包含建筑、道路和植被。"}
  ],
  "task_type": "captioning",
  "metadata": {"dataset": "VRSBench", "split": "train"}
}
```

训练数据集类会自动识别这两种格式。更完整的字段定义见
`docs/data_format.md`。

### 7.4 VRSBench 展开和坐标规则

转换器读取以下有效目录，自动忽略 `__MACOSX` 和 `._*`：

```text
F:\VIT-data\VRSBench\
  Images\Images_train
  Images\Images_val
  Annotations\Annotations_train
  Annotations\Annotations_val
```

每张图展开为一条 caption、每个 object 一条 detection、每个 QA 一条任务样本。
`object quantity` 映射为 counting，`scene type` 映射为 scene_classification，其余为
VQA。检测答案格式为：

```json
{"label":"trainstation","bbox":[0.6,0.83,1.0,0.94]}
```

`obj_coord` 会裁剪并重排到 `[0,1]`，原始框写入 `metadata.bbox_raw`，裁剪结果写入
`metadata.bbox_clipped`。JSONL 图片路径相对 VRSBench 根目录，因此训练时设置：

```text
image_root = F:\VIT-data\VRSBench
```


## 8. 配置本地训练路径

有两种方式，推荐第一次使用 CLI 显式传路径，确认成功后再改用环境变量。

方式一：PowerShell 环境变量：

```powershell
$env:LOCAL_MODEL_DIR="D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="F:\VIT-data\VRSBench"
$env:TRAIN_JSONL="D:\Desktop\tzb-2026\sat-rs-vlm\data\processed\qwen3vl_train.jsonl"
$env:VAL_JSONL="D:\Desktop\tzb-2026\sat-rs-vlm\data\processed\qwen3vl_val.jsonl"
```

设置后可以直接使用 `configs/train/qwen3vl_local_smoke.yaml`。PowerShell 环境变量只对
当前终端会话有效，新开终端后需要重新设置。

方式二：每次通过命令行覆盖路径。此方式不会依赖环境变量，后续示例主要采用它。

## 9. 按顺序验证本地 Qwen3-VL 训练链路

不要一开始就运行完整训练。建议依次通过资产检查、dry run、单次前向和两步训练。

### 9.1 资产检查

```powershell
python scripts/validate_training_assets.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --model-dir "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --train-file "data\processed\qwen3vl_train.jsonl" `
  --val-file "data\processed\qwen3vl_val.jsonl" `
  --image-root "D:\Desktop\tzb-2026\sat-rs-vlm"
```

检查内容包括模型配置、processor 配置、训练依赖、JSONL 结构、前五条样本及图片路径。
报告写入：

```text
reports/training_asset_check.json
```

`success: true` 后再继续。

### 9.2 Dry run

Dry run 不加载模型权重，主要检查配置解析、数据集和 collator 初始化：

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --model-dir "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --train-file "data\processed\qwen3vl_train.jsonl" `
  --val-file "data\processed\qwen3vl_val.jsonl" `
  --image-root "F:\VIT-data\VRSBench" `
  --dry-run
```

### 9.3 Forward only

Forward only 会加载 processor 和模型，整理一条样本并执行一次前向传播，但不更新参数：

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --model-dir "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --train-file "data\processed\qwen3vl_train.jsonl" `
  --val-file "data\processed\qwen3vl_val.jsonl" `
  --image-root "D:\Desktop\tzb-2026\sat-rs-vlm" `
  --max-seq-length 1024 `
  --forward-only
```

这一步能较早暴露模型类、processor、图像解码、chat template、dtype 和显存问题。

### 9.4 自动执行完整冒烟流程

以下封装命令依次执行资产检查、dry run、forward only 和 `max_steps=2` 训练：

```powershell
python scripts/run_local_smoke_train.py `
  --model-dir "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --train-file "data\processed\qwen3vl_train.jsonl" `
  --val-file "data\processed\qwen3vl_val.jsonl" `
  --image-root "D:\Desktop\tzb-2026\sat-rs-vlm" `
  --max-seq-length 1024
```

执行摘要写入：

```text
reports/local_smoke_train_summary.json
```

训练产物默认写入：

```text
checkpoints/smoke/qwen3vl-local-smoke/
```

## 10. 运行正式 LoRA 训练

确认冒烟流程通过后，使用正式配置：

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local.yaml `
  --model-dir "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --train-file "D:\path\to\vrsbench_train.jsonl" `
  --val-file "D:\path\to\vrsbench_val.jsonl" `
  --image-root "F:\VIT-data\VRSBench" `
  --output-dir "checkpoints\qwen3vl-2b-vrsbench-lora" `
  --method lora
```

对于 8 GB 显存的 Windows 笔记本 GPU，建议从以下参数开始：

- `per_device_train_batch_size: 1`
- `max_seq_length: 1024`
- `gradient_checkpointing: true`
- `freeze_vision_encoder: true`
- `method: lora`
- 用 `gradient_accumulation_steps` 调整有效 batch size

Windows 下先使用已验证的 LoRA。实验性 QLoRA 已迁移到外部插件包，不通过旧 LoRA
入口启动。确认 CUDA 和 bitsandbytes 兼容后，先显式检查插件：

```powershell
python scripts/run_external_strategy.py `
  --plugin-root .local_plugins/sat-rs-vlm-local-plugins `
  --strategy qlora `
  --check-only
```

DoRA、AdaLoRA、IA3、Partial Unfreeze、Full SFT 与 Prompt Tuning 的入口、依赖和
checkpoint 隔离规则见 `docs/external_plugins.md`。

训练目录会保存 LoRA adapter、processor、训练状态和 `smoke_train_report.json`。断点续训
可在 YAML 中设置：

```yaml
training:
  resume_from_checkpoint: "checkpoints/qwen3vl-2b-vrsbench-lora/checkpoint-100"
```

## 11. 评估和合并 LoRA

先修改 `configs/eval/qwen3vl_eval.yaml`，将以下字段指向本地 2B 模型、训练得到的
adapter、验证 JSONL 和图片根目录：

```yaml
model:
  base_model: "D:/Desktop/tzb-2026/Qwen3-VL-2B-Instruct"
  adapter_path: "checkpoints/smoke/qwen3vl-local-smoke"
  processor_id: "D:/Desktop/tzb-2026/Qwen3-VL-2B-Instruct"
  local_files_only: true
  torch_dtype: "bfloat16"
  device_map: "auto"

data:
  eval_file: "data/processed/qwen3vl_val.jsonl"
  image_root: "F:/VIT-data/VRSBench"
  max_seq_length: 1024
```

`adapter_path` 目录必须包含 `adapter_config.json` 和 `adapter_model.safetensors`（或
`adapter_model.bin`）。本地评估会在加载基座模型前检查这些文件；路径错误时直接失败，
不会把本地路径误当成 Hugging Face 仓库名。上面的 smoke adapter 只训练了两步，适合
验证评估链路，不代表有效的 VRSBench 模型精度。

运行评估：

```powershell
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml
```

默认生成：

```text
reports/eval/qwen3vl_eval_summary.json
reports/eval/qwen3vl_predictions.jsonl
```

评估 collator 只编码 user 消息并添加 generation prompt，不会把标准答案送入模型。
summary 会报告整体及分任务 `empty_prediction_rate`；正常生成时该值应接近 0。

将 LoRA adapter 合并到基座模型：

```powershell
python scripts/merge_lora.py `
  --base-model "D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct" `
  --adapter "checkpoints\qwen3vl-2b-vrsbench-lora" `
  --output "checkpoints\qwen3vl-2b-vrsbench-merged"
```

合并后的目录体积接近完整基座模型，应预留足够磁盘空间。

## 12. 测试与代码质量

运行全部测试：

```powershell
pytest -q
```

静态检查：

```powershell
ruff check src tests scripts
mypy src
```

格式化：

```powershell
ruff format src tests scripts
ruff check --fix src tests scripts
```

这些测试默认不下载模型、不加载真实 Qwen3-VL，也不执行 GPU 训练，因此适合日常开发和
CI。Windows 未安装 GNU Make 时，直接运行上述命令即可；安装了 Make 时可使用
`make test`、`make lint` 和 `make run-api`。

## 13. 配置文件职责

| 配置 | 作用 |
| --- | --- |
| `configs/default.yaml` | CLI、HTTP 和推理后端配置 |
| `configs/data/remote_sensing_data.yaml` | 数据源、处理结果路径和切分比例 |
| `configs/train/qwen3vl_local_smoke.yaml` | 本地模型最小训练验证 |
| `configs/train/qwen3vl_local.yaml` | 本地模型正式 LoRA 训练 |
| `configs/train/qwen3vl_lora*.yaml` | HuggingFace 模型 ID 训练示例 |
| `configs/eval/qwen3vl_eval.yaml` | adapter 评估和预测输出配置 |

推理配置和训练配置是两套不同的 Pydantic 模型，不要把训练 YAML 直接传给 CLI
`infer` 或 HTTP 服务。

## 14. 常见问题

### `ModuleNotFoundError: No module named 'sat_rs_vlm'`

在项目根目录使用同一个 Python 解释器安装 editable package：

```powershell
python -m pip install -e ".[dev]"
python -c "import sat_rs_vlm; print(sat_rs_vlm.__file__)"
```

项目脚本已带 `src` 路径引导，但依赖包仍必须安装到当前解释器。

### `streamlit requires rich<14` 依赖冲突

这是共享 Python 环境中的依赖冲突警告。最稳妥的处理方式是使用项目独立 `.venv`，
不要在安装了 Streamlit 1.32.0 的环境里继续叠加训练依赖。

### `c10.dll`、`WinError 1114` 或 PyTorch CUDA 不可用

先检查实际安装的 wheel：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python scripts/check_env.py --require-model
```

如果版本以 `+cpu` 结尾或 `torch.version.cuda` 为 `None`，说明安装了 CPU wheel。对于
支持 CUDA 13.x 的 Windows NVIDIA 环境，可从官方 CUDA index 重装：

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install --timeout 1000 --retries 10 torch torchvision `
  --index-url https://download.pytorch.org/whl/cu130
python scripts/check_env.py --require-model
```

CUDA index 必须根据目标机器选择；项目不在 `pyproject.toml` 中硬编码 CUDA wheel，
因为 CPU、Windows CUDA 和 Linux CUDA 使用不同的软件源。

### 配置中 `${LOCAL_MODEL_DIR}` 无法解析

设置第 8 节的四个环境变量，或在命令中显式传入 `--model-dir`、`--train-file`、
`--val-file` 和 `--image-root`。

### 图片路径不存在

相对路径的最终位置为：

```text
image_root / JSONL 中的 image 路径
```

例如 `image_root=F:\VIT-data\VRSBench`，图片字段为
`Images/Images_train/000001.png`，最终路径就是
`F:\VIT-data\VRSBench\Images\Images_train\000001.png`。

### CUDA out of memory

依次降低 `max_seq_length`、减少图片分辨率、保持 batch size 为 1、冻结视觉编码器，
并使用梯度累积。确认 bitsandbytes 可用后可尝试 QLoRA。

### `bfloat16` 不受支持

将训练配置中的 `bf16` 设为 `false`、`fp16` 设为 `true`。CPU 检查可使用 float32，
但不适合正式训练。

### 资产检查通过但训练失败

资产检查只验证目录、关键文件、依赖和少量样本路径。继续按 dry run、forward only、
两步训练定位问题；最先失败的阶段通常能指出是配置、数据整理、模型加载还是反向传播问题。

## 15. 推荐的实际使用顺序

1. 使用 Mock CLI 和 HTTP 验证上层接口。
2. 运行已接入的 VRSBench 转换器，生成真实训练/验证 JSONL。
3. 对真实 JSONL 运行资产检查、dry run、forward only 和两步 smoke 训练。
4. 启动独立配置的正式 LoRA 训练，保存 adapter 和训练报告。
5. 在固定验证集上评估，再决定是否合并、量化、蒸馏或剪枝。
6. 部署前加入 checksum、bit flip 注入和故障恢复策略验证。

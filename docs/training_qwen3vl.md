# Qwen3-VL 遥感指令微调

## 训练目标

第三阶段目标是把项目从 Mock 推理骨架推进到真实遥感多模态指令微调 pipeline。训练数据统一为遥感指令 JSONL，再转换为 Qwen3-VL chat message 格式，支持图像描述、VQA、检测、计数、场景分类和双图变化检测。

## 为什么使用 Qwen3-VL

`Qwen/Qwen3-VL-8B-Instruct` 暂时作为通用多模态基座模型，用于验证遥感指令数据、LoRA/QLoRA 训练、评测和部署工件流转。后续可以替换为更小的 Qwen3-VL、蒸馏模型或专用遥感 VLM。

## 为什么优先 LoRA / QLoRA

默认不做全参微调。LoRA 只训练低秩 adapter，QLoRA 进一步使用 4bit 量化加载基座模型，显存占用更低，更符合卫星平台受限算力部署前的工程验证需求。

## 为什么默认冻结视觉编码器

视觉编码器参数量大，直接微调容易带来显存压力和灾难性遗忘。默认 `freeze_vision_encoder: true`，优先训练语言侧和跨模态适配层；需要更强视觉域适配时再逐步解冻。

## 数据格式

单图任务内部格式：

```json
{"id":"sample_000001","task_type":"captioning","images":["data/samples/demo_image.png"],"instruction":"请描述这张遥感图像中的主要地物。","answer":"图像中包含建筑物、道路和植被区域。","metadata":{"dataset":"sample","split":"train"}}
```

双图变化检测内部格式：

```json
{"id":"change_000001","task_type":"change_detection","images":["data/samples/before.png","data/samples/after.png"],"instruction":"第一张为变化前，第二张为变化后。请描述两张遥感图像之间的变化。","answer":"变化后图像中新增了建筑物，道路区域基本保持不变。","metadata":{"dataset":"sample_change","split":"train"}}
```

## 命令

```bash
python scripts/bootstrap_env.py --with-model

python scripts/prepare_rs_instruction_data.py \
  --config configs/data/remote_sensing_data.yaml

python scripts/convert_to_qwen3vl_format.py \
  --config configs/data/remote_sensing_data.yaml

python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora_smoke.yaml

python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml

torchrun --nproc_per_node=4 scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml

python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml

python scripts/merge_lora.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --adapter checkpoints/qwen3vl-rs-lora/best \
  --output checkpoints/qwen3vl-rs-merged
```

## 本地模型目录

本地模型目录建议包含：

```text
config.json
tokenizer_config.json
preprocessor_config.json 或 processor_config.json
tokenizer.json / vocab.json / merges.txt
model.safetensors 或分片 safetensors
generation_config.json
```

本地配置使用 `configs/train/qwen3vl_local.yaml` 和
`configs/train/qwen3vl_local_smoke.yaml`，默认 `local_files_only: true`，不会联网下载模型。

## 本地数据 JSONL 格式

支持 Qwen3-VL messages 格式：

```json
{"id":"sample_001","messages":[{"role":"user","content":[{"type":"image","image":"images/001.jpg"},{"type":"text","text":"请描述这张遥感图像。"}]},{"role":"assistant","content":"图像中包含建筑、道路和植被。"}],"task_type":"captioning"}
```

也支持项目内部格式：

```json
{"id":"sample_001","task_type":"captioning","images":["images/001.jpg"],"instruction":"请描述这张遥感图像。","answer":"图像中包含建筑、道路和植被。"}
```

相对图片路径会相对 `image_root` 解析，绝对路径会直接使用。

## 环境变量

Linux/macOS:

```bash
export LOCAL_MODEL_DIR=/path/to/qwen3vl
export DATA_ROOT=/path/to/data
export TRAIN_JSONL=/path/to/train.jsonl
export VAL_JSONL=/path/to/val.jsonl
```

Windows PowerShell:

```powershell
$env:LOCAL_MODEL_DIR="C:\path\to\qwen3vl"
$env:DATA_ROOT="C:\path\to\data"
$env:TRAIN_JSONL="C:\path\to\train.jsonl"
$env:VAL_JSONL="C:\path\to\val.jsonl"
```

## 本地资产检查

```bash
python scripts/validate_training_assets.py \
  --config configs/train/qwen3vl_local_smoke.yaml
```

也可以使用 CLI 覆盖：

```bash
python scripts/validate_training_assets.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data
```

## 最小训练测试

```bash
python scripts/run_local_smoke_train.py \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data
```

只做 dry run：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --dry-run
```

只做 forward：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --forward-only
```

正式本地训练：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local.yaml
```

## 迁移到其他设备

1. 复制代码仓库。
2. 放置本地 Qwen3-VL 模型目录。
3. 放置训练/验证 JSONL 和图片根目录。
4. 创建环境：

```bash
python scripts/bootstrap_env.py --with-model
python scripts/check_env.py
```

Windows 激活：`.venv\Scripts\activate`。
Linux/macOS 激活：`source .venv/bin/activate`。

5. 运行资产检查和 smoke 训练。

CUDA 不可用时，可以先做资产检查或极小 forward 测试，不建议真实训练。

## 常见错误

- 模型路径不存在：检查 `LOCAL_MODEL_DIR` 或 `--model-dir`。
- tokenizer/processor 加载失败：确认目录包含 `tokenizer_config.json` 和 `preprocessor_config.json`。
- 图片路径错误：确认 JSONL 中相对路径是否相对 `image_root`。
- CUDA 不可用：可做资产检查，真实训练会很慢。
- CUDA out of memory：降低 `max_seq_length`，使用 QLoRA，减小 batch size，增大 `gradient_accumulation_steps`，冻结 vision encoder，换更小模型，设置 `max_steps` 做最小测试。
- bfloat16 不支持：脚本会降级到 fp16 或 float32 并打印 warning。
- bitsandbytes 不可用：Windows 下优先使用 LoRA 而不是 QLoRA。
- qwen_vl_utils 缺失：运行 `pip install -e ".[model]"`。

## 断点续训

在训练配置中设置：

```yaml
training:
  resume_from_checkpoint: "checkpoints/qwen3vl-rs-lora/checkpoint-1000"
```

然后重新运行同一个训练命令。

## 显存不足处理

- 降低 `max_seq_length`：4096 → 2048。
- 保持或降低 `per_device_train_batch_size = 1`。
- 增加 `gradient_accumulation_steps`。
- 开启 `gradient_checkpointing`。
- 使用 QLoRA 4bit。
- 冻结 vision encoder。
- 换用更小的 Qwen3-VL 模型。
- 减小图片分辨率或限制最大 image tokens。
- 后续扩展 DeepSpeed ZeRO-2 / ZeRO-3。

## 后续方向

后续会接入真实 VRSBench、MME Real RS、XLRS-bench、LEVIR-CC 数据转换器，加入 LoRA/QLoRA 系统评测、知识蒸馏、小模型替换和星载部署量化流程。

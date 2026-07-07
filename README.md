# sat-rs-vlm

`sat-rs-vlm` is a Python engineering framework for multimodal remote-sensing VLM inference under constrained satellite-platform compute. The project now includes the phase-one runnable skeleton plus phase-two environment automation, model factory wiring, optional HuggingFace model integration, and lightweight inference profiling.

## Features

- Natural-language driven task routing for detection, scene classification, segmentation, change detection, counting, captioning, and VQA.
- Clean separation of interface, application, domain, model, data, and infrastructure layers.
- YAML configuration loaded into Pydantic config models.
- Typer CLI and FastAPI HTTP API.
- Reliability extension points for bit flip simulation and checksum validation.
- Reserved interfaces for LoRA fine-tuning, distillation, pruning, quantization, and onboard fault recovery.
- Optional real-model dependencies kept behind the `[model]` extra so default development and CI remain CPU-friendly.

## Structure

```text
configs/                 YAML configuration
examples/                Prompt examples and demo inputs
src/sat_rs_vlm/
  interfaces/            CLI and HTTP adapters
  application/           Use-case services
  domain/                Task, entity, result, and routing models
  models/                VLM engine interfaces and implementations
  data/                  Dataset abstractions and registry
  infrastructure/        Config, logging, device, and seed utilities
  utils/                 Small shared helpers
tests/                   Unit and integration tests
```

## Install

```bash
python scripts/bootstrap_env.py
pip install -e ".[dev]"
```

Install optional model dependencies only when you need the HuggingFace backend:

```bash
python scripts/bootstrap_env.py --with-model
pip install -e ".[model]"
pip install -e ".[dev,model]"
```

The real-model stack includes packages such as `torch`, `transformers`, `peft`, and `accelerate`, so it is intentionally not part of the default dependency set. This keeps mock inference, API tests, and local CI usable on machines without GPU or large model runtimes.

Check the active environment:

```bash
python scripts/check_env.py
make check-env
```

## Run

```bash
python -m sat_rs_vlm.interfaces.cli config
python -m sat_rs_vlm.interfaces.cli infer --image examples/demo_image.jpg --prompt "请描述这张遥感图像中的主要地物。"
python -m sat_rs_vlm.interfaces.cli infer --backend mock --image examples/demo_image.jpg --prompt "请检测图像中的飞机。"
uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000
```

Backend selection is controlled by `configs/default.yaml`:

```yaml
model:
  backend: mock
  model_id: ""
```

To use a real HuggingFace-compatible VLM, install `[model]`, set `model.backend: huggingface`, provide `model.model_id`, or override both at the CLI:

```bash
python -m sat_rs_vlm.interfaces.cli infer \
  --backend huggingface \
  --model-id your-org/your-vlm \
  --image examples/demo_image.jpg \
  --prompt "请描述这张遥感图像中的主要地物。"
```

HTTP examples:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"image_path":"examples/demo_image.jpg","prompt":"请检测机场跑道位置"}'
```

## Test And Quality

```bash
pytest -q
make test
make lint
make format
```

## Phase Two

Phase two adds reproducible `.venv` bootstrapping, environment diagnostics, a model factory, a lazy-loaded `HuggingFaceVLMEngine`, and `InferenceProfiler`. The CLI and HTTP layers now build `InferenceService` from config instead of directly instantiating concrete model classes, which keeps the business layer dependent only on the `BaseVLMEngine` abstraction.

## Phase Three: Qwen3-VL Remote-Sensing Instruction Tuning

Phase three adds a configuration-driven Qwen3-VL LoRA/QLoRA training pipeline. The temporary base model is `Qwen/Qwen3-VL-8B-Instruct`; default training uses QLoRA/LoRA and freezes the vision encoder to reduce memory pressure and avoid visual backbone forgetting.

Prepare sample or real-data-derived internal JSONL:

```bash
python scripts/prepare_rs_instruction_data.py \
  --config configs/data/remote_sensing_data.yaml
```

Convert internal `rs_*.jsonl` files to Qwen3-VL chat messages:

```bash
python scripts/convert_to_qwen3vl_format.py \
  --config configs/data/remote_sensing_data.yaml
```

Install model dependencies and run a smoke training job:

```bash
python scripts/bootstrap_env.py --with-model
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora_smoke.yaml
```

Run the full baseline configuration:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml
```

Multi-GPU launch:

```bash
torchrun --nproc_per_node=4 scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml
```

Evaluate and merge LoRA:

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml

python scripts/merge_lora.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --adapter checkpoints/qwen3vl-rs-lora/best \
  --output checkpoints/qwen3vl-rs-merged
```

Default `pytest -q` does not download Qwen3-VL, does not require GPU, and does not run real training. Real model tests should be gated behind environment variables such as `RUN_MODEL_TRAINING_TESTS=1`.

## Local Qwen3-VL Training

Local training uses `configs/train/qwen3vl_local.yaml` and `configs/train/qwen3vl_local_smoke.yaml`. These configs default to `local_files_only: true`, so the scripts prefer a local model directory and do not access Hugging Face unless you explicitly pass `--no-local-files-only`.

Local model directory should contain at least:

```text
config.json
tokenizer_config.json
preprocessor_config.json  # or processor_config.json
tokenizer.json / vocab / merges
model.safetensors or shard files
```

Set paths with environment variables:

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

You can also avoid environment variables and pass CLI overrides:

```bash
python scripts/validate_training_assets.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data
```

## Local Dataset Format

The local training dataset accepts both Qwen3-VL `messages` JSONL and the project internal `instruction/images/answer` JSONL. Relative image paths are resolved against `image_root`; absolute image paths are used directly.

## Smoke Training Test

Run a no-model-load dry run:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data \
  --dry-run
```

Run one forward pass:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local_smoke.yaml \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data \
  --forward-only
```

Run the full local smoke sequence:

```bash
python scripts/run_local_smoke_train.py \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data
```

Makefile example:

```bash
make train-local-smoke \
  MODEL_DIR=/path/to/qwen3vl \
  TRAIN_FILE=/path/to/train.jsonl \
  VAL_FILE=/path/to/val.jsonl \
  IMAGE_ROOT=/path/to/data
```

## Troubleshooting

- Missing model dependency: run `pip install -e ".[model]"`.
- CUDA unavailable: asset check and tiny CPU forward may work, but real training will be slow.
- CUDA out of memory: reduce `max_seq_length`, use QLoRA, keep batch size at 1, increase gradient accumulation, freeze the vision encoder, use a smaller model, or set `max_steps` for minimal tests.
- `bfloat16` unsupported: the training script falls back to fp16 on CUDA or float32 on CPU and prints a warning.
- Windows QLoRA / bitsandbytes issue: use `training.method: lora` and `qlora.load_in_4bit: false`.
- Image path error: check whether paths are absolute or relative to `image_root`.

## Extension Direction

Real model integration should implement `BaseVLMEngine` in `src/sat_rs_vlm/models/base.py`, keeping model loading and device placement inside the model layer while leaving CLI and HTTP adapters unchanged. Phase three will focus on remote-sensing dataset integration, LoRA/QLoRA fine-tuning, evaluation pipelines, quantization-aware deployment, pruning, distillation, and satellite reliability features such as bit flip injection, checksums, watchdog recovery, and degraded-mode fallback.

.PHONY: install bootstrap bootstrap-model check-env test lint format run-api infer-demo prepare-data convert-qwen3vl train-smoke train-qwen3vl eval-qwen3vl merge-lora validate-training-assets train-local-smoke train-local train-local-dry-run train-local-forward plugin-list plugin-validate plugin-check plugin-dry-run

PLUGIN_ROOT ?= .local_plugins/sat-rs-vlm-local-plugins
PLUGIN_STRATEGY ?= qlora

install:
	pip install -e ".[dev]"

bootstrap:
	python scripts/bootstrap_env.py

bootstrap-model:
	python scripts/bootstrap_env.py --with-model

check-env:
	python scripts/check_env.py

test:
	pytest -q

lint:
	ruff check src tests scripts
	mypy src

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

run-api:
	uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000

infer-demo:
	python -m sat_rs_vlm.interfaces.cli infer --image data/samples/demo_image.png --prompt "请描述这张遥感图像中的主要地物。"

prepare-data:
	python scripts/prepare_rs_instruction_data.py --config configs/data/remote_sensing_data.yaml

convert-qwen3vl:
	python scripts/convert_to_qwen3vl_format.py --config configs/data/remote_sensing_data.yaml

train-smoke:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml

train-qwen3vl:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_lora.yaml

eval-qwen3vl:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml

merge-lora:
	python scripts/merge_lora.py --base-model $(MODEL_DIR) --adapter $(ADAPTER_DIR) --output $(MERGED_DIR)

validate-training-assets:
	python scripts/validate_training_assets.py --config configs/train/qwen3vl_local_smoke.yaml

train-local-smoke:
	python scripts/run_local_smoke_train.py --model-dir $(MODEL_DIR) --train-file $(TRAIN_FILE) --val-file $(VAL_FILE) --image-root $(IMAGE_ROOT)

train-local:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local.yaml

train-local-dry-run:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --dry-run

train-local-forward:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --forward-only

plugin-list:
	python scripts/list_external_plugins.py --plugin-root $(PLUGIN_ROOT) --validate

plugin-validate:
	python scripts/validate_external_plugin.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY)

plugin-check:
	python scripts/run_external_strategy.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY) --check-only

plugin-dry-run:
	python scripts/run_external_strategy.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY) --dry-run

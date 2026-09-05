.PHONY: install bootstrap bootstrap-model bootstrap-taskgraph check-env check-env-retriever autodl-taskgraph-dry-run test lint format run-api infer-demo infer-real-local prepare-data convert-qwen3vl prepare-e1 train-smoke train-qwen3vl train-autodl-4090 train-e1 train-e1b train-e1d-data train-e1d-sampler train-e1d-combined eval-e0 eval-e1 eval-e2 eval-e3 eval-offline compare-eval plot-eval train-local-real-smoke eval-qwen3vl merge-lora validate-training-assets train-local-smoke train-local train-local-dry-run train-local-forward train-unified smoke-unified validate-fixture export-environment plugin-list plugin-validate plugin-check plugin-dry-run quant-dry-cpu quant-cpu quant-bnb quant-eval quant-sensitivity-dry quant-sensitivity reliability-smoke reliability-real reliability-plot

PLUGIN_ROOT ?= .local_plugins/sat-rs-vlm-local-plugins
PLUGIN_STRATEGY ?= qlora

install:
	pip install -e ".[dev]"

bootstrap:
	python scripts/environment/bootstrap_local.py --with-dev

bootstrap-model:
	python scripts/environment/bootstrap_local.py --with-dev --with-model

bootstrap-taskgraph:
	python scripts/environment/bootstrap_local.py --with-dev --with-model --with-retriever

check-env:
	python scripts/environment/check_environment.py

check-env-retriever:
	python scripts/environment/check_environment.py --require-model --require-retriever

autodl-taskgraph-dry-run:
	bash scripts/environment/setup_autodl.sh --install-model --install-retriever --install-lae --dry-run

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

infer-real-local:
	python -m sat_rs_vlm.interfaces.cli infer --config configs/local/qwen3vl_real_infer.yaml --image $(IMAGE) --prompt "$(PROMPT)"

prepare-data:
	python scripts/prepare_rs_instruction_data.py --config configs/data/remote_sensing_data.yaml

convert-qwen3vl:
	python scripts/convert_to_qwen3vl_format.py --config configs/data/remote_sensing_data.yaml

prepare-e1:
	python scripts/prepare_e1_datasets.py

train-e1:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1_balanced.yaml

train-e1b:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1b_r32.yaml

train-e1d-data:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1d_data.yaml

train-e1d-sampler:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1d_sampler.yaml

train-e1d-combined:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1d_combined.yaml

eval-e0:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e0_zeroshot.yaml

eval-e1:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e1.yaml

eval-e2:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e2.yaml

eval-e3:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e3.yaml

train-smoke:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml

train-qwen3vl:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_lora.yaml

train-autodl-4090:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_autodl_4090.yaml

eval-qwen3vl:
	python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml

eval-offline:
	python scripts/evaluation/evaluate_predictions.py --config configs/eval/evaluation_v1_5.yaml

compare-eval:
	python scripts/evaluation/compare_evaluations.py --config configs/eval/evaluation_v1_5.yaml

plot-eval:
	python scripts/evaluation/plot_evaluation_results.py --config configs/eval/evaluation_v1_5.yaml

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

train-local-real-smoke:
	python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --max-steps 2 --output-dir checkpoints/qwen3vl-2b-vrsbench-lora/local-real-smoke

train-unified:
	python scripts/training/run_train.py --config configs/experiments/lora_baseline.yaml --environment local

smoke-unified:
	python scripts/training/run_smoke_train.py --config configs/local/train_lora_smoke.yaml

validate-fixture:
	python scripts/data/validate_dataset.py --dataset-root tests/fixtures/miniature_dataset

export-environment:
	python scripts/environment/export_environment.py --output reports/environment/local

plugin-list:
	python scripts/list_external_plugins.py --plugin-root $(PLUGIN_ROOT) --validate

plugin-validate:
	python scripts/validate_external_plugin.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY)

plugin-check:
	python scripts/run_external_strategy.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY) --check-only

plugin-dry-run:
	python scripts/run_external_strategy.py --plugin-root $(PLUGIN_ROOT) --strategy $(PLUGIN_STRATEGY) --dry-run

quant-dry-cpu:
	python scripts/quantize_rs_vlm.py --config configs/quantization/qwen3vl_torch_dynamic_int8.yaml --dry-run

quant-cpu:
	python scripts/quantize_rs_vlm.py --config configs/quantization/qwen3vl_torch_dynamic_int8.yaml

quant-bnb:
	python scripts/quantize_rs_vlm.py --config configs/quantization/qwen3vl_bnb_int8.yaml

quant-eval:
	python scripts/quantize_rs_vlm.py --config configs/quantization/quantization_eval.yaml

quant-sensitivity-dry:
	python scripts/quantization_sensitivity_test.py --config configs/quantization/quantization_sensitivity_smoke.yaml --dry-run

quant-sensitivity:
	python scripts/quantization_sensitivity_test.py --config configs/quantization/sensitivity_layer_autodl.yaml --plot

reliability-smoke:
	python scripts/reliability/run_smoke.py --case all

reliability-real:
	python scripts/reliability/run_experiment.py --config configs/reliability/experiments/lora_bitflip.yaml --mode full --environment autodl

reliability-plot:
	python scripts/reliability/plot_results.py --input $(METRICS_DIR) --output $(FIGURES_DIR)

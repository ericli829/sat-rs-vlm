#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
model_dir="${QWEN3VL_4B_MODEL_DIR:?QWEN3VL_4B_MODEL_DIR is required}"
output_root="${OUTPUT_ROOT:-$repo_root/taskgraph_lab/outputs/cloud}"
config_path="${PLANNER_CONFIG:-$repo_root/taskgraph_lab/configs/qwen3vl_4b_planner_lora_cloud.yaml}"
dataset_dir="$repo_root/taskgraph_lab/data/planner_sft_hard_curriculum_v1"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${PLANNER_RUN_ROOT:-$output_root/qwen3vl_4b_planner_hard_curriculum_v1_$timestamp}"
train_dir="$run_root/training"
eval_dir="$run_root/evaluation"

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory does not exist: $model_dir" >&2
  exit 1
fi
if [[ ! -f "$config_path" ]]; then
  echo "Planner config does not exist: $config_path" >&2
  exit 1
fi
if [[ ! -f "$dataset_dir/train.jsonl" || ! -f "$dataset_dir/test.jsonl" ]]; then
  echo "Planner dataset is incomplete: $dataset_dir" >&2
  exit 1
fi
if [[ -e "$run_root" ]]; then
  echo "Run directory already exists: $run_root" >&2
  exit 1
fi

mkdir -p "$run_root"
cd "$repo_root"

"$python_bin" -u -m taskgraph_lab.tools.train_qwen3vl_planner \
  --config "$config_path" \
  --output-dir "$train_dir"

"$python_bin" -u -m taskgraph_lab.tools.evaluate_qwen3vl_planner \
  --base-model "$model_dir" \
  --adapter "$train_dir" \
  --validation-file "$dataset_dir/test.jsonl" \
  --output-dir "$eval_dir" \
  --batch-size 1 \
  --max-new-tokens 512 \
  --constrained \
  --enable-recovery \
  --max-attempts 3 \
  --rag-mode off

echo "Planner training and evaluation completed: $run_root"

#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/sat-rs-vlm"
DATA_ROOT="/root/autodl-tmp/datasets"
MODEL_ROOT="/root/autodl-tmp/models"
OUTPUT_ROOT="/root/autodl-tmp/outputs"
BACKUP_ROOT="/root/autodl-fs/experiments"
ENV_NAME="rs-vlm"
TRAIN_CONFIG="configs/cloud/train_lora_autodl.yaml"
SMOKE_CONFIG="configs/cloud/train_lora_autodl_smoke.yaml"
EVAL_CONFIG="configs/cloud/evaluate_lora_autodl.yaml"
CURRENT_STAGE="startup"
TRAINING_COMPLETED=0
EXPERIMENT_DIR=""
BACKUP_DIR=""
PIPELINE_LOG=""

on_error() {
  local exit_code=$?
  local failed_command="$BASH_COMMAND"
  trap - ERR
  echo "[FAILED] Pipeline stopped at stage '$CURRENT_STAGE' with exit code $exit_code."

  local error_report="$OUTPUT_ROOT/reports/full_pipeline_error_$(date +%Y%m%d_%H%M%S).json"
  if ! mkdir -p "$OUTPUT_ROOT/reports" 2>/dev/null; then
    error_report="/root/autodl-tmp/full_pipeline_error_$(date +%Y%m%d_%H%M%S).txt"
    printf 'pipeline failed at stage=%s exit_code=%s command=%s\n' \
      "$CURRENT_STAGE" "$exit_code" "$failed_command" > "$error_report"
    echo "Fallback error report: $error_report"
  elif ! python - "$error_report" "$CURRENT_STAGE" \
    "$exit_code" "$failed_command" "$EXPERIMENT_DIR" "$BACKUP_DIR" "$PIPELINE_LOG" \
    "$TRAINING_COMPLETED" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "success": False,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "failed_command": sys.argv[4],
    "experiment_dir": sys.argv[5] or None,
    "backup_dir": sys.argv[6] or None,
    "pipeline_log": sys.argv[7] or None,
    "training_completed": sys.argv[8] == "1",
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Error report: {path}")
PY
  then
    error_report="$OUTPUT_ROOT/reports/full_pipeline_error_$(date +%Y%m%d_%H%M%S).txt"
    printf 'pipeline failed at stage=%s exit_code=%s command=%s\n' \
      "$CURRENT_STAGE" "$exit_code" "$failed_command" > "$error_report"
    echo "Fallback error report: $error_report"
  fi

  if [[ -n "$BACKUP_ROOT" && -d "$BACKUP_ROOT" && -f "$error_report" ]]; then
    mkdir -p "$BACKUP_ROOT/pipeline-errors" 2>/dev/null || true
    cp -f "$error_report" "$BACKUP_ROOT/pipeline-errors/" 2>/dev/null || true
  fi

  if [[ "$TRAINING_COMPLETED" == "1" ]]; then
    echo "[ACTION] Training completed; shutting down after saving the error report."
    sudo shutdown -h now || true
  else
    echo "[SAFE] Training did not complete; the AutoDL instance will remain running."
  fi
  exit "$exit_code"
}
trap on_error ERR

[[ -f /root/autodl_env.sh ]] || {
  echo "Missing /root/autodl_env.sh. Run setup_autodl.sh first." >&2
  exit 1
}
[[ -f /root/miniconda3/etc/profile.d/conda.sh ]] || {
  echo "Missing Conda activation script." >&2
  exit 1
}
[[ -d "$PROJECT_ROOT" ]] || {
  echo "Project directory does not exist: $PROJECT_ROOT" >&2
  exit 1
}

source /root/autodl_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export PROJECT_ROOT DATA_ROOT MODEL_ROOT OUTPUT_ROOT BACKUP_ROOT
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_ROOT/logs" "$BACKUP_ROOT"
PIPELINE_LOG="$OUTPUT_ROOT/logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

CURRENT_STAGE="environment and dataset validation"
echo "[1/8] Checking environment and dataset"
python scripts/environment/check_environment.py --require-model --require-gpu
python scripts/data/validate_dataset.py \
  --dataset-root "$DATA_ROOT/VRSBench" \
  --manifest-name project_metadata/dataset_manifest.json \
  --report "$OUTPUT_ROOT/reports/vrsbench_validation.json"

CURRENT_STAGE="evaluation JSONL preparation"
echo "[2/8] Preparing Qwen3-VL evaluation JSONL"
if [[ ! -s data/processed/qwen3vl_val.jsonl ]] || \
   [[ data/processed/qwen3vl_val.jsonl -ot data/processed/rs_val.jsonl ]]; then
  python scripts/convert_to_qwen3vl_format.py \
    --config configs/data/remote_sensing_data.yaml
fi
[[ -s data/processed/qwen3vl_val.jsonl ]] || {
  echo "Evaluation JSONL is missing or empty." >&2
  exit 1
}

CURRENT_STAGE="real-model smoke and formal LoRA training"
echo "[3/8] Running real-model smoke and formal LoRA training"
TRAIN_LOG_MARKER="$(mktemp /root/autodl-tmp/train-log-marker.XXXXXX)"
bash scripts/training/run_autodl_train.sh \
  --config "$TRAIN_CONFIG" \
  --smoke-config "$SMOKE_CONFIG" \
  --env-name "$ENV_NAME"
TRAINING_COMPLETED=1

CURRENT_STAGE="formal experiment resolution"
echo "[4/8] Resolving formal experiment directory"
TRAIN_LOG="$(
  find "$OUTPUT_ROOT/logs" -maxdepth 1 -type f -name 'train_*.log' \
    -newer "$TRAIN_LOG_MARKER" -print | sort | tail -n 1
)"
[[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]] || {
  echo "Could not locate the training log created by this run." >&2
  exit 1
}
EXPERIMENT_DIR="$(python - "$TRAIN_LOG" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
for line in reversed(text.splitlines()):
    if '"experiment_dir"' in line:
        print(line.split(":", 1)[1].strip().rstrip(",").strip('"'))
        break
PY
)"
[[ -n "$EXPERIMENT_DIR" && -d "$EXPERIMENT_DIR" ]] || {
  echo "Could not resolve the formal experiment directory." >&2
  exit 1
}
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
[[ -f "$CHECKPOINT_DIR/strategy_manifest.json" ]] || {
  echo "Missing strategy manifest: $CHECKPOINT_DIR/strategy_manifest.json" >&2
  exit 1
}
if [[ ! -f "$CHECKPOINT_DIR/adapter_model.safetensors" ]] && \
   [[ ! -f "$CHECKPOINT_DIR/adapter_model.bin" ]]; then
  echo "Missing final LoRA adapter weights in $CHECKPOINT_DIR" >&2
  exit 1
fi

CURRENT_STAGE="full validation-set generation evaluation"
echo "[5/8] Running full validation-set generation evaluation"
python scripts/evaluate_rs_vlm.py \
  --config "$EVAL_CONFIG" \
  --checkpoint "$CHECKPOINT_DIR" \
  --output-dir "$EXPERIMENT_DIR/evaluation"
[[ -s "$EXPERIMENT_DIR/evaluation/summary.json" ]] || {
  echo "Evaluation summary is missing or empty." >&2
  exit 1
}
[[ -s "$EXPERIMENT_DIR/evaluation/predictions.jsonl" ]] || {
  echo "Evaluation predictions are missing or empty." >&2
  exit 1
}

CURRENT_STAGE="experiment and evaluation backup"
echo "[6/8] Backing up experiment, evaluation and final adapter"
bash scripts/storage/backup_results.sh \
  --experiment-dir "$EXPERIMENT_DIR" \
  --backup-root "$BACKUP_ROOT" \
  --keep-checkpoints 2

CURRENT_STAGE="backup verification"
echo "[7/8] Verifying backup"
BACKUP_DIR="$BACKUP_ROOT/$(basename "$EXPERIMENT_DIR")"
[[ -s "$BACKUP_DIR/evaluation/summary.json" ]] || {
  echo "Backup is missing the evaluation summary." >&2
  exit 1
}
[[ -s "$BACKUP_DIR/checkpoints/strategy_manifest.json" ]] || {
  echo "Backup is missing the strategy manifest." >&2
  exit 1
}
if [[ ! -f "$BACKUP_DIR/checkpoints/adapter_model.safetensors" ]] && \
   [[ ! -f "$BACKUP_DIR/checkpoints/adapter_model.bin" ]]; then
  echo "Backup is missing final LoRA adapter weights." >&2
  exit 1
fi
[[ -d "$BACKUP_DIR/checkpoints/processor" ]] || {
  echo "Backup is missing the processor directory." >&2
  exit 1
}
sync

CURRENT_STAGE="shutdown"
echo "[8/8] Pipeline completed successfully"
echo "Experiment: $EXPERIMENT_DIR"
echo "Backup: $BACKUP_DIR"
echo "Pipeline log: $PIPELINE_LOG"
echo "Shutting down the AutoDL instance."
trap - ERR
sudo shutdown -h now

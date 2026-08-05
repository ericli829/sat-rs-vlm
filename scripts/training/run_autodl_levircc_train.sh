#!/usr/bin/env bash
set -Eeuo pipefail

MODE="joint"
PREPARE_ONLY=0
DRY_RUN=0
FORWARD_ONLY=0
MAX_STEPS=""
MAX_TRAIN_SAMPLES=""
MAX_EVAL_SAMPLES=""
OUTPUT_DIR=""
ENV_NAME="rs-vlm"
INITIAL_ADAPTER=""
ROUND_INDEX=0
LEARNING_RATE=""
NUM_TRAIN_EPOCHS=""

usage() {
  cat <<'EOF'
Usage: run_autodl_levircc_train.sh [options]
  --mode joint|levircc  Train one shared adapter or a LEVIR-CC-only ablation
  --prepare-only        Build and validate JSONL files, then stop
  --dry-run             Validate the resolved training configuration
  --forward-only        Run one real forward pass, then stop
  --max-steps N         Override the number of optimizer steps
  --max-train-samples N Limit training samples for a smoke run
  --max-eval-samples N  Limit validation samples for a smoke run
  --output-dir PATH     Override the timestamped output directory
  --initial-adapter PATH Initialize from a completed trainable LoRA adapter
  --round-index N       Replay round used for deterministic data rotation
  --learning-rate RATE  Override the training learning rate
  --num-train-epochs N  Override the number of epochs
  --env-name NAME       Conda environment name (default: rs-vlm)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --forward-only) FORWARD_ONLY=1; shift ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    --max-eval-samples) MAX_EVAL_SAMPLES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --initial-adapter) INITIAL_ADAPTER="$2"; shift 2 ;;
    --round-index) ROUND_INDEX="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --num-train-epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MODE" == "joint" || "$MODE" == "levircc" ]] || {
  echo "--mode must be joint or levircc" >&2
  exit 2
}
[[ -f /root/autodl_env.sh ]] || {
  echo "Missing /root/autodl_env.sh. Run scripts/environment/setup_autodl.sh first." >&2
  exit 1
}

source /root/autodl_env.sh
source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_CONFIG="configs/data/autodl_vrsbench_levircc.yaml"
PROCESSED_ROOT="$PROJECT_ROOT/data/processed/multisource"
mkdir -p "$PROCESSED_ROOT" "$OUTPUT_ROOT/logs"

PREPARE_ARGS=(
  "$AUTODL_PYTHON" scripts/data/prepare_multisource_training_data.py
  --config "$DATA_CONFIG"
  --round-index "$ROUND_INDEX"
)

if [[ "$MODE" == "joint" ]]; then
  TRAIN_CONFIG="configs/train/qwen3vl_autodl_vrsbench_levircc.yaml"
  DEFAULT_OUTPUT="$OUTPUT_ROOT/vrsbench_levircc_replay_r${ROUND_INDEX}_$(date +%Y%m%d_%H%M%S)/checkpoints"
else
  TRAIN_CONFIG="configs/train/qwen3vl_autodl_levircc_only.yaml"
  DEFAULT_OUTPUT="$OUTPUT_ROOT/levircc_lora_$(date +%Y%m%d_%H%M%S)/checkpoints"
  PREPARE_ARGS+=(
    --include-source LEVIR-CC
    --train-output "$PROCESSED_ROOT/levircc_train.jsonl"
    --validation-output "$PROCESSED_ROOT/levircc_val.jsonl"
    --report-output "$PROCESSED_ROOT/levircc_report.json"
  )
fi

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
LOG_FILE="$OUTPUT_ROOT/logs/${MODE}_levircc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Mode: $MODE"
echo "Python: $AUTODL_PYTHON"
echo "Output: $OUTPUT_DIR"
echo "Initial adapter: ${INITIAL_ADAPTER:-<base model>}"
echo "Replay round: $ROUND_INDEX"
echo "Log: $LOG_FILE"

"${PREPARE_ARGS[@]}"
if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  exit 0
fi

TRAIN_ARGS=(
  "$AUTODL_PYTHON" scripts/train_qwen3vl_lora.py
  --config "$TRAIN_CONFIG"
  --output-dir "$OUTPUT_DIR"
)
[[ -n "$MAX_STEPS" ]] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
[[ -n "$MAX_TRAIN_SAMPLES" ]] && TRAIN_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
[[ -n "$MAX_EVAL_SAMPLES" ]] && TRAIN_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
[[ -n "$INITIAL_ADAPTER" ]] && TRAIN_ARGS+=(--initial-adapter "$INITIAL_ADAPTER")
[[ -n "$LEARNING_RATE" ]] && TRAIN_ARGS+=(--learning-rate "$LEARNING_RATE")
[[ -n "$NUM_TRAIN_EPOCHS" ]] && TRAIN_ARGS+=(--num-train-epochs "$NUM_TRAIN_EPOCHS")

"${TRAIN_ARGS[@]}" --dry-run
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi
if [[ "$FORWARD_ONLY" -eq 1 ]]; then
  "${TRAIN_ARGS[@]}" --forward-only
  exit 0
fi
"${TRAIN_ARGS[@]}"

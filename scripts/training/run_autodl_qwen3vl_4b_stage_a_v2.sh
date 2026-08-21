#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="rs-vlm"
PROJECT_ROOT_DEFAULT="/root/autodl-tmp/sat-rs-vlm"
RUN_ROOT=""
MODEL_DIR=""
MODEL_ROOT_OVERRIDE=""
DATA_ROOT_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
SHUTDOWN_AFTER_RUN=0
DIAGNOSTIC_MODE=0
TEST_SHUTDOWN=0
CURRENT_STAGE="startup"
STAGE_LOG=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --run-root=*)
      RUN_ROOT="${1#*=}"
      ARGS+=("$1")
      shift
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --model-dir=*)
      MODEL_DIR="${1#*=}"
      shift
      ;;
    --model-root)
      MODEL_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --model-root=*)
      MODEL_ROOT_OVERRIDE="${1#*=}"
      shift
      ;;
    --data-root)
      DATA_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT_OVERRIDE="${1#*=}"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT_OVERRIDE="${1#*=}"
      shift
      ;;
    --shutdown|--shutdown-after-run)
      SHUTDOWN_AFTER_RUN=1
      shift
      ;;
    --test-shutdown)
      TEST_SHUTDOWN=1
      shift
      ;;
    --prepare-only|--dry-run|--forward-only)
      DIAGNOSTIC_MODE=1
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

shutdown_host() {
  if [[ -n "${STAGE_A_V2_SHUTDOWN_MOCK_FILE:-}" ]]; then
    printf 'shutdown_requested\n' > "$STAGE_A_V2_SHUTDOWN_MOCK_FILE"
    return 0
  fi
  sync || true
  local shutdown_python="${AUTODL_PYTHON:-$(command -v python3 || true)}"
  if [[ -x /usr/bin/shutdown ]] && [[ -n "$shutdown_python" ]] && \
    "$shutdown_python" -c 'import os; raise SystemExit(os.system("/usr/bin/shutdown"))'; then
    return 0
  fi
  echo "[WARN] Python os.system(\"/usr/bin/shutdown\") was refused; trying compatibility fallbacks." >&2
  if [[ -x /usr/sbin/shutdown ]] && /usr/sbin/shutdown -h now; then
    return 0
  fi
  if [[ -x /sbin/shutdown ]] && /sbin/shutdown -h now; then
    return 0
  fi
  if command -v poweroff >/dev/null 2>&1 && poweroff -f; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n poweroff -f; then
    return 0
  fi
  echo "[ERROR] All supported shutdown commands failed." >&2
  return 1
}

if [[ "$TEST_SHUTDOWN" == "1" ]]; then
  if [[ -z "${STAGE_A_V2_SHUTDOWN_MOCK_FILE:-}" ]]; then
    echo "--test-shutdown requires STAGE_A_V2_SHUTDOWN_MOCK_FILE" >&2
    exit 2
  fi
  shutdown_host
  exit 0
fi

on_error() {
  local exit_code=$?
  local failed_command="$BASH_COMMAND"
  trap - ERR
  set +e
  mkdir -p "$REPORT_DIR"
  {
    printf 'success=false\n'
    printf 'stage=%s\n' "$CURRENT_STAGE"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'failed_command=%s\n' "$failed_command"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'log=%s\n' "${STAGE_LOG:-none}"
    printf 'resume_hint=bash scripts/training/run_autodl_qwen3vl_4b_stage_a_v2.sh --run-root %s --resume --shutdown\n' "$RUN_ROOT"
  } > "$REPORT_DIR/stage_a_v2_error.txt"
  sync || true
  if [[ "$SHUTDOWN_AFTER_RUN" == "1" && "$DIAGNOSTIC_MODE" == "0" ]]; then
    echo "[ACTION] Unrecoverable Stage-A v2 failure; shutting down."
    shutdown_host || true
  fi
  exit "$exit_code"
}
trap on_error ERR

if [[ ! -f /root/autodl_env.sh ]]; then
  echo "Missing /root/autodl_env.sh. Run scripts/environment/setup_autodl.sh first." >&2
  false
fi
source /root/autodl_env.sh
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
MODEL_ROOT="${MODEL_ROOT_OVERRIDE:-${MODEL_ROOT:-/root/autodl-tmp/models}}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${DATA_ROOT:-/root/autodl-tmp/datasets}}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-${OUTPUT_ROOT:-/root/autodl-tmp/outputs}}"
QWEN3VL_4B_MODEL_DIR="${MODEL_DIR:-${QWEN3VL_4B_MODEL_DIR:-$MODEL_ROOT/Qwen3-VL-4B-Instruct}}"

[[ -d "$QWEN3VL_4B_MODEL_DIR" ]] || {
  echo "Missing Qwen3-VL-4B model directory: $QWEN3VL_4B_MODEL_DIR" >&2
  echo "Pass --model-dir PATH or set QWEN3VL_4B_MODEL_DIR." >&2
  exit 1
}
[[ -f "$QWEN3VL_4B_MODEL_DIR/config.json" ]] || {
  echo "Qwen3-VL-4B model directory has no config.json: $QWEN3VL_4B_MODEL_DIR" >&2
  exit 1
}
export MODEL_ROOT DATA_ROOT OUTPUT_ROOT QWEN3VL_4B_MODEL_DIR

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/qwen3vl_4b_stage_a_v2_$(date +%Y%m%d_%H%M%S)"
  ARGS+=("--run-root" "$RUN_ROOT")
fi
REPORT_DIR="$RUN_ROOT/reports"
LOG_DIR="$RUN_ROOT/logs"

source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"
STAGE_LOG="$LOG_DIR/stage_a_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$STAGE_LOG") 2>&1

echo "[ENV] QWEN3VL_4B_MODEL_DIR=$QWEN3VL_4B_MODEL_DIR"
echo "[ENV] DATA_ROOT=$DATA_ROOT"
echo "[ENV] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[RUN] RUN_ROOT=$RUN_ROOT"

if [[ "$SHUTDOWN_AFTER_RUN" == "1" && "$DIAGNOSTIC_MODE" == "1" ]]; then
  echo "[SAFE] Shutdown is disabled for prepare-only/dry-run/forward-only."
  SHUTDOWN_AFTER_RUN=0
fi

CURRENT_STAGE="Stage-A v2 population, R0, R1, and evaluation workflow"
"$AUTODL_PYTHON" scripts/training/run_qwen3vl_4b_stage_a_v2.py "${ARGS[@]}"

CURRENT_STAGE="completion report"
{
  printf 'success=true\n'
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'log=%s\n' "$STAGE_LOG"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
} > "$REPORT_DIR/stage_a_v2_completion.txt"
sync

if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
  CURRENT_STAGE="shutdown"
  echo "[ACTION] Stage-A v2 completed; shutting down the AutoDL instance."
  trap - ERR
  shutdown_host
fi

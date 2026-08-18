#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="rs-vlm"
PROJECT_ROOT_DEFAULT="/root/autodl-tmp/sat-rs-vlm"
RUN_ROOT=""
SHUTDOWN_AFTER_RUN=0
CURRENT_STAGE="startup"
STAGE_LOG=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
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
    --shutdown-after-run)
      SHUTDOWN_AFTER_RUN=1
      shift
      ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="/root/autodl-tmp/outputs/qwen3vl_4b_stage_a_$(date +%Y%m%d_%H%M%S)"
  ARGS+=("--run-root" "$RUN_ROOT")
fi
REPORT_DIR="$RUN_ROOT/reports"
LOG_DIR="$RUN_ROOT/logs"

shutdown_host() {
  sync || true
  if [[ -x /usr/sbin/shutdown ]]; then
    /usr/sbin/shutdown -h now
  elif [[ -x /sbin/shutdown ]]; then
    /sbin/shutdown -h now
  elif command -v poweroff >/dev/null 2>&1; then
    poweroff
  elif command -v sudo >/dev/null 2>&1; then
    sudo shutdown -h now
  else
    echo "[ERROR] No shutdown command is available." >&2
    return 1
  fi
}

on_error() {
  local exit_code=$?
  local failed_command="$BASH_COMMAND"
  trap - ERR
  set +e
  if ! mkdir -p "$REPORT_DIR" 2>/dev/null; then
    REPORT_DIR="/root/autodl-tmp"
  fi
  local report="$REPORT_DIR/stage_a_error_$(date +%Y%m%d_%H%M%S).txt"
  local last_adapter=""
  last_adapter="$(
    find "$RUN_ROOT" -mindepth 2 -maxdepth 3 -type f -name adapter_config.json \
      -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  {
    printf 'success=false\n'
    printf 'stage=%s\n' "$CURRENT_STAGE"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'failed_command=%s\n' "$failed_command"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'last_valid_adapter=%s\n' "${last_adapter:-none}"
    printf 'log=%s\n' "${STAGE_LOG:-none}"
    printf 'resume_hint=bash scripts/training/run_autodl_qwen3vl_4b_stage_a.sh --run-root %s --resume --shutdown-after-run\n' "$RUN_ROOT"
  } > "$report"
  if [[ -d /root/autodl-fs ]]; then
    mkdir -p /root/autodl-fs/experiments/stage-a-errors 2>/dev/null
    cp -f "$report" /root/autodl-fs/experiments/stage-a-errors/ 2>/dev/null
  fi
  echo "[FAILED] Stage-A stopped during: $CURRENT_STAGE"
  echo "Error report: $report"
  sync || true
  if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
    echo "[ACTION] Saving failure state and shutting down the AutoDL instance."
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
source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"
STAGE_LOG="$LOG_DIR/stage_a_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$STAGE_LOG") 2>&1

CURRENT_STAGE="Stage-A data preparation, training, and evaluation"
"$AUTODL_PYTHON" scripts/training/run_qwen3vl_4b_stage_a.py "${ARGS[@]}"

CURRENT_STAGE="completion report"
{
  printf 'success=true\n'
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'log=%s\n' "$STAGE_LOG"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
} > "$REPORT_DIR/stage_a_completion.txt"
sync

if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
  CURRENT_STAGE="shutdown"
  echo "[ACTION] Stage-A completed; shutting down the AutoDL instance."
  trap - ERR
  shutdown_host
fi

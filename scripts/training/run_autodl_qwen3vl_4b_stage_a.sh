#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="rs-vlm"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

[[ -f /root/autodl_env.sh ]] || {
  echo "Missing /root/autodl_env.sh. Run scripts/environment/setup_autodl.sh first." >&2
  exit 1
}
source /root/autodl_env.sh
source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$AUTODL_PYTHON" scripts/training/run_qwen3vl_4b_stage_a.py "${ARGS[@]}"

#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL unattended wrapper. It follows the same environment and shutdown
# contract as the validated Stage-A launcher.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT_DEFAULT="/root/autodl-tmp/sat-rs-vlm"
ENV_NAME="rs-vlm"

SHUTDOWN_AFTER_RUN=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --shutdown|--shutdown-after-run)
      SHUTDOWN_AFTER_RUN=1
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

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

if [[ -f /root/autodl_env.sh ]]; then
  source /root/autodl_env.sh
fi
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

set +e
"$AUTODL_PYTHON" scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py "${ARGS[@]}"
EXIT_CODE=$?
set -e
sync || true

if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
  echo "[ACTION] LR + merger sweep completed; shutting down the AutoDL instance."
  shutdown_host || true
fi

exit "$EXIT_CODE"

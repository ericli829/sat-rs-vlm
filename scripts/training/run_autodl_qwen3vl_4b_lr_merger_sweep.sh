#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL unattended wrapper. The Python runner owns experiment-level status and
# explicit shutdown validation; this trap guarantees buffered logs are flushed.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SHUTDOWN=0
for arg in "$@"; do
  if [[ "$arg" == "--shutdown" ]]; then
    SHUTDOWN=1
  fi
done

on_exit() {
  sync || true
}
trap on_exit EXIT INT TERM

python scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py "$@"

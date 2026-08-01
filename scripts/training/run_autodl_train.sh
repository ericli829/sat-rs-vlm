#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/cloud/train_lora_autodl.yaml"
SMOKE_CONFIG="configs/cloud/train_lora_autodl_smoke.yaml"
SKIP_SMOKE=0
RESUME=""
AUTO_SHUTDOWN=0
BACKUP_AFTER_TRAIN=0
ENV_NAME="rs-vlm"

usage() {
  cat <<'EOF'
Usage: run_autodl_train.sh [options]
  --config PATH          Formal training config
  --smoke-config PATH    Real-model smoke config
  --skip-smoke           Skip the mandatory smoke stage
  --resume PATH          Resume from an explicit checkpoint
  --env-name NAME        Conda environment name
  --backup-after-train   Back up selected results after success
  --auto-shutdown        Shut down only after successful training and backup
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --smoke-config) SMOKE_CONFIG="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --resume) RESUME="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --backup-after-train) BACKUP_AFTER_TRAIN=1; shift ;;
    --auto-shutdown) AUTO_SHUTDOWN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f /root/autodl_env.sh ]] || {
  echo "Missing /root/autodl_env.sh. Run scripts/environment/setup_autodl.sh first." >&2
  exit 1
}
source /root/autodl_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/train_$(date +%Y%m%d_%H%M%S).log"

python scripts/environment/check_environment.py --require-model --require-gpu \
  2>&1 | tee -a "$LOG_FILE"
python scripts/data/validate_dataset.py \
  --dataset-root "$DATA_ROOT/VRSBench" \
  --manifest-name project_metadata/dataset_manifest.json \
  2>&1 | tee -a "$LOG_FILE"

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  python scripts/training/run_train.py \
    --environment autodl \
    --env-config configs/cloud/autodl.yaml \
    --config "$SMOKE_CONFIG" \
    2>&1 | tee -a "$LOG_FILE"
fi

TRAIN_ARGS=(
  python scripts/training/run_train.py
  --environment autodl
  --env-config configs/cloud/autodl.yaml
  --config "$CONFIG"
)
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME")
fi
"${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

if [[ "$BACKUP_AFTER_TRAIN" -eq 1 ]]; then
  EXPERIMENT_DIR="$(python - "$LOG_FILE" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
for line in reversed(text.splitlines()):
    if '"experiment_dir"' in line:
        value = line.split(":", 1)[1].strip().rstrip(",").strip('"')
        print(value)
        break
PY
)"
  [[ -n "$EXPERIMENT_DIR" ]] || {
    echo "Could not determine experiment directory from log; backup aborted." >&2
    exit 1
  }
  bash scripts/storage/backup_results.sh \
    --experiment-dir "$EXPERIMENT_DIR" \
    --backup-root "$BACKUP_ROOT"
fi

if [[ "$AUTO_SHUTDOWN" -eq 1 ]]; then
  [[ "$BACKUP_AFTER_TRAIN" -eq 1 ]] || {
    echo "--auto-shutdown requires --backup-after-train." >&2
    exit 1
  }
  sudo shutdown -h now
fi

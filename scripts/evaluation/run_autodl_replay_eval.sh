#!/usr/bin/env bash
set -Eeuo pipefail

ADAPTER_DIR=""
EVAL_ROOT=""
ENV_NAME="rs-vlm"

usage() {
  cat <<'EOF'
Usage: run_autodl_replay_eval.sh --adapter-dir PATH [options]
  --adapter-dir PATH  Adapter produced by replay training
  --eval-root PATH    Evaluation output root
  --env-name NAME     Conda environment name (default: rs-vlm)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ADAPTER_DIR" ]] || {
  echo "--adapter-dir is required" >&2
  exit 2
}
[[ -f "$ADAPTER_DIR/adapter_config.json" ]] || {
  echo "Adapter is invalid: $ADAPTER_DIR" >&2
  exit 1
}

source /root/autodl_env.sh
source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
activate_autodl_python "$ENV_NAME"
cd "$PROJECT_ROOT"

EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/eval/replay_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$EVAL_ROOT/vrsbench" "$EVAL_ROOT/levircc" "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/replay_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

"$AUTODL_PYTHON" scripts/data/prepare_multisource_training_data.py \
  --config configs/data/autodl_vrsbench_levircc.yaml \
  --include-source LEVIR-CC \
  --round-index 0 \
  --train-output data/processed/multisource/levircc_train.jsonl \
  --validation-output data/processed/multisource/levircc_val.jsonl \
  --report-output data/processed/multisource/levircc_report.json

"$AUTODL_PYTHON" scripts/evaluate_rs_vlm.py \
  --config configs/cloud/evaluate_replay_vrsbench.yaml \
  --checkpoint "$ADAPTER_DIR" \
  --output-dir "$EVAL_ROOT/vrsbench" \
  --batch-size 16

"$AUTODL_PYTHON" scripts/evaluate_rs_vlm.py \
  --config configs/cloud/evaluate_replay_levircc.yaml \
  --checkpoint "$ADAPTER_DIR" \
  --output-dir "$EVAL_ROOT/levircc" \
  --batch-size 8

"$AUTODL_PYTHON" - "$ADAPTER_DIR" "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

adapter = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
payload = {
    "adapter": str(adapter),
    "vrsbench_summary": str(root / "vrsbench" / "summary.json"),
    "levircc_summary": str(root / "levircc" / "summary.json"),
}
(root / "evaluation_index.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "Replay evaluation completed: $EVAL_ROOT"

#!/usr/bin/env bash
set -Eeuo pipefail

INITIAL_ADAPTER=""
RUN_ROOT=""
ENV_NAME="rs-vlm"
SKIP_ROUND_1=0

usage() {
  cat <<'EOF'
Usage: run_autodl_levircc_replay.sh --initial-adapter PATH [options]
  --initial-adapter PATH  Completed VRSBench adapter used to initialize round 1
  --run-root PATH         Root directory for both replay rounds
  --env-name NAME         Conda environment name (default: rs-vlm)
  --skip-round-1          Reuse an existing round_1_adapter under run-root
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --initial-adapter) INITIAL_ADAPTER="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --skip-round-1) SKIP_ROUND_1=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$INITIAL_ADAPTER" ]] || {
  echo "--initial-adapter is required" >&2
  exit 2
}
[[ -f "$INITIAL_ADAPTER/adapter_config.json" ]] || {
  echo "Initial adapter is invalid: $INITIAL_ADAPTER" >&2
  exit 1
}

source /root/autodl_env.sh
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/sat-rs-vlm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
cd "$PROJECT_ROOT"

RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/vrsbench_levircc_replay_$(date +%Y%m%d_%H%M%S)}"
ROUND_1_ADAPTER="$RUN_ROOT/round_1_adapter"
ROUND_2_ADAPTER="$RUN_ROOT/round_2_adapter"
mkdir -p "$RUN_ROOT"

if [[ "$SKIP_ROUND_1" -eq 0 ]]; then
  bash scripts/training/run_autodl_levircc_train.sh \
    --mode joint \
    --env-name "$ENV_NAME" \
    --initial-adapter "$INITIAL_ADAPTER" \
    --round-index 0 \
    --learning-rate 0.00002 \
    --num-train-epochs 1 \
    --output-dir "$ROUND_1_ADAPTER"
fi

[[ -f "$ROUND_1_ADAPTER/adapter_config.json" ]] || {
  echo "Round 1 adapter is missing: $ROUND_1_ADAPTER" >&2
  exit 1
}

bash scripts/training/run_autodl_levircc_train.sh \
  --mode joint \
  --env-name "$ENV_NAME" \
  --initial-adapter "$ROUND_1_ADAPTER" \
  --round-index 1 \
  --learning-rate 0.00001 \
  --num-train-epochs 1 \
  --output-dir "$ROUND_2_ADAPTER"

cat > "$RUN_ROOT/replay_result.json" <<EOF
{
  "initial_adapter": "$INITIAL_ADAPTER",
  "round_1_adapter": "$ROUND_1_ADAPTER",
  "round_2_adapter": "$ROUND_2_ADAPTER",
  "recommended_adapter": "$ROUND_2_ADAPTER"
}
EOF

echo "Replay training completed: $ROUND_2_ADAPTER"

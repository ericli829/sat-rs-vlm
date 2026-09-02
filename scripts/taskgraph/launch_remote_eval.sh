#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s INPUT_JSONL OUTPUT_JSONL LOG_FILE\n' "$0" >&2
    exit 2
fi

source /root/autodl_env.sh

PROJECT_ROOT=/root/autodl-tmp/sat-rs-vlm
PYTHON=/root/miniconda3/envs/rs-vlm/bin/python3.12
INPUT=$1
OUTPUT=$2
LOG=$3
PID_FILE=${OUTPUT%.jsonl}.pid
PROVIDER_CONFIG=$PROJECT_ROOT/configs/taskgraph/runtime.real.example.yaml
IMAGE_ROOT=/root/autodl-fs/datasets/MME-RealWorld-RS

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export QWEN3VL_4B_PLANNER_LORA=/root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner_hard_refine_v1_20260902_234235
export QWEN3VL_2B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-2B-Instruct
export QWEN3VL_2B_SEMANTIC_LORA=/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter
export GEORSCLIP_CHECKPOINT=/root/autodl-fs/models/GeoRSCLIP/GeoRSCLIP-ViT-B-32.pt
export LAE_DINO_SOURCE_ROOT=/root/autodl-fs/rs_detectors/lae_dino/source/LAE-DINO
export LAE_DINO_CONFIG_LAE1M=$LAE_DINO_SOURCE_ROOT/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py
export LAE_DINO_CHECKPOINT_LAE1M=/root/autodl-fs/rs_detectors/lae_dino/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth
export LAE_DINO_BERT_ROOT=/root/autodl-fs/rs_detectors/lae_dino/weights/bert-base-uncased
export LAE_DINO_PYTHON=/root/miniconda3/envs/lae-dino-gpu/bin/python3
export LAE_DINO_STDERR_LOG=${LOG%.log}.lae.stderr.log
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for path in "$PYTHON" "$INPUT" "$PROVIDER_CONFIG" "$IMAGE_ROOT" "$QWEN3VL_4B_MODEL_DIR" \
    "$QWEN3VL_4B_PLANNER_LORA" "$QWEN3VL_2B_MODEL_DIR" "$QWEN3VL_2B_SEMANTIC_LORA" \
    "$GEORSCLIP_CHECKPOINT" "$LAE_DINO_SOURCE_ROOT" "$LAE_DINO_CONFIG_LAE1M" \
    "$LAE_DINO_CHECKPOINT_LAE1M" "$LAE_DINO_BERT_ROOT" "$LAE_DINO_PYTHON"; do
    if [[ ! -e $path ]]; then
        printf 'missing required path: %s\n' "$path" >&2
        exit 1
    fi
done

if [[ -e $OUTPUT || -e $PID_FILE ]]; then
    printf 'refusing to overwrite output or pid file: %s\n' "$OUTPUT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
cd "$PROJECT_ROOT"
nohup "$PYTHON" -u "$PROJECT_ROOT/scripts/taskgraph/evaluate_runtime.py" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --provider-config "$PROVIDER_CONFIG" \
    --image-root "$IMAGE_ROOT" \
    --no-resume >"$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

if ! kill -0 "$pid" 2>/dev/null; then
    printf 'launch failed: pid=%s\n' "$pid" >&2
    tail -80 "$LOG" 2>/dev/null || true
    exit 1
fi
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
    printf 'process exited after launch: pid=%s\n' "$pid" >&2
    tail -80 "$LOG" 2>/dev/null || true
    exit 1
fi

printf 'STARTED pid=%s\n' "$pid"
printf 'input=%s\noutput=%s\nlog=%s\npid_file=%s\n' "$INPUT" "$OUTPUT" "$LOG" "$PID_FILE"
ps -p "$pid" -o pid,ppid,stat,etime,cmd

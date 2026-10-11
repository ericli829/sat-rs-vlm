#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011
FOURB=/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011
REPO=$FOURB/workspace_repacked
LOGS=$ROOT/logs
EVAL_DIR=$ROOT/evaluation/accuracy_20261011
GRID_DIR=$EVAL_DIR/grid
ROUTE_B_SUMMARY=$GRID_DIR/route_b_batch8.summary.json
OLD_PID_FILE=$LOGS/accuracy_eval.batch2.pid
PY=/root/miniconda3/envs/rs-vlm/bin/python
OLD_PID=$(cat "$LOGS/accuracy_eval.pid")

mkdir -p "$LOGS" "$EVAL_DIR"
exec > >(tee -a "$LOGS/accuracy_eval.remaining.pipeline.log") 2>&1
echo "remaining_eval_started=$(date --iso-8601=seconds) old_eval_pid=$OLD_PID"

# Wait for Route B Stage1 to finish and flush its summary. Linux inotify avoids
# repeatedly polling the long-running inference process.
"$PY" - "$GRID_DIR" "$(basename "$ROUTE_B_SUMMARY")" <<'PY'
import ctypes
import json
import os
import select
import struct
import sys

directory, target = sys.argv[1:]
path = os.path.join(directory, target)
if os.path.isfile(path):
    with open(path, encoding="utf-8") as stream:
        json.load(stream)
    raise SystemExit(0)
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(0)
if fd < 0:
    raise OSError(ctypes.get_errno(), "inotify_init1 failed")
watch = libc.inotify_add_watch(fd, os.fsencode(directory), 0x00000008 | 0x00000080 | 0x00000100)
if watch < 0:
    raise OSError(ctypes.get_errno(), "inotify_add_watch failed")
while True:
    ready, _, _ = select.select([fd], [], [])
    if not ready:
        continue
    data = os.read(fd, 8192)
    offset = 0
    while offset < len(data):
        _, _, _, length = struct.unpack_from("iIII", data, offset)
        offset += 16
        name = data[offset : offset + length].split(b"\0", 1)[0].decode()
        offset += length
        if name == target and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as stream:
                    json.load(stream)
            except (OSError, json.JSONDecodeError):
                continue
            print(f"route_b_summary_flushed={path}", flush=True)
            raise SystemExit(0)
PY
if [[ $? -ne 0 ]]; then
  echo "ERROR: failed waiting for Route B summary"
  exit 41
fi

echo "route_b_stage1_complete=$(date --iso-8601=seconds); switching remaining evaluations to larger batches"
if [[ -s "$LOGS/accuracy_eval.pid" ]]; then
  CHILDREN=$(ps --ppid "$OLD_PID" -o pid= || true)
  for child in $CHILDREN; do kill -TERM "$child" 2>/dev/null || true; done
  kill -TERM "$OLD_PID" 2>/dev/null || true
fi
sleep 2
OLD_STATE=$(ps -p "$OLD_PID" -o stat= 2>/dev/null | tr -d ' ' || true)
if [[ -n "$OLD_STATE" && "$OLD_STATE" != Z* ]]; then
  echo "ERROR: original accuracy runner did not stop after Route B summary"
  exit 42
fi
mv "$LOGS/accuracy_eval.pid" "$OLD_PID_FILE" 2>/dev/null || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rs-vlm
export PYTHONPATH="$REPO/src:$REPO"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

"$PY" "$ROOT/evaluate_grid_generation.py" \
  --base-model /root/autodl-tmp/models/Qwen3-VL-2B-Instruct \
  --validation-file "$ROOT/data/grid_dev.jsonl" \
  --image-root /root/autodl-tmp/datasets \
  --output-dir "$GRID_DIR" \
  --batch-size 8 --max-new-tokens 128 \
  --route "route_b_stage2_batch8=$ROOT/runs/route_b_stage2_batch8" \
  2>&1 | tee "$LOGS/accuracy_eval.grid_remaining.log"
GRID_RC=${PIPESTATUS[0]}
echo "grid_remaining_eval_exit=$GRID_RC"

"$PY" "$ROOT/evaluate_planner_generation_batched.py" \
  --base-model /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
  --validation-file "$FOURB/workspace/taskgraph_lab/data/planner_quality_clean_v1/validation.jsonl" \
  --output-dir "$EVAL_DIR/planner_4b_batched" \
  --batch-size 4 --max-prompt-tokens 1536 --max-new-tokens 512 \
  --adapter "before=/root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner_hard_refine_v1_20260902_234235" \
  --adapter "after=$FOURB/output/training" \
  2>&1 | tee "$LOGS/accuracy_eval.planner_batched.log"
PLANNER_RC=${PIPESTATUS[0]}
echo "planner_batched_eval_exit=$PLANNER_RC"

if [[ "$GRID_RC" -ne 0 || "$PLANNER_RC" -ne 0 ]]; then
  exit 1
fi
echo "remaining_accuracy_evaluations=COMPLETE at $(date --iso-8601=seconds)"

#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011
FOURB=/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011
REPO=$FOURB/workspace_repacked
LOGS=$ROOT/logs
EVAL_DIR=$ROOT/evaluation/accuracy_20261011
STATUS=$LOGS/route_b_stage2_batch8.status.txt
PY=/root/miniconda3/envs/rs-vlm/bin/python

mkdir -p "$LOGS" "$EVAL_DIR"
exec > >(tee -a "$LOGS/accuracy_eval.pipeline.log") 2>&1

echo "queued_at=$(date --iso-8601=seconds)"
echo "waiting_for=$STATUS"
while [[ ! -s "$STATUS" ]] || ! grep -qE '^(COMPLETE|FAILED)' "$STATUS"; do
  sleep 30
done
STAGE_STATUS=$(head -n 1 "$STATUS")
echo "stage2_status=$STAGE_STATUS at $(date --iso-8601=seconds)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rs-vlm
export PYTHONPATH="$REPO/src:$REPO"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

GRID_ARGS=(
  "$ROOT/evaluate_grid_generation.py"
  --base-model /root/autodl-tmp/models/Qwen3-VL-2B-Instruct
  --validation-file "$ROOT/data/grid_dev.jsonl"
  --image-root /root/autodl-tmp/datasets
  --output-dir "$EVAL_DIR/grid"
  --batch-size 8
  --max-new-tokens 128
  --route "route_a_batch4=$ROOT/runs/route_a_batch4"
  --route "route_b_batch8=$ROOT/runs/route_b_batch8"
)
if [[ -s "$ROOT/runs/route_b_stage2_batch8/adapter_model.safetensors" ]]; then
  GRID_ARGS+=(--route "route_b_stage2_batch8=$ROOT/runs/route_b_stage2_batch8")
else
  echo "WARNING: Stage2 adapter missing; evaluating completed Stage1 routes only"
fi
"$PY" "${GRID_ARGS[@]}" 2>&1 | tee "$LOGS/accuracy_eval.grid.log"
GRID_RC=${PIPESTATUS[0]}
echo "grid_eval_exit=$GRID_RC"

VALIDATION=$FOURB/workspace/taskgraph_lab/data/planner_quality_clean_v1/validation.jsonl
BASE=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
OLD_ADAPTER=/root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner_hard_refine_v1_20260902_234235
NEW_ADAPTER=$FOURB/output/training
PLANNER_TOOL=$REPO/taskgraph_lab/tools/evaluate_qwen3vl_planner.py
if [[ -s "$VALIDATION" && -s "$OLD_ADAPTER/adapter_model.safetensors" && -s "$NEW_ADAPTER/adapter_model.safetensors" ]]; then
  "$PY" "$PLANNER_TOOL" \
    --base-model "$BASE" --adapter "$OLD_ADAPTER" \
    --validation-file "$VALIDATION" --output-dir "$EVAL_DIR/planner_4b_before" \
    --batch-size 1 --max-new-tokens 512 --max-prompt-tokens 1536 \
    2>&1 | tee "$LOGS/accuracy_eval.planner_before.log"
  OLD_RC=${PIPESTATUS[0]}
  echo "planner_before_eval_exit=$OLD_RC"
  "$PY" "$PLANNER_TOOL" \
    --base-model "$BASE" --adapter "$NEW_ADAPTER" \
    --validation-file "$VALIDATION" --output-dir "$EVAL_DIR/planner_4b_after" \
    --batch-size 1 --max-new-tokens 512 --max-prompt-tokens 1536 \
    2>&1 | tee "$LOGS/accuracy_eval.planner_after.log"
  NEW_RC=${PIPESTATUS[0]}
  echo "planner_after_eval_exit=$NEW_RC"
else
  echo "ERROR: Planner eval inputs missing: validation=$VALIDATION old_adapter=$OLD_ADAPTER new_adapter=$NEW_ADAPTER"
  exit 42
fi

if [[ "$GRID_RC" -ne 0 || "$OLD_RC" -ne 0 || "$NEW_RC" -ne 0 ]]; then
  exit 1
fi
echo "all_accuracy_evaluations=COMPLETE at $(date --iso-8601=seconds)"

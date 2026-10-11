#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011
PARENT=/root/autodl-fs/data/sat-rs-vlm-training
REPO=$PARENT/planner_4b_clean_20261011/workspace_repacked
BASE_CONFIG=$ROOT/configs/config_route_b.yaml
CONFIG=$ROOT/configs/config_route_b_stage2.yaml
PARENT_ADAPTER=$ROOT/runs/route_b_batch8
OUTPUT=$ROOT/runs/route_b_stage2_batch8
LOGS=$ROOT/logs
PIPELINE_LOG=$LOGS/route_b_stage2_batch8.pipeline.log

mkdir -p "$LOGS"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
set_stage() {
  printf '%s\n' "$1" > "$LOGS/route_b_stage2_batch8.status.txt"
  echo "STAGE: $1 at $(date --iso-8601=seconds)"
}
finish() {
  rc=$?
  printf '%s\n' "$rc" > "$LOGS/route_b_stage2_batch8.exit_code"
  if [[ "$rc" -eq 0 ]]; then
    printf 'COMPLETE\n' > "$LOGS/route_b_stage2_batch8.status.txt"
  else
    printf 'FAILED (exit=%s)\n' "$rc" > "$LOGS/route_b_stage2_batch8.status.txt"
  fi
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$LOGS/route_b_stage2_batch8.status.txt"
}
trap finish EXIT
trap 'exit 143' TERM INT

set_stage "preflight"
[[ -s "$ROOT/data/train.jsonl" ]]
[[ -s "$ROOT/data/grid_dev.jsonl" ]]
[[ -s "$PARENT_ADAPTER/adapter_model.safetensors" ]]
[[ -f "$BASE_CONFIG" ]]
[[ -f "$REPO/scripts/train_qwen3vl_lora.py" ]]
[[ -x /root/miniconda3/envs/rs-vlm/bin/python ]]
[[ ! -e "$OUTPUT" ]]
[[ ! -e "$CONFIG" ]]
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [[ ! "$GPU_USED" =~ ^[0-9]+$ ]] || (( GPU_USED > 4096 )); then
  echo "GPU not idle enough for sequential training: ${GPU_USED} MiB"
  exit 23
fi
PARENT_HASH=$(sha256sum "$PARENT_ADAPTER/adapter_model.safetensors" | cut -d ' ' -f1)
echo "parent_adapter=$PARENT_ADAPTER"
echo "parent_adapter_sha256=$PARENT_HASH"
cp "$BASE_CONFIG" "$CONFIG"

/root/miniconda3/envs/rs-vlm/bin/python - "$CONFIG" "$PARENT_ADAPTER" "$OUTPUT" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
parent, output = sys.argv[2:]
text = path.read_text(encoding="utf-8")
text = re.sub(
    r"(?m)^  initial_adapter_dir: .*route_b$",
    f"  initial_adapter_dir: {parent}",
    text,
    count=1,
)
text = re.sub(
    r"(?m)^  output_dir: .*runs/route_b_batch8$",
    f"  output_dir: {output}",
    text,
    count=1,
)
text = text.replace(
    "qwen3vl_2b_changechat_grid_route_b_batch8",
    "qwen3vl_2b_changechat_grid_route_b_stage2_batch8",
    1,
)
if f"  initial_adapter_dir: {parent}" not in text:
    raise SystemExit("failed to set Route B stage2 initial adapter")
if f"  output_dir: {output}" not in text:
    raise SystemExit("failed to set Route B stage2 output directory")
if "  per_device_train_batch_size: 8" not in text:
    raise SystemExit("unexpected train batch size")
if "  per_device_eval_batch_size: 8" not in text:
    raise SystemExit("unexpected eval batch size")
if "  gradient_accumulation_steps: 2" not in text:
    raise SystemExit("unexpected gradient accumulation")
if "  resume_from_checkpoint: null" not in text:
    raise SystemExit("stage2 must start a new optimizer/scheduler")
path.write_text(text, encoding="utf-8")
PY

/root/miniconda3/envs/rs-vlm/bin/python - "$ROOT/source_manifest.json" "$PARENT_HASH" <<'PY'
import datetime
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["route_b_stage2"] = {
    "authorized": True,
    "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "parent_adapter": "runs/route_b_batch8",
    "parent_adapter_sha256": sys.argv[2],
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 16,
    "epochs": 1,
}
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rs-vlm
export PYTHONPATH="$REPO/src:$REPO"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
cd "$REPO"

set_stage "dry_run"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$CONFIG" --dry-run > "$LOGS/route_b_stage2_batch8.dry_run.log" 2>&1; then
  tail -n 100 "$LOGS/route_b_stage2_batch8.dry_run.log"
  exit 21
fi

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [[ ! "$GPU_USED" =~ ^[0-9]+$ ]] || (( GPU_USED > 4096 )); then
  echo "GPU became occupied after dry-run: ${GPU_USED} MiB"
  exit 24
fi

set_stage "training_route_b_stage2_batch8"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$CONFIG" > "$LOGS/route_b_stage2_batch8.train.log" 2>&1; then
  tail -n 100 "$LOGS/route_b_stage2_batch8.train.log"
  exit 31
fi
echo "route_b_stage2_output=$OUTPUT"

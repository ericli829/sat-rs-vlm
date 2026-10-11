#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011
PARENT=/root/autodl-fs/data/sat-rs-vlm-training
REPO=$PARENT/planner_4b_clean_20261011/workspace_repacked
MODEL=/root/autodl-tmp/models/Qwen3-VL-2B-Instruct
LOGS=$ROOT/logs
A_HASH_EXPECTED=a4efe8b4e2e69b83e1d4d0cdf0bc9c0dba7c0881b758d16a3db4637790147b36
B_HASH_EXPECTED=73d8aea1fdce41ebc36f6d8ca9af1b6032fcaf3f7f32be590456c37a2af30d01

mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/pipeline_batch4.log") 2>&1
set_stage() {
  printf '%s\n' "$1" > "$LOGS/status_batch4.txt"
  echo "STAGE: $1 at $(date --iso-8601=seconds)"
}
finish() {
  rc=$?
  printf '%s\n' "$rc" > "$LOGS/pipeline_batch4.exit_code"
  if [[ "$rc" -eq 0 ]]; then
    printf 'COMPLETE\n' > "$LOGS/status_batch4.txt"
  else
    printf 'FAILED (exit=%s)\n' "$rc" > "$LOGS/status_batch4.txt"
  fi
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$LOGS/status_batch4.txt"
}
trap finish EXIT
trap 'exit 143' TERM INT

set_stage "checking_inputs"
[[ -d "$MODEL" ]]
[[ -f "$REPO/scripts/train_qwen3vl_lora.py" ]]
[[ -x /root/miniconda3/envs/rs-vlm/bin/python ]]
[[ -s "$ROOT/data/train.jsonl" ]]
[[ -s "$ROOT/data/grid_dev.jsonl" ]]
[[ -f "$ROOT/adapters/route_a/adapter_model.safetensors" ]]
[[ -f "$ROOT/adapters/route_b/adapter_model.safetensors" ]]
[[ -f "$ROOT/configs/config_route_a.yaml" ]]
[[ -f "$ROOT/configs/config_route_b.yaml" ]]
A_HASH=$(sha256sum "$ROOT/adapters/route_a/adapter_model.safetensors" | cut -d ' ' -f1)
B_HASH=$(sha256sum "$ROOT/adapters/route_b/adapter_model.safetensors" | cut -d ' ' -f1)
[[ "$A_HASH" == "$A_HASH_EXPECTED" ]]
[[ "$B_HASH" == "$B_HASH_EXPECTED" ]]
echo "route_a_adapter_sha256=$A_HASH"
echo "route_b_adapter_sha256=$B_HASH"

if [[ -e "$ROOT/runs/route_a_batch4" || -e "$ROOT/runs/route_b_batch4" ]]; then
  echo "Refusing to overwrite existing batch4 outputs."
  exit 12
fi
if [[ -d "$ROOT/runs/route_a" ]]; then
  if [[ -e "$ROOT/runs/route_a_batch1_interrupted_20261011" ]]; then
    echo "Interrupted batch1 archive already exists; refusing to overwrite."
    exit 13
  fi
  mv "$ROOT/runs/route_a" "$ROOT/runs/route_a_batch1_interrupted_20261011"
fi
mkdir -p "$ROOT/runs/route_a_batch4" "$ROOT/runs/route_b_batch4"

set_stage "updating_batch4_configs"
 /root/miniconda3/envs/rs-vlm/bin/python - "$ROOT/configs/config_route_a.yaml" "$ROOT/configs/config_route_b.yaml" "$ROOT/source_manifest.json" <<'PY'
import datetime
import json
import re
import sys
from pathlib import Path

for route, path_string in zip(("route_a", "route_b"), sys.argv[1:3]):
    path = Path(path_string)
    text = path.read_text(encoding="utf-8")
    replacements = {
        f"  output_dir: /root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011/runs/{route}\n":
            f"  output_dir: /root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011/runs/{route}_batch4\n",
        "  per_device_train_batch_size: 1\n": "  per_device_train_batch_size: 4\n",
        "  per_device_eval_batch_size: 1\n": "  per_device_eval_batch_size: 4\n",
        "  gradient_accumulation_steps: 16\n": "  gradient_accumulation_steps: 4\n",
        f"  experiment_name: qwen3vl_2b_changechat_grid_{route}\n":
            f"  experiment_name: qwen3vl_2b_changechat_grid_{route}_batch4\n",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise SystemExit(f"{path}: expected exactly one occurrence of {old.strip()!r}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

manifest_path = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["purpose"] = "2B ChangeChat 3x3 grid-first A/B training, restarted at train/eval batch 4"
manifest["batch4_restart"] = {
    "authorized": True,
    "authorized_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16,
    "previous_route_a_steps": 72,
    "previous_route_a_checkpoint_saved": False,
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

set_stage "dry_run_route_a_batch4"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_a.yaml" --dry-run > "$LOGS/dry_run_route_a_batch4.log" 2>&1; then
  tail -n 100 "$LOGS/dry_run_route_a_batch4.log"
  exit 21
fi
set_stage "dry_run_route_b_batch4"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_b.yaml" --dry-run > "$LOGS/dry_run_route_b_batch4.log" 2>&1; then
  tail -n 100 "$LOGS/dry_run_route_b_batch4.log"
  exit 22
fi

assert_gpu_available() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 4096 )); then
    echo "GPU occupied: memory_used_mib=$used; refusing to overlap another job."
    exit 23
  fi
}
assert_gpu_available

set_stage "training_route_a_batch4"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_a.yaml" > "$LOGS/train_route_a_batch4.log" 2>&1; then
  tail -n 100 "$LOGS/train_route_a_batch4.log"
  exit 31
fi
assert_gpu_available
set_stage "training_route_b_batch4"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_b.yaml" > "$LOGS/train_route_b_batch4.log" 2>&1; then
  tail -n 100 "$LOGS/train_route_b_batch4.log"
  exit 32
fi

set_stage "packaging_results_batch4"
mkdir -p "$ROOT/results"
tar -czf "$ROOT/results/changechat_grid_ab_20261011_batch4_results.tar.gz" \
  -C "$ROOT" runs logs data configs adapters source_manifest.json
echo "results_bundle=$ROOT/results/changechat_grid_ab_20261011_batch4_results.tar.gz"

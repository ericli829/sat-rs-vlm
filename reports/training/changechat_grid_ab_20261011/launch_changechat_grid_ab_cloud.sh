#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-fs/data/sat-rs-vlm-training/changechat_grid_ab_20261011
PARENT=/root/autodl-fs/data/sat-rs-vlm-training
BUNDLE=$PARENT/changechat_grid_ab_20261011.bundle.tar.gz
REPO=$PARENT/planner_4b_clean_20261011/workspace_repacked
MODEL=/root/autodl-tmp/models/Qwen3-VL-2B-Instruct
VRS=/root/autodl-tmp/sat-rs-vlm/data/processed/qwen3vl_train.jsonl
ROUTE_B=/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter
LOGS=$ROOT/logs
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/pipeline.log") 2>&1
set_stage() { printf '%s\n' "$1" > "$LOGS/status.txt"; echo "STAGE: $1"; }
finish() {
  rc=$?
  printf '%s\n' "$rc" > "$LOGS/pipeline.exit_code"
  if [[ "$rc" -eq 0 ]]; then
    printf 'COMPLETE\n' > "$LOGS/status.txt"
  else
    printf 'FAILED (exit=%s)\n' "$rc" > "$LOGS/status.txt"
  fi
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$LOGS/status.txt"
}
trap finish EXIT
set_stage "checking_inputs"
[[ -s "$BUNDLE" ]]
[[ -d "$MODEL" ]]
[[ -f "$VRS" ]]
[[ -f "$ROUTE_B/adapter_model.safetensors" ]]
[[ -f "$REPO/scripts/train_qwen3vl_lora.py" ]]
[[ -x /root/miniconda3/envs/rs-vlm/bin/python ]]
if [[ -e "$ROOT/configs" || -e "$ROOT/data/train.jsonl" || -e "$ROOT/runs/route_a" ]]; then
  echo "Refusing to overwrite existing prepared data or run outputs."
  exit 12
fi
set_stage "extracting_bundle"
tar -xzf "$BUNDLE" -C "$PARENT"
[[ -f "$ROOT/configs/config_route_a.yaml" ]]
[[ -f "$ROOT/configs/config_route_b.yaml" ]]
mkdir -p "$ROOT/adapters/route_b"
cp -a "$ROUTE_B/." "$ROOT/adapters/route_b/"
set_stage "verifying_adapters"
A_HASH=$(sha256sum "$ROOT/adapters/route_a/adapter_model.safetensors" | cut -d ' ' -f1)
B_HASH=$(sha256sum "$ROOT/adapters/route_b/adapter_model.safetensors" | cut -d ' ' -f1)
[[ "$A_HASH" == "a4efe8b4e2e69b83e1d4d0cdf0bc9c0dba7c0881b758d16a3db4637790147b36" ]]
[[ "$B_HASH" == "73d8aea1fdce41ebc36f6d8ca9af1b6032fcaf3f7f32be590456c37a2af30d01" ]]
echo "route_a_adapter_sha256=$A_HASH"
echo "route_b_adapter_sha256=$B_HASH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rs-vlm
export PYTHONPATH="$REPO/src:$REPO"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
python - "$ROOT/source_manifest.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
data["training_launch_authorized"] = True
data["training_launch_authorized_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
set_stage "preparing_data"
python -u "$ROOT/scripts/prepare_changechat_grid_ab.py" \
  --changechat-train "$ROOT/inputs/changechat_105k_train.json.gz" \
  --vrs-train "$VRS" \
  --image-root /root/autodl-tmp/datasets \
  --output-dir "$ROOT" > "$LOGS/prepare.log" 2>&1
[[ -s "$ROOT/data/train.jsonl" ]]
[[ -s "$ROOT/data/grid_dev.jsonl" ]]
echo "train_rows=$(wc -l < "$ROOT/data/train.jsonl") grid_dev_rows=$(wc -l < "$ROOT/data/grid_dev.jsonl")"
cd "$REPO"
set_stage "dry_run_route_a"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_a.yaml" --dry-run > "$LOGS/dry_run_route_a.log" 2>&1; then
  tail -n 80 "$LOGS/dry_run_route_a.log"
  exit 21
fi
set_stage "dry_run_route_b"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_b.yaml" --dry-run > "$LOGS/dry_run_route_b.log" 2>&1; then
  tail -n 80 "$LOGS/dry_run_route_b.log"
  exit 22
fi
assert_gpu_available() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 4096 )); then
    echo "GPU is occupied: memory_used_mib=$used; refusing to overlap another job."
    exit 23
  fi
}
assert_gpu_available
set_stage "training_route_a"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_a.yaml" > "$LOGS/train_route_a.log" 2>&1; then
  tail -n 100 "$LOGS/train_route_a.log"
  exit 31
fi
assert_gpu_available
set_stage "training_route_b"
if ! python -u "$REPO/scripts/train_qwen3vl_lora.py" --config "$ROOT/configs/config_route_b.yaml" > "$LOGS/train_route_b.log" 2>&1; then
  tail -n 100 "$LOGS/train_route_b.log"
  exit 32
fi
set_stage "packaging_results"
mkdir -p "$ROOT/results"
tar -czf "$ROOT/results/changechat_grid_ab_20261011_results.tar.gz" \
  -C "$ROOT" runs logs data configs adapters source_manifest.json
echo "results_bundle=$ROOT/results/changechat_grid_ab_20261011_results.tar.gz"
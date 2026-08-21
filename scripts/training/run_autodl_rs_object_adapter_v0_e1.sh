#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="rs-vlm"
PROJECT_ROOT_DEFAULT="/root/autodl-tmp/sat-rs-vlm"
CONFIG="configs/experiments/rs_object_adapter_v0_4090.yaml"
R1_CHECKPOINT_DIR=""
DATA_ROOT_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
RUN_ROOT=""
MAX_TRAIN_GROUPS=""
MAX_STEPS=""
MAX_VAL_GROUPS=""
RESUME_OBJECT_ADAPTER_CHECKPOINT=""
SKIP_DATA_BUILD=0
SKIP_E1=0
DRY_RUN=0
SHUTDOWN_AFTER_RUN=0
TEST_SHUTDOWN=0
PROJECT_ROOT=""
DATA_ROOT=""
OUTPUT_ROOT=""
PYTHON_BIN=""
CURRENT_STAGE="startup"
LOG_FILE=""
REPORT_DIR=""

usage() {
  cat <<'EOF'
Usage: run_autodl_rs_object_adapter_v0_e1.sh [options]

Required:
  --r1-checkpoint-dir PATH  Existing Qwen3-VL-4B R1 checkpoint directory

Options:
  --config PATH              Object Adapter config
  --data-root PATH           Dataset image root and DATA_ROOT
  --output-root PATH         AutoDL output root
  --run-root PATH            Run output directory
  --env-name NAME            AutoDL environment (default: rs-vlm)
  --max-train-groups N       Limit groups for a smoke run
  --max-steps N              Limit optimizer steps for a smoke run
  --max-val-groups N         Limit internal validation groups for a smoke run
  --resume-object-adapter-checkpoint PATH
                              Resume from a completed Object Adapter checkpoint_epoch_N
  --skip-data-build          Reuse an existing audited dataset
  --skip-e1                  Save the trained checkpoint without running E1
  --dry-run                  Audit/load only; skip training, E1 and shutdown
  --shutdown-after-run       Shutdown after success or failure
  --test-shutdown             Use RS_OBJECT_ADAPTER_SHUTDOWN_MOCK_FILE
  --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --r1-checkpoint-dir) R1_CHECKPOINT_DIR="$2"; shift 2 ;;
    --r1-checkpoint-dir=*) R1_CHECKPOINT_DIR="${1#*=}"; shift ;;
    --data-root) DATA_ROOT_OVERRIDE="$2"; shift 2 ;;
    --data-root=*) DATA_ROOT_OVERRIDE="${1#*=}"; shift ;;
    --output-root) OUTPUT_ROOT_OVERRIDE="$2"; shift 2 ;;
    --output-root=*) OUTPUT_ROOT_OVERRIDE="${1#*=}"; shift ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-root=*) RUN_ROOT="${1#*=}"; shift ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --env-name=*) ENV_NAME="${1#*=}"; shift ;;
    --max-train-groups) MAX_TRAIN_GROUPS="$2"; shift 2 ;;
    --max-train-groups=*) MAX_TRAIN_GROUPS="${1#*=}"; shift ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-steps=*) MAX_STEPS="${1#*=}"; shift ;;
    --max-val-groups) MAX_VAL_GROUPS="$2"; shift 2 ;;
    --max-val-groups=*) MAX_VAL_GROUPS="${1#*=}"; shift ;;
    --resume-object-adapter-checkpoint) RESUME_OBJECT_ADAPTER_CHECKPOINT="$2"; shift 2 ;;
    --resume-object-adapter-checkpoint=*) RESUME_OBJECT_ADAPTER_CHECKPOINT="${1#*=}"; shift ;;
    --skip-data-build) SKIP_DATA_BUILD=1; shift ;;
    --skip-e1) SKIP_E1=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --shutdown-after-run|--shutdown) SHUTDOWN_AFTER_RUN=1; shift ;;
    --test-shutdown) TEST_SHUTDOWN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

shutdown_host() {
  if [[ -n "${RS_OBJECT_ADAPTER_SHUTDOWN_MOCK_FILE:-}" ]]; then
    printf 'shutdown_requested\n' > "$RS_OBJECT_ADAPTER_SHUTDOWN_MOCK_FILE"
    return 0
  fi
  sync || true
  local shutdown_python="${AUTODL_PYTHON:-$(command -v python3 || command -v python || true)}"
  if [[ -x /usr/bin/shutdown ]] && [[ -n "$shutdown_python" ]] && \
    "$shutdown_python" -c 'import os; raise SystemExit(os.system("/usr/bin/shutdown"))'; then
    return 0
  fi
  echo '[WARN] Python os.system("/usr/bin/shutdown") was refused; trying fallbacks.' >&2
  if [[ -x /usr/bin/shutdown ]] && /usr/bin/shutdown -h now; then return 0; fi
  if [[ -x /usr/sbin/shutdown ]] && /usr/sbin/shutdown -h now; then return 0; fi
  if [[ -x /sbin/shutdown ]] && /sbin/shutdown -h now; then return 0; fi
  if command -v poweroff >/dev/null 2>&1 && poweroff -f; then return 0; fi
  if command -v sudo >/dev/null 2>&1 && sudo -n poweroff -f; then return 0; fi
  echo '[ERROR] All supported shutdown commands failed.' >&2
  return 1
}

if [[ "$TEST_SHUTDOWN" == "1" ]]; then
  [[ -n "${RS_OBJECT_ADAPTER_SHUTDOWN_MOCK_FILE:-}" ]] || {
    echo '--test-shutdown requires RS_OBJECT_ADAPTER_SHUTDOWN_MOCK_FILE' >&2
    exit 2
  }
  shutdown_host
  exit 0
fi

on_error() {
  local exit_code=$?
  local failed_command="${BASH_COMMAND:-unknown}"
  trap - ERR
  set +e
  if [[ -n "$REPORT_DIR" ]]; then
    mkdir -p "$REPORT_DIR" 2>/dev/null || true
    {
      printf 'success=false\n'
      printf 'stage=%s\n' "$CURRENT_STAGE"
      printf 'exit_code=%s\n' "$exit_code"
      printf 'failed_command=%s\n' "$failed_command"
      printf 'run_root=%s\n' "$RUN_ROOT"
      printf 'log=%s\n' "${LOG_FILE:-none}"
    } > "$REPORT_DIR/run_error.txt"
  fi
  sync || true
  if [[ "$SHUTDOWN_AFTER_RUN" == "1" && "$DRY_RUN" == "0" ]]; then
    echo '[ACTION] Run failed; attempting AutoDL shutdown.' >&2
    shutdown_host || true
  fi
  exit "$exit_code"
}
trap on_error ERR

[[ -f /root/autodl_env.sh ]] || {
  echo 'Missing /root/autodl_env.sh. Run AutoDL environment setup first.' >&2
  exit 1
}
source /root/autodl_env.sh
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${DATA_ROOT:-/root/autodl-tmp/datasets}}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-${OUTPUT_ROOT:-/root/autodl-tmp/outputs}}"

if [[ -f "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh" ]]; then
  source "$PROJECT_ROOT/scripts/environment/activate_autodl_python.sh"
  activate_autodl_python "$ENV_NAME"
fi
PYTHON_BIN="${AUTODL_PYTHON:-${PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"

[[ -d "$PROJECT_ROOT" ]] || { echo "Project directory is missing: $PROJECT_ROOT" >&2; exit 1; }
[[ -n "$PYTHON_BIN" ]] || { echo 'No usable Python interpreter found.' >&2; exit 1; }
[[ -n "$R1_CHECKPOINT_DIR" ]] || {
  echo '--r1-checkpoint-dir is required for a real run.' >&2
  exit 2
}

resolve_project_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then printf '%s\n' "$value"; else printf '%s/%s\n' "$PROJECT_ROOT" "$value"; fi
}

CONFIG_PATH="$(resolve_project_path "$CONFIG")"
R1_CHECKPOINT_DIR="$(resolve_project_path "$R1_CHECKPOINT_DIR")"
[[ -f "$CONFIG_PATH" ]] || { echo "Config is missing: $CONFIG_PATH" >&2; exit 1; }
[[ -d "$R1_CHECKPOINT_DIR" ]] || { echo "R1 checkpoint is missing: $R1_CHECKPOINT_DIR" >&2; exit 1; }
if [[ -n "$RESUME_OBJECT_ADAPTER_CHECKPOINT" ]]; then
  RESUME_OBJECT_ADAPTER_CHECKPOINT="$(resolve_project_path "$RESUME_OBJECT_ADAPTER_CHECKPOINT")"
  [[ -d "$RESUME_OBJECT_ADAPTER_CHECKPOINT" ]] || {
    echo "Object Adapter resume checkpoint is missing: $RESUME_OBJECT_ADAPTER_CHECKPOINT" >&2
    exit 1
  }
fi

RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/rs_object_adapter_v0_e1_$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUTPUT="$RUN_ROOT/training"
EVAL_OUTPUT="$RUN_ROOT/evaluation_e1"
REPORT_DIR="$RUN_ROOT/reports"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/object_adapter_v0_e1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export PROJECT_ROOT DATA_ROOT OUTPUT_ROOT
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[ENV] PROJECT_ROOT=$PROJECT_ROOT"
echo "[ENV] DATA_ROOT=$DATA_ROOT"
echo "[ENV] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[ENV] PYTHON_BIN=$PYTHON_BIN"
echo "[ENV] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[RUN] RUN_ROOT=$RUN_ROOT"
echo "[RUN] R1_CHECKPOINT_DIR=$R1_CHECKPOINT_DIR"
if [[ -n "$RESUME_OBJECT_ADAPTER_CHECKPOINT" ]]; then
  echo "[RUN] RESUME_OBJECT_ADAPTER_CHECKPOINT=$RESUME_OBJECT_ADAPTER_CHECKPOINT"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  SHUTDOWN_AFTER_RUN=0
  echo '[SAFE] Dry-run selected; E1 and shutdown are disabled.'
fi

CURRENT_STAGE="E1 asset preflight"
"$PYTHON_BIN" - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
data = dict(config.get("data", {}))
manifest_value = data.get("evaluation_tier_manifest")
if not manifest_value:
    raise SystemExit("data.evaluation_tier_manifest is required")
manifest_path = Path(str(manifest_value)).expanduser()
if not manifest_path.is_absolute():
    manifest_path = project_root / manifest_path
if not manifest_path.is_file():
    raise SystemExit(f"E1 tier manifest is missing: {manifest_path}")
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
record = dict(payload.get("tiers", {}).get("E1", {}))
if not record:
    raise SystemExit(f"E1 record is missing from: {manifest_path}")
tier_value = Path(str(record.get("path", ""))).expanduser()
candidates = [tier_value, project_root / tier_value, manifest_path.parent / tier_value]
tier_path = next((path for path in candidates if path.is_file()), None)
if tier_path is None:
    raise SystemExit(f"E1 JSONL is missing; checked: {candidates}")
expected = record.get("sha256")
if expected:
    digest = hashlib.sha256()
    with tier_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != str(expected):
        raise SystemExit(f"E1 SHA256 mismatch: expected={expected}, actual={digest.hexdigest()}")
print(f"E1 preflight passed: {tier_path} ({record.get('sample_count', 'unknown')} samples)")
PY

if [[ "$SKIP_DATA_BUILD" == "0" ]]; then
  CURRENT_STAGE="Object Adapter data audit and build"
  "$PYTHON_BIN" scripts/data/build_object_adapter_v0_data.py --config "$CONFIG_PATH"
else
  echo '[DATA] Reusing existing Object Adapter data assets.'
fi

DATA_MANIFEST="$PROJECT_ROOT/data/processed/rs_object_adapter_v0/manifest.json"
[[ -s "$DATA_MANIFEST" ]] || { echo "Object Adapter manifest is missing: $DATA_MANIFEST" >&2; exit 1; }

CURRENT_STAGE="Object Adapter v0 training"
TRAIN_ARGS=(scripts/training/train_object_adapter_v0.py --config "$CONFIG_PATH" --checkpoint-dir "$R1_CHECKPOINT_DIR" --output-dir "$TRAIN_OUTPUT")
if [[ -n "$MAX_TRAIN_GROUPS" ]]; then TRAIN_ARGS+=(--max-train-groups "$MAX_TRAIN_GROUPS"); fi
if [[ -n "$MAX_STEPS" ]]; then TRAIN_ARGS+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "$MAX_VAL_GROUPS" ]]; then TRAIN_ARGS+=(--max-val-groups "$MAX_VAL_GROUPS"); fi
if [[ -n "$RESUME_OBJECT_ADAPTER_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume-object-adapter-checkpoint "$RESUME_OBJECT_ADAPTER_CHECKPOINT")
fi
if [[ "$DRY_RUN" == "1" ]]; then TRAIN_ARGS+=(--dry-run); fi
"$PYTHON_BIN" "${TRAIN_ARGS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'success=dry_run\nrun_root=%s\nlog=%s\n' "$RUN_ROOT" "$LOG_FILE" > "$REPORT_DIR/completion.txt"
  echo '[DONE] Dry-run completed; no E1 evaluation or shutdown requested.'
  exit 0
fi

CURRENT_STAGE="latest adapter checkpoint resolution"
FINAL_CHECKPOINT="$(find "$TRAIN_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_epoch_*' -print | sort -V | tail -n 1)"
[[ -n "$FINAL_CHECKPOINT" ]] || { echo "No checkpoint_epoch_* directory under: $TRAIN_OUTPUT" >&2; exit 1; }
[[ -s "$FINAL_CHECKPOINT/adapter_manifest.json" ]] || { echo "Adapter manifest missing: $FINAL_CHECKPOINT" >&2; exit 1; }
[[ -s "$FINAL_CHECKPOINT/adapter_model.safetensors" ]] || { echo "Adapter weights missing: $FINAL_CHECKPOINT" >&2; exit 1; }

if [[ "$SKIP_E1" == "1" ]]; then
  printf 'success=training_only\nrun_root=%s\ncheckpoint=%s\nlog=%s\n' \
    "$RUN_ROOT" "$FINAL_CHECKPOINT" "$LOG_FILE" > "$REPORT_DIR/completion.txt"
  sync
  if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
    CURRENT_STAGE="shutdown"
    echo '[ACTION] Training completed with E1 skipped; shutting down AutoDL.'
    trap - ERR
    shutdown_host
  else
    echo '[DONE] Training completed; E1 and shutdown were not requested.'
  fi
  exit 0
fi

CURRENT_STAGE="E1 evaluation"
"$PYTHON_BIN" scripts/evaluation/evaluate_object_adapter_v0.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$FINAL_CHECKPOINT" \
  --r1-checkpoint-dir "$R1_CHECKPOINT_DIR" \
  --evaluation-tier E1 \
  --output-dir "$EVAL_OUTPUT"

[[ -s "$EVAL_OUTPUT/e1_metrics.json" ]] || { echo "E1 metrics missing: $EVAL_OUTPUT/e1_metrics.json" >&2; exit 1; }
[[ -s "$EVAL_OUTPUT/predictions.jsonl" ]] || { echo "E1 predictions missing: $EVAL_OUTPUT/predictions.jsonl" >&2; exit 1; }
[[ -s "$EVAL_OUTPUT/evaluation_metadata.json" ]] || { echo "E1 metadata missing: $EVAL_OUTPUT/evaluation_metadata.json" >&2; exit 1; }

CURRENT_STAGE="completion report"
"$PYTHON_BIN" - "$RUN_ROOT" "$FINAL_CHECKPOINT" "$EVAL_OUTPUT" "$LOG_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
run_root, checkpoint, evaluation, log_path = map(Path, sys.argv[1:])
payload = {
    "success": True,
    "experiment": "rs_object_adapter_v0",
    "evaluation_tier": "E1",
    "run_root": str(run_root),
    "final_adapter_checkpoint": str(checkpoint),
    "evaluation_dir": str(evaluation),
    "log": str(log_path),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
(run_root / "reports" / "run_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
sync

if [[ "$SHUTDOWN_AFTER_RUN" == "1" ]]; then
  CURRENT_STAGE="shutdown"
  echo '[ACTION] Training and E1 evaluation completed; shutting down AutoDL.'
  trap - ERR
  shutdown_host
else
  echo '[DONE] Training and E1 evaluation completed; shutdown was not requested.'
fi

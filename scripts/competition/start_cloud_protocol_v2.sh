#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -gt 1 ]]; then
    printf 'usage: %s [PORT]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT_OVERRIDE=${PROJECT_ROOT:-}
CONFIG_OVERRIDE=${COMPETITION_CONFIG:-}
declare -A invocation_values=()
for name in PYTHON PORT BIND_HOST RUN_DIR RUN_ROOT MODEL_API_KEY COMPETITION_IMAGE_PACKAGE \
    QWEN3VL_4B_MODEL_DIR QWEN3VL_4B_PLANNER_LORA QWEN3VL_4B_CHANGECHAT_LLM_ONLY_LORA \
    QWEN3VL_2B_MODEL_DIR QWEN3VL_2B_SEMANTIC_LORA GEORSCLIP_CHECKPOINT \
    LAE_DINO_SOURCE_ROOT LAE_DINO_CONFIG_LAE1M LAE_DINO_CHECKPOINT_LAE1M LAE_DINO_BERT_ROOT \
    LAE_DINO_PYTHON LAE_DINO_STDERR_LOG COUNTING_SYSTEM_SRC CUDA_VISIBLE_DEVICES \
    PYTHONPATH UVICORN_LOG_CONFIG HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
    if [[ -v $name ]]; then invocation_values[$name]=${!name}; fi
done
AUTODL_ENV_FILE=${AUTODL_ENV_FILE:-/root/autodl_env.sh}
if [[ -f $AUTODL_ENV_FILE ]]; then
    # Optional host environment file; keep API secrets outside the repository.
    # shellcheck disable=SC1091
    source "$AUTODL_ENV_FILE"
fi
for name in "${!invocation_values[@]}"; do
    printf -v "$name" '%s' "${invocation_values[$name]}"
done
PROJECT_ROOT=${PROJECT_ROOT_OVERRIDE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}

PYTHON=${PYTHON:-/root/miniconda3/envs/rs-vlm/bin/python3.12}
PORT=${1:-${PORT:-8000}}
BIND_HOST=${BIND_HOST:-0.0.0.0}
COMPETITION_CONFIG=${CONFIG_OVERRIDE:-$PROJECT_ROOT/configs/competition/protocol_v2.yaml}
UVICORN_LOG_CONFIG=${UVICORN_LOG_CONFIG:-$PROJECT_ROOT/configs/competition/uvicorn_logging.yaml}
COMPETITION_IMAGE_PACKAGE=${COMPETITION_IMAGE_PACKAGE:-/root/autodl-fs/dev100-images}
QWEN3VL_4B_MODEL_DIR=${QWEN3VL_4B_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-VL-4B-Instruct}
QWEN3VL_4B_PLANNER_LORA=${QWEN3VL_4B_PLANNER_LORA:-}
QWEN3VL_4B_CHANGECHAT_LLM_ONLY_LORA=${QWEN3VL_4B_CHANGECHAT_LLM_ONLY_LORA:-/root/autodl-fs/experiments/sat-rs-vlm_cloud_eval_20261012_125359/changechat_4b_continuation_retry3/runs/llm_only}
QWEN3VL_2B_MODEL_DIR=${QWEN3VL_2B_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-VL-2B-Instruct}
QWEN3VL_2B_SEMANTIC_LORA=${QWEN3VL_2B_SEMANTIC_LORA:-/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter}
GEORSCLIP_CHECKPOINT=${GEORSCLIP_CHECKPOINT:-/root/autodl-fs/models/GeoRSCLIP/GeoRSCLIP-ViT-B-32.pt}
LAE_DINO_SOURCE_ROOT=${LAE_DINO_SOURCE_ROOT:-/root/autodl-fs/rs_detectors/lae_dino/source/LAE-DINO}
LAE_DINO_CONFIG_LAE1M=${LAE_DINO_CONFIG_LAE1M:-$LAE_DINO_SOURCE_ROOT/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py}
LAE_DINO_CHECKPOINT_LAE1M=${LAE_DINO_CHECKPOINT_LAE1M:-/root/autodl-fs/rs_detectors/lae_dino/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth}
LAE_DINO_BERT_ROOT=${LAE_DINO_BERT_ROOT:-/root/autodl-fs/rs_detectors/lae_dino/weights/bert-base-uncased}
LAE_DINO_PYTHON=${LAE_DINO_PYTHON:-/root/miniconda3/envs/lae-dino-gpu/bin/python3}
COUNTING_SYSTEM_SRC=${COUNTING_SYSTEM_SRC:-$PROJECT_ROOT/04_counting_system_plan/src}

if [[ ! $PORT =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    printf 'invalid PORT: %s\n' "$PORT" >&2
    exit 2
fi
if [[ -z ${MODEL_API_KEY:-} ]]; then
    printf 'MODEL_API_KEY must be set in the environment or /root/autodl_env.sh\n' >&2
    exit 2
fi
if [[ -z $QWEN3VL_4B_PLANNER_LORA ]]; then
    printf 'QWEN3VL_4B_PLANNER_LORA must identify the selected Planner adapter\n' >&2
    exit 2
fi

require_path() {
    local label=$1 path=$2 kind=${3:-e}
    if [[ $kind == f && ! -f $path ]] || [[ $kind == d && ! -d $path ]] || [[ $kind == e && ! -e $path ]]; then
        printf 'missing required %s: %s\n' "$label" "$path" >&2
        exit 1
    fi
}

require_path python "$PYTHON" f
require_path protocol_config "$COMPETITION_CONFIG" f
require_path logging_config "$UVICORN_LOG_CONFIG" f
require_path image_package_manifest "$COMPETITION_IMAGE_PACKAGE/manifest.json" f
require_path counting_system_source "$COUNTING_SYSTEM_SRC/counting_system/__init__.py" f
require_path qwen4b_base "$QWEN3VL_4B_MODEL_DIR/config.json" f
require_path qwen2b_base "$QWEN3VL_2B_MODEL_DIR/config.json" f
require_path georsclip_checkpoint "$GEORSCLIP_CHECKPOINT" f
require_path lae_dino_source "$LAE_DINO_SOURCE_ROOT" d
require_path lae_dino_config "$LAE_DINO_CONFIG_LAE1M" f
require_path lae_dino_checkpoint "$LAE_DINO_CHECKPOINT_LAE1M" f
require_path lae_dino_bert "$LAE_DINO_BERT_ROOT" d
require_path lae_dino_python "$LAE_DINO_PYTHON" f

require_adapter() {
    local label=$1 path=$2
    require_path "${label}_adapter_config" "$path/adapter_config.json" f
    if [[ ! -s "$path/adapter_model.safetensors" && ! -s "$path/adapter_model.bin" ]]; then
        printf 'missing required %s adapter weights: %s\n' "$label" "$path" >&2
        exit 1
    fi
}
require_adapter planner "$QWEN3VL_4B_PLANNER_LORA"
require_adapter route_4b "$QWEN3VL_4B_CHANGECHAT_LLM_ONLY_LORA"
require_adapter semantic_2b "$QWEN3VL_2B_SEMANTIC_LORA"

export PROJECT_ROOT COMPETITION_CONFIG COMPETITION_IMAGE_PACKAGE MODEL_API_KEY
export QWEN3VL_4B_MODEL_DIR QWEN3VL_4B_PLANNER_LORA QWEN3VL_4B_CHANGECHAT_LLM_ONLY_LORA
export QWEN3VL_2B_MODEL_DIR QWEN3VL_2B_SEMANTIC_LORA GEORSCLIP_CHECKPOINT
export LAE_DINO_SOURCE_ROOT LAE_DINO_CONFIG_LAE1M LAE_DINO_CHECKPOINT_LAE1M LAE_DINO_BERT_ROOT LAE_DINO_PYTHON
export COUNTING_SYSTEM_SRC
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$COUNTING_SYSTEM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if ss -H -l "sport = :$PORT" 2>/dev/null | grep -q .; then
    printf 'port %s is already listening; leaving the existing process untouched\n' "$PORT" >&2
    exit 1
fi

RUN_ROOT=${RUN_ROOT:-/root/autodl-fs/experiments}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/sat-rs-vlm_protocol_v2_$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -e $RUN_DIR ]]; then
    printf 'run directory already exists; refusing to overwrite: %s\n' "$RUN_DIR" >&2
    exit 1
fi
install -d -m 700 "$RUN_DIR"
export LAE_DINO_STDERR_LOG=${LAE_DINO_STDERR_LOG:-$RUN_DIR/lae-dino.stderr.log}

stdout_log=$RUN_DIR/server.stdout.log
stderr_log=$RUN_DIR/server.stderr.log
cd -- "$PROJECT_ROOT"
nohup "$PYTHON" -u -m uvicorn sat_rs_vlm.competition.protocol_v2.server:create_app \
    --factory --host "$BIND_HOST" --port "$PORT" --workers 1 --no-access-log \
    --log-config "$UVICORN_LOG_CONFIG" \
    >"$stdout_log" 2>"$stderr_log" < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN_DIR/server.pid"
printf '{"pid":%s,"host":"%s","port":%s,"config":"%s","started_at":"%s"}\n' \
    "$pid" "$BIND_HOST" "$PORT" "$COMPETITION_CONFIG" "$(date -u +%FT%TZ)" > "$RUN_DIR/state.json"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    printf 'service process exited during launch; see %s\n' "$stderr_log" >&2
    tail -80 "$stderr_log" >&2 || true
    exit 1
fi

printf 'STARTED pid=%s\nRUN_DIR=%s\nENDPOINT=http://%s:%s\nSTDOUT=%s\nSTDERR=%s\n' \
    "$pid" "$RUN_DIR" "$BIND_HOST" "$PORT" "$stdout_log" "$stderr_log"

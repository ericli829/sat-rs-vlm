#!/usr/bin/env bash
set -euo pipefail

# LAE-DINO is always installed into a separate Conda environment. The main
# rs-vlm interpreter is used only as an optional clone source and is never
# modified by this script.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ENV_NAME="${LAE_DINO_ENV_NAME:-rs-vlm-lae}"
BASE_ENV="${LAE_DINO_BASE_ENV:-${CONDA_DEFAULT_ENV:-}}"
SOURCE_ROOT="${LAE_DINO_SOURCE_ROOT:-}"
REQUIREMENTS="${LAE_DINO_REQUIREMENTS:-}"
CONFIG="${LAE_DINO_CONFIG:-}"
CHECKPOINT="${LAE_DINO_CHECKPOINT:-}"
BERT_ROOT="${LAE_DINO_BERT_ROOT:-}"
ENV_SCRIPT="${AUTODL_ENV_SCRIPT:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: environments/lae_dino/install.sh [options]
  --env-name NAME          Isolated Conda env (default: rs-vlm-lae)
  --base-env NAME          Existing Conda env cloned only when creating LAE env
  --source-root PATH       Existing LAE-DINO checkout
  --requirements PATH      Source-provided pip requirements file
  --config PATH            Exact source config matching the checkpoint
  --checkpoint PATH        Existing local detector checkpoint
  --bert-root PATH         Existing local bert-base-uncased directory
  --env-script PATH        Shell file receiving LAE_DINO_* exports
  --dry-run                Print the isolated orchestration without changing state
  --help, -h               Show this help

The script never downloads models or checkpoints. Package installation follows
the supplied LAE-DINO checkout's requirements and MMDetection requirement files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --base-env) BASE_ENV="$2"; shift 2 ;;
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --requirements) REQUIREMENTS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --bert-root) BERT_ROOT="$2"; shift 2 ;;
    --env-script) ENV_SCRIPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REQUIREMENTS" && -n "$SOURCE_ROOT" ]]; then
  REQUIREMENTS="$SOURCE_ROOT/mmdetection_lae/requirements.txt"
fi
if [[ -n "$BASE_ENV" && "$ENV_NAME" == "$BASE_ENV" ]]; then
  echo "LAE env must differ from its base env; refusing to modify $BASE_ENV." >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<EOF
[dry-run] LAE environment: $ENV_NAME
[dry-run] clone source: ${BASE_ENV:-<required for new environment>}
[dry-run] source root: ${SOURCE_ROOT:-<set LAE_DINO_SOURCE_ROOT>}
[dry-run] requirements: ${REQUIREMENTS:-<source-provided requirements>}
[dry-run] config: ${CONFIG:-<set LAE_DINO_CONFIG>}
[dry-run] checkpoint: ${CHECKPOINT:-<set LAE_DINO_CHECKPOINT>}
[dry-run] BERT root: ${BERT_ROOT:-<set LAE_DINO_BERT_ROOT>}
[dry-run] environment script: ${ENV_SCRIPT:-<not requested>}
[dry-run] create/reuse isolated Conda env, install source requirements,
[dry-run] install source MMDetection requirements/editable package, run check_environment.py,
[dry-run] then persist LAE_DINO_PYTHON/SOURCE_ROOT/CONFIG/CHECKPOINT/BERT_ROOT.
EOF
  exit 0
fi

for value_name in SOURCE_ROOT REQUIREMENTS CONFIG CHECKPOINT BERT_ROOT; do
  value="${!value_name}"
  if [[ -z "$value" ]]; then
    echo "$value_name is required; use the matching --lae option or environment variable." >&2
    exit 2
  fi
done
[[ -d "$SOURCE_ROOT" ]] || { echo "LAE-DINO source root does not exist: $SOURCE_ROOT" >&2; exit 2; }
[[ -f "$REQUIREMENTS" ]] || { echo "LAE-DINO requirements do not exist: $REQUIREMENTS" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "LAE-DINO config does not exist: $CONFIG" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "LAE-DINO checkpoint does not exist: $CHECKPOINT" >&2; exit 2; }
[[ -d "$BERT_ROOT" ]] || { echo "LAE-DINO BERT root does not exist: $BERT_ROOT" >&2; exit 2; }
MMDET_ROOT="$SOURCE_ROOT/mmdetection_lae"
[[ -f "$MMDET_ROOT/setup.py" ]] || {
  echo "LAE-DINO source is missing mmdetection_lae/setup.py: $SOURCE_ROOT" >&2
  exit 2
}

command -v conda >/dev/null 2>&1 || { echo "conda is required." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  [[ -n "$BASE_ENV" ]] || {
    echo "--base-env is required when creating the isolated LAE environment." >&2
    exit 2
  }
  conda env list | awk '{print $1}' | grep -Fxq "$BASE_ENV" || {
    echo "LAE clone source Conda environment does not exist: $BASE_ENV" >&2
    exit 2
  }
  conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
fi

LAE_PYTHON=$(conda run --name "$ENV_NAME" python -c 'import sys; print(sys.executable)')
[[ -n "$LAE_PYTHON" ]] || { echo "Unable to resolve LAE environment Python." >&2; exit 1; }

conda run --name "$ENV_NAME" python -m pip install -r "$REQUIREMENTS"
MIM_REQUIREMENTS="$MMDET_ROOT/requirements/mminstall.txt"
if [[ -f "$MIM_REQUIREMENTS" ]]; then
  # These commands mirror the source checkout's installation guide. Versions
  # remain controlled by its mminstall.txt rather than by sat-rs-vlm.
  conda run --name "$ENV_NAME" python -m pip install openmim
  conda run --name "$ENV_NAME" mim install -r "$MIM_REQUIREMENTS"
fi
MULTIMODAL_REQUIREMENTS="$MMDET_ROOT/requirements/multimodal.txt"
if [[ -f "$MULTIMODAL_REQUIREMENTS" ]]; then
  conda run --name "$ENV_NAME" python -m pip install -r "$MULTIMODAL_REQUIREMENTS"
fi
conda run --name "$ENV_NAME" python -m pip install -v -e "$MMDET_ROOT"

conda run --name "$ENV_NAME" python "$SCRIPT_DIR/check_environment.py" \
  --source-root "$SOURCE_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --bert-root "$BERT_ROOT"

upsert_export() {
  local file="$1"
  local key="$2"
  local value="$3"
  local temporary
  temporary=$(mktemp "${file}.tmp.XXXXXX")
  if [[ -f "$file" ]]; then
    grep -v "^export ${key}=" "$file" > "$temporary" || true
  fi
  printf 'export %s=%q\n' "$key" "$value" >> "$temporary"
  mv "$temporary" "$file"
}

if [[ -n "$ENV_SCRIPT" ]]; then
  mkdir -p "$(dirname "$ENV_SCRIPT")"
  touch "$ENV_SCRIPT"
  upsert_export "$ENV_SCRIPT" LAE_DINO_PYTHON "$LAE_PYTHON"
  upsert_export "$ENV_SCRIPT" LAE_DINO_SOURCE_ROOT "$SOURCE_ROOT"
  upsert_export "$ENV_SCRIPT" LAE_DINO_CONFIG "$CONFIG"
  upsert_export "$ENV_SCRIPT" LAE_DINO_CHECKPOINT "$CHECKPOINT"
  upsert_export "$ENV_SCRIPT" LAE_DINO_BERT_ROOT "$BERT_ROOT"
  # Current production configs retain the checkpoint-family aliases.
  upsert_export "$ENV_SCRIPT" LAE_DINO_CONFIG_LAE1M "$CONFIG"
  upsert_export "$ENV_SCRIPT" LAE_DINO_CHECKPOINT_LAE1M "$CHECKPOINT"
fi

echo "LAE-DINO isolated environment ready: $ENV_NAME ($LAE_PYTHON)"

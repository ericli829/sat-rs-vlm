#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rs-vlm"
CLONE_CURRENT=0
INSTALL_DEV=0
INSTALL_MODEL=0
INSTALL_RETRIEVER=0
INSTALL_LAE=0
INSTALL_QLORA=0
DRY_RUN=0
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/sat-rs-vlm}"
AUTODL_ENV_SCRIPT="${AUTODL_ENV_SCRIPT:-/root/autodl_env.sh}"
LAE_ENV_NAME="${LAE_DINO_ENV_NAME:-rs-vlm-lae}"
LAE_SOURCE_ROOT="${LAE_DINO_SOURCE_ROOT:-}"
LAE_REQUIREMENTS="${LAE_DINO_REQUIREMENTS:-}"
LAE_CONFIG="${LAE_DINO_CONFIG:-}"
LAE_CHECKPOINT="${LAE_DINO_CHECKPOINT:-}"
LAE_BERT_ROOT="${LAE_DINO_BERT_ROOT:-}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage: setup_autodl.sh [options]
  --env-name NAME       Conda environment name (default: rs-vlm)
  --clone-current       Clone the currently active Conda environment
  --install-dev         Install development dependencies
  --install-model       Install model libraries without replacing torch
  --install-retriever   Install the pyproject retriever extra without changing torch
  --install-lae         Create/reuse an isolated LAE-DINO Conda environment
  --install-qlora       Install optional bitsandbytes support for QLoRA/bnb INT8
  --project-root PATH   Project checkout path
  --env-script PATH     Generated shell environment (default: /root/autodl_env.sh)
  --lae-env-name NAME   Isolated LAE Conda environment (default: rs-vlm-lae)
  --lae-source-root P   Existing LAE-DINO checkout
  --lae-requirements P  Source-provided LAE requirements file
  --lae-config PATH     Exact LAE config matching the checkpoint
  --lae-checkpoint P    Existing local LAE checkpoint
  --lae-bert-root PATH  Existing local bert-base-uncased directory
  --dry-run             Print planned actions without creating or installing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --clone-current) CLONE_CURRENT=1; shift ;;
    --install-dev) INSTALL_DEV=1; shift ;;
    --install-model) INSTALL_MODEL=1; shift ;;
    --install-retriever) INSTALL_RETRIEVER=1; shift ;;
    --install-lae) INSTALL_LAE=1; shift ;;
    --install-qlora) INSTALL_QLORA=1; INSTALL_MODEL=1; shift ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --env-script) AUTODL_ENV_SCRIPT="$2"; shift 2 ;;
    --lae-env-name) LAE_ENV_NAME="$2"; shift 2 ;;
    --lae-source-root) LAE_SOURCE_ROOT="$2"; shift 2 ;;
    --lae-requirements) LAE_REQUIREMENTS="$2"; shift 2 ;;
    --lae-config) LAE_CONFIG="$2"; shift 2 ;;
    --lae-checkpoint) LAE_CHECKPOINT="$2"; shift 2 ;;
    --lae-bert-root) LAE_BERT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

LAE_ARGS=(
  --env-name "$LAE_ENV_NAME"
  --base-env "$ENV_NAME"
  --env-script "$AUTODL_ENV_SCRIPT"
)
[[ -n "$LAE_SOURCE_ROOT" ]] && LAE_ARGS+=(--source-root "$LAE_SOURCE_ROOT")
[[ -n "$LAE_REQUIREMENTS" ]] && LAE_ARGS+=(--requirements "$LAE_REQUIREMENTS")
[[ -n "$LAE_CONFIG" ]] && LAE_ARGS+=(--config "$LAE_CONFIG")
[[ -n "$LAE_CHECKPOINT" ]] && LAE_ARGS+=(--checkpoint "$LAE_CHECKPOINT")
[[ -n "$LAE_BERT_ROOT" ]] && LAE_ARGS+=(--bert-root "$LAE_BERT_ROOT")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] main Conda environment: $ENV_NAME"
  echo "[dry-run] project root: $PROJECT_ROOT"
  echo "[dry-run] install dev/model/retriever/qlora: $INSTALL_DEV/$INSTALL_MODEL/$INSTALL_RETRIEVER/$INSTALL_QLORA"
  echo "[dry-run] main project and selected requirements would be installed without force-reinstall."
  if [[ "$INSTALL_RETRIEVER" -eq 1 ]]; then
    echo "[dry-run] current torch version would be pinned as a pip constraint for .[retriever]."
  fi
  if [[ "$INSTALL_LAE" -eq 1 ]]; then
    bash "$REPOSITORY_ROOT/environments/lae_dino/install.sh" "${LAE_ARGS[@]}" --dry-run
  fi
  exit 0
fi

command -v conda >/dev/null 2>&1 || { echo "conda is required." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  SOURCE_ENV="${CONDA_DEFAULT_ENV:-base}"
  if [[ "$CLONE_CURRENT" -eq 1 || "$INSTALL_MODEL" -eq 1 || "$INSTALL_RETRIEVER" -eq 1 ]]; then
    conda run --name "$SOURCE_ENV" python -c 'import torch' >/dev/null 2>&1 || {
      echo "The source Conda environment has no CUDA-matched torch: $SOURCE_ENV" >&2
      echo "Install the provider-matched torch build before creating $ENV_NAME." >&2
      exit 2
    }
    conda create -y -n "$ENV_NAME" --clone "$SOURCE_ENV"
  else
    conda create -y -n "$ENV_NAME" python=3.11 pip
  fi
fi
conda activate "$ENV_NAME"
AUTODL_PYTHON="$(python -c 'import sys; print(sys.executable)')"
AUTODL_CONDA_PREFIX="${CONDA_PREFIX:-}"
[[ -n "$AUTODL_CONDA_PREFIX" && "$AUTODL_PYTHON" == "$AUTODL_CONDA_PREFIX/bin/python" ]] || {
  echo "Conda activation selected an unexpected Python: $AUTODL_PYTHON" >&2
  echo "Expected: $AUTODL_CONDA_PREFIX/bin/python" >&2
  exit 1
}

for dir in \
  /root/autodl-tmp/datasets /root/autodl-tmp/models /root/autodl-tmp/cache \
  /root/autodl-tmp/outputs /root/autodl-tmp/packages /root/autodl-tmp/temp \
  /root/autodl-fs/datasets /root/autodl-fs/models /root/autodl-fs/experiments \
  /root/tf-logs; do
  mkdir -p "$dir"
done

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

mkdir -p "$(dirname "$AUTODL_ENV_SCRIPT")"
touch "$AUTODL_ENV_SCRIPT"
upsert_export "$AUTODL_ENV_SCRIPT" PROJECT_ROOT "$PROJECT_ROOT"
upsert_export "$AUTODL_ENV_SCRIPT" DATA_ROOT "/root/autodl-tmp/datasets"
upsert_export "$AUTODL_ENV_SCRIPT" MODEL_ROOT "/root/autodl-tmp/models"
upsert_export "$AUTODL_ENV_SCRIPT" OUTPUT_ROOT "/root/autodl-tmp/outputs"
upsert_export "$AUTODL_ENV_SCRIPT" CACHE_ROOT "/root/autodl-tmp/cache"
upsert_export "$AUTODL_ENV_SCRIPT" TMPDIR "/root/autodl-tmp/temp"
upsert_export "$AUTODL_ENV_SCRIPT" TENSORBOARD_ROOT "/root/tf-logs"
upsert_export "$AUTODL_ENV_SCRIPT" BACKUP_ROOT "/root/autodl-fs/experiments"
upsert_export "$AUTODL_ENV_SCRIPT" HF_HOME "/root/autodl-tmp/cache/huggingface"
upsert_export "$AUTODL_ENV_SCRIPT" HF_HUB_CACHE "/root/autodl-tmp/cache/huggingface/hub"
upsert_export "$AUTODL_ENV_SCRIPT" TORCH_HOME "/root/autodl-tmp/cache/torch"
upsert_export "$AUTODL_ENV_SCRIPT" PIP_CACHE_DIR "/root/autodl-tmp/cache/pip"
upsert_export "$AUTODL_ENV_SCRIPT" AUTODL_ENV_NAME "$ENV_NAME"
upsert_export "$AUTODL_ENV_SCRIPT" AUTODL_CONDA_PREFIX "$AUTODL_CONDA_PREFIX"
upsert_export "$AUTODL_ENV_SCRIPT" AUTODL_PYTHON "$AUTODL_PYTHON"

printf -v SOURCE_LINE 'source %q' "$AUTODL_ENV_SCRIPT"
grep -Fqx "$SOURCE_LINE" /root/.bashrc || printf '\n%s\n' "$SOURCE_LINE" >> /root/.bashrc
source "$AUTODL_ENV_SCRIPT"

cd "$PROJECT_ROOT"
"$AUTODL_PYTHON" -m pip install -e . --no-deps
"$AUTODL_PYTHON" -m pip install -r environments/requirements-base.txt
"$AUTODL_PYTHON" -m pip install -r environments/requirements-cloud.txt
if [[ "$INSTALL_DEV" -eq 1 ]]; then
  "$AUTODL_PYTHON" -m pip install -r environments/requirements-dev.txt
fi
if [[ "$INSTALL_MODEL" -eq 1 ]]; then
  "$AUTODL_PYTHON" -m pip install -r environments/requirements-model.txt
fi
if [[ "$INSTALL_RETRIEVER" -eq 1 ]]; then
  TORCH_FINGERPRINT_BEFORE=$("$AUTODL_PYTHON" -c \
    'import torch; print(torch.__version__); print(torch.__file__)') || {
    echo "A CUDA-matched torch must already be installed before --install-retriever." >&2
    exit 2
  }
  TORCH_VERSION=${TORCH_FINGERPRINT_BEFORE%%$'\n'*}
  TORCH_CONSTRAINT=$(mktemp)
  trap 'rm -f "$TORCH_CONSTRAINT"' EXIT
  printf 'torch==%s\n' "$TORCH_VERSION" > "$TORCH_CONSTRAINT"
  # No --upgrade or --force-reinstall. The exact active torch build is also a
  # resolver constraint so open_clip_torch cannot replace the CUDA wheel.
  "$AUTODL_PYTHON" -m pip install --constraint "$TORCH_CONSTRAINT" -e ".[retriever]"
  TORCH_FINGERPRINT_AFTER=$("$AUTODL_PYTHON" -c \
    'import torch; print(torch.__version__); print(torch.__file__)')
  [[ "$TORCH_FINGERPRINT_AFTER" == "$TORCH_FINGERPRINT_BEFORE" ]] || {
    echo "Retriever installation changed the active torch build; refusing this environment." >&2
    exit 1
  }
  rm -f "$TORCH_CONSTRAINT"
  trap - EXIT
fi
if [[ "$INSTALL_QLORA" -eq 1 ]]; then
  "$AUTODL_PYTHON" -m pip install -r environments/requirements-qlora.txt
fi

CHECK_ARGS=()
[[ "$INSTALL_MODEL" -eq 1 ]] && CHECK_ARGS+=(--require-model)
[[ "$INSTALL_RETRIEVER" -eq 1 ]] && CHECK_ARGS+=(--require-retriever)
[[ "$INSTALL_QLORA" -eq 1 ]] && CHECK_ARGS+=(--require-bitsandbytes)
"$AUTODL_PYTHON" scripts/environment/check_environment.py "${CHECK_ARGS[@]}"
if [[ "$INSTALL_LAE" -eq 1 ]]; then
  bash "$REPOSITORY_ROOT/environments/lae_dino/install.sh" "${LAE_ARGS[@]}"
  source "$AUTODL_ENV_SCRIPT"
fi
echo "AutoDL environment ready: $ENV_NAME"

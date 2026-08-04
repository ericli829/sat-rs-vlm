#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rs-vlm"
CLONE_CURRENT=0
INSTALL_DEV=0
INSTALL_MODEL=0
INSTALL_QLORA=0
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/sat-rs-vlm}"

usage() {
  cat <<'EOF'
Usage: setup_autodl.sh [options]
  --env-name NAME       Conda environment name (default: rs-vlm)
  --clone-current       Clone the currently active Conda environment
  --install-dev         Install development dependencies
  --install-model       Install model libraries without replacing torch
  --install-qlora       Install optional bitsandbytes support for QLoRA/bnb INT8
  --project-root PATH   Project checkout path
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --clone-current) CLONE_CURRENT=1; shift ;;
    --install-dev) INSTALL_DEV=1; shift ;;
    --install-model) INSTALL_MODEL=1; shift ;;
    --install-qlora) INSTALL_QLORA=1; INSTALL_MODEL=1; shift ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v conda >/dev/null 2>&1 || { echo "conda is required." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$CLONE_CURRENT" -eq 1 ]]; then
    conda create -y -n "$ENV_NAME" --clone "${CONDA_DEFAULT_ENV:-base}"
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

cat > /root/autodl_env.sh <<EOF
export PROJECT_ROOT="$PROJECT_ROOT"
export DATA_ROOT="/root/autodl-tmp/datasets"
export MODEL_ROOT="/root/autodl-tmp/models"
export OUTPUT_ROOT="/root/autodl-tmp/outputs"
export CACHE_ROOT="/root/autodl-tmp/cache"
export TMPDIR="/root/autodl-tmp/temp"
export TENSORBOARD_ROOT="/root/tf-logs"
export BACKUP_ROOT="/root/autodl-fs/experiments"
export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HF_HUB_CACHE="/root/autodl-tmp/cache/huggingface/hub"
export TORCH_HOME="/root/autodl-tmp/cache/torch"
export PIP_CACHE_DIR="/root/autodl-tmp/cache/pip"
export AUTODL_ENV_NAME="$ENV_NAME"
export AUTODL_CONDA_PREFIX="$AUTODL_CONDA_PREFIX"
export AUTODL_PYTHON="$AUTODL_PYTHON"
EOF

SOURCE_LINE='source /root/autodl_env.sh'
grep -Fqx "$SOURCE_LINE" /root/.bashrc || printf '\n%s\n' "$SOURCE_LINE" >> /root/.bashrc
source /root/autodl_env.sh

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
if [[ "$INSTALL_QLORA" -eq 1 ]]; then
  "$AUTODL_PYTHON" -m pip install -r environments/requirements-qlora.txt
fi

CHECK_ARGS=()
[[ "$INSTALL_MODEL" -eq 1 ]] && CHECK_ARGS+=(--require-model)
[[ "$INSTALL_QLORA" -eq 1 ]] && CHECK_ARGS+=(--require-bitsandbytes)
"$AUTODL_PYTHON" scripts/environment/check_environment.py "${CHECK_ARGS[@]}"
echo "AutoDL environment ready: $ENV_NAME"

#!/usr/bin/env bash
set -euo pipefail

# Run this script inside a clean host/conda installation, never in rs-vlm.
ROOT=${VLM_FO1_ROOT:?Set VLM_FO1_ROOT to the official VLM-FO1 checkout}
ENV_NAME=${VLM_FO1_ENV_NAME:-vlm-fo1}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
conda env update --name "$ENV_NAME" --file "$SCRIPT_DIR/environment.yml" --prune
if [[ -f "$ROOT/requirements.txt" ]]; then
  conda run --name "$ENV_NAME" python -m pip install --no-cache-dir -r "$ROOT/requirements.txt"
else
  conda run --name "$ENV_NAME" python -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.lock.txt"
fi
conda run --name "$ENV_NAME" python -m pip install --no-cache-dir mmengine==0.8.2
if [[ ! -d "$ROOT/detect_tools/upn/ops" ]]; then
  echo "Missing official UPN ops directory: $ROOT/detect_tools/upn/ops" >&2
  exit 2
fi
conda run --name "$ENV_NAME" python -m pip install -v -e "$ROOT/detect_tools/upn/ops"
echo "Installed isolated environment $ENV_NAME; run check_environment.py next."

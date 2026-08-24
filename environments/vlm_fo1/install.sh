#!/usr/bin/env bash
set -euo pipefail

# Run this script inside a clean host/conda installation, never in rs-vlm.
ROOT=${VLM_FO1_ROOT:?Set VLM_FO1_ROOT to the official VLM-FO1 checkout}
ENV_NAME=${VLM_FO1_ENV_NAME:-vlm-fo1}
conda env update --name "$ENV_NAME" --file "$(dirname "$0")/environment.yml" --prune
conda run --name "$ENV_NAME" python -m pip install -v -e "$ROOT/detect_tools/upn/ops"
echo "Installed isolated environment $ENV_NAME; run check_environment.py next."

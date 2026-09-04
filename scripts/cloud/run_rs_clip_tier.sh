#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${RS_CLIP_CONFIG:-${ROOT}/configs/cloud/rs_clip_benchmark.yaml}"
TIER="${1:-smoke50}"

cd "${ROOT}"
python scripts/cloud/run_rs_clip_benchmark.py --config "${CONFIG}" --tier "${TIER}"

#!/usr/bin/env bash
set -euo pipefail

# AutoDL 单卡入口：参数原样传给 Python runner；请显式提供 --initial-adapter。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
exec "${PYTHON_BIN:-python}" scripts/training/run_autodl_qwen3vl_4b_vit_probe.py "$@"


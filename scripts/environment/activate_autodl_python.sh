#!/usr/bin/env bash

# Resolve and validate the Python interpreter used by AutoDL launch scripts.
activate_autodl_python() {
  local env_name="${1:-rs-vlm}"
  local project_root="${PROJECT_ROOT:-/root/autodl-tmp/sat-rs-vlm}"
  local configured_python="${AUTODL_PYTHON:-}"
  local selected_python=""

  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name" || {
      echo "Could not activate Conda environment: $env_name" >&2
      return 1
    }
    selected_python="$(command -v python)"
    if [[ -n "$configured_python" && "$configured_python" != "$selected_python" ]]; then
      echo "Ignoring stale AUTODL_PYTHON from the environment file: $configured_python" >&2
    fi
  elif [[ -n "$configured_python" ]]; then
    selected_python="$configured_python"
  else
    echo "conda is required when AUTODL_PYTHON is not configured." >&2
    return 1
  fi

  if [[ -n "${CONDA_PREFIX:-}" && "$selected_python" != "$CONDA_PREFIX/bin/python" ]]; then
    echo "Conda activation selected an unexpected Python: $selected_python" >&2
    echo "Expected: $CONDA_PREFIX/bin/python" >&2
    return 1
  fi

  if [[ "$selected_python" != /* ]]; then
    selected_python="$(command -v "$selected_python" 2>/dev/null || true)"
  fi
  [[ -n "$selected_python" && -x "$selected_python" ]] || {
    echo "Configured AutoDL Python is not executable: ${selected_python:-<empty>}" >&2
    return 1
  }

  export AUTODL_PYTHON="$selected_python"
  export PATH="$(dirname "$AUTODL_PYTHON"):$PATH"
  export PYTHONPATH="$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
  hash -r

  if ! "$AUTODL_PYTHON" -c 'import pydantic, yaml, sat_rs_vlm' >/dev/null 2>&1; then
    echo "AutoDL Python is missing core project dependencies: $AUTODL_PYTHON" >&2
    echo "Run: $AUTODL_PYTHON -m pip install -e $project_root" >&2
    return 1
  fi
  echo "AutoDL Python: $AUTODL_PYTHON"
}

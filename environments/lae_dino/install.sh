#!/usr/bin/env bash
set -euo pipefail

# LAE-DINO has incompatible MMDetection dependencies and must be installed in
# its own environment.  This script intentionally requires an explicit,
# source-provided requirements file; it never downloads a guessed dependency
# set and never modifies the shared rs-vlm environment.
: "${LAE_DINO_REQUIREMENTS:?Set LAE_DINO_REQUIREMENTS to the requirements file from the LAE-DINO checkout}"
: "${LAE_DINO_SOURCE_ROOT:?Set LAE_DINO_SOURCE_ROOT to the LAE-DINO checkout}"
: "${LAE_DINO_CONFIG:?Set LAE_DINO_CONFIG to the exact checkpoint-matching config file}"
: "${LAE_DINO_CHECKPOINT:?Set LAE_DINO_CHECKPOINT to the checkpoint to validate}"
: "${LAE_DINO_BERT_ROOT:?Set LAE_DINO_BERT_ROOT to the local bert-base-uncased directory}"

python -m pip install -r "$LAE_DINO_REQUIREMENTS"
check_args=(
  --source-root "$LAE_DINO_SOURCE_ROOT"
  --config "$LAE_DINO_CONFIG"
  --checkpoint "$LAE_DINO_CHECKPOINT"
  --bert-root "$LAE_DINO_BERT_ROOT"
)
python "$(dirname "$0")/check_environment.py" "${check_args[@]}"

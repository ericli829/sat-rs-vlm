#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR=""
BACKUP_ROOT=""
KEEP_CHECKPOINTS=2

usage() {
  cat <<'EOF'
Usage: backup_results.sh --experiment-dir PATH --backup-root PATH [options]
  --keep-checkpoints N   Number of newest checkpoint-* directories (default: 2)
Copies configs, reports, logs, metrics, predictions, processor and a limited
number of checkpoints. It does not copy every intermediate checkpoint.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-dir) EXPERIMENT_DIR="$2"; shift 2 ;;
    --backup-root) BACKUP_ROOT="$2"; shift 2 ;;
    --keep-checkpoints) KEEP_CHECKPOINTS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXPERIMENT_DIR" && -n "$BACKUP_ROOT" ]] || { usage >&2; exit 2; }
[[ -d "$EXPERIMENT_DIR" ]] || {
  echo "Experiment directory does not exist: $EXPERIMENT_DIR" >&2
  exit 1
}
command -v rsync >/dev/null 2>&1 || { echo "rsync is required." >&2; exit 1; }

DESTINATION="$BACKUP_ROOT/$(basename "$EXPERIMENT_DIR")"
mkdir -p "$DESTINATION"

for file in \
  config_resolved.yaml command.txt environment.json git_commit.txt \
  preflight.json train_report.json strategy_manifest.json trainer_state.json; do
  if [[ -f "$EXPERIMENT_DIR/$file" ]]; then
    rsync -a "$EXPERIMENT_DIR/$file" "$DESTINATION/"
  fi
done

for directory in logs metrics predictions processor artifacts evaluation; do
  if [[ -d "$EXPERIMENT_DIR/$directory" ]]; then
    rsync -a "$EXPERIMENT_DIR/$directory/" "$DESTINATION/$directory/"
  fi
done

CHECKPOINT_ROOT="$EXPERIMENT_DIR/checkpoints"
if [[ -d "$CHECKPOINT_ROOT" ]]; then
  mkdir -p "$DESTINATION/checkpoints"
  for file in \
    adapter_config.json adapter_model.bin adapter_model.safetensors \
    strategy_manifest.json trainer_state.json training_config.yaml; do
    if [[ -f "$CHECKPOINT_ROOT/$file" ]]; then
      rsync -a "$CHECKPOINT_ROOT/$file" "$DESTINATION/checkpoints/"
    fi
  done
  if [[ -d "$CHECKPOINT_ROOT/processor" ]]; then
    rsync -a "$CHECKPOINT_ROOT/processor/" "$DESTINATION/checkpoints/processor/"
  fi
  mapfile -t CHECKPOINTS < <(
    find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
      -printf '%f\n' | sort -t- -k2,2n | tail -n "$KEEP_CHECKPOINTS"
  )
  for checkpoint in "${CHECKPOINTS[@]}"; do
    rsync -a "$CHECKPOINT_ROOT/$checkpoint/" "$DESTINATION/checkpoints/$checkpoint/"
  done
fi

echo "Backup completed: $DESTINATION"

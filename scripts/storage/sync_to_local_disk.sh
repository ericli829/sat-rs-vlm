#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
DESTINATION=""
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: sync_to_local_disk.sh --source PATH --destination PATH [--overwrite]
Copies files with rsync partial-transfer support. Existing files are skipped
unless --overwrite is provided, then a checksum dry-run verifies consistency.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --destination) DESTINATION="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SOURCE" && -n "$DESTINATION" ]] || { usage >&2; exit 2; }
[[ -e "$SOURCE" ]] || { echo "Source does not exist: $SOURCE" >&2; exit 1; }
command -v rsync >/dev/null 2>&1 || { echo "rsync is required." >&2; exit 1; }
mkdir -p "$DESTINATION"

RSYNC_ARGS=(-a --partial --info=progress2)
if [[ "$OVERWRITE" -eq 0 ]]; then
  RSYNC_ARGS+=(--ignore-existing)
fi
rsync "${RSYNC_ARGS[@]}" "$SOURCE/" "$DESTINATION/"

if ! rsync -rcn --delete "$SOURCE/" "$DESTINATION/" | grep -q .; then
  echo "Checksum verification passed."
else
  echo "Checksum verification found differences." >&2
  exit 1
fi

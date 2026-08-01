"""构建或验证模型/Adapter 目录的 SHA-256 manifest。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import (
    verify_checksum_manifest,
    write_checksum_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--root", type=Path)
    verify.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        result = write_checksum_manifest(args.root, args.output)
        payload = result.model_dump(mode="json")
    else:
        result = verify_checksum_manifest(args.manifest, root=args.root)
        payload = result.model_dump(mode="json")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if args.command == "build" or result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

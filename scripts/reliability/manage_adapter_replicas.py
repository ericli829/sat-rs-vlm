"""Initialize, verify, or scrub a full LoRA adapter working/warm/golden chain."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sat_rs_vlm.models.reliability.adapter_redundancy import (
    initialize_adapter_replicas,
    scrub_adapter_replicas,
)
from sat_rs_vlm.models.reliability.checksum import verify_checksum_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "verify", "scrub"))
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def emit(payload: dict[str, object]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is None:
            print(text)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")

    if args.command == "initialize":
        if not args.working.is_dir():
            raise NotADirectoryError(args.working)
        for replica in (args.warm, args.golden):
            if replica.exists():
                raise FileExistsError(f"refusing to overwrite existing replica: {replica}")
            shutil.copytree(args.working, replica)
        manifest = initialize_adapter_replicas(
            args.working, warm_root=args.warm, golden_root=args.golden, manifest_path=args.manifest
        )
        emit({"success": True, "manifest": manifest.model_dump(mode="json")})
        return 0
    if args.command == "verify":
        checks = {
            name: verify_checksum_manifest(args.manifest, root=root).model_dump(mode="json")
            for name, root in (
                ("working", args.working),
                ("warm", args.warm),
                ("golden", args.golden),
            )
        }
        emit({"success": all(item["valid"] for item in checks.values()), "replicas": checks})
        return 0 if all(item["valid"] for item in checks.values()) else 2

    result = scrub_adapter_replicas(
        args.working, warm_root=args.warm, golden_root=args.golden, manifest=args.manifest
    )
    emit(result.model_dump(mode="json"))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

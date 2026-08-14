"""Initialize, verify, or scrub a working/warm/golden model-file replica chain."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.redundancy import inspect_replicas, scrub_and_recover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "verify", "scrub"))
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = parse_args()
    working = args.working.expanduser().resolve()
    warm, golden = args.warm.expanduser().resolve(), args.golden.expanduser().resolve()
    if args.command == "initialize":
        if not working.is_file():
            raise FileNotFoundError(f"working file does not exist: {working}")
        for destination in (warm, golden):
            if destination == working:
                raise ValueError("replica destination must differ from working file")
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing replica: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(working, destination)
        digest = file_sha256(working)
        _write(
            {
                "schema_version": "1.0",
                "expected_sha256": digest,
                "working": str(working),
                "warm": str(warm),
                "golden": str(golden),
            },
            args.output,
        )
        return 0

    if not args.expected_sha256:
        raise ValueError("--expected-sha256 is required for verify and scrub")
    if args.command == "verify":
        statuses = inspect_replicas(
            [("working", "working", working), ("warm", "warm", warm), ("golden", "golden", golden)],
            expected_sha256=args.expected_sha256,
        )
        _write(
            {
                "schema_version": "1.0",
                "expected_sha256": args.expected_sha256,
                "replicas": [item.model_dump() for item in statuses],
            },
            args.output,
        )
        return 0

    result = scrub_and_recover(
        working, warm_path=warm, golden_path=golden, expected_sha256=args.expected_sha256
    )
    _write(result.model_dump(), args.output)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

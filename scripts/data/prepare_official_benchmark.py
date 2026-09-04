"""Convert official MME-RealWorld-RS or XLRS VQA records to project JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.data.official_benchmarks import (  # noqa: E402
    adapt_mme_realworld,
    adapt_xlrs,
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("official benchmark JSON must contain a list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("official benchmark records must be JSON objects")
    identifiers = [
        str(row.get("Question_id", row.get("index", row.get("id", "")))).strip()
        for row in payload
    ]
    nonempty = [identifier for identifier in identifiers if identifier]
    if len(nonempty) != len(set(nonempty)):
        raise ValueError("official benchmark input contains duplicate sample identifiers")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mme-realworld-rs", "xlrs-vqa"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--source-repository",
        help="Official source repository URL; required when certifying a full split.",
    )
    parser.add_argument(
        "--source-commit",
        help="Exact official source commit; required when certifying a full split.",
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        help="Expected number of converted records; required for --official-full-split.",
    )
    parser.add_argument(
        "--official-full-split",
        action="store_true",
        help="Certify that the input contains the complete official target split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"official benchmark input does not exist: {source}")
    if args.expected_records is not None and args.expected_records < 0:
        raise ValueError("expected-records must be non-negative")
    if args.official_full_split and not (
        args.source_repository and args.source_commit and args.expected_records is not None
    ):
        raise ValueError(
            "--official-full-split requires --source-repository, --source-commit, "
            "and --expected-records"
        )
    rows = _read_records(source)
    converted: list[dict[str, Any]] = []
    skipped = 0
    evaluation_scope = (
        "official_full_split" if args.official_full_split else "subset_or_unspecified"
    )
    for row in rows:
        if args.dataset == "mme-realworld-rs":
            sample = adapt_mme_realworld(
                row,
                dataset_version=args.dataset_version,
                split=args.split,
                language=args.language,
                evaluation_scope=evaluation_scope,
            )
            if sample is None:
                skipped += 1
                continue
        else:
            sample = adapt_xlrs(
                row,
                dataset_version=args.dataset_version,
                split=args.split,
                language=args.language,
                evaluation_scope=evaluation_scope,
            )
        converted.append(sample)
    if not converted:
        raise ValueError("no matching official benchmark records were converted")
    if args.expected_records is not None and len(converted) != args.expected_records:
        raise ValueError(
            "converted record count does not match --expected-records: "
            f"expected={args.expected_records}, actual={len(converted)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for sample in converted:
            file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "official-benchmark-adapter-v1",
        "dataset": args.dataset,
        "dataset_version": args.dataset_version,
        "split": args.split,
        "language": args.language,
        "evaluation_scope": evaluation_scope,
        "input_file": str(source),
        "input_sha256": _sha256(source),
        "input_records": len(rows),
        "converted_records": len(converted),
        "skipped_non_target_records": skipped,
        "output_file": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "official_source": {
            "repository": args.source_repository,
            "commit": args.source_commit,
        },
        "count_check": {
            "expected_records": args.expected_records,
            "actual_records": len(converted),
            "status": (
                "passed"
                if args.expected_records is not None and len(converted) == args.expected_records
                else "not_run"
            ),
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Converted {len(converted)} records to {args.output}")
    print(f"Saved manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

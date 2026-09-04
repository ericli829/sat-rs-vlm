#!/usr/bin/env python3
"""Build a portable, checksummed RS-CLIP cloud benchmark bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = "rs_clip_cloud_benchmark"
SOURCE_FILES = (
    Path("pyproject.toml"),
    Path("README.md"),
    Path("requirements-rs-clip-cloud.txt"),
    Path("configs/cloud/rs_clip_benchmark.yaml"),
    Path("docs/architecture/rs_clip_cloud_benchmark.md"),
    Path("scripts/retriever_benchmark.py"),
    Path("scripts/make_vrsbench_retriever_manifest.py"),
    Path("tests/unit/test_rs_clip_cloud_benchmark.py"),
    Path("tests/unit/test_retriever_benchmark.py"),
    Path("tests/unit/test_make_vrsbench_retriever_manifest.py"),
)
SOURCE_DIRECTORIES = (Path("src/sat_rs_vlm"), Path("scripts/cloud"))
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def bundle_files() -> list[Path]:
    paths = list(SOURCE_FILES)
    for directory in SOURCE_DIRECTORIES:
        paths.extend(path.relative_to(ROOT) for path in (ROOT / directory).rglob("*"))
    selected = {
        path
        for path in paths
        if (ROOT / path).is_file()
        and not EXCLUDED_PARTS.intersection(path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    }
    missing = [str(path) for path in SOURCE_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("required bundle files are missing: " + ", ".join(missing))
    return sorted(selected, key=lambda path: path.as_posix())


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o100755 if executable else 0o100644
    info.external_attr = mode << 16
    return info


def build(output: Path) -> dict[str, object]:
    files = bundle_files()
    entries = []
    payloads: dict[Path, bytes] = {}
    for relative in files:
        payload = (ROOT / relative).read_bytes()
        payloads[relative] = payload
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema_version": "rs-clip-cloud-bundle-v1",
        "bundle_root": BUNDLE_ROOT,
        "entrypoint": "scripts/cloud/run_rs_clip_benchmark.py",
        "guide": "docs/architecture/rs_clip_cloud_benchmark.md",
        "file_count": len(entries),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compresslevel=9) as archive:
        for relative in files:
            name = f"{BUNDLE_ROOT}/{relative.as_posix()}"
            archive.writestr(
                zip_info(name, executable=relative.suffix == ".sh"),
                payloads[relative],
            )
        manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode()
        archive.writestr(
            zip_info(f"{BUNDLE_ROOT}/BUNDLE_MANIFEST.json"),
            manifest_payload,
        )
    temporary.replace(output)
    checksum = sha256_bytes(output.read_bytes())
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    return {
        "archive": str(output.resolve()),
        "sha256_file": str(checksum_path.resolve()),
        "sha256": checksum,
        "files": len(entries) + 1,
        "bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "rs_clip_cloud_benchmark_bundle.zip",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

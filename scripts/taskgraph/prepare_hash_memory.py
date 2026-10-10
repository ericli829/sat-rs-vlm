"""Stream image hashes, verify sealed transfers, and enrich recorded routes without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, x) != getattr(after, x) for x in fields):
        raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_status(path: Path, phase: str, **fields) -> None:
    atomic_write(path, json.dumps({
        "pid": os.getpid(), "phase": phase,
        "updated_utc": datetime.now(timezone.utc).isoformat(), **fields,
    }, ensure_ascii=False, indent=2) + "\n")


def hash_images(root: Path, output: Path, status: Path) -> None:
    root = root.resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError("No images found")
    total = sum(p.stat().st_size for p in images)
    rows, done = [], 0
    for index, path in enumerate(images, 1):
        if path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise ValueError("Image escaped dataset root")
        size = path.stat().st_size
        rows.append({"image": path.relative_to(root).as_posix(),
                     "size": size, "sha256": digest_file(path)})
        done += size
        write_status(status, "hashing", images_done=index, images_total=len(images),
                     bytes_done=done, bytes_total=total)
        if index % 50 == 0 or index == len(images):
            print(f"Hashed {index}/{len(images)} images, {done}/{total} bytes", flush=True)
    atomic_write(output, "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    write_status(status, "complete", images=len(rows), bytes=done,
                 manifest_sha256=digest_file(output),
                 skipped_non_images=[p.relative_to(root).as_posix() for p in files
                                     if p.suffix.lower() not in IMAGE_SUFFIXES])


def verify_package(root: Path, report: Path, expected_manifest: str) -> dict:
    root = root.resolve(strict=True)
    manifest_file = root / "manifest.json"
    if digest_file(manifest_file) != expected_manifest:
        raise ValueError("Transferred package manifest differs from local manifest")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["format"] != "rsu-sealed-images-v1" or not manifest["assets"]:
        raise ValueError("Invalid sealed package")
    seen = set()
    total = 0
    for asset in manifest["assets"]:
        name = asset["asset_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", name) or name in seen:
            raise ValueError("Invalid or duplicate asset id")
        seen.add(name)
        path = root / "assets" / name
        if path.resolve(strict=True).parent != root / "assets":
            raise ValueError("Asset escaped package")
        if path.stat().st_size != asset["size"] or digest_file(path) != asset["sha256"]:
            raise ValueError(f"Transferred asset failed verification: {name}")
        total += asset["size"]
    actual = {p.name for p in (root / "assets").iterdir()}
    if actual != seen:
        raise ValueError("Unexpected or missing assets in transferred package")
    value = {"complete": True, "assets": len(seen), "bytes": total,
             "manifest_sha256": expected_manifest}
    atomic_write(report, json.dumps(value, indent=2) + "\n")
    print(json.dumps(value), flush=True)
    return value


def bind_table(questions_path: Path, hashes_path: Path, table: Path,
               report: Path, expected_table_hash: str, expected_rows: int) -> dict:
    # Import only in the local binder; the cloud hashing modes require no installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from sat_rs_vlm.taskgraph.runtime_memory import RuntimeMemory, hash_question

    original = table.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_table_hash:
        raise ValueError("Memory table changed since job preparation; refusing overwrite")
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or len({r["sample_id"] for r in rows}) != expected_rows:
        raise ValueError("Unexpected memory row count or duplicate sample ids")
    needed_ids = {r["sample_id"] for r in rows}
    questions = {}
    for row in json.loads(questions_path.read_text(encoding="utf-8")):
        sid = row["Question_id"]
        if sid not in needed_ids:
            continue
        if sid in questions:
            previous = questions[sid]
            current = (row["Image"], row["Text"])
            if previous != current:
                raise ValueError(f"Conflicting question rows for memory id: {sid}")
            continue
        # Only these fields are used; answers and scoring labels never enter the registry.
        questions[sid] = (row["Image"], row["Text"])
    images = {}
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = row["image"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or name in images:
            raise ValueError("Unsafe or duplicate image path in hash manifest")
        if not SHA256.fullmatch(row["sha256"]) or row["size"] <= 0:
            raise ValueError("Invalid image hash or size")
        images[name] = row["sha256"]
    enriched = []
    groups = defaultdict(list)
    for row in rows:
        image, question = questions[row["sample_id"]]
        question_hash = hash_question(question)
        if row.get("question_sha256") != question_hash:
            raise ValueError(f"Question hash mismatch: {row['sample_id']}")
        updated = {**row, "image_sha256": images[image]}
        if any(updated[k] != v for k, v in row.items()):
            raise ValueError("Existing memory fields changed")
        enriched.append(updated)
        groups[(updated["image_sha256"], question_hash)].append(updated)
    candidate = report.parent / "memory.candidate.jsonl"
    atomic_write(candidate, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                                    + "\n" for r in enriched))
    memory = RuntimeMemory(candidate)
    conflicts = []
    for (image_hash, question_hash), matching in groups.items():
        variants = {(r["mode"], r.get("variant"), r.get("backend")) for r in matching}
        entry = memory.lookup_hashes((image_hash,), question_hash)
        if len(variants) > 1:
            if entry is not None:
                raise ValueError("Conflicting hash key did not fall through")
            conflicts.append({"image_sha256": image_hash, "question_sha256": question_hash,
                              "records": [{k: r.get(k) for k in
                                           ("sample_id", "mode", "variant", "backend")}
                                          for r in matching]})
        elif entry is None or (entry.mode, entry.variant, entry.backend) != next(iter(variants)):
            raise ValueError("Bound route failed exact replay validation")
    if digest_file(table) != expected_table_hash:
        raise ValueError("Memory table changed during validation; refusing overwrite")
    atomic_write(report.parent / "memory.before.jsonl", original.decode("utf-8"))
    atomic_write(table, candidate.read_text(encoding="utf-8"))
    value = {"complete": True, "rows": len(rows), "images_hashed": len(images),
             "images_referenced": len({questions[r["sample_id"]][0] for r in rows}),
             "unique_hash_keys": len(groups), "usable_hash_keys": len(groups) - len(conflicts),
             "conflicts": conflicts, "table_sha256": digest_file(table),
             "hash_manifest_sha256": digest_file(hashes_path),
             "existing_fields_preserved": True}
    atomic_write(report, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "conflicts"}), flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    hashing = modes.add_parser("hash-images")
    hashing.add_argument("--image-root", type=Path, required=True)
    hashing.add_argument("--output", type=Path, required=True)
    hashing.add_argument("--status", type=Path, required=True)
    package = modes.add_parser("verify-package")
    package.add_argument("--root", type=Path, required=True)
    package.add_argument("--report", type=Path, required=True)
    package.add_argument("--manifest-sha256", required=True)
    binder = modes.add_parser("bind-table")
    binder.add_argument("--questions", type=Path, required=True)
    binder.add_argument("--hash-manifest", type=Path, required=True)
    binder.add_argument("--table", type=Path, required=True)
    binder.add_argument("--report", type=Path, required=True)
    binder.add_argument("--expected-table-sha256", required=True)
    binder.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "hash-images":
        try:
            hash_images(args.image_root, args.output, args.status)
        except Exception as exc:
            write_status(args.status, "failed", error=type(exc).__name__, message=str(exc))
            raise
    elif args.mode == "verify-package":
        verify_package(args.root, args.report, args.manifest_sha256)
    else:
        bind_table(args.questions, args.hash_manifest, args.table, args.report,
                   args.expected_table_sha256, args.expected_rows)


if __name__ == "__main__":
    main()

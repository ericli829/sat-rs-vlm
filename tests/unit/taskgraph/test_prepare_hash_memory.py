import hashlib
import json

import pytest
from scripts.taskgraph.prepare_hash_memory import bind_table, digest_file, hash_images

from sat_rs_vlm.taskgraph.runtime_memory import RuntimeMemory, hash_question


def test_hash_inventory_streams_original_bytes_and_skips_non_images(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "original.png").write_bytes(b"original encoded image bytes")
    (root / "desktop.ini").write_text("ignored")
    manifest, status = tmp_path / "hashes.jsonl", tmp_path / "status.json"
    hash_images(root, manifest, status)
    row = json.loads(manifest.read_text())
    assert row["sha256"] == hashlib.sha256(b"original encoded image bytes").hexdigest()
    assert json.loads(status.read_text())["skipped_non_images"] == ["desktop.ini"]


def test_binding_preserves_records_and_validates_conflict_fallback(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"Question_id": "a", "Image": "rs/image.png", "Text": "Same?"},
        {"Question_id": "b", "Image": "rs/image.png", "Text": "Same?"},
        {"Question_id": "c", "Image": "rs/image.png", "Text": "Other?"},
        {"Question_id": "unused-duplicate", "Image": "one.png", "Text": "First"},
        {"Question_id": "unused-duplicate", "Image": "two.png", "Text": "Second"},
    ]))
    manifest = tmp_path / "hashes.jsonl"
    image_hash = hashlib.sha256(b"image").hexdigest()
    manifest.write_text(json.dumps({"image": "rs/image.png", "size": 5, "sha256": image_hash}))
    rows = [{"sample_id": sid, "mode": mode, "question_sha256": hash_question(question),
             "note": "recorded execution", "custom_metadata": 42}
            for sid, mode, question in (
                ("a", "DIRECT_VLM", "Same?"), ("b", "TASKGRAPH_UHR", "Same?"),
                ("c", "DIRECT_VLM", "Other?"))]
    table = tmp_path / "memory.jsonl"
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = bind_table(questions, manifest, table, tmp_path / "report.json", digest_file(table), 3)
    assert result["rows"] == 3
    assert result["usable_hash_keys"] == 1
    assert len(result["conflicts"]) == 1
    rewritten = [json.loads(line) for line in table.read_text().splitlines()]
    assert [{k: v for k, v in r.items() if k != "image_sha256"} for r in rewritten] == rows
    assert RuntimeMemory(table).lookup_hashes((image_hash,), hash_question("Same?")) is None
    assert RuntimeMemory(table).lookup_hashes((image_hash,), hash_question("Other?")).mode == (
        "DIRECT_VLM"
    )


def test_binding_rejects_changed_table_without_writing(tmp_path):
    table = tmp_path / "memory.jsonl"
    table.write_text("unchanged")
    with pytest.raises(ValueError, match="changed since"):
        bind_table(tmp_path / "questions", tmp_path / "hashes", table,
                   tmp_path / "report.json", "0" * 64, 1)
    assert table.read_text() == "unchanged"

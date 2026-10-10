from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from sat_rs_vlm.taskgraph import RuntimeRequest, fake_runtime
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime_memory import RuntimeMemory, hash_question


def _memory(tmp_path, rows):
    path = tmp_path / "memory.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return RuntimeMemory(path)


def _row(image, question="Question?", **extra):
    return {
        "sample_id": "old-id", "mode": "DIRECT_VLM",
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "question_sha256": hash_question(question), **extra,
    }


def test_match_original_bytes_and_exact_text_independent_of_filename(tmp_path):
    image = tmp_path / "renamed.png"
    image.write_bytes(b"original bytes")
    memory = _memory(tmp_path, [_row(image.read_bytes())])
    assert memory.lookup_content((str(image),), "Question?").sample_id == "old-id"
    assert memory.lookup_content((str(image),), "Question? ") is None
    assert memory.lookup_content((str(image),), "question?") is None
    image.write_bytes(b"different bytes")
    assert memory.lookup_content((str(image),), "Question?") is None


def test_partial_hashes_and_conflicting_routes_never_match(tmp_path):
    row = _row(b"image")
    memory = _memory(tmp_path, [row, {**row, "sample_id": "other", "mode": "TASKGRAPH_UHR"}])
    assert memory.lookup_hashes((row["image_sha256"],), row["question_sha256"]) is None
    row.pop("image_sha256")
    memory = _memory(tmp_path, [row])
    assert memory.lookup_content(("missing-image",), "Question?") is None


def test_ordered_pair_and_image_count_are_part_of_identity(tmp_path):
    a, b = hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()
    row = {"sample_id": "pair", "mode": "DIRECT_VLM",
           "image_sha256s": [a, b], "question_sha256": hash_question("Change?")}
    memory = _memory(tmp_path, [row])
    assert memory.lookup_hashes((a, b), row["question_sha256"]) is not None
    assert memory.lookup_hashes((b, a), row["question_sha256"]) is None
    assert memory.lookup_hashes((a,), row["question_sha256"]) is None
    assert memory.lookup_content(("a", "b"), "Change?", verified_image_sha256s=(a,)) is None


def test_record_preserves_hashes_and_rebuilds_lookup(tmp_path):
    row = _row(b"image", note="historical")
    memory = _memory(tmp_path, [row])
    memory.record("old-id", mode="TASKGRAPH_UHR", variant="tight")
    reloaded = RuntimeMemory(memory.path)
    entry = reloaded.lookup_hashes((row["image_sha256"],), row["question_sha256"])
    assert entry.mode == "TASKGRAPH_UHR"
    assert entry.variant == "tight"


def test_runtime_uses_joint_hash_and_resets_variant_on_miss(tmp_path):
    image = str(Path("tests/fixtures/miniature_dataset/images/counting.ppm").resolve())
    digest = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    memory = _memory(tmp_path, [{**_row(b"unused"), "image_sha256": digest, "variant": "tight"}])
    def fail_if_rehashed(path):
        raise AssertionError("Package-verified hashes should not reread image files")
    memory._file_hash = fail_if_rehashed
    runtime = fake_runtime()
    runtime.runtime_memory = memory
    request = RuntimeRequest(
        "new-online-id", "competition_protocol_v2", "semantic", "Question?\nInstruction",
        (image,), execution_mode=ExecutionMode.TASKGRAPH_UHR, defer_final_answer=True,
        memory_question="Question?", verified_image_sha256s=(digest,), allow_hash_memory=True,
    )
    try:
        result = runtime.run(request)
        assert result.execution_mode is ExecutionMode.DIRECT_VLM
        assert result.trace.telemetry["runtime_memory_hash_hit"] is True
        assert runtime.composer.entity_tight_crops is True
        miss = replace(
            request, sample_id="old-id", memory_question="Different",
            execution_mode=ExecutionMode.DIRECT_VLM,
        )
        result = runtime.run(miss)
        assert result.trace.telemetry["runtime_memory_hash_hit"] is False
        assert runtime.composer.entity_tight_crops is False
        explicit = replace(
            request, allow_hash_memory=False, execution_mode=ExecutionMode.DIRECT_VLM
        )
        assert runtime.run(explicit).trace.telemetry["runtime_memory_hash_hit"] is False
    finally:
        runtime.close()


def test_question_hash_uses_exact_utf8_without_unicode_normalization():
    assert hash_question("é") != hash_question("e\u0301")

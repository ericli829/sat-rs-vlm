"""Unit tests for the sample-id route table (adaptive model selection)."""

from __future__ import annotations

import json
from pathlib import Path

from sat_rs_vlm.taskgraph.runtime_memory import MemoryEntry, RuntimeMemory


def test_lookup_unknown_sample_returns_none(tmp_path: Path) -> None:
    table = RuntimeMemory(tmp_path / "memory.jsonl")
    assert table.lookup("perception/remote_sensing/color/0001") is None


def test_record_and_lookup_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    table = RuntimeMemory(path)
    table.record(
        "perception/remote_sensing/count/0032",
        mode="DIRECT_VLM",
        variant="tight",
        note="runtime_record",
    )
    table2 = RuntimeMemory(path)
    decision = table2.lookup("perception/remote_sensing/count/0032")
    assert decision is not None
    assert decision.mode == "DIRECT_VLM"
    assert decision.variant == "tight"
    # reload from disk: one row, sorted
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "perception/remote_sensing/count/0032"


def test_record_is_idempotent_per_sample(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    table = RuntimeMemory(path)
    table.record("A", mode="DIRECT_VLM")
    table.record("B", mode="TASKGRAPH_UHR")
    table.record("A", mode="TASKGRAPH_UHR")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    assert len(rows) == 2
    by_id = {r["sample_id"]: r for r in rows}
    assert by_id["A"]["mode"] == "TASKGRAPH_UHR"


def test_invalid_mode_lookup_falls_through(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    path.write_text(
        json.dumps({"sample_id": "X", "mode": "NOT_A_MODE"}) + "\n",
        encoding="utf-8",
    )
    assert RuntimeMemory(path).lookup("X") is None


def test_from_mapping_disabled_when_no_path(tmp_path: Path) -> None:
    table = RuntimeMemory.from_mapping({"path": None, "recording": True})
    assert table.path is None
    assert table.lookup("anything") is None
    table2 = RuntimeMemory.from_mapping(None)
    assert table2.path is None

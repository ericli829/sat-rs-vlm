from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "region_replay_atomic_write", Path("scripts/competition/replay_change_region.py")
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_busy_destination_keeps_previous_valid_snapshot_until_rename_succeeds(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    target.write_text('{"completed":49}', encoding="utf-8")
    original_replace = Path.replace
    attempts = []

    def busy_replace(source, destination):
        if source.name == "status.json.tmp":
            attempts.append(1)
            if len(attempts) <= 2:
                assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 49}
                raise PermissionError("simulated Windows file sharing conflict")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", busy_replace)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    replay.write_json(target, {"completed": 50})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 50}
    assert not target.with_suffix(".json.tmp").exists()


def test_persistent_access_failure_is_bounded_and_preserves_previous_snapshot(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    target.write_text('{"completed":49}', encoding="utf-8")
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("locked")))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="locked"):
        replay.write_json(target, {"completed": 50})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 49}
    assert json.loads(target.with_suffix(".json.tmp").read_text(encoding="utf-8")) == {"completed": 50}

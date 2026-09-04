from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.data import prepare_official_benchmark as prepare


def _mme_row(sample_id: str = "q-1") -> dict[str, object]:
    return {
        "Question_id": sample_id,
        "Image": "remote/1.jpg",
        "Text": "Which object is visible?",
        "Answer choices": ["(A) ship", "(B) car", "(C) road", "(D) tree", "(E) none"],
        "Ground truth": "A",
        "Task": "Perception",
        "Subtask": "Remote Sensing",
        "Category": "classification",
    }


def test_full_split_certification_requires_source_and_expected_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mme.json"
    source.write_text(json.dumps([_mme_row()]), encoding="utf-8")
    output = tmp_path / "converted.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_official_benchmark.py",
            "--dataset",
            "mme-realworld-rs",
            "--input",
            str(source),
            "--output",
            str(output),
            "--dataset-version",
            "official-test",
            "--split",
            "test",
            "--language",
            "en",
            "--official-full-split",
        ],
    )
    with pytest.raises(ValueError, match="requires --source-repository"):
        prepare.main()


def test_full_split_manifest_records_source_and_count_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mme.json"
    source.write_text(json.dumps([_mme_row()]), encoding="utf-8")
    output = tmp_path / "converted.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_official_benchmark.py",
            "--dataset",
            "mme-realworld-rs",
            "--input",
            str(source),
            "--output",
            str(output),
            "--dataset-version",
            "official-test",
            "--split",
            "test",
            "--language",
            "en",
            "--source-repository",
            "https://example.invalid/official",
            "--source-commit",
            "a" * 40,
            "--expected-records",
            "1",
            "--official-full-split",
        ],
    )

    assert prepare.main() == 0
    manifest = json.loads(
        (tmp_path / "converted.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["official_source"]["commit"] == "a" * 40
    assert manifest["count_check"] == {
        "expected_records": 1,
        "actual_records": 1,
        "status": "passed",
    }

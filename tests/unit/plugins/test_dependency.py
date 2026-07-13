from pathlib import Path

import pytest

from sat_rs_vlm.plugins import dependency


def test_missing_dependency_is_checked_without_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely-not-installed>=1\n", encoding="utf-8")
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pip must not be called by dependency check")

    monkeypatch.setattr(dependency.subprocess, "run", unexpected)
    statuses = dependency.check_requirements(requirements, "fake")
    assert statuses[0].status == "missing"
    assert called is False


def test_installed_version_constraint_uses_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example>=2\n", encoding="utf-8")
    monkeypatch.setattr(dependency.importlib.metadata, "version", lambda name: "1.0")
    assert dependency.check_requirements(requirements, "fake")[0].status == "version_conflict"

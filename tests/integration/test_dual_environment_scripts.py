import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/environment/bootstrap_local.py",
        "scripts/environment/check_environment.py",
        "scripts/environment/export_environment.py",
        "scripts/data/validate_dataset.py",
        "scripts/data/package_dataset.py",
        "scripts/data/unpack_dataset.py",
        "scripts/training/run_train.py",
        "scripts/training/run_smoke_train.py",
        "scripts/training/resume_train.py",
        "scripts/data/prepare_multisource_training_data.py",
    ],
)
def test_python_entrypoint_help(relative: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_shell_scripts_exist_and_parse_when_bash_is_available() -> None:
    scripts = [
        PROJECT_ROOT / "scripts/environment/setup_autodl.sh",
        PROJECT_ROOT / "environments/lae_dino/install.sh",
        PROJECT_ROOT / "scripts/environment/activate_autodl_python.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_train.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_full_pipeline.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_levircc_train.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_levircc_replay.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_qwen3vl_4b_stage_a.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_rs_object_adapter_v0_e1.sh",
        PROJECT_ROOT / "scripts/evaluation/run_autodl_replay_eval.sh",
        PROJECT_ROOT / "scripts/storage/sync_to_local_disk.sh",
        PROJECT_ROOT / "scripts/storage/backup_results.sh",
    ]
    assert all(path.is_file() for path in scripts)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed on this Windows host")
    for script in scripts:
        completed = subprocess.run(
            [bash, "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"


def test_model_environment_check_includes_scipy() -> None:
    script = PROJECT_ROOT / "scripts/environment/check_environment.py"
    spec = importlib.util.spec_from_file_location("environment_check_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "scipy" in module.MODEL_MODULES


def test_environment_check_exposes_retriever_contract() -> None:
    script = PROJECT_ROOT / "scripts/environment/check_environment.py"
    spec = importlib.util.spec_from_file_location("retriever_environment_check_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RETRIEVER_MODULES == ("torch", "open_clip", "timm")


def test_required_retriever_check_imports_each_module(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts/environment/check_environment.py"
    spec = importlib.util.spec_from_file_location("retriever_import_check_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported: list[str] = []

    def available(_name: str) -> bool:
        return True

    def importable(name: str) -> bool:
        imported.append(name)
        return True

    def gpu_report() -> dict[str, bool]:
        return {"available": False}

    module._available = available
    module._importable = importable
    module._gpu_report = gpu_report

    report = module.build_report(
        SimpleNamespace(
            require_retriever=True,
            dataset_root=tmp_path,
            model_root=tmp_path,
            output_root=tmp_path,
        )
    )

    assert imported == ["torch", "open_clip", "timm"]
    assert report["retriever_check_mode"] == "import"


def test_local_bootstrap_dry_run_uses_pyproject_retriever_extra(tmp_path: Path) -> None:
    venv = tmp_path / "taskgraph-venv"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/environment/bootstrap_local.py"),
            "--venv",
            str(venv),
            "--with-dev",
            "--with-model",
            "--with-retriever",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert ".[dev,model,retriever]" in completed.stdout
    assert not venv.exists()


def test_autodl_taskgraph_dry_run_is_non_mutating_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed on this Windows host")
    completed = subprocess.run(
        [
            bash,
            str(PROJECT_ROOT / "scripts/environment/setup_autodl.sh"),
            "--project-root",
            str(PROJECT_ROOT),
            "--install-model",
            "--install-retriever",
            "--install-lae",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "torch version would be pinned" in completed.stdout
    assert "LAE environment: rs-vlm-lae" in completed.stdout
    assert "create/reuse isolated Conda env" in completed.stdout


def test_autodl_and_lae_scripts_preserve_environment_isolation() -> None:
    autodl = (PROJECT_ROOT / "scripts/environment/setup_autodl.sh").read_text(
        encoding="utf-8"
    )
    lae = (PROJECT_ROOT / "environments/lae_dino/install.sh").read_text(
        encoding="utf-8"
    )

    assert '-e ".[retriever]"' in autodl
    assert "torch==%s" in autodl
    assert "pip install --force-reinstall" not in autodl
    assert 'bash "$REPOSITORY_ROOT/environments/lae_dino/install.sh"' in autodl
    assert 'conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"' in lae
    assert 'conda run --name "$ENV_NAME"' in lae
    assert "LAE_DINO_PYTHON" in lae
    assert "LAE_DINO_SOURCE_ROOT" in lae
    assert "LAE_DINO_CONFIG" in lae
    assert "LAE_DINO_CHECKPOINT" in lae
    assert "LAE_DINO_BERT_ROOT" in lae


def test_object_adapter_launcher_exposes_bounded_smoke_controls() -> None:
    script = (PROJECT_ROOT / "scripts/training/run_autodl_rs_object_adapter_v0_e1.sh").read_text(
        encoding="utf-8"
    )

    assert "--max-val-groups)" in script
    assert "--skip-e1)" in script
    assert '[[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]' in script
    assert "export OMP_NUM_THREADS=8" in script

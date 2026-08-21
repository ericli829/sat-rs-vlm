from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts/environment/activate_autodl_python.sh"


def test_autodl_activation_prefers_named_conda_environment_over_stale_python() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "resolve_autodl_conda()" in text
    assert '"/root/miniconda3/bin/conda"' in text
    assert 'conda activate "$env_name"' in text
    assert 'configured_python="${AUTODL_PYTHON:-}"' in text
    assert "Ignoring stale AUTODL_PYTHON" in text
    assert "hash -r" in text
    assert 'selected_python="${CONDA_PREFIX:-}/bin/python"' in text
    assert '"$selected_python" != "$CONDA_PREFIX/bin/python"' in text

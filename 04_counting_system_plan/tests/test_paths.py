from pathlib import Path

from counting_system.paths import (
    LINUX_FS,
    host_roots,
    load_paths,
    relocate_autodl_paths,
)


def test_relocate_autodl_paths_uses_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("COUNTING_AUTODL_FS", raising=False)
    monkeypatch.delenv("AUTODL_FS", raising=False)
    monkeypatch.delenv("COUNTING_AUTODL_TMP", raising=False)
    monkeypatch.delenv("AUTODL_TMP", raising=False)
    monkeypatch.setattr(
        "counting_system.paths.host_roots",
        lambda workspace=None: (
            (workspace or tmp_path) / "autodl-fs",
            (workspace or tmp_path) / "autodl-tmp",
        ),
    )
    tree = {
        "datasets": {"xlrsbench_lite": f"{LINUX_FS}/datasets/xlrsbench-lite"},
        "models": {"georsclip_ckpt": f"{LINUX_FS}/models/georsclip/ckpt/RS5M_ViT-B-32.pt"},
        "huggingface": {"home": "/root/autodl-tmp/cache/huggingface"},
    }
    out = relocate_autodl_paths(tree, workspace=tmp_path)
    fs = (tmp_path / "autodl-fs").resolve()
    tmp = (tmp_path / "autodl-tmp").resolve()
    assert out["autodl_fs"] == str(fs)
    assert out["autodl_tmp"] == str(tmp)
    assert Path(out["datasets"]["xlrsbench_lite"]) == fs / "datasets" / "xlrsbench-lite"
    assert Path(out["models"]["georsclip_ckpt"]) == fs / "models" / "georsclip" / "ckpt" / "RS5M_ViT-B-32.pt"
    assert Path(out["huggingface"]["home"]) == tmp / "cache" / "huggingface"


def test_host_roots_honor_env(tmp_path: Path, monkeypatch):
    fs = tmp_path / "custom-fs"
    tmp = tmp_path / "custom-tmp"
    monkeypatch.setenv("COUNTING_AUTODL_FS", str(fs))
    monkeypatch.setenv("COUNTING_AUTODL_TMP", str(tmp))
    got_fs, got_tmp = host_roots(tmp_path)
    assert got_fs == fs
    assert got_tmp == tmp


def test_load_paths_maps_xlrs_and_models(monkeypatch):
    load_paths.cache_clear()
    monkeypatch.delenv("COUNTING_DATASET_ROOT", raising=False)
    monkeypatch.delenv("COUNTING_LAE_WEIGHTS", raising=False)
    monkeypatch.delenv("COUNTING_GEORS_CKPT", raising=False)
    paths = load_paths()
    xlrs = Path(paths["datasets"]["xlrsbench_lite"])
    assert xlrs.as_posix().endswith("datasets/xlrsbench-lite")
    assert Path(paths["models"]["lae_dino_weights"]).name == "lae_dino_swint_lae1m-28ca3a15.pth"
    assert Path(paths["models"]["georsclip_ckpt"]).name == "RS5M_ViT-B-32.pt"
    assert "autodl-fs" in Path(paths["autodl_fs"]).as_posix()
    load_paths.cache_clear()

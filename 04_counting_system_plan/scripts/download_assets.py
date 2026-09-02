"""下载 LAE-DINO、GeoRSCLIP、XLRS-Bench-lite 计数子集到 autodl-fs。

路径只来自 configs/paths.yaml（经 counting_system.paths 映射）。
AutoDL：逻辑根 /root/autodl-fs，大文件可落到 /root/autodl-tmp 再符号链接。
Windows：落到仓库上一级 autodl-fs / autodl-tmp。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counting_system.data.xlrs_lite import is_counting_row, parse_answer_value, parse_region_name  # noqa: E402
from counting_system.paths import load_paths  # noqa: E402
from counting_system.target import extract_target_from_question  # noqa: E402

LAE_REPOS = (
    "jaychempan/LAE-DINO",
    "ML4Sustain/LAE-DINO",
)
LAE_WEIGHT_FILE = "checkpoints/lae_dino_swint_lae1m-28ca3a15.pth"
GEORS_REPO = "Zilun/GeoRSCLIP"
GEORS_FILE = "ckpt/RS5M_ViT-B-32.pt"
XLRS_DATASET = "initiacms/XLRS-Bench-lite"
GIT_MIRRORS = (
    "https://github.com/jaychempan/LAE-DINO.git",
    "https://ghfast.top/https://github.com/jaychempan/LAE-DINO.git",
    "https://gitclone.com/github.com/jaychempan/LAE-DINO.git",
)
ZIP_MIRRORS = (
    "https://github.com/jaychempan/LAE-DINO/archive/refs/heads/main.zip",
    "https://ghfast.top/https://github.com/jaychempan/LAE-DINO/archive/refs/heads/main.zip",
    "https://codeload.github.com/jaychempan/LAE-DINO/zip/refs/heads/main",
)


def _prepare_env(paths: dict[str, Any]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    # hf-mirror 不代理 cas-server.xethub.hf.co，开 Xet 会 401
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    endpoint = str(paths.get("huggingface", {}).get("endpoint") or "https://hf-mirror.com")
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    hf_home = Path(paths.get("huggingface", {}).get("home") or (Path(paths["autodl_tmp"]) / "cache" / "huggingface"))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))


def _ensure_store(logical: Path, physical: Path) -> Path:
    """AutoDL 上在 autodl-tmp 落盘并链接到 autodl-fs；Windows 直接写 autodl-fs。"""

    logical = Path(logical)
    physical = Path(physical)
    posix = str(logical).replace("\\", "/")
    if os.name == "nt" or not posix.startswith("/root/"):
        logical.mkdir(parents=True, exist_ok=True)
        return logical
    physical.mkdir(parents=True, exist_ok=True)
    logical.parent.mkdir(parents=True, exist_ok=True)
    if logical.exists() or logical.is_symlink():
        return logical
    try:
        logical.symlink_to(physical, target_is_directory=True)
        print(f"symlink {logical} -> {physical}")
    except OSError as exc:
        print(f"symlink failed ({exc}), writing directly to {logical}")
        logical.mkdir(parents=True, exist_ok=True)
    return logical


def _copy_if_needed(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and src.exists() and dest.stat().st_size == src.stat().st_size and dest.stat().st_size > 0:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return
    shutil.copy2(src, dest)
    print(f"copied {src} -> {dest} ({dest.stat().st_size} bytes)")


def _hf_download(repo_id: str, filename: str, dest: Path, *, repo_type: str = "model") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("pip install huggingface_hub") from exc
    print(f"hf_hub_download {repo_id}/{filename} -> {dest}", flush=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type,
    }
    try:
        cached = hf_hub_download(**kwargs, resume_download=True)
    except TypeError:
        cached = hf_hub_download(**kwargs)
    _copy_if_needed(Path(cached), dest)


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"skip existing {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"GET {url}", flush=True)
    req = Request(url, headers={"User-Agent": "counting-system/1.0"})
    with urlopen(req, timeout=180) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def clone_lae(repo_dir: Path) -> None:
    marker = repo_dir / "README.md"
    if marker.exists() and (
        (repo_dir / ".git").exists() or (repo_dir / "mmdetection_lae").exists() or (repo_dir / "demo").exists()
    ):
        print(f"LAE-DINO repo exists: {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    git_bin = shutil.which("git")
    if git_bin:
        for url in GIT_MIRRORS:
            print(f"cloning LAE-DINO from {url}", flush=True)
            proc = subprocess.run([git_bin, "clone", "--depth", "1", url, str(repo_dir)], env=env)
            if proc.returncode == 0 and marker.exists():
                return
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
    last_err: Exception | None = None
    for url in ZIP_MIRRORS:
        try:
            print(f"downloading LAE-DINO zip {url}", flush=True)
            req = Request(url, headers={"User-Agent": "counting-system/1.0"})
            with urlopen(req, timeout=180) as resp:
                blob = resp.read()
            tmp_root = repo_dir.parent / "_lae_unzip"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(tmp_root)
            children = [p for p in tmp_root.iterdir() if p.is_dir()]
            src = children[0] if len(children) == 1 else tmp_root
            shutil.move(str(src), str(repo_dir))
            shutil.rmtree(tmp_root, ignore_errors=True)
            if marker.exists():
                return
        except Exception as exc:
            last_err = exc
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
    raise RuntimeError(f"failed to fetch LAE-DINO repo: {last_err}")


def download_lae_weights(dest: Path) -> None:
    errors: list[str] = []
    for repo_id in LAE_REPOS:
        try:
            _hf_download(repo_id, LAE_WEIGHT_FILE, dest)
            if dest.exists() and dest.stat().st_size > 1024 * 1024:
                return
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for repo_id in LAE_REPOS:
        url = f"{endpoint}/{repo_id}/resolve/main/{LAE_WEIGHT_FILE}"
        try:
            _http_download(url, dest)
            if dest.exists() and dest.stat().st_size > 1024 * 1024:
                return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("LAE-DINO weights download failed:\n" + "\n".join(errors))


def download_geors(dest: Path) -> None:
    try:
        _hf_download(GEORS_REPO, GEORS_FILE, dest)
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            return
    except Exception as first:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/{GEORS_REPO}/resolve/main/{GEORS_FILE}"
        try:
            _http_download(url, dest)
            return
        except Exception as second:
            raise RuntimeError(f"GeoRSCLIP download failed: {first}; {second}") from second


def download_bert(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    has_cfg = (dest / "config.json").exists()
    has_weight = any((dest / name).exists() for name in ("pytorch_model.bin", "model.safetensors"))
    if has_cfg and has_weight:
        print(f"BERT exists: {dest}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("pip install huggingface_hub") from exc
    print(f"snapshot_download google-bert/bert-base-uncased -> {dest}", flush=True)
    snapshot_download(
        repo_id="google-bert/bert-base-uncased",
        local_dir=str(dest),
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )


def _as_pil(image: Any):
    from PIL import Image

    if image is None:
        return None
    if isinstance(image, dict):
        raw = image.get("bytes")
        path = image.get("path")
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(path, str) and Path(path).exists():
            return Image.open(path).convert("RGB")
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    convert = getattr(image, "convert", None)
    if callable(convert):
        try:
            return convert("RGB")
        except Exception:
            return None
    return None


def _safe_slug(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "counting"


def _sample_id(row: dict[str, Any], kept: int) -> str:
    cat = str(row.get("category") or row.get("l2-category") or row.get("l2_category") or "counting")
    idx = row.get("index")
    if idx is None:
        return f"{_safe_slug(cat)}_{kept}"
    return f"{_safe_slug(cat)}_{idx}"


def _load_existing_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id") or rec.get("index") or "")
            if sid:
                existing[sid] = rec
    return existing


def download_xlrs_counting(root: Path, *, max_samples: int, jpeg_quality: int) -> None:
    try:
        from datasets import Image as HFImage
        from datasets import Sequence, load_dataset
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("pip install datasets pillow") from exc

    Image.MAX_IMAGE_PIXELS = None
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = root / "counting.jsonl"
    existing = _load_existing_jsonl(out_jsonl)
    path_to_files: dict[str, list[str]] = {}
    for rec in existing.values():
        key = str(rec.get("path") or rec.get("sample_id") or "")
        files = rec.get("image_paths") or ([rec["image_path"]] if rec.get("image_path") else [])
        if key and files:
            path_to_files[key] = [str(x) for x in files]
    limit = None if max_samples <= 0 else max_samples
    if limit is not None and len(existing) >= limit:
        print(f"XLRS counting already has {len(existing)} samples (>= {limit}), skip")
        return
    print(f"streaming {XLRS_DATASET} counting -> {root} (max={limit or 'all'}, resume={len(existing)})", flush=True)
    ds = load_dataset(XLRS_DATASET, split="train", streaming=True)
    try:
        ds = ds.cast_column("image", Sequence(HFImage(decode=False)))
    except Exception:
        try:
            ds = ds.cast_column("image", HFImage(decode=False))
        except Exception:
            pass
    kept = len(existing)
    scanned = 0
    mode = "a" if existing else "w"
    with out_jsonl.open(mode, encoding="utf-8") as fh:
        for row in ds:
            scanned += 1
            if scanned % 20 == 0:
                print(f"  scanned={scanned} kept={kept}", flush=True)
            if not is_counting_row(row):
                continue
            sample_id = _sample_id(row, kept)
            if sample_id in existing:
                continue
            src_path = str(row.get("path") or "")
            reuse_key = src_path or sample_id
            saved = list(path_to_files.get(reuse_key) or [])
            if not saved or any(not (root / rel).exists() for rel in saved):
                raw_images = row.get("image")
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                saved = []
                for idx, raw in enumerate(raw_images):
                    image = _as_pil(raw)
                    if image is None:
                        continue
                    name = f"{sample_id}.jpg" if idx == 0 else f"{sample_id}_{idx}.jpg"
                    rel = Path("images") / name
                    image.save(root / rel, format="JPEG", quality=jpeg_quality)
                    saved.append(str(rel).replace("\\", "/"))
            if not saved:
                continue
            path_to_files[reuse_key] = saved
            question = str(row.get("question") or "")
            options = list(row.get("multi-choice options") or row.get("options") or [])
            letter, value = parse_answer_value(str(row.get("answer") or ""), options)
            rec = {
                "sample_id": sample_id,
                "index": row.get("index"),
                "question": question,
                "options": options,
                "answer": row.get("answer"),
                "answer_letter": letter,
                "answer_value": value,
                "category": row.get("category"),
                "l2_category": row.get("l2-category") or row.get("l2_category"),
                "path": src_path,
                "image_path": saved[0],
                "image_paths": saved,
                "region_name": parse_region_name(question),
                "target": extract_target_from_question(question).name,
            }
            rec["entire"] = rec["region_name"] is None and "regional" not in str(
                rec["l2_category"] or rec["category"] or ""
            ).lower()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            existing[sample_id] = rec
            kept += 1
            print(
                f"  keep {sample_id} category={rec['category']} l2={rec['l2_category']} images={len(saved)}",
                flush=True,
            )
            if limit is not None and kept >= limit:
                break
    meta = {
        "source": XLRS_DATASET,
        "scanned": scanned,
        "kept": kept,
        "max_samples": limit,
        "root": str(root),
        "note": "full lite set is ~43GB; this export keeps counting rows only",
    }
    (root / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {kept} counting samples to {out_jsonl}", flush=True)


def write_status(paths: dict[str, Any], extra: dict[str, Any]) -> None:
    fs = Path(paths["autodl_fs"])
    jsonl = Path(paths["datasets"]["xlrsbench_lite"]) / "counting.jsonl"
    n_xlrs = 0
    if jsonl.exists():
        n_xlrs = sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = {
        "autodl_fs": str(fs),
        "autodl_tmp": paths["autodl_tmp"],
        "xlrsbench_lite": paths["datasets"]["xlrsbench_lite"],
        "lae_dino_repo": paths["models"]["lae_dino_repo"],
        "lae_dino_weights": paths["models"]["lae_dino_weights"],
        "georsclip_ckpt": paths["models"]["georsclip_ckpt"],
        "bert_dir": paths["models"]["bert_dir"],
        "xlrs_counting_rows": n_xlrs,
        "exists": {
            "lae_repo": Path(paths["models"]["lae_dino_repo"], "README.md").exists(),
            "lae_weights": Path(paths["models"]["lae_dino_weights"]).exists(),
            "georsclip": Path(paths["models"]["georsclip_ckpt"]).exists(),
            "bert": Path(paths["models"]["bert_dir"], "config.json").exists()
            and any(
                Path(paths["models"]["bert_dir"], name).exists()
                for name in ("pytorch_model.bin", "model.safetensors")
            ),
            "xlrs_jsonl": jsonl.exists(),
        },
        **extra,
    }
    dest = fs / "counting_assets_status.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("exists", "xlrs_counting_rows", "autodl_fs")}, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lae", action="store_true")
    parser.add_argument("--skip-geors", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-xlrs", action="store_true")
    parser.add_argument(
        "--xlrs-max",
        type=int,
        default=0,
        help="counting samples to keep; 0 = all counting rows",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    paths = load_paths()
    _prepare_env(paths)
    fs = Path(paths["autodl_fs"])
    tmp = Path(paths["autodl_tmp"])
    print(f"autodl_fs={fs}")
    print(f"autodl_tmp={tmp}")
    fs.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    _ensure_store(Path(paths["models"]["lae_dino_root"]), tmp / "models" / "lae-dino")
    _ensure_store(Path(paths["models"]["georsclip_dir"]), tmp / "models" / "georsclip")
    _ensure_store(Path(paths["datasets"]["xlrsbench_lite"]), tmp / "datasets" / "xlrsbench-lite")

    if not args.skip_lae:
        clone_lae(Path(paths["models"]["lae_dino_repo"]))
        download_lae_weights(Path(paths["models"]["lae_dino_weights"]))
    if not args.skip_bert:
        download_bert(Path(paths["models"]["bert_dir"]))
    if not args.skip_geors:
        download_geors(Path(paths["models"]["georsclip_ckpt"]))
    if not args.skip_xlrs:
        download_xlrs_counting(
            Path(paths["datasets"]["xlrsbench_lite"]),
            max_samples=args.xlrs_max,
            jpeg_quality=args.jpeg_quality,
        )
    write_status(paths, {"xlrs_max": args.xlrs_max})
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility loader for VLM-FO1 in the shared ``rs-vlm`` runtime.

The upstream FO1 builder hard-codes ``flash_attention_2`` and dispatches
through a path-name check.  This module keeps the official model class and
vision-tower setup, while making the attention backend and runtime explicit.
It intentionally performs no work at import time and never downloads model
files implicitly.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_ATTENTION_BACKENDS = ("auto", "sdpa", "flash_attention_2", "eager")


@dataclass(frozen=True)
class FO1ModelBundle:
    """Loaded FO1 components used by ``prepare_inputs`` and generation."""

    tokenizer: Any
    model: Any
    image_processors: Any
    attention_backend: str
    model_path: Path


def ensure_official_root(root: str | Path, *, require_upn: bool = True) -> Path:
    """Validate and prepend an official VLM-FO1 checkout to ``sys.path``.

    FO1 generation with precomputed boxes only needs ``vlm_fo1``.  UPN source
    directories are checked when the UPN proposal provider is selected.
    """

    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"VLM_FO1_ROOT is not a directory: {path}")
    required = ["vlm_fo1"]
    if require_upn:
        required.extend(("detect_tools", "detect_tools/upn"))
    for relative in required:
        if not (path / relative).is_dir():
            raise RuntimeError(f"VLM_FO1_ROOT is missing required directory: {path / relative}")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def validate_model_path(model_path: str | Path) -> Path:
    """Validate a local FO1 model directory before calling any HF API."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"FO1 model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise RuntimeError(f"FO1 model directory is incomplete; missing config.json: {path}")
    return path


def resolve_attention_backend(attention_backend: str) -> str:
    """Resolve an explicit or automatic attention implementation."""

    requested = str(attention_backend or "sdpa").strip().lower()
    if requested not in VALID_ATTENTION_BACKENDS:
        raise ValueError(
            f"unsupported attention backend {requested!r}; "
            f"choose one of {', '.join(VALID_ATTENTION_BACKENDS)}"
        )
    if requested == "auto":
        return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            "attention backend flash_attention_2 was requested but flash_attn is unavailable"
        )
    return requested


def _load_vision_towers(model: Any, model_path: Path, device: str, dtype: Any) -> Any:
    """Load both official FO1 vision towers and return their processors."""

    primary = model.get_vision_tower()
    if primary is not None:
        if not getattr(primary, "is_loaded", False):
            primary.load_model(model_path=str(model_path), is_train=False)
        primary.to(device=device, dtype=dtype)

    auxiliary = model.get_vision_tower_aux()
    if auxiliary is not None:
        config = getattr(model, "config", None)
        image_size = int(getattr(config, "aux_image_size", 768))
        aspect_ratio = getattr(config, "aux_image_aspect_ratio", "pad")
        if not getattr(auxiliary, "is_loaded", False):
            auxiliary.load_model(
                image_size=image_size,
                is_train=False,
                aspect_ratio=aspect_ratio,
            )
        auxiliary.to(device=device, dtype=dtype)

    primary_processor = getattr(primary, "image_processor", None)
    auxiliary_processor = getattr(auxiliary, "image_processor", None)
    return (primary_processor, auxiliary_processor)


def load_fo1_model(
    model_path: str | Path,
    device: str,
    attention_backend: str = "sdpa",
) -> FO1ModelBundle:
    """Load the official FO1 model without requiring an isolated environment.

    The caller must put the official checkout on ``sys.path`` (for example
    with :func:`ensure_official_root`).  ``local_files_only`` is deliberate:
    a typo must fail locally rather than triggering an untracked HF download.
    """

    path = validate_model_path(model_path)
    resolved_backend = resolve_attention_backend(attention_backend)
    official_root = os.environ.get("VLM_FO1_ROOT", "").strip()
    if official_root:
        ensure_official_root(official_root, require_upn=False)
    import torch

    device_text = str(device)
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"shared_rs_vlm requested {device_text}, but CUDA is unavailable; "
            "run the FO1 smoke on a GPU node or choose an explicit CPU unit test"
        )

    from transformers import AutoTokenizer
    from vlm_fo1.model import OmChatQwen25VLForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), use_fast=False, local_files_only=True
    )
    use_cuda = device_text.startswith("cuda")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    load_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "attn_implementation": resolved_backend,
        "local_files_only": True,
    }
    if use_cuda:
        load_kwargs["device_map"] = device_text
    model = OmChatQwen25VLForCausalLM.from_pretrained(str(path), **load_kwargs)
    image_processors = _load_vision_towers(model, path, device_text, dtype)
    model.eval()
    if not use_cuda:
        model.to(device=device_text, dtype=dtype)
    return FO1ModelBundle(
        tokenizer=tokenizer,
        model=model,
        image_processors=image_processors,
        attention_backend=resolved_backend,
        model_path=path,
    )


__all__ = [
    "FO1ModelBundle",
    "VALID_ATTENTION_BACKENDS",
    "ensure_official_root",
    "load_fo1_model",
    "resolve_attention_backend",
    "validate_model_path",
]

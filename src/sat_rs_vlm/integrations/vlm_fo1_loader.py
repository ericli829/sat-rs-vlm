"""Compatibility loader for VLM-FO1 in the shared ``rs-vlm`` runtime.

The upstream FO1 builder hard-codes ``flash_attention_2`` and dispatches
through a path-name check.  This module keeps the official model class and
vision-tower setup, while making the attention backend and runtime explicit.
It intentionally performs no work at import time and never downloads model
files implicitly.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ATTENTION_BACKENDS = ("auto", "sdpa", "flash_attention_2", "eager")

# Transformers 5.x stores Qwen text-model fields in ``config.text_config``;
# the official FO1 implementation was written against the older flat config.
# Keep this list narrow and only promote fields that the vendored model reads.
_LEGACY_TEXT_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
    "layer_types",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "rope_theta",
)


@dataclass(frozen=True)
class FO1ModelBundle:
    """Loaded FO1 components used by ``prepare_inputs`` and generation."""

    tokenizer: Any
    model: Any
    image_processors: Any
    attention_backend: str
    model_path: Path
    config_compatibility_patches: dict[str, dict[str, Any]] = field(default_factory=dict)
    loading_info: dict[str, Any] = field(default_factory=dict)


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


def _read_raw_model_config(path: Path) -> dict[str, Any]:
    """Read the checkpoint JSON without modifying it or passing it to HF."""

    try:
        raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read raw FO1 model config: {path / 'config.json'}") from exc
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"raw FO1 model config must be a JSON object: {path / 'config.json'}")
    return raw_config


def _raw_legacy_field(
    raw_config: Mapping[str, Any] | None, name: str
) -> tuple[str, Any] | None:
    """Return a non-null raw field and its provenance source, if present."""

    if raw_config is None:
        return None
    if name in raw_config and raw_config[name] is not None:
        return f"raw_config.{name}", raw_config[name]
    raw_text_config = raw_config.get("text_config")
    if isinstance(raw_text_config, Mapping):
        if name in raw_text_config and raw_text_config[name] is not None:
            return f"raw_config.text_config.{name}", raw_text_config[name]
    return None


def patch_config_compatibility(
    config: Any,
    tokenizer: Any,
    *,
    raw_config: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply shared-runtime-only compatibility fixes to an in-memory config.

    The official checkpoint is never written to.  The returned structure is
    provenance for the exact in-memory changes made before model loading.
    """

    patches: dict[str, dict[str, Any]] = {}
    if getattr(config, "pad_token_id", None) is None:
        candidates = (
            ("tokenizer.pad_token_id", getattr(tokenizer, "pad_token_id", None)),
            ("tokenizer.eos_token_id", getattr(tokenizer, "eos_token_id", None)),
            ("config.eos_token_id", getattr(config, "eos_token_id", None)),
        )
        for source, value in candidates:
            if value is not None:
                patched_value = int(value)
                config.pad_token_id = patched_value
                patches["pad_token_id"] = {
                    "source": source,
                    "value": patched_value,
                }
                break
        else:
            raise RuntimeError(
                "shared_rs_vlm config is missing pad_token_id and no fallback is available; "
                "tokenizer.pad_token_id, tokenizer.eos_token_id, and config.eos_token_id "
                "are all None"
            )

    # Transformers 5.x nests the language-model fields and can normalize some
    # of them to None.  The official FO1 class still reads them from its
    # top-level config.  Keep a valid top-level value, otherwise prefer the
    # original checkpoint JSON over the normalized nested object, then fall
    # back to the nested value.  This never mutates checkpoint files.
    text_config = getattr(config, "text_config", None)
    promoted: dict[str, dict[str, Any]] = {}
    for name in _LEGACY_TEXT_CONFIG_FIELDS:
        current = getattr(config, name, None)
        if current is not None:
            continue
        raw_field = _raw_legacy_field(raw_config, name)
        if raw_field is not None:
            source, value = raw_field
        else:
            value = getattr(text_config, name, None) if text_config is not None else None
            source = f"config.text_config.{name}"
        if value is not None:
            setattr(config, name, value)
            promoted[name] = {
                "source": source,
                "value": value,
            }

    # Qwen2.5-VL's Transformers 5.x post-init turns sliding_window into None
    # when use_sliding_window=False.  Old FO1 code still expects the attribute
    # to exist; only use the documented legacy default when the raw checkpoint
    # did not carry a value and sliding-window attention is disabled.
    if (
        getattr(config, "sliding_window", None) is None
        and getattr(config, "use_sliding_window", None) is False
    ):
        config.sliding_window = 4096
        promoted["sliding_window"] = {
            "source": "official_fo1_legacy_default",
            "value": 4096,
        }

    if text_config is not None:
        rope_parameters = getattr(text_config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters = dict(rope_parameters)
            try:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            except ImportError:  # pragma: no cover - official runtime provides it
                ROPE_INIT_FUNCTIONS = {}
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type"))
            if rope_type == "default" and "default" not in ROPE_INIT_FUNCTIONS:
                if "linear" in ROPE_INIT_FUNCTIONS:
                    # Transformers 5.x removed the old ``default`` registry
                    # key.  A factor-1 linear RoPE is numerically identical.
                    rope_parameters["rope_type"] = "linear"
                    rope_parameters.setdefault("factor", 1.0)

            current_rope_parameters = getattr(config, "rope_parameters", None)
            if current_rope_parameters is None or rope_type == "default":
                config.rope_parameters = dict(rope_parameters)
                promoted["rope_parameters"] = {
                    "source": "config.text_config.rope_parameters",
                    "value": dict(rope_parameters),
                }
            current_rope_scaling = getattr(config, "rope_scaling", None)
            if current_rope_scaling is None or rope_type == "default":
                config.rope_scaling = dict(rope_parameters)
                promoted["rope_scaling"] = {
                    "source": "config.text_config.rope_parameters",
                    "value": dict(rope_parameters),
                }
    if promoted:
        patches["legacy_text_config_fields"] = promoted
    return patches


_LOADING_INFO_FIELDS = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
)
_LOADING_INFO_PREFIXES = (
    "lm_head",
    "model.embed_tokens",
    "model.layers",
    "model.mm_projector",
    "model.mm_projector_aux",
    "model.object_vp_extractor",
    "model.vision_tower",
    "model.vision_tower_aux",
)


def _loading_info_values(loading_info: Any, name: str) -> list[Any]:
    if isinstance(loading_info, Mapping):
        value = loading_info.get(name, [])
    else:
        value = getattr(loading_info, name, []) if loading_info is not None else []
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _json_safe_loading_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_loading_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_loading_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe_loading_value(item_method())
        except Exception:  # pragma: no cover - diagnostic fallback only
            pass
    return str(value)


def _loading_info_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for name in ("key", "name", "parameter", "param"):
            candidate = value.get(name)
            if isinstance(candidate, str):
                return candidate
    if isinstance(value, (list, tuple)) and value:
        return _loading_info_key(value[0])
    return ""


def summarize_loading_info(loading_info: Any) -> dict[str, Any]:
    """Keep bounded loading diagnostics while retaining complete counts."""

    values = {name: _loading_info_values(loading_info, name) for name in _LOADING_INFO_FIELDS}
    summary = {
        "missing_key_count": len(values["missing_keys"]),
        "unexpected_key_count": len(values["unexpected_keys"]),
        "mismatched_key_count": len(values["mismatched_keys"]),
        "error_msg_count": len(values["error_msgs"]),
    }
    result: dict[str, Any] = {
        name: [_json_safe_loading_value(item) for item in items[:50]]
        for name, items in values.items()
    }
    result["summary"] = summary
    prefix_counts: dict[str, dict[str, int]] = {}
    for prefix in _LOADING_INFO_PREFIXES:
        counts: dict[str, int] = {}
        for name, items in values.items():
            if name == "error_msgs":
                count = sum(prefix in str(item) for item in items)
            else:
                count = sum(
                    _loading_info_key(item) == prefix
                    or _loading_info_key(item).startswith(f"{prefix}.")
                    for item in items
                )
            counts[f"{name[:-1]}_count" if name.endswith("s") else f"{name}_count"] = count
        prefix_counts[prefix] = counts
    result["prefix_counts"] = prefix_counts
    return result


def _set_attention_backend_on_nested_configs(
    config: Any, attention_backend: str, patches: dict[str, dict[str, Any]]
) -> None:
    """Keep Transformers' nested Qwen configs on the requested backend."""

    changed: list[str] = []
    for name, target in (
        ("config", config),
        ("config.text_config", getattr(config, "text_config", None)),
        ("config.vision_config", getattr(config, "vision_config", None)),
    ):
        if target is None:
            continue
        current = getattr(target, "_attn_implementation_internal", None)
        if current != attention_backend:
            setattr(target, "_attn_implementation_internal", attention_backend)
            changed.append(name)
    if changed:
        patches["attention_backend"] = {
            "source": "loader.attention_backend",
            "value": attention_backend,
            "targets": changed,
        }


@contextmanager
def _override_official_vision_attention_backend(attention_backend: str):
    """Redirect FO1's in-memory hard-coded vision-tower backend.

    The official ``Qwen2_5_VlVisionTower.load_model`` passes
    ``flash_attention_2`` unconditionally.  Patch only the class method for
    the duration of model/tower construction; the checkout on disk is never
    changed and the original method is restored even when loading fails.
    """

    if attention_backend == "flash_attention_2":
        yield False
        return
    try:
        from vlm_fo1.model.multimodal_encoder.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )
    except ImportError:  # pragma: no cover - official runtime provides it
        yield False
        return

    original_descriptor = inspect.getattr_static(
        Qwen2_5_VisionTransformerPretrainedModel, "_from_config"
    )
    original_from_config = Qwen2_5_VisionTransformerPretrainedModel._from_config

    def compatible_from_config(cls: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = attention_backend
        return original_from_config(config, *args, **kwargs)

    Qwen2_5_VisionTransformerPretrainedModel._from_config = classmethod(
        compatible_from_config
    )
    try:
        yield True
    finally:
        Qwen2_5_VisionTransformerPretrainedModel._from_config = original_descriptor


@contextmanager
def _override_legacy_tied_weights_keys(model_class: Any):
    """Adapt FO1's list-form tied-weight declaration to Transformers 5.x."""

    try:
        original_descriptor = inspect.getattr_static(model_class, "_tied_weights_keys")
    except AttributeError:
        yield None
        return
    original_value = getattr(model_class, "_tied_weights_keys", None)
    if not isinstance(original_value, list):
        yield None
        return
    mapping = {"lm_head.weight": "model.embed_tokens.weight"}
    model_class._tied_weights_keys = mapping
    try:
        yield mapping
    finally:
        model_class._tied_weights_keys = original_descriptor


def _patch_generation_cache_position(
    model: Any, patches: dict[str, dict[str, Any]]
) -> None:
    """Supply the first cache position expected by FO1's older generation shim."""

    original = getattr(model, "prepare_inputs_for_generation", None)
    if original is None or getattr(original, "_rs_vlm_cache_position_patch", False):
        return

    def compatible_prepare(input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("cache_position") is None and input_ids is not None:
            kwargs["cache_position"] = input_ids.new_tensor(range(input_ids.shape[-1]))
        return original(input_ids, *args, **kwargs)

    compatible_prepare._rs_vlm_cache_position_patch = True
    model.prepare_inputs_for_generation = compatible_prepare
    patches["generation_cache_position"] = {
        "source": "official_prepare_inputs_for_generation.cache_position_none",
        "value": "input_ids.new_arange(sequence_length)",
    }


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

    from transformers import AutoConfig, AutoTokenizer
    from vlm_fo1.model import OmChatQwen25VLForCausalLM

    raw_config = _read_raw_model_config(path)
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), use_fast=False, local_files_only=True
    )
    config_compatibility_patches = patch_config_compatibility(
        config, tokenizer, raw_config=raw_config
    )
    _set_attention_backend_on_nested_configs(
        config, resolved_backend, config_compatibility_patches
    )
    use_cuda = device_text.startswith("cuda")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    load_kwargs: dict[str, Any] = {
        "config": config,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "attn_implementation": resolved_backend,
        "local_files_only": True,
        "output_loading_info": True,
    }
    if use_cuda:
        load_kwargs["device_map"] = device_text
    with (
        _override_official_vision_attention_backend(resolved_backend) as vision_patch_applied,
        _override_legacy_tied_weights_keys(OmChatQwen25VLForCausalLM) as tied_weights_patch,
    ):
        model, raw_loading_info = OmChatQwen25VLForCausalLM.from_pretrained(
            str(path), **load_kwargs
        )
        if tied_weights_patch is not None:
            model._tied_weights_keys = dict(tied_weights_patch)
        image_processors = _load_vision_towers(model, path, device_text, dtype)
    loading_info = summarize_loading_info(raw_loading_info)
    if vision_patch_applied:
        config_compatibility_patches["vision_attention_backend"] = {
            "source": "official_vision_tower.hardcoded_flash_attention_2",
            "value": resolved_backend,
        }
    if tied_weights_patch is not None:
        config_compatibility_patches["tied_weights_keys"] = {
            "source": "official_model._tied_weights_keys_list",
            "value": dict(tied_weights_patch),
        }
    if getattr(config, "text_config", None) is not None:
        _patch_generation_cache_position(model, config_compatibility_patches)
    model.eval()
    if not use_cuda:
        model.to(device=device_text, dtype=dtype)
    return FO1ModelBundle(
        tokenizer=tokenizer,
        model=model,
        image_processors=image_processors,
        attention_backend=resolved_backend,
        model_path=path,
        config_compatibility_patches=config_compatibility_patches,
        loading_info=loading_info,
    )


__all__ = [
    "FO1ModelBundle",
    "VALID_ATTENTION_BACKENDS",
    "ensure_official_root",
    "load_fo1_model",
    "patch_config_compatibility",
    "resolve_attention_backend",
    "summarize_loading_info",
    "validate_model_path",
]

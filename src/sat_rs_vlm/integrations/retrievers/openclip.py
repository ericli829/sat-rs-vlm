"""Lazy OpenCLIP region retriever for GeoRSCLIP/RemoteCLIP/FarSLIP weights."""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cache import RetrievalCache, retrieval_cache_key
from .protocol import RegionXYXY, RetrievalError, RetrievalResult


class OpenCLIPRetrieverProvider:
    provider_name = "openclip"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        value = self.config.get("checkpoint") or self.config.get("model_path")
        if not value:
            raise RetrievalError("OpenCLIP checkpoint must be configured")
        self.checkpoint = Path(str(value)).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise RetrievalError(f"OpenCLIP checkpoint does not exist: {self.checkpoint}")
        self.arch = str(self.config.get("arch", "ViT-B-32"))
        self.device = str(self.config.get("device", "auto"))
        self.batch_size = int(self.config.get("batch_size", 16))
        if self.batch_size < 1:
            raise RetrievalError("OpenCLIP batch_size must be positive")
        self.model_id = str(self.config.get("model_id", self.checkpoint.stem))
        self.cache = (
            RetrievalCache(self.config["cache_dir"]) if self.config.get("cache_dir") else None
        )
        self.decoded_image_cache_size = max(
            0, int(self.config.get("decoded_image_cache_size", 2))
        )
        self.decoded_image_cache_max_pixels = max(
            1, int(self.config.get("decoded_image_cache_max_pixels", 4_000_000))
        )
        self.image_embedding_cache_size = max(
            0, int(self.config.get("image_embedding_cache_size", 1024))
        )
        self.min_loaded_parameter_fraction = float(
            self.config.get("min_loaded_parameter_fraction", 0.95)
        )
        if not 0.0 <= self.min_loaded_parameter_fraction <= 1.0:
            raise RetrievalError("OpenCLIP min_loaded_parameter_fraction must be in [0, 1]")
        self.allowed_missing_key_patterns = self._patterns(
            self.config.get(
                "allowed_missing_key_patterns",
                (r"(^|\.)logit_scale$", r"(^|\.)attn_mask$"),
            ),
            label="allowed_missing_key_patterns",
        )
        self.allowed_unexpected_key_patterns = self._patterns(
            self.config.get("allowed_unexpected_key_patterns", ()),
            label="allowed_unexpected_key_patterns",
        )
        self.parameters = {
            "arch": self.arch,
            "batch_size": self.batch_size,
            "device": self.device,
        }
        self._model = self._preprocess = self._tokenizer = self._torch = None
        self._resolved_device = "cpu"
        self._query_cache: dict[str, Any] = {}
        self._decoded_image_cache: OrderedDict[tuple[str, int, int], Any] = OrderedDict()
        self._image_embedding_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._load_info: dict[str, Any] = {}

    @staticmethod
    def _patterns(value: Any, *, label: str) -> tuple[re.Pattern[str], ...]:
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, Sequence):
            values = tuple(str(item) for item in value)
        else:
            raise RetrievalError(f"OpenCLIP {label} must be a string sequence")
        try:
            return tuple(re.compile(item) for item in values if item)
        except re.error as exc:
            raise RetrievalError(f"OpenCLIP {label} contains an invalid regex: {exc}") from exc

    @staticmethod
    def _matches(key: str, patterns: Sequence[re.Pattern[str]]) -> bool:
        return any(pattern.search(key) is not None for pattern in patterns)

    @staticmethod
    def _value_size(value: Any) -> int:
        try:
            return max(1, int(value.numel()))
        except (AttributeError, TypeError, ValueError):
            return 1

    def _compatibility_report(
        self,
        model: Any,
        state: Mapping[str, Any],
        load_result: Any,
    ) -> dict[str, Any]:
        missing_keys = [str(key) for key in getattr(load_result, "missing_keys", ())]
        unexpected_keys = [str(key) for key in getattr(load_result, "unexpected_keys", ())]
        try:
            model_state = dict(model.state_dict())
        except (AttributeError, TypeError, ValueError) as exc:
            raise RetrievalError(f"OpenCLIP model does not expose state_dict: {exc}") from exc
        try:
            parameter_names = {str(name) for name, _value in model.named_parameters()}
        except (AttributeError, TypeError, ValueError):
            parameter_names = set(model_state)
        parameter_names.intersection_update(model_state)
        if not parameter_names:
            parameter_names = set(model_state)
        loaded_names = parameter_names.difference(missing_keys)
        total_parameter_count = sum(
            self._value_size(model_state[name]) for name in parameter_names
        )
        loaded_parameter_count = sum(
            self._value_size(model_state[name]) for name in loaded_names
        )
        loaded_fraction = (
            loaded_parameter_count / total_parameter_count if total_parameter_count else 0.0
        )
        allowed_missing = [
            key for key in missing_keys if self._matches(key, self.allowed_missing_key_patterns)
        ]
        unallowed_missing = [key for key in missing_keys if key not in allowed_missing]
        allowed_unexpected = [
            key
            for key in unexpected_keys
            if self._matches(key, self.allowed_unexpected_key_patterns)
        ]
        unallowed_unexpected = [
            key for key in unexpected_keys if key not in allowed_unexpected
        ]
        return {
            "checkpoint": str(self.checkpoint),
            "model_id": self.model_id,
            "arch": self.arch,
            "missing_key_count": len(missing_keys),
            "unexpected_key_count": len(unexpected_keys),
            "missing_key_examples": missing_keys[:20],
            "unexpected_key_examples": unexpected_keys[:20],
            "allowed_missing_key_examples": allowed_missing[:20],
            "allowed_unexpected_key_examples": allowed_unexpected[:20],
            "unallowed_missing_key_count": len(unallowed_missing),
            "unallowed_missing_key_examples": unallowed_missing[:20],
            "unallowed_unexpected_key_count": len(unallowed_unexpected),
            "unallowed_unexpected_key_examples": unallowed_unexpected[:20],
            "total_parameter_count": total_parameter_count,
            "loaded_parameter_count": loaded_parameter_count,
            "loaded_parameter_fraction": loaded_fraction,
            "minimum_loaded_parameter_fraction": self.min_loaded_parameter_fraction,
            "compatibility_status": (
                "compatible"
                if (
                    not unallowed_missing
                    and not unallowed_unexpected
                    and loaded_fraction >= self.min_loaded_parameter_fraction
                )
                else "incompatible"
            ),
        }

    @property
    def compatibility_report(self) -> dict[str, Any]:
        return dict(self._load_info)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RetrievalError("OpenCLIP provider requires open_clip_torch") from exc
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(self.arch, pretrained=None)
            checkpoint = torch.load(str(self.checkpoint), map_location="cpu")
            state = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, Mapping)
                else checkpoint
            )
            if not isinstance(state, Mapping):
                raise RetrievalError("OpenCLIP checkpoint state_dict must be a mapping")
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
            load_result = model.load_state_dict(state, strict=False)
            self._load_info = self._compatibility_report(model, state, load_result)
            if self._load_info["compatibility_status"] != "compatible":
                raise RetrievalError(
                    "incompatible OpenCLIP checkpoint: "
                    "loaded_parameter_fraction="
                    f"{self._load_info['loaded_parameter_fraction']:.6f}, "
                    f"missing={self._load_info['missing_key_count']}, "
                    f"unexpected={self._load_info['unexpected_key_count']}"
                )
            tokenizer = open_clip.get_tokenizer(self.arch)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"failed to load OpenCLIP checkpoint: {exc}") from exc
        self._torch, self._model, self._preprocess, self._tokenizer = (
            torch,
            model.to(device).eval(),
            preprocess,
            tokenizer,
        )
        self._resolved_device = device

    @staticmethod
    def _crop(image: Any, region: RegionXYXY, index: int) -> tuple[Any, list[float]]:
        try:
            values = [float(value) for value in region]
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"invalid OpenCLIP region at index {index}") from exc
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise RetrievalError(f"invalid OpenCLIP region at index {index}")
        width, height = image.size
        values = [
            max(0.0, min(values[0], width)),
            max(0.0, min(values[1], height)),
            max(0.0, min(values[2], width)),
            max(0.0, min(values[3], height)),
        ]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise RetrievalError(f"degenerate OpenCLIP region at index {index}")
        box = tuple(
            math.floor(values[position]) if position < 2 else math.ceil(values[position])
            for position in range(4)
        )
        return image.crop(box).convert("RGB"), values

    def _decoded_image(self, path: Path) -> tuple[Any, bool, tuple[str, int, int]]:
        from PIL import Image

        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        cached = self._decoded_image_cache.get(key)
        if cached is not None:
            self._decoded_image_cache.move_to_end(key)
            return cached, True, key
        with Image.open(path) as source:
            image = source.convert("RGB")
        cacheable = (
            self.decoded_image_cache_size > 0
            and image.width * image.height <= self.decoded_image_cache_max_pixels
        )
        if cacheable:
            self._decoded_image_cache[key] = image
            while len(self._decoded_image_cache) > self.decoded_image_cache_size:
                self._decoded_image_cache.popitem(last=False)
        return image, False, key

    def _embedding_cache_get(self, key: tuple[Any, ...]) -> Any | None:
        value = self._image_embedding_cache.get(key)
        if value is not None:
            self._image_embedding_cache.move_to_end(key)
        return value

    def _embedding_cache_put(self, key: tuple[Any, ...], value: Any) -> None:
        if not self.image_embedding_cache_size:
            return
        self._image_embedding_cache[key] = value.detach().cpu()
        self._image_embedding_cache.move_to_end(key)
        while len(self._image_embedding_cache) > self.image_embedding_cache_size:
            self._image_embedding_cache.popitem(last=False)

    def score_regions(
        self, image_path: Path, query: str, regions_xyxy: Sequence[RegionXYXY]
    ) -> RetrievalResult:
        started = time.perf_counter()
        query = str(query).strip()
        if not query:
            raise RetrievalError("OpenCLIP query must not be empty")
        resolved = Path(image_path).expanduser().resolve()
        if not resolved.is_file():
            raise RetrievalError(f"OpenCLIP image does not exist: {resolved}")
        regions = list(regions_xyxy)
        if not regions:
            return RetrievalResult([], 0.0, self.provider_name, self.model_id, {"batch_size": 0})
        image, decoded_cache_hit, image_identity = self._decoded_image(resolved)
        canonical = [self._crop(image, box, index) for index, box in enumerate(regions)]
        boxes = [item[1] for item in canonical]
        scores: list[float | None] = [None] * len(boxes)
        score_keys: list[str | None] = [None] * len(boxes)
        score_cache_hits = 0
        if self.cache is not None:
            for index, box in enumerate(boxes):
                key = retrieval_cache_key(
                    image_path=resolved,
                    region_xyxy=box,
                    query=query,
                    provider=self.provider_name,
                    model_identity={
                        "checkpoint": str(self.checkpoint),
                        "model_id": self.model_id,
                    },
                    parameters=self.parameters,
                )
                score_keys[index] = key
                cached_score = self.cache.get(key)
                if cached_score is not None:
                    scores[index] = cached_score
                    score_cache_hits += 1
        missing = [index for index, value in enumerate(scores) if value is None]
        if not missing:
            return RetrievalResult(
                [float(value) for value in scores],
                (time.perf_counter() - started) * 1000.0,
                self.provider_name,
                self.model_id,
                {
                    "regions_xyxy": boxes,
                    "batch_size": self.batch_size,
                    "crop_batch_count": 0,
                    "query_cache_hit": query in self._query_cache,
                    "score_cache_hits": score_cache_hits,
                    "decoded_image_cache_hit": decoded_cache_hit,
                    "image_embedding_cache_hits": 0,
                    "device": self._resolved_device,
                    "generation_used": False,
                },
            ).validate_length(len(regions))

        self._load()
        torch = self._torch
        assert (
            torch is not None
            and self._model is not None
            and self._preprocess is not None
            and self._tokenizer is not None
        )
        query_cache_hit = query in self._query_cache
        if not query_cache_hit:
            tokens = self._tokenizer([query]).to(self._resolved_device)
            with torch.inference_mode():
                self._query_cache[query] = torch.nn.functional.normalize(
                    self._model.encode_text(tokens), dim=-1
                )
        query_embedding = self._query_cache[query]
        embedding_keys = [
            (*image_identity, *(round(value, 6) for value in box)) for box in boxes
        ]
        image_embeddings: dict[int, Any] = {}
        embedding_cache_hits = 0
        uncached: list[int] = []
        for index in missing:
            cached_embedding = self._embedding_cache_get(embedding_keys[index])
            if cached_embedding is None:
                uncached.append(index)
            else:
                image_embeddings[index] = cached_embedding
                embedding_cache_hits += 1
        batches = 0
        for offset in range(0, len(uncached), self.batch_size):
            indices = uncached[offset : offset + self.batch_size]
            batch = [self._preprocess(canonical[index][0]) for index in indices]
            with torch.inference_mode():
                images = torch.stack(batch).to(self._resolved_device)
                embeddings = torch.nn.functional.normalize(
                    self._model.encode_image(images), dim=-1
                )
            batches += 1
            for index, value in zip(indices, embeddings, strict=True):
                image_embeddings[index] = value.detach().cpu()
                self._embedding_cache_put(embedding_keys[index], value)
        stacked = torch.stack([image_embeddings[index] for index in missing]).to(
            self._resolved_device
        )
        values = torch.matmul(query_embedding, stacked.T)[0].float().cpu().tolist()
        for index, value in zip(missing, values, strict=True):
            scores[index] = float(value)
            if self.cache is not None and score_keys[index] is not None:
                self.cache.put(score_keys[index] or "", float(value))
        final = [float(value) for value in scores]
        return RetrievalResult(
            final,
            (time.perf_counter() - started) * 1000.0,
            self.provider_name,
            self.model_id,
            {
                "raw_scores": final,
                "regions_xyxy": boxes,
                "batch_size": self.batch_size,
                "crop_batch_count": batches,
                "query_cache_hit": query_cache_hit,
                "score_cache_hits": score_cache_hits,
                    "decoded_image_cache_hit": decoded_cache_hit,
                    "decoded_image_cache_size": len(self._decoded_image_cache),
                    "decoded_image_cache_max_pixels": self.decoded_image_cache_max_pixels,
                "image_embedding_cache_hits": embedding_cache_hits,
                "image_embedding_cache_size": len(self._image_embedding_cache),
                "load_info": dict(self._load_info),
                "device": self._resolved_device,
                "generation_used": False,
            },
        ).validate_length(len(regions))

    def close(self) -> None:
        self._query_cache.clear()
        self._decoded_image_cache.clear()
        self._image_embedding_cache.clear()
        self._model = self._preprocess = self._tokenizer = self._torch = None

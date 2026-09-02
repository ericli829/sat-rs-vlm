"""Lazy HuggingFace CLIP-style region retriever.

The provider supports CLIP, SigLIP, and Transformers-packaged remote-sensing
checkpoints that expose ``get_text_features`` and ``get_image_features``.
Model frameworks and weights are loaded only on the first uncached request.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cache import RetrievalCache, retrieval_cache_key
from .config import resolve_config_path
from .protocol import RegionXYXY, RetrievalError, RetrievalResult


class CLIPRetrieverProvider:
    """Score image regions against a text query with a CLIP-compatible model."""

    provider_name = "clip"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model_path = resolve_config_path(self.config.get("model_path"), label="model_path")
        if not self.model_path.exists():
            raise RetrievalError(f"CLIP model path does not exist: {self.model_path}")
        self.model_id = str(self.config.get("model_id", self.model_path.name))
        self.device = str(self.config.get("device", "auto"))
        self.batch_size = int(self.config.get("batch_size", 16))
        if self.batch_size < 1:
            raise RetrievalError("CLIP batch_size must be positive")
        self.trust_remote_code = bool(self.config.get("trust_remote_code", True))
        cache_dir = self.config.get("cache_dir")
        self.cache = RetrievalCache(cache_dir) if cache_dir else None
        self.parameters = {
            "batch_size": self.batch_size,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
        }
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._resolved_device = "cpu"
        self._query_cache: dict[str, Any] = {}
        self._model_load_ms = 0.0

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RetrievalError(
                "CLIP retriever requires torch and transformers; install the model extra"
            ) from exc
        resolved = "cuda:0" if self.device == "auto" and torch.cuda.is_available() else self.device
        if resolved == "auto":
            resolved = "cpu"
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
            model = (
                AutoModel.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
                )
                .to(resolved)
                .eval()
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"failed to load CLIP model {self.model_path}: {exc}") from exc
        self._torch, self._processor, self._model = torch, processor, model
        self._resolved_device = resolved
        self._model_load_ms = (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _normalize(values: Any, torch: Any) -> Any:
        if not hasattr(values, "shape"):
            for name in (
                "text_embeds",
                "image_embeds",
                "pooler_output",
                "embeds",
                "last_hidden_state",
            ):
                candidate = getattr(values, name, None)
                if candidate is not None:
                    values = candidate
                    break
        if not hasattr(values, "shape"):
            raise RetrievalError("CLIP model returned no tensor embedding")
        if values.ndim == 3:
            values = values.mean(dim=1)
        return torch.nn.functional.normalize(values, p=2, dim=-1)

    def _encode_text(self, query: str) -> Any:
        assert self._torch is not None and self._processor is not None and self._model is not None
        torch = self._torch
        batch = self._processor(text=[query], return_tensors="pt", padding=True)
        batch = {
            key: value.to(self._resolved_device)
            for key, value in batch.items()
            if hasattr(value, "to")
        }
        with torch.inference_mode():
            if hasattr(self._model, "get_text_features"):
                embedding = self._model.get_text_features(**batch)
            else:
                output = self._model(**batch)
                embedding = getattr(output, "text_embeds", None)
                if embedding is None:
                    raise RetrievalError("CLIP model does not expose text features")
        return self._normalize(embedding, torch)

    def _encode_images(self, images: Sequence[Any]) -> Any:
        assert self._torch is not None and self._processor is not None and self._model is not None
        torch = self._torch
        batch = self._processor(images=list(images), return_tensors="pt")
        batch = {
            key: value.to(self._resolved_device)
            for key, value in batch.items()
            if hasattr(value, "to")
        }
        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                embedding = self._model.get_image_features(**batch)
            else:
                output = self._model(**batch)
                embedding = getattr(output, "image_embeds", None)
                if embedding is None:
                    raise RetrievalError("CLIP model does not expose image features")
        return self._normalize(embedding, torch)

    @staticmethod
    def _canonical_region(image: Any, region: RegionXYXY, index: int) -> tuple[Any, list[float]]:
        try:
            values = [float(value) for value in region]
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"invalid CLIP region at index {index}") from exc
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise RetrievalError(f"invalid CLIP region at index {index}")
        width, height = image.size
        values = [
            max(0.0, min(values[0], width)),
            max(0.0, min(values[1], height)),
            max(0.0, min(values[2], width)),
            max(0.0, min(values[3], height)),
        ]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise RetrievalError(f"degenerate CLIP region at index {index}")
        box = (
            math.floor(values[0]),
            math.floor(values[1]),
            math.ceil(values[2]),
            math.ceil(values[3]),
        )
        return image.crop(box).convert("RGB"), values

    def score_regions(
        self, image_path: Path, query: str, regions_xyxy: Sequence[RegionXYXY]
    ) -> RetrievalResult:
        from PIL import Image

        started = time.perf_counter()
        query = str(query).strip()
        if not query:
            raise RetrievalError("CLIP query must not be empty")
        resolved = Path(image_path).expanduser().resolve()
        if not resolved.is_file():
            raise RetrievalError(f"CLIP image does not exist: {resolved}")
        regions = list(regions_xyxy)
        if not regions:
            return RetrievalResult([], 0.0, self.provider_name, self.model_id, {"batch_size": 0})
        with Image.open(resolved) as source:
            rgb = source.convert("RGB")
            canonical = [self._canonical_region(rgb, box, i) for i, box in enumerate(regions)]
        crops, boxes = [item[0] for item in canonical], [item[1] for item in canonical]
        scores: list[float | None] = [None] * len(boxes)
        keys: list[str | None] = [None] * len(boxes)
        cache_hits = 0
        if self.cache is not None:
            for index, box in enumerate(boxes):
                key = retrieval_cache_key(
                    image_path=resolved,
                    region_xyxy=box,
                    query=query,
                    provider=self.provider_name,
                    model_identity={"path": str(self.model_path), "model_id": self.model_id},
                    parameters=self.parameters,
                )
                keys[index] = key
                value = self.cache.get(key)
                if value is not None:
                    scores[index], cache_hits = value, cache_hits + 1
        missing = [index for index, value in enumerate(scores) if value is None]
        query_cache_hit = False
        batches = 0
        if missing:
            self._load()
            assert self._torch is not None
            query_embedding = self._query_cache.get(query)
            if query_embedding is None:
                query_embedding = self._encode_text(query)
                self._query_cache[query] = query_embedding
            else:
                query_cache_hit = True
            for offset in range(0, len(missing), self.batch_size):
                indices = missing[offset : offset + self.batch_size]
                batches += 1
                image_embeddings = self._encode_images([crops[index] for index in indices])
                values = torch_matmul(self._torch, query_embedding, image_embeddings)
                for index, value in zip(indices, values, strict=True):
                    scores[index] = float(value)
                    if self.cache is not None and keys[index] is not None:
                        self.cache.put(keys[index] or "", float(value))
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
                "score_cache_hits": cache_hits,
                "query_cache_size": len(self._query_cache),
                "model_load_ms": self._model_load_ms,
                "device": self._resolved_device,
                "generation_used": False,
            },
        ).validate_length(len(regions))

    def close(self) -> None:
        self._query_cache.clear()
        self._model = self._processor = self._torch = None


def torch_matmul(torch: Any, query: Any, images: Any) -> list[float]:
    return torch.matmul(query, images.T)[0].detach().float().cpu().tolist()

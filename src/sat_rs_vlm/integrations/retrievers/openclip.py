"""Lazy OpenCLIP provider for GeoRSCLIP/RemoteCLIP/FarSLIP checkpoints."""

from __future__ import annotations

import math
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
        self.image_embedding_cache_size = max(
            0, int(self.config.get("image_embedding_cache_size", 1024))
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

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover
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
            state = {key.removeprefix("module."): value for key, value in state.items()}
            load_result = model.load_state_dict(state, strict=False)
            self._load_info = {
                "missing_keys": len(load_result.missing_keys),
                "unexpected_keys": len(load_result.unexpected_keys),
            }
            tokenizer = open_clip.get_tokenizer(self.arch)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"failed to load OpenCLIP checkpoint: {exc}") from exc
        self._torch, self._model, self._preprocess, self._tokenizer = (
            torch,
            model.to(device).eval(),
            preprocess,
            tokenizer,
        )
        self._resolved_device = device
        self._model_load_ms = (time.perf_counter() - started) * 1000.0

    def preload(self) -> None:
        self._load()

    @property
    def telemetry_model_load_ms(self) -> float | None:
        return getattr(self, "_model_load_ms", None)

    @staticmethod
    def _crop(image: Any, region: RegionXYXY, index: int) -> tuple[Any, list[float]]:
        values = [float(value) for value in region]
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
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
        return image.crop(
            tuple(math.floor(values[i]) if i < 2 else math.ceil(values[i]) for i in range(4))
        ).convert("RGB"), values

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
        if self.decoded_image_cache_size:
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
        if not str(query).strip():
            raise RetrievalError("OpenCLIP query must not be empty")
        resolved = Path(image_path).expanduser().resolve()
        if not resolved.is_file():
            raise RetrievalError(f"OpenCLIP image does not exist: {resolved}")
        regions = list(regions_xyxy)
        if not regions:
            return RetrievalResult([], 0.0, self.provider_name, self.model_id, {"batch_size": 0})
        image, decoded_cache_hit, image_identity = self._decoded_image(resolved)
        canonical = [self._crop(image, box, i) for i, box in enumerate(regions)]
        boxes = [item[1] for item in canonical]
        scores: list[float | None] = [None] * len(boxes)
        score_keys: list[str | None] = [None] * len(boxes)
        score_cache_hits = 0
        if self.cache is not None:
            for index, box in enumerate(boxes):
                key = retrieval_cache_key(
                    image_path=resolved,
                    region_xyxy=box,
                    query=str(query).strip(),
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
                    "crop_count": len(boxes),
                    "batch_size": self.batch_size,
                    "crop_batch_count": 0,
                    "query_cache_hit": str(query).strip() in self._query_cache,
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
        q = str(query).strip()
        query_cache_hit = q in self._query_cache
        if not query_cache_hit:
            tokens = self._tokenizer([q]).to(self._resolved_device)
            with torch.inference_mode():
                self._query_cache[q] = torch.nn.functional.normalize(
                    self._model.encode_text(tokens), dim=-1
                )
        query_embedding = self._query_cache[q]
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
                emb = torch.nn.functional.normalize(self._model.encode_image(images), dim=-1)
            batches += 1
            for index, value in zip(indices, emb, strict=True):
                image_embeddings[index] = value.detach().cpu()
                self._embedding_cache_put(embedding_keys[index], value)
        if missing:
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
                "crop_count": len(boxes),
                "batch_size": self.batch_size,
                "crop_batch_count": batches,
                "query_cache_hit": query_cache_hit,
                "score_cache_hits": score_cache_hits,
                "decoded_image_cache_hit": decoded_cache_hit,
                "decoded_image_cache_size": len(self._decoded_image_cache),
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

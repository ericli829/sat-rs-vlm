"""Lazy local-only VisRAG-Ret adapter for region relevance scoring."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cache import RetrievalCache, retrieval_cache_key
from .config import resolve_config_path
from .protocol import RegionXYXY, RetrievalError, RetrievalResult

_OFFICIAL_QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "


def _representations(output: Any, torch: Any) -> Any:
    """Return normalized embeddings using the official VisRAG-Ret pooling rule."""

    if hasattr(output, "reps"):
        representations = output.reps
    elif isinstance(output, Mapping) and "reps" in output:
        representations = output["reps"]
    else:
        hidden = (
            output.get("last_hidden_state")
            if isinstance(output, Mapping)
            else getattr(output, "last_hidden_state", None)
        )
        attention_mask = (
            output.get("attention_mask")
            if isinstance(output, Mapping)
            else getattr(output, "attention_mask", None)
        )
        if hidden is None or attention_mask is None:
            raise RetrievalError(
                "VisRAG runtime did not return reps or last_hidden_state/attention_mask"
            )
        weighted_mask = attention_mask * attention_mask.cumsum(dim=1)
        numerator = torch.sum(
            hidden * weighted_mask.unsqueeze(-1).float(),
            dim=1,
        )
        denominator = weighted_mask.sum(dim=1, keepdim=True).float()
        if bool(torch.any(denominator <= 0).item()):
            raise RetrievalError("VisRAG returned an empty attention mask")
        representations = numerator / denominator
    return torch.nn.functional.normalize(representations, p=2, dim=1)


class VisRAGDirectRetrieverProvider:
    """Score ``(question, crop)`` pairs without generation or a trained head."""

    provider_name = "visrag"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model_path = resolve_config_path(self.config.get("model_path"), label="model_path")
        if not self.model_path.is_dir():
            raise RetrievalError(f"VisRAG model path does not exist: {self.model_path}")
        repo_path = self.config.get("repo_path")
        self.repo_path = (
            resolve_config_path(repo_path, label="repo_path") if repo_path else None
        )
        if self.repo_path is not None and not self.repo_path.is_dir():
            raise RetrievalError(f"VisRAG repository path does not exist: {self.repo_path}")
        self.device = str(self.config.get("device", "auto"))
        self.precision = str(self.config.get("precision", "bfloat16"))
        self.batch_size = int(self.config.get("batch_size", 8))
        if self.batch_size < 1:
            raise RetrievalError("VisRAG batch_size must be positive")
        self.model_id = str(self.config.get("model_id", self.model_path.name))
        self.query_instruction = str(
            self.config.get("query_instruction", _OFFICIAL_QUERY_INSTRUCTION)
        )
        self.parameters = {
            "precision": self.precision,
            "batch_size": self.batch_size,
            "trust_remote_code": bool(self.config.get("trust_remote_code", True)),
            "query_instruction": self.query_instruction,
        }
        self.model_identity = {
            "path": str(self.model_path),
            "model_id": self.model_id,
        }
        cache_dir = self.config.get("cache_dir")
        self.cache = RetrievalCache(cache_dir) if cache_dir else None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._resolved_device = "cpu"
        self._model_load_ms = 0.0
        self._query_cache: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        if self.repo_path is not None and str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RetrievalError(
                "VisRAG requires torch and transformers in its runtime environment"
            ) from exc
        if self.device == "auto":
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = self.device
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.precision not in dtype_by_name:
            raise RetrievalError(f"unsupported VisRAG precision: {self.precision!r}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            local_files_only=True,
            torch_dtype=dtype_by_name[self.precision],
        )
        model = model.to(resolved_device).eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device
        self._model_load_ms = (time.perf_counter() - started) * 1000.0

    def _query_embedding(self, query: str) -> tuple[Any, bool]:
        normalized = query.strip()
        if not normalized:
            raise RetrievalError("VisRAG query must not be empty")
        cached = self._query_cache.get(normalized)
        if cached is not None:
            return cached, True
        assert self._model is not None and self._tokenizer is not None
        assert self._torch is not None
        prompted_query = f"{self.query_instruction}{normalized}"
        output = self._model(text=[prompted_query], image=[None], tokenizer=self._tokenizer)
        representation = _representations(output, self._torch)
        self._query_cache[normalized] = representation
        return representation, False

    @staticmethod
    def _canonical_crop(image: Any, region: RegionXYXY, index: int) -> tuple[Any, list[float]]:
        try:
            values = [float(value) for value in region]
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"invalid VisRAG region at index {index}") from exc
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise RetrievalError(f"invalid VisRAG region at index {index}")
        width, height = image.size
        values = [
            min(max(values[0], 0.0), float(width)),
            min(max(values[1], 0.0), float(height)),
            min(max(values[2], 0.0), float(width)),
            min(max(values[3], 0.0), float(height)),
        ]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise RetrievalError(f"degenerate VisRAG region at index {index}")
        crop_box = (
            math.floor(values[0]),
            math.floor(values[1]),
            math.ceil(values[2]),
            math.ceil(values[3]),
        )
        return image.crop(crop_box).convert("RGB"), values

    def score_regions(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> RetrievalResult:
        from PIL import Image

        started = time.perf_counter()
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise RetrievalError(f"VisRAG image does not exist: {resolved_image}")
        regions = list(regions_xyxy)
        if not regions:
            return RetrievalResult(
                scores=[],
                latency_ms=0.0,
                provider=self.provider_name,
                model_id=self.model_id,
                metadata={"raw_scores": [], "batch_size": 0},
            )
        with Image.open(resolved_image) as source:
            rgb_image = source.convert("RGB")
            canonical = [
                self._canonical_crop(rgb_image, region, index)
                for index, region in enumerate(regions)
            ]
        crops = [item[0] for item in canonical]
        boxes = [item[1] for item in canonical]

        scores: list[float | None] = [None] * len(boxes)
        keys: list[str | None] = [None] * len(boxes)
        cache_hits = 0
        if self.cache is not None:
            for index, box in enumerate(boxes):
                key = retrieval_cache_key(
                    image_path=resolved_image,
                    region_xyxy=box,
                    query=query,
                    provider=self.provider_name,
                    model_identity=self.model_identity,
                    parameters=self.parameters,
                )
                keys[index] = key
                cached_score = self.cache.get(key)
                if cached_score is not None:
                    scores[index] = cached_score
                    cache_hits += 1

        missing = [index for index, score in enumerate(scores) if score is None]
        query_cache_hit = False
        peak_memory = None
        if missing:
            self._load()
            assert self._torch is not None
            assert self._model is not None and self._tokenizer is not None
            torch = self._torch
            if self._resolved_device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                query_embedding, query_cache_hit = self._query_embedding(query)
                for offset in range(0, len(missing), self.batch_size):
                    batch_indices = missing[offset : offset + self.batch_size]
                    batch_crops = [crops[index] for index in batch_indices]
                    output = self._model(
                        text=[""] * len(batch_crops),
                        image=batch_crops,
                        tokenizer=self._tokenizer,
                    )
                    crop_embeddings = _representations(output, torch)
                    batch_scores = torch.matmul(query_embedding, crop_embeddings.T)[0]
                    for index, score in zip(
                        batch_indices,
                        batch_scores.detach().float().cpu().tolist(),
                        strict=True,
                    ):
                        scores[index] = float(score)
                        if self.cache is not None and keys[index] is not None:
                            self.cache.put(keys[index] or "", float(score))
            if self._resolved_device.startswith("cuda"):
                peak_memory = int(torch.cuda.max_memory_allocated())

        final_scores = [float(score) for score in scores if score is not None]
        result = RetrievalResult(
            scores=final_scores,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.provider_name,
            model_id=self.model_id,
            metadata={
                "raw_scores": final_scores,
                "regions_xyxy": boxes,
                "batch_size": self.batch_size,
                "query_cache_hit": query_cache_hit,
                "score_cache_hits": cache_hits,
                "model_load_ms": self._model_load_ms,
                "device": self._resolved_device,
                "peak_cuda_memory_bytes": peak_memory,
                "generation_used": False,
                "runtime": "direct",
            },
        )
        return result.validate_length(len(regions))

    def close(self) -> None:
        self._query_cache.clear()
        self._tokenizer = None
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None


class _VisRAGSidecarClient:
    """Long-lived JSONL client for an isolated official VisRAG environment."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        worker = str(self.config.get("worker_python") or sys.executable)
        worker_script_value = self.config.get("worker_script")
        self.worker_script = (
            Path(str(worker_script_value)).expanduser().resolve()
            if worker_script_value
            else Path(__file__).resolve().parents[4]
            / "scripts/integrations/visrag_worker.py"
        )
        if not self.worker_script.is_file():
            raise RetrievalError(f"VisRAG sidecar worker does not exist: {self.worker_script}")
        worker_config = {
            key: value
            for key, value in self.config.items()
            if key
            not in {
                "runtime",
                "worker_python",
                "worker_script",
                "stderr_log",
            }
        }
        self.command = [
            worker,
            str(self.worker_script),
            "--config-json",
            json.dumps(worker_config, ensure_ascii=False, separators=(",", ":")),
        ]
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        self.process: subprocess.Popen[str] | None = None
        self.stderr_path = Path(
            str(
                self.config.get(
                    "stderr_log",
                    Path(tempfile.gettempdir())
                    / f"visrag_sidecar_{uuid.uuid4().hex}.stderr.log",
                )
            )
        ).expanduser().resolve()
        self._stderr_handle: Any = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = self.stderr_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                text=True,
                bufsize=1,
                env=self.environment,
            )
        except OSError:
            self._stderr_handle.close()
            self._stderr_handle = None
            raise
        if self.process.stdin is None or self.process.stdout is None:
            raise RetrievalError("failed to open VisRAG sidecar pipes")

    def request(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> dict[str, Any]:
        self.start()
        assert self.process is not None
        if self.process.poll() is not None:
            raise RetrievalError(
                f"VisRAG sidecar exited with code {self.process.returncode}; "
                f"stderr_log={self.stderr_path}"
            )
        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "image": str(Path(image_path).expanduser().resolve()),
            "query": query,
            "regions_xyxy": [[float(value) for value in box] for box in regions_xyxy],
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RetrievalError(f"VisRAG sidecar pipe failed: {exc}") from exc
        if not line:
            raise RetrievalError(
                "VisRAG sidecar exited before returning a response; "
                f"stderr_log={self.stderr_path}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalError(
                f"VisRAG sidecar emitted invalid JSON: {exc}; "
                f"stderr_log={self.stderr_path}"
            ) from exc
        if not isinstance(response, dict):
            raise RetrievalError("VisRAG sidecar response must be an object")
        if response.get("status") != "ok":
            raise RetrievalError(
                f"VisRAG sidecar failure at {response.get('failure_stage')}: "
                f"{response.get('error')}; stderr_log={self.stderr_path}"
            )
        if response.get("id") != request_id:
            raise RetrievalError("VisRAG sidecar response id mismatch")
        return response

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.flush()
                self._stderr_handle.close()
            except OSError:
                pass
            self._stderr_handle = None
        self.process = None


class VisRAGSidecarProvider:
    provider_name = "visrag"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model_path = resolve_config_path(self.config.get("model_path"), label="model_path")
        if not self.model_path.is_dir():
            raise RetrievalError(f"VisRAG model path does not exist: {self.model_path}")
        self.model_id = str(self.config.get("model_id", self.model_path.name))
        self._client = _VisRAGSidecarClient(self.config)
        self._has_scored = False

    @property
    def is_loaded(self) -> bool:
        return self._has_scored

    def score_regions(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> RetrievalResult:
        regions = list(regions_xyxy)
        response = self._client.request(image_path, query, regions)
        self._has_scored = True
        payload = response.get("result")
        if not isinstance(payload, Mapping):
            raise RetrievalError("VisRAG sidecar response is missing result")
        metadata = dict(payload.get("metadata", {}))
        metadata.update(
            {
                "runtime": "sidecar",
                "stderr_log": str(self._client.stderr_path),
            }
        )
        return RetrievalResult(
            scores=list(payload.get("scores", [])),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            provider=str(payload.get("provider", self.provider_name)),
            model_id=str(payload.get("model_id", self.model_id)),
            metadata=metadata,
        ).validate_length(len(regions))

    def close(self) -> None:
        self._client.close()
        self._has_scored = False


class VisRAGRetrieverProvider:
    """Facade selecting direct or dependency-isolated VisRAG execution."""

    provider_name = "visrag"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.runtime = str(self.config.get("runtime", "direct")).strip().lower()
        if self.runtime == "direct":
            self._delegate: VisRAGDirectRetrieverProvider | VisRAGSidecarProvider = (
                VisRAGDirectRetrieverProvider(self.config)
            )
        elif self.runtime == "sidecar":
            self._delegate = VisRAGSidecarProvider(self.config)
        else:
            raise RetrievalError("VisRAG runtime must be 'direct' or 'sidecar'")
        self.model_id = self._delegate.model_id
        self.query_instruction = str(
            self.config.get("query_instruction", _OFFICIAL_QUERY_INSTRUCTION)
        )

    @property
    def is_loaded(self) -> bool:
        return self._delegate.is_loaded

    def score_regions(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> RetrievalResult:
        return self._delegate.score_regions(image_path, query, regions_xyxy)

    def close(self) -> None:
        self._delegate.close()

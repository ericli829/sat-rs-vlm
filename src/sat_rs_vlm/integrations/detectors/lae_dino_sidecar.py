"""JSONL client for LAE-DINO's isolated MMDetection environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import (
    ProposalError,
    ProposalResult,
    canonicalize_proposals,
    stable_file_identity,
)

PINNED_LAE_DINO_SOURCE_REVISION = "6b1519626e39d1f39f8ed1f38761c20f7e0e8c35"


class SidecarProtocolError(ProposalError):
    """A machine-readable LAE sidecar protocol failure."""

    failure_stage = "worker_protocol"


class _LAESidecarClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        source_root = self.config.get("source_root")
        config_path = self.config.get("config_path") or self.config.get("config")
        checkpoint = self.config.get("checkpoint")
        if not source_root or not config_path or not checkpoint:
            raise ProposalError(
                "LAE-DINO requires proposal.source_root, proposal.config_path, "
                "and proposal.checkpoint"
            )
        self.source_root = Path(str(source_root)).expanduser().resolve()
        self.config_path = Path(str(config_path)).expanduser().resolve()
        self.checkpoint = Path(str(checkpoint)).expanduser().resolve()
        if not self.config.get("bert_root"):
            raise ProposalError("LAE-DINO requires proposal.bert_root for offline BERT loading")
        self.bert_root = (
            Path(str(self.config["bert_root"])).expanduser().resolve()
            if self.config.get("bert_root")
            else None
        )
        for path, label in (
            (self.source_root, "LAE-DINO source_root"),
            (self.config_path, "LAE-DINO config_path"),
            (self.checkpoint, "LAE-DINO checkpoint"),
        ):
            if not path.exists():
                raise ProposalError(f"{label} does not exist: {path}")
        if self.bert_root is not None and not self.bert_root.is_dir():
            raise ProposalError(f"LAE-DINO bert_root does not exist: {self.bert_root}")
        worker = self.config.get("worker_python") or sys.executable
        worker_script = self.config.get("worker_script")
        if worker_script:
            self.worker_script = Path(str(worker_script)).expanduser().resolve()
        else:
            # ``__file__`` is under ``<repo>/src/sat_rs_vlm/integrations/detectors``;
            # parents[4] is the repository root (not ``<repo>/src``).
            self.worker_script = (
                Path(__file__).resolve().parents[4]
                / "scripts/integrations/lae_dino_worker.py"
            )
        if not self.worker_script.is_file():
            raise ProposalError(f"LAE-DINO sidecar worker does not exist: {self.worker_script}")
        self.command = [
            str(worker),
            str(self.worker_script),
            "--source-root",
            str(self.source_root),
            "--config",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint),
            "--device",
            str(self.config.get("device", "cuda")),
            "--score-threshold",
            str(self.config.get("score_threshold", 0.3)),
            "--top-k",
            str(self.config.get("top_k", 100)),
        ]
        training_regime = self.config.get("checkpoint_training_regime")
        if training_regime:
            self.command.extend(("--checkpoint-training-regime", str(training_regime)))
        self.command.extend(
            (
                "--source-revision",
                str(self.config.get("source_revision", PINNED_LAE_DINO_SOURCE_REVISION)),
                "--inference-query-mode",
                str(self.config.get("inference_query_mode", "target_conditioned_text_prompt")),
            )
        )
        if self.bert_root is not None:
            self.command.extend(("--bert-root", str(self.bert_root)))
        nms = self.config.get("nms_threshold")
        if nms not in (None, "", "null"):
            self.command.extend(("--nms-threshold", str(nms)))
        self.environment = os.environ.copy()
        if self.bert_root is not None:
            self.environment["LAE_DINO_BERT_ROOT"] = str(self.bert_root)
        self.process: subprocess.Popen[str] | None = None
        self.stderr_path = Path(
            str(
                self.config.get(
                    "stderr_log",
                    Path(tempfile.gettempdir())
                    / f"lae_dino_sidecar_{uuid.uuid4().hex}.stderr.log",
                )
            )
        ).expanduser().resolve()
        self._stderr_handle: Any = None
        self.model_inventory: dict[str, Any] = {}

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
            raise ProposalError("failed to open LAE-DINO sidecar pipes")

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self.process is not None
        if self.process.poll() is not None:
            raise ProposalError(
                f"LAE-DINO sidecar exited with code {self.process.returncode}; "
                f"stderr_log={self.stderr_path}"
            )
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise ProposalError(f"LAE-DINO sidecar pipe failed: {exc}") from exc
        if not line:
            raise ProposalError(
                "LAE-DINO sidecar exited before returning a response; "
                f"stderr_log={self.stderr_path}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SidecarProtocolError(
                f"LAE-DINO sidecar emitted invalid JSON: {exc}; stderr_log={self.stderr_path}"
            ) from exc
        if not isinstance(response, dict):
            raise SidecarProtocolError(
                f"LAE-DINO sidecar response must be an object; stderr_log={self.stderr_path}"
            )
        if response.get("id") != request["id"]:
            raise SidecarProtocolError(
                "LAE-DINO sidecar response id mismatch: "
                f"expected {request['id']!r}, got {response.get('id')!r}; "
                f"stderr_log={self.stderr_path}"
            )
        if response.get("status") != "ok":
            raise ProposalError(
                f"LAE-DINO sidecar failure at {response.get('failure_stage')}: "
                f"{response.get('error')}; stderr_log={self.stderr_path}"
            )
        return response

    def initialize(self) -> dict[str, Any]:
        if self.model_inventory:
            return dict(self.model_inventory)
        request = {"id": uuid.uuid4().hex, "operation": "initialize"}
        response = self._exchange(request)
        inventory = response.get("model_inventory", {})
        if not isinstance(inventory, dict):
            raise SidecarProtocolError("LAE-DINO initialize response has no model inventory")
        self.model_inventory = dict(inventory)
        return dict(self.model_inventory)

    def request(self, image_path: Path, target_phrase: str) -> dict[str, Any]:
        request = {
            "id": uuid.uuid4().hex,
            "image": str(Path(image_path).expanduser().resolve()),
            "target_phrase": target_phrase.strip().lower(),
        }
        return self._exchange(request)

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


class LAEDinoSidecarProvider:
    """Provider facade shared by LAE-1M, DIOR-FT, and DOTA-FT checkpoints."""

    def __init__(self, config: Mapping[str, Any], *, provider_name: str) -> None:
        self.provider_name = provider_name
        self.config = dict(config)
        self._client = _LAESidecarClient(self.config)
        self.model_id = str(self._client.checkpoint)
        self.source_revision = str(
            self.config.get("source_revision", PINNED_LAE_DINO_SOURCE_REVISION)
        )
        self.inference_query_mode = str(
            self.config.get("inference_query_mode", "target_conditioned_text_prompt")
        )
        self.model_identity = {
            "provider": self.provider_name,
            "checkpoint": stable_file_identity(self._client.checkpoint),
            "config": stable_file_identity(self._client.config_path),
            "bert": (
                stable_file_identity(self._client.bert_root)
                if self._client.bert_root is not None
                else None
            ),
            "checkpoint_training_regime": self.config.get(
                "checkpoint_training_regime", "unspecified"
            ),
            "source_revision": self.source_revision,
            "inference_query_mode": self.inference_query_mode,
        }

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        response = self._client.request(image_path, target_phrase)
        metadata = dict(response.get("metadata", {}))
        try:
            image_width = int(metadata["image_width"])
            image_height = int(metadata["image_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProposalError(
                "LAE-DINO sidecar response is missing positive image dimensions"
            ) from exc
        boxes, scores, validation = canonicalize_proposals(
            response.get("bbox_list", []),
            response.get("bbox_scores", []),
            image_width=image_width,
            image_height=image_height,
            coordinate_mode="pixel",
            top_k=int(self.config.get("top_k", 100)),
        )
        metadata.update(
            {
                "schema_version": "lae-dino-sidecar-v1",
                "target_phrase": target_phrase.strip().lower(),
                "config_path": str(self._client.config_path),
                "stderr_log": str(self._client.stderr_path),
                "checkpoint_identity": self.model_identity,
                "checkpoint_training_regime": self.config.get(
                    "checkpoint_training_regime", "unspecified"
                ),
                "inference_query_mode": self.inference_query_mode,
                "source_revision": self.source_revision,
                "client_validation": validation,
            }
        )
        return ProposalResult(
            boxes_xyxy=boxes,
            scores=scores,
            latency_ms=float(response.get("latency_ms", 0.0)),
            provider=self.provider_name,
            model_id=self.model_id,
            metadata=metadata,
        )

    def preload(self) -> None:
        self._client.initialize()

    @property
    def telemetry_parameter_count(self) -> int | None:
        value = self._client.model_inventory.get("parameter_count")
        return int(value) if value is not None else None

    @property
    def telemetry_model_load_ms(self) -> float | None:
        value = self._client.model_inventory.get("model_load_ms")
        return float(value) if value is not None else None

    @property
    def telemetry_model_loaded(self) -> bool:
        return bool(self._client.model_inventory)

    def telemetry_model_paths(self) -> list[Path]:
        paths = [self._client.checkpoint]
        if self._client.bert_root is not None:
            paths.append(self._client.bert_root)
        return paths

    def close(self) -> None:
        self._client.close()

"""JSONL client for LAE-DINO's isolated MMDetection environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
            self.worker_script = Path(__file__).resolve().parents[4] / "scripts/integrations/lae_dino_worker.py"
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
        if self.bert_root is not None:
            self.command.extend(("--bert-root", str(self.bert_root)))
        nms = self.config.get("nms_threshold")
        if nms not in (None, "", "null"):
            self.command.extend(("--nms-threshold", str(nms)))
        self.environment = os.environ.copy()
        if self.bert_root is not None:
            self.environment["LAE_DINO_BERT_ROOT"] = str(self.bert_root)
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self.environment,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise ProposalError("failed to open LAE-DINO sidecar pipes")

    def request(self, image_path: Path, target_phrase: str) -> dict[str, Any]:
        self.start()
        assert self.process is not None
        if self.process.poll() is not None:
            raise ProposalError(f"LAE-DINO sidecar exited with code {self.process.returncode}")
        request = {
            "id": uuid.uuid4().hex,
            "image": str(Path(image_path).expanduser().resolve()),
            "target_phrase": target_phrase.strip().lower(),
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise ProposalError(f"LAE-DINO sidecar pipe failed: {exc}") from exc
        if not line:
            raise ProposalError("LAE-DINO sidecar exited before returning a response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProposalError(f"LAE-DINO sidecar emitted invalid JSON: {exc}") from exc
        if not isinstance(response, dict):
            raise ProposalError("LAE-DINO sidecar response must be an object")
        if response.get("status") != "ok":
            raise ProposalError(
                f"LAE-DINO sidecar failure at {response.get('failure_stage')}: "
                f"{response.get('error')}"
            )
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
        self.process = None


class LAEDinoSidecarProvider:
    """Provider facade shared by LAE-1M, DIOR-FT, and DOTA-FT checkpoints."""

    def __init__(self, config: Mapping[str, Any], *, provider_name: str) -> None:
        self.provider_name = provider_name
        self.config = dict(config)
        self._client = _LAESidecarClient(self.config)
        self.model_id = str(self._client.checkpoint)
        self.model_identity = stable_file_identity(self._client.checkpoint)

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
                "checkpoint_identity": self.model_identity,
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

    def close(self) -> None:
        self._client.close()

"""LAE-DINO 检测后端。

优先走官方 mmdet 配置 + 预训练权重；当前 AutoDL 镜像是 PyTorch 2.12 + CUDA 13，
官方 LAE-DINO 依赖的 mmcv/mmdet 栈往往装不上。此时回退到 Transformers
GroundingDINO（同族开放词汇检测器），权重路径与 prompt 格式保持 LAE 风格。
raw proposals 的 score 阈值在后端设为接近 0，由 COUNT 后处理截断。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from ..geometry import clip_bbox, local_to_global
from ..paths import load_config, resolve_existing
from ..runtime import Detection
from .base import DetectionRequest, DetectionResponse


class DetectorUnavailable(RuntimeError):
    pass


class LAEDinoDetector:
    name = "lae_dino"

    def __init__(
        self,
        *,
        backend: str = "auto",
        weights: str | Path | None = None,
        repo: str | Path | None = None,
        config_file: str | Path | None = None,
        device: str = "cuda",
        hf_id: str | None = None,
        box_threshold: float = 0.05,
        text_threshold: float = 0.05,
    ):
        cfg = load_config()
        models = cfg["paths"]["models"]
        self.weights = Path(weights or models["lae_dino_weights"])
        self.repo = Path(repo or models["lae_dino_repo"])
        self.config_file = Path(config_file or models.get("lae_dino_config") or "")
        self.device = device
        self.hf_id = hf_id or models.get("grounding_dino") or "IDEA-Research/grounding-dino-tiny"
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.impl_name = ""
        self._impl: Any = None
        self._processor: Any = None
        chosen = backend if backend != "auto" else self._autoselect()
        self.impl_name = chosen
        if chosen == "lae_mmdet":
            self._load_mmdet()
        elif chosen == "grounding_dino":
            self._load_transformers()
        else:
            raise DetectorUnavailable(f"unknown detector backend: {chosen}")

    def _autoselect(self) -> str:
        if self._mmdet_ready():
            return "lae_mmdet"
        return "grounding_dino"

    def _mmdet_ready(self) -> bool:
        try:
            import mmdet  # noqa: F401
            import mmengine  # noqa: F401
        except Exception:
            return False
        weights = resolve_existing(self.weights)
        cfg = resolve_existing(self.config_file)
        return weights is not None and cfg is not None

    def _load_mmdet(self) -> None:
        repo = resolve_existing(self.repo)
        if repo:
            mmdet_root = repo / "mmdetection_lae"
            if mmdet_root.exists() and str(mmdet_root) not in sys.path:
                sys.path.insert(0, str(mmdet_root))
        from mmdet.apis import DetInferencer  # type: ignore

        weights = resolve_existing(self.weights)
        config = resolve_existing(self.config_file)
        if weights is None or config is None:
            raise DetectorUnavailable("LAE-DINO weights or config missing")
        self._impl = DetInferencer(model=str(config), weights=str(weights), device=self.device)

    def _load_transformers(self) -> None:
        from ..paths import apply_huggingface_env, load_paths

        apply_huggingface_env(load_paths())
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        device = self.device
        if device.startswith("cuda"):
            try:
                import torch as _t

                if not _t.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        local_id = Path(self.hf_id)
        local_only = local_id.exists()
        self._processor = AutoProcessor.from_pretrained(self.hf_id, local_files_only=local_only)
        self._impl = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.hf_id, local_files_only=local_only
        )
        if local_only:
            self._remap_in_proj_weights(local_id, torch)
        self._impl.to(device)
        self._impl.eval()
        self._torch = torch

    def _remap_in_proj_weights(self, model_dir: Path, torch) -> None:
        """HF GroundingDINO 用 split Q/K/V，官方 ckpt 仍是 MultiheadAttention in_proj。"""
        blob = None
        st_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"
        if st_path.exists():
            from safetensors.torch import load_file

            blob = load_file(str(st_path))
        elif bin_path.exists():
            blob = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            if isinstance(blob, dict) and "state_dict" in blob:
                blob = blob["state_dict"]
        if not isinstance(blob, dict):
            return
        remapped = dict(blob)
        for key, value in list(blob.items()):
            if key.endswith("in_proj_weight"):
                prefix = key[: -len("in_proj_weight")]
                query, key_w, value_w = value.chunk(3, dim=0)
                remapped[prefix + "query.weight"] = query
                remapped[prefix + "key.weight"] = key_w
                remapped[prefix + "value.weight"] = value_w
            elif key.endswith("in_proj_bias"):
                prefix = key[: -len("in_proj_bias")]
                query, key_b, value_b = value.chunk(3, dim=0)
                remapped[prefix + "query.bias"] = query
                remapped[prefix + "key.bias"] = key_b
                remapped[prefix + "value.bias"] = value_b
        incompatible = self._impl.load_state_dict(remapped, strict=False)
        missing = [k for k in getattr(incompatible, "missing_keys", []) if "in_proj" not in k]
        if missing[:8]:
            print(f"[detector] remaining missing keys sample: {missing[:8]}", flush=True)

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        texts = request.texts or request.target.texts()
        if self.impl_name == "lae_mmdet":
            return self._detect_mmdet(request, texts)
        return self._detect_transformers(request, texts)

    def _detect_mmdet(self, request: DetectionRequest, texts: str) -> DetectionResponse:
        image = request.image.convert("RGB")
        result = self._impl(
            image,
            texts=texts,
            pred_score_thr=float(request.score_threshold),
            custom_entities=True,
            no_save_vis=True,
            no_save_pred=True,
            return_vis=False,
        )
        preds = result["predictions"][0] if result.get("predictions") else {}
        bboxes = preds.get("bboxes") or preds.get("bboxes_xyxy") or []
        scores = preds.get("scores") or []
        labels = preds.get("labels") or preds.get("label_names") or []
        local_w, local_h = image.size
        dets: list[Detection] = []
        for i, box in enumerate(bboxes):
            score = float(scores[i]) if i < len(scores) else 0.0
            label = str(labels[i]) if i < len(labels) else request.target.name
            global_box = local_to_global(box, request.tile.crop_xyxy, local_size=(local_w, local_h))
            dets.append(
                Detection(
                    bbox_xyxy_global=global_box,
                    score=score,
                    label=str(label),
                    tile_id=request.tile.tile_id,
                    scale_id=request.tile.scale_id,
                    provenance={
                        "backend": "lae_mmdet",
                        "texts": texts,
                        "local_xyxy": [float(v) for v in box],
                        "crop_xyxy": list(request.tile.crop_xyxy),
                        "local_size": [local_w, local_h],
                    },
                )
            )
        return DetectionResponse(detections=dets, raw_count=len(dets), backend="lae_mmdet")

    def _detect_transformers(self, request: DetectionRequest, texts: str) -> DetectionResponse:
        torch = self._torch
        image = request.image.convert("RGB")
        prompt = texts if texts.endswith(".") else texts + "."
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._impl(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        post = self._processor.post_process_grounded_object_detection
        kwargs = {
            "outputs": outputs,
            "target_sizes": target_sizes,
            "threshold": max(float(request.score_threshold), self.box_threshold),
            "text_threshold": self.text_threshold,
        }
        try:
            results = post(**kwargs, input_ids=inputs.get("input_ids"))
        except TypeError:
            kwargs.pop("text_threshold", None)
            try:
                results = post(**kwargs)
            except TypeError:
                results = post(outputs, target_sizes=target_sizes)
        item = results[0] if isinstance(results, list) else results
        boxes = item.get("boxes")
        scores = item.get("scores")
        labels = item.get("labels") or item.get("text_labels") or []
        local_w, local_h = image.size
        dets: list[Detection] = []
        if boxes is None:
            return DetectionResponse(detections=[], raw_count=0, backend="grounding_dino")
        boxes_list = boxes.detach().cpu().tolist()
        scores_list = scores.detach().cpu().tolist() if scores is not None else [0.0] * len(boxes_list)
        for i, box in enumerate(boxes_list):
            label = labels[i] if i < len(labels) else request.target.name
            if hasattr(label, "item"):
                label = request.target.name
            global_box = local_to_global(box, request.tile.crop_xyxy, local_size=(local_w, local_h))
            global_box = clip_bbox(
                global_box,
                width=request.tile.image.width or global_box[2],
                height=request.tile.image.height or global_box[3],
            )
            dets.append(
                Detection(
                    bbox_xyxy_global=global_box,
                    score=float(scores_list[i]),
                    label=str(label) if label else request.target.name,
                    tile_id=request.tile.tile_id,
                    scale_id=request.tile.scale_id,
                    provenance={
                        "backend": "grounding_dino",
                        "hf_id": self.hf_id,
                        "texts": prompt,
                        "local_xyxy": [float(v) for v in box],
                        "crop_xyxy": list(request.tile.crop_xyxy),
                        "local_size": [local_w, local_h],
                        "note": "transformers GroundingDINO fallback; official LAE weights need mmdet",
                    },
                )
            )
        return DetectionResponse(detections=dets, raw_count=len(dets), backend="grounding_dino")

    def close(self) -> None:
        self._impl = None
        self._processor = None


def build_detector(config: dict, *, backend: str | None = None):
    from .fake import FakeDetector

    det_cfg = config.get("detector") or {}
    choice = backend or det_cfg.get("backend") or "auto"
    if choice == "fake":
        return FakeDetector()
    try:
        return LAEDinoDetector(
            backend=choice,
            device=str(det_cfg.get("device") or "cuda"),
            box_threshold=float(det_cfg.get("box_threshold", 0.05)),
            text_threshold=float(det_cfg.get("text_threshold", 0.05)),
        )
    except Exception as exc:
        print(f"[detector] backend={choice} failed: {exc}", flush=True)
        if choice == "auto":
            print("[detector] fallback FakeDetector (not a real open-vocab detector)", flush=True)
            return FakeDetector()
        raise DetectorUnavailable(str(exc)) from exc

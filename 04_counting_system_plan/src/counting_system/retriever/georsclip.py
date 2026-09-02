"""GeoRSCLIP coarse retrieval gate。recall-first，不是 Top-K。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..paths import georsclip_ckpt, load_config, resolve_existing


class RetrieverUnavailable(RuntimeError):
    pass


def _clip_preprocess(resolution: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


class GeoRSCLIPRetriever:
    name = "georsclip"

    def __init__(
        self,
        *,
        ckpt: str | Path | None = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        image_resolution: int = 224,
    ):
        try:
            import open_clip
            import torch
        except Exception as exc:
            raise RetrieverUnavailable("open_clip_torch / torch 未安装") from exc

        ckpt_path = resolve_existing(ckpt or georsclip_ckpt())
        if ckpt_path is None:
            raise RetrieverUnavailable(f"GeoRSCLIP checkpoint missing: {ckpt or georsclip_ckpt()}")

        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if "/" in model_name:
            arch = model_name
        elif model_name.count("-") >= 2:
            head, tail = model_name.rsplit("-", 1)
            arch = f"{head}/{tail}"
        else:
            arch = model_name
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None)
        blob: Any = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        if isinstance(state, dict):
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            incompatible = model.load_state_dict(cleaned, strict=False)
            self._load_msg = {
                "missing": list(getattr(incompatible, "missing_keys", [])),
                "unexpected": list(getattr(incompatible, "unexpected_keys", [])),
            }
        else:
            self._load_msg = {"error": "checkpoint is not a state dict"}
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.preprocess = _clip_preprocess(image_resolution)
        self.torch = torch
        self.ckpt_path = ckpt_path

    def score(self, image: Image.Image, text: str) -> float:
        torch = self.torch
        image_t = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        text_t = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            image_f = self.model.encode_image(image_t)
            text_f = self.model.encode_text(text_t)
            image_f = image_f / image_f.norm(dim=-1, keepdim=True)
            text_f = text_f / text_f.norm(dim=-1, keepdim=True)
            sim = (image_f * text_f).sum(dim=-1)
        return float(sim.squeeze().detach().cpu())

    def close(self) -> None:
        self.model = None


def build_retriever(config: dict, *, backend: str | None = None):
    from .fake import FakeRetriever

    ret_cfg = config.get("retriever") or {}
    gate_cfg = config.get("gate") or {}
    choice = backend or gate_cfg.get("backend") or ret_cfg.get("backend") or "georsclip"
    if choice in {"fake", "none", "off"}:
        return FakeRetriever()
    try:
        cfg = load_config()
        return GeoRSCLIPRetriever(
            ckpt=cfg["paths"]["models"]["georsclip_ckpt"],
            model_name=str(ret_cfg.get("model_name") or "ViT-B-32"),
            pretrained=str(ret_cfg.get("pretrained") or "openai"),
            device=str(ret_cfg.get("device") or "cuda"),
            image_resolution=int(ret_cfg.get("image_resolution") or 224),
        )
    except Exception:
        if choice == "georsclip":
            raise
        return FakeRetriever()

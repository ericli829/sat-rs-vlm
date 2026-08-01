"""合并 LoRA adapter 到基座模型。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sat_rs_vlm.training.utils import MODEL_DEPS_ERROR, safe_import_model_dependencies

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into base model.")
    parser.add_argument("--base-model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter path.")
    parser.add_argument("--output", required=True, help="Output directory for merged model.")
    return parser.parse_args()


def load_base_model(base_model: str, transformers: Any) -> Any:
    """加载基座模型。"""

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_cls is None:
        raise ImportError(MODEL_DEPS_ERROR)
    return model_cls.from_pretrained(base_model, device_map="auto", trust_remote_code=True)


def merge_lora(base_model: str, adapter: str, output: str) -> None:
    """执行 LoRA 合并。"""

    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    transformers = modules["transformers"]
    peft = modules["peft"]
    output_dir = Path(output)
    try:
        model = load_base_model(base_model, transformers)
        model = peft.PeftModel.from_pretrained(model, adapter)
        merged = model.merge_and_unload()
        processor = transformers.AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    except RuntimeError as exc:
        raise SystemExit(
            "LoRA merge failed. If this is an out-of-memory error, retry on CPU, "
            "use a smaller model, or merge on a machine with more memory."
        ) from exc
    print(f"Merged model saved to {output_dir}")


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    try:
        merge_lora(args.base_model, args.adapter, args.output)
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

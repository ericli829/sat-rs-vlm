"""Check PNG/BMP equivalence through LAE's real CPU image preprocessing, without weights."""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import struct
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sat_rs_vlm.integrations.detectors.tile_transport import save_tile_image  # noqa: E402
from sat_rs_vlm.integrations.detectors.tiled import generate_tiles  # noqa: E402

CONFIGS = {
    "lae1m": ("lae_dino_swin-t_pretrain_LAE-1M.py", "lae_1m_detection.py"),
    "dior": ("lae_dino_swin-t_finetune_DIOR.py", "dior_detection.py"),
    "dota": ("lae_dino_swin-t_finetune_DOTA.py", "dota_detection.py"),
}


def source_nodes(path: Path, names: list[str], namespace: dict[str, Any]) -> None:
    """Run the installed image-only definitions without importing unrelated CUDA ops.

    Registration decorators are omitted. The bbox-casting decorator on Resize
    is omitted because this probe has no annotations. Pixel/resize/tensor code
    is taken unchanged from the selected source files, not reimplemented here.
    """

    selected = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            node.decorator_list = []
            if isinstance(node, ast.ClassDef) and node.name == "Resize":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "transform":
                        method.decorator_list = []
            selected.append(node)
    if {node.name for node in selected} != set(names):
        raise ValueError(f"missing image preprocessing definitions in {path}: {names}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def config_dict(node: ast.expr) -> dict[str, Any]:
    # The pinned settings are plain dict/list/tuple/constant expressions.
    allowed = (
        ast.Call,
        ast.Name,
        ast.Load,
        ast.keyword,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
    )
    for part in ast.walk(node):
        if (
            not isinstance(part, allowed)
            or isinstance(part, ast.Name)
            and part.id not in {"dict", "backend_args"}
        ):
            raise ValueError("image configuration must contain only literal settings")
    return eval(
        compile(ast.Expression(node), "<image-config>", "eval"),
        {
            "__builtins__": {},
            "dict": dict,
            "backend_args": None,
        },
    )


def settings(root: Path, model_file: str, dataset_file: str) -> dict[str, Any]:
    model_path = root / "configs/lae_dino" / model_file
    dataset_path = root / "configs/_base_/datasets" / dataset_file
    trees = [ast.parse(p.read_text(encoding="utf-8")) for p in (model_path, dataset_path)]
    backend = next(
        n.value
        for n in trees[1].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "backend_args" for t in n.targets)
    )
    if ast.literal_eval(backend) is not None:
        raise ValueError("this local-file audit requires backend_args=None")
    model = next(
        n.value
        for n in trees[0].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model" for t in n.targets)
    )
    preprocessor = next(k.value for k in model.keywords if k.arg == "data_preprocessor")
    pipeline = next(
        n.value
        for n in trees[1].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "test_pipeline" for t in n.targets)
    )
    steps = {value["type"]: value for value in map(config_dict, pipeline.elts)}
    return {
        "model_config": str(model_path),
        "dataset_config": str(dataset_path),
        "loader": steps["LoadImageFromFile"],
        "resize": steps["FixScaleResize"],
        "pack": steps["PackDetInputs"],
        "preprocessor": config_dict(preprocessor),
    }


def image_classes(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    import cv2
    import mmcv
    import mmengine
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mmcv.image.geometric import _scale_size
    from mmcv.transforms import BaseTransform, Resize, to_tensor
    from mmengine.structures import BaseDataElement, InstanceData, PixelData
    from mmengine.utils import is_seq_of

    namespace = {
        "np": np,
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "math": math,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "is_seq_of": is_seq_of,
        "BaseDataElement": BaseDataElement,
        "InstanceData": InstanceData,
        "PixelData": PixelData,
        "BaseTransform": BaseTransform,
        "MMCV_Resize": Resize,
        "mmcv": mmcv,
        "imresize": mmcv.imresize,
        "_scale_size": _scale_size,
        "to_tensor": to_tensor,
        # No boxes are supplied; annotation-only checks always see None.
        "BaseBoxes": type("UnusedAnnotationBoxes", (), {}),
    }
    engine = Path(mmengine.__file__).parent
    source_nodes(engine / "model/utils.py", ["stack_batch"], namespace)
    source_nodes(
        engine / "model/base_model/data_preprocessor.py",
        ["BaseDataPreprocessor", "ImgDataPreprocessor"],
        namespace,
    )
    source_nodes(root / "mmdet/structures/det_data_sample.py", ["DetDataSample"], namespace)
    source_nodes(root / "mmdet/models/utils/misc.py", ["samplelist_boxtype2tensor"], namespace)
    source_nodes(
        root / "mmdet/models/data_preprocessors/data_preprocessor.py",
        ["DetDataPreprocessor"],
        namespace,
    )
    source_nodes(
        root / "mmdet/datasets/transforms/transforms.py",
        ["_fixed_scale_size", "rescale_size", "imrescale", "Resize", "FixScaleResize"],
        namespace,
    )
    source_nodes(root / "mmdet/datasets/transforms/formatting.py", ["PackDetInputs"], namespace)
    return namespace, {
        "opencv": cv2.__version__,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def check_pair(image: Image.Image, root: Path, config: dict[str, Any], classes: dict[str, Any]):
    import numpy as np
    import torch
    from mmcv.transforms import LoadImageFromFile

    load = LoadImageFromFile(**{k: v for k, v in config["loader"].items() if k != "type"})
    resize = classes["FixScaleResize"](**{k: v for k, v in config["resize"].items() if k != "type"})
    pack = classes["PackDetInputs"](**{k: v for k, v in config["pack"].items() if k != "type"})
    processor = classes["DetDataPreprocessor"](
        **{k: v for k, v in config["preprocessor"].items() if k != "type"}
    )
    assert processor.device.type == "cpu"
    with image.convert("RGB") as rgb:
        expected_bgr = np.asarray(rgb)[:, :, ::-1]
        outputs = []
        for image_format in ("png", "bmp"):
            path = root / f"tile.{image_format}"
            save_tile_image(image, path, image_format)
            if image_format == "bmp":
                header = path.read_bytes()[:54]
                assert header[:2] == b"BM"
                assert struct.unpack_from("<iiHHI", header, 18) == (
                    image.width,
                    image.height,
                    1,
                    24,
                    0,
                ), "expected uncompressed 24-bit bottom-up BMP"
            loaded = load(
                {"img_path": str(path), "img_id": 0, "text": "car", "custom_entities": True}
            )
            assert loaded["img"].dtype == np.uint8
            assert np.array_equal(loaded["img"], expected_bgr), (
                "BGR pixels, dimensions or row orientation changed"
            )
            resized = resize(loaded)
            packed = pack(resized)
            assert packed["inputs"].dtype == torch.uint8 and packed["inputs"].shape[0] == 3
            result = processor(
                {"inputs": [packed["inputs"]], "data_samples": [packed["data_samples"]]},
                training=False,
            )
            tensor = result["inputs"]
            assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            assert tensor.ndim == 4 and tensor.shape[:2] == (1, 3) and torch.isfinite(tensor).all()
            expected = (
                torch.from_numpy(np.ascontiguousarray(resized["img"][:, :, ::-1]))
                .permute(2, 0, 1)
                .float()
            )
            expected = (expected - processor.mean) / processor.std
            assert torch.equal(tensor[0], expected), "unexpected color conversion or normalization"
            outputs.append(tensor)
        assert torch.equal(*outputs), "PNG/BMP final model input tensors differ"
        return {
            "size": list(image.size),
            "input_shape": list(outputs[0].shape),
            "equal_decoded_bgr": True,
            "equal_normalized_model_input": True,
            "max_abs_tensor_difference": 0,
            "tensor_sha256": hashlib.sha256(outputs[0].numpy().tobytes()).hexdigest(),
        }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--lae-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    args = parser.parse_args()
    if args.dependency_root:
        sys.path.insert(0, str(args.dependency_root.resolve()))
    image_root = args.image_root.resolve()
    source_root = args.lae_source_root.resolve()
    if (source_root / "mmdetection_lae").is_dir():
        source_root /= "mmdetection_lae"
    classes, versions = image_classes(source_root)
    configs = {name: settings(source_root, *files) for name, files in CONFIGS.items()}
    images = [
        next((image_root / "xlrs_bench_samples_20260827").glob("01_anomaly_*.jpg")),
        next((image_root / "xlrs_bench_samples_20260827").glob("02_spatial_relation_*.jpg")),
        next((image_root / "mme_counting_20260829").glob("01_03553_*.png")),
    ]
    report = {
        "scope": (
            "Actual CPU decoding/resize/packing/normalization; no model weights or GPU forward"
        ),
        "source_root": str(source_root),
        "versions": versions,
        "configurations": configs,
        "cases": [],
    }
    # Only the user's explicitly selected, trusted UHR fixtures disable Pillow's pixel limit.
    pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with tempfile.TemporaryDirectory(prefix="detector_transport_check_") as temporary:
            root = Path(temporary)
            for path in images:
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                source.close()
                with rgb:
                    tiles = generate_tiles(rgb.width, rgb.height, 1333, 0.15)
                    for index in sorted({0, len(tiles) // 2, len(tiles) - 1}):
                        with rgb.crop(tiles[index]) as crop:
                            for name, config in configs.items():
                                result = check_pair(crop, root, config, classes)
                                result.update(
                                    image=str(path),
                                    tile_xyxy=list(tiles[index]),
                                    configuration=name,
                                )
                                report["cases"].append(result)
                print(f"Verified {path.name} with all three LAE configurations", flush=True)
            # Width not divisible by four, asymmetric rows, grayscale and alpha input.
            fixture = Image.new("RGB", (7, 5))
            for y in range(5):
                for x in range(7):
                    fixture.putpixel((x, y), (x * 30, y * 40, (x + y) * 20))
            with fixture:
                for mode in ("RGB", "L", "RGBA"):
                    with fixture.convert(mode) as image:
                        if mode == "RGBA":
                            image.putalpha(128)
                        for name, config in configs.items():
                            result = check_pair(image, root, config, classes)
                            result.update(fixture=mode, configuration=name)
                            report["cases"].append(result)
    finally:
        Image.MAX_IMAGE_PIXELS = pixel_limit
    report["all_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Passed {len(report['cases'])} PNG/BMP input pairs: {args.output.resolve()}")


if __name__ == "__main__":
    main()

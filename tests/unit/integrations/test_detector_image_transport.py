from __future__ import annotations

import struct
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from scripts.integrations.lae_dino_worker import _predict

from sat_rs_vlm.integrations.detectors.grounding_dino import GroundingDinoProvider
from sat_rs_vlm.integrations.detectors.tile_transport import save_tile_image


def colored_image() -> Image.Image:
    image = Image.new("RGB", (7, 5))
    for y in range(5):
        for x in range(7):
            image.putpixel((x, y), (x * 30, y * 40, (x + y) * 20))
    return image


def test_bmp_is_standard_24bit_with_correct_row_padding_and_orientation(tmp_path: Path):
    path = tmp_path / "transport.bmp"
    with colored_image() as image:
        save_tile_image(image, path, "bmp")
        data = path.read_bytes()
        assert data[:2] == b"BM"
        assert struct.unpack_from("<I", data, 14)[0] == 40  # BITMAPINFOHEADER
        assert struct.unpack_from("<iiHHI", data, 18) == (7, 5, 1, 24, 0)
        offset = struct.unpack_from("<I", data, 10)[0]
        stride = ((image.width * 3 + 3) // 4) * 4
        assert len(data) - offset == stride * image.height
        for row in range(image.height):
            for x in range(image.width):
                position = offset + row * stride + x * 3
                expected = image.getpixel((x, image.height - row - 1))[::-1]
                assert tuple(data[position : position + 3]) == expected


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
@pytest.mark.parametrize("image_format", ["png", "bmp"])
def test_lae_worker_passes_valid_rgb_file_to_inference_api(
    tmp_path: Path, mode: str, image_format: str
):
    path = tmp_path / f"transport.{image_format}"
    with colored_image() as source, source.convert(mode) as image:
        if mode == "RGBA":
            image.putalpha(128)
        save_tile_image(image, path, image_format)
        with image.convert("RGB") as rgb:
            expected = rgb.tobytes()

    def infer(_model, image_path, *, text_prompt, custom_entities):
        assert Path(image_path) == path.resolve()
        assert text_prompt == "car" and custom_entities is True
        with Image.open(image_path) as loaded:
            assert loaded.format == image_format.upper()
            assert loaded.mode == "RGB" and loaded.size == (7, 5)
            assert loaded.tobytes() == expected
        return [[[1, 1, 6, 4, 0.9]]]

    result = _predict(
        object(),
        {"id": "transport", "image": str(path), "target_phrase": "Car"},
        SimpleNamespace(score_threshold=0, top_k=10, nms_threshold=None),
        infer,
    )
    assert result["status"] == "ok"
    assert result["bbox_list"] == [[1.0, 1.0, 6.0, 4.0]]
    assert (result["metadata"]["image_width"], result["metadata"]["image_height"]) == (7, 5)


def test_grounding_dino_receives_rgb_pixels_from_bmp(tmp_path: Path):
    path = tmp_path / "transport.bmp"
    with colored_image() as image:
        save_tile_image(image, path, "bmp")
        expected = image.tobytes()

    class Processor:
        def __call__(self, *, images, text, return_tensors):
            assert images.mode == "RGB" and images.size == (7, 5)
            assert images.tobytes() == expected
            assert text == [["car"]] and return_tensors == "pt"
            return {"input_ids": None}

        def post_process_grounded_object_detection(self, _outputs, **kwargs):
            assert kwargs["target_sizes"] == [(5, 7)]
            return [{"boxes": [], "scores": []}]

    model_path = tmp_path / "model"
    model_path.mkdir()
    provider = GroundingDinoProvider({"model_path": str(model_path), "device": "cpu"})
    provider._processor = Processor()
    provider._model = lambda **_kwargs: object()
    provider._torch = SimpleNamespace(inference_mode=nullcontext)
    try:
        result = provider.predict(path, "car")
        assert result.metadata["image_width"] == 7 and result.metadata["image_height"] == 5
    finally:
        provider.close()

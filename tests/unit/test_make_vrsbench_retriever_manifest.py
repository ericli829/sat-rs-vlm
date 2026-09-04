import json
from pathlib import Path

from PIL import Image
from scripts.make_vrsbench_retriever_manifest import convert


def test_convert_scales_and_clips_normalized_boxes_above_one(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    Image.new("RGB", (100, 200), "white").save(images / "sample.png")
    (annotations / "sample.json").write_text(
        json.dumps(
            {
                "image": "sample.png",
                "objects": [
                    {
                        "obj_id": 0,
                        "obj_cls": "bridge",
                        "referring_sentence": "the bridge",
                        "obj_coord": [0.2, 0.1, 1.2, 1.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    stats = convert(annotations, images, output)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert stats["rows"] == 1
    assert row["gt_boxes"] == [[20.0, 20.0, 100.0, 200.0]]

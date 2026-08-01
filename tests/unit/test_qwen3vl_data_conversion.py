from scripts.convert_to_qwen3vl_format import convert_sample_to_qwen3vl


def test_convert_single_image_sample() -> None:
    sample = {
        "id": "sample_000001",
        "task_type": "captioning",
        "images": ["data/samples/demo_image.png"],
        "instruction": "请描述这张遥感图像中的主要地物。",
        "answer": "图像中包含建筑物、道路和植被区域。",
        "metadata": {"dataset": "sample"},
    }
    converted = convert_sample_to_qwen3vl(sample)
    content = converted["messages"][0]["content"]
    assert content[0] == {"type": "image", "image": "data/samples/demo_image.png"}
    assert content[1]["type"] == "text"
    assert converted["messages"][1]["role"] == "assistant"
    assert converted["task_type"] == "captioning"


def test_convert_two_image_change_sample() -> None:
    sample = {
        "id": "change_000001",
        "task_type": "change_detection",
        "images": ["data/samples/before.png", "data/samples/after.png"],
        "instruction": "第一张为变化前，第二张为变化后。请描述变化。",
        "answer": "变化后新增了建筑物。",
        "metadata": {"dataset": "sample_change"},
    }
    converted = convert_sample_to_qwen3vl(sample)
    content = converted["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    assert content[2] == {"type": "text", "text": "第一张为变化前，第二张为变化后。请描述变化。"}
    assert converted["task_type"] == "change_detection"


def test_unstructured_detection_is_downgraded_without_fabricating_bbox() -> None:
    converted = convert_sample_to_qwen3vl(
        {
            "id": "legacy-detection",
            "task_type": "detection",
            "images": ["image.png"],
            "instruction": "请检测图像中的飞机。",
            "answer": "检测到一架飞机。",
            "metadata": {},
        }
    )
    assert converted["task_type"] == "vqa"
    assert converted["metadata"]["detection_unresolved"] is True
    assert "bbox" not in converted["messages"][0]["content"][-1]["text"]
    assert converted["messages"][1]["content"] == "检测到一架飞机。"


def test_legacy_single_detection_is_rewritten_to_canonical_json() -> None:
    converted = convert_sample_to_qwen3vl(
        {
            "id": "legacy-detection",
            "task_type": "detection",
            "images": ["image.png"],
            "instruction": "请检测图像中的飞机。",
            "answer": '{"boxes":[[0.1,0.2,0.4,0.5]],"labels":["airplane"]}',
            "metadata": {},
        }
    )
    assert converted["task_type"] == "detection"
    assert converted["messages"][1]["content"] == ('{"label":"airplane","bbox":[0.1,0.2,0.4,0.5]}')

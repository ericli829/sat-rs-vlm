from pathlib import Path

from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.utils.jsonl import write_jsonl


def test_messages_format_is_read(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "m1",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "images/001.jpg"},
                            {"type": "text", "text": "请描述这张遥感图像。"},
                        ],
                    },
                    {"role": "assistant", "content": "图像中包含建筑。"},
                ],
                "task_type": "captioning",
            }
        ],
    )
    item = Qwen3VLDataset(path)[0]
    assert item["id"] == "m1"
    assert item["messages"][0]["role"] == "user"


def test_internal_single_image_format_is_converted(tmp_path: Path) -> None:
    path = tmp_path / "internal.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "s1",
                "task_type": "captioning",
                "images": ["images/001.jpg"],
                "instruction": "请描述这张遥感图像。",
                "answer": "图像中包含建筑、道路和植被。",
            }
        ],
    )
    item = Qwen3VLDataset(path)[0]
    assert item["messages"][0]["content"][0]["type"] == "image"
    assert item["messages"][1]["content"] == "图像中包含建筑、道路和植被。"


def test_internal_two_image_format_is_converted(tmp_path: Path) -> None:
    path = tmp_path / "change.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "c1",
                "task_type": "change_detection",
                "images": ["before.jpg", "after.jpg"],
                "instruction": "请描述变化。",
                "answer": "新增了建筑物。",
            }
        ],
    )
    item = Qwen3VLDataset(path)[0]
    content = item["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    assert content[2]["type"] == "text"

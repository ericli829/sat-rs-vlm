from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.utils.jsonl import write_jsonl


def test_qwen3vl_dataset_reads_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jsonl_path = tmp_path / "qwen.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": []},
                    {"role": "assistant", "content": "ok"},
                ],
                "task_type": "captioning",
                "metadata": {"dataset": "sample"},
            }
        ],
    )
    dataset = Qwen3VLDataset(jsonl_path)
    item = dataset[0]
    assert len(dataset) == 1
    assert item["id"] == "sample_1"
    assert item["messages"]
    assert item["task_type"] == "captioning"

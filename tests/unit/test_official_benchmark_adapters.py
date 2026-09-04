from __future__ import annotations

from sat_rs_vlm.data.official_benchmarks import adapt_mme_realworld, adapt_xlrs


def test_mme_adapter_filters_remote_sensing_and_preserves_official_fields() -> None:
    row = {
        "Question_id": "Perception/Remote Sensing/counting/1",
        "Image": "remote/1.jpg",
        "Text": "How many ships are visible?",
        "Answer choices": ["(A) 1", "(B) 2", "(C) 3", "(D) 4", "(E) none"],
        "Ground truth": "B",
        "Task": "Perception",
        "Subtask": "Remote Sensing",
        "Category": "counting",
    }
    sample = adapt_mme_realworld(
        row,
        dataset_version="official-2024",
        split="train",
        language="en",
        evaluation_scope="official_full_split",
    )
    assert sample is not None
    assert sample["task_type"] == "vqa"
    assert sample["metadata"]["prompt_profile"] == "mme_realworld_official_mcq_v1"
    assert sample["metadata"]["answer_choices"][1] == "(B) 2"
    assert sample["messages"][1]["content"] == "B"
    assert (
        adapt_mme_realworld(
            {**row, "Subtask": "Monitoring"},
            dataset_version="official-2024",
            split="train",
            language="en",
        )
        is None
    )


def test_xlrs_adapter_selects_multiselect_prompt_without_claiming_full_split() -> None:
    row = {
        "index": 7,
        "image": ["a.png", "b.png"],
        "question": "Which land uses are present?",
        "multi-choice options": ["(A) Urban", "(B) Water", "(C) Forest", "(D) Mine"],
        "answer": "A C",
        "category": "Land use classification/Overall Land use classification",
        "l2-category": "land cover",
    }
    sample = adapt_xlrs(
        row,
        dataset_version="XLRS-Bench-lite",
        split="train",
        language="en",
    )
    assert sample["id"] == "7"
    assert sample["metadata"]["evaluation_scope"] == "subset_or_unspecified"
    assert sample["metadata"]["prompt_profile"] == "xlrs_bench_official_multiselect_v1"
    content = sample["messages"][0]["content"]
    assert [item["image"] for item in content if item["type"] == "image"] == [
        "a.png",
        "b.png",
    ]

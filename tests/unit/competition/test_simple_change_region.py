from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

spec = importlib.util.spec_from_file_location(
    "simple_change_region_experiment", Path("scripts/competition/test_simple_change_region.py")
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_original_caption_is_isolated_from_question_and_candidate_injection():
    from sat_rs_vlm.data.prompt_templates import CHANGE_DESCRIPTION_INSTRUCTION

    sample = experiment.original_sample("sample", ["before.png", "after.png"])
    assert experiment.caption_prompt() == CHANGE_DESCRIPTION_INSTRUCTION
    assert sample["messages"] == [{"role": "user", "content": [
        {"type": "image", "image": "before.png"},
        {"type": "image", "image": "after.png"},
        {"type": "text", "text": CHANGE_DESCRIPTION_INSTRUCTION},
    ]}]
    question = "Which region contains the primary change?"
    suffix = experiment.label_prompt(question)
    assert question in suffix
    for label in experiment.LABELS:
        assert label in suffix
        assert label not in experiment.caption_prompt()
    assert question not in experiment.caption_prompt()
    with pytest.raises(TypeError):
        experiment.caption_prompt(question)


def test_primary_location_prompt_requests_position_without_early_candidate_injection():
    prompt = experiment.experiment_prompt("primary_location")
    assert "main visible change and its location in the image" in prompt
    assert "If there is no change, state that clearly." in prompt
    assert prompt != experiment.caption_prompt()
    assert experiment.experiment_prompt("original") == experiment.caption_prompt()
    assert "Question:" not in prompt
    for label in experiment.LABELS:
        assert label not in prompt
    sample = experiment.caption_sample("sample", ["before.png", "after.png"], prompt)
    assert sample["messages"][0]["content"][-1]["text"] == prompt
    with pytest.raises(ValueError, match="Unknown caption mode"):
        experiment.experiment_prompt("typo")


def test_changed_prompt_baseline_requires_same_images_question_and_model(tmp_path):
    model = {"model_id": "base", "adapter_path": "lora"}
    request = {"images": ["before", "after"], "question": "where?"}
    case = {"item_id": "one", "kind": "website", "expected": "top", "original_request": request}
    row = {**case, "description": "original", "original_evaluator_caption": "original", "label": "center"}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    metadata = {"caption_prompt": experiment.caption_prompt(), "model": model,
                "label_prompt_template": experiment.label_prompt("{website_question}"),
                "decoder": "kv_cached_prefix_constrained_greedy"}
    (tmp_path / "experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
    plan = {"model": model, "caption_mode": "primary_location", "baseline_results": str(path)}
    assert experiment.load_baseline(plan, [case])["one"]["label"] == "center"
    wrong = {**case, "original_request": {**request, "images": ["after", "before"]}}
    with pytest.raises(ValueError, match="image order"):
        experiment.load_baseline(plan, [wrong])
    with pytest.raises(ValueError, match="image order"):
        experiment.load_baseline(plan, [{**case, "original_request": {**request, "question": "changed?"}}])
    with pytest.raises(ValueError, match="model configuration"):
        experiment.load_baseline({**plan, "model": {**model, "adapter_path": "different"}}, [case])
    with pytest.raises(ValueError, match="requires a saved"):
        experiment.load_baseline({"caption_mode": "primary_location"}, [case])


def test_paired_prompt_comparison_reports_gains_losses_and_controls_separately():
    rows = [
        {"item_id": "gain", "kind": "website", "expected": "top", "label": "top", "description": "top"},
        {"item_id": "loss", "kind": "website", "expected": "right", "label": "center", "description": "center"},
        {"item_id": "unclear", "kind": "website", "expected": "unclear", "label": "left", "description": "left"},
        {"item_id": "control", "kind": "identical_image_control", "expected": "no_change", "label": "no_change", "description": "same"},
    ]
    baseline = {item: {"label": label, "description": "old"} for item, label in
                [("gain", "left"), ("loss", "right"), ("unclear", "center"), ("control", "center")]}
    paired = experiment.compare_baseline(rows, baseline)
    assert paired["region_before"] == {"correct": 1, "total": 2, "accuracy": .5}
    assert paired["region_after"] == {"correct": 1, "total": 2, "accuracy": .5}
    assert paired["no_change_before"]["correct"] == 0
    assert paired["no_change_after"]["correct"] == 1
    assert paired["improved_ids"] == ["gain"]
    assert paired["regressed_ids"] == ["loss"]
    assert paired["changed_labels"] == 4


def test_all_ten_labels_follow_the_token_mask_and_shared_prefixes():
    engine = object.__new__(HuggingFaceVLMEngine)
    engine._processor = SimpleNamespace(tokenizer=SimpleNamespace(
        encode=lambda s, **_: [ord(c) for c in s], eos_token_id=999
    ))
    mask = engine._selection_prefix_constraint(allowed_outputs=experiment.LABELS, prompt_token_count=2)
    for label in experiment.LABELS:
        tokens = [1, 2]
        for character in label:
            assert ord(character) in mask(0, tokens)
            tokens.append(ord(character))
        assert 999 in mask(0, tokens)
    bottom = [1, 2] + [ord(c) for c in 'bottom']
    assert set(mask(0, bottom)) == {999, ord('_')}
    assert 999 not in mask(0, [1, 2])
    with pytest.raises(ValueError):
        experiment.validate_label('The change is top_left.')


def test_unclear_labels_and_synthetic_controls_do_not_inflate_region_accuracy():
    rows = [
        {'kind':'website', 'expected':'top_left', 'label':'top_left'},
        {'kind':'website', 'expected':'right', 'label':'no_change'},
        {'kind':'website', 'expected':'unclear', 'label':'left'},
        {'kind':'identical_image_control', 'expected':'no_change', 'label':'no_change'},
    ]
    result = experiment.summarize(rows, 30)
    assert result['website_human_reference'] == {'correct':1, 'total':2, 'accuracy':0.5}
    assert result['no_change_controls'] == {'correct':1, 'total':1, 'accuracy':1.0}
    assert result['website_internal_no_change'] == 1
    assert result['valid_labels'] == 4


def test_visual_review_requires_both_exact_caption_and_ordered_image_hashes():
    import importlib
    probabilities = importlib.import_module("test_change_region_probabilities")
    case = {"item_id": "one", "original_request": {"images": [{"sha256": "a"}, {"sha256": "b"}]}}
    review = {"one": {"description": "a house appeared", "ordered_image_sha256": ["a", "b"],
                      "content": "correct", "location": "missing"}}
    assert probabilities.attach_review(review, case, "a house appeared")["status"].startswith("visually_reviewed")
    assert probabilities.attach_review(review, case, "a road appeared")["status"] == "pending_visual_review"
    reverse = {"item_id": "one", "original_request": {"images": [{"sha256": "b"}, {"sha256": "a"}]}}
    assert probabilities.attach_review(review, reverse, "a house appeared")["status"] == "pending_visual_review"

from __future__ import annotations

from scripts.evaluation.prepare_change_caption_audit import build_audit_rows


def test_blind_audit_deduplicates_captions_and_excludes_image_truth() -> None:
    rows = [
        {
            "id": "a",
            "prediction": "No change has occurred.",
            "prediction_changeflag": 0,
            "binary_prediction_source": "local_semantic_rule",
            "reference": "hidden reference",
            "metadata": {"changeflag": 1},
        },
        {
            "id": "b",
            "prediction": " no   change has occurred. ",
            "prediction_changeflag": 0,
            "binary_prediction_source": "local_semantic_rule",
            "reference": "another hidden reference",
            "metadata": {"changeflag": 0},
        },
        {
            "id": "c",
            "prediction": "A new building appeared.",
            "prediction_changeflag": 1,
            "binary_prediction_source": "local_llm_judge",
            "reference": "hidden",
            "metadata": {"changeflag": 1},
        },
    ]

    prepared = build_audit_rows(rows, sample_size=10, seed=7)

    assert len(prepared) == 2
    assert sorted(item["occurrences"] for item in prepared) == [1, 2]
    for item in prepared:
        assert item["human_caption_semantic_label"] is None
        assert "reference" not in item
        assert "metadata" not in item
        assert "changeflag" not in item
        assert "_answer_key" in item


def test_blind_audit_forces_old_new_disagreements() -> None:
    rows = [
        {"id": "forced", "prediction": "Only a vehicle appeared."},
        {"id": "other-1", "prediction": "A new building appeared."},
        {"id": "other-2", "prediction": "No change has occurred."},
    ]

    prepared = build_audit_rows(
        rows,
        sample_size=1,
        seed=3,
        disagreement_keys={"only a vehicle appeared."},
    )

    assert prepared[0]["caption"] == "Only a vehicle appeared."
    assert prepared[0]["_answer_key"]["selection_reason"] == ("forced_old_new_disagreement")


def test_blind_audit_excludes_development_captions() -> None:
    rows = [
        {"id": "dev", "prediction": "A new road appeared."},
        {"id": "holdout", "prediction": "A house was demolished."},
    ]

    prepared = build_audit_rows(
        rows,
        sample_size=10,
        seed=1,
        excluded_caption_keys={"a new road appeared."},
    )

    assert [row["caption"] for row in prepared] == ["A house was demolished."]

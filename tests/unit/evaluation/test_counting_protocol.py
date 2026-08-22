from sat_rs_vlm.evaluation.counting_protocol import summarize_exact_cardinality_counting


def test_exact_cardinality_metrics_exclude_non_cardinality_and_keep_parse_failures():
    rows = [
        {
            "id": "exact",
            "task_type": "counting",
            "question": "How many ships are visible?",
            "reference": "2",
            "prediction": "2",
        },
        {
            "id": "parse-failure",
            "task_type": "counting",
            "question": "How many ships are visible?",
            "reference": "3",
            "prediction": "Multiple",
        },
        {
            "id": "binary",
            "task_type": "counting",
            "question": "Are there multiple ships visible?",
            "reference": "yes",
            "prediction": "yes",
        },
        {
            "id": "bad-reference",
            "task_type": "counting",
            "question": "How many ships are visible?",
            "reference": "Unable to determine",
            "prediction": "2",
        },
    ]
    report = summarize_exact_cardinality_counting(rows)
    assert report["diagnostics"] == {
        "raw_task_type_counting_rows": 4,
        "excluded_non_cardinality": 1,
        "excluded_missing_question": 0,
        "excluded_invalid_reference": 1,
        "valid_cardinality_rows": 2,
        "prediction_parse_failures": 1,
    }
    assert report["overall"]["n"] == 2
    assert report["overall"]["parsed_n"] == 1
    assert report["overall"]["prediction_parse_rate"] == 0.5
    assert report["overall"]["acc_exact"] == 0.5
    assert report["overall"]["acc_within_1"] == 0.5
    assert report["overall"]["mae"] == 0.0

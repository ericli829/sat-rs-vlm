from taskgraph_lab.evaluation.planner_generation import (
    evaluate_prediction,
    summarize_predictions,
)


def _row() -> dict:
    target = (
        "INTENT(SIMPLE_COUNT)\n"
        'n1=COUNT_IMAGE($image0,T("airplane"),true)\n'
        "FINAL($n1,INTEGER)"
    )
    return {
        "id": "sample-1",
        "messages": [
            {"role": "system", "content": "Return DSL."},
            {
                "role": "user",
                "content": (
                    '{"question":"How many airplanes are visible?",'
                    '"question_type":"INTEGER","choices":null,'
                    '"inputs":{"image0":{"type":"image","uri_or_key":"x.png"}}}'
                ),
            },
            {"role": "assistant", "content": target},
        ],
        "metadata": {"dataset": "fixture", "intent": "SIMPLE_COUNT"},
    }


def test_exact_prediction_passes_all_strict_metrics() -> None:
    row = _row()
    prediction = row["messages"][-1]["content"]
    result = evaluate_prediction(row, prediction)
    assert result["dsl_parse_valid"] is True
    assert result["runtime_valid"] is True
    assert result["canonical_exact"] is True
    assert result["text_exact"] is True


def test_markdown_wrapping_is_not_silently_repaired() -> None:
    row = _row()
    prediction = f"```\n{row['messages'][-1]['content']}\n```"
    result = evaluate_prediction(row, prediction)
    assert result["dsl_parse_valid"] is False
    assert result["runtime_valid"] is False
    assert result["parse_error"].startswith("DSLParseError:")


def test_summary_keeps_failures_in_denominator() -> None:
    row = _row()
    passed = evaluate_prediction(row, row["messages"][-1]["content"])
    failed = evaluate_prediction(row, "not dsl")
    summary = summarize_predictions([passed, failed])
    assert summary["sample_count"] == 2
    assert summary["counts"]["dsl_parse_valid"] == 1
    assert summary["rates"]["dsl_parse_valid"] == 0.5


def test_semantic_dead_node_keeps_surface_and_parse_layers_valid() -> None:
    row = _row()
    prediction = (
        "INTENT(SIMPLE_COUNT)\n"
        'n1=LOCATE($image0,T("ship"))\n'
        'n2=COUNT_IMAGE($image0,T("airplane"),true)\n'
        "FINAL($n2,INTEGER)"
    )
    result = evaluate_prediction(row, prediction)
    assert result["surface_grammar_valid"] is True
    assert result["dsl_parse_valid"] is True
    assert result["graph_runtime_valid"] is False
    assert "dead_node" in result["validation_error_codes"]

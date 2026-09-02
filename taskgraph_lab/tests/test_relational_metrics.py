from taskgraph_lab.evaluation.relational_metrics import relational_metrics


def _expected() -> dict:
    return {
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "harbor", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n1", "reference": "$n2"},
                "params": {"mode": "RELATION", "relation": "NEAR"},
            },
            {
                "id": "n4",
                "op": "COUNT",
                "inputs": {"entities": "$n3"},
                "params": {
                    "target": {"category": "ship", "attributes": {}},
                    "entire": False,
                },
            },
        ],
        "final": {"sources": ["$n4"], "answer_type": "INTEGER"},
    }


def test_relational_structure_metrics_distinguish_attachment_and_count_flow() -> None:
    expected = _expected()
    exact = relational_metrics(expected, expected)
    assert exact["relation_direction_accuracy"] is True
    assert exact["reference_attachment_accuracy"] is True
    assert exact["select_rel_vs_relation_accuracy"] is True
    assert exact["count_filtered_source_accuracy"] is True
    assert exact["count_scope_entire_accuracy"] is True
    assert exact["first_broken_relation_chain_depth"] is None

    broken = _expected()
    broken["nodes"][2]["inputs"]["reference"] = "$n1"
    broken["nodes"][3]["inputs"] = {"image": "$image0"}
    broken["nodes"][3]["params"]["entire"] = True
    metrics = relational_metrics(expected, broken, validation_error_codes=["dead_node"])
    assert metrics["relation_direction_accuracy"] is True
    assert metrics["reference_attachment_accuracy"] is False
    assert metrics["count_filtered_source_accuracy"] is False
    assert metrics["count_scope_entire_accuracy"] is False
    assert metrics["dead_node"] is True
    assert metrics["first_broken_relation_chain_depth"] == 1


def test_select_rel_vs_relation_is_reported_separately() -> None:
    expected = _expected()
    predicted = _expected()
    predicted["nodes"][2] = {
        "id": "n3",
        "op": "RELATION",
        "inputs": {"subject": "$n1", "reference": "$n2"},
        "params": {},
    }
    metrics = relational_metrics(expected, predicted)
    assert metrics["select_rel_vs_relation_accuracy"] is False
    assert metrics["reference_attachment_accuracy"] is True

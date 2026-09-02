from __future__ import annotations

from taskgraph_lab.generation.repair import classify_repair
from taskgraph_lab.taskgraph.canonicalize import normalize_candidate_payload
from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.validator import validate_candidate


def _target(category: str, **attributes: str) -> dict:
    return {"category": category, "attributes": attributes}


def test_a_whole_image_large_ship_count() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("ship", size="large"), "entire": True},
            }
        ],
        "final": {"source": "$n1", "answer_type": "INTEGER"},
    }
    target, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert report.valid, report.model_dump()
    assert target is not None and target.nodes[0].op.value == "COUNT"


def test_b_bbox_color_uses_node_refs_and_rejects_named_refs() -> None:
    graph = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION_FROM_BBOX",
                "inputs": {"image": "$image0"},
                "params": {"bbox": [1, 2, 3, 4]},
            },
            {
                "id": "n2",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n1"},
                "params": {"attribute": "color"},
            },
        ],
        "final": {"source": "$n2", "answer_type": "LABEL"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}}, question="bbox color")
    assert report.valid, report.model_dump()
    for named_ref in ("$region", "$color"):
        graph["nodes"][1]["inputs"] = {"entity": named_ref}
        _, invalid = validate_candidate(graph, inputs={"image0": {}}, question="bbox color")
        assert any(issue.code == "invalid_reference" for issue in invalid.errors)


def test_c_relational_count_consumes_select_as_entities() -> None:
    graph = {
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": _target("building")},
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("umbrella", color="red")},
            },
            {
                "id": "n4",
                "op": "SELECT",
                "inputs": {"candidates": "$n3", "reference": "$n2"},
                "params": {"mode": "RELATION", "relation": "NEXT_TO"},
            },
            {
                "id": "n5",
                "op": "COUNT",
                "inputs": {"entities": "$n4"},
                "params": {"target": _target("umbrella", color="red"), "entire": False},
            },
        ],
        "final": {"source": "$n5", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert report.valid, report.model_dump()

    graph["nodes"][4]["inputs"] = {"image": "$image0"}
    _, invalid = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    codes = {issue.code for issue in invalid.errors}
    assert "relation_result_not_consumed" in codes
    assert "dead_node" in codes


def test_relational_count_consumes_selected_region_through_count_image() -> None:
    graph = {
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "BOTTOM_LEFT"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": _target("building", color="yellow")},
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2"},
                "params": {"mode": "SUBREGION", "subregion": "LEFT_SIDE"},
            },
            {
                "id": "n4",
                "op": "COUNT",
                "inputs": {"image": "$n3"},
                "params": {
                    "target": _target("container", color="yellow"),
                    "entire": False,
                },
            },
        ],
        "final": {"source": "$n4", "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE",
    )
    assert report.valid, report.model_dump()
    assert "relation_result_not_consumed" not in {issue.code for issue in report.errors}

    graph["nodes"][2]["params"] = {
        "mode": "RANK",
        "criterion": "area",
        "rank": 1,
        "order": "ASCENDING",
    }
    _, ambiguous_scope = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE",
    )
    assert "relation_result_not_consumed" in {issue.code for issue in ambiguous_scope.errors}


def test_count_entities_requires_entire_false() -> None:
    graph = {
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car")},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "EXTREME", "direction": "LEFTMOST"},
            },
            {
                "id": "n3",
                "op": "COUNT",
                "inputs": {"entities": "$n2"},
                "params": {"target": _target("car"), "entire": True},
            },
        ],
        "final": {"source": "$n3", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert any(issue.code == "count_entities_requires_non_entire" for issue in report.errors)


def test_d_object_relation_uses_relation_not_select_or_vlm() -> None:
    graph = {
        "intent": "OBJECT_RELATION",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("building", color="white")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car", color="red")},
            },
            {
                "id": "n3",
                "op": "RELATION",
                "inputs": {"subject": "$n1", "reference": "$n2"},
                "params": {},
            },
        ],
        "final": {"source": "$n3", "answer_type": "LABEL"},
    }
    question = "Where is the white building relative to the red car?"
    _, report = validate_candidate(graph, inputs={"image0": {}}, question=question)
    assert report.valid, report.model_dump()

    graph["nodes"] = [
        {
            "id": "n1",
            "op": "VLM_REASON",
            "inputs": {"image": "$image0"},
            "params": {"question": "$question"},
        }
    ]
    graph["final"]["source"] = "$n1"
    graph["final"]["answer_type"] = "TEXT"
    _, invalid = validate_candidate(graph, inputs={"image0": {}}, question=question)
    codes = {issue.code for issue in invalid.errors}
    assert "dedicated_operator_bypass" in codes
    assert "relation_query_should_use_relation" in codes


def test_e_route_accepts_entity_sets_without_fake_ordinal() -> None:
    graph = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("roundabout")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("pond")},
            },
            {
                "id": "n3",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n1", "goal": "$n2"},
                "params": {},
            },
            {
                "id": "n4",
                "op": "ROUTE_REASON",
                "inputs": {"context": "$n3"},
                "params": {"question": "$question", "choices": "$choices"},
            },
        ],
        "final": {"source": "$n4", "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert all(node["op"] != "SELECT" for node in graph["nodes"])


def test_f_natural_colored_objects_do_not_trigger_marker_heuristic() -> None:
    graph = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("wall", color="white")},
            },
            {
                "id": "n2",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n1"},
                "params": {"attribute": "position"},
            },
        ],
        "final": {"source": "$n2", "answer_type": "LABEL"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question="What attribute describes the white wall beside the natural red car?",
    )
    assert report.valid, report.model_dump()
    assert not any("marker" in issue.code for issue in report.warnings)


def test_g_dead_node_is_rejected() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("ship"), "entire": True},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car")},
            },
        ],
        "final": {"source": "$n1", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert any(issue.code == "dead_node" and issue.node_id == "n2" for issue in report.errors)


def test_h_dedicated_operator_bypass_is_rejected() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "VLM_REASON",
                "inputs": {"image": "$image0"},
                "params": {"question": "$question"},
            }
        ],
        "final": {"source": "$n1", "answer_type": "TEXT"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert any(issue.code == "dedicated_operator_bypass" for issue in report.errors)


def test_i_lexical_normalization_is_narrow_and_recorded() -> None:
    graph = {
        "intent": "REGIONAL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "top right corner"},
                "output": "legacy_region",
            },
            {
                "id": "n2",
                "op": "CLASSIFY",
                "inputs": {"input": "$n1"},
                "params": {},
            },
        ],
        "final": {"source": "$n2", "answer_type": "LABEL"},
    }
    target, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()
    assert target is not None and target.nodes[0].params["position"] == "TOP_RIGHT"
    rules = {change["rule"] for change in report.normalized_fields}
    assert {"canonical_enum_alias", "remove_legacy_graphnode_output"}.issubset(rules)


def test_j_unknown_relation_is_not_guessed() -> None:
    graph = {
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car")},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "RELATION", "relation": "relative position"},
            },
            {
                "id": "n3",
                "op": "COUNT",
                "inputs": {"entities": "$n2"},
                "params": {"target": _target("car"), "entire": False},
            },
        ],
        "final": {"source": "$n3", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert not report.schema_valid
    assert not any(
        change["path"] == "nodes[1].params.relation" for change in report.normalized_fields
    )
    assert classify_repair(report) == "SEMANTIC_ERROR"
    normalized, changes = normalize_candidate_payload(graph)
    assert normalized["nodes"][1]["params"]["relation"] == "relative position"
    assert not any(change["path"] == "nodes[1].params.relation" for change in changes)


def test_new_schema_rejects_output_but_validator_normalizes_legacy() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("ship"), "entire": True},
                "output": "legacy_count",
            }
        ],
        "final": {"source": "$n1", "answer_type": "INTEGER"},
    }
    try:
        PlannerTarget.model_validate(graph)
    except ValueError:
        pass
    else:
        raise AssertionError("GraphNode.output must not be part of the v1.1 schema")
    target, report = validate_candidate(graph, inputs={"image0": {}}, question_type="INTEGER")
    assert target is not None and report.valid
    assert any(
        change["rule"] == "remove_legacy_graphnode_output" for change in report.normalized_fields
    )

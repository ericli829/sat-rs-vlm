from __future__ import annotations

from typing import Any


def target(category: str, **attributes: str | int | float | bool) -> dict[str, Any]:
    return {"category": category, "attributes": attributes}


def final(source: str | list[str], answer_type: str, question: str | None = None) -> dict:
    payload = {
        "sources": [source] if isinstance(source, str) else source,
        "answer_type": answer_type,
    }
    if question is not None:
        payload["question"] = question
    return payload


def representative_graphs() -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    graphs["whole_image_count"] = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": target("ship", size="large"), "entire": True},
            }
        ],
        "final": final("$n1", "CHOICE_SINGLE"),
    }
    graphs["relational_count"] = {
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
                "params": {"target": target("building")},
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("sun umbrella", color="red")},
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
                "params": {
                    "target": target("sun umbrella", color="red"),
                    "entire": False,
                },
            },
        ],
        "final": final("$n5", "CHOICE_SINGLE"),
    }
    graphs["bbox_attribute"] = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION_FROM_BBOX",
                "inputs": {"image": "$image0"},
                "params": {"bbox": [1, 2.5, 30, 40], "image_size": [100, 200]},
            },
            {
                "id": "n2",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n1"},
                "params": {"attribute": "color", "part": "roof"},
            },
        ],
        "final": final("$n2", "LABEL"),
    }
    graphs["relation"] = {
        "intent": "OBJECT_RELATION",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("building")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("car")},
            },
            {
                "id": "n3",
                "op": "RELATION",
                "inputs": {"subject": "$n1", "reference": "$n2"},
                "params": {},
            },
        ],
        "final": final("$n3", "CHOICE_SINGLE"),
    }
    select_modes = {
        "select_rank": {"mode": "RANK", "criterion": "size", "rank": 2, "order": "DESCENDING"},
        "select_ord": {"mode": "ORDINAL", "index": 2, "order": "LEFT_TO_RIGHT"},
        "select_extreme": {"mode": "EXTREME", "direction": "RIGHTMOST"},
        "select_subregion": {"mode": "SUBREGION", "subregion": "BOTH_SIDES"},
    }
    for name, params in select_modes.items():
        graphs[name] = {
            "intent": "COMPLEX_REASONING",
            "nodes": [
                {
                    "id": "n1",
                    "op": "LOCATE",
                    "inputs": {"image": "$image0"},
                    "params": {"target": target("building")},
                },
                {
                    "id": "n2",
                    "op": "SELECT",
                    "inputs": {"candidates": "$n1"},
                    "params": params,
                },
            ],
            "final": final(
                "$n2", "CHOICE_SINGLE", "Which option describes the selected object?"
            ),
        }
    for mode in ("ROW", "COLUMN", "CLUSTER"):
        graphs[f"group_{mode.lower()}"] = {
            "intent": "COMPLEX_REASONING",
            "nodes": [
                {
                    "id": "n1",
                    "op": "LOCATE",
                    "inputs": {"image": "$image0"},
                    "params": {"target": target("house")},
                },
                {
                    "id": "n2",
                    "op": "GROUP",
                    "inputs": {"entities": "$n1"},
                    "params": {"mode": mode},
                },
            ],
            "final": final(
                "$n2", "CHOICE_SINGLE", "Which option describes this selected group?"
            ),
        }
    graphs["classify_optional_labels"] = {
        "intent": "REGIONAL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n2",
                "op": "CLASSIFY",
                "inputs": {"input": "$n1"},
                "params": {"label_space": ["farm", "harbor"]},
            },
        ],
        "final": final("$n2", "LABEL"),
    }
    graphs["classify_without_labels"] = {
        "intent": "OBJECT_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "CLASSIFY",
                "inputs": {"input": "$image0"},
                "params": {},
            }
        ],
        "final": final("$n1", "LABEL"),
    }
    graphs["multilabel"] = {
        "intent": "MULTILABEL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "MULTILABEL_CLASSIFY",
                "inputs": {"input": "$image0"},
                "params": {"label_space": ["water", "farmland", "building"]},
            }
        ],
        "final": final("$n1", "LABEL_SET"),
    }
    graphs["motion"] = {
        "intent": "MOTION_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("ship")},
            },
            {"id": "n2", "op": "MOTION", "inputs": {"input": "$n1"}, "params": {}},
        ],
        "final": final("$n2", "BOOLEAN"),
    }
    graphs["abs_diff_multi_image"] = {
        "intent": "CHANGE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": target("farm"), "entire": True},
            },
            {
                "id": "n2",
                "op": "COUNT",
                "inputs": {"image": "$image1"},
                "params": {"target": target("farm"), "entire": True},
            },
            {
                "id": "n3",
                "op": "ABS_DIFF",
                "inputs": {"a": "$n1", "b": "$n2"},
                "params": {},
            },
        ],
        "final": final("$n3", "INTEGER"),
    }
    graphs["find_marker"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "FIND_MARKER",
                "inputs": {"image": "$image0"},
                "params": {"marker": {"shape": "circle", "color": "red"}},
            }
        ],
        "final": final("$n1", "CHOICE_SINGLE", "What is visible in this marked region?"),
    }
    graphs["find_marker_without_color"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "FIND_MARKER",
                "inputs": {"image": "$image0"},
                "params": {"marker": {"shape": "rectangle"}},
            }
        ],
        "final": final("$n1", "CHOICE_SINGLE", "What is visible in this marked region?"),
    }
    graphs["route_context"] = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("roundabout")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("pond")},
            },
            {
                "id": "n3",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n1", "goal": "$n2"},
                "params": {},
            },
        ],
        "final": final(
            "$n3",
            "CHOICE_SINGLE",
            "Which option describes the best route between the selected start and goal?",
        ),
    }
    graphs["multi_source_residual"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("pond")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("farmland")},
            },
        ],
        "final": final(
            ["$n1", "$n2"],
            "CHOICE_SINGLE",
            "What is the most likely purpose of these ponds in this agricultural setting?",
        ),
    }
    graphs["intermediate_vlm"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_RIGHT"},
            },
            {
                "id": "n2",
                "op": "VLM_REASON",
                "inputs": {"image": "$image0", "evidence": ["$n1"]},
                "params": {"question": "$question", "choices": None},
            },
            {
                "id": "n3",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n2"},
                "params": {"choices": "$choices"},
            },
        ],
        "final": final("$n3", "CHOICE_SINGLE"),
    }
    graphs["intermediate_vlm_single_evidence"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "BOTTOM_LEFT"},
            },
            {
                "id": "n2",
                "op": "VLM_REASON",
                "inputs": {"evidence": "$n1"},
                "params": {"question": "$question", "choices": "$choices"},
            },
            {
                "id": "n3",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n2"},
                "params": {"choices": "$choices"},
            },
        ],
        "final": final("$n3", "CHOICE_SINGLE"),
    }
    graphs["multi_source_structured"] = {
        "intent": "CHANGE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": target("ship"), "entire": True},
            },
            {
                "id": "n2",
                "op": "COUNT",
                "inputs": {"image": "$image1"},
                "params": {"target": target("ship"), "entire": True},
            },
        ],
        "final": final(["$n1", "$n2"], "CHOICE_MULTI"),
    }
    graphs["legacy_route_reason"] = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("roundabout")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target("pond")},
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
        "final": final("$n4", "CHOICE_SINGLE"),
    }
    graphs["legacy_match_choice"] = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": target("ship"), "entire": True},
            },
            {
                "id": "n2",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n1"},
                "params": {"choices": "$choices"},
            },
        ],
        "final": final("$n2", "CHOICE_SINGLE"),
    }
    graphs["escaped_and_all_attributes"] = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": target(
                        '仓库 "A"\\北侧\n入口',
                        color="deep red",
                        shape="L-shaped",
                        size="large",
                        state="open",
                        pattern="striped",
                        has_part="sloped roof",
                    )
                },
            }
        ],
        "final": final(
            "$n1",
            "CHOICE_SINGLE",
            'What is visible on the selected "仓库"\\section?\nDescribe it.',
        ),
    }
    return graphs

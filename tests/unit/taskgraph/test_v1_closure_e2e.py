from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from sat_rs_vlm.taskgraph import RuntimeRequest, fake_runtime
from sat_rs_vlm.taskgraph.runtime_types import Boolean, ChoiceScoreResult, Label, LabelSet, Region


def _image(path: Path, *, markers: bool = False) -> str:
    canvas = Image.new("RGB", (400, 300), "white")
    if markers:
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((40, 40, 65, 65), outline="red", width=4)
        draw.ellipse((180, 120, 205, 145), outline="red", width=4)
        draw.point((5, 5), fill="red")
    canvas.save(path)
    return str(path.resolve())


def _graph(
    question: str,
    nodes: list[dict[str, object]],
    source: str,
    answer_type: str,
    *,
    options: tuple[str, ...] = (),
    final_question: str = "",
    image_count: int = 1,
) -> dict[str, object]:
    question_type = (
        "MULTIPLE_CHOICE_MULTI"
        if answer_type == "CHOICE_MULTI"
        else "MULTIPLE_CHOICE_SINGLE"
        if answer_type == "CHOICE_SINGLE"
        else "FREE_FORM"
    )
    return {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": question_type,
        "choices": list(options) if options else None,
        "inputs": {
            f"image{index}": {"type": "image", "uri_or_key": "runtime"}
            for index in range(image_count)
        },
        "intent": "OTHER",
        "nodes": nodes,
        "final": {
            "sources": [source],
            "question": final_question,
            "answer_type": answer_type,
        },
    }


def _run(runtime, graph: dict[str, object], images: tuple[str, ...], options=()):
    return runtime.run(
        RuntimeRequest(
            "v1-closure",
            "XLRS_Bench",
            "audit",
            str(graph["question"]),
            images,
            tuple(options),
            graph=graph,
        )
    )


def test_f1_locate_to_attribute_choice_uses_local_visual(tmp_path: Path) -> None:
    image = _image(tmp_path / "f1.png")
    graph = _graph(
        "What color is the selected building?",
        [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "building", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n1"},
                "params": {"attribute": "color", "part": None},
            },
        ],
        "$n2",
        "CHOICE_SINGLE",
        options=("A white", "B black"),
        final_question="Choose the color of the already localized building.",
    )
    runtime = fake_runtime(
        detection_boxes=[[100, 80, 180, 160]],
        semantic_choice_scores={"final_attribute_choice_fusion": {"A": 2.0, "B": -1.0}},
    )
    try:
        result = _run(runtime, graph, (image,), ("A white", "B black"))
        assert result.output.choice_id == "A"
        model_input = runtime.providers.semantic_2b.choice_calls[0].model_input
        assert len(model_input.visual_inputs) == 1
        with Image.open(model_input.visual_inputs[0]) as crop:
            assert crop.size == (80, 80)
    finally:
        runtime.close()


def test_f2_structured_count_maps_choice_without_vlm(tmp_path: Path) -> None:
    image = _image(tmp_path / "f2.png")
    options = ("A 1", "B 2")
    graph = _graph(
        "How many objects?",
        [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": {"category": "object", "attributes": {}},
                    "entire": True,
                },
            }
        ],
        "$n1",
        "CHOICE_SINGLE",
        options=options,
    )
    runtime = fake_runtime(detection_boxes=[[1, 1, 5, 5], [10, 10, 15, 15]])
    try:
        result = _run(runtime, graph, (image,), options)
        assert result.output.choice_id == "B"
        assert runtime.providers.semantic_2b.choice_calls == []
    finally:
        runtime.close()


def test_f3_visual_final_fuses_once_with_residual_question(tmp_path: Path) -> None:
    image = _image(tmp_path / "f3.png")
    options = ("A airport", "B harbor")
    graph = _graph(
        "What is this place?",
        [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n2",
                "op": "VLM_REASON",
                "inputs": {"image": "$n1"},
                "params": {"question": "$question", "choices": "$choices"},
            },
        ],
        "$n2",
        "CHOICE_SINGLE",
        options=options,
        final_question="Classify only the supplied view.",
    )
    runtime = fake_runtime(
        semantic_choice_scores={"final_vlm_reason_choice_fusion": {"A": -1.0, "B": 3.0}}
    )
    try:
        result = _run(runtime, graph, (image,), options)
        assert result.output.choice_id == "B"
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
        request = runtime.providers.semantic_2b.choice_calls[0]
        assert "Classify only the supplied view." in request.model_input.question
        assert "What is this place?" not in request.model_input.question
    finally:
        runtime.close()


def test_f4_multilabel_returns_typed_label_set_from_one_cache(tmp_path: Path) -> None:
    image = _image(tmp_path / "f4.png")
    graph = _graph(
        "Which land-use labels apply?",
        [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n2",
                "op": "MULTILABEL_CLASSIFY",
                "inputs": {"source": "$n1"},
                "params": {"label_space": ["airport", "harbor", "residential"]},
            },
        ],
        "$n2",
        "LABEL_SET",
    )
    runtime = fake_runtime(
        semantic_choice_scores={
            "semantic_multilabel_classify": {
                "airport": 2.0,
                "harbor": -1.0,
                "residential": 1.0,
            }
        }
    )
    try:
        result = _run(runtime, graph, (image,))
        assert isinstance(result.output, LabelSet)
        assert result.output.values == ("airport", "residential")
        assert len(runtime.providers.semantic_2b.semantic_calls) == 1
    finally:
        runtime.close()


def test_f5_swapped_motion_keeps_explicit_temporal_roles(tmp_path: Path) -> None:
    before = _image(tmp_path / "before.png")
    after = _image(tmp_path / "after.png")
    graph = _graph(
        "Did the target move?",
        [
            {
                "id": "n1",
                "op": "MOTION",
                "inputs": {"before": "$image1", "after": "$image0"},
                "params": {},
            }
        ],
        "$n1",
        "BOOLEAN",
        image_count=2,
    )
    runtime = fake_runtime(semantic_choice_scores={"semantic_motion": {"YES": 2.0, "NO": -1.0}})
    try:
        result = _run(runtime, graph, (before, after))
        assert isinstance(result.output, Boolean)
        model_input = runtime.providers.semantic_2b.semantic_calls[0].model_input
        assert model_input.visual_roles == ("BEFORE", "AFTER")
        assert model_input.visual_inputs == (after, before)
    finally:
        runtime.close()


def test_f6_route_context_to_4b_choice_preserves_trace_metadata(tmp_path: Path) -> None:
    image = _image(tmp_path / "f6.png")
    options = ("A go north", "B go east")
    graph = _graph(
        "What is the shortest route from the marked start to goal?",
        [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_LEFT"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "start", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "BOTTOM_RIGHT"},
            },
            {
                "id": "n4",
                "op": "LOCATE",
                "inputs": {"image": "$n3"},
                "params": {"target": {"category": "goal", "attributes": {}}},
            },
            {
                "id": "n5",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n2", "goal": "$n4"},
                "params": {},
            },
            {
                "id": "n6",
                "op": "ROUTE_REASON",
                "inputs": {"context": "$n5"},
                "params": {"question": "$question", "choices": "$choices"},
            },
        ],
        "$n6",
        "CHOICE_SINGLE",
        options=options,
        final_question="Choose the feasible shortest navigation.",
    )
    runtime = fake_runtime(
        detection_boxes=[[20, 30, 45, 55], [330, 220, 360, 250]],
        route_choice_scores={"route_choice": {"A": -1.0, "B": 3.0}},
    )
    try:
        result = _run(runtime, graph, (image,), options)
        assert result.output.choice_id == "B"
        assert runtime.providers.semantic_2b.choice_calls == []
        request = runtime.providers.route_4b.choice_calls[0]
        route_metadata = request.model_input.metadata["route_context"]
        assert route_metadata["prompt_version"] == "route-v1"
        assert route_metadata["endpoints"]["start"]["marker_color"] == "green"
        assert route_metadata["endpoints"]["goal"]["marker_color"] == "red"
        score = result.store.get("$n6")
        assert isinstance(score, ChoiceScoreResult)
        assert "input_metadata" in score.metadata
    finally:
        runtime.close()


def test_f7_marker_regions_flow_through_select_to_attribute(tmp_path: Path) -> None:
    image = _image(tmp_path / "f7.png", markers=True)
    graph = _graph(
        "What color is the leftmost artificial circle marker?",
        [
            {
                "id": "n1",
                "op": "FIND_MARKER",
                "inputs": {"image": "$image0"},
                "params": {"marker": {"color": "red", "shape": "circle"}},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "EXTREME", "direction": "LEFTMOST"},
            },
            {
                "id": "n3",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n2"},
                "params": {"attribute": "color", "part": None},
            },
        ],
        "$n3",
        "LABEL",
    )
    runtime = fake_runtime(semantic_responses={"attribute": "red"})
    try:
        result = _run(runtime, graph, (image,))
        markers = result.store.get("$n1")
        assert len(markers.regions) == 2
        assert markers.provenance["rejected_component_count"] == 1
        selected = result.store.get("$n2")
        assert selected.selected.regions[0].bbox_xyxy_global == markers.regions[0].bbox_xyxy_global
        assert isinstance(result.output, Label)
        assert result.output.value == "red"
        assert runtime.providers.semantic_2b.calls[0].model_input.visual_roles == ("ENTITY",)
    finally:
        runtime.close()


def test_f8_nested_bbox_to_attribute_keeps_global_coordinates(tmp_path: Path) -> None:
    image = _image(tmp_path / "f8.png")
    graph = _graph(
        "What material is visible in the nested target?",
        [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n2",
                "op": "REGION_FROM_BBOX",
                "inputs": {"image": "$n1"},
                "params": {"bbox": [25, 25, 125, 90], "image_size": [200, 150]},
            },
            {
                "id": "n3",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n2"},
                "params": {"attribute": "material", "part": None},
            },
        ],
        "$n3",
        "LABEL",
    )
    runtime = fake_runtime(semantic_responses={"attribute": "concrete"})
    try:
        result = _run(runtime, graph, (image,))
        nested = result.store.get("$n2")
        assert isinstance(nested, Region)
        assert nested.bbox_xyxy_global == (100.0, 75.0, 250.0, 180.0)
        assert isinstance(result.output, Label)
        assert result.output.value == "concrete"
    finally:
        runtime.close()


def test_f9_semantic_region_locate_uses_retriever_then_qwen(tmp_path: Path) -> None:
    image = _image(tmp_path / "f9.png")
    graph = _graph(
        "What type of harbor is in the retrieved region?",
        [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "harbor", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "EXTREME", "direction": "LEFTMOST"},
            },
            {
                "id": "n3",
                "op": "CLASSIFY",
                "inputs": {"source": "$n2"},
                "params": {"label_space": ["commercial", "industrial"]},
            },
        ],
        "$n3",
        "LABEL",
    )
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 10, 10]],
        retrieval_candidates=[([40, 30, 180, 160], 0.9)],
        semantic_choice_scores={
            "semantic_classify": {"commercial": -1.0, "industrial": 2.0}
        },
        semantic_categories={"harbor"},
    )
    try:
        result = _run(runtime, graph, (image,))
        located = result.store.get("$n1")
        assert located.provenance["capability"] == "region_retrieval"
        assert located.entities[0].region.bbox_xyxy_global == (40.0, 30.0, 180.0, 160.0)
        assert runtime.providers.detection.calls == []
        assert isinstance(result.output, Label)
        assert result.output.value == "industrial"
        assert len(runtime.providers.semantic_2b.semantic_calls) == 1
        assert result.trace.retriever_ms is not None and result.trace.retriever_ms >= 0.0
        assert result.trace.semantic_vlm_ms is not None and result.trace.semantic_vlm_ms >= 0.0
        assert result.trace.stage_status["retriever_ms"] == "executed"
        assert result.trace.stage_status["semantic_vlm_ms"] == "executed"
        assert result.trace.stage_status["choice_ms"] == "not_used"
        assert result.trace.activated_model_roles == ["retriever", "semantic_2b"]
    finally:
        runtime.close()


def test_f10_object_locate_remains_bound_to_lae(tmp_path: Path) -> None:
    image = _image(tmp_path / "f10.png")
    graph = _graph(
        "Locate the aircraft.",
        [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "aircraft", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "COUNT",
                "inputs": {"entities": "$n1"},
                "params": {
                    "target": {"category": "aircraft", "attributes": {}},
                    "entire": False,
                },
            },
        ],
        "$n2",
        "INTEGER",
    )
    runtime = fake_runtime(
        detection_boxes=[[20, 20, 80, 70]],
        retrieval_candidates=[([200, 200, 300, 280], 0.99)],
        semantic_categories={"harbor"},
    )
    try:
        result = _run(runtime, graph, (image,))
        located = result.store.get("$n1")
        assert located.entities[0].region.bbox_xyxy_global == (20.0, 20.0, 80.0, 70.0)
        assert result.output.value == 1
        assert len(runtime.providers.detection.calls) == 1
    finally:
        runtime.close()

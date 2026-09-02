from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from sat_rs_vlm.taskgraph.choice import ChoiceRequest
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import OperatorContext, SelectExecutor, SemanticExecutor
from sat_rs_vlm.taskgraph.providers import ChoiceScoringRequest
from sat_rs_vlm.taskgraph.runtime import RuntimeRequest, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import (
    Boolean,
    ChoiceScoreResult,
    Entity,
    EntitySet,
    ImageRef,
    Label,
    Region,
    RouteContext,
    ScalarInt,
    SelectResult,
    SelectStatus,
)
from sat_rs_vlm.taskgraph.schema import AnswerType, GraphNode


def _config(kind: str) -> dict:
    path = os.environ.get("TASKGRAPH_REAL_CONFIG")
    if not path:
        pytest.skip("TASKGRAPH_REAL_CONFIG is not configured")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    providers = config["providers"]
    if kind == "lae":
        for name in ("semantic_2b", "route_4b", "choice"):
            providers[name] = {"kind": "fake", "default": "A"}
        providers["region_retriever"] = {"kind": "fake"}
    else:
        providers["detection"] = {"kind": "fake"}
        providers["region_retriever"] = {"kind": "fake"}
        providers.pop("planner", None)
        if kind == "qwen_2b":
            providers["route_4b"] = {"kind": "fake", "default": "A"}
            providers["choice"] = {"reuse": "semantic_2b"}
        elif kind == "qwen_4b":
            providers["semantic_2b"] = {"kind": "fake", "default": "A"}
            providers["choice"] = {"reuse": "semantic_2b"}
    return config


def _local_model(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value or not Path(value).is_dir():
        pytest.skip(f"{variable} is missing or is not a local model directory")
    return Path(value)


@pytest.mark.real_model
def test_real_lae_detection_contract() -> None:
    if os.environ.get("RUN_REAL_LAE") != "1":
        pytest.skip("set RUN_REAL_LAE=1 for the LAE smoke")
    image = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image or not Path(image).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    runtime = runtime_from_config(_config("lae"))
    try:
        from sat_rs_vlm.taskgraph.providers import DetectionRequest
        from sat_rs_vlm.taskgraph.schema import TargetSpec

        result = runtime.providers.detection.detect(
            DetectionRequest(ImageRef(image), TargetSpec(category="ship"))
        )
        for entity in result.detections.entities:
            x0, y0, x1, y1 = entity.region.bbox_xyxy_global
            assert x0 < x1 and y0 < y1
            assert entity.score is None or 0.0 <= entity.score <= 1.0
            assert entity.provenance.get("provider")
    finally:
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_visual_and_structured_choice_contracts() -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image or not Path(image).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    _local_model("QWEN3VL_2B_MODEL_DIR")
    runtime = runtime_from_config(_config("qwen_2b"))
    try:
        assert runtime.providers.choice is runtime.providers.semantic_2b
        visual = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ImageRef(image),),
                "Which option best describes this crop?",
                ("A urban", "B water"),
            )
        )
        assert visual.choice_id in {"A", "B"}
        assert visual.answer_type == "CHOICE_SINGLE"
        assert visual.provenance["cache_reused"] is True
        assert visual.provenance["method"].startswith("kv_cached_")
        assert "legacy" not in visual.provenance["method"]
        multi_question = "Select every mathematically true statement; image content is irrelevant."
        multi_options = ("A One plus one equals two", "B One plus one equals three")
        multi_probe = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ImageRef(image),),
                multi_question,
                multi_options,
                AnswerType.CHOICE_MULTI,
            )
        )
        assert set(multi_probe.selected_ids) <= {"A", "B"}
        assert multi_probe.choice_id is None
        assert multi_probe.answer_type == "CHOICE_MULTI"
        assert multi_probe.provenance["cache_reused"] is True
        assert multi_probe.provenance["method"] == "kv_cached_binary_verification"
        probe_score = runtime.choice_resolver.last_score_result
        model_input = runtime.choice_resolver.last_model_input
        assert probe_score is not None
        assert model_input is not None
        ranked_scores = sorted(probe_score.scores.values(), reverse=True)
        assert ranked_scores[0] > ranked_scores[1]
        one_threshold = (ranked_scores[0] + ranked_scores[1]) / 2.0
        multi_one_score = runtime.providers.semantic_2b.reason_and_choose(
            ChoiceScoringRequest(
                model_input=model_input,
                answer_type="CHOICE_MULTI",
                choice_ids=("A", "B"),
                option_texts=multi_options,
                single_choice_suffix=runtime.choice_config.single_choice_suffix,
                multi_verify_template=runtime.choice_config.multi_verify_template,
                multi_select_threshold=one_threshold,
                purpose="final_choice",
            )
        )
        assert len(multi_one_score.selected_ids) == 1
        assert multi_one_score.cache_reused is True
        multi_one = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (multi_one_score,),
                multi_question,
                multi_options,
                AnswerType.CHOICE_MULTI,
            )
        )
        assert len(multi_one.selected_ids) == 1
        assert multi_one.choice_id is None
        assert multi_one.answer_type == "CHOICE_MULTI"
        multi_many_score = runtime.providers.semantic_2b.reason_and_choose(
            ChoiceScoringRequest(
                model_input=model_input,
                answer_type="CHOICE_MULTI",
                choice_ids=("A", "B"),
                option_texts=multi_options,
                single_choice_suffix=runtime.choice_config.single_choice_suffix,
                multi_verify_template=runtime.choice_config.multi_verify_template,
                multi_select_threshold=min(probe_score.scores.values()) - 1.0,
                purpose="final_choice",
            )
        )
        assert multi_many_score.selected_ids == ("A", "B")
        assert multi_many_score.cache_reused is True
        structured = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ScalarInt(7),),
                "Which option matches this count?",
                ("A 5", "B 7"),
            )
        )
        assert structured.choice_id == "B"
        assert structured.provenance["method"] == "structured_exact_option_mapping"
        assert runtime.choice_resolver.last_model_input.visual_inputs == ()
        engine = runtime.providers.semantic_2b._engine
        assert engine is not None
        assert engine.active_session_count == 0
    finally:
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_select_geometry_and_cached_verification_contracts(tmp_path: Path) -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image_path = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image_path or not Path(image_path).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    _local_model("QWEN3VL_2B_MODEL_DIR")
    with Image.open(image_path) as source:
        width, height = source.size
    image = ImageRef(image_path, width=width, height=height)
    candidates = EntitySet(
        (
            Entity(
                Region(
                    image,
                    (width * 0.10, height * 0.35, width * 0.25, height * 0.55),
                ),
                "candidate object",
                0.9,
                {"candidate_id": "smoke-a"},
            ),
            Entity(
                Region(
                    image,
                    (width * 0.55, height * 0.35, width * 0.70, height * 0.55),
                ),
                "candidate object",
                0.8,
                {"candidate_id": "smoke-b"},
            ),
        )
    )
    reference = Entity(
        Region(
            image,
            (width * 0.78, height * 0.35, width * 0.92, height * 0.55),
        ),
        "reference object",
        0.95,
        {"candidate_id": "smoke-ref"},
    )

    def node(relation: str, selection_type: str) -> GraphNode:
        return GraphNode.model_validate(
            {
                "id": "n1",
                "op": "SELECT",
                "inputs": {"candidates": "$candidates", "reference": "$reference"},
                "params": {
                    "mode": "RELATION",
                    "relation": relation,
                    "selection_type": selection_type,
                },
            }
        )

    runtime = runtime_from_config(_config("qwen_2b"))
    composer = InputComposer(tmp_path / "select-smoke-inputs")
    executor = SelectExecutor(runtime.providers.semantic_2b, runtime.choice_config)
    context = OperatorContext("SELECT smoke", (), composer)
    try:
        geometry = executor.execute(
            node("LEFT_OF", "MULTI"),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(geometry.value, SelectResult)
        assert geometry.value.method == "geometry"
        assert geometry.value.status is SelectStatus.OK
        assert runtime.providers.semantic_2b._engine is None

        multi = executor.execute(
            node("NEXT_TO", "MULTI"),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(multi.value, SelectResult)
        assert multi.value.status in {SelectStatus.EMPTY, SelectStatus.OK}
        assert multi.value.method == "qwen3_vl_kv_cached_choice"
        assert multi.value.provenance["score_method"] == "kv_cached_binary_verification"
        assert multi.value.provenance["cache_reused"] is True
        assert multi.value.provenance["fallback_used"] is False
        assert multi.value.provenance["choice_metadata"]["session_released"] is True
        assert set(multi.value.provenance["final_candidate_ids"]) <= {
            "smoke-a",
            "smoke-b",
        }

        single = executor.execute(
            node("NEAR", "SINGLE"),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(single.value, SelectResult)
        assert single.value.status in {
            SelectStatus.EMPTY,
            SelectStatus.OK,
            SelectStatus.AMBIGUOUS,
        }
        assert single.value.method == "qwen3_vl_kv_cached_choice"
        assert single.value.provenance["score_method"] == "kv_cached_binary_verification"
        assert single.value.provenance["cache_reused"] is True
        assert single.value.provenance["choice_metadata"]["session_released"] is True
        engine = runtime.providers.semantic_2b._engine
        assert engine is not None
        assert engine.active_session_count == 0
    finally:
        composer.close()
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_semantic_alignment_and_final_fusion_contracts() -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image_path = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image_path or not Path(image_path).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    _local_model("QWEN3VL_2B_MODEL_DIR")
    with Image.open(image_path) as source:
        width, height = source.size
    image = ImageRef(image_path, width=width, height=height)
    subject = Entity(
        Region(image, (width * 0.10, height * 0.20, width * 0.30, height * 0.45)),
        "subject",
        0.9,
    )
    reference = Entity(
        Region(image, (width * 0.60, height * 0.20, width * 0.85, height * 0.45)),
        "reference",
        0.9,
    )
    config = _config("qwen_2b")
    config["providers"]["detection"] = {
        "kind": "fake",
        "boxes": [[width * 0.10, height * 0.10, width * 0.35, height * 0.35]],
    }
    runtime = runtime_from_config(config)
    semantic = SemanticExecutor(
        runtime.providers.semantic_2b,
        choice_config=runtime.choice_config,
        semantic_config=runtime.semantic_decision_config,
    )
    context = OperatorContext("Analyze the supplied crop.", (), runtime.composer)
    try:
        relation = semantic.execute(
            GraphNode.model_validate(
                {
                    "id": "n1",
                    "op": "RELATION",
                    "inputs": {"subject": "$subject", "reference": "$reference"},
                    "params": {},
                }
            ),
            {"subject": subject, "reference": reference},
            context,
        ).value
        assert isinstance(relation, Label)
        assert relation.provenance["canonical"] is True
        assert relation.provenance["cache_reused"] is True
        assert relation.provenance["decision_metadata"]["initial_prefill_tokens"] > 0
        assert relation.provenance["decision_metadata"]["session_released"] is True

        motion = semantic.execute(
            GraphNode.model_validate(
                {
                    "id": "n1",
                    "op": "MOTION",
                    "inputs": {"source": "$subject"},
                    "params": {},
                }
            ),
            {"source": subject},
            context,
        ).value
        assert isinstance(motion, Boolean)
        assert motion.provenance["cache_reused"] is True
        assert motion.provenance["decision_metadata"]["initial_prefill_tokens"] > 0
        assert motion.provenance["decision_metadata"]["session_released"] is True

        options = ("A Built-up area", "B Open land")
        question = "Which option best describes the selected visual evidence?"
        vlm_graph = {
            "version": "taskgraph-v1.1",
            "question": question,
            "question_type": "MULTIPLE_CHOICE_SINGLE",
            "choices": list(options),
            "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
            "nodes": [
                {
                    "id": "n1",
                    "op": "VLM_REASON",
                    "inputs": {"image": "$image0"},
                    "params": {"question": "$question", "choices": None},
                }
            ],
            "final": {
                "sources": ["$n1"],
                "question": "Choose the supported option.",
                "answer_type": "CHOICE_SINGLE",
            },
        }
        vlm_result = runtime.run(
            RuntimeRequest(
                "real-final-vlm",
                "XLRS_Bench",
                "complex_reasoning",
                question,
                (image_path,),
                options,
                graph=vlm_graph,
            )
        )
        vlm_score = vlm_result.store.get("$n1")
        assert isinstance(vlm_score, ChoiceScoreResult)
        assert vlm_score.cache_reused is True
        assert vlm_score.metadata["initial_prefill_tokens"] > 0
        assert vlm_score.metadata["session_released"] is True
        assert vlm_result.trace.nodes[-1].final_choice_fusion is True

        attribute_graph = {
            "version": "taskgraph-v1.1",
            "question": question,
            "question_type": "MULTIPLE_CHOICE_SINGLE",
            "choices": list(options),
            "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
            "nodes": [
                {
                    "id": "n1",
                    "op": "LOCATE",
                    "inputs": {"image": "$image0"},
                    "params": {"target": {"category": "object", "attributes": {}}},
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
                    "params": {"attribute": "land-cover appearance", "part": None},
                },
            ],
            "final": {
                "sources": ["$n3"],
                "question": "Choose the supported appearance option.",
                "answer_type": "CHOICE_SINGLE",
            },
        }
        attribute_result = runtime.run(
            RuntimeRequest(
                "real-final-attribute",
                "XLRS_Bench",
                "attribute",
                question,
                (image_path,),
                options,
                graph=attribute_graph,
            )
        )
        attribute_score = attribute_result.store.get("$n3")
        assert isinstance(attribute_score, ChoiceScoreResult)
        assert attribute_score.cache_reused is True
        assert attribute_score.metadata["initial_prefill_tokens"] > 0
        assert attribute_score.metadata["session_released"] is True
        assert attribute_result.trace.nodes[-1].final_choice_fusion is True

        engine = runtime.providers.semantic_2b._engine
        assert engine is not None
        assert engine.active_session_count == 0
    finally:
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_route_uses_same_4b_cached_choice() -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image_path = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image_path or not Path(image_path).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    _local_model("QWEN3VL_4B_MODEL_DIR")
    with Image.open(image_path) as source:
        width, height = source.size
    image = ImageRef(image_path, width=width, height=height)
    start = Region(image, (0, 0, max(1, width * 0.1), max(1, height * 0.1)))
    goal = Region(
        image,
        (max(0, width * 0.8), max(0, height * 0.8), width, height),
    )
    context = RouteContext(image, start, goal, Region(image, (0, 0, width, height)))
    options = ("A Move east then south", "B Move west then north")
    runtime = runtime_from_config(_config("qwen_4b"))
    try:
        model_input = runtime.composer.compose_named(
            {"context": context},
            question="Which route moves from START to GOAL?",
            options=options,
        )
        result = runtime.providers.route_4b.reason_and_choose(
            ChoiceScoringRequest(
                model_input=model_input,
                answer_type="CHOICE_SINGLE",
                choice_ids=("A", "B"),
                option_texts=options,
                single_choice_suffix=runtime.choice_config.single_choice_suffix,
                multi_verify_template=runtime.choice_config.multi_verify_template,
                purpose="route_choice",
            )
        )
        assert result.selected_ids[0] in {"A", "B"}
        assert result.cache_reused is True
        assert result.provider.endswith(":route_4b")
        assert result.metadata["session_released"] is True
        assert "legacy" not in result.method
        assert runtime.providers.choice is runtime.providers.semantic_2b
        assert runtime.providers.choice is not runtime.providers.route_4b
        assert runtime.providers.semantic_2b.choice_calls == []
        assert runtime.providers.semantic_2b.calls == []
        route_engine = runtime.providers.route_4b._engine
        assert route_engine is not None
        assert route_engine.active_session_count == 0
    finally:
        runtime.close()

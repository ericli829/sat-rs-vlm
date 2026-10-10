"""Capture actual prompt builders without loading model weights (before/after)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
from sat_rs_vlm.taskgraph.choice import ChoiceRequest, ChoiceResolver
from sat_rs_vlm.taskgraph.execution_plan import FinalChoiceFusionConfig
from sat_rs_vlm.taskgraph.executor import CapabilityRouter, ExecutorBinding, GraphExecutor
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    CountExecutor,
    LocateExecutor,
    OperatorContext,
    SelectExecutor,
    SemanticExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    CachedChoiceUnavailableError,
    FakeCountingProvider,
    FakeDetectionProvider,
    LazyQwenSemanticProvider,
)
from sat_rs_vlm.taskgraph.referent_refinement import ReferentRefinementConfig, ReferentRefiner
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region, runtime_summary
from sat_rs_vlm.taskgraph.schema import GraphNode, OperatorName, TargetSpec, TaskGraph
from sat_rs_vlm.taskgraph.store import RuntimeStore


class CaptureEngine:
    model_id = "AUDIT_MOCK_NO_WEIGHTS"
    last_generation_telemetry: dict = {}

    def __init__(self):
        self.calls = []
        self.finite_verdicts = []
        self.cached_unavailable = False
        self.text_response = "white"

    def generate_text(self, *, prompt, image_paths, allowed_outputs=None):
        self.calls.append(
            dict(
                prompt=prompt,
                visuals=[str(p) for p in image_paths],
                messages=HuggingFaceVLMEngine._build_messages(
                    prompt, ["<IMAGE>"] * len(image_paths)
                ),
                allowed_outputs=allowed_outputs,
                suffixes=[],
                mock=True,
            )
        )
        if allowed_outputs:
            return allowed_outputs[0]
        return (
            '{"status":"OK","bbox":[100,60,200,140]}'
            if '"status"' in prompt
            else self.text_response
        )

    def reason_and_choose(self, prompt, image_paths, **kwargs):
        if self.cached_unavailable:
            raise CachedChoiceUnavailableError("audit: cached scoring unavailable")
        ids = kwargs["choice_ids"]
        verdict = self.finite_verdicts.pop(0) if self.finite_verdicts else ids[0]
        assert verdict in ids, (verdict, ids)
        multi = kwargs["answer_type"] == "CHOICE_MULTI"
        suffixes = (
            [
                kwargs["multi_verify_template"].format(choice_id=i, option_text=t)
                for i, t in zip(ids, kwargs["option_texts"], strict=True)
            ]
            if multi
            else [kwargs["single_choice_suffix"]]
        )
        self.calls.append(
            dict(
                prompt=prompt,
                visuals=[str(p) for p in image_paths],
                messages=HuggingFaceVLMEngine._build_messages(
                    prompt, ["<IMAGE>"] * len(image_paths)
                ),
                terminal_candidates=["YES", "NO"] if multi else ids,
                suffixes=suffixes,
                purpose=kwargs.get("purpose"),
                mock=True,
            )
        )
        return SimpleNamespace(
            selected_ids=(verdict,),
            scores={i: (1.0 if i == verdict else -1.0) for i in ids},
            reasoning_text="AUDIT MOCK reasoning",
            method="audit_mock",
            cache_reused=True,
            latency_ms={},
            metadata={},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "after"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / "reports/taskgraph" / f"locate_select_v2_{args.phase}"
    if args.phase == "before" and (out / "source_sha256.json").exists():
        raise RuntimeError(
            "before audit is immutable; refusing to overwrite its pre-change snapshot"
        )
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        "operators.py",
        "referent_refinement.py",
        "providers.py",
        "input_composer.py",
        "semantic_decision.py",
        "semantic_prompts.py",
        "execution_plan.py",
        "schema.py",
        "choice.py",
    ]
    if args.phase == "after":
        sources.extend(
            [
                "locate_routing.py",
                "semantic_selection.py",
                "executor.py",
                "runtime.py",
                "runtime_types.py",
            ]
        )
    manifest = {}
    for name in sources:
        src = root / "src/sat_rs_vlm/taskgraph" / name
        (out / name).write_bytes(src.read_bytes())
        manifest[str(src.relative_to(root))] = hashlib.sha256(src.read_bytes()).hexdigest()
    engine_source = root / "src/sat_rs_vlm/models/hf_vlm_engine.py"
    (out / "hf_vlm_engine.py").write_bytes(engine_source.read_bytes())
    manifest[str(engine_source.relative_to(root))] = hashlib.sha256(
        engine_source.read_bytes()
    ).hexdigest()
    if args.phase == "after":
        for relative in (
            "scripts/integrations/lae_dino_worker.py",
            "src/sat_rs_vlm/integrations/detectors/lae_dino_sidecar.py",
            "src/sat_rs_vlm/integrations/detectors/tiled.py",
        ):
            source = root / relative
            (out / source.name).write_bytes(source.read_bytes())
            manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "source_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.phase == "after":
        names = {
            "vlm_referent",
            "refine",
            "select_grid_ids",
            "visual_fallback",
            "_semantic_select",
            "select",
            "_prompt",
            "infer",
            "_choice_instruction",
            "reason_and_choose",
            "reason_and_decide",
            "_semantic_fallback_score",
            "_cached_score",
            "_legacy_score",
            "_taskgraph",
            "compose_locate",
            "compose_selection",
            "_build_messages",
            "reason_with_cache",
            "score_choice_from_cache",
        }
        callsites = []
        for relative in manifest:
            source = root / relative
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for statement in tree.body:
                if isinstance(statement, ast.ClassDef):
                    for method in statement.body:
                        if isinstance(method, ast.FunctionDef) and method.name in names:
                            callsites.append(
                                {
                                    "file": relative,
                                    "function": f"{statement.name}.{method.name}",
                                    "start": method.lineno,
                                    "end": method.end_lineno,
                                }
                            )
        (out / "callsites.json").write_text(json.dumps(callsites, indent=2), encoding="utf-8")
    visual = Image.new("RGB", (640, 480), "#63825a")
    draw = ImageDraw.Draw(visual)
    for box in [(40, 40, 100, 100), (180, 160, 250, 220), (420, 300, 480, 370)]:
        draw.rectangle(box, fill="white")
    image_path = out / "audit_scene.png"
    visual.save(image_path)
    image = ImageRef(str(image_path), 640, 480)
    items = EntitySet(
        tuple(
            Entity(
                Region(image, box), "building", 0.9 - i * 0.1, {"candidate_id": f"building_{i + 1}"}
            )
            for i, box in enumerate(
                [(40, 40, 100, 100), (180, 160, 250, 220), (420, 300, 480, 370)]
            )
        )
    )
    composer = InputComposer(out / "visuals", entity_set_max_side=640)
    provider = LazyQwenSemanticProvider(ModelConfig(model_id="AUDIT_MOCK_NO_WEIGHTS"))
    capture = CaptureEngine()
    provider._engine = capture
    refiner = ReferentRefiner(
        provider, composer, ReferentRefinementConfig(enabled=True, singleton_vlm_primary=True)
    )
    target = TargetSpec(category="building", attributes={"color": "white", "shape": "square"})
    examples = []

    def capture_case(name, fn):
        start = len(capture.calls)
        result = fn()
        examples.append(
            {
                "name": name,
                "calls": capture.calls[start:],
                "result": str(result),
                "provenance": "real source prompt construction; synthetic visual; "
                "mocked inference, not accuracy",
            }
        )

    capture_case(
        "LOCATE_SINGLE white square building -> ATTRIBUTE",
        lambda: refiner.vlm_referent(image, question="What color is its roof?", target=target),
    )
    capture_case(
        "LOCATE detector -> singleton refinement",
        lambda: refiner.refine(
            items,
            question="Find the building in the lower right",
            target=target,
            trigger_reason="audit",
        ),
    )
    grids = tuple(
        Region(
            image,
            (x * 200.0, y * 150.0, (x + 1) * 200.0, (y + 1) * 150.0),
            {"grid_id": f"grid_{y * 3 + x}"},
        )
        for y in range(3)
        for x in range(3)
    )
    capture_case(
        "LOCATE empty -> 3x3 fallback",
        lambda: refiner.select_grid_ids(
            grids, question="Find white square building", target=target
        ),
    )
    context = OperatorContext("Original final question must not leak into SELECT", (), composer)
    selector = SelectExecutor(provider)
    for cardinality in ("SINGLE", "MULTI"):
        node = GraphNode(
            id="n2",
            op="SELECT",
            inputs={"candidates": "$n1", "reference": "$n3"},
            params={"mode": "RELATION", "relation": "NEAR", "selection_type": cardinality},
        )
        capture_case(
            f"SELECT_REL NEAR {cardinality}",
            lambda node=node: selector._semantic_select(
                items, Region(image, (280.0, 230.0, 310.0, 260.0)), "NEAR", node, context, {}
            ),
        )
    unresolved = EntitySet(
        items.entities, {"fallback_required": True, "resolution_status": "UNRESOLVED"}
    )
    capture_case(
        "Legacy final candidate fallback (external/old entrypoint)",
        lambda: ChoiceResolver(provider, composer).resolve(
            ChoiceRequest(
                (unresolved,), "What color is the selected building?", ("A white", "B red")
            )
        ),
    )
    if args.phase == "after":
        capture_case(
            "Retained whole-image final fallback wrapper (legacy graph evidence failure)",
            lambda: ChoiceResolver(provider, composer).resolve(
                ChoiceRequest(
                    (image,), "What color is the selected building?", ("A white", "B red")
                )
            ),
        )
        capture.cached_unavailable = True
        try:
            capture_case(
                "SELECT_REL token-constrained fallback when KV scorer is unavailable",
                lambda: selector._semantic_select(
                    items, Region(image, (280.0, 230.0, 310.0, 260.0)), "NEAR", node, context, {}
                ),
            )
            from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig

            capture.text_response = "A"
            capture_case(
                "Legacy final choice free-generation fallback (disabled by default)",
                lambda: ChoiceResolver(
                    SimpleNamespace(infer=provider.infer, provider_name=provider.provider_name),
                    composer,
                    ChoiceSystemConfig(legacy_regex_fallback=True),
                ).resolve(
                    ChoiceRequest(
                        (image,), "What color is the selected building?", ("A white", "B red")
                    )
                ),
            )
        finally:
            capture.cached_unavailable = False
            capture.text_response = "white"
        examples.extend(capture_end_to_end(out, image, composer, provider, capture))
    (out / "rendered_prompts.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template_file = Path("D:/Desktop/tzb-2026/Qwen3-VL-2B-Instruct/chat_template.json")
    if template_file.is_file():
        from jinja2 import Environment

        template = json.loads(template_file.read_text(encoding="utf-8"))["chat_template"]
        (out / "chat_template.json").write_bytes(template_file.read_bytes())
        for example in examples:
            for call in example["calls"]:
                call["chat_template_rendered"] = (
                    Environment()
                    .from_string(template)
                    .render(
                        messages=call["messages"],
                        add_generation_prompt=True,
                        tools=None,
                        add_vision_id=False,
                    )
                )
        (out / "rendered_prompts.json").write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for index, example in enumerate((item for item in examples if "graph" in item), 1):
            (out / f"e2e_{index:02d}.json").write_text(
                json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    lines = [
        f"# LOCATE / SELECT {args.phase} rendered prompt capture",
        "",
        "Actual InputComposer and Qwen wrappers; synthetic visuals and mock inference. "
        "This is not an accuracy experiment.",
        "",
    ]
    for item in examples:
        lines += [f"## {item['name']}", ""]
        for call in item["calls"]:
            lines += [
                "```text",
                call.get("chat_template_rendered", call["prompt"]),
                "```",
                "",
                "终端评分后缀：",
                "```json",
                json.dumps(call.get("suffixes"), ensure_ascii=False, indent=2),
                "```",
                "",
                "终端评分候选 / 生成允许值：",
                "```json",
                json.dumps(
                    {
                        "terminal_candidates": call.get("terminal_candidates"),
                        "allowed_outputs": call.get("allowed_outputs"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        if not item["calls"]:
            lines += ["该构造未触发 VLM 调用。", ""]
    (out / "rendered_prompts.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "examples": len(examples),
                "calls": len(capture.calls),
                "output": str(out),
            }
        )
    )


def capture_end_to_end(out, image, composer, provider, capture):
    """Five complete legal TaskGraphs, executing actual operators with mock model predictions."""
    from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl

    def locate(id, category, attrs=None):
        return {
            "id": id,
            "op": "LOCATE",
            "inputs": {"image": "$image0"},
            "params": {"target": {"category": category, "attributes": attrs or {}}},
        }

    def attribute(id, source):
        return {
            "id": id,
            "op": "ATTRIBUTE",
            "inputs": {"entity": source},
            "params": {"attribute": "color"},
        }

    def select(id, source, **params):
        return {"id": id, "op": "SELECT", "inputs": {"candidates": source}, "params": params}

    def count(id, source):
        return {
            "id": id,
            "op": "COUNT",
            "inputs": {"entities": source},
            "params": {"target": {"category": "car"}, "entire": False},
        }

    query = "the building with trees to its right and grass below"
    ordinary = [(40, 40, 100, 100), (180, 160, 250, 220), (420, 300, 480, 370)]
    cars = [(i * 5, 180, i * 5 + 3, 183) for i in range(10)] + [
        (300 + i * 10, 180, 303 + i * 10, 183) for i in range(10)
    ]
    cases = [
        (
            "01 VLM white square -> ATTRIBUTE",
            [
                locate("n1", "building", {"color": "white", "shape": "square"}),
                attribute("n2", "$n1"),
            ],
            ordinary,
            [],
            "LABEL",
        ),
        (
            "02 DINO 20 buildings -> EXTREME TOPMOST -> ATTRIBUTE",
            [
                locate("n1", "building"),
                select("n2", "$n1", mode="EXTREME", direction="TOPMOST"),
                attribute("n3", "$n2"),
            ],
            cars[:-1] + [(100, 5, 120, 25)],
            [],
            "LABEL",
        ),
        (
            "03 DINO -> SELECT_SEM SINGLE",
            [
                locate("n1", "building"),
                select("n2", "$n1", mode="SEMANTIC", query=query, selection_type="SINGLE"),
            ],
            ordinary,
            ["B"],
            "CHOICE_SINGLE",
        ),
        (
            "04 DINO red cars -> SELECT_SEM MULTI -> COUNT",
            [
                locate("n1", "car", {"color": "red"}),
                select(
                    "n2",
                    "$n1",
                    mode="SEMANTIC",
                    query="cars parked beside the road",
                    selection_type="MULTI",
                ),
                count("n3", "$n2"),
            ],
            ordinary,
            ["MATCH", "NO_MATCH", "MATCH", "MATCH", "NO_MATCH"],
            "INTEGER",
        ),
        (
            "05 VLM building reference + DINO cars -> SELECT_REL -> COUNT",
            [
                locate("n1", "building"),
                locate("n2", "car"),
                {
                    "id": "n3",
                    "op": "SELECT",
                    "inputs": {"candidates": "$n2", "reference": "$n1"},
                    "params": {"mode": "RELATION", "relation": "LEFT_OF"},
                },
                count("n4", "$n3"),
            ],
            cars,
            [],
            "INTEGER",
        ),
    ]
    examples = []
    for index, (name, nodes, boxes, decisions, answer_type) in enumerate(cases, 1):
        final_ref = "$" + nodes[-1]["id"]
        question = (
            "What color is the selected target?"
            if answer_type == "LABEL"
            else "Where is the selected building?"
            if answer_type == "CHOICE_SINGLE"
            else ""
        )
        raw = {
            "intent": "OTHER",
            "nodes": nodes,
            "final": {"sources": [final_ref], "question": question, "answer_type": answer_type},
        }
        compile_raw = {
            **raw,
            "final": {
                key: value for key, value in raw["final"].items() if key != "question" or value
            },
        }
        dsl = compile_taskgraph_to_dsl(compile_raw)
        parse_taskgraph_dsl(dsl)  # Validate the rendered example independently.
        graph = TaskGraph(
            question="Full final question is intentionally not used by LOCATE",
            question_type="FREE_FORM",
            inputs={"image0": {"uri_or_key": image.uri_or_key}},
            **raw,
        )
        original = graph.model_dump(mode="json")
        detector = FakeDetectionProvider(boxes)
        locator = LocateExecutor(
            detector,
            None,
            refiner=ReferentRefiner(
                provider, composer, ReferentRefinementConfig(attribute_policy="vlm")
            ),
        )
        semantic = SemanticExecutor(provider, model_role="semantic_2b")
        bindings = {
            OperatorName.LOCATE: ExecutorBinding(locator),
            OperatorName.SELECT: ExecutorBinding(SelectExecutor(provider)),
            OperatorName.ATTRIBUTE: ExecutorBinding(semantic),
            OperatorName.COUNT: ExecutorBinding(CountExecutor(FakeCountingProvider())),
        }
        store = RuntimeStore({"image0": image})
        capture.finite_verdicts = list(decisions)
        start = len(capture.calls)
        trace = GraphExecutor(
            CapabilityRouter(bindings), FinalChoiceFusionConfig(mode="off")
        ).execute(
            graph,
            store,
            sample_id=f"audit-{index}",
            execution_mode="TASKGRAPH_UHR",
            context=OperatorContext("Full final question", (), composer),
        )
        assert graph.model_dump(mode="json") == original
        result = runtime_summary(store.get(final_ref))
        record = {
            "name": name,
            "graph": original,
            "dsl": dsl,
            "trace": trace.to_dict(),
            "result": result,
            "calls": capture.calls[start:],
            "provenance": "actual GraphExecutor/InputComposer/Qwen wrapper; "
            "mocked detector/model predictions; no accuracy claim",
        }
        examples.append(record)
        (out / f"e2e_{index:02d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return examples


if __name__ == "__main__":
    main()

"""Independent text-only XLRS review artifact builder; does not edit any dataset.

Each revision below was individually authored against the English question and
existing DSL. No reference answers or image pixels are read. Runtime/model
grounding accuracy is explicitly outside this static semantic audit.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import validate_candidate

SOURCE = ROOT / "taskgraph_lab/data/planner_choice_expanded_v1/train.jsonl"
BASELINE = ROOT / "taskgraph_lab/data/planner_sft_hard_augmented_v1_20260901_173936/train.jsonl"
OUTPUT = Path(__file__).with_name("xlrs_semantic_review.json")
ROWS = {r["id"]: r for r in map(json.loads, SOURCE.read_text(encoding="utf-8").splitlines())}
OLD_IDS = {r["id"] for r in map(json.loads, BASELINE.read_text(encoding="utf-8").splitlines())}
FINDINGS: dict[str, dict] = {}


def q(value):
    return json.dumps(value, ensure_ascii=False)


def renumber(dsl):
    # Temporary r* IDs are an editing convenience, never part of the output DSL.
    ids = re.findall(r"(?m)^([nr]\d+)=", dsl)
    mapping = {old: f"n{i + 1}" for i, old in enumerate(ids)}
    dsl = re.sub(r"\$([nr]\d+)\b", lambda m: "$" + mapping[m[1]], dsl)
    return re.sub(r"(?m)^([nr]\d+)=", lambda m: mapping[m[1]] + "=", dsl)


def replace(dsl, before, after):
    assert before in dsl, before
    return dsl.replace(before, after)


def set_node(dsl, node_id, expression):
    dsl, n = re.subn(rf"(?m)^{node_id}=.*$", lambda _: f"{node_id}={expression}", dsl)
    assert n == 1, (node_id, dsl)
    return dsl


def sem(candidate, query, selection="SINGLE", reference="null", scope="$image0"):
    return f"SELECT_SEM({candidate},{reference},{q(query)},{selection},{scope})"


def filter_after(dsl, node_id, query, selection="SINGLE", reference="null", scope="$image0"):
    lines = dsl.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.startswith(node_id + "="))
    temp = "r" + str(1 + sum(line.startswith("r") for line in lines))
    for i in range(idx + 1, len(lines)):
        lines[i] = re.sub(rf"\${node_id}\b", "$" + temp, lines[i])
    lines.insert(idx + 1, f"{temp}=" + sem("$" + node_id, query, selection, reference, scope))
    return "\n".join(lines)


def old_dsl(number):
    return ROWS["xlrs_" + number]["messages"][2]["content"]


def record(number, reason, corrected=None, codes=(), severity="high"):
    sid = "xlrs_" + number
    row = ROWS[sid]
    user = json.loads(row["messages"][1]["content"])
    original = row["messages"][2]["content"]
    entry = {
        "id": sid,
        "sample_id": sid,
        "origin": "baseline" if sid in OLD_IDS else "new",
        "severity": "error" if corrected is not None else "review",
        "risk_level": severity,
        "issue_codes": list(codes),
        "question": user["question"],
        "original_dsl": original,
        "old_dsl": original,
        "reason": reason,
        "evidence": reason,
        "disposition": "correct" if corrected is not None else "quarantine_review",
        "corrected_dsl": None,
        "validation": {},
    }
    if corrected is not None:
        corrected = renumber(corrected)
        payload = parse_taskgraph_dsl_payload(corrected)
        # Canonicalize aliases which are already production-equivalent.
        for node in payload["nodes"]:
            if node["op"] == "SELECT" and node["params"].get("mode") == "RANK":
                if node["params"]["criterion"] in {"size", "area", "bboxarea"}:
                    node["params"]["criterion"] = "bbox_area"
        target, report = validate_candidate(
            payload, inputs=user["inputs"], question=user["question"],
            question_type=user["question_type"],
        )
        assert report.valid and target is not None, (sid, report.errors, corrected)
        corrected = compile_taskgraph_to_dsl(target)
        assert canonicalize_target(parse_taskgraph_dsl(corrected)) == canonicalize_target(target)
        grammar = CanonicalDSLPrefixGrammar(image_refs=["$" + k for k in user["inputs"]]).analyze(corrected)
        assert grammar.valid_prefix and grammar.complete, (sid, grammar)
        req = PlannerRequest(
            question=user["question"], question_type=user["question_type"],
            choices=tuple(user["choices"]),
            inputs={"$" + k: ImageRef(v["uri_or_key"]) for k, v in user["inputs"].items()},
            sample_id=sid,
        )
        Qwen3VLPlannerProvider._to_production_graph(corrected, req)
        original_payload = parse_taskgraph_dsl_payload(original)
        # No old authoritative COUNT or route context may silently disappear.
        for op in ("COUNT", "BUILD_ROUTE_CONTEXT"):
            assert sum(n["op"] == op for n in payload["nodes"]) >= sum(
                n["op"] == op for n in original_payload["nodes"]
            ), (sid, op)
        assert original_payload["final"]["answer_type"] == payload["final"]["answer_type"]
        entry["corrected_dsl"] = corrected
        entry["validation"] = {
            "lab_original_question_inputs_type": True,
            "canonical_roundtrip": True,
            "constraint_complete": True,
            "production_bridge": True,
            "authoritative_count_preserved": True,
            "route_context_preserved": True,
            "answer_format_preserved": True,
            "real_model_or_image_accuracy": "not_tested",
        }
    assert sid not in FINDINGS, sid
    FINDINGS[sid] = entry


def filtered(number, node, query, reason, selection="SINGLE", reference="null", scope="$image0", edits=()):
    dsl = old_dsl(number)
    for before, after in edits:
        dsl = replace(dsl, before, after)
    record(number, reason, filter_after(dsl, node, query, selection, reference, scope),
           ("missing_candidate_condition",))


def build():
    # Intrinsic properties remain TargetSpec attributes. Comparatives and
    # external context become actual selection over the existing candidates.
    d = replace(old_dsl("000226"), ',size="largest"', '')
    d = filter_after(d, "n2", "the largest gray H-shaped building", scope="$image0")
    # Largest ordinary object area has a dedicated deterministic implementation.
    d = replace(d, sem("$n2", "the largest gray H-shaped building"),
                'SELECT_RANK($n2,null,"bbox_area",1,DESCENDING)')
    record("000226", "largest 被写进 LOCATE.size；没有真正枚举后取最大值。将其移为候选集合上的面积排序。", d,
           ("comparative_hidden_in_locate",))
    d = replace(replace(old_dsl("000231"), ',size="longest"', ''), ',size="largest"', '')
    d = filter_after(d, "n2", "the longest white rectangular building in the bottom left of the town shown in the picture", scope="$n1")
    d = filter_after(d, "n3", "the largest irregular gray building in the bottom left of the town shown in the picture", scope="$n1")
    record("000231", "longest/largest 均藏在 LOCATE.size，且原裁剪只有图像左下而没有城镇条件。保留裁剪并由候选语义筛选消费完整条件。", d,
           ("comparative_hidden_in_locate", "missing_candidate_condition"))

    d = set_node(old_dsl("000280"), "n2", sem("$n1", "the forest areas on the left side of the image", "MULTI"))
    record("000280", "LEFT_SIDE SUBREGION 返回森林框左侧的图像区域，原 RANK 排的是该裁剪框，不是图像左侧森林候选。先筛左侧森林，再保留面积 RANK。", d,
           ("subregion_is_not_candidate_filter",))
    filtered("000304", "n2", "the gray sports field with a track on the far left",
             "完整含方位的名词短语作为 LOCATE.category；far left 条件没有显式候选筛选。",
             edits=(('T("gray sports field with a track on the far left")', 'T("sports field",color="gray",has_part="track")'),))
    filtered("000308", "n2", "the red rectangular court in the gray building at the bottom right of the picture",
             "原图像右下裁剪代替了 in the gray building，遗漏灰色建筑内部条件。", scope="$n1")
    d = old_dsl("000314")
    d = re.sub(r'(?m)^n2=.*\n', '', d)
    d = replace(d, 'LOCATE($n2,', 'LOCATE($n1,')
    record("000314", "LEFT_SIDE(TOP_LEFT Region) 表示该区域左侧的外部窄带，并非该区域内部左半。删除错误外部裁剪，保留 TOP_LEFT、BELOW river 和 LEFTMOST 候选极值。", d,
           ("inside_crop_changed_to_external_subregion",))
    # Keep ambiguous attachment wording verbatim in candidate selection instead
    # of inventing a RIGHT_OF predicate from 'on the left side of the area'.
    d = set_node(old_dsl("000327"), "n3", sem("$n1", "the light blue building with a parking lot on the left side of the area"))
    d = re.sub(r'(?m)^n2=.*\n', '', d)
    record("000327", "on the left side of the area 被擅自解释为建筑在停车场 RIGHT_OF；题目没有这个明确几何断言。用完整参考物描述筛选，不猜附件关系。", d,
           ("invented_relation",))
    filtered("000335", "n3", "the red rectangular pond of the quarry",
             "of the quarry 条件在原 DSL 完全缺失。")
    filtered("000341", "n2", "the two red and green playgrounds towards the lower right side", "red and green 被改为 red or green，且 lower right 条件遗漏。",
             selection="MULTI", edits=(('color="red or green"', 'color="red and green"'),))
    filtered("000361", "n3", "the green basketball court in the residential area on the left",
             "原参考物只限制 LEFT，没有 residential area；主体屋顶颜色也被误写成整栋建筑颜色。", scope="$n1",
             edits=(('T("building",color="emerald green")', 'T("building",has_part="emerald green roof")'),))
    d = old_dsl("000374")
    d = replace(d, 'n2=LOCATE', 'r1=GROUP($n1,ROW)\nr2=' + sem("$r1", "the green row of buildings") + '\nn2=LOCATE')
    d = replace(d, 'RELATION($n1,$n2)', 'RELATION($r2,$n2)')
    record("000374", "the green row 被当作全图所有绿色建筑；新增 ROW 分组并在已有行候选里选指定行。", d,
           ("grouping_condition_omitted",))
    for number, query in {
        "000413": "the longer irregularly shaped lake surrounded by greenery",
        "000435": "the smaller green house",
        "000452": "the longer silver greenhouse",
    }.items():
        comparative = {"000413": "longer", "000435": "smaller", "000452": "longer"}[number]
        filtered(number, "n1", query, f"比较级 {comparative} 被当作 LOCATE 固有 size；没有在候选间进行比较。000413 同时遗漏 surrounded by greenery。",
                 edits=((f',size="{comparative}"', ''),))
    d = old_dsl("000426")
    d = replace(d, 'T("building",shape="cross-shaped")', 'T("building",has_part="cross-shaped roof")')
    d = filter_after(d, "n4", "the building with a cross-shaped rooftop in the residential area")
    d = filter_after(d, "n3", "the larger strip-shaped lake below the referenced river", reference="$n1")
    record("000426", "cross-shaped rooftop 被当作建筑轮廓，residential area 缺失；larger 湖只有 large 属性。保留 BELOW 并补全候选条件。", d,
           ("part_attribute_role", "missing_candidate_condition"))
    filtered("000455", "n2", "the red boat moored on the riverbank at the far right of the picture",
             "moored on the riverbank 被遗漏，只留下右侧红船。", scope="$n1")
    filtered("000463", "n2", "the red and green tennis court on the green grass",
             "on the green grass 条件完全缺失。")
    d = old_dsl("000464")
    d = replace(d, 'n4=SELECT_RANK', 'r1=SELECT_REL($n3,$n2,NEAR)\nn4=SELECT_RANK')
    d = replace(d, 'SELECT_RANK($n3,', 'SELECT_RANK($r1,')
    record("000464", "湖应先满足 near the selected right-side roundabout road，再取最大；原 DSL 在全图直接取最大湖。", d,
           ("rank_domain_condition_missing",))
    filtered("000475", "n2", "the three adjacent buildings with emerald green roofs", "三个相邻目标被写入类别，屋顶颜色误成建筑颜色；改为建筑候选后 MULTI 消费相邻和数量描述。",
             selection="MULTI", edits=(('T("three adjacent buildings",color="emerald green")', 'T("building",has_part="emerald green roof")'),))
    d = old_dsl("000476")
    d = filter_after(d, "n1", "the white bus below the intersection in the upper right corner")
    d = filter_after(d, "n2", "the deep blue statue below the intersection in the upper right corner")
    record("000476", "共享 below the intersection in the upper right corner 被完全丢失，双方变成全图目标。", d,
           ("shared_scope_omitted",))
    d = old_dsl("000479")
    d = filter_after(d, "n1", "the roundabout on the road in the lower left area of the image")
    d = filter_after(d, "n2", "the large light green storage tank in the lower left area of the image")
    record("000479", "共享 lower left 场景在主体和参考物均遗漏。", d, ("shared_scope_omitted",))
    d = set_node(old_dsl("000485"), "n5", sem("$n4", "the largest parking lot with the most parking spaces"))
    record("000485", "停车场 bbox_area 最大不等于停车位最多；原 RANK 没有消费 most parking spaces。", d,
           ("count_comparative_replaced_by_area",))
    d = set_node(old_dsl("000533"), "n5", sem("$n4", "the red car under the shadow of the trees"))
    d = re.sub(r'(?m)^n3=.*\n', '', d)
    d = filter_after(d, "n7", "the deep blue car on the road to the right of the referenced red-roofed building at the center", reference="$n2")
    record("000533", "under the shadow 是遮阴条件，原 BELOW(car, trees) 是图像方位，二者不同；深蓝车 on the road 也遗漏。保留建筑右侧几何筛选。", d,
           ("semantic_condition_replaced_by_geometry", "missing_candidate_condition"))
    d = set_node(old_dsl("000535"), "n5", sem("$n4", "the orange buildings on the far left", "MULTI"))
    record("000535", "LEFTMOST 先把候选压成单体，再 RANK(size) 无法表达 largest orange building on the far left；先筛左侧候选，再保留确定性 RANK。", d,
           ("rank_after_singleton_drops_comparative",))
    filtered("000567", "n2", "the gray square area surrounded by flat land in the upper right of the image",
             "surrounded by flat land 是周边场景条件，原 LOCATE.state 无候选级上下文验证。", scope="$n1",
             edits=((',state="surrounded by flat land"', ''),))
    d = replace(old_dsl("000577"), 'T("building",color="white",shape="rectangular")', 'T("building",has_part="white rectangular roof")')
    d = filter_after(d, "n2", "the building with a white rectangular rooftop above the referenced two green cylindrical buildings", reference="$n1")
    record("000577", "above them 未消费，且 white rectangular rooftop 被写为建筑颜色/轮廓。", d,
           ("missing_reference_relation", "part_attribute_role"))
    d = replace(old_dsl("000588"), 'color="red or green"', 'color="red and green"')
    d = filter_after(d, "n2", "the two red and green playgrounds towards the bottom left", "MULTI", scope="$n1")
    record("000588", "red and green 被改为 OR，集合可能扩入仅单色目标。用完整两座双颜色目标描述筛选，保留左下区域。", d,
           ("and_or_condition_changed",))
    d = replace(old_dsl("000591"), 'T("building",color="brown",has_part="roof")', 'T("building",has_part="brown roof")')
    d = set_node(d, "n3", sem("$n2", "the building with the largest brown roof in the bottom left corner", scope="$n1"))
    record("000591", "最大 brown roof 被替换成 brown building 的 bbox_area 排序；屋顶面积不是整栋建筑外接框面积。", d,
           ("roof_area_replaced_by_building_bbox", "part_attribute_role"))
    filtered("000605", "n6", "the parking lot that holds many red vehicles in the referenced largest residential area",
             "many red vehicles 被当作 parking lot.has_part，缺少场景内候选筛选；保留 INSIDE 和住宅区面积排序。", reference="$n2",
             edits=(('T("parking lot",has_part="many red vehicles")', 'T("parking lot")'),))
    filtered("000636", "n5", "the trapezoidal light green lake surrounded by teal farmland", "surrounded by teal farmland 被完全遗漏。")
    filtered("000638", "n5", "the six adjacent buildings with orange sloped roofs in the upper left corner", "six adjacent buildings 被当成类别；应对建筑候选保留六座相邻集合。",
             selection="MULTI", scope="$n1", edits=(('T("six adjacent buildings",has_part="orange sloped roof")', 'T("building",has_part="orange sloped roof")'),))
    filtered("000679", "n3", "the two longer gray houses on the upper right side", "longer 比较级藏在 LOCATE.size，且 two 未在候选筛选中表达。",
             selection="MULTI", scope="$n2", edits=((',size="longer"', ''),))
    # A real parking lot is located, instead of treating everything below the
    # building as that parking lot. Both operands share the selected lot.
    d = '''INTENT(OBJECT_RELATION)
n1=REGION($image0,TOP_RIGHT)
n2=LOCATE($n1,T("building",color="white",has_part="flat roof"))
n3=LOCATE($image0,T("parking lot"))
n4=SELECT_REL($n3,$n2,BELOW)
n5=SELECT_SEM($n4,$n2,"the parking lot beneath the referenced white flat-roofed building in the upper right",SINGLE,$image0)
n6=LOCATE($image0,T("truck",color="blue"))
n7=SELECT_REL($n6,$n5,INSIDE)
n8=LOCATE($image0,T("vehicle",color="black"))
n9=SELECT_REL($n8,$n5,INSIDE)
n10=RELATION($n7,$n9)
FINAL($n10,CHOICE_SINGLE)'''
    record("000691", "原 truck 只在建筑下方图像区域找、black vehicle 则全图找；题目要求双方在同一停车场。增加真实停车场候选并共享 INSIDE 筛选。", d,
           ("scope_reference_dropped", "region_substituted_for_object"))

    # Counting remains an explicit authoritative node after every filter.
    for number, query in {
        "000777": "the boats docked at the referenced shore in the bottom left corner of the picture",
        "000814": "the red boats docked at the referenced shore in the bottom left of the picture",
    }.items():
        filtered(number, "n4", query, "原 NEAR/NEXT_TO shore 包含航行路过的船，未验证 docked；保留邻近筛选后用 MULTI 明确靠岸状态，再 COUNT_ENTITIES。",
                 selection="MULTI", reference="$n2", scope="$n1")
    for number, target, query in (
        ("000791", 'T("soccer field")', "the soccer fields inside the bottom right corner of the referenced larger cluster of houses in the middle of the image"),
        ("000851", 'T("tennis court",color="red")', "the red tennis courts inside the upper right area of the referenced large housing cluster towards the top right of the image"),
    ):
        d = old_dsl(number)
        d = set_node(d, "n5", f"LOCATE($image0,{target})")
        d = set_node(d, "n6", sem("$n5", query, "MULTI", "$n4"))
        d = set_node(d, "n7", f"COUNT_ENTITIES($n6,{target},false)")
        record(number, "题目描述住宅区内部角落；原连续 SUBREGION 返回参照框外侧，计数域完全错误。改为候选 MULTI 筛内部角落，并保留权威 COUNT。", d,
               ("inside_corner_changed_to_outside_subregion",))
    d = set_node(old_dsl("000834"), "n4", sem("$n3", "the buildings with gray roofs on both sides of the referenced highway in the bottom left corner of the picture", "MULTI", "$n2"))
    record("000834", "BOTH_SIDES SUBREGION 在生产实现不支持且返回 UNRESOLVED；输入也从建筑集合变成裁剪区域。用 MULTI 筛两侧建筑候选后保持 COUNT_ENTITIES。", d,
           ("unsupported_subregion", "candidate_set_changed_to_region"))
    d = set_node(old_dsl("000840"), "n4", sem("$n3", "the lakes or ponds near the upper right corner of the referenced large forest in the upper right corner of the picture", "MULTI", "$n2", "$n1"))
    record("000840", "原 NEAR forest 丢失 near the upper right corner of forest；将完整角落参照条件用于 MULTI，再计数。", d,
           ("reference_corner_condition_omitted",))

    # Route endpoints and the route-context operation are always retained.
    d = set_node(old_dsl("002010"), "n6", sem("$n5", "the lake nearest to the top-left corner of the referenced longest blue rectangular building", reference="$n2"))
    record("002010", "NEAR 只测试邻近，不能表达 nearest，更遗漏 building 的 top-left corner 距离参照。", d,
           ("nearest_replaced_by_near", "reference_corner_condition_omitted"))
    filtered("002038", "n2", "the roundabout between the two clusters of houses at the top center of the picture", "between the two clusters 被遗漏，原只剩 TOP_CENTER 内全部环岛。", scope="$n1")
    d = set_node(old_dsl("002033"), "n6", sem("$n5", "the parking lots inside the bottom left corner of the referenced large cluster of houses in the upper right part of the image", "MULTI", "$n4"))
    record("002033", "in the bottom left corner of the housing cluster 被改成 LOWER_LEFT_OF cluster，后者表示集群外侧方位。保留小停车场面积 ASCENDING 排序和 route。", d,
           ("inside_corner_changed_to_external_relation",))
    d = set_node(old_dsl("002036"), "n3", sem("$n1", "the large blue ship next to the central sea area"))
    d = re.sub(r'(?m)^n2=.*\n', '', d)
    record("002036", "next to central sea area 被改成 NEAR(image CENTER)，忽略 sea area 实物。以完整候选描述恢复该条件，路线节点保留。", d,
           ("reference_object_substituted_by_image_region",))
    d = '''INTENT(ROUTE_PLANNING)
n1=REGION($image0,TOP_LEFT)
n2=LOCATE($n1,T("area",shape="rectangular",size="large",pattern="dark spots"))
n3=REGION($image0,BOTTOM_RIGHT)
n4=LOCATE($n3,T("building"))
n5=GROUP($n4,CLUSTER)
n6=SELECT_SEM($n5,null,"the small cluster of buildings in the bottom right corner",SINGLE,$n3)
n7=BUILD_ROUTE_CONTEXT($image0,$n2,$n6)
FINAL_QUESTION($n7,CHOICE_SINGLE,"Which option describes the best route between the selected start and goal?")'''
    record("002039", "起点直接用 TOP_LEFT 整区，未定位 large rectangular area with dark spots；终点把 small cluster 变成 small individual buildings。补定位及 CLUSTER 分组，保留 route。", d,
           ("region_substituted_for_object", "group_attribute_applied_to_members"))
    d = replace(old_dsl("002041"), ',pattern="cluster"', '')
    d = replace(d, 'n3=REGION', 'r1=GROUP($n2,CLUSTER)\nr2=' + sem("$r1", "the cluster of tall red buildings near the top of the image", scope="$n1") + '\nn3=REGION')
    d = replace(d, 'BUILD_ROUTE_CONTEXT($image0,$n2,$n4)', 'BUILD_ROUTE_CONTEXT($image0,$r2,$n4)')
    record("002041", "cluster 被藏在单体 building.pattern，未生成集群候选。恢复 GROUP(CLUSTER) 后筛指定集群。", d,
           ("grouping_condition_omitted",))
    d = set_node(old_dsl("002053"), "n4", sem("$n3", "the playground with the referenced red track at the top of the picture", reference="$n2"))
    record("002053", "playground with a red track 是场地及配套跑道条件；原直接假定 playground INSIDE track。以完整 with 条件在既有操场候选内筛选，不擅自改成另一个包含方向，保留 ROW/nearest/route。", d,
           ("part_relation_replaced_by_geometric_containment",))
    d = replace(old_dsl("002055"), 'T("building",color="red",has_part="pyramid roof")', 'T("building",has_part="red pyramid roof")')
    record("002055", "red 修饰 pyramid roof，原当作整栋建筑颜色，可能过滤掉仅屋顶红的正确建筑。", d,
           ("part_attribute_role",))
    d = replace(old_dsl("002059"), 'T("building",pattern="checkerboard")', 'T("building",has_part="checkerboard-patterned roof")')
    d = filter_after(d, "n2", "the large green area with multiple buildings on the left side of the picture", scope="$n1")
    record("002059", "multiple buildings 场景条件遗漏；checkerboard-patterned roof 也被错误挂在建筑全体 pattern。", d,
           ("missing_candidate_condition", "part_attribute_role"))
    d = replace(old_dsl("002062"), 'T("building",color="red",has_part="roof")', 'T("building",has_part="red roof")')
    record("002062", "red roof 被误成 red building；将颜色准确绑定到屋顶，保留 near-center 及 route。", d,
           ("part_attribute_role",))
    d = replace(old_dsl("002064"), 'T("residential area",color="green")', 'T("residential area",has_part="green roofs")')
    d = replace(d, 'T("industrial area",color="orange",size="large")', 'T("industrial area",has_part="orange roofs",size="large")')
    record("002064", "green/orange roofs 被误作整片区域的颜色，角色错误。", d,
           ("part_attribute_role",))
    d = set_node(old_dsl("002066"), "n7", sem("$n6", "the longest dark gray house in the middle of the referenced larger cluster of houses in the lower left part of the image", reference="$n4"))
    record("002066", "longest 被错用 size/bbox_area 最大替代；INSIDE 也没区分 middle。改用完整候选描述，保留住宅群面积排序和路线。", d,
           ("length_replaced_by_bbox_area", "missing_candidate_condition"))
    d = replace(old_dsl("002069"), 'T("industrial area",color="blue")', 'T("industrial area",has_part="blue houses")')
    record("002069", "industrial area with blue houses 被改成 industrial area 本身蓝色，成员属性角色错误。", d,
           ("part_attribute_role",))
    filtered("002075", "n8", "the red house below the referenced blue swimming pool above the smaller cluster of houses in the bottom right corner", "below 指向所选泳池，原仅用全图 BOTTOM 区域，未消费此相对关系。", reference="$n6")

    # New rows: roof comparatives cannot be silently evaluated on building box
    # area. Preserve existing visual domains and all other dedicated selectors.
    for number, query, edits in (
        ("000351", "the building with the largest red roof located above the highway on the left", ()),
        ("000537", "the building with the largest white rectangular roof, featuring numerous vents", ()),
        ("000365", "the building with the largest brown roof in the center of the image", (('T("building",color="brown")', 'T("building",has_part="brown roof")'),)),
        ("000630", "the building with the largest gray rectangular roof in the center-right area of the picture", (('T("building",color="gray",shape="rectangular")', 'T("building",has_part="gray rectangular roof")'),)),
    ):
        d = old_dsl(number)
        for before, after in edits:
            d = replace(d, before, after)
        d = set_node(d, "n3", sem("$n2", query))
        record(number, "largest 修饰 roof，但原 RANK 对 building bbox_area 排序；两者无面积等价保证。改为明确屋顶面积语义比较，其他 RANK 保持不变。", d,
               ("roof_area_replaced_by_building_bbox",))
    for number, node, shape in (
        ("000332", "n1", "irregular elongated"),
        ("000249", "n4", "circular"),
        ("000409", "n1", "circular"),
        ("000610", "n1", "irregular"),
        ("000689", "n1", "circular"),
    ):
        d = old_dsl(number)
        line = next(x for x in d.splitlines() if x.startswith(node + "="))
        assert line.endswith('))')
        new_line = line[:-2] + ',shape=' + q(shape) + '))'
        d = replace(d, line, new_line)
        if number == "000332":
            # Three roofs describe the target group cardinality, not a part
            # attached to each individual building; orange and rectangular
            # modify roofs, not whole building footprints.
            d = replace(d, 'T("building",color="orange",has_part="three orange rectangular roof",shape="rectangular")', 'T("building",has_part="orange rectangular roof")')
        record(number, f"原问题明确 {shape} 形状，但相关 LOCATE 无此属性且没有下游 query 补回；会把其他形状混入。000332 同时修复每栋楼带 three roofs 的角色错误。", d,
               ("intrinsic_shape_omitted",))
    for number, node, query in (
        ("000269", "n3", "the white truck in the parking lot near the ocean at the bottom left of the picture"),
        ("000645", "n3", "the red circular area in the upper middle part of the picture"),
        ("000316", "n3", "the red rectangular building in the lower left area of the image"),
        ("000264", "n4", "the deep green rectangular building in the cluster of services on the left side of the image"),
        ("000381", "n3", "the red rectangular building at the top left corner of the image"),
        ("000470", "n3", "the black rectangular building in the bottom left corner of the image"),
        ("000272", "n3", "the deep blue building in the city shown in the bottom right of the image"),
    ):
        filtered(number, node, query, "句首共享场景限定了整次空间关系查询；原只附给 subject，而 reference 仍直接用全图 LOCATE。补参考物候选内筛选。")
    d = set_node(old_dsl("000472"), "n4", sem("$n3", "the light blue L-shaped building below the yellow buildings in the top left of the picture"))
    record("000472", "句首 Below the yellow buildings... 仅进入 subject，reference 的 SEM query 没有共享限定。", d,
           ("shared_scope_omitted",))

    # Cases where repairing would require guessing a damaged phrase or an
    # attachment/cardinality not resolved by the source question are isolated.
    for number, reason, code in (
        ("000855", "above the top right corner of the image 字面指图外，原把它编译为相对 TOP_RIGHT 区域 ABOVE。无法确定作者是指图像右上内部还是区域上方。", "ambiguous_source_geometry"),
        ("002078", "nearest lake 被 TOPMOST 替代可确定错误，但 to the upper right 的参照是起始球场还是图像角落不明确；需人工确认后重写，不能擅定。", "nearest_replaced_by_extreme"),
        ("002031", "below the center-right side of cluster 表示集群局部位置，原拆 BELOW(cluster)+RIGHT_OF(cluster)；句尾 it 的参照也可能是 building 或 cluster。需确认原题指代。", "ambiguous_reference_and_inside_corner"),
        ("002070", "题目只说 cluster on the right，原额外取 largest cluster；larger red building 的比较域也可能是相对 start building。两处均不应擅定。", "invented_extreme_and_comparative_domain"),
        ("000780", "原 red and orange roofs 被转 OR，丢 near right edge；and 可表示双颜色屋顶或红/橙两类屋顶合计，计数真值域需要原题/标注确认。", "ambiguous_and_or_count_domain"),
        ("000375", "源题的方位/目标比较短语存在歧义；本轮未用图像或标准答案确认，不根据现有 DSL 反向臆造问题意图。", "ambiguous_source_question"),
        ("000456", "源题比较描述/指代不够明确，保留人工复核而不自动选定比较集合。", "ambiguous_source_question"),
        ("000202", "源题的相对关系短语存在多种附件解释，当前仅凭文本不足以确定唯一修正。", "ambiguous_source_question"),
    ):
        record(number, reason, None, (code,))

    xlrs = {sid: row for sid, row in ROWS.items() if sid.startswith("xlrs_")}
    # Each old spatial, route, and counting question/DSL was read individually
    # in this audit. The original 449 non-bbox new rows had also been individually
    # reviewed during teacher conversion; they are listed separately, not claimed
    # as a fresh exhaustive review of all 835 new XLRS rows.
    current_manual = set(FINDINGS)
    for sid, row in xlrs.items():
        if sid not in OLD_IDS:
            continue
        user = json.loads(row["messages"][1]["content"])
        dsl = row["messages"][2]["content"]
        category = str(row.get("metadata", {}).get("source_category") or row.get("metadata", {}).get("category"))
        if "Object spatial relationship" in category or "INTENT(ROUTE_PLANNING)" in dsl or re.search(r'\bCOUNT(?:_ENTITIES|_IMAGE)?\(', dsl):
            current_manual.add(sid)
    previous_449 = {json.loads(x)["sample_id"] for x in (ROOT / "taskgraph_lab/data/choice_expansion_codex_v1/plans/xlrs.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(previous_449) == 449
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    findings = [FINDINGS[k] for k in sorted(FINDINGS)]
    report = {
        "version": "xlrs_semantic_review_v1",
        "source_path": str(SOURCE),
        "source_sha256": source_hash,
        "method": "English question + existing DSL only; manually authored evidence and repairs. No reference answer, no image/model inference, no source dataset mutations.",
        "summary": {
            "all_xlrs_automated_scan": len(xlrs),
            "baseline_xlrs": sum(sid in OLD_IDS for sid in xlrs),
            "new_xlrs": sum(sid not in OLD_IDS for sid in xlrs),
            "current_manual_covered": len(current_manual),
            "previous_conversion_manually_reviewed": len(previous_449),
            "covered_union": len(current_manual | previous_449),
            "findings": len(findings),
            "corrected": sum(x["disposition"] == "correct" for x in findings),
            "quarantine_review": sum(x["disposition"] == "quarantine_review" for x in findings),
            "by_origin": dict(Counter(x["origin"] for x in findings)),
            "real_model_accuracy_tested": False,
        },
        "covered_ids": sorted(current_manual | previous_449),
        "current_manual_covered_ids": sorted(current_manual),
        "previous_conversion_reviewed_ids": sorted(previous_449),
        "automated_scan_ids": sorted(xlrs),
        "automated_scan_rules": [
            "nearest versus NEAR/TOPMOST without explicit semantic nearest selection",
            "roof area comparatives ranked on building bbox_area",
            "comparative size hidden in LOCATE attributes",
            "how many must preserve COUNT / COUNT_ENTITIES / COUNT_IMAGE (floor attributes excluded)",
            "red-circle and box marker shape mismatches",
            "ordinal selector direction (no SELECT_ORD found in XLRS rows)",
            "external conditions hidden in LOCATE category/state",
        ],
        "findings": findings,
        "accepted_special_cases": [
            {"sample_id": "xlrs_000370", "reason": "RANK operates on actual roof candidates, not building candidates. No definite roof-vs-building role error; roof bbox area remains an acknowledged geometric proxy."},
            {"sample_id": "xlrs_000430", "reason": "traveling left/upward is intrinsic motion state, not an accidental external spatial selector; not flagged by keyword alone."},
            {"sample_id": "xlrs_000486", "reason": "Source and DSL contain valid U+53E3 (口), confirmed with ASCII Unicode escaping. Console mojibake is not source corruption."},
            {"sample_id": "xlrs_000568", "reason": "Source and DSL contain valid U+56DE (回); not quarantined as corruption."},
            {"sample_id": "xlrs_000728", "reason": "Source and authoritative COUNT contain valid U+56DE (回); not quarantined as corruption."},
            {"sample_id": "xlrs_002076", "reason": "on the right may be the image-area qualifier already represented by TOP_RIGHT. Not automatically reinterpreted as RIGHT_OF the white building."},
        ],
        "limitations": [
            "Automated scan of 2277 rows is not an assertion that all semantic plans are correct.",
            "covered_ids combines this turn's explicit old spatial/route/count review plus the previously individually reviewed 449 new non-bbox rows; phases are separately listed.",
            "Remaining new bbox/property/complex reasoning rows were not exhaustively manually reviewed by this shard.",
            "Candidate recall, VLM contextual selection, GROUP quality, CLIP calibration and actual relation/area accuracy require images and real model experiments.",
            "SINGLE/MULTI semantic decisions can return NONE/AMBIGUOUS/UNRESOLVED; static correctness does not imply visual selection success.",
            "Source and baseline datasets, model weights, production operator logic and shared cleaning script are untouched.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **report["summary"], "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    build()

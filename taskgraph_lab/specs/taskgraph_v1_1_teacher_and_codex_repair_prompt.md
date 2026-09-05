# TaskGraph v1.1 Prompt + Codex Repair Prompt

> 依据：DeepSeek V4 Flash `thinking_low` / `thinking_disabled` smoke review  
> 目标：减少 DSL 人工摩擦，让 Teacher 主要承担“语义分解”，而不是记忆大量易错格式；禁止通过 VLM_REASON 绕过已有专用 primitive。

---

# A. 新版 Teacher System Prompt（TaskGraph v1.1）

```text
You are a Remote-Sensing Visual Program Compiler.

Your task is NOT to answer the visual question.
Compile the question into a TaskGraph logical DAG for training a small local text Planner.

The Planner should only decide:
1. what logical operations are needed;
2. their dependency order;
3. which previous node feeds which later node.

The Planner must NOT decide:
- which detector/VLM/retriever/model to use;
- thresholds, tile sizes, NMS, zoom depth, GPU placement;
- how fuzzy spatial semantics such as LEFT_OF or NEAR are numerically implemented.

Those are Executor responsibilities.

==================================================
1. OUTPUT
==================================================

Return JSON only.

Output:

{
  "intent": "<IntentLabel>",
  "nodes": [
    {
      "id": "n1",
      "op": "<OperatorName>",
      "inputs": {...},
      "params": {...}
    }
  ],
  "final": {
    "source": "$nX",
    "answer_type": "<AnswerType>"
  }
}

IMPORTANT:
- There is NO user-defined output variable.
- Do NOT emit an "output" field.
- Nodes are referenced only by node id:
  $n1, $n2, ...
- Never reference names such as $region, $color, $cars.
- Every reference must exist.
- The graph must be acyclic.
- Node ids must follow dependency order.

==================================================
2. SYSTEM INPUTS
==================================================

The caller provides:
- question
- question_type
- choices
- inputs

Do not copy these fields into the output.

Available image refs are:
$image0, $image1, ...

==================================================
3. INTENT LABELS
==================================================

Use one of:

SIMPLE_COUNT
RELATIONAL_COUNT
ATTRIBUTE_QUERY
OBJECT_RELATION
OBJECT_CLASSIFICATION
REGIONAL_CLASSIFICATION
MULTILABEL_CLASSIFICATION
MOTION_QUERY
CHANGE_COUNT
ROUTE_PLANNING
COMPLEX_REASONING
OTHER

Intent is only a diagnostic label.
The executable meaning comes from nodes.

==================================================
4. TARGET SPEC
==================================================

Every target MUST be an object:

{
  "category": "...",
  "attributes": {...}
}

Never use:

"target": "large ship"

Instead use:

"target": {
  "category": "ship",
  "attributes": {
    "size": "large"
  }
}

Allowed intrinsic attributes include:

color
shape
size
state
pattern
has_part

Examples:

{
  "category": "car",
  "attributes": {"color": "deep red"}
}

{
  "category": "building",
  "attributes": {
    "color": "white",
    "has_part": "sloped roof"
  }
}

{
  "category": "building",
  "attributes": {
    "shape": "L-shaped"
  }
}

External relations are NOT attributes.

Do NOT encode:
- car next to building
- ship left of forest
- building near harbor

Use LOCATE + SELECT / RELATION.

==================================================
5. LEGAL OPERATORS
==================================================

Use only:

REGION
REGION_FROM_BBOX
FIND_MARKER

LOCATE
SELECT
GROUP

COUNT
ATTRIBUTE
CLASSIFY
MULTILABEL_CLASSIFY
MOTION
RELATION

ABS_DIFF
VLM_REASON

BUILD_ROUTE_CONTEXT
ROUTE_REASON

MATCH_CHOICE

Do not invent operators.

==================================================
6. CANONICAL ENUMS
==================================================

Use EXACT enum strings.

REGION.position:

TOP
BOTTOM
LEFT
RIGHT
CENTER
TOP_LEFT
TOP_RIGHT
BOTTOM_LEFT
BOTTOM_RIGHT
TOP_CENTER
BOTTOM_CENTER
CENTER_LEFT
CENTER_RIGHT

Examples:
"top" -> TOP
"top edge" -> TOP
"top right corner" -> TOP_RIGHT
"bottom right corner" -> BOTTOM_RIGHT
"middle right" -> CENTER_RIGHT

SELECT relation:

LEFT_OF
RIGHT_OF
ABOVE
BELOW
UPPER_LEFT_OF
UPPER_RIGHT_OF
LOWER_LEFT_OF
LOWER_RIGHT_OF
NEAR
NEXT_TO
INSIDE
OUTSIDE
BETWEEN
AROUND
IN_FRONT_OF
BEHIND

Examples:
"next to" -> NEXT_TO
"right of" -> RIGHT_OF

SELECT ordinal order:

TOP_TO_BOTTOM
BOTTOM_TO_TOP
LEFT_TO_RIGHT
RIGHT_TO_LEFT

SELECT rank order:

ASCENDING
DESCENDING

Do not output:
"top right"
"next to"
"right of"
"asc"
"relative position"

when an enum is required.

==================================================
7. OPERATOR SEMANTICS
==================================================

REGION
Input:
  image: ImageRef | Region
Params:
  position
Output:
  Region

Use only for explicit absolute image regions.

--------------------------------------------------

REGION_FROM_BBOX
Input:
  image: ImageRef
Params:
  bbox
  image_size optional
Output:
  Region

If the question explicitly gives a bounding box, use this.
Do not re-locate that same object.

--------------------------------------------------

FIND_MARKER
Input:
  image: ImageRef | Region
Params:
  marker {color?, shape}
Output:
  Region | RegionSet

IMPORTANT:
FIND_MARKER is ONLY for artificial annotation markers added to the benchmark image:
- red circle
- colored rectangle
- annotation border
- marker point

Do NOT use FIND_MARKER for natural objects such as:
- white wall
- red car
- blue roof
- green pond

--------------------------------------------------

LOCATE
Input:
  image: ImageRef | Region
Params:
  target: TargetSpec
Output:
  EntitySet

LOCATE returns a candidate set even if the language description sounds unique.

Do not add a fake SELECT just to convert EntitySet into Entity.
Executor-level modules may resolve candidate ambiguity later.

--------------------------------------------------

SELECT
Inputs:
  candidates
  optional reference
Params:
  one of:

RELATION:
{
  "mode": "RELATION",
  "relation": "<canonical relation enum>"
}

RANK:
{
  "mode": "RANK",
  "criterion": "...",
  "rank": 1,
  "order": "ASCENDING|DESCENDING"
}

ORDINAL:
{
  "mode": "ORDINAL",
  "index": 1,
  "order": "TOP_TO_BOTTOM|BOTTOM_TO_TOP|LEFT_TO_RIGHT|RIGHT_TO_LEFT"
}

EXTREME:
{
  "mode": "EXTREME",
  "direction": "LEFTMOST|RIGHTMOST|TOPMOST|BOTTOMMOST"
}

SUBREGION:
{
  "mode": "SUBREGION",
  "subregion": "LEFT_SIDE|RIGHT_SIDE|ABOVE|BELOW|INSIDE|OUTSIDE|BOTH_SIDES|AROUND"
}

SELECT means:
the relation/ranking criterion is known, and the system selects/filter candidates or creates the corresponding search region.

Fuzzy relations such as NEAR / NEXT_TO / LEFT_OF are still represented by SELECT.
The Planner does NOT decide how they are numerically interpreted.
The Executor may use geometry, a local VLM, or another semantic module.

--------------------------------------------------

GROUP
Input:
  entities: EntitySet
Params:
  mode: ROW | COLUMN | CLUSTER

Use only if the question explicitly describes rows, columns, or clusters.

--------------------------------------------------

COUNT
Input:
  one of:
  image: ImageRef | Region
  entities: EntitySet

Params MUST contain exactly:

{
  "target": TargetSpec,
  "entire": true|false
}

COUNT semantics:

A. image/Region input:
perform exhaustive instance counting inside that visual scope.

B. EntitySet input:
count the already selected/filtered target instances.

"entire": true only when the visual scope is still the whole image or a broad global scope.

"entire": false when:
- upstream REGION restricted the image;
- marker/bbox restricted the image;
- SELECT produced a local scope;
- EntitySet has already been filtered.

Do NOT add:
scope
relation
anchor
constraints
model
threshold
tile_size

--------------------------------------------------

ATTRIBUTE
Input:
  entity: Entity | EntitySet | Region
Params:
  attribute
  part optional
Output:
  Label | Boolean | ScalarFloat

EntitySet is allowed.
If multiple candidates remain, the ATTRIBUTE Executor may resolve the intended entity.

--------------------------------------------------

CLASSIFY
Input:
  Region | Entity | ImageRef
Output:
  Label

--------------------------------------------------

MULTILABEL_CLASSIFY
Input:
  ImageRef | Region
Output:
  LabelSet

--------------------------------------------------

MOTION
Input:
  Region | Entity | EntitySet
Output:
  Boolean

--------------------------------------------------

RELATION
Inputs:
  subject: Entity | EntitySet | Region
  reference: Entity | EntitySet | Region
Params:
  {}
Output:
  Label

RELATION is used when the spatial relation itself is UNKNOWN and is what the question asks for.

Example:

"Where is the white building relative to the red car?"

Correct:
LOCATE(white building)
LOCATE(red car)
RELATION(subject=building, reference=car)
MATCH_CHOICE

Incorrect:
SELECT(relation="relative position")

Difference:

SELECT:
known relation -> select/filter candidates.

RELATION:
known entities -> predict/compute the relation.

--------------------------------------------------

ABS_DIFF
Input:
  a: ScalarInt
  b: ScalarInt
Output:
  ScalarInt

--------------------------------------------------

BUILD_ROUTE_CONTEXT
Inputs:
  image: ImageRef | Region
  start: Entity | EntitySet | Region
  goal: Entity | EntitySet | Region
Output:
  RouteContext

EntitySet is allowed.
Do not create fake ORDINAL nodes merely to force LOCATE output into a single Entity.
The Route Executor may resolve ambiguity.

--------------------------------------------------

ROUTE_REASON
Input:
  context: RouteContext
Params:
  question: "$question"
  choices: "$choices"
Output:
  Answer | Label

Do not add MATCH_CHOICE after ROUTE_REASON if ROUTE_REASON already selects among the provided choices.

--------------------------------------------------

MATCH_CHOICE
Input:
  value: ScalarInt | ScalarFloat | Boolean | Label | LabelSet | Answer
Params:
  choices: "$choices"
Output:
  Answer

==================================================
8. VLM_REASON IS NOT A GENERAL ESCAPE HATCH
==================================================

Do NOT use VLM_REASON if the problem can be represented using dedicated operators.

In particular, VLM_REASON MUST NOT replace:

COUNT tasks
ATTRIBUTE tasks
OBJECT_RELATION tasks
MOTION tasks
BBOX tasks
ROUTE_PLANNING tasks
simple/regional classification tasks

Examples of forbidden shortcuts:

RELATIONAL_COUNT
-> VLM_REASON(question, choices)

OBJECT_RELATION
-> VLM_REASON(question, choices)

ROUTE_PLANNING
-> VLM_REASON(question, choices)

Use VLM_REASON only for genuinely high-level semantic reasoning such as:
- why / cause
- environmental interpretation
- function
- anomaly explanation
- social/economic inference
- open-ended complex reasoning

When VLM_REASON is used, first construct obvious relevant regions/evidence if useful.

==================================================
9. COMMON COMPILATION PATTERNS
==================================================

A. Whole-image count

"How many large ships are in the picture?"

COUNT(
  image=$image0,
  target={category:ship, attributes:{size:large}},
  entire=true
)
-> MATCH_CHOICE

No extra LOCATE is needed.

--------------------------------------------------

B. BBox color

REGION_FROM_BBOX
-> ATTRIBUTE(color)
-> MATCH_CHOICE

Use $n1, $n2 references only.

--------------------------------------------------

C. Relational count

"How many red umbrellas are next to the building at the top?"

REGION(TOP)
-> LOCATE(building)
-> LOCATE(red umbrella)
-> SELECT(
     candidates=umbrellas,
     reference=building,
     relation=NEXT_TO
   )
-> COUNT(
     entities=selected_umbrellas,
     target=red umbrella,
     entire=false
   )
-> MATCH_CHOICE

Do not ignore the SELECT result.

--------------------------------------------------

D. Object relation

"Where is building A relative to car B?"

LOCATE(A)
LOCATE(B)
-> RELATION(subject=A, reference=B)
-> MATCH_CHOICE

--------------------------------------------------

E. Route

REGION(TOP_RIGHT)
-> LOCATE(start building)

REGION(BOTTOM_RIGHT)
-> LOCATE(goal building)

-> BUILD_ROUTE_CONTEXT(
     image=$image0,
     start=start candidates,
     goal=goal candidates
   )
-> ROUTE_REASON

Do NOT invent an ORDINAL node merely because LOCATE returns EntitySet.

==================================================
10. MINIMALITY AND DATAFLOW
==================================================

Use the smallest graph that preserves all semantics.

Every non-final node should contribute to a downstream node unless it is a necessary parallel branch.

Do not generate dead nodes.

If you created:

LOCATE
-> SELECT

then the result of SELECT must actually feed the later COUNT / ATTRIBUTE / other task.

Do not perform a complex branch and then ignore it.

==================================================
11. PHYSICAL EXECUTION DETAILS ARE FORBIDDEN
==================================================

Never emit:

LAE-DINO
CLIP
VisRAG
Qwen
detector
retriever
threshold
pred_score_thr
NMS
tile_size
overlap
zoom depth
beam width
GPU/CPU scheduling
model residency

Those belong to the Executor.

==================================================
12. FINAL CHECK BEFORE OUTPUT
==================================================

Before emitting JSON, silently verify:

1. Did I use only legal operators?
2. Are all enum strings canonical?
3. Is every target a TargetSpec object?
4. Are all references only $imageX or $nX?
5. Is every node actually used?
6. Did I avoid VLM_REASON when a dedicated operator exists?
7. For "where is A relative to B", did I use RELATION rather than SELECT?
8. For relational counting, does the SELECT result feed COUNT?
9. Did I avoid inventing fake ORDINAL/RANK operations solely for type conversion?
10. Is the graph minimal?

Return JSON only.
```

---

# B. Codex Repair Prompt

```text
你现在位于仓库 ericli829/sat-rs-vlm 的 TaskGraph 实验区。

目标：
根据 DeepSeek V4 Flash smoke review 暴露的问题，修复 taskgraph_lab 的 schema、validator、prompt 和 tests。

不要实现正式 TaskGraph runtime。
不要接真实 LAE / CLIP / VLM Executor。
本次只修“训练数据生成层”。

请先阅读：

- taskgraph_lab/specs/
- taskgraph_lab/prompts/system_prompt.txt
- taskgraph_lab/taskgraph/
- taskgraph_lab/generation/
- taskgraph_lab/tests/
- prompt_review.json

重点理解 prompt_review.json 中 thinking_low / thinking_disabled 的失败案例。

============================================================
1. 本次问题总结
============================================================

当前 smoke 暴露：

A. VLM_REASON 被 Teacher 当成万能 escape hatch。
例如 relational count / object relation 可直接生成 VLM_REASON，
虽然 schema valid，但不符合系统架构。

B. prompt 与严格 schema 的 canonical enum 不匹配。
模型会生成：
"top right corner"
"next to"
"asc"
但 schema 要：
TOP_RIGHT
NEXT_TO
ASCENDING / spatial order enum

C. TargetSpec 格式不稳定。
模型经常输出：
"target": "large ship"
而不是结构化 TargetSpec。

D. GraphNode 同时存在：
id="n1"
output="region"
但只能引用 $n1，不能引用 $region，
导致 Teacher 自然地产生 invalid refs。

E. LOCATE -> EntitySet，
但 BUILD_ROUTE_CONTEXT / RELATION / ATTRIBUTE 等要求单个 Entity，
导致 Teacher 被迫发明无语义的 SELECT/ORDINAL 来做类型转换。

F. relational COUNT 数据流断裂：
SELECT 返回 EntitySet，
但 COUNT 只接受 ImageRef/Region，
因此模型无法自然表达：
LOCATE -> SELECT -> COUNT(selected instances)

G. FIND_MARKER 被误用于自然物体（如 white wall）。

H. validator 可以接受：
VLM_REASON shortcut
dead node
unused SELECT branch
等“形式合法、语义错误”的 graph。

I. repair 擅长格式修正，但会把 semantic error 强行改成合法 enum，
例如把未知 relation 错改成 RIGHT_OF。
因此 repair 不应该承担复杂重规划。

============================================================
2. Schema 修改
============================================================

2.1 删除 Planner graph 中的 GraphNode.output 字段。

目标 node：

{
  "id": "n1",
  "op": "...",
  "inputs": {...},
  "params": {...}
}

所有中间结果只允许通过：

$n1
$n2

引用。

更新：
- Pydantic schema
- serialization
- canonicalizer
- prompts
- tests
- fixtures

如果需要兼容旧数据：
可以在 loader 中临时接受 output 但 canonical export 必须删除；
不要让新 Teacher prompt 再生成 output。

------------------------------------------------------------

2.2 COUNT 输入扩展。

允许：

COUNT inputs:
A.
{"image": ImageRef|Region}

或 B.
{"entities": EntitySet}

Params 仍严格只有：

target
entire

禁止增加第三个语义参数。

当 input=entities 时：
entire 应为 false。

Type checker 必须支持。

------------------------------------------------------------

2.3 Semantic executors 接受 EntitySet。

至少修改类型签名：

ATTRIBUTE:
Entity | EntitySet | Region

MOTION:
Entity | EntitySet | Region

RELATION.subject:
Entity | EntitySet | Region

RELATION.reference:
Entity | EntitySet | Region

BUILD_ROUTE_CONTEXT.start:
Entity | EntitySet | Region

BUILD_ROUTE_CONTEXT.goal:
Entity | EntitySet | Region

理由：
LOCATE 的统一输出仍保持 EntitySet；
候选消歧属于未来 Executor，不要求 Planner 增加假的类型转换节点。

不要新增 RESOLVE_ENTITY primitive。

============================================================
3. Prompt 修改
============================================================

将新的 system_prompt.txt 更新为我提供的 TaskGraph v1.1 Teacher Prompt。

重点必须明确：

- 所有 target 必须是 TargetSpec object；
- canonical enums 的 exact spelling；
- 只能引用 $imageX / $nX；
- FIND_MARKER 只用于 benchmark artificial marker；
- SELECT vs RELATION 的明确对照；
- relational count 必须把 SELECT 结果传给 COUNT.entities；
- route 不要为 EntitySet->Entity 发明 fake ordinal；
- VLM_REASON 不能替代 COUNT / ATTRIBUTE / OBJECT_RELATION / ROUTE；
- no dead nodes；
- output 字段删除。

保留 prompt 与代码分离。

============================================================
4. Validator 增强
============================================================

新增 semantic/static checks。

4.1 Dead node / unused branch

除 final source 和合理并行依赖外，
每个 node 都必须被后续 node 或 final 使用。

如果 n3 SELECT 从未被消费：
WARNING 或 ERROR。

建议：
对于明显 semantic branch（SELECT/LOCATE）的完全 dead node -> ERROR。

------------------------------------------------------------

4.2 Dedicated operator bypass

如果 intent 是：

SIMPLE_COUNT / RELATIONAL_COUNT
但 graph 只有 VLM_REASON：
ERROR code:
dedicated_operator_bypass

如果 intent 是 ATTRIBUTE_QUERY
但没有 ATTRIBUTE，直接 VLM_REASON：
ERROR

如果 intent 是 OBJECT_RELATION
但没有 RELATION：
ERROR

如果 intent 是 ROUTE_PLANNING
但没有 BUILD_ROUTE_CONTEXT + ROUTE_REASON：
ERROR

如果 intent 是 MOTION_QUERY
但没有 MOTION：
ERROR

分类问题可根据 schema 当前设计决定是否允许 VLM fallback，
但默认优先 CLASSIFY/MULTILABEL_CLASSIFY。

------------------------------------------------------------

4.3 Relational COUNT dependency coverage

如果：
intent=RELATIONAL_COUNT

并且 graph 有 SELECT，
最终 COUNT 不依赖该 SELECT 输出（directly or through descendants）：
ERROR:
relation_result_not_consumed

============================================================
5. SELECT vs RELATION 检查
============================================================

增加 heuristic warning/error：

如果问题形式明显是：

"where is A relative to B"
"what is the position of A relative to B"

且 graph 使用 SELECT 作为最终 relation result、没有 RELATION：
ERROR/WARNING:
relation_query_should_use_relation

至少在 smoke fixture 中覆盖。

不要自动把 relation 猜成 RIGHT_OF/LEFT_OF。

============================================================
6. Marker heuristic
============================================================

FIND_MARKER 只用于显式人工 marker phrase：

red circle
red box
bounding outline
highlighted region
marked area
colored border
marker point

如果 target 只是自然物体：

white wall
red car
blue roof
green pond

不要触发 marker heuristic。

Validator 可以做保守 WARNING，
不要过度硬编码所有自然语言。

============================================================
7. Canonical enum preprocessing
============================================================

Teacher 应输出 canonical enum。

同时为了降低纯格式失败成本，可以增加“非语义 canonicalization”：

允许安全映射：

top -> TOP
top edge -> TOP
top right -> TOP_RIGHT
top right corner -> TOP_RIGHT
bottom right -> BOTTOM_RIGHT
next to -> NEXT_TO
right of -> RIGHT_OF

这类映射必须：
- deterministic；
- 仅做 lexical normalization；
- 不能改变图逻辑；
- 不能把 "relative position" 猜成 RIGHT_OF；
- 不能给未知 relation 选择任意合法 enum。

建议 canonicalizer 在 schema validation 前有一个 narrow normalization pass，
并在 metadata 记录 normalized_fields。

不要让 repair API 去承担这些低级格式修复。

============================================================
8. Repair 策略修改
============================================================

把错误分两类。

AUTO_NORMALIZABLE:
- enum capitalization / known lexical alias
- legacy output field removal
- safe node reference migration if unambiguous
- trivial TargetSpec wrapping only when semantics are unambiguous

LLM_REPAIRABLE:
- malformed JSON
- missing required simple field
- straightforward schema mismatch

SEMANTIC_ERROR:
- SELECT vs RELATION
- missing dependency
- dead branch
- VLM_REASON bypass
- invented rank/ordinal
- wrong target/anchor

SEMANTIC_ERROR 不要简单用当前 repair prompt“随便修成合法 schema”。

可以：
- reject；
或
- 使用单独 semantic-regeneration prompt 从原 question 重新生成整张 graph。

第一版优先 reject，避免污染训练数据。

============================================================
9. Tests
============================================================

基于 prompt_review.json 增加 regression tests。

至少：

A. whole-image large ship count
正确：
COUNT(target structured, entire=true)
-> MATCH_CHOICE

B. bbox color
只能 $n1 / $n2 引用；
不允许 $region / $color

C. red umbrellas next to top building
正确链路：
REGION TOP
LOCATE building
LOCATE umbrella
SELECT NEXT_TO
COUNT entities=$select_result
MATCH_CHOICE

必须检查 SELECT 被 COUNT 消费。

D. object spatial relation
LOCATE building
LOCATE car
RELATION
MATCH_CHOICE

不能：
VLM_REASON shortcut
不能：
SELECT(relative position)

E. route
REGION TOP_RIGHT
LOCATE start
REGION BOTTOM_RIGHT
LOCATE goal
BUILD_ROUTE_CONTEXT(EntitySet allowed)
ROUTE_REASON

不需要 fake ORDINAL。

F. FIND_MARKER natural wall negative case

G. dead node detection

H. dedicated_operator_bypass

I. lexical enum normalization

J. unknown relation 不得被 canonicalizer 猜成合法方向

============================================================
10. Smoke comparison
============================================================

修复后重新使用同一 5 条样本运行：

deepseek-v4-flash:
- thinking_disabled
- thinking_low

保持：
- model
- temperature
- samples
- prompt version之外的 generation config

尽量一致。

输出新的 review JSON，并比较：

initial schema valid
initial graph valid
initial type valid
semantic valid
repair count
reject count
latency
reasoning tokens

另外人工统计：

architecturally_correct_graphs / 5

不要只看 validator terminal_status。

============================================================
11. 验收
============================================================

最终报告：

1. 修改文件列表；
2. schema diff；
3. validator 新规则；
4. canonicalizer 新规则；
5. regression tests；
6. pytest 结果；
7. 新 smoke 对比；
8. thinking_disabled vs thinking_low 哪个更适合作为 teacher；
9. 仍无法自然表达的问题；
10. 是否建议冻结 TaskGraph v1.1。

不要修改 taskgraph_lab 之外已有正式工程，除非绝对必要。
```

---

# C. 训练本地 Planner 的数据量建议

这是经验估计，不是来自 smoke 文件本身。

任务特点：

- 输出空间高度受限；
- operator 只有约十几个；
- 大量题目属于有限组合；
- 重点是 schema adherence + semantic decomposition；
- 不要求 Planner 看图；
- 真实视觉工作由后端 expert 承担。

因此它比通用 reasoning SFT 的数据需求小得多。

建议分阶段：

## 1. Schema feasibility

300–500 条高质量数据。

目标：
验证 local model 能否学会：
- JSON
- operator names
- refs
- TargetSpec
- 常见 1–3 node graph

不用于最终模型。

## 2. 可用 MVP

2,000–4,000 条高质量 canonical graph。

需要覆盖：
- 所有 operator
- 简单与嵌套问题
- MME + XLRS
- route / double-image / ordinal / relational count 等长尾

对于本来具备较好 instruction-following 能力的 1B–3B text model，
这通常已经足够判断方案是否可行。

## 3. 推荐正式规模

5,000–10,000 条高质量数据。

组成可为：

- 2k–4k 真实 benchmark question graph
- 2k–4k paraphrase / synthetic composition
- 1k–2k difficult / correction / negative examples

这一规模更适合稳定训练一个本地小 Planner。

## 4. 不建议一开始追求

30k–100k 条低质量 teacher graph。

结构任务里错误标签比数据少更危险。

优先：

5k clean

而不是：

50k noisy

---

# D. 数据生成来源建议

推荐混合，不依赖一种 teacher。

## Tier 1：程序自动标注

适合规则非常明确的问题：

- bbox color
- bbox motion
- simple whole-image count
- absolute-region count
- two-image count difference
- standard route skeleton
- obvious classification

根据 dataset category + regex/template 自动生成 graph。

这部分几乎零成本，而且标签最稳定。

目标可以占：

30–50%

---

## Tier 2：强 LLM 一次生成 + Validator

适合：

- 中等 relation
- ordinal
- nested count
- attribute reference

用低温强模型生成，
然后：
schema + type + semantic validator。

目标约：

30–40%

---

## Tier 3：Agent / expensive teacher

只处理 difficult bucket：

- 多 anchor
- 嵌套 referring expression
-复杂 route endpoint
- multi-region reasoning
- 多层 ordinal + relation

流程：

question
→ teacher draft
→ validator
→ semantic critic
→ regenerate
→ final verifier

只占：

10–20%

这样成本可控。

---

# E. 最推荐的数据构造策略

不要让 Agent 从 0 生成全部 5k。

推荐：

1. 先写 deterministic annotator，吃掉最简单的 40%左右；
2. 强模型生成中等难度；
3. Validator 自动分桶；
4. 只有 failed / ambiguous bucket 进入 expensive agent；
5. 人工抽检每一类；
6. 对已确认 graph 做 paraphrase augmentation；
7. 同一个 logical graph 可配 2–4 种自然语言表达。

这样一个高质量 graph 可以扩成多个 Planner 训练样本，而不用重复做视觉标注。


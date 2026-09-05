# TaskGraph v1：遥感复杂问题 DAG 中间表示定义与训练数据生成规范

> 目标：为 MME RealWorld RS、XLRS-Bench 等高分辨率遥感问答建立统一的逻辑中间表示，使强文本模型能够把自然语言问题编译成可执行的任务图，并据此自动生成高质量训练数据。  
> DAG = Directed Acyclic Graph（有向无环图）。  
> 本文档同时面向两类读者：
>
> 1. **人**：用于介绍系统为什么这样拆、每种类型和操作是什么；
> 2. **强模型**：可直接把本文档作为大部分提示词上下文，生成 TaskGraph 标注和训练样本。

---

# 0. 设计目标

遥感高分辨率问题的困难往往不在“最终任务名称”，而在于一个问题包含多层空间限定、锚点定位、排序、属性、计数、双图比较或路线推理。

例如：

```text
How many swimming pools are there to the left of the forest
in the bottom-left corner?
```

它不是一个带复杂 `constraints` 的 COUNT，而应编译成：

```text
IMAGE
 ↓
REGION(bottom_left)
 ↓
LOCATE(forest)
 ↓
SELECT(subregion=left_of forest)
 ↓
COUNT(swimming_pool, entire=False)
```

再如：

```text
How many differences are there in the number of farms
within the areas marked by red circles in the two images?
```

它天然是一张有两条分支再汇聚的 DAG：

```text
IMAGE_0                     IMAGE_1
  ↓                           ↓
FIND_MARKER(red_circle)    FIND_MARKER(red_circle)
  ↓                           ↓
COUNT(farm, False)         COUNT(farm, False)
      \                       /
       \                     /
              ABS_DIFF
                 ↓
            MATCH_CHOICE
```

因此本系统遵循：

> **Planner 描述逻辑程序；Executor 决定物理执行策略。**

逻辑图中不应出现具体模型、阈值、切片大小、CLIP、LAE-DINO、NMS 等实现细节。

---

# 1. 总体对象关系

TaskGraph v1 由四层组成：

```text
TaskGraph
├── metadata / original question
├── inputs
├── nodes[]
└── final
```

核心类：

```text
TaskGraph
GraphNode
FinalSpec
TargetSpec
AttributeSpec

以及一组 Runtime Data Types：
ImageRef
Region
RegionSet
Entity
EntitySet
ScalarInt
ScalarFloat
Boolean
Label
LabelSet
RouteContext
Evidence
EvidenceSet
Answer
```

GraphNode 的具体语义由 `op` 和对应的 Params 类决定。

---

# 2. TaskGraph 顶层定义

## 2.1 TaskGraph

### 作用

表示一个问题完整的逻辑执行计划。

### 推荐定义

```python
class TaskGraph:
    version: str
    question: str
    question_type: QuestionType
    choices: list[str] | None
    inputs: dict[str, InputSpec]
    intent: IntentLabel | None
    nodes: list[GraphNode]
    final: FinalSpec
```

### 字段解释

| 字段 | 类型 | 生成者 | 含义 |
|---|---|---|---|
| `version` | `str` | 系统 | IR schema 版本，如 `taskgraph-v1` |
| `question` | `str` | 系统 | 原始问题，禁止 Planner 改写 |
| `question_type` | `QuestionType` | 系统 | 题型 |
| `choices` | `list[str] \| None` | 系统 | 选择题选项 |
| `inputs` | `dict[str, InputSpec]` | 系统 | 输入图像或多图 |
| `intent` | `IntentLabel \| None` | Planner | 仅用于统计、诊断，不负责真正路由 |
| `nodes` | `list[GraphNode]` | Planner | 可执行逻辑 DAG |
| `final` | `FinalSpec` | Planner | 最终答案来源 |

## 2.2 QuestionType

```python
enum QuestionType:
    MULTIPLE_CHOICE
    MULTIPLE_CHOICE_SINGLE
    MULTIPLE_CHOICE_MULTI
    FREE_FORM
    BOOLEAN
    INTEGER
```

- `MULTIPLE_CHOICE`：通用选择题；单选/多选由原始题面语义和 Teacher 的
  `final.answer_type` 表达；
- `MULTIPLE_CHOICE_SINGLE`、`MULTIPLE_CHOICE_MULTI`：仅为兼容旧版已规范化数据；
  validator 不再用这两个来源标签覆盖题面语义；
- `FREE_FORM`：自由文本；
- `BOOLEAN`：是/否；
- `INTEGER`：直接输出计数值。

## 2.3 InputSpec

```python
class InputSpec:
    type: Literal["image"]
    uri_or_key: str
```

双图时使用 `image0`、`image1` 等独立入口。输入由系统注入，不由 Planner 创建。

## 2.4 IntentLabel

`intent` 仅用于日志、采样和训练分析，不参与真实执行路由。

```python
enum IntentLabel:
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
```

---

# 3. GraphNode

```python
class GraphNode:
    id: str
    op: OperatorName
    inputs: dict[str, Ref]
    params: dict
    output: str
```

示例：

```json
{
  "id": "n3",
  "op": "COUNT",
  "inputs": {"image": "$n2"},
  "params": {
    "target": {"category": "swimming_pool", "attributes": {}},
    "entire": false
  },
  "output": "pool_count"
}
```

## 3.1 Ref

```text
$image0
$image1
$n1
$n2
```

约束：

1. `$imageX` 引用 TaskGraph.inputs；
2. `$nX` 只能引用已定义节点；
3. 图必须无环；
4. 引用对象类型必须满足 operator signature。

## 3.2 output

`output` 是便于人类阅读的变量名，不负责声明类型。真实类型由 operator 推导。

---

# 4. FinalSpec

```python
class FinalSpec:
    source: Ref
    answer_type: AnswerType
```

```python
enum AnswerType:
    CHOICE_SINGLE
    CHOICE_MULTI
    INTEGER
    BOOLEAN
    LABEL
    LABEL_SET
    TEXT
```

`CHOICE_MULTI` 定义为不定项选择：最终 Choice VLM 必须选择所有适用的原始选项，
合法结果包含一个或多个 option id。某个样本的标准答案恰好只有一个标签时，仍可属于
`CHOICE_MULTI`；`CHOICE_SINGLE` 才表示题目协议强制且仅允许一个选项。最终 VLM
prompt 必须显式写出该约束并使用 `{"choice_ids":[...]}`，不能只写含糊的
“return the option id”。

---

# 5. TargetSpec

TargetSpec 只描述**目标本身是什么**，不描述目标与外部对象的空间关系。

```python
class TargetSpec:
    category: str
    attributes: dict[str, str | int | float | bool] = {}
```

允许：

```json
{"category": "vehicle", "attributes": {"color": "red"}}
```

```json
{"category": "building", "attributes": {"shape": "L-shaped"}}
```

```json
{"category": "ship", "attributes": {"size": "large", "has_part": "helipad"}}
```

不允许：

```text
vehicle left of forest
ship in bottom-left
building near harbor
```

这些是外部关系，必须由 REGION / LOCATE / SELECT 等节点表达。

---

# 6. AttributeSpec

```python
class AttributeSpec:
    name: str
    value: str | int | float | bool | None
    part: str | None
```

例如：

```json
{"name": "color", "value": null, "part": "roof"}
```

第一版 JSON 可以继续使用简化字典；训练数据生成器应理解语义一致。

---

# 7. Runtime Data Types

Runtime Data Type 是节点之间真正传递的数据类型。Planner 只负责正确连接，不负责填内部实现细节。

## 7.1 ImageRef

```python
class ImageRef:
    id: str
    width: int | None
    height: int | None
```

原始图像引用，不在 Planner 输出里展开像素。

## 7.2 Region

```python
class Region:
    image_ref: ImageRef
    geometry: internal
```

对 Planner 而言：Region 是一张图中的某个视觉作用域。

内部可以是 bbox、polygon、mask、crop transform 或 global-local mapping，但不暴露给 Planner。

## 7.3 RegionSet

```python
class RegionSet:
    items: list[Region]
```

用于多个 marker、多块候选 ROI 等情况。

## 7.4 Entity

```python
class Entity:
    category: str | None
    region: Region
    confidence: float | None
    attributes: dict
```

内部可附带 bbox、mask、score、provider、scale、tile 等物理信息，但不进入 TaskGraph。

## 7.5 EntitySet

```python
class EntitySet:
    items: list[Entity]
```

## 7.6 ScalarInt

```python
class ScalarInt:
    value: int
```

COUNT、ABS_DIFF 的典型输出。

## 7.7 ScalarFloat

```python
class ScalarFloat:
    value: float
```

为距离、比例等未来扩展保留。

## 7.8 Boolean

```python
class Boolean:
    value: bool
```

## 7.9 Label

```python
class Label:
    value: str
```

ATTRIBUTE、CLASSIFY、RELATION 等的常见输出。

## 7.10 LabelSet

```python
class LabelSet:
    values: list[str]
```

MULTILABEL_CLASSIFY 输出。

## 7.11 RouteContext

```python
class RouteContext:
    region: Region
    start: Entity | Region
    goal: Entity | Region
    metadata: internal
```

表示围绕起点、终点和必要上下文构造好的路线推理输入。内部可包含 marker-rendered image、large-context crop、坐标变换等。

## 7.12 Evidence

```python
class Evidence:
    source: Ref
    payload: AnyTypedRuntimeObject
```

## 7.13 EvidenceSet

```python
class EvidenceSet:
    items: list[Evidence]
```

用于多分支、多区域、多图 reasoning。

## 7.14 Answer

```python
class Answer:
    value: str | int | bool | list[str]
```

---

# 8. Operator 总览

TaskGraph v1 推荐 17 个逻辑 operator：

```text
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
```

以下内容不进入逻辑 DAG：

```text
CLIP
LAE-DINO
Retriever
Zoom
Answerability Model
NMS
tile_size
score_threshold
pred_score_thr
multi-scale detector
GPU/CPU placement
```

它们属于物理执行层。

---

# 9. REGION

## 定义

```python
REGION(
    image: ImageRef | Region,
    position: RegionPosition
) -> Region
```

## RegionPosition

```python
enum RegionPosition:
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
```

必要时可扩 FAR_LEFT、FAR_RIGHT、UPPER_MIDDLE、LOWER_MIDDLE，但优先用有限规范词表。

## 输入

```text
ImageRef | Region
```

## 输出

```text
Region
```

---

# 10. REGION_FROM_BBOX

## 定义

```python
REGION_FROM_BBOX(
    image: ImageRef,
    bbox: [x1, y1, x2, y2],
    image_size: [width, height] | None
) -> Region
```

用于题目已经显式给出 bounding box 的情况。不要重新 LOCATE 同一个目标。

---

# 11. FIND_MARKER

```python
FIND_MARKER(
    image: ImageRef | Region,
    marker: MarkerSpec
) -> Region | RegionSet
```

```python
class MarkerSpec:
    color: str | None
    shape: str
```

典型 marker：

```text
red circle
red rectangle
light blue border
```

若题目明确存在多个 marker，输出 RegionSet。

---

# 12. LOCATE

```python
LOCATE(
    image: ImageRef | Region,
    target: TargetSpec
) -> EntitySet
```

即使语义上只有一个目标，v1 仍建议统一输出 EntitySet，再通过 SELECT 选出唯一目标。

LOCATE 只表达“找什么”，不表达使用 LAE、CLIP、VLM 还是 Retriever。

---

# 13. SELECT

SELECT 是 TaskGraph v1 中最重要的通用筛选算子。

```python
SELECT(
    candidates: EntitySet | Region | RegionSet,
    reference: Entity | EntitySet | Region | None,
    spec: SelectSpec
) -> Entity | EntitySet | Region | RegionSet
```

```python
class SelectSpec:
    mode: SelectMode
    relation: SpatialRelation | None
    criterion: str | None
    rank: int | None
    order: SortOrder | None
    index: int | None
    direction: ExtremeDirection | None
    subregion: SubregionType | None
```

## 13.1 SelectMode

```python
enum SelectMode:
    RELATION
    RANK
    ORDINAL
    EXTREME
    SUBREGION
```

## 13.2 SpatialRelation

```python
enum SpatialRelation:
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
```

## 13.3 RANK

用于 largest、smallest、nearest、farthest、second largest 等。

```json
{"mode": "RANK", "criterion": "size", "rank": 1, "order": "DESCENDING"}
```

## 13.4 ORDINAL

```json
{"mode": "ORDINAL", "index": 2, "order": "TOP_TO_BOTTOM"}
```

```python
enum SortOrder:
    ASCENDING
    DESCENDING
    TOP_TO_BOTTOM
    BOTTOM_TO_TOP
    LEFT_TO_RIGHT
    RIGHT_TO_LEFT
```

## 13.5 EXTREME

```python
enum ExtremeDirection:
    LEFTMOST
    RIGHTMOST
    TOPMOST
    BOTTOMMOST
```

## 13.6 SUBREGION

```python
enum SubregionType:
    LEFT_SIDE
    RIGHT_SIDE
    ABOVE
    BELOW
    INSIDE
    OUTSIDE
    BOTH_SIDES
    AROUND
```

用于从参考实体/区域构造新的几何作用域，例如“forest 左侧”“U-shaped road 内部”“highway 两侧”。

---

# 14. GROUP

```python
GROUP(
    entities: EntitySet,
    mode: GroupMode
) -> EntitySet | RegionSet
```

```python
enum GroupMode:
    ROW
    COLUMN
    CLUSTER
```

用于 rightmost column、second row、cluster 等结构。

---

# 15. COUNT

COUNT 是 detector-driven macro primitive。

```python
COUNT(
    image: ImageRef | Region,
    target: TargetSpec,
    entire: bool
) -> ScalarInt
```

## 参数

语义参数严格只有：

```text
target
entire
```

当前视觉区域通过 `inputs.image` 传入，不属于 params。

## entire

`entire=True`：当前输入仍是整图或大范围全域作用域，物理 Executor 可以启用全图候选粗筛、多尺度搜索等策略。

`entire=False`：上游 DAG 已完成主要空间约束，Executor 在当前 ROI 内完整统计。

无论 True/False，COUNT 都必须尽可能 exhaustive。

## 禁止

COUNT 不得接收：

```text
scope
relation
anchor
spatial_constraints
model
threshold
tile_size
```

## 属性

intrinsic attribute 可写进 TargetSpec，如 red car、large ship with helipad、white airplane。

外部空间关系必须由上游 DAG 处理。

---

# 16. ATTRIBUTE

```python
ATTRIBUTE(
    entity: Entity | EntitySet | Region,
    attribute: str,
    part: str | None
) -> Label | Boolean | ScalarFloat
```

典型属性：color、shape、material、state、orientation、texture。

`part` 用于 roof color、wall color、flag color、vehicle body color 等。

---

# 17. CLASSIFY

```python
CLASSIFY(
    input: Region | Entity | ImageRef,
    label_space: list[str] | None
) -> Label
```

用于 bbox 内目标类别、red-circle 区域 land-use、单标签分类。

---

# 18. MULTILABEL_CLASSIFY

```python
MULTILABEL_CLASSIFY(
    input: ImageRef | Region,
    label_space: list[str]
) -> LabelSet
```

用于 overall land-use 多选题。

---

# 19. MOTION

```python
MOTION(
    input: Region | Entity
) -> Boolean
```

典型链路：

```text
REGION_FROM_BBOX
↓
MOTION
↓
MATCH_CHOICE
```

---

# 20. RELATION

```python
RELATION(
    subject: Entity | Region,
    reference: Entity | Region
) -> Label
```

与 SELECT 的区别：

```text
SELECT：已知 relation，筛对象
RELATION：已知两个对象，求 relation
```

---

# 21. ABS_DIFF

```python
ABS_DIFF(
    a: ScalarInt,
    b: ScalarInt
) -> ScalarInt
```

数学定义：

```text
|a - b|
```

用于双时相 counting difference。

---

# 22. VLM_REASON

VLM = Vision-Language Model（视觉语言模型）。

```python
VLM_REASON(
    image: ImageRef | Region | None,
    evidence: EvidenceSet | list[Ref] | None,
    question: str,
    choices: list[str] | None
) -> Label | Boolean | Text | Answer
```

用于不能安全降解为几何/检测程序的高层因果、环境、功能、社会经济或异常解释问题。

原则：能由 COUNT、RELATION、ABS_DIFF、MATCH_CHOICE 等确定性节点完成时，不要直接交给 VLM_REASON。

---

# 23. BUILD_ROUTE_CONTEXT

```python
BUILD_ROUTE_CONTEXT(
    image: ImageRef | Region,
    start: Entity | Region,
    goal: Entity | Region
) -> RouteContext
```

它只表达“生成适合路线推理的上下文”。内部可做 large-context crop、marker、resize、坐标变换等，但这些不进入 TaskGraph。

---

# 24. ROUTE_REASON

```python
ROUTE_REASON(
    context: RouteContext,
    question: str,
    choices: list[str]
) -> Answer | Label
```

第一版不要求 Planner 把每个 intersection / fork / U-turn / roundabout 拆成几十个 node。

---

# 25. MATCH_CHOICE

```python
MATCH_CHOICE(
    value: ScalarInt | Boolean | Label | LabelSet | Answer,
    choices: list[str]
) -> Answer
```

如果上游已经得到确定性结果，例如 COUNT=25，而选项 C=25，则直接映射到 C，不再让语言模型重新推理一次。

---

# 26. Operator 类型签名总表

| Operator | 输入 | 参数 | 输出 |
|---|---|---|---|
| `REGION` | `ImageRef \| Region` | `position` | `Region` |
| `REGION_FROM_BBOX` | `ImageRef` | `bbox, image_size?` | `Region` |
| `FIND_MARKER` | `ImageRef \| Region` | `marker` | `Region \| RegionSet` |
| `LOCATE` | `ImageRef \| Region` | `target` | `EntitySet` |
| `SELECT` | `EntitySet/Region/RegionSet + optional reference` | `SelectSpec` | `Entity/EntitySet/Region/RegionSet` |
| `GROUP` | `EntitySet` | `mode` | `EntitySet \| RegionSet` |
| `COUNT` | `ImageRef \| Region` | `target, entire` | `ScalarInt` |
| `ATTRIBUTE` | `Entity/EntitySet/Region` | `attribute, part?` | `Label/Boolean/ScalarFloat` |
| `CLASSIFY` | `Region/Entity/ImageRef` | `label_space?` | `Label` |
| `MULTILABEL_CLASSIFY` | `ImageRef \| Region` | `label_space` | `LabelSet` |
| `MOTION` | `Region \| Entity` | 无 | `Boolean` |
| `RELATION` | `subject + reference` | 无 | `Label` |
| `ABS_DIFF` | `ScalarInt + ScalarInt` | 无 | `ScalarInt` |
| `VLM_REASON` | `Image/Region + optional Evidence` | `question, choices?` | `Answer/Label/Boolean/Text` |
| `BUILD_ROUTE_CONTEXT` | `image + start + goal` | 无 | `RouteContext` |
| `ROUTE_REASON` | `RouteContext` | `question, choices` | `Answer/Label` |
| `MATCH_CHOICE` | `Scalar/Boolean/Label/LabelSet/Answer` | `choices` | `Answer` |

---

# 27. 类型检查规则

Validator 至少检查：

1. 所有 `$nX` / `$imageX` 引用存在；
2. 图无环；
3. operator 名称合法；
4. 输入类型满足 signature；
5. params 满足 operator schema；
6. final.source 存在；
7. final.answer_type 与源类型兼容；
8. COUNT.params 严格只有 target + entire；
9. 不允许 model、threshold、tile_size 等物理字段；
10. bbox 题优先 REGION_FROM_BBOX；
11. marker 题优先 FIND_MARKER；
12. external relation 不得隐藏在 TargetSpec；
13. 确定性计算优先于 VLM_REASON；
14. 多选题输出必须能表达 LabelSet / CHOICE_MULTI。

---

# 28. 典型编译规则

## 28.1 absolute region

```text
"in the lower left"
→ REGION(BOTTOM_LEFT)
```

## 28.2 anchor object

```text
"below the forest"
→ LOCATE(forest)
→ SELECT(SUBREGION, BELOW)
```

## 28.3 nested relation

```text
"parking lot in the inner circle of the U-shaped road
near the water and next to the playground"
```

应分层：

```text
LOCATE(U-shaped road)
LOCATE(water)
LOCATE(playground)
SELECT road candidates near water
SELECT remaining candidates next_to playground
SELECT subregion inside
LOCATE(parking_lot)
```

不要生成一个超长 target string。

## 28.4 ordinal

```text
"the second building area from top to bottom"
→ LOCATE(building area)
→ SELECT(ORDINAL, index=2, TOP_TO_BOTTOM)
```

## 28.5 superlative

```text
"largest roundabout"
→ LOCATE(roundabout)
→ SELECT(RANK, criterion=size, rank=1, descending)
```

## 28.6 intrinsic attribute

```text
"red cars"
```

可以进入 TargetSpec。具体由 detector 直接属性 prompt，还是 detect+attribute filter，属于物理 Executor。

## 28.7 part attribute

```text
"roof color of the building"
→ LOCATE(building)
→ ATTRIBUTE(color, part=roof)
```

## 28.8 change count

```text
difference in number between two images
→ image0 branch COUNT
→ image1 branch COUNT
→ ABS_DIFF
```

## 28.9 route

```text
shortest driving route from A to B
→ LOCATE(A)
→ LOCATE(B)
→ BUILD_ROUTE_CONTEXT
→ ROUTE_REASON
```

---

# 29. MME 示例：简单区域计数

问题：

```text
How many airplanes are there in the lower left area of this picture?
```

DAG：

```json
{
  "intent": "SIMPLE_COUNT",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {"image": "$image0"},
      "params": {"position": "BOTTOM_LEFT"},
      "output": "bottom_left"
    },
    {
      "id": "n2",
      "op": "COUNT",
      "inputs": {"image": "$n1"},
      "params": {
        "target": {"category": "airplane", "attributes": {}},
        "entire": false
      },
      "output": "airplane_count"
    },
    {
      "id": "n3",
      "op": "MATCH_CHOICE",
      "inputs": {"value": "$n2"},
      "params": {"choices": "$choices"},
      "output": "answer"
    }
  ],
  "final": {"source": "$n3", "answer_type": "CHOICE_SINGLE"}
}
```

---

# 30. MME 示例：复杂嵌套 Counting

问题：

```text
How many cars are there in the parking lot in the inner circle
of the U-shaped road near the water and next to the playground
in the middle right area of this picture?
```

推荐逻辑：

```text
IMAGE
↓
REGION(middle_right)
├─ LOCATE(U-shaped road)
├─ LOCATE(water)
└─ LOCATE(playground)

U-shaped road candidates
↓ SELECT(near water)
↓ SELECT(next_to playground)
↓ SELECT(subregion=inside)
↓ LOCATE(parking_lot)
↓ COUNT(car, False)
↓ MATCH_CHOICE
```

- `water` 和 `playground` 是 anchor；
- `U-shaped road` 是需要消歧的中间对象；
- `parking lot` 是最终 count scope；
- `car` 才是 COUNT target。

---

# 31. MME 示例：序数 + 分组

问题：

```text
How many houses are there in the rightmost column of the second
building area from top to bottom in the area on the right of the picture?
```

推荐：

```text
REGION(right)
↓
LOCATE(building_area)
↓
SELECT(ORDINAL, index=2, TOP_TO_BOTTOM)
↓
LOCATE(house)
↓
GROUP(COLUMN)
↓
SELECT(EXTREME, RIGHTMOST)
↓
COUNT(house, False)
↓
MATCH_CHOICE
```

---

# 32. MME 示例：属性 + anchor

问题：

```text
What color is the roof of the L-shaped building
in the harbor area above the picture?
```

推荐：

```text
REGION(top)
↓
LOCATE(harbor)
↓
LOCATE(building, shape=L-shaped)
↓
SELECT(INSIDE / NEAR harbor)
↓
ATTRIBUTE(color, part=roof)
↓
MATCH_CHOICE
```

---

# 33. XLRS 示例：给定 bbox 属性

问题形式：

```text
Determine the color of the object within the given reference bounding box.
Image resolution: ...
Bounding box: [...]
```

推荐：

```text
REGION_FROM_BBOX
↓
ATTRIBUTE(color)
↓
MATCH_CHOICE
```

---

# 34. XLRS 示例：双图 red-circle counting difference

```text
$image0                   $image1
   ↓                         ↓
FIND_MARKER(red_circle)    FIND_MARKER(red_circle)
   ↓                         ↓
COUNT(farm, False)         COUNT(farm, False)
      \                       /
       \                     /
              ABS_DIFF
                 ↓
            MATCH_CHOICE
```

这是典型 DAG merge。

---

# 35. XLRS 示例：Route Planning

问题：

```text
What is the shortest route from the largest circular roundabout
in the cluster of houses in the top right corner of the picture
to the triangular green pond above it?
```

推荐：

```text
REGION(top_right)
↓
LOCATE(roundabout)
↓
SELECT(RANK, largest)
↓
LOCATE(triangular green pond)
↓
BUILD_ROUTE_CONTEXT
↓
ROUTE_REASON
↓
MATCH_CHOICE
```

第一版不把每个候选路线里的 turn / fork / roundabout 再拆成微操作。

---

# 36. XLRS 示例：多区域 evidence reasoning

```text
In the middle on the right side, the lower left corner,
and on both sides of the highway in the farmland areas,
there are multiple lakes or ponds of varying sizes. Why does this occur?
```

推荐：

```text
                       IMAGE
              ┌─────────┼─────────┐
              ↓         ↓         ↓
REGION(mid_right) REGION(low_left) LOCATE(highway)
      ↓                ↓               ↓
LOCATE(pond)       LOCATE(pond)   SELECT(BOTH_SIDES)
                                      ↓
                                LOCATE(farmland)
                                      ↓
                                  LOCATE(pond)
              \          |          /
               \         |         /
                   VLM_REASON
                       ↓
                  MATCH_CHOICE
```

---

# 37. 训练数据格式

建议 teacher 生成样本：

```json
{
  "sample_id": "...",
  "input": {
    "question": "...",
    "question_type": "...",
    "choices": [...],
    "images": [...]
  },
  "target": {
    "version": "taskgraph-v1",
    "intent": "...",
    "nodes": [...],
    "final": {...}
  },
  "metadata": {
    "dataset": "MME_RealWorld_RS | XLRS_Bench",
    "source_category": "...",
    "difficulty": "simple | medium | complex",
    "generation_method": "teacher"
  }
}
```

真正训练 Planner 时，建议模型只生成：

```text
intent
nodes
final
```

question / choices / inputs 由系统直接注入，减少复制错误。

---

# 38. GraphQuality

```python
enum GraphQuality:
    VALID
    VALID_BUT_NON_MINIMAL
    SEMANTICALLY_AMBIGUOUS
    TYPE_ERROR
    LOGIC_ERROR
```

SFT 数据优先只使用 `VALID`，以及人工确认过的 `VALID_BUT_NON_MINIMAL`。

---

# 39. 自动 Validator 建议

```text
1. JSON parse
2. schema valid
3. all refs exist
4. graph acyclic
5. operator exists
6. input types compatible
7. params match operator schema
8. final source exists
9. answer_type compatible
10. COUNT only target + entire
11. no physical executor fields
12. bbox questions use REGION_FROM_BBOX unless有充分理由
13. marker questions use FIND_MARKER unless有充分理由
14. external relations are not hidden inside TargetSpec
15. MCQ deterministic outputs normally end with MATCH_CHOICE
```

---

# 40. 最小化原则

对于同一个问题可能存在多个等价 DAG。训练数据应偏好：

> **语义清楚、节点较少、每个节点职责单一、不过度拆分。**

简单问题不要过度分解；复杂 anchor 不得为了节点少而塞成超长 target string。

---

# 41. 物理 Executor 与逻辑 DAG 的边界

以下字段禁止出现在 Planner 生成图中：

```json
{
  "model": "LAE-DINO",
  "pred_score_thr": 0.05,
  "tile_size": 1024,
  "overlap": 0.15,
  "retriever": "RemoteCLIP",
  "route_model": "Qwen3-VL-4B"
}
```

这些属于：

```text
Capability Router
CountExecutor
LocatorExecutor
RouteExecutor
ModelPool
```

TaskGraph 只说明：

```text
做什么
依赖什么
结果传给谁
```

---

# 42. 推荐代码结构

```text
taskgraph/
├── schema/
│   ├── graph.py
│   ├── node.py
│   ├── targets.py
│   ├── runtime_types.py
│   ├── enums.py
│   └── operators/
│       ├── region.py
│       ├── locate.py
│       ├── select.py
│       ├── count.py
│       ├── attribute.py
│       ├── classify.py
│       ├── reasoning.py
│       └── route.py
│
├── validation/
│   ├── schema_validator.py
│   ├── type_checker.py
│   └── graph_validator.py
│
├── execution/
│   ├── executor.py
│   ├── registry.py
│   └── runtime_store.py
│
└── training/
    ├── serialize.py
    ├── teacher_generation.py
    └── quality_checks.py
```

---

# 43. 第一阶段实现建议

先不要接真实专家模型。

```text
Question
↓
Strong Teacher LLM
↓
TaskGraph JSON
↓
Schema Validator
↓
Type Checker
↓
Fake Executor / Trace Executor
↓
人工检查
```

优先采样八类：

```text
1. absolute-region count
2. nested relational count
3. ordinal / rank / extreme
4. part attribute
5. given bbox
6. visual marker
7. two-image branch + merge
8. route / multi-region reasoning
```

这些结构稳定后再冻结 `taskgraph-v1`。

---

# 44. 给强模型生成训练数据的 Prompt

以下提示词可直接放在本文档之后使用。

```text
你是一个“遥感视觉程序编译器（Remote-Sensing Visual Program Compiler）”。

你的任务不是回答问题本身，而是把输入的遥感问题编译为 TaskGraph v1 的逻辑 DAG，用于训练一个文本 Planner。

你必须严格遵循上文《TaskGraph v1：遥感复杂问题 DAG 中间表示定义与训练数据生成规范》。

====================
一、总原则
====================

1. 只描述“逻辑上需要做什么”，不要描述“具体模型怎么做”。

2. 禁止在输出中出现任何物理执行细节，例如：
   - LAE-DINO
   - CLIP
   - RemoteCLIP
   - VisRAG
   - Qwen
   - detector threshold
   - pred_score_thr
   - tile size
   - overlap
   - NMS
   - GPU / CPU placement
   - zoom depth
   - beam width

3. 图必须是 DAG（Directed Acyclic Graph，有向无环图）。

4. 每个 node 必须职责单一，使用规范中定义的 operator。

5. 不要直接回答题目答案。

6. 不要输出自然语言分析、解释、思维过程或 markdown。
   最终只输出合法 JSON。

====================
二、输入/参数边界
====================

GraphNode 使用：

{
  "id": "...",
  "op": "...",
  "inputs": {...},
  "params": {...},
  "output": "..."
}

inputs 表示 runtime 数据流。
params 只表示该逻辑操作自身的语义参数。

例如 COUNT：

COUNT 的语义参数严格只有：

{
  "target": TargetSpec,
  "entire": true/false
}

当前图像或 ROI 必须通过：

"inputs": {"image": "$..."}

传入。

绝对禁止给 COUNT 添加：

scope
relation
anchor
constraints
model
threshold
tile_size

====================
三、TargetSpec
====================

TargetSpec 可以包含：

category
以及目标自身的 intrinsic attributes，例如：

color
shape
size
state
pattern
has_part

允许：

red car
large ship
L-shaped building
ship with helipad

不允许把外部空间关系塞入 TargetSpec，例如：

vehicle left of forest
ship in bottom-left
building near harbor

这些必须拆成 REGION / LOCATE / SELECT 等节点。

====================
四、强制编译规则
====================

A. 问题包含绝对区域：
"in the bottom-left"
"upper right"
"far left"
优先生成 REGION。

B. 问题显式给 Bounding Box：
优先生成 REGION_FROM_BBOX。
不要重新 LOCATE 同一个目标。

C. 问题提到人工标记：
"red circle"
"red box"
"light blue border"
优先生成 FIND_MARKER。

D. 外部空间关系：
left of
right of
above
below
near
next to
inside
outside
both sides
使用 SELECT。

E. superlative / ranking：
largest
smallest
nearest
farthest
second largest
使用 SELECT(mode=RANK)。

F. ordinal：
first road from top to bottom
second area from left to right
使用 SELECT(mode=ORDINAL)。

G. extreme：
leftmost
rightmost
topmost
bottommost
使用 SELECT(mode=EXTREME)。

H. row / column / cluster：
使用 GROUP。

I. 计数：
最终目标实例数量由 COUNT 完成。

COUNT(target, entire=True)：
只有当当前传入视觉作用域仍是整图/大范围全域搜索时使用。

COUNT(target, entire=False)：
当上游已经通过 REGION / SELECT / marker / bbox 等明确缩小作用域时使用。

COUNT 永远是 exhaustive；
entire 只决定物理执行策略，而不是是否完整统计。

J. 属性：
"What color..."
"What shape..."
优先 ATTRIBUTE。
如果问 roof / wall / flag / body 等部件属性：
使用 ATTRIBUTE 的 part 参数。

K. 单标签分类：
使用 CLASSIFY。

L. 多标签 land-use：
使用 MULTILABEL_CLASSIFY。

M. motion：
给定目标/ROI 判断运动状态，用 MOTION。

N. 求两个实体相对位置：
用 RELATION。

O. 双图数量差：
两条 branch 分别 COUNT，再 ABS_DIFF。

P. route：
先定位 start 和 goal，
然后 BUILD_ROUTE_CONTEXT，
最后 ROUTE_REASON。

不要把路线中的每一个 turn / fork / intersection 都拆成 node，
除非未来 schema 明确增加相关 primitive。

Q. MCQ：
如果上游已经得到确定性值/标签，通常最后用 MATCH_CHOICE。

R. 高层因果、环境、功能、经济、社会、异常解释：
使用 VLM_REASON。
若问题中有明显局部 evidence，应先 REGION / LOCATE evidence，再汇入 VLM_REASON。

====================
五、最小化原则
====================

选择满足语义的最小清晰 DAG。

简单问题不要过度分解。

例如：
How many airplanes are there in the lower left area?

正确：
REGION(bottom_left)
→ COUNT(airplane, false)
→ MATCH_CHOICE

不要额外寻找 airport、runway 等无关中间对象。

但复杂嵌套关系必须显式拆解，不能为了少节点而把所有描述塞进一个 target string。

====================
六、类型要求
====================

必须遵守 operator 输入输出类型。

主要类型：
ImageRef
Region
RegionSet
Entity
EntitySet
ScalarInt
ScalarFloat
Boolean
Label
LabelSet
RouteContext
EvidenceSet
Answer

关键签名：

REGION:
ImageRef|Region -> Region

REGION_FROM_BBOX:
ImageRef -> Region

FIND_MARKER:
ImageRef|Region -> Region|RegionSet

LOCATE:
ImageRef|Region -> EntitySet

SELECT:
EntitySet|Region|RegionSet (+ optional reference)
-> Entity|EntitySet|Region|RegionSet

GROUP:
EntitySet -> EntitySet|RegionSet

COUNT:
ImageRef|Region -> ScalarInt

ATTRIBUTE:
Entity|EntitySet|Region -> Label|Boolean|ScalarFloat

CLASSIFY:
Region|Entity|ImageRef -> Label

MULTILABEL_CLASSIFY:
ImageRef|Region -> LabelSet

MOTION:
Region|Entity -> Boolean

RELATION:
subject + reference -> Label

ABS_DIFF:
ScalarInt + ScalarInt -> ScalarInt

BUILD_ROUTE_CONTEXT:
image + start + goal -> RouteContext

ROUTE_REASON:
RouteContext -> Answer|Label

MATCH_CHOICE:
Scalar/Boolean/Label/LabelSet/Answer -> Answer

====================
七、输出格式
====================

系统已经提供：
- question
- question_type
- choices
- inputs

你只输出：

{
  "intent": "...",
  "nodes": [...],
  "final": {
    "source": "$nX",
    "answer_type": "..."
  }
}

节点 id 使用：
n1
n2
n3
...

引用格式：
$image0
$image1
$n1
$n2

必须确保：
- 所有引用存在
- 无环
- final.source 存在
- 类型兼容
- operator 名称合法
- params 符合 schema

====================
八、遇到歧义
====================

如果存在两种合理 DAG：

1. 优先选择较少节点但语义仍清楚的方案；
2. 不凭空增加图中没有被问题要求的 anchor；
3. 不假设具体 detector 的能力；
4. 不为提高模型性能而利用数据集答案模式；
5. 不使用 ground-truth answer 反推程序结构；
6. 程序必须仅由 question、choices 和可见输入描述推导。

====================
九、当前样本
====================

Question:
{{QUESTION}}

Question type:
{{QUESTION_TYPE}}

Choices:
{{CHOICES}}

Inputs:
{{INPUTS}}

请生成 TaskGraph v1 JSON。
```

---

# 45. 使用建议

用于 teacher 数据生成时，可以把：

```text
本文档全文
+
上面的 Teacher Prompt
+
单个数据集样本
```

作为上下文。

更经济的方式：

1. system prompt 放本文档 schema 的精简版；
2. few-shot 放 8~20 个高质量典型 DAG；
3. user message 只传 question / choices / image keys；
4. teacher 生成 JSON；
5. Validator 自动过滤；
6. 对复杂样本人工抽检。

---

# 46. 一句话定义

> **TaskGraph v1 是一个面向高分辨率遥感问答的强类型逻辑中间表示：文本 Planner 只负责把自然语言问题编译成“先做什么、结果传给谁、最后如何汇聚”的 DAG；所有模型选择、切图、检索、阈值和资源调度则由独立物理 Executor 完成。**

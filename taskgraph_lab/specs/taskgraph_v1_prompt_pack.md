# TaskGraph v1 Prompt Pack
## 遥感问题 → TaskGraph 训练数据生成专用提示词

> 配套规范：`taskgraph_v1_schema_and_teacher_prompt.md`  
> 用途：将 MME RealWorld RS、XLRS-Bench 等遥感问题编译为 TaskGraph v1 训练标签。  
> DAG = Directed Acyclic Graph（有向无环图）  
> IR = Intermediate Representation（中间表示）  
> MCQ = Multiple-Choice Question（选择题）  
> VLM = Vision-Language Model（视觉语言模型）

---

# 1. 推荐调用结构

训练数据生成时建议固定成三层：

```text
System Prompt
    ↓
少量 Few-shot Examples（可选）
    ↓
User Prompt：单个待标注样本
```

不要每个样本都重复完整设计文档。

推荐：

```text
System Prompt：
    使用本文第 2 节的精简 schema

Few-shot：
    选择 4–8 个覆盖不同逻辑结构的高质量样本

User Prompt：
    只传 question / question_type / choices / inputs
```

Teacher 只生成：

```json
{
  "intent": "...",
  "nodes": [...],
  "final": {...}
}
```

不要让 Teacher 重复：

```text
question
choices
inputs
```

这些由系统保留。

---

# 2. System Prompt：TaskGraph v1 精简 Schema

下面这一段建议直接作为 Teacher 的 **System Prompt**。

```text
You are a Remote-Sensing Visual Program Compiler.

Your task is NOT to answer the remote-sensing question.
Your task is to compile the question into a valid TaskGraph v1 logical DAG for training a text Planner.

TaskGraph v1 is a typed logical intermediate representation.
The Planner specifies WHAT must be done, in what dependency order, and how intermediate results flow.
The physical Executor decides HOW each operation is implemented.

==================================================
1. OUTPUT CONTRACT
==================================================

Output JSON only.
Do not output markdown.
Do not output explanations.
Do not answer the original question.
Do not output hidden reasoning or chain-of-thought.

Output exactly:

{
  "intent": "<IntentLabel>",
  "nodes": [
    {
      "id": "n1",
      "op": "<OperatorName>",
      "inputs": {...},
      "params": {...},
      "output": "<readable_name>"
    }
  ],
  "final": {
    "source": "$nX",
    "answer_type": "<AnswerType>"
  }
}

Node ids must be n1, n2, n3, ... in dependency order.

References:
$image0, $image1, ...
$n1, $n2, ...

Every reference must exist.
The graph must be acyclic.
final.source must exist.

==================================================
2. SYSTEM-PROVIDED FIELDS
==================================================

The caller already provides:
- question
- question_type
- choices
- inputs

Do NOT copy them into your output.

==================================================
3. INTENT LABELS
==================================================

Choose one:

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
Do not use it as a substitute for an executable graph.

==================================================
4. RUNTIME DATA TYPES
==================================================

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

The output type of each node is determined by its operator.
Do not invent new runtime types.

==================================================
5. TARGET SPEC
==================================================

TargetSpec:

{
  "category": "<object/category>",
  "attributes": {
    "<intrinsic_attribute>": "<value>"
  }
}

Intrinsic attributes may include:
color
shape
size
state
pattern
has_part

Examples allowed:
red vehicle
large ship
L-shaped building
ship with helipad

External spatial relations are NOT TargetSpec attributes.

Do NOT encode:
vehicle left of forest
ship in bottom-left
building near harbor

Use REGION / LOCATE / SELECT for external spatial relations.

==================================================
6. LEGAL OPERATORS
==================================================

Only use these operators:

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
7. OPERATOR SIGNATURES
==================================================

REGION
Input:
  image: ImageRef | Region
Params:
  position
Output:
  Region

Allowed normalized positions include:
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

--------------------------------------------------

REGION_FROM_BBOX
Input:
  image: ImageRef
Params:
  bbox: [x1, y1, x2, y2]
  image_size: [width, height] | null
Output:
  Region

Use this when the question explicitly provides a reference bounding box.
Do not re-locate the same object.

--------------------------------------------------

FIND_MARKER
Input:
  image: ImageRef | Region
Params:
  marker:
    {
      "color": "<color|null>",
      "shape": "<shape>"
    }
Output:
  Region | RegionSet

Use for explicit visual markers such as:
red circle
red rectangle
colored border

--------------------------------------------------

LOCATE
Input:
  image: ImageRef | Region
Params:
  target: TargetSpec
Output:
  EntitySet

LOCATE means "find this entity/category in the current visual scope".
Do not specify which detector/retriever/model performs it.

--------------------------------------------------

SELECT
Inputs:
  candidates: EntitySet | Region | RegionSet
  reference: Entity | EntitySet | Region   [optional]
Params:
  one SelectSpec
Output:
  Entity | EntitySet | Region | RegionSet

Select modes:

A. RELATION
{
  "mode": "RELATION",
  "relation": "<SpatialRelation>"
}

Relations:
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

B. RANK
{
  "mode": "RANK",
  "criterion": "size|distance|...",
  "rank": <1-based integer>,
  "order": "ASCENDING|DESCENDING"
}

Use for:
largest
smallest
nearest
farthest
second largest

C. ORDINAL
{
  "mode": "ORDINAL",
  "index": <1-based integer>,
  "order": "TOP_TO_BOTTOM|BOTTOM_TO_TOP|LEFT_TO_RIGHT|RIGHT_TO_LEFT"
}

Use for:
first road from top to bottom
second area from left to right

D. EXTREME
{
  "mode": "EXTREME",
  "direction": "LEFTMOST|RIGHTMOST|TOPMOST|BOTTOMMOST"
}

E. SUBREGION
{
  "mode": "SUBREGION",
  "subregion": "LEFT_SIDE|RIGHT_SIDE|ABOVE|BELOW|INSIDE|OUTSIDE|BOTH_SIDES|AROUND"
}

SUBREGION creates a visual scope from a reference region/entity.

--------------------------------------------------

GROUP
Input:
  entities: EntitySet
Params:
  mode: "ROW|COLUMN|CLUSTER"
Output:
  EntitySet | RegionSet

Use only when the question explicitly refers to spatial grouping such as:
row
column
cluster

--------------------------------------------------

COUNT
Input:
  image: ImageRef | Region
Params MUST contain exactly:
  target: TargetSpec
  entire: boolean
Output:
  ScalarInt

COUNT must NOT receive:
scope
relation
anchor
constraints
model
threshold
tile_size
retriever
detector

COUNT is always exhaustive within its current input visual scope.

entire=true:
the current input is still the whole image or a broad global scope.

entire=false:
upstream DAG operations have already restricted the visual scope.

External spatial constraints must be resolved before COUNT.

--------------------------------------------------

ATTRIBUTE
Input:
  entity: Entity | EntitySet | Region
Params:
  attribute: "<attribute_name>"
  part: "<part_name|null>"
Output:
  Label | Boolean | ScalarFloat

Examples:
roof color
object color
shape
orientation

--------------------------------------------------

CLASSIFY
Input:
  input: Region | Entity | ImageRef
Params:
  label_space: [labels] | null
Output:
  Label

Use for single-label classification.

--------------------------------------------------

MULTILABEL_CLASSIFY
Input:
  input: ImageRef | Region
Params:
  label_space: [labels]
Output:
  LabelSet

Use when multiple labels can simultaneously be correct.

--------------------------------------------------

MOTION
Input:
  input: Region | Entity
Params:
  {}
Output:
  Boolean

--------------------------------------------------

RELATION
Inputs:
  subject: Entity | Region
  reference: Entity | Region
Params:
  {}
Output:
  Label

RELATION computes an unknown relation between two known objects.

Difference:
SELECT knows the relation and selects/filter objects.
RELATION knows the objects and computes the relation.

--------------------------------------------------

ABS_DIFF
Inputs:
  a: ScalarInt
  b: ScalarInt
Params:
  {}
Output:
  ScalarInt

Computes |a-b|.

--------------------------------------------------

VLM_REASON
Inputs:
  image: ImageRef | Region                [optional]
  evidence: one or more upstream refs     [optional]
Params:
  question: "$question"
  choices: "$choices" | null
Output:
  Answer | Label | Boolean | Text

Use only when high-level visual/semantic reasoning is genuinely required:
cause
function
environmental inference
social/economic interpretation
anomaly explanation
open-ended complex reasoning

Do not use VLM_REASON when deterministic operators can solve the problem.

--------------------------------------------------

BUILD_ROUTE_CONTEXT
Inputs:
  image: ImageRef | Region
  start: Entity | Region
  goal: Entity | Region
Params:
  {}
Output:
  RouteContext

It means:
construct a suitable large-context visual representation for route reasoning.

Do not specify crop margins, resize sizes, models, or road extraction details.

--------------------------------------------------

ROUTE_REASON
Input:
  context: RouteContext
Params:
  question: "$question"
  choices: "$choices"
Output:
  Answer | Label

For TaskGraph v1, do not decompose every turn, fork, roundabout, or intersection inside route options into graph nodes.

--------------------------------------------------

MATCH_CHOICE
Input:
  value: ScalarInt | ScalarFloat | Boolean | Label | LabelSet | Answer
Params:
  choices: "$choices"
Output:
  Answer

If upstream logic already produced a deterministic count/label/boolean, prefer MATCH_CHOICE instead of asking a language model to select the option again.

==================================================
8. IMPORTANT COMPILATION RULES
==================================================

A. Absolute image region:
"bottom-left", "upper right", "middle right"
→ REGION

B. Explicit bounding box:
→ REGION_FROM_BBOX
Do not re-locate the same object.

C. Explicit visual marker:
"red circle", "red rectangle", "colored border"
→ FIND_MARKER

D. External spatial constraints:
left of
right of
above
below
near
next to
inside
outside
both sides
→ SELECT

E. Superlative/rank:
largest
smallest
nearest
farthest
second largest
→ SELECT mode=RANK

F. Ordinal:
first/second/third ... from top to bottom / left to right
→ SELECT mode=ORDINAL

G. Extreme:
leftmost/rightmost/topmost/bottommost
→ SELECT mode=EXTREME

H. Rows/columns/clusters:
→ GROUP

I. Counting:
spatial restrictions must be resolved upstream.
Then COUNT(target, entire).

J. Intrinsic attributes:
red car
L-shaped building
large ship
may be represented inside TargetSpec.

K. Part attributes:
roof color
wall color
→ ATTRIBUTE with part.

L. Difference in counts between two images:
parallel branches
→ COUNT
→ ABS_DIFF

M. Route planning:
locate start
locate goal
→ BUILD_ROUTE_CONTEXT
→ ROUTE_REASON

N. Multi-region high-level reasoning:
construct relevant region/entity branches first when useful,
then merge evidence into VLM_REASON.

O. Multiple-choice:
if a deterministic value/label is produced,
normally finish with MATCH_CHOICE.

==================================================
9. PHYSICAL EXECUTION DETAILS ARE FORBIDDEN
==================================================

Never output physical implementation details such as:

LAE-DINO
CLIP
RemoteCLIP
GeoRSCLIP
VisRAG
Qwen
detector
retriever
NMS
pred_score_thr
score threshold
tile_size
overlap
multi-scale detector
zoom depth
beam width
answerability threshold
GPU
CPU
model loading/offloading

These belong to the physical Executor / Capability Router.

==================================================
10. MINIMALITY
==================================================

Generate the smallest clear DAG that preserves the full semantics.

Do not add unnecessary anchors.

Example:
"How many airplanes are in the lower-left area?"
should be:

REGION(bottom_left)
→ COUNT(airplane, false)

Do not additionally locate airport/runway unless the question requires them.

However, do not compress complex nested references into one giant TargetSpec.

==================================================
11. TYPE AND LOGIC RULES
==================================================

- Every ref must exist.
- Graph must be acyclic.
- Inputs must match operator signatures.
- COUNT params must be exactly target + entire.
- External spatial relations cannot be hidden inside TargetSpec.
- Explicit bbox should use REGION_FROM_BBOX.
- Explicit marker should use FIND_MARKER.
- Deterministic operations are preferred over VLM_REASON.
- Do not use the ground-truth answer to infer the graph.
- Do not exploit dataset-specific answer patterns.
- Infer the graph only from question, choices, and provided input metadata.

==================================================
12. FINAL ANSWER TYPE
==================================================

Use one of:

CHOICE_SINGLE
CHOICE_MULTI
INTEGER
BOOLEAN
LABEL
LABEL_SET
TEXT

CHOICE_MULTI is indeterminate multi-select: select every applicable original
option, and allow one or more selected option ids. CHOICE_SINGLE allows exactly
one selected option. The final Choice VLM prompt must state this explicitly.

Return JSON only.
```

---

# 3. 单样本 User Prompt

实际生成一个样本时，User Prompt 建议极短：

```text
Compile the following sample into TaskGraph v1.

Question:
{{QUESTION}}

Question type:
{{QUESTION_TYPE}}

Choices:
{{CHOICES_OR_NULL}}

Inputs:
{{INPUTS_JSON}}
```

示例：

```text
Compile the following sample into TaskGraph v1.

Question:
How many airplanes are there in the lower left area of this picture?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) 15", "(B) 20", "(C) 25", "(D) 30"]

Inputs:
{
  "image0": {
    "type": "image",
    "uri_or_key": "sample.png"
  }
}
```

---

# 4. Few-shot 示例 1：Absolute-region Count

### Input

```text
Question:
How many airplanes are there in the lower left area of this picture?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) 15", "(B) 20", "(C) 25", "(D) 30"]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "SIMPLE_COUNT",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "BOTTOM_LEFT"
      },
      "output": "bottom_left_region"
    },
    {
      "id": "n2",
      "op": "COUNT",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "airplane",
          "attributes": {}
        },
        "entire": false
      },
      "output": "airplane_count"
    },
    {
      "id": "n3",
      "op": "MATCH_CHOICE",
      "inputs": {
        "value": "$n2"
      },
      "params": {
        "choices": "$choices"
      },
      "output": "answer"
    }
  ],
  "final": {
    "source": "$n3",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

---

# 5. Few-shot 示例 2：Given BBox Attribute

### Input

```text
Question:
Determine the color of the object within the given reference bounding box.
Image resolution: 4096 x 4096.
Bounding box: [2559, 1895, 2610, 1928].

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) white", "(B) gray", "(C) red", "(D) blue"]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "ATTRIBUTE_QUERY",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION_FROM_BBOX",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "bbox": [2559, 1895, 2610, 1928],
        "image_size": [4096, 4096]
      },
      "output": "reference_region"
    },
    {
      "id": "n2",
      "op": "ATTRIBUTE",
      "inputs": {
        "entity": "$n1"
      },
      "params": {
        "attribute": "color",
        "part": null
      },
      "output": "object_color"
    },
    {
      "id": "n3",
      "op": "MATCH_CHOICE",
      "inputs": {
        "value": "$n2"
      },
      "params": {
        "choices": "$choices"
      },
      "output": "answer"
    }
  ],
  "final": {
    "source": "$n3",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

---

# 6. Few-shot 示例 3：Nested Relational Count

### Input

```text
Question:
How many swimming pools are there to the left of the forest
in the bottom-left corner?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) 1", "(B) 2", "(C) 3", "(D) 4"]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "RELATIONAL_COUNT",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "BOTTOM_LEFT"
      },
      "output": "bottom_left_region"
    },
    {
      "id": "n2",
      "op": "LOCATE",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "forest",
          "attributes": {}
        }
      },
      "output": "forest_candidates"
    },
    {
      "id": "n3",
      "op": "SELECT",
      "inputs": {
        "candidates": "$n1",
        "reference": "$n2"
      },
      "params": {
        "mode": "SUBREGION",
        "subregion": "LEFT_SIDE"
      },
      "output": "left_of_forest_region"
    },
    {
      "id": "n4",
      "op": "COUNT",
      "inputs": {
        "image": "$n3"
      },
      "params": {
        "target": {
          "category": "swimming_pool",
          "attributes": {}
        },
        "entire": false
      },
      "output": "pool_count"
    },
    {
      "id": "n5",
      "op": "MATCH_CHOICE",
      "inputs": {
        "value": "$n4"
      },
      "params": {
        "choices": "$choices"
      },
      "output": "answer"
    }
  ],
  "final": {
    "source": "$n5",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

---

# 7. Few-shot 示例 4：Two-image Marker + Count Difference

### Input

```text
Question:
How many differences are there in the number of farms
within the areas marked by red circles in the two images?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) 3", "(B) 4", "(C) 5", "(D) 6"]

Inputs:
{
  "image0": {"type": "image", "uri_or_key": "before.png"},
  "image1": {"type": "image", "uri_or_key": "after.png"}
}
```

### Target

```json
{
  "intent": "CHANGE_COUNT",
  "nodes": [
    {
      "id": "n1",
      "op": "FIND_MARKER",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "marker": {
          "color": "red",
          "shape": "circle"
        }
      },
      "output": "marked_region_before"
    },
    {
      "id": "n2",
      "op": "COUNT",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "farm",
          "attributes": {}
        },
        "entire": false
      },
      "output": "farm_count_before"
    },
    {
      "id": "n3",
      "op": "FIND_MARKER",
      "inputs": {
        "image": "$image1"
      },
      "params": {
        "marker": {
          "color": "red",
          "shape": "circle"
        }
      },
      "output": "marked_region_after"
    },
    {
      "id": "n4",
      "op": "COUNT",
      "inputs": {
        "image": "$n3"
      },
      "params": {
        "target": {
          "category": "farm",
          "attributes": {}
        },
        "entire": false
      },
      "output": "farm_count_after"
    },
    {
      "id": "n5",
      "op": "ABS_DIFF",
      "inputs": {
        "a": "$n2",
        "b": "$n4"
      },
      "params": {},
      "output": "count_difference"
    },
    {
      "id": "n6",
      "op": "MATCH_CHOICE",
      "inputs": {
        "value": "$n5"
      },
      "params": {
        "choices": "$choices"
      },
      "output": "answer"
    }
  ],
  "final": {
    "source": "$n6",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

---

# 8. Few-shot 示例 5：Ordinal + Group + Count

### Input

```text
Question:
How many houses are there in the rightmost column of the second
building area from top to bottom in the area on the right of the picture?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) 5", "(B) 6", "(C) 7", "(D) 8"]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "RELATIONAL_COUNT",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "RIGHT"
      },
      "output": "right_region"
    },
    {
      "id": "n2",
      "op": "LOCATE",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "building_area",
          "attributes": {}
        }
      },
      "output": "building_areas"
    },
    {
      "id": "n3",
      "op": "SELECT",
      "inputs": {
        "candidates": "$n2"
      },
      "params": {
        "mode": "ORDINAL",
        "index": 2,
        "order": "TOP_TO_BOTTOM"
      },
      "output": "second_building_area"
    },
    {
      "id": "n4",
      "op": "LOCATE",
      "inputs": {
        "image": "$n3"
      },
      "params": {
        "target": {
          "category": "house",
          "attributes": {}
        }
      },
      "output": "houses"
    },
    {
      "id": "n5",
      "op": "GROUP",
      "inputs": {
        "entities": "$n4"
      },
      "params": {
        "mode": "COLUMN"
      },
      "output": "house_columns"
    },
    {
      "id": "n6",
      "op": "SELECT",
      "inputs": {
        "candidates": "$n5"
      },
      "params": {
        "mode": "EXTREME",
        "direction": "RIGHTMOST"
      },
      "output": "rightmost_column"
    },
    {
      "id": "n7",
      "op": "COUNT",
      "inputs": {
        "image": "$n6"
      },
      "params": {
        "target": {
          "category": "house",
          "attributes": {}
        },
        "entire": false
      },
      "output": "house_count"
    },
    {
      "id": "n8",
      "op": "MATCH_CHOICE",
      "inputs": {
        "value": "$n7"
      },
      "params": {
        "choices": "$choices"
      },
      "output": "answer"
    }
  ],
  "final": {
    "source": "$n8",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

---

# 9. Few-shot 示例 6：Route Planning

### Input

```text
Question:
What is the shortest route from the largest circular roundabout
in the cluster of houses in the top right corner of the picture
to the triangular green pond above it?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
[
  "(A) ...",
  "(B) ...",
  "(C) ...",
  "(D) ..."
]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "ROUTE_PLANNING",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "TOP_RIGHT"
      },
      "output": "top_right_region"
    },
    {
      "id": "n2",
      "op": "LOCATE",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "roundabout",
          "attributes": {
            "shape": "circular"
          }
        }
      },
      "output": "roundabouts"
    },
    {
      "id": "n3",
      "op": "SELECT",
      "inputs": {
        "candidates": "$n2"
      },
      "params": {
        "mode": "RANK",
        "criterion": "size",
        "rank": 1,
        "order": "DESCENDING"
      },
      "output": "largest_roundabout"
    },
    {
      "id": "n4",
      "op": "LOCATE",
      "inputs": {
        "image": "$n1"
      },
      "params": {
        "target": {
          "category": "pond",
          "attributes": {
            "shape": "triangular",
            "color": "green"
          }
        }
      },
      "output": "pond_candidates"
    },
    {
      "id": "n5",
      "op": "SELECT",
      "inputs": {
        "candidates": "$n4",
        "reference": "$n3"
      },
      "params": {
        "mode": "RELATION",
        "relation": "ABOVE"
      },
      "output": "goal_pond"
    },
    {
      "id": "n6",
      "op": "BUILD_ROUTE_CONTEXT",
      "inputs": {
        "image": "$image0",
        "start": "$n3",
        "goal": "$n5"
      },
      "params": {},
      "output": "route_context"
    },
    {
      "id": "n7",
      "op": "ROUTE_REASON",
      "inputs": {
        "context": "$n6"
      },
      "params": {
        "question": "$question",
        "choices": "$choices"
      },
      "output": "route_answer"
    }
  ],
  "final": {
    "source": "$n7",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

这里不再额外 `MATCH_CHOICE`，因为 `ROUTE_REASON` 本身已经根据 choices 输出选择题答案。

---

# 10. Few-shot 示例 7：Multi-region Complex Reasoning

### Input

```text
Question:
In the middle on the right side, the lower left corner,
and on both sides of the highway in the farmland areas shown in the image,
there are multiple lakes or ponds of varying sizes. Why does this occur?

Question type:
MULTIPLE_CHOICE_SINGLE

Choices:
["(A) ...", "(B) ...", "(C) ...", "(D) ..."]

Inputs:
{"image0": {"type": "image", "uri_or_key": "sample.png"}}
```

### Target

```json
{
  "intent": "COMPLEX_REASONING",
  "nodes": [
    {
      "id": "n1",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "CENTER_RIGHT"
      },
      "output": "middle_right_region"
    },
    {
      "id": "n2",
      "op": "REGION",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "position": "BOTTOM_LEFT"
      },
      "output": "bottom_left_region"
    },
    {
      "id": "n3",
      "op": "LOCATE",
      "inputs": {
        "image": "$image0"
      },
      "params": {
        "target": {
          "category": "highway",
          "attributes": {}
        }
      },
      "output": "highway"
    },
    {
      "id": "n4",
      "op": "SELECT",
      "inputs": {
        "candidates": "$image0",
        "reference": "$n3"
      },
      "params": {
        "mode": "SUBREGION",
        "subregion": "BOTH_SIDES"
      },
      "output": "highway_side_regions"
    },
    {
      "id": "n5",
      "op": "VLM_REASON",
      "inputs": {
        "image": "$image0",
        "evidence": ["$n1", "$n2", "$n4"]
      },
      "params": {
        "question": "$question",
        "choices": "$choices"
      },
      "output": "reasoning_answer"
    }
  ],
  "final": {
    "source": "$n5",
    "answer_type": "CHOICE_SINGLE"
  }
}
```

这里没有强行增加不必要的 `LOCATE(pond)` / `LOCATE(farmland)`，因为问题的最终任务是因果解释；如果后续实验发现显式局部 evidence 有帮助，再允许更细的 evidence graph。

---

# 11. 推荐 Few-shot 组合

不建议每次把所有示例都塞进去。

### 基础 Teacher

使用：

```text
示例 1：absolute-region count
示例 2：bbox attribute
示例 4：two-image branch + merge
示例 6：route
```

### 复杂结构 Teacher

增加：

```text
示例 3：relation + subregion
示例 5：ordinal + group
示例 7：multi-region reasoning
```

总量控制在：

```text
4–7 examples
```

通常已经足够建立 schema 行为。

---

# 12. 批量训练数据生成 Prompt

如果一次请求让 Teacher 生成多个样本，建议不要让模型输出一个巨大自然语言回复，而是输出 JSONL 风格对象数组。

System Prompt 保持第 2 节不变。

User Prompt：

```text
Compile every sample below into TaskGraph v1.

Return a JSON array.
The i-th output object must correspond to the i-th input sample.
Do not skip samples.
Do not merge samples.
Do not answer any original question.

Samples:
{{SAMPLES_JSON}}
```

每个输入样本：

```json
{
  "sample_id": "xlrs_001979",
  "question": "...",
  "question_type": "MULTIPLE_CHOICE_SINGLE",
  "choices": ["...", "..."],
  "inputs": {
    "image0": {...},
    "image1": {...}
  }
}
```

建议 batch 不要过大。

对于复杂问题：

```text
8–16 samples / request
```

对于简单 bbox / counting：

```text
16–32 samples / request
```

更容易保持结构稳定。

---

# 13. Validator Repair Prompt

Teacher 输出经过自动 Validator 后，如果只存在 schema / type / reference 错误，不建议直接丢弃。

可进行一次 repair。

### System Prompt

继续使用第 2 节相同 System Prompt。

### User Prompt

```text
The previously generated TaskGraph is invalid.

Original sample:

Question:
{{QUESTION}}

Question type:
{{QUESTION_TYPE}}

Choices:
{{CHOICES}}

Inputs:
{{INPUTS}}

Invalid TaskGraph:
{{INVALID_GRAPH}}

Validator errors:
{{VALIDATOR_ERRORS}}

Repair the graph.

Requirements:
1. Preserve the original semantic plan whenever it is correct.
2. Fix only what is needed to satisfy TaskGraph v1.
3. Do not answer the original question.
4. Do not add physical execution details.
5. Return the complete repaired JSON object only.
```

只允许：

```text
1 次 repair
```

仍失败则进入人工 review / discard。

---

# 14. Semantic Review Prompt

Schema valid 不代表逻辑一定正确。

可以用另一个强模型做 semantic reviewer。

### System Prompt

```text
You are a TaskGraph semantic reviewer.

You are given:
1. a remote-sensing question,
2. choices and input metadata,
3. a TaskGraph v1 candidate.

Judge whether the graph faithfully represents the question semantics.

Do NOT solve the visual question.
Do NOT use the ground-truth answer.

Check:
- missing constraints
- invented constraints
- wrong target
- wrong anchor
- incorrect spatial scope
- incorrect ordering/ranking
- external relations hidden inside TargetSpec
- unnecessary VLM_REASON
- unnecessary nodes
- missing branch/merge logic
- COUNT entire flag misuse
- route start/goal interpretation

Return JSON only:

{
  "verdict": "VALID|VALID_BUT_NON_MINIMAL|SEMANTICALLY_AMBIGUOUS|LOGIC_ERROR",
  "issues": [
    {
      "node": "nX|null",
      "type": "...",
      "message": "short explanation"
    }
  ]
}
```

---

# 15. 训练集最终建议格式

Teacher + Validator + Reviewer 完成后：

```json
{
  "sample_id": "xlrs_001979",
  "input": {
    "question": "...",
    "question_type": "MULTIPLE_CHOICE_SINGLE",
    "choices": [...],
    "inputs": {...}
  },
  "target": {
    "intent": "CHANGE_COUNT",
    "nodes": [...],
    "final": {...}
  },
  "metadata": {
    "schema_version": "taskgraph-v1",
    "teacher_model": "...",
    "validator_passed": true,
    "repair_count": 0,
    "semantic_review": "VALID",
    "dataset": "XLRS_Bench",
    "source_category": "Counting/Counting with changing detection"
  }
}
```

Planner 的监督目标只需要：

```json
{
  "intent": "...",
  "nodes": [...],
  "final": {...}
}
```

---

# 16. 推荐数据生成流水线

```text
MME / XLRS raw QA
        ↓
normalize question_type / choices / inputs
        ↓
Teacher generation
        ↓
JSON/schema validation
        ↓
type check
        ↓
graph cycle/reference check
        ↓
invalid?
   ┌────┴────┐
  yes       no
   ↓         ↓
repair once  semantic review
   ↓         ↓
validate     quality label
       \     /
        \   /
      accepted graphs
           ↓
      human spot check
           ↓
       SFT dataset
```

SFT = Supervised Fine-Tuning（监督微调）。

---

# 17. 推荐生成策略

第一阶段不要追求一次生成全部 3080 条。

先构建一个覆盖结构的 seed set：

```text
absolute region count           30
entire-image count              20
nested relational count         40
ordinal/rank/extreme            30
bbox attribute/class/motion     30
marker region                   20
multi-image count/change        20
spatial relation                20
single/multi-label classify     20
route planning                  30
complex multi-region reasoning  30
```

约：

```text
290 samples
```

人工抽检并修改 schema 后，再冻结 TaskGraph v1。

---

# 18. 强模型调用参数建议

对于 TaskGraph generation：

```text
temperature: 0–0.2
top_p: low/default
JSON schema / structured output: enabled if available
max_output_tokens: 给复杂 DAG 留足空间
```

目标是：

```text
稳定
可执行
最小化
```

而不是生成多样性。

对于同一问题如需 uncertainty estimation，可以独立运行：

```text
3 次低温生成
```

比较 graph semantic consistency。

但正式训练标签最好只保留经过 Validator/Reviewer 的一个 canonical graph。

---

# 19. 不建议放入 System Prompt 的内容

下面这些虽然在完整设计文档中重要，但不需要反复喂给 Teacher：

```text
LAE threshold experiments
CLIP distillation
VisRAG optimization
multi-scale COUNT implementation
Answerability training
ModelPool
CPU/GPU resource scheduling
route 4B/2B experiments
具体代码目录
```

Teacher 只需要逻辑 schema。

这样可以显著缩短上下文并减少模型把物理实现误写进 DAG 的概率。

---

# 20. 与完整 Schema 文档的关系

完整文档：

```text
taskgraph_v1_schema_and_teacher_prompt.md
```

用途：

```text
人类设计参考
schema 冻结
代码实现
类型定义
系统说明
```

本 Prompt Pack：

```text
taskgraph_v1_prompt_pack.md
```

用途：

```text
Teacher system prompt
few-shot
单样本生成
batch generation
repair
semantic review
训练数据流水线
```

如果二者定义冲突：

> **以完整 TaskGraph v1 Schema 文档为准，并同步修改 Prompt Pack。**

---

# 21. 最简实际调用版本

如果只需要立即测试一个强模型，直接使用：

```text
System:
第 2 节 System Prompt

User:
Compile the following sample into TaskGraph v1.

Question:
{{QUESTION}}

Question type:
{{QUESTION_TYPE}}

Choices:
{{CHOICES}}

Inputs:
{{INPUTS}}
```

如果 zero-shot 不稳定，再在 System 与 User 之间加入：

```text
Few-shot 示例 1
Few-shot 示例 3
Few-shot 示例 4
Few-shot 示例 6
```

即可开始生成第一批训练数据。

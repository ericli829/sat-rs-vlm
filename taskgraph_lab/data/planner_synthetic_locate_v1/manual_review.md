# 人工审核样本

共 44 条；以下为完整问题、实际 Planner 输入、Gold DSL 和结构化生成规则。没有审核真实视觉答案。

所有样本实际使用的共同 system 消息（下方逐条完整列出 user 和 assistant）：
~~~text
Compile one remote-sensing question into taskgraph-v1.1-dsl-v1. Return only the
DSL program: no JSON, Markdown, explanation, or predicted answer. The input JSON
contains question, coarse question_type, original choices, and image inputs.
Never copy choices into the program or use a ground-truth answer.

Use only $image0, $image1, ... and sequential n1, n2, ... references. Emit an
optional INTENT(enum), one contributing node per line, then exactly one
FINAL(source-or-list,answer_type) or
FINAL_QUESTION(source-or-list,answer_type,"static residual question"). Final
sources must be $nX references, never raw images.

For a bounding-box localization request, use FINAL($nX,BBOX) with exactly one
single-target visual source and omit a residual final question. Prefer the
shortest faithful path: LOCATE for category/intrinsic attributes, REGION then
LOCATE for an explicit image area, or a singleton SELECT for an extreme,
ordinal, or external semantic condition. BBOX is not a choice answer. Its
source must be LOCATE, REGION/REGION_BBOX, or SELECT_RANK/ORD/EXTREME/SUBREGION
or SELECT_SEM(...,SINGLE). Do not pass GROUP, SELECT_REL, SELECT_SEM MULTI, or
multiple sources directly to BBOX. Runtime must reject zero/multiple targets
and unresolved evidence; never predict bbox coordinate values in the program.

Allowed node surfaces are REGION, REGION_BBOX, FIND_MARKER, LOCATE with T(...),
SELECT_REL, SELECT_SEM, SELECT_RANK, SELECT_ORD, SELECT_EXTREME, SELECT_SUBREGION, GROUP,
COUNT, COUNT_IMAGE, COUNT_ENTITIES, ATTRIBUTE, CLASSIFY, MULTILABEL_CLASSIFY,
MOTION, RELATION, ABS_DIFF, VLM_REASON, BUILD_ROUTE_CONTEXT, and ROUTE_REASON.
Follow the canonical argument roles learned from the examples; use JSON strings,
true/false/null, enum symbols, lists, and T("category",attribute=value).

Prefer dedicated deterministic operators over VLM_REASON. COUNT is mandatory
and authoritative for counting. Relational counting must filter with SELECT and
feed that result to COUNT_ENTITIES or COUNT_IMAGE. Preserve explicit category
alternatives, attributes, positions, markers, bounding boxes, relations, ranks,
ordinals, groups, and route endpoints. SELECT_RANK criterion must be exactly
the quoted string "bbox_area" or "score": rank by the bounding-box area
(largest/smallest object) or by detector score only. Use FINAL_QUESTION only
when visual or semantic evidence still needs a final judgment; never include
runtime values. When a complex
question needs both a COUNT result and visual evidence, keep COUNT upstream and feed the mixed
evidence into one VLM_REASON node; use only that node as the final source.

CHOICE_MULTI is indeterminate multi-select: the downstream Choice VLM selects
every applicable option and may return one or more ids. CHOICE_SINGLE permits
exactly one id. Infer this from question semantics, never legacy question_type
cardinality labels.

For a natural-language condition over existing candidates, use
SELECT_SEM(candidates,reference-or-null,"complete natural language query",SINGLE|MULTI[,scope]).
The optional fifth argument is an ImageRef or preceding Region reference and
maps only to SELECT.inputs.scope. Omit it when no explicit scope is needed;
do not insert a default or null. Scope and reference are separate input roles.
The string belongs to SELECT.params.query, never to inputs. Preserve the full
condition (including surrounding trees, grass, roads, shapes, and spatial context).
Do not create extra LOCATE nodes for background nouns solely to express this
condition. LOCATE produces the category candidates with their intrinsic TargetSpec
attributes; SELECT consumes the filtering condition. SINGLE may be ambiguous or
unresolved; MULTI may select no candidates. Geometry REL/RANK/ORD/EXTREME and
SUBREGION remain available and retain their meanings. Plan all logical steps
explicitly; the executor chooses only physical LOCATE backends from their uses.
~~~

## synthetic-locate-v1-00000

Provide a bounding box for the railway station shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Provide a bounding box for the railway station shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("railway station"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 6,
  "category": "railway station",
  "context_index": 4,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action6",
  "template_id": "train:category:a6:c4"
}
~~~

## synthetic-locate-v1-00001

Specify the bounds of the water tower with trees immediately to its right shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Specify the bounds of the water tower with trees immediately to its right shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("water tower"))
n2=SELECT_SEM($n1,null,"the water tower with trees immediately to its right",SINGLE)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 9,
  "category": "water tower",
  "condition": "with trees immediately to its right",
  "context_index": 4,
  "family": "semantic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action9",
  "template_id": "train:semantic:a9:c4"
}
~~~

## synthetic-locate-v1-00002

Locate the wind turbine in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Locate the wind turbine in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("wind turbine"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "wind turbine",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action0",
  "template_id": "train:category:a0:c2"
}
~~~

## synthetic-locate-v1-00003

Mark the location of the train in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Mark the location of the train in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("train"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 8,
  "category": "train",
  "context_index": 1,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action8",
  "template_id": "train:category:a8:c1"
}
~~~

## synthetic-locate-v1-00004

Which part contains the hangar in the displayed view?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Which part contains the hangar in the displayed view?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("hangar"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "hangar",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action2",
  "template_id": "validation:category:a2:c2"
}
~~~

## synthetic-locate-v1-00006

Provide a bounding box for the oil tank in the lower-left region in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Provide a bounding box for the oil tank in the lower-left region in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=REGION($image0,BOTTOM_LEFT)
n2=LOCATE($n1,T("oil tank"))
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 6,
  "category": "oil tank",
  "context_index": 1,
  "family": "region",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "region": "BOTTOM_LEFT",
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action6",
  "template_id": "train:region:a6:c1"
}
~~~

## synthetic-locate-v1-00007

Locate the fishing boat in the right half in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Locate the fishing boat in the right half in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=REGION($image0,RIGHT)
n2=LOCATE($n1,T("fishing boat"))
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "fishing boat",
  "context_index": 1,
  "family": "region",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "region": "RIGHT",
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action0",
  "template_id": "train:region:a0:c1"
}
~~~

## synthetic-locate-v1-00010

Indicate the position of the cargo ship with containers in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Indicate the position of the cargo ship with containers in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("cargo ship",has_part="containers"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 4,
  "category": "cargo ship",
  "context_index": 2,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "has_part": "containers"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "cargo ship with containers",
  "template_family": "train:action4",
  "template_id": "train:intrinsic:a4:c2"
}
~~~

## synthetic-locate-v1-00012

Which part contains the residential building in this aerial scene?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Which part contains the residential building in this aerial scene?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("residential building"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "residential building",
  "context_index": 1,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action2",
  "template_id": "validation:category:a2:c1"
}
~~~

## synthetic-locate-v1-00014

Provide a bounding box for the red container in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Provide a bounding box for the red container in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("container",color="red"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 6,
  "category": "container",
  "context_index": 2,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "color": "red"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "red container",
  "template_family": "train:action6",
  "template_id": "train:intrinsic:a6:c2"
}
~~~

## synthetic-locate-v1-00015

Indicate the position of the green sports field shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Indicate the position of the green sports field shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("sports field",color="green"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 4,
  "category": "sports field",
  "context_index": 4,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "color": "green"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "green sports field",
  "template_family": "train:action4",
  "template_id": "train:intrinsic:a4:c4"
}
~~~

## synthetic-locate-v1-00016

Mark the location of the white airplane shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Mark the location of the white airplane shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("airplane",color="white"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 8,
  "category": "airplane",
  "context_index": 4,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "color": "white"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "white airplane",
  "template_family": "train:action8",
  "template_id": "train:intrinsic:a8:c4"
}
~~~

## synthetic-locate-v1-00017

Which part contains the basketball court in the displayed view?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Which part contains the basketball court in the displayed view?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("basketball court"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "basketball court",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action2",
  "template_id": "validation:category:a2:c2"
}
~~~

## synthetic-locate-v1-00018

Locate the stadium in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Locate the stadium in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("stadium"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "stadium",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action0",
  "template_id": "train:category:a0:c2"
}
~~~

## synthetic-locate-v1-00019

Indicate the position of the topmost bus in the image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Indicate the position of the topmost bus in the image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("bus"))
n2=SELECT_EXTREME($n1,null,TOPMOST)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 4,
  "category": "bus",
  "context_index": 0,
  "extreme": "TOPMOST",
  "family": "extreme",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action4",
  "template_id": "train:extreme:a4:c0"
}
~~~

## synthetic-locate-v1-00020

Provide a bounding box for the vehicle beside a paved road in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Provide a bounding box for the vehicle beside a paved road in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("vehicle"))
n2=SELECT_SEM($n1,null,"the vehicle beside a paved road",SINGLE)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 6,
  "category": "vehicle",
  "condition": "beside a paved road",
  "context_index": 2,
  "family": "semantic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action6",
  "template_id": "train:semantic:a6:c2"
}
~~~

## synthetic-locate-v1-00021

Mark the location of the antenna in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Mark the location of the antenna in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("antenna"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 8,
  "category": "antenna",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action8",
  "template_id": "train:category:a8:c2"
}
~~~

## synthetic-locate-v1-00026

Point out the square helipad in the provided scene.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Point out the square helipad in the provided scene.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("helipad",shape="square"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 7,
  "category": "helipad",
  "context_index": 3,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "shape": "square"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "square helipad",
  "template_family": "train:action7",
  "template_id": "train:intrinsic:a7:c3"
}
~~~

## synthetic-locate-v1-00027

What are the bounds of the yacht in the supplied image?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"What are the bounds of the yacht in the supplied image?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("yacht"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 1,
  "category": "yacht",
  "context_index": 3,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action1",
  "template_id": "validation:category:a1:c3"
}
~~~

## synthetic-locate-v1-00030

Mark the location of the tall chimney in the image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Mark the location of the tall chimney in the image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("chimney",size="tall"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 8,
  "category": "chimney",
  "context_index": 0,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "size": "tall"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "tall chimney",
  "template_family": "train:action8",
  "template_id": "train:intrinsic:a8:c0"
}
~~~

## synthetic-locate-v1-00031

Where is the bottommost warehouse in the satellite photograph?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Where is the bottommost warehouse in the satellite photograph?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("warehouse"))
n2=SELECT_EXTREME($n1,null,BOTTOMMOST)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "warehouse",
  "context_index": 0,
  "extreme": "BOTTOMMOST",
  "family": "extreme",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action0",
  "template_id": "validation:extreme:a0:c0"
}
~~~

## synthetic-locate-v1-00032

Identify the football field in the image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Identify the football field in the image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("football field"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "football field",
  "context_index": 0,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action2",
  "template_id": "train:category:a2:c0"
}
~~~

## synthetic-locate-v1-00034

Give the bounding box of the car shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Give the bounding box of the car shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("car"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 3,
  "category": "car",
  "context_index": 4,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action3",
  "template_id": "train:category:a3:c4"
}
~~~

## synthetic-locate-v1-00042

Identify the greenhouse with trees immediately to its right in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Identify the greenhouse with trees immediately to its right in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("greenhouse"))
n2=SELECT_SEM($n1,null,"the greenhouse with trees immediately to its right",SINGLE)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "greenhouse",
  "condition": "with trees immediately to its right",
  "context_index": 2,
  "family": "semantic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action2",
  "template_id": "train:semantic:a2:c2"
}
~~~

## synthetic-locate-v1-00043

Which part contains the industrial building in the satellite photograph?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Which part contains the industrial building in the satellite photograph?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("industrial building"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "industrial building",
  "context_index": 0,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action2",
  "template_id": "validation:category:a2:c0"
}
~~~

## synthetic-locate-v1-00046

Locate the large building in the image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Locate the large building in the image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("building",size="large"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "building",
  "context_index": 0,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "size": "large"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "large building",
  "template_family": "train:action0",
  "template_id": "train:intrinsic:a0:c0"
}
~~~

## synthetic-locate-v1-00048

Point out the rectangular swimming pool shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Point out the rectangular swimming pool shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("swimming pool",shape="rectangular"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 7,
  "category": "swimming pool",
  "context_index": 4,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "shape": "rectangular"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "rectangular swimming pool",
  "template_family": "train:action7",
  "template_id": "train:intrinsic:a7:c4"
}
~~~

## synthetic-locate-v1-00050

Specify the bounds of the ship in the provided scene.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Specify the bounds of the ship in the provided scene.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("ship"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 9,
  "category": "ship",
  "context_index": 3,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action9",
  "template_id": "train:category:a9:c3"
}
~~~

## synthetic-locate-v1-00055

Indicate the position of the tennis court in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Indicate the position of the tennis court in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("tennis court"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 4,
  "category": "tennis court",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action4",
  "template_id": "train:category:a4:c2"
}
~~~

## synthetic-locate-v1-00058

Find the bridge in the provided scene.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Find the bridge in the provided scene.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("bridge"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 1,
  "category": "bridge",
  "context_index": 3,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action1",
  "template_id": "train:category:a1:c3"
}
~~~

## synthetic-locate-v1-00061

Mark the location of the silo in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Mark the location of the silo in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("silo"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 8,
  "category": "silo",
  "context_index": 1,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action8",
  "template_id": "train:category:a8:c1"
}
~~~

## synthetic-locate-v1-00063

Indicate the position of the large parking area shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Indicate the position of the large parking area shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("parking area",size="large"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 4,
  "category": "parking area",
  "context_index": 4,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "size": "large"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "large parking area",
  "template_family": "train:action4",
  "template_id": "train:intrinsic:a4:c4"
}
~~~

## synthetic-locate-v1-00072

Determine where the dock in the upper half is in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Determine where the dock in the upper half is in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=REGION($image0,TOP)
n2=LOCATE($n1,T("dock"))
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 5,
  "category": "dock",
  "context_index": 1,
  "family": "region",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "region": "TOP",
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action5",
  "template_id": "train:region:a5:c1"
}
~~~

## synthetic-locate-v1-00081

Give the bounding box of the tower in the overhead view.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Give the bounding box of the tower in the overhead view.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("tower"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 3,
  "category": "tower",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action3",
  "template_id": "train:category:a3:c2"
}
~~~

## synthetic-locate-v1-00089

What are the bounds of the runway in the displayed view?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"What are the bounds of the runway in the displayed view?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("runway"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 1,
  "category": "runway",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action1",
  "template_id": "validation:category:a1:c2"
}
~~~

## synthetic-locate-v1-00119

Identify the truck with a trailer in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Identify the truck with a trailer in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("truck",has_part="trailer"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "truck",
  "context_index": 1,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "has_part": "trailer"
  },
  "seed": 20261010,
  "split": "train",
  "target_phrase": "truck with a trailer",
  "template_family": "train:action2",
  "template_id": "train:intrinsic:a2:c1"
}
~~~

## synthetic-locate-v1-00163

What are the bounds of the solar panel in the displayed view?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"What are the bounds of the solar panel in the displayed view?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("solar panel"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 1,
  "category": "solar panel",
  "context_index": 2,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action1",
  "template_id": "validation:category:a1:c2"
}
~~~

## synthetic-locate-v1-00177

Identify the storage tank shown here.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Identify the storage tank shown here.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("storage tank"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "storage tank",
  "context_index": 4,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action2",
  "template_id": "train:category:a2:c4"
}
~~~

## synthetic-locate-v1-00214

Where is the roundabout visible here?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Where is the roundabout visible here?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("roundabout"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "roundabout",
  "context_index": 4,
  "family": "category",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action0",
  "template_id": "validation:category:a0:c4"
}
~~~

## synthetic-locate-v1-00374

Which part contains the tall transmission tower in the supplied image?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Which part contains the tall transmission tower in the supplied image?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("transmission tower",size="tall"))
FINAL($n1,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 2,
  "category": "transmission tower",
  "context_index": 3,
  "family": "intrinsic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {
    "size": "tall"
  },
  "seed": 20261010,
  "split": "validation",
  "target_phrase": "tall transmission tower",
  "template_family": "validation:action2",
  "template_id": "validation:intrinsic:a2:c3"
}
~~~

## synthetic-locate-v1-00013

Point out the fourth railway station from right to left in this satellite image.

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Point out the fourth railway station from right to left in this satellite image.\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("railway station"))
n2=SELECT_ORD($n1,null,4,RIGHT_TO_LEFT)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 7,
  "category": "railway station",
  "context_index": 1,
  "family": "ordinal",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "order": "RIGHT_TO_LEFT",
  "ordinal": 4,
  "seed": 20261010,
  "split": "train",
  "template_family": "train:action7",
  "template_id": "train:ordinal:a7:c1"
}
~~~

## synthetic-locate-v1-00029

Where is the hangar in the top-center region in the supplied image?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Where is the hangar in the top-center region in the supplied image?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=REGION($image0,TOP_CENTER)
n2=LOCATE($n1,T("hangar"))
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "hangar",
  "context_index": 3,
  "family": "region",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "region": "TOP_CENTER",
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action0",
  "template_id": "validation:region:a0:c3"
}
~~~

## synthetic-locate-v1-00176

Where is the third warehouse from bottom to top in the satellite photograph?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"Where is the third warehouse from bottom to top in the satellite photograph?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("warehouse"))
n2=SELECT_ORD($n1,null,3,BOTTOM_TO_TOP)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 0,
  "category": "warehouse",
  "context_index": 0,
  "family": "ordinal",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "order": "BOTTOM_TO_TOP",
  "ordinal": 3,
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action0",
  "template_id": "validation:ordinal:a0:c0"
}
~~~

## synthetic-locate-v1-00038

What are the bounds of the hangar with an apron above it visible here?

Planner 输入：
~~~json
{"choices":null,"inputs":{"image0":{"type":"image","uri_or_key":"synthetic\\image0.png"}},"question":"What are the bounds of the hangar with an apron above it visible here?\nResolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.","question_type":"FREE_FORM"}
~~~

Gold DSL：
~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("hangar"))
n2=SELECT_SEM($n1,null,"the hangar with an apron above it",SINGLE)
FINAL($n2,BBOX)
~~~

生成规则：
~~~json
{
  "action_index": 1,
  "category": "hangar",
  "condition": "with an apron above it",
  "context_index": 4,
  "family": "semantic",
  "generator_version": "planner-synthetic-locate-v1",
  "intended_output_contract": "bbox",
  "intrinsic_attributes": {},
  "seed": 20261010,
  "split": "validation",
  "template_family": "validation:action1",
  "template_id": "validation:semantic:a1:c4"
}
~~~

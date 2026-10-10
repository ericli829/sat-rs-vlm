# LOCATE / SELECT after rendered prompt capture

Actual InputComposer and Qwen wrappers; synthetic visuals and mock inference. This is not an accuracy experiment.

## LOCATE_SINGLE white square building -> ATTRIBUTE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: LOCATE_SCOPE

Question:
Task: LOCATE_SINGLE. Localize one target object in this remote-sensing visual.
Target category: building
Intrinsic attributes: {"color": "white", "shape": "square"}
Actual supplied visual size: 640 x 480 pixels.
Search scope: the supplied image or crop, and nothing outside it.
Use ONLY the target definition supplied to this LOCATE node. Do not solve the overall question, use downstream SELECT conditions, or invent additional landmarks.
If one well-supported matching object is identifiable, return exactly {"status":"OK","bbox":[x1,y1,x2,y2]}.
If none is visible, return exactly {"status":"NONE"}.
If several plausible targets cannot be distinguished, return exactly {"status":"AMBIGUOUS"}.
If evidence is insufficient, return exactly {"status":"UNRESOLVED"}.
The bbox uses pixel XYXY of the ACTUAL supplied visual, top-left origin. No Markdown, prose, or extra fields.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

## LOCATE detector -> singleton refinement

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.9; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.8; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.7; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

Question:
You are verifying which supplied candidate matches the referring expression. Do not answer the original multiple-choice question. Do not search outside the supplied candidates. Select exactly one candidate only if the evidence supports it.

Original referring question: Find the building in the lower right
Target specification: category=building; attributes={'color': 'white', 'shape': 'square'}

Options:
Candidate A: building_1
Candidate B: building_2
Candidate C: building_3

Analyze the visual evidence, question, and all candidate options carefully. Reason through the problem before making the final decision. The final option will be selected in a separate constrained step.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nFinal choice:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "A",
    "B",
    "C"
  ],
  "allowed_outputs": null
}
```

## LOCATE empty -> 3x3 fallback

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C/D/E/F/G/H/I

[CANDIDATES]
type: RegionSet
boxes_xyxy_global:
- [0.0, 0.0, 200.0, 150.0]
- [200.0, 0.0, 400.0, 150.0]
- [400.0, 0.0, 600.0, 150.0]
- [0.0, 150.0, 200.0, 300.0]
- [200.0, 150.0, 400.0, 300.0]
- [400.0, 150.0, 600.0, 300.0]
- [0.0, 300.0, 200.0, 450.0]
- [200.0, 300.0, 400.0, 450.0]
- [400.0, 300.0, 600.0, 450.0]

Question:
Select one to three grid cells that most likely contain the target. Use only the marked 3x3 overview and do not answer the original question.

Original referring expression: Find white square building
Target specification: category=building; attributes={'color': 'white', 'shape': 'square'}

Options:
Grid A: top-left
Grid B: top-center
Grid C: top-right
Grid D: middle-left
Grid E: center
Grid F: middle-right
Grid G: bottom-left
Grid H: bottom-center
Grid I: bottom-right

Analyze the visual evidence, question, and all candidate options carefully. Reason through the problem before making the final decision. The final option will be selected in a separate constrained step.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "Verify the following option independently.\n\nCandidate option A: Grid A: top-left\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option B: Grid B: top-center\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option C: Grid C: top-right\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option D: Grid D: middle-left\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option E: Grid E: center\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option F: Grid F: middle-right\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option G: Grid G: bottom-left\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option H: Grid H: bottom-center\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option I: Grid I: bottom-right\nIs this option a correct answer to the original question?\nAnswer YES or NO:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "YES",
    "NO"
  ],
  "allowed_outputs": null
}
```

## SELECT_REL NEAR SINGLE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C
REFERENCE: REF

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.9; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.8; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.7; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[REFERENCE]
type: Region
bbox_xyxy_global: [280.0, 230.0, 310.0, 260.0]

Question:
Analyze which candidate objects satisfy relation NEAR relative to the marked reference. Use candidate labels only as visual references during reasoning. A separate constrained verification step will determine the final selection.

Analyze which candidate object or objects satisfy the requested relation. Use candidate labels only as visual references during reasoning. A separate constrained step will determine the final selection.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "Verify the following option independently.\n\nCandidate option A: Candidate A: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option B: Candidate B: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option C: Candidate C: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "YES",
    "NO"
  ],
  "allowed_outputs": null
}
```

## SELECT_REL NEAR MULTI

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C
REFERENCE: REF

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.9; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.8; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.7; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[REFERENCE]
type: Region
bbox_xyxy_global: [280.0, 230.0, 310.0, 260.0]

Question:
Analyze which candidate objects satisfy relation NEAR relative to the marked reference. Use candidate labels only as visual references during reasoning. A separate constrained verification step will determine the final selection.

Analyze which candidate object or objects satisfy the requested relation. Use candidate labels only as visual references during reasoning. A separate constrained step will determine the final selection.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "Verify the following option independently.\n\nCandidate option A: Candidate A: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option B: Candidate B: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:",
  "Verify the following option independently.\n\nCandidate option C: Candidate C: building\nIs this option a correct answer to the original question?\nAnswer YES or NO:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "YES",
    "NO"
  ],
  "allowed_outputs": null
}
```

## Legacy final candidate fallback (external/old entrypoint)

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.9; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.8; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.7; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

Question:
Use only the supplied bounded visual candidates to resolve the remaining referring expression and answer the residual question. Do not search outside the supplied candidates and do not infer a candidate that is not shown.

Original question: What color is the selected building?
The candidate resolver could not establish a reliable singleton. Choose the answer option supported by the supplied candidates.

Options:
A white
B red

Analyze the visual evidence, question, and all candidate options carefully. Reason through the problem before making the final decision. The final option will be selected in a separate constrained step.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nFinal choice:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "A",
    "B"
  ],
  "allowed_outputs": null
}
```

## Retained whole-image final fallback wrapper (legacy graph evidence failure)

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: RESULT_1

Question:
What color is the selected building?

Options:
A white
B red

Analyze the visual evidence, question, and all candidate options carefully. Reason through the problem before making the final decision. The final option will be selected in a separate constrained step.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nFinal choice:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "A",
    "B"
  ],
  "allowed_outputs": null
}
```

## SELECT_REL token-constrained fallback when KV scorer is unavailable

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C
REFERENCE: REF

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.9; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.8; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.7; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[REFERENCE]
type: Region
bbox_xyxy_global: [280.0, 230.0, 310.0, 260.0]

Question:
Analyze which candidate objects satisfy relation NEAR relative to the marked reference. Use candidate labels only as visual references during reasoning. A separate constrained verification step will determine the final selection.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": [
    "NONE",
    "A",
    "B",
    "C",
    "A,B",
    "A,C",
    "B,C",
    "A,B,C"
  ]
}
```

## Legacy final choice free-generation fallback (disabled by default)

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: RESULT_1

Question:
What color is the selected building?

Options:
A white
B red<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

## 01 VLM white square -> ATTRIBUTE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: LOCATE_SCOPE

Question:
Task: LOCATE_SINGLE. Localize one target object in this remote-sensing visual.
Target category: building
Intrinsic attributes: {"color": "white", "shape": "square"}
Actual supplied visual size: 640 x 480 pixels.
Search scope: the supplied image or crop, and nothing outside it.
Use ONLY the target definition supplied to this LOCATE node. Do not solve the overall question, use downstream SELECT conditions, or invent additional landmarks.
If one well-supported matching object is identifiable, return exactly {"status":"OK","bbox":[x1,y1,x2,y2]}.
If none is visible, return exactly {"status":"NONE"}.
If several plausible targets cannot be distinguished, return exactly {"status":"AMBIGUOUS"}.
If evidence is insufficient, return exactly {"status":"UNRESOLVED"}.
The bbox uses pixel XYXY of the ACTUAL supplied visual, top-left origin. No Markdown, prose, or extra fields.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: ENTITY

Structured results:
[ENTITY]
type: EntitySet
entities:
- index: 0; label: building; score: None; bbox_xyxy_global: [100.0, 60.0, 200.0, 140.0]

Question:
Determine the color of the already selected target. The upstream target selection is authoritative; do not localize a replacement.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

## 02 DINO 20 buildings -> EXTREME TOPMOST -> ATTRIBUTE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: ENTITY

Structured results:
[ENTITY]
type: Entity
label: building
score: 0.8
bbox_xyxy_global: [100.0, 5.0, 120.0, 25.0]

Question:
Determine the color of the already selected target. The upstream target selection is authoritative; do not localize a replacement.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

## 03 DINO -> SELECT_SEM SINGLE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: building; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: building; score: 0.98; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: building; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[SCOPE]
type: Region
bbox_xyxy_global: [0.0, 0.0, 640.0, 480.0]

Question:
Task: SELECT_SEM_SINGLE
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
the building with trees to its right and grass below

Choose exactly one marked candidate only when it uniquely satisfies the complete query. Choose NONE if none matches, AMBIGUOUS if several match, UNRESOLVED if evidence is insufficient.

Reason about all marked candidates and the full query. A finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nSELECT_SEM_SINGLE decision:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "A",
    "B",
    "C",
    "NONE",
    "AMBIGUOUS",
    "UNRESOLVED"
  ],
  "allowed_outputs": null
}
```

## 04 DINO red cars -> SELECT_SEM MULTI -> COUNT

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: car; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: car; score: 0.98; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: car; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

Question:
Task: SELECT_SEM_MULTI
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
Match the category and ALL intrinsic attributes of this LOCATE TargetSpec: {"attributes": {"color": "red"}, "category": "car"}

Evaluate every marked candidate against the complete query. For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. Empty sets are valid; uncertainty is not a negative.

Verify ONLY candidate A against the complete query. Reason from the supplied context; a three-state finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nCandidate A semantic verdict:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "MATCH",
    "NO_MATCH",
    "UNCERTAIN"
  ],
  "allowed_outputs": null
}
```

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: car; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: car; score: 0.98; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: car; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

Question:
Task: SELECT_SEM_MULTI
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
Match the category and ALL intrinsic attributes of this LOCATE TargetSpec: {"attributes": {"color": "red"}, "category": "car"}

Evaluate every marked candidate against the complete query. For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. Empty sets are valid; uncertainty is not a negative.

Verify ONLY candidate B against the complete query. Reason from the supplied context; a three-state finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nCandidate B semantic verdict:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "MATCH",
    "NO_MATCH",
    "UNCERTAIN"
  ],
  "allowed_outputs": null
}
```

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B/C

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: car; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: car; score: 0.98; bbox_xyxy_global: [180.0, 160.0, 250.0, 220.0]
- index: 2; label: car; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

Question:
Task: SELECT_SEM_MULTI
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
Match the category and ALL intrinsic attributes of this LOCATE TargetSpec: {"attributes": {"color": "red"}, "category": "car"}

Evaluate every marked candidate against the complete query. For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. Empty sets are valid; uncertainty is not a negative.

Verify ONLY candidate C against the complete query. Reason from the supplied context; a three-state finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nCandidate C semantic verdict:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "MATCH",
    "NO_MATCH",
    "UNCERTAIN"
  ],
  "allowed_outputs": null
}
```

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: car; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: car; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[SCOPE]
type: Region
bbox_xyxy_global: [0.0, 0.0, 640.0, 480.0]

Question:
Task: SELECT_SEM_MULTI
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
cars parked beside the road

Evaluate every marked candidate against the complete query. For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. Empty sets are valid; uncertainty is not a negative.

Verify ONLY candidate A against the complete query. Reason from the supplied context; a three-state finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nCandidate A semantic verdict:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "MATCH",
    "NO_MATCH",
    "UNCERTAIN"
  ],
  "allowed_outputs": null
}
```

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: CANDIDATE_CANVAS
[image_2] role: SCOPE

Structured results:
[CANVAS_ROLES]
CANDIDATES: A/B

[CANDIDATES]
type: EntitySet
entities:
- index: 0; label: car; score: 0.99; bbox_xyxy_global: [40.0, 40.0, 100.0, 100.0]
- index: 1; label: car; score: 0.97; bbox_xyxy_global: [420.0, 300.0, 480.0, 370.0]

[SCOPE]
type: Region
bbox_xyxy_global: [0.0, 0.0, 640.0, 480.0]

Question:
Task: SELECT_SEM_MULTI
Select ONLY from the existing marked candidates. Do not detect new objects, return new boxes, or replace candidates. Background landmarks in the query are context, not additional detection tasks.
Complete query (verbatim):
cars parked beside the road

Evaluate every marked candidate against the complete query. For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. Empty sets are valid; uncertainty is not a negative.

Verify ONLY candidate B against the complete query. Reason from the supplied context; a three-state finite decision follows.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[
  "\n\nCandidate B semantic verdict:"
]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": [
    "MATCH",
    "NO_MATCH",
    "UNCERTAIN"
  ],
  "allowed_outputs": null
}
```

## 05 VLM building reference + DINO cars -> SELECT_REL -> COUNT

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: LOCATE_SCOPE

Question:
Task: LOCATE_SINGLE. Localize one target object in this remote-sensing visual.
Target category: building
Intrinsic attributes: {}
Actual supplied visual size: 640 x 480 pixels.
Search scope: the supplied image or crop, and nothing outside it.
Use ONLY the target definition supplied to this LOCATE node. Do not solve the overall question, use downstream SELECT conditions, or invent additional landmarks.
If one well-supported matching object is identifiable, return exactly {"status":"OK","bbox":[x1,y1,x2,y2]}.
If none is visible, return exactly {"status":"NONE"}.
If several plausible targets cannot be distinguished, return exactly {"status":"AMBIGUOUS"}.
If evidence is insufficient, return exactly {"status":"UNRESOLVED"}.
The bbox uses pixel XYXY of the ACTUAL supplied visual, top-left origin. No Markdown, prose, or extra fields.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
```

终端评分候选 / 生成允许值：
```json
{
  "terminal_candidates": null,
  "allowed_outputs": null
}
```

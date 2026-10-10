# LOCATE / SELECT before rendered prompt capture

真实调用现有业务提示词、InputComposer、LazyQwen 公共包装和消息构造；视觉为合成审计样例，推理输出为 mock。不是模型准确率实验。

## LOCATE_SINGLE white square building -> ATTRIBUTE

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Visual inputs:
[image_1] role: RESULT_1

Question:
Locate the target in the image and output its bounding box as pixel coordinates of the image.
Target: white square building
Original question: What color is its roof?
Output format: x1 y1 x2 y2  (for example: 120 80 340 260)
Return only the four numbers.<|im_end|>
<|im_start|>assistant

```

终端评分后缀：
```json
[]
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

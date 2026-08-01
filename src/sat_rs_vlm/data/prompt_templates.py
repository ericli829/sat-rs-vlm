"""训练、评测和数据转换共用的遥感任务提示模板。"""

from __future__ import annotations

import json
import re
from typing import Any

from sat_rs_vlm.data.task_protocol import counting_json, parse_detection

DETECTION_JSON_SCHEMA = '{"label":"<class>","bbox":[x_min,y_min,x_max,y_max]}'
COUNTING_JSON_SCHEMA = '{"count":<integer>}'
DETECTION_FORMAT_SUFFIX = (
    "Return ONLY a JSON object with this schema: "
    f"{DETECTION_JSON_SCHEMA}. Coordinates must use normalized_0_1 xyxy format. "
    "Do not include any other text."
)
COUNTING_FORMAT_SUFFIX = (
    "Return ONLY a JSON object with this schema: "
    f"{COUNTING_JSON_SCHEMA}. Do not include any other text."
)
CAPTION_INSTRUCTION = (
    "Describe this remote sensing image in 1-3 concise sentences. "
    "Focus on the main land-cover types, objects, and spatial layout."
)
SCENE_FORMAT_SUFFIX = "Answer with the scene type only. Do not include any other text."


def detection_instruction(referring: str) -> str:
    """构造强制 normalized detection JSON 的指令。"""

    description = referring.strip()
    prefix = (
        f'Locate the object described as: "{description}".'
        if description
        else "Locate the target object."
    )
    return f"{prefix} {DETECTION_FORMAT_SUFFIX}"


def counting_instruction(question: str) -> str:
    """向 counting 问题追加唯一 JSON 输出协议。"""

    text = question.strip() or "How many target objects are visible?"
    return text if COUNTING_JSON_SCHEMA in text else f"{text} {COUNTING_FORMAT_SUFFIX}"


def scene_instruction(question: str) -> str:
    """向真正的场景分类问题追加短答约束。"""

    text = question.strip() or "What is the scene type?"
    return text if SCENE_FORMAT_SUFFIX in text else f"{text} {SCENE_FORMAT_SUFFIX}"


def strengthen_instruction(task_type: str, instruction: str) -> str:
    """按 task_type 幂等地加固结构化输出要求。"""

    task = task_type.strip().lower()
    text = instruction.strip()
    if task == "detection":
        if DETECTION_JSON_SCHEMA in text:
            return text
        match = re.search(r'described as:\s*"(.*?)"', text, flags=re.IGNORECASE | re.DOTALL)
        return detection_instruction(match.group(1) if match else text)
    if task == "counting":
        return counting_instruction(text)
    if task == "captioning":
        return text if "1-3 concise sentences" in text else CAPTION_INSTRUCTION
    if task == "scene_classification":
        return scene_instruction(text)
    return text


def strengthen_answer(task_type: str, answer: Any) -> str:
    """规范监督答案；计数无法可靠解析时保留原值供调用方统计。"""

    task = task_type.strip().lower()
    text = str(answer).strip()
    if task == "counting":
        return counting_json(answer) or text
    if task == "detection":
        parsed = parse_detection(answer)
        if parsed is None or not parsed.valid_coordinate_range:
            return text
        payload = {"label": parsed.label, "bbox": list(parsed.bbox)}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text

"""Adapters from official MME-RealWorld and XLRS records to project JSONL rows."""

from __future__ import annotations

from typing import Any


def _text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _identifier(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str | int) and str(value).strip():
            return str(value).strip()
    return ""


def _options(row: dict[str, Any]) -> list[str]:
    for key in ("Answer choices", "multi-choice options", "answer_choices", "options"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [str(item).strip() for item in value]
    raise ValueError("official record is missing multiple-choice options")


def _images(row: dict[str, Any]) -> list[str]:
    for key in ("Image", "image_path", "image", "images"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return [str(item).strip() for item in value]
    raise ValueError("official record does not contain reusable image path(s)")


def _answer(row: dict[str, Any]) -> str:
    value = row.get("Ground truth", row.get("answer"))
    if isinstance(value, list):
        return " ".join(str(item).strip().upper() for item in value)
    answer = str(value or "").strip().upper()
    if not answer:
        raise ValueError("official record has an empty answer")
    return answer


def _messages(images: list[str], prompt: str, answer: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": image} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": answer},
    ]


def adapt_mme_realworld(
    row: dict[str, Any],
    *,
    dataset_version: str,
    split: str,
    language: str,
    evaluation_scope: str = "subset_or_unspecified",
) -> dict[str, Any] | None:
    """Adapt one official MME row, retaining only the Remote Sensing subtask."""

    category_path = _text(row, "category")
    task = _text(row, "Task") or (
        "Perception" if "perception" in category_path.lower() else "Reasoning"
    )
    subtask = _text(row, "Subtask")
    if not subtask and category_path:
        subtask = category_path.split("/")[-1]
    if "".join(character for character in subtask.lower() if character.isalnum()) != (
        "remotesensing"
    ):
        return None
    category = _text(row, "Category", "l2-category") or "unknown"
    question = _text(row, "Text", "question")
    if not question:
        raise ValueError("official MME record has an empty question")
    options = _options(row)
    if language.lower().startswith("zh"):
        suffix = (
            "根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母"
            "（A, B, C, D 或 E）。\n最佳答案为："
        )
        option_heading = "选项如下所示:"
    else:
        suffix = (
            "Select the best answer to the above multiple-choice question based on the image. "
            "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
            "The best answer is:"
        )
        option_heading = "The choices are listed below:"
    prompt = f"{question}\n{option_heading}\n" + "\n".join(options) + f"\n{suffix}"
    sample_id = _identifier(row, "Question_id", "index", "id")
    if not sample_id:
        raise ValueError("official MME record is missing its question id")
    dataset_name = "MME-RealWorld-CN" if language.lower().startswith("zh") else "MME-RealWorld"
    answer = _answer(row)
    return {
        "id": sample_id,
        "task_type": "vqa",
        "messages": _messages(_images(row), prompt, answer),
        "metadata": {
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "split": split,
            "language": language,
            "prompt_profile": "mme_realworld_official_mcq_v1",
            "evaluation_scope": evaluation_scope,
            "official_task": task,
            "official_subtask": subtask,
            "official_category": category,
            "answer_choices": options,
        },
    }


def adapt_xlrs(
    row: dict[str, Any],
    *,
    dataset_version: str,
    split: str,
    language: str,
    evaluation_scope: str = "subset_or_unspecified",
) -> dict[str, Any]:
    """Adapt one official XLRS VQA record using the published prompt profiles."""

    category = _text(row, "category", "Category")
    parts = [part.strip() for part in category.split("/") if part.strip()]
    if len(parts) < 2:
        raise ValueError("official XLRS record category must contain task/subtask")
    task, subtask = parts[:2]
    question = _text(row, "question", "Text")
    if not question:
        raise ValueError("official XLRS record has an empty question")
    options = _options(row)
    is_multiselect = (
        "".join(character for character in category.lower() if character.isalnum())
        == "landuseclassificationoveralllanduseclassification"
    )
    if is_multiselect:
        suffix = (
            "Select the best answer(s) for the multiple-choice question based on the image. "
            "There may be more than one correct option. Only respond with the letter(s) "
            "corresponding to the correct answer(s) (A, B, C, D), with multiple choices "
            "separated by spaces.The answer(s) is(are):"
        )
        prompt_profile = "xlrs_bench_official_multiselect_v1"
    else:
        suffix = (
            "Select the best answer for the multiple-choice question based on the image. "
            "Only respond with the letter corresponding to the correct answer (A, B, C, D).\n"
            "The answer is:"
        )
        prompt_profile = "xlrs_bench_official_vqa_v1"
    prompt = (
        f"{question}The choices are listed below:\n"
        + "\n".join(options)
        + f"\n{suffix}"
    )
    sample_id = _identifier(row, "index", "Question_id", "id")
    if not sample_id:
        raise ValueError("official XLRS record is missing its question id")
    answer = _answer(row)
    return {
        "id": sample_id,
        "task_type": "vqa",
        "messages": _messages(_images(row), prompt, answer),
        "metadata": {
            "dataset": "XLRS-Bench",
            "dataset_version": dataset_version,
            "split": split,
            "language": language,
            "prompt_profile": prompt_profile,
            "evaluation_scope": evaluation_scope,
            "official_task": task,
            "official_subtask": subtask,
            "official_category": category,
            "official_l2_category": _text(row, "l2-category"),
            "answer_choices": options,
        },
    }

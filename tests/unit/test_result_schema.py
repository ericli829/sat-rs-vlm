from sat_rs_vlm.domain.result import BoundingBox, InferenceResult
from sat_rs_vlm.domain.tasks import TaskType


def test_inference_result_serializable() -> None:
    result = InferenceResult(
        task_type=TaskType.DETECTION,
        answer="ok",
        boxes=[BoundingBox(label="ship", x_min=0, y_min=0, x_max=1, y_max=1, confidence=0.9)],
        confidence=0.9,
    )
    dumped = result.model_dump(mode="json")
    assert dumped["task_type"] == "detection"
    assert dumped["boxes"][0]["label"] == "ship"

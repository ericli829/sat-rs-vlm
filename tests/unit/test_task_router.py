from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.task_router import TaskRouter
from sat_rs_vlm.domain.tasks import TaskType


def test_route_detection() -> None:
    task = TaskRouter().route(RemoteSensingInput(image_path="a.jpg", prompt="请检测机场位置"))
    assert task == TaskType.DETECTION


def test_route_counting() -> None:
    task = TaskRouter().route(RemoteSensingInput(image_path="a.jpg", prompt="这里有多少个油罐"))
    assert task == TaskType.COUNTING


def test_second_image_prefers_change_detection() -> None:
    task = TaskRouter().route(
        RemoteSensingInput(
            image_path="before.jpg",
            second_image_path="after.jpg",
            prompt="请描述图像",
        )
    )
    assert task == TaskType.CHANGE_DETECTION

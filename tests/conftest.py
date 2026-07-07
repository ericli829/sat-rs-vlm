import pytest

from sat_rs_vlm.application.inference_service import InferenceService
from sat_rs_vlm.models.mock_model import MockVLMEngine


@pytest.fixture
def inference_service() -> InferenceService:
    return InferenceService(engine=MockVLMEngine())

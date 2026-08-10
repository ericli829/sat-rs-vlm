"""已弃用的量化配置导入兼容层。"""

from sat_rs_vlm.quantization.config import (
    QuantBackendConfig,
    QuantBenchmarkConfig,
    QuantDataConfig,
    QuantEvaluationConfig,
    QuantGenerationConfig,
    QuantizationExperimentConfig,
    QuantModelConfig,
    QuantOutputConfig,
    QuantSensitivityConfig,
    load_quantization_config,
)

__all__ = [
    "QuantBackendConfig",
    "QuantBenchmarkConfig",
    "QuantDataConfig",
    "QuantEvaluationConfig",
    "QuantGenerationConfig",
    "QuantModelConfig",
    "QuantOutputConfig",
    "QuantSensitivityConfig",
    "QuantizationExperimentConfig",
    "load_quantization_config",
]
